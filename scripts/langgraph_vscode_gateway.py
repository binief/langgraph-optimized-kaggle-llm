import asyncio
import hashlib
import json
import os
import re
import sqlite3
import time
import traceback
import uuid
from collections import Counter
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional, Set
from typing_extensions import TypedDict

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from langgraph.graph import END, START, StateGraph

SOLVER_BASE = os.environ.get("SOLVER_BASE", "http://127.0.0.1:8000/v1")
COMPACTOR_BASE = os.environ.get("COMPACTOR_BASE", "http://127.0.0.1:8001/v1")

# Watermarks for 32K-token context (~3.5-4 chars/tok)
SOFT_WATERMARK_CHARS = int(os.environ.get("SOFT_WATERMARK_CHARS", "72000"))   # ~60%: background pre-compaction
HIGH_WATERMARK_CHARS = int(os.environ.get("HIGH_WATERMARK_CHARS", "92000"))   # ~75%: apply compaction
LOW_WATERMARK_CHARS = int(os.environ.get("LOW_WATERMARK_CHARS", "48000"))     # ~40%: target tail size
MAX_THINK_CHARS = int(os.environ.get("MAX_THINK_CHARS", "12000"))
DYNAMIC_SAMPLING = os.environ.get("DYNAMIC_SAMPLING", "1") == "1"
CACHE_DB_PATH = os.environ.get("CACHE_DB_PATH", "/tmp/compaction_cache.db")

http_client: Optional[httpx.AsyncClient] = None
bg_precompact_inflight: Set[str] = set()
bg_tasks: Set[asyncio.Task] = set()

METRICS: Dict[str, Any] = {
    "requests_total": 0,
    "stream_requests": 0,
    "lightweight_offloaded": 0,
    "pruned_messages": 0,
    "precompactions_triggered": 0,
    "compactions_applied": 0,
    "compaction_cache_hits": 0,
    "errors_detected": 0,
    "stuck_loops_broken": 0,
    "thinking_loops_aborted": 0,
    "ttft_samples": 0,
    "ttft_total_ms": 0.0,
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    global http_client
    http_client = httpx.AsyncClient(
        timeout=httpx.Timeout(connect=10.0, read=600.0, write=60.0, pool=30.0),
        limits=httpx.Limits(max_connections=30, max_keepalive_connections=15),
    )
    yield
    for t in list(bg_tasks):
        t.cancel()
    await http_client.aclose()


# ==============================================================================
# Persistent SQLite Summary Cache
# ==============================================================================
def init_db():
    conn = sqlite3.connect(CACHE_DB_PATH)
    conn.execute(
        "CREATE TABLE IF NOT EXISTS summaries (hash TEXT PRIMARY KEY, summary TEXT, created_at DATETIME DEFAULT CURRENT_TIMESTAMP)"
    )
    conn.commit()
    conn.close()


def get_cached_summary(key_hash: str) -> Optional[str]:
    conn = sqlite3.connect(CACHE_DB_PATH)
    row = conn.execute("SELECT summary FROM summaries WHERE hash = ?", (key_hash,)).fetchone()
    conn.close()
    return row[0] if row else None


def put_cached_summary(key_hash: str, summary: str):
    conn = sqlite3.connect(CACHE_DB_PATH)
    conn.execute("INSERT OR REPLACE INTO summaries (hash, summary) VALUES (?, ?)", (key_hash, summary))
    conn.commit()
    conn.close()


init_db()


# ==============================================================================
# Helpers: Token Budget, File/Error Ledger, Loop Detection, Lightweight Offload
# ==============================================================================
FILE_PATH_RE = re.compile(
    r"(?:^|[\s\"'`(])([A-Za-z0-9_.\-/]+?\.(?:py|js|ts|tsx|jsx|json|yaml|yml|toml|md|cpp|c|h|hpp|rs|go|java|sh|sql|css|html|ipynb))(?=$|[\s\"'`),:])"
)
ERROR_LINE_RE = re.compile(r"(error|fail|failed|exception|traceback|assert|warn|enoent|line \d+)", re.IGNORECASE)
ERROR_RE = re.compile(
    r"(traceback \(most recent call last\)|syntaxerror|typeerror|nameerror|referenceerror|"
    r"compilation failed|error:|failed|exit code [1-9]|assertionerror|exception|enoent)",
    re.IGNORECASE,
)
LIGHTWEIGHT_RE = re.compile(
    r"(generate a (?:concise |short )?title|summarize (?:this|the) (?:conversation|chat)|"
    r"commit message|suggest a branch name|title for this conversation)",
    re.IGNORECASE,
)


def msg_chars(m: Dict[str, Any]) -> int:
    c = m.get("content")
    total = len(c) if isinstance(c, str) else len(json.dumps(c)) if c is not None else 0
    if m.get("tool_calls"):
        total += len(json.dumps(m["tool_calls"]))
    return total


def approx_chars(messages: List[Dict[str, Any]]) -> int:
    return sum(msg_chars(m) for m in messages)


def build_file_error_ledger(msgs: List[Dict[str, Any]]) -> str:
    files: List[str] = []
    seen_files: Set[str] = set()
    recent_errors: List[str] = []

    def add_file(p: str):
        p = p.strip()
        if p and len(p) < 140 and p not in seen_files and not p.startswith("http"):
            seen_files.add(p)
            files.append(p)

    for m in msgs:
        for tc in m.get("tool_calls") or []:
            args = tc.get("function", {}).get("arguments", "")
            if isinstance(args, str):
                for match in FILE_PATH_RE.findall(args):
                    add_file(match)
        c = m.get("content")
        text = c if isinstance(c, str) else json.dumps(c) if c is not None else ""
        for match in FILE_PATH_RE.findall(text[:4000]):
            add_file(match)
        if m.get("role") in ("tool", "user") and ERROR_RE.search(text):
            for line in text.splitlines():
                if ERROR_LINE_RE.search(line) and 10 <= len(line.strip()) <= 200:
                    recent_errors.append(line.strip())

    parts = []
    if files:
        parts.append("Files touched/referenced: " + ", ".join(files[-20:]))
    if recent_errors:
        dedup_errs = list(dict.fromkeys(recent_errors[-6:]))
        parts.append("Recent errors observed:\n- " + "\n- ".join(dedup_errs))
    return "\n".join(parts)


def is_lightweight_request(body: Dict[str, Any]) -> bool:
    if body.get("tools"):
        return False
    msgs = body.get("messages") or []
    if len(msgs) > 4 or approx_chars(msgs) > 6000:
        return False
    combined = "\n".join(str(m.get("content") or "") for m in msgs)
    return bool(LIGHTWEIGHT_RE.search(combined))


def detect_thinking_loop(think_buf: str) -> Optional[str]:
    if not think_buf:
        return None
    if len(think_buf) >= MAX_THINK_CHARS:
        return f"thinking exceeded {MAX_THINK_CHARS} chars"
    if len(think_buf) >= 240:
        tail = think_buf[-80:]
        if think_buf[-240:] == tail * 3:
            return "consecutive tail repetition"
    chunks = re.split(r"(?:\n+|(?<=[.!?])\s+)", think_buf[-5000:])
    norm_counts = Counter()
    for ch in chunks:
        norm = re.sub(r"\s+", " ", ch).strip().lower()
        if len(norm) >= 45:
            norm_counts[norm] += 1
            if norm_counts[norm] >= 3:
                return f"repeated thinking phrase ({norm[:50]}...)"
    return None


def normalize_content(c: Any) -> Any:
    if isinstance(c, list):
        parts = []
        for item in c:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(item.get("text", ""))
            elif isinstance(item, str):
                parts.append(item)
            else:
                return c
        return "\n".join(parts)
    return c


def normalize_payload(body: Dict[str, Any]) -> Dict[str, Any]:
    p = dict(body)
    msgs = []
    for m in p.get("messages", []):
        m2 = dict(m)
        if m2.get("role") == "developer":
            m2["role"] = "system"
        if "content" in m2:
            m2["content"] = normalize_content(m2["content"])
        msgs.append(m2)
    p["messages"] = msgs

    if "max_completion_tokens" in p and "max_tokens" not in p:
        p["max_tokens"] = p.pop("max_completion_tokens")
    else:
        p.pop("max_completion_tokens", None)

    for k in ("store", "metadata", "service_tier", "reasoning_effort", "prompt_cache_key"):
        p.pop(k, None)

    p.setdefault("presence_penalty", 0.4)
    p.setdefault("repeat_penalty", 1.05)
    return p


def serialize_for_compactor(msgs: List[Dict[str, Any]]) -> str:
    lines = []
    for m in msgs:
        role = m.get("role", "unknown").upper()
        c = m.get("content") or ""
        if not isinstance(c, str):
            c = json.dumps(c)
        if len(c) > 2000:
            c = c[:1200] + "\n...[trimmed]...\n" + c[-500:]
        if m.get("tool_calls"):
            c += "\n[ToolCalls]: " + json.dumps(m["tool_calls"])[:800]
        lines.append(f"[{role}]\n{c}")
    return "\n\n".join(lines)[-22000:]


async def summarize_with_compactor(msgs_to_compact: List[Dict[str, Any]]) -> str:
    transcript = serialize_for_compactor(msgs_to_compact)
    key_hash = hashlib.sha256(transcript.encode("utf-8")).hexdigest()
    cached = get_cached_summary(key_hash)
    if cached:
        METRICS["compaction_cache_hits"] += 1
        return cached

    compactor_prompt = (
        "Compress the following earlier coding-agent conversation into a dense, factual working memory.\n"
        "Preserve ONLY:\n"
        "1. Decisions, requirements, and constraints.\n"
        "2. What was tried, what succeeded, and what failed.\n"
        "3. Current progress and next pending step.\n"
        "Be concise (under 350 words). No fluff.\n\n"
        f"{transcript}"
    )

    try:
        r = await http_client.post(
            f"{COMPACTOR_BASE}/chat/completions",
            json={
                "model": "qwen3-1.7b-compactor",
                "messages": [{"role": "user", "content": compactor_prompt}],
                "temperature": 0.1,
                "max_tokens": 512,
                "chat_template_kwargs": {"enable_thinking": False},
            },
        )
        r.raise_for_status()
        summary = r.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        summary = f"(Auto-compacted {len(msgs_to_compact)} older messages; compactor fallback: {e})"

    put_cached_summary(key_hash, summary)
    return summary


# ==============================================================================
# LangGraph State & Nodes
# ==============================================================================
class GatewayState(TypedDict):
    payload: Dict[str, Any]
    pruned_count: int
    precompact_started: bool
    compacted: bool
    has_error: bool
    is_stuck_loop: bool
    enable_thinking: bool
    thinking_loop_retried: bool
    response_json: Optional[Dict[str, Any]]


# Node 1: Smart Tool-Output Pruning (zero LLM cost, keeps recent 4 turns untouched)
def prune_tool_outputs_node(state: GatewayState) -> dict:
    payload = dict(state["payload"])
    msgs = [dict(m) for m in payload.get("messages", [])]
    if len(msgs) <= 6:
        return {"payload": payload, "pruned_count": 0}

    pruned = 0
    cutoff = len(msgs) - 4
    for i in range(cutoff):
        m = msgs[i]
        if m.get("role") == "tool" and isinstance(m.get("content"), str) and len(m["content"]) > 1800:
            lines = m["content"].splitlines()
            if len(lines) > 70:
                head_lines = lines[:30]
                tail_lines = lines[-30:]
                err_lines = [ln for ln in lines[30:-30] if ERROR_LINE_RE.search(ln)][:30]
                omitted = len(lines) - 60 - len(err_lines)
                parts = head_lines + [f"\n...[pruned {max(omitted, 0)} non-error lines]..."]
                if err_lines:
                    parts += ["[Kept matching error/warn lines:]"] + err_lines
                parts += ["...[continued]..."] + tail_lines
                m["content"] = "\n".join(parts)
                pruned += 1

    if pruned:
        METRICS["pruned_messages"] += pruned
        payload["messages"] = msgs
    return {"payload": payload, "pruned_count": pruned}


def split_head_older_tail(msgs: List[Dict[str, Any]]):
    head = []
    for m in msgs:
        if m.get("role") == "system" and len(head) < 2:
            head.append(m)
        else:
            break
    body = msgs[len(head):]

    kept_tail: List[Dict[str, Any]] = []
    budget = approx_chars(head)
    split_idx = len(body)
    for i in range(len(body) - 1, -1, -1):
        c = msg_chars(body[i])
        if kept_tail and budget + c > LOW_WATERMARK_CHARS:
            break
        kept_tail.insert(0, body[i])
        budget += c
        split_idx = i

    while split_idx > 0 and kept_tail and kept_tail[0].get("role") == "tool":
        split_idx -= 1
        kept_tail.insert(0, body[split_idx])

    return head, body[:split_idx], kept_tail


# Node 2: Soft Background Pre-Compaction (60%) + Hard Watermark Compaction (75%) + Ledger
async def compact_context_node(state: GatewayState) -> dict:
    payload = dict(state["payload"])
    msgs = [dict(m) for m in payload.get("messages", [])]
    total = approx_chars(msgs)

    if total <= SOFT_WATERMARK_CHARS or len(msgs) <= 6:
        return {"payload": payload, "compacted": False, "precompact_started": False}

    head, older, kept_tail = split_head_older_tail(msgs)
    if not older:
        return {"payload": payload, "compacted": False, "precompact_started": False}

    # Soft watermark: pre-warm SQLite cache in the background without altering current prompt prefix
    if total <= HIGH_WATERMARK_CHARS:
        transcript = serialize_for_compactor(older)
        key_hash = hashlib.sha256(transcript.encode("utf-8")).hexdigest()
        if not get_cached_summary(key_hash) and key_hash not in bg_precompact_inflight:
            bg_precompact_inflight.add(key_hash)
            METRICS["precompactions_triggered"] += 1

            async def _bg():
                try:
                    await summarize_with_compactor(older)
                finally:
                    bg_precompact_inflight.discard(key_hash)

            t = asyncio.create_task(_bg())
            bg_tasks.add(t)
            t.add_done_callback(bg_tasks.discard)
        return {"payload": payload, "compacted": False, "precompact_started": True}

    # Hard watermark: apply ledger + cached (or freshly generated) summary
    ledger = build_file_error_ledger(older)
    summary = await summarize_with_compactor(older)
    METRICS["compactions_applied"] += 1

    content_blocks = [f"[Compacted Earlier Session Memory ({len(older)} turns)]"]
    if ledger:
        content_blocks.append(f"## Deterministic File & Error Ledger\n{ledger}")
    content_blocks.append(f"## Summary\n{summary}")

    summary_msg = {"role": "system", "content": "\n\n".join(content_blocks)}
    payload["messages"] = head + [summary_msg] + kept_tail
    return {"payload": payload, "compacted": True, "precompact_started": False}


# Node 3: Error Detection, Stuck-Loop Breaker, Dynamic Sampling & Thinking Router
def analyze_and_route_node(state: GatewayState) -> dict:
    payload = dict(state["payload"])
    msgs = list(payload.get("messages", []))

    last_msg = msgs[-1] if msgs else {}
    last_role = last_msg.get("role", "")
    last_content = last_msg.get("content", "")
    last_text = last_content if isinstance(last_content, str) else json.dumps(last_content)

    has_error = bool(ERROR_RE.search(last_text))
    if has_error:
        METRICS["errors_detected"] += 1

    assistant_tool_sigs = []
    for m in reversed(msgs):
        if m.get("role") == "assistant" and m.get("tool_calls"):
            sig = json.dumps(
                [(tc.get("function", {}).get("name"), tc.get("function", {}).get("arguments")) for tc in m["tool_calls"]],
                sort_keys=True,
            )
            assistant_tool_sigs.append(sig)
            if len(assistant_tool_sigs) == 2:
                break
    is_stuck = len(assistant_tool_sigs) == 2 and assistant_tool_sigs[0] == assistant_tool_sigs[1]
    if is_stuck:
        METRICS["stuck_loops_broken"] += 1

    enable_thinking = (last_role == "user") or has_error or is_stuck

    hints = []
    if has_error:
        hints.append(
            "An error/failure was observed in the latest output. Diagnose the root cause from the traceback/message before editing."
        )
    if is_stuck:
        hints.append(
            "WARNING: You repeated the exact same tool call twice. Do NOT repeat it again—inspect different lines, check paths, or change strategy."
        )

    if hints:
        insert_at = 1 if msgs and msgs[0].get("role") == "system" else 0
        msgs.insert(insert_at, {"role": "system", "content": "[LangGraph Guard] " + " ".join(hints)})
        payload["messages"] = msgs

    kwargs = dict(payload.get("chat_template_kwargs") or {})
    if "enable_thinking" not in kwargs:
        kwargs["enable_thinking"] = enable_thinking
    payload["chat_template_kwargs"] = kwargs

    if DYNAMIC_SAMPLING:
        if enable_thinking:
            payload["temperature"] = 0.6
            payload["top_p"] = 0.95
        else:
            payload["temperature"] = 0.15
            payload["top_p"] = 0.9

    return {
        "payload": payload,
        "has_error": has_error,
        "is_stuck_loop": is_stuck,
        "enable_thinking": bool(kwargs["enable_thinking"]),
    }


# Node 4 & 5: Non-streaming Solver Call & Tool-Call Repair
async def call_solver_node(state: GatewayState) -> dict:
    payload = dict(state["payload"])
    payload["model"] = "qwen3.8-27b"
    payload["stream"] = False
    r = await http_client.post(f"{SOLVER_BASE}/chat/completions", json=payload)
    if r.status_code != 200:
        raise RuntimeError(f"Solver returned {r.status_code}: {r.text}")
    resp = r.json()

    msg = (resp.get("choices") or [{}])[0].get("message", {})
    reasoning = msg.get("reasoning_content") or ""
    content = msg.get("content") or ""
    if not reasoning and "<think>" in content:
        reasoning = content.split("<think>", 1)[1].split("</think>", 1)[0]

    loop_reason = detect_thinking_loop(reasoning)
    stuck_in_think = (
        (resp.get("choices") or [{}])[0].get("finish_reason") == "length"
        and not (msg.get("tool_calls") or content.replace(f"<think>{reasoning}</think>", "").strip())
    )

    retried = False
    if (loop_reason or stuck_in_think) and payload.get("chat_template_kwargs", {}).get("enable_thinking", True):
        METRICS["thinking_loops_aborted"] += 1
        print(f"[LangGraph] Non-stream thinking loop ({loop_reason or 'length_exhausted'}), retrying directly...", flush=True)
        retry_payload = dict(payload)
        retry_kwargs = dict(retry_payload.get("chat_template_kwargs") or {})
        retry_kwargs["enable_thinking"] = False
        retry_payload["chat_template_kwargs"] = retry_kwargs
        retry_payload["temperature"] = 0.2
        r2 = await http_client.post(f"{SOLVER_BASE}/chat/completions", json=retry_payload)
        if r2.status_code == 200:
            resp = r2.json()
            retried = True

    return {"response_json": resp, "thinking_loop_retried": retried}


def repair_tool_calls_node(state: GatewayState) -> dict:
    resp = state.get("response_json")
    if not resp:
        return {}
    for choice in resp.get("choices", []):
        msg = choice.get("message", {})
        for tc in msg.get("tool_calls") or []:
            fn = tc.get("function", {})
            args = fn.get("arguments")
            if isinstance(args, str):
                try:
                    json.loads(args)
                except json.JSONDecodeError:
                    cleaned = args.strip().strip("`")
                    cleaned = re.sub(r",\s*([}\]])", r"\1", cleaned)
                    try:
                        json.loads(cleaned)
                        fn["arguments"] = cleaned
                    except Exception:
                        pass
    return {"response_json": resp}


prep_wf = StateGraph(GatewayState)
prep_wf.add_node("prune_tools", prune_tool_outputs_node)
prep_wf.add_node("compact_context", compact_context_node)
prep_wf.add_node("analyze_and_route", analyze_and_route_node)
prep_wf.add_edge(START, "prune_tools")
prep_wf.add_edge("prune_tools", "compact_context")
prep_wf.add_edge("compact_context", "analyze_and_route")
prep_wf.add_edge("analyze_and_route", END)
prep_graph = prep_wf.compile()

full_wf = StateGraph(GatewayState)
full_wf.add_node("prune_tools", prune_tool_outputs_node)
full_wf.add_node("compact_context", compact_context_node)
full_wf.add_node("analyze_and_route", analyze_and_route_node)
full_wf.add_node("call_solver", call_solver_node)
full_wf.add_node("repair_tools", repair_tool_calls_node)
full_wf.add_edge(START, "prune_tools")
full_wf.add_edge("prune_tools", "compact_context")
full_wf.add_edge("compact_context", "analyze_and_route")
full_wf.add_edge("analyze_and_route", "call_solver")
full_wf.add_edge("call_solver", "repair_tools")
full_wf.add_edge("repair_tools", END)
full_graph = full_wf.compile()


# ==============================================================================
# FastAPI OpenAI-Compatible Endpoint (:8080)
# ==============================================================================
app = FastAPI(title="LangGraph Dual-Model Coding Gateway", lifespan=lifespan)


@app.get("/health")
async def health():
    status = {"status": "ok", "solver": SOLVER_BASE, "compactor": COMPACTOR_BASE}
    for name, url in (("solver_up", f"{SOLVER_BASE[:-3]}/health"), ("compactor_up", f"{COMPACTOR_BASE[:-3]}/health")):
        try:
            r = await http_client.get(url, timeout=2.0)
            status[name] = (r.status_code == 200)
        except Exception:
            status[name] = False
    return status


@app.get("/metrics")
def metrics():
    avg_ttft = (METRICS["ttft_total_ms"] / METRICS["ttft_samples"]) if METRICS["ttft_samples"] else 0.0
    return {**METRICS, "avg_ttft_ms": round(avg_ttft, 1)}


@app.api_route("/v1/models", methods=["GET", "HEAD"])
def list_models():
    return {
        "object": "list",
        "data": [{"id": "qwen3.8-27b", "object": "model", "owned_by": "langgraph-gateway"}],
    }


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    t_req = time.monotonic()
    METRICS["requests_total"] += 1
    raw_body = await request.json()
    body = normalize_payload(raw_body)
    stream = bool(body.get("stream", False))

    # Fast-path offload lightweight title/commit/summary requests to :8001 so 27B slot stays free
    if is_lightweight_request(body):
        METRICS["lightweight_offloaded"] += 1
        off_payload = dict(body)
        off_payload["model"] = "qwen3-1.7b-compactor"
        off_payload["chat_template_kwargs"] = {"enable_thinking": False}
        off_payload["temperature"] = 0.2
        print("[LangGraph] Offloaded lightweight request to compactor (:8001)", flush=True)
        if stream:
            req = http_client.build_request("POST", f"{COMPACTOR_BASE}/chat/completions", json=off_payload)
            up = await http_client.send(req, stream=True)
            if up.status_code != 200:
                err_b = (await up.aread()).decode("utf-8", errors="replace")
                await up.aclose()
                return JSONResponse(status_code=up.status_code, content={"error": {"message": err_b}})

            async def _off_stream():
                try:
                    async for raw in up.aiter_bytes():
                        if raw:
                            yield raw
                finally:
                    await up.aclose()

            return StreamingResponse(_off_stream(), media_type="text/event-stream")
        r = await http_client.post(f"{COMPACTOR_BASE}/chat/completions", json=off_payload)
        return JSONResponse(status_code=r.status_code, content=r.json())

    init_state: GatewayState = {
        "payload": body,
        "pruned_count": 0,
        "precompact_started": False,
        "compacted": False,
        "has_error": False,
        "is_stuck_loop": False,
        "enable_thinking": True,
        "thinking_loop_retried": False,
        "response_json": None,
    }

    if stream:
        METRICS["stream_requests"] += 1
        state_out = await prep_graph.ainvoke(init_state)
        print(
            f"[LangGraph] msgs={len(state_out['payload'].get('messages', []))} "
            f"pruned={state_out['pruned_count']} precompact={state_out['precompact_started']} "
            f"compacted={state_out['compacted']} err={state_out['has_error']} "
            f"stuck={state_out['is_stuck_loop']} think={state_out['enable_thinking']} stream=True",
            flush=True,
        )
        prepared_payload = dict(state_out["payload"])
        prepared_payload["model"] = "qwen3.8-27b"
        prepared_payload["stream"] = True

        req = http_client.build_request("POST", f"{SOLVER_BASE}/chat/completions", json=prepared_payload)
        upstream_resp = await http_client.send(req, stream=True)

        if upstream_resp.status_code != 200:
            err_body = (await upstream_resp.aread()).decode("utf-8", errors="replace")
            await upstream_resp.aclose()
            print(f"[Upstream {upstream_resp.status_code}] {err_body}", flush=True)
            return JSONResponse(status_code=upstream_resp.status_code, content={"error": {"message": err_body}})

        async def sse_proxy():
            first_token_recorded = False

            async def stream_attempt(resp_obj, check_thinking_loop: bool, retry_flag: dict):
                nonlocal first_token_recorded
                queue: asyncio.Queue = asyncio.Queue()
                loop_state = {"reason": None, "saw_inline_think": False, "closed_inline_think": False, "saw_reasoning_field": False}

                async def reader():
                    nonlocal first_token_recorded
                    think_buf = ""
                    in_inline_think = False
                    buf = ""
                    try:
                        async for raw in resp_obj.aiter_bytes():
                            if not raw:
                                continue
                            if not first_token_recorded:
                                first_token_recorded = True
                                METRICS["ttft_samples"] += 1
                                METRICS["ttft_total_ms"] += (time.monotonic() - t_req) * 1000.0
                            text = raw.decode("utf-8", errors="replace")
                            buf += text

                            while "\n" in buf:
                                line, buf = buf.split("\n", 1)
                                stripped = line.strip()
                                if check_thinking_loop and stripped.startswith("data:") and stripped != "data: [DONE]":
                                    try:
                                        evt = json.loads(stripped[5:].strip())
                                        delta = (evt.get("choices") or [{}])[0].get("delta", {})
                                        rc = delta.get("reasoning_content")
                                        ct = delta.get("content")
                                        if rc:
                                            loop_state["saw_reasoning_field"] = True
                                            think_buf += rc
                                        elif ct:
                                            if "<think>" in ct:
                                                in_inline_think = True
                                                loop_state["saw_inline_think"] = True
                                            if in_inline_think:
                                                think_buf += ct
                                            if "</think>" in ct:
                                                in_inline_think = False
                                                loop_state["closed_inline_think"] = True
                                        if think_buf and (rc or in_inline_think):
                                            reason = detect_thinking_loop(think_buf)
                                            if reason:
                                                loop_state["reason"] = reason
                                                return
                                    except Exception:
                                        pass
                                if stripped == "data: [DONE]" and check_thinking_loop and loop_state["reason"]:
                                    return
                                await queue.put((line + "\n").encode("utf-8"))
                    except Exception as exc:
                        print("[SSE stream error]", traceback.format_exc(), flush=True)
                        err_evt = f'data: {json.dumps({"error": {"message": str(exc)}})}\n\n'
                        await queue.put(err_evt.encode("utf-8"))
                    finally:
                        if buf and not loop_state["reason"]:
                            await queue.put(buf.encode("utf-8"))
                        await resp_obj.aclose()
                        await queue.put(None)

                task = asyncio.create_task(reader())
                try:
                    while True:
                        try:
                            item = await asyncio.wait_for(queue.get(), timeout=5.0)
                        except asyncio.TimeoutError:
                            yield b": keepalive\n\n"
                            continue
                        if item is None:
                            break
                        if check_thinking_loop and b"data: [DONE]" in item:
                            continue
                        yield item
                finally:
                    if not task.done():
                        task.cancel()

                if loop_state["reason"]:
                    METRICS["thinking_loops_aborted"] += 1
                    print(f"[LangGraph] Thinking loop aborted ({loop_state['reason']}); retrying with enable_thinking=False", flush=True)
                    cid = f"chatcmpl-{uuid.uuid4().hex[:12]}"
                    if loop_state["saw_inline_think"] and not loop_state["closed_inline_think"]:
                        close_evt = {
                            "id": cid, "object": "chat.completion.chunk", "created": int(time.time()),
                            "model": "qwen3.8-27b",
                            "choices": [{"index": 0, "delta": {"content": "\n</think>\n\n"}, "finish_reason": None}],
                        }
                        yield f"data: {json.dumps(close_evt)}\n\n".encode("utf-8")
                    elif loop_state["saw_reasoning_field"]:
                        note_evt = {
                            "id": cid, "object": "chat.completion.chunk", "created": int(time.time()),
                            "model": "qwen3.8-27b",
                            "choices": [{"index": 0, "delta": {"reasoning_content": "\n[Thinking loop stopped — answering directly]\n"}, "finish_reason": None}],
                        }
                        yield f"data: {json.dumps(note_evt)}\n\n".encode("utf-8")
                    retry_flag["triggered"] = True
                    return

            yield b": connected\n\n"
            should_check_loop = bool(prepared_payload.get("chat_template_kwargs", {}).get("enable_thinking", True))
            retry_flag = {"triggered": False}
            async for chunk in stream_attempt(upstream_resp, check_thinking_loop=should_check_loop, retry_flag=retry_flag):
                yield chunk

            if retry_flag["triggered"]:
                retry_payload = dict(prepared_payload)
                rk = dict(retry_payload.get("chat_template_kwargs") or {})
                rk["enable_thinking"] = False
                retry_payload["chat_template_kwargs"] = rk
                retry_payload["temperature"] = 0.2
                r_req = http_client.build_request("POST", f"{SOLVER_BASE}/chat/completions", json=retry_payload)
                retry_resp = await http_client.send(r_req, stream=True)
                if retry_resp.status_code != 200:
                    err_b = (await retry_resp.aread()).decode("utf-8", errors="replace")
                    await retry_resp.aclose()
                    yield f'data: {json.dumps({"error": {"message": err_b}})}\n\ndata: [DONE]\n\n'.encode("utf-8")
                    return
                async for chunk in stream_attempt(retry_resp, check_thinking_loop=False, retry_flag={"triggered": False}):
                    yield chunk
            elif should_check_loop:
                yield b"data: [DONE]\n\n"

        return StreamingResponse(
            sse_proxy(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    try:
        state_out = await full_graph.ainvoke(init_state)
        print(
            f"[LangGraph] msgs={len(state_out['payload'].get('messages', []))} "
            f"pruned={state_out['pruned_count']} precompact={state_out['precompact_started']} "
            f"compacted={state_out['compacted']} err={state_out['has_error']} "
            f"stuck={state_out['is_stuck_loop']} think={state_out['enable_thinking']} "
            f"think_retry={state_out['thinking_loop_retried']} stream=False",
            flush=True,
        )
        return JSONResponse(state_out["response_json"])
    except Exception as e:
        print("[Gateway error]", traceback.format_exc(), flush=True)
        return JSONResponse(status_code=500, content={"error": {"message": str(e)}})