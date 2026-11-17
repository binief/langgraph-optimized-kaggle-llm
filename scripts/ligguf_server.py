import asyncio
import os
import subprocess
import tempfile
import time
import uuid
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

LIGGUF_BIN = "/kaggle/working/ligguf/ligguf-fast"
MODEL_PATH = os.environ.get("LIGGUF_MODEL_PATH", "/kaggle/working/models/Qwen3-0.6B-Q8_0.gguf")
CONTEXT_LEN = int(os.environ.get("LIGGUF_CTX", "2048"))
USE_CHATML = os.environ.get("LIGGUF_CHATML", "1") == "1"

app = FastAPI(title="LiGGUF OpenAI-Compatible API")
gen_lock = asyncio.Lock()  # serialize CPU inference requests


class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    model: str = "ligguf"
    messages: List[Message]
    max_tokens: Optional[int] = 256
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.9
    top_k: Optional[int] = 40


class CompletionRequest(BaseModel):
    model: str = "ligguf"
    prompt: str
    max_tokens: Optional[int] = 256
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.9
    top_k: Optional[int] = 40


def format_chat(messages: List[Message]) -> str:
    if USE_CHATML:
        parts = [f"<|im_start|>{m.role}\n{m.content}<|im_end|>" for m in messages]
        parts.append("<|im_start|>assistant\n<think>\n\n</think>\n\n")
        return "\n".join(parts)
    lines = [f"{m.role.capitalize()}: {m.content}" for m in messages]
    lines.append("Assistant:")
    return "\n".join(lines)


def run_ligguf(prompt: str, max_tokens: int, temperature: float, top_p: float, top_k: int) -> str:
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as tf:
        tf.write(prompt)
        prompt_file = tf.name

    try:
        cmd = [
            LIGGUF_BIN,
            "-m", MODEL_PATH,
            "-c", str(CONTEXT_LEN),
            "-f", prompt_file,
            "-n", str(max_tokens),
            "-K", str(top_k),
            "-T", str(top_p),
            "-M", str(max(temperature, 0.0)),
        ]
        if temperature <= 0:
            cmd.append("-G")

        res = subprocess.run(cmd, capture_output=True, text=True, check=True)
        out = res.stdout

        if "<<<GEN_START>>>" in out:
            out = out.split("<<<GEN_START>>>", 1)[1]
        if "\n\nInference-only tok/s:" in out:
            out = out.split("\n\nInference-only tok/s:", 1)[0]

        for stop_tok in ("<|im_end|>", "<|endoftext|>"):
            if stop_tok in out:
                out = out.split(stop_tok, 1)[0]

        return out.strip()
    finally:
        if os.path.exists(prompt_file):
            os.remove(prompt_file)


@app.get("/health")
async def health():
    return {"status": "ok", "model": MODEL_PATH}


@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [{"id": "ligguf", "object": "model", "owned_by": "local"}],
    }


@app.post("/v1/chat/completions")
async def chat_completions(req: ChatRequest):
    prompt = format_chat(req.messages)
    async with gen_lock:
        try:
            text = await asyncio.to_thread(
                run_ligguf,
                prompt,
                req.max_tokens or 256,
                req.temperature if req.temperature is not None else 0.7,
                req.top_p if req.top_p is not None else 0.9,
                req.top_k if req.top_k is not None else 40,
            )
        except subprocess.CalledProcessError as e:
            raise HTTPException(status_code=500, detail=e.stderr or str(e))

    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": req.model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": "stop",
            }
        ],
    }


@app.post("/v1/completions")
async def completions(req: CompletionRequest):
    async with gen_lock:
        try:
            text = await asyncio.to_thread(
                run_ligguf,
                req.prompt,
                req.max_tokens or 256,
                req.temperature if req.temperature is not None else 0.7,
                req.top_p if req.top_p is not None else 0.9,
                req.top_k if req.top_k is not None else 40,
            )
        except subprocess.CalledProcessError as e:
            raise HTTPException(status_code=500, detail=e.stderr or str(e))

    return {
        "id": f"cmpl-{uuid.uuid4().hex[:12]}",
        "object": "text_completion",
        "created": int(time.time()),
        "model": req.model,
        "choices": [{"index": 0, "text": text, "finish_reason": "stop"}],
    }