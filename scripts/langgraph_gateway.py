import os
import time
import uuid
from typing import Annotated, List, Optional
from typing_extensions import TypedDict

import httpx
from fastapi import FastAPI
from pydantic import BaseModel

from langchain_core.messages import SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

LLM_BASE = os.environ.get("LLM_BASE", "http://127.0.0.1:8000/v1")
DOWNSTREAM_API_BASE = os.environ.get("DOWNSTREAM_API_BASE", "http://127.0.0.1:9000")

# --- 1. Wrap REST API endpoints as typed LangChain tools ---
@tool
def get_order(order_id: str) -> dict:
    """Fetch order details (including customer_id, amount, and status) by order_id (e.g. ORD-500)."""
    r = httpx.get(f"{DOWNSTREAM_API_BASE}/orders/{order_id}", timeout=10)
    r.raise_for_status()
    return r.json()

@tool
def get_customer(customer_id: str) -> dict:
    """Fetch customer profile, account tier (standard/enterprise), and status by customer_id (e.g. C-101)."""
    r = httpx.get(f"{DOWNSTREAM_API_BASE}/customers/{customer_id}", timeout=10)
    r.raise_for_status()
    return r.json()

@tool
def create_support_ticket(customer_id: str, priority: str, summary: str) -> dict:
    """Create a support ticket. priority must be exactly one of: 'low', 'medium', 'high'."""
    r = httpx.post(
        f"{DOWNSTREAM_API_BASE}/tickets",
        json={"customer_id": customer_id, "priority": priority, "summary": summary},
        timeout=10,
    )
    r.raise_for_status()
    return r.json()

tools = [get_order, get_customer, create_support_ticket]

# --- 2. Connect ChatOpenAI to local llama-server ---
llm = ChatOpenAI(
    base_url=LLM_BASE,
    api_key="not-needed",
    model="qwen3.8-27b",
    temperature=0.1,
)
llm_with_tools = llm.bind_tools(tools)

SYSTEM_PROMPT = SystemMessage(
    content=(
        "You are an operations assistant with API tools. "
        "Always verify order and customer details via tools before creating tickets. "
        "For enterprise customers with delayed orders, set ticket priority to 'high'. "
        "Summarize all tool results concisely."
    )
)

# --- 3. Build LangGraph StateGraph ---
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]

def agent_node(state: AgentState):
    msgs = [SYSTEM_PROMPT] + state["messages"]
    response = llm_with_tools.invoke(msgs)
    return {"messages": [response]}

builder = StateGraph(AgentState)
builder.add_node("agent", agent_node)
builder.add_node("tools", ToolNode(tools, handle_tool_errors=True))
builder.add_edge(START, "agent")
builder.add_conditional_edges("agent", tools_condition)
builder.add_edge("tools", "agent")

graph = builder.compile(checkpointer=MemorySaver())

# --- 4. Expose via FastAPI (custom /workflow + OpenAI /v1/chat/completions) ---
app = FastAPI(title="LangGraph + Qwen3.8-27B Orchestrator")

class WorkflowReq(BaseModel):
    query: str
    thread_id: str = "default"
    recursion_limit: int = 12

class ChatMsg(BaseModel):
    role: str
    content: str

class OpenAIChatReq(BaseModel):
    model: str = "qwen3.8-27b-agent"
    messages: List[ChatMsg]
    user: Optional[str] = "default"

@app.get("/health")
def health():
    return {"status": "ok", "llm_base": LLM_BASE, "api_base": DOWNSTREAM_API_BASE}

@app.post("/workflow")
async def run_workflow(req: WorkflowReq):
    out = await graph.ainvoke(
        {"messages": [("user", req.query)]},
        config={"configurable": {"thread_id": req.thread_id}, "recursion_limit": req.recursion_limit},
    )
    last_human_idx = max((i for i, m in enumerate(out["messages"]) if m.type == "human"), default=0)
    tool_trace = []
    for m in out["messages"][last_human_idx:]:
        if getattr(m, "tool_calls", None):
            tool_trace.append({"called": [tc["name"] for tc in m.tool_calls]})
        elif m.type == "tool":
            tool_trace.append({"tool": m.name, "output": m.content})
    return {
        "reply": out["messages"][-1].content,
        "steps": len(tool_trace),
        "trace": tool_trace,
    }

@app.get("/v1/models")
def list_models():
    return {"object": "list", "data": [{"id": "qwen3.8-27b-agent", "object": "model"}]}

@app.post("/v1/chat/completions")
async def openai_chat(req: OpenAIChatReq):
    thread_id = req.user or "openai-default"
    last_user = next((m.content for m in reversed(req.messages) if m.role == "user"), "")
    out = await graph.ainvoke(
        {"messages": [("user", last_user)]},
        config={"configurable": {"thread_id": thread_id}, "recursion_limit": 12},
    )
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": req.model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": out["messages"][-1].content},
                "finish_reason": "stop",
            }
        ],
    }