import os
import uuid
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from langchain_core.messages import HumanMessage
from langgraph.types import Command
from pydantic import BaseModel, Field

load_dotenv()

_graph = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _graph
    from graph_builder import compile_app, make_checkpointer

    _graph = compile_app(make_checkpointer())
    yield


app = FastAPI(title="Auto-Market-Research Swarm", lifespan=lifespan)


class StartRunBody(BaseModel):
    query: str = Field(..., description="Mã CP hoặc câu hỏi nghiên cứu")
    thread_id: Optional[str] = None


class ResumeBody(BaseModel):
    action: str = Field(..., description="approve hoặc reject")
    feedback: str = ""


def _cfg(thread_id: str) -> Dict[str, Any]:
    return {"configurable": {"thread_id": thread_id}}


def _json_safe_values(vals: Any) -> Dict[str, Any]:
    if not isinstance(vals, dict):
        return {}
    msgs = vals.get("messages") or []
    last = ""
    if msgs:
        last = str(getattr(msgs[-1], "content", "") or "")
    return {
        "raw_data": vals.get("raw_data"),
        "chart_path": vals.get("chart_path"),
        "status": vals.get("status"),
        "revision_count": vals.get("revision_count"),
        "sender": vals.get("sender"),
        "last_message": last,
    }


@app.post("/runs/start")
def start_run(body: StartRunBody):
    if _graph is None:
        raise HTTPException(503, "Graph chưa sẵn sàng")
    tid = body.thread_id or str(uuid.uuid4())
    config = _cfg(tid)
    nodes: List[str] = []
    for chunk in _graph.stream({"messages": [HumanMessage(content=body.query)]}, config):
        nodes.extend(k for k in chunk if k != "__interrupt__")
    snap = _graph.get_state(config)
    return {
        "thread_id": tid,
        "nodes_touched": nodes,
        "pending_interrupts": [getattr(i, "value", i) for i in snap.interrupts],
        "next": snap.next,
        "values": _json_safe_values(snap.values),
    }


@app.get("/runs/{thread_id}/state")
def run_state(thread_id: str):
    if _graph is None:
        raise HTTPException(503, "Graph chưa sẵn sàng")
    snap = _graph.get_state(_cfg(thread_id))
    return {
        "values": _json_safe_values(snap.values),
        "next": snap.next,
        "interrupts": [getattr(i, "value", i) for i in snap.interrupts],
    }


@app.post("/runs/{thread_id}/resume")
def resume_run(thread_id: str, body: ResumeBody):
    if _graph is None:
        raise HTTPException(503, "Graph chưa sẵn sàng")
    action = body.action.strip().lower()
    if action not in {"approve", "reject"}:
        raise HTTPException(400, "action phải là approve hoặc reject")
    payload = {"action": action, "feedback": body.feedback}
    config = _cfg(thread_id)
    nodes: List[str] = []
    for chunk in _graph.stream(Command(resume=payload), config):
        nodes.extend(k for k in chunk if k != "__interrupt__")
    snap = _graph.get_state(config)
    return {
        "nodes_touched": nodes,
        "pending_interrupts": [getattr(i, "value", i) for i in snap.interrupts],
        "next": snap.next,
        "values": _json_safe_values(snap.values),
    }


@app.get("/health")
def health():
    return {"ok": True, "checkpointer": os.getenv("AMR_CHECKPOINTER", "memory")}
