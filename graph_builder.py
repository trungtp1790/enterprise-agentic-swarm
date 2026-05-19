import os
import sqlite3
import warnings
from pathlib import Path

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph

try:
    from langgraph.checkpoint.sqlite import SqliteSaver
except ModuleNotFoundError:  # LangGraph cũ hoặc bản tối giản không gói sqlite
    SqliteSaver = None  # type: ignore[misc, assignment]

from agents_logic import analyst_node, critic_node, human_review_node, researcher_node
from state_schema import AgentState

_ROOT = Path(__file__).resolve().parent
_DATA = _ROOT / "data"
_DEFAULT_SQLITE = _DATA / "checkpoints.db"


def make_checkpointer():
    mode = os.getenv("AMR_CHECKPOINTER", "memory").strip().lower()
    if mode == "sqlite":
        if SqliteSaver is None:
            warnings.warn(
                "Không import được langgraph.checkpoint.sqlite (langgraph quá cũ hoặc thiếu extra). "
                "Chạy: pip install -U 'langgraph>=1.0' hoặc dùng .venv của project. "
                "Tạm thời dùng InMemorySaver.",
                stacklevel=2,
            )
            return InMemorySaver()
        _DATA.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(_DEFAULT_SQLITE), check_same_thread=False)
        return SqliteSaver(conn)
    return InMemorySaver()


def review_routing(state: AgentState) -> str:
    status = (state.get("status") or "").strip().upper()
    if status == "APPROVE":
        return "end"
    return "retry"


def build_workflow() -> StateGraph:
    workflow = StateGraph(AgentState)
    workflow.add_node("researcher", researcher_node)
    workflow.add_node("analyst", analyst_node)
    workflow.add_node("human_review", human_review_node)
    workflow.add_node("critic", critic_node)

    workflow.add_edge(START, "researcher")
    workflow.add_edge("researcher", "analyst")
    workflow.add_edge("analyst", "human_review")
    workflow.add_edge("human_review", "critic")
    workflow.add_conditional_edges(
        "critic",
        review_routing,
        {"end": END, "retry": "researcher"},
    )
    return workflow


def compile_app(checkpointer=None):
    """
    Biên dịch graph với checkpointer.
    HITL: dùng interrupt() trong node human_review (tương thích Command(resume=...)).
    Module 4 gợi ý interrupt_before=['critic']; ở đây tách bước human_review + interrupt
    để Streamlit có thể gửi approve/reject có cấu trúc trước khi chạy Critic.
    """
    cp = checkpointer or make_checkpointer()
    return build_workflow().compile(checkpointer=cp)


_checkpointer_singleton = None
_compiled_singleton = None


def get_app_graph():
    """Singleton đơn giản cho Streamlit / API (một process = một checkpointer)."""
    from dotenv import load_dotenv

    load_dotenv()
    global _checkpointer_singleton, _compiled_singleton
    if _compiled_singleton is None:
        _checkpointer_singleton = make_checkpointer()
        _compiled_singleton = compile_app(_checkpointer_singleton)
    return _compiled_singleton
