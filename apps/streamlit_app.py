import os
import sys
import uuid
from pathlib import Path
from typing import Any, Dict

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import httpx
import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langgraph.types import Command

load_dotenv(_ROOT / ".env")

USE_API = os.getenv("AMR_USE_API", "0").strip() in {"1", "true", "yes"}
API_BASE = os.getenv("AMR_API_BASE", "http://127.0.0.1:8000").rstrip("/")


@st.cache_resource
def _local_graph():
    from amr_swarm.graph_builder import get_app_graph

    return get_app_graph()


def _thread_config():
    if "thread_id" not in st.session_state:
        st.session_state.thread_id = str(uuid.uuid4())
    return {"configurable": {"thread_id": st.session_state.thread_id}}


def _render_stream_chunk(output: Dict[str, Any], status) -> None:
    for node_name, state_update in output.items():
        if node_name == "__interrupt__":
            continue
        status.write(f"**{node_name}** đã cập nhật state.")
        if isinstance(state_update, dict) and state_update.get("chart_path") and node_name == "analyst":
            p = state_update["chart_path"]
            if os.path.isfile(p):
                st.image(p, caption="Biểu đồ (Analyst)")


def _run_local_stream(graph, payload: Any, config: Dict[str, Any], status) -> None:
    for output in graph.stream(payload, config=config):
        _render_stream_chunk(output, status)


def _api_start(query: str, thread_id: str) -> Dict[str, Any]:
    with httpx.Client(timeout=600.0) as client:
        r = client.post(
            f"{API_BASE}/runs/start",
            json={"query": query, "thread_id": thread_id},
        )
        r.raise_for_status()
        return r.json()


def _api_resume(thread_id: str, action: str, feedback: str) -> Dict[str, Any]:
    with httpx.Client(timeout=600.0) as client:
        r = client.post(
            f"{API_BASE}/runs/{thread_id}/resume",
            json={"action": action, "feedback": feedback},
        )
        r.raise_for_status()
        return r.json()


st.set_page_config(page_title="Auto-Market-Research Swarm", layout="wide")
st.title("Auto-Market-Research Swarm")
st.caption("Researcher → Analyst → HITL → Critic (LangGraph + Streamlit)")

with st.sidebar:
    st.header("Cấu hình")
    user_query = st.text_area(
        "Mã cổ phiếu / ngành / câu hỏi:",
        placeholder="Ví dụ: Phân tích nhanh cổ phiếu VNM",
        height=100,
    )
    if st.button("Luồng thread mới", help="Thread mới cho checkpoint"):
        st.session_state.thread_id = str(uuid.uuid4())
        st.success(f"thread_id mới: {st.session_state.thread_id}")
    start_btn = st.button("Chạy phân tích", type="primary", use_container_width=True)
    st.divider()
    try:
        from amr_swarm.agents_logic import get_active_llm_label

        st.caption(f"LLM đang dùng: **{get_active_llm_label()}**")
    except Exception as exc:
        st.warning(str(exc))
    st.markdown(
        "`AMR_USE_API=1`: gọi FastAPI (`uvicorn apps.api:app`). "
        "Mặc định chạy graph trong process Streamlit."
    )

config = _thread_config()

if start_btn and user_query.strip():
    with st.status("Đang chạy graph...", expanded=True) as status:
        try:
            if USE_API:
                data = _api_start(user_query.strip(), st.session_state.thread_id)
                status.write("Đã gọi API /runs/start")
                for n in data.get("nodes_touched") or []:
                    status.write(f"**{n}**")
            else:
                graph = _local_graph()
                _run_local_stream(
                    graph,
                    {"messages": [HumanMessage(content=user_query.strip())]},
                    config,
                    status,
                )
        except Exception as e:
            err = str(e)
            status.update(label=f"Lỗi: {e}", state="error")
            if (
                "insufficient_quota" in err
                or "Error code: 429" in err
                or "RESOURCE_EXHAUSTED" in err
            ):
                st.error(
                    "**Hết quota / rate limit (429)** — API tạm chặn do free tier.\n\n"
                    "Đợi ~1 phút rồi chạy lại; hoặc đổi `AMR_GEMINI_MODEL` trong `.env`."
                )
            elif "404" in err and "NOT_FOUND" in err:
                st.error(
                    "**Model Gemini không tồn tại (404).**\n\n"
                    "Trong `.env` dùng tên hợp lệ, ví dụ:\n"
                    "`AMR_GEMINI_MODEL=gemini-2.0-flash-lite`\n"
                    "`AMR_GEMINI_FALLBACK_MODELS=gemini-2.0-flash`\n\n"
                    "Tránh `gemini-1.5-flash` (không còn trên API v1beta)."
                )
            elif "contents are required" in err.lower():
                st.error(
                    "**Gemini từ chối request thiếu tin nhắn người dùng.** "
                    "Hãy **restart Streamlit** (Ctrl+C rồi chạy lại)."
                )
            else:
                st.error(err)
        else:
            status.update(label="Tạm dừng hoặc hoàn tất bước hiện tại", state="complete")

interrupts: tuple = ()
values: Dict[str, Any] = {}

if USE_API:
    try:
        with httpx.Client(timeout=30.0) as client:
            r = client.get(f"{API_BASE}/runs/{st.session_state.thread_id}/state")
            if r.status_code == 200:
                data = r.json()
                interrupts = tuple(data.get("interrupts") or [])
                values = data.get("values") or {}
    except Exception:
        pass
else:
    graph = _local_graph()
    snap = graph.get_state(config)
    interrupts = snap.interrupts
    values = snap.values if isinstance(snap.values, dict) else {}

if interrupts:
    st.warning("Hệ thống đang chờ duyệt (trước khi chạy Critic).")
    raw_preview = (values.get("raw_data") or "") if isinstance(values, dict) else ""
    draft_preview = ""
    if isinstance(values, dict) and values.get("messages"):
        msgs = values.get("messages") or []
        if msgs:
            draft_preview = str(getattr(msgs[-1], "content", "") or "")
    combined = f"{raw_preview}\n{draft_preview}".lower()
    if "[search_failed]" in combined or "connectionerror" in combined or "lỗi tavily" in combined:
        st.error(
            "**Chưa có dữ liệu thị trường thật** — Tavily lỗi mạng/DNS (`api.tavily.com`).\n\n"
            "Nên bấm **Yêu cầu sửa (Reject)** hoặc **Luồng thread mới**, sửa mạng/key, rồi chạy lại. "
            "Không nên **Duyệt** nếu Analyst chỉ mô tả lỗi kết nối."
        )
    intr0 = interrupts[0]
    intr_val = getattr(intr0, "value", intr0)
    if isinstance(intr_val, dict):
        with st.expander("Payload interrupt (preview Analyst)", expanded=False):
            st.json(intr_val)

    draft = ""
    if isinstance(values, dict):
        draft = values.get("last_message") or ""
        if not draft and values.get("messages"):
            msgs = values.get("messages") or []
            if msgs:
                draft = str(getattr(msgs[-1], "content", "") or "")

    if draft:
        with st.expander("Bản nháp Analyst (text)", expanded=True):
            st.markdown(draft[:12000])

    c1, c2 = st.columns(2)
    with c1:
        if st.button("Duyệt (Approve) — chạy Critic"):
            with st.spinner("Đang resume..."):
                try:
                    if USE_API:
                        _api_resume(st.session_state.thread_id, "approve", "")
                    else:
                        graph = _local_graph()
                        _run_local_stream(
                            graph, Command(resume={"action": "approve"}), config, st.empty()
                        )
                    st.success("Đã gửi approve.")
                    st.rerun()
                except Exception as e:
                    st.error(str(e))
    with c2:
        fb = st.text_input("Feedback khi từ chối", key="hitl_feedback")
        if st.button("Yêu cầu sửa (Reject)"):
            with st.spinner("Đang resume..."):
                try:
                    if USE_API:
                        _api_resume(st.session_state.thread_id, "reject", fb)
                    else:
                        graph = _local_graph()
                        _run_local_stream(
                            graph,
                            Command(resume={"action": "reject", "feedback": fb}),
                            config,
                            st.empty(),
                        )
                    st.success("Đã gửi reject + feedback.")
                    st.rerun()
                except Exception as e:
                    st.error(str(e))

if not interrupts and isinstance(values, dict) and (
    values.get("last_message") or values.get("messages") or values.get("raw_data")
):
    st.subheader("Báo cáo / trạng thái cuối")
    status = (values.get("status") or "").strip().upper()
    if status:
        st.info(f"Critic: **{status}**")
    raw = values.get("raw_data") or ""
    if raw:
        st.markdown("### Dữ liệu tóm tắt (Researcher)")
        st.markdown(raw[:20000])
    chart_path = values.get("chart_path")
    if chart_path and os.path.isfile(chart_path):
        st.image(chart_path, caption="Biểu đồ đính kèm")
