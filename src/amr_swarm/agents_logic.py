import os
from typing import Any, Dict, List

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_google_genai import ChatGoogleGenerativeAI

try:
    from langchain_openai import ChatOpenAI
except ModuleNotFoundError:  # pragma: no cover
    ChatOpenAI = None  # type: ignore

from amr_swarm.llm_utils import (
    invoke_with_retry,
    is_model_not_found_error,
    is_rate_limit_error,
    llm_pause,
    should_try_fallback_model,
)
from amr_swarm.state_schema import AgentState, CriticDecision, ResearcherOutput
from amr_swarm.tools_config import (
    DEFAULT_CHART_PATH,
    SEARCH_FAILED_PREFIX,
    is_search_failure,
    python_sandbox_tool,
    web_search_tool,
)

DEFAULT_GEMINI_MODEL = "gemini-1.5-flash"
# Google đã gỡ gemini-1.5-* khỏi Generative Language API; alias chính thức còn chạy được.
GEMINI_15_FLASH_API_MODEL = "gemini-flash-latest"
_LEGACY_GEMINI_MODEL_ALIASES: Dict[str, str] = {
    "gemini-1.5-flash": GEMINI_15_FLASH_API_MODEL,
    "gemini-1.5-flash-latest": GEMINI_15_FLASH_API_MODEL,
    "gemini-1.5-flash-002": GEMINI_15_FLASH_API_MODEL,
    "gemini-1.5-flash-8b": "gemini-2.0-flash-lite",
    "gemini-1.5-pro": "gemini-2.0-flash",
    "gemini-1.5-pro-latest": "gemini-2.0-flash",
}
MAX_TOOL_ROUNDS = int(os.getenv("AMR_MAX_TOOL_ROUNDS", "2"))
_warned_gemini_aliases: set[str] = set()


def _env(name: str) -> str:
    return (os.getenv(name) or "").strip()


def _message_text(content: Any) -> str:
    """Chuẩn hóa content (Gemini hay trả list block) thành chuỗi."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for block in content:
            if isinstance(block, dict):
                if block.get("type") == "text":
                    parts.append(str(block.get("text", "")))
                elif "text" in block:
                    parts.append(str(block["text"]))
            elif block:
                parts.append(str(block))
        return "\n".join(p for p in parts if p).strip()
    return str(content)


def _gemini_chat_messages(system: str, human: str) -> List[Any]:
    """Gemini bắt buộc có ít nhất một user turn — không gửi chỉ SystemMessage."""
    human = (human or "").strip() or "Thực hiện nhiệm vụ theo hướng dẫn hệ thống."
    return [SystemMessage(content=system), HumanMessage(content=human)]


def resolve_gemini_model(model: str) -> str:
    """Chuyển tên model cũ (1.5) sang ID còn được API hỗ trợ."""
    name = model.strip()
    if not name:
        return GEMINI_15_FLASH_API_MODEL
    resolved = _LEGACY_GEMINI_MODEL_ALIASES.get(name, name)
    if resolved != name and name not in _warned_gemini_aliases:
        _warned_gemini_aliases.add(name)
        print(
            f"[Gemini] '{name}' is retired on the API; using '{resolved}' "
            f"(keep AMR_GEMINI_MODEL={name} if you prefer that label)."
        )
    return resolved


def _gemini_model_chain() -> List[str]:
    configured = _env("AMR_GEMINI_MODEL") or DEFAULT_GEMINI_MODEL
    primary = resolve_gemini_model(configured)
    extra = _env("AMR_GEMINI_FALLBACK_MODELS") or "gemini-2.0-flash,gemini-2.0-flash-lite"
    chain = [primary]
    for m in extra.split(","):
        m = resolve_gemini_model(m.strip())
        if m and m not in chain:
            chain.append(m)
    return chain


def resolve_llm_provider() -> str:
    explicit = _env("AMR_LLM_PROVIDER").lower()
    if explicit in {"gemini", "google"}:
        return "gemini"
    if explicit == "openai":
        return "openai"
    if _env("OPENAI_API_KEY") and ChatOpenAI is not None:
        return "openai"
    return "gemini"


def get_active_llm_label() -> str:
    provider = resolve_llm_provider()
    if provider == "openai":
        return f"OpenAI ({_env('AMR_OPENAI_MODEL') or 'gpt-4o-mini'})"
    configured = _env("AMR_GEMINI_MODEL") or DEFAULT_GEMINI_MODEL
    resolved = _gemini_model_chain()[0]
    if configured != resolved:
        return f"Gemini ({configured} -> {resolved})"
    return f"Gemini ({resolved})"


def _build_gemini(model: str) -> ChatGoogleGenerativeAI:
    key = _env("GOOGLE_API_KEY") or _env("GEMINI_API_KEY")
    if not key:
        raise RuntimeError(
            "Thiếu GOOGLE_API_KEY. Thêm key Gemini vào .env hoặc đặt AMR_LLM_PROVIDER=openai."
        )
    api_model = resolve_gemini_model(model)
    return ChatGoogleGenerativeAI(
        model=api_model,
        temperature=0.1,
        google_api_key=key,
    )


def _build_openai() -> ChatOpenAI:
    if ChatOpenAI is None:
        raise RuntimeError("Cài langchain-openai: pip install langchain-openai")
    key = _env("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("Thiếu OPENAI_API_KEY.")
    return ChatOpenAI(
        model=_env("AMR_OPENAI_MODEL") or "gpt-4o-mini",
        temperature=0.1,
        api_key=key,
    )


_llm = None
_llm_provider: str | None = None
_llm_model: str | None = None


def get_llm(model: str | None = None):
    global _llm, _llm_provider, _llm_model
    provider = resolve_llm_provider()
    target_model = model or (_gemini_model_chain()[0] if provider == "gemini" else "openai")
    if _llm is None or _llm_provider != provider or _llm_model != target_model:
        _llm = _build_gemini(target_model) if provider == "gemini" else _build_openai()
        _llm_provider = provider
        _llm_model = target_model
    return _llm


def _tool_loop(
    llm,
    tools: List[Any],
    system: SystemMessage,
    prior_messages: List[Any],
    tools_by_name: Dict[str, Any],
    *,
    label: str,
    max_rounds: int = MAX_TOOL_ROUNDS,
) -> List[Any]:
    msgs: List[Any] = [system, *prior_messages]
    llm_tools = llm.bind_tools(tools)
    for round_idx in range(max_rounds):
        ai: AIMessage = invoke_with_retry(
            lambda m=msgs: llm_tools.invoke(m),
            label=f"{label}:round{round_idx + 1}",
        )
        msgs.append(ai)
        calls = getattr(ai, "tool_calls", None) or []
        if not calls:
            return msgs
        for tc in calls:
            name = tc.get("name")
            tid = tc.get("id") or "tool-call"
            args = tc.get("args") or {}
            tool_fn = tools_by_name.get(name)
            if tool_fn is None:
                out = f"Không tìm thấy tool: {name}"
            else:
                try:
                    out = tool_fn.invoke(args)
                except Exception as e:
                    out = f"Lỗi thực thi tool {name}: {e!s}"
            msgs.append(ToolMessage(content=str(out), tool_call_id=str(tid), name=str(name)))
    return msgs


def _last_ai_message_text(state: AgentState) -> str:
    for m in reversed(state["messages"]):
        if isinstance(m, AIMessage) and m.content:
            return _message_text(m.content)
    return ""


def _last_human_text(state: AgentState) -> str:
    for m in reversed(state["messages"]):
        if isinstance(m, HumanMessage):
            return str(m.content)
    return ""


def _run_with_model_fallback(fn, *, label: str):
    """Thử lần lượt các model Gemini khi 429 hoặc model không tồn tại (404)."""
    if resolve_llm_provider() != "gemini":
        return invoke_with_retry(fn, label=label)

    last_err: BaseException | None = None
    for model_name in _gemini_model_chain():
        try:
            return invoke_with_retry(lambda m=model_name: fn(m), label=f"{label}:{model_name}")
        except Exception as exc:
            last_err = exc
            if should_try_fallback_model(exc):
                reason = "404 NOT_FOUND" if is_model_not_found_error(exc) else "429 quota"
                print(f"[{label}] Model {model_name} ({reason}) — thử model dự phòng...")
                continue
            raise
    raise last_err  # type: ignore[misc]


def _structured_output(schema, messages, *, label: str):
    def _call(model_name: str):
        llm = get_llm(model_name)
        return llm.with_structured_output(schema).invoke(messages)

    if resolve_llm_provider() == "gemini":
        return _run_with_model_fallback(_call, label=label)
    return invoke_with_retry(lambda: _call(None), label=label)  # type: ignore[arg-type]


def researcher_node(state: AgentState):
    llm_pause()
    query = _last_human_text(state)
    search_blob = web_search_tool.invoke({"query": query})

    if is_search_failure(search_blob):
        return {
            "raw_data": (
                f"{SEARCH_FAILED_PREFIX}\n"
                f"Truy vấn: {query}\n"
                f"{search_blob}\n\n"
                "Không có dữ liệu thị trường để tóm tắt. Vui lòng sửa mạng/Tavily và chạy lại."
            ),
            "sender": "researcher",
        }

    pack: ResearcherOutput = _structured_output(
        ResearcherOutput,
        _gemini_chat_messages(
            (
                "Bạn là Researcher thị trường. Chỉ tóm tắt khách quan kết quả tìm kiếm. "
                "Không bịa số liệu. Không khuyến nghị mua/bán. "
                "Nếu kết quả tìm kiếm báo lỗi hoặc rỗng, ghi rõ không có dữ liệu."
            ),
            (
                f"Yêu cầu: {query}\n\n"
                f"Kết quả web_search:\n{search_blob}\n\n"
                "Trả về summary và sources (URL nếu có)."
            ),
        ),
        label="researcher",
    )
    summary_block = pack.summary.strip()
    if pack.sources:
        summary_block += "\n\nNguồn: " + "; ".join(pack.sources)
    return {"raw_data": summary_block, "sender": "researcher"}


def analyst_node(state: AgentState):
    llm_pause()
    chart_hint = str(DEFAULT_CHART_PATH).replace("\\", "/")
    raw_data = state.get("raw_data", "") or ""

    if is_search_failure(raw_data):
        return {
            "messages": [
                AIMessage(
                    content=(
                        "Không thể phân tích hoặc vẽ biểu đồ: Researcher chưa có dữ liệu thị trường "
                        "(lỗi Tavily / mạng). Sửa kết nối rồi bấm **Luồng thread mới** và chạy lại."
                    ),
                    name="analyst",
                )
            ],
            "sender": "analyst",
            "chart_path": "",
        }

    analyst_msgs = _gemini_chat_messages(
        (
            "Bạn là Analyst dữ liệu. Đọc raw_data từ Researcher trong tin nhắn người dùng.\n"
            "Nhiệm vụ: phân tích logic, dùng python_sandbox_tool TỐI ĐA 1 lần để vẽ biểu đồ "
            f"(matplotlib). Lưu PNG tại: {chart_hint}\n"
            "Trước plt.savefig: plt.figure(). Trả lời ngắn: ý nghĩa biểu đồ và số liệu chính.\n"
            "KHÔNG phân tích thông điệp lỗi API/mạng như số liệu tài chính."
        ),
        f"RAW_DATA:\n{raw_data}\n\nHãy phân tích và vẽ biểu đồ theo hướng dẫn.",
    )
    tools_map = {python_sandbox_tool.name: python_sandbox_tool}

    def _run_analyst(model_name: str):
        llm = get_llm(model_name)
        return _tool_loop(
            llm,
            [python_sandbox_tool],
            analyst_msgs[0],
            analyst_msgs[1:],
            tools_map,
            label=f"analyst:{model_name}",
            max_rounds=2,
        )

    if resolve_llm_provider() == "gemini":
        msgs = _run_with_model_fallback(_run_analyst, label="analyst")
    else:
        msgs = invoke_with_retry(lambda: _run_analyst(None), label="analyst")  # type: ignore[arg-type]

    final_ai = msgs[-1]
    if not isinstance(final_ai, AIMessage):
        final_ai = AIMessage(content=str(final_ai))
    return {
        "messages": [AIMessage(content=_message_text(final_ai.content), name="analyst")],
        "sender": "analyst",
        "chart_path": str(DEFAULT_CHART_PATH),
    }


def human_review_node(state: AgentState):
    from langgraph.types import Command, interrupt

    payload = {
        "kind": "pre_critic_review",
        "chart_path": state.get("chart_path") or "",
        "analyst_preview": _last_ai_message_text(state)[:6000],
    }
    decision = interrupt(payload)
    if not isinstance(decision, dict):
        decision = {"action": "approve"}
    action = str(decision.get("action", "approve")).lower()
    if action == "reject":
        fb = (decision.get("feedback") or "").strip() or "Người duyệt yêu cầu làm lại."
        return Command(
            update={
                "messages": [
                    HumanMessage(
                        content=f"[HITL — trước Critic] Phản hồi người duyệt: {fb}"
                    )
                ],
                "revision_count": int(state.get("revision_count") or 0) + 1,
                "sender": "human_review",
            },
            goto="researcher",
        )
    return {"sender": "human_review"}


def critic_node(state: AgentState):
    llm_pause()
    rev_count = int(state.get("revision_count") or 0)
    decision: CriticDecision = _structured_output(
        CriticDecision,
        _gemini_chat_messages(
            (
                "Bạn là Critic khó tính. Không dùng tool.\n"
                "Nếu revision_count >= 3 thì BẮT BUỘC APPROVE.\n"
                "Trả về APPROVE hoặc REJECT + feedback."
            ),
            (
                f"Dữ liệu thô (Researcher):\n{state.get('raw_data', '')}\n\n"
                f"Bản phân tích (Analyst):\n{_last_ai_message_text(state)}\n\n"
                f"Số lần revision: {rev_count}\n\n"
                "Hãy đánh giá và trả về quyết định."
            ),
        ),
        label="critic",
    )

    rev = int(state.get("revision_count") or 0)
    if decision.decision == "REJECT":
        rev += 1

    return {
        "status": decision.decision,
        "revision_count": rev,
        "sender": "critic",
        "messages": [
            HumanMessage(
                content=f"[Critic] {decision.decision}. Ghi chú: {decision.feedback}"
            )
        ],
    }
