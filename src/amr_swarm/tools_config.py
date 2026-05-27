import os
import re
import time
from typing import Any, List

from langchain_core.tools import tool
from langchain_experimental.tools import PythonREPLTool

from amr_swarm.paths import CHARTS_DIR, DEFAULT_CHART_PATH

CHARTS_DIR.mkdir(parents=True, exist_ok=True)

SEARCH_FAILED_PREFIX = "[SEARCH_FAILED]"
_python_repl = PythonREPLTool()

_SEARCH_FAIL_MARKERS = (
    "connectionerror",
    "max retries exceeded",
    "getaddrinfo failed",
    "failed to resolve",
    "lỗi tavily",
    "chưa có tavily_api_key",
    SEARCH_FAILED_PREFIX.lower(),
)


def is_search_failure(text: str) -> bool:
    """True nếu web_search không trả được kết quả hữu ích."""
    t = (text or "").strip().lower()
    if not t or t.startswith(SEARCH_FAILED_PREFIX.lower()):
        return True
    if t.startswith("[{") and "connectionerror" in t:
        return True
    return any(m in t for m in _SEARCH_FAIL_MARKERS)


def _format_tavily_results(response: Any) -> str:
    if isinstance(response, dict):
        parts: List[str] = []
        for hit in response.get("results") or []:
            title = hit.get("title") or ""
            url = hit.get("url") or ""
            content = (hit.get("content") or "")[:1200]
            parts.append(f"- {title}\n  URL: {url}\n  {content}")
        if parts:
            return "\n\n".join(parts)
        answer = (response.get("answer") or "").strip()
        if answer:
            return answer
    if isinstance(response, list):
        return str(response)
    return str(response)


def _tavily_search_with_retry(query: str) -> str:
    from tavily import TavilyClient

    api_key = (os.getenv("TAVILY_API_KEY") or "").strip()
    if not api_key:
        return (
            f"{SEARCH_FAILED_PREFIX} Chưa có TAVILY_API_KEY trong .env — "
            "không gọi được Tavily."
        )

    client = TavilyClient(api_key=api_key)
    retries = int(os.getenv("AMR_TAVILY_MAX_RETRIES", "3"))
    delay = float(os.getenv("AMR_TAVILY_RETRY_SEC", "2"))
    last_err: Exception | None = None

    for attempt in range(retries):
        try:
            response = client.search(
                query=query,
                search_depth=os.getenv("AMR_TAVILY_DEPTH", "basic"),
                max_results=int(os.getenv("AMR_TAVILY_MAX_RESULTS", "5")),
            )
            blob = _format_tavily_results(response).strip()
            if blob and not is_search_failure(blob):
                return blob
            return f"{SEARCH_FAILED_PREFIX} Tavily trả về rỗng cho truy vấn: {query}"
        except Exception as exc:
            last_err = exc
            err_s = str(exc).lower()
            transient = any(
                m in err_s
                for m in (
                    "connection",
                    "timeout",
                    "getaddrinfo",
                    "resolve",
                    "max retries",
                    "temporarily",
                )
            )
            if transient and attempt < retries - 1:
                time.sleep(delay * (attempt + 1))
                continue
            break

    hint = (
        "Kiểm tra: (1) TAVILY_API_KEY, (2) mạng/DNS tới api.tavily.com, "
        "(3) firewall/VPN. Thử: nslookup api.tavily.com"
    )
    return f"{SEARCH_FAILED_PREFIX} Lỗi Tavily: {last_err!s}. {hint}"


def _strip_code_fences(code: str) -> str:
    text = code.strip()
    text = re.sub(r"^```(?:python)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


@tool
def web_search_tool(query: str) -> str:
    """
    Tìm kiếm web thời gian thực qua Tavily.
    Đầu vào: truy vấn ngắn gọn (ví dụ: "FPT revenue Q3 2025").
    """
    try:
        return _tavily_search_with_retry(query)
    except Exception as e:
        return f"{SEARCH_FAILED_PREFIX} Lỗi Tavily: {e!s}"


@tool
def python_sandbox_tool(code: str) -> str:
    """
    Thực thi Python cục bộ (pandas/matplotlib). Trả về stdout hoặc thông báo lỗi.
    Code nên lưu figure vào thư mục charts (đường dẫn gợi ý trong system prompt Analyst).
    """
    try:
        cleaned = _strip_code_fences(code)
        if not cleaned:
            return "Không có code để chạy."
        out = _python_repl.invoke({"query": cleaned})
        out_s = out if isinstance(out, str) else str(out)
        if not out_s.strip():
            return "Code chạy xong; không có output (thêm print() nếu cần xem kết quả trung gian)."
        return out_s
    except Exception as e:
        return f"Lỗi khi chạy Python sandbox: {e!s}\nHãy sửa code và thử lại."
