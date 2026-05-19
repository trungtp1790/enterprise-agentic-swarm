"""Retry & rate-limit helpers cho LLM (Gemini free tier hay trả 429)."""
import os
import re
import time
from typing import Any, Callable, TypeVar

T = TypeVar("T")


def _env(name: str, default: str = "") -> str:
    return (os.getenv(name) or default).strip()


def is_rate_limit_error(exc: BaseException) -> bool:
    text = str(exc).lower()
    markers = (
        "429",
        "resource_exhausted",
        "resource exhausted",
        "rate limit",
        "quota",
        "too many requests",
        "insufficient_quota",
    )
    return any(m in text for m in markers)


def is_model_not_found_error(exc: BaseException) -> bool:
    text = str(exc).lower()
    return (
        "404" in text
        or "not_found" in text
        or "is not found" in text
        or "not supported for generatecontent" in text
    )


def should_try_fallback_model(exc: BaseException) -> bool:
    return is_rate_limit_error(exc) or is_model_not_found_error(exc)


def parse_retry_seconds(exc: BaseException, default: float = 30.0) -> float:
    text = str(exc)
    m = re.search(r"retry in ([\d.]+)s", text, re.I)
    if m:
        return float(m.group(1)) + 1.0
    m = re.search(r"RetryDelay.*?seconds:\s*([\d.]+)", text, re.I | re.S)
    if m:
        return float(m.group(1)) + 1.0
    return default


def invoke_with_retry(
    fn: Callable[[], T],
    *,
    label: str = "LLM",
    max_retries: int | None = None,
) -> T:
    retries = max_retries if max_retries is not None else int(_env("AMR_LLM_MAX_RETRIES", "5"))
    base_delay = float(_env("AMR_LLM_RETRY_BASE_SEC", "2"))
    last_exc: BaseException | None = None

    for attempt in range(retries + 1):
        try:
            return fn()
        except Exception as exc:
            last_exc = exc
            if is_model_not_found_error(exc):
                raise
            if not is_rate_limit_error(exc) or attempt >= retries:
                raise
            wait = max(parse_retry_seconds(exc), base_delay * (2**attempt))
            print(f"[{label}] Rate limit (429) — đợi {wait:.0f}s rồi thử lại ({attempt + 1}/{retries})...")
            time.sleep(wait)

    raise last_exc  # type: ignore[misc]


def llm_pause() -> None:
    """Nghỉ ngắn giữa các node để tránh vượt RPM free tier."""
    delay = float(_env("AMR_LLM_DELAY_SEC", "3"))
    if delay > 0:
        time.sleep(delay)
