"""Thin, testable wrapper around the Google Gen AI SDK.

Responsibilities:
* bounded retries with exponential backoff + full jitter for transient errors
  (HTTP 429/500/502/503/504, timeouts, connection errors);
* no retries for permanent errors (400/401/403/404: bad request, bad key,
  unknown model), which are raised immediately;
* timing each call with a monotonic clock;
* extracting token usage from ``response.usage_metadata`` (None when the API
  does not report it - counts are never invented).

Anything implementing ``generate(prompt, *, model, json_schema=None)`` and
returning an ``LLMResponse`` can stand in for this client in tests.
"""

from __future__ import annotations

import random
import re
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

from app import config
from app.errors import ConfigurationError, EmptyModelResponseError, ExternalServiceError

TRANSIENT_HTTP_CODES = {408, 429, 500, 502, 503, 504}


@dataclass
class TokenUsage:
    input_tokens: int | None = None
    output_tokens: int | None = None  # visible answer tokens
    thinking_tokens: int | None = None  # billed at the output rate
    total_tokens: int | None = None

    @property
    def billable_output_tokens(self) -> int | None:
        if self.output_tokens is None and self.thinking_tokens is None:
            return None
        return (self.output_tokens or 0) + (self.thinking_tokens or 0)

    def to_dict(self) -> dict:
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "thinking_tokens": self.thinking_tokens,
            "total_tokens": self.total_tokens,
        }


@dataclass
class LLMResponse:
    text: str
    model: str
    latency_ms: float  # duration of the successful attempt only
    total_latency_ms: float  # including failed attempts and backoff waits
    attempts: int
    usage: TokenUsage | None = None
    retry_errors: list[str] = field(default_factory=list)
    finish_reason: str | None = None  # e.g. "STOP", "MAX_TOKENS", "SAFETY"; None if not reported


class LLMClient(Protocol):
    def generate(self, prompt: str, *, model: str, json_schema: dict | None = None) -> LLMResponse: ...


def backoff_delay(attempt: int, base: float, maximum: float, rng: Callable[[], float] = random.random) -> float:
    """Full-jitter exponential backoff: uniform(0, min(max, base * 2**attempt))."""
    return rng() * min(maximum, base * (2**attempt))


def _parse_retry_delay(details: Any) -> float | None:
    """Read google.rpc.RetryInfo.retryDelay (e.g. "17s") from an error payload."""
    try:
        for item in details.get("error", {}).get("details", []):
            delay = item.get("retryDelay")
            if isinstance(delay, str):
                match = re.fullmatch(r"(\d+(?:\.\d+)?)s", delay)
                if match:
                    return float(match.group(1))
    except AttributeError:
        return None
    return None


def _transport_error_types() -> tuple[type[BaseException], ...]:
    types: list[type[BaseException]] = [TimeoutError, ConnectionError]
    for module_name in ("httpx", "httpx2"):
        try:
            module = __import__(module_name)
            types.append(module.TransportError)
        except (ImportError, AttributeError):
            continue
    return tuple(types)


def extract_finish_reason(response: Any) -> str | None:
    candidates = getattr(response, "candidates", None) or []
    if not candidates:
        return None
    reason = getattr(candidates[0], "finish_reason", None)
    if reason is None:
        return None
    return getattr(reason, "name", None) or str(reason)


def extract_usage(response: Any) -> TokenUsage | None:
    meta = getattr(response, "usage_metadata", None)
    if meta is None:
        return None
    usage = TokenUsage(
        input_tokens=getattr(meta, "prompt_token_count", None),
        output_tokens=getattr(meta, "candidates_token_count", None),
        thinking_tokens=getattr(meta, "thoughts_token_count", None),
        total_tokens=getattr(meta, "total_token_count", None),
    )
    if all(v is None for v in usage.to_dict().values()):
        return None
    return usage


class GeminiClient:
    def __init__(
        self,
        api_key: str | None = None,
        *,
        max_retries: int | None = None,
        base_delay: float | None = None,
        max_delay: float | None = None,
        timeout_s: float | None = None,
        min_interval_s: float | None = None,
        sdk_client: Any = None,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.perf_counter,
        rng: Callable[[], float] = random.random,
    ):
        self.max_retries = config.MAX_RETRIES if max_retries is None else max_retries
        self.base_delay = config.RETRY_BASE_DELAY_S if base_delay is None else base_delay
        self.max_delay = config.RETRY_MAX_DELAY_S if max_delay is None else max_delay
        self.min_interval_s = config.MIN_REQUEST_INTERVAL_S if min_interval_s is None else min_interval_s
        self._sleep, self._clock, self._rng = sleep, clock, rng
        self._last_request_at: float | None = None
        self._lock = threading.Lock()

        if sdk_client is not None:
            self._sdk = sdk_client
        else:
            key = api_key or config.GEMINI_API_KEY
            if not key:
                raise ConfigurationError("GEMINI_API_KEY is not set.")
            from google import genai
            from google.genai import types

            timeout_ms = int((config.REQUEST_TIMEOUT_S if timeout_s is None else timeout_s) * 1000)
            self._sdk = genai.Client(api_key=key, http_options=types.HttpOptions(timeout=timeout_ms))

    def _throttle(self) -> None:
        if self.min_interval_s <= 0:
            return
        with self._lock:
            now = self._clock()
            if self._last_request_at is not None:
                wait = self.min_interval_s - (now - self._last_request_at)
                if wait > 0:
                    self._sleep(wait)
            self._last_request_at = self._clock()

    def _build_config(self, json_schema: dict | None) -> Any:
        if json_schema is None:
            return None
        from google.genai import types

        return types.GenerateContentConfig(response_mime_type="application/json", response_json_schema=json_schema)

    def generate(self, prompt: str, *, model: str, json_schema: dict | None = None) -> LLMResponse:
        from google.genai import errors as genai_errors

        transport_errors = _transport_error_types()
        started = self._clock()
        retry_errors: list[str] = []
        gen_config = self._build_config(json_schema)

        for attempt in range(self.max_retries + 1):
            self._throttle()
            attempt_start = self._clock()
            try:
                kwargs = {"model": model, "contents": prompt}
                if gen_config is not None:
                    kwargs["config"] = gen_config
                response = self._sdk.models.generate_content(**kwargs)
            except genai_errors.APIError as exc:
                code = getattr(exc, "code", None)
                status = getattr(exc, "status", None)
                summary = f"HTTP {code} {status}"
                if code not in TRANSIENT_HTTP_CODES:
                    hint = {
                        400: "invalid request or invalid API key",
                        401: "authentication failed - check GEMINI_API_KEY",
                        403: "permission denied - check GEMINI_API_KEY and API access",
                        404: f"model {model!r} not found - check GEMINI_MODEL / JUDGE_MODEL",
                    }.get(code, "non-retryable API error")
                    raise ExternalServiceError(
                        f"{summary}: {hint}. {getattr(exc, 'message', '')}".strip(),
                        transient=False,
                        status_code=code,
                        status=status,
                        attempts=attempt + 1,
                    ) from exc
                retry_errors.append(summary)
                if attempt >= self.max_retries:
                    raise ExternalServiceError(
                        f"{summary} after {attempt + 1} attempt(s): {getattr(exc, 'message', exc)}",
                        transient=True,
                        status_code=code,
                        status=status,
                        attempts=attempt + 1,
                    ) from exc
                delay = backoff_delay(attempt, self.base_delay, self.max_delay, self._rng)
                server_delay = _parse_retry_delay(getattr(exc, "details", None))
                if server_delay is not None:
                    delay = max(delay, min(server_delay, self.max_delay))
                self._sleep(delay)
                continue
            except (genai_errors.UnknownApiResponseError, *transport_errors) as exc:
                # Network failures and unparseable responses (e.g. a proxy's HTML
                # error page) are external problems that may succeed on retry.
                summary = f"{type(exc).__name__}: {exc}"
                retry_errors.append(summary)
                if attempt >= self.max_retries:
                    raise ExternalServiceError(
                        f"Network error after {attempt + 1} attempt(s): {summary}",
                        transient=True,
                        attempts=attempt + 1,
                    ) from exc
                self._sleep(backoff_delay(attempt, self.base_delay, self.max_delay, self._rng))
                continue

            finished = self._clock()
            text = getattr(response, "text", None)
            if not text or not text.strip():
                raise EmptyModelResponseError(
                    "Model returned an empty response (possibly blocked by safety filters).",
                    attempts=attempt + 1,
                )
            return LLMResponse(
                text=text,
                model=model,
                latency_ms=(finished - attempt_start) * 1000,
                total_latency_ms=(finished - started) * 1000,
                attempts=attempt + 1,
                usage=extract_usage(response),
                retry_errors=retry_errors,
                finish_reason=extract_finish_reason(response),
            )

        raise AssertionError("unreachable")  # pragma: no cover


_default_client: LLMClient | None = None


def get_llm_client() -> LLMClient:
    global _default_client
    if _default_client is None:
        _default_client = GeminiClient()
    return _default_client


def set_llm_client(client: LLMClient | None) -> None:
    global _default_client
    _default_client = client
