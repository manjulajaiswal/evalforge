"""Groq client (OpenAI-compatible Chat Completions API).

Implements the same LLMClient protocol as GeminiClient, so the RAG pipeline,
judges and evaluator are unchanged. Uses httpx directly (already a dependency
of google-genai) rather than adding another SDK.

Error policy (mirrors GeminiClient):
* 429 with a short wait (per-minute limits): retried, honouring retry-after.
* 429 for a daily quota (RPD/TPD) or a wait longer than the retry budget:
  permanent "quota exhausted" error, so the run aborts immediately.
* 500/502/503/504, timeouts, connection errors: retried with backoff.
* 400 json_validate_failed (model broke JSON mode): JudgeOutputError, so the
  judge's re-ask logic handles it like any malformed judge output.
* Other 4xx (bad key, unknown model, bad request): permanent, not retried.

Endpoint: https://api.groq.com/openai/v1/chat/completions
"""

from __future__ import annotations

import random
import re
import threading
import time
from typing import Any, Callable

import httpx

from app import config
from app.errors import ConfigurationError, EmptyModelResponseError, ExternalServiceError, JudgeOutputError
from app.gemini_client import TRANSIENT_HTTP_CODES, LLMResponse, TokenUsage, backoff_delay

GROQ_CHAT_URL = "https://api.groq.com/openai/v1/chat/completions"

_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.S | re.I)


def parse_wait_seconds(text: str) -> float | None:
    """Parse Groq's 'Please try again in 7m28.8s' / '1h2m3s' / '450ms' hints."""
    match = re.search(r"try again in\s+((?:\d+(?:\.\d+)?(?:h|ms|m|s))+)", text or "")
    if not match:
        return None
    total = 0.0
    for value, unit in re.findall(r"(\d+(?:\.\d+)?)(ms|h|m|s)", match.group(1)):
        total += float(value) * {"h": 3600, "m": 60, "s": 1, "ms": 0.001}[unit]
    return total


def extract_groq_usage(payload: dict) -> TokenUsage | None:
    usage = payload.get("usage")
    if not isinstance(usage, dict):
        return None
    prompt = usage.get("prompt_tokens")
    completion = usage.get("completion_tokens")
    details = usage.get("completion_tokens_details") or {}
    reasoning = details.get("reasoning_tokens") if isinstance(details, dict) else None
    if prompt is None and completion is None:
        return None
    # completion_tokens includes reasoning tokens; split them so they are not double counted.
    visible = None if completion is None else completion - (reasoning or 0)
    return TokenUsage(
        input_tokens=prompt,
        output_tokens=visible,
        thinking_tokens=reasoning,
        total_tokens=usage.get("total_tokens"),
    )


class GroqClient:
    def __init__(
        self,
        api_key: str | None = None,
        *,
        max_retries: int | None = None,
        base_delay: float | None = None,
        max_delay: float | None = None,
        timeout_s: float | None = None,
        min_interval_s: float | None = None,
        reasoning_effort: str | None = None,
        http_client: httpx.Client | None = None,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.perf_counter,
        rng: Callable[[], float] = random.random,
    ):
        key = api_key or config.GROQ_API_KEY
        if not key:
            raise ConfigurationError("GROQ_API_KEY is not set.")
        self._headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
        self.max_retries = config.MAX_RETRIES if max_retries is None else max_retries
        self.base_delay = config.RETRY_BASE_DELAY_S if base_delay is None else base_delay
        self.max_delay = config.RETRY_MAX_DELAY_S if max_delay is None else max_delay
        self.min_interval_s = config.MIN_REQUEST_INTERVAL_S if min_interval_s is None else min_interval_s
        self.reasoning_effort = config.GROQ_REASONING_EFFORT if reasoning_effort is None else reasoning_effort
        timeout = config.REQUEST_TIMEOUT_S if timeout_s is None else timeout_s
        self._http = http_client or httpx.Client(timeout=timeout)
        self._sleep, self._clock, self._rng = sleep, clock, rng
        self._last_request_at: float | None = None
        self._lock = threading.Lock()

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

    def _body(self, prompt: str, model: str, json_schema: dict | None) -> dict:
        body: dict[str, Any] = {"model": model, "messages": [{"role": "user", "content": prompt}]}
        if json_schema is not None:
            # JSON mode; the schema itself is enforced by app/judge.py's validators.
            body["response_format"] = {"type": "json_object"}
        if self.reasoning_effort and model.startswith("openai/gpt-oss"):
            body["reasoning_effort"] = self.reasoning_effort
        return body

    def generate(self, prompt: str, *, model: str, json_schema: dict | None = None) -> LLMResponse:
        started = self._clock()
        retry_errors: list[str] = []
        body = self._body(prompt, model, json_schema)

        for attempt in range(self.max_retries + 1):
            self._throttle()
            attempt_start = self._clock()
            try:
                response = self._http.post(GROQ_CHAT_URL, headers=self._headers, json=body)
            except httpx.TransportError as exc:
                summary = f"{type(exc).__name__}: {exc}"
                retry_errors.append(summary)
                if attempt >= self.max_retries:
                    raise ExternalServiceError(
                        f"Network error after {attempt + 1} attempt(s): {summary}", transient=True, attempts=attempt + 1
                    ) from exc
                self._sleep(backoff_delay(attempt, self.base_delay, self.max_delay, self._rng))
                continue

            code = response.status_code
            if code == 200:
                finished = self._clock()
                try:
                    payload = response.json()
                    choice = payload["choices"][0]
                    text = choice["message"].get("content") or ""
                except (ValueError, KeyError, IndexError, TypeError) as exc:
                    raise ExternalServiceError(
                        f"Unexpected Groq response format: {exc}", transient=False, status_code=code, attempts=attempt + 1
                    ) from exc
                text = _THINK_BLOCK.sub("", text).strip()
                if not text:
                    raise EmptyModelResponseError("Model returned an empty response.", attempts=attempt + 1)
                finish = choice.get("finish_reason")
                return LLMResponse(
                    text=text,
                    model=model,
                    latency_ms=(finished - attempt_start) * 1000,
                    total_latency_ms=(finished - started) * 1000,
                    attempts=attempt + 1,
                    usage=extract_groq_usage(payload),
                    retry_errors=retry_errors,
                    # Normalise OpenAI-style "stop"/"length" to the names the evaluator checks.
                    finish_reason={"stop": "STOP", "length": "MAX_TOKENS"}.get(finish, finish.upper() if finish else None),
                )

            message, error_code = self._error_details(response)
            summary = f"HTTP {code}{f' {error_code}' if error_code else ''}"

            if code == 400 and error_code == "json_validate_failed":
                raise JudgeOutputError(f"Model output failed Groq JSON mode validation: {message[:200]}")

            if code == 429:
                retry_errors.append(summary)
                wait = _retry_after(response) or parse_wait_seconds(message)
                daily = bool(re.search(r"per day|\(RPD\)|\(TPD\)", message, re.I))
                if daily or (wait is not None and wait > self.max_delay):
                    hint = f" Groq asks to retry in about {wait / 60:.0f} min." if wait else ""
                    raise ExternalServiceError(
                        f"{summary}: API quota exhausted (daily or long-window limit).{hint} {message}".strip(),
                        transient=False,
                        status_code=code,
                        attempts=attempt + 1,
                    )
                if attempt >= self.max_retries:
                    raise ExternalServiceError(
                        f"{summary} after {attempt + 1} attempt(s): {message}", transient=True, status_code=code,
                        attempts=attempt + 1,
                    )
                delay = backoff_delay(attempt, self.base_delay, self.max_delay, self._rng)
                if wait is not None:
                    delay = max(delay, min(wait + 0.5, self.max_delay))
                self._sleep(delay)
                continue

            if code in TRANSIENT_HTTP_CODES:
                retry_errors.append(summary)
                if attempt >= self.max_retries:
                    raise ExternalServiceError(
                        f"{summary} after {attempt + 1} attempt(s): {message}", transient=True, status_code=code,
                        attempts=attempt + 1,
                    )
                self._sleep(backoff_delay(attempt, self.base_delay, self.max_delay, self._rng))
                continue

            hint = {
                400: "invalid request",
                401: "authentication failed - check GROQ_API_KEY",
                403: "permission denied - check GROQ_API_KEY and model access",
                404: f"model {model!r} not found - check GENERATION_MODEL / JUDGE_MODEL",
            }.get(code, "non-retryable API error")
            raise ExternalServiceError(
                f"{summary}: {hint}. {message}".strip(), transient=False, status_code=code, attempts=attempt + 1
            )

        raise AssertionError("unreachable")  # pragma: no cover

    @staticmethod
    def _error_details(response: httpx.Response) -> tuple[str, str | None]:
        try:
            error = response.json().get("error", {})
            return str(error.get("message", "")), error.get("code")
        except (ValueError, AttributeError):
            return response.text[:500], None


def _retry_after(response: httpx.Response) -> float | None:
    raw = response.headers.get("retry-after")
    try:
        return float(raw) if raw is not None else None
    except ValueError:
        return None
