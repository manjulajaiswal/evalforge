"""GeminiClient behaviour with a fake SDK object (no network)."""

from types import SimpleNamespace

import httpx
import pytest
from google.genai import errors as genai_errors

from app.errors import ConfigurationError, EmptyModelResponseError, ExternalServiceError
from app.gemini_client import GeminiClient, backoff_delay, extract_usage


def api_error(code, status, retry_delay=None):
    details = []
    if retry_delay:
        details.append({"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": retry_delay})
    payload = {"error": {"code": code, "message": f"{status} message", "status": status, "details": details}}
    cls = genai_errors.ServerError if code >= 500 else genai_errors.ClientError
    return cls(code, payload)


def ok_response(text="answer", usage=True):
    meta = SimpleNamespace(prompt_token_count=100, candidates_token_count=20, thoughts_token_count=50, total_token_count=170)
    return SimpleNamespace(text=text, usage_metadata=meta if usage else None)


class FakeSDK:
    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.calls = []
        self.models = self

    def generate_content(self, **kwargs):
        self.calls.append(kwargs)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


def make_client(outcomes, **kwargs):
    sleeps = []
    sdk = FakeSDK(outcomes)
    client = GeminiClient(
        sdk_client=sdk, max_retries=kwargs.pop("max_retries", 3), base_delay=1.0, max_delay=8.0,
        min_interval_s=kwargs.pop("min_interval_s", 0.0), sleep=sleeps.append, rng=lambda: 1.0, **kwargs,
    )
    return client, sdk, sleeps


def test_success_returns_text_usage_and_timing():
    client, sdk, sleeps = make_client([ok_response()])
    r = client.generate("prompt", model="gemini-3.6-flash")
    assert r.text == "answer" and r.attempts == 1 and sleeps == []
    assert r.usage.input_tokens == 100 and r.usage.billable_output_tokens == 70
    assert r.latency_ms >= 0 and sdk.calls[0]["model"] == "gemini-3.6-flash"


def test_transient_503_is_retried_with_exponential_backoff():
    client, sdk, sleeps = make_client([api_error(503, "UNAVAILABLE"), api_error(503, "UNAVAILABLE"), ok_response()])
    r = client.generate("p", model="m")
    assert r.attempts == 3 and len(sdk.calls) == 3
    assert sleeps == [1.0, 2.0]  # rng=1.0 -> upper bound of full jitter: base * 2**attempt
    assert r.retry_errors == ["HTTP 503 UNAVAILABLE", "HTTP 503 UNAVAILABLE"]


def test_retries_are_bounded():
    client, sdk, sleeps = make_client([api_error(503, "UNAVAILABLE")] * 4, max_retries=3)
    with pytest.raises(ExternalServiceError) as exc:
        client.generate("p", model="m")
    assert exc.value.transient and exc.value.attempts == 4 and exc.value.status_code == 503
    assert len(sdk.calls) == 4 and len(sleeps) == 3


@pytest.mark.parametrize("code,status", [(400, "INVALID_ARGUMENT"), (401, "UNAUTHENTICATED"), (403, "PERMISSION_DENIED"), (404, "NOT_FOUND")])
def test_permanent_errors_are_not_retried(code, status):
    client, sdk, sleeps = make_client([api_error(code, status)])
    with pytest.raises(ExternalServiceError) as exc:
        client.generate("p", model="m")
    assert not exc.value.transient and exc.value.status_code == code
    assert len(sdk.calls) == 1 and sleeps == []


def test_429_honours_server_retry_delay():
    client, _, sleeps = make_client([api_error(429, "RESOURCE_EXHAUSTED", retry_delay="7s"), ok_response()])
    client.generate("p", model="m")
    assert sleeps == [7.0]


def test_network_errors_are_retried():
    client, _, sleeps = make_client([httpx.ConnectError("boom"), ok_response()])
    assert client.generate("p", model="m").attempts == 2 and len(sleeps) == 1


def test_empty_response_is_an_error_not_an_answer():
    client, _, _ = make_client([ok_response(text="")])
    with pytest.raises(EmptyModelResponseError):
        client.generate("p", model="m")


def test_missing_usage_is_reported_as_none():
    client, _, _ = make_client([ok_response(usage=False)])
    assert client.generate("p", model="m").usage is None
    assert extract_usage(SimpleNamespace(usage_metadata=SimpleNamespace())) is None


def test_json_mode_is_requested_for_schemas():
    client, sdk, _ = make_client([ok_response("{}")])
    client.generate("p", model="m", json_schema={"type": "object"})
    cfg = sdk.calls[0]["config"]
    assert cfg.response_mime_type == "application/json" and cfg.response_json_schema == {"type": "object"}


def test_throttle_spaces_requests():
    t = iter([0.0, 0.0, 0.0, 0.0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
    client, _, sleeps = make_client([ok_response(), ok_response()], min_interval_s=1.0, clock=lambda: next(t))
    client.generate("p", model="m")
    client.generate("p", model="m")
    assert sleeps and sleeps[0] == pytest.approx(0.9)


def test_backoff_is_capped():
    assert backoff_delay(10, 2.0, 60.0, rng=lambda: 1.0) == 60.0
    assert backoff_delay(0, 2.0, 60.0, rng=lambda: 0.5) == 1.0


def test_missing_api_key_is_a_configuration_error():
    with pytest.raises(ConfigurationError, match="GEMINI_API_KEY"):
        GeminiClient()
