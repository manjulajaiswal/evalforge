"""GroqClient behaviour against a mocked HTTP transport (no network)."""

import json

import httpx
import pytest

from app import config, gemini_client
from app.errors import ConfigurationError, EmptyModelResponseError, ExternalServiceError, JudgeOutputError
from app.groq_client import GROQ_CHAT_URL, GroqClient, parse_wait_seconds


def ok(content="answer", finish="stop", usage=True, reasoning=40):
    body = {"choices": [{"message": {"role": "assistant", "content": content}, "finish_reason": finish}]}
    if usage:
        body["usage"] = {"prompt_tokens": 300, "completion_tokens": 60, "total_tokens": 360,
                         "completion_tokens_details": {"reasoning_tokens": reasoning}}
    return httpx.Response(200, json=body)


def err(code, message="", error_code=None, retry_after=None):
    headers = {"retry-after": str(retry_after)} if retry_after is not None else {}
    return httpx.Response(code, json={"error": {"message": message, "type": "x", "code": error_code}}, headers=headers)


def make(responses, **kw):
    queue = list(responses)
    seen = []

    def handler(request: httpx.Request):
        seen.append(json.loads(request.content))
        item = queue.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    sleeps = []
    client = GroqClient(
        api_key="test-key", http_client=httpx.Client(transport=httpx.MockTransport(handler)),
        max_retries=kw.pop("max_retries", 3), base_delay=1.0, max_delay=60.0, min_interval_s=0.0,
        sleep=sleeps.append, rng=lambda: 1.0, **kw,
    )
    return client, seen, sleeps


def test_success_text_usage_and_request_body():
    client, seen, _ = make([ok()])
    r = client.generate("prompt", model="openai/gpt-oss-20b")
    assert r.text == "answer" and r.attempts == 1 and r.finish_reason == "STOP"
    assert r.usage.input_tokens == 300 and r.usage.output_tokens == 20 and r.usage.thinking_tokens == 40
    assert r.usage.billable_output_tokens == 60  # reasoning counted once, as output
    assert seen[0]["model"] == "openai/gpt-oss-20b" and seen[0]["messages"][0]["content"] == "prompt"
    assert seen[0]["reasoning_effort"] == "low" and "response_format" not in seen[0]


def test_json_mode_requested_for_judge_calls():
    client, seen, _ = make([ok('{"a": 1}')])
    client.generate("Return JSON only.", model="openai/gpt-oss-120b", json_schema={"type": "object"})
    assert seen[0]["response_format"] == {"type": "json_object"}


def test_reasoning_effort_only_sent_to_gpt_oss_models():
    client, seen, _ = make([ok()])
    client.generate("p", model="llama-3.3-70b-versatile")
    assert "reasoning_effort" not in seen[0]


def test_think_blocks_are_stripped_and_length_maps_to_max_tokens():
    client, _, _ = make([ok("<think>secret reasoning</think>\nFinal answer.", finish="length")])
    r = client.generate("p", model="m")
    assert r.text == "Final answer." and r.finish_reason == "MAX_TOKENS"


def test_per_minute_429_is_retried_honouring_retry_after():
    client, seen, sleeps = make([err(429, "Rate limit reached ... tokens per minute (TPM)", retry_after=7), ok()])
    r = client.generate("p", model="m")
    assert r.attempts == 2 and sleeps == [7.5] and r.retry_errors == ["HTTP 429"]


def test_daily_quota_fails_fast():
    msg = "Rate limit reached for model `openai/gpt-oss-120b` on tokens per day (TPD): Limit 200000. Please try again in 7m28.8s."
    client, seen, sleeps = make([err(429, msg, error_code="rate_limit_exceeded")])
    with pytest.raises(ExternalServiceError) as exc:
        client.generate("p", model="m")
    assert not exc.value.transient and "quota exhausted" in str(exc.value) and len(seen) == 1 and sleeps == []


def test_long_wait_429_fails_fast():
    client, seen, _ = make([err(429, "slow down", retry_after=3600)])
    with pytest.raises(ExternalServiceError) as exc:
        client.generate("p", model="m")
    assert not exc.value.transient and len(seen) == 1


def test_server_errors_and_network_errors_are_retried_then_bounded():
    client, _, sleeps = make([err(503, "over capacity"), httpx.ConnectError("boom"), ok()])
    assert client.generate("p", model="m").attempts == 3 and len(sleeps) == 2
    client, seen, _ = make([err(503)] * 4, max_retries=3)
    with pytest.raises(ExternalServiceError) as exc:
        client.generate("p", model="m")
    assert exc.value.transient and len(seen) == 4


@pytest.mark.parametrize("code", [400, 401, 403, 404])
def test_permanent_errors_are_not_retried(code):
    client, seen, sleeps = make([err(code, "nope")])
    with pytest.raises(ExternalServiceError) as exc:
        client.generate("p", model="m")
    assert not exc.value.transient and exc.value.status_code == code and len(seen) == 1 and sleeps == []


def test_json_validate_failed_becomes_judge_output_error():
    client, _, _ = make([err(400, "Failed to generate JSON", error_code="json_validate_failed")])
    with pytest.raises(JudgeOutputError):
        client.generate("p", model="m", json_schema={"type": "object"})


def test_empty_content_is_an_error_and_missing_usage_is_none():
    client, _, _ = make([ok("   ")])
    with pytest.raises(EmptyModelResponseError):
        client.generate("p", model="m")
    client, _, _ = make([ok(usage=False)])
    assert client.generate("p", model="m").usage is None


def test_unexpected_payload_is_reported():
    client, _, _ = make([httpx.Response(200, json={"weird": True})])
    with pytest.raises(ExternalServiceError, match="Unexpected Groq response"):
        client.generate("p", model="m")


def test_parse_wait_seconds():
    assert parse_wait_seconds("Please try again in 7m28.8s.") == pytest.approx(448.8)
    assert parse_wait_seconds("try again in 1h2m3s") == pytest.approx(3723)
    assert parse_wait_seconds("try again in 450ms") == pytest.approx(0.45)
    assert parse_wait_seconds("no hint") is None


def test_missing_key_is_configuration_error(monkeypatch):
    monkeypatch.setattr("app.config.GROQ_API_KEY", None)
    with pytest.raises(ConfigurationError, match="GROQ_API_KEY"):
        GroqClient()


def test_provider_selects_client(monkeypatch):
    monkeypatch.setattr("app.config.LLM_PROVIDER", "groq")
    monkeypatch.setattr("app.config.GROQ_API_KEY", "k")
    gemini_client.set_llm_client(None)
    assert type(gemini_client.get_llm_client()).__name__ == "GroqClient"


def test_groq_models_have_prices():
    from datetime import date

    from app.pricing import price_for

    assert price_for(config.GROQ_GENERATION_MODEL_DEFAULT, date(2026, 10, 28)) is not None
    assert price_for(config.GROQ_JUDGE_MODEL_DEFAULT, date(2026, 10, 28)) is not None


def test_url_is_groq_openai_compatible_endpoint():
    assert GROQ_CHAT_URL == "https://api.groq.com/openai/v1/chat/completions"


def test_full_cli_run_on_groq_with_mocked_api(monkeypatch, ingested_store, report_paths, golden_cases):
    """End-to-end wiring: run_eval -> GroqClient -> (mock) Groq API -> judges -> report."""
    from scripts.run_eval import main
    from tests.fakes import extract_question

    answers = {c.question: c.expected_answer for c in golden_cases}

    def handler(request: httpx.Request):
        body = json.loads(request.content)
        prompt = body["messages"][0]["content"]
        if "response_format" not in body:  # RAG generation
            content = answers[extract_question(prompt)]
        elif "fact-checker" in prompt:
            content = json.dumps({"claims": [{"claim": "c", "label": "supported", "evidence": "e"}], "abstained": False})
        else:
            content = json.dumps({"verdict": "correct", "correctness_score": 1.0, "relevance_score": 0.95,
                                  "abstained": False, "issues": [], "reasoning": "ok"})
        return httpx.Response(200, json={
            "choices": [{"message": {"content": content}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 400, "completion_tokens": 80, "total_tokens": 480,
                      "completion_tokens_details": {"reasoning_tokens": 50}},
        })

    monkeypatch.setattr("app.config.LLM_PROVIDER", "groq")
    monkeypatch.setattr("app.config.GENERATION_MODEL", "openai/gpt-oss-20b")
    monkeypatch.setattr("app.config.JUDGE_MODEL", "openai/gpt-oss-120b")
    llm = GroqClient(api_key="k", http_client=httpx.Client(transport=httpx.MockTransport(handler)), min_interval_s=0)
    code = main(["--ids", "q001,q002,q017", "--quiet"], llm=llm, store=ingested_store, outputs=report_paths,
                run_mode="test-fake")
    report = json.loads(report_paths.json.read_text())
    assert report["summary"]["counts"]["passed"] == 3 and code in (0, 1)
    assert report["config"]["provider"] == "groq" and "groq" in report["config"]["pricing"]["source"]
    assert report["config"]["judge_model"] == "openai/gpt-oss-120b"
    expected_gen_cost = (400 * 0.075 + 80 * 0.30) / 1e6
    assert report["summary"]["cost_usd"]["mean_pipeline_cost_per_query"] == pytest.approx(expected_gen_cost)
    assert "**Provider:** groq" in report_paths.markdown.read_text()
