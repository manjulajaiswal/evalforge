import json
from datetime import date

import pytest

from app.errors import JudgeOutputError
from app.gemini_client import TokenUsage
from app.judge import judge_correctness, judge_faithfulness, parse_correctness, parse_faithfulness
from app.pricing import estimate_cost_usd, price_for
from tests.fakes import FakeLLM

GOOD_CORRECT = {"verdict": "correct", "correctness_score": 1.0, "relevance_score": 0.9, "abstained": False, "issues": [], "reasoning": "ok"}


# --- parsing / validation ---------------------------------------------------------

def test_parse_faithfulness_valid_and_score():
    text = json.dumps({"claims": [{"claim": "A", "label": "supported", "evidence": "e"},
                                  {"claim": "B", "label": "unsupported", "evidence": ""}], "abstained": False})
    claims, abstained = parse_faithfulness(text)
    assert [c.label for c in claims] == ["supported", "unsupported"] and abstained is False


def test_parse_strips_markdown_fences():
    parse_correctness("```json\n" + json.dumps(GOOD_CORRECT) + "\n```")


@pytest.mark.parametrize(
    "payload,message",
    [
        ("not json", "valid JSON"),
        ("[1]", "must be an object"),
        ({"claims": "x", "abstained": False}, "'claims' must be a list"),
        ({"claims": [{"claim": "A", "label": "maybe", "evidence": ""}], "abstained": False}, "label must be one of"),
        ({"claims": [{"claim": "", "label": "supported", "evidence": ""}], "abstained": False}, "non-empty"),
        ({"claims": [], "abstained": "no"}, "'abstained'"),
    ],
)
def test_malformed_faithfulness_output(payload, message):
    with pytest.raises(JudgeOutputError, match=message):
        parse_faithfulness(payload if isinstance(payload, str) else json.dumps(payload))


@pytest.mark.parametrize(
    "changes,message",
    [
        ({"verdict": "great"}, "'verdict' must be one of"),
        ({"correctness_score": 1.5}, "between 0 and 1"),
        ({"relevance_score": "high"}, "must be a number"),
        ({"correctness_score": True}, "must be a number"),
        ({"verdict": "correct", "correctness_score": 0.1}, "inconsistent"),
        ({"verdict": "incorrect", "correctness_score": 0.9}, "inconsistent"),
        ({"issues": "none"}, "'issues' must be a list"),
    ],
)
def test_malformed_correctness_output(changes, message):
    with pytest.raises(JudgeOutputError, match=message):
        parse_correctness(json.dumps({**GOOD_CORRECT, **changes}))


# --- judge calls -------------------------------------------------------------------

def test_faithfulness_judge_never_sees_the_gold_answer():
    llm = FakeLLM()
    judge_faithfulness(llm, "How long?", ["Refunds within 7 days."], "Within 7 days.")
    prompt = llm.calls[0]["prompt"]
    assert "Refunds within 7 days." in prompt and "Within 7 days." in prompt
    assert "REFERENCE" not in prompt and "expected" not in prompt.lower()


def test_correctness_judge_gets_reference_and_answerability_note():
    llm = FakeLLM()
    judge_correctness(llm, "Q?", "GOLD ANSWER", "model answer", should_answer=False)
    prompt = llm.calls[0]["prompt"]
    assert "GOLD ANSWER" in prompt and "NOT available" in prompt


def test_malformed_judge_output_is_retried_then_succeeds(monkeypatch):
    monkeypatch.setattr("app.config.JUDGE_MAX_FORMAT_RETRIES", 1)
    replies = iter(["oops", GOOD_CORRECT])
    llm = FakeLLM(correct_fn=lambda *a: next(replies))
    result = judge_correctness(llm, "Q?", "A", "A", True)
    assert result.verdict == "correct" and result.call.format_retries == 1
    assert result.call.usage.input_tokens == 800  # tokens of the malformed reply are still counted


def test_persistently_malformed_judge_output_raises(monkeypatch):
    monkeypatch.setattr("app.config.JUDGE_MAX_FORMAT_RETRIES", 1)
    llm = FakeLLM(faith_fn=lambda *a: "garbage")
    with pytest.raises(JudgeOutputError, match="after 2 judge response"):
        judge_faithfulness(llm, "Q", ["ctx"], "ans")


def test_faithfulness_score_semantics():
    llm = FakeLLM(faith_fn=lambda *a: {"claims": [], "abstained": True})
    result = judge_faithfulness(llm, "Q", ["ctx"], "I don't have enough information to answer that.")
    assert result.score is None and not result.has_hallucination
    llm = FakeLLM(faith_fn=lambda *a: {"claims": [{"claim": "a", "label": "supported", "evidence": ""},
                                                 {"claim": "b", "label": "contradicted", "evidence": ""}], "abstained": False})
    result = judge_faithfulness(llm, "Q", ["ctx"], "a and b")
    assert result.score == 0.5 and result.has_hallucination and len(result.contradicted_claims) == 1


# --- pricing -----------------------------------------------------------------------

def test_price_window_changes_on_2027_01_01():
    assert price_for("gemini-3.6-flash", date(2026, 12, 31)).input_per_million == 0.75
    assert price_for("gemini-3.6-flash", date(2027, 1, 1)).output_per_million == 7.50


def test_cost_includes_thinking_tokens_as_output():
    usage = TokenUsage(input_tokens=1_000_000, output_tokens=100_000, thinking_tokens=900_000, total_tokens=2_000_000)
    assert estimate_cost_usd("gemini-3.6-flash", usage, date(2026, 10, 26)) == pytest.approx(0.75 + 3.75)


def test_cost_is_unavailable_rather_than_guessed():
    usage = TokenUsage(10, 10, 0, 20)
    assert estimate_cost_usd("some-unknown-model", usage, date(2026, 10, 26)) is None
    assert estimate_cost_usd("gemini-3.6-flash", None, date(2026, 10, 26)) is None
    assert estimate_cost_usd("gemini-3.6-flash", TokenUsage(None, 5, 0, 5), date(2026, 10, 26)) is None
