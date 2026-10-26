"""Evaluation engine with fake LLM + fake embedder (no network)."""

from datetime import date

import pytest

from app.dataset import GoldenCase
from app.errors import ConfigurationError, ExternalServiceError
from app.evaluator import EvalSettings, aggregate, evaluate_case, run_evaluation
from tests.fakes import FakeClock, FakeLLM

ABSTAIN = "I don't have enough information to answer that."


def case(id="c1", question="How many days do customers have to request a refund?", answer="Refunds within 7 days of purchase.",
         should_answer=True, facts=None, category="refund", relevant=None):
    return GoldenCase(
        id=id, question=question, expected_answer=answer,
        expected_source="refund_policy.md" if should_answer else None,
        category=category, difficulty="easy", should_answer=should_answer,
        relevant_sources=relevant if relevant is not None else (["refund_policy.md"] if should_answer else []),
        expected_facts=facts or [],
    )


WINDOW = [{"type": "duration", "min": 7, "max": 7, "unit": "day"}]


def settings(llm, store, **kw):
    kw.setdefault("top_k", 3)
    return EvalSettings(llm=llm, store=store, generation_model="gemini-3.6-flash", judge_model="gemini-3.6-flash",
                        pricing_date=date(2026, 10, 26), clock=FakeClock(), **kw)


def test_correct_answer_passes_with_full_record(ingested_store):
    c = case(facts=WINDOW)
    record = evaluate_case(c, settings(FakeLLM(answers={c.question: "You can request a refund within 7 days."}), ingested_store))
    assert record["status"] == "passed" and record["failure_reasons"] == []
    assert record["answer_evaluation"]["facts"]["passed"] is True
    assert record["retrieval"]["metrics"]["hit"] == 1.0
    assert [ch["rank"] for ch in record["retrieval"]["chunks"]] == [1, 2, 3]
    lat = record["latency_ms"]
    assert set(lat) == {"retrieval", "generation", "generation_including_retries", "end_to_end", "judge_total"}
    assert lat["end_to_end"] == pytest.approx(lat["retrieval"] + lat["generation_including_retries"])
    assert lat["end_to_end"] >= lat["generation"]
    assert record["tokens"]["generation"]["input_tokens"] == 400
    assert record["cost_usd"]["generation"] == pytest.approx((400 * 0.75 + 130 * 3.75) / 1e6)


def test_wrong_number_fails_even_when_judge_says_correct(ingested_store):
    """Deterministic checks are hard constraints: a lenient judge cannot pass a wrong number."""
    c = case(facts=WINDOW)
    record = evaluate_case(c, settings(FakeLLM(answers={c.question: "Refunds within 30 days."}), ingested_store))
    assert record["status"] == "failed" and record["answer_evaluation"]["decided_by"] == "deterministic"
    assert "answer_quality" in record["failure_categories"]
    assert any("7 days" in r or "Missing" in r for r in record["failure_reasons"])


def test_unnecessary_abstention_fails(ingested_store):
    c = case()
    record = evaluate_case(c, settings(FakeLLM(answers={c.question: ABSTAIN}), ingested_store))
    assert record["status"] == "failed"
    assert record["failure_reasons"][0].startswith("Unnecessary abstention")


def test_unanswerable_question_abstention_passes_and_has_no_retrieval_metrics(ingested_store):
    c = case(question="Do you ship internationally?", answer=ABSTAIN + " Not mentioned.", should_answer=False, category="shipping")
    record = evaluate_case(c, settings(FakeLLM(answers={c.question: ABSTAIN}), ingested_store))
    assert record["status"] == "passed" and record["answer_evaluation"]["decided_by"] == "deterministic"
    assert record["retrieval"]["metrics"] is None and record["faithfulness"]["score"] is None


def test_answering_an_unanswerable_question_with_invented_detail_fails(ingested_store):
    c = case(question="Do you ship internationally?", answer=ABSTAIN, should_answer=False)
    llm = FakeLLM(
        answers={c.question: "Yes, international shipping takes 10 days."},
        correct_fn=lambda *a: {"verdict": "incorrect", "correctness_score": 0.0, "relevance_score": 0.8,
                               "abstained": False, "issues": ["invented international policy"], "reasoning": "x"},
        faith_fn=lambda *a: {"claims": [{"claim": "International shipping takes 10 days", "label": "unsupported", "evidence": ""}],
                             "abstained": False},
    )
    record = evaluate_case(c, settings(llm, ingested_store))
    assert record["status"] == "failed"
    assert set(record["failure_categories"]) == {"answer_quality", "faithfulness"}
    assert record["faithfulness"]["has_hallucination"]


def test_correct_answer_with_unsupported_extra_claim_fails_on_faithfulness(ingested_store):
    c = case()
    llm = FakeLLM(
        answers={c.question: "7 days, and we also give free gifts."},
        faith_fn=lambda *a: {"claims": [{"claim": "7 days", "label": "supported", "evidence": "x"},
                                        {"claim": "free gifts", "label": "unsupported", "evidence": ""}], "abstained": False},
    )
    record = evaluate_case(c, settings(llm, ingested_store))
    assert record["status"] == "failed" and record["failure_categories"] == ["faithfulness"]
    assert record["faithfulness"]["score"] == 0.5


def test_retrieval_miss_is_reported(ingested_store):
    c = case(question="How long does standard shipping take?", relevant=["shipping_policy.md"])
    record = evaluate_case(c, settings(FakeLLM(answers={c.question: "3 to 5 business days."}), ingested_store, top_k=1))
    assert record["retrieval"]["metrics"]["hit"] == 1.0
    c2 = case(id="c2", question="How long does standard shipping take?", relevant=["account_policy.md"])
    record2 = evaluate_case(c2, settings(FakeLLM(answers={c2.question: "x"}), ingested_store, top_k=1))
    assert "retrieval" in record2["failure_categories"]


def test_partial_abstention_is_graded_by_judge_not_failed_as_abstention(ingested_store):
    """'I don't have enough information... but the policy says X' is a nuanced answer, not a refusal."""
    c = case(question="Can I get a refund for a product that arrived damaged?")
    mixed = ("I don't have enough information to answer that. The policy only says refunds are available "
             "if the product has not been used extensively or damaged.")
    record = evaluate_case(c, settings(FakeLLM(answers={c.question: mixed}), ingested_store))
    assert record["status"] == "passed" and record["answer_evaluation"]["decided_by"] == "judge"
    assert record["answer_evaluation"]["abstained_canonical"] and not record["answer_evaluation"]["pure_abstention"]


def test_abstention_plus_invented_detail_on_unanswerable_needs_judge_agreement(ingested_store):
    c = case(question="Is express shipping free?", answer=ABSTAIN, should_answer=False)
    answer = ABSTAIN + " To get free express shipping, select Express at checkout."
    llm = FakeLLM(answers={c.question: answer},
                  correct_fn=lambda *a: {"verdict": "incorrect", "correctness_score": 0.0, "relevance_score": 0.5,
                                         "abstained": True, "issues": ["invented"], "reasoning": "x"})
    record = evaluate_case(c, settings(llm, ingested_store))
    assert record["status"] == "failed" and record["answer_evaluation"]["decided_by"] == "judge"


def test_truncated_generation_is_flagged(ingested_store):
    class Truncating(FakeLLM):
        def generate(self, prompt, *, model, json_schema=None):
            r = super().generate(prompt, model=model, json_schema=json_schema)
            if json_schema is None:
                r.finish_reason = "MAX_TOKENS"
            return r

    record = evaluate_case(case(), settings(Truncating(answers={case().question: "7 days"}), ingested_store))
    assert record["generation"]["possibly_truncated"] is True
    assert aggregate([record])["answer"]["possibly_truncated_answers"] == 1


# --- failure isolation ----------------------------------------------------------------


def test_empty_model_response_is_a_case_error_and_does_not_abort(ingested_store):
    from app.errors import EmptyModelResponseError

    cases = [case(id=f"c{i}", question=f"q{i}?") for i in range(3)]
    llm = FakeLLM(answer_fn=lambda q, p: EmptyModelResponseError("blocked") if q == "q0?" else "within 7 days")
    result = run_evaluation(cases, settings(llm, ingested_store))
    assert [r["status"] for r in result["cases"]] == ["error", "passed", "passed"] and not result["aborted"]
    assert result["cases"][0]["error"]["type"] == "empty_response"

def failing_on(question_to_exc):
    def answer_fn(question, prompt):
        return question_to_exc.get(question, "Refunds within 7 days.")
    return answer_fn


def test_transient_api_failure_is_isolated_and_not_a_quality_failure(ingested_store):
    cases = [case(id=f"c{i}", question=f"refund question {i}?") for i in range(4)]
    exc = ExternalServiceError("HTTP 503 UNAVAILABLE after 5 attempt(s)", transient=True, status_code=503, attempts=5)
    llm = FakeLLM(answer_fn=failing_on({"refund question 1?": exc}))
    result = run_evaluation(cases, settings(llm, ingested_store))
    statuses = [r["status"] for r in result["cases"]]
    assert statuses == ["passed", "error", "passed", "passed"] and not result["aborted"]
    err = result["cases"][1]["error"]
    assert err["type"] == "infrastructure" and err["status_code"] == 503 and err["transient"] is True
    summary = aggregate(result["cases"])
    assert summary["counts"]["errored"] == 1 and summary["counts"]["failed"] == 0
    assert summary["answer"]["answer_accuracy"] == 1.0  # computed over completed cases only


def test_permanent_api_error_aborts_remaining_cases(ingested_store):
    cases = [case(id=f"c{i}", question=f"refund question {i}?") for i in range(3)]
    exc = ExternalServiceError("HTTP 401: authentication failed", transient=False, status_code=401)
    result = run_evaluation(cases, settings(FakeLLM(answer_fn=failing_on({"refund question 0?": exc})), ingested_store))
    assert [r["status"] for r in result["cases"]] == ["error", "not_run", "not_run"]
    assert result["aborted"] and "permanent API error" in result["abort_reason"]


def test_consecutive_transient_failures_trip_the_circuit_breaker(ingested_store):
    cases = [case(id=f"c{i}", question=f"q{i}?") for i in range(5)]
    exc = ExternalServiceError("503", transient=True, status_code=503)
    llm = FakeLLM(answer_fn=lambda q, p: exc)
    result = run_evaluation(cases, settings(llm, ingested_store, max_consecutive_infra_failures=2))
    assert [r["status"] for r in result["cases"]] == ["error", "error", "not_run", "not_run", "not_run"]


def test_configuration_error_mid_run_aborts(ingested_store):
    cases = [case(id="a", question="qa?"), case(id="b", question="qb?")]
    llm = FakeLLM(answer_fn=lambda q, p: ConfigurationError("embedding model missing"))
    result = run_evaluation(cases, settings(llm, ingested_store))
    assert [r["status"] for r in result["cases"]] == ["error", "not_run"]
    assert result["cases"][0]["error"]["type"] == "configuration"


def test_malformed_judge_output_is_a_judge_failure(ingested_store, monkeypatch):
    monkeypatch.setattr("app.config.JUDGE_MAX_FORMAT_RETRIES", 0)
    llm = FakeLLM(correct_fn=lambda *a: "not json")
    result = run_evaluation([case()], settings(llm, ingested_store))
    assert result["cases"][0]["status"] == "error" and result["cases"][0]["error"]["type"] == "judge_failure"


def test_unexpected_exception_is_recorded_as_evaluator_error_with_traceback(ingested_store):
    llm = FakeLLM(answer_fn=lambda q, p: KeyError("bug"))
    result = run_evaluation([case()], settings(llm, ingested_store))
    error = result["cases"][0]["error"]
    assert error["type"] == "evaluator_error" and "KeyError" in error["traceback"]


# --- aggregation ------------------------------------------------------------------

def test_aggregate_metrics_and_denominators(ingested_store):
    cases = [
        case(id="a", question="qa?", facts=WINDOW),
        case(id="b", question="qb?", facts=WINDOW),
        case(id="u", question="qu?", answer=ABSTAIN, should_answer=False),
    ]
    llm = FakeLLM(answers={"qa?": "within 7 days", "qb?": "within 30 days", "qu?": ABSTAIN})
    result = run_evaluation(cases, settings(llm, ingested_store))
    s = aggregate(result["cases"])
    assert s["counts"] == {"selected": 3, "attempted": 3, "completed": 3, "passed": 2, "failed": 1, "errored": 0,
                           "not_run": 0, "errors_by_type": {}}
    assert s["answer"]["answer_accuracy"] == pytest.approx(2 / 3)
    assert s["answer"]["answerable_accuracy"] == 0.5 and s["answer"]["abstention_accuracy"] == 1.0
    assert s["answer"]["deterministic_fact_pass_rate"] == 0.5 and s["answer"]["cases_with_fact_checks"] == 2
    assert s["retrieval"]["n_labeled"] == 2  # unanswerable excluded
    assert s["faithfulness"]["n_scored"] == 2 and s["faithfulness"]["n_without_claims"] == 1
    assert s["latency_ms"]["n"] == 3 and s["latency_ms"]["end_to_end"]["p95"] is not None
    assert s["cost_usd"]["pipeline_total"] == pytest.approx(3 * (400 * 0.75 + 130 * 3.75) / 1e6)


def test_cost_is_unavailable_if_any_case_lacks_usage(ingested_store):
    result = run_evaluation([case()], settings(FakeLLM(usage=None), ingested_store))
    s = aggregate(result["cases"])
    assert s["cost_usd"]["pipeline_total"] is None and s["tokens"]["generation_input"] is None


def test_aggregate_of_empty_run_has_no_fake_numbers():
    s = aggregate([])
    assert s["answer"]["answer_accuracy"] is None and s["latency_ms"]["end_to_end"]["p95"] is None
