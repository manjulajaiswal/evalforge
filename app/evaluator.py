"""Evaluation engine: run golden cases through the RAG pipeline and score them.

Per case:
  1. retrieve()                  - app.rag_pipeline (the system under test)
  2. generate_answer_detailed()  - app.rag_pipeline (the system under test)
  3. deterministic checks        - app.metrics (abstention wording, typed facts)
  4. faithfulness judge          - app.judge (context + answer, no gold answer)
  5. correctness judge           - app.judge (gold answer + answer)
  6. retrieval metrics           - app.metrics (needs relevant_sources labels)

A case PASSES only if its answer is judged correct (after deterministic hard
checks) AND the faithfulness judge found no unsupported or contradicted
claims. Errors are never counted as passes or as quality failures: they get
status "error" with an error_type, and the run is marked incomplete.
"""

from __future__ import annotations

import time
import traceback
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Callable

from app import config
from app.dataset import GoldenCase, load_and_validate
from app.errors import ConfigurationError, EmptyModelResponseError, ExternalServiceError, JudgeOutputError
from app.gemini_client import LLMClient, TokenUsage
from app.judge import judge_correctness, judge_faithfulness
from app.metrics import (
    check_facts,
    is_canonical_abstention,
    is_pure_abstention,
    mean,
    percentile,
    retrieval_metrics,
    semantic_similarity,
)
from app.pricing import estimate_cost_usd
from app.rag_pipeline import generate_answer_detailed, retrieve
from app.vector_store import VectorStore

# error_type values
INFRASTRUCTURE = "infrastructure"  # Gemini API unavailable, auth failure, network
EMPTY_RESPONSE = "empty_response"  # model returned no text (e.g. safety block)
CONFIGURATION = "configuration"  # e.g. embedding model cannot be loaded
JUDGE_FAILURE = "judge_failure"  # judge returned malformed output
EVALUATOR_ERROR = "evaluator_error"  # unexpected exception in EvalForge code (a bug)


@dataclass
class EvalSettings:
    llm: LLMClient
    store: VectorStore | None = None
    top_k: int = field(default_factory=lambda: config.TOP_K)
    generation_model: str = field(default_factory=lambda: config.GENERATION_MODEL)
    judge_model: str = field(default_factory=lambda: config.JUDGE_MODEL)
    pricing_date: date = field(default_factory=date.today)
    max_consecutive_infra_failures: int = field(default_factory=lambda: config.MAX_CONSECUTIVE_INFRA_FAILURES)
    clock: Callable[[], float] = time.perf_counter
    compute_semantic_similarity: bool = True
    progress: Callable[[int, int, dict], None] | None = None


def _usage_dict(usage: TokenUsage | None) -> dict | None:
    return usage.to_dict() if usage is not None else None


def _base_record(case: GoldenCase) -> dict:
    return {
        "id": case.id,
        "question": case.question,
        "category": case.category,
        "difficulty": case.difficulty,
        "tags": case.tags,
        "should_answer": case.should_answer,
        "expected_answer": case.expected_answer,
        "expected_source": case.expected_source,
        "relevant_sources": case.relevant_sources,
        "status": "not_run",
        "failure_reasons": [],
        "failure_categories": [],
        "error": None,
        "answer": None,
        "retrieval": None,
        "answer_evaluation": None,
        "faithfulness": None,
        "latency_ms": None,
        "generation": None,
        "tokens": None,
        "cost_usd": None,
    }


def evaluate_case(case: GoldenCase, settings: EvalSettings) -> dict:
    """Run one case end to end. Exceptions propagate to run_evaluation."""
    record = _base_record(case)

    # 1-2. The system under test.
    retrieval = retrieve(case.question, top_k=settings.top_k, store=settings.store)
    generation = generate_answer_detailed(
        case.question, retrieval.contexts, llm=settings.llm, model=settings.generation_model
    )
    # End-to-end = what a user waits for: query embedding + vector search +
    # model call, including any retry backoff. Each part is timed with a
    # monotonic clock (time.perf_counter) where it happens.
    end_to_end_ms = retrieval.latency_ms + generation.total_latency_ms
    answer = generation.text
    record["answer"] = answer

    retrieved_sources = retrieval.sources
    r_metrics = retrieval_metrics(retrieved_sources, case.relevant_sources)
    record["retrieval"] = {
        "top_k": settings.top_k,
        "chunks": [
            {"rank": c.rank, "chunk_id": c.chunk_id, "source": c.source, "distance": c.distance, "text": c.text}
            for c in retrieval.chunks
        ],
        "metrics": r_metrics,
    }
    record["generation"] = {
        "model": generation.model,
        "attempts": generation.attempts,
        "retry_errors": generation.retry_errors,
        "finish_reason": generation.finish_reason,
        # STOP (or unreported) is normal; MAX_TOKENS etc. means the answer may be cut off.
        "possibly_truncated": generation.finish_reason not in (None, "STOP"),
    }

    # 3. Deterministic checks. Only a *pure* abstention (nothing but the
    # "not enough information" sentence) is decided deterministically; an
    # answer that abstains on part of the question and answers the rest is
    # graded by the judge.
    abstained_canonical = is_canonical_abstention(answer)
    pure_abstention = is_pure_abstention(answer)
    facts = check_facts(case.expected_facts, answer)
    similarity = semantic_similarity(case.expected_answer, answer) if settings.compute_semantic_similarity else None

    # 4-5. LLM judges (raise JudgeOutputError / ExternalServiceError on failure).
    faith = judge_faithfulness(settings.llm, case.question, retrieval.contexts, answer, model=settings.judge_model)
    correct = judge_correctness(
        settings.llm, case.question, case.expected_answer, answer, case.should_answer, model=settings.judge_model
    )

    # Combine: deterministic checks are hard constraints applied before the judge verdict.
    reasons: list[str] = []
    categories: list[str] = []
    if case.should_answer:
        if pure_abstention:
            answer_correct, decided_by = False, "deterministic"
            reasons.append("Unnecessary abstention: the answer is in the knowledge base but the system declined.")
        elif facts["passed"] is False:
            answer_correct, decided_by = False, "deterministic"
            reasons.append("Missing or wrong required fact(s): " + "; ".join(facts["missing"]))
        else:
            answer_correct, decided_by = correct.verdict == "correct", "judge"
    else:
        if pure_abstention:
            answer_correct, decided_by = True, "deterministic"
        else:
            answer_correct, decided_by = correct.verdict == "correct", "judge"
    if not answer_correct and decided_by == "judge":
        detail = "; ".join(correct.issues) or correct.reasoning
        label = "Answered an unanswerable question" if not case.should_answer else "Judge verdict"
        reasons.append(f"{label}: {correct.verdict}. {detail}".strip())
    if not answer_correct:
        categories.append("answer_quality")

    if faith.has_hallucination:
        categories.append("faithfulness")
        for c in faith.contradicted_claims:
            reasons.append(f"Contradicts retrieved context: {c.claim}")
        for c in faith.unsupported_claims:
            reasons.append(f"Not supported by retrieved context: {c.claim}")

    if r_metrics is not None and r_metrics["missing_sources"]:
        # Informational: retrieval misses are reported and gated in aggregate,
        # but a case is judged on its answer.
        categories.append("retrieval")
        reasons.append("Relevant source(s) not retrieved: " + ", ".join(r_metrics["missing_sources"]))

    passed = answer_correct and not faith.has_hallucination
    record["status"] = "passed" if passed else "failed"
    record["failure_reasons"] = reasons
    record["failure_categories"] = categories

    record["answer_evaluation"] = {
        "correct": answer_correct,
        "decided_by": decided_by,
        "abstained_canonical": abstained_canonical,
        "pure_abstention": pure_abstention,
        "facts": facts,
        "semantic_similarity": similarity,
        "judge": {
            "verdict": correct.verdict,
            "correctness_score": correct.correctness_score,
            "relevance_score": correct.relevance_score,
            "abstained": correct.abstained,
            "issues": correct.issues,
            "reasoning": correct.reasoning,
            "format_retries": correct.call.format_retries,
        },
    }
    record["faithfulness"] = {
        "score": faith.score,
        "claim_count": len(faith.claims),
        "has_hallucination": faith.has_hallucination,
        "abstained": faith.abstained,
        "claims": [{"claim": c.claim, "label": c.label, "evidence": c.evidence} for c in faith.claims],
        "format_retries": faith.call.format_retries,
    }

    judge_tokens = _sum_usage([faith.call.usage, correct.call.usage])
    generation_cost = estimate_cost_usd(generation.model, generation.usage, settings.pricing_date)
    faith_cost = estimate_cost_usd(faith.call.model, faith.call.usage, settings.pricing_date)
    correct_cost = estimate_cost_usd(correct.call.model, correct.call.usage, settings.pricing_date)
    record["latency_ms"] = {
        "retrieval": retrieval.latency_ms,
        "generation": generation.latency_ms,
        "generation_including_retries": generation.total_latency_ms,
        "end_to_end": end_to_end_ms,
        "judge_total": faith.call.latency_ms + correct.call.latency_ms,
    }
    # API attempts per judge call (> 1 means retries, usually HTTP 429 rate limiting).
    record["judge_calls"] = {
        "faithfulness_attempts": faith.call.attempts,
        "correctness_attempts": correct.call.attempts,
    }
    record["tokens"] = {"generation": _usage_dict(generation.usage), "judge": _usage_dict(judge_tokens)}
    record["cost_usd"] = {
        "generation": generation_cost,
        "judge": None if faith_cost is None or correct_cost is None else faith_cost + correct_cost,
    }
    return record


def _sum_usage(usages: list[TokenUsage | None]) -> TokenUsage | None:
    if any(u is None for u in usages):
        return None
    total = TokenUsage(0, 0, 0, 0)
    for u in usages:
        for name in ("input_tokens", "output_tokens", "thinking_tokens", "total_tokens"):
            setattr(total, name, getattr(total, name) + (getattr(u, name) or 0))
    return total


def _error_record(case: GoldenCase, error_type: str, exc: BaseException, partial: dict | None = None) -> dict:
    record = _base_record(case)
    record["status"] = "error"
    error: dict[str, Any] = {"type": error_type, "exception": type(exc).__name__, "message": str(exc)}
    if isinstance(exc, ExternalServiceError):
        error.update(status_code=exc.status_code, transient=exc.transient, attempts=exc.attempts)
    if error_type == EVALUATOR_ERROR:
        error["traceback"] = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))[-4000:]
    record["error"] = error
    record["failure_reasons"] = [f"{error_type}: {exc}"]
    return record


def run_evaluation(cases: list[GoldenCase], settings: EvalSettings) -> dict:
    """Evaluate every case, isolating per-case failures.

    Returns {"cases": [...], "aborted": bool, "abort_reason": str | None,
    "duration_s": float}. Aborts early (remaining cases -> "not_run") on a
    permanent API error (bad key, unknown model), a configuration error, or
    too many consecutive infrastructure/evaluator failures.
    """
    records: list[dict] = []
    aborted, abort_reason = False, None
    consecutive_failures = 0
    started = settings.clock()

    for index, case in enumerate(cases):
        if aborted:
            not_run = _base_record(case)
            not_run["failure_reasons"] = [f"Not run: {abort_reason}"]
            records.append(not_run)
            continue
        try:
            record = evaluate_case(case, settings)
            consecutive_failures = 0
        except EmptyModelResponseError as exc:
            # One blocked/empty answer is a per-case error, not a reason to stop the run.
            record = _error_record(case, EMPTY_RESPONSE, exc)
            consecutive_failures += 1
        except ExternalServiceError as exc:
            record = _error_record(case, INFRASTRUCTURE, exc)
            consecutive_failures += 1
            if not exc.transient:
                aborted, abort_reason = True, f"permanent API error on case {case.id}: {exc}"
        except ConfigurationError as exc:
            record = _error_record(case, CONFIGURATION, exc)
            aborted, abort_reason = True, f"configuration error on case {case.id}: {exc}"
        except JudgeOutputError as exc:
            record = _error_record(case, JUDGE_FAILURE, exc)
            consecutive_failures = 0
        except Exception as exc:  # noqa: BLE001 - recorded with traceback and reported as an evaluator bug
            record = _error_record(case, EVALUATOR_ERROR, exc)
            consecutive_failures += 1

        if not aborted and consecutive_failures >= settings.max_consecutive_infra_failures:
            aborted = True
            abort_reason = f"{consecutive_failures} consecutive cases failed with errors (last: {record['id']})"
        records.append(record)
        if settings.progress:
            try:
                settings.progress(index + 1, len(cases), record)
            except Exception:  # noqa: BLE001 - progress output is cosmetic; never lose results over it
                pass

    return {
        "cases": records,
        "aborted": aborted,
        "abort_reason": abort_reason,
        "duration_s": settings.clock() - started,
    }


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def _rate(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def aggregate(records: list[dict]) -> dict:
    completed = [r for r in records if r["status"] in ("passed", "failed")]
    answerable = [r for r in completed if r["should_answer"]]
    unanswerable = [r for r in completed if not r["should_answer"]]
    errors = [r for r in records if r["status"] == "error"]

    def correct(r):
        return r["answer_evaluation"]["correct"]

    with_facts = [r for r in completed if r["answer_evaluation"]["facts"]["checked"]]
    retrieval_rows = [r["retrieval"]["metrics"] for r in completed if r["retrieval"]["metrics"] is not None]
    faith_scores = [r["faithfulness"]["score"] for r in completed if r["faithfulness"]["score"] is not None]

    e2e = [r["latency_ms"]["end_to_end"] for r in completed]
    gen = [r["latency_ms"]["generation"] for r in completed]
    ret = [r["latency_ms"]["retrieval"] for r in completed]
    judge_lat = [r["latency_ms"]["judge_total"] for r in completed]

    def token_sum(kind: str, field_name: str):
        values = [r["tokens"][kind] for r in completed]
        if not values or any(v is None or v.get(field_name) is None for v in values):
            return None
        return sum(v[field_name] for v in values)

    def cost_values(kind: str):
        values = [r["cost_usd"][kind] for r in completed]
        if not values or any(v is None for v in values):
            return None  # all-or-nothing: a partial sum would understate cost
        return values

    gen_costs = cost_values("generation")
    judge_costs = cost_values("judge")

    by_category: dict[str, dict] = {}
    for r in records:
        bucket = by_category.setdefault(r["category"], {"total": 0, "completed": 0, "passed": 0, "correct": 0})
        bucket["total"] += 1
        if r["status"] in ("passed", "failed"):
            bucket["completed"] += 1
            bucket["passed"] += r["status"] == "passed"
            bucket["correct"] += correct(r)
    for bucket in by_category.values():
        bucket["accuracy"] = _rate(bucket["correct"], bucket["completed"])
        bucket["pass_rate"] = _rate(bucket["passed"], bucket["completed"])

    return {
        "counts": {
            "selected": len(records),
            "attempted": sum(r["status"] != "not_run" for r in records),
            "completed": len(completed),
            "passed": sum(r["status"] == "passed" for r in records),
            "failed": sum(r["status"] == "failed" for r in records),
            "errored": len(errors),
            "not_run": sum(r["status"] == "not_run" for r in records),
            "errors_by_type": _count(e["error"]["type"] for e in errors),
        },
        "answer": {
            "pass_rate": _rate(sum(r["status"] == "passed" for r in completed), len(completed)),
            "answer_accuracy": _rate(sum(correct(r) for r in completed), len(completed)),
            "answerable_accuracy": _rate(sum(correct(r) for r in answerable), len(answerable)),
            "abstention_accuracy": _rate(sum(correct(r) for r in unanswerable), len(unanswerable)),
            "unnecessary_abstention_rate": _rate(
                sum(r["answer_evaluation"]["pure_abstention"] for r in answerable), len(answerable)
            ),
            "mean_correctness_score": mean(r["answer_evaluation"]["judge"]["correctness_score"] for r in completed),
            "mean_answer_relevance": mean(r["answer_evaluation"]["judge"]["relevance_score"] for r in completed),
            "deterministic_fact_pass_rate": _rate(
                sum(bool(r["answer_evaluation"]["facts"]["passed"]) for r in with_facts), len(with_facts)
            ),
            "cases_with_fact_checks": len(with_facts),
            "possibly_truncated_answers": sum(bool(r["generation"].get("possibly_truncated")) for r in completed),
            "mean_semantic_similarity": mean(r["answer_evaluation"]["semantic_similarity"] for r in completed),
            "n_answerable": len(answerable),
            "n_unanswerable": len(unanswerable),
        },
        "faithfulness": {
            "mean_faithfulness": mean(faith_scores),
            "n_scored": len(faith_scores),
            "n_without_claims": len(completed) - len(faith_scores),
            "hallucination_rate": _rate(sum(r["faithfulness"]["has_hallucination"] for r in completed), len(completed)),
            "cases_with_contradictions": sum(
                any(c["label"] == "contradicted" for c in r["faithfulness"]["claims"]) for r in completed
            ),
            "cases_with_unsupported_claims": sum(
                any(c["label"] == "unsupported" for c in r["faithfulness"]["claims"]) for r in completed
            ),
        },
        "retrieval": {
            "n_labeled": len(retrieval_rows),
            "source_hit_rate": mean(m["hit"] for m in retrieval_rows),
            "mean_source_precision": mean(m["precision"] for m in retrieval_rows),
            "mean_source_recall": mean(m["recall"] for m in retrieval_rows),
            "mrr": mean(m["reciprocal_rank"] for m in retrieval_rows),
        },
        "latency_ms": {
            "n": len(completed),
            "end_to_end": {"p50": percentile(e2e, 50), "p95": percentile(e2e, 95), "mean": mean(e2e)},
            "generation": {"p50": percentile(gen, 50), "p95": percentile(gen, 95), "mean": mean(gen)},
            "retrieval": {"p50": percentile(ret, 50), "p95": percentile(ret, 95), "mean": mean(ret)},
            "judge_per_case": {"p50": percentile(judge_lat, 50), "p95": percentile(judge_lat, 95)},
            "generation_retries": sum(r["generation"]["attempts"] - 1 for r in completed),
            "judge_retries": sum(
                sum(r.get("judge_calls", {}).values()) - len(r.get("judge_calls", {})) for r in completed
            ),
            "percentile_method": "linear interpolation (numpy 'linear' / PERCENTILE.INC)",
        },
        "tokens": {
            "generation_input": token_sum("generation", "input_tokens"),
            "generation_output": token_sum("generation", "output_tokens"),
            "generation_thinking": token_sum("generation", "thinking_tokens"),
            "judge_input": token_sum("judge", "input_tokens"),
            "judge_output": token_sum("judge", "output_tokens"),
            "judge_thinking": token_sum("judge", "thinking_tokens"),
        },
        "cost_usd": {
            "pipeline_total": sum(gen_costs) if gen_costs is not None else None,
            "mean_pipeline_cost_per_query": mean(gen_costs) if gen_costs is not None else None,
            "judge_total": sum(judge_costs) if judge_costs is not None else None,
            "run_total": (sum(gen_costs) + sum(judge_costs)) if gen_costs is not None and judge_costs is not None else None,
        },
        "by_category": dict(sorted(by_category.items())),
    }


def _count(values) -> dict:
    out: dict = {}
    for v in values:
        out[v] = out.get(v, 0) + 1
    return out


# ---------------------------------------------------------------------------
# Backwards-compatible helpers used by scripts/test_evaluator.py
# ---------------------------------------------------------------------------


def load_golden_dataset() -> list[dict]:
    """Validated golden cases as plain dicts (original return type)."""
    return [case.to_dict() for case in load_and_validate()]


def run_test_case(test_case: dict) -> dict:
    """Evaluate one case dict with the default (live) clients."""
    from app.gemini_client import get_llm_client

    case = GoldenCase(**{k: v for k, v in test_case.items() if k in GoldenCase.__dataclass_fields__})
    if not case.relevant_sources and case.should_answer and case.expected_source:
        case.relevant_sources = [case.expected_source]
    return evaluate_case(case, EvalSettings(llm=get_llm_client()))
