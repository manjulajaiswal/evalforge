"""Quality gates and exit-code policy.

Gate statuses
  pass     metric available and within threshold
  fail     metric outside threshold
  fail     metric unavailable for an enabled gate ("missing metrics never pass")
  skipped  gate does not apply, with an explicit reason (only used for the
           regression gate when no compatible baseline exists, and for gates
           whose population is empty by design, e.g. abstention accuracy on a
           run that selected no unanswerable cases)

Exit codes (scripts/run_eval.py)
  0  evaluation completed and every gate passed
  1  evaluation completed and at least one quality gate failed
  2  invalid configuration or dataset (nothing, or not everything, could run)
  3  incomplete: external-service failures (Gemini API errors, malformed
     judge output) prevented some cases from being evaluated
  4  evaluation-engine failure (unexpected exception in EvalForge code)
"""

from __future__ import annotations

EXIT_OK = 0
EXIT_QUALITY_GATE_FAILED = 1
EXIT_CONFIG_OR_DATASET = 2
EXIT_INCOMPLETE_EXTERNAL = 3
EXIT_ENGINE_ERROR = 4

EXIT_CODE_MEANINGS = {
    EXIT_OK: "completed; all quality gates passed",
    EXIT_QUALITY_GATE_FAILED: "completed; one or more quality gates failed",
    EXIT_CONFIG_OR_DATASET: "invalid configuration or dataset",
    EXIT_INCOMPLETE_EXTERNAL: "incomplete; external service failures (API errors or malformed judge output)",
    EXIT_ENGINE_ERROR: "evaluation engine error (bug in EvalForge)",
}

# Metrics that are legitimately undefined when the run contains no cases of
# the relevant kind (keyed by metric name).
_POPULATION = {
    "abstention_accuracy": ("answer", "n_unanswerable"),
    "source_hit_rate": ("retrieval", "n_labeled"),
    "mean_faithfulness": ("faithfulness", "n_scored"),  # e.g. a run of only correct abstentions
}


def flatten_metrics(summary: dict) -> dict:
    """Map gate metric names to values from an aggregate() summary."""
    return {
        "answer_accuracy": summary["answer"]["answer_accuracy"],
        "mean_answer_relevance": summary["answer"]["mean_answer_relevance"],
        "mean_faithfulness": summary["faithfulness"]["mean_faithfulness"],
        "hallucination_rate": summary["faithfulness"]["hallucination_rate"],
        "abstention_accuracy": summary["answer"]["abstention_accuracy"],
        "source_hit_rate": summary["retrieval"]["source_hit_rate"],
        "p95_end_to_end_latency_ms": summary["latency_ms"]["end_to_end"]["p95"],
        "mean_pipeline_cost_per_query_usd": summary["cost_usd"]["mean_pipeline_cost_per_query"],
    }


def evaluate_gates(
    summary: dict,
    gates: dict[str, dict],
    baseline: dict | None = None,
    baseline_note: str = "",
    retrieval_informative: bool = True,
) -> list[dict]:
    """Evaluate each gate.

    retrieval_informative=False (top-k >= chunks in the store, so every source
    is always retrieved) marks source-level retrieval gates as skipped with
    that reason instead of reporting a meaningless pass.
    """
    metrics = flatten_metrics(summary)
    if baseline is not None and metrics["answer_accuracy"] is not None:
        base_acc = baseline.get("metrics", {}).get("answer_accuracy")
        metrics["answer_accuracy_drop_vs_baseline"] = (
            None if base_acc is None else base_acc - metrics["answer_accuracy"]
        )

    results = []
    for name, gate in gates.items():
        metric = gate["metric"]
        entry = {"gate": name, "metric": metric, "op": gate["op"], "threshold": gate["threshold"], "value": None}

        if metric == "answer_accuracy_drop_vs_baseline" and baseline is None:
            entry.update(status="skipped", reason=baseline_note or "no compatible baseline")
            results.append(entry)
            continue
        if metric == "source_hit_rate" and not retrieval_informative:
            entry.update(
                value=metrics.get(metric),
                status="skipped",
                reason="non-informative: top-k >= chunks in the store, so every source is always retrieved",
            )
            results.append(entry)
            continue
        if metric in _POPULATION:
            section, key = _POPULATION[metric]
            if summary[section][key] == 0 and summary["counts"]["completed"] > 0:
                entry.update(status="skipped", reason=f"no applicable cases in this run ({key}=0)")
                results.append(entry)
                continue

        value = metrics.get(metric)
        entry["value"] = value
        if value is None:
            entry.update(status="fail", reason="metric unavailable (missing metrics never pass)")
        elif (gate["op"] == ">=" and value >= gate["threshold"]) or (gate["op"] == "<=" and value <= gate["threshold"]):
            entry.update(status="pass", reason="")
        else:
            entry.update(status="fail", reason=f"{value:.4g} is not {gate['op']} {gate['threshold']}")
        results.append(entry)
    return results


def determine_exit_code(counts: dict, gate_results: list[dict]) -> int:
    errors = counts.get("errors_by_type", {})
    if errors.get("evaluator_error"):
        return EXIT_ENGINE_ERROR
    if errors.get("configuration"):
        return EXIT_CONFIG_OR_DATASET
    if counts.get("errored") or counts.get("not_run"):
        return EXIT_INCOMPLETE_EXTERNAL
    if any(g["status"] == "fail" for g in gate_results):
        return EXIT_QUALITY_GATE_FAILED
    return EXIT_OK
