"""Build, write and compare evaluation reports.

Artifacts (paths from app/config.py, overridable for tests):
  reports/latest_report.json   full machine-readable report (overwritten each run)
  reports/latest_report.md     human-readable summary (overwritten each run)
  reports/history.jsonl        one summary line per run (append-only)
  reports/baseline.json        pinned baseline for the regression gate
                               (written only with --update-baseline)

Reports never contain the API key or any environment variable values.
"""

from __future__ import annotations

import hashlib
import json
import platform
import uuid
from datetime import datetime
from pathlib import Path

from app.quality_gates import EXIT_CODE_MEANINGS

REPORT_SCHEMA_VERSION = 1

LIMITATIONS = [
    "Faithfulness means 'supported by the retrieved context', not 'true in the real world'. "
    "A faithful answer can repeat an incorrect document; a true statement can be unfaithful if it was not retrieved.",
    "Correctness, relevance and faithfulness are graded by an LLM judge. Judges can be wrong, are not deterministic, "
    "and by default use the same model family as the system under test (possible self-preference bias).",
    "Deterministic fact checks only cover facts annotated in expected_facts (numbers, times, key phrases).",
    "Semantic similarity (MiniLM cosine) is a diagnostic only; it rates '30 days' and '7 days' as similar.",
    "Retrieval metrics are source-level (file name), not chunk-level.",
    "Latency percentiles from small samples are unstable; see the sample size n next to each value.",
    "Costs are estimates at paid Standard-tier list prices; free-tier usage is not billed.",
    "Generation is not deterministic, so repeated runs can differ.",
]


def new_run_id(now: datetime) -> str:
    return now.strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:6]


def compatibility_key(run_config: dict) -> str:
    """Runs are comparable only if dataset, case selection, models, prompt and index match."""
    relevant = {
        k: run_config.get(k)
        for k in (
            "dataset_fingerprint",
            "case_ids_fingerprint",
            "generation_model",
            "judge_model",
            "prompt_version",
            "top_k",
            "index_config",
        )
    }
    return hashlib.sha256(json.dumps(relevant, sort_keys=True).encode()).hexdigest()[:16]


def case_ids_fingerprint(case_ids: list[str]) -> str:
    return hashlib.sha256("\n".join(sorted(case_ids)).encode()).hexdigest()[:16]


def build_report(
    *,
    run_id: str,
    started_at: datetime,
    run_config: dict,
    dataset_summary: dict,
    run_result: dict,
    summary: dict,
    gate_results: list[dict],
    exit_code: int,
    baseline_info: dict,
    run_mode: str = "live",
) -> dict:
    counts = summary["counts"]
    complete = counts["errored"] == 0 and counts["not_run"] == 0
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "run_id": run_id,
        "run_mode": run_mode,  # "live" for real API runs; tests use "test-fake"
        "started_at": started_at.isoformat(),
        "duration_s": round(run_result["duration_s"], 3),
        "status": "completed" if complete else ("aborted" if run_result["aborted"] else "incomplete"),
        "abort_reason": run_result["abort_reason"],
        "exit_code": exit_code,
        "exit_meaning": EXIT_CODE_MEANINGS[exit_code],
        "config": {**run_config, "compatibility_key": compatibility_key(run_config)},
        "environment": {"python": platform.python_version(), "platform": platform.platform(terse=True)},
        "dataset": dataset_summary,
        "summary": summary,
        "quality_gates": gate_results,
        # Only a complete run can pass its gates; partial data never counts as a pass.
        "gates_passed": complete and all(g["status"] != "fail" for g in gate_results),
        "baseline": baseline_info,
        "infrastructure_failures": [
            {"id": r["id"], **r["error"]} for r in run_result["cases"] if r["status"] == "error"
        ],
        "failed_cases": [
            {"id": r["id"], "categories": r["failure_categories"], "reasons": r["failure_reasons"]}
            for r in run_result["cases"]
            if r["status"] == "failed"
        ],
        "limitations": LIMITATIONS,
        "cases": run_result["cases"],
    }


def history_entry(report: dict) -> dict:
    s = report["summary"]
    return {
        "run_id": report["run_id"],
        "run_mode": report["run_mode"],
        "started_at": report["started_at"],
        "status": report["status"],
        "exit_code": report["exit_code"],
        "compatibility_key": report["config"]["compatibility_key"],
        "generation_model": report["config"]["generation_model"],
        "judge_model": report["config"]["judge_model"],
        "top_k": report["config"]["top_k"],
        "counts": s["counts"],
        "metrics": {
            "pass_rate": s["answer"]["pass_rate"],
            "answer_accuracy": s["answer"]["answer_accuracy"],
            "abstention_accuracy": s["answer"]["abstention_accuracy"],
            "mean_answer_relevance": s["answer"]["mean_answer_relevance"],
            "mean_faithfulness": s["faithfulness"]["mean_faithfulness"],
            "hallucination_rate": s["faithfulness"]["hallucination_rate"],
            "source_hit_rate": s["retrieval"]["source_hit_rate"],
            "mean_source_precision": s["retrieval"]["mean_source_precision"],
            "p50_latency_ms": s["latency_ms"]["end_to_end"]["p50"],
            "p95_latency_ms": s["latency_ms"]["end_to_end"]["p95"],
            "run_cost_usd": s["cost_usd"]["run_total"],
        },
        "gates_passed": report["gates_passed"],
    }


def load_baseline(path: Path, compat_key: str) -> tuple[dict | None, str]:
    """Return (baseline, note). Baseline is None if missing or incompatible."""
    if not path.is_file():
        return None, f"no baseline file at {path.name} (create one with --update-baseline)"
    try:
        baseline = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return None, f"baseline file is not valid JSON ({exc.msg}); ignoring it"
    if baseline.get("compatibility_key") != compat_key:
        return None, "baseline is incompatible (different dataset, case selection, model, prompt or index)"
    return baseline, f"compared with baseline run {baseline.get('run_id')}"


def write_reports(report: dict, json_path: Path, md_path: Path, history_path: Path) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    md_path.write_text(render_markdown(report), encoding="utf-8")
    with open(history_path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(history_entry(report), ensure_ascii=False) + "\n")


def write_baseline(report: dict, path: Path) -> None:
    entry = history_entry(report)
    entry["compatibility_key"] = report["config"]["compatibility_key"]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(entry, indent=2), encoding="utf-8")


def read_history(path: Path) -> list[dict]:
    """Read history.jsonl, skipping (and reporting) corrupt lines rather than failing."""
    entries = []
    if not path.is_file():
        return entries
    for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            entries.append(json.loads(line))
        except json.JSONDecodeError:
            entries.append({"run_id": f"<corrupt line {i}>", "corrupt": True})
    return entries


# ---------------------------------------------------------------------------
# Markdown
# ---------------------------------------------------------------------------


def _fmt(value, kind: str = "num") -> str:
    if value is None:
        return "n/a"
    if kind == "pct":
        return f"{value * 100:.1f}%"
    if kind == "ms":
        return f"{value:,.0f} ms"
    if kind == "usd":
        return f"${value:.6f}"
    if kind == "int":
        return f"{value:,}"
    return f"{value:.3f}"


def render_markdown(report: dict) -> str:
    s = report["summary"]
    c = s["counts"]
    a, f, r, lat, tok, cost = s["answer"], s["faithfulness"], s["retrieval"], s["latency_ms"], s["tokens"], s["cost_usd"]
    cfg = report["config"]
    lines = [
        "# EvalForge evaluation report",
        "",
        f"- **Run:** `{report['run_id']}` ({report['run_mode']}), started {report['started_at']}, took {report['duration_s']:.1f} s",
        f"- **Status:** {report['status']} | **Exit code:** {report['exit_code']} ({report['exit_meaning']})",
        f"- **Model:** `{cfg['generation_model']}` | **Judge:** `{cfg['judge_model']}` | **top-k:** {cfg['top_k']} | "
        f"**Prompt:** `{cfg['prompt_version']}`",
        f"- **Index:** {json.dumps(cfg.get('index_config'))}",
        f"- **Dataset:** {report['dataset']['total']} cases in file, {c['selected']} selected "
        f"(fingerprint `{cfg['dataset_fingerprint']}`)",
    ]
    if report["abort_reason"]:
        lines.append(f"- **Aborted:** {report['abort_reason']}")
    lines += [
        "",
        "## Outcome",
        "",
        "| Selected | Completed | Passed | Failed | Errored | Not run |",
        "|---:|---:|---:|---:|---:|---:|",
        f"| {c['selected']} | {c['completed']} | {c['passed']} | {c['failed']} | {c['errored']} | {c['not_run']} |",
        "",
    ]
    if c["errored"] or c["not_run"]:
        lines += [
            "> **This run is incomplete.** Metrics below cover only completed cases and must not be read as a full "
            "evaluation.",
            "",
        ]
    lines += [
        "## Quality gates",
        "",
        "| Gate | Metric | Value | Threshold | Status | Note |",
        "|---|---|---:|---:|---|---|",
    ]
    for g in report["quality_gates"]:
        value = "n/a" if g["value"] is None else f"{g['value']:.4g}"
        lines.append(
            f"| {g['gate']} | `{g['metric']}` | {value} | {g['op']} {g['threshold']} | **{g['status'].upper()}** | {g.get('reason', '')} |"
        )
    lines += [
        "",
        "Thresholds are provisional until calibrated against a live baseline (see README).",
        "",
        "## Answer quality",
        "",
        f"- Pass rate (correct and no unsupported claims): {_fmt(a['pass_rate'], 'pct')}",
        f"- Answer accuracy (all completed cases): {_fmt(a['answer_accuracy'], 'pct')}",
        f"- Answerable accuracy: {_fmt(a['answerable_accuracy'], 'pct')} (n={a['n_answerable']})",
        f"- Abstention accuracy on unanswerable cases: {_fmt(a['abstention_accuracy'], 'pct')} (n={a['n_unanswerable']})",
        f"- Unnecessary abstention rate: {_fmt(a['unnecessary_abstention_rate'], 'pct')}",
        f"- Mean judge correctness score: {_fmt(a['mean_correctness_score'])}",
        f"- Mean answer relevance: {_fmt(a['mean_answer_relevance'])}",
        f"- Deterministic fact checks passed: {_fmt(a['deterministic_fact_pass_rate'], 'pct')} "
        f"(n={a['cases_with_fact_checks']})",
        f"- Mean semantic similarity to reference (diagnostic only): {_fmt(a['mean_semantic_similarity'])}",
        "",
        "## Faithfulness and hallucination",
        "",
        f"- Mean faithfulness (supported claims / claims): {_fmt(f['mean_faithfulness'])} "
        f"(n={f['n_scored']}; {f['n_without_claims']} answers made no claims)",
        f"- Hallucination rate (cases with any unsupported or contradicted claim): {_fmt(f['hallucination_rate'], 'pct')}",
        f"- Cases with contradicted claims: {f['cases_with_contradictions']}; with unsupported claims: "
        f"{f['cases_with_unsupported_claims']}",
        "",
        "## Retrieval (source level)",
        "",
        f"- Labeled cases: {r['n_labeled']} (unanswerable cases have no relevant source and are excluded)",
        f"- Hit rate: {_fmt(r['source_hit_rate'], 'pct')} | Precision: {_fmt(r['mean_source_precision'])} | "
        f"Recall: {_fmt(r['mean_source_recall'])} | MRR: {_fmt(r['mrr'])}",
        "",
        "## Latency",
        "",
        f"Sample size n={lat['n']}. Percentiles: {lat['percentile_method']}.",
        "",
        "| Stage | p50 | p95 | mean |",
        "|---|---:|---:|---:|",
    ]
    for stage in ("end_to_end", "retrieval", "generation"):
        row = lat[stage]
        lines.append(f"| {stage} | {_fmt(row['p50'], 'ms')} | {_fmt(row['p95'], 'ms')} | {_fmt(row['mean'], 'ms')} |")
    lines.append(
        f"| judge (eval overhead, not user-facing) | {_fmt(lat['judge_per_case']['p50'], 'ms')} | "
        f"{_fmt(lat['judge_per_case']['p95'], 'ms')} | |"
    )
    lines += [
        "",
        f"API retries: generation {lat.get('generation_retries', 'n/a')}, judge {lat.get('judge_retries', 'n/a')} "
        "(retries are usually HTTP 429 rate limiting or 503 overload). "
        f"One-off embedding-model warm-up before the run: {_fmt(cfg.get('embedding_warmup_ms'), 'ms')} "
        "(excluded from per-query latency).",
    ]
    if lat["n"] and lat["n"] < 20:
        lines += ["", f"> Only {lat['n']} samples: percentiles are preliminary."]
    lines += [
        "",
        "## Tokens and estimated cost",
        "",
        f"- Generation tokens: input {_fmt(tok['generation_input'], 'int')}, output {_fmt(tok['generation_output'], 'int')}, "
        f"thinking {_fmt(tok['generation_thinking'], 'int')}",
        f"- Judge tokens: input {_fmt(tok['judge_input'], 'int')}, output {_fmt(tok['judge_output'], 'int')}, "
        f"thinking {_fmt(tok['judge_thinking'], 'int')}",
        f"- Estimated pipeline cost: total {_fmt(cost['pipeline_total'], 'usd')}, per query "
        f"{_fmt(cost['mean_pipeline_cost_per_query'], 'usd')}",
        f"- Estimated judge cost: {_fmt(cost['judge_total'], 'usd')} | Estimated run total: {_fmt(cost['run_total'], 'usd')}",
        f"- Pricing source: {cfg.get('pricing', {}).get('source')} (verified {cfg.get('pricing', {}).get('verified_on')}); "
        "paid Standard-tier list prices, thinking tokens billed as output. 'n/a' = usage or pricing unavailable.",
        "",
        "## By category",
        "",
        "| Category | Selected | Completed | Accuracy | Pass rate |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, b in s["by_category"].items():
        lines.append(f"| {name} | {b['total']} | {b['completed']} | {_fmt(b['accuracy'], 'pct')} | {_fmt(b['pass_rate'], 'pct')} |")

    lines += ["", "## Baseline comparison", "", report["baseline"].get("note", "")]
    if report["baseline"].get("run_id"):
        lines.append(
            f"\nBaseline answer accuracy {_fmt(report['baseline'].get('answer_accuracy'), 'pct')} vs this run "
            f"{_fmt(a['answer_accuracy'], 'pct')}."
        )

    if report["infrastructure_failures"]:
        lines += ["", "## Errors (not counted as passes or quality failures)", "", "| Case | Type | Detail |", "|---|---|---|"]
        for e in report["infrastructure_failures"]:
            lines.append(f"| {e['id']} | {e['type']} | {str(e['message'])[:200].replace('|', '/')} |")

    if report["failed_cases"]:
        lines += ["", "## Failed cases", ""]
        by_id = {r["id"]: r for r in report["cases"]}
        for fc in report["failed_cases"]:
            case = by_id[fc["id"]]
            lines += [
                f"### {fc['id']} ({case['category']}, {case['difficulty']}) - {', '.join(fc['categories'])}",
                "",
                f"- **Question:** {case['question']}",
                f"- **Expected:** {case['expected_answer']}",
                f"- **Actual:** {(case['answer'] or '').strip()}",
                f"- **Retrieved sources:** {', '.join(ch['source'] for ch in case['retrieval']['chunks'])}",
            ]
            lines += [f"- {reason}" for reason in fc["reasons"]]
            lines.append("")

    lines += ["", "## Where to investigate", ""]
    lines += _recommendations(report) or ["- Nothing flagged."]
    lines += ["", "## Limitations", ""] + [f"- {item}" for item in report["limitations"]]
    return "\n".join(lines) + "\n"


def _recommendations(report: dict) -> list[str]:
    s = report["summary"]
    out = []
    counts = s["counts"]["errors_by_type"]
    if counts.get("infrastructure"):
        out.append("- API errors occurred: check Gemini status/quota, then re-run; consider GEMINI_MIN_REQUEST_INTERVAL_S.")
    if counts.get("judge_failure"):
        out.append("- The judge returned malformed output: inspect the errored cases; consider a different JUDGE_MODEL.")
    if counts.get("evaluator_error"):
        out.append("- An evaluator bug was hit: see the traceback in latest_report.json.")
    r = s["retrieval"]
    if r["source_hit_rate"] is not None and r["source_hit_rate"] < 1:
        out.append("- Some relevant sources were not retrieved: compare chunking/top-k with scripts/retrieval_experiment.py.")
    if r["mean_source_precision"] is not None and r["mean_source_precision"] < 0.5:
        out.append(
            "- Low source precision: top-k returns many irrelevant sources (with whole-document chunks, top-k = number "
            "of documents returns everything)."
        )
    a = s["answer"]
    if a["unnecessary_abstention_rate"]:
        out.append("- The model abstained on answerable questions: check whether the right chunk was retrieved.")
    if a["abstention_accuracy"] is not None and a["abstention_accuracy"] < 1:
        out.append("- The model answered some unanswerable questions: review faithfulness claims for invented details.")
    if s["faithfulness"]["hallucination_rate"]:
        out.append("- Unsupported/contradicted claims found: read the claims list per case (the judge can also be wrong).")
    return out
