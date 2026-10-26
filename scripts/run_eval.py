"""Run the live golden-dataset evaluation.

Usage (from the repository root, virtual environment active):

    python -m scripts.run_eval                    # full dataset
    python -m scripts.run_eval --limit 5          # first 5 cases (cheap smoke run)
    python -m scripts.run_eval --ids q001,q002    # specific cases
    python -m scripts.run_eval --category refund  # one category
    python -m scripts.run_eval --update-baseline  # also pin this run as the regression baseline

Requires GEMINI_API_KEY and an ingested vector store (python -m scripts.ingest).
Exit codes are documented in app/quality_gates.py and the README.
"""

from __future__ import annotations

import argparse
import sys
import time
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from app import config
from app.dataset import dataset_fingerprint, load_and_validate, summarize
from app.embedder import generate_embeddings
from app.errors import ConfigurationError, DatasetValidationError
from app.evaluator import EvalSettings, aggregate, run_evaluation
from app.gemini_client import LLMClient, get_llm_client
from app.pricing import PRICING_CURRENCY, PRICING_SOURCE_URL, PRICING_VERIFIED_ON
from app.quality_gates import (
    EXIT_CODE_MEANINGS,
    EXIT_CONFIG_OR_DATASET,
    EXIT_ENGINE_ERROR,
    EXIT_OK,
    EXIT_QUALITY_GATE_FAILED,
    determine_exit_code,
    evaluate_gates,
)
from app.rag_pipeline import PROMPT_VERSION
from app.report_generator import (
    build_report,
    case_ids_fingerprint,
    compatibility_key,
    load_baseline,
    new_run_id,
    write_baseline,
    write_reports,
)
from app.vector_store import VectorStore, get_vector_store


@dataclass
class OutputPaths:
    json: Path = config.LATEST_REPORT_JSON
    markdown: Path = config.LATEST_REPORT_MD
    history: Path = config.HISTORY_PATH
    baseline: Path = config.BASELINE_PATH


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the EvalForge golden-dataset evaluation against Gemini.")
    parser.add_argument("--dataset", type=Path, default=config.GOLDEN_DATASET_PATH, help="Path to the JSONL dataset.")
    parser.add_argument("--ids", help="Comma-separated case IDs to run.")
    parser.add_argument("--category", help="Only run cases in this category.")
    parser.add_argument("--limit", type=int, help="Run at most this many cases (after other filters).")
    parser.add_argument("--top-k", type=int, default=None, help=f"Chunks to retrieve (default {config.TOP_K}).")
    parser.add_argument("--update-baseline", action="store_true", help="Pin this run as the regression baseline.")
    parser.add_argument("--quiet", action="store_true", help="Do not print per-case progress.")
    return parser.parse_args(argv)


def select_cases(cases, args):
    selected = cases
    if args.ids:
        wanted = [i.strip() for i in args.ids.split(",") if i.strip()]
        known = {c.id for c in cases}
        unknown = [i for i in wanted if i not in known]
        if unknown:
            raise ConfigurationError(f"Unknown case id(s): {', '.join(unknown)}")
        selected = [c for c in selected if c.id in set(wanted)]
    if args.category:
        selected = [c for c in selected if c.category == args.category]
    if args.limit is not None:
        if args.limit < 1:
            raise ConfigurationError("--limit must be at least 1")
        selected = selected[: args.limit]
    if not selected:
        raise ConfigurationError("No cases match the given filters.")
    return selected


def safe_console() -> None:
    """Never crash on characters the console cannot encode (e.g. cp1252 on Windows)."""
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="replace")


def main(argv: list[str] | None = None, **kwargs) -> int:
    """Entry point. Any unexpected exception is reported as an engine error (exit 4),
    never as exit 1, which is reserved for genuine quality-gate failures."""
    safe_console()
    try:
        return _main(argv, **kwargs)
    except SystemExit:
        raise
    except Exception:  # noqa: BLE001 - last-resort handler; prints the full traceback
        traceback.print_exc()
        print(f"Exit code {EXIT_ENGINE_ERROR}: {EXIT_CODE_MEANINGS[EXIT_ENGINE_ERROR]}", file=sys.stderr)
        return EXIT_ENGINE_ERROR


def _main(
    argv: list[str] | None = None,
    *,
    llm: LLMClient | None = None,
    store: VectorStore | None = None,
    outputs: OutputPaths | None = None,
    compute_semantic_similarity: bool = True,
    run_mode: str = "live",
) -> int:
    args = parse_args(argv)
    outputs = outputs or OutputPaths()
    started_at = datetime.now(timezone.utc)

    # 1. Dataset
    try:
        all_cases = load_and_validate(args.dataset)
        cases = select_cases(all_cases, args)
    except DatasetValidationError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return EXIT_CONFIG_OR_DATASET
    except ConfigurationError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return EXIT_CONFIG_OR_DATASET

    # 2. Configuration (the key is only required when the real client will be used)
    top_k = args.top_k if args.top_k is not None else config.TOP_K
    problems = config.validate_config(require_api_key=llm is None)
    if top_k < 1:
        problems.append("--top-k must be at least 1")
    if problems:
        for p in problems:
            print(f"ERROR: {p}", file=sys.stderr)
        return EXIT_CONFIG_OR_DATASET

    # 3. Vector store must be populated
    store = store or get_vector_store()
    if store.count() == 0:
        print(
            "ERROR: the vector store is empty. Run `python -m scripts.ingest` first.",
            file=sys.stderr,
        )
        return EXIT_CONFIG_OR_DATASET

    # 4. Warm up the embedding model so its one-off load time (seconds) is not
    # counted as the first question's retrieval latency.
    warmup_start = time.perf_counter()
    try:
        generate_embeddings(["warm-up"])
    except ConfigurationError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return EXIT_CONFIG_OR_DATASET
    embedding_warmup_ms = (time.perf_counter() - warmup_start) * 1000

    try:
        llm = llm or get_llm_client()
    except ConfigurationError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return EXIT_CONFIG_OR_DATASET

    run_config = {
        "generation_model": config.GEMINI_MODEL,
        "judge_model": config.JUDGE_MODEL,
        "embedding_model": config.EMBEDDING_MODEL_NAME,
        "prompt_version": PROMPT_VERSION,
        "top_k": top_k,
        "index_config": store.metadata(),
        "dataset_path": str(Path(args.dataset).resolve().relative_to(config.ROOT_DIR))
        if Path(args.dataset).resolve().is_relative_to(config.ROOT_DIR)
        else str(args.dataset),
        "dataset_fingerprint": dataset_fingerprint(args.dataset),
        "case_ids_fingerprint": case_ids_fingerprint([c.id for c in cases]),
        "filters": {"ids": args.ids, "category": args.category, "limit": args.limit},
        "retries": {
            "max_retries": config.MAX_RETRIES,
            "base_delay_s": config.RETRY_BASE_DELAY_S,
            "max_delay_s": config.RETRY_MAX_DELAY_S,
            "min_request_interval_s": config.MIN_REQUEST_INTERVAL_S,
            "max_consecutive_failures": config.MAX_CONSECUTIVE_INFRA_FAILURES,
            "judge_max_format_retries": config.JUDGE_MAX_FORMAT_RETRIES,
        },
        "embedding_warmup_ms": round(embedding_warmup_ms, 1),  # one-off startup cost, excluded from per-query latency
        "pricing": {"source": PRICING_SOURCE_URL, "verified_on": PRICING_VERIFIED_ON, "currency": PRICING_CURRENCY},
        "quality_gates": config.QUALITY_GATES,
    }

    def progress(done: int, total: int, record: dict) -> None:
        if args.quiet:
            return
        status = record["status"].upper()
        detail = ""
        if record["status"] == "error":
            detail = f" [{record['error']['type']}] {record['error']['message'][:120]}"
        elif record["status"] == "failed":
            detail = f" {'; '.join(record['failure_reasons'])[:160]}"
        print(f"[{done}/{total}] {record['id']}: {status}{detail}", flush=True)

    print(
        f"Evaluating {len(cases)} case(s) with {config.GEMINI_MODEL} (judge {config.JUDGE_MODEL}), "
        f"top-k={top_k}, store has {store.count()} chunks.",
        flush=True,
    )
    settings = EvalSettings(
        llm=llm,
        store=store,
        top_k=top_k,
        generation_model=config.GEMINI_MODEL,
        judge_model=config.JUDGE_MODEL,
        pricing_date=started_at.date(),
        compute_semantic_similarity=compute_semantic_similarity,
        progress=progress,
    )
    run_result = run_evaluation(cases, settings)
    summary = aggregate(run_result["cases"])

    compat = compatibility_key(run_config)
    baseline, baseline_note = load_baseline(outputs.baseline, compat)
    gate_results = evaluate_gates(
        summary, config.QUALITY_GATES, baseline, baseline_note, retrieval_informative=store.count() > top_k
    )
    exit_code = determine_exit_code(summary["counts"], gate_results)

    baseline_info = {"note": baseline_note}
    if baseline:
        baseline_info.update(run_id=baseline.get("run_id"), answer_accuracy=baseline.get("metrics", {}).get("answer_accuracy"))

    report = build_report(
        run_id=new_run_id(started_at),
        started_at=started_at,
        run_config=run_config,
        dataset_summary=summarize(all_cases),
        run_result=run_result,
        summary=summary,
        gate_results=gate_results,
        exit_code=exit_code,
        baseline_info=baseline_info,
        run_mode=run_mode,
    )
    write_reports(report, outputs.json, outputs.markdown, outputs.history)

    if args.update_baseline:
        if exit_code in (EXIT_OK, EXIT_QUALITY_GATE_FAILED):
            write_baseline(report, outputs.baseline)
            print(f"Baseline updated: {outputs.baseline}")
        else:
            print("Baseline NOT updated: the run was incomplete.", file=sys.stderr)

    c = summary["counts"]
    print()
    print(
        f"Completed {c['completed']}/{c['selected']} | passed {c['passed']} | failed {c['failed']} | "
        f"errored {c['errored']} | not run {c['not_run']}"
    )
    for g in gate_results:
        value = "n/a" if g["value"] is None else f"{g['value']:.4g}"
        print(f"  gate {g['gate']:<22} {g['status'].upper():<8} value={value} {g['op']} {g['threshold']} {g.get('reason', '')}")
    print(f"Reports: {outputs.json} and {outputs.markdown}")
    print(f"Exit code {exit_code}: {EXIT_CODE_MEANINGS[exit_code]}")
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
