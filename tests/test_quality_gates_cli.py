"""Quality gates, exit codes, reports and the controlled regression demonstration."""

import copy
import json
import os
import subprocess
import sys

import pytest

from app import config
from app.errors import ExternalServiceError
from app.quality_gates import (
    EXIT_CONFIG_OR_DATASET,
    EXIT_ENGINE_ERROR,
    EXIT_INCOMPLETE_EXTERNAL,
    EXIT_OK,
    EXIT_QUALITY_GATE_FAILED,
    determine_exit_code,
    evaluate_gates,
)
from app.report_generator import read_history
from scripts.run_eval import main
from tests.fakes import FakeLLM

SMALL = ["--ids", "q001,q002,q003,q068,q069", "--quiet"]


def run(argv, llm, store, paths, **kw):
    return main(argv, llm=llm, store=store, outputs=paths, run_mode="test-fake", **kw)


# --- gate logic --------------------------------------------------------------------

def summary_with(**metrics):
    base = {
        "counts": {"completed": 10, "errored": 0, "not_run": 0},
        "answer": {"answer_accuracy": 0.9, "mean_answer_relevance": 0.9, "abstention_accuracy": 1.0, "n_unanswerable": 2},
        "faithfulness": {"mean_faithfulness": 0.95, "hallucination_rate": 0.0, "n_scored": 8},
        "retrieval": {"source_hit_rate": 1.0, "n_labeled": 8},
        "latency_ms": {"end_to_end": {"p95": 1200.0}},
        "cost_usd": {"mean_pipeline_cost_per_query": 0.001},
    }
    for path, value in metrics.items():
        section, key = path.split("__")
        if section == "latency_ms":
            base[section]["end_to_end"][key] = value
        else:
            base[section][key] = value
    return base


def gate_status(results, name):
    return next(g for g in results if g["gate"] == name)


def test_all_gates_pass_on_good_metrics():
    results = evaluate_gates(summary_with(), config.QUALITY_GATES)
    assert all(g["status"] in ("pass", "skipped") for g in results)
    assert gate_status(results, "accuracy_regression")["status"] == "skipped"


def test_threshold_violation_fails_gate():
    results = evaluate_gates(summary_with(faithfulness__hallucination_rate=0.5), config.QUALITY_GATES)
    assert gate_status(results, "hallucination_rate")["status"] == "fail"


def test_unavailable_metric_never_passes():
    results = evaluate_gates(summary_with(cost_usd__mean_pipeline_cost_per_query=None), config.QUALITY_GATES)
    g = gate_status(results, "cost_per_query")
    assert g["status"] == "fail" and "unavailable" in g["reason"]


def test_gate_is_skipped_only_when_population_is_empty():
    results = evaluate_gates(summary_with(answer__n_unanswerable=0, answer__abstention_accuracy=None), config.QUALITY_GATES)
    assert gate_status(results, "abstention_accuracy")["status"] == "skipped"


def test_faithfulness_gate_skipped_when_no_answer_made_claims():
    s = summary_with(faithfulness__mean_faithfulness=None, faithfulness__n_scored=0)
    assert gate_status(evaluate_gates(s, config.QUALITY_GATES), "faithfulness")["status"] == "skipped"


def test_source_hit_rate_marked_non_informative_when_top_k_covers_store():
    g = gate_status(evaluate_gates(summary_with(), config.QUALITY_GATES, retrieval_informative=False), "source_hit_rate")
    assert g["status"] == "skipped" and "non-informative" in g["reason"]


def test_regression_gate_against_baseline():
    baseline = {"metrics": {"answer_accuracy": 0.95}}
    ok = evaluate_gates(summary_with(answer__answer_accuracy=0.92), config.QUALITY_GATES, baseline)
    bad = evaluate_gates(summary_with(answer__answer_accuracy=0.80), config.QUALITY_GATES, baseline)
    assert gate_status(ok, "accuracy_regression")["status"] == "pass"
    assert gate_status(bad, "accuracy_regression")["status"] == "fail"


@pytest.mark.parametrize(
    "counts,gate_fail,expected",
    [
        ({"errored": 0, "not_run": 0, "errors_by_type": {}}, False, EXIT_OK),
        ({"errored": 0, "not_run": 0, "errors_by_type": {}}, True, EXIT_QUALITY_GATE_FAILED),
        ({"errored": 1, "not_run": 0, "errors_by_type": {"infrastructure": 1}}, False, EXIT_INCOMPLETE_EXTERNAL),
        ({"errored": 1, "not_run": 0, "errors_by_type": {"judge_failure": 1}}, True, EXIT_INCOMPLETE_EXTERNAL),
        ({"errored": 1, "not_run": 3, "errors_by_type": {"configuration": 1}}, False, EXIT_CONFIG_OR_DATASET),
        ({"errored": 1, "not_run": 0, "errors_by_type": {"evaluator_error": 1, "infrastructure": 2}}, True, EXIT_ENGINE_ERROR),
    ],
)
def test_exit_code_precedence(counts, gate_fail, expected):
    gates = [{"status": "fail" if gate_fail else "pass"}]
    assert determine_exit_code(counts, gates) == expected


# --- CLI end to end (fakes) ---------------------------------------------------------

def test_cli_success_writes_reports_and_history(oracle_llm, ingested_store, report_paths):
    code = run(SMALL, oracle_llm, ingested_store, report_paths)
    report = json.loads(report_paths.json.read_text())
    assert report["run_mode"] == "test-fake" and report["status"] == "completed"
    assert report["summary"]["counts"]["completed"] == 5 and len(report["cases"]) == 5
    assert report["dataset"]["total"] >= 100
    # The oracle answers correctly; the fake embedder + whole-document chunks give
    # source precision 1/3, so every gate except cost/latency is evaluated on real logic.
    assert code == (EXIT_OK if report["gates_passed"] else EXIT_QUALITY_GATE_FAILED)
    md = report_paths.markdown.read_text()
    assert "# EvalForge evaluation report" in md and "Quality gates" in md and "Limitations" in md
    history = read_history(report_paths.history)
    assert len(history) == 1 and history[0]["run_id"] == report["run_id"]


def test_controlled_regression_demo(oracle_llm, ingested_store, report_paths, golden_cases):
    """1) known-good run -> 0; 2) inject a regression -> gate fails, exit 1; 3) restore -> 0."""
    assert run(SMALL + ["--update-baseline"], oracle_llm, ingested_store, report_paths) in (EXIT_OK, EXIT_QUALITY_GATE_FAILED)
    good = json.loads(report_paths.json.read_text())
    good_gate_failures = {g["gate"] for g in good["quality_gates"] if g["status"] == "fail"}

    # Regression: the model starts claiming a 30-day refund window.
    broken_answers = {c.question: c.expected_answer for c in golden_cases}
    broken_answers[golden_cases[0].question] = "Customers can request a refund within 30 days of purchase."
    broken_answers[golden_cases[1].question] = "Standard shipping takes 10 business days."
    code = run(SMALL, FakeLLM(answers=broken_answers), ingested_store, report_paths)
    broken = json.loads(report_paths.json.read_text())
    assert code == EXIT_QUALITY_GATE_FAILED
    failed = {g["gate"] for g in broken["quality_gates"] if g["status"] == "fail"}
    assert {"answer_accuracy", "accuracy_regression"} <= failed
    assert broken["summary"]["counts"]["failed"] == 2

    # Restore -> back to the known-good outcome.
    assert run(SMALL, oracle_llm, ingested_store, report_paths) == (EXIT_OK if not good_gate_failures else EXIT_QUALITY_GATE_FAILED)
    restored = json.loads(report_paths.json.read_text())
    assert gate_status(restored["quality_gates"], "accuracy_regression")["status"] == "pass"
    assert len(read_history(report_paths.history)) == 3


def test_tightened_gate_fails_and_restored_gate_passes(oracle_llm, ingested_store, report_paths, monkeypatch):
    gates = copy.deepcopy(config.QUALITY_GATES)
    for name in ("cost_per_query", "p95_latency", "source_hit_rate"):
        gates.pop(name)
    monkeypatch.setattr("app.config.QUALITY_GATES", gates)
    assert run(SMALL, oracle_llm, ingested_store, report_paths) == EXIT_OK
    strict = copy.deepcopy(gates)
    strict["answer_relevance"]["threshold"] = 0.99  # fake judge reports 0.9
    monkeypatch.setattr("app.config.QUALITY_GATES", strict)
    assert run(SMALL, oracle_llm, ingested_store, report_paths) == EXIT_QUALITY_GATE_FAILED
    monkeypatch.setattr("app.config.QUALITY_GATES", gates)
    assert run(SMALL, oracle_llm, ingested_store, report_paths) == EXIT_OK


def test_cli_api_outage_is_reported_as_incomplete_not_passed(ingested_store, report_paths):
    outage = ExternalServiceError("HTTP 503 UNAVAILABLE after 5 attempt(s)", transient=True, status_code=503, attempts=5)
    code = run(SMALL, FakeLLM(answer_fn=lambda q, p: outage), ingested_store, report_paths)
    report = json.loads(report_paths.json.read_text())
    assert code == EXIT_INCOMPLETE_EXTERNAL
    assert report["status"] == "aborted" and report["summary"]["counts"]["passed"] == 0
    assert all(g["status"] != "pass" for g in report["quality_gates"] if g["metric"] != "answer_accuracy_drop_vs_baseline")
    assert "incomplete" in report_paths.markdown.read_text().lower()


def test_incomplete_run_never_reports_gates_passed(ingested_store, report_paths):
    outage = ExternalServiceError("down", transient=True, status_code=503)
    run(SMALL, FakeLLM(answer_fn=lambda q, p: outage), ingested_store, report_paths)
    report = json.loads(report_paths.json.read_text())
    assert report["gates_passed"] is False and read_history(report_paths.history)[-1]["gates_passed"] is False


def test_unexpected_exception_exits_4_not_1(oracle_llm, ingested_store, report_paths, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("disk full")

    monkeypatch.setattr("scripts.run_eval.write_reports", boom)
    assert run(SMALL, oracle_llm, ingested_store, report_paths) == EXIT_ENGINE_ERROR


def test_unencodable_characters_do_not_crash_on_windows_consoles(ingested_store, report_paths, tmp_path):
    """Reproduces the cp1252 crash: run the CLI in a subprocess with a cp1252 stdout."""
    script = tmp_path / "probe.py"
    script.write_text(
        "import sys\n"
        "from app import embedder\n"
        "from app.vector_store import make_in_memory_store\n"
        "from app.ingestion import ingest_knowledge_base\n"
        "from tests.fakes import FakeEmbedder, FakeLLM\n"
        "from scripts.run_eval import main, OutputPaths\n"
        "from pathlib import Path\n"
        "embedder.set_embedding_model(FakeEmbedder())\n"
        "s = make_in_memory_store(); ingest_knowledge_base(s)\n"
        "llm = FakeLLM(answer_fn=lambda q, p: 'Refunds → 7 days ≠ 30 days',\n"
        "              correct_fn=lambda *a: {'verdict': 'incorrect', 'correctness_score': 0.0, 'relevance_score': 0.5,\n"
        "                                     'abstained': False, 'issues': ['7 ≠ 30 → wrong'], 'reasoning': '≠'})\n"
        f"d = Path({str(tmp_path)!r})\n"
        "sys.exit(main(['--ids', 'q001'], llm=llm, store=s, outputs=OutputPaths(d/'r.json', d/'r.md', d/'h.jsonl', d/'b.json'), run_mode='test-fake'))\n",
        encoding="utf-8",
    )
    env = {**os.environ, "PYTHONIOENCODING": "cp1252", "GEMINI_API_KEY": "", "PYTHONPATH": str(config.ROOT_DIR)}
    proc = subprocess.run([sys.executable, str(script)], cwd=config.ROOT_DIR, env=env, capture_output=True, timeout=120)
    assert proc.returncode == EXIT_QUALITY_GATE_FAILED, proc.stderr.decode("utf-8", "replace")
    assert (tmp_path / "r.json").exists()


def test_cli_rejects_empty_store(oracle_llm, store, report_paths, capsys):
    assert run(SMALL, oracle_llm, store, report_paths) == EXIT_CONFIG_OR_DATASET
    assert "scripts.ingest" in capsys.readouterr().err
    assert not report_paths.json.exists()


def test_cli_rejects_unknown_ids_and_bad_dataset(oracle_llm, ingested_store, report_paths, tmp_path):
    assert run(["--ids", "nope"], oracle_llm, ingested_store, report_paths) == EXIT_CONFIG_OR_DATASET
    bad = tmp_path / "bad.jsonl"
    bad.write_text("{oops\n")
    assert run(["--dataset", str(bad)], oracle_llm, ingested_store, report_paths) == EXIT_CONFIG_OR_DATASET


def test_baseline_not_updated_from_incomplete_run(ingested_store, report_paths):
    outage = ExternalServiceError("down", transient=True, status_code=503)
    run(SMALL + ["--update-baseline"], FakeLLM(answer_fn=lambda q, p: outage), ingested_store, report_paths)
    assert not report_paths.baseline.exists()


def test_reports_never_contain_the_api_key(oracle_llm, ingested_store, report_paths, monkeypatch):
    secret = "AIza-test-secret-value-123"
    monkeypatch.setenv("GEMINI_API_KEY", secret)
    monkeypatch.setattr("app.config.GEMINI_API_KEY", secret)
    run(SMALL, oracle_llm, ingested_store, report_paths)
    for path in (report_paths.json, report_paths.markdown, report_paths.history):
        assert secret not in path.read_text()


# --- real process exit codes ---------------------------------------------------------

@pytest.mark.parametrize(
    "env_overrides,message",
    [
        ({"GEMINI_API_KEY": ""}, "GEMINI_API_KEY is not set"),
        ({"GEMINI_API_KEY": "your_gemini_api_key_here"}, "placeholder"),
        ({"GEMINI_API_KEY": "x" * 30, "TOP_K": "abc"}, "TOP_K must be an integer"),
    ],
)
def test_run_eval_process_config_errors_exit_2(env_overrides, message):
    env = {**os.environ, **env_overrides}
    proc = subprocess.run([sys.executable, "-m", "scripts.run_eval", "--limit", "1"], cwd=config.ROOT_DIR,
                          env=env, capture_output=True, text=True, timeout=120)
    assert proc.returncode == EXIT_CONFIG_OR_DATASET, proc.stderr
    assert message in proc.stderr


def test_run_eval_process_without_api_key_exits_2():
    env = {**os.environ, "GEMINI_API_KEY": ""}  # empty value also overrides any local .env
    proc = subprocess.run([sys.executable, "-m", "scripts.run_eval", "--limit", "1"], cwd=config.ROOT_DIR,
                          env=env, capture_output=True, text=True, timeout=120)
    assert proc.returncode == EXIT_CONFIG_OR_DATASET, proc.stderr
    assert "GEMINI_API_KEY is not set" in proc.stderr
