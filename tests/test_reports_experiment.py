import json
from datetime import datetime, timezone

from app.evaluator import aggregate
from app.quality_gates import evaluate_gates
from app.report_generator import (
    build_report,
    compatibility_key,
    history_entry,
    load_baseline,
    read_history,
    render_markdown,
    write_baseline,
)
from scripts import retrieval_experiment


def empty_report(**overrides):
    summary = aggregate([])
    report = build_report(
        run_id="r1", started_at=datetime(2026, 10, 26, tzinfo=timezone.utc),
        run_config={"generation_model": "m", "judge_model": "m", "prompt_version": "p", "top_k": 3,
                    "dataset_fingerprint": "d", "case_ids_fingerprint": "c", "index_config": {}},
        dataset_summary={"total": 0}, run_result={"cases": [], "aborted": True, "abort_reason": "x", "duration_s": 0.0},
        summary=summary, gate_results=evaluate_gates(summary, {"a": {"metric": "answer_accuracy", "op": ">=", "threshold": 0.8}}),
        exit_code=3, baseline_info={"note": "none"}, run_mode="test-fake",
    )
    report.update(overrides)
    return report


def test_markdown_renders_with_all_metrics_missing():
    md = render_markdown(empty_report())
    assert "n/a" in md and "**FAIL**" in md and "metric unavailable" in md


def test_history_entry_is_json_serialisable_and_compact():
    entry = history_entry(empty_report())
    assert json.loads(json.dumps(entry))["run_id"] == "r1" and "cases" not in entry


def test_read_history_tolerates_corrupt_lines(tmp_path):
    path = tmp_path / "h.jsonl"
    path.write_text('{"run_id": "a"}\nnot json\n\n{"run_id": "b"}\n')
    entries = read_history(path)
    assert [e["run_id"] for e in entries] == ["a", "<corrupt line 2>", "b"]
    assert read_history(tmp_path / "missing.jsonl") == []


def test_compatibility_key_changes_with_model_or_dataset():
    base = {"generation_model": "m", "dataset_fingerprint": "d", "top_k": 3}
    assert compatibility_key(base) == compatibility_key(dict(base))
    assert compatibility_key(base) != compatibility_key({**base, "generation_model": "other"})
    assert compatibility_key(base) != compatibility_key({**base, "dataset_fingerprint": "e"})


def test_baseline_roundtrip_and_incompatibility(tmp_path):
    report = empty_report()
    report["config"]["compatibility_key"] = "k1"
    write_baseline(report, tmp_path / "b.json")
    assert load_baseline(tmp_path / "b.json", "k1")[0]["run_id"] == "r1"
    baseline, note = load_baseline(tmp_path / "b.json", "k2")
    assert baseline is None and "incompatible" in note
    (tmp_path / "bad.json").write_text("{")
    assert load_baseline(tmp_path / "bad.json", "k1")[0] is None


def test_retrieval_experiment_compares_configurations(golden_cases):
    configs = [
        {"strategy": "character", "chunk_size": 500, "overlap": 50, "top_k": 3},
        {"strategy": "paragraph", "chunk_size": 500, "overlap": 0, "top_k": 1},
    ]
    report = retrieval_experiment.run(configs, golden_cases)
    whole_docs, paragraphs = report["results"]
    assert whole_docs["chunks_in_store"] == 3 and paragraphs["chunks_in_store"] == 12
    assert whole_docs["is_baseline"] is True
    # With 3 whole-document chunks and k=3 everything is retrieved: recall is
    # trivially 1.0 and precision is ~1/3. The numbers come from the fake
    # embedder here, so only these structural facts are asserted.
    assert whole_docs["recall"] == 1.0 and whole_docs["precision"] < 0.5
    assert whole_docs["n_cases"] == paragraphs["n_cases"] == sum(c.should_answer for c in golden_cases)
