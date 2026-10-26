import json
import subprocess
import sys

import pytest

from app import config
from app.dataset import load_and_validate, summarize
from app.errors import DatasetValidationError
from app.metrics import check_facts, is_canonical_abstention

ORIGINAL = {
    "q001": ("How many days do customers have to request a refund?", "Customers can request a refund within 7 days of purchase.", "refund_policy.md"),
    "q002": ("How long does standard shipping take?", "Standard shipping usually takes 3 to 5 business days.", "shipping_policy.md"),
    "q003": ("How can users reset their password?", "Users can reset their password using the forgot password option on the login page.", "account_policy.md"),
}


def write(tmp_path, rows):
    path = tmp_path / "golden.jsonl"
    path.write_text("\n".join(r if isinstance(r, str) else json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    return path


def good(**overrides):
    row = {
        "id": "x1",
        "question": "Q?",
        "expected_answer": "A.",
        "expected_source": "refund_policy.md",
        "category": "refund",
        "difficulty": "easy",
    }
    row.update(overrides)
    return row


# --- the real dataset ------------------------------------------------------------

def test_real_dataset_is_valid_and_large_enough(golden_cases):
    assert len(golden_cases) >= 100
    assert len({c.id for c in golden_cases}) == len(golden_cases)


def test_original_cases_are_preserved(golden_cases):
    by_id = {c.id: c for c in golden_cases}
    for case_id, (question, answer, source) in ORIGINAL.items():
        case = by_id[case_id]
        assert (case.question, case.expected_answer, case.expected_source) == (question, answer, source)


def test_dataset_covers_required_case_types(golden_cases):
    summary = summarize(golden_cases)
    assert summary["unanswerable"] >= 20
    for tag in ("paraphrase", "numeric", "condition", "exception", "ambiguous", "false_premise", "near_miss", "multi_source", "retrieval_trap"):
        assert summary["by_tag"].get(tag, 0) >= 1, tag
    assert set(summary["by_category"]) >= {"refund", "shipping", "account", "cross_policy", "out_of_scope"}


def test_expected_answers_satisfy_their_own_fact_checks(golden_cases):
    """If a gold answer fails its own deterministic checks, the check is too brittle."""
    for case in golden_cases:
        if case.expected_facts:
            result = check_facts(case.expected_facts, case.expected_answer)
            assert result["passed"], (case.id, result["missing"])


def test_unanswerable_cases_are_consistent(golden_cases):
    for case in golden_cases:
        if not case.should_answer:
            assert case.relevant_sources == [] and case.expected_source is None and not case.expected_facts
            assert is_canonical_abstention(case.expected_answer), case.id
        else:
            assert case.expected_source in case.relevant_sources


def test_no_duplicate_questions(golden_cases):
    questions = [c.question.strip().lower() for c in golden_cases]
    assert len(set(questions)) == len(questions)


# --- validation errors -------------------------------------------------------------

def test_original_schema_rows_still_valid(tmp_path):
    cases = load_and_validate(write(tmp_path, [good()]))
    assert cases[0].should_answer is True and cases[0].relevant_sources == ["refund_policy.md"]


def test_malformed_json_reports_file_and_line(tmp_path):
    path = write(tmp_path, [good(), "{not json", good(id="x2")])
    with pytest.raises(DatasetValidationError) as exc:
        load_and_validate(path)
    assert any("golden.jsonl:2" in issue and "invalid JSON" in issue for issue in exc.value.issues)


def test_duplicate_ids(tmp_path):
    with pytest.raises(DatasetValidationError, match="duplicate id 'x1'.*line 1"):
        load_and_validate(write(tmp_path, [good(), good()]))


@pytest.mark.parametrize(
    "row,message",
    [
        ({k: v for k, v in good().items() if k != "question"}, "missing required field 'question'"),
        (good(question="  "), "'question' must be a non-empty string"),
        (good(category="billing"), "category must be one of"),
        (good(difficulty="extreme"), "difficulty must be one of"),
        (good(expected_source=None), "may only be null when should_answer is false"),
        (good(expected_source="nope.md"), "not a file in the knowledge base"),
        (good(should_answer="yes"), "'should_answer' must be true or false"),
        (good(relevant_sources=["shipping_policy.md"]), "expected_source must be included"),
        (good(relevant_sources="refund_policy.md"), "must be a list"),
        (good(expected_facts=[{"type": "duration", "min": 1}]), "expected_facts[0]"),
        (good(should_answer=False, expected_source=None, expected_facts=[{"type": "phrase", "any": ["x"]}]), "must not have expected_facts"),
        (good(tags="x"), "'tags' must be a list"),
        (good(extra=1), "unknown field"),
        ([1, 2], "must be a JSON object"),
    ],
)
def test_schema_violations(tmp_path, row, message):
    with pytest.raises(DatasetValidationError) as exc:
        load_and_validate(write(tmp_path, [row]))
    assert any(message in issue for issue in exc.value.issues), exc.value.issues


def test_all_issues_are_reported_not_just_the_first(tmp_path):
    with pytest.raises(DatasetValidationError) as exc:
        load_and_validate(write(tmp_path, [good(category="?"), good(id="x2", difficulty="?")]))
    assert len(exc.value.issues) == 2


def test_empty_and_missing_files(tmp_path):
    with pytest.raises(DatasetValidationError, match="no test cases"):
        load_and_validate(write(tmp_path, [""]))
    with pytest.raises(DatasetValidationError, match="file not found"):
        load_and_validate(tmp_path / "missing.jsonl")


def test_validate_dataset_command_exit_codes(tmp_path):
    ok = subprocess.run([sys.executable, "-m", "scripts.validate_dataset"], cwd=config.ROOT_DIR, capture_output=True, text=True)
    assert ok.returncode == 0, ok.stderr
    assert "Dataset is valid" in ok.stdout
    bad = subprocess.run(
        [sys.executable, "-m", "scripts.validate_dataset", str(write(tmp_path, ["{bad"]))],
        cwd=config.ROOT_DIR, capture_output=True, text=True,
    )
    assert bad.returncode == 2 and "golden.jsonl:1" in bad.stderr
