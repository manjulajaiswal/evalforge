"""Dashboard smoke tests with Streamlit's AppTest (renders headlessly, no browser)."""

import json

import pytest
from streamlit.testing.v1 import AppTest

from app import config
from tests.fakes import FakeLLM

APP = str(config.ROOT_DIR / "dashboard" / "streamlit_app.py")


def render(monkeypatch, reports_dir):
    monkeypatch.setenv("EVALFORGE_REPORTS_DIR", str(reports_dir))
    app = AppTest.from_file(APP, default_timeout=60)
    app.run()
    assert not app.exception, app.exception
    return app


def test_dashboard_without_reports(monkeypatch, tmp_path):
    app = render(monkeypatch, tmp_path)
    assert any("No evaluation report" in i.value for i in app.info)


def test_dashboard_with_invalid_report(monkeypatch, tmp_path):
    (tmp_path / "latest_report.json").write_text("{broken")
    app = render(monkeypatch, tmp_path)
    assert app.error


@pytest.mark.parametrize("break_answers", [False, True])
def test_dashboard_renders_real_report_structure(monkeypatch, tmp_path, ingested_store, golden_cases, break_answers):
    from scripts.run_eval import OutputPaths, main

    answers = {c.question: c.expected_answer for c in golden_cases}
    if break_answers:
        answers[golden_cases[0].question] = "30 days"
    paths = OutputPaths(tmp_path / "latest_report.json", tmp_path / "latest_report.md",
                        tmp_path / "history.jsonl", tmp_path / "baseline.json")
    for _ in range(2):
        main(["--ids", "q001,q002,q068", "--quiet"], llm=FakeLLM(answers=answers), store=ingested_store,
             outputs=paths, run_mode="test-fake")
    app = render(monkeypatch, tmp_path)
    assert any("not by a live model run" in w.value for w in app.warning)  # fake runs are labelled
    assert len(app.dataframe) >= 2


def test_dashboard_tolerates_missing_optional_sections(monkeypatch, tmp_path):
    (tmp_path / "latest_report.json").write_text(json.dumps({"run_id": "x", "run_mode": "live", "status": "incomplete"}))
    render(monkeypatch, tmp_path)
