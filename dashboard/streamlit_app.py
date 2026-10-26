"""EvalForge dashboard: a read-only view of the latest report and run history.

Start (repository root, virtual environment active):
    streamlit run dashboard/streamlit_app.py

It only displays what is in reports/latest_report.json and
reports/history.jsonl; it never runs evaluations or invents data.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:  # `streamlit run` puts dashboard/ on sys.path, not the repo root
    sys.path.insert(0, str(ROOT))

from app import config  # noqa: E402
from app.report_generator import read_history  # noqa: E402

REPORTS_DIR = Path(os.getenv("EVALFORGE_REPORTS_DIR", str(config.REPORTS_DIR)))


def fmt(value, kind="num"):
    if value is None:
        return "n/a"
    if kind == "pct":
        return f"{value * 100:.1f}%"
    if kind == "ms":
        return f"{value:,.0f} ms"
    if kind == "usd":
        return f"${value:.5f}"
    return f"{value:.3f}"


def load_report(path: Path):
    if not path.is_file():
        return None, None
    try:
        return json.loads(path.read_text(encoding="utf-8")), None
    except json.JSONDecodeError as exc:
        return None, f"{path.name} is not valid JSON: {exc.msg}"


st.set_page_config(page_title="EvalForge", layout="wide")
st.title("EvalForge evaluation dashboard")

report, error = load_report(REPORTS_DIR / "latest_report.json")
if error:
    st.error(error)
    st.stop()
if report is None:
    st.info(
        "No evaluation report found yet. Run `python -m scripts.ingest` and then "
        "`python -m scripts.run_eval` to create reports/latest_report.json."
    )
    st.stop()

summary = report.get("summary", {})
counts = summary.get("counts", {})
answer = summary.get("answer", {})
faith = summary.get("faithfulness", {})
retrieval = summary.get("retrieval", {})
latency = summary.get("latency_ms", {})
cost = summary.get("cost_usd", {})
cfg = report.get("config", {})

if report.get("run_mode") != "live":
    st.warning(f"This report was produced in '{report.get('run_mode')}' mode, not by a live model run. "
               "Its numbers say nothing about real model quality.")

status = report.get("status", "unknown")
st.caption(
    f"Run `{report.get('run_id')}` started {report.get('started_at')} | model `{cfg.get('generation_model')}` | "
    f"judge `{cfg.get('judge_model')}` | top-k {cfg.get('top_k')} | dataset {report.get('dataset', {}).get('total')} cases"
)
headline = f"Status: **{status}**, exit code {report.get('exit_code')} ({report.get('exit_meaning', '')})"
if status == "completed" and report.get("gates_passed"):
    st.success(headline)
elif status == "completed":
    st.error(headline + " - one or more quality gates failed")
else:
    st.error(headline + " - the run is incomplete; metrics cover completed cases only")
if report.get("abort_reason"):
    st.write(f"Aborted: {report['abort_reason']}")

cols = st.columns(5)
for col, (label, key) in zip(cols, [("Selected", "selected"), ("Completed", "completed"), ("Passed", "passed"),
                                    ("Failed", "failed"), ("Errored / not run", None)], strict=True):
    value = counts.get(key) if key else f"{counts.get('errored', 0)} / {counts.get('not_run', 0)}"
    col.metric(label, value if value is not None else "n/a")

st.subheader("Quality")
cols = st.columns(4)
cols[0].metric("Answer accuracy", fmt(answer.get("answer_accuracy"), "pct"))
cols[1].metric("Abstention accuracy", fmt(answer.get("abstention_accuracy"), "pct"), help=f"n={answer.get('n_unanswerable')}")
cols[2].metric("Mean faithfulness", fmt(faith.get("mean_faithfulness")), help=f"n={faith.get('n_scored')}")
cols[3].metric("Hallucination rate", fmt(faith.get("hallucination_rate"), "pct"))
cols = st.columns(4)
cols[0].metric("Answer relevance", fmt(answer.get("mean_answer_relevance")))
cols[1].metric("Source hit rate", fmt(retrieval.get("source_hit_rate"), "pct"), help=f"n={retrieval.get('n_labeled')}")
cols[2].metric("Source precision", fmt(retrieval.get("mean_source_precision")))
cols[3].metric("Fact checks passed", fmt(answer.get("deterministic_fact_pass_rate"), "pct"))

st.subheader("Latency and cost")
e2e = latency.get("end_to_end", {}) or {}
cols = st.columns(4)
cols[0].metric("p50 end-to-end", fmt(e2e.get("p50"), "ms"), help=f"n={latency.get('n')}")
cols[1].metric("p95 end-to-end", fmt(e2e.get("p95"), "ms"), help=f"n={latency.get('n')}")
cols[2].metric("Est. cost per query", fmt(cost.get("mean_pipeline_cost_per_query"), "usd"))
cols[3].metric("Est. run cost (incl. judge)", fmt(cost.get("run_total"), "usd"))
if latency.get("n") and latency["n"] < 20:
    st.caption(f"Only {latency['n']} latency samples: percentiles are preliminary.")
st.caption("Costs are estimates at paid Standard-tier list prices; 'n/a' means usage or pricing was unavailable.")

st.subheader("Quality gates")
gates = report.get("quality_gates", [])
if gates:
    st.dataframe(pd.DataFrame(gates)[["gate", "metric", "value", "op", "threshold", "status", "reason"]], hide_index=True)
else:
    st.write("No gate results in this report.")

by_category = summary.get("by_category", {})
if by_category:
    st.subheader("By category")
    st.dataframe(pd.DataFrame.from_dict(by_category, orient="index"))

errors = report.get("infrastructure_failures", [])
if errors:
    st.subheader("Errors (not counted as passes or quality failures)")
    st.dataframe(pd.DataFrame(errors)[["id", "type", "message"]], hide_index=True)

failed = report.get("failed_cases", [])
st.subheader(f"Failed cases ({len(failed)})")
cases_by_id = {c["id"]: c for c in report.get("cases", [])}
for item in failed:
    case = cases_by_id.get(item["id"], {})
    with st.expander(f"{item['id']}: {case.get('question', '')}"):
        st.write("**Expected:**", case.get("expected_answer"))
        st.write("**Actual:**", case.get("answer"))
        chunks = (case.get("retrieval") or {}).get("chunks", [])
        st.write("**Retrieved sources:**", ", ".join(c["source"] for c in chunks) or "none")
        for reason in item.get("reasons", []):
            st.write("-", reason)

st.subheader("History")
history = [h for h in read_history(REPORTS_DIR / "history.jsonl") if not h.get("corrupt")]
key = cfg.get("compatibility_key")
comparable = [h for h in history if h.get("compatibility_key") == key and h.get("run_mode") == report.get("run_mode")]
if len(comparable) >= 2:
    trend = pd.DataFrame(
        [{"run": h["started_at"], **{k: h["metrics"].get(k) for k in ("answer_accuracy", "hallucination_rate", "source_hit_rate")}}
         for h in comparable]
    ).set_index("run")
    st.line_chart(trend)
    st.caption(f"{len(comparable)} comparable runs (same dataset, cases, models, prompt and index).")
else:
    st.write("Not enough comparable runs for a trend yet (need 2 with the same configuration).")
if history:
    st.dataframe(
        pd.DataFrame([{"run_id": h["run_id"], "mode": h.get("run_mode"), "status": h.get("status"),
                       "exit": h.get("exit_code"), "accuracy": h.get("metrics", {}).get("answer_accuracy"),
                       "gates_passed": h.get("gates_passed")} for h in history]),
        hide_index=True,
    )
