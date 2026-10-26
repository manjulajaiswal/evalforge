"""Central configuration for EvalForge.

Every path is absolute and derived from the repository root, so commands
behave the same no matter which directory they are launched from.

Values that a user may reasonably change per environment (model names, retry
settings, top-k) can be overridden with environment variables or a local
``.env`` file. Secrets (the API key) are only ever read from the environment.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parent.parent

# Load <repo>/.env explicitly (not the current working directory's .env).
load_dotenv(ROOT_DIR / ".env")


# Problems found while parsing environment variables. They are not raised at
# import time (that would crash every command); validate_config() reports them.
_ENV_ERRORS: list[str] = []


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw)
    except ValueError:
        _ENV_ERRORS.append(f"Environment variable {name} must be an integer, got {raw!r}.")
        return default


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return float(raw)
    except ValueError:
        _ENV_ERRORS.append(f"Environment variable {name} must be a number, got {raw!r}.")
        return default


# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
DATA_DIR = ROOT_DIR / "data"
KNOWLEDGE_BASE_DIR = DATA_DIR / "knowledge_base"
GOLDEN_DATASET_PATH = DATA_DIR / "golden_dataset.jsonl"

REPORTS_DIR = ROOT_DIR / "reports"
LATEST_REPORT_JSON = REPORTS_DIR / "latest_report.json"
LATEST_REPORT_MD = REPORTS_DIR / "latest_report.md"
HISTORY_PATH = REPORTS_DIR / "history.jsonl"
BASELINE_PATH = REPORTS_DIR / "baseline.json"

# Single source of truth for the vector database location. Ingestion,
# retrieval and evaluation all use this path.
CHROMA_DIR = ROOT_DIR / ".chroma"
CHROMA_COLLECTION_NAME = "evalforge_knowledge"

# --------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# gemini-3.6-flash: stable per https://ai.google.dev/gemini-api/docs/models/gemini-3.6-flash
# (checked 2026-10-26). gemini-3.5-flash is deprecated and routed to 3.6.
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.6-flash")

# Model used as LLM-as-a-judge. Defaults to the generation model.
JUDGE_MODEL = os.getenv("JUDGE_MODEL", GEMINI_MODEL)

EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

# --------------------------------------------------------------------------
# Retrieval / chunking (the baseline configuration of the system under test)
# --------------------------------------------------------------------------
TOP_K = _env_int("TOP_K", 3)
CHUNK_STRATEGY = os.getenv("CHUNK_STRATEGY", "character")  # "character" | "paragraph"
CHUNK_SIZE = _env_int("CHUNK_SIZE", 500)
CHUNK_OVERLAP = _env_int("CHUNK_OVERLAP", 50)

# --------------------------------------------------------------------------
# API reliability
# --------------------------------------------------------------------------
MAX_RETRIES = _env_int("GEMINI_MAX_RETRIES", 4)  # retries after the first attempt
RETRY_BASE_DELAY_S = _env_float("GEMINI_RETRY_BASE_DELAY_S", 2.0)
RETRY_MAX_DELAY_S = _env_float("GEMINI_RETRY_MAX_DELAY_S", 60.0)
REQUEST_TIMEOUT_S = _env_float("GEMINI_REQUEST_TIMEOUT_S", 60.0)
# Minimum spacing between API requests (helps stay under free-tier RPM limits).
MIN_REQUEST_INTERVAL_S = _env_float("GEMINI_MIN_REQUEST_INTERVAL_S", 0.0)
# Abort the run after this many consecutive cases fail for infrastructure reasons.
MAX_CONSECUTIVE_INFRA_FAILURES = _env_int("MAX_CONSECUTIVE_INFRA_FAILURES", 3)
# Re-ask the judge this many extra times if it returns malformed JSON.
JUDGE_MAX_FORMAT_RETRIES = _env_int("JUDGE_MAX_FORMAT_RETRIES", 1)

# --------------------------------------------------------------------------
# Quality-gate candidates
# --------------------------------------------------------------------------
# The four constants below are the project's original candidate thresholds.
# They are kept for backward compatibility and feed QUALITY_GATES.
MIN_ANSWER_RELEVANCY = 0.70
MIN_FAITHFULNESS = 0.70
MAX_HALLUCINATION_RATE = 0.20
MAX_P95_LATENCY_MS = 5000

# PROVISIONAL: these thresholds have not yet been calibrated against a live
# baseline run. See README > "Quality gates" for the calibration procedure.
QUALITY_GATES: dict[str, dict] = {
    "answer_accuracy": {"metric": "answer_accuracy", "op": ">=", "threshold": 0.80},
    "answer_relevance": {"metric": "mean_answer_relevance", "op": ">=", "threshold": MIN_ANSWER_RELEVANCY},
    "faithfulness": {"metric": "mean_faithfulness", "op": ">=", "threshold": MIN_FAITHFULNESS},
    "hallucination_rate": {"metric": "hallucination_rate", "op": "<=", "threshold": MAX_HALLUCINATION_RATE},
    "abstention_accuracy": {"metric": "abstention_accuracy", "op": ">=", "threshold": 0.80},
    "source_hit_rate": {"metric": "source_hit_rate", "op": ">=", "threshold": 0.90},
    "p95_latency": {"metric": "p95_end_to_end_latency_ms", "op": "<=", "threshold": MAX_P95_LATENCY_MS},
    "cost_per_query": {"metric": "mean_pipeline_cost_per_query_usd", "op": "<=", "threshold": 0.01},
    # Compared against reports/baseline.json; skipped when no compatible baseline exists.
    "accuracy_regression": {"metric": "answer_accuracy_drop_vs_baseline", "op": "<=", "threshold": 0.05},
}

# The answer the RAG prompt asks the model to give when the context is insufficient.
ABSTENTION_TEXT = "I don't have enough information to answer that."


def validate_config(require_api_key: bool = True) -> list[str]:
    """Return a list of configuration problems (empty list means valid)."""
    problems: list[str] = list(_ENV_ERRORS)
    if require_api_key and not GEMINI_API_KEY:
        problems.append(
            "GEMINI_API_KEY is not set. Copy .env.example to .env and add your key "
            "(or set the environment variable)."
        )
    elif require_api_key and GEMINI_API_KEY.strip().lower().startswith("your_"):
        problems.append("GEMINI_API_KEY still contains the placeholder from .env.example; paste your real key.")
    if not GEMINI_MODEL.strip():
        problems.append("GEMINI_MODEL is empty.")
    if not JUDGE_MODEL.strip():
        problems.append("JUDGE_MODEL is empty.")
    if TOP_K < 1:
        problems.append(f"TOP_K must be >= 1, got {TOP_K}.")
    if CHUNK_STRATEGY not in {"character", "paragraph"}:
        problems.append(f"CHUNK_STRATEGY must be 'character' or 'paragraph', got {CHUNK_STRATEGY!r}.")
    if CHUNK_SIZE < 1 or CHUNK_OVERLAP < 0 or CHUNK_OVERLAP >= CHUNK_SIZE:
        problems.append(
            f"Invalid chunking: CHUNK_SIZE={CHUNK_SIZE}, CHUNK_OVERLAP={CHUNK_OVERLAP} "
            "(need size >= 1 and 0 <= overlap < size)."
        )
    if MAX_RETRIES < 0:
        problems.append("GEMINI_MAX_RETRIES must be >= 0.")
    if RETRY_BASE_DELAY_S < 0 or RETRY_MAX_DELAY_S < RETRY_BASE_DELAY_S:
        problems.append("Retry delays must satisfy 0 <= base <= max.")
    if MIN_REQUEST_INTERVAL_S < 0:
        problems.append("GEMINI_MIN_REQUEST_INTERVAL_S must be >= 0.")
    for name, gate in QUALITY_GATES.items():
        if gate["op"] not in (">=", "<="):
            problems.append(f"Quality gate {name}: op must be '>=' or '<='.")
        if not isinstance(gate["threshold"], (int, float)):
            problems.append(f"Quality gate {name}: threshold must be numeric.")
    return problems
