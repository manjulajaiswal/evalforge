from pathlib import Path
from dotenv import load_dotenv
import os

load_dotenv()

ROOT_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = ROOT_DIR / "data"
KNOWLEDGE_BASE_DIR = DATA_DIR / "knowledge_base"
GOLDEN_DATASET_PATH = DATA_DIR / "golden_dataset.jsonl"

REPORTS_DIR = ROOT_DIR / "reports"
LATEST_REPORT_JSON = REPORTS_DIR / "latest_report.json"
LATEST_REPORT_MD = REPORTS_DIR / "latest_report.md"
HISTORY_PATH = REPORTS_DIR / "history.jsonl"

CHROMA_DIR = ROOT_DIR / ".chroma"


GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.6-flash")


MIN_ANSWER_RELEVANCY = 0.70
MIN_FAITHFULNESS = 0.70
MAX_HALLUCINATION_RATE = 0.20
MAX_P95_LATENCY_MS = 5000