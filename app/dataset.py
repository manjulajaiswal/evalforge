"""Golden dataset loading and validation.

Schema (one JSON object per line):

Required (original fields)
    id               str, unique, non-empty
    question         str, non-empty
    expected_answer  str, non-empty (for unanswerable cases: the ideal abstention)
    expected_source  str | null  (null only when should_answer is false)
    category         one of CATEGORIES
    difficulty       one of DIFFICULTIES

Optional (added for evaluation quality; defaults keep the original rows valid)
    should_answer     bool, default true. false = the knowledge base does not
                      contain the answer and the system should say so.
    relevant_sources  list[str], default [expected_source] (or [] when
                      should_answer is false). Every source that contains
                      information needed to answer. Used for precision/recall.
    expected_facts    list of fact specs checked deterministically
                      (see app/metrics.py: check_fact). Default [].
    tags              list[str], free-form labels (paraphrase, numeric, ...).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

from app.config import GOLDEN_DATASET_PATH, KNOWLEDGE_BASE_DIR
from app.errors import DatasetValidationError
from app.metrics import FACT_TYPES, validate_fact_spec

CATEGORIES = {"refund", "shipping", "account", "cross_policy", "out_of_scope"}
DIFFICULTIES = {"easy", "medium", "hard"}
REQUIRED_FIELDS = ("id", "question", "expected_answer", "expected_source", "category", "difficulty")
OPTIONAL_FIELDS = ("should_answer", "relevant_sources", "expected_facts", "tags")


@dataclass
class GoldenCase:
    id: str
    question: str
    expected_answer: str
    expected_source: str | None
    category: str
    difficulty: str
    should_answer: bool = True
    relevant_sources: list[str] = field(default_factory=list)
    expected_facts: list[dict] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    line_number: int = 0

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "question": self.question,
            "expected_answer": self.expected_answer,
            "expected_source": self.expected_source,
            "category": self.category,
            "difficulty": self.difficulty,
            "should_answer": self.should_answer,
            "relevant_sources": list(self.relevant_sources),
            "expected_facts": list(self.expected_facts),
            "tags": list(self.tags),
        }


def _validate_row(row: object, where: str, known_sources: set[str] | None) -> tuple[list[str], GoldenCase | None]:
    issues: list[str] = []
    if not isinstance(row, dict):
        return [f"{where}: each line must be a JSON object"], None

    for name in REQUIRED_FIELDS:
        if name not in row:
            issues.append(f"{where}: missing required field '{name}'")
    unknown = set(row) - set(REQUIRED_FIELDS) - set(OPTIONAL_FIELDS)
    if unknown:
        issues.append(f"{where}: unknown field(s) {sorted(unknown)}")
    if issues:
        return issues, None

    for name in ("id", "question", "expected_answer"):
        if not isinstance(row[name], str) or not row[name].strip():
            issues.append(f"{where}: '{name}' must be a non-empty string")

    should_answer = row.get("should_answer", True)
    if not isinstance(should_answer, bool):
        issues.append(f"{where}: 'should_answer' must be true or false")
        should_answer = True

    source = row["expected_source"]
    if source is None:
        if should_answer:
            issues.append(f"{where}: 'expected_source' may only be null when should_answer is false")
    elif not isinstance(source, str) or not source.strip():
        issues.append(f"{where}: 'expected_source' must be a non-empty string or null")
    elif known_sources is not None and source not in known_sources:
        issues.append(f"{where}: expected_source '{source}' is not a file in the knowledge base")

    if row["category"] not in CATEGORIES:
        issues.append(f"{where}: category must be one of {sorted(CATEGORIES)}, got {row['category']!r}")
    if row["difficulty"] not in DIFFICULTIES:
        issues.append(f"{where}: difficulty must be one of {sorted(DIFFICULTIES)}, got {row['difficulty']!r}")

    default_relevant = [source] if (should_answer and isinstance(source, str)) else []
    relevant = row.get("relevant_sources", default_relevant)
    if not isinstance(relevant, list) or not all(isinstance(s, str) and s.strip() for s in relevant):
        issues.append(f"{where}: 'relevant_sources' must be a list of file names")
        relevant = default_relevant
    else:
        if len(set(relevant)) != len(relevant):
            issues.append(f"{where}: 'relevant_sources' contains duplicates")
        if known_sources is not None:
            for s in relevant:
                if s not in known_sources:
                    issues.append(f"{where}: relevant source '{s}' is not a file in the knowledge base")
        if should_answer and not relevant:
            issues.append(f"{where}: answerable cases need at least one relevant source")
        if isinstance(source, str) and should_answer and source not in relevant:
            issues.append(f"{where}: expected_source must be included in relevant_sources")

    facts = row.get("expected_facts", [])
    if not isinstance(facts, list):
        issues.append(f"{where}: 'expected_facts' must be a list")
        facts = []
    for i, fact in enumerate(facts):
        for problem in validate_fact_spec(fact):
            issues.append(f"{where}: expected_facts[{i}]: {problem}")
    if facts and not should_answer:
        issues.append(f"{where}: unanswerable cases must not have expected_facts")

    tags = row.get("tags", [])
    if not isinstance(tags, list) or not all(isinstance(t, str) for t in tags):
        issues.append(f"{where}: 'tags' must be a list of strings")
        tags = []

    if issues:
        return issues, None
    return [], GoldenCase(
        id=row["id"],
        question=row["question"],
        expected_answer=row["expected_answer"],
        expected_source=source,
        category=row["category"],
        difficulty=row["difficulty"],
        should_answer=should_answer,
        relevant_sources=list(relevant),
        expected_facts=list(facts),
        tags=list(tags),
    )


def load_and_validate(path: Path | None = None, knowledge_base_dir: Path | None = KNOWLEDGE_BASE_DIR) -> list[GoldenCase]:
    """Parse and validate the dataset. Raises DatasetValidationError listing every problem."""
    path = Path(path) if path is not None else GOLDEN_DATASET_PATH
    if not path.is_file():
        raise DatasetValidationError([f"{path}: file not found"])

    known_sources = None
    if knowledge_base_dir is not None and Path(knowledge_base_dir).is_dir():
        known_sources = {p.name for p in Path(knowledge_base_dir).glob("*.md")}

    issues: list[str] = []
    cases: list[GoldenCase] = []
    seen: dict[str, int] = {}
    with open(path, "r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            where = f"{path.name}:{line_number}"
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                issues.append(f"{where}: invalid JSON ({exc.msg} at column {exc.colno})")
                continue
            row_issues, case = _validate_row(row, where, known_sources)
            issues.extend(row_issues)
            if case is None:
                continue
            case.line_number = line_number
            if case.id in seen:
                issues.append(f"{where}: duplicate id '{case.id}' (first used on line {seen[case.id]})")
                continue
            seen[case.id] = line_number
            cases.append(case)

    if not cases and not issues:
        issues.append(f"{path}: dataset contains no test cases")
    if issues:
        raise DatasetValidationError(issues)
    return cases


def dataset_fingerprint(path: Path | None = None) -> str:
    """Content hash with normalised line endings (CRLF on Windows == LF on Linux)."""
    path = Path(path) if path is not None else GOLDEN_DATASET_PATH
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()[:16]


def summarize(cases: list[GoldenCase]) -> dict:
    def count(key) -> dict:
        out: dict = {}
        for case in cases:
            values = key(case)
            for value in values if isinstance(values, list) else [values]:
                out[value] = out.get(value, 0) + 1
        return dict(sorted(out.items(), key=lambda kv: str(kv[0])))

    return {
        "total": len(cases),
        "by_category": count(lambda c: c.category),
        "by_difficulty": count(lambda c: c.difficulty),
        "answerable": sum(c.should_answer for c in cases),
        "unanswerable": sum(not c.should_answer for c in cases),
        "with_expected_facts": sum(bool(c.expected_facts) for c in cases),
        "by_tag": count(lambda c: c.tags),
        "fact_types_known": sorted(FACT_TYPES),
    }
