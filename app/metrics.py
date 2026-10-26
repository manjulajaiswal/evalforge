"""Deterministic metrics.

Everything here is pure Python: no network calls and no randomness. That
makes these checks cheap, reproducible and unit-testable, and they act as
hard constraints alongside the LLM judge (app/judge.py).

Contents
* semantic_similarity     cosine similarity of MiniLM embeddings, clipped to
                          [-1, 1]. Diagnostic only: MiniLM rates "within 30
                          days" as very similar to "within 7 days", so it is
                          not used as a pass/fail signal.
* refund_days_match       original refund-window check (kept for
                          compatibility; check_fact supersedes it).
* check_fact              typed fact checks used by expected_facts in the
                          golden dataset: duration, count, clock_time, phrase.
* is_canonical_abstention detects the prompt's fixed "I don't have enough
                          information" answer.
* retrieval_metrics       source-level hit / precision / recall / reciprocal rank.
* percentile              linear-interpolation percentile with sample count.
"""

from __future__ import annotations

import math
import re
from typing import Iterable, Sequence

import numpy as np

# ---------------------------------------------------------------------------
# Semantic similarity (original)
# ---------------------------------------------------------------------------


def semantic_similarity(expected, actual):
    from app.embedder import generate_embeddings

    embeddings = generate_embeddings([expected, actual])
    expected_embedding = embeddings[0]
    actual_embedding = embeddings[1]
    similarity = np.dot(expected_embedding, actual_embedding) / (
        np.linalg.norm(expected_embedding) * np.linalg.norm(actual_embedding)
    )
    # Floating-point rounding can produce values like 1.0000001; cosine
    # similarity is defined on [-1, 1].
    return float(np.clip(similarity, -1.0, 1.0))


# ---------------------------------------------------------------------------
# Number parsing helpers
# ---------------------------------------------------------------------------

WORD_NUMBERS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
    "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13,
    "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18,
    "nineteen": 19, "twenty": 20, "a": 1, "an": 1,
}
ORDINALS = {
    "first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5, "sixth": 6,
    "seventh": 7, "eighth": 8, "ninth": 9, "tenth": 10,
    "1st": 1, "2nd": 2, "3rd": 3, "4th": 4, "5th": 5, "6th": 6, "7th": 7, "8th": 8, "9th": 9, "10th": 10,
}
_NUM = r"(\d+|" + "|".join(sorted((w for w in WORD_NUMBERS if w not in {"a", "an"}), key=len, reverse=True)) + r"|an?)"


def _to_int(token: str) -> int:
    token = token.lower()
    if token.isdigit():
        return int(token)
    if token in WORD_NUMBERS:
        return WORD_NUMBERS[token]
    return ORDINALS[token]


def normalize_text(text: str) -> str:
    text = text.lower().replace("’", "'").replace("‘", "'")
    text = re.sub(r"\b(\w+)n't\b", lambda m: {"wo": "will", "ca": "can"}.get(m.group(1), m.group(1)) + " not", text)
    text = text.replace("cannot", "can not")
    text = re.sub(r"[\"'`*_]", "", text)
    text = re.sub(r"[-‐-―/]", " ", text)
    text = re.sub(r"[^a-z0-9:.\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


# ---------------------------------------------------------------------------
# Original refund-period check (kept; see check_fact for the general version)
# ---------------------------------------------------------------------------


def refund_days_match(expected, actual):
    def extract_days(text):
        text = text.lower()
        word_to_number = {
            "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
            "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
        }
        match = re.search(
            r"\b(\d+|one|two|three|four|five|six|seven|eight|nine|ten)"
            r"\s+(?:calendar\s+)?days?\b",
            text,
        )
        if match:
            value = match.group(1)
            return int(value) if value.isdigit() else word_to_number[value]
        if re.search(r"\bone\s+week\b", text):
            return 7
        return None

    expected_days = extract_days(expected)
    actual_days = extract_days(actual)
    return expected_days is not None and actual_days is not None and expected_days == actual_days


# ---------------------------------------------------------------------------
# Typed fact checks
# ---------------------------------------------------------------------------

_N = _NUM[1:-1]
# A number or a range ("5 to 7", "one or two", "between 3 and 5"), optionally followed by a unit.
_DURATION_RE = re.compile(
    rf"\b(?:between\s+(?P<b_lo>{_N})\s+and\s+(?P<b_hi>{_N})|(?P<lo>{_N})(?:\s*(?:to|through|or)\s*(?P<hi>{_N}))?)"
    r"(?:\s*(?P<qual>(?:business|working|calendar)\s+)?(?P<unit>days?|weeks?)\b)?"
)
# Words allowed between a bare number/range and the unit it borrows,
# e.g. "3 to 5 [and express] 1 to 2 business days" or "7 [and] 14 days respectively".
_BORROW_GAP = re.compile(r"^\s*(?:,|and|or|while|whereas)?\s*(?:[a-z]+\s+){0,2}$")


def _unit_of(qual: str, unit: str) -> tuple[str, int] | None:
    """(unit, multiplier) or None for units the policies never use ("business weeks")."""
    is_week = unit.startswith("week")
    if qual in ("business", "working"):
        return None if is_week else ("business_day", 1)
    return ("day", 7 if is_week else 1)


def extract_durations(text: str) -> list[tuple[int, int, str]]:
    """Find durations like '7 days', 'one week', '5 to 7 business days'.

    Returns (min, max, unit) with unit 'day' or 'business_day'. Weeks are
    converted to days; a single number is returned as (n, n, unit). A bare
    number or range borrows the unit of a nearby duration in the same
    sentence ("3 to 5 and 1 to 2 business days", "7 and 14 days
    respectively"). Ranges with min > max are discarded.
    """
    text = re.sub(r"\(\s*\d+\s*\)", " ", text)  # "seven (7) days" -> "seven days"
    # Keep numeric ranges ("5-7", "5–7") before normalisation turns dashes into spaces.
    text = re.sub(rf"(?i)\b{_NUM}\s*[-‐-―]\s*{_NUM}\b", r"\1 to \2", text)
    text = normalize_text(text)
    found: list[tuple[int, int, str]] = []
    for sentence in re.split(r"[;!?]|\.(?=\s|$)", text):
        items = []  # [low, high, unit|None, start, end]
        for m in _DURATION_RE.finditer(sentence):
            low_tok = m.group("b_lo") or m.group("lo")
            high_tok = m.group("b_hi") or m.group("hi")
            if low_tok in ("a", "an") and not m.group("unit"):
                continue
            low = _to_int(low_tok)
            high = _to_int(high_tok) if high_tok else low
            unit = None
            if m.group("unit"):
                parsed = _unit_of((m.group("qual") or "").strip(), m.group("unit"))
                if parsed is None:
                    continue
                unit = parsed
            items.append([low, high, unit, m.start(), m.end()])
        for i, item in enumerate(items):
            if item[2] is None:
                nxt = next((j for j in range(i + 1, len(items)) if items[j][2] is not None), None)
                prv = next((j for j in range(i - 1, -1, -1) if items[j][2] is not None), None)
                if nxt is not None and _BORROW_GAP.match(sentence[item[4]:items[nxt][3]]):
                    item[2] = items[nxt][2]
                elif prv is not None and _BORROW_GAP.match(sentence[items[prv][4]:item[3]]):
                    item[2] = items[prv][2]
        for low, high, unit, _, _ in items:
            if unit is None or low > high:
                continue
            name, mult = unit
            found.append((low * mult, high * mult, name))
    return found


def _count_matches(nouns: Sequence[str], text: str) -> list[int]:
    """Numbers attached to a noun: '5 failed login attempts', 'fifth attempt', 'attempts: 5'."""
    noun_alt = "|".join(re.escape(normalize_text(n)) for n in nouns)
    ordinal_alt = "|".join(ORDINALS)
    forward = re.compile(rf"\b({_N}|{ordinal_alt})\s+(?:[a-z]+\s+){{0,4}}?(?:{noun_alt})s?\b")
    backward = re.compile(rf"\b(?:{noun_alt})s?\s*(?::|=|of)\s*({_N})\b")
    normalized = normalize_text(text)
    return [_to_int(m.group(1)) for regex in (forward, backward) for m in regex.finditer(normalized)]


def extract_clock_hours(text: str) -> list[tuple[int, int]]:
    """Return (hour24, minute) pairs for '6 PM', '6:00 p.m.', '6.00pm', "6 o'clock in the evening", '18:00'."""
    lowered = text.lower().replace("’", "'").replace("o'clock", "oclock").replace("o clock", "oclock")
    times = []
    word_hours = "|".join(w for w in WORD_NUMBERS if w not in {"a", "an"})
    pm_words = r"in\s+the\s+(?:evening|afternoon)|at\s+night|tonight"
    am_words = r"in\s+the\s+morning"
    pattern = (
        rf"\b(\d{{1,2}}|{word_hours})(?:[:.](\d{{2}}))?\s*(?:oclock\s*)?"
        rf"(a\.?\s?m\.?|p\.?\s?m\.?|{pm_words}|{am_words})(?![a-z])"
    )
    for m in re.finditer(pattern, lowered):
        hour, minute = _to_int(m.group(1)), int(m.group(2) or 0)
        if not 1 <= hour <= 12 or minute > 59:
            continue
        marker = m.group(3)
        is_pm = marker.startswith("p") or bool(re.match(pm_words, marker))
        hour = (hour % 12) + (12 if is_pm else 0)
        times.append((hour, minute))
    for m in re.finditer(r"\b([01]?\d|2[0-3]):([0-5]\d)\b(?!\s*(?:[ap]\.?\s?m|in the|at night))", lowered):
        times.append((int(m.group(1)), int(m.group(2))))
    return times


FACT_TYPES = {"duration", "count", "clock_time", "phrase"}


def validate_fact_spec(fact: object) -> list[str]:
    """Return problems with a fact spec (empty list = valid)."""
    if not isinstance(fact, dict):
        return ["fact must be an object"]
    kind = fact.get("type")
    if kind not in FACT_TYPES:
        return [f"type must be one of {sorted(FACT_TYPES)}"]
    problems = []
    if not isinstance(fact.get("description", ""), str):
        problems.append("description must be a string")
    if kind == "duration":
        lo, hi = fact.get("min"), fact.get("max")
        if not (isinstance(lo, int) and isinstance(hi, int) and 0 < lo <= hi):
            problems.append("duration needs integer min/max with 0 < min <= max")
        if fact.get("unit") not in ("day", "business_day"):
            problems.append("duration unit must be 'day' or 'business_day'")
    elif kind == "count":
        if not (isinstance(fact.get("value"), int) and fact["value"] > 0):
            problems.append("count needs a positive integer 'value'")
        nouns = fact.get("nouns")
        if not (isinstance(nouns, list) and nouns and all(isinstance(n, str) and n for n in nouns)):
            problems.append("count needs a non-empty 'nouns' list")
    elif kind == "clock_time":
        if not (isinstance(fact.get("hour"), int) and 0 <= fact["hour"] <= 23):
            problems.append("clock_time needs 'hour' 0-23")
        if not (isinstance(fact.get("minute", 0), int) and 0 <= fact.get("minute", 0) <= 59):
            problems.append("clock_time 'minute' must be 0-59")
    elif kind == "phrase":
        options = fact.get("any")
        if not (isinstance(options, list) and options and all(isinstance(o, str) and o.strip() for o in options)):
            problems.append("phrase needs a non-empty 'any' list of strings")
    return problems


def describe_fact(fact: dict) -> str:
    if fact.get("description"):
        return fact["description"]
    kind = fact["type"]
    if kind == "duration":
        span = str(fact["min"]) if fact["min"] == fact["max"] else f"{fact['min']}-{fact['max']}"
        return f"{span} {'business days' if fact['unit'] == 'business_day' else 'days'}"
    if kind == "count":
        return f"{fact['value']} {fact['nouns'][0]}(s)"
    if kind == "clock_time":
        return f"{fact['hour']:02d}:{fact.get('minute', 0):02d}"
    return " / ".join(fact["any"])


def check_fact(fact: dict, answer: str) -> bool:
    """True if ``answer`` states the fact. Unknown/invalid specs raise ValueError."""
    problems = validate_fact_spec(fact)
    if problems:
        raise ValueError("; ".join(problems))
    kind = fact["type"]
    if kind == "duration":
        return (fact["min"], fact["max"], fact["unit"]) in extract_durations(answer)
    if kind == "count":
        return fact["value"] in _count_matches(fact["nouns"], answer)
    if kind == "clock_time":
        return (fact["hour"], fact.get("minute", 0)) in extract_clock_hours(answer)
    def words_only(text: str) -> str:
        return " " + re.sub(r"\s+", " ", re.sub(r"[.:]", " ", normalize_text(text))).strip() + " "

    normalized = words_only(answer)
    return any(words_only(option) in normalized for option in fact["any"])


def check_facts(facts: Iterable[dict], answer: str) -> dict:
    results = [{"fact": describe_fact(f), "type": f["type"], "present": check_fact(f, answer)} for f in facts]
    return {
        "checked": len(results),
        "passed": all(r["present"] for r in results) if results else None,
        "missing": [r["fact"] for r in results if not r["present"]],
        "results": results,
    }


# ---------------------------------------------------------------------------
# Abstention
# ---------------------------------------------------------------------------

_ABSTENTION_PATTERNS = (
    "do not have enough information",
    "not enough information to answer",
    "insufficient information to answer",
)


def is_canonical_abstention(answer: str) -> bool:
    """True if the answer contains the prompt's prescribed 'not enough information' wording."""
    normalized = normalize_text(answer)
    return any(pattern in normalized for pattern in _ABSTENTION_PATTERNS)


def is_pure_abstention(answer: str) -> bool:
    """True if the answer is ONLY an abstention (no other substantive content).

    "I don't have enough information to answer that." -> True
    "I don't have enough information. The policy only says X." -> False
    Mixed answers are graded by the judge, not by this deterministic rule.
    """
    if not is_canonical_abstention(answer):
        return False
    sentences = [s for s in re.split(r"(?<=[.!?])\s+", answer.strip()) if s.strip()]
    remaining = [s for s in sentences if not is_canonical_abstention(s)]
    return len(re.findall(r"[a-z0-9]+", normalize_text(" ".join(remaining)))) <= 3


# ---------------------------------------------------------------------------
# Retrieval metrics (source level)
# ---------------------------------------------------------------------------


def retrieval_metrics(retrieved_sources: Sequence[str], relevant_sources: Sequence[str]) -> dict | None:
    """Source-level retrieval metrics for one query.

    retrieved_sources: source of each retrieved chunk, in rank order (may repeat).
    relevant_sources:  sources annotated as containing needed information.

    Definitions (U = unique retrieved sources, R = relevant sources):
      hit        1 if |U n R| > 0 else 0
      precision  |U n R| / |U|   (0.0 when nothing was retrieved)
      recall     |U n R| / |R|
      rr         1 / rank of the first chunk whose source is relevant (0 if none)

    Returns None when R is empty (e.g. unanswerable questions): recall is
    undefined without relevance labels, and precision against an empty
    relevant set would penalise retrieval for something it cannot fix.
    """
    relevant = set(relevant_sources)
    if not relevant:
        return None
    unique_retrieved = list(dict.fromkeys(retrieved_sources))
    hits = [s for s in unique_retrieved if s in relevant]
    first_rank = next((i + 1 for i, s in enumerate(retrieved_sources) if s in relevant), None)
    return {
        "hit": 1.0 if hits else 0.0,
        "precision": len(hits) / len(unique_retrieved) if unique_retrieved else 0.0,
        "recall": len(hits) / len(relevant),
        "reciprocal_rank": 1.0 / first_rank if first_rank else 0.0,
        "first_relevant_rank": first_rank,
        "retrieved_unique_sources": unique_retrieved,
        "missing_sources": sorted(relevant - set(unique_retrieved)),
        "irrelevant_sources": [s for s in unique_retrieved if s not in relevant],
    }


# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------


def percentile(values: Sequence[float], q: float) -> float | None:
    """q-th percentile (0-100) by linear interpolation between closest ranks.

    Same as numpy.percentile(method="linear") / Excel PERCENTILE.INC.
    Returns None for an empty sample.
    """
    data = sorted(v for v in values if v is not None and not math.isnan(v))
    if not data:
        return None
    if not 0 <= q <= 100:
        raise ValueError("q must be between 0 and 100")
    position = (len(data) - 1) * q / 100
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(data[lower])
    return float(data[lower] + (data[upper] - data[lower]) * (position - lower))


def mean(values: Iterable[float | None]) -> float | None:
    data = [v for v in values if v is not None]
    return sum(data) / len(data) if data else None
