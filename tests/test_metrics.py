import numpy as np
import pytest

from app import embedder
from app.metrics import (
    check_fact,
    check_facts,
    extract_clock_hours,
    extract_durations,
    is_canonical_abstention,
    mean,
    percentile,
    refund_days_match,
    retrieval_metrics,
    semantic_similarity,
    validate_fact_spec,
)

REFUND_WINDOW = {"type": "duration", "min": 7, "max": 7, "unit": "day"}
REFUND_PROCESSING = {"type": "duration", "min": 5, "max": 7, "unit": "business_day"}
LOCK = {"type": "count", "value": 5, "nouns": ["attempt", "tries", "try"]}
CUTOFF = {"type": "clock_time", "hour": 18}


# --- semantic similarity -----------------------------------------------------

class ConstantEmbedder:
    def encode(self, texts):
        v = np.array([0.1, 0.2, 0.3], dtype=np.float32) * 1.0000001
        return np.stack([v for _ in texts])


def test_semantic_similarity_is_clipped_to_valid_range():
    embedder.set_embedding_model(ConstantEmbedder())
    assert semantic_similarity("a", "a") <= 1.0


def test_semantic_similarity_orders_related_text_higher():
    s_same = semantic_similarity("refund within 7 days", "refund within 7 days")
    s_other = semantic_similarity("refund within 7 days", "express shipping takes 1 business day")
    assert s_same == pytest.approx(1.0) and s_other < s_same


# --- original refund check (behaviour preserved) -----------------------------

def test_refund_days_match_original_behaviour():
    expected = "Customers can request a refund within 7 days of purchase."
    assert refund_days_match(expected, "Customers have one week from the date they purchased the product.")
    assert not refund_days_match(expected, "Customers can request a refund within 30 days of purchase.")
    assert not refund_days_match(expected, "No number here.")


# --- durations ---------------------------------------------------------------

@pytest.mark.parametrize(
    "text,expected",
    [
        ("within 7 days of purchase", (7, 7, "day")),
        ("you have one week", (7, 7, "day")),
        ("a 7-day window", (7, 7, "day")),
        ("seven (7) calendar days", (7, 7, "day")),
        ("within 5 to 7 business days", (5, 7, "business_day")),
        ("5-7 business days", (5, 7, "business_day")),
        ("5–7 working days", (5, 7, "business_day")),
        ("between 3 and 5 business days", (3, 5, "business_day")),
        ("processed within two weeks", (14, 14, "day")),
    ],
)
def test_extract_durations(text, expected):
    assert expected in extract_durations(text)


def test_business_days_are_not_calendar_days():
    assert not check_fact(REFUND_WINDOW, "You can request a refund within 7 business days.")
    assert check_fact(REFUND_PROCESSING, "Refunds take 5 to 7 business days.")
    assert not check_fact(REFUND_PROCESSING, "Refunds take 7 business days.")


def test_wrong_number_fails_even_if_wording_is_similar():
    assert not check_fact(REFUND_WINDOW, "Customers can request a refund within 30 days of purchase.")


# --- counts / times / phrases -------------------------------------------------

@pytest.mark.parametrize(
    "text,ok",
    [
        ("locked after 5 failed login attempts", True),
        ("after five consecutive unsuccessful sign-in attempts", True),
        ("on the fifth failed attempt", True),
        ("after 5 tries", True),
        ("after 3 failed attempts", False),
    ],
)
def test_count_fact(text, ok):
    assert check_fact(LOCK, text) is ok


@pytest.mark.parametrize("text", ["after 6 PM", "after 6:00 p.m.", "after 18:00", "after 6pm", "after six p.m."])
def test_clock_time_variants(text):
    assert check_fact(CUTOFF, text)


def test_clock_time_rejects_other_hours():
    assert not check_fact(CUTOFF, "after 5 PM")
    assert extract_clock_hours("12 AM and 12 PM") == [(0, 0), (12, 0)]


def test_phrase_fact_normalises_case_punctuation_and_quotes():
    fact = {"type": "phrase", "any": ["forgot password", "forgotten password"]}
    assert check_fact(fact, "Click “Forgot Password?” on the login page.")
    assert check_fact(fact, "Use the forgot-password option.")
    assert not check_fact(fact, "Contact support to reset it.")
    assert check_fact({"type": "phrase", "any": ["original payment method"]}, "Back to the original payment method.")


def test_check_facts_summary():
    result = check_facts([REFUND_WINDOW, REFUND_PROCESSING], "Request within 7 days; processing takes 3 days.")
    assert result["checked"] == 2 and result["passed"] is False and len(result["missing"]) == 1
    assert check_facts([], "anything")["passed"] is None


@pytest.mark.parametrize(
    "spec",
    [
        {"type": "nope"},
        {"type": "duration", "min": 7, "max": 5, "unit": "day"},
        {"type": "duration", "min": 7, "max": 7, "unit": "weeks"},
        {"type": "count", "value": 0, "nouns": ["x"]},
        {"type": "count", "value": 3, "nouns": []},
        {"type": "clock_time", "hour": 25},
        {"type": "phrase", "any": []},
        "not a dict",
    ],
)
def test_invalid_fact_specs_are_rejected(spec):
    assert validate_fact_spec(spec)
    with pytest.raises(ValueError):
        check_fact(spec, "text")


# --- abstention ----------------------------------------------------------------

@pytest.mark.parametrize(
    "text,expected",
    [
        ("I don't have enough information to answer that.", True),
        ("I don’t have enough information to answer that.", True),
        ("I do not have enough information.", True),
        ("Refunds are allowed within 7 days.", False),
        ("The policy does not mention it.", False),
    ],
)
def test_canonical_abstention(text, expected):
    assert is_canonical_abstention(text) is expected


# --- retrieval metrics -----------------------------------------------------------

def test_retrieval_metrics_definitions():
    m = retrieval_metrics(["refund.md", "shipping.md", "refund.md"], ["refund.md"])
    assert m["hit"] == 1.0 and m["precision"] == 0.5 and m["recall"] == 1.0 and m["reciprocal_rank"] == 1.0
    assert m["irrelevant_sources"] == ["shipping.md"] and m["missing_sources"] == []


def test_retrieval_metrics_multi_source_and_rank():
    m = retrieval_metrics(["account.md", "shipping.md"], ["shipping.md", "refund.md"])
    assert m["recall"] == 0.5 and m["precision"] == 0.5 and m["reciprocal_rank"] == 0.5
    assert m["missing_sources"] == ["refund.md"]


def test_retrieval_metrics_conventions():
    assert retrieval_metrics(["a.md"], []) is None  # no labels -> undefined, not 0
    empty = retrieval_metrics([], ["a.md"])
    assert empty["precision"] == 0.0 and empty["recall"] == 0.0 and empty["hit"] == 0.0


# --- aggregation helpers ----------------------------------------------------------

def test_percentile_matches_numpy_linear():
    data = [120.0, 80.0, 300.0, 95.0, 101.0, 2500.0]
    for q in (0, 50, 95, 100):
        assert percentile(data, q) == pytest.approx(float(np.percentile(data, q)))
    assert percentile([], 95) is None
    assert percentile([42.0], 95) == 42.0
    with pytest.raises(ValueError):
        percentile([1.0], 101)


def test_mean_ignores_missing_values():
    assert mean([1.0, None, 3.0]) == 2.0
    assert mean([None]) is None
