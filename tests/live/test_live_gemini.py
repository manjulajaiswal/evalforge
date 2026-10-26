"""LIVE integration tests: these call the real Gemini API and cost (a little) money.

Not part of the default suite. Run explicitly:

    pytest -m live

They are skipped when GEMINI_API_KEY is not set.
"""

import os

import pytest

pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(not os.environ.get("GEMINI_API_KEY") and not os.path.exists(".env"), reason="GEMINI_API_KEY not set"),
]


def test_minimal_generation_returns_text_and_usage():
    from app import config
    from app.gemini_client import GeminiClient

    if not config.GEMINI_API_KEY:
        pytest.skip("GEMINI_API_KEY not set")
    response = GeminiClient().generate("Reply with the single word: pong", model=config.GEMINI_MODEL)
    assert response.text.strip()
    assert response.usage is not None and response.usage.input_tokens and response.usage.input_tokens > 0
    assert response.latency_ms > 0


def test_judge_returns_schema_valid_json():
    from app import config
    from app.gemini_client import GeminiClient
    from app.judge import judge_faithfulness

    if not config.GEMINI_API_KEY:
        pytest.skip("GEMINI_API_KEY not set")
    result = judge_faithfulness(
        GeminiClient(),
        "How long do I have to request a refund?",
        ["Customers can request a refund within 7 days of purchase."],
        "You can request a refund within 7 days of purchase.",
    )
    assert result.claims and all(c.label in ("supported", "unsupported", "contradicted") for c in result.claims)
