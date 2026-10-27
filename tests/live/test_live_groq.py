"""LIVE Groq test (opt-in: pytest -m live). Skipped without GROQ_API_KEY."""

import os

import pytest

pytestmark = [pytest.mark.live, pytest.mark.skipif(not os.environ.get("GROQ_API_KEY"), reason="GROQ_API_KEY not set")]


def test_groq_generation_and_json_judge():
    from app import config
    from app.groq_client import GroqClient
    from app.judge import judge_faithfulness

    client = GroqClient(api_key=os.environ["GROQ_API_KEY"])
    response = client.generate("Reply with the single word: pong", model=config.GROQ_GENERATION_MODEL_DEFAULT)
    assert response.text.strip() and response.usage is not None
    result = judge_faithfulness(client, "Refund window?", ["Refunds within 7 days of purchase."],
                                "Refunds within 7 days.", model=config.GROQ_JUDGE_MODEL_DEFAULT)
    assert result.claims
