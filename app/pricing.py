"""Model pricing used to *estimate* API cost.

Source: Gemini Developer API pricing, Standard (paid) tier, text input.
URL: https://ai.google.dev/gemini-api/docs/pricing
Verified: 2026-10-26. Currency: USD per 1,000,000 tokens.

Material conditions:
* Output prices include thinking tokens (the page labels them "Output price
  (including thinking tokens)"), so thinking tokens are billed as output.
* gemini-3.6-flash list prices change on 2027-01-01 (both rates double); the
  rate is chosen by the date of the evaluation run.
* The free tier is "Free of charge". Estimates here are what the same usage
  would cost at paid list prices, not what a free-tier user is
  billed.
* Context caching, batch and priority tiers are not modelled.

A model that is not listed here gets cost = None ("unavailable"); no other
model's price is ever substituted.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from app.gemini_client import TokenUsage

PRICING_SOURCE_URL = "https://ai.google.dev/gemini-api/docs/pricing"
PRICING_VERIFIED_ON = "2026-10-26"
PRICING_CURRENCY = "USD"


@dataclass(frozen=True)
class PriceWindow:
    valid_from: date | None  # inclusive; None = no lower bound
    valid_until: date | None  # inclusive; None = no upper bound
    input_per_million: float
    output_per_million: float


MODEL_PRICES: dict[str, list[PriceWindow]] = {
    "gemini-3.6-flash": [
        PriceWindow(None, date(2026, 12, 31), 0.75, 3.75),
        PriceWindow(date(2027, 1, 1), None, 1.50, 7.50),
    ],
    "gemini-3.8-flash": [
        PriceWindow(None, date(2026, 12, 31), 0.75, 3.75),
        PriceWindow(date(2027, 1, 1), None, 1.50, 7.50),
    ],
    "gemini-3.5-flash-lite": [PriceWindow(None, None, 0.30, 2.50)],
    "gemini-3.1-flash-lite": [PriceWindow(None, None, 0.25, 1.50)],
    # Groq on-demand prices. Groq's own pricing page could not be fetched from the
    # build environment; these values agree across two independent trackers
    # (benchlm.ai, read 2026-10-20; eesel.ai, 2026-06-08) and should be
    # re-checked at https://groq.com/pricing. Reasoning tokens count as output.
    "openai/gpt-oss-20b": [PriceWindow(None, None, 0.075, 0.30)],
    "openai/gpt-oss-120b": [PriceWindow(None, None, 0.15, 0.60)],
}

PRICING_SOURCES = {
    "gemini": {"source": PRICING_SOURCE_URL, "verified_on": PRICING_VERIFIED_ON, "currency": PRICING_CURRENCY},
    "groq": {
        "source": "https://groq.com/pricing (values via benchlm.ai/free-tier/groq and eesel.ai/blog/groq-pricing)",
        "verified_on": "2026-10-28 (secondary sources)",
        "currency": PRICING_CURRENCY,
    },
}


def price_for(model: str, on: date) -> PriceWindow | None:
    for window in MODEL_PRICES.get(model, []):
        if (window.valid_from is None or on >= window.valid_from) and (
            window.valid_until is None or on <= window.valid_until
        ):
            return window
    return None


def estimate_cost_usd(model: str, usage: TokenUsage | None, on: date) -> float | None:
    """Estimated cost of one call, or None when usage or pricing is unknown."""
    if usage is None or usage.input_tokens is None or usage.billable_output_tokens is None:
        return None
    window = price_for(model, on)
    if window is None:
        return None
    return (
        usage.input_tokens * window.input_per_million + usage.billable_output_tokens * window.output_per_million
    ) / 1_000_000
