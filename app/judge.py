"""LLM-as-a-judge evaluations.

Two separate judge calls are made per case, on purpose:

1. Faithfulness judge - sees ONLY the question, the retrieved context and the
   generated answer. It never sees the gold answer, so its verdict measures
   "is every claim supported by what the system retrieved?", not "does it
   match our expected answer?".
2. Correctness judge - sees the question, the gold (expected) answer and the
   generated answer. It grades correctness and relevance.

Both judges must return JSON matching a schema. The response is requested in
JSON mode and then validated again here; anything malformed raises
JudgeOutputError (after a configurable number of re-asks) so that a broken
judge never silently produces a passing score.

Limitations (documented in the README): the judge is itself an LLM and can be
wrong; by default it is the same model family as the system under test,
which may bias it toward its own phrasing; its verdicts are not deterministic.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from app import config
from app.errors import JudgeOutputError
from app.gemini_client import LLMClient, LLMResponse, TokenUsage

CLAIM_LABELS = ("supported", "unsupported", "contradicted")
VERDICTS = ("correct", "partially_correct", "incorrect")

FAITHFULNESS_SCHEMA = {
    "type": "object",
    "properties": {
        "claims": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "claim": {"type": "string"},
                    "label": {"type": "string", "enum": list(CLAIM_LABELS)},
                    "evidence": {"type": "string"},
                },
                "required": ["claim", "label", "evidence"],
            },
        },
        "abstained": {"type": "boolean"},
    },
    "required": ["claims", "abstained"],
}

CORRECTNESS_SCHEMA = {
    "type": "object",
    "properties": {
        "verdict": {"type": "string", "enum": list(VERDICTS)},
        "correctness_score": {"type": "number"},
        "relevance_score": {"type": "number"},
        "abstained": {"type": "boolean"},
        "issues": {"type": "array", "items": {"type": "string"}},
        "reasoning": {"type": "string"},
    },
    "required": ["verdict", "correctness_score", "relevance_score", "abstained", "issues", "reasoning"],
}

FAITHFULNESS_PROMPT = """You are a strict fact-checker for a customer-support assistant.

Decide whether each factual claim in the ANSWER is supported by the CONTEXT.
Use only the CONTEXT. Do not use outside knowledge, and do not judge whether
the claim is true in the real world.

Steps:
1. Split the ANSWER into atomic factual claims (one fact each). Ignore
   greetings, filler and offers of further help.
2. Label each claim:
   - "supported": the CONTEXT states it or it follows directly from the CONTEXT.
   - "contradicted": the CONTEXT states something incompatible with it.
   - "unsupported": the CONTEXT neither states nor implies it.
3. In "evidence" quote the context sentence used (empty string if none).
4. Set "abstained" to true if the ANSWER declines to answer or says the
   information is not available, false otherwise. A statement that the
   context does not contain some information is not a factual claim about
   the policy; do not list it as a claim.

If the ANSWER makes no factual claims, return an empty "claims" list.

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:
{answer}

Return JSON only."""

CORRECTNESS_PROMPT = """You are grading a customer-support assistant's answer against a reference answer.

QUESTION:
{question}

REFERENCE ANSWER (ground truth written by the test author):
{expected}

{answerability_note}

ASSISTANT ANSWER:
{answer}

Grade with this rubric:
- "verdict":
  - "correct": conveys every key fact of the reference (numbers, units such as
    "business days" vs "days", conditions and exceptions) and adds nothing
    that conflicts with it. Different wording is fine. Extra details that are
    consistent with the reference are fine.
  - "partially_correct": some key facts are right but at least one is missing
    or imprecise, with nothing directly wrong.
  - "incorrect": a key fact is wrong, the answer contradicts the reference,
    it answers a different question, or it abstains when the reference gives
    an answer.
- "correctness_score": 1.0 = correct, 0.5 = partially correct, 0.0 = incorrect
  (use intermediate values only if clearly justified).
- "relevance_score": 0.0-1.0, how directly the answer addresses the question
  that was asked, regardless of correctness (1.0 = fully on-topic and focused).
- "abstained": true if the assistant declined to answer or said the
  information is not available.
- "issues": short list of specific problems (empty list if none).
- "reasoning": one or two sentences.

Return JSON only."""

ANSWERABLE_NOTE = "The reference shows the information IS available, so the assistant should answer."
UNANSWERABLE_NOTE = (
    "The information requested is NOT available to the assistant. The correct behaviour is to say "
    "so (abstain) without inventing details. Treat an abstention as correct. Treat any invented "
    "specifics (numbers, procedures, policies not in the reference) as incorrect."
)


@dataclass
class Claim:
    claim: str
    label: str
    evidence: str


@dataclass
class JudgeCall:
    """Bookkeeping for one judge invocation (latency, tokens, retries)."""

    model: str
    latency_ms: float
    attempts: int
    format_retries: int
    usage: TokenUsage | None


@dataclass
class FaithfulnessResult:
    claims: list[Claim]
    abstained: bool
    call: JudgeCall

    @property
    def score(self) -> float | None:
        """Supported claims / all claims. None when the answer makes no claims."""
        if not self.claims:
            return None
        return sum(c.label == "supported" for c in self.claims) / len(self.claims)

    @property
    def unsupported_claims(self) -> list[Claim]:
        return [c for c in self.claims if c.label == "unsupported"]

    @property
    def contradicted_claims(self) -> list[Claim]:
        return [c for c in self.claims if c.label == "contradicted"]

    @property
    def has_hallucination(self) -> bool:
        return bool(self.unsupported_claims or self.contradicted_claims)


@dataclass
class CorrectnessResult:
    verdict: str
    correctness_score: float
    relevance_score: float
    abstained: bool
    issues: list[str]
    reasoning: str
    call: JudgeCall
    raw: dict = field(default_factory=dict)


def _strip_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else ""
        if text.rstrip().endswith("```"):
            text = text.rstrip()[:-3]
    return text.strip()


def _parse_json_object(text: str) -> dict:
    try:
        data = json.loads(_strip_fences(text))
    except json.JSONDecodeError as exc:
        raise JudgeOutputError(f"Judge did not return valid JSON: {exc.msg}") from exc
    if not isinstance(data, dict):
        raise JudgeOutputError("Judge JSON must be an object")
    return data


def _number_in_unit_range(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise JudgeOutputError(f"'{name}' must be a number")
    if not 0.0 <= float(value) <= 1.0:
        raise JudgeOutputError(f"'{name}' must be between 0 and 1, got {value}")
    return float(value)


def parse_faithfulness(text: str) -> tuple[list[Claim], bool]:
    data = _parse_json_object(text)
    claims_raw = data.get("claims")
    if not isinstance(claims_raw, list):
        raise JudgeOutputError("'claims' must be a list")
    if not isinstance(data.get("abstained"), bool):
        raise JudgeOutputError("'abstained' must be true or false")
    claims = []
    for i, item in enumerate(claims_raw):
        if not isinstance(item, dict):
            raise JudgeOutputError(f"claims[{i}] must be an object")
        claim, label, evidence = item.get("claim"), item.get("label"), item.get("evidence", "")
        if not isinstance(claim, str) or not claim.strip():
            raise JudgeOutputError(f"claims[{i}].claim must be a non-empty string")
        if label not in CLAIM_LABELS:
            raise JudgeOutputError(f"claims[{i}].label must be one of {CLAIM_LABELS}, got {label!r}")
        if not isinstance(evidence, str):
            raise JudgeOutputError(f"claims[{i}].evidence must be a string")
        claims.append(Claim(claim.strip(), label, evidence))
    return claims, data["abstained"]


def parse_correctness(text: str) -> dict:
    data = _parse_json_object(text)
    if data.get("verdict") not in VERDICTS:
        raise JudgeOutputError(f"'verdict' must be one of {VERDICTS}, got {data.get('verdict')!r}")
    out = {
        "verdict": data["verdict"],
        "correctness_score": _number_in_unit_range(data.get("correctness_score"), "correctness_score"),
        "relevance_score": _number_in_unit_range(data.get("relevance_score"), "relevance_score"),
    }
    if (out["verdict"] == "correct" and out["correctness_score"] < 0.5) or (
        out["verdict"] == "incorrect" and out["correctness_score"] > 0.5
    ):
        raise JudgeOutputError(
            f"verdict {out['verdict']!r} is inconsistent with correctness_score {out['correctness_score']}"
        )
    if not isinstance(data.get("abstained"), bool):
        raise JudgeOutputError("'abstained' must be true or false")
    issues = data.get("issues", [])
    if not isinstance(issues, list) or not all(isinstance(i, str) for i in issues):
        raise JudgeOutputError("'issues' must be a list of strings")
    reasoning = data.get("reasoning", "")
    if not isinstance(reasoning, str):
        raise JudgeOutputError("'reasoning' must be a string")
    out.update(abstained=data["abstained"], issues=issues, reasoning=reasoning)
    return out


def _call_and_parse(llm: LLMClient, prompt: str, schema: dict, parser, model: str, max_format_retries: int):
    """Call the judge; re-ask up to max_format_retries times on malformed output.

    Returns (parsed, JudgeCall). Token usage and latency are summed across
    re-asks so the cost of malformed responses is still counted.
    """
    total_latency = 0.0
    total_attempts = 0
    # All-or-nothing: if any response lacks usage data, the total is unknown
    # (a partial sum would understate tokens and cost).
    usage_sum: TokenUsage | None = TokenUsage(0, 0, 0, 0)
    last_error: JudgeOutputError | None = None
    for format_try in range(max_format_retries + 1):
        response: LLMResponse = llm.generate(prompt, model=model, json_schema=schema)
        total_latency += response.total_latency_ms
        total_attempts += response.attempts
        if response.usage is None or usage_sum is None:
            usage_sum = None
        else:
            for name in ("input_tokens", "output_tokens", "thinking_tokens", "total_tokens"):
                setattr(usage_sum, name, (getattr(usage_sum, name) or 0) + (getattr(response.usage, name) or 0))
        try:
            parsed = parser(response.text)
            return parsed, JudgeCall(model, total_latency, total_attempts, format_try, usage_sum)
        except JudgeOutputError as exc:
            last_error = exc
    raise JudgeOutputError(f"{last_error} (after {max_format_retries + 1} judge response(s))")


def judge_faithfulness(
    llm: LLMClient, question: str, contexts: list[str], answer: str, *, model: str | None = None
) -> FaithfulnessResult:
    context_block = "\n\n".join(f"[{i + 1}] {c.strip()}" for i, c in enumerate(contexts)) or "(no context retrieved)"
    prompt = FAITHFULNESS_PROMPT.format(context=context_block, question=question, answer=answer)
    (claims, abstained), call = _call_and_parse(
        llm, prompt, FAITHFULNESS_SCHEMA, parse_faithfulness, model or config.JUDGE_MODEL, config.JUDGE_MAX_FORMAT_RETRIES
    )
    return FaithfulnessResult(claims=claims, abstained=abstained, call=call)


def judge_correctness(
    llm: LLMClient, question: str, expected: str, answer: str, should_answer: bool, *, model: str | None = None
) -> CorrectnessResult:
    prompt = CORRECTNESS_PROMPT.format(
        question=question,
        expected=expected,
        answer=answer,
        answerability_note=ANSWERABLE_NOTE if should_answer else UNANSWERABLE_NOTE,
    )
    parsed, call = _call_and_parse(
        llm, prompt, CORRECTNESS_SCHEMA, parse_correctness, model or config.JUDGE_MODEL, config.JUDGE_MAX_FORMAT_RETRIES
    )
    return CorrectnessResult(call=call, raw=dict(parsed), **parsed)
