"""Offline stand-ins for the embedding model and Gemini.

These fakes exist so the evaluation engine can be tested without network
access or an API key. Results produced with them say nothing about real LLM
quality; reports produced in tests are labelled run_mode="test-fake".
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Callable

import numpy as np

from app.gemini_client import LLMResponse, TokenUsage
from app.judge import CORRECTNESS_SCHEMA, FAITHFULNESS_SCHEMA

DIM = 256
_STOP = {"the", "a", "an", "is", "are", "to", "of", "and", "or", "my", "i", "do", "does", "can", "how", "what", "be", "in", "on", "for", "it", "if", "you", "your"}


class FakeEmbedder:
    """Deterministic hashed bag-of-words embeddings (unit-normalised).

    Texts that share words get similar vectors, so retrieval behaves sensibly
    in tests without downloading a model.
    """

    def encode(self, texts):
        out = np.zeros((len(texts), DIM))
        for i, text in enumerate(texts):
            for word in re.findall(r"[a-z]+", text.lower()):
                if word in _STOP:
                    continue
                word = word.rstrip("s")
                h = int(hashlib.md5(word.encode()).hexdigest(), 16)
                out[i, h % DIM] += 1.0
            norm = np.linalg.norm(out[i])
            if norm:
                out[i] /= norm
            else:
                out[i, 0] = 1.0
        return out


def extract_question(prompt: str) -> str:
    match = re.search(r"Question:\n(.*?)\n\nAnswer:", prompt, re.S)
    return match.group(1).strip() if match else ""


def extract_section(prompt: str, name: str) -> str:
    match = re.search(rf"{name}:\n(.*?)\n\n", prompt, re.S)
    return match.group(1).strip() if match else ""


class FakeLLM:
    """Scriptable fake for the LLMClient protocol.

    answer_fn(question, prompt) -> str | Exception   for RAG generation
    faith_fn(question, answer, prompt) -> dict | str | Exception
    correct_fn(question, answer, prompt) -> dict | str | Exception
    Defaults: echo the gold answer for the question; judge everything
    supported/correct.
    """

    def __init__(
        self,
        answers: dict[str, str] | None = None,
        answer_fn: Callable | None = None,
        faith_fn: Callable | None = None,
        correct_fn: Callable | None = None,
        usage: TokenUsage | None | str = "default",
        latency_ms: float = 120.0,
    ):
        usage = TokenUsage(400, 30, 100, 530) if usage == "default" else usage
        self.answers = answers or {}
        self.answer_fn = answer_fn
        self.faith_fn = faith_fn
        self.correct_fn = correct_fn
        self.usage = usage
        self.latency_ms = latency_ms
        self.calls: list[dict] = []

    def _respond(self, value, model):
        if isinstance(value, BaseException):
            raise value
        text = value if isinstance(value, str) else json.dumps(value)
        return LLMResponse(text=text, model=model, latency_ms=self.latency_ms, total_latency_ms=self.latency_ms,
                           attempts=1, usage=self.usage)

    def generate(self, prompt: str, *, model: str, json_schema: dict | None = None) -> LLMResponse:
        kind = "faith" if json_schema is FAITHFULNESS_SCHEMA else "correct" if json_schema is CORRECTNESS_SCHEMA else "answer"
        self.calls.append({"kind": kind, "prompt": prompt, "model": model})
        if kind == "answer":
            question = extract_question(prompt)
            value = self.answer_fn(question, prompt) if self.answer_fn else self.answers.get(question, "I don't have enough information to answer that.")
            return self._respond(value, model)
        question = extract_section(prompt, "QUESTION")
        match = re.search(r"(?:^|\n)(?:ASSISTANT )?ANSWER:\n(.*?)\n\n(?:Return JSON only|Grade with)", prompt, re.S)
        answer = match.group(1).strip() if match else ""
        if kind == "faith":
            value = self.faith_fn(question, answer, prompt) if self.faith_fn else {
                "claims": [] if "enough information" in answer else [{"claim": answer, "label": "supported", "evidence": "x"}],
                "abstained": "enough information" in answer,
            }
        else:
            value = self.correct_fn(question, answer, prompt) if self.correct_fn else {
                "verdict": "correct", "correctness_score": 1.0, "relevance_score": 0.9,
                "abstained": "enough information" in answer, "issues": [], "reasoning": "ok",
            }
        return self._respond(value, model)


class FakeClock:
    """Monotonic clock that advances a fixed step per call."""

    def __init__(self, step_s: float = 0.05):
        self.t = 0.0
        self.step = step_s

    def __call__(self) -> float:
        self.t += self.step
        return self.t
