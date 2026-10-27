from __future__ import annotations

import os

import pytest

from app import embedder, gemini_client, vector_store
from app.dataset import load_and_validate
from app.ingestion import ingest_knowledge_base
from tests.fakes import FakeEmbedder, FakeLLM


@pytest.fixture(autouse=True)
def offline_environment(request, monkeypatch):
    """Every offline test runs with the fake embedder and no real Gemini client.

    The real API key is removed so an accidental live call fails loudly
    instead of spending money. Tests marked `live` are left untouched.
    """
    if request.node.get_closest_marker("live"):
        yield
        return
    for name in ("GEMINI_API_KEY", "GROQ_API_KEY"):
        monkeypatch.delenv(name, raising=False)
        monkeypatch.setattr(f"app.config.{name}", None)
    monkeypatch.setattr("app.config.LLM_PROVIDER", "gemini")
    embedder.set_embedding_model(FakeEmbedder())
    gemini_client.set_llm_client(None)
    vector_store.set_vector_store(None)
    yield
    embedder.set_embedding_model(None)
    gemini_client.set_llm_client(None)
    vector_store.set_vector_store(None)


@pytest.fixture
def store():
    s = vector_store.make_in_memory_store()
    yield s
    s.client.delete_collection(s.collection_name)


@pytest.fixture
def ingested_store(store):
    ingest_knowledge_base(store)
    return store


@pytest.fixture(scope="session")
def golden_cases():
    return load_and_validate()


@pytest.fixture
def oracle_llm(golden_cases):
    """Answers every question with its gold answer (a perfect system)."""
    return FakeLLM(answers={c.question: c.expected_answer for c in golden_cases})


@pytest.fixture
def report_paths(tmp_path):
    from scripts.run_eval import OutputPaths

    return OutputPaths(
        json=tmp_path / "latest_report.json",
        markdown=tmp_path / "latest_report.md",
        history=tmp_path / "history.jsonl",
        baseline=tmp_path / "baseline.json",
    )


def has_live_key() -> bool:
    return bool(os.environ.get("GEMINI_API_KEY"))
