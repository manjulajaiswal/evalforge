from pathlib import Path

import numpy as np

from app import config
from app.ingestion import chunk_id, ingest_knowledge_base
from app.rag_pipeline import retrieve, retrieve_context
from app.vector_store import get_vector_store, set_vector_store


def write_kb(directory: Path, files: dict[str, str]) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    for p in directory.glob("*.md"):
        p.unlink()
    for name, text in files.items():
        (directory / name).write_text(text, encoding="utf-8")
    return directory


def test_default_store_uses_configured_absolute_path():
    assert config.CHROMA_DIR.is_absolute()
    assert config.CHROMA_DIR == config.ROOT_DIR / ".chroma"


def test_ingestion_is_idempotent(store):
    first = ingest_knowledge_base(store)
    second = ingest_knowledge_base(store)
    assert first["chunks"] == second["chunks"] == store.count() == 3
    assert second["new"] == 0 and second["deleted_stale"] == 0


def test_edited_document_replaces_stale_text(store, tmp_path):
    kb = write_kb(tmp_path / "kb", {"a.md": "Refunds within 7 days."})
    ingest_knowledge_base(store, directory=kb)
    write_kb(kb, {"a.md": "Refunds within 10 days."})
    ingest_knowledge_base(store, directory=kb)
    docs = store.collection.get(ids=[chunk_id("a.md", 0)])["documents"]
    assert docs == ["Refunds within 10 days."]


def test_removed_document_chunks_are_deleted(store, tmp_path):
    kb = write_kb(tmp_path / "kb", {"a.md": "alpha", "b.md": "beta"})
    ingest_knowledge_base(store, directory=kb)
    write_kb(kb, {"a.md": "alpha"})
    stats = ingest_knowledge_base(store, directory=kb)
    assert stats["deleted_stale"] == 1 and store.count() == 1


def test_rebuild_and_clear(store):
    ingest_knowledge_base(store)
    store.clear()
    assert store.count() == 0
    stats = ingest_knowledge_base(store, rebuild=True)
    assert stats["total"] == 3


def test_index_config_is_recorded(store):
    ingest_knowledge_base(store, strategy="paragraph")
    meta = store.metadata()
    assert meta["chunk_strategy"] == "paragraph" and meta["chunk_count"] == store.count() == 12
    assert meta["embedding_model"] == config.EMBEDDING_MODEL_NAME


def test_whole_document_chunks_with_top_k_3_return_every_document(ingested_store):
    """Reproduces the historical observation: a refund question 'retrieved'
    account and shipping documents. With 3 single-chunk documents and
    top-k=3, every query returns all three, independent of ranking quality."""
    result = retrieve("How many days do customers have to request a refund?", top_k=3, store=ingested_store)
    assert sorted(result.sources) == ["account_policy.md", "refund_policy.md", "shipping_policy.md"]
    assert result.chunks[0].source == "refund_policy.md"  # ranking itself is fine (fake embedder)
    assert [c.rank for c in result.chunks] == [1, 2, 3]
    assert result.chunks[0].distance <= result.chunks[1].distance


def test_original_retrieve_context_interface(ingested_store):
    set_vector_store(ingested_store)
    assert get_vector_store() is ingested_store
    contexts, sources = retrieve_context("express shipping", n_results=2)
    assert len(contexts) == 2 and all(set(s) == {"source"} for s in sources)


def test_upsert_accepts_numpy_embeddings(store):
    store.upsert(["x"], ["text"], np.array([[0.1, 0.2, 0.3]]), [{"source": "s.md"}])
    assert store.count() == 1
