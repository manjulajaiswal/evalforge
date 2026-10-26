"""Knowledge-base ingestion: load -> chunk -> embed -> store.

Running ingestion twice with the same inputs leaves the store unchanged.
Changing a document updates its chunks; deleting a document removes them.
The chunking/embedding configuration is recorded in the collection metadata
so that evaluation reports can state exactly what was indexed.
"""

from __future__ import annotations

from pathlib import Path

from app import config
from app.chunker import chunk_with_strategy
from app.embedder import MODEL_NAME, generate_embeddings
from app.loader import load_documents
from app.vector_store import VectorStore, get_vector_store


def chunk_id(source: str, chunk_index: int) -> str:
    return f"{source}::chunk_{chunk_index}"


def ingest_knowledge_base(
    store: VectorStore | None = None,
    *,
    directory: Path | None = None,
    strategy: str | None = None,
    chunk_size: int | None = None,
    overlap: int | None = None,
    rebuild: bool = False,
) -> dict:
    store = store or get_vector_store()
    strategy = strategy or config.CHUNK_STRATEGY
    chunk_size = chunk_size if chunk_size is not None else config.CHUNK_SIZE
    overlap = overlap if overlap is not None else config.CHUNK_OVERLAP

    documents = load_documents(directory)
    chunks = chunk_with_strategy(documents, strategy, chunk_size, overlap)
    texts = [chunk["content"] for chunk in chunks]
    embeddings = generate_embeddings(texts)

    if rebuild:
        store.clear()

    ids = [chunk_id(c["source"], c["chunk_index"]) for c in chunks]
    metadatas = [{"source": c["source"], "chunk_index": c["chunk_index"]} for c in chunks]
    stats = store.sync_chunks(ids, texts, embeddings, metadatas)

    index_config = {
        "chunk_strategy": strategy,
        "chunk_size": chunk_size,
        "chunk_overlap": overlap,
        "embedding_model": MODEL_NAME,
        "document_count": len(documents),
        "chunk_count": len(chunks),
    }
    store.set_metadata(index_config)

    return {
        **stats,
        "documents": len(documents),
        "chunks": len(chunks),
        "embedding_dimensions": int(len(embeddings[0])) if len(embeddings) else 0,
        "index_config": index_config,
    }
