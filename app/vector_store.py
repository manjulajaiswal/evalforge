"""ChromaDB persistence and similarity search.

Design notes
------------
* The database lives at ``config.CHROMA_DIR`` (absolute path) so ingestion
  and evaluation always share one store, whatever the working directory.
* ``sync_chunks`` makes ingestion idempotent: chunks are *upserted* (so edited
  documents replace stale text; plain ``add`` silently keeps the old text for
  an existing ID) and chunks that no longer exist are deleted.
* Embeddings are always supplied by app.embedder, so the collection is
  created with ``embedding_function=None`` and Chroma never downloads its own
  default model.
* Distance metric: Chroma's default (squared L2). MiniLM embeddings are
  unit-normalised, so the ranking is identical to cosine similarity.
"""

from __future__ import annotations

from typing import Any, Sequence

from app.config import CHROMA_COLLECTION_NAME, CHROMA_DIR

COLLECTION_NAME = CHROMA_COLLECTION_NAME


class VectorStore:
    def __init__(self, client: Any, collection_name: str = COLLECTION_NAME):
        self.client = client
        self.collection_name = collection_name
        self.collection = client.get_or_create_collection(name=collection_name, embedding_function=None)

    # -- writes ------------------------------------------------------------
    def upsert(self, ids, documents, embeddings, metadatas) -> None:
        if ids:
            self.collection.upsert(
                ids=list(ids),
                documents=list(documents),
                embeddings=[list(map(float, e)) for e in embeddings],
                metadatas=list(metadatas),
            )

    def sync_chunks(self, ids, documents, embeddings, metadatas) -> dict:
        """Make the collection contain exactly these chunks. Returns counts."""
        if len(set(ids)) != len(ids):
            raise ValueError("Chunk IDs must be unique")
        existing = set(self.collection.get(include=[])["ids"])
        stale = sorted(existing - set(ids))
        self.upsert(ids, documents, embeddings, metadatas)
        if stale:
            self.collection.delete(ids=stale)
        return {
            "upserted": len(ids),
            "new": len(set(ids) - existing),
            "deleted_stale": len(stale),
            "total": self.count(),
        }

    def set_metadata(self, metadata: dict) -> None:
        self.collection.modify(metadata=metadata)

    def clear(self) -> None:
        """Remove every chunk by dropping and recreating the collection."""
        self.client.delete_collection(self.collection_name)
        self.collection = self.client.get_or_create_collection(name=self.collection_name, embedding_function=None)

    # -- reads -------------------------------------------------------------
    def count(self) -> int:
        return self.collection.count()

    def metadata(self) -> dict:
        return dict(self.collection.metadata or {})

    def search(self, query_embedding: Sequence[float], n_results: int = 3) -> dict:
        return self.collection.query(
            query_embeddings=[list(map(float, query_embedding))],
            n_results=n_results,
            include=["documents", "metadatas", "distances"],
        )


_default_store: VectorStore | None = None


def get_vector_store() -> VectorStore:
    """The persistent store at CHROMA_DIR (created lazily)."""
    global _default_store
    if _default_store is None:
        import chromadb

        CHROMA_DIR.mkdir(parents=True, exist_ok=True)
        _default_store = VectorStore(chromadb.PersistentClient(path=str(CHROMA_DIR)))
    return _default_store


def set_vector_store(store: VectorStore | None) -> None:
    """Inject a store (tests and experiments use in-memory clients)."""
    global _default_store
    _default_store = store


def make_in_memory_store(collection_name: str | None = None) -> VectorStore:
    """In-memory store for tests/experiments.

    Chroma's EphemeralClient instances share one backend within a process, so
    each store gets a unique collection name unless one is given.
    """
    import uuid

    import chromadb

    return VectorStore(chromadb.EphemeralClient(), collection_name or f"mem_{uuid.uuid4().hex[:12]}")


# -- backwards-compatible module-level API --------------------------------
def add_documents(ids, documents, embeddings, metadatas) -> None:
    """Insert or update chunks (upsert, so repeated runs do not fail or go stale)."""
    get_vector_store().upsert(ids, documents, embeddings, metadatas)


def clear_collection() -> None:
    get_vector_store().clear()


def search(query_embedding, n_results: int = 3) -> dict:
    return get_vector_store().search(query_embedding, n_results=n_results)
