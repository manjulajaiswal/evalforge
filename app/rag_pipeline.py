"""The RAG application under test: retrieve context, then ask Gemini.

This is the only RAG implementation in the project; the evaluator calls these
functions rather than re-implementing retrieval or generation.

``retrieve_context`` and ``generate_answer`` keep their original signatures.
``retrieve`` and ``generate_answer_detailed`` return the extra information
(ranks, distances, timings, token usage) that evaluation needs.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass

from app import config
from app.embedder import generate_embeddings
from app.gemini_client import LLMClient, LLMResponse, get_llm_client
from app.vector_store import VectorStore, get_vector_store

PROMPT_TEMPLATE = """
You are a helpful customer support assistant.

Answer the user's question using ONLY the information provided in the context below.

If the answer cannot be found in the context, say:
"I don't have enough information to answer that."

Do not make up information.

Context:
{context}

Question:
{question}

Answer:
"""

# Recorded in reports so that runs with different prompts are not compared.
PROMPT_VERSION = hashlib.sha256(PROMPT_TEMPLATE.encode("utf-8")).hexdigest()[:12]


@dataclass
class RetrievedChunk:
    rank: int  # 1 = most similar
    chunk_id: str
    source: str
    text: str
    distance: float | None


@dataclass
class RetrievalResult:
    chunks: list[RetrievedChunk]
    latency_ms: float  # query embedding + vector search

    @property
    def contexts(self) -> list[str]:
        return [c.text for c in self.chunks]

    @property
    def sources(self) -> list[str]:
        return [c.source for c in self.chunks]


def build_prompt(question: str, contexts: list[str]) -> str:
    return PROMPT_TEMPLATE.format(context="\n\n".join(contexts), question=question)


def retrieve(question: str, top_k: int | None = None, store: VectorStore | None = None) -> RetrievalResult:
    top_k = config.TOP_K if top_k is None else top_k
    store = store or get_vector_store()
    start = time.perf_counter()
    query_embedding = generate_embeddings([question])[0]
    results = store.search(query_embedding, n_results=top_k)
    latency_ms = (time.perf_counter() - start) * 1000

    ids = results["ids"][0]
    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = (results.get("distances") or [[None] * len(ids)])[0]
    chunks = [
        RetrievedChunk(
            rank=i + 1,
            chunk_id=ids[i],
            source=(metadatas[i] or {}).get("source", "<unknown>"),
            text=documents[i],
            distance=None if distances[i] is None else float(distances[i]),
        )
        for i in range(len(ids))
    ]
    return RetrievalResult(chunks=chunks, latency_ms=latency_ms)


def retrieve_context(question, n_results=3):
    """Original interface: returns (list of chunk texts, list of metadata dicts)."""
    result = retrieve(question, top_k=n_results)
    return result.contexts, [{"source": c.source} for c in result.chunks]


def generate_answer_detailed(
    question: str, contexts: list[str], *, llm: LLMClient | None = None, model: str | None = None
) -> LLMResponse:
    llm = llm or get_llm_client()
    return llm.generate(build_prompt(question, contexts), model=model or config.GEMINI_MODEL)


def generate_answer(question, contexts):
    """Original interface: returns only the answer text."""
    return generate_answer_detailed(question, contexts).text
