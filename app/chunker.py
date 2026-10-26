"""Split documents into chunks for embedding.

``chunk_documents`` is the original fixed-size character chunker and remains
the default. ``chunk_documents_by_paragraph`` is an alternative strategy used
only when CHUNK_STRATEGY=paragraph, so its effect can be measured with
scripts/retrieval_experiment.py before anyone adopts it.
"""

from __future__ import annotations

import re
from typing import Dict, List


def _validate(chunk_size: int, overlap: int) -> None:
    if chunk_size < 1:
        raise ValueError("Chunk size must be at least 1")
    if overlap < 0:
        raise ValueError("Overlap must not be negative")
    if overlap >= chunk_size:
        raise ValueError("Overlap must be smaller than chunk size")


def chunk_documents(
    documents: List[Dict],
    chunk_size: int = 500,
    overlap: int = 50,
) -> List[Dict]:
    """Fixed-size character windows that advance by ``chunk_size - overlap``.

    Each chunk carries ``source``, ``content`` and a per-source ``chunk_index``.
    Whitespace-only windows are skipped.
    """
    _validate(chunk_size, overlap)
    step = chunk_size - overlap
    chunks: List[Dict] = []

    for document in documents:
        content = document["content"]
        source = document["source"]
        index = 0

        for start in range(0, len(content), step):
            chunk_text = content[start:start + chunk_size]
            if not chunk_text.strip():
                continue
            chunks.append({"source": source, "content": chunk_text, "chunk_index": index})
            index += 1
            # The previous window already reached the end of the document; any
            # further window would only repeat its tail.
            if start + chunk_size >= len(content):
                break

    return chunks


def chunk_documents_by_paragraph(
    documents: List[Dict],
    chunk_size: int = 500,
    overlap: int = 0,
) -> List[Dict]:
    """One chunk per Markdown paragraph (blank-line separated).

    Headings are prepended to the paragraphs that follow them, so each chunk
    keeps its topic (e.g. "# Refund Policy"). Paragraphs longer than
    ``chunk_size`` fall back to character windows. ``overlap`` only applies to
    that fallback.
    """
    _validate(chunk_size, overlap)
    chunks: List[Dict] = []

    for document in documents:
        source = document["source"]
        heading = ""
        index = 0
        for block in re.split(r"\n\s*\n", document["content"]):
            block = block.strip()
            if not block:
                continue
            if block.startswith("#") and "\n" not in block:
                heading = block
                continue
            text = f"{heading}\n{block}" if heading else block
            if len(text) <= chunk_size:
                pieces = [text]
            else:
                pieces = [
                    piece["content"]
                    for piece in chunk_documents(
                        [{"source": source, "content": text}], chunk_size, overlap
                    )
                ]
            for piece in pieces:
                chunks.append({"source": source, "content": piece, "chunk_index": index})
                index += 1

    return chunks


def chunk_with_strategy(documents: List[Dict], strategy: str, chunk_size: int, overlap: int) -> List[Dict]:
    if strategy == "character":
        return chunk_documents(documents, chunk_size, overlap)
    if strategy == "paragraph":
        return chunk_documents_by_paragraph(documents, chunk_size, overlap)
    raise ValueError(f"Unknown chunk strategy: {strategy!r}")
