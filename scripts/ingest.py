"""Load the knowledge base, chunk it, embed it and store it in ChromaDB.

Usage (repository root, virtual environment active; no API key needed):

    python -m scripts.ingest            # sync: safe to run repeatedly
    python -m scripts.ingest --rebuild  # drop the collection and rebuild it from scratch

Run it after cloning, and again whenever a file in data/knowledge_base/ or the
chunking settings change.
"""

from __future__ import annotations

import argparse
import sys

from app import config
from app.errors import ConfigurationError
from app.ingestion import ingest_knowledge_base


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Ingest the knowledge base into ChromaDB.")
    parser.add_argument("--rebuild", action="store_true", help="Delete and recreate the collection first.")
    args = parser.parse_args(argv)

    try:
        stats = ingest_knowledge_base(rebuild=args.rebuild)
    except ConfigurationError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    print(f"Loaded {stats['documents']} documents")
    print(f"Created {stats['chunks']} chunks ({stats['index_config']['chunk_strategy']}, "
          f"size {stats['index_config']['chunk_size']}, overlap {stats['index_config']['chunk_overlap']})")
    print(f"Embedding dimensions: {stats['embedding_dimensions']}")
    print(f"New chunks: {stats['new']} | Updated/unchanged: {stats['upserted'] - stats['new']} | "
          f"Stale chunks removed: {stats['deleted_stale']}")
    print(f"Documents stored in ChromaDB at {config.CHROMA_DIR} ({stats['total']} chunks in collection)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
