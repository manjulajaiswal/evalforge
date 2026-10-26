from __future__ import annotations

from pathlib import Path
from typing import Dict, List

from app.config import KNOWLEDGE_BASE_DIR
from app.errors import ConfigurationError


def load_documents(directory: Path | None = None) -> List[Dict]:
    """Load every Markdown file in the knowledge-base directory, sorted by name.

    Returns dicts with ``source`` (the file name) and ``content``.
    Raises ConfigurationError if the directory is missing or has no .md files,
    because ingesting an empty knowledge base would make every evaluation fail
    for a confusing reason.
    """
    directory = Path(directory) if directory is not None else KNOWLEDGE_BASE_DIR
    if not directory.is_dir():
        raise ConfigurationError(f"Knowledge-base directory not found: {directory}")

    documents: List[Dict] = []
    for file_path in sorted(directory.glob("*.md")):
        content = file_path.read_text(encoding="utf-8")
        if not content.strip():
            continue
        documents.append({"source": file_path.name, "content": content})

    if not documents:
        raise ConfigurationError(f"No non-empty .md files found in {directory}")
    return documents
