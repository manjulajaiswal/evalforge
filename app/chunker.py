from typing import List, Dict


def chunk_documents(
    documents: List[Dict],
    chunk_size: int = 500,
    overlap: int = 50,
):
    chunks = []

    for document in documents:
        content = document["content"]
        source = document["source"]

        if overlap >= chunk_size:
            raise ValueError("Overlap must be smaller than chunk size")

        step = chunk_size - overlap

        for start in range(0, len(content), step):
            chunk_text = content[start:start + chunk_size]

            if not chunk_text.strip():
                continue

            chunks.append(
                {
                    "source": source,
                    "content": chunk_text,
                }
            )

    return chunks