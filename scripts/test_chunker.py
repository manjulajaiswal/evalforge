from app.chunker import chunk_documents
from app.loader import load_documents

documents = load_documents()
chunks = chunk_documents(documents)

print(f"Loaded {len(documents)} documents")
print(f"Created {len(chunks)} chunks")

for chunk in chunks:
    print("=" * 50)
    print(f"Source: {chunk['source']}")
    print(chunk["content"])
