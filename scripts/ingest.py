from app.loader import load_documents
from app.chunker import chunk_documents
from app.embedder import generate_embeddings
from app.vector_store import add_documents

documents = load_documents()

print(f"Loaded {len(documents)} documents")

chunks = chunk_documents(documents)

print(f"Created {len(chunks)} chunks")

texts = [chunk["content"] for chunk in chunks]

embeddings = generate_embeddings(texts)

print(f"Generated {len(embeddings)} embeddings")
print(f"Embedding dimensions: {len(embeddings[0])}")


ids = [
    f"{chunk['source']}_chunk_{i}"
    for i, chunk in enumerate(chunks)
]

metadatas = [
    {"source": chunk["source"]}
    for chunk in chunks
]

add_documents(
    ids=ids,
    documents=texts,
    embeddings=embeddings,
    metadatas=metadatas,
)

print("Documents stored in ChromaDB")