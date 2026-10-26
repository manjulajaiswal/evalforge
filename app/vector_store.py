import chromadb

client = chromadb.PersistentClient(path="./chroma_db")

COLLECTION_NAME = "evalforge_knowledge"

collection = client.get_or_create_collection(
    name=COLLECTION_NAME
)


def add_documents(ids, documents, embeddings, metadatas):
    collection.add(
        ids=ids,
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas,
    )


def clear_collection():
    collection.delete(
        where={}
    )

def search(query_embedding, n_results=3):
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        include=["documents", "metadatas", "distances"],
    )

    return results