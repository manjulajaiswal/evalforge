from app.loader import load_documents

documents = load_documents()

print(f"Loaded {len(documents)} documents")

for document in documents:
    print("=" * 50)
    print(document["source"])
    print(document["content"])
