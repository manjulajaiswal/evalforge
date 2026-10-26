from app.embedder import generate_embeddings

texts = [
    "Customers can request a refund within 7 days.",
    "A refund can be requested within one week.",
    "Standard shipping takes 3 to 5 business days."
]

embeddings = generate_embeddings(texts)

print("Number of embeddings:", len(embeddings))
print("Embedding dimensions:", len(embeddings[0]))
