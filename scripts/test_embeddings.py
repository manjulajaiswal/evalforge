from sentence_transformers import SentenceTransformer
from sentence_transformers import SentenceTransformer, util
model = SentenceTransformer("all-MiniLM-L6-v2")

print("Embedding model loaded successfully")

texts = [
    "Customers can request a refund within 7 days.",
    "A refund can be requested within one week.",
    "Standard shipping takes 3 to 5 business days."
]

embeddings = model.encode(texts)

similarity_ab = util.cos_sim(embeddings[0], embeddings[1])
similarity_ac = util.cos_sim(embeddings[0], embeddings[2])
similarity_bc = util.cos_sim(embeddings[1], embeddings[2])

print("\nSimilarity A ↔ B:", similarity_ab.item())
print("Similarity A ↔ C:", similarity_ac.item())
print("Similarity B ↔ C:", similarity_bc.item())