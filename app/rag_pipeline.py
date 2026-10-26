from app.embedder import generate_embeddings
from app.vector_store import search
from google import genai
from app.config import GEMINI_API_KEY, GEMINI_MODEL

from app.config import GEMINI_API_KEY
from app.embedder import generate_embeddings
from app.vector_store import search


client = genai.Client(api_key=GEMINI_API_KEY)


def retrieve_context(question, n_results=3):
    query_embedding = generate_embeddings([question])[0]

    results = search(
        query_embedding,
        n_results=n_results,
    )

    contexts = results["documents"][0]
    sources = results["metadatas"][0]

    return contexts, sources


def generate_answer(question, contexts):
    context = "\n\n".join(contexts)

    prompt = f"""
You are a helpful customer support assistant.

Answer the user's question using ONLY the information provided in the context below.

If the answer cannot be found in the context, say:
"I don't have enough information to answer that."

Do not make up information.

Context:
{context}

Question:
{question}

Answer:
"""

    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
    )

    return response.text