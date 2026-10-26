import json

from app.config import GOLDEN_DATASET_PATH
from app.embedder import generate_embeddings
from app.vector_store import search
from app.rag_pipeline import generate_answer
from app.metrics import semantic_similarity, refund_days_match

def load_golden_dataset():
    test_cases = []

    with open(GOLDEN_DATASET_PATH, "r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                test_cases.append(json.loads(line))

    return test_cases


def run_test_case(test_case):
    question = test_case["question"]

    # 1. Convert the question into an embedding
    query_embedding = generate_embeddings([question])[0]

    # 2. Search ChromaDB for relevant context
    results = search(query_embedding)

    # 3. Extract retrieved documents
    contexts = results["documents"][0]

    # 4. Generate an answer using Gemini + retrieved context
    answer = generate_answer(question, contexts)

    # 5. Compare generated answer with expected answer
    similarity = semantic_similarity(
    test_case["expected_answer"],
    answer
    )

    refund_period_match = refund_days_match(
    test_case["expected_answer"],
    answer
    )
    similarity_pass = similarity >= 0.80

    refund_period_pass = (
        refund_period_match
        if "refund" in question.lower()
        else True
    )

    passed = similarity_pass and refund_period_pass

    return {
        "id": test_case["id"],
        "question": question,
        "expected_answer": test_case["expected_answer"],
        "expected_source": test_case["expected_source"],
        "actual_answer": answer,
        "similarity": similarity,
        "refund_period_match": refund_period_match,
        "similarity_pass": similarity_pass,
        "refund_period_pass": refund_period_pass,
        "passed": passed,
    }