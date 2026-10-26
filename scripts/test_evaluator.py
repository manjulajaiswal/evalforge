"""Manual LIVE smoke check: run the first few golden cases and print results.

Calls the real Gemini API (needs GEMINI_API_KEY and an ingested store).
Usage: python -m scripts.test_evaluator [N]   (default N=3)
For the full evaluation with reports and quality gates use scripts.run_eval.
"""

import sys

from app.evaluator import load_golden_dataset, run_test_case

limit = int(sys.argv[1]) if len(sys.argv) > 1 else 3
test_cases = load_golden_dataset()
print(f"Loaded {len(test_cases)} test cases; running the first {limit}")

for test_case in test_cases[:limit]:
    result = run_test_case(test_case)
    judge = result["answer_evaluation"]["judge"]
    print("=" * 60)
    print("ID:", result["id"])
    print("Question:", result["question"])
    print("Expected:", result["expected_answer"])
    print("Actual:", result["answer"])
    print("Retrieved sources:", [c["source"] for c in result["retrieval"]["chunks"]])
    print("Fact checks:", result["answer_evaluation"]["facts"]["results"])
    print("Judge verdict:", judge["verdict"], "| relevance:", judge["relevance_score"])
    print("Faithfulness:", result["faithfulness"]["score"], "| hallucination:", result["faithfulness"]["has_hallucination"])
    print("Semantic similarity (diagnostic):", result["answer_evaluation"]["semantic_similarity"])
    print("Result:", result["status"].upper(), "|", "; ".join(result["failure_reasons"]))
