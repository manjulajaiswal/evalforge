from app.evaluator import load_golden_dataset, run_test_case


test_cases = load_golden_dataset()

print(f"Loaded {len(test_cases)} test cases")

for test_case in test_cases:
    result = run_test_case(test_case)

    print("=" * 60)
    print("ID:", result["id"])
    print("Question:", result["question"])
    print("Expected:", result["expected_answer"])
    print("Actual:", result["actual_answer"])
    print("Similarity:", result["similarity"])
    print("Similarity check:", result["similarity_pass"])
    print("Refund period check:", result["refund_period_pass"])
    print("Result:", "PASS" if result["passed"] else "FAIL")