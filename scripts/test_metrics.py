from app.metrics import refund_days_match
from app.metrics import semantic_similarity


expected = "Customers can request a refund within 7 days of purchase."

correct_different = "Customers have one week from the date they purchased the product to request a refund."

incorrect = "Customers can request a refund within 30 days of purchase."

unrelated = "Standard shipping takes 3 to 5 business days."


print("Correct but differently worded:")
print(semantic_similarity(expected, correct_different))

print()

print("Incorrect:")
print(semantic_similarity(expected, incorrect))

print()

print("Unrelated:")
print(semantic_similarity(expected, unrelated))

expected = "Customers can request a refund within 7 days of purchase."

correct_different = (
    "Customers have one week from the date they purchased "
    "the product to request a refund."
)

incorrect = "Customers can request a refund within 30 days of purchase."

print("\nRefund period validation:")

print(
    "Correct paraphrase:",
    refund_days_match(expected, correct_different)
)

print(
    "Incorrect refund period:",
    refund_days_match(expected, incorrect)
)