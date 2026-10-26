import numpy as np

from app.embedder import generate_embeddings


def semantic_similarity(expected, actual):
    embeddings = generate_embeddings([expected, actual])

    expected_embedding = embeddings[0]
    actual_embedding = embeddings[1]

    similarity = np.dot(
        expected_embedding,
        actual_embedding
    ) / (
        np.linalg.norm(expected_embedding)
        * np.linalg.norm(actual_embedding)
    )

    return float(similarity)

def refund_days_match(expected, actual):
    import re

    def extract_days(text):
        text = text.lower()

        word_to_number = {
            "one": 1,
            "two": 2,
            "three": 3,
            "four": 4,
            "five": 5,
            "six": 6,
            "seven": 7,
            "eight": 8,
            "nine": 9,
            "ten": 10,
        }

        match = re.search(
            r"\b(\d+|one|two|three|four|five|six|seven|eight|nine|ten)"
            r"\s+(?:calendar\s+)?days?\b",
            text,
        )

        if match:
            value = match.group(1)
            return int(value) if value.isdigit() else word_to_number[value]

        if re.search(r"\bone\s+week\b", text):
            return 7

        return None

    expected_days = extract_days(expected)
    actual_days = extract_days(actual)

    return (
        expected_days is not None
        and actual_days is not None
        and expected_days == actual_days
    )