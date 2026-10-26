"""Validate data/golden_dataset.jsonl and print a coverage summary.

Usage: python -m scripts.validate_dataset [path]
Exit code 0 = valid, 2 = invalid. No API key or model download needed.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from app.dataset import load_and_validate, summarize
from app.errors import DatasetValidationError


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    path = Path(argv[0]) if argv else None
    try:
        cases = load_and_validate(path)
    except DatasetValidationError as exc:
        print(f"INVALID: {exc}", file=sys.stderr)
        return 2
    print(f"Dataset is valid: {len(cases)} cases")
    print(json.dumps(summarize(cases), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
