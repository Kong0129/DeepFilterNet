"""Score Chinese character error rate against a human-checked transcript.

Predictions CSV columns: case,strategy,prediction. This script does not run ASR.
Example: python scripts/evaluate_cer.py --reference transcript.txt --predictions predictions.csv
"""

import argparse
import csv
import unicodedata
from pathlib import Path


def normalize(text):
    """Ignore spaces and punctuation, but keep Chinese characters and numbers."""
    return "".join(
        char.casefold()
        for char in text
        if not (char.isspace() or unicodedata.category(char).startswith(("P", "S")))
    )


def edit_distance(reference, prediction):
    """Minimum character insertions, deletions and substitutions."""
    previous = list(range(len(prediction) + 1))
    for row, expected in enumerate(reference, start=1):
        current = [row]
        for column, observed in enumerate(prediction, start=1):
            current.append(min(
                previous[column] + 1,                 # delete a reference character
                current[column - 1] + 1,              # insert a predicted character
                previous[column - 1] + (expected != observed),  # substitute
            ))
        previous = current
    return previous[-1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("results/cer_scores.csv"))
    args = parser.parse_args()

    reference = normalize(args.reference.read_text(encoding="utf-8-sig"))
    if not reference:
        parser.error("Reference transcript is empty after normalization")

    with args.predictions.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        required = {"case", "strategy", "prediction"}
        if not required.issubset(reader.fieldnames or []):
            parser.error("Predictions CSV needs case,strategy,prediction columns")
        rows = []
        for item in reader:
            prediction = normalize(item["prediction"])
            errors = edit_distance(reference, prediction)
            rows.append({
                "case": item["case"],
                "strategy": item["strategy"],
                "reference_chars": len(reference),
                "edit_errors": errors,
                "cer": errors / len(reference),
                "prediction": item["prediction"],
            })

    if not rows:
        parser.error("Predictions CSV has no data rows")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Scored {len(rows)} predictions; saved {args.output}")


if __name__ == "__main__":
    main()
