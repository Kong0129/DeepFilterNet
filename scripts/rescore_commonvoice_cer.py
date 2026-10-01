"""Re-score saved Common Voice ASR texts after script/number normalization.

This does not rerun DeepFilterNet or Whisper. The original CSV stays intact.
Requires opencc-python-reimplemented==0.1.7 in scripts/_local_packages.
"""

import csv
import re
import sys
import unicodedata
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "_local_packages"))

from opencc import OpenCC

from evaluate_cer import edit_distance, normalize


ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = ROOT / "results" / "commonvoice_6speakers.csv"
OUTPUT_PATH = ROOT / "results" / "commonvoice_6speakers_cer_normalized.csv"
CONVERTER = OpenCC("t2s")
DIGITS = "零一二三四五六七八九"


def number_to_chinese(number):
    """Convert a small written integer (e.g. 20 -> 二十) for CER only."""
    value = int(number)
    if value == 0:
        return DIGITS[0]
    if value >= 10000:
        # Avoid guessing how a long number should be read; preserve its digits.
        return number
    units = ("", "十", "百", "千")
    padded = f"{value:04d}"
    pieces = []
    pending_zero = False
    for index, char in enumerate(padded):
        digit = int(char)
        if digit == 0:
            pending_zero = bool(pieces)
            continue
        if pending_zero:
            pieces.append("零")
            pending_zero = False
        pieces.append(DIGITS[digit] + units[3 - index])
    result = "".join(pieces)
    return result[1:] if result.startswith("一十") else result


def canonicalize(text):
    text = unicodedata.normalize("NFKC", text)
    text = CONVERTER.convert(text)
    text = re.sub(r"\d+", lambda match: number_to_chinese(match.group()), text)
    return normalize(text)


def score(reference, prediction):
    reference = canonicalize(reference)
    if not reference:
        raise ValueError("Empty reference after normalization")
    return edit_distance(reference, canonicalize(prediction)) / len(reference)


def main():
    with INPUT_PATH.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"No data rows in {INPUT_PATH}")
    for row in rows:
        reference = row["reference_text"]
        row["clean_cer"] = score(reference, row["clean_asr_text"])
        row["input_cer"] = score(reference, row["input_asr_text"])
        row["output_cer"] = score(reference, row["output_asr_text"])
        row["cer_improvement"] = float(row["input_cer"]) - float(row["output_cer"])
        row["cer_normalization"] = "OpenCC t2s; Arabic integers to Chinese numerals"
    with OUTPUT_PATH.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"已重算 {len(rows)} 行 CER: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
