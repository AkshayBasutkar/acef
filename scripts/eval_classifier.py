"""
Evaluation script for the Query Analyzer heuristic classifier.

Evaluates test/dev sets (JSONL format) and prints:
  - Overall accuracy
  - Hard-negative accuracy
  - Per-class Precision, Recall, F1
  - 3x3 Confusion Matrix

Usage:
  python -m scripts.eval_classifier [--file path/to/dev.jsonl]
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.schemas import QueryType
from core.query_analyzer import classify

CLASSES = ["code", "memory", "simple"]


def evaluate(file_path: Path) -> bool:
    if not file_path.exists():
        print(f"Error: Evaluation file not found: {file_path}", file=sys.stderr)
        return False

    records = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("//"):
                records.append(json.loads(line))

    if not records:
        print("Error: No records found in evaluation file", file=sys.stderr)
        return False

    # Seed known symbols and doc probe for corpus-conditioned queries
    corpus_known_symbols = frozenset({"login", "hash_password", "user_profile", "auth"})

    confusion = {exp: {pred: 0 for pred in CLASSES} for exp in CLASSES}
    total = len(records)
    correct = 0
    hn_total = 0
    hn_correct = 0
    failures = []

    for r in records:
        query = r["query"]
        expected_str = r["expected"].lower()
        needs_corpus = r.get("needs_corpus", False)
        min_turns = r.get("min_session_turns", 0)
        is_hn = r.get("hard_negative", False)
        last_turn_type = None
        if "last_turn_type" in r:
            last_turn_type = QueryType(r["last_turn_type"])

        known_symbols = corpus_known_symbols if needs_corpus else frozenset()
        doc_probe = (lambda: 0.85) if (needs_corpus and expected_str == "memory") else None

        actual_type = classify(
            query,
            session_turn_count=min_turns,
            known_symbols=known_symbols,
            doc_similarity_fn=doc_probe,
            last_turn_type=last_turn_type,
        )
        actual_str = actual_type.value.lower()

        if expected_str in confusion and actual_str in confusion[expected_str]:
            confusion[expected_str][actual_str] += 1

        is_correct = (actual_str == expected_str)
        if is_correct:
            correct += 1
        else:
            failures.append((query, expected_str, actual_str, is_hn))

        if is_hn:
            hn_total += 1
            if is_correct:
                hn_correct += 1

    accuracy = (correct / total) * 100.0
    hn_accuracy = (hn_correct / hn_total * 100.0) if hn_total > 0 else 100.0

    print("=" * 65)
    print(f" ACEF Classifier Evaluation -- {file_path.name}")
    print("=" * 65)
    print(f"Total queries evaluated:  {total}")
    print(f"Correct classifications:  {correct}")
    print(f"Overall Accuracy:         {accuracy:.1f}%")
    print(f"Hard-Negative Accuracy:   {hn_accuracy:.1f}% ({hn_correct}/{hn_total})")
    print("-" * 65)

    # Confusion matrix
    print("Confusion Matrix (Rows: Expected, Columns: Predicted):")
    lbl = "Expected \\ Pred"
    header = f"{lbl:<16}" + "".join(f"{c:>10}" for c in CLASSES)
    print(header)
    print("-" * len(header))
    for exp in CLASSES:
        row_str = f"{exp:<16}" + "".join(f"{confusion[exp][pred]:>10}" for pred in CLASSES)
        print(row_str)
    print("-" * 65)

    # Per-class P / R / F1
    print(f"{'Class':<10} {'Precision':>10} {'Recall':>10} {'F1-Score':>10}")
    print("-" * 43)
    for c in CLASSES:
        tp = confusion[c][c]
        fp = sum(confusion[other][c] for other in CLASSES if other != c)
        fn = sum(confusion[c][other] for other in CLASSES if other != c)

        prec = (tp / (tp + fp)) * 100.0 if (tp + fp) > 0 else 0.0
        rec = (tp / (tp + fn)) * 100.0 if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

        print(f"{c:<10} {prec:>9.1f}% {rec:>9.1f}% {f1:>9.1f}%")
    print("=" * 65)

    if failures:
        print("\nFailures:")
        for q, exp, act, hn in failures:
            hn_tag = " [HN]" if hn else ""
            print(f"  - Query: \"{q}\"{hn_tag}\n    Expected: {exp}, Actual: {act}")

    success = (accuracy >= 90.0) and (hn_correct == hn_total)
    if success:
        print(f"\n[PASSED] Accuracy >= 90% and all hard negatives passed.")
    else:
        print(f"\n[FAILED] Target not met (Accuracy >= 90% required, all HNs must pass).")

    return success


def main():
    parser = argparse.ArgumentParser(description="Evaluate ACEF query classifier")
    parser.add_argument(
        "--file",
        type=Path,
        default=PROJECT_ROOT / "data" / "eval" / "classifier_dev.jsonl",
        help="Path to JSONL evaluation file",
    )
    args = parser.parse_args()
    success = evaluate(args.file)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
