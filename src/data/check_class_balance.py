"""
check_class_balance.py

Prints how many images each species has in the final metadata.csv.
Run this before training to decide whether you need weighted loss or
oversampling to handle class imbalance (Phase 2).

Run from project root:
    python src/data/check_class_balance.py
"""

import csv
from collections import Counter
from pathlib import Path

METADATA_PATH = Path("data/processed/metadata.csv")


def main():
    with open(METADATA_PATH, "r", newline="") as f:
        rows = list(csv.DictReader(f))

    counts = Counter(row["species_label"] for row in rows)
    total = sum(counts.values())

    print(f"Total images: {total}\n")
    print(f"{'Species':<12} {'Count':>8} {'Percent':>10}")
    print("-" * 32)
    for species, count in counts.most_common():
        percent = 100 * count / total
        print(f"{species:<12} {count:>8} {percent:>9.1f}%")

    largest = counts.most_common(1)[0]
    smallest = counts.most_common()[-1]
    ratio = largest[1] / smallest[1]
    print(f"\nImbalance ratio (largest/smallest): {ratio:.1f}x "
          f"({largest[0]}: {largest[1]} vs {smallest[0]}: {smallest[1]})")


if __name__ == "__main__":
    main()