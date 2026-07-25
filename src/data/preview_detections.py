"""
preview_detections.py

Quick visual sanity-check: draws MegaDetector's bounding box + confidence
score on top of a few sample images, so you can eyeball whether detections
look correct. Saves annotated copies to data/processed/preview/ instead of
popping up windows (simpler and works the same on any machine).

Run from project root:
    python src/data/preview_detections.py
"""

import csv
import ast
import random
from pathlib import Path

from PIL import Image, ImageDraw

METADATA_PATH = Path("data/processed/metadata.csv")
OUTPUT_DIR = Path("data/processed/preview")
NUM_SAMPLES = 8  # how many random images to preview


def main():
    with open(METADATA_PATH, "r", newline="") as f:
        rows = list(csv.DictReader(f))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    sample = random.sample(rows, min(NUM_SAMPLES, len(rows)))

    for row in sample:
        image_path = Path(row["image_path"])
        if not image_path.exists():
            print(f"Missing: {image_path}")
            continue

        img = Image.open(image_path).convert("RGB")
        draw = ImageDraw.Draw(img)

        bbox_str = row["bbox"]
        if bbox_str and bbox_str != "None":
            x1, y1, x2, y2 = ast.literal_eval(bbox_str)
            draw.rectangle([x1, y1, x2, y2], outline="red", width=4)

        label = f"{row['species_label']} ({float(row['detection_confidence']):.2f})"
        draw.text((10, 10), label, fill="red")

        out_path = OUTPUT_DIR / f"preview_{image_path.stem}.jpg"
        img.save(out_path)
        print(f"Saved {out_path}  ->  label: {label}")


if __name__ == "__main__":
    main()