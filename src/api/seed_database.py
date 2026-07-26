import csv
import random
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image

from .database import init_db, SessionLocal, Prediction
from .inference import load_model, predict_species

METADATA_PATH = Path("data/processed/metadata.csv")
NUM_SEED_IMAGES = 300 


def parse_timestamp(timestamp_str):
    """
    Parses the capture timestamp from metadata.csv (format from the
    original CCT-20 annotations, e.g. '2011-05-19 13:12:00'). Falls back
    to current time if parsing fails, rather than crashing the whole run
    over one malformed row.
    """
    try:
        return datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
    except (ValueError, TypeError):
        return datetime.now(timezone.utc)


def main():
    print("Loading model...")
    init_db()
    load_model()

    print(f"Loading metadata from {METADATA_PATH} ...")
    with open(METADATA_PATH, "r", newline="") as f:
        rows = list(csv.DictReader(f))

    sample = random.sample(rows, min(NUM_SEED_IMAGES, len(rows)))
    print(f"Seeding database with {len(sample)} predictions...")

    db = SessionLocal()
    logged = 0
    skipped = 0

    for i, row in enumerate(sample, start=1):
        image_path = Path(row["image_path"])
        if not image_path.exists():
            skipped += 1
            continue

        try:
            image = Image.open(image_path)
            result = predict_species(image)
        except Exception as e:
            print(f"  Skipping {image_path.name}: {e}")
            skipped += 1
            continue

        record = Prediction(
            image_filename=image_path.name,
            species=result["species"],
            confidence=result["confidence"],
            needs_review=result["needs_review"],
            latitude=float(row["latitude"]),
            longitude=float(row["longitude"]),
            created_at=parse_timestamp(row["timestamp"]),
        )
        db.add(record)
        logged += 1

        if i % 50 == 0:
            db.commit()  
            print(f"  [{i}/{len(sample)}] processed")

    db.commit()
    db.close()

    print(f"\nDone. Logged {logged} predictions, skipped {skipped}.")


if __name__ == "__main__":
    main()