"""
export_snapshot.py

Exports your local, fully-seeded database to a CSV snapshot that gets
committed to the repo. The deployed dashboard (Streamlit Community Cloud)
has no access to your local wildsense.db, so it falls back to reading this
snapshot instead — see app.py's load_predictions(), which tries the live
database first and only falls back to this file if that's empty or
unavailable.

Run from project root:
    python -m src.api.export_snapshot
"""

import csv
from pathlib import Path

from .database import SessionLocal, Prediction

OUTPUT_PATH = Path("data/processed/predictions_snapshot.csv")


def main():
    db = SessionLocal()
    records = db.query(Prediction).order_by(Prediction.created_at).all()
    db.close()

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = ["species", "confidence", "needs_review", "latitude", "longitude", "created_at"]
    with open(OUTPUT_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in records:
            writer.writerow({
                "species": r.species,
                "confidence": r.confidence,
                "needs_review": r.needs_review,
                "latitude": r.latitude,
                "longitude": r.longitude,
                "created_at": r.created_at.isoformat(),
            })

    print(f"Exported {len(records)} records to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()