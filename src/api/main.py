"""
main.py

The FastAPI application. Exposes:
  POST /predict     — upload an image, get back a species prediction
                       (every prediction also gets logged to the database)
  GET  /predictions  — list recent logged predictions (useful for testing,
                        and a preview of what Phase 5's dashboard will read)
  GET  /health       — simple check that the API is running

Run from project root:
    uvicorn src.api.main:app --reload
"""

import io

from fastapi import FastAPI, UploadFile, File, Form, Depends
from PIL import Image
from sqlalchemy.orm import Session

from .database import init_db, get_db, Prediction
from .inference import load_model, predict_species

app = FastAPI(title="WildSense API")


@app.on_event("startup")
def startup():
    """Runs once when the API starts — loads the model and sets up the DB."""
    init_db()
    load_model()


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    latitude: float = Form(None),
    longitude: float = Form(None),
    db: Session = Depends(get_db),
):
    """
    Accepts an uploaded image (and optional latitude/longitude), runs the
    species classifier, logs the result to the database, and returns the
    prediction as JSON.
    """
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes))

    result = predict_species(image)

    # Log this prediction to the database, regardless of confidence —
    # low-confidence predictions still get logged, just flagged for review
    # via needs_review, rather than being silently dropped.
    record = Prediction(
        image_filename=file.filename,
        species=result["species"],
        confidence=result["confidence"],
        needs_review=result["needs_review"],
        latitude=latitude,
        longitude=longitude,
    )
    db.add(record)
    db.commit()
    db.refresh(record)

    return {
        "id": record.id,
        "species": result["species"],
        "confidence": result["confidence"],
        "needs_review": result["needs_review"],
    }


@app.get("/predictions")
def list_predictions(limit: int = 20, db: Session = Depends(get_db)):
    """
    Returns the most recent logged predictions. Mainly useful right now
    for confirming /predict is actually writing to the database — Phase 5's
    dashboard will eventually query this same table directly.
    """
    records = (
        db.query(Prediction)
        .order_by(Prediction.created_at.desc())
        .limit(limit)
        .all()
    )
    return [
        {
            "id": r.id,
            "species": r.species,
            "confidence": r.confidence,
            "needs_review": r.needs_review,
            "latitude": r.latitude,
            "longitude": r.longitude,
            "created_at": r.created_at.isoformat(),
        }
        for r in records
    ]