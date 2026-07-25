"""
inference.py

Loads the trained model once (expensive — happens at API startup, not per
request) and provides a single function to run prediction on one image.
"""

import sys
from pathlib import Path

import torch
from PIL import Image
from huggingface_hub import hf_hub_download

# Reuse the exact same transform and model-building logic from training,
# so preprocessing at inference time matches preprocessing during training
# exactly. Mismatched preprocessing is a classic, hard-to-spot bug — the
# model would run without errors but give quietly wrong predictions.
sys.path.append(str(Path(__file__).parent.parent / "training"))
from model import build_model
from dataset import load_metadata, build_label_mapping
from config import val_transform, DEVICE, METADATA_PATH, CHECKPOINT_DIR

CHECKPOINT_PATH = CHECKPOINT_DIR / "best_model.pt"

# Hugging Face Hub fallback — used when the checkpoint isn't present locally
# (e.g. on a fresh deploy where models/ isn't in git). Swap in your own
# repo if you're following along with a different upload.
HF_REPO_ID = "Soumybug/wildsense-resnet50"
HF_FILENAME = "best_model.pt"

# Below this confidence, a prediction gets flagged for human review instead
# of being logged as a confirmed sighting. This is the triage bucket
# feature — mirrors how real conservation teams would actually use a tool
# like this: as an assistant that flags uncertainty, not an oracle.
TRIAGE_CONFIDENCE_THRESHOLD = 0.5

_model = None
_idx_to_species = None


def load_model():
    """
    Loads the model and label mapping into memory once. Called at API
    startup (see main.py). Rebuilds idx_to_species the same deterministic
    way training did (sorted species names from metadata.csv), so index
    numbers line up correctly with the trained checkpoint's output layer.

    Checks for the checkpoint locally first (fast, no network needed
    during normal local development). If it's not there — e.g. on a fresh
    deploy, since models/ is gitignored and never committed — downloads it
    from Hugging Face Hub instead.
    """
    global _model, _idx_to_species

    rows = load_metadata(METADATA_PATH)
    species_to_idx, idx_to_species = build_label_mapping(rows)
    _idx_to_species = idx_to_species

    if CHECKPOINT_PATH.exists():
        checkpoint_path = CHECKPOINT_PATH
        print(f"Loading model from local file: {checkpoint_path}")
    else:
        print(f"Local checkpoint not found. Downloading {HF_FILENAME} "
              f"from Hugging Face Hub ({HF_REPO_ID})...")
        checkpoint_path = hf_hub_download(repo_id=HF_REPO_ID, filename=HF_FILENAME)
        print(f"Downloaded to {checkpoint_path}")

    model = build_model(num_classes=len(species_to_idx), freeze_backbone=False)
    model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    _model = model

    print(f"Model loaded. {len(idx_to_species)} species: {list(idx_to_species.values())}")


def predict_species(image: Image.Image):
    """
    Runs the model on a single PIL image and returns a dict with the
    predicted species, confidence score, and whether it needs human review.
    """
    if _model is None:
        raise RuntimeError("Model not loaded. Call load_model() at startup.")

    image_tensor = val_transform(image.convert("RGB")).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = _model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)[0]
        confidence, predicted_idx = torch.max(probabilities, dim=0)

    species = _idx_to_species[predicted_idx.item()]
    confidence = float(confidence.item())
    needs_review = confidence < TRIAGE_CONFIDENCE_THRESHOLD

    return {
        "species": species,
        "confidence": round(confidence, 4),
        "needs_review": needs_review,
    }