"""
inference.py

Loads the trained model and runs predictions — now using ONNX Runtime
instead of full PyTorch. This is a dependency-weight optimization only:
the model's behavior is identical, since the ONNX file was exported
directly from the trained PyTorch checkpoint (see export_to_onnx.py).

Why this rewrite exists: torch + torchvision are large, general-purpose
training libraries. Running inference — just a forward pass on already-
trained weights — doesn't need any of that machinery. Deploying with
onnxruntime instead cuts memory usage enough to fit Render's free tier
(512MB), which the full torch-based version could not do even after
removing the redundant ImageNet pretrained-weight download.
"""

from pathlib import Path

import numpy as np
from PIL import Image
import onnxruntime as ort
from huggingface_hub import hf_hub_download

from .labels import load_metadata, build_label_mapping

METADATA_PATH = Path("data/processed/metadata.csv")
CHECKPOINT_DIR = Path("models")
CHECKPOINT_PATH = CHECKPOINT_DIR / "best_model.onnx"

# Hugging Face Hub fallback — used when the checkpoint isn't present locally
# (e.g. on a fresh deploy where models/ isn't in git).
HF_REPO_ID = "Soumybug/wildsense-resnet50"
HF_FILENAME = "best_model.onnx"

IMAGE_SIZE = 224
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# Below this confidence, a prediction gets flagged for human review instead
# of being logged as a confirmed sighting — the triage bucket feature.
TRIAGE_CONFIDENCE_THRESHOLD = 0.5

_session = None
_input_name = None
_idx_to_species = None


def load_model():
    """
    Loads the ONNX model into an onnxruntime session and rebuilds the
    species label mapping. Checks for the checkpoint locally first; if
    it's not there, downloads it from Hugging Face Hub.
    """
    global _session, _input_name, _idx_to_species

    rows = load_metadata(METADATA_PATH)
    _, idx_to_species = build_label_mapping(rows)
    _idx_to_species = idx_to_species

    if CHECKPOINT_PATH.exists():
        checkpoint_path = str(CHECKPOINT_PATH)
        print(f"Loading model from local file: {checkpoint_path}")
    else:
        print(f"Local checkpoint not found. Downloading {HF_FILENAME} "
              f"from Hugging Face Hub ({HF_REPO_ID})...")
        checkpoint_path = hf_hub_download(repo_id=HF_REPO_ID, filename=HF_FILENAME)
        print(f"Downloaded to {checkpoint_path}")

    _session = ort.InferenceSession(checkpoint_path, providers=["CPUExecutionProvider"])
    _input_name = _session.get_inputs()[0].name

    print(f"Model loaded. {len(idx_to_species)} species: {list(idx_to_species.values())}")


def _preprocess(image: Image.Image) -> np.ndarray:
    """
    Replicates the exact preprocessing used during training (resize to
    224x224, scale to 0-1, normalize with ImageNet mean/std) but using
    PIL + numpy instead of torchvision transforms, since we no longer
    depend on torchvision at inference time.
    """
    image = image.convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR)
    array = np.asarray(image).astype(np.float32) / 255.0  # HWC, [0,1]
    array = (array - IMAGENET_MEAN) / IMAGENET_STD          # normalize
    array = array.transpose(2, 0, 1)                        # HWC -> CHW
    array = np.expand_dims(array, axis=0).astype(np.float32)  # add batch dim
    return array


def _softmax(logits: np.ndarray) -> np.ndarray:
    """Numerically stable softmax, since we no longer have torch.softmax."""
    exp = np.exp(logits - np.max(logits))
    return exp / exp.sum()


def predict_species(image: Image.Image):
    """
    Runs the model on a single PIL image and returns a dict with the
    predicted species, confidence score, and whether it needs human review.
    """
    if _session is None:
        raise RuntimeError("Model not loaded. Call load_model() at startup.")

    input_array = _preprocess(image)
    outputs = _session.run(None, {_input_name: input_array})
    logits = outputs[0][0]  # first output, first (only) item in the batch

    probabilities = _softmax(logits)
    predicted_idx = int(np.argmax(probabilities))
    confidence = float(probabilities[predicted_idx])

    species = _idx_to_species[predicted_idx]
    needs_review = confidence < TRIAGE_CONFIDENCE_THRESHOLD

    return {
        "species": species,
        "confidence": round(confidence, 4),
        "needs_review": needs_review,
    }