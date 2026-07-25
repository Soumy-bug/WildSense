"""
evaluate.py

Loads the best trained model checkpoint, re-runs it on the validation set,
and saves two artifacts for your README/documentation:
  - results/confusion_matrix.png  (heatmap image)
  - results/classification_report.txt (per-species precision/recall/F1)

Uses the exact same train/val split as train.py (same seed), so these
results match what training actually evaluated on.

Run from project root:
    python src/training/evaluate.py
"""

import sys
from pathlib import Path

import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix

sys.path.append(str(Path(__file__).parent))
from dataset import load_metadata, build_label_mapping, stratified_split, WildSenseDataset
from model import build_model
from train import (
    val_transform, IMAGES_ROOT, BATCH_SIZE, VAL_FRACTION,
    METADATA_PATH, CHECKPOINT_DIR, DEVICE,
)

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

CHECKPOINT_PATH = CHECKPOINT_DIR / "best_model.pt"


def plot_confusion_matrix(cm, species_names, output_path):
    """
    Saves a heatmap image of the confusion matrix using matplotlib.
    Darker cells = higher counts. Diagonal should be dark if the model
    is doing well; off-diagonal darkness reveals specific confusions.
    """
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, cmap="Blues")

    ax.set_xticks(range(len(species_names)))
    ax.set_yticks(range(len(species_names)))
    ax.set_xticklabels(species_names, rotation=45, ha="right")
    ax.set_yticklabels(species_names)
    ax.set_xlabel("Predicted species")
    ax.set_ylabel("True species")
    ax.set_title("WildSense — Validation Confusion Matrix")

    # Write the actual count in each cell, with contrasting text color
    # so numbers stay readable on both light and dark cells.
    threshold = cm.max() / 2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            color = "white" if cm[i, j] > threshold else "black"
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color=color)

    fig.colorbar(im, ax=ax, label="Number of images")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    print(f"Saved confusion matrix image to {output_path}")


def main():
    print("Rebuilding the same train/val split used during training...")
    rows = load_metadata(METADATA_PATH)
    species_to_idx, idx_to_species = build_label_mapping(rows)
    _, val_rows = stratified_split(rows, VAL_FRACTION)  # same seed=42 default as training

    val_dataset = WildSenseDataset(val_rows, species_to_idx, val_transform, images_root=IMAGES_ROOT)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    print(f"Loading model checkpoint from {CHECKPOINT_PATH} ...")
    model = build_model(num_classes=len(species_to_idx), freeze_backbone=False, pretrained=False).to(DEVICE)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
    model.eval()

    print("Running inference on validation set...")
    all_labels, all_preds = [], []
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(DEVICE)
            outputs = model(images)
            preds = outputs.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    species_names = [idx_to_species[i] for i in range(len(idx_to_species))]

    # ---- Classification report (text file) ----
    report_text = classification_report(
        all_labels, all_preds, target_names=species_names, zero_division=0
    )
    report_path = RESULTS_DIR / "classification_report.txt"
    with open(report_path, "w") as f:
        f.write("WildSense — Validation Classification Report\n")
        f.write("=" * 50 + "\n\n")
        f.write(report_text)
    print(f"Saved classification report to {report_path}")
    print("\n" + report_text)

    # ---- Confusion matrix (image) ----
    cm = confusion_matrix(all_labels, all_preds)
    plot_confusion_matrix(cm, species_names, RESULTS_DIR / "confusion_matrix.png")


if __name__ == "__main__":
    main()