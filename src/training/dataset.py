"""
dataset.py

Phase 2 data loading. Handles:
  - reading metadata.csv (built in Phase 1)
  - building a species-name <-> integer-index mapping (models need integer labels)
  - splitting into train/val sets (stratified, so each species is represented
    proportionally in both sets)
  - a PyTorch Dataset class that loads and transforms images on demand
"""

import csv
from pathlib import Path

from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset


def load_metadata(metadata_path):
    """Reads metadata.csv into a list of dicts."""
    with open(metadata_path, "r", newline="") as f:
        rows = list(csv.DictReader(f))
    return rows


def build_label_mapping(rows):
    """
    Builds species_name -> integer index and the reverse mapping.
    Models output numbers, not strings, so we need this conversion in
    both directions (encoding labels for training, decoding predictions
    back to species names for evaluation/reporting).
    """
    species_names = sorted(set(row["species_label"] for row in rows))
    species_to_idx = {name: i for i, name in enumerate(species_names)}
    idx_to_species = {i: name for name, i in species_to_idx.items()}
    return species_to_idx, idx_to_species


def stratified_split(rows, val_fraction=0.2, seed=42):
    """
    Splits rows into train/val sets, preserving each species' proportion
    in both sets (e.g. if skunk is 3% of the data, it stays ~3% in both
    train and val, rather than accidentally ending up almost entirely in
    one set). This matters a lot with imbalanced classes like ours.
    """
    labels = [row["species_label"] for row in rows]
    train_rows, val_rows = train_test_split(
        rows,
        test_size=val_fraction,
        stratify=labels,
        random_state=seed,
    )
    return train_rows, val_rows


class WildSenseDataset(Dataset):
    """
    PyTorch Dataset wrapping our metadata rows. For each item, loads the
    image from disk, applies the given transform (resize/augment/normalize),
    and returns (image_tensor, label_index).

    images_root: optional override for where images actually live. If set,
    only the filename from metadata.csv is used, joined onto this root.
    This makes the dataset portable — metadata.csv can be built on one
    machine (e.g. Windows, storing "data\\raw\\images\\...\\xyz.jpg") and
    used on another (e.g. Colab's Linux filesystem) without path breakage,
    as long as the same images are available somewhere under images_root.
    """

    def __init__(self, rows, species_to_idx, transform, images_root=None):
        self.rows = rows
        self.species_to_idx = species_to_idx
        self.transform = transform
        self.images_root = Path(images_root) if images_root else None

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]

        if self.images_root:
            image_path = self.images_root / Path(row["image_path"]).name
        else:
            image_path = row["image_path"]

        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)
        label = self.species_to_idx[row["species_label"]]
        return image, label