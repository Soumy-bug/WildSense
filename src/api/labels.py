"""
labels.py

Pure-stdlib functions for loading metadata.csv and building the species
name <-> index mapping. Deliberately has zero dependencies beyond Python's
standard library (csv, pathlib) — split out from dataset.py specifically
so that inference.py can use these without indirectly requiring torch or
scikit-learn, which dataset.py needs for its Dataset class and stratified
split but which the deployed API has no actual use for.
"""

import csv


def load_metadata(metadata_path):
    """Reads metadata.csv into a list of dicts."""
    with open(metadata_path, "r", newline="") as f:
        rows = list(csv.DictReader(f))
    return rows


def build_label_mapping(rows):
    """
    Builds species_name -> integer index and the reverse mapping.
    Sorted alphabetically for a deterministic, reproducible ordering that
    matches whatever order was used when the model was trained.
    """
    species_names = sorted(set(row["species_label"] for row in rows))
    species_to_idx = {name: i for i, name in enumerate(species_names)}
    idx_to_species = {i: name for name, i in species_to_idx.items()}
    return species_to_idx, idx_to_species