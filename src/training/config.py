"""
config.py

Shared, lightweight configuration used by BOTH train.py (local training)
and inference.py (API serving). Deliberately has zero heavy or
training-only dependencies (no wandb, no matplotlib) so that importing
this file — even indirectly, through inference.py — never pulls in
packages the deployed API doesn't actually need.

This file exists because of a real deployment bug: inference.py originally
imported directly from train.py to reuse the val_transform and a few
constants, which meant loading the API also loaded wandb (used only for
training metric logging), breaking deployment on a minimal requirements
file. Splitting the shared pieces out here fixes that at the root instead
of adding every training dependency to the deployment requirements file.
"""

from pathlib import Path

import torch
from torchvision import transforms

METADATA_PATH = Path("data/processed/metadata.csv")
CHECKPOINT_DIR = Path("models")
CHECKPOINT_DIR.mkdir(exist_ok=True)

IMAGE_SIZE = 224
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

val_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])