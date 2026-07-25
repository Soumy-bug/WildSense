"""
train.py

Phase 2 main training script. Fine-tunes ResNet50 on our filtered camera
trap species dataset, handling class imbalance with weighted loss, and
evaluating with per-species precision/recall/F1 (not just overall accuracy).

Training happens in two stages:
  Stage 1 (EPOCHS_FROZEN epochs): only the new classifier head trains,
      backbone frozen. Fast, gets the head to a reasonable starting point.
  Stage 2 (EPOCHS_FINETUNE epochs): whole network unfrozen, trains at a
      much lower learning rate to gently adapt pretrained features.

Run from project root (wildsense/) with venv activated:
    python src/training/train.py
"""

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.metrics import classification_report, confusion_matrix
import wandb

# Make sibling modules importable when running this file directly
sys.path.append(str(Path(__file__).parent))
from dataset import load_metadata, build_label_mapping, stratified_split, WildSenseDataset
from model import build_model, unfreeze_backbone


# -----------------------------------------------------------------------
# CONFIG
# -----------------------------------------------------------------------

METADATA_PATH = Path("data/processed/metadata.csv")
CHECKPOINT_DIR = Path("models")
CHECKPOINT_DIR.mkdir(exist_ok=True)

# Override where images actually live, since metadata.csv's stored paths
# were written on Windows and won't resolve directly on Colab's Linux
# filesystem. Set to None to use the paths in metadata.csv as-is (e.g. when
# running locally on the same machine that built metadata.csv).
# On Colab, set this to wherever you unzipped your images, e.g.:
#   IMAGES_ROOT = Path("/content/images/eccv_18_all_images_sm")
IMAGES_ROOT = None

IMAGE_SIZE = 224          # standard input size for ResNet50
BATCH_SIZE = 32
VAL_FRACTION = 0.2

EPOCHS_FROZEN = 5         # stage 1: train only the classifier head
EPOCHS_FINETUNE = 5       # stage 2: fine-tune the whole network
LR_FROZEN = 1e-3          # higher LR is fine, only a small new layer trains
LR_FINETUNE = 1e-5        # much lower, to avoid wrecking pretrained features

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

WANDB_PROJECT = "wildsense"


# -----------------------------------------------------------------------
# TRANSFORMS
# -----------------------------------------------------------------------
# ImageNet normalization stats — required because ResNet50 was pretrained
# expecting inputs normalized this way. Using different stats would confuse
# the pretrained layers.
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(),      # cheap, effective augmentation:
                                             # animals can face either direction
    transforms.ColorJitter(brightness=0.2, contrast=0.2),  # camera traps see
                                             # widely varying lighting conditions
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

val_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    # No augmentation for validation — we want to evaluate on realistic,
    # unmodified images to get a true read on performance.
])


def compute_class_weights(train_rows, species_to_idx):
    """
    Computes inverse-frequency weights per class, so rare species (like
    skunk) contribute more to the loss than common ones (like opossum).
    This directly addresses the 12.1x imbalance found in Phase 1.
    """
    counts = np.zeros(len(species_to_idx))
    for row in train_rows:
        counts[species_to_idx[row["species_label"]]] += 1

    # Inverse frequency, then normalize so weights average to ~1
    weights = 1.0 / counts
    weights = weights / weights.sum() * len(counts)
    return torch.tensor(weights, dtype=torch.float32)


def run_epoch(model, loader, criterion, optimizer, training):
    """
    Runs one full pass over the given loader. If training=True, updates
    model weights; if False, just evaluates (used for validation).
    Returns average loss and lists of true/predicted labels for metrics.
    """
    model.train() if training else model.eval()

    total_loss = 0.0
    all_labels = []
    all_preds = []

    torch.set_grad_enabled(training)
    for images, labels in loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        if training:
            optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        if training:
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())
    torch.set_grad_enabled(True)

    avg_loss = total_loss / len(loader.dataset)
    return avg_loss, all_labels, all_preds


def evaluate_and_log(labels, preds, idx_to_species, epoch, split_name):
    """
    Prints and logs per-species precision/recall/F1 plus overall macro F1.
    Macro F1 (average of each class's F1, unweighted) matters more than
    accuracy here — it doesn't let good performance on common species like
    opossum hide poor performance on rare ones like skunk.
    """
    species_names = [idx_to_species[i] for i in range(len(idx_to_species))]
    report = classification_report(
        labels, preds, target_names=species_names, output_dict=True, zero_division=0
    )
    macro_f1 = report["macro avg"]["f1-score"]

    print(f"\n[{split_name}] Epoch {epoch} — macro F1: {macro_f1:.3f}")
    for name in species_names:
        p, r, f1 = report[name]["precision"], report[name]["recall"], report[name]["f1-score"]
        print(f"    {name:<10} precision={p:.2f}  recall={r:.2f}  f1={f1:.2f}")

    wandb.log({
        f"{split_name}_macro_f1": macro_f1,
        "epoch": epoch,
    })

    return macro_f1


def main():
    print("Loading metadata and building train/val split...")
    rows = load_metadata(METADATA_PATH)
    species_to_idx, idx_to_species = build_label_mapping(rows)
    train_rows, val_rows = stratified_split(rows, VAL_FRACTION)
    print(f"Train: {len(train_rows)} images, Val: {len(val_rows)} images, "
          f"{len(species_to_idx)} species")

    train_dataset = WildSenseDataset(train_rows, species_to_idx, train_transform, images_root=IMAGES_ROOT)
    val_dataset = WildSenseDataset(val_rows, species_to_idx, val_transform, images_root=IMAGES_ROOT)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    class_weights = compute_class_weights(train_rows, species_to_idx).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    model = build_model(num_classes=len(species_to_idx), freeze_backbone=True).to(DEVICE)

    wandb.init(project=WANDB_PROJECT, config={
        "epochs_frozen": EPOCHS_FROZEN,
        "epochs_finetune": EPOCHS_FINETUNE,
        "batch_size": BATCH_SIZE,
        "lr_frozen": LR_FROZEN,
        "lr_finetune": LR_FINETUNE,
    })

    best_macro_f1 = 0.0
    epoch_counter = 0

    # ---------------- Stage 1: train classifier head only ----------------
    print("\n=== Stage 1: training classifier head (backbone frozen) ===")
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=LR_FROZEN
    )

    for _ in range(EPOCHS_FROZEN):
        epoch_counter += 1
        train_loss, _, _ = run_epoch(model, train_loader, criterion, optimizer, training=True)
        val_loss, val_labels, val_preds = run_epoch(model, val_loader, criterion, optimizer, training=False)
        print(f"\nEpoch {epoch_counter} — train_loss={train_loss:.3f}  val_loss={val_loss:.3f}")
        macro_f1 = evaluate_and_log(val_labels, val_preds, idx_to_species, epoch_counter, "val")

        if macro_f1 > best_macro_f1:
            best_macro_f1 = macro_f1
            torch.save(model.state_dict(), CHECKPOINT_DIR / "best_model.pt")
            print(f"  New best model saved (macro F1={macro_f1:.3f})")

    # ---------------- Stage 2: fine-tune whole network ----------------
    print("\n=== Stage 2: fine-tuning full network (low learning rate) ===")
    model = unfreeze_backbone(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR_FINETUNE)

    for _ in range(EPOCHS_FINETUNE):
        epoch_counter += 1
        train_loss, _, _ = run_epoch(model, train_loader, criterion, optimizer, training=True)
        val_loss, val_labels, val_preds = run_epoch(model, val_loader, criterion, optimizer, training=False)
        print(f"\nEpoch {epoch_counter} — train_loss={train_loss:.3f}  val_loss={val_loss:.3f}")
        macro_f1 = evaluate_and_log(val_labels, val_preds, idx_to_species, epoch_counter, "val")

        if macro_f1 > best_macro_f1:
            best_macro_f1 = macro_f1
            torch.save(model.state_dict(), CHECKPOINT_DIR / "best_model.pt")
            print(f"  New best model saved (macro F1={macro_f1:.3f})")

    # ---------------- Final confusion matrix ----------------
    print("\n=== Final confusion matrix (last epoch's validation predictions) ===")
    species_names = [idx_to_species[i] for i in range(len(idx_to_species))]
    cm = confusion_matrix(val_labels, val_preds)
    print(f"{'':<10}" + "".join(f"{n[:8]:>9}" for n in species_names))
    for i, row in enumerate(cm):
        print(f"{species_names[i]:<10}" + "".join(f"{v:>9}" for v in row))

    print(f"\nTraining complete. Best validation macro F1: {best_macro_f1:.3f}")
    print(f"Best model saved to {CHECKPOINT_DIR / 'best_model.pt'}")


if __name__ == "__main__":
    main()