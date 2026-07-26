"""
export_to_onnx.py

Converts the trained PyTorch checkpoint to ONNX format. ONNX is a
portable model format that can run with the much lighter `onnxruntime`
library instead of full PyTorch — meaningfully reduces memory and
dependency footprint at deployment time, which is what fixed our
out-of-memory crash on Render's free tier.

Run once, locally, after training:
    python src/training/export_to_onnx.py
"""

import sys
from pathlib import Path

import torch

sys.path.append(str(Path(__file__).parent))
from model import build_model
from dataset import load_metadata, build_label_mapping
from config import METADATA_PATH, CHECKPOINT_DIR, DEVICE

CHECKPOINT_PATH = CHECKPOINT_DIR / "best_model.pt"
ONNX_OUTPUT_PATH = CHECKPOINT_DIR / "best_model.onnx"


def main():
    rows = load_metadata(METADATA_PATH)
    species_to_idx, _ = build_label_mapping(rows)

    model = build_model(num_classes=len(species_to_idx), freeze_backbone=False, pretrained=False)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
    model.eval()

    dummy_input = torch.randn(1, 3, 224, 224)

    torch.onnx.export(
        model,
        dummy_input,
        str(ONNX_OUTPUT_PATH),
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
        opset_version=18,
        external_data=False,  # embed weights directly in the .onnx file
                                # instead of splitting them into a separate
                                # .onnx.data file (which is the default,
                                # and was making the main file look tiny).
    )

    print(f"Exported ONNX model to {ONNX_OUTPUT_PATH}")
    print(f"Original .pt size vs new .onnx size:")
    print(f"  {CHECKPOINT_PATH}: {CHECKPOINT_PATH.stat().st_size / 1e6:.1f} MB")
    print(f"  {ONNX_OUTPUT_PATH}: {ONNX_OUTPUT_PATH.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()