import json
import csv
from pathlib import Path

from PytorchWildlife.models import detection as pw_detection


# -----------------------------------------------------------------------
# CONFIG — edit these to match your setup
# -----------------------------------------------------------------------

# Path to the CCT-20 annotation/split file you downloaded from LILA
ANNOTATIONS_PATH = Path("data/raw/eccv_18_annotation_files/train_annotations.json")

# Folder where you extracted the benchmark images
IMAGES_DIR = Path("data/raw/images/eccv_18_all_images_sm")

# Where the output CSV will be written
OUTPUT_CSV = Path("data/processed/megadetector_results.csv")

# Pick 5-8 species to start with. These must match the category "name"
# field exactly as it appears in your annotations.json (see Step 1 below
# if you're not sure what names are available).
CHOSEN_SPECIES = [
    "opossum",
    "raccoon",
    "coyote",
    "bobcat",
    "squirrel",
    "skunk",
]

# MegaDetector confidence threshold — image is "kept" if the top detection
# confidence is >= this value. Start at 0.3 and adjust after spot-checking
# (see Phase 1 guide, Step 4).
CONFIDENCE_THRESHOLD = 0.3


def load_annotations(annotations_path):
    """
    Loads the CCT-20 annotation file (COCO Camera Traps format) and returns
    three lookup dictionaries so we can join images -> labels -> file paths.
    """
    print(f"Loading annotations from {annotations_path} ...")
    with open(annotations_path, "r") as f:
        data = json.load(f)

    # category_id -> species name, e.g. {7: "coyote"}
    category_id_to_name = {cat["id"]: cat["name"] for cat in data["categories"]}

    # image_id -> image record (has file_name, date_captured, location, etc.)
    image_id_to_info = {img["id"]: img for img in data["images"]}

    # image_id -> category_id  (CCT-20 has one label per image in this file)
    image_id_to_category = {}
    for ann in data["annotations"]:
        image_id_to_category[ann["image_id"]] = ann["category_id"]

    print(f"Found {len(category_id_to_name)} categories, "
          f"{len(image_id_to_info)} images, "
          f"{len(image_id_to_category)} annotations.")

    return category_id_to_name, image_id_to_info, image_id_to_category


def filter_to_chosen_species(category_id_to_name, image_id_to_info,
                              image_id_to_category, chosen_species):
    """
    Returns a list of dicts, one per image that belongs to one of our
    chosen species. Each dict has: image_path, species_label, timestamp.
    """
    # Which category IDs correspond to our chosen species names?
    chosen_ids = {
        cat_id for cat_id, name in category_id_to_name.items()
        if name in chosen_species
    }

    if not chosen_ids:
        raise ValueError(
            "None of CHOSEN_SPECIES matched category names in annotations.json. "
            "Print category_id_to_name.values() to see the exact names available."
        )

    selected_images = []
    for image_id, category_id in image_id_to_category.items():
        if category_id not in chosen_ids:
            continue  # not one of our chosen species, skip it

        img_info = image_id_to_info.get(image_id)
        if img_info is None:
            continue  # annotation points to an image we don't have info for

        selected_images.append({
            "image_id": image_id,
            "image_path": str(IMAGES_DIR / img_info["file_name"]),
            "species_label": category_id_to_name[category_id],
            "timestamp": img_info.get("date_captured", ""),
        })

    print(f"Selected {len(selected_images)} images across "
          f"{len(chosen_species)} chosen species.")
    return selected_images


def run_detector_on_images(selected_images, model, threshold):
    """
    Runs MegaDetector on each selected image and returns a list of result
    dicts ready to write to CSV. Images that fail to load are skipped with
    a warning printed, rather than crashing the whole run.
    """
    results = []
    total = len(selected_images)

    for i, item in enumerate(selected_images, start=1):
        image_path = item["image_path"]

        if not Path(image_path).exists():
            print(f"  [{i}/{total}] WARNING: file not found, skipping: {image_path}")
            continue

        try:
            detection = model.single_image_detection(image_path)
        except Exception as e:
            print(f"  [{i}/{total}] WARNING: detection failed for {image_path}: {e}")
            continue

        # single_image_detection returns a dict with a "detections" key holding
        # a supervision.Detections object. Confidence scores and boxes live as
        # attributes on that object (.confidence, .xyxy), not as dict keys.
        dets = detection.get("detections")
        has_detection = dets is not None and len(dets.xyxy) > 0

        if has_detection:
            best_confidence = float(max(dets.confidence))
            best_index = int(dets.confidence.argmax())
            bbox = dets.xyxy[best_index].tolist()
        else:
            best_confidence = 0.0
            bbox = None

        detected = best_confidence >= threshold

        results.append({
            "image_path": image_path,
            "species_label": item["species_label"],
            "timestamp": item["timestamp"],
            "detected": detected,
            "detection_confidence": round(float(best_confidence), 4),
            "bbox": bbox,
        })

        if i % 50 == 0 or i == total:
            print(f"  [{i}/{total}] processed")

    return results


def write_results_csv(results, output_path):
    """Writes the final results list to a CSV file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = ["image_path", "species_label", "timestamp",
                  "detected", "detection_confidence", "bbox"]

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    kept = sum(1 for r in results if r["detected"])
    print(f"\nWrote {len(results)} rows to {output_path}")
    print(f"  {kept} images passed the detection threshold ({CONFIDENCE_THRESHOLD})")
    print(f"  {len(results) - kept} images were filtered out as likely empty/false triggers")


def main():
    category_id_to_name, image_id_to_info, image_id_to_category = load_annotations(
        ANNOTATIONS_PATH
    )

    selected_images = filter_to_chosen_species(
        category_id_to_name, image_id_to_info, image_id_to_category, CHOSEN_SPECIES
    )

    print("\nLoading MegaDetector (first run will download model weights)...")
    model = pw_detection.MegaDetectorV6(version="MDV6-yolov10-c")

    print("\nRunning detection on selected images...")
    results = run_detector_on_images(selected_images, model, CONFIDENCE_THRESHOLD)

    write_results_csv(results, OUTPUT_CSV)


if __name__ == "__main__":
    main()