"""
build_metadata.py

Phase 1, Step 5: builds the final clean metadata.csv that Phase 2's training
script will load. Joins together:
  - megadetector_results.csv (from run_megadetector.py) — which images passed
    the "is there an animal here?" filter
  - train_annotations.json — to recover each image's camera location ID
  - simulated GPS coordinates — one fixed (fake) lat/long per camera location

Output: data/processed/metadata.csv
    Columns: image_path, species_label, timestamp, latitude, longitude,
             detection_confidence, bbox

Run this from the project root (wildsense/) with your venv activated:
    python src/data/build_metadata.py
"""

import json
import csv
import hashlib
from pathlib import Path


# -----------------------------------------------------------------------
# CONFIG
# -----------------------------------------------------------------------

ANNOTATIONS_PATH = Path("data/raw/eccv_18_annotation_files/train_annotations.json")
DETECTION_RESULTS_PATH = Path("data/processed/megadetector_results.csv")
OUTPUT_CSV = Path("data/processed/metadata.csv")

# Only keep rows that MegaDetector confirmed had an animal (detected == True).
# These are the images Phase 2 will actually train on.
KEEP_ONLY_DETECTED = True

# Base coordinates to scatter simulated camera locations around. CCT-20 cameras
# are in the southwestern US, so we center roughly on southern Arizona.
# These are NOT real camera coordinates — purely for demo/dashboard purposes.
BASE_LATITUDE = 31.9
BASE_LONGITUDE = -110.0

# How far (in degrees) simulated locations can scatter from the base point.
# ~0.5 degrees is roughly 50km, giving a reasonably spread-out looking map
# without coordinates ending up unrealistically far apart.
SCATTER_RANGE = 0.5


def load_location_lookup(annotations_path):
    """
    Builds a dict mapping image file_name -> location ID, by reading the
    original annotation file. We need this because megadetector_results.csv
    only has image_path, not the camera location — but we need location to
    assign consistent simulated GPS per camera.
    """
    print(f"Loading location info from {annotations_path} ...")
    with open(annotations_path, "r") as f:
        data = json.load(f)

    filename_to_location = {}
    for img in data["images"]:
        filename_to_location[img["file_name"]] = img.get("location", "unknown")

    print(f"Loaded location info for {len(filename_to_location)} images.")
    return filename_to_location


def simulate_coordinates(location_id):
    """
    Deterministically generates a fake (but consistent) lat/long for a given
    location ID. Same location_id always produces the same coordinates, so
    all images from one camera cluster together on the map — which is what
    you'd expect from a real camera trap deployment.

    Uses a hash of the location_id to generate a repeatable offset, so we
    don't need to store a separate lookup table anywhere.
    """
    # Hash the location_id into two numbers we can use as pseudo-random
    # but deterministic offsets.
    hash_bytes = hashlib.md5(str(location_id).encode()).digest()
    lat_offset = (hash_bytes[0] / 255.0 - 0.5) * 2 * SCATTER_RANGE
    lon_offset = (hash_bytes[1] / 255.0 - 0.5) * 2 * SCATTER_RANGE

    latitude = round(BASE_LATITUDE + lat_offset, 6)
    longitude = round(BASE_LONGITUDE + lon_offset, 6)
    return latitude, longitude


def load_detection_results(detection_results_path):
    """Reads megadetector_results.csv into a list of dicts."""
    print(f"Loading detection results from {detection_results_path} ...")
    with open(detection_results_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    print(f"Loaded {len(rows)} rows.")
    return rows


def build_final_metadata(detection_rows, filename_to_location):
    """
    Joins detection results with location info and simulated GPS, filters
    to only detected==True rows if configured, and returns the final list
    of row dicts ready to write out.
    """
    # Cache so we don't recompute simulated coordinates for the same
    # location more than once (keeps output perfectly consistent).
    location_coords_cache = {}

    final_rows = []
    skipped_not_detected = 0
    skipped_no_location = 0

    for row in detection_rows:
        detected = row["detected"] == "True"

        if KEEP_ONLY_DETECTED and not detected:
            skipped_not_detected += 1
            continue

        image_filename = Path(row["image_path"]).name
        location_id = filename_to_location.get(image_filename)

        if location_id is None:
            skipped_no_location += 1
            continue

        if location_id not in location_coords_cache:
            location_coords_cache[location_id] = simulate_coordinates(location_id)
        latitude, longitude = location_coords_cache[location_id]

        final_rows.append({
            "image_path": row["image_path"],
            "species_label": row["species_label"],
            "timestamp": row["timestamp"],
            "latitude": latitude,
            "longitude": longitude,
            "detection_confidence": row["detection_confidence"],
            "bbox": row["bbox"],
        })

    print(f"Built {len(final_rows)} final metadata rows.")
    print(f"  Skipped {skipped_not_detected} rows (not detected as containing an animal)")
    print(f"  Skipped {skipped_no_location} rows (no matching location found)")
    print(f"  Simulated GPS assigned to {len(location_coords_cache)} unique camera locations")

    return final_rows


def write_metadata_csv(rows, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = ["image_path", "species_label", "timestamp",
                  "latitude", "longitude", "detection_confidence", "bbox"]

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nWrote {len(rows)} rows to {output_path}")


def main():
    filename_to_location = load_location_lookup(ANNOTATIONS_PATH)
    detection_rows = load_detection_results(DETECTION_RESULTS_PATH)
    final_rows = build_final_metadata(detection_rows, filename_to_location)
    write_metadata_csv(final_rows, OUTPUT_CSV)


if __name__ == "__main__":
    main()