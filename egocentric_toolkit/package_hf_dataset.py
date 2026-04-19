"""
=============================================================
SCRIPT 2: HuggingFace Dataset Packager
=============================================================
Usage:
    python package_hf_dataset.py

What it does:
    • Loads master_dataset.json from /dataset/final_dataset/
    • Converts it to a HuggingFace datasets.Dataset with a typed schema
    • Splits into train/val/test DatasetDicts
    • Saves to disk at /dataset/final_dataset/hf_dataset/
    • Writes a standalone load_dataset.py convenience loader

Dependencies:
    pip install datasets pandas
=============================================================
"""

import json
import sys
from pathlib import Path

import pandas as pd

# ─── CONFIG ────────────────────────────────────────────────
FINAL_DATASET_DIR = Path("/Users/mannatsaini/Desktop/my_robotics_data/final_dataset")
MASTER_JSON       = FINAL_DATASET_DIR / "master_dataset.json"
HF_DATASET_DIR    = FINAL_DATASET_DIR / "hf_dataset"
# ───────────────────────────────────────────────────────────

CYAN  = "\033[96m"
GREEN = "\033[92m"
YELLOW= "\033[93m"
RED   = "\033[91m"
BOLD  = "\033[1m"
RESET = "\033[0m"


def build_hf_features():
    """
    Define the HuggingFace Features schema for the dataset.

    bbox_annotations is stored as a JSON string to stay type-safe
    while preserving the full nested structure.  Downstream users
    can json.loads() it as needed.
    """
    try:
        from datasets import (
            Features, Value, ClassLabel, Sequence
        )
    except ImportError:
        print(f"{RED}ERROR: datasets library not installed.\n"
              f"  Run: pip install datasets{RESET}")
        sys.exit(1)

    return Features({
        "segment_id":          Value("string"),
        "video_id":            Value("string"),
        "video_path":          Value("string"),
        "video_exists":        Value("bool"),
        "action_label":        Value("string"),
        "start_time":          Value("float32"),
        "end_time":            Value("float32"),
        "duration":            Value("float32"),
        "start_frame":         Value("int32"),
        "end_frame":           Value("int32"),
        "objects_present":     Sequence(Value("string")),
        # Full bbox JSON stored as string to handle nested variability
        "bbox_annotations_json": Value("string"),
        "nl_description":      Value("string"),
        "description_version": Value("string"),
        "task_category":       ClassLabel(names=["kitchen", "cleaning", "other"]),
        "source":              Value("string"),
        "split":               Value("string"),
    })


def load_master(path: Path) -> list[dict]:
    if not path.exists():
        print(f"{RED}ERROR: {path} not found. Run build_dataset.py first.{RESET}")
        sys.exit(1)
    with open(path) as f:
        return json.load(f)


def normalise_record(record: dict) -> dict:
    """
    Coerce record fields to match the HF schema exactly.

    • bbox_annotations list → JSON string
    • task_category string  → valid ClassLabel name (fallback to 'other')
    • int fields safely cast
    """
    valid_task_cats = {"kitchen", "cleaning", "other"}
    task_cat = record.get("task_category", "other")
    if task_cat not in valid_task_cats:
        task_cat = "other"

    return {
        "segment_id":            record.get("segment_id", ""),
        "video_id":              record.get("video_id", ""),
        "video_path":            record.get("video_path", ""),
        "video_exists":          bool(record.get("video_exists", False)),
        "action_label":          record.get("action_label", "unknown"),
        "start_time":            float(record.get("start_time", 0.0)),
        "end_time":              float(record.get("end_time",   0.0)),
        "duration":              float(record.get("duration",   0.0)),
        "start_frame":           int(record.get("start_frame", 0)),
        "end_frame":             int(record.get("end_frame",   0)),
        "objects_present":       list(record.get("objects_present", [])),
        # Serialise nested bbox list to JSON string for schema safety
        "bbox_annotations_json": json.dumps(record.get("bbox_annotations", [])),
        "nl_description":        record.get("nl_description", ""),
        "description_version":   record.get("description_version", "none"),
        "task_category":         task_cat,
        "source":                record.get("source", "custom_recorded"),
        "split":                 record.get("split", "train"),
    }


def run() -> None:
    try:
        from datasets import Dataset, DatasetDict
    except ImportError:
        print(f"{RED}ERROR: datasets library not found.\n"
              f"  Run: pip install datasets{RESET}")
        sys.exit(1)

    HF_DATASET_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\n{BOLD}Loading master_dataset.json…{RESET}")
    records = load_master(MASTER_JSON)
    print(f"  {len(records)} records loaded")

    # ── Normalise ─────────────────────────────────────────────
    norm_records = [normalise_record(r) for r in records]

    features = build_hf_features()

    # ── Split into DatasetDict ────────────────────────────────
    train_rows = [r for r in norm_records if r["split"] == "train"]
    val_rows   = [r for r in norm_records if r["split"] == "val"]
    test_rows  = [r for r in norm_records if r["split"] == "test"]

    print(f"\n{CYAN}Split sizes:  "
          f"train={len(train_rows)}  val={len(val_rows)}  test={len(test_rows)}{RESET}")

    def rows_to_dataset(rows: list[dict]) -> Dataset:
        """Convert a list of flat dicts to a HF Dataset with the typed schema."""
        if not rows:
            # Return an empty dataset with the correct schema
            return Dataset.from_dict(
                {k: [] for k in features.keys()},
                features=features,
            )
        # Pivot list-of-dicts → dict-of-lists  (HF Dataset.from_dict format)
        col_dict = {k: [r[k] for r in rows] for k in rows[0].keys()}
        return Dataset.from_dict(col_dict, features=features)

    dataset_dict = DatasetDict({
        "train": rows_to_dataset(train_rows),
        "val":   rows_to_dataset(val_rows),
        "test":  rows_to_dataset(test_rows),
    })

    # ── Save ──────────────────────────────────────────────────
    print(f"\n{BOLD}Saving HuggingFace DatasetDict to {HF_DATASET_DIR}…{RESET}")
    dataset_dict.save_to_disk(str(HF_DATASET_DIR))
    print(f"{GREEN}✔ Dataset saved.{RESET}")

    # ── Write the convenience loader script ───────────────────
    loader_path = FINAL_DATASET_DIR / "load_dataset.py"
    loader_content = '''\
"""
Convenience loader for the egocentric robotics HuggingFace dataset.

Usage:
    python load_dataset.py

Or in your own code:
    from datasets import load_from_disk
    import json

    dataset = load_from_disk("./hf_dataset")

    # Access splits
    train_ds = dataset["train"]
    val_ds   = dataset["val"]
    test_ds  = dataset["test"]

    # Decode nested bbox annotations
    for example in train_ds:
        bboxes = json.loads(example["bbox_annotations_json"])
        print(example["segment_id"], example["action_label"])
        print("Objects:", example["objects_present"])
        print("Description:", example["nl_description"])
        print("BBox frames:", len(bboxes))
"""

from datasets import load_from_disk
import json
from pathlib import Path

DATASET_PATH = Path(__file__).parent / "hf_dataset"

dataset = load_from_disk(str(DATASET_PATH))

print(f"Loaded dataset splits: {list(dataset.keys())}")
for split, ds in dataset.items():
    print(f"  {split}: {len(ds)} examples")

# Preview first training example
if len(dataset["train"]) > 0:
    example = dataset["train"][0]
    print("\\n─── First training example ───")
    print(f"  segment_id   : {example['segment_id']}")
    print(f"  action_label : {example['action_label']}")
    print(f"  task_category: {example['task_category']}")
    print(f"  duration     : {example['duration']:.2f}s")
    print(f"  objects      : {example['objects_present']}")
    print(f"  description  : {example['nl_description'][:80]}...")
    bboxes = json.loads(example["bbox_annotations_json"])
    print(f"  bbox frames  : {len(bboxes)}")
'''
    loader_path.write_text(loader_content)
    print(f"{GREEN}✔ Loader script written → {loader_path}{RESET}")

    print(f"\n{BOLD}To load the dataset anywhere:{RESET}")
    print(f"  {CYAN}from datasets import load_from_disk")
    print(f"  dataset = load_from_disk('{HF_DATASET_DIR}'){RESET}\n")


if __name__ == "__main__":
    run()
