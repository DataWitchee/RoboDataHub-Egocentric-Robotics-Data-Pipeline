"""
=============================================================
SCRIPT 1: Master Dataset Builder
=============================================================
Usage:
    python build_dataset.py

What it does:
    • Loads segments.json, bbox_annotations.json, nl_descriptions.json
    • Merges all records by segment_id into a single unified record
    • Assigns task_category (kitchen / cleaning) from action label
    • Applies 80/10/10 train/val/test split deterministically
    • Saves master_dataset.json and master_dataset.csv

Dependencies:
    pip install pandas
=============================================================
"""

import json
import sys
import random
from pathlib import Path
from collections import defaultdict

import pandas as pd

# ─── CONFIG ────────────────────────────────────────────────
SEGMENTS_JSON     = Path("/Users/mannatsaini/Desktop/my_robotics_data/segments/segments.json")
BBOX_JSON         = Path("/Users/mannatsaini/Desktop/my_robotics_data/annotations/bbox_annotations.json")
NL_JSON           = Path("/Users/mannatsaini/Desktop/my_robotics_data/descriptions/nl_descriptions.json")
FINAL_DATASET_DIR = Path("/Users/mannatsaini/Desktop/my_robotics_data/final_dataset")
SEGMENTS_BASE_DIR = Path("/Users/mannatsaini/Desktop/my_robotics_data/segments")   # root used in relative video_path

SPLIT_RATIOS  = {"train": 0.80, "val": 0.10, "test": 0.10}
RANDOM_SEED   = 42

# Map action labels → task category
KITCHEN_ACTIONS  = {"chopping", "stirring", "pouring", "boiling", "cooking",
                    "cutting", "mixing", "peeling", "frying", "grilling"}
CLEANING_ACTIONS = {"wiping", "washing", "scrubbing", "picking_up",
                    "putting_away", "sweeping", "mopping", "dusting", "cleaning"}
# ───────────────────────────────────────────────────────────

CYAN  = "\033[96m"
GREEN = "\033[92m"
YELLOW= "\033[93m"
RED   = "\033[91m"
BOLD  = "\033[1m"
DIM   = "\033[2m"
RESET = "\033[0m"


def load_json(path: Path) -> list | dict:
    """Load a JSON file, exiting if not found."""
    if not path.exists():
        print(f"{YELLOW}⚠  Missing: {path}  (continuing with empty data){RESET}")
        return []
    with open(path) as f:
        return json.load(f)


# ─── Loaders & Index Builders ──────────────────────────────

def build_segment_index(segments_raw: list) -> dict[str, dict]:
    """
    Flatten segments.json (nested by video) into a flat dict
    keyed on segment_id.
    Returns: {segment_id: {action_label, start_time, end_time, source_video}}
    """
    index = {}
    for video_entry in segments_raw:
        vid_id = video_entry.get("video_id", "unknown")
        source = video_entry.get("source", "")
        for seg in video_entry.get("segments", []):
            sid = seg["segment_id"]
            index[sid] = {
                "video_id":    vid_id,
                "source_file": source,
                "action_label":seg.get("action_label", "unknown"),
                "start_time":  seg.get("start_time", 0.0),
                "end_time":    seg.get("end_time",   0.0),
                "start_frame": seg.get("start_frame", 0),
                "end_frame":   seg.get("end_frame",   0),
            }
    return index


def build_bbox_index(bbox_raw: list) -> dict[str, list]:
    """
    Index bbox_annotations.json: segment_id → list of frame annotation dicts.
    """
    return {entry["segment_id"]: entry.get("frames", [])
            for entry in bbox_raw}


def build_objects_index(bbox_raw: list) -> dict[str, list]:
    """
    Derive the unique set of detected object classes per segment.
    """
    idx = {}
    for entry in bbox_raw:
        sid = entry["segment_id"]
        unique = set()
        for frame in entry.get("frames", []):
            for obj in frame.get("objects", []):
                unique.add(obj["class"])
        idx[sid] = sorted(unique)
    return idx


def build_nl_index(nl_raw: list) -> dict[str, dict]:
    """
    Index nl_descriptions.json: segment_id → {nl_description, description_version}
    """
    return {
        entry["segment_id"]: {
            "nl_description":      entry.get("nl_description", ""),
            "description_version": entry.get("description_version", "unknown"),
        }
        for entry in nl_raw
    }


def infer_task_category(action_label: str) -> str:
    """Map an action label to 'kitchen', 'cleaning', or 'other'."""
    label = action_label.lower()
    if label in KITCHEN_ACTIONS:
        return "kitchen"
    if label in CLEANING_ACTIONS:
        return "cleaning"
    return "other"


def find_video_file(segment_id: str) -> str | None:
    """
    Search for the segment's video file under SEGMENTS_BASE_DIR.
    Returns a relative path string or None.
    """
    # Check both flat and nested layouts
    for candidate in [
        SEGMENTS_BASE_DIR / f"{segment_id}.mp4",
        SEGMENTS_BASE_DIR / segment_id.split("_")[0] / f"{segment_id}.mp4",
    ]:
        if candidate.exists():
            return str(candidate.relative_to(SEGMENTS_BASE_DIR.parent))
    return None


def assign_splits(segment_ids: list[str], seed: int = RANDOM_SEED) -> dict[str, str]:
    """
    Deterministically assign each segment_id to train / val / test.
    Uses a seeded shuffle for reproducibility.
    """
    ids = list(segment_ids)
    rng = random.Random(seed)
    rng.shuffle(ids)

    n = len(ids)
    n_train = int(n * SPLIT_RATIOS["train"])
    n_val   = int(n * SPLIT_RATIOS["val"])

    splits = {}
    for i, sid in enumerate(ids):
        if i < n_train:
            splits[sid] = "train"
        elif i < n_train + n_val:
            splits[sid] = "val"
        else:
            splits[sid] = "test"
    return splits


# ─── Main ──────────────────────────────────────────────────

def run() -> None:
    FINAL_DATASET_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\n{BOLD}Loading source JSON files…{RESET}")
    segments_raw = load_json(SEGMENTS_JSON)
    bbox_raw     = load_json(BBOX_JSON)
    nl_raw       = load_json(NL_JSON)

    seg_index  = build_segment_index(segments_raw)
    bbox_index = build_bbox_index(bbox_raw)
    obj_index  = build_objects_index(bbox_raw)
    nl_index   = build_nl_index(nl_raw)

    # Collect all known segment IDs across all data sources
    all_segment_ids = sorted(
        seg_index.keys()
        | bbox_index.keys()
        | nl_index.keys()
    )
    splits = assign_splits(all_segment_ids)

    print(f"{CYAN}Building master records for {len(all_segment_ids)} segment(s)…{RESET}")
    records = []

    for sid in all_segment_ids:
        seg        = seg_index.get(sid, {})
        nl         = nl_index.get(sid, {})
        bbox_frames= bbox_index.get(sid, [])
        objects    = obj_index.get(sid, [])
        action     = seg.get("action_label", "unknown")
        start      = seg.get("start_time", 0.0)
        end        = seg.get("end_time",   0.0)
        duration   = round(end - start, 3)
        video_file = find_video_file(sid)

        record = {
            "segment_id":       sid,
            "video_id":         seg.get("video_id", "unknown"),
            "video_path":       video_file or f"segments/{sid}.mp4",
            "video_exists":     video_file is not None,
            "action_label":     action,
            "start_time":       start,
            "end_time":         end,
            "duration":         duration,
            "start_frame":      seg.get("start_frame", 0),
            "end_frame":        seg.get("end_frame",   0),
            "objects_present":  objects,
            "bbox_annotations": bbox_frames,
            "nl_description":   nl.get("nl_description", ""),
            "description_version": nl.get("description_version", "none"),
            "task_category":    infer_task_category(action),
            "source":           "custom_recorded",
            "split":            splits.get(sid, "train"),
        }
        records.append(record)

    # ── Save full JSON ────────────────────────────────────────
    json_out = FINAL_DATASET_DIR / "master_dataset.json"
    with open(json_out, "w") as f:
        json.dump(records, f, indent=2)
    print(f"{GREEN}✔ master_dataset.json → {json_out}{RESET}")

    # ── Save flat CSV (bbox_annotations column serialized as str) ─
    flat_records = []
    for r in records:
        flat = {k: v for k, v in r.items() if k != "bbox_annotations"}
        flat["objects_present"]  = "|".join(r["objects_present"])
        flat["bbox_frame_count"] = len(r["bbox_annotations"])
        flat_records.append(flat)

    csv_out = FINAL_DATASET_DIR / "master_dataset.csv"
    pd.DataFrame(flat_records).to_csv(csv_out, index=False)
    print(f"{GREEN}✔ master_dataset.csv  → {csv_out}{RESET}")

    # ── Summary ───────────────────────────────────────────────
    split_counts = {"train": 0, "val": 0, "test": 0}
    for r in records:
        split_counts[r["split"]] += 1

    print(f"\n{BOLD}{CYAN}{'─'*50}{RESET}")
    print(f"  Total segments : {len(records)}")
    print(f"  Train / Val / Test : "
          f"{split_counts['train']} / {split_counts['val']} / {split_counts['test']}")
    print(f"  Task categories: "
          f"{len({r['task_category'] for r in records})}")
    print(f"  Action labels  : "
          f"{len({r['action_label'] for r in records})}")
    print(f"{BOLD}{CYAN}{'─'*50}{RESET}\n")


if __name__ == "__main__":
    run()
