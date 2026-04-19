"""
=============================================================
BONUS SCRIPT: LeRobot Format Converter
=============================================================
Usage:
    python convert_lerobot.py

What it does:
    • Loads master_dataset.json from /dataset/final_dataset/
    • Converts each segment into a LeRobot-compatible episode folder
    • Extracts frames at 1 FPS and writes them as JPGs
    • Writes per-frame action metadata as a JSON Lines file
    • Generates a top-level meta_data.json following LeRobot conventions
    • Outputs to /dataset/final_dataset/lerobot_dataset/

LeRobot schema reference:
    https://github.com/huggingface/lerobot

LeRobot episode structure produced:
    lerobot_dataset/
        meta_data.json
        episodes/
            video001_seg001/
                frames/
                    frame_000000.jpg
                    frame_000001.jpg
                    ...
                episode_data.jsonl    ← one JSON object per frame
        dataset.json                  ← master index

Dependencies:
    pip install opencv-python
=============================================================
"""

import json
import sys
from pathlib import Path
from datetime import date

import cv2

# ─── CONFIG ────────────────────────────────────────────────
FINAL_DATASET_DIR = Path("/Users/mannatsaini/Desktop/my_robotics_data/final_dataset")
MASTER_JSON       = FINAL_DATASET_DIR / "master_dataset.json"
LEROBOT_DIR       = FINAL_DATASET_DIR / "lerobot_dataset"
SEGMENTS_ROOT     = Path("/Users/mannatsaini/Desktop/my_robotics_data")

# Sampling rate for frame extraction (frames per second saved to disk)
OUTPUT_FPS = 1
# ───────────────────────────────────────────────────────────

CYAN  = "\033[96m"
GREEN = "\033[92m"
YELLOW= "\033[93m"
RED   = "\033[91m"
BOLD  = "\033[1m"
DIM   = "\033[2m"
RESET = "\033[0m"


def load_master(path: Path) -> list[dict]:
    if not path.exists():
        print(f"{RED}ERROR: {path} not found. Run build_dataset.py first.{RESET}")
        sys.exit(1)
    with open(path) as f:
        return json.load(f)


def extract_frames_at_fps(
    video_path: Path,
    output_dir: Path,
    target_fps: float,
) -> list[dict]:
    """
    Extract frames from a video at `target_fps` and save as JPGs.

    Returns a list of frame metadata dicts:
        {frame_index, original_frame_number, timestamp, frame_path}
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return []

    src_fps      = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, int(src_fps / target_fps))

    output_dir.mkdir(parents=True, exist_ok=True)

    frame_metas = []
    saved_idx   = 0
    frame_num   = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_num % step == 0:
            filename   = f"frame_{saved_idx:06d}.jpg"
            frame_path = output_dir / filename
            cv2.imwrite(str(frame_path), frame)

            frame_metas.append({
                "frame_index":         saved_idx,
                "original_frame_number": frame_num,
                "timestamp":           round(frame_num / src_fps, 4),
                "frame_path":          str(frame_path.relative_to(LEROBOT_DIR)),
            })
            saved_idx += 1

        frame_num += 1

    cap.release()
    return frame_metas


def flatten_bbox(bbox_frames: list[dict]) -> dict[int, list]:
    """
    Build a fast lookup mapping original_frame_number → list of objects.
    """
    idx = {}
    for frame in bbox_frames:
        fn = frame.get("frame_number", 0)
        idx[fn] = frame.get("objects", [])
    return idx


def write_episode(
    record: dict,
    episode_dir: Path,
    episode_index: int,
) -> dict:
    """
    Convert one master_dataset record into a LeRobot episode.

    Writes:
        frames/frame_NNNNNN.jpg
        episode_data.jsonl      (one JSON object per extracted frame)

    Returns the episode index entry for dataset.json.
    """
    seg_id     = record["segment_id"]
    video_path = SEGMENTS_ROOT / record["video_path"]
    action     = record["action_label"]
    objects    = record["objects_present"]
    description= record["nl_description"]
    bbox_lookup= flatten_bbox(record.get("bbox_annotations", []))

    frames_dir = episode_dir / "frames"

    # Skip if source video is missing
    if not video_path.exists():
        print(f"  {YELLOW}⚠  Video not found: {video_path} — writing metadata-only episode{RESET}")
        frame_metas = []
    else:
        frame_metas = extract_frames_at_fps(video_path, frames_dir, OUTPUT_FPS)

    # Write episode_data.jsonl — one row per frame
    jsonl_path = episode_dir / "episode_data.jsonl"
    with open(jsonl_path, "w") as f:
        for fm in frame_metas:
            orig_fn = fm["original_frame_number"]
            # Best-effort bbox lookup (nearest annotated frame)
            bbox_objs = bbox_lookup.get(orig_fn, [])

            row = {
                # LeRobot standard keys
                "episode_index":    episode_index,
                "frame_index":      fm["frame_index"],
                "timestamp":        fm["timestamp"],
                "frame_path":       fm["frame_path"],

                # Action / observation context
                "action":           action,
                "task_description": description,

                # Object annotations (may be empty between sampled frames)
                "detected_objects": bbox_objs,

                # Metadata carried through
                "segment_id":       seg_id,
                "task_category":    record["task_category"],
                "objects_present":  objects,
            }
            f.write(json.dumps(row) + "\n")

    return {
        "episode_id":    ep_id_str(episode_index),
        "segment_id":    seg_id,
        "action_label":  action,
        "task_category": record["task_category"],
        "split":         record["split"],
        "start_time":    record["start_time"],
        "end_time":      record["end_time"],
        "duration":      record["duration"],
        "n_frames":      len(frame_metas),
        "episode_dir":   str(episode_dir.relative_to(LEROBOT_DIR)),
    }


def ep_id_str(i: int) -> str:
    return f"episode_{i:06d}"


def run() -> None:
    LEROBOT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\n{BOLD}Loading master_dataset.json…{RESET}")
    records = load_master(MASTER_JSON)
    print(f"  {len(records)} records to convert\n")

    episodes_dir = LEROBOT_DIR / "episodes"
    episodes_dir.mkdir(exist_ok=True)

    episode_index_entries = []
    action_set = set()
    total_frames = 0

    for idx, record in enumerate(records):
        seg_id     = record["segment_id"]
        ep_dir     = episodes_dir / seg_id
        action_set.add(record["action_label"])

        print(f"[{idx+1:>4}/{len(records)}] {seg_id}  ({record['action_label']})")
        entry = write_episode(record, ep_dir, idx)
        total_frames += entry["n_frames"]
        episode_index_entries.append(entry)

    # ── dataset.json — master index ───────────────────────────
    dataset_json = LEROBOT_DIR / "dataset.json"
    with open(dataset_json, "w") as f:
        json.dump(episode_index_entries, f, indent=2)

    # ── meta_data.json — LeRobot standard ────────────────────
    split_counts = {}
    for e in episode_index_entries:
        s = e["split"]
        split_counts[s] = split_counts.get(s, 0) + 1

    meta = {
        "dataset_name":     "EgoKitchen-Robotics",
        "version":          "1.0.0",
        "date_created":     date.today().isoformat(),
        "source":           "custom_egocentric_recordings",
        "license":          "cc-by-4.0",
        "total_episodes":   len(records),
        "total_frames":     total_frames,
        "output_fps":       OUTPUT_FPS,
        "action_categories":sorted(action_set),
        "n_action_categories": len(action_set),
        "splits":           split_counts,
        "observation_keys": ["frame_path", "detected_objects"],
        "action_keys":      ["action", "task_description"],
        "episode_dir_structure": "episodes/<segment_id>/frames/frame_NNNNNN.jpg",
        "annotation_format":    "episodes/<segment_id>/episode_data.jsonl",
        "pipeline": {
            "segmentation":  "PySceneDetect + CLIP",
            "object_detect": "YOLOv8n + ByteTrack",
            "captioning":    "BLIP-2 Opt-2.7B",
            "description":   "Claude 3.5 Sonnet",
        }
    }

    meta_path = LEROBOT_DIR / "meta_data.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n{BOLD}{GREEN}✔ LeRobot dataset written to: {LEROBOT_DIR}{RESET}")
    print(f"  Episodes   : {len(records)}")
    print(f"  Total frames extracted : {total_frames}")
    print(f"  meta_data.json → {meta_path}")
    print(f"  dataset.json  → {dataset_json}\n")


if __name__ == "__main__":
    run()
