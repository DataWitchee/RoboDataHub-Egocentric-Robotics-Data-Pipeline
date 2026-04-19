"""
=============================================================
SCRIPT 1: YOLOv8 Object Detection
=============================================================
Usage:
    python annotate_yolo.py

What it does:
    • Scans /dataset/segments/ for segmented MP4 clips
    • Runs YOLOv8 object detection on every Nth frame
    • Filters detections to kitchen/cleaning related COCO classes
    • Saves per-frame annotations to /dataset/annotations/bbox_annotations.json

Dependencies:
    pip install ultralytics opencv-python pandas
=============================================================
"""

import json
import sys
from pathlib import Path

import cv2
import torch
from ultralytics import YOLO

# ─── CONFIG ────────────────────────────────────────────────
SEGMENTS_DIR    = Path("/Users/mannatsaini/Desktop/my_robotics_data/segments")
ANNOTATIONS_DIR = Path("/Users/mannatsaini/Desktop/my_robotics_data/annotations")

# Run detection on every Nth frame
FRAME_STEP = 5

# Minimum confidence threshold
CONF_THRESH = 0.25

# COCO classes relevant to kitchen/household tasks
# 39: bottle, 41: cup, 42: fork, 43: knife, 44: spoon, 45: bowl
# 46-55: food items (banana, apple, sandwich, etc.)
# 60: dining table, 78: microwave, 79: oven, 80: toaster, 81: sink, 82: refrigerator
KITCHEN_CLASSES = [39, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 60, 78, 79, 80, 81, 82]
# ───────────────────────────────────────────────────────────

# ─── ANSI colours ──────────────────────────────────────────
CYAN  = "\033[96m"
GREEN = "\033[92m"
YELLOW= "\033[93m"
RED   = "\033[91m"
BOLD  = "\033[1m"
DIM   = "\033[2m"
RESET = "\033[0m"


def run(segments_dir: Path = SEGMENTS_DIR,
        annotations_dir: Path = ANNOTATIONS_DIR) -> None:
    if not segments_dir.exists():
        print(f"{RED}ERROR: SEGMENTS_DIR '{segments_dir}' does not exist.{RESET}")
        sys.exit(1)

    annotations_dir.mkdir(parents=True, exist_ok=True)
    out_json = annotations_dir / "bbox_annotations.json"

    # Load YOLOv8 model (yolov8n is fast, yolov8s/m are more accurate)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n{BOLD}Loading YOLOv8n on {device}…{RESET}")
    model = YOLO("yolov8n.pt")

    videos = sorted([p for p in segments_dir.rglob("*.mp4") if p.is_file()])
    print(f"{CYAN}Found {len(videos)} segment(s) to process.{RESET}\n")

    all_annotations = []
    total_objects = 0

    for idx, video_path in enumerate(videos, 1):
        segment_id = video_path.stem
        print(f"{BOLD}[{idx}/{len(videos)}] Processing {segment_id}{RESET}")

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"  {RED}Cannot open {video_path.name}{RESET}")
            continue

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        segment_data = {
            "segment_id": segment_id,
            "video_path": str(video_path),
            "frames": []
        }

        frame_idx = 0
        obj_counter = 1

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % FRAME_STEP == 0:
                timestamp = frame_idx / fps if fps > 0 else 0.0

                # Run inference
                results = model(frame, verbose=False, classes=KITCHEN_CLASSES, conf=CONF_THRESH)
                
                frame_objects = []
                for result in results:
                    boxes = result.boxes
                    for box in boxes:
                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        conf = float(box.conf[0])
                        cls_id = int(box.cls[0])
                        cls_name = model.names[cls_id]

                        frame_objects.append({
                            "object_id": f"obj_{obj_counter:04d}",
                            "class": cls_name,
                            "bbox": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)],
                            "confidence": round(conf, 4)
                        })
                        obj_counter += 1
                        total_objects += 1

                segment_data["frames"].append({
                    "frame_number": frame_idx,
                    "timestamp": round(timestamp, 3),
                    "objects": frame_objects
                })

            frame_idx += 1

        cap.release()
        all_annotations.append(segment_data)
        print(f"  → Found {obj_counter - 1} objects across {len(segment_data['frames'])} sampled frames.")

    # Save to JSON
    with open(out_json, "w") as f:
        json.dump(all_annotations, f, indent=2)

    print(f"\n{BOLD}{GREEN}✔ bbox_annotations.json saved to: {out_json}{RESET}")
    print(f"  Total objects detected: {total_objects}")


if __name__ == "__main__":
    run()
