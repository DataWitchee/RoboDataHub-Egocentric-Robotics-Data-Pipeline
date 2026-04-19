"""
=============================================================
SCRIPT 3: Bounding Box Annotation Visualizer
=============================================================
Usage:
    python visualize_annotations.py

What it does:
    • Loads tracked bounding boxes from /dataset/annotations/tracked_bbox_annotations.json
    • For each segmented clip, draws the bounding boxes, tracking IDs, and confidences
    • Saves the annotated clip as a new video in /dataset/previews/annotations/
    • Saves a sample annotated frame per segment as a JPG for quick review

Dependencies:
    pip install opencv-python numpy pandas
=============================================================
"""

import json
import sys
from pathlib import Path
import random

import cv2
import numpy as np

# ─── CONFIG ────────────────────────────────────────────────
ANNOTATIONS_JSON = Path("/Users/mannatsaini/Desktop/my_robotics_data/annotations/tracked_bbox_annotations.json")
PREVIEWS_DIR     = Path("/Users/mannatsaini/Desktop/my_robotics_data/previews/annotations")

# Colors for different classes to make them distinct
# Format is BGR
CLASS_COLORS = {
    "bottle": (200, 130, 0), "cup": (100, 200, 0), "fork": (100, 100, 200),
    "knife": (0, 0, 200), "spoon": (150, 150, 200), "bowl": (0, 200, 200),
    "sink": (250, 150, 100), "refrigerator": (200, 200, 200), 
    "microwave": (50, 100, 150), "oven": (100, 50, 150), "toaster": (150, 50, 100)
}
DEFAULT_COLOR = (0, 255, 0) # Green for anything else
# ───────────────────────────────────────────────────────────

CYAN  = "\033[96m"
GREEN = "\033[92m"
YELLOW= "\033[93m"
RED   = "\033[91m"
BOLD  = "\033[1m"
DIM   = "\033[2m"
RESET = "\033[0m"


def load_annotations(json_path: Path) -> list[dict]:
    if not json_path.exists():
        print(f"{RED}ERROR: Annotations not found at '{json_path}'{RESET}")
        sys.exit(1)
    with open(json_path) as f:
        return json.load(f)


def visualize_segment(segment_data: dict, output_dir: Path):
    segment_id = segment_data.get("segment_id")
    video_path = segment_data.get("video_path")
    frames_data = segment_data.get("frames", [])

    if not Path(video_path).exists():
        print(f"  {YELLOW}Video not found at {video_path}, skipping visualization.{RESET}")
        return

    # Create mapping of frame number to objects for quick lookup
    annotated_frames = {f["frame_number"]: f["objects"] for f in frames_data}
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  {RED}Cannot open video: {video_path}{RESET}")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Output video writer setup
    out_video_path = output_dir / f"{segment_id}_annotated.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(out_video_path), fourcc, fps, (width, height))

    # Pick a random annotated frame to save as JPG
    frames_with_detections = [f for f, objs in annotated_frames.items() if len(objs) > 0]
    sample_frame_idx = random.choice(frames_with_detections) if frames_with_detections else 0

    frame_idx = 0
    saved_sample = False

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Only draw if we saved annotations for this frame
        # (Since we ran tracking on all but saved only every FRAME_STEP)
        # To make a smooth video, we could interpolate boxes, but for now we only overlay
        # on frames we actually saved data for. Unannotated frames will stay untouched.
        # Alternatively, we just display the nearest past annotation.
        
        # Simple nearest-past-frame logic to keep boxes visible
        past_keys = [k for k in annotated_frames.keys() if k <= frame_idx]
        nearest_key = max(past_keys) if past_keys else None

        if nearest_key is not None:
            objects = annotated_frames[nearest_key]
            for obj in objects:
                x1, y1, x2, y2 = map(int, obj["bbox"])
                cls_name = obj["class"]
                track_id = obj["object_id"]
                conf = obj["confidence"]

                color = CLASS_COLORS.get(cls_name, DEFAULT_COLOR)
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                
                # Label background
                label = f"{cls_name} {track_id} {conf:.2f}"
                (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(frame, (x1, y1 - 20), (x1 + w, y1), color, -1)
                
                # Label text
                cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        out.write(frame)
        
        if frame_idx == sample_frame_idx and not saved_sample:
            sample_img_path = output_dir / f"{segment_id}_sample.jpg"
            cv2.imwrite(str(sample_img_path), frame)
            saved_sample = True

        frame_idx += 1

    cap.release()
    out.release()
    print(f"  {GREEN}✔ Created {out_video_path.name}{RESET}")


def run(json_path: Path = ANNOTATIONS_JSON,
        out_dir: Path = PREVIEWS_DIR) -> None:
    all_data = load_annotations(json_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{BOLD}Annotating {len(all_data)} video segments…{RESET}")
    for idx, segment_data in enumerate(all_data, 1):
        print(f"[{idx}/{len(all_data)}] Rendering {segment_data['segment_id']}")
        visualize_segment(segment_data, out_dir)

    print(f"\n{BOLD}{GREEN}Done! Visualizations saved to: {out_dir}{RESET}")

if __name__ == "__main__":
    run()
