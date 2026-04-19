"""
=============================================================
SCRIPT 2: YOLOv8 + ByteTrack Object Tracking
=============================================================
Usage:
    python track_yolo.py

What it does:
    • Scans /dataset/segments/ for segmented MP4 clips
    • Runs YOLOv8 with built-in ByteTrack module on every frame (to maintain IDs)
    • Samples the tracking results every Nth frame (for efficiency in output)
    • Filters detections to kitchen/cleaning related COCO classes
    • Saves tracked annotations (with stable object_ids) to /dataset/annotations/tracked_bbox_annotations.json

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

# How often to write frames to the JSON output
# Note: Tracking runs on EVERY frame internally to maintain ID consistency, 
# but we only save the results every Nth frame to keep the JSON small.
FRAME_STEP = 5

CONF_THRESH = 0.25

KITCHEN_CLASSES = [39, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 60, 78, 79, 80, 81, 82]
# ───────────────────────────────────────────────────────────

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
    out_json = annotations_dir / "tracked_bbox_annotations.json"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n{BOLD}Loading YOLOv8n (Tracking) on {device}…{RESET}")
    model = YOLO("yolov8n.pt")

    videos = sorted([p for p in segments_dir.rglob("*.mp4") if p.is_file()])
    print(f"{CYAN}Found {len(videos)} segment(s) to process.{RESET}\n")

    all_annotations = []
    total_objects_saved = 0

    for idx, video_path in enumerate(videos, 1):
        segment_id = video_path.stem
        print(f"{BOLD}[{idx}/{len(videos)}] Tracking {segment_id}{RESET}")

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"  {RED}Cannot open {video_path.name}{RESET}")
            continue

        fps = cap.get(cv2.CAP_PROP_FPS)

        segment_data = {
            "segment_id": segment_id,
            "video_path": str(video_path),
            "frames": []
        }

        frame_idx = 0
        objects_in_segment = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Run tracking on EVERY frame to maintain temporal continuity
            # persist=True keeps the tracking state across frames
            results = model.track(frame, persist=True, verbose=False, tracker="bytetrack.yaml", 
                                  classes=KITCHEN_CLASSES, conf=CONF_THRESH)

            # But only save results every FRAME_STEP frames
            if frame_idx % FRAME_STEP == 0:
                timestamp = frame_idx / fps if fps > 0 else 0.0
                frame_objects = []
                
                # We know there's only one frame of results because we passed a single frame
                result = results[0]  
                
                if result.boxes is not None and result.boxes.id is not None:
                    boxes = result.boxes.xyxy.cpu().numpy()
                    track_ids = result.boxes.id.int().cpu().numpy()
                    classes = result.boxes.cls.int().cpu().numpy()
                    confs = result.boxes.conf.cpu().numpy()
                    
                    for box, track_id, cls_id, conf in zip(boxes, track_ids, classes, confs):
                        x1, y1, x2, y2 = box.tolist()
                        cls_name = model.names[cls_id]

                        frame_objects.append({
                            "object_id": f"track_{track_id:04d}",
                            "class": cls_name,
                            "bbox": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)],
                            "confidence": round(float(conf), 4)
                        })
                        objects_in_segment += 1
                        total_objects_saved += 1

                segment_data["frames"].append({
                    "frame_number": frame_idx,
                    "timestamp": round(timestamp, 3),
                    "objects": frame_objects
                })

            frame_idx += 1

        cap.release()
        
        # Reset tracking state for the next video clip so IDs don't carry over between independent segments
        # Note: If memory buildup occurs, one could reload the model or clear Ultralytics track buffers
        all_annotations.append(segment_data)
        print(f"  → Tracked {objects_in_segment} instances across {len(segment_data['frames'])} sampled frames.")

    with open(out_json, "w") as f:
        json.dump(all_annotations, f, indent=2)

    print(f"\n{BOLD}{GREEN}✔ tracked_bbox_annotations.json saved to: {out_json}{RESET}")
    print(f"  Total bounded object instances saved: {total_objects_saved}")


if __name__ == "__main__":
    run()
