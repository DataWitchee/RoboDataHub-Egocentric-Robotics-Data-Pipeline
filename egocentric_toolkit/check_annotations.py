"""
=============================================================
SCRIPT 4: Annotation Quality Checker
=============================================================
Usage:
    python check_annotations.py

What it does:
    • Loads tracked bounding boxes from /dataset/annotations/tracked_bbox_annotations.json
    • Flags frames with zero detections
    • Flags segments displaying drops in confidence (< 0.5 mean confidence)
    • Aggregates total objects detected, class distribution, and avg conf per class
    • Saves segment-level quality reports to /dataset/annotations/annotations_summary.csv

Dependencies:
    pip install pandas
=============================================================
"""

import json
import sys
from pathlib import Path
from collections import defaultdict

import pandas as pd

# ─── CONFIG ────────────────────────────────────────────────
ANNOTATIONS_JSON = Path("/Users/mannatsaini/Desktop/my_robotics_data/annotations/tracked_bbox_annotations.json")
ANNOTATIONS_DIR  = Path("/Users/mannatsaini/Desktop/my_robotics_data/annotations")

LOW_CONFIDENCE_THRESH = 0.5
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


def analyze_annotations(all_data: list[dict], out_dir: Path):
    class_counts = defaultdict(int)
    class_confs = defaultdict(list)
    total_objects = 0
    total_frames_checked = 0
    frames_with_zero_detections = 0

    segment_reports = []

    for seg in all_data:
        segment_id = seg["segment_id"]
        frames = seg.get("frames", [])
        
        seg_confs = []
        seg_zero_frames = 0
        seg_objects = 0

        for frame in frames:
            total_frames_checked += 1
            objs = frame.get("objects", [])
            
            if not objs:
                frames_with_zero_detections += 1
                seg_zero_frames += 1
            
            for obj in objs:
                cls_name = obj["class"]
                conf = obj["confidence"]
                
                class_counts[cls_name] += 1
                class_confs[cls_name].append(conf)
                seg_confs.append(conf)
                
                seg_objects += 1
                total_objects += 1

        avg_seg_conf = sum(seg_confs) / len(seg_confs) if seg_confs else 0.0
        
        issues = []
        if seg_zero_frames > 0:
            issues.append(f"{seg_zero_frames} blank frames")
        if avg_seg_conf > 0 and avg_seg_conf < LOW_CONFIDENCE_THRESH:
            issues.append("low overall confidence")

        segment_reports.append({
            "segment_id": segment_id,
            "total_frames_sampled": len(frames),
            "empty_frames": seg_zero_frames,
            "total_objects": seg_objects,
            "average_confidence": round(avg_seg_conf, 3),
            "issues": " | ".join(issues) if issues else "None"
        })

    # Summary Output
    print(f"\n{BOLD}{CYAN}{'═'*60}")
    print("  ANNOTATION QUALITY REPORT")
    print(f"{'═'*60}{RESET}\n")

    print(f"  Total Segments Checked     : {len(all_data)}")
    print(f"  Total Frames Sampled       : {total_frames_checked}")
    print(f"  Frames with Zero Objects   : {YELLOW}{frames_with_zero_detections}{RESET} ({(frames_with_zero_detections/max(total_frames_checked,1))*100:.1f}%)")
    print(f"  Total Box Instances        : {total_objects}")
    
    print(f"\n  {BOLD}Class Distribution & Avg Confidence:{RESET}")
    if not class_counts:
        print(f"    {YELLOW}No objects detected across the dataset.{RESET}")
    else:
        for cls_name, count in sorted(class_counts.items(), key=lambda x: x[1], reverse=True):
            avg_conf = sum(class_confs[cls_name]) / len(class_confs[cls_name])
            bar = "█" * int(count / max(class_counts.values()) * 30)
            print(f"    {cls_name:<15} {count:>5} instances  {avg_conf:.2f} conf  {bar:<30}")

    df = pd.DataFrame(segment_reports)
    out_csv = out_dir / "annotations_summary.csv"
    df.to_csv(out_csv, index=False)

    print(f"\n{BOLD}{GREEN}✔ Summary report saved to: {out_csv}{RESET}")
    
    flagged = df[df["issues"] != "None"]
    if len(flagged) > 0:
        print(f"\n{RED}⚠ WARNING: {len(flagged)} segment(s) flagged for potential issues.{RESET}")
        for _, row in flagged.head(5).iterrows():
            print(f"    → {row['segment_id']}: {row['issues']}")
        if len(flagged) > 5:
            print(f"    → ... and {len(flagged) - 5} more.")


def run(json_path: Path = ANNOTATIONS_JSON,
        out_dir: Path = ANNOTATIONS_DIR) -> None:
    all_data = load_annotations(json_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    analyze_annotations(all_data, out_dir)


if __name__ == "__main__":
    run()
