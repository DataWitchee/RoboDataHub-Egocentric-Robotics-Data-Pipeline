import os
import json
import pandas as pd
import numpy as np
from pathlib import Path

# ==========================================
# CONFIGURABLE PATHS
# ==========================================
DATASET_ROOT = Path(os.getenv("DATASET_ROOT", str(Path.home() / "Desktop" / "my_robotics_data")))
SEGMENTS_FILE = DATASET_ROOT / "segments" / "segments.json"
# We prefer tracked annotations if available
ANNOTATIONS_FILE_TRACKED = DATASET_ROOT / "annotations" / "tracked_bbox_annotations.json"
ANNOTATIONS_FILE_RAW = DATASET_ROOT / "annotations" / "bbox_annotations.json"
DESCRIPTIONS_FILE = DATASET_ROOT / "descriptions" / "nl_descriptions.json"
OUTPUT_CSV = DATASET_ROOT / "final_dataset" / "submission.csv"

# Make sure output directory exists
OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)

def load_json(path):
    if path.exists():
        with open(path, 'r') as f:
            return json.load(f)
    return []

def main():
    print("Loading JSON files...")
    segments_data = load_json(SEGMENTS_FILE)
    
    anno_path = ANNOTATIONS_FILE_TRACKED if ANNOTATIONS_FILE_TRACKED.exists() else ANNOTATIONS_FILE_RAW
    annotations_data = load_json(anno_path)
    
    descriptions_data = load_json(DESCRIPTIONS_FILE)

    # 1. Parse Segments
    flat_segments = []
    for video_entry in segments_data:
        video_id = video_entry.get("video_id", "")
        for seg in video_entry.get("segments", []):
            start = seg.get("start_time", 0.0)
            end = seg.get("end_time", 0.0)
            flat_segments.append({
                "segment_id": seg.get("segment_id", ""),
                "video_path": f"segments/{seg.get('segment_id', '')}.mp4",
                "action_label": seg.get("action_label", "unknown"),
                "task_category": "kitchen", # Fixed category for the hackathon
                "start_time": start,
                "end_time": end,
                "duration_sec": round(end - start, 2)
            })
    
    df_segments = pd.DataFrame(flat_segments)

    # 2. Parse Annotations
    anno_rows = []
    for entry in annotations_data:
        sid = entry.get("segment_id", "")
        unique_objs = set()
        total_bboxes = 0
        confidences = []
        
        for frame in entry.get("frames", []):
            objs = frame.get("objects", [])
            total_bboxes += len(objs)
            for obj in objs:
                unique_objs.add(obj.get("class", "unknown"))
                if "confidence" in obj:
                    confidences.append(obj["confidence"])
                elif "conf" in obj:
                    confidences.append(obj["conf"])
                    
        avg_conf = round(sum(confidences) / len(confidences), 2) if confidences else 0.0
        
        anno_rows.append({
            "segment_id": sid,
            "objects_detected": ", ".join(sorted(unique_objs)),
            "object_count": len(unique_objs),
            "bbox_count": total_bboxes,
            "avg_confidence": avg_conf
        })
        
    df_anno = pd.DataFrame(anno_rows) if anno_rows else pd.DataFrame(columns=["segment_id", "objects_detected", "object_count", "bbox_count", "avg_confidence"])

    # 3. Parse Descriptions
    desc_rows = []
    for entry in descriptions_data:
        desc = entry.get("nl_description", "")
        desc_rows.append({
            "segment_id": entry.get("segment_id", ""),
            "nl_description": desc,
            "description_length": len(desc.split()),
            "description_source": entry.get("description_version", "template")
        })
    df_desc = pd.DataFrame(desc_rows) if desc_rows else pd.DataFrame(columns=["segment_id", "nl_description", "description_length", "description_source"])

    # MERGE ALL DATA
    print("Merging dataframes...")
    if not df_segments.empty:
        df = df_segments.merge(df_anno, on="segment_id", how="left")
        df = df.merge(df_desc, on="segment_id", how="left")
    else:
        print("No segments found! Output will be empty.")
        df = pd.DataFrame()

    # Assign random splits (80/10/10) reliably
    if not df.empty:
        splits = np.random.choice(["train", "val", "test"], p=[0.8, 0.1, 0.1], size=len(df))
        df["split"] = splits

        # ==========================================
        # CSV CLEANING STEP
        # ==========================================
        print("Cleaning CSV data...")
        
        # Fill missing values
        numeric_cols = ["start_time", "end_time", "duration_sec", "object_count", "bbox_count", "avg_confidence", "description_length"]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col]).fillna(0)
                
        text_cols = ["action_label", "objects_detected", "nl_description", "description_source"]
        for col in text_cols:
            if col in df.columns:
                df[col] = df[col].astype(str).replace("nan", "N/A").fillna("N/A")
                # Strip extra whitespace from text columns
                df[col] = df[col].str.strip()

        # Remove duplicates
        df = df.drop_duplicates(subset=["segment_id"])

        # Sort rows
        if "video_path" in df.columns and "start_time" in df.columns:
            df = df.sort_values(by=["video_path", "start_time"])

        # Create precise column order
        cols_order = [
            "segment_id", "video_path", "action_label", "task_category", 
            "start_time", "end_time", "duration_sec", "objects_detected",
            "object_count", "bbox_count", "avg_confidence", "nl_description",
            "description_length", "description_source", "split"
        ]
        # Just in case some column is missing entirely, filter to intersect
        cols_order = [c for c in cols_order if c in df.columns]
        df = df[cols_order]

    # Save to disk
    df.to_csv(OUTPUT_CSV, index=False)
    
    # ==========================================
    # SUBMISSION SUMMARY PRINTER
    # ==========================================
    try:
        tot_segments = len(df)
        actions = ", ".join(df["action_label"].unique())
        tot_objects = int(df["bbox_count"].sum())
        avg_dur = round(df["duration_sec"].mean(), 1) if not df["duration_sec"].empty else 0
        desc_gen = len(df[df["nl_description"] != "N/A"])
        
        # count description sources safely
        source_counts = df["description_source"].value_counts()
        claude_refined = source_counts.get("claude-refined", source_counts.get("claude", 0))
        templates = source_counts.get("template", 0)
        
        split_counts = df["split"].value_counts()
        train_c = split_counts.get("train", 0)
        val_c = split_counts.get("val", 0)
        test_c = split_counts.get("test", 0)
        
        # Verify missing values
        missing_count = df.isna().sum().sum()

        print("\n============================================")
        print("HACKATHON SUBMISSION DATASET SUMMARY")
        print("============================================")
        print(f"Total Segments         : {tot_segments}")
        print(f"Action Categories      : {actions}")
        print(f"Total Objects Detected : {tot_objects}")
        print(f"Avg Segment Duration   : {avg_dur}s")
        print(f"Descriptions Generated : {desc_gen}")
        print(f"Claude Refined         : {claude_refined}")
        print(f"Template Generated     : {templates}")
        print(f"Train / Val / Test     : {train_c} / {val_c} / {test_c}")
        print(f"Missing Values         : {missing_count}")
        print(f"CSV Path               : {OUTPUT_CSV}")
        print("============================================")
        print("READY FOR SUBMISSION ✅")
        print("============================================\n")
    except Exception as e:
        print(f"Summary failed (empty dataset?): {e}")

if __name__ == "__main__":
    main()
