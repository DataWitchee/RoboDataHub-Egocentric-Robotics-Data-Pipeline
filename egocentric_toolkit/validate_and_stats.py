import pandas as pd
import os
from pathlib import Path

# ==========================================
# CONFIGURABLE PATHS
# ==========================================
DATASET_ROOT = Path(os.getenv("DATASET_ROOT", str(Path.home() / "Desktop" / "my_robotics_data")))
SUBMISSION_CSV = DATASET_ROOT / "final_dataset" / "submission.csv"
STATS_CSV = DATASET_ROOT / "final_dataset" / "submission_stats.csv"

def generate_stats(df):
    """Generates submission_stats.csv per the requirements."""
    print(f"Generating stats for {len(df)} segments...")
    
    # Calculate groupings
    # Ensure numeric columns 
    for col in ["duration_sec", "object_count", "description_length"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    stats = df.groupby("action_label").agg(
        count=("segment_id", "count"),
        avg_duration=("duration_sec", "mean"),
        avg_objects=("object_count", "mean"),
        avg_description_length=("description_length", "mean")
    ).reset_index()

    # Round numeric columns for cleanliness
    stats["avg_duration"] = stats["avg_duration"].round(1)
    stats["avg_objects"] = stats["avg_objects"].round(1)
    stats["avg_description_length"] = stats["avg_description_length"].round(0).astype(int)
    
    # Sort by count descending
    stats = stats.sort_values("count", ascending=False)
    
    stats.to_csv(STATS_CSV, index=False)
    print(f"Stats saved to: {STATS_CSV}")

def validate_submission(df):
    """Validates the CSV against hackathon rules."""
    print("\nRunning CSV Validation...")
    errors = []
    
    # 1. Required columns
    required_cols = [
        "segment_id", "video_path", "action_label", "task_category", 
        "start_time", "end_time", "duration_sec", "objects_detected",
        "object_count", "bbox_count", "avg_confidence", "nl_description",
        "description_length", "description_source", "split"
    ]
    
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        errors.append(f"Missing required columns: {missing_cols}")

    # 2. No empty segment_id or nl_description
    if "segment_id" in df.columns:
        empty_segs = df["segment_id"].isna().sum() + (df["segment_id"] == "").sum()
        if empty_segs > 0:
            errors.append(f"Found {empty_segs} rows with missing segment_id")
            
    if "nl_description" in df.columns:
        empty_desc = df["nl_description"].isna().sum() + (df["nl_description"] == "").sum() + (df["nl_description"] == "N/A").sum()
        if empty_desc > 0:
            errors.append(f"Found {empty_desc} rows with missing 'nl_description' (or N/A)")

    # 3. Valid timing
    if "start_time" in df.columns and "end_time" in df.columns:
        invalid_times = len(df[df["start_time"] >= df["end_time"]])
        if invalid_times > 0:
            errors.append(f"Found {invalid_times} rows where start_time >= end_time")

    # 4. Valid split
    if "split" in df.columns:
        valid_splits = {"train", "val", "test"}
        invalid_splits = df[~df["split"].isin(valid_splits)]
        if len(invalid_splits) > 0:
            bad_vals = invalid_splits["split"].unique()
            errors.append(f"Found invalid split values: {bad_vals}. Must be train, val, or test.")
            
    # Output Results
    if errors:
        print("❌ VALIDATION FAILED! The following issues were found:")
        for err in errors:
            print(f"  - {err}")
    else:
        print("✅ CSV is valid and ready for submission")


def main():
    if not SUBMISSION_CSV.exists():
        print(f"Error: {SUBMISSION_CSV} does not exist. Run create_submission.py first.")
        return
        
    df = pd.read_csv(SUBMISSION_CSV)
    
    generate_stats(df)
    validate_submission(df)

if __name__ == "__main__":
    main()
