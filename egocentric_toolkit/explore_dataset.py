"""
=============================================================
SCRIPT 1: Egocentric Video Dataset Explorer
=============================================================
Usage:
    python explore_dataset.py /path/to/video/folder

Outputs:
    - Prints a rich summary table to the terminal
    - Saves dataset_summary.csv in the video folder
=============================================================
"""

import os
import sys
import cv2
import pandas as pd
from pathlib import Path
from datetime import timedelta


# ─── ANSI colors for a nicer terminal experience ───────────
GREEN  = "\033[92m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
RED    = "\033[91m"
BOLD   = "\033[1m"
RESET  = "\033[0m"


def scan_videos(root_folder: str) -> list[Path]:
    """
    Recursively scan a folder for all .mp4 and .avi video files.

    Args:
        root_folder: Path to the root directory to search.

    Returns:
        Sorted list of Path objects for every video found.
    """
    root = Path(root_folder)
    if not root.exists():
        print(f"{RED}ERROR: Folder '{root_folder}' does not exist.{RESET}")
        sys.exit(1)

    video_extensions = {".mp4", ".avi"}
    videos = sorted([
        p for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() in video_extensions
    ])

    print(f"{CYAN}Found {len(videos)} video(s) in '{root_folder}'{RESET}\n")
    return videos


def extract_metadata(video_path: Path) -> dict:
    """
    Extract metadata from a single video file using OpenCV.

    Args:
        video_path: Path object pointing to the video file.

    Returns:
        Dictionary with metadata fields, or error flags if unreadable.
    """
    meta = {
        "filename":    video_path.name,
        "filepath":    str(video_path),
        "extension":   video_path.suffix.lower(),
        "file_size_mb": round(video_path.stat().st_size / (1024 ** 2), 2),
        "duration_sec": None,
        "duration_hms": None,
        "fps":          None,
        "width":        None,
        "height":       None,
        "resolution":   None,
        "total_frames": None,
        "readable":     False,
    }

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        cap.release()
        return meta  # readable stays False

    fps          = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration_sec = total_frames / fps if fps > 0 else 0

    meta.update({
        "fps":          round(fps, 2),
        "width":        width,
        "height":       height,
        "resolution":   f"{width}x{height}",
        "total_frames": total_frames,
        "duration_sec": round(duration_sec, 2),
        "duration_hms": str(timedelta(seconds=int(duration_sec))),
        "readable":     True,
    })

    cap.release()
    return meta


def build_summary_table(video_paths: list[Path]) -> pd.DataFrame:
    """
    Extract metadata for every video and return a tidy DataFrame.

    Args:
        video_paths: List of video Path objects.

    Returns:
        pandas DataFrame with one row per video.
    """
    print(f"{BOLD}Extracting metadata...{RESET}")
    records = []

    for i, vp in enumerate(video_paths, 1):
        print(f"  [{i:>3}/{len(video_paths)}] {vp.name}", end="\r")
        records.append(extract_metadata(vp))

    print()  # newline after the progress line
    return pd.DataFrame(records)


def print_summary(df: pd.DataFrame) -> None:
    """
    Print a human-readable summary table and aggregate statistics
    to the terminal.

    Args:
        df: Summary DataFrame built by build_summary_table().
    """
    readable = df[df["readable"]]
    corrupt  = df[~df["readable"]]

    # ── per-video table ──────────────────────────────────────
    display_cols = [
        "filename", "resolution", "fps", "duration_hms",
        "total_frames", "file_size_mb"
    ]
    print(f"\n{BOLD}{CYAN}{'─'*70}")
    print(" DATASET SUMMARY TABLE")
    print(f"{'─'*70}{RESET}")
    pd.set_option("display.max_colwidth", 35)
    pd.set_option("display.width", 120)
    print(readable[display_cols].to_string(index=False))

    # ── aggregate stats ──────────────────────────────────────
    print(f"\n{BOLD}{GREEN}{'─'*70}")
    print(" AGGREGATE STATISTICS")
    print(f"{'─'*70}{RESET}")
    if not readable.empty:
        print(f"  Total videos      : {len(df)}")
        print(f"  Readable videos   : {len(readable)}")
        print(f"  Corrupt/unreadable: {len(corrupt)}")
        print(f"  Total size (GB)   : {readable['file_size_mb'].sum() / 1024:.3f}")
        print(f"  Total duration    : {str(timedelta(seconds=int(readable['duration_sec'].sum())))}")
        print(f"  Avg FPS           : {readable['fps'].mean():.2f}")
        print(f"  Avg duration (s)  : {readable['duration_sec'].mean():.2f}")
        print(f"  Unique resolutions: {readable['resolution'].nunique()}")
        print(f"  Resolutions found : {', '.join(readable['resolution'].unique())}")

    if not corrupt.empty:
        print(f"\n{RED}  ⚠ Corrupt files:{RESET}")
        for _, row in corrupt.iterrows():
            print(f"    • {row['filepath']}")


def save_csv(df: pd.DataFrame, root_folder: str) -> None:
    """
    Save the summary DataFrame as a CSV file in the root folder.

    Args:
        df:          Summary DataFrame.
        root_folder: Path to the root video directory (CSV saved here).
    """
    out_path = Path(root_folder) / "dataset_summary.csv"
    df.to_csv(out_path, index=False)
    print(f"\n{BOLD}{GREEN}✔ Saved: {out_path}{RESET}")


# ─── MAIN ──────────────────────────────────────────────────
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"{YELLOW}Usage: python explore_dataset.py <video_folder>{RESET}")
        sys.exit(1)

    video_folder = sys.argv[1]

    videos = scan_videos(video_folder)

    if not videos:
        print(f"{YELLOW}No .mp4 or .avi files found. Exiting.{RESET}")
        sys.exit(0)

    summary_df = build_summary_table(videos)
    print_summary(summary_df)
    save_csv(summary_df, video_folder)
