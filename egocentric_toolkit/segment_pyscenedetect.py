"""
=============================================================
SCRIPT 1: PySceneDetect Scene-Boundary Segmenter
=============================================================
Usage:
    python segment_pyscenedetect.py

What it does:
    • Scans /dataset/raw_videos/ recursively for .mp4 / .avi files
    • Runs PySceneDetect's ContentDetector on each video
    • Splits every detected scene into a separate clip in /dataset/segments/
    • Names clips:  video001_seg001.mp4, video001_seg002.mp4 …
    • Writes a scene manifest:  /dataset/segments/scenes_manifest.csv

Dependencies:
    pip install scenedetect[opencv] opencv-python pandas
=============================================================
"""

import os
import sys
import csv
import shutil
import subprocess
from pathlib import Path

import cv2
import pandas as pd
from scenedetect import open_video, SceneManager, split_video_ffmpeg
from scenedetect.detectors import ContentDetector

# ─── CONFIG — edit these paths to match your setup ─────────
RAW_VIDEOS_DIR  = Path("/Users/mannatsaini/Desktop/my_robotics_data/raw_videos")
SEGMENTS_DIR    = Path("/Users/mannatsaini/Desktop/my_robotics_data/segments")

# ContentDetector threshold:
#   Lower  → more sensitive (detects subtle cuts)
#   Higher → less sensitive (only hard cuts)
CONTENT_THRESHOLD = 27.0

# Minimum scene length in seconds to keep (drops flash frames)
MIN_SCENE_LEN_SEC = 1.0

# Output video encoding params for FFmpeg split
FFMPEG_ARGS = "-c:v libx264 -preset fast -crf 22 -c:a aac"
# ───────────────────────────────────────────────────────────


# ─── ANSI colours ──────────────────────────────────────────
CYAN  = "\033[96m"
GREEN = "\033[92m"
YELLOW= "\033[93m"
RED   = "\033[91m"
BOLD  = "\033[1m"
DIM   = "\033[2m"
RESET = "\033[0m"


def scan_videos(root: Path) -> list[Path]:
    """Recursively find all .mp4 and .avi files under root."""
    exts = {".mp4", ".avi"}
    videos = sorted(p for p in root.rglob("*")
                    if p.is_file() and p.suffix.lower() in exts)
    print(f"{CYAN}Found {len(videos)} video(s) in '{root}'{RESET}")
    return videos


def detect_scenes(video_path: Path) -> list[tuple]:
    """
    Run PySceneDetect ContentDetector on a single video.

    Args:
        video_path: Path to the source video.

    Returns:
        List of (start_timecode, end_timecode) scene tuples.
        Returns [] if the file cannot be opened.
    """
    try:
        video = open_video(str(video_path))
    except Exception as exc:
        print(f"  {RED}Cannot open {video_path.name}: {exc}{RESET}")
        return []

    scene_manager = SceneManager()
    scene_manager.add_detector(
        ContentDetector(
            threshold=CONTENT_THRESHOLD,
            min_scene_len=int(MIN_SCENE_LEN_SEC *
                              video.frame_rate),   # frames, not seconds
        )
    )

    scene_manager.detect_scenes(video=video, show_progress=False)
    scenes = scene_manager.get_scene_list()
    return scenes


def split_scenes(
    video_path: Path,
    scenes: list[tuple],
    video_id: str,
    out_dir: Path,
) -> list[dict]:
    """
    Split a video into scene clips using FFmpeg via PySceneDetect.

    Clips are saved as: <out_dir>/<video_id>_seg<NNN>.mp4

    Args:
        video_path: Source video path.
        scenes:     Scene list from detect_scenes().
        video_id:   String identifier used in output filenames.
        out_dir:    Directory where clips will be written.

    Returns:
        List of dicts with metadata for each saved segment.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    records = []
    for i, (start_tc, end_tc) in enumerate(scenes, 1):
        seg_name  = f"{video_id}_seg{i:03d}.mp4"
        seg_path  = out_dir / seg_name
        start_sec = start_tc.get_seconds()
        end_sec   = end_tc.get_seconds()
        duration  = round(end_sec - start_sec, 3)

        # FFmpeg to extract the clip precisely
        cmd = [
            "ffmpeg", "-y",
            "-i", str(video_path),
            "-ss", str(start_sec),
            "-to", str(end_sec),
            *FFMPEG_ARGS.split(),
            str(seg_path),
        ]
        result = subprocess.run(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        status = "OK" if result.returncode == 0 else "FAILED"
        size_mb = round(seg_path.stat().st_size / (1024**2), 2) if seg_path.exists() else 0

        records.append({
            "video_id":    video_id,
            "segment_id":  seg_name,
            "start_sec":   round(start_sec, 3),
            "end_sec":     round(end_sec, 3),
            "duration_sec":duration,
            "start_frame": start_tc.get_frames(),
            "end_frame":   end_tc.get_frames(),
            "output_path": str(seg_path),
            "file_size_mb":size_mb,
            "status":      status,
        })

        colour = GREEN if status == "OK" else RED
        print(f"    {colour}[{i:>3}] {seg_name}  "
              f"({start_sec:.2f}s → {end_sec:.2f}s, {duration:.2f}s){RESET}")

    return records


def assign_video_id(video_path: Path, index: int) -> str:
    """
    Generate a zero-padded video ID string.

    Args:
        video_path: Not used directly; kept for future name-based IDs.
        index:      1-based position of the video in the scan order.

    Returns:
        String like 'video001'.
    """
    return f"video{index:03d}"


def run(raw_dir: Path = RAW_VIDEOS_DIR,
        seg_dir: Path = SEGMENTS_DIR) -> None:
    """
    Main segmentation pipeline.

    Iterates over all videos, detects scenes, splits clips,
    and writes scenes_manifest.csv.
    """
    if not raw_dir.exists():
        print(f"{RED}ERROR: RAW_VIDEOS_DIR '{raw_dir}' does not exist.{RESET}")
        sys.exit(1)

    seg_dir.mkdir(parents=True, exist_ok=True)
    videos = scan_videos(raw_dir)

    if not videos:
        print(f"{YELLOW}No videos found. Exiting.{RESET}")
        return

    all_records: list[dict] = []

    for idx, vp in enumerate(videos, 1):
        vid_id = assign_video_id(vp, idx)
        print(f"\n{BOLD}[{idx}/{len(videos)}] {vp.name}  →  {vid_id}{RESET}")

        scenes = detect_scenes(vp)
        print(f"  Detected {len(scenes)} scene(s)")

        if not scenes:
            print(f"  {YELLOW}No scenes detected — skipping.{RESET}")
            continue

        records = split_scenes(vp, scenes, vid_id, seg_dir / vid_id)
        all_records.extend(records)

    # ── Write manifest CSV ───────────────────────────────────
    if all_records:
        manifest_path = seg_dir / "scenes_manifest.csv"
        pd.DataFrame(all_records).to_csv(manifest_path, index=False)
        print(f"\n{BOLD}{GREEN}✔ Manifest saved: {manifest_path}{RESET}")

    total_segs = len(all_records)
    ok_segs    = sum(1 for r in all_records if r["status"] == "OK")
    avg_dur    = (sum(r["duration_sec"] for r in all_records) / total_segs
                  if total_segs else 0)

    print(f"\n{BOLD}{CYAN}{'─'*50}")
    print(f"  DONE — {ok_segs}/{total_segs} segments saved")
    print(f"  Average duration : {avg_dur:.2f}s")
    print(f"  Output directory : {seg_dir}")
    print(f"{'─'*50}{RESET}")


if __name__ == "__main__":
    run()
