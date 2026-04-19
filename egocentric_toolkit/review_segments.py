"""
=============================================================
SCRIPT 4: Segment Visual Review — Annotated Thumbnail Grids
=============================================================
Usage:
    python review_segments.py

What it does:
    • Reads /dataset/segments/segments.json
    • For each video, opens the corresponding raw video file
    • Extracts the middle frame of every segment
    • Overlays: action label, duration, confidence, segment ID
    • Saves one grid PNG per video to /dataset/previews/segments/
    • Display resolution of each thumbnail is configurable below

Dependencies:
    pip install opencv-python matplotlib

Note:
    The script looks for the raw video by matching 'video_id'
    (video001, video002 …) to the alphabetically-sorted list of
    videos in RAW_VIDEOS_DIR.  If your naming differs, adjust
    the build_video_map() function.
=============================================================
"""

import json
import sys
import math
from pathlib import Path

import cv2
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec

# ─── CONFIG ────────────────────────────────────────────────
RAW_VIDEOS_DIR   = Path("/Users/mannatsaini/Desktop/my_robotics_data/raw_videos")
SEGMENTS_JSON    = Path("/Users/mannatsaini/Desktop/my_robotics_data/segments/segments.json")
PREVIEW_OUT_DIR  = Path("/Users/mannatsaini/Desktop/my_robotics_data/previews/segments")

# Thumbnail dimensions (pixels) inside the grid
THUMB_W = 320
THUMB_H = 180

# Maximum thumbnails per row in the grid
GRID_COLS = 4

# Overlay text style
FONT_SCALE    = 0.55
FONT_THICKNESS= 1
LABEL_COLOR   = (255, 255, 255)   # white text
BAR_COLOR     = (15, 15, 50)      # dark navy bar behind text
# ───────────────────────────────────────────────────────────


# ─── Action → colour mapping for label bars ────────────────
LABEL_COLORS: dict[str, tuple[int, int, int]] = {
    "chopping":   (220,  60,  60),
    "stirring":   (230, 160,  30),
    "pouring":    ( 60, 130, 220),
    "boiling":    (200,  80, 200),
    "wiping":     ( 50, 180, 120),
    "washing":    ( 40, 160, 200),
    "scrubbing":  (180, 100,  40),
    "picking_up": ( 90, 180,  50),
    "putting_away":( 70, 100, 200),
    "idle":       (100, 100, 100),
}
DEFAULT_COLOR = (130, 130, 130)


# ─── Helpers ───────────────────────────────────────────────
def build_video_map(raw_dir: Path) -> dict[str, Path]:
    """
    Build a mapping from video_id string (e.g. 'video001') to
    the actual file path, by sorting all found videos alphabetically.

    Args:
        raw_dir: Root directory containing raw videos.

    Returns:
        Dict like {'video001': Path('/dataset/raw_videos/…/foo.mp4'), …}
    """
    exts   = {".mp4", ".avi"}
    videos = sorted(
        p for p in raw_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in exts
    )
    return {f"video{i:03d}": vp for i, vp in enumerate(videos, 1)}


def load_segments_json(json_path: Path) -> list[dict]:
    """Load and return the segments.json content."""
    if not json_path.exists():
        print(f"ERROR: segments.json not found at '{json_path}'")
        sys.exit(1)
    with open(json_path) as f:
        return json.load(f)


def extract_middle_frame(
    video_path: Path,
    start_frame: int,
    end_frame: int,
) -> np.ndarray | None:
    """
    Open the video and read the frame exactly halfway between
    start_frame and end_frame.

    Args:
        video_path:  Path to the source video.
        start_frame: Segment start frame index.
        end_frame:   Segment end frame index.

    Returns:
        BGR uint8 ndarray, or None if the seek fails.
    """
    mid_frame = (start_frame + end_frame) // 2
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None

    cap.set(cv2.CAP_PROP_POS_FRAMES, mid_frame)
    ret, frame = cap.read()
    cap.release()

    return frame if ret else None


def annotate_thumbnail(
    frame: np.ndarray,
    segment_id: str,
    action_label: str,
    duration: float,
    confidence: float | None,
) -> np.ndarray:
    """
    Resize the frame to THUMB_W × THUMB_H and overlay a coloured
    annotation bar at the bottom with label, duration, confidence.

    Args:
        frame:       BGR image array from OpenCV.
        segment_id:  e.g. 'video001_seg003'.
        action_label: Human-readable action name.
        duration:    Segment duration in seconds.
        confidence:  CLIP confidence score (may be None).

    Returns:
        Annotated BGR uint8 thumbnail.
    """
    thumb = cv2.resize(frame, (THUMB_W, THUMB_H), interpolation=cv2.INTER_AREA)

    # ── Coloured header bar (top) ─────────────────────────
    bar_color = LABEL_COLORS.get(action_label, DEFAULT_COLOR)
    bar_h     = 22
    cv2.rectangle(thumb, (0, 0), (THUMB_W, bar_h), bar_color, -1)

    # Action label in header
    cv2.putText(
        thumb, action_label.upper(),
        (6, bar_h - 6),
        cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE,
        (255, 255, 255), FONT_THICKNESS, cv2.LINE_AA,
    )

    # ── Footer bar (bottom) ───────────────────────────────
    footer_h = 20
    y0 = THUMB_H - footer_h
    overlay = thumb.copy()
    cv2.rectangle(overlay, (0, y0), (THUMB_W, THUMB_H), BAR_COLOR, -1)
    cv2.addWeighted(overlay, 0.75, thumb, 0.25, 0, thumb)

    conf_str = f"conf={confidence:.2f}" if confidence is not None else ""
    footer_text = f"{segment_id}  {duration:.1f}s  {conf_str}"
    cv2.putText(
        thumb, footer_text,
        (4, THUMB_H - 5),
        cv2.FONT_HERSHEY_SIMPLEX, 0.38,
        LABEL_COLOR, 1, cv2.LINE_AA,
    )

    return thumb


def build_grid(thumbnails: list[np.ndarray], video_id: str) -> np.ndarray:
    """
    Arrange a list of BGR thumbnails into a grid image.
    Missing cells in the last row are filled with a dark placeholder.

    Args:
        thumbnails: List of annotated BGR thumbnails.
        video_id:   Used as the grid title.

    Returns:
        Single BGR uint8 image containing the full grid.
    """
    n_cols  = GRID_COLS
    n_rows  = math.ceil(len(thumbnails) / n_cols)

    # Pad list so every row is full
    placeholder = np.zeros((THUMB_H, THUMB_W, 3), dtype=np.uint8)
    placeholder[:] = (20, 20, 30)   # dark navy

    padded = thumbnails + [placeholder] * (n_rows * n_cols - len(thumbnails))

    rows = []
    for r in range(n_rows):
        row_imgs = [padded[r * n_cols + c] for c in range(n_cols)]
        rows.append(np.hstack(row_imgs))

    grid = np.vstack(rows)

    # ── Title bar on top ──────────────────────────────────
    title_h = 36
    title_bar = np.zeros((title_h, grid.shape[1], 3), dtype=np.uint8)
    title_bar[:] = (10, 10, 26)

    title_text = f"  {video_id}  —  {len(thumbnails)} segments"
    cv2.putText(
        title_bar, title_text,
        (12, title_h - 10),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7,
        (160, 200, 255), 1, cv2.LINE_AA,
    )
    return np.vstack([title_bar, grid])


def save_grid(grid: np.ndarray, video_id: str, out_dir: Path) -> None:
    """Save the grid image as a PNG file."""
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{video_id}_segment_review.png"
    cv2.imwrite(str(out_path), grid)
    print(f"  Saved grid → {out_path}")


# ─── MAIN ──────────────────────────────────────────────────
def run(
    raw_dir:   Path = RAW_VIDEOS_DIR,
    json_path: Path = SEGMENTS_JSON,
    out_dir:   Path = PREVIEW_OUT_DIR,
) -> None:
    """
    For each video in segments.json, build and save a thumbnail
    grid PNG to /dataset/previews/segments/.
    """
    video_map    = build_video_map(raw_dir)
    all_entries  = load_segments_json(json_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Found {len(all_entries)} video entries in segments.json\n")

    for entry in all_entries:
        video_id  = entry["video_id"]
        segments  = entry.get("segments", [])
        video_path = video_map.get(video_id)

        print(f"[{video_id}]  {len(segments)} segment(s)", end="")

        if not video_path:
            print(f"  — raw video not found in '{raw_dir}', skipping")
            continue

        print(f"  ←  {video_path.name}")
        thumbnails: list[np.ndarray] = []

        for seg in segments:
            seg_id  = seg["segment_id"]
            label   = seg.get("action_label", "unknown")
            start_f = seg.get("start_frame", 0)
            end_f   = seg.get("end_frame", start_f + 1)
            start_t = seg.get("start_time", 0.0)
            end_t   = seg.get("end_time",   0.0)
            conf    = seg.get("confidence")
            duration= round(end_t - start_t, 2)

            frame = extract_middle_frame(video_path, start_f, end_f)

            if frame is None:
                # Use dark placeholder if frame extraction fails
                frame = np.zeros((THUMB_H * 4, THUMB_W * 4, 3), dtype=np.uint8)
                frame[:] = (30, 10, 10)
                cv2.putText(
                    frame, "UNREADABLE",
                    (60, THUMB_H * 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5,
                    (60, 60, 180), 2,
                )

            thumb = annotate_thumbnail(frame, seg_id, label, duration, conf)
            thumbnails.append(thumb)

        if thumbnails:
            grid = build_grid(thumbnails, video_id)
            save_grid(grid, video_id, out_dir)
        else:
            print(f"  No thumbnails generated for {video_id}")

    print(f"\nAll grids saved to: {out_dir}")


if __name__ == "__main__":
    run()
