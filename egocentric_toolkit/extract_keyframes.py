"""
=============================================================
SCRIPT 2: Keyframe Extractor + Thumbnail Grid Viewer
=============================================================
Usage:
    python extract_keyframes.py /path/to/video/folder

What it does:
    - Extracts 5 evenly-spaced keyframes per video
    - Saves them as JPGs in <video_folder>/previews/<video_stem>/
    - Displays a grid of thumbnails (matplotlib) for quick review
=============================================================
"""

import os
import sys
import cv2
import math
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path


# ─── Config ────────────────────────────────────────────────
KEYFRAMES_PER_VIDEO = 5          # number of sampled frames
THUMB_WIDTH         = 320        # resize width for grid display
THUMB_HEIGHT        = 180        # resize height for grid display
JPEG_QUALITY        = 90         # 0-100, higher = larger file
PREVIEWS_SUBDIR     = "previews" # sub-folder name inside video root


# ─── Helpers ───────────────────────────────────────────────
def scan_videos(root_folder: str) -> list[Path]:
    """Recursively find all .mp4 and .avi files under root_folder."""
    root = Path(root_folder)
    if not root.exists():
        print(f"ERROR: Folder '{root_folder}' does not exist.")
        sys.exit(1)

    return sorted([
        p for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() in {".mp4", ".avi"}
    ])


def extract_keyframes(
    video_path: Path,
    output_dir: Path,
    n_frames: int = KEYFRAMES_PER_VIDEO,
) -> list[tuple[int, np.ndarray]]:
    """
    Sample n_frames evenly spaced keyframes from a video and save
    them as JPGs.

    Args:
        video_path: Path to the source video.
        output_dir: Directory where JPGs will be written.
        n_frames:   Number of frames to sample.

    Returns:
        List of (frame_index, BGR np.ndarray) tuples for the
        thumbnail grid renderer.  Returns [] on failure.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"  ⚠  Cannot open: {video_path.name}")
        return []

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Guard against very short videos (fewer frames than requested)
    n = min(n_frames, total_frames)

    # Evenly-spaced indices across the whole clip
    # e.g. for 5 frames in a 100-frame video → [10, 30, 50, 70, 90]
    indices = [int(i * total_frames / n) for i in range(n)]
    # Clamp to valid range
    indices = [min(idx, total_frames - 1) for idx in indices]

    output_dir.mkdir(parents=True, exist_ok=True)
    extracted = []

    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue

        # Build output path: previews/<video_stem>/frame_<idx>.jpg
        out_file = output_dir / f"frame_{idx:06d}.jpg"
        cv2.imwrite(
            str(out_file),
            frame,
            [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY],
        )
        extracted.append((idx, frame))

    cap.release()
    return extracted


def make_thumbnail(frame: np.ndarray) -> np.ndarray:
    """
    Resize a BGR frame to the thumbnail dimensions and convert to RGB
    for matplotlib.

    Args:
        frame: OpenCV BGR image array.

    Returns:
        RGB uint8 array of shape (THUMB_HEIGHT, THUMB_WIDTH, 3).
    """
    thumb = cv2.resize(frame, (THUMB_WIDTH, THUMB_HEIGHT),
                       interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(thumb, cv2.COLOR_BGR2RGB)


def display_thumbnail_grid(
    all_frames: list[tuple[str, list[tuple[int, np.ndarray]]]],
) -> None:
    """
    Display a matplotlib grid of thumbnails.

    Layout:  one row per video, one column per keyframe.
    The video filename is shown as the row label.

    Args:
        all_frames: List of (video_name, [(frame_idx, frame), ...]).
    """
    n_videos  = len(all_frames)
    n_cols    = KEYFRAMES_PER_VIDEO

    if n_videos == 0:
        print("No frames to display.")
        return

    fig_width  = n_cols * 3.0
    fig_height = n_videos * 2.0 + 0.5   # +0.5 for the top title

    fig = plt.figure(figsize=(fig_width, fig_height), facecolor="#1a1a2e")
    fig.suptitle(
        "Egocentric Dataset — Keyframe Previews",
        fontsize=14, fontweight="bold", color="white", y=0.98,
    )

    gs = gridspec.GridSpec(
        n_videos, n_cols,
        figure=fig,
        hspace=0.45, wspace=0.05,
        left=0.12,             # leave space for row labels
    )

    for row_idx, (video_name, frames) in enumerate(all_frames):
        for col_idx in range(n_cols):
            ax = fig.add_subplot(gs[row_idx, col_idx])
            ax.set_facecolor("#0f0f1a")

            if col_idx < len(frames):
                frame_idx, frame_bgr = frames[col_idx]
                thumb = make_thumbnail(frame_bgr)
                ax.imshow(thumb)
                ax.set_title(
                    f"f{frame_idx}",
                    fontsize=7, color="#aaaacc", pad=2,
                )
            else:
                # Filler for videos with fewer frames than n_cols
                ax.text(
                    0.5, 0.5, "—",
                    ha="center", va="center",
                    color="#555577", transform=ax.transAxes,
                )

            ax.axis("off")

            # Row label on the leftmost cell only
            if col_idx == 0:
                ax.set_ylabel(
                    video_name[:28] + ("…" if len(video_name) > 28 else ""),
                    fontsize=7, color="#ccccee", rotation=0,
                    labelpad=4, ha="right", va="center",
                )
                ax.yaxis.set_label_coords(-0.02, 0.5)

    plt.savefig(
        Path(sys.argv[1]) / "previews" / "thumbnail_grid.png",
        dpi=120, bbox_inches="tight", facecolor=fig.get_facecolor(),
    )
    print("\n✔ Thumbnail grid saved to previews/thumbnail_grid.png")
    plt.show()


# ─── MAIN ──────────────────────────────────────────────────
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python extract_keyframes.py <video_folder>")
        sys.exit(1)

    video_folder = sys.argv[1]
    previews_root = Path(video_folder) / PREVIEWS_SUBDIR

    videos = scan_videos(video_folder)

    if not videos:
        print("No .mp4 or .avi files found.")
        sys.exit(0)

    print(f"Processing {len(videos)} video(s) → keyframes saved in '{previews_root}'\n")

    all_frames: list[tuple[str, list]] = []

    for i, vp in enumerate(videos, 1):
        out_dir = previews_root / vp.stem
        print(f"[{i:>3}/{len(videos)}] {vp.name}")
        frames = extract_keyframes(vp, out_dir, KEYFRAMES_PER_VIDEO)
        print(f"        → {len(frames)} frame(s) saved to {out_dir}")
        all_frames.append((vp.name, frames))

    print(f"\nBuilding thumbnail grid for {len(all_frames)} video(s)…")
    display_thumbnail_grid(all_frames)
