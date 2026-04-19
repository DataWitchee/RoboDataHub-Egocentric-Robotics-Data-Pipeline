"""
=============================================================
SCRIPT 3: Data Quality Checker
=============================================================
Usage:
    python quality_checker.py /path/to/video/folder

Flags:
    ✗  Videos shorter than 10 seconds
    ✗  Videos with height below 480p
    ✗  Duplicate filenames (same name, possibly different paths)
    ✗  Corrupted or unreadable files

Outputs:
    - Colour-coded terminal report
    - quality_report.csv saved in the video folder
=============================================================
"""

import sys
import cv2
import pandas as pd
from pathlib import Path
from collections import Counter
from datetime import timedelta


# ─── Thresholds ────────────────────────────────────────────
MIN_DURATION_SEC   = 10     # flag if shorter than this
MIN_HEIGHT_PX      = 480    # flag if vertical res is below this


# ─── ANSI colours ──────────────────────────────────────────
RED    = "\033[91m"
GREEN  = "\033[92m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
DIM    = "\033[2m"
RESET  = "\033[0m"

OK   = f"{GREEN}✔  OK{RESET}"
FAIL = f"{RED}✗  FAIL{RESET}"
WARN = f"{YELLOW}⚠  WARN{RESET}"


# ─── Helpers ───────────────────────────────────────────────
def scan_videos(root_folder: str) -> list[Path]:
    """Recursively find all .mp4 and .avi files."""
    root = Path(root_folder)
    if not root.exists():
        print(f"{RED}ERROR: '{root_folder}' does not exist.{RESET}")
        sys.exit(1)
    return sorted([
        p for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() in {".mp4", ".avi"}
    ])


def check_video(path: Path) -> dict:
    """
    Run all quality checks on a single video file.

    Returns a dict with per-check boolean flags and raw metadata.
    """
    result = {
        "filename":      path.name,
        "filepath":      str(path),
        "file_size_mb":  round(path.stat().st_size / (1024 ** 2), 2),
        # raw values
        "duration_sec":  None,
        "fps":           None,
        "width":         None,
        "height":        None,
        "resolution":    None,
        # quality flags (True = passes check, False = flagged)
        "readable":      False,
        "long_enough":   False,
        "high_enough":   False,
        "is_duplicate":  False,   # filled in later with global scan
        # human-readable issue summary
        "issues":        [],
    }

    cap = cv2.VideoCapture(str(path))

    # ── Check 1: Is the file readable? ─────────────────────
    if not cap.isOpened():
        result["issues"].append("Corrupted/unreadable")
        cap.release()
        return result

    fps          = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration_sec = total_frames / fps if fps > 0 else 0

    result.update({
        "readable":     True,
        "duration_sec": round(duration_sec, 2),
        "fps":          round(fps, 2),
        "width":        width,
        "height":       height,
        "resolution":   f"{width}x{height}",
    })
    cap.release()

    # ── Check 2: Duration ≥ MIN_DURATION_SEC ───────────────
    if duration_sec >= MIN_DURATION_SEC:
        result["long_enough"] = True
    else:
        result["issues"].append(
            f"Too short ({duration_sec:.1f}s < {MIN_DURATION_SEC}s)"
        )

    # ── Check 3: Resolution ≥ 480p ──────────────────────────
    if height >= MIN_HEIGHT_PX:
        result["high_enough"] = True
    else:
        result["issues"].append(
            f"Low resolution ({height}p < {MIN_HEIGHT_PX}p)"
        )

    return result


def flag_duplicates(records: list[dict]) -> list[dict]:
    """
    Mark records whose filenames appear more than once.

    Mutates each dict in-place and returns the list.
    """
    name_counts = Counter(r["filename"] for r in records)
    for r in records:
        if name_counts[r["filename"]] > 1:
            r["is_duplicate"] = True
            r["issues"].append("Duplicate filename")
    return records


def print_report(records: list[dict]) -> None:
    """Print a colour-coded quality report to the terminal."""

    total    = len(records)
    flagged  = [r for r in records if r["issues"]]
    clean    = [r for r in records if not r["issues"]]

    print(f"\n{BOLD}{CYAN}{'═'*70}")
    print("  DATA QUALITY REPORT — Egocentric Video Dataset")
    print(f"{'═'*70}{RESET}\n")

    # ── Per-video results ───────────────────────────────────
    for r in records:
        status = FAIL if r["issues"] else OK
        print(f"  {status}  {r['filename']}")
        if r["readable"]:
            print(
                f"       {DIM}{r['resolution']} | "
                f"{r['fps']} FPS | "
                f"{r['duration_sec']}s | "
                f"{r['file_size_mb']} MB{RESET}"
            )
        for issue in r["issues"]:
            print(f"       {RED}→ {issue}{RESET}")

    # ── Summary banner ──────────────────────────────────────
    print(f"\n{BOLD}{'─'*70}{RESET}")
    print(f"  Total videos checked : {total}")
    print(f"  {GREEN}Clean                : {len(clean)}{RESET}")
    print(f"  {RED}Flagged              : {len(flagged)}{RESET}")

    # Break down issue types
    type_counts: dict[str, int] = {}
    for r in flagged:
        for issue in r["issues"]:
            key = issue.split("(")[0].strip()   # normalise the key
            type_counts[key] = type_counts.get(key, 0) + 1

    if type_counts:
        print(f"\n  {BOLD}Issue breakdown:{RESET}")
        for issue_type, count in sorted(type_counts.items(), key=lambda x: -x[1]):
            print(f"    • {issue_type}: {count}")

    print(f"{BOLD}{'─'*70}{RESET}\n")


def save_report(records: list[dict], root_folder: str) -> None:
    """
    Save a flat quality report CSV to root_folder/quality_report.csv.

    Converts the 'issues' list column to a pipe-separated string for CSV.
    """
    df = pd.DataFrame(records)
    df["issues"] = df["issues"].apply(lambda x: " | ".join(x) if x else "None")

    # Re-order columns for readability
    cols = [
        "filename", "filepath", "readable",
        "resolution", "fps", "duration_sec", "file_size_mb",
        "long_enough", "high_enough", "is_duplicate", "issues",
    ]
    df = df[[c for c in cols if c in df.columns]]
    out = Path(root_folder) / "quality_report.csv"
    df.to_csv(out, index=False)
    print(f"{BOLD}{GREEN}✔ Quality report saved to: {out}{RESET}")


# ─── MAIN ──────────────────────────────────────────────────
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"{YELLOW}Usage: python quality_checker.py <video_folder>{RESET}")
        sys.exit(1)

    video_folder = sys.argv[1]
    videos = scan_videos(video_folder)

    if not videos:
        print(f"{YELLOW}No .mp4 or .avi files found. Exiting.{RESET}")
        sys.exit(0)

    print(f"{CYAN}Checking {len(videos)} video(s)…{RESET}")

    records = [check_video(vp) for vp in videos]
    records = flag_duplicates(records)

    print_report(records)
    save_report(records, video_folder)
