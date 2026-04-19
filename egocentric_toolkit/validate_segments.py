"""
=============================================================
SCRIPT 3: Segment Validation & Statistics
=============================================================
Usage:
    python validate_segments.py

What it does:
    • Loads /dataset/segments/segments.json
    • Checks every segment for:
        - Too short  : duration < MIN_DURATION_SEC (default 1 s)
        - Too long   : duration > MAX_DURATION_SEC (default 30 s)
        - Overlapping: segments whose time ranges intersect
    • Prints a coloured terminal report
    • Saves /dataset/segments/segments_summary.csv
    • Prints label distribution and aggregate statistics

Dependencies:
    pip install pandas
=============================================================
"""

import json
import sys
from pathlib import Path
from collections import Counter

import pandas as pd

# ─── CONFIG ────────────────────────────────────────────────
SEGMENTS_JSON   = Path("/Users/mannatsaini/Desktop/my_robotics_data/segments/segments.json")
SEGMENTS_DIR    = Path("/Users/mannatsaini/Desktop/my_robotics_data/segments")

MIN_DURATION_SEC = 1.0    # segments shorter than this are flagged
MAX_DURATION_SEC = 30.0   # segments longer  than this are flagged
# ───────────────────────────────────────────────────────────


# ─── ANSI colours ──────────────────────────────────────────
CYAN  = "\033[96m"
GREEN = "\033[92m"
YELLOW= "\033[93m"
RED   = "\033[91m"
BOLD  = "\033[1m"
DIM   = "\033[2m"
RESET = "\033[0m"

PASS = f"{GREEN}PASS{RESET}"
FAIL = f"{RED}FAIL{RESET}"
WARN = f"{YELLOW}WARN{RESET}"


# ─── Helpers ───────────────────────────────────────────────
def load_segments(json_path: Path) -> list[dict]:
    """
    Load segments.json and flatten into a list of segment dicts.

    Each dict gets a 'video_id' field injected for traceability.

    Returns:
        Flat list of segment dicts with 'video_id' added.
    """
    if not json_path.exists():
        print(f"{RED}ERROR: segments.json not found at '{json_path}'{RESET}")
        sys.exit(1)

    with open(json_path) as f:
        data = json.load(f)

    flat = []
    for video_entry in data:
        vid = video_entry.get("video_id", "unknown")
        for seg in video_entry.get("segments", []):
            seg = seg.copy()
            seg["video_id"] = vid         # inject for traceability
            flat.append(seg)

    print(f"{CYAN}Loaded {len(flat)} segment(s) from {len(data)} video(s){RESET}\n")
    return flat


def compute_duration(seg: dict) -> float:
    """Compute duration in seconds from start_time / end_time."""
    return round(seg["end_time"] - seg["start_time"], 3)


def detect_overlaps(segments_for_video: list[dict]) -> list[str]:
    """
    Find segment IDs that overlap with any subsequent segment
    (within the same video).

    Two segments [a_start, a_end) and [b_start, b_end) overlap iff
        a_start < b_end  AND  b_start < a_end

    Args:
        segments_for_video: Segments belonging to one video,
                            sorted by start_time.

    Returns:
        List of segment_id strings that participate in an overlap.
    """
    flagged = set()
    n = len(segments_for_video)
    for i in range(n):
        for j in range(i + 1, n):
            a = segments_for_video[i]
            b = segments_for_video[j]
            if a["start_time"] < b["end_time"] and b["start_time"] < a["end_time"]:
                flagged.add(a["segment_id"])
                flagged.add(b["segment_id"])
    return list(flagged)


def validate(flat_segments: list[dict]) -> pd.DataFrame:
    """
    Run all quality checks and return an annotated DataFrame.

    Columns added:
        duration_sec, too_short, too_long, overlapping, issues
    """
    rows = []

    # Group by video for overlap detection
    by_video: dict[str, list[dict]] = {}
    for seg in flat_segments:
        by_video.setdefault(seg["video_id"], []).append(seg)

    # Detect overlaps per video
    overlap_ids: set[str] = set()
    for vid_segs in by_video.values():
        sorted_segs = sorted(vid_segs, key=lambda s: s["start_time"])
        overlap_ids.update(detect_overlaps(sorted_segs))

    # Build per-segment result rows
    for seg in flat_segments:
        dur        = compute_duration(seg)
        too_short  = dur < MIN_DURATION_SEC
        too_long   = dur > MAX_DURATION_SEC
        overlapping= seg["segment_id"] in overlap_ids

        issues = []
        if too_short:
            issues.append(f"too short ({dur:.2f}s < {MIN_DURATION_SEC}s)")
        if too_long:
            issues.append(f"too long ({dur:.2f}s > {MAX_DURATION_SEC}s)")
        if overlapping:
            issues.append("overlaps with another segment")

        rows.append({
            "video_id":    seg["video_id"],
            "segment_id":  seg["segment_id"],
            "action_label":seg.get("action_label", "unknown"),
            "start_time":  seg["start_time"],
            "end_time":    seg["end_time"],
            "duration_sec":dur,
            "start_frame": seg.get("start_frame"),
            "end_frame":   seg.get("end_frame"),
            "confidence":  seg.get("confidence"),
            "too_short":   too_short,
            "too_long":    too_long,
            "overlapping": overlapping,
            "issues":      " | ".join(issues) if issues else "None",
        })

    return pd.DataFrame(rows)


def print_report(df: pd.DataFrame) -> None:
    """
    Print a detailed, colour-coded validation report to the terminal.
    """
    n_total     = len(df)
    n_short     = df["too_short"].sum()
    n_long      = df["too_long"].sum()
    n_overlap   = df["overlapping"].sum()
    n_flagged   = (df["issues"] != "None").sum()
    n_clean     = n_total - n_flagged

    # ── Per-segment section ─────────────────────────────────
    print(f"\n{BOLD}{CYAN}{'═'*72}")
    print("  SEGMENT VALIDATION REPORT")
    print(f"{'═'*72}{RESET}\n")

    for _, row in df.iterrows():
        ok = row["issues"] == "None"
        status_str = PASS if ok else FAIL
        conf_str   = f"conf={row['confidence']:.2f}" if row["confidence"] is not None else ""
        print(
            f"  {status_str}  {row['segment_id']:<22}"
            f"  {row['action_label']:<14}"
            f"  {row['duration_sec']:>6.2f}s"
            f"  {conf_str}"
        )
        if not ok:
            for issue in row["issues"].split(" | "):
                print(f"           {RED}→ {issue}{RESET}")

    # ── Summary banner ──────────────────────────────────────
    avg_dur = df["duration_sec"].mean()
    med_dur = df["duration_sec"].median()
    total_t = df["duration_sec"].sum()

    print(f"\n{BOLD}{'─'*72}{RESET}")
    print(f"\n  {BOLD}STATISTICS{RESET}")
    print(f"  Total segments         : {n_total}")
    print(f"  {GREEN}Clean segments         : {n_clean}{RESET}")
    print(f"  {RED}Flagged segments       : {n_flagged}{RESET}")
    print(f"    • Too short (<{MIN_DURATION_SEC}s)  : {n_short}")
    print(f"    • Too long  (>{MAX_DURATION_SEC}s) : {n_long}")
    print(f"    • Overlapping          : {n_overlap}")
    print(f"  Average duration        : {avg_dur:.2f}s")
    print(f"  Median  duration        : {med_dur:.2f}s")
    print(f"  Total content duration  : {total_t:.1f}s ({total_t/60:.1f} min)")

    # ── Label distribution ──────────────────────────────────
    print(f"\n  {BOLD}ACTION LABEL DISTRIBUTION{RESET}")
    label_counts = df["action_label"].value_counts()
    for label, count in label_counts.items():
        bar   = "█" * int(count / max(label_counts) * 30)
        pct   = count / n_total * 100
        print(f"    {label:<18} {count:>4}  {bar:<30}  {pct:.1f}%")

    print(f"\n{BOLD}{'─'*72}{RESET}\n")


def save_summary(df: pd.DataFrame, out_dir: Path) -> None:
    """Save the annotated DataFrame as segments_summary.csv."""
    out_path = out_dir / "segments_summary.csv"
    df.to_csv(out_path, index=False)
    print(f"{BOLD}{GREEN}✔ Summary saved: {out_path}{RESET}")


# ─── MAIN ──────────────────────────────────────────────────
if __name__ == "__main__":
    flat_segments = load_segments(SEGMENTS_JSON)
    df = validate(flat_segments)
    print_report(df)
    save_summary(df, SEGMENTS_DIR)
