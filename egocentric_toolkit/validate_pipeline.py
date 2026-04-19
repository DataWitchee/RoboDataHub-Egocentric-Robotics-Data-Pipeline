"""
=============================================================
SCRIPT 4: Final Pipeline Validation
=============================================================
Usage:
    python validate_pipeline.py

What it does:
    • Loads master_dataset.json
    • Per segment checks:
        - Video file exists on disk
        - Action label is set (not 'unknown')
        - At least 1 object annotation present
        - NL description is non-empty
    • Computes a completeness score (0–100%) per segment
    • Prints a final summary banner
    • Saves validation_report.csv to /dataset/final_dataset/

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
FINAL_DATASET_DIR = Path("/Users/mannatsaini/Desktop/my_robotics_data/final_dataset")
MASTER_JSON       = FINAL_DATASET_DIR / "master_dataset.json"
DATASET_ROOT      = Path("/Users/mannatsaini/Desktop/my_robotics_data")     # used to resolve video_path
# ───────────────────────────────────────────────────────────

CYAN  = "\033[96m"
GREEN = "\033[92m"
YELLOW= "\033[93m"
RED   = "\033[91m"
BOLD  = "\033[1m"
DIM   = "\033[2m"
RESET = "\033[0m"


# ─── Checks ────────────────────────────────────────────────

CHECKS = {
    "has_video":       "Video file exists on disk",
    "has_action":      "Action label is set & not 'unknown'",
    "has_objects":     "At least 1 object annotation",
    "has_description": "Non-empty NL description",
}


def check_record(record: dict) -> dict[str, bool]:
    """
    Run the four completeness checks on a single record.

    Returns a dict of {check_name: bool}.
    """
    video_path  = DATASET_ROOT / record.get("video_path", "")
    action      = record.get("action_label", "unknown")
    objects     = record.get("objects_present", [])
    description = record.get("nl_description", "").strip()

    bbox_frame_count = sum(
        len(f.get("objects", []))
        for f in record.get("bbox_annotations", [])
    )
    has_objects = len(objects) > 0 or bbox_frame_count > 0

    return {
        "has_video":       video_path.exists() or record.get("video_exists", False),
        "has_action":      bool(action) and action.lower() != "unknown",
        "has_objects":     has_objects,
        "has_description": len(description) >= 10,
    }


def completeness_score(checks: dict[str, bool]) -> int:
    """Return a 0-100 completeness score (25 pts per passed check)."""
    return int(sum(checks.values()) / len(checks) * 100)


# ─── Main ──────────────────────────────────────────────────

def run() -> None:
    if not MASTER_JSON.exists():
        print(f"{RED}ERROR: {MASTER_JSON} not found. Run build_dataset.py first.{RESET}")
        sys.exit(1)

    with open(MASTER_JSON) as f:
        records = json.load(f)

    rows = []
    issue_counters = {k: 0 for k in CHECKS}
    split_counter  = Counter()

    for record in records:
        sid    = record["segment_id"]
        checks = check_record(record)
        score  = completeness_score(checks)
        issues = [CHECKS[k] for k, v in checks.items() if not v]

        split_counter[record.get("split", "unknown")] += 1
        for k, v in checks.items():
            if not v:
                issue_counters[k] += 1

        rows.append({
            "segment_id":       sid,
            "action_label":     record.get("action_label", ""),
            "split":            record.get("split", ""),
            "duration_sec":     record.get("duration", 0.0),
            "completeness_pct": score,
            **{f"check_{k}": v for k, v in checks.items()},
            "issues": " | ".join(issues) if issues else "None",
        })

    df = pd.DataFrame(rows)

    # ── Save validation report ─────────────────────────────
    out_csv = FINAL_DATASET_DIR / "validation_report.csv"
    df.to_csv(out_csv, index=False)

    # ── Derived summary metrics ────────────────────────────
    total        = len(df)
    fully_ok     = int((df["completeness_pct"] == 100).sum())
    pct_ok       = fully_ok / total * 100 if total else 0
    avg_duration = df["duration_sec"].mean()
    n_actions    = df["action_label"].nunique()
    flagged      = df[df["completeness_pct"] < 100]

    # ── Summary banner ─────────────────────────────────────
    dataset_ready = "YES ✓" if pct_ok >= 90 else "NO — review flagged records"
    ready_colour  = GREEN if pct_ok >= 90 else RED

    print(f"\n{BOLD}{CYAN}{'═'*52}")
    print("  FINAL PIPELINE VALIDATION SUMMARY")
    print(f"{'═'*52}{RESET}\n")

    print(f"  Total segments        : {BOLD}{total}{RESET}")
    print(f"  Fully complete        : {GREEN}{fully_ok} ({pct_ok:.1f}%){RESET}")

    for k, label in CHECKS.items():
        bad = issue_counters[k]
        colour = RED if bad > 0 else GREEN
        short  = label.split("(")[0].strip()
        print(f"  Missing {short:<20}: {colour}{bad}{RESET}")

    print(f"\n  Train / Val / Test    : "
          f"{split_counter.get('train', 0)} / "
          f"{split_counter.get('val',   0)} / "
          f"{split_counter.get('test',  0)}")
    print(f"  Action categories     : {n_actions}")
    print(f"  Avg segment duration  : {avg_duration:.1f}s")
    print(f"\n  {BOLD}Dataset ready         : "
          f"{ready_colour}{dataset_ready}{RESET}")

    if len(flagged) > 0:
        print(f"\n{YELLOW}  ─── Flagged Records (first 10) ───{RESET}")
        for _, row in flagged.head(10).iterrows():
            print(f"    [{row['completeness_pct']:>3}%] {row['segment_id']:<24}"
                  f"  {DIM}{row['issues']}{RESET}")
        if len(flagged) > 10:
            print(f"    … and {len(flagged) - 10} more (see validation_report.csv)")

    print(f"\n{GREEN}{BOLD}  ✔ Validation report saved → {out_csv}{RESET}\n")


if __name__ == "__main__":
    run()
