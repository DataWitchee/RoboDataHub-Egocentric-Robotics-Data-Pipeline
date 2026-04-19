"""
=============================================================
SCRIPT 3: Description Quality Checker
=============================================================
Usage:
    python check_descriptions.py

What it does:
    • Loads nl_descriptions.json 
    • Flags short descriptions (< 10 words).
    • Flags entirely empty descriptions.
    • Computes API vs Template generator split statistics.
    • Saves segment-level quality reports to /dataset/descriptions/descriptions_summary.csv

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
DESCRIPTIONS_JSON = Path("/Users/mannatsaini/Desktop/my_robotics_data/descriptions/nl_descriptions.json")
DESCRIPTIONS_DIR  = Path("/Users/mannatsaini/Desktop/my_robotics_data/descriptions")

SHORT_WORDS_THRESH = 10
# ───────────────────────────────────────────────────────────

CYAN  = "\033[96m"
GREEN = "\033[92m"
YELLOW= "\033[93m"
RED   = "\033[91m"
BOLD  = "\033[1m"
DIM   = "\033[2m"
RESET = "\033[0m"


def load_descriptions(json_path: Path) -> list[dict]:
    if not json_path.exists():
        print(f"{RED}ERROR: Descriptions missing at '{json_path}'. Run refine_descriptions.py first.{RESET}")
        sys.exit(1)
    with open(json_path) as f:
        return json.load(f)


def analyze_descriptions(all_data: list[dict], out_dir: Path):
    segment_reports = []
    
    total_words = 0
    empty_count = 0
    short_count = 0
    version_counts = Counter()

    for seg in all_data:
        segment_id = seg["segment_id"]
        desc = seg.get("nl_description", "")
        version = seg.get("description_version", "unknown")
        
        words = desc.split() if desc else []
        word_count = len(words)
        
        version_counts[version] += 1
        total_words += word_count
        
        issues = []
        if word_count == 0:
            issues.append("empty description")
            empty_count += 1
        elif word_count < SHORT_WORDS_THRESH:
            issues.append(f"short description ({word_count} wds < {SHORT_WORDS_THRESH})")
            short_count += 1

        segment_reports.append({
            "segment_id": segment_id,
            "action_label": seg.get("action_label", ""),
            "object_count": len(seg.get("objects_present", [])),
            "description_version": version,
            "word_count": word_count,
            "issues": " | ".join(issues) if issues else "None"
        })

    total_segments = len(all_data)
    avg_words = total_words / total_segments if total_segments > 0 else 0

    print(f"\n{BOLD}{CYAN}{'═'*60}")
    print("  DESCRIPTION QUALITY REPORT")
    print(f"{'═'*60}{RESET}\n")

    print(f"  Total Descriptions Generated : {total_segments}")
    print(f"  Average Word Count           : {avg_words:.1f} words")
    
    print(f"\n  {BOLD}Generator Distribution:{RESET}")
    for ver, cnt in version_counts.items():
        pct = (cnt / total_segments) * 100
        print(f"    • {ver:<15} : {cnt} ({pct:.1f}%)")

    print(f"\n  {BOLD}Quality Flags:{RESET}")
    print(f"    • Empty Descriptions       : {YELLOW if empty_count > 0 else GREEN}{empty_count}{RESET}")
    print(f"    • Short (<10 words)        : {YELLOW if short_count > 0 else GREEN}{short_count}{RESET}")

    df = pd.DataFrame(segment_reports)
    out_csv = out_dir / "descriptions_summary.csv"
    df.to_csv(out_csv, index=False)

    print(f"\n{BOLD}{GREEN}✔ Summary report saved to: {out_csv}{RESET}")
    
    flagged = df[df["issues"] != "None"]
    if len(flagged) > 0:
        print(f"\n{RED}⚠ WARNING: {len(flagged)} segment(s) flagged for brief/missing captions.{RESET}")
        for _, row in flagged.head(5).iterrows():
            print(f"    → {row['segment_id']}: {row['issues']}")


def run(json_path: Path = DESCRIPTIONS_JSON,
        out_dir: Path = DESCRIPTIONS_DIR) -> None:
    all_data = load_descriptions(json_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    analyze_descriptions(all_data, out_dir)


if __name__ == "__main__":
    run()
