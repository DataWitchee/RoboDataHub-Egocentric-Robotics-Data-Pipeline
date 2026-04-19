"""
=============================================================
SCRIPT 5: run_all.py — One-shot pipeline runner
=============================================================
Usage:
    python run_all.py /path/to/video/folder

Runs all four tools in the correct order:
  1. setup_structure  → create /dataset directories
  2. explore_dataset  → extract metadata + save dataset_summary.csv
  3. quality_checker  → flag bad videos  + save quality_report.csv
  4. extract_keyframes → sample keyframes + display thumbnail grid
=============================================================
"""

import sys
import subprocess
from pathlib import Path

SCRIPTS = [
    ("setup_structure.py",  "📁  Creating folder structure"),
    ("explore_dataset.py",  "📊  Extracting video metadata"),
    ("quality_checker.py",  "🔍  Running quality checks"),
    ("extract_keyframes.py","🖼   Extracting keyframes & building thumbnail grid"),
]

CYAN  = "\033[96m"
GREEN = "\033[92m"
RED   = "\033[91m"
BOLD  = "\033[1m"
RESET = "\033[0m"


def main():
    if len(sys.argv) < 2:
        print("Usage: python run_all.py <video_folder>")
        sys.exit(1)

    video_folder = sys.argv[1]
    script_dir   = Path(__file__).parent

    print(f"\n{BOLD}{CYAN}{'═'*60}")
    print("  Egocentric Dataset Toolkit — Full Pipeline")
    print(f"{'═'*60}{RESET}\n")

    for script, label in SCRIPTS:
        print(f"{BOLD}{label}…{RESET}")
        result = subprocess.run(
            [sys.executable, str(script_dir / script), video_folder],
            capture_output=False,
        )
        if result.returncode != 0:
            print(f"{RED}✗  {script} exited with code {result.returncode}{RESET}\n")
        else:
            print(f"{GREEN}✔  Done{RESET}\n")

    print(f"{BOLD}{GREEN}Pipeline complete!{RESET}")
    print(f"  • Metadata   → {video_folder}/dataset_summary.csv")
    print(f"  • QA report  → {video_folder}/quality_report.csv")
    print(f"  • Previews   → {video_folder}/previews/")


if __name__ == "__main__":
    main()
