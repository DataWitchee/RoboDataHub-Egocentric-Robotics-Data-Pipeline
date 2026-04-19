"""
=============================================================
run_segmentation_pipeline.py — Full Segmentation Pipeline Runner
=============================================================
Usage:
    python run_segmentation_pipeline.py [--scenedetect] [--clip] [--validate] [--review] [--all]

Flags:
    --scenedetect   Run PySceneDetect boundary segmentation
    --clip          Run CLIP action classification segmentation
    --validate      Run segment validation & statistics
    --review        Run visual thumbnail grid review
    --all           Run all four steps in order

Example:
    python run_segmentation_pipeline.py --all
    python run_segmentation_pipeline.py --clip --validate --review
=============================================================
"""

import sys
import subprocess
from pathlib import Path

CYAN  = "\033[96m"
GREEN = "\033[92m"
RED   = "\033[91m"
BOLD  = "\033[1m"
RESET = "\033[0m"

SCRIPT_DIR = Path(__file__).parent

STEPS = {
    "--scenedetect": ("segment_pyscenedetect.py", "🎬  PySceneDetect boundary segmentation"),
    "--clip":        ("segment_clip.py",           "🤖  CLIP action classification"),
    "--validate":    ("validate_segments.py",      "✅  Segment validation & statistics"),
    "--review":      ("review_segments.py",        "🖼   Visual thumbnail review"),
}


def run_step(script_name: str, label: str) -> bool:
    print(f"\n{BOLD}{label}…{RESET}")
    result = subprocess.run(
        [sys.executable, str(SCRIPT_DIR / script_name)],
        capture_output=False,
    )
    if result.returncode == 0:
        print(f"{GREEN}✔  Done{RESET}")
        return True
    else:
        print(f"{RED}✗  {script_name} failed (exit code {result.returncode}){RESET}")
        return False


def main():
    args = set(sys.argv[1:])

    if not args or "--help" in args or "-h" in args:
        print(__doc__)
        sys.exit(0)

    run_all = "--all" in args
    selected = [(flag, info) for flag, info in STEPS.items()
                if run_all or flag in args]

    if not selected:
        print(f"{RED}No valid step flags given. Use --all or one of:{RESET}")
        for flag in STEPS:
            print(f"  {flag}")
        sys.exit(1)

    print(f"\n{BOLD}{CYAN}{'═'*60}")
    print("  Robotics Segmentation Pipeline")
    print(f"{'═'*60}{RESET}")

    results = {}
    for flag, (script, label) in selected:
        results[script] = run_step(script, label)

    print(f"\n{BOLD}{'─'*60}{RESET}")
    print("  Pipeline Summary")
    print(f"{BOLD}{'─'*60}{RESET}")
    for script, ok in results.items():
        icon = f"{GREEN}✔{RESET}" if ok else f"{RED}✗{RESET}"
        print(f"  {icon}  {script}")
    print()


if __name__ == "__main__":
    main()
