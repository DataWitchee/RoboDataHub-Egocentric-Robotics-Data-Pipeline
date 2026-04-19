"""
=============================================================
run_annotation_pipeline.py — Full Annotation Pipeline Runner
=============================================================
Usage:
    python run_annotation_pipeline.py [--detect] [--track] [--check] [--visualize] [--all]

Flags:
    --detect      Run YOLOv8 per-frame detection without tracking
    --track       Run YOLOv8 + ByteTrack to generate stable object IDs
    --check       Run annotation quality checker and stats
    --visualize   Generate annotated video clips and sample frames
    --all         Run all four steps in order

Example:
    python run_annotation_pipeline.py --all
    python run_annotation_pipeline.py --track --visualize
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
    "--detect":    ("annotate_yolo.py",         "🔍  YOLOv8 Object Detection"),
    "--track":     ("track_yolo.py",            "🕵🏻‍♂️  YOLOv8 + ByteTrack Object Tracking"),
    "--visualize": ("visualize_annotations.py", "🎥  Render annotated videos"),
    "--check":     ("check_annotations.py",     "✅  Verify annotation quality & generate stats"),
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
    
    # Process commands in intended logical order according to STEPS dict definition
    selected = [(flag, info) for flag, info in STEPS.items()
                if run_all or flag in args]

    if not selected:
        print(f"{RED}No valid step flags given. Use --all or one of:{RESET}")
        for flag in STEPS:
            print(f"  {flag}")
        sys.exit(1)

    print(f"\n{BOLD}{CYAN}{'═'*60}")
    print("  Robotics Bounding Box Annotation Pipeline")
    print(f"{'═'*60}{RESET}")

    results = {}
    for flag, (script, label) in selected:
        results[script] = run_step(script, label)

    print(f"\n{BOLD}{'─'*60}{RESET}")
    print("  Pipeline Summary")
    print(f"{BOLD}{'─'*60}{RESET}")
    
    for flag, (script, _) in selected:
        ok = results.get(script, False)
        icon = f"{GREEN}✔{RESET}" if ok else f"{RED}✗{RESET}"
        print(f"  {icon}  {script}")
    print()


if __name__ == "__main__":
    main()
