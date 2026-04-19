"""
=============================================================
run_final_pipeline.py — Phase 5: Dataset Packaging Runner
=============================================================
Usage:
    python run_final_pipeline.py [--build] [--package] [--card] [--validate] [--lerobot] [--all]

Flags:
    --build     Merge all JSON sources → master_dataset.json / .csv
    --package   Convert master_dataset → HuggingFace DatasetDict + loader
    --card      Generate HuggingFace README.md dataset card
    --validate  Run completeness checks and print final summary
    --lerobot   Export to LeRobot episode format (bonus)
    --all       Run all five steps in order

Example:
    python run_final_pipeline.py --all
    python run_final_pipeline.py --build --validate
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
    "--build":    ("build_dataset.py",         "🔧  Merging all sources into master_dataset"),
    "--package":  ("package_hf_dataset.py",    "📦  Packaging HuggingFace DatasetDict"),
    "--card":     ("generate_dataset_card.py", "📝  Generating dataset card (README.md)"),
    "--validate": ("validate_pipeline.py",     "✅  Running final pipeline validation"),
    "--lerobot":  ("convert_lerobot.py",       "🤖  Exporting to LeRobot format"),
}


def run_step(script: str, label: str) -> bool:
    print(f"\n{BOLD}{label}…{RESET}")
    result = subprocess.run(
        [sys.executable, str(SCRIPT_DIR / script)],
        capture_output=False,
    )
    if result.returncode == 0:
        print(f"{GREEN}✔  Done{RESET}")
        return True
    print(f"{RED}✗  {script} failed (exit code {result.returncode}){RESET}")
    return False


def main() -> None:
    args = set(sys.argv[1:])
    if not args or "--help" in args or "-h" in args:
        print(__doc__)
        sys.exit(0)

    run_all  = "--all" in args
    selected = [(flag, info) for flag, info in STEPS.items()
                if run_all or flag in args]

    if not selected:
        print(f"{RED}No valid flags. Use --all or one of:{RESET}")
        for flag in STEPS:
            print(f"  {flag}")
        sys.exit(1)

    print(f"\n{BOLD}{CYAN}{'═'*58}")
    print("  Robotics Dataset — Final Packaging Pipeline")
    print(f"{'═'*58}{RESET}")

    results = {script: run_step(script, label)
               for _, (script, label) in selected}

    print(f"\n{BOLD}{'─'*58}{RESET}")
    print("  Summary")
    print(f"{BOLD}{'─'*58}{RESET}")
    for _, (script, _) in selected:
        icon = f"{GREEN}✔{RESET}" if results.get(script) else f"{RED}✗{RESET}"
        print(f"  {icon}  {script}")
    print()


if __name__ == "__main__":
    main()
