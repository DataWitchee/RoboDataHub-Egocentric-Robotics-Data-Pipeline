"""
=============================================================
run_descriptions_pipeline.py — Full NL Description Pipeline Runner
=============================================================
Usage:
    python run_descriptions_pipeline.py [--raw] [--refine] [--check] [--all]

Flags:
    --raw         Run BLIP-2 to extract raw image captions for segment keyframes.
    --refine      Run Anthropic Claude API (or fallback) to synthesize NL descriptions.
    --check       Run description quality checker and stats.
    --all         Run all three steps in order.

Example:
    python run_descriptions_pipeline.py --all
    python run_descriptions_pipeline.py --refine --check
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
    "--raw":    ("generate_raw_captions.py", "📸  BLIP-2 Raw Keyframe Captioning"),
    "--refine": ("refine_descriptions.py",   "🧠  Claude Description Refinement & Synthesis"),
    "--check":  ("check_descriptions.py",    "✅  Verify caption quality & compute stats"),
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
    print("  Robotics Natural Language Description Pipeline")
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
