"""
=============================================================
SCRIPT 2: Description Refinement via Claude (with Fallback)
=============================================================
Usage:
    python refine_descriptions.py

What it does:
    • Loads raw_captions.json, segments.json, and tracked_bbox_annotations.json
    • For each segment, calls Claude API (anthropic Python SDK) to generate a cohesive NL description
    • If API fails or key is missing, falls back to a deterministic template.
    • Outputs /dataset/descriptions/nl_descriptions.json

Dependencies:
    pip install anthropic python-dotenv
=============================================================
"""

import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

# ─── CONFIG ────────────────────────────────────────────────
DESCRIPTIONS_DIR = Path("/Users/mannatsaini/Desktop/my_robotics_data/descriptions")
SEGMENTS_JSON    = Path("/Users/mannatsaini/Desktop/my_robotics_data/segments/segments.json")
ANNOTATIONS_JSON = Path("/Users/mannatsaini/Desktop/my_robotics_data/annotations/tracked_bbox_annotations.json")
RAW_CAPTIONS_JSON= Path("/Users/mannatsaini/Desktop/my_robotics_data/descriptions/raw_captions.json")

# Model configuration
CLAUDE_MODEL     = "claude-3-5-sonnet-20241022" 
MAX_TOKENS       = 150
# ───────────────────────────────────────────────────────────

CYAN  = "\033[96m"
GREEN = "\033[92m"
YELLOW= "\033[93m"
RED   = "\033[91m"
BOLD  = "\033[1m"
RESET = "\033[0m"


def load_json(path: Path) -> dict | list:
    if not path.exists():
        print(f"{RED}ERROR: Expected JSON file missing: {path}{RESET}")
        return []
    with open(path) as f:
        return json.load(f)


def build_metadata_maps(segments: list, annotations: list) -> tuple[dict, dict]:
    """Build fast lookup dicts mapping segment_id -> action_label and objects_present"""
    
    # Extract action labels
    action_map = {}
    for entry in segments:
        for seg in entry.get("segments", []):
            action_map[seg["segment_id"]] = seg["action_label"]

    # Extract unique objects mapped to the segment
    object_map = {}
    for entry in annotations:
        seg_id = entry.get("segment_id")
        unique_classes = set()
        for f in entry.get("frames", []):
            for obj in f.get("objects", []):
                unique_classes.add(obj["class"])
        object_map[seg_id] = list(unique_classes)
        
    return action_map, object_map


def generate_template_description(action: str, objects: list[str]) -> str:
    """Fallback generator avoiding API calls."""
    if objects:
        obs = ", ".join(objects[:-1]) + f" and {objects[-1]}" if len(objects) > 1 else objects[0]
        return f"The person performs {action} using {obs} in a kitchen/cleaning context."
    return f"The person performs {action} in a kitchen/cleaning context without notable objects."


def ask_claude(client, prompt: str) -> str:
    """Send refined context to Anthropic's Claude to combine into human-language."""
    try:
        response = client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=MAX_TOKENS,
            system="You are an AI generating cohesive natural language descriptions for an egocentric robotic video dataset. Respond strictly with the generated sentence in one line without conversational filler.",
            messages=[{"role": "user", "content": prompt}]
        )
        return response.content[0].text.strip()
    except Exception as e:
        print(f"  {RED}Claude API Error: {e}{RESET}")
        return None


def run(desc_dir: Path = DESCRIPTIONS_DIR) -> None:
    # Environment Setup
    load_dotenv()
    api_key = os.getenv("ANTHROPIC_API_KEY")
    client  = None
    
    if api_key and api_key != "your-anthropic-api-key-here":
        try:
            import anthropic
            client = anthropic.Anthropic(api_key=api_key)
            print(f"{GREEN}✔ Anthropic SDK loaded properly.{RESET}")
        except ImportError:
            print(f"{YELLOW}⚠ anthropic library missing. Run: pip install anthropic{RESET}")
            print(f"{YELLOW}⚠ Failing back to template generator.{RESET}")
    else:
        print(f"{YELLOW}⚠ ANTHROPIC_API_KEY missing in .env. Using fallback template generator.{RESET}")

    raw_captions = load_json(RAW_CAPTIONS_JSON)
    segments_raw = load_json(SEGMENTS_JSON)
    annos_raw    = load_json(ANNOTATIONS_JSON)

    action_map, object_map = build_metadata_maps(segments_raw, annos_raw)
    
    out_json = desc_dir / "nl_descriptions.json"
    results = []

    print(f"\n{BOLD}Generating descriptions for {len(raw_captions)} segments…{RESET}")
    
    for idx, entry in enumerate(raw_captions, 1):
        seg_id = entry["segment_id"]
        captions = entry.get("raw_captions", [])
        
        action = action_map.get(seg_id, "unknown action")
        objects = object_map.get(seg_id, [])

        print(f"[{idx}/{len(raw_captions)}] {seg_id} | Action: {action} | Objects: {len(objects)}")
        
        nl_desc = None
        version = "template"
        
        if client:
            prompt = (
                f"I am analyzing a video segment labeled as '{action}'.\n"
                f"Objects detected in the frame: {', '.join(objects) if objects else 'None'}.\n"
                f"Here are three sequential raw visual captions describing the scene at start, middle, and end:\n"
                f"1. {captions[0] if len(captions)>0 else 'N/A'}\n"
                f"2. {captions[1] if len(captions)>1 else 'N/A'}\n"
                f"3. {captions[2] if len(captions)>2 else 'N/A'}\n\n"
                f"Condense all this information into one clean, structured natural language description written in present tense starting with 'The person...'"
            )
            
            nl_desc = ask_claude(client, prompt)
            if nl_desc:
                version = "claude-refined"

        if not nl_desc:
            nl_desc = generate_template_description(action, objects)

        print(f"  → Result [{version}]: {nl_desc}")
        
        results.append({
            "segment_id": seg_id,
            "action_label": action,
            "objects_present": objects,
            "raw_captions": captions,
            "nl_description": nl_desc,
            "description_version": version
        })

    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{BOLD}{GREEN}✔ nl_descriptions.json saved to: {out_json}{RESET}")


if __name__ == "__main__":
    run()
