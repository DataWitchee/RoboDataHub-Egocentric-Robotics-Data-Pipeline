"""
=============================================================
SCRIPT 1: BLIP-2 Raw Caption Generation
=============================================================
Usage:
    python generate_raw_captions.py

What it does:
    • Scans /dataset/segments/ for segmented MP4 clips.
    • Extracts 3 keyframes (Start, Middle, End) per segment.
    • Uses the BLIP-2 vision-language model to generate a raw image caption for each keyframe in zero-shot mode.
    • Saves the combined raw captions to /dataset/descriptions/raw_captions.json.

Dependencies:
    pip install transformers torch torchvision opencv-python Pillow pandas
=============================================================
"""

import json
import sys
from pathlib import Path

import cv2
import torch
from PIL import Image

# ─── CONFIG ────────────────────────────────────────────────
SEGMENTS_DIR    = Path("/Users/mannatsaini/Desktop/my_robotics_data/segments")
DESCRIPTIONS_DIR= Path("/Users/mannatsaini/Desktop/my_robotics_data/descriptions")

# Model configuration: Opting for BLIP-2 OPT-2.7b backend
# Note: For faster testing, replace with SalesForce/blip-image-captioning-base
BLIP2_MODEL_ID  = "Salesforce/blip2-opt-2.7b"
# ───────────────────────────────────────────────────────────

CYAN  = "\033[96m"
GREEN = "\033[92m"
YELLOW= "\033[93m"
RED   = "\033[91m"
BOLD  = "\033[1m"
DIM   = "\033[2m"
RESET = "\033[0m"


def extract_keyframes(video_path: Path) -> list[Image.Image]:
    """Extract 3 frames from a video (0%, 50%, 100%). Returns PIL Images."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return []

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = [0, total_frames // 2, max(0, total_frames - 1)]
    
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            # OpenCV parses as BGR, PIL requires RGB
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(rgb))
            
    cap.release()
    return frames


def build_captions(processor, model, device: str, frames: list[Image.Image]) -> list[str]:
    captions = []
    for img in frames:
        # Prompt guides the VQA nature of BLIP2
        inputs = processor(images=img, return_tensors="pt").to(device, torch.float16)
        
        generated_ids = model.generate(**inputs, max_new_tokens=40)
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        captions.append(generated_text)
    return captions


def run(segments_dir: Path = SEGMENTS_DIR,
        descriptions_dir: Path = DESCRIPTIONS_DIR) -> None:
    if not segments_dir.exists():
        print(f"{RED}ERROR: SEGMENTS_DIR '{segments_dir}' does not exist.{RESET}")
        sys.exit(1)

    try:
        from transformers import Blip2Processor, Blip2ForConditionalGeneration
    except ImportError:
        print(f"{RED}ERROR: transformers not installed.\n  Run: pip install transformers{RESET}")
        sys.exit(1)

    descriptions_dir.mkdir(parents=True, exist_ok=True)
    out_json = descriptions_dir / "raw_captions.json"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n{BOLD}Loading BLIP-2 ({BLIP2_MODEL_ID}) on {device}. This may take a minute...{RESET}")
    
    # FP16 to save memory explicitly on GPUs
    dtype = torch.float16 if device == "cuda" else torch.float32
    
    processor = Blip2Processor.from_pretrained(BLIP2_MODEL_ID)
    model = Blip2ForConditionalGeneration.from_pretrained(BLIP2_MODEL_ID, torch_dtype=dtype)
    model.to(device).eval()

    videos = sorted([p for p in segments_dir.rglob("*.mp4") if p.is_file()])
    print(f"{CYAN}Found {len(videos)} segment(s) to process.{RESET}\n")

    all_captions = []

    for idx, video_path in enumerate(videos, 1):
        segment_id = video_path.stem
        print(f"{BOLD}[{idx}/{len(videos)}] Captioning {segment_id}{RESET}")

        frames = extract_keyframes(video_path)
        if not frames:
             print(f"  {RED}Failed to read frames.{RESET}")
             continue

        captions = build_captions(processor, model, device, frames)
        if len(captions) == 3:
            print(f"  → Start : {captions[0]}")
            print(f"  → Middle: {captions[1]}")
            print(f"  → End   : {captions[2]}")

            all_captions.append({
                "segment_id": segment_id,
                "video_path": str(video_path),
                "raw_captions": captions
            })

    with open(out_json, "w") as f:
        json.dump(all_captions, f, indent=2)

    print(f"\n{BOLD}{GREEN}✔ raw_captions.json saved to: {out_json}{RESET}")


if __name__ == "__main__":
    run()
