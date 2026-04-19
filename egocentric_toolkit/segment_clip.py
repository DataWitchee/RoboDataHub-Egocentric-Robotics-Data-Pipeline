"""
=============================================================
SCRIPT 2: CLIP-Based Action Segmenter
=============================================================
Usage:
    python segment_clip.py

What it does:
    • Scans /dataset/raw_videos/ for MP4/AVI files
    • Loads OpenCLIP (ViT-B/32, pretrained on LAION-400M)
    • Samples one frame every SAMPLE_INTERVAL_SEC seconds
    • Classifies each sampled frame against ACTION_LABELS using
      zero-shot CLIP text prompts
    • Merges consecutive frames of the same predicted action
      into a segment (with hysteresis smoothing)
    • Writes /dataset/segments/segments.json in the required schema

Dependencies:
    pip install open_clip_torch torch torchvision opencv-python

Note on hardware:
    • GPU  → fast (~2-5 min/video at 1080p)
    • CPU  → slow but functional (use SAMPLE_INTERVAL_SEC ≥ 2.0)
=============================================================
"""

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

# ─── CONFIG ────────────────────────────────────────────────
RAW_VIDEOS_DIR  = Path("/Users/mannatsaini/Desktop/my_robotics_data/raw_videos")
SEGMENTS_DIR    = Path("/Users/mannatsaini/Desktop/my_robotics_data/segments")

# How often to sample frames (in seconds between samples)
SAMPLE_INTERVAL_SEC = 1.0

# CLIP model config
CLIP_MODEL      = "ViT-B-32"
CLIP_PRETRAINED = "laion400m_e32"   # good zero-shot generaliser

# Consecutive-frame smoothing window: a segment must persist for at
# least this many frames before a label change is accepted.
# Prevents noisy single-frame mislabels from fragmenting segments.
SMOOTHING_WINDOW = 3

# Minimum segment duration to keep in the output JSON (seconds)
MIN_SEG_DURATION = 0.5

# Action labels — CLIP reads free-form text so these can be
# descriptive phrases for better embedding quality.
ACTION_LABELS = [
    "a person chopping vegetables with a knife",
    "a person stirring food in a pot or pan",
    "a person pouring liquid into a container",
    "a person boiling water or cooking on a stove",
    "a person wiping a surface with a cloth",
    "a person washing dishes or hands under running water",
    "a person scrubbing a surface vigorously",
    "a person picking up an object from a surface",
    "a person putting an object away or placing it down",
    "a person standing idle or not performing a task",
]

# Canonical short label names (index-matched to ACTION_LABELS above)
SHORT_LABELS = [
    "chopping",
    "stirring",
    "pouring",
    "boiling",
    "wiping",
    "washing",
    "scrubbing",
    "picking_up",
    "putting_away",
    "idle",
]
# ───────────────────────────────────────────────────────────


# ─── ANSI colours ──────────────────────────────────────────
CYAN  = "\033[96m"
GREEN = "\033[92m"
YELLOW= "\033[93m"
RED   = "\033[91m"
BOLD  = "\033[1m"
DIM   = "\033[2m"
RESET = "\033[0m"


# ─── Model loading ─────────────────────────────────────────
def load_clip_model(device: str) -> tuple:
    """
    Load OpenCLIP model, preprocessing transform, and pre-encode
    the action label text prompts.

    Args:
        device: 'cuda' or 'cpu'.

    Returns:
        (model, preprocess, text_features) where text_features is a
        (N_labels, embed_dim) float32 tensor already normalised.
    """
    try:
        import open_clip
    except ImportError:
        print(f"{RED}ERROR: open_clip not installed.\n"
              f"  Run:  pip install open-clip-torch{RESET}")
        sys.exit(1)

    print(f"  Loading CLIP ({CLIP_MODEL} / {CLIP_PRETRAINED}) on {device}…")
    model, _, preprocess = open_clip.create_model_and_transforms(
        CLIP_MODEL, pretrained=CLIP_PRETRAINED
    )
    model = model.to(device).eval()

    tokenizer = open_clip.get_tokenizer(CLIP_MODEL)

    with torch.no_grad():
        text_tokens   = tokenizer(ACTION_LABELS).to(device)
        text_features = model.encode_text(text_tokens)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        text_features = text_features.float()

    print(f"  Encoded {len(ACTION_LABELS)} action prompt(s)")
    return model, preprocess, text_features


# ─── Frame sampling & inference ────────────────────────────
def sample_and_classify(
    video_path: Path,
    model,
    preprocess,
    text_features: torch.Tensor,
    device: str,
) -> list[dict]:
    """
    Sample frames from a video and classify each with CLIP.

    Args:
        video_path:    Path to the source video.
        model:         OpenCLIP model (eval mode).
        preprocess:    CLIP image preprocessing transform.
        text_features: Pre-encoded, normalised text embeddings.
        device:        'cuda' or 'cpu'.

    Returns:
        List of dicts, one per sampled frame:
            {frame_idx, timestamp_sec, label, label_idx, confidence}
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"  {RED}Cannot open {video_path.name}{RESET}")
        return []

    fps          = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step_frames  = max(1, int(fps * SAMPLE_INTERVAL_SEC))

    frame_results = []
    sampled_count = 0

    for frame_idx in range(0, total_frames, step_frames):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, bgr = cap.read()
        if not ret:
            break

        # Convert BGR → RGB PIL image for CLIP
        rgb   = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        pil   = Image.fromarray(rgb)
        image = preprocess(pil).unsqueeze(0).to(device)

        with torch.no_grad():
            img_features = model.encode_image(image)
            img_features = img_features / img_features.norm(dim=-1, keepdim=True)
            img_features = img_features.float()

            # Cosine similarities → softmax probabilities
            logits       = (img_features @ text_features.T).squeeze(0)
            probs        = logits.softmax(dim=-1).cpu().numpy()

        best_idx    = int(np.argmax(probs))
        confidence  = float(probs[best_idx])

        frame_results.append({
            "frame_idx":     frame_idx,
            "timestamp_sec": round(frame_idx / fps, 3),
            "label_idx":     best_idx,
            "label":         SHORT_LABELS[best_idx],
            "confidence":    round(confidence, 4),
        })
        sampled_count += 1

        # Progress every 50 samples
        if sampled_count % 50 == 0:
            pct = int(frame_idx / total_frames * 100)
            print(f"\r    {DIM}Progress: {pct}%  ({frame_idx}/{total_frames} frames){RESET}",
                  end="", flush=True)

    cap.release()
    print(f"\r    Sampled {sampled_count} frame(s)        ")
    return frame_results


# ─── Smoothing & segment merging ───────────────────────────
def smooth_labels(frame_results: list[dict], window: int) -> list[dict]:
    """
    Apply a simple majority-vote sliding window to reduce label noise.

    Args:
        frame_results: Output of sample_and_classify().
        window:        Smoothing window size (in frames).

    Returns:
        New list with 'label' and 'label_idx' updated in-place copy.
    """
    n       = len(frame_results)
    smoothed = [r.copy() for r in frame_results]
    half_w  = window // 2

    for i in range(n):
        lo = max(0, i - half_w)
        hi = min(n, i + half_w + 1)
        window_labels = [frame_results[j]["label_idx"] for j in range(lo, hi)]
        # Majority vote
        best = max(set(window_labels), key=window_labels.count)
        smoothed[i]["label_idx"] = best
        smoothed[i]["label"]     = SHORT_LABELS[best]

    return smoothed


def merge_into_segments(
    frame_results: list[dict],
    video_id: str,
    fps: float,
) -> list[dict]:
    """
    Group consecutive frames sharing the same label into segments.

    Args:
        frame_results: Smoothed list from smooth_labels().
        video_id:      String identifier for the video.
        fps:           Video frame rate (for end_frame calculation).

    Returns:
        List of segment dicts matching the required JSON schema.
    """
    if not frame_results:
        return []

    segments     = []
    seg_idx      = 1
    current      = frame_results[0]
    seg_start    = current
    seg_confidences = [current["confidence"]]

    for fr in frame_results[1:]:
        if fr["label"] == current["label"]:
            # Still same action — extend segment
            seg_confidences.append(fr["confidence"])
            current = fr
        else:
            # Label changed → close current segment
            duration = current["timestamp_sec"] - seg_start["timestamp_sec"]
            if duration >= MIN_SEG_DURATION:
                segments.append({
                    "segment_id":  f"{video_id}_seg{seg_idx:03d}",
                    "action_label":seg_start["label"],
                    "start_time":  seg_start["timestamp_sec"],
                    "end_time":    current["timestamp_sec"],
                    "start_frame": seg_start["frame_idx"],
                    "end_frame":   current["frame_idx"],
                    "confidence":  round(float(np.mean(seg_confidences)), 4),
                })
                seg_idx += 1

            # Start new segment
            seg_start       = fr
            seg_confidences = [fr["confidence"]]
            current         = fr

    # Close the final segment
    duration = current["timestamp_sec"] - seg_start["timestamp_sec"]
    if duration >= MIN_SEG_DURATION:
        segments.append({
            "segment_id":  f"{video_id}_seg{seg_idx:03d}",
            "action_label":seg_start["label"],
            "start_time":  seg_start["timestamp_sec"],
            "end_time":    current["timestamp_sec"],
            "start_frame": seg_start["frame_idx"],
            "end_frame":   current["frame_idx"],
            "confidence":  round(float(np.mean(seg_confidences)), 4),
        })

    return segments


# ─── MAIN ──────────────────────────────────────────────────
def run(raw_dir: Path = RAW_VIDEOS_DIR,
        seg_dir: Path = SEGMENTS_DIR) -> None:
    """
    Full CLIP segmentation pipeline over all videos.
    Writes segments.json inside seg_dir.
    """
    if not raw_dir.exists():
        print(f"{RED}ERROR: RAW_VIDEOS_DIR '{raw_dir}' does not exist.{RESET}")
        sys.exit(1)

    seg_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n{BOLD}Device: {device}{RESET}")

    model, preprocess, text_features = load_clip_model(device)

    exts   = {".mp4", ".avi"}
    videos = sorted(p for p in raw_dir.rglob("*")
                    if p.is_file() and p.suffix.lower() in exts)

    print(f"{CYAN}Found {len(videos)} video(s){RESET}\n")
    all_output = []

    for idx, vp in enumerate(videos, 1):
        video_id = f"video{idx:03d}"
        print(f"{BOLD}[{idx}/{len(videos)}] {vp.name}  →  {video_id}{RESET}")

        cap = cv2.VideoCapture(str(vp))
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()

        frame_results = sample_and_classify(vp, model, preprocess,
                                             text_features, device)
        if not frame_results:
            continue

        smoothed  = smooth_labels(frame_results, window=SMOOTHING_WINDOW)
        segments  = merge_into_segments(smoothed, video_id, fps)

        print(f"  → {len(segments)} segment(s) detected")
        for s in segments:
            dur = round(s["end_time"] - s["start_time"], 2)
            print(f"    {DIM}{s['segment_id']:20s} "
                  f"{s['action_label']:15s} "
                  f"{s['start_time']:.1f}s→{s['end_time']:.1f}s "
                  f"({dur}s)  conf={s['confidence']:.2f}{RESET}")

        all_output.append({
            "video_id": video_id,
            "source":   str(vp),
            "segments": segments,
        })

    # ── Write segments.json ──────────────────────────────────
    out_json = seg_dir / "segments.json"
    with open(out_json, "w") as f:
        json.dump(all_output, f, indent=2)

    total_segs = sum(len(v["segments"]) for v in all_output)
    print(f"\n{BOLD}{GREEN}✔ segments.json saved: {out_json}{RESET}")
    print(f"  Total segments: {total_segs}")


if __name__ == "__main__":
    run()
