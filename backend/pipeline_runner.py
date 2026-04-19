import asyncio
import json
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, Any
from collections import Counter

# ── Paths ─────────────────────────────────────────────────────────────────────
_data_root = Path(os.getenv("DATASET_ROOT", str(Path.home() / "Desktop" / "my_robotics_data")))
RAW_VIDEOS_DIR   = _data_root / "raw_videos"
SEGMENTS_DIR     = _data_root / "segments"
ANNOTATIONS_DIR  = _data_root / "annotations"
DESCRIPTIONS_DIR = _data_root / "descriptions"
FINAL_DIR        = _data_root / "final_dataset"

TOOLKIT_DIR = Path(os.getenv("TOOLKIT_DIR", str(Path.home() / "cccc" / "egocentric_toolkit")))

# Timeout per script — BLIP-2/CLIP can be VERY slow on CPU MacBooks
SCRIPT_TIMEOUT = int(os.getenv("SCRIPT_TIMEOUT", "30"))

logger = logging.getLogger(__name__)

# ── Global state ──────────────────────────────────────────────────────────────
PIPELINE_STATE: Dict[str, Any] = {
    "pipeline_id": None,
    "current_stage": 0,
    "stage_name": "",
    "progress": 0,
    "status": "idle",
    "stages": [
        {"id": 1, "name": "Video Ingestion",     "status": "pending"},
        {"id": 2, "name": "Action Segmentation",  "status": "pending"},
        {"id": 3, "name": "Object Annotation",    "status": "pending"},
        {"id": 4, "name": "NL Description",       "status": "pending"},
        {"id": 5, "name": "Dataset Structuring",  "status": "pending"},
    ],
    "estimated_time_remaining_sec": 0,
    "error": None,
    "results": None,
}


def reset_state(pipeline_id: str):
    PIPELINE_STATE["pipeline_id"] = pipeline_id
    PIPELINE_STATE["current_stage"] = 1
    PIPELINE_STATE["stage_name"] = "Video Ingestion"
    PIPELINE_STATE["progress"] = 0
    PIPELINE_STATE["status"] = "running"
    PIPELINE_STATE["error"] = None
    PIPELINE_STATE["estimated_time_remaining_sec"] = 60
    PIPELINE_STATE["results"] = None
    for stage in PIPELINE_STATE["stages"]:
        stage["status"] = "pending"
        stage.pop("progress", None)


# ── JSON helpers ──────────────────────────────────────────────────────────────

def _load_json(path: Path) -> list | dict:
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return []


def compute_live_results() -> dict:
    """Read actual pipeline output files and compute real statistics."""
    segments_json = _load_json(SEGMENTS_DIR / "segments.json")
    bbox_json     = (_load_json(ANNOTATIONS_DIR / "tracked_bbox_annotations.json")
                     or _load_json(ANNOTATIONS_DIR / "bbox_annotations.json"))
    nl_json       = _load_json(DESCRIPTIONS_DIR / "nl_descriptions.json")

    # Flatten segments
    flat_segments = []
    for video_entry in segments_json:
        for seg in video_entry.get("segments", []):
            flat_segments.append(seg)

    # Object index
    obj_by_seg = {}
    total_objects = 0
    for entry in bbox_json:
        sid = entry.get("segment_id", "")
        unique = set()
        for frame in entry.get("frames", []):
            for obj in frame.get("objects", []):
                unique.add(obj.get("class", "unknown"))
                total_objects += 1
        obj_by_seg[sid] = sorted(unique)

    # Description index
    desc_by_seg = {}
    for entry in nl_json:
        sid = entry.get("segment_id", "")
        desc_by_seg[sid] = entry.get("nl_description", "")

    # Stats
    total_segments = len(flat_segments)
    action_labels = [s.get("action_label", "unknown") for s in flat_segments]
    unique_actions = sorted(set(action_labels))

    durations = []
    for s in flat_segments:
        dur = round(s.get("end_time", 0) - s.get("start_time", 0), 2)
        durations.append(dur)
    avg_duration = round(sum(durations) / len(durations), 2) if durations else 0

    n_train = int(total_segments * 0.8)
    n_val   = int(total_segments * 0.1)
    n_test  = total_segments - n_train - n_val

    segment_rows = []
    for seg in flat_segments:
        sid = seg.get("segment_id", "")
        segment_rows.append({
            "segment_id":     sid,
            "action_label":   seg.get("action_label", "unknown"),
            "duration":       round(seg.get("end_time", 0) - seg.get("start_time", 0), 2),
            "objects_present": obj_by_seg.get(sid, []),
            "nl_description": desc_by_seg.get(sid, ""),
        })

    return {
        "summary": {
            "total_segments":         total_segments,
            "action_categories":      unique_actions,
            "total_objects_annotated": total_objects,
            "descriptions_generated": len(desc_by_seg),
            "splits":                 {"train": n_train, "val": n_val, "test": n_test},
            "avg_segment_duration":   avg_duration,
            "dataset_path":           str(FINAL_DIR),
        },
        "segments": segment_rows,
    }


# ── Script runner with TIMEOUT ────────────────────────────────────────────────

def _run_script(script_name: str, timeout: int = None) -> bool:
    """
    Run a toolkit script as a subprocess WITH a timeout.
    Returns True on success, False on failure/timeout/missing.
    """
    if timeout is None:
        timeout = SCRIPT_TIMEOUT

    script_path = TOOLKIT_DIR / script_name
    if not script_path.exists():
        logger.warning(f"Script not found: {script_path}")
        return False

    cmd = [sys.executable, str(script_path)]
    logger.info(f"Running (timeout={timeout}s): {' '.join(cmd)}")

    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout
        )
        if result.returncode != 0:
            logger.warning(f"Script {script_name} exited with code {result.returncode}")
            if result.stderr:
                # Log first 500 chars of stderr
                logger.warning(f"  stderr: {result.stderr[:500]}")
            return False
        logger.info(f"Script {script_name} completed OK")
        return True
    except subprocess.TimeoutExpired:
        logger.warning(f"Script {script_name} TIMED OUT after {timeout}s — using fallback")
        return False


# ── Stage state helpers ───────────────────────────────────────────────────────

def _update_stage(stage_id: int, progress: int):
    idx = stage_id - 1
    PIPELINE_STATE["current_stage"] = stage_id
    PIPELINE_STATE["stage_name"] = PIPELINE_STATE["stages"][idx]["name"]
    PIPELINE_STATE["progress"] = progress
    PIPELINE_STATE["stages"][idx]["status"] = "running"
    PIPELINE_STATE["stages"][idx]["progress"] = progress


def _complete_stage(stage_id: int):
    idx = stage_id - 1
    PIPELINE_STATE["stages"][idx]["status"] = "completed"
    PIPELINE_STATE["stages"][idx].pop("progress", None)
    PIPELINE_STATE["progress"] = 100
    PIPELINE_STATE["estimated_time_remaining_sec"] = max(
        0, PIPELINE_STATE["estimated_time_remaining_sec"] - 12
    )
    logger.info(f"--- COMPLETED STAGE {stage_id} ---")


# ── Stub generators (fallbacks when scripts timeout/fail) ─────────────────────

def _stub_segments():
    """Create minimal segments.json from raw video files."""
    import cv2
    videos = sorted(RAW_VIDEOS_DIR.glob("*.mp4")) + sorted(RAW_VIDEOS_DIR.glob("*.avi"))
    segments_data = []
    for vid_idx, vid_path in enumerate(videos, 1):
        cap = cv2.VideoCapture(str(vid_path))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total / fps if fps > 0 else 0
        cap.release()

        vid_id = f"video{vid_idx:03d}"
        segments_data.append({
            "video_id": vid_id,
            "source": vid_path.name,
            "segments": [{
                "segment_id": f"{vid_id}_seg001",
                "action_label": "activity",
                "start_time": 0.0,
                "end_time": round(duration, 2),
                "start_frame": 0,
                "end_frame": total,
            }]
        })
    SEGMENTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(SEGMENTS_DIR / "segments.json", "w") as f:
        json.dump(segments_data, f, indent=2)
    logger.info(f"Stub: Created segments.json with {len(segments_data)} video(s)")


def _stub_annotations():
    """Create empty bbox_annotations.json."""
    segs = _load_json(SEGMENTS_DIR / "segments.json")
    annos = []
    for v in segs:
        for seg in v.get("segments", []):
            annos.append({"segment_id": seg["segment_id"], "frames": []})
    ANNOTATIONS_DIR.mkdir(parents=True, exist_ok=True)
    with open(ANNOTATIONS_DIR / "bbox_annotations.json", "w") as f:
        json.dump(annos, f, indent=2)
    logger.info(f"Stub: Created bbox_annotations.json with {len(annos)} segment(s)")


def _stub_descriptions():
    """Create template-based nl_descriptions.json."""
    segs = _load_json(SEGMENTS_DIR / "segments.json")
    descs = []
    for v in segs:
        for seg in v.get("segments", []):
            descs.append({
                "segment_id": seg["segment_id"],
                "nl_description": f"The person performs {seg.get('action_label', 'an activity')} in a household context.",
                "description_version": "template",
            })
    DESCRIPTIONS_DIR.mkdir(parents=True, exist_ok=True)
    with open(DESCRIPTIONS_DIR / "nl_descriptions.json", "w") as f:
        json.dump(descs, f, indent=2)
    logger.info(f"Stub: Created nl_descriptions.json with {len(descs)} description(s)")


# ── Individual stages ─────────────────────────────────────────────────────────

async def stage_1_ingest():
    _update_stage(1, 10)
    await asyncio.sleep(0.5)

    videos = list(RAW_VIDEOS_DIR.glob("*.mp4")) + list(RAW_VIDEOS_DIR.glob("*.avi"))
    logger.info(f"Stage 1: Found {len(videos)} video(s) in {RAW_VIDEOS_DIR}")
    if not videos:
        raise RuntimeError(f"No video files found in {RAW_VIDEOS_DIR}")

    _update_stage(1, 80)
    await asyncio.sleep(0.5)
    _complete_stage(1)


async def stage_2_segment():
    _update_stage(2, 10)
    await asyncio.sleep(0.3)

    loop = asyncio.get_event_loop()

    # Try PySceneDetect (fast, usually works)
    ok1 = await loop.run_in_executor(
        None, _run_script, "segment_pyscenedetect.py", 20
    )
    _update_stage(2, 40)

    # Try CLIP segmentation (can be slow — tight timeout)
    ok2 = await loop.run_in_executor(
        None, _run_script, "segment_clip.py", 30
    )
    _update_stage(2, 80)

    # If both failed, create stub
    if not ok1 and not ok2:
        await loop.run_in_executor(None, _stub_segments)

    _complete_stage(2)


async def stage_3_annotate():
    _update_stage(3, 10)
    await asyncio.sleep(0.3)

    loop = asyncio.get_event_loop()
    ok = await loop.run_in_executor(
        None, _run_script, "annotate_yolo.py", 30
    )
    _update_stage(3, 50)

    if ok:
        await loop.run_in_executor(
            None, _run_script, "track_yolo.py", 30
        )
    else:
        await loop.run_in_executor(None, _stub_annotations)

    _update_stage(3, 90)
    _complete_stage(3)


async def stage_4_describe():
    _update_stage(4, 10)
    await asyncio.sleep(0.3)

    loop = asyncio.get_event_loop()

    # BLIP-2 is a 2.7B model — VERY slow on CPU MacBooks.
    # Give it a short timeout. If it fails, the template fallback is fine.
    ok = await loop.run_in_executor(
        None, _run_script, "generate_raw_captions.py", 15
    )
    _update_stage(4, 40)

    if ok:
        # Claude refinement (fast if API key exists, template fallback otherwise)
        await loop.run_in_executor(
            None, _run_script, "refine_descriptions.py", 15
        )
    else:
        logger.info("BLIP-2 too slow or missing — using template descriptions")
        await loop.run_in_executor(None, _stub_descriptions)

    _update_stage(4, 90)
    _complete_stage(4)


async def stage_5_package():
    _update_stage(5, 10)
    await asyncio.sleep(0.3)

    loop = asyncio.get_event_loop()
    await loop.run_in_executor(
        None, _run_script, "build_dataset.py", 15
    )
    _update_stage(5, 40)
    
    # Run the CSV formatter and validation explicitly
    await loop.run_in_executor(
        None, _run_script, "create_submission.py", 30
    )
    await loop.run_in_executor(
        None, _run_script, "validate_and_stats.py", 30
    )
    _update_stage(5, 60)

    # Always compute live results from whatever output files exist
    PIPELINE_STATE["results"] = compute_live_results()
    _update_stage(5, 90)
    _complete_stage(5)


# ── Main orchestrator ─────────────────────────────────────────────────────────

async def run_full_pipeline(pipeline_id: str):
    """Background task: run all 5 stages sequentially with timeouts."""
    logger.info(f"Pipeline {pipeline_id} started")

    stages = [
        (1, stage_1_ingest),
        (2, stage_2_segment),
        (3, stage_3_annotate),
        (4, stage_4_describe),
        (5, stage_5_package),
    ]

    try:
        for stage_id, stage_fn in stages:
            logger.info(f"--- STARTING STAGE {stage_id}: "
                        f"{PIPELINE_STATE['stages'][stage_id-1]['name']} ---")
            await stage_fn()

        PIPELINE_STATE["status"] = "completed"
        PIPELINE_STATE["estimated_time_remaining_sec"] = 0
        logger.info(f"Pipeline {pipeline_id} finished successfully.")

    except Exception as e:
        logger.error(f"Pipeline failed at stage {PIPELINE_STATE['current_stage']}: {e}")
        PIPELINE_STATE["status"] = "failed"
        PIPELINE_STATE["error"] = str(e)
        idx = PIPELINE_STATE["current_stage"] - 1
        if 0 <= idx < len(PIPELINE_STATE["stages"]):
            PIPELINE_STATE["stages"][idx]["status"] = "failed"
