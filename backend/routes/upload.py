from pathlib import Path
from fastapi import APIRouter, UploadFile
from typing import List
from utils.file_utils import validate_and_save_file
import logging
import os

router = APIRouter()
logger = logging.getLogger(__name__)

# ── Path config ─────────────────────────────────────────────────────────────
# Default to ~/Desktop/my_robotics_data/raw_videos so it always works locally.
# Override by setting the RAW_VIDEOS_DIR environment variable.
_default = Path.home() / "Desktop" / "my_robotics_data" / "raw_videos"
RAW_VIDEOS_DIR = Path(os.getenv("RAW_VIDEOS_DIR", str(_default)))
RAW_VIDEOS_DIR.mkdir(parents=True, exist_ok=True)     # auto-create on startup

logger.info(f"Upload destination: {RAW_VIDEOS_DIR}")


@router.post("/upload")
async def upload_videos(videos: List[UploadFile]):
    """
    Accepts one or more video files via multipart/form-data.
    Field name must be 'videos'  — matches FormData.append('videos', ...) in the frontend.
    """
    if not videos:
        return {"uploaded": [], "total_files": 0, "total_size_mb": 0.0, "status": "no_files"}

    logger.info(f"Received upload request: {len(videos)} file(s).")

    total_size_mb = 0.0
    uploaded_files = []

    for video in videos:
        size_mb = validate_and_save_file(video, RAW_VIDEOS_DIR)
        total_size_mb += size_mb
        uploaded_files.append(video.filename)
        logger.info(f"  Saved: {video.filename} ({size_mb:.2f} MB)")

    logger.info(f"Upload complete: {len(uploaded_files)} files, {total_size_mb:.2f} MB total.")

    return {
        "uploaded": uploaded_files,
        "total_files": len(uploaded_files),
        "total_size_mb": round(total_size_mb, 2),
        "status": "ready"
    }
