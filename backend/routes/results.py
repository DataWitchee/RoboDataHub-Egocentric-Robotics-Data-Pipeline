import os
import zipfile
from datetime import datetime
from pathlib import Path
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pipeline_runner import PIPELINE_STATE, compute_live_results
from utils.zip_utils import create_zip_archive

router = APIRouter()

_data_root = Path(os.getenv("DATASET_ROOT", str(Path.home() / "Desktop" / "my_robotics_data")))
FINAL_DATASET_DIR = _data_root / "final_dataset"


@router.get("/results")
async def get_results():
    """
    Returns the LIVE pipeline results computed from actual output files.
    No hardcoded data — everything comes from segments.json,
    bbox_annotations.json, and nl_descriptions.json.
    """
    # 1. First try cached results from the pipeline run
    cached = PIPELINE_STATE.get("results")
    if cached:
        result_data = cached
    else:
        # 2. Otherwise compute fresh from whatever files exist on disk
        result_data = compute_live_results()

    # 3. If absolutely nothing exists, return a clear error
    if result_data["summary"]["total_segments"] == 0:
        raise HTTPException(
            status_code=404,
            detail="No pipeline results found. Run the pipeline first."
        )

    # 4. Attach raw CSV string if it exists
    csv_path = FINAL_DATASET_DIR / "submission.csv"
    if csv_path.exists():
        with open(csv_path, "r") as f:
            result_data["csv_content"] = f.read()
    else:
        result_data["csv_content"] = ""

    return result_data


@router.get("/download")
async def download_dataset():
    """
    Zips the entire final_dataset/ folder and returns it as a download.
    If final_dataset doesn't exist, zips whatever data directories we have.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    zip_filename = f"robodata_dataset_{timestamp}.zip"
    zip_filepath = f"/tmp/{zip_filename}"

    if FINAL_DATASET_DIR.exists() and any(FINAL_DATASET_DIR.iterdir()):
        create_zip_archive(FINAL_DATASET_DIR, zip_filepath)
    elif _data_root.exists():
        # Zip the entire data root if final_dataset wasn't generated
        create_zip_archive(_data_root, zip_filepath)
    else:
        # Last resort: create a zip with a note
        with zipfile.ZipFile(zip_filepath, 'w') as zipf:
            zipf.writestr("README.txt",
                          "No dataset files found. Run the pipeline first.")

    return FileResponse(
        path=zip_filepath,
        filename=zip_filename,
        media_type="application/zip",
    )
