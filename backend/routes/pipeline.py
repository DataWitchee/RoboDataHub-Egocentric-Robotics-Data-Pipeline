from datetime import datetime
from fastapi import APIRouter, BackgroundTasks
from pipeline_runner import PIPELINE_STATE, run_full_pipeline, reset_state

router = APIRouter()

@router.post("/run-pipeline")
async def trigger_pipeline(background_tasks: BackgroundTasks):
    """
    Triggers the full pipeline as a background task.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    pipeline_id = f"run_{timestamp}"
    
    reset_state(pipeline_id)
    
    background_tasks.add_task(run_full_pipeline, pipeline_id)
    
    return {
        "message": "Pipeline started",
        "pipeline_id": pipeline_id
    }

@router.get("/pipeline-status")
async def get_pipeline_status():
    """
    Returns real-time pipeline state.
    """
    # Create the response dictionary matching the frontend expectations exactly
    current_idx = PIPELINE_STATE["current_stage"] - 1 if PIPELINE_STATE["current_stage"] > 0 else 0
    stages_response = []
    
    for idx, stage in enumerate(PIPELINE_STATE["stages"]):
        stage_dict = {
            "id": stage["id"],
            "name": stage["name"],
            "status": stage["status"]
        }
        # Include progress only if the stage is currently running
        if stage["status"] == "running" and "progress" in stage:
             stage_dict["progress"] = stage["progress"]
        stages_response.append(stage_dict)

    return {
        "pipeline_id": PIPELINE_STATE["pipeline_id"],
        "current_stage": PIPELINE_STATE["current_stage"],
        "stage_name": PIPELINE_STATE["stage_name"],
        "progress": PIPELINE_STATE["progress"],
        "status": PIPELINE_STATE["status"],
        "stages": stages_response,
        "estimated_time_remaining_sec": PIPELINE_STATE["estimated_time_remaining_sec"],
        "error": PIPELINE_STATE["error"]
    }
