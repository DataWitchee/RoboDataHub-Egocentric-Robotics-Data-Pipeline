"""
Pydantic data models for the Learning Memory System.

These models define the exact schema for tasks stored in memory.
Using Pydantic makes it trivial to swap JSON storage for a real
database (Postgres, Mongo, etc.) later — the validation layer stays
the same.
"""

from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime


# ── Atomic sub-structures ─────────────────────────────────────────────────────

class ActionItem(BaseModel):
    """A single detected action with its confidence score."""
    action_name: str = Field(..., description="Name of the detected action (e.g. 'grasping', 'pouring')")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Model confidence 0–1")


class StepItem(BaseModel):
    """One temporal step within a task (a sub-segment of the action)."""
    step_description: str = Field(..., description="Human-readable description of the step")
    start_time: float = Field(..., ge=0.0, description="Start time in seconds")
    end_time: float = Field(..., ge=0.0, description="End time in seconds")


class FailureItem(BaseModel):
    """A predicted failure mode for the task."""
    probability: float = Field(..., ge=0.0, le=1.0, description="Failure probability 0–1")
    reason: str = Field(..., description="Why this failure might occur")


# ── Full task record ──────────────────────────────────────────────────────────

class TaskRecord(BaseModel):
    """
    A complete task as stored in the Learning Memory.

    This is the atomic unit of memory — one processed video segment's
    full analysis: what actions were detected, what steps were extracted,
    what failures are predicted, and the robot-readable intent.
    """
    task_id: str = Field(..., description="Unique identifier for this task")
    actions: List[ActionItem] = Field(default_factory=list, description="Detected actions with confidence")
    steps: List[StepItem] = Field(default_factory=list, description="Sequential steps extracted from the video")
    failures: List[FailureItem] = Field(default_factory=list, description="Predicted failure modes")
    intent: str = Field(default="", description="Robot-readable intent string")
    timestamp: str = Field(
        default_factory=lambda: datetime.now().isoformat(),
        description="ISO timestamp of when this task was stored"
    )


# ── API request / response models ────────────────────────────────────────────

class QueryRequest(BaseModel):
    """Request body for the similarity query endpoint."""
    actions: List[str] = Field(..., description="List of action names to search for (e.g. ['grasping', 'pouring'])")
    top_k: int = Field(default=3, ge=1, le=50, description="Number of most similar tasks to return")


class SimilarTaskResult(BaseModel):
    """A single search result with similarity score attached."""
    task: TaskRecord
    similarity: float = Field(..., ge=0.0, le=1.0, description="Jaccard similarity score")


class QueryResponse(BaseModel):
    """Response from the similarity query endpoint."""
    query_actions: List[str]
    similar_tasks: List[SimilarTaskResult]
    total_memories: int = Field(..., description="Total number of tasks in memory")


class MemoryStatsResponse(BaseModel):
    """High-level stats about the memory system."""
    total_tasks: int
    unique_actions: List[str]
    oldest_task: Optional[str] = None
    newest_task: Optional[str] = None
