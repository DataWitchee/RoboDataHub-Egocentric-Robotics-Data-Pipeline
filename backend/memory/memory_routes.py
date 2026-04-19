"""
FastAPI routes for the Learning Memory System.

Endpoints:
  GET  /memory/           → list all stored tasks
  GET  /memory/stats      → memory statistics
  GET  /memory/{task_id}  → get a single task by ID
  POST /memory/store      → store a new task
  POST /memory/query      → similarity search
  DELETE /memory/{task_id} → delete a task
  DELETE /memory/          → clear all memory
"""

import logging
from fastapi import APIRouter, HTTPException

from memory.memory_manager import MemoryManager
from memory.models import (
    TaskRecord,
    QueryRequest,
    QueryResponse,
    SimilarTaskResult,
    MemoryStatsResponse,
)

router = APIRouter(prefix="/memory", tags=["Learning Memory"])
logger = logging.getLogger(__name__)

# Singleton memory manager — shared across all requests
memory = MemoryManager()


# ── GET  /memory/ — List all tasks ────────────────────────────────────────────

@router.get("/", response_model=list[TaskRecord])
async def get_all_memories():
    """
    Retrieve every task currently stored in the learning memory.
    Returns an empty list if no tasks have been stored yet.
    """
    tasks = memory.get_all_memories()
    logger.info(f"GET /memory/ → returning {len(tasks)} task(s)")
    return tasks


# ── GET  /memory/stats — Memory statistics ────────────────────────────────────

@router.get("/stats", response_model=MemoryStatsResponse)
async def get_memory_stats():
    """
    Return high-level statistics: total tasks, unique actions,
    oldest and newest timestamps.
    """
    return memory.get_stats()


# ── GET  /memory/{task_id} — Get a single task ───────────────────────────────

@router.get("/{task_id}", response_model=TaskRecord)
async def get_task(task_id: str):
    """Retrieve a single task by its ID."""
    task = memory.get_task_by_id(task_id)
    if not task:
        raise HTTPException(status_code=404, detail=f"Task '{task_id}' not found in memory.")
    return task


# ── POST /memory/store — Store a task ─────────────────────────────────────────

@router.post("/store", response_model=TaskRecord)
async def store_task(task: TaskRecord):
    """
    Store a new task into the learning memory.

    If a task with the same task_id already exists, it will be
    overwritten (upsert behavior).

    This endpoint is called automatically by the pipeline after
    processing a video, but can also be used manually for testing.
    """
    stored = memory.store_task(task)
    logger.info(f"POST /memory/store → stored task '{stored.task_id}'")
    return stored


# ── POST /memory/query — Similarity search ───────────────────────────────────

@router.post("/query", response_model=QueryResponse)
async def query_similar_tasks(request: QueryRequest):
    """
    Find the most similar past tasks using Jaccard similarity
    on action names.

    Input:
      { "actions": ["grasping", "pouring"], "top_k": 3 }

    Output:
      {
        "query_actions": ["grasping", "pouring"],
        "similar_tasks": [ { "task": {...}, "similarity": 0.67 }, ... ],
        "total_memories": 42
      }

    Only tasks with similarity > 0 are returned (completely
    unrelated tasks are omitted).
    """
    results = memory.query_similar_tasks(
        actions=request.actions,
        top_k=request.top_k,
    )

    similar = [
        SimilarTaskResult(task=task, similarity=score)
        for task, score in results
    ]

    all_tasks = memory.get_all_memories()
    logger.info(
        f"POST /memory/query → query={request.actions}, "
        f"found {len(similar)} similar task(s) out of {len(all_tasks)}"
    )

    return QueryResponse(
        query_actions=request.actions,
        similar_tasks=similar,
        total_memories=len(all_tasks),
    )


# ── DELETE /memory/{task_id} — Delete a single task ──────────────────────────

@router.delete("/{task_id}")
async def delete_task(task_id: str):
    """Remove a task from memory by its ID."""
    deleted = memory.delete_task(task_id)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Task '{task_id}' not found.")
    return {"deleted": task_id, "status": "ok"}


# ── DELETE /memory/ — Clear all memory ────────────────────────────────────────

@router.delete("/")
async def clear_memory():
    """Wipe all tasks from memory. Use with caution."""
    memory.clear_memory()
    logger.info("DELETE /memory/ → memory cleared")
    return {"status": "memory_cleared"}
