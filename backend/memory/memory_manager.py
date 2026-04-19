"""
Learning Memory Manager — the "brain" of the system.

This module handles:
  • Persistent JSON storage of processed tasks
  • Loading / saving the memory file
  • Storing new tasks
  • Jaccard-similarity-based retrieval of past tasks

Design decision: The JSON file acts as a lightweight database.
To swap it for Postgres/Mongo/Redis later, only this file needs
to change — the Pydantic models and API routes stay the same.
"""

import json
import logging
from pathlib import Path
from typing import List, Tuple

from memory.models import TaskRecord

logger = logging.getLogger(__name__)

# Default location for the memory file
DEFAULT_MEMORY_PATH = Path(__file__).parent.parent / "data" / "memory.json"


class MemoryManager:
    """
    Core intelligence module.

    Responsibilities:
      1. Persist tasks to a JSON file (simulates a database)
      2. Retrieve all memories
      3. Find similar past tasks using Jaccard similarity
    """

    def __init__(self, memory_path: Path = DEFAULT_MEMORY_PATH):
        self.memory_path = memory_path
        # Auto-create the data directory and file if they don't exist
        self._ensure_file()
        logger.info(f"MemoryManager initialized — storage: {self.memory_path}")

    # ── File I/O ──────────────────────────────────────────────────────────────

    def _ensure_file(self):
        """Create the data directory and an empty JSON array if the file is missing."""
        self.memory_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.memory_path.exists():
            self.memory_path.write_text("[]")
            logger.info(f"Created new memory file: {self.memory_path}")

    def load_memory(self) -> List[TaskRecord]:
        """
        Load all tasks from the JSON file.
        Returns a list of validated TaskRecord objects.
        """
        try:
            raw = json.loads(self.memory_path.read_text())
            tasks = [TaskRecord(**entry) for entry in raw]
            logger.debug(f"Loaded {len(tasks)} task(s) from memory")
            return tasks
        except (json.JSONDecodeError, Exception) as e:
            logger.error(f"Failed to load memory: {e} — resetting to empty")
            self.memory_path.write_text("[]")
            return []

    def save_memory(self, tasks: List[TaskRecord]):
        """
        Persist the full task list back to the JSON file.
        Each TaskRecord is serialized via .model_dump() for clean JSON output.
        """
        data = [task.model_dump() for task in tasks]
        self.memory_path.write_text(json.dumps(data, indent=2))
        logger.debug(f"Saved {len(tasks)} task(s) to memory")

    # ── Core operations ───────────────────────────────────────────────────────

    def store_task(self, task: TaskRecord) -> TaskRecord:
        """
        Add a new task to persistent memory.

        If a task with the same task_id already exists, it is overwritten
        (upsert behavior) so that re-processing a video updates the record
        instead of duplicating it.
        """
        tasks = self.load_memory()

        # Upsert: replace existing task with same ID, or append new
        existing_ids = {t.task_id for t in tasks}
        if task.task_id in existing_ids:
            tasks = [t if t.task_id != task.task_id else task for t in tasks]
            logger.info(f"Updated existing task in memory: {task.task_id}")
        else:
            tasks.append(task)
            logger.info(f"Stored new task in memory: {task.task_id}")

        self.save_memory(tasks)
        return task

    def get_all_memories(self) -> List[TaskRecord]:
        """Return every task currently in memory."""
        return self.load_memory()

    def get_task_by_id(self, task_id: str) -> TaskRecord | None:
        """Look up a single task by its ID."""
        for task in self.load_memory():
            if task.task_id == task_id:
                return task
        return None

    def delete_task(self, task_id: str) -> bool:
        """Remove a task from memory. Returns True if it existed."""
        tasks = self.load_memory()
        before = len(tasks)
        tasks = [t for t in tasks if t.task_id != task_id]
        if len(tasks) < before:
            self.save_memory(tasks)
            logger.info(f"Deleted task: {task_id}")
            return True
        return False

    def clear_memory(self):
        """Wipe all tasks from memory."""
        self.save_memory([])
        logger.info("Memory cleared")

    # ── Similarity search ─────────────────────────────────────────────────────

    @staticmethod
    def _jaccard_similarity(set_a: set, set_b: set) -> float:
        """
        Jaccard similarity = |A ∩ B| / |A ∪ B|

        Returns a value between 0.0 (no overlap) and 1.0 (identical sets).
        This is a clean, interpretable metric for comparing action sets
        without needing heavy ML embeddings.
        """
        if not set_a and not set_b:
            return 0.0
        intersection = set_a & set_b
        union = set_a | set_b
        return len(intersection) / len(union)

    def query_similar_tasks(
        self,
        actions: List[str],
        top_k: int = 3,
    ) -> List[Tuple[TaskRecord, float]]:
        """
        Find the most similar past tasks based on Jaccard similarity
        of their action sets.

        Args:
            actions: List of action names to search for (e.g. ["grasping", "pouring"])
            top_k:   Number of top results to return

        Returns:
            List of (TaskRecord, similarity_score) tuples, sorted by
            similarity descending.  Only tasks with similarity > 0 are
            included (we skip completely unrelated memories).
        """
        # Normalize query actions to lowercase for case-insensitive matching
        query_set = {a.strip().lower() for a in actions if a.strip()}

        if not query_set:
            return []

        tasks = self.load_memory()
        scored: List[Tuple[TaskRecord, float]] = []

        for task in tasks:
            # Build the action set for this stored task (also lowercase)
            task_actions = {
                a.action_name.strip().lower()
                for a in task.actions
                if a.action_name.strip()
            }

            sim = self._jaccard_similarity(query_set, task_actions)

            # Only keep tasks with non-zero similarity
            if sim > 0:
                scored.append((task, round(sim, 4)))

        # Sort by similarity descending, then by timestamp descending (newest first)
        scored.sort(key=lambda x: (-x[1], x[0].timestamp), reverse=False)

        return scored[:top_k]

    # ── Stats ─────────────────────────────────────────────────────────────────

    def get_stats(self) -> dict:
        """Return high-level statistics about the memory."""
        tasks = self.load_memory()
        all_actions = set()
        for t in tasks:
            for a in t.actions:
                all_actions.add(a.action_name.lower())

        timestamps = sorted([t.timestamp for t in tasks]) if tasks else []

        return {
            "total_tasks": len(tasks),
            "unique_actions": sorted(all_actions),
            "oldest_task": timestamps[0] if timestamps else None,
            "newest_task": timestamps[-1] if timestamps else None,
        }
