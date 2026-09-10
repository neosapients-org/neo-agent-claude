"""Mem0 history / audit trail API.

Mem0 tracks every memory operation in SQLite:
  ADD, UPDATE, DELETE, NONE
Each entry: memory_id, old_memory, new_memory, event, timestamps, actor_id, role.

This module provides a clean API to access this audit trail.
It wraps Mem0 internal SQLite access with defensive try/except so
that Mem0 version upgrades don't break the consumer.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_HISTORY_SELECT_COLS = (
    "id, memory_id, old_memory, new_memory, event, "
    "created_at, updated_at, is_deleted, actor_id, role"
)


def _row_to_dict(row: tuple) -> Dict[str, Any]:
    """Convert a raw SQLite row tuple to a dict."""
    return {
        "id": row[0],
        "memory_id": row[1],
        "old_memory": row[2],
        "new_memory": row[3],
        "event": row[4],
        "created_at": row[5],
        "updated_at": row[6],
        "is_deleted": bool(row[7]) if row[7] is not None else False,
        "actor_id": row[8],
        "role": row[9],
    }


def _get_db_connection(mem0_obj: Any):
    """Safely obtain the SQLite db object from a Mem0 instance.

    Returns (db, lock) on success or (None, None) if unavailable.
    Works across Mem0 v0.1.x and v0.2.x internal layouts.
    """
    db = getattr(mem0_obj, "db", None)
    if db is None:
        return None, None
    lock = getattr(db, "_lock", None)
    conn = getattr(db, "connection", None)
    if conn is None:
        return None, None
    return db, lock


class MemoryHistoryManager:
    """Clean API for Mem0's SQLite audit trail.

    Requires the underlying AsyncMemory instance from NeoMemoryConnector.
    """

    def __init__(self, mem0_memory: Any) -> None:
        self._mem0 = mem0_memory

    async def get_memory_history(self, memory_id: str) -> List[Dict[str, Any]]:
        """Get full change history for a specific memory."""
        if not self._mem0:
            return []
        try:
            return await self._mem0.history(memory_id)
        except Exception as e:
            logger.warning(f"[History] Failed for {memory_id}: {e}")
            return []

    def get_all_history(
        self,
        limit: int = 100,
        event_filter: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Get all history entries (newest first)."""
        db, lock = _get_db_connection(self._mem0)
        if db is None:
            return []
        try:
            conn = db.connection
            if lock:
                lock.acquire()
            try:
                if event_filter:
                    cur = conn.execute(
                        f"SELECT {_HISTORY_SELECT_COLS} FROM history "
                        "WHERE event = ? ORDER BY COALESCE(updated_at, created_at) DESC LIMIT ?",
                        (event_filter.upper(), limit),
                    )
                else:
                    cur = conn.execute(
                        f"SELECT {_HISTORY_SELECT_COLS} FROM history "
                        "ORDER BY COALESCE(updated_at, created_at) DESC LIMIT ?",
                        (limit,),
                    )
                return [_row_to_dict(r) for r in cur.fetchall()]
            finally:
                if lock:
                    lock.release()
        except Exception as e:
            logger.warning(f"[History] get_all failed: {e}")
            return []

    def get_stats(self) -> Dict[str, Any]:
        """Get summary statistics of memory operations."""
        db, lock = _get_db_connection(self._mem0)
        if db is None:
            return {"total": 0, "by_event": {}, "history_enabled": False}
        try:
            conn = db.connection
            if lock:
                lock.acquire()
            try:
                cur = conn.execute(
                    "SELECT event, COUNT(*) FROM history GROUP BY event"
                )
                by_event = {r[0]: r[1] for r in cur.fetchall()}
                total = sum(by_event.values())
            finally:
                if lock:
                    lock.release()
            return {
                "total": total,
                "by_event": by_event,
                "history_enabled": True,
                "history_db_path": getattr(db, "db_path", ":memory:"),
                "persistent": getattr(db, "db_path", ":memory:") != ":memory:",
            }
        except Exception as e:
            logger.warning(f"[History] get_stats failed: {e}")
            return {
                "total": 0,
                "by_event": {},
                "history_enabled": False,
                "error": str(e),
            }
