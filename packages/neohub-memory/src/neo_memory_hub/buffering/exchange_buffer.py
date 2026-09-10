"""Session-scoped exchange buffer for batching memory extraction.

Instead of running LLM fact extraction after every message, this buffer
collects (query, response) pairs per session and flushes them together when:
  1. flush_threshold exchanges reached
  2. Buffered text exceeds max_buffer_chars
  3. Session is idle for idle_timeout_seconds (configurable)
  4. Session ends (explicit flush_session call)
"""
from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class BufferedExchange:
    """A single buffered query+response pair."""

    query: str
    response: str
    user_id: str
    tenant_id: Optional[str] = None
    session_id: Optional[str] = None
    investor_name: Optional[str] = None
    agent_id: Optional[str] = None
    dept_id: Optional[str] = None
    timestamp: float = field(default_factory=time.time)


class ExchangeBuffer:
    """Session-scoped buffer. Thread-safe via asyncio.Lock per session.

    Supports four flush triggers:
    - **Count threshold**: Flushes when `flush_threshold` exchanges accumulate.
    - **Char threshold**: Flushes when total buffered chars >= `max_buffer_chars`.
    - **Idle timeout**: Flushes when no new exchange arrives for
      `idle_timeout_seconds` (default 60s). Requires calling `start_idle_monitor()`
      or the connector handles it automatically.
    - **Explicit flush**: `flush_session()` for session-end cleanup.

    Usage:
        buffer = ExchangeBuffer(flush_threshold=10, idle_timeout_seconds=60)
        ready = await buffer.add_exchange(query="...", response="...", user_id="u1")
        if ready:
            text = buffer.consolidate_exchanges(ready)
            # Send text through extraction pipeline
    """

    def __init__(
        self,
        flush_threshold: int = 10,
        max_buffer_chars: int = 50000,
        flush_on_session_end: bool = True,
        idle_timeout_seconds: float = 60.0,
    ) -> None:
        self._flush_threshold = flush_threshold
        self._max_buffer_chars = max_buffer_chars
        self._flush_on_session_end = flush_on_session_end
        self._idle_timeout_seconds = idle_timeout_seconds
        self._buffers: Dict[str, List[BufferedExchange]] = {}
        self._locks: Dict[str, asyncio.Lock] = {}
        self._last_activity: Dict[str, float] = {}
        self._idle_tasks: Dict[str, asyncio.Task] = {}
        self._flush_callback: Optional[Callable] = None

    @property
    def idle_timeout_seconds(self) -> float:
        return self._idle_timeout_seconds

    def set_flush_callback(self, callback: Callable) -> None:
        """Set the callback invoked when idle timeout triggers a flush.

        The callback receives (session_key: str, exchanges: List[BufferedExchange])
        and should be an async callable.
        """
        self._flush_callback = callback

    def _session_key(self, user_id: str, session_id: Optional[str]) -> str:
        return f"{user_id}:{session_id or 'default'}"

    def _get_lock(self, key: str) -> asyncio.Lock:
        if key not in self._locks:
            self._locks[key] = asyncio.Lock()
        return self._locks[key]

    async def add_exchange(
        self,
        query: str,
        response: str,
        user_id: str,
        tenant_id: Optional[str] = None,
        session_id: Optional[str] = None,
        investor_name: Optional[str] = None,
        agent_id: Optional[str] = None,
        dept_id: Optional[str] = None,
    ) -> Optional[List[BufferedExchange]]:
        """Add exchange. Returns exchanges to flush if threshold reached."""
        key = self._session_key(user_id, session_id)
        lock = self._get_lock(key)

        async with lock:
            if key not in self._buffers:
                self._buffers[key] = []

            self._buffers[key].append(
                BufferedExchange(
                    query=query,
                    response=response,
                    user_id=user_id,
                    tenant_id=tenant_id,
                    session_id=session_id,
                    investor_name=investor_name,
                    agent_id=agent_id,
                    dept_id=dept_id,
                )
            )

            # Update last activity timestamp
            self._last_activity[key] = time.time()

            buffer = self._buffers[key]
            total_chars = sum(len(e.query) + len(e.response) for e in buffer)

            if (
                len(buffer) >= self._flush_threshold
                or total_chars >= self._max_buffer_chars
            ):
                self._cancel_idle_timer(key)
                return self._drain(key)

        # (Re)start idle timer outside lock
        self._restart_idle_timer(key)
        return None

    async def flush_session(
        self, user_id: str, session_id: Optional[str] = None
    ) -> Optional[List[BufferedExchange]]:
        """Flush all buffered exchanges for a session."""
        if not self._flush_on_session_end:
            return None
        key = self._session_key(user_id, session_id)
        lock = self._get_lock(key)
        async with lock:
            self._cancel_idle_timer(key)
            return self._drain(key)

    def _drain(self, key: str) -> Optional[List[BufferedExchange]]:
        buffer = self._buffers.pop(key, None)
        self._last_activity.pop(key, None)
        # Clean up completed idle task reference to prevent memory leak
        self._idle_tasks.pop(key, None)
        return buffer if buffer else None

    # =========================================================================
    # Idle Timeout
    # =========================================================================

    def _restart_idle_timer(self, key: str) -> None:
        """Cancel existing idle timer for key and start a new one."""
        if self._idle_timeout_seconds <= 0:
            return
        self._cancel_idle_timer(key)
        try:
            loop = asyncio.get_running_loop()
            self._idle_tasks[key] = loop.create_task(
                self._idle_flush_task(key)
            )
        except RuntimeError:
            # No running event loop (e.g. during sync tests)
            pass

    def _cancel_idle_timer(self, key: str) -> None:
        """Cancel pending idle flush timer for a session."""
        task = self._idle_tasks.pop(key, None)
        if task and not task.done():
            task.cancel()

    async def _idle_flush_task(self, key: str) -> None:
        """Wait for idle_timeout_seconds, then flush if still idle."""
        try:
            await asyncio.sleep(self._idle_timeout_seconds)
        except asyncio.CancelledError:
            return

        lock = self._get_lock(key)
        async with lock:
            last = self._last_activity.get(key, 0)
            elapsed = time.time() - last
            if elapsed < self._idle_timeout_seconds:
                # Activity happened while we were sleeping; re-schedule
                return

            exchanges = self._drain(key)
            if exchanges and self._flush_callback:
                try:
                    await self._flush_callback(key, exchanges)
                except Exception as e:
                    logger.error(f"[ExchangeBuffer] Idle flush callback failed: {e}")

    def get_last_activity(
        self, user_id: str, session_id: Optional[str] = None
    ) -> Optional[float]:
        """Get the timestamp of the last exchange for a session."""
        key = self._session_key(user_id, session_id)
        return self._last_activity.get(key)

    def get_idle_seconds(
        self, user_id: str, session_id: Optional[str] = None
    ) -> float:
        """Get seconds since last activity for a session (0.0 if no buffer)."""
        last = self.get_last_activity(user_id, session_id)
        if last is None:
            return 0.0
        return time.time() - last

    # =========================================================================
    # Utilities
    # =========================================================================

    def get_buffer_size(
        self, user_id: str, session_id: Optional[str] = None
    ) -> int:
        key = self._session_key(user_id, session_id)
        return len(self._buffers.get(key, []))

    def consolidate_exchanges(self, exchanges: List[BufferedExchange]) -> str:
        """Consolidate multiple exchanges into LLM-ready transcript text."""
        parts = []
        for i, ex in enumerate(exchanges, 1):
            parts.append(f"--- Exchange {i} ---")
            parts.append(f"User: {ex.query}")
            parts.append(f"Assistant: {ex.response}")
            parts.append("")
        return "\n".join(parts)

    def cleanup_session(
        self, user_id: str, session_id: Optional[str] = None
    ) -> None:
        key = self._session_key(user_id, session_id)
        self._cancel_idle_timer(key)
        self._buffers.pop(key, None)
        self._locks.pop(key, None)
        self._last_activity.pop(key, None)

    @property
    def active_sessions(self) -> int:
        return len(self._buffers)
