"""
Tests for operation callback hooks on NeoMemoryConnector (v0.2 — CR-9).
"""

from __future__ import annotations

import asyncio

import pytest
from unittest.mock import AsyncMock, MagicMock, call

from neo_memory_hub.integrations.connector import (
    ConnectorConfig,
    NeoMemoryConnector,
    OperationLog,
)


@pytest.fixture
def mock_async_memory():
    """Create mock AsyncMemory."""
    memory = AsyncMock()
    memory.add = AsyncMock(
        return_value={"results": [{"id": "m1", "memory": "fact", "event": "ADD"}]}
    )
    memory.search = AsyncMock(return_value={"results": []})
    memory.get_all = AsyncMock(return_value={"results": []})
    memory.delete = AsyncMock(return_value={"success": True})
    return memory


def _make_connector(mock_memory, on_operation=None) -> NeoMemoryConnector:
    connector = NeoMemoryConnector(
        config=ConnectorConfig(
            log_operations=True,
            use_storage_validation=False,
            on_operation=on_operation,
        )
    )
    connector._memory = mock_memory
    connector._initialized = True
    return connector


class TestOperationHooks:
    """Tests for on_operation callback."""

    @pytest.mark.asyncio
    async def test_sync_callback_invoked(self, mock_async_memory):
        """Synchronous callback should be called after operation."""
        callback = MagicMock()
        connector = _make_connector(mock_async_memory, on_operation=callback)

        await connector.add("hello", user_id="u1")

        assert callback.called
        log_entry = callback.call_args[0][0]
        assert log_entry["operation"] == "add"
        assert log_entry["success"] is True

    @pytest.mark.asyncio
    async def test_async_callback_invoked(self, mock_async_memory):
        """Async callback should be scheduled (ensure_future)."""
        received_logs: list[OperationLog] = []

        async def async_cb(log: OperationLog) -> None:
            received_logs.append(log)

        connector = _make_connector(mock_async_memory, on_operation=async_cb)

        await connector.add("hello", user_id="u1")
        # Give the event loop a chance to run the coroutine
        await asyncio.sleep(0.05)

        assert len(received_logs) >= 1
        assert received_logs[0]["operation"] == "add"

    @pytest.mark.asyncio
    async def test_callback_receives_correct_fields(self, mock_async_memory):
        """Callback should receive well-formed OperationLog."""
        callback = MagicMock()
        connector = _make_connector(mock_async_memory, on_operation=callback)

        await connector.add("data", user_id="u1", agent_id="a1")

        log_entry = callback.call_args[0][0]
        assert "timestamp" in log_entry
        assert "request_id" in log_entry
        assert log_entry["user_id"] == "u1"
        assert log_entry["agent_id"] == "a1"
        assert log_entry["error"] is None

    @pytest.mark.asyncio
    async def test_callback_error_does_not_crash(self, mock_async_memory):
        """If callback raises, the operation should still succeed."""

        def bad_callback(log: OperationLog) -> None:
            raise RuntimeError("Callback boom!")

        connector = _make_connector(mock_async_memory, on_operation=bad_callback)

        # Should NOT raise
        result = await connector.add("hello", user_id="u1")
        assert "results" in result

    @pytest.mark.asyncio
    async def test_no_callback_is_fine(self, mock_async_memory):
        """Connector with no callback should work without errors."""
        connector = _make_connector(mock_async_memory, on_operation=None)

        result = await connector.add("hello", user_id="u1")
        assert "results" in result

    @pytest.mark.asyncio
    async def test_callback_on_search(self, mock_async_memory):
        """Callback should fire for search operations too."""
        callback = MagicMock()
        connector = _make_connector(mock_async_memory, on_operation=callback)

        await connector.search("query", user_id="u1")

        log_entry = callback.call_args[0][0]
        assert log_entry["operation"] == "search"

    @pytest.mark.asyncio
    async def test_callback_on_failed_operation(self, mock_async_memory):
        """Callback should fire even when the operation fails."""
        mock_async_memory.add = AsyncMock(side_effect=RuntimeError("DB error"))
        callback = MagicMock()
        connector = _make_connector(mock_async_memory, on_operation=callback)

        with pytest.raises(RuntimeError, match="DB error"):
            await connector.add("fail", user_id="u1")

        log_entry = callback.call_args[0][0]
        assert log_entry["success"] is False
        assert "DB error" in log_entry["error"]
