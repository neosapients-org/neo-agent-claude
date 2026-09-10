"""Tests for MemoryHistoryManager."""
from unittest.mock import MagicMock, AsyncMock

import pytest

from neo_memory_hub.core.history import MemoryHistoryManager


class TestMemoryHistoryManagerInit:
    """Test initialization."""

    def test_init_with_none(self):
        mgr = MemoryHistoryManager(None)
        assert mgr._mem0 is None

    def test_init_with_mock(self):
        mock_mem0 = MagicMock()
        mgr = MemoryHistoryManager(mock_mem0)
        assert mgr._mem0 is mock_mem0


class TestGetMemoryHistory:
    """Test get_memory_history."""

    @pytest.mark.asyncio
    async def test_returns_empty_when_no_mem0(self):
        mgr = MemoryHistoryManager(None)
        result = await mgr.get_memory_history("mem_123")
        assert result == []

    @pytest.mark.asyncio
    async def test_calls_mem0_history(self):
        mock_mem0 = AsyncMock()
        mock_mem0.history = AsyncMock(return_value=[
            {"id": "h1", "event": "ADD", "memory_id": "mem_123"},
        ])
        mgr = MemoryHistoryManager(mock_mem0)
        result = await mgr.get_memory_history("mem_123")
        assert len(result) == 1
        assert result[0]["event"] == "ADD"
        mock_mem0.history.assert_called_once_with("mem_123")

    @pytest.mark.asyncio
    async def test_handles_exception_gracefully(self):
        mock_mem0 = AsyncMock()
        mock_mem0.history = AsyncMock(side_effect=Exception("DB error"))
        mgr = MemoryHistoryManager(mock_mem0)
        result = await mgr.get_memory_history("mem_123")
        assert result == []


class TestGetAllHistory:
    """Test get_all_history."""

    def test_returns_empty_when_no_mem0(self):
        mgr = MemoryHistoryManager(None)
        result = mgr.get_all_history()
        assert result == []

    def test_returns_empty_when_no_db(self):
        mock_mem0 = MagicMock(spec=[])  # No 'db' attribute
        mgr = MemoryHistoryManager(mock_mem0)
        result = mgr.get_all_history()
        assert result == []

    def test_returns_history_entries(self):
        mock_cursor = MagicMock()
        mock_cursor.fetchall.return_value = [
            ("h1", "mem1", "old text", "new text", "UPDATE", "2024-01-01", "2024-01-02", 0, "user1", "user"),
            ("h2", "mem2", None, "new text", "ADD", "2024-01-01", None, 0, "user1", "user"),
        ]
        mock_db = MagicMock()
        mock_db._lock = MagicMock()
        mock_db._lock.__enter__ = MagicMock(return_value=None)
        mock_db._lock.__exit__ = MagicMock(return_value=False)
        mock_db.connection.execute.return_value = mock_cursor

        mock_mem0 = MagicMock()
        mock_mem0.db = mock_db

        mgr = MemoryHistoryManager(mock_mem0)
        result = mgr.get_all_history(limit=10)

        assert len(result) == 2
        assert result[0]["event"] == "UPDATE"
        assert result[0]["memory_id"] == "mem1"
        assert result[1]["event"] == "ADD"
        assert result[1]["is_deleted"] is False

    def test_event_filter(self):
        mock_cursor = MagicMock()
        mock_cursor.fetchall.return_value = [
            ("h1", "mem1", None, "text", "ADD", "2024-01-01", None, 0, "u1", "user"),
        ]
        mock_db = MagicMock()
        mock_db._lock = MagicMock()
        mock_db._lock.__enter__ = MagicMock(return_value=None)
        mock_db._lock.__exit__ = MagicMock(return_value=False)
        mock_db.connection.execute.return_value = mock_cursor

        mock_mem0 = MagicMock()
        mock_mem0.db = mock_db

        mgr = MemoryHistoryManager(mock_mem0)
        result = mgr.get_all_history(limit=10, event_filter="ADD")

        assert len(result) == 1
        # Verify the SQL was called with filter
        call_args = mock_db.connection.execute.call_args
        assert "WHERE event = ?" in call_args[0][0]
        assert call_args[0][1] == ("ADD", 10)

    def test_handles_exception(self):
        mock_db = MagicMock()
        mock_db._lock = MagicMock()
        mock_db._lock.__enter__ = MagicMock(return_value=None)
        mock_db._lock.__exit__ = MagicMock(return_value=False)
        mock_db.connection.execute.side_effect = Exception("SQL error")

        mock_mem0 = MagicMock()
        mock_mem0.db = mock_db

        mgr = MemoryHistoryManager(mock_mem0)
        result = mgr.get_all_history()
        assert result == []


class TestGetStats:
    """Test get_stats."""

    def test_returns_disabled_when_no_mem0(self):
        mgr = MemoryHistoryManager(None)
        stats = mgr.get_stats()
        assert stats["total"] == 0
        assert stats["history_enabled"] is False

    def test_returns_stats(self):
        mock_cursor = MagicMock()
        mock_cursor.fetchall.return_value = [
            ("ADD", 10),
            ("UPDATE", 5),
            ("DELETE", 2),
        ]
        mock_db = MagicMock()
        mock_db._lock = MagicMock()
        mock_db._lock.__enter__ = MagicMock(return_value=None)
        mock_db._lock.__exit__ = MagicMock(return_value=False)
        mock_db.connection.execute.return_value = mock_cursor
        mock_db.db_path = "/path/to/history.db"

        mock_mem0 = MagicMock()
        mock_mem0.db = mock_db

        mgr = MemoryHistoryManager(mock_mem0)
        stats = mgr.get_stats()

        assert stats["total"] == 17
        assert stats["by_event"] == {"ADD": 10, "UPDATE": 5, "DELETE": 2}
        assert stats["history_enabled"] is True
        assert stats["persistent"] is True

    def test_memory_db_not_persistent(self):
        mock_cursor = MagicMock()
        mock_cursor.fetchall.return_value = [("ADD", 3)]
        mock_db = MagicMock()
        mock_db._lock = MagicMock()
        mock_db._lock.__enter__ = MagicMock(return_value=None)
        mock_db._lock.__exit__ = MagicMock(return_value=False)
        mock_db.connection.execute.return_value = mock_cursor
        mock_db.db_path = ":memory:"

        mock_mem0 = MagicMock()
        mock_mem0.db = mock_db

        mgr = MemoryHistoryManager(mock_mem0)
        stats = mgr.get_stats()
        assert stats["persistent"] is False

    def test_handles_exception(self):
        mock_db = MagicMock()
        mock_db._lock = MagicMock()
        mock_db._lock.__enter__ = MagicMock(return_value=None)
        mock_db._lock.__exit__ = MagicMock(return_value=False)
        mock_db.connection.execute.side_effect = Exception("SQL error")

        mock_mem0 = MagicMock()
        mock_mem0.db = mock_db

        mgr = MemoryHistoryManager(mock_mem0)
        stats = mgr.get_stats()
        assert stats["total"] == 0
        assert stats["history_enabled"] is False
        assert "error" in stats
