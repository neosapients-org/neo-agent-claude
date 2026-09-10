"""
Unit tests for NeoMemoryConnector.

Tests the connector with mocked AsyncMemory to verify:
- All Mem0 operations are properly delegated
- Value-add methods work correctly
- Operation logging functions properly
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from neo_memory_hub.integrations.connector import (
    NeoMemoryConnector,
    ConnectorConfig,
)


class TestNeoMemoryConnectorUnit:
    """Unit tests for NeoMemoryConnector with mocked AsyncMemory."""

    @pytest.fixture
    def mock_async_memory(self):
        """Create mock AsyncMemory."""
        memory = AsyncMock()
        memory.add = AsyncMock(return_value={
            "results": [{"id": "mem-1", "memory": "Test fact", "event": "ADD"}]
        })
        memory.search = AsyncMock(return_value={
            "results": [
                {"id": "mem-1", "memory": "User prefers dark mode", "score": 0.9},
                {"id": "mem-2", "memory": "User is a developer", "score": 0.8},
            ]
        })
        memory.get = AsyncMock(return_value={
            "id": "mem-1", "memory": "User prefers dark mode"
        })
        memory.get_all = AsyncMock(return_value={
            "results": [
                {"id": "mem-1", "memory": "Fact 1"},
                {"id": "mem-2", "memory": "Fact 2"},
            ]
        })
        memory.update = AsyncMock(return_value={
            "message": "Memory updated successfully!"
        })
        memory.delete = AsyncMock(return_value={
            "message": "Memory deleted successfully!"
        })
        memory.delete_all = AsyncMock(return_value={
            "message": "Memories deleted successfully!"
        })
        return memory

    @pytest.fixture
    def connector(self, mock_async_memory):
        """Create connector with mocked memory."""
        connector = NeoMemoryConnector(
            config=ConnectorConfig(
                log_operations=True,
                use_storage_validation=False,
            )
        )
        connector._memory = mock_async_memory
        connector._initialized = True
        return connector

    # =========================================================================
    # Core Mem0 Operations
    # =========================================================================

    @pytest.mark.asyncio
    async def test_add(self, connector, mock_async_memory):
        """add() should delegate to AsyncMemory."""
        result = await connector.add(
            "User prefers dark mode",
            user_id="alex",
            categories=["preferences"],
        )

        assert "results" in result
        assert len(result["results"]) == 1
        mock_async_memory.add.assert_called_once()
        call_kwargs = mock_async_memory.add.call_args[1]
        assert call_kwargs["user_id"] == "alex"
        assert call_kwargs["categories"] == ["preferences"]

    @pytest.mark.asyncio
    async def test_add_with_infer_false(self, connector, mock_async_memory):
        """add() with infer=False should pass through."""
        await connector.add(
            "Raw fact",
            user_id="alex",
            infer=False,
        )

        call_kwargs = mock_async_memory.add.call_args[1]
        assert call_kwargs["infer"] is False

    @pytest.mark.asyncio
    async def test_search(self, connector, mock_async_memory):
        """search() should delegate to AsyncMemory."""
        result = await connector.search(
            "preferences",
            user_id="alex",
            limit=5,
        )

        assert "results" in result
        assert len(result["results"]) == 2
        mock_async_memory.search.assert_called_once()

    @pytest.mark.asyncio
    async def test_search_with_threshold(self, connector, mock_async_memory):
        """search() should pass threshold to AsyncMemory."""
        await connector.search(
            "preferences",
            user_id="alex",
            threshold=0.5,
        )

        call_kwargs = mock_async_memory.search.call_args[1]
        assert call_kwargs["threshold"] == 0.5

    @pytest.mark.asyncio
    async def test_get(self, connector, mock_async_memory):
        """get() should retrieve memory by ID."""
        result = await connector.get("mem-1")

        assert result is not None
        assert result["id"] == "mem-1"
        mock_async_memory.get.assert_called_once_with("mem-1")

    @pytest.mark.asyncio
    async def test_get_all(self, connector, mock_async_memory):
        """get_all() should list all memories."""
        result = await connector.get_all(user_id="alex", limit=50)

        assert "results" in result
        assert len(result["results"]) == 2
        call_kwargs = mock_async_memory.get_all.call_args[1]
        assert call_kwargs["user_id"] == "alex"
        assert call_kwargs["limit"] == 50

    @pytest.mark.asyncio
    async def test_update(self, connector, mock_async_memory):
        """update() should update memory."""
        result = await connector.update("mem-1", "Updated content")

        assert "message" in result
        mock_async_memory.update.assert_called_once_with("mem-1", "Updated content")

    @pytest.mark.asyncio
    async def test_delete(self, connector, mock_async_memory):
        """delete() should delete memory."""
        result = await connector.delete("mem-1")

        assert "message" in result
        mock_async_memory.delete.assert_called_once_with("mem-1")

    @pytest.mark.asyncio
    async def test_delete_all(self, connector, mock_async_memory):
        """delete_all() should delete all memories for scope."""
        result = await connector.delete_all(user_id="alex")

        assert "message" in result
        call_kwargs = mock_async_memory.delete_all.call_args[1]
        assert call_kwargs["user_id"] == "alex"

    # =========================================================================
    # Value-Add Operations
    # =========================================================================

    @pytest.mark.asyncio
    async def test_build_context(self, connector, mock_async_memory):
        """build_context() should search and format."""
        context = await connector.build_context(
            "preferences",
            user_id="alex",
        )

        assert isinstance(context, str)
        assert "User prefers dark mode" in context
        mock_async_memory.search.assert_called_once()

    @pytest.mark.asyncio
    async def test_build_context_with_scores(self, connector, mock_async_memory):
        """build_context() should include scores when requested."""
        context = await connector.build_context(
            "preferences",
            user_id="alex",
            include_scores=True,
        )

        assert "relevance:" in context

    @pytest.mark.asyncio
    async def test_store_exchange(self, connector, mock_async_memory):
        """store_exchange() should store user-assistant pair."""
        result = await connector.store_exchange(
            user_message="Hello",
            assistant_response="Hi there!",
            user_id="alex",
        )

        assert "results" in result
        # Should call add once with messages list
        mock_async_memory.add.assert_called_once()
        # Messages are passed as first positional arg or as keyword
        call_args, call_kwargs = mock_async_memory.add.call_args
        messages = call_args[0] if call_args else call_kwargs.get("messages", [])
        assert len(messages) == 2
        assert messages[0]["role"] == "user"
        assert messages[1]["role"] == "assistant"

    @pytest.mark.asyncio
    async def test_store_fact(self, connector, mock_async_memory):
        """store_fact() should store with infer=False."""
        result = await connector.store_fact(
            "User's favorite color is blue",
            user_id="alex",
            categories=["preferences"],
        )

        assert "results" in result
        call_kwargs = mock_async_memory.add.call_args[1]
        assert call_kwargs["infer"] is False
        assert "fact" in call_kwargs["metadata"]

    @pytest.mark.asyncio
    async def test_store_preference(self, connector, mock_async_memory):
        """store_preference() should store with preference category."""
        result = await connector.store_preference(
            "User prefers dark mode",
            user_id="alex",
        )

        assert "results" in result
        call_kwargs = mock_async_memory.add.call_args[1]
        assert call_kwargs["infer"] is False
        assert "preference" in call_kwargs["categories"]

    # =========================================================================
    # Format Memories
    # =========================================================================

    def test_format_memories_empty(self, connector):
        """format_memories() should handle empty results."""
        result = connector.format_memories({"results": []})
        assert result == ""

    def test_format_memories_basic(self, connector):
        """format_memories() should format results."""
        search_results = {
            "results": [
                {"memory": "Fact 1", "score": 0.9},
                {"memory": "Fact 2", "score": 0.8},
            ]
        }
        result = connector.format_memories(search_results)

        assert "Relevant information from memory:" in result
        assert "- Fact 1" in result
        assert "- Fact 2" in result

    def test_format_memories_with_scores(self, connector):
        """format_memories() should include scores."""
        search_results = {
            "results": [{"memory": "Fact 1", "score": 0.9}]
        }
        result = connector.format_memories(search_results, include_scores=True)

        assert "relevance: 0.90" in result

    # =========================================================================
    # Operation Logging
    # =========================================================================

    @pytest.mark.asyncio
    async def test_operation_logging(self, connector, mock_async_memory):
        """Operations should be logged."""
        await connector.add("Test", user_id="alex")

        logs = connector.get_operation_logs()
        assert len(logs) == 1
        assert logs[0]["operation"] == "add"
        assert logs[0]["success"] is True
        assert logs[0]["user_id"] == "alex"

    @pytest.mark.asyncio
    async def test_operation_logging_filter(self, connector, mock_async_memory):
        """get_operation_logs() should filter by operation type."""
        await connector.add("Test", user_id="alex")
        await connector.search("query", user_id="alex")

        add_logs = connector.get_operation_logs(operation="add")
        assert len(add_logs) == 1
        assert add_logs[0]["operation"] == "add"

        search_logs = connector.get_operation_logs(operation="search")
        assert len(search_logs) == 1
        assert search_logs[0]["operation"] == "search"

    def test_clear_operation_logs(self, connector):
        """clear_operation_logs() should clear all logs."""
        connector._operation_logs = [{"test": "log"}]
        connector.clear_operation_logs()
        assert len(connector._operation_logs) == 0

    # =========================================================================
    # Lifecycle
    # =========================================================================

    def test_ensure_initialized_raises(self):
        """_ensure_initialized() should raise if not initialized."""
        connector = NeoMemoryConnector()

        with pytest.raises(RuntimeError, match="not initialized"):
            connector._ensure_initialized()

    @pytest.mark.asyncio
    async def test_context_manager(self):
        """Should work as async context manager."""
        with patch("neo_memory_hub.integrations.connector.AsyncMemory") as MockMemory:
            mock_memory = AsyncMock()
            mock_memory.initialize = AsyncMock()
            mock_memory.close = AsyncMock()
            MockMemory.return_value = mock_memory

            async with NeoMemoryConnector() as connector:
                assert connector._initialized is True

            mock_memory.close.assert_called_once()
