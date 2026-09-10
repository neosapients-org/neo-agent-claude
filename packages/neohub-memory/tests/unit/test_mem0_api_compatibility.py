"""
Test mem0 API compatibility - verify we correctly pass parameters to mem0.

This test suite verifies all the bug fixes related to mem0 API compatibility:
1. categories parameter - should be in metadata, not direct param
2. Input validation - production grade validation
3. Parameter pass-through to mem0
"""

import pytest
from unittest.mock import AsyncMock

from neo_memory_hub.integrations.memory import AsyncMemory


class TestMem0APICompatibility:
    """Test that our AsyncMemory wrapper correctly handles mem0 API constraints."""

    @pytest.mark.asyncio
    async def test_async_add_moves_categories_to_metadata(self):
        """BUG FIX: categories should be in metadata, not passed directly to mem0.add()."""
        m = AsyncMemory()
        m._mem0 = AsyncMock()
        m._initialized = True

        # Call with categories
        await m.add("test content", user_id="user1", categories=["persona", "security"])

        # Verify mem0.add was called without categories param
        m._mem0.add.assert_awaited_once()
        call_args = m._mem0.add.call_args
        
        # First arg should be messages
        assert call_args[0][0] == "test content"
        
        # kwargs should NOT have categories key
        assert "categories" not in call_args[1]
        
        # But metadata should have categories
        assert "metadata" in call_args[1]
        assert call_args[1]["metadata"]["categories"] == ["persona", "security"]

    @pytest.mark.asyncio
    async def test_async_search_does_not_pass_categories(self):
        """BUG FIX: categories should be used for post-filtering, not passed to mem0.search()."""
        m = AsyncMemory()
        m._mem0 = AsyncMock()
        m._initialized = True
        
        # Mock return value with categories in metadata
        m._mem0.search.return_value = {
            "results": [
                {"memory": "mem1", "metadata": {"categories": ["persona"]}},
                {"memory": "mem2", "metadata": {"categories": ["security"]}},
                {"memory": "mem3", "metadata": {"categories": ["persona", "security"]}},
            ]
        }

        # Search with categories filter
        result = await m.search("query", user_id="user1", categories=["persona"])

        # Verify mem0.search was called without categories param
        m._mem0.search.assert_awaited_once()
        call_args = m._mem0.search.call_args
        assert "categories" not in call_args[1]
        
        # Verify post-filtering worked
        assert len(result["results"]) == 2
        for mem in result["results"]:
            assert "persona" in mem["metadata"]["categories"]

    @pytest.mark.asyncio
    async def test_async_add_validates_empty_string(self):
        """Production grade: Should validate empty input."""
        m = AsyncMemory()
        m._mem0 = AsyncMock()  # Set mock object
        m._initialized = True

        with pytest.raises(ValueError, match="messages cannot be empty"):
            await m.add("")

        with pytest.raises(ValueError, match="messages cannot be empty"):
            await m.add("   ")

    @pytest.mark.asyncio
    async def test_async_search_validates_empty_query(self):
        """Production grade: Should validate empty query."""
        m = AsyncMemory()
        m._mem0 = AsyncMock()  # Set mock object
        m._initialized = True

        with pytest.raises(ValueError, match="query cannot be empty"):
            await m.search("")

        with pytest.raises(ValueError, match="query cannot be empty"):
            await m.search("   ")

    @pytest.mark.asyncio
    async def test_async_update_validates_empty_inputs(self):
        """Production grade: Should validate empty memory_id and data."""
        m = AsyncMemory()
        m._mem0 = AsyncMock()  # Set mock object
        m._initialized = True

        with pytest.raises(ValueError, match="memory_id cannot be empty"):
            await m.update("", "data")

        with pytest.raises(ValueError, match="data cannot be empty"):
            await m.update("mem_123", "")

    # =========================================================================
    # Parameter Verification Tests
    # =========================================================================

    @pytest.mark.asyncio
    async def test_async_search_passes_only_supported_params(self):
        """Verify we only pass parameters that mem0.AsyncMemory.search() supports."""
        m = AsyncMemory()
        m._mem0 = AsyncMock()
        m._initialized = True
        m._mem0.search.return_value = {"results": []}

        # Call with all our params
        await m.search(
            "query",
            user_id="user1",
            agent_id="agent1",
            run_id="run1",
            limit=20,
            filters={"f1": "v1"},
            metadata_filters={"mf1": "v1"},
            rerank=True,
            threshold=0.7,
        )

        # Verify only supported params were passed
        call_kwargs = m._mem0.search.call_args[1]
        supported_keys = {
            "user_id",
            "agent_id",
            "run_id",
            "limit",
            "filters",
            "metadata_filters",
            "threshold",
            "rerank",
        }
        assert set(call_kwargs.keys()).issubset(supported_keys)


class TestThresholdWorkaround:
    """G-03: threshold=0.0 should be converted to 0.001 in NeoMemoryConnector."""

    def _make_connector(self):
        from neo_memory_hub.integrations.connector import NeoMemoryConnector, ConnectorConfig
        from neo_memory_hub.integrations.memory import MemoryConfig

        mem_config = MemoryConfig(
            vector_store={"provider": "qdrant", "config": {"url": "http://localhost:6333", "collection_name": "test", "embedding_model_dims": 1536}},
            llm={"provider": "openai", "config": {"model": "gpt-4o-mini"}},
            embedder={"provider": "openai", "config": {"model": "text-embedding-3-small"}},
        )
        connector = NeoMemoryConnector(
            memory_config=mem_config,
            config=ConnectorConfig(default_limit=10, default_threshold=0.1),
        )
        connector._memory = AsyncMock()
        connector._memory.search = AsyncMock(return_value={"results": []})
        connector._initialized = True
        return connector

    @pytest.mark.asyncio
    async def test_zero_threshold_converted(self):
        connector = self._make_connector()
        await connector.search("query", user_id="u1", threshold=0.0)

        call_kwargs = connector._memory.search.call_args[1]
        assert call_kwargs["threshold"] == 0.001

    @pytest.mark.asyncio
    async def test_nonzero_threshold_unchanged(self):
        connector = self._make_connector()
        await connector.search("query", user_id="u1", threshold=0.5)

        call_kwargs = connector._memory.search.call_args[1]
        assert call_kwargs["threshold"] == 0.5

    @pytest.mark.asyncio
    async def test_none_threshold_uses_default(self):
        connector = self._make_connector()
        await connector.search("query", user_id="u1", threshold=None)

        call_kwargs = connector._memory.search.call_args[1]
        assert call_kwargs["threshold"] == 0.1
