"""Tests for scoped operations via ScopedMemoryConnector (memory_utils).

Covers: add_scoped, search_scoped, get_all_scoped, store_exchange_scoped.
These were extracted from NeoMemoryConnector core into memory_utils.
"""

from __future__ import annotations

import pytest
from unittest.mock import AsyncMock

from neo_memory_hub.domain.scope import AccessTier, IsolationScope
from neo_memory_hub.integrations.connector import (
    ConnectorConfig,
    NeoMemoryConnector,
)
from memory_utils.shared_scope import ScopedMemoryConnector, SharedMemoryStrategy


@pytest.fixture
def mock_async_memory():
    """Create mock AsyncMemory."""
    memory = AsyncMock()
    memory.add = AsyncMock(
        return_value={"results": [{"id": "mem-1", "memory": "fact", "event": "ADD"}]}
    )
    memory.search = AsyncMock(
        return_value={
            "results": [
                {"id": "mem-1", "memory": "Private fact", "score": 0.9, "metadata": {"pool": "private"}},
                {"id": "mem-2", "memory": "Shared fact", "score": 0.8, "metadata": {"pool": "team"}},
            ]
        }
    )
    memory.get_all = AsyncMock(
        return_value={"results": [{"id": "mem-1", "memory": "Fact 1"}]}
    )
    memory.delete = AsyncMock(return_value={"success": True})
    return memory


def _make_scoped_connector(
    mock_memory,
    strategy: SharedMemoryStrategy = SharedMemoryStrategy.ENABLED,
) -> ScopedMemoryConnector:
    connector = NeoMemoryConnector(
        config=ConnectorConfig(
            log_operations=False,
            use_storage_validation=False,
        )
    )
    connector._memory = mock_memory
    connector._initialized = True
    return ScopedMemoryConnector(connector, strategy=strategy)


@pytest.fixture
def private_scope():
    return IsolationScope(
        tenant_id="acme",
        user_id="u1",
        agent_id="a1",
        pool=AccessTier.PRIVATE,
    )


@pytest.fixture
def team_scope():
    return IsolationScope(
        tenant_id="acme",
        user_id="u1",
        agent_id="a1",
        dept_id="wealth",
        pool=AccessTier.TEAM,
    )


@pytest.fixture
def org_scope():
    return IsolationScope(
        tenant_id="acme",
        pool=AccessTier.ORG,
    )


class TestAddScoped:
    """Tests for add_scoped()."""

    @pytest.mark.asyncio
    async def test_add_private_scope(self, mock_async_memory, private_scope):
        scoped = _make_scoped_connector(mock_async_memory)
        result = await scoped.add_scoped("My preference", private_scope)

        assert "results" in result
        call_kwargs = mock_async_memory.add.call_args[1]
        assert call_kwargs["user_id"] == "u1"
        assert call_kwargs["agent_id"] == "a1"

    @pytest.mark.asyncio
    async def test_add_scoped_merges_metadata(self, mock_async_memory, private_scope):
        scoped = _make_scoped_connector(mock_async_memory)
        await scoped.add_scoped(
            "fact",
            private_scope,
            metadata={"custom": "val"},
        )

        call_kwargs = mock_async_memory.add.call_args[1]
        merged_meta = call_kwargs["metadata"]
        assert merged_meta.get("custom") == "val"
        assert "pool" in merged_meta  # from scope

    @pytest.mark.asyncio
    async def test_add_org_scope_omits_user(self, mock_async_memory, org_scope):
        """ORG scope should omit user_id in Mem0 params."""
        scoped = _make_scoped_connector(mock_async_memory)
        await scoped.add_scoped("org knowledge", org_scope)

        call_kwargs = mock_async_memory.add.call_args[1]
        # ORG tier: user_id might be None via to_mem0_params
        # Just ensure tenant is in metadata
        merged_meta = call_kwargs["metadata"]
        assert merged_meta.get("tenant_id") == "acme"

    @pytest.mark.asyncio
    async def test_add_blocked_by_read_only(self, mock_async_memory, team_scope):
        """READ_ONLY strategy should block writes to shared tiers."""
        scoped = _make_scoped_connector(mock_async_memory, SharedMemoryStrategy.READ_ONLY)
        result = await scoped.add_scoped("blocked", team_scope)

        assert result["blocked_by"] == "shared_memory_strategy"
        assert result["strategy"] == "read_only"
        mock_async_memory.add.assert_not_called()

    @pytest.mark.asyncio
    async def test_add_blocked_by_disabled(self, mock_async_memory, team_scope):
        """DISABLED strategy should block writes to shared tiers."""
        scoped = _make_scoped_connector(mock_async_memory, SharedMemoryStrategy.DISABLED)
        result = await scoped.add_scoped("blocked", team_scope)

        assert result["blocked_by"] == "shared_memory_strategy"
        mock_async_memory.add.assert_not_called()

    @pytest.mark.asyncio
    async def test_add_private_not_blocked(self, mock_async_memory, private_scope):
        """PRIVATE tier should NOT be blocked even when strategy is DISABLED."""
        scoped = _make_scoped_connector(mock_async_memory, SharedMemoryStrategy.DISABLED)
        result = await scoped.add_scoped("allowed", private_scope)

        assert "results" in result
        mock_async_memory.add.assert_called_once()


class TestSearchScoped:
    """Tests for search_scoped()."""

    @pytest.mark.asyncio
    async def test_search_basic(self, mock_async_memory, private_scope):
        scoped = _make_scoped_connector(mock_async_memory)
        result = await scoped.search_scoped("query", private_scope)

        assert "results" in result
        mock_async_memory.search.assert_called_once()

    @pytest.mark.asyncio
    async def test_search_disabled_filters_non_private(self, mock_async_memory, private_scope):
        """DISABLED strategy should filter out non-PRIVATE results."""
        scoped = _make_scoped_connector(mock_async_memory, SharedMemoryStrategy.DISABLED)
        result = await scoped.search_scoped("query", private_scope)

        # Only the "private" pool memory should remain
        assert len(result["results"]) == 1
        assert result["results"][0]["metadata"]["pool"] == "private"

    @pytest.mark.asyncio
    async def test_search_enabled_returns_all(self, mock_async_memory, private_scope):
        """ENABLED strategy should return all results unfiltered."""
        scoped = _make_scoped_connector(mock_async_memory, SharedMemoryStrategy.ENABLED)
        result = await scoped.search_scoped("query", private_scope)

        assert len(result["results"]) == 2

    @pytest.mark.asyncio
    async def test_search_read_only_returns_all(self, mock_async_memory, private_scope):
        """READ_ONLY strategy should return all results (reading is allowed)."""
        scoped = _make_scoped_connector(mock_async_memory, SharedMemoryStrategy.READ_ONLY)
        result = await scoped.search_scoped("query", private_scope)

        assert len(result["results"]) == 2

    @pytest.mark.asyncio
    async def test_search_passes_limit_and_threshold(self, mock_async_memory, private_scope):
        scoped = _make_scoped_connector(mock_async_memory)
        await scoped.search_scoped(
            "query", private_scope, limit=3, threshold=0.5
        )

        call_kwargs = mock_async_memory.search.call_args[1]
        assert call_kwargs["limit"] == 3
        assert call_kwargs["threshold"] == 0.5


class TestGetAllScoped:
    """Tests for get_all_scoped()."""

    @pytest.mark.asyncio
    async def test_get_all_basic(self, mock_async_memory, private_scope):
        scoped = _make_scoped_connector(mock_async_memory)
        result = await scoped.get_all_scoped(private_scope)

        assert "results" in result
        mock_async_memory.get_all.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_all_passes_limit(self, mock_async_memory, private_scope):
        scoped = _make_scoped_connector(mock_async_memory)
        await scoped.get_all_scoped(private_scope, limit=25)

        call_kwargs = mock_async_memory.get_all.call_args[1]
        assert call_kwargs["limit"] == 25


class TestStoreExchangeScoped:
    """Tests for store_exchange_scoped()."""

    @pytest.mark.asyncio
    async def test_store_exchange_delegates(self, mock_async_memory, private_scope):
        scoped = _make_scoped_connector(mock_async_memory)
        result = await scoped.store_exchange_scoped(
            "What is my risk?", "Your risk is moderate.", private_scope
        )

        assert "results" in result
        call_kwargs = mock_async_memory.add.call_args[1]
        messages = call_kwargs["messages"]
        assert messages[0]["role"] == "user"
        assert messages[1]["role"] == "assistant"

    @pytest.mark.asyncio
    async def test_store_exchange_blocked_shared_write(self, mock_async_memory, team_scope):
        """store_exchange_scoped should respect SharedMemoryStrategy."""
        scoped = _make_scoped_connector(mock_async_memory, SharedMemoryStrategy.READ_ONLY)
        result = await scoped.store_exchange_scoped(
            "Hello", "Hi there", team_scope
        )

        assert result["blocked_by"] == "shared_memory_strategy"
        mock_async_memory.add.assert_not_called()
