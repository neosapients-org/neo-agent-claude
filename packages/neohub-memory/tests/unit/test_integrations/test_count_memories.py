"""
Tests for count_memories() on NeoMemoryConnector (v0.2).
"""

from __future__ import annotations

import pytest
from unittest.mock import AsyncMock

from neo_memory_hub.domain.scope import AccessTier, IsolationScope
from neo_memory_hub.integrations.connector import (
    ConnectorConfig,
    NeoMemoryConnector,
)


@pytest.fixture
def mock_async_memory():
    """Create mock AsyncMemory with subject-tagged memories."""
    memory = AsyncMock()
    memory.get_all = AsyncMock(
        return_value={
            "results": [
                {"id": "m1", "memory": "Fact A", "metadata": {"subject_id": "CUST_99"}},
                {"id": "m2", "memory": "Fact B", "metadata": {"subject_id": "CUST_99"}},
                {"id": "m3", "memory": "Fact C", "metadata": {"subject_id": "CUST_42"}},
                {"id": "m4", "memory": "Fact D", "metadata": {}},
            ]
        }
    )
    return memory


def _make_connector(mock_memory) -> NeoMemoryConnector:
    connector = NeoMemoryConnector(
        config=ConnectorConfig(
            log_operations=False,
            use_storage_validation=False,
        )
    )
    connector._memory = mock_memory
    connector._initialized = True
    return connector


class TestCountMemories:
    """Tests for count_memories()."""

    @pytest.mark.asyncio
    async def test_total_count(self, mock_async_memory):
        connector = _make_connector(mock_async_memory)
        result = await connector.count_memories(user_id="u1")

        assert result["total"] == 4
        assert "by_subject" not in result

    @pytest.mark.asyncio
    async def test_group_by_subject(self, mock_async_memory):
        connector = _make_connector(mock_async_memory)
        result = await connector.count_memories(
            user_id="u1", group_by_subject=True
        )

        assert result["total"] == 4
        subjects = result["by_subject"]
        assert subjects["CUST_99"] == 2
        assert subjects["CUST_42"] == 1
        assert subjects["_unknown"] == 1  # m4 has no subject_id

    @pytest.mark.asyncio
    async def test_empty_results(self):
        memory = AsyncMock()
        memory.get_all = AsyncMock(return_value={"results": []})

        connector = _make_connector(memory)
        result = await connector.count_memories(user_id="u1")

        assert result["total"] == 0

    @pytest.mark.asyncio
    async def test_scoped_count(self, mock_async_memory):
        """count_memories with scope should pass scope params."""
        scope = IsolationScope(
            tenant_id="acme",
            user_id="u1",
            agent_id="a1",
            pool=AccessTier.PRIVATE,
        )
        connector = _make_connector(mock_async_memory)
        result = await connector.count_memories(scope=scope)

        call_kwargs = mock_async_memory.get_all.call_args[1]
        assert call_kwargs["user_id"] == "u1"
        assert result["total"] == 4

    @pytest.mark.asyncio
    async def test_group_by_subject_empty(self):
        memory = AsyncMock()
        memory.get_all = AsyncMock(return_value={"results": []})

        connector = _make_connector(memory)
        result = await connector.count_memories(
            user_id="u1", group_by_subject=True
        )

        assert result["total"] == 0
        assert result["by_subject"] == {}
