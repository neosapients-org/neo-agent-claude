"""
Tests for cleanup_expired() on NeoMemoryConnector (v0.2).
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from unittest.mock import AsyncMock

from neo_memory_hub.domain.scope import AccessTier, IsolationScope
from neo_memory_hub.integrations.connector import (
    ConnectorConfig,
    NeoMemoryConnector,
)


@pytest.fixture
def mock_async_memory():
    """Create mock AsyncMemory with mixed expired/active memories."""
    now = datetime.now(UTC)
    past = (now - timedelta(days=10)).isoformat()
    future = (now + timedelta(days=10)).isoformat()

    memory = AsyncMock()
    memory.get_all = AsyncMock(
        return_value={
            "results": [
                {"id": "m1", "memory": "Expired", "metadata": {"expires_at": past}},
                {"id": "m2", "memory": "Active", "metadata": {"expires_at": future}},
                {"id": "m3", "memory": "No TTL", "metadata": {}},
                {"id": "m4", "memory": "Low salience", "metadata": {"salience": 0.1}},
                {"id": "m5", "memory": "High salience", "metadata": {"salience": 0.9}},
            ]
        }
    )
    memory.delete = AsyncMock(return_value={"success": True})
    return memory


def _make_connector(mock_memory) -> NeoMemoryConnector:
    connector = NeoMemoryConnector(
        config=ConnectorConfig(
            log_operations=True,
            use_storage_validation=False,
        )
    )
    connector._memory = mock_memory
    connector._initialized = True
    return connector


class TestCleanupExpired:
    """Tests for cleanup_expired()."""

    @pytest.mark.asyncio
    async def test_dry_run_returns_preview(self, mock_async_memory):
        """Dry run should report expired memories but not delete them."""
        connector = _make_connector(mock_async_memory)
        result = await connector.cleanup_expired(dry_run=True)

        assert result["dry_run"] is True
        assert result["scanned"] == 5
        assert result["expired"] == 1  # Only m1
        assert "m1" in result["deleted_ids"]
        # Should NOT have called delete
        mock_async_memory.delete.assert_not_called()

    @pytest.mark.asyncio
    async def test_actual_delete(self, mock_async_memory):
        """Non-dry-run should actually delete expired memories."""
        connector = _make_connector(mock_async_memory)
        result = await connector.cleanup_expired(dry_run=False)

        assert result["dry_run"] is False
        assert result["expired"] == 1
        # Should have called delete for m1
        mock_async_memory.delete.assert_called()
        deleted_ids = [c.args[0] for c in mock_async_memory.delete.call_args_list]
        assert "m1" in deleted_ids

    @pytest.mark.asyncio
    async def test_non_expired_preserved(self, mock_async_memory):
        """Active and no-TTL memories should NOT be deleted."""
        connector = _make_connector(mock_async_memory)
        result = await connector.cleanup_expired(dry_run=False)

        deleted_ids = result["deleted_ids"]
        assert "m2" not in deleted_ids  # future
        assert "m3" not in deleted_ids  # no TTL

    @pytest.mark.asyncio
    async def test_salience_threshold(self, mock_async_memory):
        """Should also clean low-salience memories when threshold given."""
        connector = _make_connector(mock_async_memory)
        result = await connector.cleanup_expired(
            salience_threshold=0.5, dry_run=True
        )

        assert result["low_salience"] == 1  # m4 has salience 0.1
        assert "m4" in result["deleted_ids"]
        assert "m5" not in result["deleted_ids"]  # 0.9 > 0.5

    @pytest.mark.asyncio
    async def test_expiry_and_salience_combined(self, mock_async_memory):
        """Expired AND low-salience should both be counted."""
        connector = _make_connector(mock_async_memory)
        result = await connector.cleanup_expired(
            salience_threshold=0.5, dry_run=True
        )

        assert result["expired"] == 1
        assert result["low_salience"] == 1
        assert len(result["deleted_ids"]) == 2

    @pytest.mark.asyncio
    async def test_scoped_cleanup(self, mock_async_memory):
        """cleanup_expired with scope should pass scope params to get_all."""
        scope = IsolationScope(
            tenant_id="acme",
            user_id="u1",
            agent_id="a1",
            pool=AccessTier.PRIVATE,
        )
        connector = _make_connector(mock_async_memory)
        await connector.cleanup_expired(scope=scope, dry_run=True)

        call_kwargs = mock_async_memory.get_all.call_args[1]
        assert call_kwargs["user_id"] == "u1"

    @pytest.mark.asyncio
    async def test_user_id_filter(self, mock_async_memory):
        """cleanup_expired with user_id should pass it to get_all."""
        connector = _make_connector(mock_async_memory)
        await connector.cleanup_expired(user_id="u1", dry_run=True)

        call_kwargs = mock_async_memory.get_all.call_args[1]
        assert call_kwargs["user_id"] == "u1"

    @pytest.mark.asyncio
    async def test_empty_results(self):
        """Should handle zero memories gracefully."""
        memory = AsyncMock()
        memory.get_all = AsyncMock(return_value={"results": []})
        memory.delete = AsyncMock()

        connector = _make_connector(memory)
        result = await connector.cleanup_expired(dry_run=False)

        assert result["scanned"] == 0
        assert result["expired"] == 0
        assert result["deleted_ids"] == []
        memory.delete.assert_not_called()

    @pytest.mark.asyncio
    async def test_malformed_expires_at_ignored(self):
        """Memories with unparseable expires_at should be skipped."""
        memory = AsyncMock()
        memory.get_all = AsyncMock(
            return_value={
                "results": [
                    {"id": "bad", "memory": "Bad date", "metadata": {"expires_at": "not-a-date"}},
                ]
            }
        )
        memory.delete = AsyncMock()

        connector = _make_connector(memory)
        result = await connector.cleanup_expired(dry_run=False)

        assert result["expired"] == 0
        assert "bad" not in result["deleted_ids"]
