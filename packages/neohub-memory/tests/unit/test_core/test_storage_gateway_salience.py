"""
Tests for StorageGateway salience gating (CR-6).
"""

import pytest

from neo_memory_hub.core.storage_gateway import (
    StorageDecision,
    StorageGateway,
    StorageGatewayConfig,
    WorthinessResult,
)


class TestSalienceGate:
    """Tests for the salience threshold in StorageGateway.evaluate()."""

    def test_no_salience_defaults_to_store(self) -> None:
        """Without salience param, existing behavior unchanged — STORE."""
        gw = StorageGateway(StorageGatewayConfig(min_salience_threshold=0.7))
        result = gw.evaluate("A valid memory content")
        assert result.decision == StorageDecision.STORE

    def test_salience_above_threshold_stores(self) -> None:
        """Salience above threshold should STORE."""
        gw = StorageGateway(StorageGatewayConfig(min_salience_threshold=0.7))
        result = gw.evaluate("Important fact", salience=0.85)
        assert result.decision == StorageDecision.STORE

    def test_salience_below_threshold_skips(self) -> None:
        """Salience below threshold should SKIP."""
        gw = StorageGateway(StorageGatewayConfig(min_salience_threshold=0.7))
        result = gw.evaluate("Low importance chat", salience=0.3)
        assert result.decision == StorageDecision.SKIP
        assert "Salience too low" in result.reason

    def test_salience_equal_to_threshold_stores(self) -> None:
        """Salience exactly at threshold should STORE (not strictly less)."""
        gw = StorageGateway(StorageGatewayConfig(min_salience_threshold=0.5))
        result = gw.evaluate("Borderline content", salience=0.5)
        assert result.decision == StorageDecision.STORE

    def test_zero_threshold_accepts_all(self) -> None:
        """Threshold of 0.0 should accept everything (no gating)."""
        gw = StorageGateway(StorageGatewayConfig(min_salience_threshold=0.0))
        result = gw.evaluate("Anything at all", salience=0.01)
        assert result.decision == StorageDecision.STORE

    def test_salience_none_with_high_threshold_stores(self) -> None:
        """salience=None should skip the check even with high threshold."""
        gw = StorageGateway(StorageGatewayConfig(min_salience_threshold=0.9))
        result = gw.evaluate("No salience provided", salience=None)
        assert result.decision == StorageDecision.STORE

    def test_existing_content_validation_still_works(self) -> None:
        """Existing checks (empty, too short, too long) should still work."""
        gw = StorageGateway(StorageGatewayConfig(min_salience_threshold=0.5))

        # Empty content
        result = gw.evaluate("")
        assert result.decision == StorageDecision.SKIP

        # Too short
        result = gw.evaluate("ab")
        assert result.decision == StorageDecision.SKIP

        # Too long
        gw2 = StorageGateway(StorageGatewayConfig(max_content_length=10))
        result = gw2.evaluate("This is way too long for the limit")
        assert result.decision == StorageDecision.SKIP

    def test_salience_check_after_content_validation(self) -> None:
        """Content validation should run before salience check."""
        gw = StorageGateway(StorageGatewayConfig(min_salience_threshold=0.5))
        # Empty content should SKIP even with high salience
        result = gw.evaluate("", salience=1.0)
        assert result.decision == StorageDecision.SKIP
        assert "empty" in result.reason.lower()

    def test_default_config_no_salience_gating(self) -> None:
        """Default config should have threshold 0.0 (no gating)."""
        config = StorageGatewayConfig()
        assert config.min_salience_threshold == 0.0

    def test_high_threshold_blocks_most(self) -> None:
        """High threshold should block most content."""
        gw = StorageGateway(StorageGatewayConfig(min_salience_threshold=0.95))
        assert gw.evaluate("Good content", salience=0.9).decision == StorageDecision.SKIP
        assert gw.evaluate("Great content", salience=0.96).decision == StorageDecision.STORE
