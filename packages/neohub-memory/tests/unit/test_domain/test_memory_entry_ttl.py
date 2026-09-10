"""
Tests for MemoryEntry TTL support (CR-5).
"""

from datetime import UTC, datetime, timedelta

import pytest

from neo_memory_hub.domain.memory import MemoryEntry, utcnow
from neo_memory_hub.domain.scope import IsolationScope
from neo_memory_hub.domain.types import MemoryType


class TestMemoryEntryTTL:
    """Tests for expires_at, is_expired, and calculate_expiry."""

    def _make_entry(self, **kwargs: object) -> MemoryEntry:
        scope = IsolationScope(tenant_id="acme", user_id="u1")
        defaults = {"content": "Test memory", "scope": scope}
        defaults.update(kwargs)
        return MemoryEntry(**defaults)  # type: ignore[arg-type]

    def test_expires_at_default_none(self) -> None:
        """expires_at should default to None."""
        entry = self._make_entry()
        assert entry.expires_at is None

    def test_not_expired_when_none(self) -> None:
        """is_expired should be False when expires_at is None (permanent)."""
        entry = self._make_entry()
        assert entry.is_expired is False

    def test_expired_when_in_past(self) -> None:
        """is_expired should be True when expires_at is in the past."""
        past = utcnow() - timedelta(hours=1)
        entry = self._make_entry(expires_at=past)
        assert entry.is_expired is True

    def test_not_expired_when_in_future(self) -> None:
        """is_expired should be False when expires_at is in the future."""
        future = utcnow() + timedelta(hours=1)
        entry = self._make_entry(expires_at=future)
        assert entry.is_expired is False

    def test_calculate_expiry_episodic(self) -> None:
        """EPISODIC should default to 365 days."""
        now = utcnow()
        expiry = MemoryEntry.calculate_expiry(MemoryType.EPISODIC, created_at=now)
        assert expiry is not None
        delta = expiry - now
        assert abs(delta.days - 365) <= 1

    def test_calculate_expiry_persona_permanent(self) -> None:
        """PERSONA should return None (permanent)."""
        expiry = MemoryEntry.calculate_expiry(MemoryType.PERSONA)
        assert expiry is None

    def test_calculate_expiry_working_short(self) -> None:
        """WORKING should default to 1 day."""
        now = utcnow()
        expiry = MemoryEntry.calculate_expiry(MemoryType.WORKING, created_at=now)
        assert expiry is not None
        delta = expiry - now
        assert abs(delta.days - 1) <= 0

    def test_calculate_expiry_custom_override(self) -> None:
        """custom_retention_days should override the default."""
        now = utcnow()
        expiry = MemoryEntry.calculate_expiry(
            MemoryType.EPISODIC, created_at=now, custom_retention_days=30
        )
        assert expiry is not None
        delta = expiry - now
        assert abs(delta.days - 30) <= 0

    def test_calculate_expiry_defaults_to_now(self) -> None:
        """If no created_at is given, should use current time."""
        before = utcnow()
        expiry = MemoryEntry.calculate_expiry(MemoryType.WORKING)
        after = utcnow()
        assert expiry is not None
        assert before + timedelta(days=1) <= expiry <= after + timedelta(days=1, seconds=1)

    def test_calculate_expiry_conversation(self) -> None:
        """CONVERSATION should default to 90 days."""
        now = utcnow()
        expiry = MemoryEntry.calculate_expiry(MemoryType.CONVERSATION, created_at=now)
        assert expiry is not None
        assert abs((expiry - now).days - 90) <= 1
