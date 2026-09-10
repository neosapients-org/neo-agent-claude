"""
Tests for AccessTier enum (CR-1).
"""

import pytest

from neo_memory_hub.domain.scope import AccessTier


class TestAccessTier:
    """Tests for the AccessTier enum."""

    def test_all_six_tiers_exist(self) -> None:
        """All 6 tier values should be defined."""
        assert AccessTier.PRIVATE == "private"
        assert AccessTier.SHARED == "shared"
        assert AccessTier.TEAM == "team"
        assert AccessTier.ORG == "org"
        assert AccessTier.SYSTEM == "system"
        assert AccessTier.RESTRICTED == "restricted"

    def test_tier_count(self) -> None:
        """Should have exactly 6 tiers."""
        assert len(AccessTier) == 6

    def test_construct_from_string(self) -> None:
        """Should construct tier from string value."""
        assert AccessTier("team") == AccessTier.TEAM
        assert AccessTier("org") == AccessTier.ORG
        assert AccessTier("restricted") == AccessTier.RESTRICTED

    def test_construct_from_invalid_string(self) -> None:
        """Invalid string should raise ValueError."""
        with pytest.raises(ValueError):
            AccessTier("invalid_tier")

    def test_json_serialization(self) -> None:
        """Tiers should serialize to string values."""
        assert AccessTier.TEAM.value == "team"
        assert AccessTier.ORG.value == "org"
        assert AccessTier.RESTRICTED.value == "restricted"

    def test_is_str_enum(self) -> None:
        """AccessTier should be a string enum."""
        assert isinstance(AccessTier.PRIVATE, str)
        assert AccessTier.PRIVATE == "private"

    def test_tier_ordering_in_list(self) -> None:
        """All tiers should be iterable."""
        tiers = list(AccessTier)
        assert len(tiers) == 6
        assert AccessTier.PRIVATE in tiers
        assert AccessTier.RESTRICTED in tiers
