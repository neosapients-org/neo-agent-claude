"""
Tests for IsolationScope v0.2 — CR-2 (dept_id/subject_id), CR-3 (can_access), CR-4 (serialization).
"""

import pytest
from pydantic import ValidationError as PydanticValidationError

from neo_memory_hub.core.exceptions import AccessDeniedError
from neo_memory_hub.domain.scope import AccessTier, IsolationScope


# ============================================================================
# CR-2: New fields (dept_id, subject_id)
# ============================================================================


class TestIsolationScopeFields:
    """Tests for new dept_id and subject_id fields."""

    def test_create_with_dept_and_subject(self) -> None:
        """Should create scope with dept_id and subject_id."""
        scope = IsolationScope(
            tenant_id="acme",
            dept_id="wealth",
            user_id="rm_001",
            agent_id="vic_l3",
            subject_id="cust_sharma_99",
            pool=AccessTier.TEAM,
        )
        assert scope.dept_id == "wealth"
        assert scope.subject_id == "cust_sharma_99"

    def test_dept_and_subject_default_none(self) -> None:
        """dept_id and subject_id should default to None."""
        scope = IsolationScope(tenant_id="acme", user_id="user1")
        assert scope.dept_id is None
        assert scope.subject_id is None

    def test_dept_id_validation_alphanumeric(self) -> None:
        """dept_id should validate alphanumeric + underscore + hyphen."""
        scope = IsolationScope(tenant_id="acme", dept_id="wealth-mgmt_1", pool=AccessTier.TEAM)
        assert scope.dept_id == "wealth-mgmt_1"

    def test_dept_id_validation_invalid_chars(self) -> None:
        """dept_id with invalid characters should be rejected."""
        with pytest.raises(PydanticValidationError, match="Invalid identifier format"):
            IsolationScope(tenant_id="acme", dept_id="wealth mgmt!", pool=AccessTier.TEAM)

    def test_subject_id_max_128_chars(self) -> None:
        """subject_id should accept up to 128 characters."""
        long_id = "a" * 128
        scope = IsolationScope(tenant_id="acme", user_id="user1", subject_id=long_id)
        assert scope.subject_id == long_id

    def test_subject_id_over_128_rejected(self) -> None:
        """subject_id over 128 characters should be rejected."""
        with pytest.raises(PydanticValidationError):
            IsolationScope(tenant_id="acme", user_id="user1", subject_id="a" * 129)

    def test_team_tier_requires_dept_id(self) -> None:
        """TEAM tier should require dept_id."""
        with pytest.raises(PydanticValidationError, match="TEAM tier requires dept_id"):
            IsolationScope(tenant_id="acme", user_id="user1", pool=AccessTier.TEAM)

    def test_team_tier_with_dept_id_succeeds(self) -> None:
        """TEAM tier with dept_id should succeed."""
        scope = IsolationScope(
            tenant_id="acme", dept_id="wealth", user_id="user1", pool=AccessTier.TEAM
        )
        assert scope.pool == AccessTier.TEAM

    def test_org_tier_allows_agent_without_user(self) -> None:
        """ORG tier should allow agent_id without user_id (like SYSTEM)."""
        scope = IsolationScope(
            tenant_id="acme", agent_id="global_bot", pool=AccessTier.ORG
        )
        assert scope.agent_id == "global_bot"

    def test_existing_constructors_still_work(self) -> None:
        """Existing scopes without new fields should still work."""
        scope = IsolationScope(
            tenant_id="acme",
            user_id="user1",
            agent_id="agent1",
            session_id="sess1",
            pool=AccessTier.SHARED,
        )
        assert scope.tenant_id == "acme"
        assert scope.dept_id is None
        assert scope.subject_id is None


# ============================================================================
# CR-3: can_access() — Memory Firewall
# ============================================================================


class TestCanAccess:
    """Tests for the 6-tier access control matrix."""

    def _make_scope(self, **kwargs: object) -> IsolationScope:
        """Helper to create scopes with defaults."""
        defaults = {"tenant_id": "acme"}
        defaults.update(kwargs)
        return IsolationScope(**defaults)  # type: ignore[arg-type]

    # --- Cross-tenant ---

    def test_cross_tenant_always_denied(self) -> None:
        """Cross-tenant access should always be denied for all tiers."""
        requester = self._make_scope(tenant_id="tenant_a", user_id="user1")
        for tier in AccessTier:
            if tier == AccessTier.TEAM:
                target = self._make_scope(
                    tenant_id="tenant_b", dept_id="d", pool=tier
                )
            else:
                target = self._make_scope(tenant_id="tenant_b", pool=tier)
            assert requester.can_access(target) is False, f"Cross-tenant should deny {tier}"

    # --- RESTRICTED ---

    def test_restricted_always_denied(self) -> None:
        """RESTRICTED memories should never be accessible."""
        requester = self._make_scope(user_id="user1", agent_id="agent1")
        target = self._make_scope(
            user_id="user1", agent_id="agent1", pool=AccessTier.RESTRICTED
        )
        assert requester.can_access(target) is False

    # --- SYSTEM ---

    def test_system_accessible_by_any_tenant_user(self) -> None:
        """SYSTEM memories should be accessible by anyone in the tenant."""
        requester = self._make_scope(user_id="user1")
        target = self._make_scope(pool=AccessTier.SYSTEM)
        assert requester.can_access(target) is True

    # --- ORG ---

    def test_org_accessible_by_any_tenant_user(self) -> None:
        """ORG memories should be accessible by anyone in the tenant."""
        requester = self._make_scope(user_id="user1")
        target = self._make_scope(pool=AccessTier.ORG)
        assert requester.can_access(target) is True

    def test_org_cross_tenant_denied(self) -> None:
        """ORG memories from different tenant should be denied."""
        requester = self._make_scope(tenant_id="a", user_id="user1")
        target = self._make_scope(tenant_id="b", pool=AccessTier.ORG)
        assert requester.can_access(target) is False

    # --- TEAM ---

    def test_team_same_dept_allowed(self) -> None:
        """TEAM memories should be accessible by same department."""
        requester = self._make_scope(dept_id="wealth", user_id="user1")
        target = self._make_scope(
            dept_id="wealth", user_id="user2", pool=AccessTier.TEAM
        )
        assert requester.can_access(target) is True

    def test_team_different_dept_denied(self) -> None:
        """TEAM memories should be denied for different department."""
        requester = self._make_scope(dept_id="wealth", user_id="user1")
        target = self._make_scope(
            dept_id="hr", user_id="user2", pool=AccessTier.TEAM
        )
        assert requester.can_access(target) is False

    def test_team_missing_dept_denied(self) -> None:
        """TEAM access should be denied if either scope lacks dept_id."""
        requester = self._make_scope(user_id="user1")  # no dept_id
        target = self._make_scope(
            dept_id="wealth", user_id="user2", pool=AccessTier.TEAM
        )
        assert requester.can_access(target) is False

    # --- SHARED ---

    def test_shared_same_user_allowed(self) -> None:
        """SHARED memories should be accessible by same user."""
        requester = self._make_scope(user_id="user1", agent_id="a1")
        target = self._make_scope(user_id="user1", pool=AccessTier.SHARED)
        assert requester.can_access(target) is True

    def test_shared_different_user_denied(self) -> None:
        """SHARED memories should be denied for different user."""
        requester = self._make_scope(user_id="user1")
        target = self._make_scope(user_id="user2", pool=AccessTier.SHARED)
        assert requester.can_access(target) is False

    def test_shared_none_user_denied(self) -> None:
        """SHARED access should be denied if requester has no user_id."""
        requester = self._make_scope(pool=AccessTier.SYSTEM)  # no user_id
        target = self._make_scope(user_id="user1", pool=AccessTier.SHARED)
        assert requester.can_access(target) is False

    # --- PRIVATE ---

    def test_private_same_user_agent_allowed(self) -> None:
        """PRIVATE memories should be accessible by same user+agent."""
        requester = self._make_scope(user_id="user1", agent_id="agent1")
        target = self._make_scope(
            user_id="user1", agent_id="agent1", pool=AccessTier.PRIVATE
        )
        assert requester.can_access(target) is True

    def test_private_different_agent_denied(self) -> None:
        """PRIVATE memories should be denied for different agent."""
        requester = self._make_scope(user_id="user1", agent_id="agent_a")
        target = self._make_scope(
            user_id="user1", agent_id="agent_b", pool=AccessTier.PRIVATE
        )
        assert requester.can_access(target) is False

    def test_private_none_agent_denied(self) -> None:
        """PRIVATE access should be denied if requester has no agent_id."""
        requester = self._make_scope(user_id="user1")  # no agent_id
        target = self._make_scope(
            user_id="user1", agent_id="agent1", pool=AccessTier.PRIVATE
        )
        assert requester.can_access(target) is False

    def test_private_none_user_denied(self) -> None:
        """PRIVATE access should be denied if requester has no user_id."""
        requester = self._make_scope(pool=AccessTier.SYSTEM)  # no user/agent
        target = self._make_scope(
            user_id="user1", agent_id="agent1", pool=AccessTier.PRIVATE
        )
        assert requester.can_access(target) is False


class TestCheckAccess:
    """Tests for check_access() which raises on denial."""

    def test_raises_access_denied_error(self) -> None:
        """check_access() should raise AccessDeniedError on denial."""
        requester = IsolationScope(tenant_id="acme", user_id="user1")
        target = IsolationScope(
            tenant_id="acme",
            user_id="user1",
            agent_id="agent1",
            pool=AccessTier.RESTRICTED,
        )

        with pytest.raises(AccessDeniedError) as exc_info:
            requester.check_access(target)

        assert "Access denied" in str(exc_info.value)
        assert exc_info.value.details.get("requester_scope") is not None
        assert exc_info.value.details.get("target_scope") is not None

    def test_no_exception_on_allowed(self) -> None:
        """check_access() should not raise when access is allowed."""
        requester = IsolationScope(tenant_id="acme", user_id="user1")
        target = IsolationScope(tenant_id="acme", pool=AccessTier.SYSTEM)
        requester.check_access(target)  # Should not raise


# ============================================================================
# CR-4: Serialization & Builder Methods
# ============================================================================


class TestSerialization:
    """Tests for to_mem0_params, to_mem0_filters, to_filter_dict, builders."""

    def test_to_mem0_params_includes_dept_and_subject(self) -> None:
        """to_mem0_params() should include dept_id and subject_id in metadata."""
        scope = IsolationScope(
            tenant_id="acme",
            dept_id="wealth",
            user_id="rm_001",
            agent_id="vic",
            subject_id="cust_99",
            pool=AccessTier.TEAM,
        )
        params = scope.to_mem0_params()
        assert params["user_id"] == "rm_001"
        assert params["agent_id"] == "vic"
        meta = params["metadata"]
        assert meta["dept_id"] == "wealth"
        assert meta["subject_id"] == "cust_99"
        assert meta["pool"] == "team"

    def test_to_mem0_params_org_omits_user_agent(self) -> None:
        """ORG tier should omit user_id but provide synthetic agent_id for Mem0."""
        scope = IsolationScope(tenant_id="acme", pool=AccessTier.ORG)
        params = scope.to_mem0_params()
        assert "user_id" not in params
        assert params["agent_id"] == "__org_acme__"
        assert params["metadata"]["pool"] == "org"

    def test_to_mem0_params_system_omits_user_agent(self) -> None:
        """SYSTEM tier should omit user_id but provide synthetic agent_id for Mem0."""
        scope = IsolationScope(tenant_id="acme", pool=AccessTier.SYSTEM)
        params = scope.to_mem0_params()
        assert "user_id" not in params
        assert params["agent_id"] == "__org_acme__"

    def test_to_mem0_filters_per_tier(self) -> None:
        """to_mem0_filters() should produce correct filters per tier."""
        # PRIVATE
        scope = IsolationScope(
            tenant_id="acme", user_id="u1", agent_id="a1", pool=AccessTier.PRIVATE
        )
        f = scope.to_mem0_filters()
        assert f["pool"] == "private"
        assert f["user_id"] == "u1"
        assert f["agent_id"] == "a1"

        # TEAM
        scope = IsolationScope(
            tenant_id="acme", dept_id="wealth", user_id="u1", pool=AccessTier.TEAM
        )
        f = scope.to_mem0_filters()
        assert f["pool"] == "team"
        assert f["dept_id"] == "wealth"

        # ORG
        scope = IsolationScope(tenant_id="acme", pool=AccessTier.ORG)
        f = scope.to_mem0_filters()
        assert f["pool"] == "org"
        assert "user_id" not in f

    def test_to_mem0_filters_includes_subject_id(self) -> None:
        """Filters should include subject_id when set."""
        scope = IsolationScope(
            tenant_id="acme", user_id="u1", subject_id="cust_99"
        )
        f = scope.to_mem0_filters()
        assert f["subject_id"] == "cust_99"

    def test_to_filter_dict_includes_all_fields(self) -> None:
        """to_filter_dict() should include all non-None fields."""
        scope = IsolationScope(
            tenant_id="acme",
            dept_id="wealth",
            user_id="u1",
            agent_id="a1",
            session_id="s1",
            subject_id="cust_99",
            pool=AccessTier.TEAM,
        )
        d = scope.to_filter_dict()
        assert d["tenant_id"] == "acme"
        assert d["dept_id"] == "wealth"
        assert d["user_id"] == "u1"
        assert d["agent_id"] == "a1"
        assert d["session_id"] == "s1"
        assert d["subject_id"] == "cust_99"
        assert d["pool"] == "team"

    def test_to_composite_key_new_format(self) -> None:
        """Composite key format: tenant:dept:user."""
        scope = IsolationScope(
            tenant_id="acme", dept_id="wealth", user_id="u1", pool=AccessTier.TEAM
        )
        assert scope.to_composite_key() == "acme:wealth:u1"

    def test_to_composite_key_no_dept(self) -> None:
        """Composite key without dept_id."""
        scope = IsolationScope(tenant_id="acme", user_id="u1")
        assert scope.to_composite_key() == "acme:u1"

    def test_to_composite_key_tenant_only(self) -> None:
        """Composite key with only tenant."""
        scope = IsolationScope(tenant_id="acme", pool=AccessTier.SYSTEM)
        assert scope.to_composite_key() == "acme"


class TestBuilders:
    """Tests for builder methods."""

    def _base_scope(self) -> IsolationScope:
        return IsolationScope(
            tenant_id="acme",
            dept_id="wealth",
            user_id="u1",
            agent_id="a1",
            subject_id="cust_1",
            pool=AccessTier.TEAM,
        )

    def test_with_dept(self) -> None:
        """with_dept() should create new scope with new dept_id."""
        base = self._base_scope()
        new = base.with_dept("hr")
        assert new.dept_id == "hr"
        assert new.tenant_id == base.tenant_id
        assert new.user_id == base.user_id
        assert new.subject_id == base.subject_id

    def test_with_subject(self) -> None:
        """with_subject() should create new scope with new subject_id."""
        base = self._base_scope()
        new = base.with_subject("cust_99")
        assert new.subject_id == "cust_99"
        assert new.dept_id == base.dept_id

    def test_with_pool_preserves_new_fields(self) -> None:
        """with_pool() should preserve dept_id and subject_id."""
        base = self._base_scope()
        new = base.with_pool(AccessTier.ORG)
        assert new.pool == AccessTier.ORG
        assert new.dept_id == "wealth"
        assert new.subject_id == "cust_1"

    def test_with_session_preserves_new_fields(self) -> None:
        """with_session() should preserve dept_id and subject_id."""
        base = self._base_scope()
        new = base.with_session("sess_42")
        assert new.session_id == "sess_42"
        assert new.dept_id == "wealth"
        assert new.subject_id == "cust_1"
