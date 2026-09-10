"""Tests for PoolRoutingHook ABC and SharedScopePoolRouter implementation."""
import pytest
from unittest.mock import AsyncMock, MagicMock

from neo_memory_hub.hooks.callbacks import PoolRoutingHook


class TestPoolRoutingHookABC:
    """Verify PoolRoutingHook ABC contract."""

    def test_cannot_instantiate_directly(self):
        """PoolRoutingHook is abstract."""
        with pytest.raises(TypeError, match="abstract"):
            PoolRoutingHook()

    def test_partial_implementation_fails(self):
        """Missing any abstract method raises TypeError."""

        class Partial(PoolRoutingHook):
            def resolve_store_params(self, *args, **kwargs):
                return {}
            # Missing: get_pool_search_tasks, extend_profile, format_pool_sections

        with pytest.raises(TypeError, match="abstract"):
            Partial()

    def test_complete_implementation_works(self):
        """Complete subclass can be instantiated."""

        class Complete(PoolRoutingHook):
            def resolve_store_params(self, *args, **kwargs):
                return {}

            def get_pool_search_tasks(self, *args, **kwargs):
                return [], []

            def extend_profile(self, *args, **kwargs):
                return {}

            def format_pool_sections(self, *args, **kwargs):
                return ""

        hook = Complete()
        assert hook is not None


class TestSharedScopePoolRouter:
    """Tests for the SharedScopePoolRouter implementation."""

    @pytest.fixture
    def router(self):
        from memory_utils.shared_scope.pool_routing import (
            PoolSearchConfig,
            SharedScopePoolRouter,
        )
        return SharedScopePoolRouter(
            pools=[
                PoolSearchConfig(pool="team", threshold=0.2, limit=5),
                PoolSearchConfig(pool="org", threshold=0.2, limit=5),
            ],
            default_pool="private",
        )

    @pytest.fixture
    def router_no_pools(self):
        from memory_utils.shared_scope.pool_routing import SharedScopePoolRouter
        return SharedScopePoolRouter()

    # ==== resolve_store_params ====

    def test_private_pool_no_change(self, router):
        """Private pool: user_id unchanged."""
        result = router.resolve_store_params(
            fact_pool="private",
            user_id="user1",
            agent_id="agent1",
            tenant_id="tenant1",
            dept_id="dept1",
            metadata={"pool": "private"},
        )
        assert result["user_id"] == "user1"
        assert result["agent_id"] == "agent1"

    def test_team_pool_sentinel_user_id(self, router):
        """Team pool: user_id becomes __team_{dept_id}__."""
        result = router.resolve_store_params(
            fact_pool="team",
            user_id="user1",
            agent_id="agent1",
            tenant_id="tenant1",
            dept_id="wealth",
            metadata={"pool": "team"},
        )
        assert result["user_id"] == "__team_wealth__"
        assert result["metadata"]["dept_id"] == "wealth"

    def test_team_pool_default_dept(self, router):
        """Team pool without dept_id uses 'default'."""
        result = router.resolve_store_params(
            fact_pool="team",
            user_id="user1",
            agent_id="agent1",
            tenant_id="tenant1",
            dept_id=None,
            metadata={"pool": "team"},
        )
        assert result["user_id"] == "__team_default__"

    def test_org_pool_sentinel_user_id(self, router):
        """Org pool: user_id becomes __org_{tenant_id}__."""
        result = router.resolve_store_params(
            fact_pool="org",
            user_id="user1",
            agent_id="agent1",
            tenant_id="acme_corp",
            dept_id=None,
            metadata={"pool": "org"},
        )
        assert result["user_id"] == "__org_acme_corp__"

    def test_shared_pool_same_user(self, router):
        """Shared pool: user_id stays the same."""
        result = router.resolve_store_params(
            fact_pool="shared",
            user_id="user1",
            agent_id="agent1",
            tenant_id="tenant1",
            dept_id=None,
            metadata={"pool": "shared"},
        )
        assert result["user_id"] == "user1"

    def test_none_pool_uses_default(self, router):
        """None pool falls back to default (private)."""
        result = router.resolve_store_params(
            fact_pool=None,
            user_id="user1",
            agent_id="agent1",
            tenant_id="tenant1",
            dept_id=None,
            metadata={},
        )
        assert result["user_id"] == "user1"

    # ==== get_pool_search_tasks ====

    def test_pool_search_tasks_creates_correct_tasks(self, router):
        """Creates one search task per configured pool."""
        mock_search = AsyncMock(return_value={"results": []})

        tasks, labels = router.get_pool_search_tasks(
            query="test query",
            search_fn=mock_search,
            user_id="user1",
            agent_id="agent1",
            tenant_id="tenant1",
            dept_id="wealth",
            allow_cross_agent=True,
        )
        assert len(tasks) == 2
        assert labels == ["team", "org"]

    def test_pool_search_tasks_empty_when_no_pools(self, router_no_pools):
        """No pools configured: empty tasks."""
        mock_search = AsyncMock(return_value={"results": []})

        tasks, labels = router_no_pools.get_pool_search_tasks(
            query="test",
            search_fn=mock_search,
            user_id="user1",
            agent_id="agent1",
            tenant_id="tenant1",
            dept_id=None,
            allow_cross_agent=True,
        )
        assert len(tasks) == 0
        assert len(labels) == 0

    @pytest.mark.asyncio
    async def test_pool_search_team_uses_sentinel(self, router):
        """Team pool search uses sentinel user_id."""
        calls = []

        async def capture_search(**kwargs):
            calls.append(kwargs)
            return {"results": []}

        tasks, labels = router.get_pool_search_tasks(
            query="test",
            search_fn=capture_search,
            user_id="user1",
            agent_id="agent1",
            tenant_id="tenant1",
            dept_id="wealth",
            allow_cross_agent=True,
        )

        import asyncio
        await asyncio.gather(*tasks)

        team_call = calls[0]  # first pool is team
        assert team_call["user_id"] == "__team_wealth__"
        assert team_call["agent_id"] is None
        assert team_call["metadata_filters"]["dept_id"] == "wealth"

    @pytest.mark.asyncio
    async def test_pool_search_org_uses_sentinel(self, router):
        """Org pool search uses sentinel user_id."""
        calls = []

        async def capture_search(**kwargs):
            calls.append(kwargs)
            return {"results": []}

        tasks, labels = router.get_pool_search_tasks(
            query="test",
            search_fn=capture_search,
            user_id="user1",
            agent_id="agent1",
            tenant_id="acme_corp",
            dept_id=None,
            allow_cross_agent=True,
        )

        import asyncio
        await asyncio.gather(*tasks)

        org_call = calls[1]  # second pool is org
        assert org_call["user_id"] == "__org_acme_corp__"
        assert org_call["agent_id"] is None

    # ==== extend_profile ====

    def test_extend_profile_adds_team_section(self, router):
        """Team memories are added to profile."""
        profile = {"persona": [{"detail": "test"}]}
        by_category = {
            "persona": [{"id": "1", "memory": "likes X"}],
            "team": [{"id": "2", "memory": "department rule Y"}],
        }
        result = router.extend_profile(profile, by_category)
        assert "team" in result
        assert result["team"][0]["directive"] == "department rule Y"
        assert result["team"][0]["pool"] == "team"

    def test_extend_profile_adds_org_section(self, router):
        """Org memories are added to profile."""
        profile = {}
        by_category = {
            "org": [{"id": "3", "memory": "firm-wide policy Z"}],
        }
        result = router.extend_profile(profile, by_category)
        assert "org" in result
        assert result["org"][0]["directive"] == "firm-wide policy Z"

    def test_extend_profile_no_pool_mems(self, router):
        """No pool memories: profile unchanged."""
        profile = {"persona": [{"detail": "test"}]}
        by_category = {"persona": [{"id": "1", "memory": "likes X"}]}
        result = router.extend_profile(profile, by_category)
        assert "team" not in result
        assert "org" not in result

    # ==== format_pool_sections ====

    def test_format_pool_sections_team_and_org(self, router):
        """Formats both team and org sections."""
        profile = {
            "team": [{"directive": "rule A"}, {"directive": "rule B"}],
            "org": [{"directive": "mandate C"}],
        }
        result = router.format_pool_sections(profile)
        assert "## DEPARTMENT DIRECTIVES (TEAM)" in result
        assert "- rule A" in result
        assert "- rule B" in result
        assert "## FIRM-WIDE DIRECTIVES (ORG)" in result
        assert "- mandate C" in result

    def test_format_pool_sections_empty(self, router):
        """No pool sections: empty string."""
        profile = {"persona": [{"detail": "test"}]}
        result = router.format_pool_sections(profile)
        assert result == ""

    def test_format_pool_sections_team_only(self, router):
        """Only team section present."""
        profile = {
            "team": [{"directive": "dept rule"}],
        }
        result = router.format_pool_sections(profile)
        assert "DEPARTMENT DIRECTIVES" in result
        assert "FIRM-WIDE DIRECTIVES" not in result
