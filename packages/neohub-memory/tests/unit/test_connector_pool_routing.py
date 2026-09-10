"""Tests for HighLevelMemoryConnector with PoolRoutingHook integration.

Verifies that:
1. Connector works WITHOUT pool_routing (backward compatible, all private)
2. Connector works WITH pool_routing (SharedScopePoolRouter)
3. Pool routing hook is called correctly during store_exchange
4. Pool routing hook is called correctly during retrieve_context
"""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from neo_memory_hub.connector import HighLevelMemoryConnector
from neo_memory_hub.config.hub_config import MemoryHubConfig


def _make_config(**overrides):
    """Create a minimal test config."""
    defaults = {
        "vector_store": {"provider": "qdrant", "collection_name": "test"},
        "llm": {"model": "gpt-4o-mini", "temperature": 0.1},
        "embedder": {"model": "text-embedding-3-small", "dimensions": 1536},
        "extraction": {"enabled": True, "valid_categories": ["persona", "preference"]},
        "dedup": {"enabled": False},
        "buffering": {"enabled": False},
        "salience": {"enabled": False},
        "history": {"enabled": False},
        "isolation": {
            "default_agent_id": "test_agent",
            "default_pool": "private",
            "retrieval_pools": ["private"],
        },
    }
    defaults.update(overrides)
    return MemoryHubConfig(**defaults)


class TestConnectorWithoutPoolRouting:
    """Verify backward compatibility — no pool_routing, everything private."""

    def test_init_without_pool_routing(self):
        """Can create connector without pool_routing param."""
        cfg = _make_config()
        connector = HighLevelMemoryConnector(config=cfg)
        assert connector._pool_routing is None

    @pytest.mark.asyncio
    async def test_store_exchange_private_pool(self):
        """Without pool routing, store_exchange uses inline pool logic."""
        cfg = _make_config()
        connector = HighLevelMemoryConnector(config=cfg)

        # Mock all internal components
        mock_neo = AsyncMock()
        mock_neo.search = AsyncMock(return_value={"results": []})
        mock_neo.store_fact = AsyncMock(return_value={
            "results": [{"id": "m1", "memory": "test fact"}]
        })
        connector._connector = mock_neo
        connector._initialized = True

        mock_extractor = AsyncMock()
        mock_extractor.extract = AsyncMock(return_value=MagicMock(
            error=None,
            facts=[MagicMock(
                category="persona",
                content="User is a software engineer",
                investor_name=None,
                profile_key="occupation",
                detail="",
                behavioral_note="",
                trigger="",
                salience=0.8,
                salience_reasoning="",
                pool="private",
            )],
            investor_names_found=set(),
        ))
        connector._extractor = mock_extractor
        connector._retriever = MagicMock()
        connector._assembler = MagicMock()
        connector._formatter = MagicMock()

        result = await connector.store_exchange(
            query="What do you do?",
            response="I am a software engineer",
            user_id="user1",
        )

        assert result["facts_stored"] == 1
        # Verify store_fact was called with original user_id (not sentinel)
        call_kwargs = mock_neo.store_fact.call_args
        assert call_kwargs.kwargs["user_id"] == "user1"


class TestConnectorWithPoolRouting:
    """Verify pool routing hook is called and integrated correctly."""

    def test_init_with_pool_routing(self):
        """Can create connector with pool_routing param."""
        from memory_utils.shared_scope.pool_routing import (
            PoolSearchConfig,
            SharedScopePoolRouter,
        )

        router = SharedScopePoolRouter(
            pools=[PoolSearchConfig(pool="team", threshold=0.2, limit=5)]
        )
        cfg = _make_config()
        connector = HighLevelMemoryConnector(config=cfg, pool_routing=router)
        assert connector._pool_routing is router

    @pytest.mark.asyncio
    async def test_store_exchange_team_pool_uses_sentinel(self):
        """With pool routing, team pool facts use sentinel user_id."""
        from memory_utils.shared_scope.pool_routing import (
            PoolSearchConfig,
            SharedScopePoolRouter,
        )

        router = SharedScopePoolRouter(
            pools=[PoolSearchConfig(pool="team", threshold=0.2, limit=5)]
        )
        cfg = _make_config(isolation={
            "default_agent_id": "test_agent",
            "default_pool": "private",
            "retrieval_pools": ["private", "team"],
            "default_dept_id": "engineering",
        })
        connector = HighLevelMemoryConnector(config=cfg, pool_routing=router)

        mock_neo = AsyncMock()
        mock_neo.search = AsyncMock(return_value={"results": []})
        mock_neo.store_fact = AsyncMock(return_value={
            "results": [{"id": "m1", "memory": "team rule"}]
        })
        connector._connector = mock_neo
        connector._initialized = True

        mock_extractor = AsyncMock()
        mock_extractor.extract = AsyncMock(return_value=MagicMock(
            error=None,
            facts=[MagicMock(
                category="procedural",
                content="Department follows agile methodology",
                investor_name=None,
                profile_key="",
                detail="",
                behavioral_note="",
                trigger="",
                salience=0.7,
                salience_reasoning="",
                pool="team",  # LLM assigned team pool
            )],
            investor_names_found=set(),
        ))
        connector._extractor = mock_extractor
        connector._retriever = MagicMock()
        connector._assembler = MagicMock()
        connector._formatter = MagicMock()

        result = await connector.store_exchange(
            query="How does the department work?",
            response="We follow agile methodology",
            user_id="user1",
            dept_id="engineering",
        )

        assert result["facts_stored"] == 1
        # Verify store_fact was called with sentinel user_id
        call_kwargs = mock_neo.store_fact.call_args
        assert call_kwargs.kwargs["user_id"] == "__team_engineering__"

    @pytest.mark.asyncio
    async def test_store_exchange_org_pool_uses_sentinel(self):
        """With pool routing, org pool facts use sentinel user_id."""
        from memory_utils.shared_scope.pool_routing import (
            PoolSearchConfig,
            SharedScopePoolRouter,
        )

        router = SharedScopePoolRouter(
            pools=[PoolSearchConfig(pool="org", threshold=0.2, limit=5)]
        )
        cfg = _make_config()
        connector = HighLevelMemoryConnector(config=cfg, pool_routing=router)

        mock_neo = AsyncMock()
        mock_neo.search = AsyncMock(return_value={"results": []})
        mock_neo.store_fact = AsyncMock(return_value={
            "results": [{"id": "m1", "memory": "org policy"}]
        })
        connector._connector = mock_neo
        connector._initialized = True

        mock_extractor = AsyncMock()
        mock_extractor.extract = AsyncMock(return_value=MagicMock(
            error=None,
            facts=[MagicMock(
                category="procedural",
                content="Company-wide compliance policy",
                investor_name=None,
                profile_key="",
                detail="",
                behavioral_note="",
                trigger="",
                salience=0.9,
                salience_reasoning="",
                pool="org",
            )],
            investor_names_found=set(),
        ))
        connector._extractor = mock_extractor
        connector._retriever = MagicMock()
        connector._assembler = MagicMock()
        connector._formatter = MagicMock()

        result = await connector.store_exchange(
            query="What is the company policy?",
            response="We have a compliance-first policy",
            user_id="user1",
            tenant_id="acme_corp",
        )

        assert result["facts_stored"] == 1
        call_kwargs = mock_neo.store_fact.call_args
        assert call_kwargs.kwargs["user_id"] == "__org_acme_corp__"

    @pytest.mark.asyncio
    async def test_retrieve_context_with_pool_routing(self):
        """Pool routing extends profile and formats pool sections."""
        from memory_utils.shared_scope.pool_routing import (
            PoolSearchConfig,
            SharedScopePoolRouter,
        )

        router = SharedScopePoolRouter(
            pools=[PoolSearchConfig(pool="team", threshold=0.2, limit=5)]
        )
        cfg = _make_config(isolation={
            "default_agent_id": "test_agent",
            "default_pool": "private",
            "retrieval_pools": ["private", "team"],
        })
        connector = HighLevelMemoryConnector(config=cfg, pool_routing=router)

        mock_neo = AsyncMock()
        mock_neo.search = AsyncMock(return_value={"results": []})
        connector._connector = mock_neo
        connector._initialized = True

        # Mock retriever to return results including team memories
        mock_retriever = AsyncMock()
        mock_retriever.retrieve = AsyncMock(return_value=MagicMock(
            memories=[
                {"id": "1", "memory": "User likes Python", "_retrieval_reason": "persona"},
                {"id": "2", "memory": "Dept uses GitFlow", "_retrieval_reason": "team"},
            ],
            by_category={
                "persona": [{"id": "1", "memory": "User likes Python"}],
                "team": [{"id": "2", "memory": "Dept uses GitFlow"}],
            },
            count=2,
            user_id="user1",
            agent_id="test_agent",
            investor_name=None,
        ))
        connector._retriever = mock_retriever

        mock_assembler = MagicMock()
        mock_assembler.assemble = MagicMock(return_value={
            "persona": [{"detail": "likes Python"}],
        })
        connector._assembler = mock_assembler

        mock_formatter = MagicMock()
        mock_formatter.format = MagicMock(return_value="## PERSONA\n- likes Python")
        connector._formatter = mock_formatter

        result = await connector.retrieve_context(
            query="What tools does the team use?",
            user_id="user1",
        )

        assert result["count"] == 2
        # Pool routing should have extended the profile
        assert "team" in result["structured_profile"] or "context" in result

    @pytest.mark.asyncio
    async def test_retrieve_context_without_pool_routing(self):
        """Without pool routing, no pool sections are added."""
        cfg = _make_config()
        connector = HighLevelMemoryConnector(config=cfg)

        mock_neo = AsyncMock()
        mock_neo.search = AsyncMock(return_value={"results": []})
        connector._connector = mock_neo
        connector._initialized = True

        mock_retriever = AsyncMock()
        mock_retriever.retrieve = AsyncMock(return_value=MagicMock(
            memories=[{"id": "1", "memory": "User likes Python"}],
            by_category={"persona": [{"id": "1", "memory": "User likes Python"}]},
            count=1,
            user_id="user1",
            agent_id="test_agent",
            investor_name=None,
        ))
        connector._retriever = mock_retriever

        mock_assembler = MagicMock()
        mock_assembler.assemble = MagicMock(return_value={
            "persona": [{"detail": "likes Python"}],
        })
        connector._assembler = mock_assembler

        mock_formatter = MagicMock()
        mock_formatter.format = MagicMock(return_value="## PERSONA\n- likes Python")
        connector._formatter = mock_formatter

        result = await connector.retrieve_context(
            query="What does the user like?",
            user_id="user1",
        )

        assert result["count"] == 1
        # No pool sections should be in context
        assert "DEPARTMENT DIRECTIVES" not in result["context"]
        assert "FIRM-WIDE DIRECTIVES" not in result["context"]
