"""Unit tests for neomem_mcp Phase 2 — remaining 12 tools, resource, handler coverage.

Tests the MCP layer in isolation — mocks all library calls.
Does NOT require Qdrant, OpenAI, or any external service.
"""

import json
import os
import tempfile
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import yaml

from neomem_mcp.models import CapabilityDefinition


# --------------------------------------------------------------------------- #
#  Shared helpers
# --------------------------------------------------------------------------- #


def _make_cap(**overrides) -> CapabilityDefinition:
    """Create a CapabilityDefinition with defaults + overrides."""
    defaults = {
        "name": "memory.test",
        "mcp_tool_name": "neomem_test",
        "required_package": "neo_memory_hub",
        "connector": "high_level",
        "method": "test_method",
        "description": "Test tool",
        "input_schema": {
            "type": "object",
            "properties": {
                "user_id": {"type": "string", "description": "User ID"},
            },
            "required": ["user_id"],
        },
    }
    defaults.update(overrides)
    return CapabilityDefinition(**defaults)


def _make_ctx(hlc=None, low_level=None, scoped=None, scorer=None, capabilities=None):
    """Create a mock MCP context with AppContext."""
    app = MagicMock()
    app.hlc = hlc
    app.low_level = low_level
    app.scoped = scoped
    app.scorer = scorer
    app.capabilities = capabilities or []

    ctx = MagicMock()
    ctx.request_context.lifespan_context = app
    return ctx


# --------------------------------------------------------------------------- #
#  Test: Phase 2 capability YAMLs — all 12 new + 1 resource parse correctly
# --------------------------------------------------------------------------- #


class TestPhase2CapabilityYAMLs:
    """Validate all Phase 2 capability YAML files."""

    CAPS_DIR = None

    @pytest.fixture(autouse=True)
    def _set_caps_dir(self):
        from neomem_mcp.loader import get_bundled_capabilities_dir
        TestPhase2CapabilityYAMLs.CAPS_DIR = get_bundled_capabilities_dir()

    PHASE2_TOOLS = [
        "neomem_buffer_exchange",
        "neomem_flush_buffer",
        "neomem_get_all",
        "neomem_update_memory",
        "neomem_delete_memory",
        "neomem_delete_all",
        "neomem_cleanup_expired",
        "neomem_count_memories",
        "neomem_score_salience",
        "neomem_get_history",
        "neomem_store_scoped",
        "neomem_search_scoped",
    ]

    def _load_yaml(self, tool_name: str) -> dict:
        path = os.path.join(self.CAPS_DIR, f"{tool_name}.capability.yaml")
        with open(path) as f:
            return yaml.safe_load(f)

    def test_all_phase2_yamls_exist(self):
        for name in self.PHASE2_TOOLS:
            path = os.path.join(self.CAPS_DIR, f"{name}.capability.yaml")
            assert os.path.exists(path), f"Missing YAML: {name}"

    def test_config_resource_yaml_exists(self):
        path = os.path.join(self.CAPS_DIR, "neomem_config.capability.yaml")
        assert os.path.exists(path)

    @pytest.mark.parametrize("tool_name", [
        "neomem_buffer_exchange",
        "neomem_flush_buffer",
        "neomem_get_all",
        "neomem_update_memory",
        "neomem_delete_memory",
        "neomem_delete_all",
        "neomem_cleanup_expired",
        "neomem_count_memories",
        "neomem_score_salience",
        "neomem_get_history",
        "neomem_store_scoped",
        "neomem_search_scoped",
    ])
    def test_yaml_has_required_fields(self, tool_name):
        data = self._load_yaml(tool_name)
        for field in ["name", "mcp_tool_name", "required_package", "method",
                       "description", "input_schema", "version"]:
            assert field in data, f"{tool_name} missing field: {field}"

    def test_scoped_tools_require_memory_utils(self):
        for name in ["neomem_store_scoped", "neomem_search_scoped"]:
            data = self._load_yaml(name)
            assert data["required_package"] == "memory_utils"
            assert data["connector"] == "scoped"

    def test_destructive_tools_have_hint(self):
        for name in ["neomem_delete_memory", "neomem_delete_all", "neomem_cleanup_expired"]:
            data = self._load_yaml(name)
            assert data["annotations"]["destructiveHint"] is True

    def test_readonly_tools_have_hint(self):
        for name in ["neomem_get_all", "neomem_count_memories",
                      "neomem_score_salience", "neomem_get_history"]:
            data = self._load_yaml(name)
            assert data["annotations"]["readOnlyHint"] is True

    def test_config_resource_has_mcp_type(self):
        data = self._load_yaml("neomem_config")
        assert data["mcp_type"] == "resource"
        assert data["mcp_resource_uri"] == "neomem://config"

    def test_all_22_tools_present(self):
        """Verify the full set of 22 tools + 1 resource."""
        from neomem_mcp.loader import load_capabilities
        caps = load_capabilities(self.CAPS_DIR)
        tool_names = {c.mcp_tool_name for c in caps if c.mcp_type == "tool"}
        resource_names = {c.mcp_tool_name for c in caps if c.mcp_type == "resource"}

        expected_tools = {
            # Phase 1
            "neomem_store_exchange", "neomem_retrieve_context", "neomem_search",
            "neomem_store_fact", "neomem_get_memory",
            # Phase 2
            "neomem_buffer_exchange", "neomem_flush_buffer", "neomem_get_all",
            "neomem_update_memory", "neomem_delete_memory", "neomem_delete_all",
            "neomem_cleanup_expired", "neomem_count_memories",
            "neomem_score_salience", "neomem_get_history",
            "neomem_store_scoped", "neomem_search_scoped",
            # Phase 3 — full coverage
            "neomem_store_preference", "neomem_get_all_history",
            "neomem_get_history_stats", "neomem_get_all_scoped",
            "neomem_add_scoped",
        }
        assert tool_names == expected_tools
        assert "neomem_config" in resource_names


# --------------------------------------------------------------------------- #
#  Test: Buffer exchange + flush handlers
# --------------------------------------------------------------------------- #


class TestBufferHandlers:
    """Test buffer_exchange and flush_session_buffer tool handlers."""

    @pytest.mark.asyncio
    async def test_buffer_exchange_success(self):
        from neomem_mcp._handlers import resolve_handler

        hlc = AsyncMock()
        hlc.buffer_exchange = AsyncMock(return_value={
            "buffered": True,
            "buffer_size": 3,
            "auto_flushed": False,
        })

        cap = _make_cap(
            name="memory.buffer_exchange",
            mcp_tool_name="neomem_buffer_exchange",
            method="buffer_exchange",
            input_schema={
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "response": {"type": "string"},
                    "user_id": {"type": "string"},
                    "session_id": {"type": "string"},
                },
                "required": ["query", "response", "user_id"],
            },
        )
        handler = resolve_handler(cap)
        ctx = _make_ctx(hlc=hlc)

        result = await handler(ctx, query="hello", response="world", user_id="u1")
        parsed = json.loads(result)

        assert parsed["buffered"] is True
        assert parsed["buffer_size"] == 3
        hlc.buffer_exchange.assert_called_once_with(
            query="hello", response="world", user_id="u1"
        )

    @pytest.mark.asyncio
    async def test_flush_buffer_success(self):
        from neomem_mcp._handlers import resolve_handler

        hlc = AsyncMock()
        hlc.flush_session_buffer = AsyncMock(return_value={
            "flushed_count": 5,
            "total_facts_stored": 8,
            "results": [{"id": "1"}, {"id": "2"}],
        })

        cap = _make_cap(
            name="memory.flush_buffer",
            mcp_tool_name="neomem_flush_buffer",
            method="flush_session_buffer",
            input_schema={
                "type": "object",
                "properties": {
                    "user_id": {"type": "string"},
                    "session_id": {"type": "string"},
                },
                "required": ["user_id"],
            },
        )
        handler = resolve_handler(cap)
        ctx = _make_ctx(hlc=hlc)

        result = await handler(ctx, user_id="u1", session_id="s1")
        parsed = json.loads(result)

        assert parsed["flushed_count"] == 5
        assert parsed["total_facts_stored"] == 8
        hlc.flush_session_buffer.assert_called_once_with(
            user_id="u1", session_id="s1"
        )


# --------------------------------------------------------------------------- #
#  Test: CRUD handlers (get_all, update, delete, delete_all)
# --------------------------------------------------------------------------- #


class TestCRUDHandlers:
    """Test get_all, update, delete, delete_all tool handlers."""

    @pytest.mark.asyncio
    async def test_get_all_success(self):
        from neomem_mcp._handlers import resolve_handler

        hlc = AsyncMock()
        hlc.get_all = AsyncMock(return_value={
            "results": [
                {"id": "m1", "memory": "fact one"},
                {"id": "m2", "memory": "fact two"},
            ]
        })

        cap = _make_cap(
            method="get_all",
            input_schema={
                "type": "object",
                "properties": {"user_id": {"type": "string"}},
                "required": ["user_id"],
            },
        )
        handler = resolve_handler(cap)
        ctx = _make_ctx(hlc=hlc)

        result = await handler(ctx, user_id="u1")
        parsed = json.loads(result)

        assert len(parsed["results"]) == 2
        hlc.get_all.assert_called_once_with(user_id="u1")

    @pytest.mark.asyncio
    async def test_update_memory_success(self):
        from neomem_mcp._handlers import resolve_handler

        hlc = AsyncMock()
        hlc.update = AsyncMock(return_value={
            "id": "m1",
            "memory": "updated fact",
        })

        cap = _make_cap(
            method="update",
            input_schema={
                "type": "object",
                "properties": {
                    "memory_id": {"type": "string"},
                    "data": {"type": "string"},
                },
                "required": ["memory_id", "data"],
            },
        )
        handler = resolve_handler(cap)
        ctx = _make_ctx(hlc=hlc)

        result = await handler(ctx, memory_id="m1", data="updated fact")
        parsed = json.loads(result)

        assert parsed["memory"] == "updated fact"
        hlc.update.assert_called_once_with(memory_id="m1", data="updated fact")

    @pytest.mark.asyncio
    async def test_delete_memory_success(self):
        from neomem_mcp._handlers import resolve_handler

        hlc = AsyncMock()
        hlc.delete = AsyncMock(return_value={"deleted": True, "id": "m1"})

        cap = _make_cap(
            method="delete",
            input_schema={
                "type": "object",
                "properties": {"memory_id": {"type": "string"}},
                "required": ["memory_id"],
            },
        )
        handler = resolve_handler(cap)
        ctx = _make_ctx(hlc=hlc)

        result = await handler(ctx, memory_id="m1")
        parsed = json.loads(result)

        assert parsed["deleted"] is True
        hlc.delete.assert_called_once_with(memory_id="m1")

    @pytest.mark.asyncio
    async def test_delete_all_success(self):
        from neomem_mcp._handlers import resolve_handler

        hlc = AsyncMock()
        hlc.delete_all = AsyncMock(return_value={"deleted_count": 15})

        cap = _make_cap(
            method="delete_all",
            input_schema={
                "type": "object",
                "properties": {
                    "user_id": {"type": "string"},
                    "agent_id": {"type": "string"},
                },
                "required": ["user_id"],
            },
        )
        handler = resolve_handler(cap)
        ctx = _make_ctx(hlc=hlc)

        result = await handler(ctx, user_id="u1")
        parsed = json.loads(result)

        assert parsed["deleted_count"] == 15
        hlc.delete_all.assert_called_once_with(user_id="u1")


# --------------------------------------------------------------------------- #
#  Test: Cleanup and count handlers
# --------------------------------------------------------------------------- #


class TestMaintenanceHandlers:
    """Test cleanup_expired and count_memories tool handlers."""

    @pytest.mark.asyncio
    async def test_cleanup_expired_success(self):
        from neomem_mcp._handlers import resolve_handler

        hlc = AsyncMock()
        hlc.cleanup_expired = AsyncMock(return_value={
            "scanned": 100,
            "expired": 3,
            "low_salience": 2,
            "deleted_ids": ["d1", "d2", "d3", "d4", "d5"],
            "dry_run": False,
        })

        cap = _make_cap(
            method="cleanup_expired",
            input_schema={
                "type": "object",
                "properties": {
                    "user_id": {"type": "string"},
                    "dry_run": {"type": "boolean"},
                },
                "required": [],
            },
        )
        handler = resolve_handler(cap)
        ctx = _make_ctx(hlc=hlc)

        result = await handler(ctx, user_id="u1", dry_run=True)
        parsed = json.loads(result)

        assert parsed["scanned"] == 100
        assert parsed["expired"] == 3
        assert len(parsed["deleted_ids"]) == 5
        hlc.cleanup_expired.assert_called_once_with(user_id="u1", dry_run=True)

    @pytest.mark.asyncio
    async def test_cleanup_expired_no_args(self):
        from neomem_mcp._handlers import resolve_handler

        hlc = AsyncMock()
        hlc.cleanup_expired = AsyncMock(return_value={
            "scanned": 0, "expired": 0, "low_salience": 0,
            "deleted_ids": [], "dry_run": False,
        })

        cap = _make_cap(
            method="cleanup_expired",
            input_schema={
                "type": "object",
                "properties": {"user_id": {"type": "string"}},
                "required": [],
            },
        )
        handler = resolve_handler(cap)
        ctx = _make_ctx(hlc=hlc)

        result = await handler(ctx)
        parsed = json.loads(result)
        assert parsed["scanned"] == 0

    @pytest.mark.asyncio
    async def test_count_memories_success(self):
        from neomem_mcp._handlers import resolve_handler

        hlc = AsyncMock()
        hlc.count_memories = AsyncMock(return_value={"total": 42})

        cap = _make_cap(
            method="count_memories",
            input_schema={
                "type": "object",
                "properties": {"user_id": {"type": "string"}},
                "required": [],
            },
        )
        handler = resolve_handler(cap)
        ctx = _make_ctx(hlc=hlc)

        result = await handler(ctx, user_id="u1")
        parsed = json.loads(result)

        assert parsed["total"] == 42
        hlc.count_memories.assert_called_once_with(user_id="u1")


# --------------------------------------------------------------------------- #
#  Test: Scorer handler
# --------------------------------------------------------------------------- #


class TestScorerHandler:
    """Test score_salience tool handler via SalienceScorer."""

    @pytest.mark.asyncio
    async def test_score_facts_multiple(self):
        from neomem_mcp._handlers import resolve_handler

        scorer = AsyncMock()
        scored_fact_1 = MagicMock()
        scored_fact_1.model_dump = MagicMock(return_value={
            "text": "User prefers Rust",
            "salience": 0.8,
            "reasoning": "Strong preference signal",
        })
        scored_fact_2 = MagicMock()
        scored_fact_2.model_dump = MagicMock(return_value={
            "text": "User works at Anthropic",
            "salience": 0.6,
            "reasoning": "Professional context",
        })
        scorer.score_facts = AsyncMock(return_value=[scored_fact_1, scored_fact_2])

        cap = _make_cap(
            connector="scorer",
            method="score_facts",
            input_schema={
                "type": "object",
                "properties": {
                    "facts": {"type": "array"},
                    "context": {"type": "string"},
                },
                "required": ["facts"],
            },
        )
        handler = resolve_handler(cap)
        ctx = _make_ctx(scorer=scorer)

        result = await handler(
            ctx, facts=["User prefers Rust", "User works at Anthropic"],
            context="programming discussion",
        )
        parsed = json.loads(result)

        assert len(parsed["results"]) == 2
        assert parsed["results"][0]["salience"] == 0.8
        scorer.score_facts.assert_called_once_with(
            ["User prefers Rust", "User works at Anthropic"],
            context="programming discussion",
        )

    @pytest.mark.asyncio
    async def test_score_facts_single_string_converted(self):
        from neomem_mcp._handlers import resolve_handler

        scorer = AsyncMock()
        scored = MagicMock()
        scored.model_dump = MagicMock(return_value={
            "text": "test", "salience": 0.5, "reasoning": "neutral"
        })
        scorer.score_facts = AsyncMock(return_value=[scored])

        cap = _make_cap(
            connector="scorer",
            method="score_facts",
            input_schema={
                "type": "object",
                "properties": {"facts": {"type": "array"}},
                "required": ["facts"],
            },
        )
        handler = resolve_handler(cap)
        ctx = _make_ctx(scorer=scorer)

        # Pass a string instead of list — handler should convert
        result = await handler(ctx, facts="single fact")
        parsed = json.loads(result)

        scorer.score_facts.assert_called_once_with(
            ["single fact"], context=None,
        )

    @pytest.mark.asyncio
    async def test_score_single(self):
        from neomem_mcp._handlers import resolve_handler

        scorer = AsyncMock()
        scored = MagicMock()
        scored.model_dump = MagicMock(return_value={
            "text": "high value fact", "salience": 0.9, "reasoning": "critical"
        })
        scorer.score_single = AsyncMock(return_value=scored)

        cap = _make_cap(
            connector="scorer",
            method="score_single",
            input_schema={
                "type": "object",
                "properties": {
                    "fact": {"type": "string"},
                    "context": {"type": "string"},
                },
                "required": ["fact"],
            },
        )
        handler = resolve_handler(cap)
        ctx = _make_ctx(scorer=scorer)

        result = await handler(ctx, fact="high value fact", context="meeting")
        parsed = json.loads(result)

        assert parsed["salience"] == 0.9
        scorer.score_single.assert_called_once_with(
            "high value fact", context="meeting",
        )


# --------------------------------------------------------------------------- #
#  Test: History handler
# --------------------------------------------------------------------------- #


class TestHistoryHandler:
    """Test get_history tool handler."""

    @pytest.mark.asyncio
    async def test_get_history_success(self):
        from neomem_mcp._handlers import resolve_handler

        hlc = AsyncMock()
        hlc.history = AsyncMock(return_value=[
            {"event": "ADD", "timestamp": "2026-03-20T10:00:00", "data": "fact1"},
            {"event": "UPDATE", "timestamp": "2026-03-21T11:00:00", "data": "fact1_v2"},
        ])

        cap = _make_cap(
            method="history",
            input_schema={
                "type": "object",
                "properties": {"memory_id": {"type": "string"}},
                "required": ["memory_id"],
            },
        )
        handler = resolve_handler(cap)
        ctx = _make_ctx(hlc=hlc)

        result = await handler(ctx, memory_id="m1")
        parsed = json.loads(result)

        assert len(parsed["results"]) == 2
        assert parsed["results"][0]["event"] == "ADD"
        hlc.history.assert_called_once_with(memory_id="m1")

    @pytest.mark.asyncio
    async def test_get_history_empty(self):
        from neomem_mcp._handlers import resolve_handler

        hlc = AsyncMock()
        hlc.history = AsyncMock(return_value=[])

        cap = _make_cap(
            method="history",
            input_schema={
                "type": "object",
                "properties": {"memory_id": {"type": "string"}},
                "required": ["memory_id"],
            },
        )
        handler = resolve_handler(cap)
        ctx = _make_ctx(hlc=hlc)

        result = await handler(ctx, memory_id="m_nonexistent")
        parsed = json.loads(result)
        assert parsed["results"] == []


# --------------------------------------------------------------------------- #
#  Test: Scoped handlers
# --------------------------------------------------------------------------- #


class TestScopedHandlers:
    """Test store_scoped and search_scoped tool handlers."""

    @pytest.mark.asyncio
    async def test_store_scoped_success(self):
        from neomem_mcp._handlers import resolve_handler

        scoped = AsyncMock()
        scoped.store_exchange_scoped = AsyncMock(return_value={
            "facts_stored": 2,
            "scope": {"tenant_id": "acme", "user_id": "u1", "pool": "private"},
        })

        cap = _make_cap(
            connector="scoped",
            method="store_exchange_scoped",
            required_package="memory_utils",
            input_schema={
                "type": "object",
                "properties": {
                    "user_message": {"type": "string"},
                    "assistant_response": {"type": "string"},
                    "user_id": {"type": "string"},
                    "tenant_id": {"type": "string"},
                    "pool": {"type": "string"},
                },
                "required": ["user_message", "assistant_response", "user_id"],
            },
        )
        handler = resolve_handler(cap)
        ctx = _make_ctx(scoped=scoped)

        result = await handler(
            ctx,
            user_message="I like Python",
            assistant_response="Python is great!",
            user_id="u1",
            tenant_id="acme",
        )
        parsed = json.loads(result)

        assert parsed["facts_stored"] == 2
        # Verify scope was built from flat params
        scoped.store_exchange_scoped.assert_called_once()
        call_kwargs = scoped.store_exchange_scoped.call_args
        scope_arg = call_kwargs.kwargs.get("scope")
        assert scope_arg is not None
        assert scope_arg.tenant_id == "acme"
        assert scope_arg.user_id == "u1"

    @pytest.mark.asyncio
    async def test_search_scoped_success(self):
        from neomem_mcp._handlers import resolve_handler

        scoped = AsyncMock()
        scoped.search_scoped = AsyncMock(return_value={
            "results": [
                {"id": "s1", "memory": "scoped fact", "score": 0.9}
            ]
        })

        cap = _make_cap(
            connector="scoped",
            method="search_scoped",
            required_package="memory_utils",
            input_schema={
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "user_id": {"type": "string"},
                    "tenant_id": {"type": "string"},
                    "limit": {"type": "integer"},
                },
                "required": ["query", "user_id"],
            },
        )
        handler = resolve_handler(cap)
        ctx = _make_ctx(scoped=scoped)

        result = await handler(
            ctx, query="Python", user_id="u1", tenant_id="acme", limit=5,
        )
        parsed = json.loads(result)

        assert len(parsed["results"]) == 1
        scoped.search_scoped.assert_called_once()
        call_kwargs = scoped.search_scoped.call_args
        assert call_kwargs.kwargs["query"] == "Python"
        assert call_kwargs.kwargs["limit"] == 5

    @pytest.mark.asyncio
    async def test_scoped_not_installed(self):
        from neomem_mcp._handlers import resolve_handler

        cap = _make_cap(
            connector="scoped",
            method="store_exchange_scoped",
            required_package="memory_utils",
            input_schema={
                "type": "object",
                "properties": {
                    "user_message": {"type": "string"},
                    "user_id": {"type": "string"},
                },
                "required": ["user_message", "user_id"],
            },
        )
        handler = resolve_handler(cap)
        ctx = _make_ctx(scoped=None)

        result = await handler(ctx, user_message="test", user_id="u1")
        parsed = json.loads(result)
        assert parsed["error"] == "PackageNotInstalled"

    @pytest.mark.asyncio
    async def test_scoped_with_strategy_override(self):
        from neomem_mcp._handlers import resolve_handler

        scoped = AsyncMock()
        scoped.search_scoped = AsyncMock(return_value={"results": []})
        scoped._strategy = None

        cap = _make_cap(
            connector="scoped",
            method="search_scoped",
            required_package="memory_utils",
            input_schema={
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "user_id": {"type": "string"},
                    "strategy": {"type": "string"},
                },
                "required": ["query", "user_id"],
            },
        )
        handler = resolve_handler(cap)
        ctx = _make_ctx(scoped=scoped)

        # strategy override should be popped and applied to scoped connector
        with patch("neomem_mcp._handlers.SharedMemoryStrategy", create=True) as MockStrategy:
            with patch.dict("sys.modules", {"memory_utils.shared_scope": MagicMock()}):
                result = await handler(
                    ctx, query="test", user_id="u1", strategy="ENABLED",
                )
        # Should not crash
        assert json.loads(result) is not None


# --------------------------------------------------------------------------- #
#  Test: Dynamic signature building
# --------------------------------------------------------------------------- #


class TestDynamicSignature:
    """Test that _build_signature creates proper function signatures."""

    def test_signature_with_required_and_optional(self):
        from neomem_mcp._handlers import _build_signature
        from mcp.server.fastmcp import Context
        import inspect

        schema = {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "limit": {"type": "integer"},
                "dry_run": {"type": "boolean"},
            },
            "required": ["query"],
        }

        sig = _build_signature(schema)
        params = dict(sig.parameters)

        assert "ctx" in params
        assert params["ctx"].annotation is Context

        assert "query" in params
        assert params["query"].default is inspect.Parameter.empty  # required
        assert params["query"].annotation is str

        assert "limit" in params
        assert params["limit"].default is None  # optional
        assert params["limit"].kind == inspect.Parameter.KEYWORD_ONLY

        assert "dry_run" in params
        assert params["dry_run"].default is None

    def test_signature_no_required_fields(self):
        from neomem_mcp._handlers import _build_signature

        schema = {
            "type": "object",
            "properties": {
                "user_id": {"type": "string"},
            },
            "required": [],
        }

        sig = _build_signature(schema)
        params = dict(sig.parameters)

        assert params["user_id"].default is None

    def test_signature_empty_schema(self):
        from neomem_mcp._handlers import _build_signature

        sig = _build_signature({"type": "object", "properties": {}, "required": []})
        params = dict(sig.parameters)

        assert len(params) == 1  # just ctx
        assert "ctx" in params

    def test_signature_array_type(self):
        from neomem_mcp._handlers import _build_signature

        schema = {
            "type": "object",
            "properties": {
                "facts": {"type": "array"},
            },
            "required": ["facts"],
        }

        sig = _build_signature(schema)
        params = dict(sig.parameters)
        assert params["facts"].annotation is list


# --------------------------------------------------------------------------- #
#  Test: Server bootstrap with all 18 capabilities
# --------------------------------------------------------------------------- #


class TestServerBootstrapPhase2:
    """Test server bootstrap with all Phase 2 capabilities."""

    def test_create_server_registers_all_core_tools(self):
        from neomem_mcp.server import create_mcp_server
        from neomem_mcp.config import MCPServerConfig
        from neomem_mcp.loader import get_bundled_capabilities_dir

        caps_dir = get_bundled_capabilities_dir()

        with patch("neomem_mcp.server.detect_installed_packages") as mock_detect:
            mock_detect.return_value = {"neo_memory_hub"}
            config = MCPServerConfig(capabilities_dir=caps_dir)
            server = create_mcp_server(server_config=config, capabilities_dir=caps_dir)

        # Verify it's a FastMCP instance
        assert server is not None

    @patch("neomem_mcp.server.app_lifespan")
    def test_scoped_tools_skipped_without_memory_utils(self, mock_lifespan):
        """Scoped tools require memory_utils — should be skipped in standard env."""
        from neomem_mcp.loader import load_capabilities, detect_installed_packages, filter_capabilities, get_bundled_capabilities_dir

        caps = load_capabilities(get_bundled_capabilities_dir())
        installed = detect_installed_packages()

        # memory_utils may or may not be installed
        registered, skipped = filter_capabilities(caps, installed, mcp_type="tool")

        scoped_names = {"neomem_store_scoped", "neomem_search_scoped"}

        if "memory_utils" not in installed:
            skipped_names = {c.mcp_tool_name for c in skipped}
            assert scoped_names.issubset(skipped_names)
        else:
            registered_names = {c.mcp_tool_name for c in registered}
            assert scoped_names.issubset(registered_names)


# --------------------------------------------------------------------------- #
#  Test: End-to-end handler pipeline for Phase 2 tools
# --------------------------------------------------------------------------- #


class TestEndToEndPhase2Pipeline:
    """Full pipeline tests: load YAML → resolve handler → mock call → verify."""

    CAPS_DIR = None

    @pytest.fixture(autouse=True)
    def _set_caps_dir(self):
        from neomem_mcp.loader import get_bundled_capabilities_dir
        TestEndToEndPhase2Pipeline.CAPS_DIR = get_bundled_capabilities_dir()

    def _load_cap(self, tool_name: str) -> CapabilityDefinition:
        path = os.path.join(self.CAPS_DIR, f"{tool_name}.capability.yaml")
        with open(path) as f:
            data = yaml.safe_load(f)
        return CapabilityDefinition.from_dict(data)

    @pytest.mark.asyncio
    async def test_buffer_exchange_pipeline(self):
        from neomem_mcp._handlers import resolve_handler

        cap = self._load_cap("neomem_buffer_exchange")
        hlc = AsyncMock()
        hlc.buffer_exchange = AsyncMock(return_value={"buffered": True})
        ctx = _make_ctx(hlc=hlc)

        handler = resolve_handler(cap)
        result = await handler(ctx, query="hi", response="hello", user_id="u1")
        parsed = json.loads(result)
        assert parsed["buffered"] is True

    @pytest.mark.asyncio
    async def test_flush_buffer_pipeline(self):
        from neomem_mcp._handlers import resolve_handler

        cap = self._load_cap("neomem_flush_buffer")
        hlc = AsyncMock()
        hlc.flush_session_buffer = AsyncMock(return_value={"flushed_count": 3})
        ctx = _make_ctx(hlc=hlc)

        handler = resolve_handler(cap)
        result = await handler(ctx, user_id="u1")
        parsed = json.loads(result)
        assert parsed["flushed_count"] == 3

    @pytest.mark.asyncio
    async def test_get_all_pipeline(self):
        from neomem_mcp._handlers import resolve_handler

        cap = self._load_cap("neomem_get_all")
        hlc = AsyncMock()
        hlc.get_all = AsyncMock(return_value={"results": [{"id": "m1"}]})
        ctx = _make_ctx(hlc=hlc)

        handler = resolve_handler(cap)
        result = await handler(ctx, user_id="u1")
        parsed = json.loads(result)
        assert len(parsed["results"]) == 1

    @pytest.mark.asyncio
    async def test_update_memory_pipeline(self):
        from neomem_mcp._handlers import resolve_handler

        cap = self._load_cap("neomem_update_memory")
        hlc = AsyncMock()
        hlc.update = AsyncMock(return_value={"id": "m1", "memory": "new text"})
        ctx = _make_ctx(hlc=hlc)

        handler = resolve_handler(cap)
        result = await handler(ctx, memory_id="m1", data="new text")
        parsed = json.loads(result)
        assert parsed["memory"] == "new text"

    @pytest.mark.asyncio
    async def test_delete_memory_pipeline(self):
        from neomem_mcp._handlers import resolve_handler

        cap = self._load_cap("neomem_delete_memory")
        hlc = AsyncMock()
        hlc.delete = AsyncMock(return_value={"deleted": True})
        ctx = _make_ctx(hlc=hlc)

        handler = resolve_handler(cap)
        result = await handler(ctx, memory_id="m1")
        parsed = json.loads(result)
        assert parsed["deleted"] is True

    @pytest.mark.asyncio
    async def test_delete_all_pipeline(self):
        from neomem_mcp._handlers import resolve_handler

        cap = self._load_cap("neomem_delete_all")
        hlc = AsyncMock()
        hlc.delete_all = AsyncMock(return_value={"deleted_count": 10})
        ctx = _make_ctx(hlc=hlc)

        handler = resolve_handler(cap)
        result = await handler(ctx, user_id="u1")
        parsed = json.loads(result)
        assert parsed["deleted_count"] == 10

    @pytest.mark.asyncio
    async def test_cleanup_expired_pipeline(self):
        from neomem_mcp._handlers import resolve_handler

        cap = self._load_cap("neomem_cleanup_expired")
        hlc = AsyncMock()
        hlc.cleanup_expired = AsyncMock(return_value={
            "scanned": 50, "expired": 2, "deleted_ids": ["x1", "x2"],
        })
        ctx = _make_ctx(hlc=hlc)

        handler = resolve_handler(cap)
        result = await handler(ctx, dry_run=False)
        parsed = json.loads(result)
        assert parsed["scanned"] == 50

    @pytest.mark.asyncio
    async def test_count_memories_pipeline(self):
        from neomem_mcp._handlers import resolve_handler

        cap = self._load_cap("neomem_count_memories")
        hlc = AsyncMock()
        hlc.count_memories = AsyncMock(return_value={"total": 99})
        ctx = _make_ctx(hlc=hlc)

        handler = resolve_handler(cap)
        result = await handler(ctx, user_id="u1")
        parsed = json.loads(result)
        assert parsed["total"] == 99

    @pytest.mark.asyncio
    async def test_score_salience_pipeline(self):
        from neomem_mcp._handlers import resolve_handler

        cap = self._load_cap("neomem_score_salience")
        scorer = AsyncMock()
        scored = MagicMock()
        scored.model_dump = MagicMock(return_value={
            "text": "test", "salience": 0.7, "reasoning": "moderate"
        })
        scorer.score_facts = AsyncMock(return_value=[scored])
        ctx = _make_ctx(scorer=scorer)

        handler = resolve_handler(cap)
        result = await handler(ctx, facts=["test"])
        parsed = json.loads(result)
        assert parsed["results"][0]["salience"] == 0.7

    @pytest.mark.asyncio
    async def test_get_history_pipeline(self):
        from neomem_mcp._handlers import resolve_handler

        cap = self._load_cap("neomem_get_history")
        hlc = AsyncMock()
        hlc.history = AsyncMock(return_value=[
            {"event": "ADD", "timestamp": "2026-03-20T10:00:00"}
        ])
        ctx = _make_ctx(hlc=hlc)

        handler = resolve_handler(cap)
        result = await handler(ctx, memory_id="m1")
        parsed = json.loads(result)
        assert parsed["results"][0]["event"] == "ADD"


# --------------------------------------------------------------------------- #
#  Test: Error handling edge cases
# --------------------------------------------------------------------------- #


class TestErrorEdgeCases:
    """Test error handling patterns for Phase 2 tools."""

    @pytest.mark.asyncio
    async def test_method_not_found_on_connector(self):
        from neomem_mcp._handlers import resolve_handler

        hlc = MagicMock(spec=[])  # no attributes

        cap = _make_cap(method="nonexistent_method")
        handler = resolve_handler(cap)
        ctx = _make_ctx(hlc=hlc)

        result = await handler(ctx, user_id="u1")
        parsed = json.loads(result)
        assert parsed["error"] == "MethodNotFound"

    @pytest.mark.asyncio
    async def test_scorer_method_not_found(self):
        from neomem_mcp._handlers import resolve_handler

        scorer = MagicMock(spec=[])

        cap = _make_cap(
            connector="scorer",
            method="nonexistent_score",
            input_schema={
                "type": "object",
                "properties": {"fact": {"type": "string"}},
                "required": ["fact"],
            },
        )
        handler = resolve_handler(cap)
        ctx = _make_ctx(scorer=scorer)

        result = await handler(ctx, fact="test")
        parsed = json.loads(result)
        assert parsed["error"] == "InternalError"

    @pytest.mark.asyncio
    async def test_storage_error_mapped(self):
        from neomem_mcp._handlers import resolve_handler
        from neo_memory_hub.core.exceptions import StorageError

        hlc = AsyncMock()
        hlc.test_method = AsyncMock(side_effect=StorageError("Qdrant down"))

        cap = _make_cap()
        handler = resolve_handler(cap)
        ctx = _make_ctx(hlc=hlc)

        result = await handler(ctx, user_id="u1")
        parsed = json.loads(result)
        assert parsed["error"] == "StorageError"
        assert "Qdrant" in parsed["message"]

    @pytest.mark.asyncio
    async def test_not_found_error_mapped(self):
        from neomem_mcp._handlers import resolve_handler
        from neo_memory_hub.core.exceptions import NotFoundError

        hlc = AsyncMock()
        hlc.test_method = AsyncMock(side_effect=NotFoundError("memory", "m1"))

        cap = _make_cap()
        handler = resolve_handler(cap)
        ctx = _make_ctx(hlc=hlc)

        result = await handler(ctx, user_id="u1")
        parsed = json.loads(result)
        assert parsed["error"] == "NotFoundError"

    @pytest.mark.asyncio
    async def test_access_denied_error_mapped(self):
        from neomem_mcp._handlers import resolve_handler
        from neo_memory_hub.core.exceptions import AccessDeniedError

        hlc = AsyncMock()
        hlc.test_method = AsyncMock(side_effect=AccessDeniedError("No access"))

        cap = _make_cap()
        handler = resolve_handler(cap)
        ctx = _make_ctx(hlc=hlc)

        result = await handler(ctx, user_id="u1")
        parsed = json.loads(result)
        assert parsed["error"] == "AccessDeniedError"

    @pytest.mark.asyncio
    async def test_handler_serializes_pydantic_model(self):
        from neomem_mcp._handlers import resolve_handler

        hlc = AsyncMock()
        model_result = MagicMock()
        model_result.model_dump = MagicMock(return_value={
            "field_a": "value_a", "field_b": 42,
        })
        hlc.test_method = AsyncMock(return_value=model_result)

        cap = _make_cap()
        handler = resolve_handler(cap)
        ctx = _make_ctx(hlc=hlc)

        result = await handler(ctx, user_id="u1")
        parsed = json.loads(result)
        assert parsed["field_a"] == "value_a"
        assert parsed["field_b"] == 42


# --------------------------------------------------------------------------- #
#  Test: Phase 3 — store_preference (low_level connector)
# --------------------------------------------------------------------------- #


class TestStorePreferenceHandler:
    """Tests for neomem_store_preference tool handler."""

    @pytest.mark.asyncio
    async def test_store_preference_success(self):
        from neomem_mcp._handlers import resolve_handler

        low_level = AsyncMock()
        low_level.store_preference = AsyncMock(return_value={
            "results": [{"id": "pref-1", "memory": "prefers dark mode", "event": "ADD"}],
        })

        cap = _make_cap(
            name="memory.store_preference",
            mcp_tool_name="neomem_store_preference",
            connector="low_level",
            method="store_preference",
            input_schema={
                "type": "object",
                "properties": {
                    "preference": {"type": "string"},
                    "user_id": {"type": "string"},
                    "categories": {"type": "array"},
                },
                "required": ["preference", "user_id"],
            },
        )
        handler = resolve_handler(cap)
        ctx = _make_ctx(low_level=low_level)

        result = await handler(ctx, preference="prefers dark mode", user_id="u1")
        parsed = json.loads(result)
        assert "results" in parsed
        assert parsed["results"][0]["id"] == "pref-1"
        low_level.store_preference.assert_called_once_with(
            preference="prefers dark mode", user_id="u1",
        )

    @pytest.mark.asyncio
    async def test_store_preference_with_categories(self):
        from neomem_mcp._handlers import resolve_handler

        low_level = AsyncMock()
        low_level.store_preference = AsyncMock(return_value={
            "results": [{"id": "pref-2", "memory": "likes ETFs", "event": "ADD"}],
        })

        cap = _make_cap(
            name="memory.store_preference",
            mcp_tool_name="neomem_store_preference",
            connector="low_level",
            method="store_preference",
            input_schema={
                "type": "object",
                "properties": {
                    "preference": {"type": "string"},
                    "user_id": {"type": "string"},
                    "categories": {"type": "array"},
                },
                "required": ["preference", "user_id"],
            },
        )
        handler = resolve_handler(cap)
        ctx = _make_ctx(low_level=low_level)

        result = await handler(ctx, preference="likes ETFs", user_id="u1", categories=["investment"])
        parsed = json.loads(result)
        assert parsed["results"][0]["id"] == "pref-2"
        low_level.store_preference.assert_called_once_with(
            preference="likes ETFs", user_id="u1", categories=["investment"],
        )


# --------------------------------------------------------------------------- #
#  Test: Phase 3 — get_all_history (sync method on high_level)
# --------------------------------------------------------------------------- #


class TestGetAllHistoryHandler:
    """Tests for neomem_get_all_history tool handler (sync method)."""

    @pytest.mark.asyncio
    async def test_get_all_history_default(self):
        from neomem_mcp._handlers import resolve_handler

        hlc = MagicMock()
        hlc.get_all_history = MagicMock(return_value=[
            {"memory_id": "m1", "event": "ADD", "timestamp": "2026-01-01T00:00:00"},
            {"memory_id": "m2", "event": "UPDATE", "timestamp": "2026-01-02T00:00:00"},
        ])

        cap = _make_cap(
            name="memory.get_all_history",
            mcp_tool_name="neomem_get_all_history",
            connector="high_level",
            method="get_all_history",
            input_schema={
                "type": "object",
                "properties": {
                    "limit": {"type": "integer"},
                    "event_filter": {"type": "string"},
                },
                "required": [],
            },
        )
        handler = resolve_handler(cap)
        ctx = _make_ctx(hlc=hlc)

        result = await handler(ctx)
        parsed = json.loads(result)
        assert "results" in parsed
        assert len(parsed["results"]) == 2
        hlc.get_all_history.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_all_history_with_filter(self):
        from neomem_mcp._handlers import resolve_handler

        hlc = MagicMock()
        hlc.get_all_history = MagicMock(return_value=[
            {"memory_id": "m1", "event": "ADD", "timestamp": "2026-01-01T00:00:00"},
        ])

        cap = _make_cap(
            name="memory.get_all_history",
            mcp_tool_name="neomem_get_all_history",
            connector="high_level",
            method="get_all_history",
            input_schema={
                "type": "object",
                "properties": {
                    "limit": {"type": "integer"},
                    "event_filter": {"type": "string"},
                },
                "required": [],
            },
        )
        handler = resolve_handler(cap)
        ctx = _make_ctx(hlc=hlc)

        result = await handler(ctx, limit=50, event_filter="ADD")
        parsed = json.loads(result)
        assert len(parsed["results"]) == 1
        hlc.get_all_history.assert_called_once_with(limit=50, event_filter="ADD")


# --------------------------------------------------------------------------- #
#  Test: Phase 3 — get_history_stats (sync method on high_level)
# --------------------------------------------------------------------------- #


class TestGetHistoryStatsHandler:
    """Tests for neomem_get_history_stats tool handler (sync method)."""

    @pytest.mark.asyncio
    async def test_get_history_stats_success(self):
        from neomem_mcp._handlers import resolve_handler

        hlc = MagicMock()
        hlc.get_history_stats = MagicMock(return_value={
            "total": 42,
            "ADD": 30,
            "UPDATE": 8,
            "DELETE": 4,
            "history_enabled": True,
        })

        cap = _make_cap(
            name="memory.get_history_stats",
            mcp_tool_name="neomem_get_history_stats",
            connector="high_level",
            method="get_history_stats",
            input_schema={"type": "object", "properties": {}, "required": []},
        )
        handler = resolve_handler(cap)
        ctx = _make_ctx(hlc=hlc)

        result = await handler(ctx)
        parsed = json.loads(result)
        assert parsed["total"] == 42
        assert parsed["ADD"] == 30
        assert parsed["history_enabled"] is True
        hlc.get_history_stats.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_history_stats_disabled(self):
        from neomem_mcp._handlers import resolve_handler

        hlc = MagicMock()
        hlc.get_history_stats = MagicMock(return_value={
            "total": 0, "history_enabled": False,
        })

        cap = _make_cap(
            name="memory.get_history_stats",
            mcp_tool_name="neomem_get_history_stats",
            connector="high_level",
            method="get_history_stats",
            input_schema={"type": "object", "properties": {}, "required": []},
        )
        handler = resolve_handler(cap)
        ctx = _make_ctx(hlc=hlc)

        result = await handler(ctx)
        parsed = json.loads(result)
        assert parsed["history_enabled"] is False


# --------------------------------------------------------------------------- #
#  Test: Phase 3 — get_all_scoped (scoped connector)
# --------------------------------------------------------------------------- #


class TestGetAllScopedHandler:
    """Tests for neomem_get_all_scoped tool handler."""

    @pytest.mark.asyncio
    async def test_get_all_scoped_success(self):
        from neomem_mcp._handlers import resolve_handler

        scoped = AsyncMock()
        scoped.get_all_scoped = AsyncMock(return_value={
            "results": [
                {"id": "s1", "memory": "team fact 1"},
                {"id": "s2", "memory": "team fact 2"},
            ],
        })

        cap = _make_cap(
            name="memory.get_all_scoped",
            mcp_tool_name="neomem_get_all_scoped",
            required_package="memory_utils",
            connector="scoped",
            method="get_all_scoped",
            input_schema={
                "type": "object",
                "properties": {
                    "user_id": {"type": "string"},
                    "tenant_id": {"type": "string"},
                    "agent_id": {"type": "string"},
                    "dept_id": {"type": "string"},
                    "pool": {"type": "string"},
                    "limit": {"type": "integer"},
                },
                "required": ["user_id"],
            },
        )
        handler = resolve_handler(cap)
        ctx = _make_ctx(scoped=scoped)

        result = await handler(ctx, user_id="u1", tenant_id="t1", pool="shared", limit=50)
        parsed = json.loads(result)
        assert "results" in parsed
        assert len(parsed["results"]) == 2
        scoped.get_all_scoped.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_all_scoped_not_installed(self):
        from neomem_mcp._handlers import resolve_handler

        cap = _make_cap(
            name="memory.get_all_scoped",
            mcp_tool_name="neomem_get_all_scoped",
            required_package="memory_utils",
            connector="scoped",
            method="get_all_scoped",
            input_schema={
                "type": "object",
                "properties": {"user_id": {"type": "string"}},
                "required": ["user_id"],
            },
        )
        handler = resolve_handler(cap)
        ctx = _make_ctx(scoped=None)  # memory_utils not installed

        result = await handler(ctx, user_id="u1")
        parsed = json.loads(result)
        assert parsed["error"] == "PackageNotInstalled"


# --------------------------------------------------------------------------- #
#  Test: Phase 3 — add_scoped (scoped connector)
# --------------------------------------------------------------------------- #


class TestAddScopedHandler:
    """Tests for neomem_add_scoped tool handler."""

    @pytest.mark.asyncio
    async def test_add_scoped_success(self):
        from neomem_mcp._handlers import resolve_handler

        scoped = AsyncMock()
        scoped.add_scoped = AsyncMock(return_value={
            "results": [{"id": "as-1", "memory": "team policy doc", "event": "ADD"}],
        })

        cap = _make_cap(
            name="memory.add_scoped",
            mcp_tool_name="neomem_add_scoped",
            required_package="memory_utils",
            connector="scoped",
            method="add_scoped",
            input_schema={
                "type": "object",
                "properties": {
                    "messages": {"type": "string"},
                    "user_id": {"type": "string"},
                    "tenant_id": {"type": "string"},
                    "agent_id": {"type": "string"},
                    "dept_id": {"type": "string"},
                    "pool": {"type": "string"},
                    "categories": {"type": "array"},
                    "infer": {"type": "boolean"},
                },
                "required": ["messages", "user_id"],
            },
        )
        handler = resolve_handler(cap)
        ctx = _make_ctx(scoped=scoped)

        result = await handler(
            ctx, messages="Team guideline: always verify client identity",
            user_id="u1", tenant_id="acme", dept_id="compliance", pool="team",
        )
        parsed = json.loads(result)
        assert "results" in parsed
        assert parsed["results"][0]["id"] == "as-1"
        scoped.add_scoped.assert_called_once()

    @pytest.mark.asyncio
    async def test_add_scoped_with_infer_false(self):
        from neomem_mcp._handlers import resolve_handler

        scoped = AsyncMock()
        scoped.add_scoped = AsyncMock(return_value={
            "results": [{"id": "as-2", "memory": "raw fact", "event": "ADD"}],
        })

        cap = _make_cap(
            name="memory.add_scoped",
            mcp_tool_name="neomem_add_scoped",
            required_package="memory_utils",
            connector="scoped",
            method="add_scoped",
            input_schema={
                "type": "object",
                "properties": {
                    "messages": {"type": "string"},
                    "user_id": {"type": "string"},
                    "tenant_id": {"type": "string"},
                    "pool": {"type": "string"},
                    "infer": {"type": "boolean"},
                },
                "required": ["messages", "user_id"],
            },
        )
        handler = resolve_handler(cap)
        ctx = _make_ctx(scoped=scoped)

        result = await handler(
            ctx, messages="Explicit raw fact", user_id="u2", infer=False,
        )
        parsed = json.loads(result)
        assert parsed["results"][0]["id"] == "as-2"


# --------------------------------------------------------------------------- #
#  Test: Phase 3 — capability YAML validation for new tools
# --------------------------------------------------------------------------- #


class TestPhase3CapabilityYAMLs:
    """Validate the 5 new Phase 3 capability YAML files."""

    CAPS_DIR = None

    @pytest.fixture(autouse=True)
    def _set_caps_dir(self):
        from neomem_mcp.loader import get_bundled_capabilities_dir
        TestPhase3CapabilityYAMLs.CAPS_DIR = get_bundled_capabilities_dir()

    @pytest.mark.parametrize("yaml_file,expected_tool,expected_connector", [
        ("neomem_store_preference.capability.yaml", "neomem_store_preference", "low_level"),
        ("neomem_get_all_history.capability.yaml", "neomem_get_all_history", "high_level"),
        ("neomem_get_history_stats.capability.yaml", "neomem_get_history_stats", "high_level"),
        ("neomem_get_all_scoped.capability.yaml", "neomem_get_all_scoped", "scoped"),
        ("neomem_add_scoped.capability.yaml", "neomem_add_scoped", "scoped"),
    ])
    def test_yaml_structure(self, yaml_file, expected_tool, expected_connector):
        path = os.path.join(self.CAPS_DIR, yaml_file)
        assert os.path.exists(path), f"Missing: {yaml_file}"
        with open(path) as f:
            data = yaml.safe_load(f)
        assert data["mcp_tool_name"] == expected_tool
        assert data["connector"] == expected_connector
        assert "input_schema" in data
        assert "description" in data
        assert data.get("version") == "0.3.0"

    def test_store_preference_schema_has_required_fields(self):
        path = os.path.join(self.CAPS_DIR, "neomem_store_preference.capability.yaml")
        with open(path) as f:
            data = yaml.safe_load(f)
        props = data["input_schema"]["properties"]
        assert "preference" in props
        assert "user_id" in props
        assert set(data["input_schema"]["required"]) == {"preference", "user_id"}

    def test_scoped_tools_require_memory_utils(self):
        for name in ["neomem_get_all_scoped", "neomem_add_scoped"]:
            path = os.path.join(self.CAPS_DIR, f"{name}.capability.yaml")
            with open(path) as f:
                data = yaml.safe_load(f)
            assert data["required_package"] == "memory_utils"


# --------------------------------------------------------------------------- #
#  Test: sync method handling in handler
# --------------------------------------------------------------------------- #


class TestSyncMethodHandling:
    """Verify the handler correctly dispatches sync methods."""

    @pytest.mark.asyncio
    async def test_sync_method_called_without_await(self):
        """get_all_history and get_history_stats are sync — handler must not await them."""
        from neomem_mcp._handlers import resolve_handler

        hlc = MagicMock()
        hlc.get_history_stats = MagicMock(return_value={"total": 5})

        cap = _make_cap(
            name="memory.get_history_stats",
            mcp_tool_name="neomem_get_history_stats",
            connector="high_level",
            method="get_history_stats",
            input_schema={"type": "object", "properties": {}, "required": []},
        )
        handler = resolve_handler(cap)
        ctx = _make_ctx(hlc=hlc)

        result = await handler(ctx)
        parsed = json.loads(result)
        assert parsed["total"] == 5
        hlc.get_history_stats.assert_called_once()
