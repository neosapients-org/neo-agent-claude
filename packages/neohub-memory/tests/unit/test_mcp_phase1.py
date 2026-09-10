"""Unit tests for neomem_mcp Phase 1 — models, config, loader, handlers, server, cli.

Tests the MCP layer in isolation — mocks all library calls.
Does NOT require Qdrant, OpenAI, or any external service.
"""

import json
import os
import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import yaml

# --------------------------------------------------------------------------- #
#  Test: models.py
# --------------------------------------------------------------------------- #


class TestCapabilityDefinition:
    """Tests for CapabilityDefinition dataclass."""

    def test_from_dict_minimal(self):
        from neomem_mcp.models import CapabilityDefinition

        data = {
            "name": "memory.test",
            "mcp_tool_name": "neomem_test",
            "required_package": "neo_memory_hub",
            "method": "test_method",
            "description": "A test tool",
        }
        cap = CapabilityDefinition.from_dict(data)

        assert cap.name == "memory.test"
        assert cap.mcp_tool_name == "neomem_test"
        assert cap.required_package == "neo_memory_hub"
        assert cap.method == "test_method"
        assert cap.description == "A test tool"
        assert cap.connector == "high_level"  # default
        assert cap.execution_mode == "optional"  # default
        assert cap.hook_position is None
        assert cap.mcp_type == "tool"
        assert cap.version == "0.3.0"
        assert cap.input_schema == {}

    def test_from_dict_full(self):
        from neomem_mcp.models import CapabilityDefinition

        data = {
            "name": "memory.store_exchange",
            "mcp_tool_name": "neomem_store_exchange",
            "required_package": "neo_memory_hub",
            "connector": "high_level",
            "method": "store_exchange",
            "description": "Store a conversation exchange",
            "execution_mode": "mandatory",
            "hook_position": "post",
            "input_schema": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
            "annotations": {"readOnlyHint": False},
            "version": "0.3.0",
            "mcp_type": "tool",
        }
        cap = CapabilityDefinition.from_dict(data, source_file="/caps/test.yaml")

        assert cap.execution_mode == "mandatory"
        assert cap.hook_position == "post"
        assert cap.input_schema["type"] == "object"
        assert cap.source_file == "/caps/test.yaml"

    def test_to_dict(self):
        from neomem_mcp.models import CapabilityDefinition

        cap = CapabilityDefinition(
            name="memory.test",
            mcp_tool_name="neomem_test",
            required_package="neo_memory_hub",
            connector="high_level",
            method="test",
            description="test",
        )
        d = cap.to_dict()
        assert d["name"] == "memory.test"
        assert d["mcp_tool_name"] == "neomem_test"
        assert "mcp_resource_uri" not in d  # only included when set

    def test_to_dict_with_resource_uri(self):
        from neomem_mcp.models import CapabilityDefinition

        cap = CapabilityDefinition(
            name="memory.config",
            mcp_tool_name="",
            required_package="neo_memory_hub",
            connector="high_level",
            method="get_config",
            description="Config resource",
            mcp_type="resource",
            mcp_resource_uri="neomem://config",
        )
        d = cap.to_dict()
        assert d["mcp_resource_uri"] == "neomem://config"


class TestToolMetadata:
    """Tests for ToolMetadata dataclass."""

    def test_is_mandatory(self):
        from neomem_mcp.models import ToolMetadata

        tm = ToolMetadata(
            name="test",
            description="test",
            execution_mode="mandatory",
            hook_position="post",
            required_package="neo_memory_hub",
            connector="high_level",
            version="0.3.0",
        )
        assert tm.is_mandatory()
        assert tm.is_post_hook()
        assert not tm.is_pre_hook()

    def test_is_optional(self):
        from neomem_mcp.models import ToolMetadata

        tm = ToolMetadata(
            name="test",
            description="test",
            execution_mode="optional",
            hook_position=None,
            required_package="neo_memory_hub",
            connector="high_level",
            version="0.3.0",
        )
        assert not tm.is_mandatory()
        assert not tm.is_pre_hook()
        assert not tm.is_post_hook()


# --------------------------------------------------------------------------- #
#  Test: config.py
# --------------------------------------------------------------------------- #


class TestMCPServerConfig:
    """Tests for MCPServerConfig."""

    def test_defaults(self):
        from neomem_mcp.config import MCPServerConfig

        config = MCPServerConfig()
        assert config.host == "127.0.0.1"
        assert config.port == 18432
        assert config.transport == "streamable-http"
        assert config.qdrant_url == "http://localhost:6335"
        assert config.llm_model == "gpt-4o-mini"
        assert config.salience_enabled is True

    def test_from_env(self):
        from neomem_mcp.config import MCPServerConfig

        env = {
            "NEOMEM_MCP_HOST": "0.0.0.0",
            "NEOMEM_MCP_PORT": "9999",
            "NEOMEM_MCP_TRANSPORT": "stdio",
            "NEOMEM_QDRANT_URL": "http://qdrant:6334",
            "NEOMEM_LLM_MODEL": "gpt-4o",
            "NEOMEM_SALIENCE_ENABLED": "false",
        }
        with patch.dict(os.environ, env, clear=False):
            config = MCPServerConfig.from_env()

        assert config.host == "0.0.0.0"
        assert config.port == 9999
        assert config.transport == "stdio"
        assert config.qdrant_url == "http://qdrant:6334"
        assert config.llm_model == "gpt-4o"
        assert config.salience_enabled is False

    def test_from_yaml(self):
        from neomem_mcp.config import MCPServerConfig

        yaml_content = {
            "host": "10.0.0.1",
            "port": 8888,
            "qdrant_url": "http://qdrant-prod:6335",
            "llm_model": "gpt-4o",
        }

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            yaml.dump(yaml_content, f)
            f.flush()
            # Remove any env overrides that would shadow the YAML values
            env_overrides = {
                "NEOMEM_MCP_HOST": "",
                "NEOMEM_MCP_PORT": "",
                "NEOMEM_QDRANT_URL": "",
                "NEOMEM_LLM_MODEL": "",
            }
            with patch.dict(os.environ, {}, clear=False):
                # Temporarily remove env vars that would override YAML
                for k in env_overrides:
                    os.environ.pop(k, None)
                config = MCPServerConfig.from_yaml(f.name)

        os.unlink(f.name)
        assert config.port == 8888
        assert config.qdrant_url == "http://qdrant-prod:6335"

    def test_from_yaml_missing_file(self):
        from neomem_mcp.config import MCPServerConfig

        config = MCPServerConfig.from_yaml("/nonexistent/path.yaml")
        # Falls back to from_env defaults
        assert config.host == "127.0.0.1"

    def test_build_memory_config(self):
        from neomem_mcp.config import MCPServerConfig, build_memory_config

        config = MCPServerConfig(
            qdrant_url="http://qdrant:6335",
            qdrant_collection="test_collection",
            llm_model="gpt-4o",
            llm_temperature=0.2,
        )
        mem_config = build_memory_config(config)

        # Returns a MemoryHubConfig Pydantic object, not a dict
        assert mem_config.vector_store.provider == "qdrant"
        assert mem_config.vector_store.qdrant_url == "http://qdrant:6335"
        assert mem_config.vector_store.collection_name == "test_collection"
        assert mem_config.llm.model == "gpt-4o"
        assert mem_config.llm.temperature == 0.2


# --------------------------------------------------------------------------- #
#  Test: loader.py
# --------------------------------------------------------------------------- #


class TestLoader:
    """Tests for capability loading and package detection."""

    def _create_capability_yaml(self, dir_path: str, name: str, data: dict) -> str:
        """Helper to write a capability YAML file."""
        filepath = os.path.join(dir_path, f"{name}.capability.yaml")
        with open(filepath, "w") as f:
            yaml.dump(data, f)
        return filepath

    def test_load_capabilities_from_directory(self):
        from neomem_mcp.loader import load_capabilities

        with tempfile.TemporaryDirectory() as tmpdir:
            self._create_capability_yaml(tmpdir, "tool_a", {
                "name": "a",
                "mcp_tool_name": "neomem_a",
                "required_package": "neo_memory_hub",
                "method": "do_a",
                "description": "Tool A",
            })
            self._create_capability_yaml(tmpdir, "tool_b", {
                "name": "b",
                "mcp_tool_name": "neomem_b",
                "required_package": "memory_utils",
                "method": "do_b",
                "description": "Tool B",
            })

            caps = load_capabilities(tmpdir)
            assert len(caps) == 2
            assert caps[0].name == "a"
            assert caps[1].name == "b"

    def test_load_capabilities_empty_directory(self):
        from neomem_mcp.loader import load_capabilities

        with tempfile.TemporaryDirectory() as tmpdir:
            caps = load_capabilities(tmpdir)
            assert caps == []

    def test_load_capabilities_missing_directory(self):
        from neomem_mcp.loader import load_capabilities

        caps = load_capabilities("/nonexistent/path/to/caps")
        assert caps == []

    def test_load_capabilities_invalid_yaml(self):
        from neomem_mcp.loader import load_capabilities

        with tempfile.TemporaryDirectory() as tmpdir:
            # Write invalid YAML
            filepath = os.path.join(tmpdir, "bad.capability.yaml")
            with open(filepath, "w") as f:
                f.write("{{{invalid yaml")

            caps = load_capabilities(tmpdir)
            assert caps == []  # skip invalid files

    def test_load_capabilities_missing_required_field(self):
        from neomem_mcp.loader import load_capabilities

        with tempfile.TemporaryDirectory() as tmpdir:
            self._create_capability_yaml(tmpdir, "incomplete", {
                "description": "Missing required fields",
            })
            caps = load_capabilities(tmpdir)
            assert caps == []  # skip files with missing fields

    def test_detect_installed_packages(self):
        from neomem_mcp.loader import detect_installed_packages

        installed = detect_installed_packages()
        assert "neo_memory_hub" in installed  # should be importable in test env

    def test_filter_capabilities(self):
        from neomem_mcp.loader import filter_capabilities
        from neomem_mcp.models import CapabilityDefinition

        caps = [
            CapabilityDefinition(
                name="a", mcp_tool_name="neomem_a",
                required_package="neo_memory_hub",
                connector="high_level", method="a", description="A",
            ),
            CapabilityDefinition(
                name="b", mcp_tool_name="neomem_b",
                required_package="fake_package",
                connector="high_level", method="b", description="B",
            ),
        ]

        registerable, skipped = filter_capabilities(
            caps, {"neo_memory_hub"}
        )
        assert len(registerable) == 1
        assert registerable[0].name == "a"
        assert len(skipped) == 1
        assert skipped[0].name == "b"

    def test_filter_capabilities_by_type(self):
        from neomem_mcp.loader import filter_capabilities
        from neomem_mcp.models import CapabilityDefinition

        caps = [
            CapabilityDefinition(
                name="tool", mcp_tool_name="neomem_tool",
                required_package="neo_memory_hub",
                connector="high_level", method="do", description="T",
                mcp_type="tool",
            ),
            CapabilityDefinition(
                name="res", mcp_tool_name="",
                required_package="neo_memory_hub",
                connector="high_level", method="get", description="R",
                mcp_type="resource",
            ),
        ]

        tools, _ = filter_capabilities(caps, {"neo_memory_hub"}, mcp_type="tool")
        assert len(tools) == 1
        assert tools[0].name == "tool"

        resources, _ = filter_capabilities(caps, {"neo_memory_hub"}, mcp_type="resource")
        assert len(resources) == 1
        assert resources[0].name == "res"


# --------------------------------------------------------------------------- #
#  Test: _handlers.py
# --------------------------------------------------------------------------- #


class TestHandlers:
    """Tests for the handler bridge layer."""

    def _make_cap(self, **overrides) -> "CapabilityDefinition":
        from neomem_mcp.models import CapabilityDefinition

        defaults = {
            "name": "memory.test",
            "mcp_tool_name": "neomem_test",
            "required_package": "neo_memory_hub",
            "connector": "high_level",
            "method": "test_method",
            "description": "Test tool",
        }
        defaults.update(overrides)
        return CapabilityDefinition(**defaults)

    def _make_ctx(self, hlc=None, low_level=None, scoped=None, scorer=None):
        """Create a mock MCP context with AppContext."""
        app = MagicMock()
        app.hlc = hlc
        app.low_level = low_level
        app.scoped = scoped
        app.scorer = scorer

        ctx = MagicMock()
        ctx.request_context.lifespan_context = app
        return ctx

    @pytest.mark.asyncio
    async def test_high_level_handler_success(self):
        from neomem_mcp._handlers import resolve_handler

        hlc = AsyncMock()
        hlc.store_exchange = AsyncMock(return_value={
            "facts_stored": 3,
            "facts_skipped": 1,
            "facts_extracted": 4,
        })

        cap = self._make_cap(method="store_exchange")
        handler = resolve_handler(cap)
        ctx = self._make_ctx(hlc=hlc)

        result = await handler(ctx, query="test", response="answer", user_id="u1")
        parsed = json.loads(result)

        assert parsed["facts_stored"] == 3
        assert parsed["facts_skipped"] == 1
        hlc.store_exchange.assert_called_once_with(
            query="test", response="answer", user_id="u1"
        )

    @pytest.mark.asyncio
    async def test_handler_returns_none_result(self):
        from neomem_mcp._handlers import resolve_handler

        hlc = AsyncMock()
        hlc.get = AsyncMock(return_value=None)

        cap = self._make_cap(method="get")
        handler = resolve_handler(cap)
        ctx = self._make_ctx(hlc=hlc)

        result = await handler(ctx, memory_id="test123")
        parsed = json.loads(result)
        assert parsed["result"] is None

    @pytest.mark.asyncio
    async def test_handler_returns_list(self):
        from neomem_mcp._handlers import resolve_handler

        hlc = AsyncMock()
        hlc.test_method = AsyncMock(return_value=["a", "b", "c"])

        cap = self._make_cap()
        handler = resolve_handler(cap)
        ctx = self._make_ctx(hlc=hlc)

        result = await handler(ctx)
        parsed = json.loads(result)
        assert parsed["results"] == ["a", "b", "c"]

    @pytest.mark.asyncio
    async def test_handler_error_boundary(self):
        from neomem_mcp._handlers import resolve_handler

        hlc = AsyncMock()
        hlc.test_method = AsyncMock(side_effect=RuntimeError("boom"))

        cap = self._make_cap()
        handler = resolve_handler(cap)
        ctx = self._make_ctx(hlc=hlc)

        result = await handler(ctx)
        parsed = json.loads(result)

        assert parsed["error"] == "InternalError"
        assert "neomem_test" in parsed["message"]

    @pytest.mark.asyncio
    async def test_handler_validation_error(self):
        from neomem_mcp._handlers import resolve_handler

        hlc = AsyncMock()
        # Simulate library ValidationError
        from neo_memory_hub.core.exceptions import ValidationError

        hlc.test_method = AsyncMock(side_effect=ValidationError("user_id required"))

        cap = self._make_cap()
        handler = resolve_handler(cap)
        ctx = self._make_ctx(hlc=hlc)

        result = await handler(ctx)
        parsed = json.loads(result)
        assert parsed["error"] == "ValidationError"
        assert "user_id" in parsed["message"]

    @pytest.mark.asyncio
    async def test_scoped_handler_not_installed(self):
        from neomem_mcp._handlers import resolve_handler

        cap = self._make_cap(connector="scoped", method="add_scoped")
        handler = resolve_handler(cap)
        ctx = self._make_ctx(scoped=None)  # memory_utils not installed

        result = await handler(ctx, messages="test", tenant_id="t1")
        parsed = json.loads(result)
        assert parsed["error"] == "PackageNotInstalled"

    @pytest.mark.asyncio
    async def test_scoped_handler_success(self):
        from neomem_mcp._handlers import resolve_handler

        scoped = AsyncMock()
        scoped.add_scoped = AsyncMock(return_value={"results": [{"id": "123"}]})

        cap = self._make_cap(connector="scoped", method="add_scoped")
        handler = resolve_handler(cap)
        ctx = self._make_ctx(scoped=scoped)

        result = await handler(
            ctx, messages="test fact", tenant_id="t1", user_id="u1", pool="private"
        )
        parsed = json.loads(result)
        assert parsed["results"][0]["id"] == "123"
        # Verify scope was constructed from flat params
        scoped.add_scoped.assert_called_once()
        call_kwargs = scoped.add_scoped.call_args
        assert call_kwargs.kwargs.get("messages") == "test fact"

    @pytest.mark.asyncio
    async def test_scorer_handler(self):
        from neomem_mcp._handlers import resolve_handler

        scorer = AsyncMock()
        mock_fact = MagicMock()
        mock_fact.text = "User likes equity"
        mock_fact.salience = 0.9
        mock_fact.reasoning = "high relevance"
        mock_fact.__dict__ = {"text": "User likes equity", "salience": 0.9, "reasoning": "high relevance"}
        scorer.score_facts = AsyncMock(return_value=[mock_fact])

        cap = self._make_cap(connector="scorer", method="score_facts")
        handler = resolve_handler(cap)
        ctx = self._make_ctx(scorer=scorer)

        result = await handler(ctx, facts=["User likes equity"])
        parsed = json.loads(result)
        assert "results" in parsed

    @pytest.mark.asyncio
    async def test_handler_method_not_found(self):
        from neomem_mcp._handlers import resolve_handler

        hlc = MagicMock(spec=[])  # no methods at all

        cap = self._make_cap(method="nonexistent_method")
        handler = resolve_handler(cap)
        ctx = self._make_ctx(hlc=hlc)

        result = await handler(ctx)
        parsed = json.loads(result)
        assert parsed["error"] == "MethodNotFound"

    def test_serialize_result_dict(self):
        from neomem_mcp._handlers import _serialize_result

        assert _serialize_result({"key": "val"}) == {"key": "val"}

    def test_serialize_result_none(self):
        from neomem_mcp._handlers import _serialize_result

        assert _serialize_result(None) == {"result": None}

    def test_serialize_result_string(self):
        from neomem_mcp._handlers import _serialize_result

        assert _serialize_result("hello") == {"result": "hello"}

    def test_serialize_result_int(self):
        from neomem_mcp._handlers import _serialize_result

        assert _serialize_result(42) == {"result": 42}

    def test_serialize_result_pydantic_model(self):
        from neomem_mcp._handlers import _serialize_result

        model = MagicMock()
        model.model_dump = MagicMock(return_value={"a": 1, "b": 2})

        assert _serialize_result(model) == {"a": 1, "b": 2}

    def test_make_error_response_generic(self):
        from neomem_mcp._handlers import _make_error_response

        result = _make_error_response(RuntimeError("boom"), "test_tool")
        assert result["error"] == "InternalError"
        assert "test_tool" in result["message"]
        # Must NOT contain the actual error message — security boundary
        assert "boom" not in result["message"]


# --------------------------------------------------------------------------- #
#  Test: server.py — create_mcp_server registration
# --------------------------------------------------------------------------- #


class TestServerBootstrap:
    """Tests for MCP server creation and tool registration."""

    def test_create_mcp_server_with_capabilities(self):
        """Verify server registers tools from capability YAMLs."""
        from neomem_mcp.config import MCPServerConfig

        from neomem_mcp.loader import get_bundled_capabilities_dir
        caps_dir = get_bundled_capabilities_dir()

        with patch("neomem_mcp.server.detect_installed_packages") as mock_detect:
            mock_detect.return_value = {"neo_memory_hub"}

            config = MCPServerConfig(capabilities_dir=caps_dir)
            from neomem_mcp.server import create_mcp_server

            mcp = create_mcp_server(server_config=config, capabilities_dir=caps_dir)

        # Should have registered the 5 Phase 1 tools
        assert mcp is not None

    def test_create_mcp_server_empty_caps(self):
        """Server with no capabilities should still start (0 tools)."""
        from neomem_mcp.config import MCPServerConfig

        with tempfile.TemporaryDirectory() as tmpdir:
            config = MCPServerConfig(capabilities_dir=tmpdir)
            from neomem_mcp.server import create_mcp_server

            mcp = create_mcp_server(server_config=config, capabilities_dir=tmpdir)
            assert mcp is not None


# --------------------------------------------------------------------------- #
#  Test: Capability YAMLs — validate all Phase 1 YAMLs parse correctly
# --------------------------------------------------------------------------- #


class TestCapabilityYAMLs:
    """Validate the actual capability YAML files in capabilities/."""

    CAPS_DIR = None  # resolved in setup

    @pytest.fixture(autouse=True)
    def _set_caps_dir(self):
        from neomem_mcp.loader import get_bundled_capabilities_dir
        TestCapabilityYAMLs.CAPS_DIR = get_bundled_capabilities_dir()

    def test_all_yaml_files_parse(self):
        from neomem_mcp.loader import load_capabilities

        caps = load_capabilities(self.CAPS_DIR)
        assert len(caps) >= 5, f"Expected at least 5 capabilities, got {len(caps)}"

    def test_store_exchange_capability(self):
        cap = self._load_cap("neomem_store_exchange")
        assert cap.mcp_tool_name == "neomem_store_exchange"
        assert cap.required_package == "neo_memory_hub"
        assert cap.connector == "high_level"
        assert cap.method == "store_exchange"
        assert cap.execution_mode == "mandatory"
        assert cap.hook_position == "post"
        assert "query" in cap.input_schema["properties"]
        assert "response" in cap.input_schema["properties"]
        assert "user_id" in cap.input_schema["properties"]
        assert "query" in cap.input_schema["required"]
        assert "response" in cap.input_schema["required"]
        assert "user_id" in cap.input_schema["required"]

    def test_retrieve_context_capability(self):
        cap = self._load_cap("neomem_retrieve_context")
        assert cap.execution_mode == "mandatory"
        assert cap.hook_position == "pre"
        assert cap.method == "retrieve_context"
        assert cap.annotations.get("readOnlyHint") is True

    def test_search_capability(self):
        cap = self._load_cap("neomem_search")
        assert cap.execution_mode == "optional"
        assert cap.hook_position is None or cap.hook_position == "null"
        assert cap.method == "search"

    def test_store_fact_capability(self):
        cap = self._load_cap("neomem_store_fact")
        assert cap.execution_mode == "mandatory"
        assert cap.hook_position == "post"
        assert cap.method == "store_fact"
        assert "fact" in cap.input_schema["properties"]

    def test_get_memory_capability(self):
        cap = self._load_cap("neomem_get_memory")
        assert cap.execution_mode == "optional"
        assert cap.method == "get"
        assert "memory_id" in cap.input_schema["properties"]

    def test_all_capabilities_have_required_fields(self):
        from neomem_mcp.loader import load_capabilities

        caps = load_capabilities(self.CAPS_DIR)
        for cap in caps:
            assert cap.name, f"Missing name in {cap.source_file}"
            assert cap.mcp_tool_name or cap.mcp_type == "resource", (
                f"Missing mcp_tool_name in {cap.source_file}"
            )
            assert cap.required_package, f"Missing required_package in {cap.source_file}"
            assert cap.method or cap.mcp_type == "resource", (
                f"Missing method in {cap.source_file}"
            )
            assert cap.description, f"Missing description in {cap.source_file}"
            assert cap.version, f"Missing version in {cap.source_file}"

    def _load_cap(self, tool_name: str):
        from neomem_mcp.loader import load_capabilities

        caps = load_capabilities(self.CAPS_DIR)
        for cap in caps:
            if cap.mcp_tool_name == tool_name:
                return cap
        pytest.fail(f"Capability {tool_name} not found in {self.CAPS_DIR}")


# --------------------------------------------------------------------------- #
#  Test: CLI
# --------------------------------------------------------------------------- #


class TestCLI:
    """Tests for CLI commands."""

    def test_validate_runs_without_crash(self):
        """Validate command should complete without errors."""
        import argparse

        from neomem_mcp.cli import cmd_validate

        args = argparse.Namespace(
            capabilities_dir=None  # use default bundled path
        )
        # Should not raise
        cmd_validate(args)

    def test_validate_with_missing_caps_dir(self):
        """Validate with missing caps dir should report warning but not crash."""
        import argparse

        from neomem_mcp.cli import cmd_validate

        args = argparse.Namespace(capabilities_dir="/nonexistent")
        # Will print warning but should not crash
        with pytest.raises(SystemExit) as exc_info:
            cmd_validate(args)
        assert exc_info.value.code == 1  # errors found (no capabilities)


# --------------------------------------------------------------------------- #
#  Test: Integration — end-to-end handler pipeline
# --------------------------------------------------------------------------- #


class TestEndToEndHandlerPipeline:
    """Full pipeline: YAML → CapabilityDefinition → Handler → Mock library call."""

    @pytest.mark.asyncio
    async def test_store_exchange_full_pipeline(self):
        """Load real YAML, create handler, call with mock connector."""
        from neomem_mcp._handlers import resolve_handler
        from neomem_mcp.loader import load_capabilities, get_bundled_capabilities_dir

        caps_dir = get_bundled_capabilities_dir()
        caps = load_capabilities(caps_dir)
        store_cap = next(c for c in caps if c.mcp_tool_name == "neomem_store_exchange")

        handler = resolve_handler(store_cap)

        # Mock the AppContext
        hlc = AsyncMock()
        hlc.store_exchange = AsyncMock(return_value={
            "facts_stored": 2,
            "facts_skipped": 0,
            "facts_extracted": 2,
            "results": [
                {"id": "mem_1", "memory": "Rajesh prefers equity"},
                {"id": "mem_2", "memory": "Rajesh has 2.3M portfolio"},
            ],
            "investor_name": "Rajesh Kumar",
        })

        ctx = MagicMock()
        ctx.request_context.lifespan_context.hlc = hlc

        result = await handler(
            ctx,
            query="Tell me about Rajesh",
            response="Rajesh is a conservative investor with $2.3M portfolio",
            user_id="rm_001",
            investor_name="Rajesh Kumar",
        )
        parsed = json.loads(result)

        assert parsed["facts_stored"] == 2
        assert parsed["investor_name"] == "Rajesh Kumar"
        assert len(parsed["results"]) == 2

        hlc.store_exchange.assert_called_once_with(
            query="Tell me about Rajesh",
            response="Rajesh is a conservative investor with $2.3M portfolio",
            user_id="rm_001",
            investor_name="Rajesh Kumar",
        )

    @pytest.mark.asyncio
    async def test_retrieve_context_full_pipeline(self):
        """Load real YAML, create handler, call with mock connector."""
        from neomem_mcp._handlers import resolve_handler
        from neomem_mcp.loader import load_capabilities, get_bundled_capabilities_dir

        caps_dir = get_bundled_capabilities_dir()
        caps = load_capabilities(caps_dir)
        retrieve_cap = next(c for c in caps if c.mcp_tool_name == "neomem_retrieve_context")

        handler = resolve_handler(retrieve_cap)

        hlc = AsyncMock()
        hlc.retrieve_context = AsyncMock(return_value={
            "context": "Known facts about user:\n- Prefers equity...",
            "memories": [{"id": "m1", "memory": "Prefers equity"}],
            "count": 1,
            "user_id": "rm_001",
        })

        ctx = MagicMock()
        ctx.request_context.lifespan_context.hlc = hlc

        result = await handler(ctx, query="What does user prefer?", user_id="rm_001")
        parsed = json.loads(result)

        assert "context" in parsed
        assert parsed["count"] == 1

    @pytest.mark.asyncio
    async def test_search_full_pipeline(self):
        from neomem_mcp._handlers import resolve_handler
        from neomem_mcp.loader import load_capabilities, get_bundled_capabilities_dir

        caps_dir = get_bundled_capabilities_dir()
        caps = load_capabilities(caps_dir)
        search_cap = next(c for c in caps if c.mcp_tool_name == "neomem_search")

        handler = resolve_handler(search_cap)

        hlc = AsyncMock()
        hlc.search = AsyncMock(return_value={
            "results": [
                {"id": "m1", "memory": "User likes conservative funds", "score": 0.92},
            ]
        })

        ctx = MagicMock()
        ctx.request_context.lifespan_context.hlc = hlc

        result = await handler(ctx, query="What funds does user like?", user_id="rm_001")
        parsed = json.loads(result)
        assert len(parsed["results"]) == 1
        assert parsed["results"][0]["score"] == 0.92

    @pytest.mark.asyncio
    async def test_get_memory_full_pipeline(self):
        from neomem_mcp._handlers import resolve_handler
        from neomem_mcp.loader import load_capabilities, get_bundled_capabilities_dir

        caps_dir = get_bundled_capabilities_dir()
        caps = load_capabilities(caps_dir)
        get_cap = next(c for c in caps if c.mcp_tool_name == "neomem_get_memory")

        handler = resolve_handler(get_cap)

        hlc = AsyncMock()
        hlc.get = AsyncMock(return_value={"id": "mem123", "memory": "A fact"})

        ctx = MagicMock()
        ctx.request_context.lifespan_context.hlc = hlc

        result = await handler(ctx, memory_id="mem123")
        parsed = json.loads(result)
        assert parsed["id"] == "mem123"
