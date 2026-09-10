"""MCP Server for neo_memory_hub — pure registration engine.

Contains ZERO tool definitions. Reads capability YAMLs at startup,
checks which packages are installed, and registers only the intersection.

Usage:
    neomem-mcp serve --port 18432
    neomem-mcp serve --transport stdio
"""

import json
import logging
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any, Optional

from mcp.server.fastmcp import FastMCP

from neomem_mcp._handlers import resolve_handler
from neomem_mcp.config import MCPServerConfig, build_memory_config
from neomem_mcp.loader import (
    detect_installed_packages,
    filter_capabilities,
    load_capabilities,
    resolve_capabilities_dir,
)
from neomem_mcp.models import CapabilityDefinition

logger = logging.getLogger(__name__)


@dataclass
class AppContext:
    """Resources that persist across all MCP tool calls.

    Created once in the lifespan, shared by all handlers.
    """

    hlc: Any            # HighLevelMemoryConnector
    low_level: Any      # NeoMemoryConnector (hlc._connector)
    scoped: Any | None  # ScopedMemoryConnector (None if memory_utils not installed)
    scorer: Any | None  # SalienceScorer
    capabilities: list[CapabilityDefinition]  # registered capabilities (for metadata)


@asynccontextmanager
async def app_lifespan(server: FastMCP) -> AsyncIterator[AppContext]:
    """Manage HighLevelMemoryConnector lifecycle.

    Initializes on startup, closes on shutdown. All tools share the
    same connector instance — no connection leak.
    """
    from dotenv import load_dotenv

    from neo_memory_hub import HighLevelMemoryConnector, SalienceScorer

    load_dotenv()  # Load .env for OPENAI_API_KEY etc.

    server_config = MCPServerConfig.from_env()
    config = build_memory_config(server_config)

    hlc = HighLevelMemoryConnector(config=config)
    await hlc.initialize()
    logger.info("HighLevelMemoryConnector initialized")

    scorer = SalienceScorer()

    # Optionally create ScopedMemoryConnector if memory_utils is installed
    scoped = None
    try:
        from memory_utils.shared_scope import ScopedMemoryConnector, SharedMemoryStrategy

        scoped = ScopedMemoryConnector(
            hlc._connector,
            strategy=SharedMemoryStrategy.ENABLED,
        )
        logger.info("memory_utils detected — scoped tools available")
    except ImportError:
        logger.info("memory_utils not installed — scoped tools disabled")

    # Load capabilities for metadata exposure
    caps_dir = resolve_capabilities_dir(server_config.capabilities_dir)
    all_caps = load_capabilities(caps_dir)
    installed = detect_installed_packages()
    registered_caps, _ = filter_capabilities(all_caps, installed, mcp_type="tool")

    try:
        yield AppContext(
            hlc=hlc,
            low_level=hlc._connector,
            scoped=scoped,
            scorer=scorer,
            capabilities=registered_caps,
        )
    finally:
        await hlc.close()
        logger.info("HighLevelMemoryConnector closed")


def create_mcp_server(
    server_config: Optional[MCPServerConfig] = None,
    capabilities_dir: Optional[str] = None,
) -> FastMCP:
    """Create and bootstrap the MCP server.

    1. Create FastMCP with lifespan for connector management
    2. Load all *.capability.yaml files
    3. Check which packages are installed
    4. Register tools whose required_package is installed
    5. Return the configured FastMCP server

    Args:
        server_config: Server configuration. Defaults to from_env().
        capabilities_dir: Override capabilities directory path.

    Returns:
        Configured FastMCP server ready to run.
    """
    if server_config is None:
        server_config = MCPServerConfig.from_env()

    caps_dir = resolve_capabilities_dir(capabilities_dir or server_config.capabilities_dir)

    mcp = FastMCP(
        "neomem_mcp",
        lifespan=app_lifespan,
        json_response=True,
        host=server_config.host,
        port=server_config.port,
    )

    # Load capabilities and register tools
    capabilities = load_capabilities(caps_dir)
    installed = detect_installed_packages()

    tools_registered, tools_skipped = filter_capabilities(
        capabilities, installed, mcp_type="tool"
    )

    for cap in tools_registered:
        handler_fn = resolve_handler(cap)
        mcp.tool(
            name=cap.mcp_tool_name,
            description=cap.description,
        )(handler_fn)
        logger.info("Registered tool: %s", cap.mcp_tool_name)

    # Register resources
    resources_registered, resources_skipped = filter_capabilities(
        capabilities, installed, mcp_type="resource"
    )
    for cap in resources_registered:
        _register_resource(mcp, cap)

    total_registered = len(tools_registered) + len(resources_registered)
    total_skipped = len(tools_skipped) + len(resources_skipped)
    logger.info(
        "MCP bootstrap complete: %d registered (%d tools, %d resources), %d skipped",
        total_registered,
        len(tools_registered),
        len(resources_registered),
        total_skipped,
    )

    return mcp


def _register_resource(mcp: FastMCP, cap: CapabilityDefinition) -> None:
    """Register an MCP resource from a capability definition."""
    if not cap.mcp_resource_uri:
        logger.warning("Resource %s has no mcp_resource_uri, skipping", cap.name)
        return

    @mcp.resource(cap.mcp_resource_uri)
    async def resource_handler() -> str:
        """Dynamic resource handler."""
        try:
            # Config resource — return server config summary
            if cap.method is None or cap.method == "null":
                config_data = {
                    "server": {"name": "neomem_mcp", "version": "0.1.0"},
                    "resource": cap.name,
                }
                return json.dumps(config_data, indent=2, default=str)
            return "{}"
        except Exception as e:
            logger.exception("Error in resource %s", cap.name)
            return json.dumps({"error": type(e).__name__})

    resource_handler.__name__ = f"resource_{cap.name}"
    resource_handler.__doc__ = cap.description
    logger.info("Registered resource: %s → %s", cap.name, cap.mcp_resource_uri)
