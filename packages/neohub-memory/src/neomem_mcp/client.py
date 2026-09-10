"""Universal MCP client for neomem server.

Contains ZERO tool-specific logic. Discovers and calls whatever the
server has registered.

Usage:
    client = NeoMemMCPClient()
    await client.connect()
    result = await client.call_tool("neomem_store_exchange", {...})
    await client.close()
"""

import json
import logging
from typing import Any, Optional

from neomem_mcp.models import ToolMetadata

logger = logging.getLogger(__name__)


class NeoMemMCPClient:
    """Universal MCP client — no tool-specific methods."""

    def __init__(self, host: str = "localhost", port: int = 18432):
        self.host = host
        self.port = port
        self._session = None
        self._transport_ctx = None

    async def connect(self) -> None:
        """Connect to the MCP server via Streamable HTTP."""
        from mcp import ClientSession
        from mcp.client.streamable_http import streamablehttp_client

        url = f"http://{self.host}:{self.port}/mcp"
        self._transport_ctx = streamablehttp_client(url)
        read, write, _ = await self._transport_ctx.__aenter__()
        self._session = ClientSession(read, write)
        await self._session.__aenter__()
        await self._session.initialize()
        logger.info("Connected to MCP server at %s", url)

    async def list_tools(self) -> list:
        """List all registered tools (raw MCP tool objects)."""
        self._ensure_connected()
        result = await self._session.list_tools()
        return result.tools

    async def list_tools_with_metadata(self) -> list[ToolMetadata]:
        """List tools with execution_mode and hook_position metadata.

        Frameworks use this to auto-wire mandatory vs optional tools.
        """
        tools = await self.list_tools()
        metadata_list = []
        for t in tools:
            annotations = t.annotations or {}
            # annotations may be a ToolAnnotations model, extract dict-like
            ann_dict = {}
            if hasattr(annotations, "model_dump"):
                ann_dict = annotations.model_dump()
            elif isinstance(annotations, dict):
                ann_dict = annotations

            metadata_list.append(
                ToolMetadata(
                    name=t.name,
                    description=t.description or "",
                    execution_mode=ann_dict.get("execution_mode", "optional"),
                    hook_position=ann_dict.get("hook_position"),
                    required_package=ann_dict.get("required_package", ""),
                    connector=ann_dict.get("connector", ""),
                    version=ann_dict.get("version", ""),
                    input_schema=t.inputSchema or {},
                )
            )
        return metadata_list

    async def call_tool(self, name: str, arguments: dict) -> dict:
        """Call a tool by name with arguments.

        Args:
            name: Tool name (e.g. "neomem_store_exchange").
            arguments: Tool arguments as a dict.

        Returns:
            Parsed JSON response as a dict.
        """
        self._ensure_connected()
        result = await self._session.call_tool(name, arguments)

        for content in result.content:
            if hasattr(content, "text"):
                try:
                    return json.loads(content.text)
                except json.JSONDecodeError:
                    return {"result": content.text}
        return {"result": None}

    async def read_resource(self, uri: str) -> str:
        """Read a resource by URI."""
        self._ensure_connected()
        from pydantic import AnyUrl

        result = await self._session.read_resource(AnyUrl(uri))
        for content in result.contents:
            if hasattr(content, "text"):
                return content.text
        return ""

    async def close(self) -> None:
        """Close the MCP connection."""
        if self._session:
            await self._session.__aexit__(None, None, None)
            self._session = None
        if self._transport_ctx:
            await self._transport_ctx.__aexit__(None, None, None)
            self._transport_ctx = None
        logger.info("MCP client disconnected")

    async def __aenter__(self) -> "NeoMemMCPClient":
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.close()

    def _ensure_connected(self) -> None:
        if self._session is None:
            raise RuntimeError(
                "Not connected. Call await client.connect() first or use 'async with'."
            )
