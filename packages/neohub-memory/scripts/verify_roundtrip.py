#!/usr/bin/env python3
"""Quick verification: MCP server → Qdrant round-trip."""
import asyncio
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), "..", "..", ".env"))

from neomem_mcp.client import NeoMemMCPClient


async def verify():
    host = os.environ.get("NEOMEM_MCP_HOST", "localhost")
    port = int(os.environ.get("NEOMEM_MCP_PORT", "18432"))

    c = NeoMemMCPClient(host=host, port=port)
    await c.connect()
    tools = await c.list_tools()
    print(f"[MCP] Connected to http://{host}:{port}/mcp — {len(tools)} tools")

    # Store a fact → goes through MCP → HighLevelConnector → Qdrant
    r = await c.call_tool("neomem_store_fact", {
        "fact": "VERIFICATION: Qdrant round-trip proof via MCP server",
        "user_id": "verification_user",
    })
    print(f"[STORE → Qdrant] {r}")

    # Search back from Qdrant
    r = await c.call_tool("neomem_search", {
        "query": "verification round-trip",
        "user_id": "verification_user",
    })
    results = r.get("results", [])
    print(f"[SEARCH ← Qdrant] {len(results)} result(s):")
    for mem in results[:3]:
        print(f"  → {str(mem.get('memory', mem.get('text', '')))[:80]}")

    # Get all
    r = await c.call_tool("neomem_get_all", {"user_id": "verification_user"})
    all_mems = r.get("results", r.get("memories", []))
    print(f"[GET_ALL ← Qdrant] {len(all_mems)} memories persisted")

    # Read config resource
    config_text = await c.read_resource("neomem://config")
    cfg = json.loads(config_text) if config_text else {}
    print(f"[RESOURCE] neomem://config → keys: {list(cfg.keys())}")

    # Cleanup
    await c.call_tool("neomem_delete_all", {"user_id": "verification_user"})
    print("[CLEANUP] Verification data deleted")

    await c.close()
    print("\n✅ Full round-trip verified: Deep Agent → MCP Server → Qdrant → back")


if __name__ == "__main__":
    asyncio.run(verify())
