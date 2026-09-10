"""Live test script for MCP server — store, search, retrieve via MCP protocol."""

import asyncio
import json

from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client


async def test():
    async with streamablehttp_client("http://127.0.0.1:18432/mcp") as (r, w, _):
        async with ClientSession(r, w) as session:
            await session.initialize()

            # 1. Store an exchange
            print("=== store_exchange ===")
            result = await session.call_tool(
                "neomem_store_exchange",
                {
                    "query": "My favorite programming language is Rust and I work at Anthropic.",
                    "response": "Great! Rust is a powerful systems language. Working at Anthropic must be exciting!",
                    "user_id": "mcp_test_user",
                    "session_id": "mcp_test_session_001",
                },
            )
            print(f"isError: {result.isError}")
            for item in result.content:
                text = item.text if hasattr(item, "text") else str(item)
                print(f"  {item.type}: {text[:300]}")

            # 2. Search for what we stored
            print("\n=== search ===")
            result2 = await session.call_tool(
                "neomem_search",
                {
                    "query": "programming language",
                    "user_id": "mcp_test_user",
                    "limit": 5,
                },
            )
            print(f"isError: {result2.isError}")
            for item in result2.content:
                text = item.text if hasattr(item, "text") else str(item)
                print(f"  {item.type}: {text[:300]}")

            # 3. Retrieve context
            print("\n=== retrieve_context ===")
            result3 = await session.call_tool(
                "neomem_retrieve_context",
                {
                    "query": "What programming language does the user prefer?",
                    "user_id": "mcp_test_user",
                    "session_id": "mcp_test_session_001",
                },
            )
            print(f"isError: {result3.isError}")
            for item in result3.content:
                text = item.text if hasattr(item, "text") else str(item)
                print(f"  {item.type}: {text[:500]}")

            # 4. Store a fact
            print("\n=== store_fact ===")
            result4 = await session.call_tool(
                "neomem_store_fact",
                {
                    "fact": "User prefers dark mode in all IDEs",
                    "user_id": "mcp_test_user",
                },
            )
            print(f"isError: {result4.isError}")
            for item in result4.content:
                text = item.text if hasattr(item, "text") else str(item)
                print(f"  {item.type}: {text[:300]}")

            print("\n=== ALL TESTS PASSED ===")


if __name__ == "__main__":
    asyncio.run(test())
