"""Quick debug: test store_scoped & cleanup_expired via MCP."""
import json, anyio
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

async def test():
    async with streamablehttp_client("http://localhost:18434/mcp") as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()

            # Test store_scoped
            r = await session.call_tool("neomem_store_scoped", {
                "user_message": "Test msg",
                "assistant_response": "Test resp",
                "pool": "private",
                "user_id": "test_user",
                "agent_id": "test_agent",
            })
            print("store_scoped:")
            for c in r.content:
                if hasattr(c, "text"):
                    print(c.text[:500])

            # Test cleanup_expired
            r2 = await session.call_tool("neomem_cleanup_expired", {
                "user_id": "test_user",
            })
            print("\ncleanup_expired:")
            for c in r2.content:
                if hasattr(c, "text"):
                    print(c.text[:500])

            # Test score_salience
            r3 = await session.call_tool("neomem_score_salience", {
                "facts": ["Client wants to sell everything due to panic"],
                "context": "Market volatility discussion",
            })
            print("\nscore_salience:")
            for c in r3.content:
                if hasattr(c, "text"):
                    print(c.text[:500])

anyio.run(test)
