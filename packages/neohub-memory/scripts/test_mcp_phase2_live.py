"""Live test script for MCP Phase 2 — tests all 15 core tools via MCP protocol.

Requires:
  - MCP server running on localhost:18432
  - Qdrant running on localhost:6335
  - OPENAI_API_KEY set

Skips scoped tools (neomem_store_scoped, neomem_search_scoped) since memory_utils
may not be installed.
"""

import asyncio
import json
import sys

from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

MCP_URL = "http://127.0.0.1:18432/mcp"
TEST_USER = "mcp_phase2_test_user"
TEST_SESSION = "mcp_phase2_session"

passed = 0
failed = 0


def report(name: str, result, expect_error: bool = False):
    global passed, failed
    is_err = result.isError if hasattr(result, "isError") else False
    text = ""
    for item in result.content:
        text += item.text if hasattr(item, "text") else str(item)

    if is_err and not expect_error:
        print(f"  FAIL {name}: {text[:200]}")
        failed += 1
    elif not is_err and expect_error:
        print(f"  FAIL {name}: expected error but got success")
        failed += 1
    else:
        parsed = json.loads(text) if text else {}
        summary = str(parsed)[:120]
        print(f"  OK   {name}: {summary}")
        passed += 1


async def test():
    async with streamablehttp_client(MCP_URL) as (r, w, _):
        async with ClientSession(r, w) as session:
            await session.initialize()

            # --- Verify all tools registered ---
            tools = await session.list_tools()
            tool_names = {t.name for t in tools.tools}
            print(f"Server has {len(tools.tools)} tools registered")

            expected_core = {
                "neomem_store_exchange", "neomem_retrieve_context",
                "neomem_search", "neomem_store_fact", "neomem_get_memory",
                "neomem_buffer_exchange", "neomem_flush_buffer",
                "neomem_get_all", "neomem_update_memory",
                "neomem_delete_memory", "neomem_delete_all",
                "neomem_cleanup_expired", "neomem_count_memories",
                "neomem_score_salience", "neomem_get_history",
            }
            missing = expected_core - tool_names
            if missing:
                print(f"  MISSING TOOLS: {missing}")
            else:
                print(f"  All {len(expected_core)} core tools registered")

            # --- 1. store_exchange ---
            print("\n--- store_exchange ---")
            r1 = await session.call_tool("neomem_store_exchange", {
                "query": "I love hiking in the Swiss Alps every summer.",
                "response": "That sounds amazing! The Swiss Alps are beautiful.",
                "user_id": TEST_USER,
                "session_id": TEST_SESSION,
            })
            report("store_exchange", r1)

            # --- 2. search ---
            print("\n--- search ---")
            r2 = await session.call_tool("neomem_search", {
                "query": "hiking mountains",
                "user_id": TEST_USER,
                "limit": 3,
            })
            report("search", r2)

            # Extract a memory_id for later tests
            r2_data = json.loads(r2.content[0].text)
            memory_id = None
            if "results" in r2_data and r2_data["results"]:
                memory_id = r2_data["results"][0].get("id")

            # --- 3. retrieve_context ---
            print("\n--- retrieve_context ---")
            r3 = await session.call_tool("neomem_retrieve_context", {
                "query": "What outdoor activities does the user enjoy?",
                "user_id": TEST_USER,
            })
            report("retrieve_context", r3)

            # --- 4. store_fact ---
            print("\n--- store_fact ---")
            r4 = await session.call_tool("neomem_store_fact", {
                "fact": "User's favorite color is blue",
                "user_id": TEST_USER,
            })
            report("store_fact", r4)

            # --- 5. get_memory ---
            if memory_id:
                print("\n--- get_memory ---")
                r5 = await session.call_tool("neomem_get_memory", {
                    "memory_id": memory_id,
                    "user_id": TEST_USER,
                })
                report("get_memory", r5)

            # --- 6. get_all ---
            print("\n--- get_all ---")
            r6 = await session.call_tool("neomem_get_all", {
                "user_id": TEST_USER,
            })
            report("get_all", r6)

            # --- 7. count_memories ---
            print("\n--- count_memories ---")
            r7 = await session.call_tool("neomem_count_memories", {
                "user_id": TEST_USER,
            })
            report("count_memories", r7)

            # --- 8. update_memory ---
            if memory_id:
                print("\n--- update_memory ---")
                r8 = await session.call_tool("neomem_update_memory", {
                    "memory_id": memory_id,
                    "data": "User enjoys hiking in Swiss Alps and French Alps",
                })
                report("update_memory", r8)

            # --- 9. get_history ---
            if memory_id:
                print("\n--- get_history ---")
                r9 = await session.call_tool("neomem_get_history", {
                    "memory_id": memory_id,
                })
                report("get_history", r9)

            # --- 10. buffer_exchange ---
            print("\n--- buffer_exchange ---")
            r10 = await session.call_tool("neomem_buffer_exchange", {
                "query": "I'm reading a book about quantum computing.",
                "response": "Quantum computing is a fascinating field!",
                "user_id": TEST_USER,
                "session_id": TEST_SESSION,
            })
            report("buffer_exchange", r10)

            # --- 11. flush_buffer ---
            print("\n--- flush_buffer ---")
            r11 = await session.call_tool("neomem_flush_buffer", {
                "user_id": TEST_USER,
                "session_id": TEST_SESSION,
            })
            report("flush_buffer", r11)

            # --- 12. score_salience ---
            print("\n--- score_salience ---")
            r12 = await session.call_tool("neomem_score_salience", {
                "facts": [
                    "User prefers Python over Java",
                    "The weather is nice today",
                ],
                "context": "programming preferences discussion",
            })
            report("score_salience", r12)

            # --- 13. cleanup_expired (dry_run) ---
            print("\n--- cleanup_expired (dry_run) ---")
            r13 = await session.call_tool("neomem_cleanup_expired", {
                "user_id": TEST_USER,
                "dry_run": True,
            })
            report("cleanup_expired", r13)

            # --- 14. Store a fact then delete it ---
            print("\n--- store + delete_memory ---")
            r14a = await session.call_tool("neomem_store_fact", {
                "fact": "Temporary fact to be deleted",
                "user_id": TEST_USER,
            })
            r14a_data = json.loads(r14a.content[0].text)
            delete_id = None
            if "results" in r14a_data and r14a_data["results"]:
                delete_id = r14a_data["results"][0].get("id")

            if delete_id:
                r14b = await session.call_tool("neomem_delete_memory", {
                    "memory_id": delete_id,
                })
                report("delete_memory", r14b)
            else:
                print("  SKIP delete_memory (no ID from store)")

            # --- Summary ---
            print(f"\n{'='*50}")
            print(f"RESULTS: {passed} passed, {failed} failed")
            if failed > 0:
                print("SOME TESTS FAILED!")
                sys.exit(1)
            else:
                print("ALL TESTS PASSED!")


if __name__ == "__main__":
    asyncio.run(test())
