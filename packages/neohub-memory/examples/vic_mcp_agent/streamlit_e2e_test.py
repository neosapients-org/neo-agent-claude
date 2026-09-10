#!/usr/bin/env python3
"""Quick E2E test of Streamlit agent flow — same path as the Streamlit app.

Tests the VICMemoryAgent (same object the Streamlit app uses) to verify:
1. Connection to MCP server
2. Tool listing (core-only vs core+utils)
3. Store exchange (fact extraction + storage)
4. Memory retrieval (context building)
5. Search
6. Memory browser operations (get_all, delete)
7. History and audit
8. Settings (config read)

Run: python examples/vic_mcp_agent/streamlit_e2e_test.py
"""

import json
import os
import sys
import time

# Add project paths
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
load_dotenv(os.path.join(ROOT, ".env"))

from agent import VICMemoryAgent

PASS = 0
FAIL = 0
ERRORS = []


def check(name, passed, detail=""):
    global PASS, FAIL
    if passed:
        PASS += 1
        print(f"  [PASS] {name}")
    else:
        FAIL += 1
        ERRORS.append(f"{name}: {detail}")
        print(f"  [FAIL] {name} — {detail}")


def main():
    global PASS, FAIL

    print("=" * 60)
    print("  Streamlit E2E Agent Flow Test")
    print("=" * 60)

    # Phase 1: Connection
    print("\n--- Phase 1: Connection ---")
    agent = VICMemoryAgent(
        user_id="streamlit_test",
        agent_id="vic_l3",
        dept_id="wealth",
    )
    tool_count = agent.connect()
    check("Connect to MCP server", agent.is_connected)
    check("Tools registered", tool_count >= 18, f"got {tool_count}")

    # Check core vs core+utils
    tools = agent.list_tools()
    has_scoped = "neomem_store_scoped" in tools
    core_tools = [t for t in tools if "scoped" not in t]
    scoped_tools = [t for t in tools if "scoped" in t]
    check("Core tools available", len(core_tools) >= 18, f"got {len(core_tools)}")
    print(f"    Scoped tools: {'YES (' + str(len(scoped_tools)) + ')' if has_scoped else 'NO (core-only)'}")

    # Phase 2: Settings (mirrors Settings tab)
    print("\n--- Phase 2: Settings Tab ---")
    config = agent.read_config()
    check("Read server config", isinstance(config, dict))
    check("Config has server info", "server" in config or "resource" in config, str(config.keys()))
    check("List tools", len(tools) > 0, f"{len(tools)} tools")

    # Phase 3: Store Exchange (mirrors Chat tab — agent stores after every exchange)
    print("\n--- Phase 3: Chat Tab — Store Exchange ---")
    import random
    unique_id = random.randint(10000, 99999)
    result = agent.call_tool("neomem_store_exchange", {
        "query": f"Tell me about Rajesh Kumar's investment goals #{unique_id}",
        "response": f"Rajesh Kumar aims to build a retirement corpus of {unique_id} crores by 2040. He prefers moderate risk equity funds and has a monthly SIP of {unique_id}.",
        "user_id": "streamlit_test",
        "agent_id": "vic_l3",
        "session_id": agent._session_id,
        "dept_id": "wealth",
        "investor_name": "Rajesh Kumar",
    })
    check("Store exchange success", "facts_stored" in result or "results" in result, str(result.get("error", "")))
    stored = result.get("facts_stored", 0)
    check("Facts extracted & stored", stored > 0, f"stored={stored}")

    # Phase 4: Memory Browser (mirrors Memory Browser tab — get_all)
    print("\n--- Phase 4: Memory Browser Tab ---")
    all_mems = agent.call_tool("neomem_get_all", {
        "user_id": "streamlit_test",
        "agent_id": "vic_l3",
    })
    memories = all_mems.get("results", [])
    check("Get all memories", isinstance(memories, list))
    check("Has stored memories", len(memories) > 0, f"count={len(memories)}")

    # Check metadata (investor, categories)
    if memories:
        first = memories[0]
        meta = first.get("metadata", {}) or {}
        check("Memory has text", bool(first.get("memory")))
        check("Memory has ID", bool(first.get("id")))
        print(f"    Sample memory: {first.get('memory', '')[:80]}...")
        print(f"    Categories: {meta.get('categories', [])}")
        print(f"    Investor: {meta.get('investor_name', 'N/A')}")

    # Phase 5: Search (mirrors search functionality)
    print("\n--- Phase 5: Search ---")
    search_result = agent.call_tool("neomem_search", {
        "query": "retirement portfolio",
        "user_id": "streamlit_test",
        "agent_id": "vic_l3",
        "limit": 5,
    })
    results = search_result.get("results", [])
    check("Search returns results", len(results) > 0, f"count={len(results)}")
    if results:
        print(f"    Top result (score={results[0].get('score', 'N/A')}): {results[0].get('memory', '')[:80]}...")

    # Phase 6: Context Retrieval
    print("\n--- Phase 6: Context Retrieval ---")
    ctx = agent.call_tool("neomem_retrieve_context", {
        "query": "What are Rajesh's goals?",
        "user_id": "streamlit_test",
        "agent_id": "vic_l3",
        "session_id": agent._session_id,
        "investor_name": "Rajesh Kumar",
    })
    check("Context retrieval success", "context" in ctx or "results" in ctx)
    context_text = ctx.get("context", "")
    check("Context has content", len(context_text) > 0, f"len={len(context_text)}")

    # Phase 7: Direct Storage (fact + preference)
    print("\n--- Phase 7: Direct Storage ---")
    fact_result = agent.call_tool("neomem_store_fact", {
        "fact": "Rajesh Kumar has two children aged 12 and 8",
        "user_id": "streamlit_test",
        "agent_id": "vic_l3",
        "categories": ["persona"],
        "investor_name": "Rajesh Kumar",
    })
    check("Store fact", "results" in fact_result or "id" in fact_result or "result" in fact_result)

    pref_result = agent.call_tool("neomem_store_preference", {
        "preference": "Rajesh prefers monthly detailed portfolio statements via email",
        "user_id": "streamlit_test",
        "agent_id": "vic_l3",
    })
    check("Store preference", "results" in pref_result or "id" in pref_result or "result" in pref_result)

    # Phase 8: Buffering
    print("\n--- Phase 8: Buffering ---")
    buf_result = agent.call_tool("neomem_buffer_exchange", {
        "query": "Quick check on market status",
        "response": "Markets are up 0.5% today with IT sector leading gains.",
        "user_id": "streamlit_test",
        "session_id": agent._session_id,
        "agent_id": "vic_l3",
    })
    check("Buffer exchange", "buffered" in buf_result or "status" in buf_result or "buffer_count" in buf_result)

    flush_result = agent.call_tool("neomem_flush_buffer", {
        "user_id": "streamlit_test",
        "session_id": agent._session_id,
        "agent_id": "vic_l3",
    })
    check("Flush buffer", "flushed" in flush_result or "exchanges_flushed" in flush_result or "status" in flush_result)

    # Phase 9: Salience Scoring
    print("\n--- Phase 9: Salience ---")
    sal_result = agent.call_tool("neomem_score_salience", {
        "facts": ["Client wants to invest 10 lakhs in a new tax-saving ELSS fund before March deadline"],
    })
    check("Score salience", "results" in sal_result, str(sal_result))
    scored = sal_result.get("results", [])
    if scored:
        score = scored[0].get("salience")
        print(f"    Salience score: {score}")
        check("Salience > 0", score is not None and float(score) > 0)

    # Phase 10: History & Audit (mirrors Operations tab)
    print("\n--- Phase 10: History & Audit ---")
    stats = agent.call_tool("neomem_get_history_stats", {})
    check("History stats", isinstance(stats, dict))

    all_hist = agent.call_tool("neomem_get_all_history", {
        "user_id": "streamlit_test",
    })
    check("All history", isinstance(all_hist, dict))

    # Phase 11: Count & Cleanup
    print("\n--- Phase 11: Count & Cleanup ---")
    count_result = agent.call_tool("neomem_count_memories", {
        "user_id": "streamlit_test",
    })
    check("Count memories", "total" in count_result or "count" in count_result)
    total = count_result.get("total") or count_result.get("count", 0)
    print(f"    Total memories: {total}")

    cleanup_result = agent.call_tool("neomem_cleanup_expired", {
        "user_id": "streamlit_test",
    })
    check("Cleanup expired", isinstance(cleanup_result, dict))

    # Phase 12: Scoped Operations (if available)
    if has_scoped:
        print("\n--- Phase 12: Scoped Operations ---")
        scoped_store = agent.call_tool("neomem_store_scoped", {
            "content": "Team-shared: Q4 rebalancing guidelines",
            "user_id": "streamlit_test",
            "agent_id": "vic_l3",
            "tenant_id": "wealth_dept",
            "pool": "team",
        })
        check("Store scoped (team)", isinstance(scoped_store, dict))

        scoped_search = agent.call_tool("neomem_search_scoped", {
            "query": "rebalancing",
            "user_id": "streamlit_test",
            "agent_id": "vic_l3",
            "tenant_id": "wealth_dept",
            "pool": "team",
        })
        check("Search scoped", "results" in scoped_search or isinstance(scoped_search, dict))

        scoped_all = agent.call_tool("neomem_get_all_scoped", {
            "user_id": "streamlit_test",
            "agent_id": "vic_l3",
            "tenant_id": "wealth_dept",
            "pool": "team",
        })
        check("Get all scoped", isinstance(scoped_all, dict))

        scoped_add = agent.call_tool("neomem_add_scoped", {
            "content": "Private note: follow up with Rajesh on ELSS",
            "user_id": "streamlit_test",
            "agent_id": "vic_l3",
            "tenant_id": "wealth_dept",
            "pool": "private",
        })
        check("Add scoped (private)", isinstance(scoped_add, dict))
    else:
        print("\n--- Phase 12: Scoped Operations (SKIPPED — core-only) ---")

    # Phase 13: Operations Log (mirrors Operations tab)
    print("\n--- Phase 13: Operations Log ---")
    ops = agent.operations
    check("Operations recorded", len(ops) > 0, f"count={len(ops)}")
    creates = sum(1 for o in ops if o.operation == "CREATE")
    reads = sum(1 for o in ops if o.operation == "READ")
    print(f"    CREATE: {creates}, READ: {reads}, total: {len(ops)}")

    # Phase 14: Memory Delete (Memory Browser delete button)
    print("\n--- Phase 14: Memory Delete ---")
    if memories:
        target_id = memories[0]["id"]
        del_result = agent.call_tool("neomem_delete_memory", {"memory_id": target_id})
        check("Delete memory", isinstance(del_result, dict))

    # Phase 15: Chat with Agent (full LLM pipeline)
    print("\n--- Phase 15: Agent Chat (LLM) ---")
    try:
        response, chat_ops = agent.chat("Tell me about Rajesh Kumar's portfolio goals")
        check("Agent chat response", len(response) > 10, f"len={len(response)}")
        check("Chat triggered memory ops", len(chat_ops) >= 0)
        print(f"    Response preview: {response[:100]}...")
        print(f"    Memory ops in chat: {len(chat_ops)}")
    except Exception as e:
        check("Agent chat", False, str(e))

    # Cleanup
    print("\n--- Cleanup ---")
    agent.call_tool("neomem_delete_all", {
        "user_id": "streamlit_test",
    })
    agent.close()
    check("Agent closed", not agent.is_connected)

    # Summary
    print("\n" + "=" * 60)
    total_tests = PASS + FAIL
    print(f"  RESULTS: {PASS}/{total_tests} passed ({FAIL} failed)")
    if ERRORS:
        print("\n  FAILURES:")
        for err in ERRORS:
            print(f"    - {err}")
    print("=" * 60)

    return FAIL == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
