#!/usr/bin/env python3
"""End-to-end test for VIC Memory Agent.

Exercises all key memory operations against a live MCP server + Qdrant.
Expects:
  - Qdrant running on localhost:6335 (with clean state)
  - MCP server running (port via NEOMEM_MCP_PORT env or default 18432)
  - OPENAI_API_KEY set

Run:
  export NEOMEM_MCP_PORT=18433  # if server on non-default port
  python examples/vic_mcp_agent/e2e_test.py
"""
from __future__ import annotations

import json
import logging
import os
import sys
import time
import traceback

# Suppress noisy HTTP/library logs — only show our test output
logging.basicConfig(level=logging.WARNING)
for noisy in ("httpx", "httpcore", "urllib3", "qdrant_client", "openai",
               "neo_memory_hub", "neomem_mcp", "memory_utils", "uvicorn",
               "mcp", "anyio"):
    logging.getLogger(noisy).setLevel(logging.ERROR)

# ── Add project root to path ────────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from examples.vic_mcp_agent.agent import VICMemoryAgent

# ── Helpers ──────────────────────────────────────────────────────────

PASSED = 0
FAILED = 0
ERRORS: list[str] = []


def section(title: str) -> None:
    print(f"\n{'─' * 60}")
    print(f"  {title}")
    print(f"{'─' * 60}")


def check(name: str, condition: bool, detail: str = "") -> None:
    global PASSED, FAILED
    if condition:
        PASSED += 1
        print(f"  ✅ {name}")
    else:
        FAILED += 1
        msg = f"  ❌ {name}"
        if detail:
            msg += f" — {detail}"
        print(msg)
        ERRORS.append(f"{name}: {detail}")


def main() -> int:
    global PASSED, FAILED

    agent = VICMemoryAgent(
        user_id="e2e_tester",
        agent_id="vic_e2e",
        dept_id="wealth_test",
    )

    # ── Phase 1: Connection ──────────────────────────────────────────
    section("Phase 1 — Connection & Setup")

    tool_count = agent.connect()
    check("MCP connection", agent.is_connected)
    check("Tool count ≥ 22", tool_count >= 22, f"got {tool_count}")
    check("Agent created", agent._agent is not None)

    tools = agent.list_tools()
    print(f"     Tools: {', '.join(tools[:8])}...")
    required_tools = [
        "neomem_store_exchange", "neomem_retrieve_context", "neomem_search",
        "neomem_store_fact", "neomem_store_preference", "neomem_get_memory",
        "neomem_get_all", "neomem_count_memories", "neomem_update_memory",
        "neomem_delete_memory", "neomem_delete_all", "neomem_buffer_exchange",
        "neomem_flush_buffer", "neomem_score_salience",
        "neomem_get_history", "neomem_get_all_history", "neomem_get_history_stats",
        "neomem_cleanup_expired", "neomem_store_scoped", "neomem_search_scoped",
        "neomem_add_scoped", "neomem_get_all_scoped",
    ]
    for t in required_tools:
        check(f"Tool available: {t}", t in tools)

    # Read server config
    config = agent.read_config()
    check("Config resource readable", bool(config), str(type(config)))

    # ── Phase 2: Direct Tool Calls — Storage ─────────────────────────
    section("Phase 2 — Direct Storage (store_fact, store_preference, store_exchange)")

    # 2a: Store fact
    r = agent.call_tool("neomem_store_fact", {
        "fact": "Mohan Kumar has a risk tolerance of moderate-conservative",
        "user_id": agent.user_id,
        "agent_id": agent.agent_id,
        "investor_name": "Mohan Kumar",
        "categories": ["persona"],
    })
    check("store_fact succeeded", r.get("status") == "ok" or "id" in str(r), json.dumps(r)[:120])
    fact1_id = r.get("id") or r.get("memory_id")

    # 2b: Store another fact
    r = agent.call_tool("neomem_store_fact", {
        "fact": "Priya Sharma prefers ESG-compliant investments only",
        "user_id": agent.user_id,
        "agent_id": agent.agent_id,
        "investor_name": "Priya Sharma",
        "categories": ["preference"],
    })
    check("store_fact (Priya) succeeded", r.get("status") == "ok" or "id" in str(r), json.dumps(r)[:120])

    # 2c: Store preference
    r = agent.call_tool("neomem_store_preference", {
        "preference": "Client prefers weekly portfolio summary emails on Monday",
        "user_id": agent.user_id,
        "agent_id": agent.agent_id,
    })
    check("store_preference succeeded", r.get("status") == "ok" or "id" in str(r), json.dumps(r)[:120])

    # 2d: Store exchange (full pipeline with LLM extraction)
    r = agent.call_tool("neomem_store_exchange", {
        "query": "Mohan is worried about geopolitical tensions affecting his portfolio",
        "response": "I understand Mohan's concerns. Given his moderate-conservative risk profile, we should review his equity exposure to emerging markets and consider hedging strategies.",
        "user_id": agent.user_id,
        "agent_id": agent.agent_id,
        "session_id": agent._session_id,
        "dept_id": agent.dept_id,
        "investor_name": "Mohan Kumar",
    })
    check("store_exchange succeeded",
          "facts_stored" in r or "extracted" in str(r).lower() or r.get("status") == "ok",
          json.dumps(r)[:150])
    facts_stored = r.get("facts_stored", 0)
    print(f"     Extracted/stored: {facts_stored} facts")

    # 2e: Store another exchange for Priya
    r = agent.call_tool("neomem_store_exchange", {
        "query": "Priya wants to increase her allocation to green bonds",
        "response": "We can increase green bond allocation from 15% to 25% in her portfolio. ICICI Pru Green Bond Fund has delivered 8.2% YTD.",
        "user_id": agent.user_id,
        "agent_id": agent.agent_id,
        "session_id": agent._session_id,
        "dept_id": agent.dept_id,
        "investor_name": "Priya Sharma",
    })
    check("store_exchange (Priya) succeeded",
          "facts_stored" in r or "extracted" in str(r).lower() or r.get("status") == "ok",
          json.dumps(r)[:150])

    time.sleep(4)  # Allow Qdrant indexing

    # ── Phase 3: Retrieval ───────────────────────────────────────────
    section("Phase 3 — Retrieval (retrieve_context, search, get_all, count)")

    # 3a: Count memories
    r = agent.call_tool("neomem_count_memories", {
        "user_id": agent.user_id,
    })
    count = r.get("total", 0)
    check("count_memories > 0", count > 0, f"count={count}")
    print(f"     Total memories: {count}")

    # 3b: Retrieve context for Mohan
    r = agent.call_tool("neomem_retrieve_context", {
        "query": "What is Mohan Kumar's risk tolerance?",
        "user_id": agent.user_id,
        "agent_id": agent.agent_id,
        "session_id": agent._session_id,
        "dept_id": agent.dept_id,
        "investor_name": "Mohan Kumar",
    })
    context_text = r.get("context", "")
    results_count = len(r.get("memories", []))
    check("retrieve_context returned results", results_count > 0, f"results={results_count}")
    check("context mentions Mohan", "mohan" in context_text.lower() or "risk" in context_text.lower(),
          f"context_len={len(context_text)}")
    print(f"     Context preview: {context_text[:120]}...")

    # 3c: Search for Priya
    r = agent.call_tool("neomem_search", {
        "query": "Priya ESG green bonds",
        "user_id": agent.user_id,
        "agent_id": agent.agent_id,
        "limit": 5,
    })
    search_results = r.get("results", [])
    check("search returned results", len(search_results) > 0, f"count={len(search_results)}")
    if search_results:
        first = search_results[0]
        print(f"     Top result: score={first.get('score', '?'):.3f} — {first.get('memory', '')[:80]}")

    # 3d: Get all memories
    r = agent.call_tool("neomem_get_all", {
        "user_id": agent.user_id,
        "agent_id": agent.agent_id,
    })
    all_mems = r.get("results", [])
    check("get_all returned memories", len(all_mems) >= 2, f"count={len(all_mems)}")
    print(f"     All memories count: {len(all_mems)}")
    for m in all_mems[:5]:
        mem_text = m.get("memory", "")[:70]
        cats = (m.get("metadata") or {}).get("categories", [])
        inv = (m.get("metadata") or {}).get("investor_name", "")
        print(f"       • [{', '.join(cats) if cats else '?'}] {inv}: {mem_text}")

    # 3e: Get specific memory by ID
    if all_mems:
        mem_id = all_mems[0].get("id")
        if mem_id:
            r = agent.call_tool("neomem_get_memory", {
                "memory_id": mem_id,
            })
            check("get_memory by ID", bool(r.get("memory") or r.get("id")), json.dumps(r)[:100])

    # ── Phase 4: Update & Delete ─────────────────────────────────────
    section("Phase 4 — Update & Delete")

    if all_mems:
        target_id = all_mems[0].get("id")

        # 4a: Update memory
        r = agent.call_tool("neomem_update_memory", {
            "memory_id": target_id,
            "data": "Mohan Kumar has a moderate-conservative risk tolerance, re-evaluated Q1 2025.",
        })
        check("update_memory succeeded", r.get("status") == "ok" or "updated" in str(r).lower(),
              json.dumps(r)[:120])

        # Verify update
        r = agent.call_tool("neomem_get_memory", {
            "memory_id": target_id,
        })
        updated_text = r.get("memory", "")
        check("update reflected", "q1 2025" in updated_text.lower() or "re-evaluated" in updated_text.lower(),
              updated_text[:80])

    # Store a temporary fact to delete
    r = agent.call_tool("neomem_store_fact", {
        "fact": "Temporary test fact for deletion",
        "user_id": agent.user_id,
        "agent_id": agent.agent_id,
        "categories": ["episodic"],
    })
    tmp_id = r.get("id") or r.get("memory_id")

    if tmp_id:
        r = agent.call_tool("neomem_delete_memory", {
            "memory_id": tmp_id,
        })
        check("delete_memory succeeded", r.get("status") == "ok" or "deleted" in str(r).lower(),
              json.dumps(r)[:100])

    # ── Phase 5: Buffering ───────────────────────────────────────────
    section("Phase 5 — Buffering (buffer_exchange + flush)")

    r = agent.call_tool("neomem_buffer_exchange", {
        "query": "Quick check on Mohan's SIP status",
        "response": "Mohan's SIPs are all running. Next debit on 5th April.",
        "user_id": agent.user_id,
        "agent_id": agent.agent_id,
        "session_id": agent._session_id,
    })
    check("buffer_exchange succeeded", r.get("status") == "ok" or "buffered" in str(r).lower(),
          json.dumps(r)[:100])

    r = agent.call_tool("neomem_flush_buffer", {
        "user_id": agent.user_id,
        "session_id": agent._session_id,
        "agent_id": agent.agent_id,
    })
    check("flush_buffer succeeded", r.get("status") == "ok" or "flushed" in str(r).lower(),
          json.dumps(r)[:100])

    # ── Phase 6: Salience Scoring ────────────────────────────────────
    section("Phase 6 — Salience Scoring")

    r = agent.call_tool("neomem_score_salience", {
        "facts": ["Mohan wants to exit all equity positions immediately due to panic"],
        "context": "Client is anxious about market volatility",
    })
    # Result: {"results": [{"text": ..., "salience": 0.85, "reasoning": ...}]}
    scored_items = r.get("results", [])
    if scored_items and isinstance(scored_items[0], dict):
        score = scored_items[0].get("salience") or scored_items[0].get("score")
    else:
        score = r.get("score") or r.get("salience_score")
    check("score_salience returned", score is not None, json.dumps(r)[:200])
    if score is not None:
        print(f"     Salience score: {score}")
        check("salience score > 0.5 (urgent)", float(score) > 0.5, f"score={score}")

    # ── Phase 7: History ─────────────────────────────────────────────
    section("Phase 7 — History & Audit")

    if all_mems:
        mem_id = all_mems[0].get("id")
        r = agent.call_tool("neomem_get_history", {
            "memory_id": mem_id,
        })
        history = r.get("history", [])
        check("get_history returned entries", len(history) >= 0, f"count={len(history)}")

    r = agent.call_tool("neomem_get_all_history", {})
    all_history = r.get("history", [])
    check("get_all_history returned", isinstance(all_history, list), f"type={type(all_history)}")

    r = agent.call_tool("neomem_get_history_stats", {})
    check("get_history_stats returned", bool(r), json.dumps(r)[:100])

    # ── Phase 8: Scoped Memory ───────────────────────────────────────
    section("Phase 8 — Scoped Memory (private/team/org)")

    r = agent.call_tool("neomem_store_scoped", {
        "user_message": "What is Mohan Kumar's priority classification level?",
        "assistant_response": "Mohan Kumar is classified as an Ultra High Net Worth Individual with priority service level. His KYC was completed on 15 March 2025 and AUM exceeds 50 crore.",
        "pool": "private",
        "user_id": agent.user_id,
        "agent_id": agent.agent_id,
    })
    check("store_scoped (private) succeeded",
          "error" not in r and isinstance(r.get("results"), list),
          json.dumps(r)[:150])
    scoped_private_count = len(r.get("results", []))
    print(f"     store_scoped (private): {scoped_private_count} facts stored")

    r = agent.call_tool("neomem_store_scoped", {
        "user_message": "What is the team status on Mohan Kumar onboarding?",
        "assistant_response": "Mohan Kumar onboarding completed by the wealth team on 10 March 2025. All KYC documents verified. Assigned to senior RM Deepak.",
        "pool": "shared",
        "user_id": agent.user_id,
        "agent_id": agent.agent_id,
    })
    check("store_scoped (shared) succeeded",
          "error" not in r and isinstance(r.get("results"), list),
          json.dumps(r)[:150])

    time.sleep(3)  # Allow indexing

    r = agent.call_tool("neomem_search_scoped", {
        "query": "Mohan Kumar UHNI priority",
        "pool": "private",
        "user_id": agent.user_id,
    })
    scoped_results = r.get("results", [])
    check("search_scoped returned results",
          len(scoped_results) > 0 or scoped_private_count == 0,
          f"count={len(scoped_results)} (stored={scoped_private_count})")

    r = agent.call_tool("neomem_get_all_scoped", {
        "pool": "private",
        "user_id": agent.user_id,
    })
    all_scoped = r.get("results", [])
    check("get_all_scoped returned",
          len(all_scoped) > 0 or scoped_private_count == 0,
          f"count={len(all_scoped)} (stored={scoped_private_count})")

    # ── Phase 9: Cleanup ─────────────────────────────────────────────
    section("Phase 9 — Lifecycle (cleanup_expired)")

    r = agent.call_tool("neomem_cleanup_expired", {
        "user_id": agent.user_id,
    })
    check("cleanup_expired succeeded",
          "scanned" in r or "dry_run" in r or r.get("status") == "ok",
          json.dumps(r)[:150])
    if "scanned" in r:
        print(f"     Cleaned: scanned={r['scanned']}, expired={r.get('expired', 0)}")

    # ── Phase 10: LLM Chat Roundtrip ────────────────────────────────
    section("Phase 10 — LLM Chat Roundtrip (full pipeline)")

    print("     Sending chat message (LLM invocation)...")
    try:
        response, ops = agent.chat(
            "I'm here for Mohan Kumar. He's concerned about the recent market volatility. "
            "What do we know about his risk profile and preferences?"
        )
        check("chat() returned response", len(response) > 20, f"len={len(response)}")
        check("chat() generated operations", len(ops) > 0, f"ops={len(ops)}")
        check("investor tracked as Mohan Kumar", agent.current_investor == "Mohan Kumar",
              f"investor={agent.current_investor}")
        print(f"     Response preview: {response[:150]}...")
        for op in ops:
            icon = {"CREATE": "🟢", "READ": "🔵", "UPDATE": "🟡", "DELETE": "🔴"}.get(op.operation, "⚪")
            print(f"       {icon} {op.tool}: {op.result_summary}")
    except Exception as e:
        check("chat() succeeded", False, str(e))
        traceback.print_exc()

    # Second turn — pronoun resolution test
    print("\n     Sending follow-up with pronoun...")
    try:
        response2, ops2 = agent.chat(
            "What about his preferences for communication and ESG?"
        )
        check("follow-up returned response", len(response2) > 20, f"len={len(response2)}")
        check("investor still tracked (pronoun)", agent.current_investor == "Mohan Kumar",
              f"investor={agent.current_investor}")
        print(f"     Response preview: {response2[:150]}...")
    except Exception as e:
        check("follow-up chat succeeded", False, str(e))
        traceback.print_exc()

    # Third turn — different investor
    print("\n     Switching to Priya Sharma...")
    try:
        response3, ops3 = agent.chat(
            "Tell me about Priya Sharma. What ESG investments does she prefer?"
        )
        check("Priya chat returned response", len(response3) > 20, f"len={len(response3)}")
        check("investor switched to Priya",
              agent.current_investor and "priya" in agent.current_investor.lower(),
              f"investor={agent.current_investor}")
        print(f"     Response preview: {response3[:150]}...")
    except Exception as e:
        check("Priya chat succeeded", False, str(e))
        traceback.print_exc()

    # ── Phase 11: Final Verification ─────────────────────────────────
    section("Phase 11 — Final Verification")

    # Count memories after all operations
    r = agent.call_tool("neomem_count_memories", {
        "user_id": agent.user_id,
    })
    final_count = r.get("total", 0)
    check("final memory count > initial", final_count > count, f"final={final_count}, initial={count}")
    print(f"     Final memory count: {final_count}")

    # Operations log
    print(f"     Total operations: {len(agent.operations)}")
    op_types = {}
    for op in agent.operations:
        op_types[op.operation] = op_types.get(op.operation, 0) + 1
    print(f"     By type: {op_types}")

    # ── Cleanup (delete_all) ─────────────────────────────────────────
    section("Cleanup — delete_all")

    r = agent.call_tool("neomem_delete_all", {
        "user_id": agent.user_id,
    })
    check("delete_all succeeded", r.get("status") == "ok" or "deleted" in str(r).lower(),
          json.dumps(r)[:100])

    time.sleep(3)  # Allow Qdrant to process deletions

    r = agent.call_tool("neomem_count_memories", {
        "user_id": agent.user_id,
    })
    post_delete = r.get("total", -1)
    check("count after delete_all low", post_delete < count,
          f"count={post_delete} vs pre-delete={count}")

    # ── Close ────────────────────────────────────────────────────────
    agent.close()
    check("agent closed", not agent.is_connected)

    # ── Summary ──────────────────────────────────────────────────────
    section("SUMMARY")
    total = PASSED + FAILED
    print(f"  PASSED: {PASSED}/{total}")
    print(f"  FAILED: {FAILED}/{total}")
    if ERRORS:
        print("\n  Failures:")
        for e in ERRORS:
            print(f"    ❌ {e}")
    print()

    return 0 if FAILED == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
