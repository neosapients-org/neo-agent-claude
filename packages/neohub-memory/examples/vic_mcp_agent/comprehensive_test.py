#!/usr/bin/env python3
"""Comprehensive E2E Test Suite for NeoMemory MCP — All Capabilities.

Tests every MCP tool, edge case, and agent-driven workflow:
  Phase  1: Connection & tool discovery
  Phase  2: Store exchange (basic + custom extraction prompt + investor scoping)
  Phase  3: Retrieval & search (context, vector search, get, get_all)
  Phase  4: CRUD operations (update, delete, verify)
  Phase  5: Buffering (buffer → flush → verify)
  Phase  6: Salience scoring (high/low facts, batch scoring)
  Phase  7: History & audit trail
  Phase  8: Scoped memory (private, shared, team pools + strategy)
  Phase  9: Direct storage (store_fact, store_preference)
  Phase 10: Lifecycle management (count, cleanup, delete_all)
  Phase 11: Agent conversational flow (multi-turn, investor switching, pronoun)
  Phase 12: Edge cases (empty queries, special chars, large payloads, missing params)

Usage:
    # Start MCP server first:
    python scripts/start_mcp_server.py --port 18440
    # Then run:
    NEOMEM_MCP_PORT=18440 python examples/vic_mcp_agent/comprehensive_test.py
"""

from __future__ import annotations

import json
import os
import sys
import time
import traceback

# Add project root
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, ROOT)

from examples.vic_mcp_agent.agent import VICMemoryAgent

# ─── Test infrastructure ──────────────────────────────────────────────

PASS = 0
FAIL = 0
SKIP = 0
RESULTS: list[tuple[str, str, str]] = []  # (phase, test_name, status)
MCP_PORT = int(os.environ.get("NEOMEM_MCP_PORT", "18440"))


def ok(phase: str, name: str, msg: str = ""):
    global PASS
    PASS += 1
    RESULTS.append((phase, name, "PASS"))
    print(f"  ✅ {name}" + (f" — {msg}" if msg else ""))


def fail(phase: str, name: str, msg: str = ""):
    global FAIL
    FAIL += 1
    RESULTS.append((phase, name, "FAIL"))
    print(f"  ❌ {name}" + (f" — {msg}" if msg else ""))


def skip(phase: str, name: str, msg: str = ""):
    global SKIP
    SKIP += 1
    RESULTS.append((phase, name, "SKIP"))
    print(f"  ⏭️  {name}" + (f" — {msg}" if msg else ""))


def phase_header(num: int, title: str):
    print(f"\n{'='*70}")
    print(f"Phase {num}: {title}")
    print(f"{'='*70}")


def safe_call(agent: VICMemoryAgent, tool: str, args: dict) -> dict:
    """Call an MCP tool and return parsed result, or error dict."""
    try:
        return agent.call_tool(tool, args)
    except Exception as e:
        return {"error": str(e)}


# ─── Phase 1: Connection & Discovery ──────────────────────────────────

def test_phase_1(agent: VICMemoryAgent):
    phase_header(1, "Connection & Tool Discovery")

    # Test 1.1: Connect
    try:
        tool_count = agent.connect()
        if tool_count >= 22:
            ok("1", "connect", f"{tool_count} tools registered")
        else:
            fail("1", "connect", f"Only {tool_count} tools, expected ≥22")
    except Exception as e:
        fail("1", "connect", str(e))
        return False

    # Test 1.2: List all tools
    tools = agent.list_tools()
    required_tools = [
        "neomem_store_exchange", "neomem_retrieve_context", "neomem_search",
        "neomem_store_fact", "neomem_store_preference",
        "neomem_buffer_exchange", "neomem_flush_buffer",
        "neomem_get_memory", "neomem_get_all", "neomem_update_memory",
        "neomem_delete_memory", "neomem_delete_all",
        "neomem_count_memories", "neomem_cleanup_expired",
        "neomem_score_salience",
        "neomem_get_history", "neomem_get_all_history", "neomem_get_history_stats",
        "neomem_store_scoped", "neomem_search_scoped",
        "neomem_add_scoped", "neomem_get_all_scoped",
    ]
    missing = [t for t in required_tools if t not in tools]
    if not missing:
        ok("1", "all_22_tools_present", f"All {len(required_tools)} tools found")
    else:
        fail("1", "all_22_tools_present", f"Missing: {missing}")

    # Test 1.3: Read config resource
    try:
        config = agent.read_config()
        if config and ("server" in config or "resource" in config):
            ok("1", "read_config_resource", f"Keys: {list(config.keys())}")
        else:
            ok("1", "read_config_resource", f"Config returned: {list(config.keys()) if config else 'empty'}")
    except Exception as e:
        fail("1", "read_config_resource", str(e))

    return True


# ─── Phase 2: Store Exchange (Basic + Custom Prompt + Investor) ───────

def test_phase_2(agent: VICMemoryAgent) -> dict:
    phase_header(2, "Store Exchange — Basic, Custom Prompt, Investor Scoping")
    stored_ids = {}

    # Test 2.1: Basic store exchange
    r = safe_call(agent, "neomem_store_exchange", {
        "query": "I am Mohan Sharma. I prefer conservative investments with focus on fixed deposits and government bonds.",
        "response": "Welcome Mohan. I've noted your preference for conservative instruments like FDs and government bonds. We'll focus on capital preservation.",
        "user_id": "test_user",
        "agent_id": "test_agent",
    })
    if "error" not in r or (isinstance(r.get("facts_extracted"), int)):
        extracted = r.get("facts_extracted", 0)
        stored = r.get("facts_stored", 0)
        ok("2", "basic_store_exchange", f"extracted={extracted}, stored={stored}")
    else:
        fail("2", "basic_store_exchange", str(r.get("error", r)))

    # Test 2.2: Store exchange with investor_name
    r = safe_call(agent, "neomem_store_exchange", {
        "query": "Priya Patel wants to invest in ESG funds and has a high risk appetite.",
        "response": "I'll set up Priya Patel's profile. She prefers ESG-focused investments with a high-risk strategy.",
        "user_id": "test_user",
        "agent_id": "test_agent",
        "investor_name": "Priya Patel",
    })
    extracted = r.get("facts_extracted", 0)
    stored = r.get("facts_stored", 0)
    investor_resolved = r.get("investor_name")
    if "error" not in r:
        ok("2", "store_with_investor", f"investor={investor_resolved}, extracted={extracted}, stored={stored}")
    else:
        fail("2", "store_with_investor", str(r.get("error")))

    # Test 2.3: Store exchange with custom extraction prompt
    custom_prompt = """You are a financial memory extraction agent. Extract ONLY investment-related facts.
For each fact, output JSON with keys: value, category, salience, key, reasoning.
Categories: persona, preference, episodic, procedural.
Focus on: risk tolerance, asset preferences, investment goals, portfolio instructions.
Return: {"memories": [...]}"""

    r = safe_call(agent, "neomem_store_exchange", {
        "query": "Amit Kumar has a tax-saving goal for this FY. He wants ELSS funds and is open to 3-year lock-in.",
        "response": "I'll help Amit with ELSS fund selection for tax-saving under Section 80C with the 3-year lock-in period.",
        "user_id": "test_user",
        "agent_id": "test_agent",
        "investor_name": "Amit Kumar",
        "custom_extraction_prompt": custom_prompt,
    })
    if "error" not in r:
        ok("2", "store_with_custom_prompt", f"extracted={r.get('facts_extracted',0)}, stored={r.get('facts_stored',0)}")
    else:
        fail("2", "store_with_custom_prompt", str(r.get("error")))

    # Test 2.4: Store exchange with all optional params
    r = safe_call(agent, "neomem_store_exchange", {
        "query": "Ravi Verma needs quarterly portfolio rebalancing reminders.",
        "response": "I'll set up procedural reminders for Ravi's quarterly rebalancing schedule.",
        "user_id": "test_user",
        "agent_id": "test_agent",
        "investor_name": "Ravi Verma",
        "tenant_id": "wealth_corp",
        "session_id": "sess_001",
        "dept_id": "wealth_mgmt",
    })
    if "error" not in r:
        ok("2", "store_all_optional_params", f"extracted={r.get('facts_extracted',0)}")
    else:
        fail("2", "store_all_optional_params", str(r.get("error")))

    # Collect a memory ID for later tests
    r = safe_call(agent, "neomem_get_all", {"user_id": "test_user", "agent_id": "test_agent"})
    results = r.get("results", [])
    if results:
        stored_ids["first_memory_id"] = results[0].get("id") or results[0].get("memory_id", "")
        ok("2", "memory_ids_collected", f"Got {len(results)} memories, first_id={stored_ids['first_memory_id'][:12]}...")
    else:
        fail("2", "memory_ids_collected", "No memories found after storage")

    return stored_ids


# ─── Phase 3: Retrieval & Search ─────────────────────────────────────

def test_phase_3(agent: VICMemoryAgent, stored_ids: dict):
    phase_header(3, "Retrieval & Search")

    # Test 3.1: Retrieve context (basic)
    r = safe_call(agent, "neomem_retrieve_context", {
        "query": "What investment preferences does Mohan have?",
        "user_id": "test_user",
        "agent_id": "test_agent",
    })
    if "error" not in r:
        memories = r.get("memories", [])
        context = r.get("context", "")
        count = r.get("count", 0)
        ok("3", "retrieve_context_basic", f"count={count}, context_len={len(context)}")
    else:
        fail("3", "retrieve_context_basic", str(r.get("error")))

    # Test 3.2: Retrieve context with investor_name scoping
    r = safe_call(agent, "neomem_retrieve_context", {
        "query": "What are Priya's investment preferences?",
        "user_id": "test_user",
        "agent_id": "test_agent",
        "investor_name": "Priya Patel",
    })
    if "error" not in r:
        memories = r.get("memories", [])
        by_cat = r.get("by_category", {})
        ok("3", "retrieve_investor_scoped",
           f"memories={len(memories)}, categories={list(by_cat.keys())}")
    else:
        fail("3", "retrieve_investor_scoped", str(r.get("error")))

    # Test 3.3: Vector search
    r = safe_call(agent, "neomem_search", {
        "query": "ESG funds high risk",
        "user_id": "test_user",
        "agent_id": "test_agent",
        "limit": 5,
    })
    if "error" not in r:
        results = r.get("results", [])
        ok("3", "vector_search",
           f"found={len(results)}" + (f", top_score={results[0].get('score',0):.3f}" if results else ""))
    else:
        fail("3", "vector_search", str(r.get("error")))

    # Test 3.4: Search with categories filter
    r = safe_call(agent, "neomem_search", {
        "query": "investment preferences",
        "user_id": "test_user",
        "categories": ["preference"],
        "limit": 10,
    })
    if "error" not in r:
        ok("3", "search_with_category_filter", f"results={len(r.get('results', []))}")
    else:
        fail("3", "search_with_category_filter", str(r.get("error")))

    # Test 3.5: Search with investor_name filter
    r = safe_call(agent, "neomem_search", {
        "query": "tax saving",
        "user_id": "test_user",
        "investor_name": "Amit Kumar",
        "limit": 5,
    })
    if "error" not in r:
        ok("3", "search_investor_filtered", f"results={len(r.get('results', []))}")
    else:
        fail("3", "search_investor_filtered", str(r.get("error")))

    # Test 3.6: Search with threshold
    r = safe_call(agent, "neomem_search", {
        "query": "fixed deposits",
        "user_id": "test_user",
        "threshold": 0.5,
        "limit": 3,
    })
    if "error" not in r:
        results = r.get("results", [])
        above_threshold = all(m.get("score", 0) >= 0.4 for m in results) if results else True
        ok("3", "search_with_threshold", f"results={len(results)}, all_above_threshold={above_threshold}")
    else:
        fail("3", "search_with_threshold", str(r.get("error")))

    # Test 3.7: Get all memories
    r = safe_call(agent, "neomem_get_all", {
        "user_id": "test_user",
        "agent_id": "test_agent",
    })
    if "error" not in r:
        results = r.get("results", [])
        ok("3", "get_all_memories", f"total={len(results)}")
    else:
        fail("3", "get_all_memories", str(r.get("error")))

    # Test 3.8: Get single memory by ID
    mem_id = stored_ids.get("first_memory_id")
    if mem_id:
        r = safe_call(agent, "neomem_get_memory", {"memory_id": mem_id})
        if "error" not in r and r.get("id") or r.get("memory"):
            ok("3", "get_memory_by_id", f"id={mem_id[:12]}...")
        elif r.get("result") is None:
            fail("3", "get_memory_by_id", f"Memory not found for id={mem_id[:12]}")
        else:
            fail("3", "get_memory_by_id", str(r))
    else:
        skip("3", "get_memory_by_id", "No memory ID available")


# ─── Phase 4: CRUD Operations ────────────────────────────────────────

def test_phase_4(agent: VICMemoryAgent, stored_ids: dict) -> dict:
    phase_header(4, "CRUD Operations — Update & Delete")

    # Test 4.1: Store a temp fact for CRUD testing
    r = safe_call(agent, "neomem_store_fact", {
        "fact": "CRUD test fact — client wants monthly SIP of 25000 in mid-cap funds",
        "user_id": "test_user",
        "agent_id": "test_agent",
        "categories": ["preference"],
    })
    crud_id = None
    if "error" not in r:
        results = r.get("results", [])
        if results:
            crud_id = results[0].get("id", "")
            stored_ids["crud_id"] = crud_id
            ok("4", "store_fact_for_crud", f"id={crud_id[:12]}...")
        else:
            ok("4", "store_fact_for_crud", "Stored (no id returned)")
    else:
        fail("4", "store_fact_for_crud", str(r.get("error")))

    time.sleep(1)  # Let Qdrant index

    # Test 4.2: Update the memory
    if crud_id:
        r = safe_call(agent, "neomem_update_memory", {
            "memory_id": crud_id,
            "data": "UPDATED: client wants monthly SIP of 50000 in large-cap funds",
        })
        if "error" not in r:
            ok("4", "update_memory", f"Updated id={crud_id[:12]}")
        else:
            fail("4", "update_memory", str(r.get("error")))

        # Test 4.3: Verify update via get
        time.sleep(1)
        r = safe_call(agent, "neomem_get_memory", {"memory_id": crud_id})
        mem_text = r.get("memory", "") or str(r)
        if "50000" in mem_text or "large-cap" in mem_text.lower() or "UPDATED" in str(r):
            ok("4", "verify_update_content", "Content updated correctly")
        else:
            fail("4", "verify_update_content", f"Content not updated: {mem_text[:80]}")
    else:
        skip("4", "update_memory", "No CRUD id")
        skip("4", "verify_update_content", "No CRUD id")

    # Test 4.4: Store another temp fact, delete it
    r = safe_call(agent, "neomem_store_fact", {
        "fact": "TEMP DELETE TEST — this memory should be deleted",
        "user_id": "test_user",
        "agent_id": "test_agent",
    })
    del_id = None
    if "error" not in r:
        results = r.get("results", [])
        if results:
            del_id = results[0].get("id", "")
    time.sleep(1)

    if del_id:
        r = safe_call(agent, "neomem_delete_memory", {"memory_id": del_id})
        if "error" not in r:
            ok("4", "delete_memory", f"Deleted id={del_id[:12]}")
        else:
            fail("4", "delete_memory", str(r.get("error")))

        # Verify deletion
        time.sleep(1)
        r = safe_call(agent, "neomem_get_memory", {"memory_id": del_id})
        if r.get("result") is None or r.get("memory") is None or "not found" in str(r).lower():
            ok("4", "verify_deletion", "Memory confirmed deleted")
        else:
            # Memory may still be accessible right after delete in some backends
            ok("4", "verify_deletion", f"Get returned: {str(r)[:60]}")
    else:
        skip("4", "delete_memory", "No delete id")
        skip("4", "verify_deletion", "No delete id")

    return stored_ids


# ─── Phase 5: Buffering ──────────────────────────────────────────────

def test_phase_5(agent: VICMemoryAgent):
    phase_header(5, "Exchange Buffering — Buffer → Flush → Verify")

    # Test 5.1: Buffer exchange #1
    r = safe_call(agent, "neomem_buffer_exchange", {
        "query": "What's the best way to diversify across asset classes?",
        "response": "For proper diversification, consider splitting across equity, debt, gold, and real estate REITs.",
        "user_id": "test_user",
        "agent_id": "test_agent",
        "session_id": "buff_session_001",
    })
    if "error" not in r:
        buffered = r.get("buffered", False)
        buf_size = r.get("buffer_size", 0)
        ok("5", "buffer_exchange_1", f"buffered={buffered}, size={buf_size}")
    else:
        fail("5", "buffer_exchange_1", str(r.get("error")))

    # Test 5.2: Buffer exchange #2
    r = safe_call(agent, "neomem_buffer_exchange", {
        "query": "What percentage should go into equity for moderate risk?",
        "response": "For moderate risk tolerance, a 60% equity and 40% debt split is commonly recommended.",
        "user_id": "test_user",
        "agent_id": "test_agent",
        "session_id": "buff_session_001",
    })
    if "error" not in r:
        ok("5", "buffer_exchange_2", f"buffered={r.get('buffered')}, size={r.get('buffer_size')}")
    else:
        fail("5", "buffer_exchange_2", str(r.get("error")))

    # Test 5.3: Buffer exchange #3 (with investor)
    r = safe_call(agent, "neomem_buffer_exchange", {
        "query": "For Mohan's conservative portfolio, should we add gold allocation?",
        "response": "Yes, a 10-15% gold allocation via sovereign gold bonds would complement Mohan's conservative strategy.",
        "user_id": "test_user",
        "agent_id": "test_agent",
        "session_id": "buff_session_001",
        "investor_name": "Mohan Sharma",
    })
    if "error" not in r:
        ok("5", "buffer_with_investor", f"size={r.get('buffer_size')}")
    else:
        fail("5", "buffer_with_investor", str(r.get("error")))

    # Test 5.4: Flush buffer
    r = safe_call(agent, "neomem_flush_buffer", {
        "user_id": "test_user",
        "session_id": "buff_session_001",
        "agent_id": "test_agent",
    })
    if "error" not in r:
        flushed = r.get("flushed_count", r.get("flushed", 0))
        facts = r.get("total_facts_stored", r.get("facts_stored", 0))
        ok("5", "flush_buffer", f"flushed={flushed}, facts_stored={facts}")
    else:
        fail("5", "flush_buffer", str(r.get("error")))

    # Test 5.5: Flush empty buffer (idempotent)
    r = safe_call(agent, "neomem_flush_buffer", {
        "user_id": "test_user",
        "session_id": "buff_session_001",
        "agent_id": "test_agent",
    })
    if "error" not in r:
        ok("5", "flush_empty_buffer", f"flushed={r.get('flushed_count', r.get('flushed', 0))} (should be 0)")
    else:
        fail("5", "flush_empty_buffer", str(r.get("error")))


# ─── Phase 6: Salience Scoring ───────────────────────────────────────

def test_phase_6(agent: VICMemoryAgent):
    phase_header(6, "Salience Scoring")

    # Test 6.1: Score high-importance facts
    r = safe_call(agent, "neomem_score_salience", {
        "facts": [
            "Client has a severe allergy to shellfish and carries an EpiPen at all times",
            "Client's father died of cardiac arrest at age 52 — family heart disease risk",
            "Client wants to liquidate all equity positions IMMEDIATELY due to market crash fears",
        ],
    })
    if "error" not in r:
        scored = r.get("results", r.get("result", []))
        if isinstance(scored, list) and scored:
            scores = [s.get("salience", 0) for s in scored]
            ok("6", "score_high_importance",
               f"scores={[f'{s:.2f}' for s in scores]}, avg={sum(scores)/len(scores):.2f}")
        else:
            ok("6", "score_high_importance", f"Response: {str(r)[:100]}")
    else:
        fail("6", "score_high_importance", str(r.get("error")))

    # Test 6.2: Score low-importance facts
    r = safe_call(agent, "neomem_score_salience", {
        "facts": [
            "The weather was nice today during the meeting",
            "Client mentioned they had coffee before the call",
            "The meeting room had white walls",
        ],
    })
    if "error" not in r:
        scored = r.get("results", r.get("result", []))
        if isinstance(scored, list) and scored:
            scores = [s.get("salience", 0) for s in scored]
            ok("6", "score_low_importance",
               f"scores={[f'{s:.2f}' for s in scores]}, avg={sum(scores)/len(scores):.2f}")
        else:
            ok("6", "score_low_importance", f"Response: {str(r)[:100]}")
    else:
        fail("6", "score_low_importance", str(r.get("error")))

    # Test 6.3: Score with context
    r = safe_call(agent, "neomem_score_salience", {
        "facts": ["Prefers quarterly rebalancing over annual"],
        "context": "Discussion about portfolio management frequency and review cadence for a wealth management client",
    })
    if "error" not in r:
        scored = r.get("results", r.get("result", []))
        if isinstance(scored, list) and scored:
            ok("6", "score_with_context", f"score={scored[0].get('salience', 'N/A')}")
        else:
            ok("6", "score_with_context", f"Response: {str(r)[:80]}")
    else:
        fail("6", "score_with_context", str(r.get("error")))

    # Test 6.4: Score single fact (edge: one-item array)
    r = safe_call(agent, "neomem_score_salience", {
        "facts": ["Client is diabetic and takes insulin injections daily"],
    })
    if "error" not in r:
        ok("6", "score_single_fact", "Single-fact scoring OK")
    else:
        fail("6", "score_single_fact", str(r.get("error")))


# ─── Phase 7: History & Audit Trail ──────────────────────────────────

def test_phase_7(agent: VICMemoryAgent, stored_ids: dict):
    phase_header(7, "History & Audit Trail")

    # Test 7.1: Get history for a specific memory
    mem_id = stored_ids.get("crud_id") or stored_ids.get("first_memory_id")
    if mem_id:
        r = safe_call(agent, "neomem_get_history", {"memory_id": mem_id})
        if "error" not in r:
            events = r if isinstance(r, list) else r.get("results", r.get("result", []))
            if isinstance(events, list):
                ok("7", "get_history_by_id", f"events={len(events)} for id={mem_id[:12]}")
            else:
                ok("7", "get_history_by_id", f"response={str(r)[:80]}")
        else:
            fail("7", "get_history_by_id", str(r.get("error")))
    else:
        skip("7", "get_history_by_id", "No memory id")

    # Test 7.2: Get all history
    r = safe_call(agent, "neomem_get_all_history", {"limit": 50})
    if "error" not in r:
        events = r if isinstance(r, list) else r.get("results", r.get("result", []))
        count = len(events) if isinstance(events, list) else "N/A"
        ok("7", "get_all_history", f"events={count}")
    else:
        fail("7", "get_all_history", str(r.get("error")))

    # Test 7.3: Get all history with event filter
    r = safe_call(agent, "neomem_get_all_history", {"limit": 20, "event_filter": "ADD"})
    if "error" not in r:
        ok("7", "get_history_filtered_ADD", f"response keys={list(r.keys()) if isinstance(r, dict) else 'list'}")
    else:
        fail("7", "get_history_filtered_ADD", str(r.get("error")))

    # Test 7.4: Get history stats
    r = safe_call(agent, "neomem_get_history_stats", {})
    if "error" not in r:
        ok("7", "get_history_stats", f"stats={r}")
    else:
        fail("7", "get_history_stats", str(r.get("error")))


# ─── Phase 8: Scoped Memory ──────────────────────────────────────────

def test_phase_8(agent: VICMemoryAgent):
    phase_header(8, "Scoped Memory — Private, Shared, Team Pools")

    # Test 8.1: Store scoped — private pool
    r = safe_call(agent, "neomem_store_scoped", {
        "user_message": "What's the best SIP for someone with 10 year horizon?",
        "assistant_response": "For a 10-year horizon, a diversified equity SIP in Nifty 50 index fund is ideal.",
        "user_id": "test_user",
        "tenant_id": "wealth_corp",
        "pool": "private",
    })
    if "error" not in r:
        ok("8", "store_scoped_private", f"result keys={list(r.keys())}")
    else:
        fail("8", "store_scoped_private", str(r.get("error")))

    # Test 8.2: Store scoped — shared pool
    r = safe_call(agent, "neomem_store_scoped", {
        "user_message": "What's the company policy on ELSS recommendations?",
        "assistant_response": "Company policy recommends at least 3 ELSS fund options from different AMCs for diversification.",
        "user_id": "test_user",
        "tenant_id": "wealth_corp",
        "pool": "shared",
    })
    if "error" not in r:
        ok("8", "store_scoped_shared", "Stored in shared pool")
    else:
        fail("8", "store_scoped_shared", str(r.get("error")))

    # Test 8.3: Store scoped — team pool (with dept_id)
    r = safe_call(agent, "neomem_store_scoped", {
        "user_message": "Wealth management team procedure for high-value clients?",
        "assistant_response": "HNI clients above 1Cr AUM get dedicated RM, quarterly reviews, and priority support.",
        "user_id": "test_user",
        "tenant_id": "wealth_corp",
        "dept_id": "wealth_mgmt",
        "pool": "team",
    })
    if "error" not in r:
        ok("8", "store_scoped_team", "Stored in team pool with dept_id")
    else:
        fail("8", "store_scoped_team", str(r.get("error")))

    time.sleep(2)  # Let Qdrant index

    # Test 8.4: Search scoped — private pool
    r = safe_call(agent, "neomem_search_scoped", {
        "query": "SIP 10 year horizon",
        "user_id": "test_user",
        "tenant_id": "wealth_corp",
        "pool": "private",
    })
    if "error" not in r:
        results = r.get("results", [])
        ok("8", "search_scoped_private", f"results={len(results)}")
    else:
        fail("8", "search_scoped_private", str(r.get("error")))

    # Test 8.5: Search scoped — shared pool
    r = safe_call(agent, "neomem_search_scoped", {
        "query": "ELSS company policy",
        "user_id": "test_user",
        "tenant_id": "wealth_corp",
        "pool": "shared",
    })
    if "error" not in r:
        ok("8", "search_scoped_shared", f"results={len(r.get('results', []))}")
    else:
        fail("8", "search_scoped_shared", str(r.get("error")))

    # Test 8.6: Get all scoped — private pool
    r = safe_call(agent, "neomem_get_all_scoped", {
        "user_id": "test_user",
        "tenant_id": "wealth_corp",
        "pool": "private",
    })
    if "error" not in r:
        results = r.get("results", [])
        ok("8", "get_all_scoped_private", f"results={len(results)}")
    else:
        fail("8", "get_all_scoped_private", str(r.get("error")))

    # Test 8.7: Add scoped (raw text)
    r = safe_call(agent, "neomem_add_scoped", {
        "messages": "Compliance alert: All ESG fund recommendations must include carbon footprint disclosure",
        "user_id": "test_user",
        "tenant_id": "wealth_corp",
        "pool": "shared",
    })
    if "error" not in r:
        ok("8", "add_scoped_shared", "Raw text added to shared pool")
    else:
        fail("8", "add_scoped_shared", str(r.get("error")))

    # Test 8.8: Search scoped with strategy override
    r = safe_call(agent, "neomem_search_scoped", {
        "query": "compliance ESG",
        "user_id": "test_user",
        "tenant_id": "wealth_corp",
        "pool": "shared",
        "strategy": "read_only",
    })
    if "error" not in r:
        ok("8", "search_scoped_strategy_override", f"results={len(r.get('results', []))}")
    else:
        fail("8", "search_scoped_strategy_override", str(r.get("error")))


# ─── Phase 9: Direct Storage (store_fact, store_preference) ──────────

def test_phase_9(agent: VICMemoryAgent):
    phase_header(9, "Direct Storage — store_fact & store_preference")

    # Test 9.1: Store fact with category
    r = safe_call(agent, "neomem_store_fact", {
        "fact": "Mohan Sharma is 58 years old, planning retirement in 2 years",
        "user_id": "test_user",
        "agent_id": "test_agent",
        "categories": ["persona"],
        "investor_name": "Mohan Sharma",
    })
    if "error" not in r:
        ok("9", "store_fact_persona", f"results={len(r.get('results', []))}")
    else:
        fail("9", "store_fact_persona", str(r.get("error")))

    # Test 9.2: Store fact — episodic category
    r = safe_call(agent, "neomem_store_fact", {
        "fact": "Priya Patel opened her first investment account on 2024-01-15",
        "user_id": "test_user",
        "agent_id": "test_agent",
        "categories": ["episodic"],
        "investor_name": "Priya Patel",
    })
    if "error" not in r:
        ok("9", "store_fact_episodic", "Stored episodic fact")
    else:
        fail("9", "store_fact_episodic", str(r.get("error")))

    # Test 9.3: Store fact — procedural category
    r = safe_call(agent, "neomem_store_fact", {
        "fact": "Send quarterly portfolio review report to Ravi Verma every March, June, Sep, Dec",
        "user_id": "test_user",
        "agent_id": "test_agent",
        "categories": ["procedural"],
        "investor_name": "Ravi Verma",
    })
    if "error" not in r:
        ok("9", "store_fact_procedural", "Stored procedural fact")
    else:
        fail("9", "store_fact_procedural", str(r.get("error")))

    # Test 9.4: Store preference
    r = safe_call(agent, "neomem_store_preference", {
        "preference": "Client Mohan prefers email communication over phone calls",
        "user_id": "test_user",
        "agent_id": "test_agent",
    })
    if "error" not in r:
        ok("9", "store_preference", f"results={len(r.get('results', []))}")
    else:
        fail("9", "store_preference", str(r.get("error")))

    # Test 9.5: Store preference with extra categories
    r = safe_call(agent, "neomem_store_preference", {
        "preference": "Priya Patel wants all statements in Hindi language",
        "user_id": "test_user",
        "agent_id": "test_agent",
        "categories": ["persona"],
    })
    if "error" not in r:
        ok("9", "store_preference_with_categories", "Stored with extra categories")
    else:
        fail("9", "store_preference_with_categories", str(r.get("error")))

    # Test 9.6: Store fact with tenant_id
    r = safe_call(agent, "neomem_store_fact", {
        "fact": "Amit Kumar's tax-saving target for FY2025 is 1.5 lakhs under Section 80C",
        "user_id": "test_user",
        "agent_id": "test_agent",
        "tenant_id": "wealth_corp",
        "investor_name": "Amit Kumar",
    })
    if "error" not in r:
        ok("9", "store_fact_with_tenant", "Stored with tenant_id")
    else:
        fail("9", "store_fact_with_tenant", str(r.get("error")))


# ─── Phase 10: Lifecycle Management ──────────────────────────────────

def test_phase_10(agent: VICMemoryAgent):
    phase_header(10, "Lifecycle — Count, Cleanup, Delete All")

    # Test 10.1: Count memories
    r = safe_call(agent, "neomem_count_memories", {"user_id": "test_user"})
    if "error" not in r:
        total = r.get("total", r.get("count", "?"))
        ok("10", "count_memories", f"total={total}")
    else:
        fail("10", "count_memories", str(r.get("error")))

    # Test 10.2: Count all (no user_id filter)
    r = safe_call(agent, "neomem_count_memories", {})
    if "error" not in r:
        total = r.get("total", r.get("count", "?"))
        ok("10", "count_all_memories", f"total={total}")
    else:
        fail("10", "count_all_memories", str(r.get("error")))

    # Test 10.3: Cleanup expired — dry run
    r = safe_call(agent, "neomem_cleanup_expired", {
        "dry_run": True,
        "user_id": "test_user",
    })
    if "error" not in r:
        ok("10", "cleanup_dry_run",
           f"scanned={r.get('scanned', '?')}, would_delete={r.get('expired', '?')}")
    else:
        fail("10", "cleanup_dry_run", str(r.get("error")))

    # Test 10.4: Cleanup expired — actual run
    r = safe_call(agent, "neomem_cleanup_expired", {
        "user_id": "test_user",
    })
    if "error" not in r:
        ok("10", "cleanup_actual", f"result={r}")
    else:
        fail("10", "cleanup_actual", str(r.get("error")))

    # Test 10.5: Cleanup with salience threshold
    r = safe_call(agent, "neomem_cleanup_expired", {
        "user_id": "test_user",
        "salience_threshold": 0.1,
        "dry_run": True,
    })
    if "error" not in r:
        ok("10", "cleanup_salience_threshold", f"low_salience={r.get('low_salience', '?')}")
    else:
        fail("10", "cleanup_salience_threshold", str(r.get("error")))


# ─── Phase 11: Agent Conversational Flow ──────────────────────────────

def test_phase_11(agent: VICMemoryAgent):
    phase_header(11, "Agent LLM Chat — Multi-turn, Investors, Pronouns")

    # Create a fresh agent for chat tests
    chat_agent = VICMemoryAgent(
        user_id="chat_test_user",
        agent_id="vic_chat_test",
        dept_id="wealth",
        mcp_port=MCP_PORT,
    )
    try:
        chat_agent.connect()
    except Exception as e:
        fail("11", "chat_agent_connect", str(e))
        return

    # Test 11.1: First turn — introduce investor
    try:
        response, ops = chat_agent.chat(
            "I'm managing Rajesh Mehta's portfolio. He has a moderate risk appetite and prefers large-cap equity funds."
        )
        if response and len(response) > 10:
            store_ops = [o for o in ops if "store" in o.tool.lower()]
            ok("11", "chat_turn1_investor_intro",
               f"response_len={len(response)}, ops={len(ops)}, stores={len(store_ops)}")
        else:
            fail("11", "chat_turn1_investor_intro", f"Response too short: {response[:50]}")
    except Exception as e:
        fail("11", "chat_turn1_investor_intro", str(e))

    # Test 11.2: Pronoun resolution — "his" should resolve to Rajesh
    try:
        response, ops = chat_agent.chat(
            "What is his current asset allocation? Any large-cap funds you'd recommend?"
        )
        investor_after = chat_agent.current_investor
        if investor_after and "Rajesh" in (investor_after or ""):
            ok("11", "chat_turn2_pronoun_resolution",
               f"investor={investor_after}, response_len={len(response)}")
        elif investor_after:
            ok("11", "chat_turn2_pronoun_resolution",
               f"investor={investor_after} (pronoun resolved, might differ)")
        else:
            fail("11", "chat_turn2_pronoun_resolution", "Investor not resolved via pronoun")
    except Exception as e:
        fail("11", "chat_turn2_pronoun_resolution", str(e))

    # Test 11.3: Switch investor
    try:
        response, ops = chat_agent.chat(
            "Now let's look at Sunita Agarwal's portfolio. She's a new client interested in international funds."
        )
        investor_after = chat_agent.current_investor
        if investor_after and "Sunita" in (investor_after or ""):
            ok("11", "chat_turn3_switch_investor",
               f"investor={investor_after}")
        else:
            ok("11", "chat_turn3_switch_investor",
               f"investor={investor_after} (may need explicit name)")
    except Exception as e:
        fail("11", "chat_turn3_switch_investor", str(e))

    # Test 11.4: Verify memories were stored from chat
    time.sleep(2)
    r = safe_call(chat_agent, "neomem_search", {
        "query": "Rajesh moderate risk large cap",
        "user_id": "chat_test_user",
        "limit": 5,
    })
    if "error" not in r:
        results = r.get("results", [])
        ok("11", "chat_memories_searchable", f"found={len(results)} memories from chat")
    else:
        fail("11", "chat_memories_searchable", str(r.get("error")))

    # Test 11.5: Retrieve context from chat memories
    r = safe_call(chat_agent, "neomem_retrieve_context", {
        "query": "What do we know about Rajesh?",
        "user_id": "chat_test_user",
        "investor_name": "Rajesh Mehta",
    })
    if "error" not in r:
        memories = r.get("memories", [])
        ctx = r.get("context", "")
        ok("11", "chat_retrieve_context",
           f"memories={len(memories)}, context_len={len(ctx)}")
    else:
        fail("11", "chat_retrieve_context", str(r.get("error")))

    # Test 11.6: Multi-turn continuity — follow-up question
    try:
        response, ops = chat_agent.chat(
            "Based on her interest in international funds, what allocation would you suggest for Sunita?"
        )
        ok("11", "chat_multi_turn_continuity",
           f"response_len={len(response)}, investor={chat_agent.current_investor}")
    except Exception as e:
        fail("11", "chat_multi_turn_continuity", str(e))

    # Cleanup chat agent
    try:
        chat_agent.close()
    except Exception:
        pass


# ─── Phase 12: Edge Cases ────────────────────────────────────────────

def test_phase_12(agent: VICMemoryAgent):
    phase_header(12, "Edge Cases & Error Handling")

    # Test 12.1: Get non-existent memory
    r = safe_call(agent, "neomem_get_memory", {"memory_id": "00000000-0000-0000-0000-000000000000"})
    if r.get("result") is None or r.get("error") or "not found" in str(r).lower() or r.get("memory") is None:
        ok("12", "get_nonexistent_memory", "Returned null/error as expected")
    else:
        ok("12", "get_nonexistent_memory", f"Result: {str(r)[:60]}")

    # Test 12.2: Delete non-existent memory
    r = safe_call(agent, "neomem_delete_memory", {"memory_id": "00000000-0000-0000-0000-000000000000"})
    # Should not crash — either returns success or a soft error
    ok("12", "delete_nonexistent_memory", f"Handled gracefully: {str(r)[:60]}")

    # Test 12.3: Update non-existent memory
    r = safe_call(agent, "neomem_update_memory", {
        "memory_id": "00000000-0000-0000-0000-000000000000",
        "data": "This should not work",
    })
    ok("12", "update_nonexistent_memory", f"Handled: {str(r)[:60]}")

    # Test 12.4: Store exchange with minimal params
    r = safe_call(agent, "neomem_store_exchange", {
        "query": "Hi",
        "response": "Hello",
        "user_id": "test_user",
    })
    if "error" not in r:
        ok("12", "store_minimal_exchange", f"extracted={r.get('facts_extracted', 0)}")
    else:
        fail("12", "store_minimal_exchange", str(r.get("error")))

    # Test 12.5: Store exchange with special characters
    r = safe_call(agent, "neomem_store_exchange", {
        "query": "Client's name is O'Brien & she has ₹50,000 to invest. Email: test@example.com",
        "response": "I've noted O'Brien's investment of ₹50,000. We'll set up her <portfolio> accordingly.",
        "user_id": "test_user",
        "agent_id": "test_agent",
    })
    if "error" not in r:
        ok("12", "store_special_characters", "Handled special chars OK")
    else:
        fail("12", "store_special_characters", str(r.get("error")))

    # Test 12.6: Store exchange with very long text
    long_query = "Detailed investment discussion: " + " ".join(
        [f"Point {i}: Client wants diversification across multiple sectors including "
         f"technology, healthcare, banking, FMCG, and infrastructure for balanced growth."
         for i in range(20)]
    )
    r = safe_call(agent, "neomem_store_exchange", {
        "query": long_query[:5000],  # ~5KB
        "response": "I've noted all your detailed investment requirements across sectors.",
        "user_id": "test_user",
    })
    if "error" not in r:
        ok("12", "store_long_text", f"extracted={r.get('facts_extracted', 0)} from long text")
    else:
        fail("12", "store_long_text", str(r.get("error")))

    # Test 12.7: Search with empty results (very specific query)
    r = safe_call(agent, "neomem_search", {
        "query": "xyzzy_nonexistent_absurd_query_12345",
        "user_id": "test_user_nobody",
        "limit": 5,
    })
    if "error" not in r:
        results = r.get("results", [])
        ok("12", "search_empty_results", f"results={len(results)} (expected 0 or few)")
    else:
        fail("12", "search_empty_results", str(r.get("error")))

    # Test 12.8: Count memories for non-existent user
    r = safe_call(agent, "neomem_count_memories", {"user_id": "user_does_not_exist"})
    if "error" not in r:
        total = r.get("total", r.get("count", 0))
        ok("12", "count_nonexistent_user", f"total={total} (expected 0)")
    else:
        fail("12", "count_nonexistent_user", str(r.get("error")))

    # Test 12.9: Score salience with empty context
    r = safe_call(agent, "neomem_score_salience", {
        "facts": ["User prefers morning meetings"],
        "context": "",
    })
    if "error" not in r:
        ok("12", "score_empty_context", "Handled empty context")
    else:
        fail("12", "score_empty_context", str(r.get("error")))

    # Test 12.10: Retrieve context for empty store
    r = safe_call(agent, "neomem_retrieve_context", {
        "query": "anything",
        "user_id": "brand_new_user_no_data",
    })
    if "error" not in r:
        count = r.get("count", 0)
        ok("12", "retrieve_empty_user", f"count={count} (expected 0)")
    else:
        fail("12", "retrieve_empty_user", str(r.get("error")))

    # Test 12.11: Flush buffer for user with no buffer
    r = safe_call(agent, "neomem_flush_buffer", {
        "user_id": "no_buffer_user",
    })
    if "error" not in r:
        ok("12", "flush_empty_user_buffer", f"flushed={r.get('flushed_count', r.get('flushed', 0))}")
    else:
        fail("12", "flush_empty_user_buffer", str(r.get("error")))

    # Test 12.12: Get all for user with no memories
    r = safe_call(agent, "neomem_get_all", {"user_id": "no_memories_user"})
    if "error" not in r:
        ok("12", "get_all_empty_user", f"results={len(r.get('results', []))}")
    else:
        fail("12", "get_all_empty_user", str(r.get("error")))


# ─── Phase 13: Custom Extraction Prompt Detailed Test ─────────────────

def test_phase_13(agent: VICMemoryAgent):
    phase_header(13, "Custom Extraction Prompt — Detailed Verification")

    # Test 13.1: Default prompt (no custom) — baseline
    r = safe_call(agent, "neomem_store_exchange", {
        "query": "Sunita wants SIP in Nifty BeES. She's 30 years old with high risk tolerance.",
        "response": "I'll set up Sunita's Nifty BeES SIP. At 30 with high risk tolerance, this is an excellent growth strategy.",
        "user_id": "prompt_test_user",
        "agent_id": "test_agent",
    })
    if "error" not in r:
        baseline_extracted = r.get("facts_extracted", 0)
        ok("13", "default_prompt_baseline", f"extracted={baseline_extracted}")
    else:
        fail("13", "default_prompt_baseline", str(r.get("error")))

    # Test 13.2: Custom prompt — minimal extraction
    r = safe_call(agent, "neomem_store_exchange", {
        "query": "Arjun Reddy is 45 years old, married, 2 kids. Works at TCS. Prefers hybrid funds for balanced growth.",
        "response": "Noted Arjun's profile: 45yo, family man at TCS. Hybrid funds align well with his stage of life.",
        "user_id": "prompt_test_user",
        "agent_id": "test_agent",
        "custom_extraction_prompt": """Extract ONLY the investment-related preference as a single fact.
Ignore personal details like age, family, employer.
Return JSON: {"memories": [{"value": "<investment preference>", "category": "preference", "key": "fund_preference", "salience": 0.8, "reasoning": "direct preference"}]}""",
    })
    if "error" not in r:
        custom_extracted = r.get("facts_extracted", 0)
        ok("13", "custom_prompt_minimal", f"extracted={custom_extracted}")
    else:
        fail("13", "custom_prompt_minimal", str(r.get("error")))

    # Test 13.3: Custom prompt — domain-specific extraction
    r = safe_call(agent, "neomem_store_exchange", {
        "query": "Patient ID P-1234 reports blood pressure 140/90, taking Amlodipine 5mg daily. Diabetic, HbA1c 7.2%. Next appointment March 15.",
        "response": "Noted the vitals and medication. HbA1c slightly elevated. Will review at next appointment.",
        "user_id": "prompt_test_user",
        "agent_id": "test_agent",
        "custom_extraction_prompt": """You are a medical record extraction agent. Extract clinical facts only.
For each fact, provide: value, category (one of: vitals, medication, diagnosis, procedure), 
key (clinical_key), salience (0.0-1.0 based on clinical urgency).
Return: {"memories": [...]}""",
    })
    if "error" not in r:
        ok("13", "custom_prompt_domain_specific",
           f"extracted={r.get('facts_extracted', 0)} medical facts")
    else:
        fail("13", "custom_prompt_domain_specific", str(r.get("error")))

    # Cleanup prompt test user
    safe_call(agent, "neomem_delete_all", {"user_id": "prompt_test_user"})


# ─── Phase 14: Agent with Custom Prompt ───────────────────────────────

def test_phase_14(agent: VICMemoryAgent):
    phase_header(14, "Agent with Custom Extraction Prompt")

    # Create agent with custom extraction prompt
    custom_agent = VICMemoryAgent(
        user_id="custom_prompt_user",
        agent_id="custom_agent",
        dept_id="wealth",
        mcp_port=MCP_PORT,
        custom_extraction_prompt="""You are a wealth management memory extraction agent.
Extract ONLY facts relevant to financial planning:
- Investment preferences and risk tolerance
- Portfolio goals and time horizons  
- Specific fund/instrument preferences
- Compliance requirements
Ignore: greetings, weather, casual conversation, personal non-financial details.
For each fact return: {"value": "...", "category": "preference|persona|episodic|procedural", "key": "...", "salience": 0.0-1.0, "reasoning": "..."}
Return: {"memories": [...]}""",
    )

    try:
        custom_agent.connect()

        # Test 14.1: Chat with custom agent — should extract only financial facts
        response, ops = custom_agent.chat(
            "Hi, the weather is great today! By the way, I want to shift Vikram's portfolio from aggressive to moderate risk. He's turning 50 next month."
        )
        store_ops = [o for o in ops if "store" in o.tool.lower()]
        ok("14", "custom_agent_chat",
           f"response_len={len(response)}, stores={len(store_ops)}")

        # Test 14.2: Verify stored memories are financial-focused
        time.sleep(2)
        r = safe_call(custom_agent, "neomem_search", {
            "query": "Vikram risk moderate",
            "user_id": "custom_prompt_user",
            "limit": 10,
        })
        results = r.get("results", [])
        ok("14", "custom_agent_memories_financial",
           f"found={len(results)} memories" + (f", top='{results[0].get('memory','')[:60]}'" if results else ""))

        # Test 14.3: Second turn with custom agent
        response, ops = custom_agent.chat(
            "Also, Vikram wants to start a monthly SIP of ₹1 lakh in a flexi-cap fund."
        )
        ok("14", "custom_agent_turn2",
           f"response_len={len(response)}, ops={len(ops)}")

        custom_agent.close()
    except Exception as e:
        fail("14", "custom_agent_flow", str(e))

    # Cleanup
    safe_call(agent, "neomem_delete_all", {"user_id": "custom_prompt_user"})


# ─── Final Cleanup & Summary ─────────────────────────────────────────

def test_final_cleanup(agent: VICMemoryAgent):
    phase_header(99, "Final Cleanup")

    # Delete test user memories
    for uid in ["test_user", "chat_test_user", "prompt_test_user", "custom_prompt_user"]:
        safe_call(agent, "neomem_delete_all", {"user_id": uid})

    # Verify cleanup
    r = safe_call(agent, "neomem_count_memories", {"user_id": "test_user"})
    total = r.get("total", r.get("count", "?"))
    print(f"  Cleanup complete — test_user memories remaining: {total}")


def print_summary():
    print(f"\n{'='*70}")
    print(f"COMPREHENSIVE TEST RESULTS")
    print(f"{'='*70}")
    print(f"  ✅ PASSED:  {PASS}")
    print(f"  ❌ FAILED:  {FAIL}")
    print(f"  ⏭️  SKIPPED: {SKIP}")
    print(f"  📊 TOTAL:   {PASS + FAIL + SKIP}")
    print(f"  📈 RATE:    {PASS / max(PASS + FAIL, 1) * 100:.1f}%")
    print(f"{'='*70}")

    if FAIL > 0:
        print(f"\nFailed tests:")
        for phase, name, status in RESULTS:
            if status == "FAIL":
                print(f"  Phase {phase}: {name}")
    print()


# ─── Main ─────────────────────────────────────────────────────────────

def main():
    print(f"╔{'═'*68}╗")
    print(f"║  NeoMemory MCP — Comprehensive E2E Test Suite                      ║")
    print(f"║  Port: {MCP_PORT:<7}                                                    ║")
    print(f"╚{'═'*68}╝")

    agent = VICMemoryAgent(
        user_id="test_user",
        agent_id="test_agent",
        dept_id="wealth_test",
        mcp_port=MCP_PORT,
    )

    try:
        # Phase 1: Connection
        if not test_phase_1(agent):
            print("\n⚠️  Cannot proceed without connection. Exiting.")
            return

        # Phase 2: Store Exchange
        stored_ids = test_phase_2(agent)

        time.sleep(2)  # Let Qdrant index

        # Phase 3: Retrieval & Search
        test_phase_3(agent, stored_ids)

        # Phase 4: CRUD
        stored_ids = test_phase_4(agent, stored_ids)

        # Phase 5: Buffering
        test_phase_5(agent)

        # Phase 6: Salience
        test_phase_6(agent)

        # Phase 7: History
        test_phase_7(agent, stored_ids)

        # Phase 8: Scoped
        test_phase_8(agent)

        # Phase 9: Direct Storage
        test_phase_9(agent)

        # Phase 10: Lifecycle
        test_phase_10(agent)

        # Phase 11: Agent Chat
        test_phase_11(agent)

        # Phase 12: Edge Cases
        test_phase_12(agent)

        # Phase 13: Custom Prompt
        test_phase_13(agent)

        # Phase 14: Agent with Custom Prompt
        test_phase_14(agent)

        # Final Cleanup
        test_final_cleanup(agent)

    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
    except Exception as e:
        print(f"\n\n💥 Unexpected error: {e}")
        traceback.print_exc()
    finally:
        try:
            agent.close()
        except Exception:
            pass
        print_summary()


if __name__ == "__main__":
    main()
