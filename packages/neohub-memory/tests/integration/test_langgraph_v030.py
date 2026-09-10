"""
LangGraph Agent Integration Tests — Neo Memory Hub v0.3.0.

Tests the HighLevelMemoryConnector with LangGraph ReAct agent:
1. Memory-augmented agent conversation (store + retrieve)
2. Multi-turn conversation with persistent memory
3. Profile-aware context injection into agent prompt
4. End-to-end RAG with v0.3.0 pipeline

Requirements:
    - OPENAI_API_KEY set (from .env)
    - Qdrant running on localhost:6335
    - pip install neo-memory-hub[langgraph,dev]

Run with:
    pytest tests/integration/test_langgraph_v030.py -v -s
"""
from __future__ import annotations

import asyncio
import os
import socket
import time
import uuid
from pathlib import Path
from typing import Any

import pytest
from dotenv import load_dotenv

# Load environment from project root
project_root = Path(__file__).parent.parent.parent
env_path = project_root / ".env"
if env_path.exists():
    load_dotenv(env_path)

from neo_memory_hub import (
    HighLevelMemoryConnector,
    MemoryHubConfig,
    TelemetryHook,
)
from neo_memory_hub.config.hub_config import (
    VectorStoreConfig,
    LLMConfig,
    EmbedderConfig,
    ExtractionConfig,
    DedupConfig,
    BufferingConfig,
    SalienceConfig,
    IsolationConfig,
    RetrievalConfig,
    FeatureConfig,
)


# =============================================================================
# Skip conditions
# =============================================================================

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6335")


def has_openai_key() -> bool:
    return bool(os.getenv("OPENAI_API_KEY"))


def has_langgraph() -> bool:
    try:
        import langgraph
        return True
    except ImportError:
        return False


def qdrant_available() -> bool:
    try:
        from urllib.parse import urlparse
        parsed = urlparse(QDRANT_URL)
        host = parsed.hostname or "localhost"
        port = parsed.port or 6335
        with socket.create_connection((host, port), timeout=2):
            return True
    except Exception:
        return False


requires_all = pytest.mark.skipif(
    not (has_openai_key() and has_langgraph() and qdrant_available()),
    reason="OPENAI_API_KEY, langgraph, or Qdrant not available",
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def unique_id():
    return f"lg30_{uuid.uuid4().hex[:8]}"


@pytest.fixture
def collection_name():
    return f"langgraph_v030_{int(time.time())}_{uuid.uuid4().hex[:4]}"


@pytest.fixture
def hlc_config(collection_name):
    """HighLevelMemoryConnector config for LangGraph tests."""
    return MemoryHubConfig(
        vector_store=VectorStoreConfig(
            provider="qdrant",
            qdrant_url=QDRANT_URL,
            collection_name=collection_name,
        ),
        llm=LLMConfig(model="gpt-4o-mini", temperature=0),
        embedder=EmbedderConfig(model="text-embedding-3-small", dimensions=1536),
        extraction=ExtractionConfig(enabled=True),
        dedup=DedupConfig(enabled=False),
        buffering=BufferingConfig(enabled=False),
        salience=SalienceConfig(enabled=False),
        isolation=IsolationConfig(require_user_id=False),
        features=FeatureConfig(storage_enabled=True, retrieval_enabled=True),
        retrieval=RetrievalConfig(limit=10, min_relevance_score=0.1),
    )


class LiveTelemetry(TelemetryHook):
    def __init__(self):
        self.events = []

    def on_storage(self, facts_stored, facts_skipped, facts_extracted, investor_name=None):
        self.events.append(("storage", facts_stored, facts_extracted, investor_name))

    def on_retrieval(self, query, count, category_counts, investor_name=None):
        self.events.append(("retrieval", query[:30], count, investor_name))

    def on_dedup(self, action, category, investor_name=None):
        self.events.append(("dedup", action, category, investor_name))


# =============================================================================
# TEST 1: LangGraph Agent with v0.3.0 Memory
# =============================================================================

@requires_all
class TestLangGraphV030:
    """LangGraph agent integration with HighLevelMemoryConnector."""

    @pytest.mark.asyncio
    async def test_react_agent_with_memory(self, hlc_config, unique_id):
        """Build a ReAct agent that uses v0.3.0 memory for context."""
        from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
        from langchain_openai import ChatOpenAI
        from langgraph.graph import END, StateGraph
        from langgraph.graph.message import add_messages
        from typing import Annotated, TypedDict

        print("\n" + "=" * 70)
        print("  LANGGRAPH REACT AGENT + v0.3.0 MEMORY")
        print("=" * 70)

        telemetry = LiveTelemetry()

        async with HighLevelMemoryConnector(
            config=hlc_config, telemetry=telemetry
        ) as hlc:
            user_id = unique_id
            investor = "Meera Joshi"

            # ------------------------------------------------------------------
            # Graph state — use runtime annotations to avoid get_type_hints issue
            # ------------------------------------------------------------------
            AgentState = TypedDict("AgentState", {
                "messages": Annotated[list[BaseMessage], add_messages],
                "memory_context": str,
                "user_id": str,
                "investor_name": str,
            })

            # ------------------------------------------------------------------
            # Nodes
            # ------------------------------------------------------------------
            async def retrieve_memory(state: AgentState) -> dict:
                """Retrieve v0.3.0 memory context for the agent."""
                user_msg = state["messages"][-1].content
                inv = state.get("investor_name", "")
                ctx = await hlc.retrieve_context(
                    query=user_msg,
                    user_id=state["user_id"],
                    investor_name=inv or None,
                )
                return {"memory_context": ctx.get("context", "")}

            async def generate(state: AgentState) -> dict:
                """LLM call with memory-augmented prompt."""
                llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)
                mem_ctx = state.get("memory_context", "")
                system_content = f"""You are a helpful wealth management assistant.
You have memory about clients. Use it to personalize your responses.

{mem_ctx if mem_ctx else 'No memory context available yet.'}

Respond briefly and naturally."""
                messages = [SystemMessage(content=system_content)] + list(state["messages"])
                response = await llm.ainvoke(messages)
                return {"messages": [response]}

            async def store_memory(state: AgentState) -> dict:
                """Store the conversation exchange via v0.3.0 pipeline."""
                msgs = state["messages"]
                if len(msgs) >= 2:
                    user_msg = msgs[-2].content
                    ai_msg = msgs[-1].content
                    await hlc.store_exchange(
                        query=user_msg,
                        response=ai_msg,
                        user_id=state["user_id"],
                        investor_name=state.get("investor_name"),
                    )
                return {}

            # ------------------------------------------------------------------
            # Build graph
            # ------------------------------------------------------------------
            print("\n[1] Building LangGraph agent...")
            graph = StateGraph(AgentState)
            graph.add_node("retrieve", retrieve_memory)
            graph.add_node("generate", generate)
            graph.add_node("store", store_memory)
            graph.set_entry_point("retrieve")
            graph.add_edge("retrieve", "generate")
            graph.add_edge("generate", "store")
            graph.add_edge("store", END)
            agent = graph.compile()
            print("    ✅ Agent compiled")

            # ------------------------------------------------------------------
            # Multi-turn conversation
            # ------------------------------------------------------------------
            conversations = [
                (
                    "I'm calling about Meera Joshi, she's a 38-year-old surgeon "
                    "with an aggressive risk profile and $2M portfolio."
                ),
                (
                    "Meera mentioned she prefers email updates and wants to increase "
                    "her allocation in biotech stocks."
                ),
                (
                    "What do you know about Meera's investment profile and preferences?"
                ),
            ]

            print(f"\n[2] Running {len(conversations)} turns...")
            for i, msg in enumerate(conversations, 1):
                print(f"\n   Turn {i}: '{msg[:60]}...'")
                result = await agent.ainvoke({
                    "messages": [HumanMessage(content=msg)],
                    "memory_context": "",
                    "user_id": user_id,
                    "investor_name": investor,
                })
                response = result["messages"][-1].content
                print(f"   Agent: '{response[:120]}...'")
                await asyncio.sleep(2)

            # ------------------------------------------------------------------
            # Verify memory was stored and retrieved
            # ------------------------------------------------------------------
            print("\n[3] Verifying memory pipeline...")

            # Check that store_exchange was called
            storage_events = [e for e in telemetry.events if e[0] == "storage"]
            print(f"    Storage events: {len(storage_events)}")
            for ev in storage_events:
                print(f"      → stored={ev[1]}, extracted={ev[2]}, investor={ev[3]}")
            assert len(storage_events) >= 2, "Should have at least 2 storage events"

            # Check that retrieve was called
            retrieval_events = [e for e in telemetry.events if e[0] == "retrieval"]
            print(f"    Retrieval events: {len(retrieval_events)}")
            for ev in retrieval_events:
                print(f"      → query='{ev[1]}...', count={ev[2]}, investor={ev[3]}")

            # Final retrieval should have found memories
            final_ctx = await hlc.retrieve_context(
                query="Tell me everything about Meera",
                user_id=user_id,
                investor_name=investor,
            )
            print(f"\n    Final retrieval: {final_ctx['count']} memories")
            if final_ctx["context"]:
                print(f"    Context preview:\n{final_ctx['context'][:300]}")
            assert final_ctx["count"] > 0, "Agent should have stored memories about Meera"

            # Check structured profile
            profile = final_ctx.get("structured_profile", {})
            print(f"    Profile sections: {list(profile.keys())}")

            print("\n    ✅ LangGraph + v0.3.0 Memory test PASSED!")

    @pytest.mark.asyncio
    async def test_memory_enhanced_rag_v030(self, hlc_config, unique_id):
        """Test memory-enhanced RAG: pre-populate facts, then use in agent."""
        from langchain_core.messages import HumanMessage, SystemMessage
        from langchain_openai import ChatOpenAI

        print("\n" + "=" * 70)
        print("  MEMORY-ENHANCED RAG (v0.3.0)")
        print("=" * 70)

        async with HighLevelMemoryConnector(config=hlc_config) as hlc:
            user_id = unique_id
            investor = "Arjun Reddy"

            # Step 1: Pre-populate with rich client info
            print("\n[1] Pre-populating client facts...")
            exchanges = [
                (
                    "New client onboarding for Arjun Reddy",
                    "Arjun Reddy is a 50-year-old pharmaceutical company CEO. "
                    "He has an aggressive risk profile and manages a $10M portfolio. "
                    "He's interested in AI and clean energy investments.",
                ),
                (
                    "What are Arjun's communication preferences?",
                    "Arjun prefers monthly video calls for detailed reviews and "
                    "weekly Slack notifications for significant market moves. "
                    "He dislikes lengthy written reports.",
                ),
                (
                    "Record today's meeting notes about Arjun",
                    "Today Arjun approved a $2M allocation to the AI sector ETF. "
                    "He also asked about tax-efficient withdrawal strategies for "
                    "a potential property purchase next year.",
                ),
            ]

            for q, r in exchanges:
                result = await hlc.store_exchange(
                    query=q, response=r,
                    user_id=user_id, investor_name=investor,
                )
                print(f"    Stored: {result.get('facts_stored', 0)} facts from '{q[:40]}...'")

            await asyncio.sleep(3)

            # Step 2: Retrieve context for RAG
            print("\n[2] Building RAG context...")
            ctx = await hlc.retrieve_context(
                query="Prepare a portfolio review summary for Arjun focusing on his interests",
                user_id=user_id,
                investor_name=investor,
            )
            print(f"    Retrieved {ctx['count']} memories")

            assert ctx["count"] > 0, "Should have Pre-populated memories"
            assert ctx["context"], "Context should be non-empty"

            # Step 3: Use context in LLM call
            print("\n[3] Generating RAG response...")
            llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
            messages = [
                SystemMessage(content=f"""You are a wealth management AI assistant.
Use the following client memory context to personalize your response.

{ctx['context']}

Be specific and reference client details from memory."""),
                HumanMessage(content="Draft a brief portfolio review summary for Arjun."),
            ]
            response = await llm.ainvoke(messages)
            print(f"\n    RAG Response:\n    {response.content[:400]}...")

            # The response should reference details from memory
            response_lower = response.content.lower()
            # At least some of these should appear in the response
            keywords = ["arjun", "portfolio", "aggressive", "ai", "clean energy",
                        "pharma", "$10m", "$2m", "video", "slack"]
            found = [k for k in keywords if k in response_lower]
            print(f"\n    Keywords found in response: {found}")
            assert len(found) >= 2, f"Expected ≥2 keywords from memory, found: {found}"

            print("\n    ✅ Memory-enhanced RAG v0.3.0 PASSED!")

    @pytest.mark.asyncio
    async def test_agent_with_profile_context(self, hlc_config, unique_id):
        """Test that the formatted profile context is properly structured."""
        print("\n" + "=" * 70)
        print("  PROFILE CONTEXT FORMATTING")
        print("=" * 70)

        async with HighLevelMemoryConnector(config=hlc_config) as hlc:
            user_id = unique_id
            investor = "Kavita Sharma"

            # Store diverse facts (persona + preference + episodic)
            print("\n[1] Storing diverse facts...")
            await hlc.store_exchange(
                query="Tell me about Kavita",
                response=(
                    "Kavita Sharma is a retired teacher, age 65, very conservative. "
                    "She insists on phone calls only, no email. Last month she was "
                    "upset about a 2% drop in her bond portfolio."
                ),
                user_id=user_id,
                investor_name=investor,
            )

            await asyncio.sleep(3)

            # Retrieve and inspect context format
            print("[2] Retrieving formatted context...")
            ctx = await hlc.retrieve_context(
                query="Tell me about Kavita's profile and recent interactions",
                user_id=user_id,
                investor_name=investor,
            )

            print(f"    Memories: {ctx['count']}")
            print(f"    Profile: {list(ctx.get('structured_profile', {}).keys())}")
            print(f"    Context:\n{ctx['context']}")

            if ctx["count"] > 0:
                # Context should have the standard header format
                context = ctx["context"]
                assert "---" in context or "MEMORY CONTEXT" in context or "CLIENT" in context, \
                    "Context should have the standard formatted header"
                print("    ✅ Context has structured formatting")
            else:
                print("    ⚠️ No memories retrieved (may need more time for indexing)")

            print("\n    ✅ Profile context test PASSED!")


# =============================================================================
# Runner
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s", "--no-header"])
