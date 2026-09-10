"""
LangGraph Agent Integration Tests.

Tests the NeoMemoryConnector integration with LangGraph workflows:
- Memory retrieval during conversation
- Fact extraction and storage
- Context building for agent

Requirements:
- OPENAI_API_KEY set (from .env)
- Qdrant running (Docker)
- langgraph installed

Run with: pytest tests/integration/test_langgraph.py -v -s
"""

import asyncio
import os
import socket
import uuid
from pathlib import Path
from typing import Annotated, TypedDict

import pytest
from dotenv import load_dotenv

# Load environment from project root
project_root = Path(__file__).parent.parent.parent
env_path = project_root / ".env"
if env_path.exists():
    load_dotenv(env_path)


# =============================================================================
# Skip conditions
# =============================================================================

def has_openai_key() -> bool:
    """Check if OpenAI API key is available."""
    return bool(os.getenv("OPENAI_API_KEY"))


def has_langgraph() -> bool:
    """Check if langgraph is installed."""
    try:
        import langgraph
        return True
    except ImportError:
        return False


def qdrant_available() -> bool:
    """Check if Qdrant server is available."""
    qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6335")
    try:
        from urllib.parse import urlparse
        parsed = urlparse(qdrant_url)
        host = parsed.hostname or "localhost"
        port = parsed.port or 6335
        with socket.create_connection((host, port), timeout=1):
            return True
    except Exception:
        return False


requires_infrastructure = pytest.mark.skipif(
    not (has_openai_key() and has_langgraph()),
    reason="OPENAI_API_KEY not set or langgraph not installed"
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def unique_id():
    """Generate unique ID for test isolation."""
    return uuid.uuid4().hex[:8]


@pytest.fixture
def memory_config():
    """Create memory configuration using Qdrant."""
    from neo_memory_hub import MemoryConfig
    
    qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6335")
    
    if not qdrant_available():
        pytest.skip("Qdrant not available")
    
    return MemoryConfig(
        vector_store={
            "provider": "qdrant",
            "config": {
                "collection_name": f"langgraph_test_{uuid.uuid4().hex[:8]}",
                "embedding_model_dims": 1536,
                "url": qdrant_url,
            },
        },
        llm={
            "provider": "openai",
            "config": {
                "model": "gpt-4o-mini",
                "temperature": 0,
            },
        },
        embedder={
            "provider": "openai",
            "config": {
                "model": "text-embedding-3-small",
            },
        },
        version="v1.1",
    )


# =============================================================================
# LangGraph Integration Tests
# =============================================================================

@requires_infrastructure
class TestLangGraphIntegration:
    """Tests for LangGraph agent integration."""

    @pytest.mark.asyncio
    async def test_agent_with_memory(self, memory_config, unique_id):
        """Test LangGraph agent with NeoMemoryConnector."""
        from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
        from langchain_openai import ChatOpenAI
        from langgraph.graph import END, StateGraph
        from langgraph.graph.message import add_messages

        from neo_memory_hub import NeoMemoryConnector

        print("\n" + "=" * 60)
        print("LANGGRAPH AGENT WITH MEMORY TEST")
        print("=" * 60)

        # State definition
        class AgentState(TypedDict):
            messages: Annotated[list[BaseMessage], add_messages]
            user_id: str
            memory_context: str

        # Create connector
        connector = NeoMemoryConnector(memory_config=memory_config)
        await connector.initialize()

        user_id = f"langgraph_user_{unique_id}"

        # Node: Retrieve memories
        async def retrieve_memories(state: AgentState) -> dict:
            """Retrieve relevant memories for the conversation."""
            user_message = state["messages"][-1].content
            context = await connector.build_context(
                user_message,
                user_id=user_id,
                limit=3,
            )
            return {"memory_context": context if context else "No previous context."}

        # Node: Generate response
        async def generate_response(state: AgentState) -> dict:
            """Generate response using LLM with memory context."""
            llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
            system = f"""You are a helpful assistant with memory.
            
{state['memory_context']}

Respond briefly and naturally."""
            messages = [SystemMessage(content=system)] + list(state["messages"])
            response = await llm.ainvoke(messages)
            return {"messages": [response]}

        # Node: Store memory
        async def store_memory(state: AgentState) -> dict:
            """Store the conversation exchange as memory."""
            messages = state["messages"]
            if len(messages) >= 2:
                user_msg = messages[-2].content
                assistant_msg = messages[-1].content
                await connector.store_exchange(
                    user_message=user_msg,
                    assistant_response=assistant_msg,
                    user_id=user_id,
                    metadata={"source": "langgraph"},
                )
            return {}

        # Build graph
        print("\n[1] Building LangGraph workflow...")
        graph = StateGraph(AgentState)
        graph.add_node("retrieve", retrieve_memories)
        graph.add_node("generate", generate_response)
        graph.add_node("store", store_memory)

        graph.set_entry_point("retrieve")
        graph.add_edge("retrieve", "generate")
        graph.add_edge("generate", "store")
        graph.add_edge("store", END)

        agent = graph.compile()
        print("    ✅ Agent compiled")

        # Run conversation
        print("\n[2] Running conversation...")
        
        conversations = [
            f"[{unique_id}] Hi! My name is Alice and I'm a data scientist.",
            "I work primarily with Python and machine learning.",
            "What do you remember about me?",
        ]

        for i, message in enumerate(conversations, 1):
            print(f"\n   Turn {i}: '{message[:50]}...'")

            result = await agent.ainvoke(
                {
                    "messages": [HumanMessage(content=message)],
                    "user_id": user_id,
                    "memory_context": "",
                }
            )

            response = result["messages"][-1].content
            print(f"   Response: '{response[:100]}...'")

            await asyncio.sleep(1)  # Rate limiting

        # Verify memories were stored
        print("\n[3] Verifying stored memories...")
        all_memories = await connector.get_all(user_id=user_id)
        memories_list = all_memories.get("results", [])
        print(f"    Total memories stored: {len(memories_list)}")

        assert len(memories_list) > 0, "Should have stored at least one memory"

        # Check if key facts were extracted
        memory_text = " ".join(m.get("memory", "").lower() for m in memories_list)
        print(f"    Memory content preview: {memory_text[:200]}...")

        # Cleanup
        await connector.close()
        print("\n✅ LangGraph agent integration test passed!")

    @pytest.mark.asyncio
    async def test_memory_enhanced_rag(self, memory_config, unique_id):
        """Test memory-enhanced RAG pattern."""
        from neo_memory_hub import NeoMemoryConnector

        print("\n" + "=" * 60)
        print("MEMORY-ENHANCED RAG TEST")
        print("=" * 60)

        async with NeoMemoryConnector(memory_config=memory_config) as connector:
            user_id = f"rag_user_{unique_id}"

            # Pre-populate with user facts
            print("\n[1] Pre-populating user facts...")
            facts = [
                "User's name is Bob",
                "User is a senior software engineer",
                "User prefers TypeScript over JavaScript",
                "User works on cloud infrastructure",
            ]
            
            for fact in facts:
                await connector.store_fact(
                    f"[{unique_id}] {fact}",
                    user_id=user_id,
                )
            
            await asyncio.sleep(1)

            # Build context for a query
            print("\n[2] Building RAG context...")
            query = "What technologies does the user work with?"
            context = await connector.build_context(
                query,
                user_id=user_id,
                limit=5,
                include_scores=True,
            )

            print(f"    Query: {query}")
            print(f"    Context:\n{context}")

            assert len(context) > 0, "Should build non-empty context"
            assert "Relevant information" in context

        print("\n✅ Memory-enhanced RAG test passed!")


# =============================================================================
# Main runner
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
