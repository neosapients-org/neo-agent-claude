"""
End-to-End Integration Tests for NeoMemoryConnector.

These tests verify the complete memory lifecycle with real infrastructure:
- Store → Retrieve → Update → Delete
- User isolation
- Context building
- Operation logging

Requirements:
- OPENAI_API_KEY set (from .env)
- Qdrant running (Docker) or use in-memory mode

Run with: pytest tests/integration/test_e2e_connector.py -v -s
"""

import asyncio
import os
import socket
import uuid
from pathlib import Path

import pytest
from dotenv import load_dotenv

# Load environment from project root
project_root = Path(__file__).parent.parent.parent
env_path = project_root / ".env"
if env_path.exists():
    load_dotenv(env_path)

from neo_memory_hub import NeoMemoryConnector, MemoryConfig
from neo_memory_hub.integrations.connector import ConnectorConfig


# =============================================================================
# Skip conditions
# =============================================================================

def has_openai_key() -> bool:
    """Check if OpenAI API key is available."""
    return bool(os.getenv("OPENAI_API_KEY"))


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


requires_openai = pytest.mark.skipif(
    not has_openai_key(),
    reason="OPENAI_API_KEY not set"
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
    qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6335")
    
    # Check if Qdrant is available
    if not qdrant_available():
        pytest.skip("Qdrant not available - start Docker with: docker-compose up -d")
    
    return MemoryConfig(
        vector_store={
            "provider": "qdrant",
            "config": {
                "collection_name": f"e2e_test_{uuid.uuid4().hex[:8]}",
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


@pytest.fixture
def connector_config():
    """Create connector configuration."""
    return ConnectorConfig(
        log_operations=True,
        default_limit=10,
    )


# =============================================================================
# E2E Tests: NeoMemoryConnector
# =============================================================================

@requires_openai
class TestNeoMemoryConnectorE2E:
    """End-to-end tests for NeoMemoryConnector."""

    @pytest.mark.asyncio
    async def test_memory_lifecycle(self, memory_config, connector_config, unique_id):
        """Test complete memory lifecycle: add → search → update → delete."""
        print("\n" + "=" * 60)
        print("MEMORY LIFECYCLE TEST")
        print("=" * 60)
        
        async with NeoMemoryConnector(
            memory_config=memory_config,
            config=connector_config,
        ) as connector:
            user_id = f"test_user_{unique_id}"
            
            # 1. ADD
            print("\n[1] Adding memory...")
            add_result = await connector.add(
                f"[{unique_id}] I love programming in Python and building AI systems",
                user_id=user_id,
                metadata={"test_id": unique_id},
            )
            print(f"    Result: {add_result}")
            assert add_result is not None
            
            # Wait for indexing
            await asyncio.sleep(1)
            
            # 2. SEARCH
            print("\n[2] Searching memories...")
            search_result = await connector.search(
                "What programming language does the user like?",
                user_id=user_id,
                limit=5,
            )
            print(f"    Found: {search_result}")
            
            results = search_result.get("results", [])
            assert len(results) > 0, "Should find at least one memory"
            
            # Verify Python is mentioned
            memory_text = " ".join(r.get("memory", "") for r in results).lower()
            assert "python" in memory_text, "Should find Python-related memory"
            
            # 3. GET ALL
            print("\n[3] Listing all memories...")
            all_memories = await connector.get_all(user_id=user_id)
            all_list = all_memories.get("results", [])
            print(f"    Total: {len(all_list)} memories")
            assert len(all_list) > 0
            
            # 4. UPDATE (if memory ID available)
            if results:
                memory_id = results[0].get("id")
                if memory_id:
                    print(f"\n[4] Updating memory {memory_id}...")
                    update_result = await connector.update(
                        memory_id,
                        "I love programming in Python, Rust, and Go",
                    )
                    print(f"    Update result: {update_result}")
            
            # 5. DELETE
            if results:
                memory_id = results[0].get("id")
                if memory_id:
                    print(f"\n[5] Deleting memory {memory_id}...")
                    delete_result = await connector.delete(memory_id)
                    print(f"    Delete result: {delete_result}")
            
            # 6. Operation logs
            print("\n[6] Checking operation logs...")
            logs = connector.get_operation_logs()
            print(f"    Total operations logged: {len(logs)}")
            for log in logs:
                print(f"    - {log['operation']}: success={log['success']}")
        
        print("\n✅ Memory lifecycle test passed!")

    @pytest.mark.asyncio
    async def test_user_isolation(self, memory_config, connector_config, unique_id):
        """Test that memories are isolated between users."""
        print("\n" + "=" * 60)
        print("USER ISOLATION TEST")
        print("=" * 60)
        
        async with NeoMemoryConnector(
            memory_config=memory_config,
            config=connector_config,
        ) as connector:
            user_a = f"user_a_{unique_id}"
            user_b = f"user_b_{unique_id}"
            
            # Store memory for User A
            print("\n[1] Storing memory for User A...")
            await connector.add(
                f"[{unique_id}] User A's secret: I prefer vim",
                user_id=user_a,
            )
            
            # Store memory for User B
            print("[2] Storing memory for User B...")
            await connector.add(
                f"[{unique_id}] User B's secret: I prefer emacs",
                user_id=user_b,
            )
            
            await asyncio.sleep(1)
            
            # User A searches - should NOT see User B's data
            print("\n[3] User A searching for editor preferences...")
            results_a = await connector.search(
                "What editor is preferred?",
                user_id=user_a,
                limit=10,
            )
            
            for result in results_a.get("results", []):
                memory = result.get("memory", "").lower()
                assert "emacs" not in memory, f"User A should NOT see User B's data! Found: {memory}"
            
            print("    ✅ User A cannot see User B's data")
            
            # User B searches - should NOT see User A's data
            print("[4] User B searching for editor preferences...")
            results_b = await connector.search(
                "What editor is preferred?",
                user_id=user_b,
                limit=10,
            )
            
            for result in results_b.get("results", []):
                memory = result.get("memory", "").lower()
                assert "vim" not in memory, f"User B should NOT see User A's data! Found: {memory}"
            
            print("    ✅ User B cannot see User A's data")
        
        print("\n✅ User isolation test passed!")

    @pytest.mark.asyncio
    async def test_build_context(self, memory_config, connector_config, unique_id):
        """Test building context from memories."""
        print("\n" + "=" * 60)
        print("BUILD CONTEXT TEST")
        print("=" * 60)
        
        async with NeoMemoryConnector(
            memory_config=memory_config,
            config=connector_config,
        ) as connector:
            user_id = f"context_user_{unique_id}"
            
            # Store some facts
            print("\n[1] Storing facts...")
            await connector.store_fact(
                "User's favorite programming language is Python",
                user_id=user_id,
            )
            await connector.store_preference(
                "User prefers dark mode in all applications",
                user_id=user_id,
            )
            
            await asyncio.sleep(1)
            
            # Build context
            print("\n[2] Building context...")
            context = await connector.build_context(
                "programming preferences",
                user_id=user_id,
                limit=5,
                include_scores=True,
            )
            
            print(f"    Context:\n{context}")
            assert len(context) > 0, "Should build non-empty context"
            assert "Relevant information" in context
        
        print("\n✅ Build context test passed!")

    @pytest.mark.asyncio
    async def test_store_exchange(self, memory_config, connector_config, unique_id):
        """Test storing conversation exchanges."""
        print("\n" + "=" * 60)
        print("STORE EXCHANGE TEST")
        print("=" * 60)
        
        async with NeoMemoryConnector(
            memory_config=memory_config,
            config=connector_config,
        ) as connector:
            user_id = f"exchange_user_{unique_id}"
            
            # Store an exchange
            print("\n[1] Storing conversation exchange...")
            result = await connector.store_exchange(
                user_message=f"[{unique_id}] Hi! I'm a data scientist working on ML projects.",
                assistant_response="That's great! Machine learning is a fascinating field.",
                user_id=user_id,
            )
            print(f"    Result: {result}")
            
            await asyncio.sleep(1)
            
            # Search for extracted facts
            print("\n[2] Searching for extracted facts...")
            search_result = await connector.search(
                "What does the user work on?",
                user_id=user_id,
            )
            
            results = search_result.get("results", [])
            print(f"    Found {len(results)} memories")
            
            for r in results:
                print(f"    - {r.get('memory', '')[:80]}...")
        
        print("\n✅ Store exchange test passed!")

    @pytest.mark.asyncio
    async def test_delete_all_for_user(self, memory_config, connector_config, unique_id):
        """Test deleting all memories for a user."""
        print("\n" + "=" * 60)
        print("DELETE ALL TEST")
        print("=" * 60)
        
        async with NeoMemoryConnector(
            memory_config=memory_config,
            config=connector_config,
        ) as connector:
            user_id = f"delete_all_user_{unique_id}"
            
            # Store multiple memories
            print("\n[1] Storing multiple memories...")
            for i in range(3):
                await connector.add(
                    f"[{unique_id}] Test fact number {i}",
                    user_id=user_id,
                )
            
            await asyncio.sleep(1)
            
            # Verify memories exist
            all_before = await connector.get_all(user_id=user_id)
            count_before = len(all_before.get("results", []))
            print(f"    Memories before delete: {count_before}")
            assert count_before > 0
            
            # Delete all
            print("\n[2] Deleting all memories...")
            delete_result = await connector.delete_all(user_id=user_id)
            print(f"    Delete result: {delete_result}")
            
            # Verify deletion
            await asyncio.sleep(1)
            all_after = await connector.get_all(user_id=user_id)
            count_after = len(all_after.get("results", []))
            print(f"    Memories after delete: {count_after}")
            assert count_after == 0, "All memories should be deleted"
        
        print("\n✅ Delete all test passed!")


# =============================================================================
# Main runner
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
