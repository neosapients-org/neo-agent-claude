"""Test configuration and fixtures."""

# Fix protobuf conflicts - MUST be before any other imports
import os
import sys

# Load environment variables first
from dotenv import load_dotenv

load_dotenv()

import asyncio
from collections.abc import Generator
from unittest.mock import AsyncMock, MagicMock

import pytest

from neo_memory_hub.config.settings import Settings
from neo_memory_hub.domain.memory import MemoryEntry, MemoryResult
from neo_memory_hub.domain.scope import IsolationScope, AccessTier
from neo_memory_hub.domain.types import MemoryType


@pytest.fixture(scope="session")
def event_loop() -> Generator[asyncio.AbstractEventLoop, None, None]:
    """Create event loop for async tests."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def settings() -> Settings:
    """Create test settings."""
    return Settings()


@pytest.fixture
def sample_scope() -> IsolationScope:
    """Create a sample isolation scope."""
    return IsolationScope(
        tenant_id="test_tenant",
        user_id="test_user",
        agent_id="test_agent",
        pool=AccessTier.SHARED,
    )


@pytest.fixture
def sample_scope_minimal() -> IsolationScope:
    """Create a minimal isolation scope (tenant only)."""
    return IsolationScope(
        tenant_id="test_tenant",
        pool=AccessTier.SYSTEM,
    )


@pytest.fixture
def sample_memory_entry(sample_scope: IsolationScope) -> MemoryEntry:
    """Create a sample memory entry."""
    return MemoryEntry(
        content="This is a test memory about user preferences.",
        memory_type=MemoryType.PERSONA,
        scope=sample_scope,
        importance=7.5,
        metadata={"source": "test"},
    )


@pytest.fixture
def sample_memory_result(sample_memory_entry: MemoryEntry) -> MemoryResult:
    """Create a sample memory result."""
    return MemoryResult.from_entry(
        entry=sample_memory_entry,
        relevance_score=0.85,
        token_count=20,
        retrieval_method="vector",
    )


@pytest.fixture
def mock_mem0_memory() -> AsyncMock:
    """Create a mock Mem0 AsyncMemory instance."""
    memory = AsyncMock()

    # Configure default return values
    memory.add.return_value = {"results": [{"id": "test-memory-id"}]}
    memory.search.return_value = {"results": []}
    memory.get.return_value = None
    memory.get_all.return_value = {"results": []}
    memory.update.return_value = {"success": True}
    memory.delete.return_value = {"success": True}

    return memory


# Markers for test categories
def pytest_configure(config: pytest.Config) -> None:
    """Configure pytest markers."""
    config.addinivalue_line("markers", "unit: Unit tests (fast, no external dependencies)")
    config.addinivalue_line(
        "markers", "integration: Integration tests (requires external services)"
    )
    config.addinivalue_line("markers", "e2e: End-to-end tests")
