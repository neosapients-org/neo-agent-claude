"""Abstract base class for memory storage engines.

Defines the contract that all memory engines must implement.
Current implementation: Mem0 (via AsyncMemory).
Future: Zep, LangMem, custom engines.

All upper layers (extraction, dedup, retrieval, formatting, buffering)
are engine-agnostic — they call through NeoMemoryConnector which
delegates to this interface.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class MemoryEngine(ABC):
    """Abstract interface for the underlying memory storage engine.

    Implementations must provide async lifecycle and CRUD operations.
    The engine handles vector storage, embedding, and basic memory
    management. All higher-level logic (extraction, dedup, salience,
    retrieval strategies) lives in the layers above.

    Implementations:
        - AsyncMemory (current — wraps mem0ai)

    Example::

        class Mem0Engine(MemoryEngine):
            async def initialize(self) -> None:
                from mem0 import AsyncMemory
                import inspect as _inspect
                result = AsyncMemory.from_config(config)
                self._mem0 = (await result) if _inspect.isawaitable(result) else result

            async def add(self, messages, **kwargs) -> dict:
                return await self._mem0.add(messages, **kwargs)
    """

    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the engine (connect to vector DB, load models, etc.)."""
        ...

    @abstractmethod
    async def add(
        self,
        messages: str | list[dict[str, str]] | list[str],
        *,
        user_id: str | None = None,
        agent_id: str | None = None,
        run_id: str | None = None,
        metadata: dict[str, Any] | None = None,
        infer: bool = True,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Add memories from messages.

        Args:
            messages: Text or list of message dicts to store.
            user_id: User identifier for scoping.
            agent_id: Agent identifier for scoping.
            run_id: Run/session identifier for scoping.
            metadata: Additional metadata to attach.
            infer: Whether to use LLM to extract facts.

        Returns:
            Dict with 'results' key containing stored memory entries.
        """
        ...

    @abstractmethod
    async def search(
        self,
        query: str,
        *,
        user_id: str | None = None,
        agent_id: str | None = None,
        run_id: str | None = None,
        limit: int | None = None,
        filters: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Search memories by semantic similarity.

        Args:
            query: Search query text.
            user_id: Scope to this user.
            agent_id: Scope to this agent.
            run_id: Scope to this run/session.
            limit: Maximum results.
            filters: Metadata filters.

        Returns:
            Dict with 'results' key containing matched memories.
        """
        ...

    @abstractmethod
    async def get(self, memory_id: str) -> dict[str, Any] | None:
        """Get a single memory by ID.

        Returns:
            Memory dict or None if not found.
        """
        ...

    @abstractmethod
    async def get_all(
        self,
        *,
        user_id: str | None = None,
        agent_id: str | None = None,
        run_id: str | None = None,
        limit: int = 100,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Get all memories for a scope.

        Returns:
            Dict with 'results' key containing all memories.
        """
        ...

    @abstractmethod
    async def update(self, memory_id: str, data: str) -> dict[str, Any]:
        """Update a memory's content.

        Returns:
            Dict with update result.
        """
        ...

    @abstractmethod
    async def delete(self, memory_id: str) -> dict[str, Any]:
        """Delete a memory by ID.

        Returns:
            Dict with deletion result.
        """
        ...

    @abstractmethod
    async def close(self) -> None:
        """Clean up resources (close connections, etc.)."""
        ...
