"""
Neo Memory Hub - Universal Memory Connector.

This is the **primary public API** for Neo Memory Hub.
It provides full Mem0 parity plus Neo-specific value-adds.

Architecture:
    Agentic Framework → NeoMemoryConnector → Mem0 AsyncMemory → Storage

Features:
- Full Mem0 API parity (add, search, get, get_all, update, delete, delete_all, reset)
- Context building for LLM prompts
- Convenience methods (store_exchange, store_fact, store_preference)
- Operation logging for debugging
- Optional noise filtering via StorageGateway

Usage:
    >>> from neo_memory_hub import NeoMemoryConnector
    >>>
    >>> # Initialize
    >>> connector = NeoMemoryConnector()
    >>> await connector.initialize()
    >>>
    >>> # Full Mem0 parity
    >>> await connector.add("User likes dark mode", user_id="alex")
    >>> results = await connector.search("preferences", user_id="alex")
    >>> memory = await connector.get(memory_id)
    >>> all_mems = await connector.get_all(user_id="alex")
    >>> await connector.update(memory_id, "Updated content")
    >>> await connector.delete(memory_id)
    >>>
    >>> # Neo value-adds
    >>> context = await connector.build_context("query", user_id="alex")
    >>> await connector.store_exchange(user_msg, assistant_msg, user_id="alex")
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Any, Awaitable, Callable, TypedDict

from neo_memory_hub.domain.scope import IsolationScope
from neo_memory_hub.integrations.memory import AsyncMemory, MemoryConfig


logger = logging.getLogger(__name__)


# =============================================================================
# Types
# =============================================================================


class MemoryItem(TypedDict, total=False):
    """A single memory item returned by connector."""

    id: str
    memory: str
    score: float
    created_at: str
    updated_at: str
    metadata: dict[str, Any]
    # Graph relations (if graph memory enabled)
    relations: list[dict[str, Any]]


class OperationLog(TypedDict):
    """Log entry for memory operations."""

    operation: str
    timestamp: str
    request_id: str
    user_id: str | None
    agent_id: str | None
    run_id: str | None
    details: dict[str, Any]
    success: bool
    error: str | None


# Callback type for operation hooks (CR-9)
OperationCallback = Callable[[OperationLog], Awaitable[None] | None]


@dataclass
class ConnectorConfig:
    """Configuration for NeoMemoryConnector."""

    # Retrieval settings
    default_limit: int = 10
    default_threshold: float | None = None  # None = no threshold

    # Storage settings
    use_storage_validation: bool = True  # Validate content before storing

    # Summary layer: when True, fact reads (search/get_all) are filtered to
    # type="fact" so the per-user summary point never leaks into fact results.
    # Only safe once existing points have been backfilled with a type — the
    # high-level connector sets this in lockstep with running that backfill.
    exclude_non_fact_points: bool = False

    # Context settings
    max_context_memories: int = 5
    context_prefix: str = "Relevant information from memory:"

    # Logging
    log_operations: bool = True
    max_operation_logs: int = 100

    # Operation hook (v0.2) — called after every operation completes
    on_operation: OperationCallback | None = None


# =============================================================================
# NeoMemoryConnector - Primary Public API
# =============================================================================


@dataclass
class NeoMemoryConnector:
    """
    Universal Memory Connector for Neo Memory Hub.

    This is the **single public API** for all memory operations.
    It provides full Mem0 parity plus Neo-specific value-adds.

    Usage:
        >>> connector = NeoMemoryConnector()
        >>> await connector.initialize()
        >>>
        >>> # Store memories
        >>> await connector.add("User prefers dark mode", user_id="alex")
        >>>
        >>> # Search memories
        >>> results = await connector.search("preferences", user_id="alex")
        >>>
        >>> # Build context for LLM
        >>> context = await connector.build_context("What are my preferences?", user_id="alex")
    """

    # Configuration
    memory_config: MemoryConfig | None = None
    config: ConnectorConfig = field(default_factory=ConnectorConfig)

    # Internal state
    _memory: AsyncMemory | None = field(default=None, repr=False)
    _initialized: bool = field(default=False, repr=False)
    _operation_logs: list[OperationLog] = field(default_factory=list, repr=False)
    _request_counter: int = field(default=0, repr=False)

    # Optional storage gateway for validation
    _storage_gateway: Any = field(default=None, repr=False)

    # =========================================================================
    # Lifecycle
    # =========================================================================

    async def initialize(self) -> None:
        """
        Initialize the connector.

        Must be called before any operations.
        """
        if self._initialized:
            return

        self._memory = AsyncMemory(config=self.memory_config)
        await self._memory.initialize()

        # Initialize storage gateway for validation
        if self.config.use_storage_validation:
            try:
                from neo_memory_hub.core.storage_gateway import StorageGateway

                self._storage_gateway = StorageGateway()
            except ImportError:
                logger.debug("StorageGateway not available, skipping validation")

        self._initialized = True
        logger.info("NeoMemoryConnector initialized")

    async def close(self) -> None:
        """Close the connector."""
        if self._memory:
            await self._memory.close()
        self._memory = None
        self._initialized = False

    async def __aenter__(self) -> NeoMemoryConnector:
        """Async context manager entry."""
        await self.initialize()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Async context manager exit."""
        await self.close()

    def _ensure_initialized(self) -> None:
        """Ensure connector is initialized."""
        if not self._initialized or self._memory is None:
            raise RuntimeError(
                "NeoMemoryConnector not initialized. Call await initialize() first "
                "or use async context manager: async with NeoMemoryConnector() as connector:"
            )

    # =========================================================================
    # CORE MEM0 OPERATIONS (Full Parity)
    # =========================================================================

    async def add(
        self,
        messages: str | list[dict[str, str]] | list[str],
        *,
        user_id: str | None = None,
        agent_id: str | None = None,
        run_id: str | None = None,
        metadata: dict[str, Any] | None = None,
        filters: dict[str, Any] | None = None,
        categories: list[str] | None = None,
        infer: bool = True,
        prompt: str | None = None,
        llm: dict[str, Any] | None = None,
        memory_type: str | None = None,
    ) -> dict[str, Any]:
        """
        Add memories with optional LLM fact extraction.

        This is the primary method for storing memories. When infer=True (default),
        Mem0's LLM will extract facts from the content. When infer=False, content
        is stored as-is without processing.

        Args:
            messages: Content to store. Can be:
                - A string: "User likes dark mode"
                - A list of message dicts: [{"role": "user", "content": "..."}]
                - A list of strings: ["fact 1", "fact 2"]
            user_id: User identifier for memory isolation
            agent_id: Agent identifier for agent-specific memories
            run_id: Run/session identifier
            metadata: Additional metadata to store
            filters: Additional filters
            categories: Categories to tag the memory
            infer: Whether to use LLM for fact extraction (default: True)
            prompt: Per-call prompt override for fact extraction
            llm: Per-call LLM override (provider/config)
            memory_type: Type of memory (stored in metadata)

        Returns:
            Dict with "results" key containing added memory info:
            {
                "results": [
                    {"id": "mem_123", "memory": "User likes dark mode", "event": "ADD"}
                ]
            }

        Example:
            >>> await connector.add("User prefers dark mode", user_id="alex")
            >>> await connector.add([
            ...     {"role": "user", "content": "I love hiking"},
            ...     {"role": "assistant", "content": "Great! I'll remember that."}
            ... ], user_id="alex")
        """
        self._ensure_initialized()
        request_id = self._generate_request_id()

        self._log_operation(
            operation="add",
            request_id=request_id,
            user_id=user_id,
            agent_id=agent_id,
            run_id=run_id,
            details={
                "infer": infer,
                "has_categories": bool(categories),
                "has_metadata": bool(metadata),
            },
        )

        try:
            result = await self._memory.add(
                messages=messages,
                user_id=user_id,
                agent_id=agent_id,
                run_id=run_id,
                metadata=metadata,
                filters=filters,
                categories=categories,
                infer=infer,
                prompt=prompt,
                llm=llm,
                memory_type=memory_type,
            )

            self._update_log(
                request_id,
                success=True,
                details={"results_count": len(result.get("results", []))},
            )

            return result

        except Exception as e:
            self._update_log(request_id, success=False, error=str(e))
            logger.error(f"add() failed: {e}")
            raise

    async def search(
        self,
        query: str,
        *,
        user_id: str | None = None,
        agent_id: str | None = None,
        run_id: str | None = None,
        limit: int | None = None,
        filters: dict[str, Any] | None = None,
        metadata_filters: dict[str, Any] | None = None,
        categories: list[str] | None = None,
        threshold: float | None = None,
        rerank: bool = False,
    ) -> dict[str, Any]:
        """
        Search memories with optional reranking.

        Args:
            query: Search query
            user_id: Filter by user
            agent_id: Filter by agent
            run_id: Filter by run/session
            limit: Maximum results (default: config.default_limit)
            filters: Additional filters
            metadata_filters: Metadata-specific filters
            categories: Filter by categories (client-side post-filtering)
            threshold: Minimum similarity score (0.0-1.0)
            rerank: Whether to use reranker

        Returns:
            Dict with "results" and optionally "relations" keys:
            {
                "results": [{"id": "...", "memory": "...", "score": 0.85}],
                "relations": [...]  # If graph memory enabled
            }

        Example:
            >>> results = await connector.search("preferences", user_id="alex")
            >>> for mem in results["results"]:
            ...     print(f"{mem['memory']} (score: {mem['score']:.2f})")
        """
        self._ensure_initialized()
        request_id = self._generate_request_id()
        limit = limit or self.config.default_limit

        # Keep the per-user summary point out of fact search results.
        if self.config.exclude_non_fact_points:
            metadata_filters = dict(metadata_filters or {})
            metadata_filters.setdefault("type", "fact")

        self._log_operation(
            operation="search",
            request_id=request_id,
            user_id=user_id,
            agent_id=agent_id,
            run_id=run_id,
            details={"query": query[:100], "limit": limit, "rerank": rerank},
        )

        try:
            # Workaround: Mem0 treats threshold=0.0 as falsy, skipping all
            # results. Convert 0.0 to 0.001 so it behaves as "no threshold".
            effective_threshold = threshold if threshold is not None else self.config.default_threshold
            if effective_threshold == 0.0:
                effective_threshold = 0.001

            result = await self._memory.search(
                query=query,
                user_id=user_id,
                agent_id=agent_id,
                run_id=run_id,
                limit=limit,
                filters=filters,
                metadata_filters=metadata_filters,
                categories=categories,
                threshold=effective_threshold,
                rerank=rerank,
            )

            self._update_log(
                request_id,
                success=True,
                details={"results_count": len(result.get("results", []))},
            )

            return result

        except Exception as e:
            self._update_log(request_id, success=False, error=str(e))
            logger.error(f"search() failed: {e}")
            raise

    async def get(self, memory_id: str) -> dict[str, Any] | None:
        """
        Get a specific memory by ID.

        Args:
            memory_id: Memory ID to retrieve

        Returns:
            Memory dict or None if not found

        Example:
            >>> memory = await connector.get("mem_123")
            >>> if memory:
            ...     print(memory["memory"])
        """
        self._ensure_initialized()
        request_id = self._generate_request_id()

        self._log_operation(
            operation="get",
            request_id=request_id,
            user_id=None,
            agent_id=None,
            run_id=None,
            details={"memory_id": memory_id},
        )

        try:
            result = await self._memory.get(memory_id)
            self._update_log(request_id, success=True, details={"found": result is not None})
            return result

        except Exception as e:
            self._update_log(request_id, success=False, error=str(e))
            logger.error(f"get() failed: {e}")
            raise

    async def get_all(
        self,
        *,
        user_id: str | None = None,
        agent_id: str | None = None,
        run_id: str | None = None,
        limit: int = 100,
        filters: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Get all memories for a scope with pagination.

        Args:
            user_id: Filter by user
            agent_id: Filter by agent
            run_id: Filter by run/session
            limit: Maximum results
            filters: Additional filters

        Returns:
            Dict with "results" key containing all matching memories

        Example:
            >>> all_memories = await connector.get_all(user_id="alex")
            >>> print(f"Found {len(all_memories['results'])} memories")
        """
        self._ensure_initialized()
        request_id = self._generate_request_id()

        # Keep the per-user summary point out of fact listings.
        if self.config.exclude_non_fact_points:
            filters = dict(filters or {})
            filters.setdefault("type", "fact")

        self._log_operation(
            operation="get_all",
            request_id=request_id,
            user_id=user_id,
            agent_id=agent_id,
            run_id=run_id,
            details={"limit": limit},
        )

        try:
            result = await self._memory.get_all(
                user_id=user_id,
                agent_id=agent_id,
                run_id=run_id,
                limit=limit,
                filters=filters,
            )

            self._update_log(
                request_id,
                success=True,
                details={"results_count": len(result.get("results", []))},
            )

            return result

        except Exception as e:
            self._update_log(request_id, success=False, error=str(e))
            logger.error(f"get_all() failed: {e}")
            raise

    async def update(self, memory_id: str, data: str) -> dict[str, Any]:
        """
        Update a specific memory.

        Args:
            memory_id: ID of memory to update
            data: New content

        Returns:
            Update result

        Example:
            >>> result = await connector.update("mem_123", "Updated preference")
            >>> print(result["message"])
        """
        self._ensure_initialized()
        request_id = self._generate_request_id()

        self._log_operation(
            operation="update",
            request_id=request_id,
            user_id=None,
            agent_id=None,
            run_id=None,
            details={"memory_id": memory_id, "data_length": len(data)},
        )

        try:
            result = await self._memory.update(memory_id, data)
            self._update_log(request_id, success=True)
            return result

        except Exception as e:
            self._update_log(request_id, success=False, error=str(e))
            logger.error(f"update() failed: {e}")
            raise

    async def delete(self, memory_id: str) -> dict[str, Any]:
        """
        Delete a specific memory.

        Args:
            memory_id: ID of memory to delete

        Returns:
            Delete result

        Example:
            >>> result = await connector.delete("mem_123")
            >>> print(result["message"])
        """
        self._ensure_initialized()
        request_id = self._generate_request_id()

        self._log_operation(
            operation="delete",
            request_id=request_id,
            user_id=None,
            agent_id=None,
            run_id=None,
            details={"memory_id": memory_id},
        )

        try:
            result = await self._memory.delete(memory_id)
            self._update_log(request_id, success=True)
            return result

        except Exception as e:
            self._update_log(request_id, success=False, error=str(e))
            logger.error(f"delete() failed: {e}")
            raise

    async def delete_all(
        self,
        *,
        user_id: str | None = None,
        agent_id: str | None = None,
        run_id: str | None = None,
    ) -> dict[str, Any]:
        """
        Delete all memories for a scope.

        Args:
            user_id: Delete all for user
            agent_id: Delete all for agent
            run_id: Delete all for run/session

        Returns:
            Delete result

        Example:
            >>> result = await connector.delete_all(user_id="alex")
            >>> print(result["message"])
        """
        self._ensure_initialized()
        request_id = self._generate_request_id()

        self._log_operation(
            operation="delete_all",
            request_id=request_id,
            user_id=user_id,
            agent_id=agent_id,
            run_id=run_id,
            details={},
        )

        try:
            result = await self._memory.delete_all(
                user_id=user_id,
                agent_id=agent_id,
                run_id=run_id,
            )
            self._update_log(request_id, success=True)
            return result

        except Exception as e:
            self._update_log(request_id, success=False, error=str(e))
            logger.error(f"delete_all() failed: {e}")
            raise

    async def reset(self) -> None:
        """
        Reset/clear ALL memories.

        WARNING: This is destructive and cannot be undone.
        """
        self._ensure_initialized()
        request_id = self._generate_request_id()

        self._log_operation(
            operation="reset",
            request_id=request_id,
            user_id=None,
            agent_id=None,
            run_id=None,
            details={"warning": "Clearing all memories"},
        )

        try:
            # Mem0's reset() doesn't exist on AsyncMemory, use delete_all with no filters
            # Actually, let's check if the sync version has it and call appropriately
            if hasattr(self._memory, "_mem0") and hasattr(self._memory._mem0, "reset"):
                await self._memory._mem0.reset()
            else:
                # Fallback: delete_all without scope deletes everything
                await self._memory.delete_all()

            self._update_log(request_id, success=True)

        except Exception as e:
            self._update_log(request_id, success=False, error=str(e))
            logger.error(f"reset() failed: {e}")
            raise

    # =========================================================================
    # HELPER METHODS
    # =========================================================================

    def format_memories(
        self,
        search_results: dict[str, Any],
        *,
        include_scores: bool = False,
        include_relations: bool = False,
        prefix: str | None = None,
    ) -> str:
        """
        Format search results as a context string for LLM prompts.

        Args:
            search_results: Results from search()
            include_scores: Include relevance scores
            include_relations: Include graph relations
            prefix: Custom prefix (default: config.context_prefix)

        Returns:
            Formatted string for system prompt injection

        Example:
            >>> results = await connector.search("preferences", user_id="alex")
            >>> context = connector.format_memories(results)
            >>> # Use in LLM prompt
            >>> response = llm.invoke([SystemMessage(content=context)] + messages)
        """
        results = search_results.get("results", [])
        if not results:
            return ""

        prefix = prefix or self.config.context_prefix
        lines = [prefix]

        for mem in results:
            memory_text = mem.get("memory", "")
            if not memory_text:
                continue

            if include_scores:
                score = mem.get("score", 0.0)
                lines.append(f"- {memory_text} (relevance: {score:.2f})")
            else:
                lines.append(f"- {memory_text}")

        # Add relations if requested and available
        if include_relations:
            relations = search_results.get("relations", [])
            if relations:
                lines.append("\nRelated entities:")
                for rel in relations[:5]:  # Limit to top 5 relations
                    if isinstance(rel, dict):
                        source = rel.get("source", "")
                        target = rel.get("target", "")
                        relation = rel.get("relation", "related_to")
                        if source and target:
                            lines.append(f"- {source} → {relation} → {target}")

        return "\n".join(lines)

    # =========================================================================
    # NEO VALUE-ADD OPERATIONS
    # =========================================================================

    async def build_context(
        self,
        query: str,
        *,
        user_id: str | None = None,
        agent_id: str | None = None,
        run_id: str | None = None,
        limit: int | None = None,
        include_scores: bool = False,
        include_relations: bool = False,
    ) -> str:
        """
        Build LLM-ready context from memories.

        This is a convenience method that combines search + format.

        Args:
            query: Search query
            user_id: Filter by user
            agent_id: Filter by agent
            run_id: Filter by run/session
            limit: Maximum memories (default: config.max_context_memories)
            include_scores: Include relevance scores in output
            include_relations: Include graph relations

        Returns:
            Formatted context string for LLM system prompt

        Example:
            >>> context = await connector.build_context("preferences", user_id="alex")
            >>> response = llm.invoke([SystemMessage(content=context), user_message])
        """
        limit = limit or self.config.max_context_memories

        results = await self.search(
            query=query,
            user_id=user_id,
            agent_id=agent_id,
            run_id=run_id,
            limit=limit,
        )

        return self.format_memories(
            results,
            include_scores=include_scores,
            include_relations=include_relations,
        )

    async def store_exchange(
        self,
        user_message: str,
        assistant_response: str,
        *,
        user_id: str | None = None,
        agent_id: str | None = None,
        run_id: str | None = None,
        infer: bool = True,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Store a user-assistant exchange.

        This is a convenience method for storing conversation pairs.

        Args:
            user_message: User's message
            assistant_response: Assistant's response
            user_id: User identifier
            agent_id: Agent identifier
            run_id: Run/session identifier
            infer: Whether to use LLM for fact extraction
            metadata: Additional metadata

        Returns:
            Add result

        Example:
            >>> await connector.store_exchange(
            ...     "What's my favorite color?",
            ...     "Based on our conversation, you prefer blue.",
            ...     user_id="alex"
            ... )
        """
        messages = [
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": assistant_response},
        ]

        return await self.add(
            messages=messages,
            user_id=user_id,
            agent_id=agent_id,
            run_id=run_id,
            infer=infer,
            metadata=metadata,
        )

    async def store_fact(
        self,
        fact: str,
        *,
        user_id: str | None = None,
        agent_id: str | None = None,
        categories: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Store a specific fact with raw storage (no LLM processing).

        Use this when you have an already-extracted fact.

        Args:
            fact: The fact to store
            user_id: User identifier
            agent_id: Agent identifier
            categories: Categories to tag the fact
            metadata: Additional metadata

        Returns:
            Add result

        Example:
            >>> await connector.store_fact(
            ...     "User's favorite programming language is Python",
            ...     user_id="alex",
            ...     categories=["preferences", "technical"]
            ... )
        """
        full_metadata = dict(metadata or {})

        return await self.add(
            messages=fact,
            user_id=user_id,
            agent_id=agent_id,
            categories=categories,
            metadata=full_metadata,
            infer=False,  # Raw storage - fact is already extracted
        )

    async def store_preference(
        self,
        preference: str,
        *,
        user_id: str | None = None,
        agent_id: str | None = None,
        categories: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Store a user preference.

        Convenience wrapper with appropriate categorization.

        Args:
            preference: The preference to store
            user_id: User identifier
            categories: Additional categories
            metadata: Additional metadata

        Returns:
            Add result

        Example:
            >>> await connector.store_preference(
            ...     "User prefers dark mode interfaces",
            ...     user_id="alex"
            ... )
        """
        all_categories = ["preference"]
        if categories:
            all_categories.extend(categories)

        full_metadata = {"preference": True, **(metadata or {})}

        return await self.add(
            messages=preference,
            user_id=user_id,
            agent_id=agent_id,
            categories=all_categories,
            metadata=full_metadata,
            infer=False,  # Preferences are explicit
        )

    # =========================================================================
    # OPERATION LOGGING (for debugging/observability)
    # =========================================================================

    def _generate_request_id(self) -> str:
        """Generate unique request ID."""
        self._request_counter += 1
        return f"req_{self._request_counter}_{uuid.uuid4().hex[:8]}"

    def _log_operation(
        self,
        operation: str,
        request_id: str,
        user_id: str | None,
        agent_id: str | None,
        run_id: str | None,
        details: dict[str, Any],
    ) -> None:
        """Log an operation."""
        if not self.config.log_operations:
            return

        log_entry: OperationLog = {
            "operation": operation,
            "timestamp": datetime.now(UTC).isoformat(),
            "request_id": request_id,
            "user_id": user_id,
            "agent_id": agent_id,
            "run_id": run_id,
            "details": details,
            "success": False,
            "error": None,
        }
        self._operation_logs.append(log_entry)

        # Keep only last N logs
        if len(self._operation_logs) > self.config.max_operation_logs:
            self._operation_logs = self._operation_logs[-self.config.max_operation_logs :]

    def _update_log(
        self,
        request_id: str,
        success: bool,
        details: dict[str, Any] | None = None,
        error: str | None = None,
    ) -> None:
        """Update log entry with result and invoke callback hook."""
        for log in reversed(self._operation_logs):
            if log["request_id"] == request_id:
                log["success"] = success
                log["error"] = error
                if details:
                    log["details"].update(details)

                # Invoke operation callback (CR-9)
                if self.config.on_operation:
                    try:
                        result = self.config.on_operation(log)
                        if asyncio.iscoroutine(result):
                            asyncio.ensure_future(result)
                    except Exception as cb_err:
                        logger.warning(f"Operation callback failed: {cb_err}")
                break

    def get_operation_logs(
        self,
        limit: int = 20,
        operation: str | None = None,
    ) -> list[OperationLog]:
        """
        Get recent operation logs for debugging.

        Args:
            limit: Maximum logs to return
            operation: Filter by operation type

        Returns:
            List of operation logs (newest first)
        """
        logs = self._operation_logs.copy()

        if operation:
            logs = [log for log in logs if log["operation"] == operation]

        return list(reversed(logs))[:limit]

    def clear_operation_logs(self) -> None:
        """Clear all operation logs."""
        self._operation_logs.clear()

    # =========================================================================
    # LIFECYCLE OPERATIONS (v0.2)
    # =========================================================================

    async def cleanup_expired(
        self,
        *,
        user_id: str | None = None,
        scope: IsolationScope | None = None,
        dry_run: bool = False,
        batch_size: int = 100,
        salience_threshold: float | None = None,
    ) -> dict[str, Any]:
        """
        Clean up expired memories based on TTL metadata.

        Scans memories and deletes those with `expires_at` in the past.
        Optionally also cleans low-salience memories.

        Args:
            user_id: Filter to a specific user (optional)
            scope: Filter to a specific scope (optional, overrides user_id)
            dry_run: If True, returns preview without deleting
            batch_size: Maximum memories to scan per batch
            salience_threshold: Also delete memories with salience below this

        Returns:
            Dict with cleanup results:
            {
                "scanned": N,
                "expired": N,
                "low_salience": N,
                "deleted_ids": [...],
                "dry_run": bool,
            }
        """
        self._ensure_initialized()

        # Build get_all params
        kwargs: dict[str, Any] = {"limit": batch_size}
        if scope is not None:
            params = scope.to_mem0_params()
            kwargs["user_id"] = params.get("user_id")
            kwargs["agent_id"] = params.get("agent_id")
            kwargs["run_id"] = params.get("run_id")
            kwargs["filters"] = scope.to_mem0_filters()
        elif user_id is not None:
            kwargs["user_id"] = user_id

        try:
            all_memories = await self.get_all(**kwargs)
        except Exception:
            # mem0 requires at least one identity param — if none provided, return empty
            if user_id is None and scope is None:
                return {
                    "scanned": 0, "expired": 0, "low_salience": 0,
                    "deleted_ids": [], "dry_run": dry_run,
                }
            raise
        results_list = all_memories.get("results", [])

        now = datetime.now(UTC)
        expired_ids: list[str] = []
        low_salience_ids: list[str] = []

        for mem in results_list:
            mem_id = mem.get("id")
            if not mem_id:
                continue

            meta = mem.get("metadata") or {}

            # Check expiry
            expires_at_str = meta.get("expires_at")
            if expires_at_str:
                try:
                    expires_at = datetime.fromisoformat(expires_at_str)
                    if expires_at < now:
                        expired_ids.append(mem_id)
                        continue
                except (ValueError, TypeError):
                    pass

            # Check low salience
            if salience_threshold is not None:
                salience = meta.get("salience")
                if salience is not None and float(salience) < salience_threshold:
                    low_salience_ids.append(mem_id)

        all_to_delete = expired_ids + low_salience_ids

        if not dry_run:
            for mem_id in all_to_delete:
                try:
                    await self.delete(mem_id)
                except Exception as e:
                    logger.warning(f"Failed to delete memory {mem_id}: {e}")

        return {
            "scanned": len(results_list),
            "expired": len(expired_ids),
            "low_salience": len(low_salience_ids),
            "deleted_ids": all_to_delete,
            "dry_run": dry_run,
        }

    async def count_memories(
        self,
        *,
        user_id: str | None = None,
        scope: IsolationScope | None = None,
        group_by_subject: bool = False,
    ) -> dict[str, Any]:
        """
        Count memories for a scope, optionally grouped by subject_id.

        Useful for triggering memory consolidation when counts get high.

        Args:
            user_id: Filter to a specific user
            scope: Filter to a specific scope (overrides user_id)
            group_by_subject: Group counts by subject_id metadata

        Returns:
            Dict: {"total": N} or {"total": N, "by_subject": {"CUST_99": 12, ...}}
        """
        self._ensure_initialized()

        kwargs: dict[str, Any] = {"limit": 1000}
        if scope is not None:
            params = scope.to_mem0_params()
            kwargs["user_id"] = params.get("user_id")
            kwargs["agent_id"] = params.get("agent_id")
            kwargs["run_id"] = params.get("run_id")
            kwargs["filters"] = scope.to_mem0_filters()
        elif user_id is not None:
            kwargs["user_id"] = user_id

        try:
            all_memories = await self.get_all(**kwargs)
        except Exception:
            # mem0 requires at least one identity param — if none provided, return 0
            if user_id is None and scope is None:
                return {"total": 0}
            raise
        results_list = all_memories.get("results", [])
        total = len(results_list)

        result: dict[str, Any] = {"total": total}

        if group_by_subject:
            by_subject: dict[str, int] = {}
            for mem in results_list:
                subject = mem.get("metadata", {}).get("subject_id", "_unknown")
                by_subject[subject] = by_subject.get(subject, 0) + 1
            result["by_subject"] = by_subject

        return result


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    "ConnectorConfig",
    "MemoryItem",
    "NeoMemoryConnector",
    "OperationCallback",
    "OperationLog",
]
