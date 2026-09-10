"""Scope-aware memory connector wrapping NeoMemoryConnector.

Adds enterprise scope isolation (add_scoped, search_scoped, etc.)
and SharedMemoryStrategy write-guards on top of the core connector.
"""

from __future__ import annotations

import logging
from typing import Any

from neo_memory_hub.domain.scope import AccessTier, IsolationScope
from neo_memory_hub.integrations.connector import NeoMemoryConnector

from memory_utils.shared_scope.strategy import SharedMemoryStrategy

logger = logging.getLogger(__name__)


class ScopedMemoryConnector:
    """Wraps NeoMemoryConnector with scope-aware operations.

    Usage:
        >>> from memory_utils.shared_scope import ScopedMemoryConnector
        >>> connector = NeoMemoryConnector()
        >>> scoped = ScopedMemoryConnector(connector)
        >>> await scoped.add_scoped("fact", scope)
    """

    def __init__(
        self,
        connector: NeoMemoryConnector,
        strategy: SharedMemoryStrategy = SharedMemoryStrategy.ENABLED,
    ) -> None:
        self._connector = connector
        self._strategy = strategy

    @property
    def connector(self) -> NeoMemoryConnector:
        return self._connector

    @property
    def strategy(self) -> SharedMemoryStrategy:
        return self._strategy

    def _check_shared_write(
        self, scope: IsolationScope
    ) -> dict[str, Any] | None:
        """Check if writing to the given scope is allowed under SharedMemoryStrategy.

        Returns None if allowed, or a blocked-result dict if not.
        """
        if scope.pool in (AccessTier.TEAM, AccessTier.ORG, AccessTier.SYSTEM):
            if self._strategy in (SharedMemoryStrategy.DISABLED, SharedMemoryStrategy.READ_ONLY):
                logger.info(
                    f"Shared memory write blocked "
                    f"(strategy={self._strategy.value}, tier={scope.pool.value})"
                )
                return {
                    "results": [],
                    "blocked_by": "shared_memory_strategy",
                    "strategy": self._strategy.value,
                    "tier": scope.pool.value,
                }
        return None

    async def add_scoped(
        self,
        messages: str | list[dict[str, str]] | list[str],
        scope: IsolationScope,
        *,
        metadata: dict[str, Any] | None = None,
        categories: list[str] | None = None,
        infer: bool = True,
        prompt: str | None = None,
        llm: dict[str, Any] | None = None,
        memory_type: str | None = None,
    ) -> dict[str, Any]:
        """Add memories with enterprise scope isolation.

        Extracts Mem0 parameters from the scope and enforces
        SharedMemoryStrategy before delegating to connector.add().
        """
        blocked = self._check_shared_write(scope)
        if blocked is not None:
            return blocked

        params = scope.to_mem0_params()
        scope_meta = params.pop("metadata", {})
        merged_metadata = {**scope_meta, **(metadata or {})}

        return await self._connector.add(
            messages=messages,
            user_id=params.get("user_id"),
            agent_id=params.get("agent_id"),
            run_id=params.get("run_id"),
            metadata=merged_metadata,
            categories=categories,
            infer=infer,
            prompt=prompt,
            llm=llm,
            memory_type=memory_type,
        )

    async def search_scoped(
        self,
        query: str,
        scope: IsolationScope,
        *,
        limit: int | None = None,
        categories: list[str] | None = None,
        threshold: float | None = None,
        rerank: bool = False,
    ) -> dict[str, Any]:
        """Search memories with enterprise scope isolation.

        If SharedMemoryStrategy is DISABLED, only PRIVATE results are returned.
        """
        params = scope.to_mem0_params()
        filters = scope.to_mem0_filters()

        result = await self._connector.search(
            query=query,
            user_id=params.get("user_id"),
            agent_id=params.get("agent_id"),
            run_id=params.get("run_id"),
            limit=limit,
            filters=filters,
            categories=categories,
            threshold=threshold,
            rerank=rerank,
        )

        if self._strategy == SharedMemoryStrategy.DISABLED:
            results = result.get("results", [])
            result["results"] = [
                r
                for r in results
                if r.get("metadata", {}).get("pool") in (None, AccessTier.PRIVATE.value)
            ]

        return result

    async def get_all_scoped(
        self,
        scope: IsolationScope,
        *,
        limit: int = 100,
    ) -> dict[str, Any]:
        """Get all memories for an enterprise scope."""
        params = scope.to_mem0_params()
        filters = scope.to_mem0_filters()

        return await self._connector.get_all(
            user_id=params.get("user_id"),
            agent_id=params.get("agent_id"),
            run_id=params.get("run_id"),
            limit=limit,
            filters=filters,
        )

    async def store_exchange_scoped(
        self,
        user_message: str,
        assistant_response: str,
        scope: IsolationScope,
        *,
        infer: bool = True,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Store a user-assistant exchange with scope isolation."""
        messages = [
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": assistant_response},
        ]
        return await self.add_scoped(
            messages=messages,
            scope=scope,
            infer=infer,
            metadata=metadata,
        )
