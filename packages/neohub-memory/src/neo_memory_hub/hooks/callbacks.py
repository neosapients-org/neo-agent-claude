"""Hook interfaces for application-specific extensions.

Applications implement these to inject domain-specific behavior
without modifying library code.

Hooks:
    EntityResolver — domain-specific entity name extraction
    TelemetryHook — observability callbacks (retrieval, storage, dedup)
    PoolRoutingHook — shared-scope pool routing (memory_utils extension point)
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional, Tuple


class EntityResolver(ABC):
    """Hook for domain-specific entity resolution.

    Applications implement this to resolve entity names from queries
    (e.g., investor name resolution in wealth management).
    """

    @abstractmethod
    async def resolve(
        self,
        query: str,
        session_context: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        """Resolve entity name from query text.

        Returns:
            Entity name string, or None if no specific entity identified.
        """
        ...


class TelemetryHook(ABC):
    """Hook for observability/telemetry integration."""

    def on_retrieval(
        self,
        query: str,
        memories_count: int,
        by_category: Dict[str, int],
        investor_name: Optional[str] = None,
    ) -> None:
        """Called after memory retrieval."""

    def on_storage(
        self,
        facts_stored: int,
        facts_skipped: int,
        facts_extracted: int,
        investor_name: Optional[str] = None,
    ) -> None:
        """Called after memory storage."""

    def on_dedup(
        self,
        action: str,
        category: str,
        investor_name: Optional[str] = None,
    ) -> None:
        """Called after dedup decision."""


class PoolRoutingHook(ABC):
    """Extension point for shared-scope pool routing.

    If no implementation is registered with HighLevelMemoryConnector,
    all memories are stored and retrieved as PRIVATE (single-user,
    single-agent). This is the seam between memory_core and
    memory_utils/shared_scope.

    Implementations should be provided by memory_utils and injected
    via dependency injection — memory_core never imports memory_utils.
    """

    @abstractmethod
    def resolve_store_params(
        self,
        fact_pool: Optional[str],
        user_id: str,
        agent_id: str,
        tenant_id: Optional[str],
        dept_id: Optional[str],
        metadata: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Resolve storage parameters for pool-aware memory storage.

        When pool is "team" or "org", this replaces user_id with sentinel
        values (e.g. ``__team_{dept_id}__``, ``__org_{tenant_id}__``) so
        that team/org memories are stored under a shared synthetic user.

        Args:
            fact_pool: Pool assigned to the fact (private/shared/team/org).
            user_id: Original user identifier.
            agent_id: Original agent identifier.
            tenant_id: Tenant identifier.
            dept_id: Department identifier.
            metadata: Metadata dict to potentially augment.

        Returns:
            Dict with keys: ``user_id``, ``agent_id``, ``metadata``
        """
        ...

    @abstractmethod
    def get_pool_search_tasks(
        self,
        query: str,
        search_fn: Callable,
        user_id: str,
        agent_id: str,
        tenant_id: Optional[str],
        dept_id: Optional[str],
        allow_cross_agent: bool,
    ) -> Tuple[List[Any], List[str]]:
        """Return additional search coroutines for pool-tier retrieval.

        These tasks are gathered alongside the core category search tasks.

        Args:
            query: Search query.
            search_fn: Async search callable (NeoMemoryConnector.search).
            user_id: User identifier.
            agent_id: Agent identifier.
            tenant_id: Tenant identifier.
            dept_id: Department identifier for team pool.
            allow_cross_agent: Whether shared pool skips agent_id.

        Returns:
            Tuple of (list of coroutines, list of label strings).
        """
        ...

    @abstractmethod
    def extend_profile(
        self,
        profile: Dict[str, Any],
        by_category: Dict[str, List[Dict[str, Any]]],
    ) -> Dict[str, Any]:
        """Extend the assembled profile with team/org/shared sections.

        Called after ProfileAssembler.assemble() to add pool-tier sections
        (e.g. DEPARTMENT DIRECTIVES, FIRM-WIDE DIRECTIVES).

        Args:
            profile: Profile dict from ProfileAssembler.assemble().
            by_category: Memories grouped by category/pool label.

        Returns:
            Updated profile dict with pool-tier sections added.
        """
        ...

    @abstractmethod
    def format_pool_sections(
        self,
        profile: Dict[str, Any],
        entity_name: Optional[str] = None,
    ) -> str:
        """Format pool-tier sections as markdown for context injection.

        Called after ContextFormatter.format() to append team/org sections.

        Args:
            profile: Profile dict including pool-tier entries.
            entity_name: Entity name for personalization.

        Returns:
            Markdown string with pool-tier sections.
        """
        ...
