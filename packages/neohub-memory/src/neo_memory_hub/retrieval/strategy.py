"""Multi-category parallel retrieval strategy.

Fires multiple vector searches in parallel (one per category), deduplicates
across results, and supports pool-based routing via PoolRoutingHook.
"""
from __future__ import annotations

import asyncio
import logging
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional

if TYPE_CHECKING:
    from neo_memory_hub.hooks.callbacks import PoolRoutingHook

logger = logging.getLogger(__name__)


@dataclass
class CategorySearchConfig:
    """Configuration for a single category search."""

    name: str  # Category name (e.g., "persona")
    threshold: float = 0.2  # Minimum similarity score
    limit: int = 5  # Max results for this category
    query_override: Optional[str] = None  # Alternative query text
    investor_aware_threshold: Optional[float] = None  # Lower threshold when investor filter active
    query_hint: Optional[str] = None  # Hint words appended when investor_name is known


@dataclass
class RetrievalResult:
    """Result of multi-category retrieval."""

    memories: List[Dict[str, Any]] = field(default_factory=list)
    by_category: Dict[str, List[Dict[str, Any]]] = field(default_factory=dict)
    count: int = 0
    user_id: str = ""
    agent_id: str = ""
    investor_name: Optional[str] = None


class MultiCategoryRetriever:
    """Parallel multi-category memory retrieval.

    Fires multiple vector searches concurrently via asyncio.gather,
    deduplicates across results, and supports pool-based retrieval.

    Usage:
        retriever = MultiCategoryRetriever(
            categories=[
                CategorySearchConfig(name="persona", threshold=0.2, limit=5),
                CategorySearchConfig(name="preference", threshold=0.15, limit=5),
                CategorySearchConfig(name="episodic", threshold=0.2, limit=5),
                CategorySearchConfig(
                    name="procedural", threshold=0.30, limit=5,
                    investor_aware_threshold=0.15,
                ),
            ],
        )
        result = await retriever.retrieve(
            query="...", search_fn=connector.search,
            user_id="rm_001", agent_id="vic_l3",
            investor_name="Senthil Kumar",
        )
    """

    def __init__(
        self,
        categories: Optional[List[CategorySearchConfig]] = None,
        parallel: bool = True,
    ) -> None:
        self._categories = categories if categories is not None else [
            CategorySearchConfig(name="persona", threshold=0.20, limit=5),
            CategorySearchConfig(name="preference", threshold=0.15, limit=5),
            CategorySearchConfig(name="episodic", threshold=0.20, limit=5),
            CategorySearchConfig(
                name="procedural",
                threshold=0.30,
                limit=5,
                investor_aware_threshold=0.15,
            ),
        ]
        self._parallel = parallel

    @staticmethod
    def normalize_investor_name(name: Optional[str]) -> Optional[str]:
        """Normalize entity name for case-insensitive matching.
        Returns None if name is empty/null-like.
        """
        if not name or not isinstance(name, str):
            return None
        name = name.strip()
        if not name or name.lower() in (
            "unknown",
            "unknown client",
            "none",
            "null",
        ):
            return None
        return name.title()

    @staticmethod
    def investor_name_to_subject_id(name: Optional[str]) -> Optional[str]:
        """Convert entity name to IsolationScope-compatible subject_id.
        Replaces spaces/special chars with underscores.
        """
        if not name or not isinstance(name, str):
            return None
        name = name.strip()
        if not name:
            return None
        subject = re.sub(r"[^a-zA-Z0-9_-]", "_", name)
        subject = re.sub(r"_+", "_", subject).strip("_")
        return subject or None

    async def retrieve(
        self,
        query: str,
        search_fn: Callable,
        user_id: str,
        agent_id: str,
        investor_name: Optional[str] = None,
        tenant_id: Optional[str] = None,
        allow_cross_agent: bool = True,
        dept_id: Optional[str] = None,
        pool_routing: Optional["PoolRoutingHook"] = None,
    ) -> RetrievalResult:
        """Run parallel multi-category retrieval.

        Args:
            query: User's query for semantic matching.
            search_fn: Async search callable (NeoMemoryConnector.search signature).
            user_id: RM/user identifier.
            agent_id: Agent identifier.
            investor_name: Entity name for filtering (None = skip entity-specific).
            tenant_id: Tenant for multi-tenancy.
            allow_cross_agent: Whether shared pool skips agent_id.
            dept_id: Department for team pool.
            pool_routing: Optional PoolRoutingHook for shared-scope pool searches.

        Returns:
            RetrievalResult with all retrieved memories and grouping.
        """
        investor_name = self.normalize_investor_name(investor_name)

        # Build metadata filters for PRIVATE pool
        private_filters: Dict[str, Any] = {}
        if tenant_id:
            private_filters["tenant_id"] = tenant_id
        if investor_name:
            private_filters["investor_name"] = investor_name
        private_filters = private_filters or None

        # Build category search tasks
        tasks = []
        labels = []

        for cat_cfg in self._categories:
            # Determine threshold
            threshold = cat_cfg.threshold
            if investor_name and cat_cfg.investor_aware_threshold is not None:
                threshold = cat_cfg.investor_aware_threshold

            # Determine query — enhance with hints when investor is known
            cat_query = cat_cfg.query_override or query
            if investor_name and not cat_cfg.query_override and cat_cfg.query_hint:
                cat_query = f"{investor_name} {query} {cat_cfg.query_hint}"

            tasks.append(
                search_fn(
                    query=cat_query,
                    user_id=user_id,
                    agent_id=agent_id,
                    categories=[cat_cfg.name],
                    limit=cat_cfg.limit,
                    threshold=threshold,
                    metadata_filters=private_filters,
                )
            )
            labels.append(cat_cfg.name)

        # Pool-tier searches: use PoolRoutingHook if provided
        if pool_routing:
            pool_tasks, pool_labels = pool_routing.get_pool_search_tasks(
                query=query,
                search_fn=search_fn,
                user_id=user_id,
                agent_id=agent_id,
                tenant_id=tenant_id,
                dept_id=dept_id,
                allow_cross_agent=allow_cross_agent,
            )
            tasks.extend(pool_tasks)
            labels.extend(pool_labels)

        # Execute
        if self._parallel:
            all_results = await asyncio.gather(*tasks, return_exceptions=True)
        else:
            all_results = []
            for t in tasks:
                try:
                    all_results.append(await t)
                except Exception as e:
                    all_results.append(e)

        # Process results
        seen_ids: set = set()
        all_memories: List[Dict[str, Any]] = []
        by_category: Dict[str, List[Dict[str, Any]]] = {
            label: [] for label in labels
        }

        for label, result_obj in zip(labels, all_results):
            if isinstance(result_obj, BaseException):
                logger.warning(f"[Retriever] {label} search failed: {result_obj}")
                continue
            if result_obj is None:
                logger.warning(f"[Retriever] {label} search returned None")
                continue
            for item in result_obj.get("results", []):
                memory_id = item.get("id")
                if memory_id and memory_id not in seen_ids:
                    seen_ids.add(memory_id)
                    item["_retrieval_reason"] = label
                    all_memories.append(item)
                    by_category.setdefault(label, []).append(item)

        return RetrievalResult(
            memories=all_memories,
            by_category=by_category,
            count=len(all_memories),
            user_id=user_id,
            agent_id=agent_id,
            investor_name=investor_name,
        )
