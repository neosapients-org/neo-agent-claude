"""Pool routing implementation for shared-scope memory.

Implements the PoolRoutingHook ABC from memory_core to provide:
- Sentinel user_id creation for TEAM/ORG pools
- Pool-tier search tasks for multi-pool retrieval
- Profile extension with team/org/shared sections
- Context formatting for department and firm-wide directives
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

from neo_memory_hub.hooks.callbacks import PoolRoutingHook

logger = logging.getLogger(__name__)


@dataclass
class PoolSearchConfig:
    """Configuration for a pool-tier search (shared/team/org)."""

    pool: str  # "shared", "team", "org"
    threshold: float = 0.2
    limit: int = 5
    sentinel_user_id: Optional[str] = None  # Override user_id for team/org


class SharedScopePoolRouter(PoolRoutingHook):
    """PoolRoutingHook implementation for shared-scope memory.

    Routes memory storage and retrieval across PRIVATE, SHARED, TEAM,
    and ORG pools using sentinel user_ids for cross-user isolation.

    Usage::

        from memory_utils.shared_scope import SharedScopePoolRouter, PoolSearchConfig

        router = SharedScopePoolRouter(
            pools=[
                PoolSearchConfig(pool="team", threshold=0.2, limit=5),
                PoolSearchConfig(pool="org", threshold=0.2, limit=5),
            ],
        )
        connector = HighLevelMemoryConnector(config=cfg, pool_routing=router)
    """

    def __init__(
        self,
        pools: Optional[List[PoolSearchConfig]] = None,
        default_pool: str = "private",
    ) -> None:
        self._pools = pools or []
        self._default_pool = default_pool

    def resolve_store_params(
        self,
        fact_pool: Optional[str],
        user_id: str,
        agent_id: str,
        tenant_id: Optional[str],
        dept_id: Optional[str],
        metadata: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Route storage to the correct user_id based on pool assignment."""
        pool = fact_pool or self._default_pool

        if pool == "team":
            return {
                "user_id": f"__team_{dept_id or 'default'}__",
                "agent_id": agent_id,
                "metadata": {**metadata, "dept_id": dept_id} if dept_id else metadata,
            }
        elif pool == "org":
            return {
                "user_id": f"__org_{tenant_id or 'default'}__",
                "agent_id": agent_id,
                "metadata": metadata,
            }
        elif pool == "shared":
            # Shared pool: same user, but accessible by all agents of that user
            return {
                "user_id": user_id,
                "agent_id": agent_id,
                "metadata": metadata,
            }

        # Default: private pool — no routing changes
        return {
            "user_id": user_id,
            "agent_id": agent_id,
            "metadata": metadata,
        }

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
        """Build pool-tier search tasks for parallel retrieval."""
        tasks: List[Any] = []
        labels: List[str] = []

        for pool_cfg in self._pools:
            pool_filters: Dict[str, Any] = {"pool": pool_cfg.pool}
            if tenant_id:
                pool_filters["tenant_id"] = tenant_id

            pool_user_id = user_id
            pool_agent_id: Optional[str] = agent_id

            if pool_cfg.pool == "shared":
                pool_agent_id = None if allow_cross_agent else agent_id
            elif pool_cfg.pool == "team":
                if dept_id:
                    pool_filters["dept_id"] = dept_id
                pool_user_id = (
                    pool_cfg.sentinel_user_id
                    or f"__team_{dept_id or 'default'}__"
                )
                pool_agent_id = None
            elif pool_cfg.pool == "org":
                pool_user_id = (
                    pool_cfg.sentinel_user_id
                    or f"__org_{tenant_id or 'default'}__"
                )
                pool_agent_id = None

            tasks.append(
                search_fn(
                    query=query,
                    user_id=pool_user_id,
                    agent_id=pool_agent_id,
                    limit=pool_cfg.limit,
                    threshold=pool_cfg.threshold,
                    metadata_filters=pool_filters,
                )
            )
            labels.append(pool_cfg.pool)

        return tasks, labels

    def extend_profile(
        self,
        profile: Dict[str, Any],
        by_category: Dict[str, List[Dict[str, Any]]],
    ) -> Dict[str, Any]:
        """Add team/org/shared sections to the assembled profile."""
        for pool_key in ("team", "org", "shared"):
            pool_mems = by_category.get(pool_key, [])
            if pool_mems:
                items = []
                for mem in pool_mems:
                    content = mem.get("memory", "")
                    items.append({
                        "directive": content,
                        "pool": pool_key,
                        "id": mem.get("id"),
                    })
                profile[pool_key] = items
        return profile

    def format_pool_sections(
        self,
        profile: Dict[str, Any],
        entity_name: Optional[str] = None,
    ) -> str:
        """Format team/org pool sections as markdown."""
        sections: List[str] = []

        _pool_headers = {
            "team": (
                "## DEPARTMENT DIRECTIVES (TEAM)",
                "Department-wide procedures shared across all users.",
            ),
            "org": (
                "## FIRM-WIDE DIRECTIVES (ORG)",
                "Organisation-wide mandates. Factor these into recommendations.",
            ),
        }

        for pool_key, (header, descriptor) in _pool_headers.items():
            items = profile.get(pool_key, [])
            if not items:
                continue
            section_lines = [header, f"*{descriptor}*", ""]
            for item in items:
                directive = item.get("directive", "")
                if directive:
                    section_lines.append(f"- {directive}")
            sections.append("\n".join(section_lines))

        return "\n\n".join(sections)
