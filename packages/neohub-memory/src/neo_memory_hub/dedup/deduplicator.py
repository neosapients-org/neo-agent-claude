"""Pre-store memory deduplication with LLM conflict resolution.

Before storing a new fact, searches for similar existing memories in the same
scope (user_id + entity_name + category) and uses an LLM to decide:
  - REPLACE: new fact supersedes old → delete old, store new
  - MERGE: combine into richer memory → delete old, store merged
  - KEEP_EXISTING: old is sufficient → skip new
  - NONE: unrelated → store new alongside existing
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional

from neo_memory_hub.extraction.prompts import DEFAULT_DEDUP_PROMPT

logger = logging.getLogger(__name__)


@dataclass
class DedupDecision:
    """Result of a dedup conflict resolution."""

    action: str  # REPLACE, MERGE, KEEP_EXISTING, NONE
    updated_memory: Optional[str] = None  # For REPLACE/MERGE: the new text
    reason: str = ""
    existing_memory_id: Optional[str] = None  # ID of the memory being acted upon


class MemoryDeduplicator:
    """Pre-store dedup layer. Stateless — all state lives in the vector store.

    Usage:
        dedup = MemoryDeduplicator(similarity_threshold=0.55)
        decision = await dedup.find_and_resolve(
            new_fact="Senthil prefers debt",
            category="preference",
            investor_name="Senthil Kumar",
            user_id="rm_001",
            agent_id="vic_l3",
            search_fn=connector.search,
        )
        if decision.action == "KEEP_EXISTING":
            # Skip storing
        elif decision.action == "REPLACE":
            await connector.delete(decision.existing_memory_id)
            await connector.store_fact(decision.updated_memory, ...)
    """

    def __init__(
        self,
        similarity_threshold: float = 0.55,
        max_candidates: int = 5,
        llm_model: str = "gpt-4o-mini",
        temperature: float = 0.1,
        dedup_prompt: Optional[str] = None,
    ) -> None:
        self._similarity_threshold = similarity_threshold
        self._max_candidates = max_candidates
        self._llm_model = llm_model
        self._temperature = temperature
        self._dedup_prompt = dedup_prompt or DEFAULT_DEDUP_PROMPT

    async def find_and_resolve(
        self,
        new_fact: str,
        category: str,
        investor_name: Optional[str],
        user_id: str,
        agent_id: str,
        search_fn: Callable,
        metadata_filters: Optional[Dict[str, Any]] = None,
    ) -> DedupDecision:
        """Search for similar existing memories and resolve conflicts.

        Args:
            new_fact: The new fact text to check for duplicates.
            category: Memory category.
            investor_name: Entity name for scoped search (None for team/org).
            user_id: Effective user_id (may be sentinel for team/org).
            agent_id: Agent identifier.
            search_fn: Async callable for vector search.
            metadata_filters: Additional metadata filters.

        Returns:
            DedupDecision with action and optional updated_memory text.
        """
        # Build scoped filters
        filters = dict(metadata_filters or {})
        if investor_name:
            filters["investor_name"] = investor_name

        candidates = await search_fn(
            query=new_fact,
            user_id=user_id,
            agent_id=agent_id,
            categories=[category],
            limit=self._max_candidates,
            threshold=self._similarity_threshold,
            metadata_filters=filters if filters else None,
        )

        results = candidates.get("results", [])
        logger.info(
            f"[Dedup] query='{new_fact[:60]}...', category={category}, "
            f"investor={investor_name}, found={len(results)} candidates"
        )
        if not results:
            return DedupDecision(
                action="NONE", reason="no similar existing memories found"
            )

        return await self._resolve_conflict(
            new_fact, results, category, investor_name
        )

    async def _resolve_conflict(
        self,
        new_fact: str,
        existing_memories: List[Dict[str, Any]],
        category: str,
        investor_name: Optional[str],
    ) -> DedupDecision:
        """Call LLM to decide how new fact relates to existing memories."""
        from openai import AsyncOpenAI

        existing_context = []
        for mem in existing_memories:
            content = (
                mem.get("memory")
                or mem.get("data")
                or mem.get("content")
                or ""
            )
            meta = mem.get("metadata", {}) or {}
            existing_context.append(
                {
                    "id": mem.get("id", ""),
                    "content": content,
                    "investor_name": meta.get("investor_name", "unknown"),
                    "category": (
                        (meta.get("categories") or [None])[0]
                        if isinstance(meta.get("categories"), list)
                        else meta.get("memory_type", "unknown")
                    ),
                    "stored_at": meta.get("stored_at", "unknown"),
                }
            )

        current_time = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        user_message = (
            f"EXISTING MEMORIES:\n```json\n{json.dumps(existing_context, indent=2)}\n```\n\n"
            f"NEW FACT (current time: {current_time}):\n"
            f"- Content: {new_fact}\n"
            f"- Investor: {investor_name or 'unknown'}\n"
            f"- Category: {category}\n\n"
            f"Which existing memory (if any) conflicts with or duplicates this new fact? "
            f"Return your decision as JSON."
        )

        try:
            client = AsyncOpenAI()
            resp = await client.chat.completions.create(
                model=self._llm_model,
                temperature=self._temperature,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": self._dedup_prompt},
                    {"role": "user", "content": user_message},
                ],
            )
            raw = resp.choices[0].message.content or "{}"
            decision = json.loads(raw)

            action = decision.get("action", "NONE").upper()
            if action not in {"REPLACE", "MERGE", "KEEP_EXISTING", "NONE"}:
                action = "NONE"

            target_id = decision.get("target_memory_id")
            if not target_id and action in ("REPLACE", "MERGE", "KEEP_EXISTING"):
                target_id = (
                    existing_memories[0].get("id") if existing_memories else None
                )

            return DedupDecision(
                action=action,
                updated_memory=decision.get("updated_memory"),
                reason=decision.get("reason", ""),
                existing_memory_id=target_id,
            )
        except Exception as e:
            logger.warning(f"[Dedup] LLM resolution failed: {e}, defaulting to NONE")
            return DedupDecision(action="NONE", reason=f"LLM error: {e}")
