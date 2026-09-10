"""Direct-Qdrant per-user preference SUMMARY layer.

Facts continue to flow through Mem0 unchanged. Summary points, however, need a
DETERMINISTIC id (one per user, overwritten in place) which Mem0 cannot provide
— it assigns random UUIDs. So summary points are written/read directly through
the raw Qdrant client that Mem0 already holds, reusing Mem0's embedder so the
summary vector lives in the SAME collection as the facts (same dimension, same
named-vector layout — we go through ``vector_store.insert`` which builds the
named/dense + optional bm25 vectors for us).

Access path (already used elsewhere, e.g. connector.py history wiring):
    mem0.vector_store.client          # sync QdrantClient
    mem0.vector_store.collection_name
    mem0.vector_store.insert/get/list  # named-vector aware helpers
    mem0.embedding_model.embed(text, "add")

All sync Qdrant/embedder calls are run via ``asyncio.to_thread`` so they never
block the event loop.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable, List, Optional

logger = logging.getLogger(__name__)

# Payload discriminator values.
TYPE_FACT = "fact"
TYPE_SUMMARY = "summary"

# Our own recency field. Mem0 writes its own created_at/updated_at into point
# payloads, so we use a namespaced key to avoid any ambiguity, and fall back to
# Mem0's fields for legacy points that predate this layer.
UPDATED_AT_KEY = "mem_updated_at"

MergeFn = Callable[[str, List[str]], Awaitable[str]]
SummarizeFn = Callable[[List[str]], Awaitable[str]]


def summary_id(user_id: str) -> str:
    """Deterministic point id for a user's single summary point."""
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, f"summary::{user_id}"))


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class SummaryStore:
    """Manages the per-user summary point in the shared Qdrant collection."""

    def __init__(
        self,
        mem0: Any,
        *,
        merge_fn: MergeFn,
        summarize_fn: SummarizeFn,
        profile_key: str = "__summary__",
    ) -> None:
        self._mem0 = mem0
        self._merge_fn = merge_fn
        self._summarize_fn = summarize_fn
        self._profile_key = profile_key

    # --- raw handles -------------------------------------------------------
    @property
    def _vs(self) -> Any:
        return self._mem0.vector_store

    @property
    def _client(self) -> Any:
        return self._mem0.vector_store.client

    @property
    def _collection(self) -> str:
        return self._mem0.vector_store.collection_name

    async def _embed(self, text: str) -> List[float]:
        return await asyncio.to_thread(self._mem0.embedding_model.embed, text, "add")

    # --- indexes & backfill (idempotent) -----------------------------------
    async def ensure_indexes(self) -> None:
        """Create KEYWORD payload indexes on user_id, type, profile_key.

        Idempotent: re-creating an existing index raises in Qdrant, which we
        swallow. Local/in-memory Qdrant may not support payload indexes — also
        swallowed (filtering still works without an index, just slower).
        """
        for field in ("user_id", "type", "profile_key"):
            try:
                await asyncio.to_thread(
                    self._client.create_payload_index,
                    collection_name=self._collection,
                    field_name=field,
                    field_schema="keyword",
                )
                logger.info("[summary] ensured payload index on %s", field)
            except Exception as e:  # already exists / unsupported on local
                logger.debug("[summary] index on %s not created: %s", field, e)

    async def backfill_type(self, batch_size: int = 256) -> int:
        """Set type="fact" on every existing point that lacks a type field.

        Idempotent: points that already have a type are not matched, so re-runs
        are no-ops. Returns the number of points updated.
        """
        from qdrant_client import models

        flt = models.Filter(
            must=[models.IsEmptyCondition(is_empty=models.PayloadField(key="type"))]
        )
        updated = 0
        offset = None
        while True:
            points, offset = await asyncio.to_thread(
                self._client.scroll,
                collection_name=self._collection,
                scroll_filter=flt,
                limit=batch_size,
                offset=offset,
                with_payload=False,
                with_vectors=False,
            )
            if not points:
                break
            ids = [p.id for p in points]
            await asyncio.to_thread(
                self._client.set_payload,
                collection_name=self._collection,
                payload={"type": TYPE_FACT},
                points=ids,
            )
            updated += len(ids)
            if offset is None:
                break
        if updated:
            logger.info("[summary] backfilled type=fact on %d legacy point(s)", updated)
        return updated

    # --- read --------------------------------------------------------------
    async def get_summary(self, user_id: str) -> str:
        """Return the user's summary text, or "" if none. No vector search."""
        try:
            rec = await asyncio.to_thread(self._vs.get, summary_id(user_id))
        except Exception as e:
            logger.debug("[summary] get_summary failed for %s: %s", user_id, e)
            return ""
        if not rec:
            return ""
        payload = getattr(rec, "payload", None) or {}
        return payload.get("data") or ""

    async def delete_summary(self, user_id: str) -> None:
        """Delete the user's summary point (used when all their facts are cleared)."""
        from qdrant_client import models

        try:
            await asyncio.to_thread(
                self._client.delete,
                collection_name=self._collection,
                points_selector=models.PointIdsList(points=[summary_id(user_id)]),
            )
        except Exception as e:
            logger.debug("[summary] delete_summary failed for %s: %s", user_id, e)

    # --- write -------------------------------------------------------------
    async def _upsert_summary(self, user_id: str, text: str) -> None:
        text = (text or "").strip()
        if not text:
            return
        vector = await self._embed(text)
        now = utc_now_iso()
        payload = {
            "data": text,
            "type": TYPE_SUMMARY,
            "user_id": user_id,
            "profile_key": self._profile_key,
            UPDATED_AT_KEY: now,
            "created_at": now,
            "updated_at": now,
        }
        # insert() upserts by id (overwrites the single summary point in place)
        # and builds the collection's named/dense + bm25 vectors from payload.
        await asyncio.to_thread(
            self._vs.insert,
            vectors=[vector],
            payloads=[payload],
            ids=[summary_id(user_id)],
        )

    async def update_summary(self, user_id: str, new_facts: List[str]) -> str:
        """Incrementally fold this turn's new facts into the user's summary."""
        new_facts = [f for f in (new_facts or []) if f and f.strip()]
        if not new_facts:
            return await self.get_summary(user_id)
        old = await self.get_summary(user_id)
        merged = await self._merge_fn(old, new_facts)
        if merged and merged.strip() and merged.strip() != (old or "").strip():
            await self._upsert_summary(user_id, merged)
        return merged or old or ""

    async def rebuild_summary(
        self,
        user_id: str,
        max_facts: int = 1000,
        extra_facts: Optional[List[str]] = None,
    ) -> str:
        """Rebuild the summary from scratch over ALL of the user's facts.

        ``extra_facts`` (the texts just stored this turn) are unioned in as the
        newest items, so a freshly-stored fact is never missed if Qdrant hasn't
        finished indexing it by the time we scroll.
        """
        try:
            scrolled = await asyncio.to_thread(
                self._vs.list,
                {"user_id": user_id, "type": TYPE_FACT},
                max_facts,
            )
        except Exception as e:
            logger.warning("[summary] rebuild scroll failed for %s: %s", user_id, e)
            return await self.get_summary(user_id)

        # vector_store.list() returns client.scroll()'s (points, next_offset).
        points = scrolled[0] if isinstance(scrolled, tuple) else scrolled
        facts_with_ts = []
        for p in points or []:
            payload = getattr(p, "payload", None) or {}
            text = payload.get("data")
            if not text:
                continue
            ts = (
                payload.get(UPDATED_AT_KEY)
                or payload.get("updated_at")
                or payload.get("created_at")
                or ""
            )
            facts_with_ts.append((ts, text))

        facts_with_ts.sort(key=lambda t: t[0])  # oldest → newest
        sorted_facts = [text for _, text in facts_with_ts]

        # Union in this turn's just-stored facts (newest), de-duplicating exact
        # text matches already present from the scroll.
        if extra_facts:
            seen = {f.strip() for f in sorted_facts}
            for f in extra_facts:
                if f and f.strip() and f.strip() not in seen:
                    sorted_facts.append(f)
                    seen.add(f.strip())

        # No facts left (e.g. the last preference was just deleted) → there is
        # nothing to summarize; drop the stale summary point so it can't keep
        # surfacing deleted preferences at retrieval time.
        if not sorted_facts:
            await self.delete_summary(user_id)
            return ""

        summary = await self._summarize_fn(sorted_facts)
        if summary and summary.strip():
            await self._upsert_summary(user_id, summary)
        return summary or ""
