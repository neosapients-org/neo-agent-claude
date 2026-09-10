"""Unit tests for the per-user preference SUMMARY layer.

No real Qdrant or OpenAI: the Mem0 instance is faked (vector_store + embedder)
and merge/summarize are stubbed, so these exercise SummaryStore logic only.
"""
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from neo_memory_hub.summary import (
    TYPE_SUMMARY,
    UPDATED_AT_KEY,
    SummaryStore,
    summary_id,
)


def _rec(point_id, payload):
    return SimpleNamespace(id=point_id, payload=payload)


def _fake_mem0():
    """A stand-in for Mem0's AsyncMemory exposing the bits SummaryStore uses."""
    vs = MagicMock()
    vs.collection_name = "test_collection"
    vs.get.return_value = None
    vs.list.return_value = ([], None)
    vs.insert.return_value = None
    client = MagicMock()
    client.scroll.return_value = ([], None)
    vs.client = client
    embedding_model = MagicMock()
    embedding_model.embed.return_value = [0.1, 0.2, 0.3, 0.4]
    return SimpleNamespace(vector_store=vs, embedding_model=embedding_model)


def _store(mem0, merge_fn=None, summarize_fn=None):
    async def _default_merge(old, facts):
        return (old + " | " if old else "") + "; ".join(facts)

    async def _default_summarize(facts):
        return " | ".join(facts)

    return SummaryStore(
        mem0,
        merge_fn=merge_fn or _default_merge,
        summarize_fn=summarize_fn or _default_summarize,
    )


# --- summary_id -----------------------------------------------------------
class TestSummaryId:
    def test_deterministic(self):
        assert summary_id("u1") == summary_id("u1")

    def test_distinct_per_user(self):
        assert summary_id("u1") != summary_id("u2")


# --- get_summary ----------------------------------------------------------
class TestGetSummary:
    async def test_returns_data(self):
        mem0 = _fake_mem0()
        mem0.vector_store.get.return_value = _rec(summary_id("u1"), {"data": "likes PDF"})
        store = _store(mem0)
        assert await store.get_summary("u1") == "likes PDF"
        mem0.vector_store.get.assert_called_once_with(summary_id("u1"))

    async def test_empty_when_missing(self):
        store = _store(_fake_mem0())
        assert await store.get_summary("u1") == ""


# --- update_summary -------------------------------------------------------
class TestUpdateSummary:
    async def test_upserts_with_deterministic_id_and_summary_payload(self):
        mem0 = _fake_mem0()
        captured = {}

        async def merge(old, facts):
            return "benchmark Nifty 500"

        store = _store(mem0, merge_fn=merge)
        await store.update_summary("u1", ["benchmark Nifty 500"])

        mem0.embedding_model.embed.assert_called_once()
        mem0.vector_store.insert.assert_called_once()
        kwargs = mem0.vector_store.insert.call_args.kwargs
        assert kwargs["ids"] == [summary_id("u1")]
        payload = kwargs["payloads"][0]
        assert payload["type"] == TYPE_SUMMARY
        assert payload["data"] == "benchmark Nifty 500"
        assert payload["user_id"] == "u1"
        assert payload["profile_key"] == "__summary__"
        assert UPDATED_AT_KEY in payload

    async def test_noop_when_merge_unchanged(self):
        mem0 = _fake_mem0()
        mem0.vector_store.get.return_value = _rec(summary_id("u1"), {"data": "same"})

        async def merge(old, facts):
            return "same"

        store = _store(mem0, merge_fn=merge)
        await store.update_summary("u1", ["something"])
        mem0.vector_store.insert.assert_not_called()

    async def test_no_facts_skips_merge_and_upsert(self):
        mem0 = _fake_mem0()
        merge = MagicMock()
        store = _store(mem0, merge_fn=merge)
        await store.update_summary("u1", [])
        merge.assert_not_called()
        mem0.vector_store.insert.assert_not_called()


# --- rebuild_summary ------------------------------------------------------
class TestRebuildSummary:
    async def test_sorts_facts_oldest_to_newest(self):
        mem0 = _fake_mem0()
        # Deliberately out of order; newest last after sort.
        mem0.vector_store.list.return_value = (
            [
                _rec("b", {"data": "newer", UPDATED_AT_KEY: "2026-03-01T00:00:00+00:00"}),
                _rec("a", {"data": "older", UPDATED_AT_KEY: "2026-01-01T00:00:00+00:00"}),
                _rec("c", {"data": "newest", UPDATED_AT_KEY: "2026-06-01T00:00:00+00:00"}),
            ],
            None,
        )
        seen = {}

        async def summarize(facts):
            seen["facts"] = list(facts)
            return "rebuilt"

        store = _store(mem0, summarize_fn=summarize)
        result = await store.rebuild_summary("u1")

        assert seen["facts"] == ["older", "newer", "newest"]
        assert result == "rebuilt"
        mem0.vector_store.insert.assert_called_once()
        assert mem0.vector_store.insert.call_args.kwargs["ids"] == [summary_id("u1")]

    async def test_extra_facts_unioned_without_duplicates(self):
        mem0 = _fake_mem0()
        # Scroll already returns one fact; extra_facts adds a new one + a dup.
        mem0.vector_store.list.return_value = (
            [_rec("a", {"data": "older", UPDATED_AT_KEY: "2026-01-01T00:00:00+00:00"})],
            None,
        )
        seen = {}

        async def summarize(facts):
            seen["facts"] = list(facts)
            return "rebuilt"

        store = _store(mem0, summarize_fn=summarize)
        await store.rebuild_summary("u1", extra_facts=["older", "brand new fact"])

        # "older" not duplicated; "brand new fact" appended as newest.
        assert seen["facts"] == ["older", "brand new fact"]

    async def test_deletes_summary_when_no_facts_remain(self):
        mem0 = _fake_mem0()
        mem0.vector_store.list.return_value = ([], None)  # all facts gone
        seen = {}

        async def summarize(facts):
            seen["called"] = True
            return "should not be used"

        store = _store(mem0, summarize_fn=summarize)
        result = await store.rebuild_summary("u1")

        assert result == ""
        assert "called" not in seen  # no LLM call when there are no facts
        mem0.vector_store.insert.assert_not_called()
        # The stale summary point is deleted instead of left behind.
        mem0.vector_store.client.delete.assert_called_once()

    async def test_falls_back_to_mem0_timestamp(self):
        mem0 = _fake_mem0()
        mem0.vector_store.list.return_value = (
            [
                _rec("b", {"data": "newer", "created_at": "2026-05-01T00:00:00+00:00"}),
                _rec("a", {"data": "older", "created_at": "2026-01-01T00:00:00+00:00"}),
            ],
            None,
        )
        seen = {}

        async def summarize(facts):
            seen["facts"] = list(facts)
            return "x"

        await _store(mem0, summarize_fn=summarize).rebuild_summary("u1")
        assert seen["facts"] == ["older", "newer"]


# --- ensure_indexes / backfill (idempotency) ------------------------------
class TestIndexesAndBackfill:
    async def test_ensure_indexes_swallows_already_exists(self):
        mem0 = _fake_mem0()
        mem0.vector_store.client.create_payload_index.side_effect = Exception("exists")
        store = _store(mem0)
        # Must not raise even when every index already exists.
        await store.ensure_indexes()
        await store.ensure_indexes()
        assert mem0.vector_store.client.create_payload_index.call_count == 6  # 3 fields x 2 runs

    async def test_backfill_tags_untyped_then_is_noop(self):
        mem0 = _fake_mem0()
        scroll = mem0.vector_store.client.scroll
        # First run: one page of untyped points, then empty.
        scroll.side_effect = [
            ([_rec("p1", None), _rec("p2", None)], None),
        ]
        store = _store(mem0)
        updated = await store.backfill_type()
        assert updated == 2
        sp = mem0.vector_store.client.set_payload.call_args.kwargs
        assert sp["payload"] == {"type": "fact"}
        assert sp["points"] == ["p1", "p2"]

        # Second run: nothing left untyped → no-op, no set_payload, no error.
        scroll.side_effect = [([], None)]
        mem0.vector_store.client.set_payload.reset_mock()
        assert await store.backfill_type() == 0
        mem0.vector_store.client.set_payload.assert_not_called()


class TestDeleteSummary:
    async def test_delete_summary_targets_deterministic_id(self):
        mem0 = _fake_mem0()
        store = _store(mem0)
        await store.delete_summary("u1")
        mem0.vector_store.client.delete.assert_called_once()
        sel = mem0.vector_store.client.delete.call_args.kwargs["points_selector"]
        assert summary_id("u1") in sel.points
