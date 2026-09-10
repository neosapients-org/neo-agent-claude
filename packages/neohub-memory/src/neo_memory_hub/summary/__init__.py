"""Per-user preference SUMMARY layer for neo_memory_hub.

A single deterministic-ID point per user, co-located with granular facts in the
same Qdrant collection (payload type="summary" vs "fact"), merged from facts on
each turn with recency-based conflict resolution.
"""

from .llm import llm_merge, llm_summarize
from .store import (
    TYPE_FACT,
    TYPE_SUMMARY,
    UPDATED_AT_KEY,
    SummaryStore,
    summary_id,
    utc_now_iso,
)

__all__ = [
    "SummaryStore",
    "summary_id",
    "utc_now_iso",
    "llm_merge",
    "llm_summarize",
    "TYPE_FACT",
    "TYPE_SUMMARY",
    "UPDATED_AT_KEY",
]
