"""Memory module — mem0-powered memory via HighLevelMemoryConnector.

- Retrieval: runs on first message of a session only (then cached in state)
- Storage: background task after each response (non-blocking)
- Category tagging: LLM-assigned per fact (no rule-based classification)
"""

import logging
import os
import time
from typing import Optional

logger = logging.getLogger(__name__)

try:
    from ns_probe import observe
except ImportError:
    def observe(*a, **kw):
        def _id(fn): return fn
        return _id

from neo_memory_hub import HighLevelMemoryConnector
from neo_memory_hub.config.hub_config import (
    MemoryHubConfig,
    VectorStoreConfig,
    LLMConfig,
    EmbedderConfig,
    RetrievalConfig,
    ExtractionConfig,
    SalienceConfig,
    IsolationConfig,
)

from ...config import config as agent_config


# Layer 1 — domain-specific extraction prompt.
# Replaces neo_memory_hub's generic default. All four memory types are allowed,
# but NONE of them may carry point-in-time financial values. Live figures
# (balances, allocations, holdings, prices, portfolio values, returns) are ALWAYS
# fetched fresh from MCP, so persisting them would only let stale numbers leak
# back into future answers and mislead the model.
agent_EXTRACTION_PROMPT = """You extract long-term memories for a wealth-management assistant.

Extract durable facts about the client and how to serve them. Each memory has a
category, entity name, key, value, and salience score.

## Categories
- persona — WHO the client is: identity, risk attitude, life stage, family
  context, goals, behavioral drivers. (Durable traits of the person.)
- preference — HOW they want to be served: communication style, report/format
  preferences, channels, exclusions, recurring asks about delivery. (Service
  delivery choices, not tasks.)
- episodic — significant EVENTS, DECISIONS, or reasoning: what happened or what
  was decided and WHY, described qualitatively, without figures. (A record of
  the past — already occurred.)
- procedural — HOW the assistant performs its work: reusable skills, workflows,
  and operating procedures. How to decompose a query into sub-queries, the
  order to call tools/MCP in, how to resolve dependencies between steps, how to
  compare/aggregate results, and how to format output. (Stable know-how reused
  across many queries — NOT client-specific reminders, NOT facts, NOT events.)

## persona vs preference (decide carefully — one fact, ONE category)
- persona = a description of the CLIENT (identity, life stage, family, risk attitude,
  financial goals/intent). Answers "who/what are they?"
- preference = an instruction for the ASSISTANT (how to communicate, format, deliver, or
  which products to recommend/exclude). Answers "what should I do?"
- A client's investment ATTITUDE or GOAL ("risk-averse", "capital preservation", "wants to
  invest conservatively", "prefers mutual funds and bonds") is persona — NOT preference.
- Never store the same idea under both categories. If a sentence has both a trait and a
  service instruction, split into one persona fact and one preference fact.

## DO NOT STORE (critical — applies to EVERY category)
- Any monetary amount, balance, net worth, AUM, portfolio/current value, invested amount, NAV or price.
- Any allocation, percentage split, return %, gain/loss figure, or holding quantity.
- Any point-in-time / transient figure that can change between sessions.
- Greetings, filler, acknowledgements, or restated questions.
If a candidate fact's MEANING depends on a number that is a financial value, DROP it entirely,
OR rephrase it qualitatively (e.g. "decided to rebalance toward debt" — NOT "moved 5% into debt").
These values are always fetched live from the platform — never persist them.

## Rules
- Prefer fewer, high-quality memories. When in doubt, drop it.
- "investor_name" attributes WHO a fact is about — set it carefully:
  - persona / episodic facts ABOUT an investor → set "investor_name" to that
    investor's full name.
  - preference / procedural facts (the advisor's OWN working style, or how they
    want data served/presented) → set "investor_name" to null. These belong to
    the advisor, NEVER to the client currently being discussed. Do not copy the
    client's name onto them just because a client was mentioned.
- Keys ("key"): lowercase_snake_case naming the DIMENSION, not the value, and STABLE across
  restatements so an updated value lands on the same key:
  - Name the dimension: "report_format" (NOT "pdf"), "benchmark" (NOT "nifty50"),
    "communication_style", "review_frequency".
  - For exclusions / restricted items, use a PER-ITEM key so one never clobbers another:
    "exclude_crypto", "exclude_tobacco" (NOT a shared "exclusions"). Same for allowed
    product scopes: "allow_pms", "allow_mutual_funds".
  - Use the SAME key when the advisor changes the value of an existing preference
    (e.g. benchmark Nifty 50 → Nifty 500 both use "benchmark").
- Salience: 0.0 (trivial) to 1.0 (critical).
- Pool is always "private".

## OUTPUT FORMAT
Return JSON: {"memories": [
  {
    "category": "persona" | "preference" | "episodic" | "procedural",
    "pool": "private",
    "investor_name": "Full Name or null",
    "key": "descriptive_key",
    "value": "concise durable fact (no financial values)",
    "detail": "optional context",
    "behavioral_note": "optional for episodic",
    "trigger": "optional for procedural",
    "salience": 0.75,
    "reasoning": "brief justification"
  }
]}
If there is nothing durable to store, return {"memories": []}.
"""


# Gap 3 — extraction prompt for the explicit "Save to permanent memory" action.
# The input is the RM's OWN durable working memory (typed in the RM's Memory box), so every
# item is ADVISOR-owned (investor_name=null) and is classified as either a preference or a
# procedural workflow — never a client fact, persona, or event.
SAVE_MEMORY_EXTRACTION_PROMPT = """The text below is the advisor's (RM's) OWN durable working
memory, typed deliberately to be saved permanently. Extract each distinct item and classify it.

## Classify each item as exactly ONE of:
- preference — WHAT the advisor wants: a delivery/communication/style choice, or a
  recommend/exclude rule. Examples: "concise, no jargon", "reports as PDF",
  "benchmark against Nifty 500", "never recommend crypto", "prioritize retirement planning".
- procedural — HOW the assistant should carry out a task: a reusable, multi-step
  workflow/method (it has steps, ordering, or a "when X, do Y" structure). Examples:
  "when comparing clients, fetch each separately then aggregate", "for a portfolio request,
  fetch holdings first, then show a table, then summary metrics".
  Tell: if it describes a method/steps → procedural; a single what/how-delivered choice → preference.

## Rules
- EVERY item is ADVISOR-owned: pool "private", investor_name = null. Never attribute to a client.
- Do NOT store point-in-time financial values (amounts, balances, %, NAV, prices). If meaning
  depends on such a number, rephrase qualitatively.
- Split multiple distinct items into separate memories; keep each concise.
- "key" = lowercase_snake_case naming the DIMENSION, not the value, stable across restatements
  (so updating a value reuses the same key):
  - dimension keys: "report_format" (not "pdf"), "benchmark" (not "nifty50"),
    "communication_style", "review_frequency", "compare_clients_workflow".
  - exclusions / restricted items use a PER-ITEM key so one never clobbers another:
    "exclude_crypto", "exclude_tobacco" (not a shared "exclusions").
- For procedural items, you may add "trigger" (when the workflow applies).
- Salience: 0.6–1.0 (these were explicitly saved, so they matter).

## OUTPUT FORMAT
Return JSON: {"memories": [
  {"category": "preference" | "procedural", "pool": "private", "investor_name": null,
   "key": "dimension_key", "value": "concise durable item", "detail": "optional",
   "trigger": "optional for procedural", "salience": 0.8, "reasoning": "brief"}
]}
If there is nothing durable to store, return {"memories": []}.
"""


class MemoryManager:
    """Mem0-backed memory manager using HighLevelMemoryConnector.

    Storage uses the full HLC pipeline: LLM fact extraction → salience gating
    → dedup → per-fact category tagging in Qdrant. No rule-based classification.
    """

    def __init__(self):
        self._connector: Optional[HighLevelMemoryConnector] = None
        self._ready = False

    async def initialize(self):
        """Initialize the HighLevelMemoryConnector."""
        # Kill-switch: skip connecting entirely when memory is disabled. With _ready left
        # False, every retrieve/store/retrieve_raw below short-circuits to a no-op.
        if not agent_config.memory_enabled:
            logger.info("[memory] Memory DISABLED (MEMORY_ENABLED=false) — skipping init")
            return

        qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")

        config = MemoryHubConfig(
            vector_store=VectorStoreConfig(
                provider="qdrant",
                collection_name="agent_memories",
                qdrant_url=qdrant_url,
                embedding_dims=1536,
            ),
            llm=LLMConfig(model=agent_config.openai_model_fast, temperature=0.1),
            embedder=EmbedderConfig(model="text-embedding-3-small", dimensions=1536),
            # Facts are stored as terse tokens ("PDF", "email"), so their cosine
            # similarity against natural-language questions is low (~0.07-0.20).
            # The library defaults (persona 0.20 / preference 0.15) drop valid
            # matches for verbose queries. Lower the cutoffs so recall is robust
            # to phrasing. (Per-user memory volume is small, so the extra recall
            # does not introduce noticeable noise.)
            # Gap 2: preferences are no longer fetched semantically — they come from a
            # complete metadata scroll (preference_full_fetch) — so the preference
            # threshold here is moot; persona/episodic/procedural stay semantic.
            retrieval=RetrievalConfig(
                limit=5,
                min_relevance_score=0.05,
                thresholds={
                    "persona": 0.05,
                    "preference": 0.05,
                    "episodic": 0.05,
                    "procedural": 0.05,
                },
                preference_full_fetch=True,
                preference_advisor_cap=25,
                preference_investor_cap=15,
                # Drive RM preference/procedural retrieval from the per-user summary
                # (one compact block) instead of scrolling every fact. Falls back to
                # the fact scrolls when the summary is empty (e.g. before first build).
                preference_from_summary=True,
            ),
            # Don't persist the verbose LLM salience justification on every point;
            # it's debug-only and is the largest text field in the payload.
            # Only persist facts the extractor scores as at least moderately
            # salient (>= 0.5). Lower-value/trivial facts are dropped at the gate.
            salience=SalienceConfig(store_reasoning=False, min_threshold=0.5),
            extraction=ExtractionConfig(
                enabled=True,
                prompt=agent_EXTRACTION_PROMPT,
                valid_categories=["persona", "preference", "episodic", "procedural"],
            ),
            # Advisor-only mode: store + retrieve only advisor-owned memories; client
            # (investor) memories are never written or read. Driven by MEMORY_ADVISOR_ONLY.
            isolation=IsolationConfig(advisor_only=agent_config.memory_advisor_only),
        )

        if agent_config.memory_advisor_only:
            logger.info("[memory] ADVISOR-ONLY mode enabled — client memories ignored")

        self._connector = HighLevelMemoryConnector(config=config)
        await self._connector.initialize()
        self._ready = True

    async def retrieve(self, query: str, user_id: str, investor_name: Optional[str] = None, client_recall_only: bool = False) -> str:
        """Retrieve relevant memories for the user. Called once at session start."""
        ctx, _ = await self.retrieve_with_raw(query, user_id, investor_name, client_recall_only)
        return ctx

    async def retrieve_with_raw(self, query: str, user_id: str, investor_name: Optional[str] = None, client_recall_only: bool = False) -> tuple:
        """Single retrieve_context call returning BOTH the formatted context string and the
        raw memory items (for the debug panel). Lets the graph's once-per-session memory
        fetch feed the debug panel too — instead of a second, redundant retrieve_raw call
        that contends with the graph and delays its start on the first turn."""
        if not self._ready or not self._connector:
            return "", []

        try:
            result = await self._connector.retrieve_context(
                query, user_id=user_id, investor_name=investor_name,
                client_recall_only=client_recall_only,
            )
            return result.get("context", ""), result.get("memories", [])
        except Exception:
            return "", []

    async def get_summary(self, user_id: str) -> str:
        """Return the user's preference SUMMARY text (deterministic point fetch, no
        vector search), or "" if none/disabled. Cheap enough to call inline in
        enrichment so preferences can shape the query sent to MCP."""
        if not self._ready or not self._connector:
            return ""
        try:
            return await self._connector.get_summary(user_id)
        except Exception:
            return ""

    async def retrieve_raw(self, query: str, user_id: str, investor_name: Optional[str] = None) -> list:
        """Retrieve raw memory items for a user. Used for the debug panel."""
        if not self._ready or not self._connector:
            return []

        try:
            result = await self._connector.retrieve_context(
                query, user_id=user_id, investor_name=investor_name
            )
            return result.get("memories", [])
        except Exception:
            return []

    async def list_all_memories(self, user_id: str, limit: int = 100) -> list:
        """List ALL stored long-term memories for a user (not query-scoped).

        Used by the debug panel's "Stored (all)" view so it reflects the true
        contents of Qdrant regardless of the current message.
        """
        if not self._ready or not self._connector:
            return []
        try:
            result = await self._connector.get_all(user_id=user_id, limit=limit)
            return result.get("results", [])
        except Exception:
            return []

    async def delete_memory(self, memory_id: str, user_id: str) -> bool:
        """Delete a single stored memory by id, but only if it belongs to user_id.

        The ownership check stops a caller from deleting another user's memory by guessing
        an id (the backend derives user_id from the authenticated user)."""
        if not self._ready or not self._connector or not memory_id:
            return False
        try:
            mem = await self._connector.get(memory_id)
            if not mem:
                return False
            owner = mem.get("user_id") or (mem.get("metadata", {}) or {}).get("user_id")
            if owner and owner != user_id:
                return False  # not the requester's memory — refuse
            # Pass user_id so the connector rebuilds the summary from the
            # remaining facts (so a deleted preference stops being applied).
            await self._connector.delete(memory_id, user_id=user_id)
            return True
        except Exception as e:
            logger.error("[memory.delete_memory] Failed: %s", str(e))
            return False

    async def purge_investor_memories(self, user_id: str) -> int:
        """Delete all client-scoped (investor) memories for a user, keeping advisor ones.

        Used to clean out existing client memories after switching to advisor-only mode."""
        if not self._ready or not self._connector:
            return 0
        try:
            result = await self._connector.delete_investor_memories(user_id=user_id)
            return result.get("deleted", 0)
        except Exception as e:
            logger.error("[memory.purge_investor_memories] Failed: %s", str(e), exc_info=True)
            return 0

    async def clear_user_memories(self, user_id: str) -> bool:
        """Delete all long-term memories for a user."""
        if not self._ready or not self._connector:
            return False

        try:
            await self._connector.delete_all(user_id=user_id)
            return True
        except Exception:
            return False

    
    async def save_preference(self, text: str, user_id: str) -> dict:
        """Gap 3 — promote RM-typed text into DURABLE advisor memory in Qdrant.

        Runs the save-extraction prompt (advisor-scoped; classifies each item as preference or
        procedural with a granular profile_key) through the full store pipeline, which applies
        conservative UPSERT-by-slot (a new value supersedes the same-slot item per category
        instead of duplicating)."""
        if not self._ready or not self._connector or not text or not text.strip():
            return {"facts_stored": 0, "facts_skipped": 0}
        try:
            result = await self._connector.store_exchange(
                query="[Saved RM memory]",
                response=text,
                user_id=user_id,
                investor_name=None,
                custom_extraction_prompt=SAVE_MEMORY_EXTRACTION_PROMPT,
                # Gap 3b: keep semantic dedup ON as the backstop. UPSERT-by-slot (with the
                # existing-keys prompt injection) handles same-slot replacement; dedup catches
                # any reworded same-concept save the LLM still keyed differently. Scoped to
                # subject_type=advisor + category, so client memories are never touched.
                skip_dedup=False,
            )
            return {
                "facts_stored": result.get("facts_stored", 0),
                "facts_skipped": result.get("facts_skipped", 0),
            }
        except Exception as e:
            logger.error("[memory.save_preference] Failed: %s", str(e), exc_info=True)
            return {"facts_stored": 0, "facts_skipped": 0, "error": str(e)}

    async def store(self, query: str, response: str, user_id: str, investor_name: Optional[str] = None,
                    force_investor_scope: bool = False, exclude_response: bool = False):
        """Store a conversation exchange.

        Delegates to HighLevelMemoryConnector.store_exchange() which runs the full
        pipeline: LLM fact extraction → salience gating → dedup → per-fact
        category tagging in Qdrant. No rule-based classification needed.

        force_investor_scope=True (client_note exchanges) attributes preference/persona facts
        to the named client instead of the advisor — so "store his investing preferences"
        saves under the client, not the RM.

        exclude_response=True (skill-driven turns) mines facts from the RM's OWN message only,
        keeping the skill's generated output out of extraction. The RM's words can still carry
        durable persona/preference, but the skill's data output never lands in memory.
        """
        if not self._ready or not self._connector:
            return

        try:
            await self._connector.store_exchange(
                query=query,
                response="" if exclude_response else response,
                user_id=user_id,
                investor_name=investor_name,
                force_investor_scope=force_investor_scope,
            )
        except Exception as e:
            logger.error("[memory.store] Failed to store memory: %s", str(e), exc_info=True)


# Singleton instance
memory_manager = MemoryManager()
