"""HighLevelMemoryConnector — the primary API for neo_memory_hub v0.3.0.

Orchestrates:
- LLM fact extraction (extraction/)
- Salience gating (core/storage_gateway.py)
- Pre-store dedup (dedup/)
- Exchange buffering (buffering/)
- Multi-category retrieval (retrieval/strategy.py)
- Context formatting (retrieval/formatter.py)
- History API (core/history.py)

Usage:
    from neo_memory_hub import HighLevelMemoryConnector, load_config

    config = load_config("memory.yaml")
    async with HighLevelMemoryConnector(config=config) as memory:
        await memory.store_exchange(query="...", response="...", user_id="u1")
        result = await memory.retrieve_context(query="...", user_id="u1")
"""
from __future__ import annotations

import asyncio
import logging
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from neo_memory_hub.buffering import BufferedExchange, ExchangeBuffer
from neo_memory_hub.config.hub_config import MemoryHubConfig, load_config
from neo_memory_hub.core.history import MemoryHistoryManager
from neo_memory_hub.core.storage_gateway import (
    StorageDecision,
    StorageGateway,
    StorageGatewayConfig,
)
from neo_memory_hub.dedup import DedupDecision, MemoryDeduplicator
from neo_memory_hub.domain.memory import MemoryEntry
from neo_memory_hub.domain.types import MemoryType
from neo_memory_hub.extraction import ExtractedFact, FactExtractor
from neo_memory_hub.hooks.callbacks import EntityResolver, PoolRoutingHook, TelemetryHook
from neo_memory_hub.integrations.connector import ConnectorConfig, NeoMemoryConnector
from neo_memory_hub.integrations.memory import MemoryConfig
from neo_memory_hub.retrieval.formatter import ContextFormatter, ProfileAssembler
from neo_memory_hub.retrieval.strategy import (
    CategorySearchConfig,
    MultiCategoryRetriever,
    RetrievalResult,
)
from neo_memory_hub.summary import SummaryStore, llm_merge, llm_summarize

logger = logging.getLogger(__name__)


class HighLevelMemoryConnector:
    """Complete memory management API for agentic applications.

    This is the single class applications need to interact with.
    It wraps NeoMemoryConnector (low-level Mem0 API) and adds:
    - LLM fact extraction pipeline
    - Pre-store dedup
    - Exchange buffering
    - Multi-category parallel retrieval
    - Rich context formatting + profile assembly
    - Salience gating
    - Mem0 history API
    - Hook points for entity resolution and telemetry
    """

    def __init__(
        self,
        config: Optional[MemoryHubConfig] = None,
        config_path: Optional[str] = None,
        entity_resolver: Optional[EntityResolver] = None,
        telemetry: Optional[TelemetryHook] = None,
        pool_routing: Optional[PoolRoutingHook] = None,
    ) -> None:
        self._config = config or load_config(config_path=config_path)
        self._entity_resolver = entity_resolver
        self._telemetry = telemetry
        self._pool_routing = pool_routing

        self._connector: Optional[NeoMemoryConnector] = None
        self._extractor: Optional[FactExtractor] = None
        self._deduplicator: Optional[MemoryDeduplicator] = None
        self._buffer: Optional[ExchangeBuffer] = None
        self._retriever: Optional[MultiCategoryRetriever] = None
        self._assembler: Optional[ProfileAssembler] = None
        self._formatter: Optional[ContextFormatter] = None
        self._gateway: Optional[StorageGateway] = None
        self._history: Optional[MemoryHistoryManager] = None
        self._summary: Optional[SummaryStore] = None
        self._initialized = False

    # =========================================================================
    # Lifecycle
    # =========================================================================

    async def initialize(self) -> None:
        if self._initialized:
            return

        cfg = self._config

        # Build Mem0 MemoryConfig from MemoryHubConfig
        mem0_config = self._build_mem0_config()
        connector_config = ConnectorConfig(
            default_limit=cfg.retrieval.limit,
            default_threshold=cfg.retrieval.min_relevance_score,
            # When the summary layer is on, fact reads exclude the summary point.
            # Safe because initialize() backfills type on existing points below.
            exclude_non_fact_points=cfg.summary.enabled,
        )
        self._connector = NeoMemoryConnector(
            memory_config=mem0_config,
            config=connector_config,
        )
        await self._connector.initialize()

        # Fact extractor
        if cfg.extraction.enabled:
            self._extractor = FactExtractor(
                llm_model=cfg.llm.model,
                temperature=cfg.llm.temperature,
                extraction_prompt=cfg.extraction.prompt,
                valid_categories=cfg.extraction.valid_categories,
                default_salience=cfg.salience.default_score,
            )

        # Deduplicator
        if cfg.dedup.enabled:
            self._deduplicator = MemoryDeduplicator(
                similarity_threshold=cfg.dedup.similarity_threshold,
                max_candidates=cfg.dedup.max_candidates,
                llm_model=cfg.dedup.model or cfg.llm.model,
                temperature=cfg.dedup.temperature,
                dedup_prompt=cfg.dedup.prompt,
            )

        # Exchange buffer
        if cfg.buffering.enabled:
            self._buffer = ExchangeBuffer(
                flush_threshold=cfg.buffering.flush_threshold,
                max_buffer_chars=cfg.buffering.max_buffer_chars,
                flush_on_session_end=cfg.buffering.flush_on_session_end,
                idle_timeout_seconds=cfg.buffering.idle_timeout_seconds,
            )
            self._buffer.set_flush_callback(self._on_idle_flush)

        # Retriever
        categories = [
            CategorySearchConfig(
                name=cat,
                threshold=cfg.retrieval.thresholds.get(cat, 0.2),
                limit=5,
                investor_aware_threshold=cfg.retrieval.investor_aware_thresholds.get(cat),
                query_hint=cfg.retrieval.query_hints.get(cat),
            )
            for cat in cfg.extraction.valid_categories
        ]
        self._retriever = MultiCategoryRetriever(
            categories=categories,
        )

        # Formatter
        self._assembler = ProfileAssembler()
        self._formatter = ContextFormatter(
            header="---CLIENT MEMORY CONTEXT---",
            max_chars=cfg.storage.max_context_chars,
        )

        # Storage gateway
        if cfg.salience.enabled:
            self._gateway = StorageGateway(
                StorageGatewayConfig(
                    min_salience_threshold=cfg.salience.min_threshold,
                    min_content_length=cfg.salience.min_content_length,
                    max_content_length=cfg.salience.max_content_length,
                )
            )

        # History
        try:
            mem0_raw = getattr(self._connector._memory, "_mem0", None)
            if mem0_raw:
                self._history = MemoryHistoryManager(mem0_raw)
        except Exception:
            pass

        # Per-user preference summary layer (same Qdrant collection, payload
        # type="summary"). Fully best-effort: a failure here never blocks init.
        if cfg.summary.enabled:
            try:
                mem0_raw = getattr(self._connector._memory, "_mem0", None)
                if mem0_raw:
                    summary_model = cfg.summary.model or cfg.llm.model
                    summary_temp = cfg.summary.temperature

                    async def _merge(old: str, facts: List[str]) -> str:
                        return await llm_merge(
                            old, facts, model=summary_model, temperature=summary_temp
                        )

                    async def _summarize(facts: List[str]) -> str:
                        return await llm_summarize(
                            facts, model=summary_model, temperature=summary_temp
                        )

                    self._summary = SummaryStore(
                        mem0_raw,
                        merge_fn=_merge,
                        summarize_fn=_summarize,
                        profile_key=cfg.summary.profile_key,
                    )
                    await self._summary.ensure_indexes()
                    if cfg.summary.backfill_on_init:
                        await self._summary.backfill_type()
            except Exception as e:
                logger.warning("[HighLevelMemoryConnector] summary init failed: %s", e)
                self._summary = None

        self._initialized = True
        logger.info("[HighLevelMemoryConnector] Initialized")

    async def close(self) -> None:
        if self._connector:
            await self._connector.close()
        self._initialized = False

    async def __aenter__(self):
        await self.initialize()
        return self

    async def __aexit__(self, *args):
        await self.close()

    @property
    def is_ready(self) -> bool:
        return self._initialized and self._connector is not None

    @property
    def config(self) -> MemoryHubConfig:
        return self._config

    def _get_agent_id(self, agent_id: Optional[str] = None) -> str:
        return agent_id or self._config.isolation.default_agent_id

    # =========================================================================
    # Mem0 Config Builder
    # =========================================================================

    def _build_mem0_config(self) -> MemoryConfig:
        """Build low-level MemoryConfig from MemoryHubConfig."""
        cfg = self._config
        vs = cfg.vector_store

        if vs.provider == "pgvector":
            return MemoryConfig.for_pgvector(
                connection_string=vs.pgvector_url
                or os.getenv("NEO_MEMORY_POSTGRES_URL", ""),
                collection_name=vs.collection_name,
                embedding_model_dims=cfg.embedder.dimensions,
                llm={
                    "provider": "openai",
                    "config": {
                        "model": cfg.llm.model,
                        "temperature": cfg.llm.temperature,
                    },
                },
                embedder={
                    "provider": "openai",
                    "config": {"model": cfg.embedder.model},
                },
                max_embedding_input_tokens=cfg.embedder.max_input_tokens,
                max_embedding_input_chars=cfg.embedder.max_input_chars,
                history_db_path=cfg.history.db_path
                if cfg.history.enabled
                else None,
                version="v1.1",
            )

        vector_store_config = {
            "provider": vs.provider,
            "config": {
                "url": vs.qdrant_url
                if vs.provider == "qdrant"
                else vs.milvus_url,
                "collection_name": vs.collection_name,
                "embedding_model_dims": cfg.embedder.dimensions,
            },
        }

        return MemoryConfig(
            vector_store=vector_store_config,
            llm={
                "provider": "openai",
                "config": {
                    "model": cfg.llm.model,
                    "temperature": cfg.llm.temperature,
                },
            },
            embedder={
                "provider": "openai",
                "config": {"model": cfg.embedder.model},
            },
            max_embedding_input_tokens=cfg.embedder.max_input_tokens,
            max_embedding_input_chars=cfg.embedder.max_input_chars,
            history_db_path=cfg.history.db_path
            if cfg.history.enabled
            else None,
            version="v1.1",
        )

    # =========================================================================
    # STORAGE: store_exchange (the main storage pipeline)
    # =========================================================================

    async def store_exchange(
        self,
        query: str,
        response: str,
        user_id: str,
        tenant_id: Optional[str] = None,
        session_id: Optional[str] = None,
        investor_name: Optional[str] = None,
        agent_id: Optional[str] = None,
        dept_id: Optional[str] = None,
        custom_extraction_prompt: Optional[str] = None,
        force_investor_scope: bool = False,
        skip_dedup: bool = False,
    ) -> Dict[str, Any]:
        """Store a conversation exchange with full pipeline.

        Pipeline:
        1. LLM extracts facts with salience scores
        2. StorageGateway evaluates each fact (salience threshold + content length)
        3. Pre-store dedup checks for similar existing memories (unless skip_dedup)
        4. Only facts passing gate + dedup are stored

        Args:
            query: User's query.
            response: Assistant's response.
            user_id: User identifier.
            tenant_id: Tenant for multi-tenancy.
            session_id: Session identifier.
            investor_name: Entity name for memory isolation. When None, the
                EntityResolver hook is called as a fallback before extraction.
            agent_id: Agent identifier (uses config default if None).
            dept_id: Department identifier for team-pool routing. Overrides
                ``isolation.default_dept_id`` from config when provided.
            custom_extraction_prompt: Per-call extraction prompt override.
                When provided, replaces the default extraction prompt for
                this call only.

        Returns:
            Dict with facts_stored, facts_skipped, facts_extracted, investor_name.
        """
        cfg = self._config
        if not self.is_ready or not cfg.features.storage_enabled:
            return {"facts_stored": 0, "facts_skipped": 0, "results": []}

        if cfg.isolation.require_user_id and not user_id:
            return {"facts_stored": 0, "facts_skipped": 0, "results": []}
        if cfg.isolation.strict_tenant_isolation and not tenant_id:
            return {"facts_stored": 0, "facts_skipped": 0, "results": []}

        effective_agent_id = self._get_agent_id(agent_id)
        # GAP-2: per-call dept_id overrides config default
        effective_dept_id = dept_id or cfg.isolation.default_dept_id

        # Advisor-only mode: drop any client scope so every extracted fact is attributed
        # to the advisor (subject_type="advisor"). Skip entity resolution entirely.
        if cfg.isolation.advisor_only:
            investor_name = None
            force_investor_scope = False

        # GAP-1: EntityResolver fallback — call resolver before extraction
        # when investor_name was not explicitly provided
        if not investor_name and not cfg.isolation.advisor_only and self._entity_resolver:
            try:
                investor_name = await self._entity_resolver.resolve(query)
            except Exception as _er_exc:
                logger.warning(
                    f"[HighLevel] EntityResolver failed in store_exchange: {_er_exc}"
                )

        # Step 1: Extract facts
        if not self._extractor:
            return {"facts_stored": 0, "facts_skipped": 0, "results": []}

        # Gap 3b: make the LLM key-aware. For an advisor save (no investor scope), inject the
        # user's EXISTING profile_keys so a reworded restatement REUSES the same slot instead
        # of minting a new key — which is what makes UPSERT-by-slot reliably replace the value.
        effective_prompt = custom_extraction_prompt
        if (
            cfg.retrieval.inject_existing_keys
            and not investor_name
            and not force_investor_scope
        ):
            existing_keys = await self._existing_pref_keys(
                store_user_id=user_id, agent_id=effective_agent_id
            )
            if existing_keys:
                effective_prompt = self._augment_prompt_with_keys(
                    effective_prompt or self._extractor._build_system_prompt(),
                    existing_keys,
                )

        extraction = await self._extractor.extract(
            query=query,
            response=response,
            max_input_chars=cfg.storage.max_fact_extraction_chars,
            prompt=effective_prompt,
        )
        if extraction.error or not extraction.facts:
            return {
                "facts_stored": 0,
                "facts_skipped": 0,
                "results": [],
                "investor_name": investor_name,
            }

        # Resolve investor: explicit/resolved > LLM-extracted
        llm_investor = next(
            (f.investor_name for f in extraction.facts if f.investor_name),
            None,
        )
        if not investor_name:
            investor_name = llm_investor
        investor_name = MultiCategoryRetriever.normalize_investor_name(
            investor_name
        )

        results = []
        skipped = []
        stored_fact_texts: List[str] = []

        # Advisor-style facts describe HOW the logged-in user wants to work, not
        # an investor. They must never inherit the request-level investor name of
        # whatever client happened to be under discussion — otherwise an advisor
        # preference gets stamped with a client's name and leaks across clients.
        ADVISOR_CATEGORIES = {"preference", "procedural"}

        for fact in extraction.facts:
            fact_investor = MultiCategoryRetriever.normalize_investor_name(
                fact.investor_name
            )
            # Advisor-only mode: store ONLY the RM's/advisor's own data — DROP any fact
            # that is about a client. A fact is client-data when the extractor attributed
            # it to a named investor, OR it is a client-descriptive category (persona =
            # who the client is, episodic = events about the client). Only the advisor's
            # own preference/procedural facts are kept.
            if cfg.isolation.advisor_only and (
                fact_investor or fact.category not in ADVISOR_CATEGORIES
            ):
                skipped.append(
                    {
                        "category": fact.category,
                        "content": fact.content[:80],
                        "salience": fact.salience,
                        "reason": "advisor_only: client-scoped fact dropped",
                    }
                )
                continue
            # Fall back to the request-level investor when the extractor didn't attribute
            # the fact. Normally only client-attribute categories (persona/episodic) inherit
            # it — advisor prefs stay investor-less. BUT when the caller signals this exchange
            # is a CLIENT NOTE (force_investor_scope), the user is stating facts ABOUT a named
            # client, so preference/persona facts should attach to that client too (fixes
            # "store his investing preferences" attributing to the advisor instead of the client).
            if not fact_investor and (
                force_investor_scope or fact.category not in ADVISOR_CATEGORIES
            ):
                fact_investor = investor_name

            # Step 2: Salience gate
            if cfg.salience.enabled and self._gateway:
                gate = self._gateway.evaluate(
                    fact.content, salience=fact.salience
                )
                if gate.decision == StorageDecision.SKIP:
                    skipped.append(
                        {
                            "category": fact.category,
                            "content": fact.content[:80],
                            "salience": fact.salience,
                            "reason": gate.reason,
                        }
                    )
                    continue

            # Build metadata
            metadata = {
                "memory_type": fact.category,
                "stored_at": datetime.now(timezone.utc).strftime(
                    "%Y-%m-%d %H:%M UTC"
                ),
                "salience_score": fact.salience,
                # Summary layer: tag every point as a fact and stamp a recency
                # timestamp the summary rebuild sorts on (newest fact wins on
                # conflict). Namespaced to avoid Mem0's own created_at/updated_at.
                "type": "fact",
                "mem_updated_at": datetime.now(timezone.utc).isoformat(),
            }
            if fact.salience_reasoning and cfg.salience.store_reasoning:
                metadata["salience_reasoning"] = fact.salience_reasoning
            if fact.profile_key:
                metadata["profile_key"] = fact.profile_key
            if fact.detail:
                metadata["detail"] = fact.detail
            if fact.behavioral_note:
                metadata["behavioral_note"] = fact.behavioral_note
            if fact.trigger:
                metadata["trigger"] = fact.trigger
            if fact_investor:
                # Fact is about a specific investor/client.
                metadata["subject_type"] = "investor"
                metadata["investor_name"] = fact_investor
                subj = MultiCategoryRetriever.investor_name_to_subject_id(
                    fact_investor
                )
                if subj:
                    metadata["subject_id"] = subj
            else:
                # Fact is about the advisor (the logged-in user) themselves.
                metadata["subject_type"] = "advisor"
            if tenant_id:
                metadata["tenant_id"] = tenant_id
            if session_id:
                metadata["session_id"] = session_id

            # Pool assignment
            valid_pools = {"private", "shared", "team", "org"}
            fact_pool = (
                fact.pool
                if (fact.pool and fact.pool in valid_pools)
                else cfg.isolation.default_pool
            )
            metadata["pool"] = fact_pool
            if effective_dept_id:
                metadata["dept_id"] = effective_dept_id

            # TTL
            type_map = {
                "persona": MemoryType.PERSONA,
                "preference": MemoryType.PERSONA,
                "episodic": MemoryType.EPISODIC,
                "procedural": MemoryType.PROCEDURAL,
                "conversational": MemoryType.CONVERSATION,
                "feedback": MemoryType.FEEDBACK,
            }
            mem_type = type_map.get(fact.category)
            if mem_type:
                expiry = MemoryEntry.calculate_expiry(mem_type)
                if expiry:
                    metadata["expires_at"] = expiry.isoformat()

            # Importance from config
            type_cfg = cfg.memory_types.get(fact.category)
            if type_cfg:
                metadata["importance"] = type_cfg.importance

            # Effective user_id (sentinel for team/org via pool routing hook)
            store_user_id = user_id
            if self._pool_routing:
                routed = self._pool_routing.resolve_store_params(
                    fact_pool=fact_pool,
                    user_id=user_id,
                    agent_id=effective_agent_id,
                    tenant_id=tenant_id,
                    dept_id=effective_dept_id,
                    metadata=metadata,
                )
                store_user_id = routed["user_id"]
                effective_agent_id = routed.get("agent_id", effective_agent_id)
                metadata = routed.get("metadata", metadata)
            else:
                # No pool routing — default to private
                pool = metadata.get("pool", "private")
                if pool == "team":
                    store_user_id = (
                        f"__team_{metadata.get('dept_id', 'default')}__"
                    )
                elif pool == "org":
                    store_user_id = (
                        f"__org_{metadata.get('tenant_id', 'default')}__"
                    )

            # Gap 3: conservative slot UPSERT — for an advisor preference/procedural item
            # carrying a profile_key, supersede any existing advisor item in the SAME slot
            # AND category so an updated value replaces the old one (rather than coexisting
            # as a contradictory duplicate). Runs before dedup; client memories and other
            # categories are untouched.
            if (
                cfg.retrieval.preference_upsert_by_slot
                and fact.category in ("preference", "procedural")
                and not fact_investor
                and fact.profile_key
            ):
                await self._delete_advisor_pref_in_slot(
                    store_user_id, effective_agent_id, fact.profile_key, fact.category
                )

            # Step 3: Dedup (skipped for explicit box-saves where UPSERT-by-slot
            # already prevents duplicates — avoids false cross-scope matches)
            fact_to_store = fact.content
            if self._deduplicator and cfg.dedup.enabled and not skip_dedup:
                try:
                    dedup_filters = (
                        {"tenant_id": tenant_id} if tenant_id else {}
                    )
                    # Scope dedup to same subject_type so advisor saves
                    # don't get rejected by similar investor memories.
                    dedup_filters["subject_type"] = (
                        "investor" if fact_investor else "advisor"
                    )
                    decision = await self._deduplicator.find_and_resolve(
                        new_fact=fact.content,
                        category=fact.category,
                        investor_name=fact_investor,
                        user_id=store_user_id,
                        agent_id=effective_agent_id,
                        search_fn=self._connector.search,
                        metadata_filters=dedup_filters or None,
                    )
                    if self._telemetry:
                        self._telemetry.on_dedup(
                            decision.action, fact.category, fact_investor
                        )

                    if decision.action == "KEEP_EXISTING":
                        skipped.append(
                            {
                                "category": fact.category,
                                "content": fact.content[:80],
                                "reason": f"dedup:KEEP_EXISTING — {decision.reason}",
                            }
                        )
                        continue
                    if (
                        decision.action == "REPLACE"
                        and decision.existing_memory_id
                    ):
                        await self._connector.delete(
                            decision.existing_memory_id
                        )
                        if decision.updated_memory:
                            fact_to_store = decision.updated_memory
                    elif (
                        decision.action == "MERGE"
                        and decision.existing_memory_id
                    ):
                        if decision.updated_memory:
                            await self._connector.delete(
                                decision.existing_memory_id
                            )
                            fact_to_store = decision.updated_memory
                except Exception as e:
                    logger.warning(
                        f"[HighLevel] Dedup failed: {e}, storing as new"
                    )

            # Step 4: Store
            try:
                result = await self._connector.store_fact(
                    fact=fact_to_store,
                    user_id=store_user_id,
                    agent_id=effective_agent_id,
                    categories=[fact.category],
                    metadata=metadata,
                )
                if result.get("results"):
                    results.extend(result["results"])
                stored_fact_texts.append(fact_to_store)
            except Exception as e:
                logger.warning(f"[HighLevel] Store failed: {e}")

        # Per-user preference summary: REBUILD it from the user's CURRENT facts
        # (not an incremental merge of just this turn's facts). The facts are the
        # deduped source of truth — rebuilding keeps the summary in lockstep with
        # them, so a fact that was superseded/cleared/UPSERT-replaced can never
        # linger as a stale line. Best-effort; never let a summary failure affect
        # fact storage. Keyed on the original user_id (the summary belongs to the
        # user, not a pool-routed sentinel).
        if self._summary is not None and stored_fact_texts:
            try:
                await self._summary.rebuild_summary(user_id, extra_facts=stored_fact_texts)
            except Exception as e:
                logger.warning(f"[HighLevel] Summary rebuild failed: {e}")

        if self._telemetry:
            self._telemetry.on_storage(
                len(results),
                len(skipped),
                len(extraction.facts),
                investor_name,
            )

        return {
            "facts_stored": len(results),
            "facts_skipped": len(skipped),
            "facts_extracted": len(extraction.facts),
            "results": results,
            "skipped": skipped,
            "investor_name": investor_name,
        }

    # =========================================================================
    # RETRIEVAL: preference scroll (Gap 2)
    # =========================================================================

    @staticmethod
    def _pref_field(item: Dict[str, Any], key: str, default: Any = None) -> Any:
        """Read a metadata field from a get_all record, tolerating both shapes
        (nested under 'metadata' or flattened onto the item itself)."""
        meta = item.get("metadata") or {}
        if key in meta:
            return meta[key]
        return item.get(key, default)

    async def _fetch_all_preferences(
        self,
        user_id: str,
        agent_id: str,
        investor_name: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Fetch the COMPLETE preference set by metadata scroll (not semantic search).

        Returns advisor (RM) preferences plus, when a client is in scope, that client's
        preferences — each ordered by salience desc then recency, and capped. Records are
        normalised to the shape ProfileAssembler expects (content + metadata dict with
        memory_type / subject_type / investor_name).
        """
        cfg = self._config
        result = await self._connector.get_all(
            user_id=user_id, agent_id=agent_id, limit=500
        )
        records = result.get("results", []) or []

        target_investor = MultiCategoryRetriever.normalize_investor_name(
            investor_name
        )

        advisor: List[Dict[str, Any]] = []
        client: List[Dict[str, Any]] = []
        for item in records:
            if self._pref_field(item, "memory_type") != "preference":
                continue
            subject = self._pref_field(item, "subject_type")
            inv = MultiCategoryRetriever.normalize_investor_name(
                self._pref_field(item, "investor_name")
            )
            content = (
                item.get("memory")
                or item.get("data")
                or item.get("content")
                or ""
            )
            if not content:
                continue
            normalised = {
                "memory": content,
                "metadata": {
                    "memory_type": "preference",
                    "subject_type": subject,
                    "investor_name": inv,
                    "profile_key": self._pref_field(item, "profile_key"),
                    "salience_score": self._pref_field(item, "salience_score", 0.5),
                    "stored_at": self._pref_field(item, "stored_at", ""),
                    "detail": self._pref_field(item, "detail", ""),
                },
                "_retrieval_reason": "preference",
            }
            if subject == "advisor":
                advisor.append(normalised)
            elif subject == "investor" and target_investor and inv == target_investor:
                # Only the in-scope client's preferences — never leak another client's.
                client.append(normalised)

        def _rank(mems: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
            return sorted(
                mems,
                key=lambda m: (
                    m["metadata"].get("salience_score") or 0.0,
                    m["metadata"].get("stored_at") or "",
                ),
                reverse=True,
            )

        advisor = _rank(advisor)[: cfg.retrieval.preference_advisor_cap]
        client = _rank(client)[: cfg.retrieval.preference_investor_cap]
        return advisor + client

    async def _fetch_advisor_procedural(
        self,
        user_id: str,
        agent_id: str,
        investor_name: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Fetch procedural workflows by metadata scroll (not semantic search).

        Advisor procedural items (investor_name=null) are the RM's reusable operating
        procedures and apply to EVERY client query — but semantic retrieval filters by
        investor_name when a client is in scope, which would exclude them. Scroll them in
        directly so workflows always reach generation. Includes the in-scope client's own
        procedural items too, if any. Carries the 'trigger' field the formatter renders.
        """
        cfg = self._config
        result = await self._connector.get_all(
            user_id=user_id, agent_id=agent_id, limit=500
        )
        records = result.get("results", []) or []
        target = MultiCategoryRetriever.normalize_investor_name(investor_name)

        advisor: List[Dict[str, Any]] = []
        client: List[Dict[str, Any]] = []
        for item in records:
            if self._pref_field(item, "memory_type") != "procedural":
                continue
            subject = self._pref_field(item, "subject_type")
            inv = MultiCategoryRetriever.normalize_investor_name(
                self._pref_field(item, "investor_name")
            )
            content = (
                item.get("memory")
                or item.get("data")
                or item.get("content")
                or ""
            )
            if not content:
                continue
            normalised = {
                "memory": content,
                "metadata": {
                    "memory_type": "procedural",
                    "subject_type": subject,
                    "investor_name": inv,
                    "profile_key": self._pref_field(item, "profile_key"),
                    "salience_score": self._pref_field(item, "salience_score", 0.5),
                    "stored_at": self._pref_field(item, "stored_at", ""),
                    "detail": self._pref_field(item, "detail", ""),
                    "trigger": self._pref_field(item, "trigger", ""),
                },
                "_retrieval_reason": "procedural",
            }
            if subject == "advisor":
                advisor.append(normalised)
            elif subject == "investor" and target and inv == target:
                client.append(normalised)

        def _rank(mems: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
            return sorted(
                mems,
                key=lambda m: (
                    m["metadata"].get("salience_score") or 0.0,
                    m["metadata"].get("stored_at") or "",
                ),
                reverse=True,
            )

        advisor = _rank(advisor)[: cfg.retrieval.preference_advisor_cap]
        client = _rank(client)[: cfg.retrieval.preference_investor_cap]
        return advisor + client

    async def _delete_advisor_pref_in_slot(
        self, user_id: str, agent_id: str, profile_key: str, category: str
    ) -> int:
        """Gap 3 conservative UPSERT: delete existing advisor memories that occupy the SAME
        slot (profile_key) in the SAME category for this RM, so a new save/value supersedes
        the old one. Only touches advisor memories with an EXACT category + profile_key match
        — never another slot, another category, or a client's memory. Returns count deleted."""
        if not profile_key:
            return 0
        try:
            res = await self._connector.get_all(
                user_id=user_id, agent_id=agent_id, limit=500
            )
        except Exception as _exc:
            logger.warning(f"[HighLevel] UPSERT slot lookup failed: {_exc}")
            return 0
        deleted = 0
        for item in res.get("results", []) or []:
            if self._pref_field(item, "memory_type") != category:
                continue
            if self._pref_field(item, "subject_type") != "advisor":
                continue
            if (self._pref_field(item, "profile_key") or "") != profile_key:
                continue
            mid = item.get("id")
            if mid:
                try:
                    await self._connector.delete(mid)
                    deleted += 1
                except Exception as _del_exc:
                    logger.warning(
                        f"[HighLevel] UPSERT delete failed for {mid}: {_del_exc}"
                    )
        return deleted

    async def _existing_pref_keys(self, store_user_id: str, agent_id: str) -> List[str]:
        """Gap 3b: distinct profile_keys already stored for this advisor's preference/
        procedural memories. Fed into the extraction prompt so a reworded restatement of
        the same preference reuses the existing slot instead of minting a new key. Scoped
        to advisor subject only (never a client's keys). Cheap metadata scroll, no LLM."""
        cap = self._config.retrieval.inject_existing_keys_cap
        try:
            res = await self._connector.get_all(
                user_id=store_user_id, agent_id=agent_id, limit=500
            )
        except Exception as _exc:
            logger.warning(f"[HighLevel] existing-keys lookup failed: {_exc}")
            return []
        seen: List[str] = []
        for item in res.get("results", []) or []:
            if self._pref_field(item, "subject_type") != "advisor":
                continue
            if self._pref_field(item, "memory_type") not in ("preference", "procedural"):
                continue
            key = (self._pref_field(item, "profile_key") or "").strip()
            if key and key not in seen:
                seen.append(key)
                if len(seen) >= cap:
                    break
        return seen

    @staticmethod
    def _augment_prompt_with_keys(prompt: str, existing_keys: List[str]) -> str:
        """Append the user's existing profile_keys to the extraction prompt with a reuse
        instruction. Keeps the slot stable across LLM wording drift."""
        keys_block = "\n".join(f"- {k}" for k in existing_keys)
        return (
            f"{prompt}\n\n"
            "## EXISTING KEYS for this user\n"
            "If a new item is the SAME dimension as one of these, REUSE that EXACT key "
            "(so the updated value supersedes the old one). Only mint a NEW "
            "lowercase_snake_case key if the item is a genuinely different dimension:\n"
            f"{keys_block}"
        )

    # =========================================================================
    # RETRIEVAL: retrieve_context
    # =========================================================================

    async def retrieve_context(
        self,
        query: str,
        user_id: str,
        agent_id: Optional[str] = None,
        tenant_id: Optional[str] = None,
        session_id: Optional[str] = None,
        investor_name: Optional[str] = None,
        dept_id: Optional[str] = None,
        client_recall_only: bool = False,
    ) -> Dict[str, Any]:
        """Retrieve memory context with multi-category strategy.

        Args:
            query: The search query / current user message.
            user_id: User identifier.
            agent_id: Agent identifier (uses config default if None).
            tenant_id: Tenant for multi-tenancy.
            session_id: Session identifier (informational only for retrieval).
            investor_name: Entity name filter. When None, the EntityResolver
                hook is called as a fallback.
            dept_id: Department identifier for team-pool routing. Overrides
                ``isolation.default_dept_id`` from config when provided.

        Returns:
            Dict with keys: context, memories, by_category, structured_profile,
            count, user_id, agent_id, investor_name.
        """
        cfg = self._config
        effective_agent_id = self._get_agent_id(agent_id)
        empty = {
            "context": "",
            "memories": [],
            "by_category": {},
            "structured_profile": {},
            "count": 0,
            "user_id": user_id,
            "agent_id": effective_agent_id,
            "investor_name": investor_name,
        }

        if not self.is_ready or not cfg.features.retrieval_enabled:
            return empty
        if cfg.isolation.require_user_id and not user_id:
            return empty
        if cfg.isolation.strict_tenant_isolation and not tenant_id:
            return empty

        # GAP-3: per-call dept_id overrides config default
        effective_dept_id = dept_id or cfg.isolation.default_dept_id

        # Advisor-only mode: ignore client scope entirely. With investor_name forced to
        # None the advisor self-recall filter below strips all investor-scoped memories,
        # so only advisor-owned memories reach generation.
        if cfg.isolation.advisor_only:
            investor_name = None
            client_recall_only = False

        # GAP-1: EntityResolver fallback when investor_name not provided
        if not investor_name and not cfg.isolation.advisor_only and self._entity_resolver:
            try:
                investor_name = await self._entity_resolver.resolve(query)
            except Exception as _er_exc:
                logger.warning(
                    f"[HighLevel] EntityResolver failed in retrieve_context: {_er_exc}"
                )

        investor_name = MultiCategoryRetriever.normalize_investor_name(
            investor_name
        )
        # Reflect resolved/normalized investor_name in the empty fallback dict
        # so early-return paths (no retrieval results) still surface it.
        empty["investor_name"] = investor_name

        # Run multi-category retrieval
        retrieval_result = await self._retriever.retrieve(
            query=query,
            search_fn=self._connector.search,
            user_id=user_id,
            agent_id=effective_agent_id,
            investor_name=investor_name,
            tenant_id=tenant_id,
            allow_cross_agent=cfg.isolation.allow_cross_agent_sharing,
            dept_id=effective_dept_id,
            pool_routing=self._pool_routing,
        )

        memories = list(retrieval_result.memories)
        by_category = dict(retrieval_result.by_category)

        # Summary-driven retrieval: the per-user summary REPLACES the preference +
        # procedural full-fetch (it already blends both for the RM into one block).
        # Falls back to the fact scrolls below when the summary is empty.
        summary_text = ""
        if cfg.retrieval.preference_from_summary and self._summary is not None:
            try:
                summary_text = (await self._summary.get_summary(user_id) or "").strip()
            except Exception as _sum_exc:
                logger.warning(
                    f"[HighLevel] get_summary failed, falling back to fact scroll: {_sum_exc}"
                )
                summary_text = ""
        use_summary = bool(summary_text)

        # Gap 2: preferences are constraints that must ALL apply, so fetch the COMPLETE set
        # by metadata scroll instead of trusting semantic top-k (which can drop a preference
        # whose wording is unrelated to the current question — e.g. "never recommend crypto"
        # on a query about index funds). Replace any semantically-retrieved preferences with
        # the scrolled set so they appear once and in full.
        if cfg.retrieval.preference_full_fetch and not use_summary:
            try:
                pref_memories = await self._fetch_all_preferences(
                    user_id=user_id,
                    agent_id=effective_agent_id,
                    investor_name=investor_name,
                )
                # Drop semantic preferences (avoid double-counting), keep other categories.
                memories = [
                    m for m in memories
                    if (m.get("metadata", {}) or {}).get("memory_type") != "preference"
                ]
                by_category.pop("preference", None)
                if pref_memories:
                    memories.extend(pref_memories)
                    by_category["preference"] = pref_memories
            except Exception as _pref_exc:
                logger.warning(
                    f"[HighLevel] Preference scroll failed, falling back to semantic: {_pref_exc}"
                )

        # Procedural workflows are advisor-owned (investor_name=null) and apply to every client
        # query, but semantic retrieval filters them out when a client is in scope. Scroll them
        # in the same way preferences are, so a saved workflow always reaches generation.
        if cfg.retrieval.procedural_full_fetch and not use_summary:
            try:
                proc_memories = await self._fetch_advisor_procedural(
                    user_id=user_id,
                    agent_id=effective_agent_id,
                    investor_name=investor_name,
                )
                # Drop semantic procedural (avoid double-counting), keep other categories.
                memories = [
                    m for m in memories
                    if (m.get("metadata", {}) or {}).get("memory_type") != "procedural"
                ]
                by_category.pop("procedural", None)
                if proc_memories:
                    memories.extend(proc_memories)
                    by_category["procedural"] = proc_memories
            except Exception as _proc_exc:
                logger.warning(
                    f"[HighLevel] Procedural scroll failed, falling back to semantic: {_proc_exc}"
                )

        # Advisor self-recall (no investor in scope): strip ALL client-scoped
        # memories so persona/episodic facts about clients don't leak into the
        # advisor's own recall and get mis-attributed.
        if not investor_name:
            memories = [
                m for m in memories
                if (m.get("metadata", {}) or {}).get("subject_type") != "investor"
            ]
            by_category = {
                cat: [
                    m for m in mems
                    if (m.get("metadata", {}) or {}).get("subject_type") != "investor"
                ]
                for cat, mems in by_category.items()
            }
            by_category = {k: v for k, v in by_category.items() if v}

        # Client recall ("What do you remember about Rohan?"): strip advisor-scoped
        # memories so the advisor's own working-style prefs (concise, Nifty 500) don't
        # appear as if they belong to the client.
        if client_recall_only and investor_name:
            memories = [
                m for m in memories
                if (m.get("metadata", {}) or {}).get("subject_type") != "advisor"
            ]
            by_category = {
                cat: [
                    m for m in mems
                    if (m.get("metadata", {}) or {}).get("subject_type") != "advisor"
                ]
                for cat, mems in by_category.items()
            }
            by_category = {k: v for k, v in by_category.items() if v}

        # Nothing to return only when there are neither memories NOR a summary block.
        if not memories and not use_summary:
            return empty

        # Assemble profile + format context from the (non-summary) memories.
        profile: Dict[str, Any] = {}
        context = ""
        if memories:
            profile = self._assembler.assemble(
                memories, entity_name=investor_name
            )

            # Extend profile with pool-tier sections if pool routing is active
            if self._pool_routing:
                profile = self._pool_routing.extend_profile(
                    profile, by_category
                )

            # Format context
            context = self._formatter.format(profile, entity_name=investor_name)

            # Append pool-tier sections if pool routing is active
            if self._pool_routing:
                pool_sections = self._pool_routing.format_pool_sections(
                    profile, entity_name=investor_name
                )
                if pool_sections:
                    context = context + "\n" + pool_sections

        # Prepend the summary block when summary-driven retrieval is active. This is
        # the RM's complete durable preference/workflow profile (replaces the scrolls).
        if use_summary:
            summary_block = f"---RM PREFERENCE PROFILE---\n{summary_text}"
            context = f"{summary_block}\n\n{context}".strip() if context else summary_block

        if self._telemetry:
            self._telemetry.on_retrieval(
                query,
                len(memories),
                {k: len(v) for k, v in by_category.items()},
                investor_name,
            )

        return {
            "context": context,
            "memories": memories,
            "by_category": by_category,
            "structured_profile": profile,
            "count": len(memories),
            "user_id": user_id,
            "agent_id": effective_agent_id,
            "investor_name": investor_name,
        }

    # =========================================================================
    # BUFFERING: buffer_exchange / flush_session
    # =========================================================================

    async def buffer_exchange(
        self,
        query: str,
        response: str,
        user_id: str,
        tenant_id: Optional[str] = None,
        session_id: Optional[str] = None,
        investor_name: Optional[str] = None,
        agent_id: Optional[str] = None,
        dept_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Buffer an exchange. Auto-flushes when threshold reached or idle timeout fires.

        Args:
            dept_id: Department identifier for team-pool routing. Overrides
                ``isolation.default_dept_id`` from config when provided.
        """
        if not self._buffer:
            return await self.store_exchange(
                query=query,
                response=response,
                user_id=user_id,
                tenant_id=tenant_id,
                session_id=session_id,
                investor_name=investor_name,
                agent_id=agent_id,
                dept_id=dept_id,
            )

        ready = await self._buffer.add_exchange(
            query=query,
            response=response,
            user_id=user_id,
            tenant_id=tenant_id,
            session_id=session_id,
            investor_name=investor_name,
            agent_id=agent_id,
            dept_id=dept_id,
        )
        if ready:
            asyncio.create_task(
                self._flush_exchanges(ready, user_id, session_id, agent_id)
            )
            return {
                "buffered": False,
                "flushing": True,
                "exchanges_flushed": len(ready),
            }

        return {
            "buffered": True,
            "flushing": False,
            "buffer_size": self._buffer.get_buffer_size(user_id, session_id),
        }

    async def flush_session_buffer(
        self,
        user_id: str,
        session_id: Optional[str] = None,
        agent_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Flush remaining buffered exchanges for a session."""
        if not self._buffer:
            return {"flushed": 0}
        exchanges = await self._buffer.flush_session(user_id, session_id)
        if not exchanges:
            return {"flushed": 0}
        result = await self._flush_exchanges(
            exchanges, user_id, session_id, agent_id
        )
        self._buffer.cleanup_session(user_id, session_id)
        return result

    async def _flush_exchanges(
        self,
        exchanges: List[BufferedExchange],
        user_id: str,
        session_id: Optional[str],
        agent_id: Optional[str],
    ) -> Dict[str, Any]:
        if not exchanges:
            return {"flushed": 0}
        text = self._buffer.consolidate_exchanges(exchanges)
        investor = next(
            (e.investor_name for e in reversed(exchanges) if e.investor_name),
            None,
        )
        tenant = next(
            (e.tenant_id for e in reversed(exchanges) if e.tenant_id), None
        )
        # Resolve agent_id from exchanges if not explicitly provided
        if not agent_id:
            agent_id = next(
                (e.agent_id for e in reversed(exchanges) if e.agent_id), None
            )
        # GAP-4: propagate dept_id from buffered exchanges
        dept_id = next(
            (e.dept_id for e in reversed(exchanges) if e.dept_id), None
        )
        try:
            result = await self.store_exchange(
                query="[Consolidated session transcript]",
                response=text,
                user_id=user_id,
                tenant_id=tenant,
                session_id=session_id,
                investor_name=investor,
                agent_id=agent_id,
                dept_id=dept_id,
            )
            result["exchanges_flushed"] = len(exchanges)
            return result
        except Exception as e:
            logger.error(f"[HighLevel] Flush failed: {e}")
            return {"flushed": 0, "error": str(e)}

    async def _on_idle_flush(
        self, session_key: str, exchanges: List[BufferedExchange]
    ) -> None:
        """Callback invoked by ExchangeBuffer when idle timeout fires."""
        if not exchanges:
            return
        # Extract routing info from the buffered exchanges
        user_id = exchanges[0].user_id
        session_id = exchanges[0].session_id
        agent_id = next(
            (e.agent_id for e in reversed(exchanges) if e.agent_id), None
        )
        logger.info(
            f"[HighLevel] Idle flush triggered for session {session_key} "
            f"({len(exchanges)} exchanges)"
        )
        await self._flush_exchanges(exchanges, user_id, session_id, agent_id)

    # =========================================================================
    # LOW-LEVEL CRUD (delegate to NeoMemoryConnector)
    # =========================================================================

    def _enforce_isolation(
        self, user_id: Optional[str], tenant_id: Optional[str] = None
    ) -> None:
        """Raise ValueError if isolation policies are violated."""
        cfg = self._config.isolation
        if cfg.require_user_id and not user_id:
            raise ValueError(
                "user_id is required (isolation.require_user_id=True)"
            )
        if cfg.strict_tenant_isolation and not tenant_id:
            raise ValueError(
                "tenant_id is required (isolation.strict_tenant_isolation=True)"
            )

    async def search(
        self,
        query: str,
        *,
        user_id: str,
        agent_id: Optional[str] = None,
        tenant_id: Optional[str] = None,
        investor_name: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        self._enforce_isolation(user_id, tenant_id)
        effective_agent_id = self._get_agent_id(agent_id)
        # Inject tenant_id and investor_name into metadata_filters
        mf = kwargs.pop("metadata_filters", None) or {}
        if tenant_id:
            mf["tenant_id"] = tenant_id
        if investor_name:
            mf["investor_name"] = MultiCategoryRetriever.normalize_investor_name(
                investor_name
            )
        if mf:
            kwargs["metadata_filters"] = mf
        return await self._connector.search(
            query, user_id=user_id, agent_id=effective_agent_id, **kwargs
        )

    async def get_summary(self, user_id: str) -> str:
        """Return the user's preference summary text, or "" if none/disabled.

        Deterministic-ID retrieve, no vector search. The summary is built from
        the user's facts and updated on each store_exchange (recency wins)."""
        if self._summary is None:
            return ""
        return await self._summary.get_summary(user_id)

    async def rebuild_summary(self, user_id: str) -> str:
        """Rebuild the user's summary from scratch over ALL their facts.

        Scrolls every type="fact" point for the user, sorts oldest→newest, and
        regenerates the summary (later facts override earlier conflicts). Returns
        the new summary text (or "" if disabled/no facts)."""
        if self._summary is None:
            return ""
        return await self._summary.rebuild_summary(user_id)

    async def get(self, memory_id: str) -> Optional[Dict[str, Any]]:
        return await self._connector.get(memory_id)

    async def get_all(
        self,
        *,
        user_id: str,
        agent_id: Optional[str] = None,
        tenant_id: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        self._enforce_isolation(user_id, tenant_id)
        effective_agent_id = self._get_agent_id(agent_id)
        return await self._connector.get_all(
            user_id=user_id, agent_id=effective_agent_id, **kwargs
        )

    async def update(self, memory_id: str, data: str) -> Dict[str, Any]:
        return await self._connector.update(memory_id, data=data)

    async def delete(self, memory_id: str, *, user_id: Optional[str] = None) -> Dict[str, Any]:
        """Delete a single memory by id.

        When the summary layer is on, the user's summary is rebuilt from the
        REMAINING facts afterward so a deleted preference stops being applied.
        ``user_id`` is needed to scope the rebuild; if not passed we read it from
        the point's payload before deleting (best-effort)."""
        # Resolve the owning user_id (for summary rebuild) before the point is gone.
        rebuild_uid = user_id
        if self._summary is not None and not rebuild_uid:
            try:
                pt = await self._connector.get(memory_id)
                if pt:
                    rebuild_uid = pt.get("user_id") or (pt.get("metadata", {}) or {}).get("user_id")
            except Exception:
                rebuild_uid = None

        result = await self._connector.delete(memory_id)

        if self._summary is not None and rebuild_uid:
            try:
                await self._summary.rebuild_summary(rebuild_uid)
            except Exception as e:
                logger.warning(f"[HighLevel] summary rebuild after delete failed: {e}")
        return result

    async def delete_all(
        self,
        *,
        user_id: str,
        agent_id: Optional[str] = None,
        tenant_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Delete all memories for a user/agent scope."""
        self._enforce_isolation(user_id, tenant_id)
        effective_agent_id = self._get_agent_id(agent_id)
        result = await self._connector.delete_all(
            user_id=user_id, agent_id=effective_agent_id
        )
        # Facts are gone → drop the derived summary point too (else it orphans
        # and keeps surfacing cleared preferences at retrieval time).
        if self._summary is not None:
            try:
                await self._summary.delete_summary(user_id)
            except Exception as e:
                logger.warning(f"[HighLevel] summary delete after clear-all failed: {e}")
        return result

    async def delete_investor_memories(
        self,
        *,
        user_id: str,
        agent_id: Optional[str] = None,
        tenant_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Purge all client-scoped (subject_type="investor") memories for a user/agent.

        Advisor-owned memories are left untouched. Used to clean up existing client
        memories when switching to advisor-only mode. Returns {"deleted": <count>}.
        """
        self._enforce_isolation(user_id, tenant_id)
        effective_agent_id = self._get_agent_id(agent_id)
        res = await self._connector.get_all(
            user_id=user_id, agent_id=effective_agent_id, limit=10000
        )
        deleted = 0
        for item in res.get("results", []) or []:
            if self._pref_field(item, "subject_type") != "investor":
                continue
            mid = item.get("id")
            if not mid:
                continue
            try:
                await self._connector.delete(mid)
                deleted += 1
            except Exception as _del_exc:
                logger.warning(
                    f"[HighLevel] delete_investor_memories failed for {mid}: {_del_exc}"
                )
        return {"deleted": deleted}

    async def store_fact(
        self,
        fact: str,
        *,
        user_id: str,
        agent_id: Optional[str] = None,
        tenant_id: Optional[str] = None,
        investor_name: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        self._enforce_isolation(user_id, tenant_id)
        effective_agent_id = self._get_agent_id(agent_id)
        # Inject tenant_id and investor_name into metadata
        meta = kwargs.pop("metadata", None) or {}
        if tenant_id:
            meta["tenant_id"] = tenant_id
        if investor_name:
            meta["investor_name"] = MultiCategoryRetriever.normalize_investor_name(
                investor_name
            )
        if meta:
            kwargs["metadata"] = meta
        return await self._connector.store_fact(
            fact, user_id=user_id, agent_id=effective_agent_id, **kwargs
        )

    # =========================================================================
    # HISTORY
    # =========================================================================

    async def history(self, memory_id: str) -> List[Dict[str, Any]]:
        if self._history:
            return await self._history.get_memory_history(memory_id)
        return []

    def get_all_history(
        self, limit: int = 100, event_filter: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        if self._history:
            return self._history.get_all_history(limit, event_filter)
        return []

    def get_history_stats(self) -> Dict[str, Any]:
        if self._history:
            return self._history.get_stats()
        return {"total": 0, "history_enabled": False}

    # =========================================================================
    # LIFECYCLE
    # =========================================================================

    async def cleanup_expired(
        self, user_id: Optional[str] = None, **kwargs
    ) -> Dict[str, Any]:
        if self._connector:
            return await self._connector.cleanup_expired(
                user_id=user_id, **kwargs
            )
        return {}

    async def count_memories(
        self, user_id: Optional[str] = None, **kwargs
    ) -> Dict[str, Any]:
        if self._connector:
            return await self._connector.count_memories(
                user_id=user_id, **kwargs
            )
        return {"total": 0}
