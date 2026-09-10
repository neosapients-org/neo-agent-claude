"""Comprehensive configuration for neo_memory_hub.

Supports loading from:
- YAML file with ${ENV_VAR:-default} resolution
- Python dict
- Programmatic construction
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class VectorStoreConfig(BaseModel):
    """Vector store backend configuration."""

    provider: str = "qdrant"
    collection_name: str = "memories"
    qdrant_url: str = "http://localhost:6333"
    milvus_url: str = "http://localhost:19530"
    pgvector_url: Optional[str] = None
    embedding_dims: int = 1536


class LLMConfig(BaseModel):
    """LLM for memory operations (extraction, dedup)."""

    model: str = "gpt-4o-mini"
    temperature: float = 0.1


class EmbedderConfig(BaseModel):
    """Embedding model configuration."""

    model: str = "text-embedding-3-small"
    dimensions: int = 1536
    max_input_tokens: int = 7000
    max_input_chars: int = 40000


class MemoryTypeConfig(BaseModel):
    """Per-type configuration."""

    importance: float = 5.0
    categories: List[str] = Field(default_factory=list)
    description: str = ""


class RetrievalConfig(BaseModel):
    """Retrieval tuning."""

    limit: int = 10
    min_relevance_score: float = 0.1
    thresholds: Dict[str, float] = Field(
        default_factory=lambda: {
            "persona": 0.20,
            "preference": 0.15,
            "episodic": 0.20,
            "procedural": 0.30,
        }
    )
    investor_aware_thresholds: Dict[str, float] = Field(
        default_factory=lambda: {
            "persona": 0.03,
            "preference": 0.03,
            "episodic": 0.03,
            "procedural": 0.15,
        }
    )
    query_hints: Dict[str, str] = Field(
        default_factory=lambda: {
            "persona": "personality behavior risk profile temperament background",
            "preference": "communication preferences reporting style excluded sectors",
            "episodic": "events decisions moments meetings calls updates",
            "procedural": "reminders alerts procedures tasks notifications recurring",
        }
    )
    profile_categories: List[str] = Field(
        default_factory=lambda: ["persona", "preference"]
    )
    # Gap 2 — preferences are CONSTRAINTS that should ALL apply, so they are fetched by a
    # metadata scroll (complete set), not semantic top-k (which can silently drop one whose
    # wording doesn't match the query). Caps are a safety guardrail against runaway growth;
    # truncation, if it ever triggers, drops lowest-salience-then-oldest first.
    preference_full_fetch: bool = True
    preference_advisor_cap: int = 25   # max RM/advisor preferences injected
    preference_investor_cap: int = 15  # max per in-scope client
    # Gap 3 — conservative UPSERT: a newly-stored advisor preference REPLACES any existing
    # advisor preference in the SAME slot (profile_key), so updating a preference (e.g.
    # benchmark Nifty 50 -> Nifty 500) supersedes the old value instead of accumulating a
    # contradictory duplicate. Different slots are always kept; never deletes on no-match.
    preference_upsert_by_slot: bool = True
    # Gap 3b — before extracting an advisor save, inject the user's EXISTING profile_keys
    # into the extraction prompt so the LLM REUSES the same slot for a reworded restatement
    # of the same preference (e.g. "low-risk" -> "moderate risk") instead of minting a new
    # key. Makes the exact-match UPSERT-by-slot above reliable across LLM wording drift.
    # Dynamic + self-reconciling: the vocabulary IS the user's current keys, so it grows
    # with new preferences while collapsing restatements onto the existing slot.
    inject_existing_keys: bool = True
    inject_existing_keys_cap: int = 40  # max existing keys listed in the prompt
    # Advisor procedural workflows (investor_name=null) are reusable operating procedures that
    # apply to EVERY client query. Semantic retrieval filters by investor_name when a client is
    # in scope, which excludes them — so, like preferences, fetch them by metadata scroll so a
    # workflow always reaches generation regardless of which client is being asked about.
    procedural_full_fetch: bool = True
    # Summary-driven retrieval: when True (and the summary layer is enabled), the per-user
    # preference SUMMARY REPLACES both the preference and procedural full-fetch — the summary
    # already blends both for the RM into one compact block. Falls back to the fact scrolls
    # when the summary is empty (e.g. a brand-new RM, or one whose summary hasn't been built/
    # rebuilt yet). Trades the exactness of the per-fact scroll for a compact, coherent profile.
    preference_from_summary: bool = False


class StorageConfig(BaseModel):
    """Storage limits."""

    max_context_chars: int = 8000
    max_fact_extraction_chars: int = 12000
    max_episodic_response_chars: int = 3000


class SalienceConfig(BaseModel):
    """Salience gating."""

    enabled: bool = True
    min_threshold: float = 0.3
    default_score: float = 0.5
    min_content_length: int = 3
    max_content_length: int = 10000
    store_reasoning: bool = True


class ExtractionConfig(BaseModel):
    """Fact extraction pipeline config."""

    enabled: bool = True
    prompt: Optional[str] = None
    valid_categories: List[str] = Field(
        default_factory=lambda: ["persona", "preference", "episodic", "procedural"]
    )


class DedupConfig(BaseModel):
    """Pre-store deduplication config."""

    enabled: bool = True
    similarity_threshold: float = 0.55
    max_candidates: int = 5
    model: Optional[str] = None  # Falls back to llm.model
    temperature: float = 0.1
    prompt: Optional[str] = None


class BufferingConfig(BaseModel):
    """Exchange buffering config."""

    enabled: bool = True
    flush_threshold: int = 10
    max_buffer_chars: int = 50000
    flush_on_session_end: bool = True
    idle_timeout_seconds: float = 60.0


class IsolationConfig(BaseModel):
    """Memory isolation / multi-tenancy."""

    default_agent_id: str = "default_agent"
    strict_tenant_isolation: bool = False
    allow_cross_agent_sharing: bool = True
    require_user_id: bool = True
    default_pool: str = "private"
    default_dept_id: Optional[str] = None
    retrieval_pools: List[str] = Field(default_factory=lambda: ["private"])
    # Advisor-only mode: when True, every stored fact is attributed to the advisor
    # (the logged-in user) regardless of any client/investor in the exchange, and
    # retrieval ignores investor scope so only advisor-owned memories surface.
    # Client (subject_type="investor") memories are never written or read.
    advisor_only: bool = False


class HistoryConfig(BaseModel):
    """Mem0 SQLite audit trail."""

    enabled: bool = True
    db_path: str = ":memory:"


class StorageRoutingConfig(BaseModel):
    """Which categories go to which backend.

    NOTE: The library stores all categories via Qdrant (vector store).
    ``postgres_categories`` is a hint for application layers that manage
    their own PostgreSQL conversation tables.  The library does NOT
    automatically route storage calls based on this config — it is the
    application's responsibility to check ``should_use_qdrant(category)``
    before calling store methods.
    """

    postgres_categories: List[str] = Field(
        default_factory=lambda: ["conversational"]
    )
    qdrant_categories: List[str] = Field(
        default_factory=lambda: ["persona", "preference", "episodic", "procedural"]
    )


class FeatureConfig(BaseModel):
    """Feature toggles."""

    retrieval_enabled: bool = True
    storage_enabled: bool = True


class SummaryConfig(BaseModel):
    """Per-user preference SUMMARY layer.

    A single deterministic-ID point per user that lives in the SAME Qdrant
    collection as the granular facts (distinguished by payload type="summary").
    It is merged from newly stored facts on each turn, resolving conflicts by
    recency (newest fact wins). Facts remain the append-only source of truth.
    """

    # Master switch. When False the connector behaves exactly as before
    # (no summary points written/read, no payload index changes).
    enabled: bool = True
    # LLM model for merge/summarize. None → fall back to the main llm.model.
    model: Optional[str] = None
    temperature: float = 0.1
    # Stable payload slot value used by the single summary point per user.
    profile_key: str = "__summary__"
    # Run the one-time type backfill on existing points during initialize().
    backfill_on_init: bool = True


class MemoryHubConfig(BaseModel):
    """Complete neo_memory_hub configuration.

    This is the SINGLE config object passed to HighLevelMemoryConnector.
    Can be loaded from YAML, dict, or constructed in code.
    """

    vector_store: VectorStoreConfig = Field(default_factory=VectorStoreConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    embedder: EmbedderConfig = Field(default_factory=EmbedderConfig)
    memory_types: Dict[str, MemoryTypeConfig] = Field(default_factory=dict)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    storage: StorageConfig = Field(default_factory=StorageConfig)
    storage_routing: StorageRoutingConfig = Field(default_factory=StorageRoutingConfig)
    salience: SalienceConfig = Field(default_factory=SalienceConfig)
    extraction: ExtractionConfig = Field(default_factory=ExtractionConfig)
    dedup: DedupConfig = Field(default_factory=DedupConfig)
    buffering: BufferingConfig = Field(default_factory=BufferingConfig)
    isolation: IsolationConfig = Field(default_factory=IsolationConfig)
    history: HistoryConfig = Field(default_factory=HistoryConfig)
    features: FeatureConfig = Field(default_factory=FeatureConfig)
    summary: SummaryConfig = Field(default_factory=SummaryConfig)

    def get_valid_categories(self) -> set:
        return set(
            self.storage_routing.postgres_categories
            + self.storage_routing.qdrant_categories
        )

    def should_use_qdrant(self, category: str) -> bool:
        return category in self.storage_routing.qdrant_categories


def _resolve_env_vars(value: Any) -> Any:
    """Resolve ${VAR_NAME:-default_value} syntax in config values.
    """
    if isinstance(value, str) and value.startswith("${") and "}" in value:
        content = value[2 : value.index("}")]
        if ":-" in content:
            var_name, default = content.split(":-", 1)
        else:
            var_name = content
            default = ""
        if default == "null":
            default = None
        env_value = os.getenv(var_name)
        return env_value if env_value is not None else default
    elif isinstance(value, dict):
        return {k: _resolve_env_vars(v) for k, v in value.items()}
    elif isinstance(value, list):
        return [_resolve_env_vars(v) for v in value]
    return value


def load_config(
    config_path: str | Path | None = None, config_dict: dict | None = None
) -> MemoryHubConfig:
    """Load MemoryHubConfig from YAML file or dict.

    Args:
        config_path: Path to YAML config file.
        config_dict: Raw dict (overrides file if both provided).

    Returns:
        MemoryHubConfig instance
    """
    if config_dict:
        resolved = _resolve_env_vars(config_dict)
        return MemoryHubConfig(**resolved)

    if config_path:
        path = Path(config_path)
        if path.exists():
            with open(path) as f:
                raw = yaml.safe_load(f) or {}
            resolved = _resolve_env_vars(raw)
            return MemoryHubConfig(**resolved)
        else:
            logger.warning(f"Config file not found: {path}, using defaults")

    return MemoryHubConfig()
