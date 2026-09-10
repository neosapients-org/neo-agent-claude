"""
Neo Memory Hub - Intelligent Memory Management for AI Agents

The **primary API** is `NeoMemoryConnector` - a single interface for all memory operations.

Quick Start:
    >>> from neo_memory_hub import NeoMemoryConnector
    >>>
    >>> async with NeoMemoryConnector() as connector:
    ...     # Store memories
    ...     await connector.add("User prefers dark mode", user_id="alex")
    ...
    ...     # Search memories  
    ...     results = await connector.search("preferences", user_id="alex")
    ...
    ...     # Build context for LLM
    ...     context = await connector.build_context("query", user_id="alex")

Features:
- Full Mem0 API parity (add, search, get, get_all, update, delete, delete_all, reset)
- Enterprise memory isolation with 6-tier AccessTier model
- Salience gating via StorageGateway
- TTL / expiry lifecycle management
- Operation audit hooks
- Context building for LLM prompts
- Framework-agnostic design (works with LangChain, LangGraph, CrewAI, etc.)
"""

__version__ = "0.3.0"
__author__ = "Neo Memory Hub Team"
__license__ = "MIT"

# =============================================================================
# Primary API - NeoMemoryConnector
# =============================================================================

from neo_memory_hub.integrations.connector import (
    ConnectorConfig,
    MemoryItem,
    NeoMemoryConnector,
    OperationLog,
)

# Expose MemoryConfig for configuring the connector
from neo_memory_hub.integrations.memory import MemoryConfig

# =============================================================================
# Domain Models
# =============================================================================

from neo_memory_hub.domain.memory import MemoryEntry, MemoryResult
from neo_memory_hub.domain.scope import AccessTier, IsolationScope
from neo_memory_hub.domain.types import MemoryType

# =============================================================================
# Core Utilities
# =============================================================================

from neo_memory_hub.core.salience import SalienceScorer, SalienceScorerConfig, ScoredFact
from neo_memory_hub.core.storage_gateway import StorageDecision, StorageGateway, StorageGatewayConfig, WorthinessResult

# =============================================================================
# Exceptions
# =============================================================================

from neo_memory_hub.core.exceptions import (
    AccessDeniedError,
    ConfigurationError,
    NeoMemoryError,
    StorageError,
    ValidationError,
)

# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Version
    "__version__",
    # Primary API
    "NeoMemoryConnector",
    "ConnectorConfig",
    "MemoryItem",
    "OperationLog",
    "MemoryConfig",
    # Domain Models
    "AccessTier",
    "IsolationScope",
    "MemoryEntry",
    "MemoryResult",
    "MemoryType",
    # Core Utilities
    "SalienceScorer",
    "SalienceScorerConfig",
    "ScoredFact",
    "StorageDecision",
    "StorageGateway",
    "StorageGatewayConfig",
    "WorthinessResult",
    # Exceptions
    "AccessDeniedError",
    "NeoMemoryError",
    "StorageError",
    "ValidationError",
    "ConfigurationError",
]

# =============================================================================
# v0.3.0 — High-Level API
# =============================================================================

from neo_memory_hub.connector import HighLevelMemoryConnector
from neo_memory_hub.config.hub_config import MemoryHubConfig, load_config
from neo_memory_hub.extraction import FactExtractor, ExtractedFact, ExtractionResult
from neo_memory_hub.dedup import MemoryDeduplicator, DedupDecision
from neo_memory_hub.buffering import ExchangeBuffer, BufferedExchange
from neo_memory_hub.retrieval.strategy import (
    MultiCategoryRetriever,
    CategorySearchConfig,
    RetrievalResult,
)
from neo_memory_hub.retrieval.formatter import ProfileAssembler, ContextFormatter
from neo_memory_hub.hooks import EntityResolver, PoolRoutingHook, TelemetryHook
from neo_memory_hub.integrations.base import MemoryEngine

__all__ += [
    "HighLevelMemoryConnector",
    "MemoryHubConfig",
    "load_config",
    "FactExtractor",
    "ExtractedFact",
    "ExtractionResult",
    "MemoryDeduplicator",
    "DedupDecision",
    "ExchangeBuffer",
    "BufferedExchange",
    "MultiCategoryRetriever",
    "CategorySearchConfig",
    "RetrievalResult",
    "ProfileAssembler",
    "ContextFormatter",
    "EntityResolver",
    "PoolRoutingHook",
    "TelemetryHook",
    "MemoryEngine",
]
