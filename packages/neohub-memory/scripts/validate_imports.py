#!/usr/bin/env python3
"""Quick import validation."""
from neo_memory_hub.config.hub_config import MemoryHubConfig, load_config
from neo_memory_hub.extraction import FactExtractor, ExtractedFact, ExtractionResult
from neo_memory_hub.extraction.prompts import DEFAULT_EXTRACTION_PROMPT, DEFAULT_DEDUP_PROMPT
from neo_memory_hub.dedup import MemoryDeduplicator, DedupDecision
from neo_memory_hub.buffering import ExchangeBuffer, BufferedExchange
from neo_memory_hub.retrieval.strategy import MultiCategoryRetriever, CategorySearchConfig, RetrievalResult
from neo_memory_hub.retrieval.formatter import ProfileAssembler, ContextFormatter
from neo_memory_hub.core.history import MemoryHistoryManager
from neo_memory_hub.hooks import EntityResolver, TelemetryHook
from neo_memory_hub.connector import HighLevelMemoryConnector

# Quick sanity checks
cfg = MemoryHubConfig()
print(f"Default config: provider={cfg.vector_store.provider}, llm={cfg.llm.model}")
cfg2 = load_config(config_dict={"llm": {"model": "gpt-4o"}})
print(f"Custom config: llm={cfg2.llm.model}")
print(f"Valid categories: {cfg.get_valid_categories()}")

# Test top-level import
from neo_memory_hub import HighLevelMemoryConnector as HLC, load_config as lc
print("Top-level import OK")
print("ALL IMPORTS PASS")
