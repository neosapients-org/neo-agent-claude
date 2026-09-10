"""MCPServerConfig — configuration for the neomem MCP server.

Reads from environment variables and/or mcp.config.yaml.
Builds MemoryHubConfig from these settings for the library layer.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import yaml

# Canonical capabilities live inside the neomem_mcp package.
_DEFAULT_CAPS_DIR = str(Path(__file__).parent / "capabilities")


@dataclass
class MCPServerConfig:
    """Configuration for the neomem MCP server process."""

    # Server transport
    host: str = "127.0.0.1"
    port: int = 18432
    transport: str = "streamable-http"  # "streamable-http" | "stdio"

    # Capabilities directory (defaults to bundled package path)
    capabilities_dir: str = _DEFAULT_CAPS_DIR

    # Vector store
    qdrant_url: str = "http://localhost:6335"
    qdrant_collection: str = "neo_memory"

    # LLM
    llm_model: str = "gpt-4o-mini"
    llm_temperature: float = 0.1

    # Embedder
    embedder_model: str = "text-embedding-3-small"
    embedder_dimensions: int = 1536

    # Features
    salience_enabled: bool = True
    dedup_enabled: bool = True
    buffering_enabled: bool = True
    extraction_enabled: bool = True

    # Logging
    log_level: str = "INFO"

    @classmethod
    def from_env(cls) -> "MCPServerConfig":
        """Build config from environment variables."""
        return cls(
            host=os.environ.get("NEOMEM_MCP_HOST", "127.0.0.1"),
            port=int(os.environ.get("NEOMEM_MCP_PORT", "18432")),
            transport=os.environ.get("NEOMEM_MCP_TRANSPORT", "streamable-http"),
            capabilities_dir=os.environ.get("NEOMEM_CAPABILITIES_DIR", _DEFAULT_CAPS_DIR),
            qdrant_url=os.environ.get("NEOMEM_QDRANT_URL", "http://localhost:6335"),
            qdrant_collection=os.environ.get("NEOMEM_QDRANT_COLLECTION", "neo_memory"),
            llm_model=os.environ.get("NEOMEM_LLM_MODEL", "gpt-4o-mini"),
            llm_temperature=float(os.environ.get("NEOMEM_LLM_TEMPERATURE", "0.1")),
            embedder_model=os.environ.get("NEOMEM_EMBEDDER_MODEL", "text-embedding-3-small"),
            embedder_dimensions=int(os.environ.get("NEOMEM_EMBEDDER_DIMENSIONS", "1536")),
            salience_enabled=os.environ.get("NEOMEM_SALIENCE_ENABLED", "true").lower() == "true",
            dedup_enabled=os.environ.get("NEOMEM_DEDUP_ENABLED", "true").lower() == "true",
            buffering_enabled=os.environ.get("NEOMEM_BUFFERING_ENABLED", "true").lower() == "true",
            extraction_enabled=os.environ.get("NEOMEM_EXTRACTION_ENABLED", "true").lower() == "true",
            log_level=os.environ.get("NEOMEM_LOG_LEVEL", "INFO"),
        )

    @classmethod
    def from_yaml(cls, config_path: str) -> "MCPServerConfig":
        """Build config from a YAML file, with env var overrides."""
        path = Path(config_path)
        if not path.exists():
            return cls.from_env()

        with open(path) as f:
            data = yaml.safe_load(f) or {}

        # Start from YAML values, let env vars override
        base = cls(
            host=data.get("host", "127.0.0.1"),
            port=data.get("port", 18432),
            transport=data.get("transport", "streamable-http"),
            capabilities_dir=data.get("capabilities_dir", _DEFAULT_CAPS_DIR),
            qdrant_url=data.get("qdrant_url", "http://localhost:6335"),
            qdrant_collection=data.get("qdrant_collection", "neo_memory"),
            llm_model=data.get("llm_model", "gpt-4o-mini"),
            llm_temperature=data.get("llm_temperature", 0.1),
            embedder_model=data.get("embedder_model", "text-embedding-3-small"),
            embedder_dimensions=data.get("embedder_dimensions", 1536),
            salience_enabled=data.get("salience_enabled", True),
            dedup_enabled=data.get("dedup_enabled", True),
            buffering_enabled=data.get("buffering_enabled", True),
            extraction_enabled=data.get("extraction_enabled", True),
            log_level=data.get("log_level", "INFO"),
        )

        # Env vars override YAML values
        env = cls.from_env()
        for fld in [
            "host", "port", "transport", "capabilities_dir", "qdrant_url",
            "qdrant_collection", "llm_model", "llm_temperature",
            "embedder_model", "embedder_dimensions", "log_level",
        ]:
            env_key = f"NEOMEM_{fld.upper()}" if fld != "host" else "NEOMEM_MCP_HOST"
            if fld == "port":
                env_key = "NEOMEM_MCP_PORT"
            elif fld == "transport":
                env_key = "NEOMEM_MCP_TRANSPORT"
            elif fld == "capabilities_dir":
                env_key = "NEOMEM_CAPABILITIES_DIR"
            else:
                env_key = f"NEOMEM_{fld.upper()}"

            if env_key in os.environ:
                setattr(base, fld, getattr(env, fld))

        return base


def build_memory_config(server_config: MCPServerConfig) -> Any:
    """Build a MemoryHubConfig from MCPServerConfig.

    Uses typed Pydantic sub-config objects directly (matching the
    pattern used in integration tests) instead of dicts, which avoids
    issues with load_config() dict merging.
    """
    from neo_memory_hub.config.hub_config import (
        BufferingConfig,
        DedupConfig,
        EmbedderConfig,
        ExtractionConfig,
        FeatureConfig,
        IsolationConfig,
        LLMConfig,
        MemoryHubConfig,
        RetrievalConfig,
        SalienceConfig,
        VectorStoreConfig,
    )

    return MemoryHubConfig(
        vector_store=VectorStoreConfig(
            provider="qdrant",
            qdrant_url=server_config.qdrant_url,
            collection_name=server_config.qdrant_collection,
        ),
        llm=LLMConfig(
            model=server_config.llm_model,
            temperature=server_config.llm_temperature,
        ),
        embedder=EmbedderConfig(
            model=server_config.embedder_model,
            dimensions=server_config.embedder_dimensions,
        ),
        extraction=ExtractionConfig(enabled=server_config.extraction_enabled),
        dedup=DedupConfig(enabled=server_config.dedup_enabled),
        buffering=BufferingConfig(enabled=server_config.buffering_enabled),
        salience=SalienceConfig(
            enabled=server_config.salience_enabled,
            min_threshold=0.2,
        ),
        isolation=IsolationConfig(require_user_id=False),
        features=FeatureConfig(storage_enabled=True, retrieval_enabled=True),
        retrieval=RetrievalConfig(limit=10, min_relevance_score=0.1),
    )
