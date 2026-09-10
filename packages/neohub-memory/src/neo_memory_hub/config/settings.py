"""Pydantic Settings for Neo Memory Hub configuration."""

from functools import lru_cache
from typing import Any, Literal

from pydantic import AliasChoices, Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class SystemSettings(BaseSettings):
    """System-level configuration."""

    name: str = Field(default="neo-memory-hub", description="Application name")
    version: str = Field(default="0.1.0", description="Application version")
    environment: Literal["development", "staging", "production", "test"] = Field(
        default="development", description="Runtime environment"
    )
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="INFO", description="Logging level"
    )

    model_config = SettingsConfigDict(
        env_prefix="NEO_MEMORY_",
        case_sensitive=False,
    )


class MilvusSettings(BaseSettings):
    """Milvus vector database configuration."""

    host: str = Field(default="localhost", description="Milvus server host")
    port: int = Field(default=19530, description="Milvus server port")
    collection_name: str = Field(default="neo_memories", description="Collection name for memories")
    index_type: Literal["IVF_FLAT", "HNSW", "IVF_SQ8"] = Field(
        default="HNSW", description="Vector index type"
    )
    metric_type: Literal["COSINE", "L2", "IP"] = Field(
        default="COSINE", description="Distance metric type"
    )
    dimension: int = Field(default=1536, description="Embedding dimension")
    pool_size: int = Field(default=10, description="Connection pool size")
    timeout: int = Field(default=30, description="Operation timeout in seconds")

    # HNSW index parameters
    hnsw_m: int = Field(
        default=16, alias="M", description="HNSW M parameter (connections per node)"
    )
    hnsw_ef_construction: int = Field(
        default=256, alias="efConstruction", description="HNSW efConstruction parameter"
    )
    hnsw_ef_search: int = Field(default=64, alias="ef", description="HNSW ef parameter for search")

    model_config = SettingsConfigDict(
        env_prefix="NEO_MEMORY_MILVUS_",
        case_sensitive=False,
    )


class QdrantSettings(BaseSettings):
    """Qdrant vector database configuration."""

    url: str | None = Field(
        default=None,
        description="Qdrant server URL (omit to use in-memory/local client)",
    )
    api_key: SecretStr | None = Field(default=None, description="Qdrant API key")
    collection_name: str = Field(default="neo_memories", description="Collection name for memories")

    model_config = SettingsConfigDict(
        env_prefix="NEO_MEMORY_QDRANT_",
        case_sensitive=False,
    )


class VectorStoreSettings(BaseSettings):
    """Vector store provider selection."""

    provider: Literal["milvus", "qdrant", "pgvector"] = Field(
        default="milvus",
        description="Vector store provider (milvus, qdrant, or pgvector)",
    )

    model_config = SettingsConfigDict(
        env_prefix="NEO_MEMORY_VECTOR_STORE_",
        case_sensitive=False,
    )


class RedisSettings(BaseSettings):
    """Redis cache and session memory configuration."""

    url: str = Field(default="redis://localhost:6379", description="Redis connection URL")
    password: SecretStr | None = Field(default=None, description="Redis password")
    db: int = Field(default=0, description="Redis database number")
    default_ttl: int = Field(default=3600, description="Default TTL in seconds")
    working_memory_ttl: int = Field(default=86400, description="Working memory TTL (24 hours)")
    query_cache_ttl: int = Field(default=300, description="Query cache TTL (5 minutes)")

    # Redis as vector store for session memory (mem0 config)
    session_collection_name: str = Field(
        default="session_memories", description="Collection name for session memories"
    )
    enabled_for_session: bool = Field(
        default=True, description="Enable Redis as vector store for session memory"
    )

    model_config = SettingsConfigDict(
        env_prefix="NEO_MEMORY_REDIS_",
        case_sensitive=False,
    )


class PostgresSettings(BaseSettings):
    """PostgreSQL metadata store configuration."""

    enabled: bool = Field(default=True, description="Enable PostgreSQL metadata store")
    url: str = Field(
        default="postgresql+asyncpg://neo:neo_secret@localhost:5432/neo_memory",
        description="PostgreSQL connection URL",
        validation_alias=AliasChoices("DATABASE_URL", "NEO_MEMORY_POSTGRES_URL"),
    )
    pool_size: int = Field(default=20, description="SQLAlchemy connection pool size")
    max_overflow: int = Field(default=10, description="SQLAlchemy max overflow connections")

    model_config = SettingsConfigDict(
        env_prefix="NEO_MEMORY_POSTGRES_",
        case_sensitive=False,
    )


class EmbeddingSettings(BaseSettings):
    """Embedding provider configuration."""

    provider: Literal[
        "openai",
        "ollama",
        "huggingface",
        "azure_openai",
        "gemini",
        "vertexai",
        "together",
        "lmstudio",
        "langchain",
        "aws_bedrock",
        "fastembed",
    ] = Field(default="openai", description="Embedding provider")
    model: str = Field(default="text-embedding-3-small", description="Embedding model name")
    api_key: SecretStr | None = Field(default=None, description="API key for the provider")
    base_url: str | None = Field(
        default=None, description="Base URL for API (Ollama, custom endpoints)"
    )
    dimensions: int = Field(default=1536, description="Embedding dimensions")
    batch_size: int = Field(default=100, description="Batch size for embedding generation")
    cache_enabled: bool = Field(default=True, description="Enable embedding caching")

    model_config = SettingsConfigDict(
        env_prefix="NEO_MEMORY_EMBEDDING_",
        case_sensitive=False,
    )


class LLMSettings(BaseSettings):
    """LLM provider configuration."""

    provider: Literal[
        "openai",
        "anthropic",
        "azure_openai",
        "ollama",
        "together",
        "groq",
        "aws_bedrock",
        "litellm",
        "gemini",
        "deepseek",
        "xai",
        "sarvam",
        "lmstudio",
        "vllm",
        "langchain",
    ] = Field(default="openai", description="LLM provider")
    model: str = Field(default="gpt-4o-mini", description="Model name")
    api_key: SecretStr | None = Field(default=None, description="API key for the provider")
    base_url: str | None = Field(default=None, description="Base URL for API")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="Sampling temperature")
    max_tokens: int = Field(default=1000, description="Maximum tokens in response")
    timeout: int = Field(default=30, description="Request timeout in seconds")
    max_retries: int = Field(default=3, description="Maximum retry attempts")

    model_config = SettingsConfigDict(
        env_prefix="NEO_MEMORY_LLM_",
        case_sensitive=False,
    )


class RankingSettings(BaseSettings):
    """Multi-factor ranking configuration."""

    strategy: Literal["multi_factor", "relevance_only"] = Field(
        default="multi_factor", description="Ranking strategy"
    )
    relevance_weight: float = Field(
        default=0.5, ge=0.0, le=1.0, description="Weight for relevance score (alpha)"
    )
    recency_weight: float = Field(
        default=0.3, ge=0.0, le=1.0, description="Weight for recency score (beta)"
    )
    importance_weight: float = Field(
        default=0.2, ge=0.0, le=1.0, description="Weight for importance score (gamma)"
    )
    recency_decay_lambda: float = Field(default=0.01, ge=0.0, description="Recency decay parameter")

    @field_validator("importance_weight")
    @classmethod
    def validate_weights_sum(cls, v: float, info: Any) -> float:
        """Validate that weights sum to 1.0."""
        data = info.data
        total = data.get("relevance_weight", 0.5) + data.get("recency_weight", 0.3) + v
        if abs(total - 1.0) > 0.001:
            raise ValueError(
                f"Ranking weights must sum to 1.0, got {total}. "
                f"(relevance={data.get('relevance_weight')}, "
                f"recency={data.get('recency_weight')}, importance={v})"
            )
        return v

    model_config = SettingsConfigDict(
        env_prefix="NEO_MEMORY_RANKING_",
        case_sensitive=False,
    )


class RetrievalSettings(BaseSettings):
    """Retrieval configuration."""

    default_limit: int = Field(default=10, ge=1, le=100, description="Default retrieval limit")
    max_limit: int = Field(default=100, ge=1, le=1000, description="Maximum retrieval limit")
    min_relevance_score: float = Field(
        default=0.3, ge=0.0, le=1.0, description="Minimum relevance score filter"
    )
    ranking: RankingSettings = Field(
        default_factory=RankingSettings, description="Ranking configuration"
    )

    model_config = SettingsConfigDict(
        env_prefix="NEO_MEMORY_RETRIEVAL_",
        case_sensitive=False,
    )


class Settings(BaseSettings):
    """
    Root settings for Neo Memory Hub.

    Settings are loaded from:
    1. Environment variables (NEO_MEMORY_*)
    2. .env file
    3. YAML config file (via ConfigLoader)
    4. Default values
    """

    system: SystemSettings = Field(default_factory=SystemSettings, description="System settings")
    vector_store: VectorStoreSettings = Field(
        default_factory=VectorStoreSettings, description="Vector store selection"
    )
    milvus: MilvusSettings = Field(default_factory=MilvusSettings, description="Milvus settings")
    qdrant: QdrantSettings = Field(default_factory=QdrantSettings, description="Qdrant settings")
    redis: RedisSettings = Field(default_factory=RedisSettings, description="Redis settings")
    postgres: PostgresSettings = Field(
        default_factory=PostgresSettings, description="PostgreSQL settings"
    )
    embedding: EmbeddingSettings = Field(
        default_factory=EmbeddingSettings, description="Embedding settings"
    )
    llm: LLMSettings = Field(default_factory=LLMSettings, description="LLM settings")
    retrieval: RetrievalSettings = Field(
        default_factory=RetrievalSettings, description="Retrieval settings"
    )

    # Mem0 advanced options (forwarded into MemoryConfig)
    mem0_history_db_path: str | None = Field(
        default=None, description="Path to Mem0 SQLite history DB for audit trail"
    )
    mem0_graph_store: dict[str, Any] | None = Field(
        default=None, description="Mem0 graph store configuration (e.g., Neo4j)"
    )
    mem0_custom_fact_extraction_prompt: str | None = Field(
        default=None, description="Custom fact extraction system prompt for Mem0"
    )
    mem0_custom_update_memory_prompt: str | None = Field(
        default=None, description="Custom update decision prompt for Mem0"
    )
    mem0_reranker: dict[str, Any] | None = Field(
        default=None, description="Mem0 reranker configuration"
    )
    mem0_version: str = Field(default="v1.1", description="Mem0 API version")

    model_config = SettingsConfigDict(
        env_prefix="NEO_MEMORY_",
        env_nested_delimiter="__",
        case_sensitive=False,
    )


@lru_cache
def get_settings() -> Settings:
    """
    Get cached settings instance.

    Uses LRU cache to ensure settings are loaded only once.

    Returns:
        Settings instance
    """
    return Settings()
