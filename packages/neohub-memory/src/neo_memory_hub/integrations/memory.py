"""
Neo Memory Hub - Async Memory Interface.

Provides AsyncMemory (async wrapper around Mem0) and MemoryConfig.

Usage:
    >>> from neo_memory_hub.integrations.memory import AsyncMemory, MemoryConfig
    >>> m = AsyncMemory()
    >>> await m.initialize()
    >>> await m.add("User likes dark mode", user_id="alex")
    >>> results = await m.search("preferences", user_id="alex")
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any

from neo_memory_hub.integrations.mem0_patches import (
    patch_mem0_milvus_update,
    patch_mem0_qdrant_update,
)


logger = logging.getLogger(__name__)

patch_mem0_milvus_update()
patch_mem0_qdrant_update()


# =============================================================================
# Configuration Types
# =============================================================================


@dataclass
class MemoryConfig:
    """
    Configuration for Memory instance.

    Fully compatible with Mem0 v1.0+ configuration including:
    - Custom prompts for fact extraction and memory updates
    - History tracking via SQLite (history_db_path)
    - Graph store for entity relationships (Neo4j, etc.)
    - Reranker configuration
    - API version control

    Supported LLM Providers (via Mem0):
    - openai, anthropic, azure_openai, ollama, groq, together,
    - aws_bedrock, litellm, gemini, deepseek, xai, sarvam,
    - lmstudio, vllm, langchain

    Supported Embedder Providers (via Mem0):
    - openai, ollama, huggingface, azure_openai, gemini, vertexai,
    - together, lmstudio, langchain, aws_bedrock, fastembed

    Supported Vector Stores (via Mem0):
    - milvus, qdrant, chroma, pinecone, weaviate, faiss,
    - elasticsearch, pgvector, redis, supabase, azure_ai_search

    API keys are resolved in this order:
    1. Explicit api_key in config
    2. Environment variable (OPENAI_API_KEY, ANTHROPIC_API_KEY, etc.)
    3. Error if neither found

    This ensures keys come from client/environment, never hardcoded.

    Example:
        >>> config = MemoryConfig(
        ...     llm={"provider": "anthropic", "config": {"model": "claude-3-5-sonnet-20241022"}},
        ...     custom_fact_extraction_prompt="Extract facts about user preferences...",
        ...     history_db_path="~/.neohub/history.db",
        ...     graph_store={
        ...         "provider": "neo4j",
        ...         "config": {"url": "bolt://localhost:7687", "username": "neo4j", "password": "..."}
        ...     }
        ... )
    """

    # Vector store configuration
    vector_store: dict[str, Any] = field(
        default_factory=lambda: {
            # Default to Qdrant for true zero-config usage.
            # Mem0 will use a local/in-memory Qdrant client when no URL is provided.
            "provider": "qdrant",
            "config": {
                "collection_name": "neo_memories",
                "embedding_model_dims": 1536,
            },
        }
    )

    # LLM configuration (for fact extraction)
    llm: dict[str, Any] = field(
        default_factory=lambda: {
            "provider": "openai",
            "config": {
                "model": "gpt-4o-mini",
                "temperature": 0.1,
                # api_key resolved from environment
            },
        }
    )

    # Embedder configuration
    embedder: dict[str, Any] = field(
        default_factory=lambda: {
            "provider": "openai",
            "config": {
                "model": "text-embedding-3-small",
                # api_key resolved from environment
            },
        }
    )

    # Guardrails: prevent accidental embedding of massive checkpoints/transcripts.
    # NOTE: Mem0 embeds message content even when `infer=False`.
    max_embedding_input_tokens: int = 7000
    max_embedding_input_chars: int = 40_000

    # Optional reranker (Mem0 v1.0+)
    # Providers: "cohere", "sentence_transformer", "llm"
    reranker: dict[str, Any] | None = None

    # =========================================================================
    # NEW: Mem0 v1.0+ Configuration Options
    # =========================================================================

    # History database path for SQLite audit trail (Mem0 tracks all memory operations)
    # Set to a path like "~/.neohub/history.db" to enable
    history_db_path: str | None = None

    # Graph store for entity relationships (Optional)
    # Providers: "neo4j", "memgraph", "neptune", "kuzu"
    # Example: {"provider": "neo4j", "config": {"url": "bolt://localhost:7687", ...}}
    graph_store: dict[str, Any] | None = None

    # API version for Mem0 (default v1.1, v1.0 no longer supported)
    version: str = "v1.1"

    # Custom prompt for fact extraction (user-supplied, domain-agnostic)
    # Use this to customize what facts are extracted from conversations.
    # Example: "Extract user preferences, constraints, and goals..."
    custom_fact_extraction_prompt: str | None = None

    # Custom prompt for memory update decisions (Mem0 v1.0+)
    # Use this to customize ADD/UPDATE/DELETE/NONE logic.
    custom_update_memory_prompt: str | None = None

    @classmethod
    def for_redis(
        cls,
        *,
        redis_url: str | None = None,
        collection_name: str = "neo_memories",
        embedding_model_dims: int = 1536,
        llm: dict[str, Any] | None = None,
        embedder: dict[str, Any] | None = None,
        reranker: dict[str, Any] | None = None,
        history_db_path: str | None = None,
        graph_store: dict[str, Any] | None = None,
        version: str = "v1.1",
        custom_fact_extraction_prompt: str | None = None,
        custom_update_memory_prompt: str | None = None,
        max_embedding_input_tokens: int | None = None,
        max_embedding_input_chars: int | None = None,
    ) -> MemoryConfig:
        """Create a MemoryConfig pre-wired for Redis vector storage."""
        resolved_redis_url = redis_url or os.getenv("REDIS_URL") or "redis://localhost:6379"
        config = cls()
        config.vector_store = {
            "provider": "redis",
            "config": {
                "redis_url": resolved_redis_url,
                "collection_name": collection_name,
                "embedding_model_dims": embedding_model_dims,
            },
        }

        if llm is not None:
            config.llm = llm
        if embedder is not None:
            config.embedder = embedder
        if reranker is not None:
            config.reranker = reranker
        if history_db_path is not None:
            config.history_db_path = history_db_path
        if graph_store is not None:
            config.graph_store = graph_store
        if version is not None:
            config.version = version
        if custom_fact_extraction_prompt is not None:
            config.custom_fact_extraction_prompt = custom_fact_extraction_prompt
        if custom_update_memory_prompt is not None:
            config.custom_update_memory_prompt = custom_update_memory_prompt
        if max_embedding_input_tokens is not None:
            config.max_embedding_input_tokens = max_embedding_input_tokens
        if max_embedding_input_chars is not None:
            config.max_embedding_input_chars = max_embedding_input_chars

        return config

    @classmethod
    def for_pgvector(
        cls,
        *,
        connection_string: str | None = None,
        dbname: str = "postgres",
        collection_name: str = "neo_memories",
        embedding_model_dims: int = 1536,
        user: str | None = None,
        password: str | None = None,
        host: str | None = None,
        port: int | None = None,
        diskann: bool = False,
        hnsw: bool = True,
        minconn: int = 1,
        maxconn: int = 5,
        sslmode: str | None = None,
        llm: dict[str, Any] | None = None,
        embedder: dict[str, Any] | None = None,
        reranker: dict[str, Any] | None = None,
        history_db_path: str | None = None,
        graph_store: dict[str, Any] | None = None,
        version: str = "v1.1",
        custom_fact_extraction_prompt: str | None = None,
        custom_update_memory_prompt: str | None = None,
        max_embedding_input_tokens: int | None = None,
        max_embedding_input_chars: int | None = None,
    ) -> MemoryConfig:
        """Create a MemoryConfig pre-wired for PostgreSQL pgvector storage."""

        def _normalize_connection_string(value: str | None) -> str | None:
            if not value:
                return None
            prefixes = (
                "postgresql+asyncpg://",
                "postgresql+psycopg://",
                "postgresql+psycopg2://",
                "postgresql+pg8000://",
                "postgresql+psycopg2cffi://",
                "postgresql+py-postgresql://",
                "postgresql+pygresql://",
            )
            for prefix in prefixes:
                if value.startswith(prefix):
                    return "postgresql://" + value[len(prefix) :]
            return value

        resolved_connection_string = _normalize_connection_string(
            connection_string
            or os.getenv("DATABASE_URL")
            or os.getenv("NEO_MEMORY_POSTGRES_URL")
        )

        config = cls()
        vector_config: dict[str, Any] = {
            "collection_name": collection_name,
            "embedding_model_dims": embedding_model_dims,
            "diskann": diskann,
            "hnsw": hnsw,
            "minconn": minconn,
            "maxconn": maxconn,
        }

        if sslmode is not None:
            vector_config["sslmode"] = sslmode

        if resolved_connection_string:
            vector_config["connection_string"] = resolved_connection_string
        else:
            vector_config.update(
                {
                    "dbname": dbname,
                    "user": user,
                    "password": password,
                    "host": host,
                    "port": port,
                }
            )

        config.vector_store = {
            "provider": "pgvector",
            "config": vector_config,
        }

        if llm is not None:
            config.llm = llm
        if embedder is not None:
            config.embedder = embedder
        if reranker is not None:
            config.reranker = reranker
        if history_db_path is not None:
            config.history_db_path = history_db_path
        if graph_store is not None:
            config.graph_store = graph_store
        if version is not None:
            config.version = version
        if custom_fact_extraction_prompt is not None:
            config.custom_fact_extraction_prompt = custom_fact_extraction_prompt
        if custom_update_memory_prompt is not None:
            config.custom_update_memory_prompt = custom_update_memory_prompt
        if max_embedding_input_tokens is not None:
            config.max_embedding_input_tokens = max_embedding_input_tokens
        if max_embedding_input_chars is not None:
            config.max_embedding_input_chars = max_embedding_input_chars

        return config

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any]) -> MemoryConfig:
        """Create config from dictionary."""
        return cls(
            vector_store=config_dict.get("vector_store", cls().vector_store),
            llm=config_dict.get("llm", cls().llm),
            embedder=config_dict.get("embedder", cls().embedder),
            reranker=config_dict.get("reranker"),
            history_db_path=config_dict.get("history_db_path"),
            graph_store=config_dict.get("graph_store"),
            version=config_dict.get("version", "v1.1"),
            custom_fact_extraction_prompt=config_dict.get("custom_fact_extraction_prompt"),
            custom_update_memory_prompt=config_dict.get("custom_update_memory_prompt"),
            max_embedding_input_tokens=int(
                config_dict.get("max_embedding_input_tokens", cls().max_embedding_input_tokens)
            ),
            max_embedding_input_chars=int(
                config_dict.get("max_embedding_input_chars", cls().max_embedding_input_chars)
            ),
        )

    def to_mem0_config(self) -> dict[str, Any]:
        """Convert to Mem0-compatible config dict."""
        config: dict[str, Any] = {}

        # Vector store
        config["vector_store"] = self.vector_store

        # LLM with API key resolution
        llm_config = dict(self.llm)
        if "config" in llm_config:
            llm_config["config"] = dict(llm_config["config"])
            if "api_key" not in llm_config["config"]:
                api_key = self._resolve_api_key(llm_config.get("provider", "openai"))
                if api_key:
                    llm_config["config"]["api_key"] = api_key
        config["llm"] = llm_config

        # Embedder with API key resolution
        emb_config = dict(self.embedder)
        if "config" in emb_config:
            emb_config["config"] = dict(emb_config["config"])
            if "api_key" not in emb_config["config"]:
                api_key = self._resolve_api_key(emb_config.get("provider", "openai"))
                if api_key:
                    emb_config["config"]["api_key"] = api_key
        config["embedder"] = emb_config

        # Optional: Reranker
        if self.reranker:
            config["reranker"] = self.reranker

        # NEW: History database path (Mem0 v1.0+ - SQLite audit trail)
        if self.history_db_path:
            config["history_db_path"] = self.history_db_path

        # NEW: Graph store (Neo4j, etc.)
        if self.graph_store:
            config["graph_store"] = self.graph_store

        # NEW: API version
        config["version"] = self.version

        # NEW: Custom prompts for fact extraction and update logic
        if self.custom_fact_extraction_prompt:
            config["custom_fact_extraction_prompt"] = self.custom_fact_extraction_prompt
        if self.custom_update_memory_prompt:
            config["custom_update_memory_prompt"] = self.custom_update_memory_prompt

        return config

    def _resolve_api_key(self, provider: str) -> str | None:
        """Resolve API key from environment for a provider."""
        env_vars = {
            "openai": "OPENAI_API_KEY",
            "anthropic": "ANTHROPIC_API_KEY",
            "cohere": "COHERE_API_KEY",
            "azure_openai": "AZURE_OPENAI_API_KEY",
            "together": "TOGETHER_API_KEY",
            "groq": "GROQ_API_KEY",
        }
        env_var = env_vars.get(provider.lower())
        if env_var:
            return os.getenv(env_var)
        return None


def _estimate_tokens_for_embedding(text: str, model: str | None = None) -> int:
    """Best-effort token estimation for embedding input size checks."""
    if not text:
        return 0
    try:
        import tiktoken  # type: ignore

        enc = None
        if model:
            try:
                enc = tiktoken.encoding_for_model(model)
            except Exception:
                enc = None
        if enc is None:
            # OpenAI embedding models commonly use cl100k_base.
            enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except Exception:
        # Fallback: ~4 chars/token for English-ish text.
        return len(text) // 4 + 1


def _ensure_embedding_input_within_limits(
    *,
    content: str,
    config: MemoryConfig,
) -> None:
    if not content:
        return

    if config.max_embedding_input_chars > 0 and len(content) > config.max_embedding_input_chars:
        raise ValueError(
            "Input too large for embedding. This usually happens when passing full conversation "
            "history or checkpoints into Mem0. Store raw transcripts/checkpoints outside the vector "
            "store and embed only summaries/facts."
        )

    model = None
    try:
        model = str(config.embedder.get("config", {}).get("model"))
    except Exception:
        model = None

    if config.max_embedding_input_tokens > 0:
        tokens = _estimate_tokens_for_embedding(content, model=model)
        if tokens > config.max_embedding_input_tokens:
            raise ValueError(
                "Input too large for embedding token limits. Mem0 embeds message content even when "
                "`infer=False`, so you cannot store a full transcript/checkpoint via Mem0 without "
                "hitting embedding limits. Store raw data outside the vector store and embed only "
                "distilled summaries/facts."
            )



# =============================================================================
# Async Memory Class
# =============================================================================


class AsyncMemory:
    """
    Async version of Memory interface.

    Provides the same API as Memory but with async methods for use in
    async contexts like LangGraph nodes.

    Usage:
        >>> m = AsyncMemory()
        >>> await m.add("User likes Python", user_id="alex")
        >>> results = await m.search("preferences", user_id="alex")
    """

    def __init__(
        self,
        config: MemoryConfig | None = None,
    ):
        """Initialize AsyncMemory."""
        self.config = config or MemoryConfig()
        self._mem0: Any = None
        self._initialized = False

    @classmethod
    def from_config(cls, config_dict: dict[str, Any]) -> AsyncMemory:
        """Create AsyncMemory from config dictionary."""
        mem_config = MemoryConfig.from_dict(config_dict)
        return cls(config=mem_config)

    @classmethod
    async def from_config_async(cls, config_dict: dict[str, Any]) -> AsyncMemory:
        """Create and initialize AsyncMemory from config."""
        instance = cls.from_config(config_dict)
        await instance.initialize()
        return instance

    async def initialize(self) -> None:
        """Initialize the async memory backend."""
        if self._initialized:
            return

        try:
            import inspect

            from mem0 import AsyncMemory as Mem0AsyncMemory

            mem0_config = self.config.to_mem0_config()
            result = Mem0AsyncMemory.from_config(mem0_config)
            # mem0ai >=1.0.10 made from_config synchronous
            if inspect.isawaitable(result):
                self._mem0 = await result
            else:
                self._mem0 = result
            self._initialized = True
            logger.info("AsyncMemory initialized successfully")

        except ImportError as e:
            raise RuntimeError("mem0ai is not installed. Install with: pip install mem0ai") from e
        except Exception as e:
            logger.error(f"Failed to initialize AsyncMemory: {e}")
            raise

    def _ensure_initialized(self) -> None:
        """Ensure memory is initialized."""
        if not self._initialized or self._mem0 is None:
            raise RuntimeError(
                "AsyncMemory not initialized. Call await initialize() first "
                "or use AsyncMemory.from_config_async()"
            )

    async def add(
        self,
        messages: str | list[dict[str, str]] | list[str],
        user_id: str | None = None,
        agent_id: str | None = None,
        run_id: str | None = None,
        metadata: dict[str, Any] | None = None,
        filters: dict[str, Any] | None = None,
        infer: bool = True,
        categories: list[str] | None = None,
        memory_type: str | None = None,
        prompt: str | None = None,
        llm: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Add memories asynchronously."""
        self._ensure_initialized()

        # Input validation for production grade
        if not messages or (isinstance(messages, str) and not messages.strip()):
            raise ValueError("messages cannot be empty")
        if isinstance(messages, list):
            if not messages:
                raise ValueError("messages list cannot be empty")
            if all(isinstance(m, str) for m in messages) and all(not m.strip() for m in messages):
                raise ValueError("messages cannot contain only empty strings")

        # Size guardrails: Mem0 embeds message content even when infer=False.
        if isinstance(messages, str):
            _ensure_embedding_input_within_limits(content=messages, config=self.config)
        elif isinstance(messages, list):
            for item in messages:
                if isinstance(item, str):
                    _ensure_embedding_input_within_limits(content=item, config=self.config)
                elif isinstance(item, dict):
                    content = str(item.get("content", ""))
                    _ensure_embedding_input_within_limits(content=content, config=self.config)

        # Build metadata with categories (mem0 doesn't support categories as direct param)
        final_metadata = metadata.copy() if metadata else {}
        if categories:
            final_metadata["categories"] = categories

        # Build kwargs for mem0.add()
        # NOTE: AsyncMemory.add() supports: user_id, agent_id, run_id, metadata, infer, memory_type, prompt, llm
        # NOT supported: categories (moved to metadata), filters
        kwargs: dict[str, Any] = {"infer": infer}
        if user_id:
            kwargs["user_id"] = user_id
        if agent_id:
            kwargs["agent_id"] = agent_id
        if run_id:
            kwargs["run_id"] = run_id
        if final_metadata:
            kwargs["metadata"] = final_metadata
        if memory_type:
            kwargs["memory_type"] = memory_type
        if prompt:
            kwargs["prompt"] = prompt
        if llm:
            kwargs["llm"] = llm

        return await self._mem0.add(messages, **kwargs)

    async def add_with_images(
        self,
        messages: list[dict[str, Any]],
        images: list[str],
        user_id: str | None = None,
        agent_id: str | None = None,
        run_id: str | None = None,
        metadata: dict[str, Any] | None = None,
        infer: bool = True,
        categories: list[str] | None = None,
        memory_type: str | None = None,
    ) -> dict[str, Any]:
        """Add memories with image content (multimodal)."""
        enriched = [{**msg, "images": images} if "images" not in msg else msg for msg in messages]
        return await self.add(
            messages=enriched,
            user_id=user_id,
            agent_id=agent_id,
            run_id=run_id,
            metadata=metadata,
            infer=infer,
            categories=categories,
            memory_type=memory_type,
        )

    async def search(
        self,
        query: str,
        user_id: str | None = None,
        agent_id: str | None = None,
        run_id: str | None = None,
        limit: int = 10,
        filters: dict[str, Any] | None = None,
        metadata_filters: dict[str, Any] | None = None,
        rerank: bool = False,
        threshold: float | None = None,
        categories: list[str] | None = None,
    ) -> dict[str, Any]:
        """
        Search memories asynchronously.

        Args:
            query: Search query
            user_id: Filter by user
            agent_id: Filter by agent
            run_id: Filter by run/session
            limit: Maximum results
            filters: Additional metadata filters
            metadata_filters: Metadata filters (AsyncMemory supports this)
            rerank: Whether to use reranker (Mem0 v1.0+)
            threshold: Minimum similarity score threshold (0.0-1.0)
            categories: Filter by categories (client-side post-filtering)

        Returns:
            Dict with "results" key containing matching memories
        """
        self._ensure_initialized()

        # Input validation for production grade
        if not query or not query.strip():
            raise ValueError("query cannot be empty")

        # Size guardrails: Mem0 will embed the query.
        _ensure_embedding_input_within_limits(content=query, config=self.config)

        # Build kwargs for mem0.search()
        # NOTE: mem0 v2.0+ requires user_id/agent_id/run_id inside filters, not top-level.
        kwargs: dict[str, Any] = {"limit": limit, "rerank": rerank}

        # Merge all filters into a single dict for Qdrant payload filtering
        merged_filters = dict(filters) if filters else {}
        if user_id:
            merged_filters["user_id"] = user_id
        if agent_id:
            merged_filters["agent_id"] = agent_id
        if run_id:
            merged_filters["run_id"] = run_id
        if metadata_filters:
            merged_filters.update(metadata_filters)
        if merged_filters:
            kwargs["filters"] = merged_filters

        if threshold is not None:
            kwargs["threshold"] = threshold

        result = await self._mem0.search(query, **kwargs)

        # Post-filter by categories if specified
        if categories and result.get("results"):
            filtered_results = []
            for mem in result["results"]:
                mem_metadata = mem.get("metadata", {}) or {}
                mem_categories = mem_metadata.get("categories", [])
                if mem_categories and any(cat in mem_categories for cat in categories):
                    filtered_results.append(mem)
            result["results"] = filtered_results

        return result

    async def get(self, memory_id: str) -> dict[str, Any] | None:
        """Get memory by ID."""
        self._ensure_initialized()
        return await self._mem0.get(memory_id)

    async def get_all(
        self,
        user_id: str | None = None,
        agent_id: str | None = None,
        run_id: str | None = None,
        limit: int = 100,
        filters: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Get all memories."""
        self._ensure_initialized()

        # mem0 v2.0+ requires entity params inside filters
        merged_filters = dict(filters) if filters else {}
        if user_id:
            merged_filters["user_id"] = user_id
        if agent_id:
            merged_filters["agent_id"] = agent_id
        if run_id:
            merged_filters["run_id"] = run_id

        kwargs: dict[str, Any] = {"limit": limit}
        if merged_filters:
            kwargs["filters"] = merged_filters

        return await self._mem0.get_all(**kwargs)

    async def update(
        self,
        memory_id: str,
        data: str,
    ) -> dict[str, Any]:
        """
        Update a memory.

        Args:
            memory_id: ID of memory to update
            data: New content
        """
        self._ensure_initialized()

        # Input validation
        if not memory_id or not memory_id.strip():
            raise ValueError("memory_id cannot be empty")
        if not data or not data.strip():
            raise ValueError("data cannot be empty")

        # Size guardrails: update content is re-embedded by Mem0/vector store.
        _ensure_embedding_input_within_limits(content=data, config=self.config)

        return await self._mem0.update(memory_id, data)

    async def delete(self, memory_id: str) -> dict[str, Any]:
        """Delete a memory."""
        self._ensure_initialized()
        return await self._mem0.delete(memory_id)

    async def delete_all(
        self,
        user_id: str | None = None,
        agent_id: str | None = None,
        run_id: str | None = None,
    ) -> dict[str, Any]:
        """Delete all memories for user/agent."""
        self._ensure_initialized()

        # mem0 2.0.x's delete_all takes entity params directly (user_id/
        # agent_id/run_id), not a `filters` dict. Pass only the ones provided.
        kwargs: dict[str, Any] = {}
        if user_id:
            kwargs["user_id"] = user_id
        if agent_id:
            kwargs["agent_id"] = agent_id
        if run_id:
            kwargs["run_id"] = run_id

        return await self._mem0.delete_all(**kwargs)

    async def close(self) -> None:
        """Close the memory backend."""
        self._mem0 = None
        self._initialized = False

    async def __aenter__(self) -> AsyncMemory:
        """Async context manager entry."""
        await self.initialize()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Async context manager exit."""
        await self.close()

    # Helper methods (same as sync version)

    def format_memories(
        self,
        search_results: dict[str, Any],
        prefix: str = "Relevant information from memory:",
    ) -> str:
        """Format search results as context string."""
        results = search_results.get("results", [])
        if not results:
            return ""

        lines = [prefix]
        for mem in results:
            memory_text = mem.get("memory", "")
            if memory_text:
                lines.append(f"- {memory_text}")

        return "\n".join(lines)


# =============================================================================
# Exports
# =============================================================================


__all__ = [
    "AsyncMemory",
    "MemoryConfig",
]
