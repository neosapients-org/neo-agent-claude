"""
Retrieval context and configuration models.

Provides dataclasses for configuring retrieval behavior including
token budgets, ranking options, and filtering criteria.
"""

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Any

from neo_memory_hub.domain.types import MemoryType


class RetrievalStrategy(str, Enum):
    """Strategy for combining search results."""

    RELEVANCE_ONLY = "relevance_only"  # Just vector similarity
    MULTI_FACTOR = "multi_factor"  # Relevance + Recency + Importance
    RECENCY_BIASED = "recency_biased"  # Higher weight on recency


@dataclass
class TokenBudgetConfig:
    """
    Configuration for token budget allocation.

    Default budget assumes 8K context window with allocations:
    - system_prompt: 500 tokens (instructions, persona)
    - core_memory: 400 tokens (always-loaded persona/security)
    - retrieved_memories: 3000 tokens (search results)
    - conversation_history: 3000 tokens (chat history)
    - response_buffer: 1100 tokens (model output)

    Attributes:
        total_budget: Total available tokens (context window)
        system_prompt: Tokens for system instructions
        core_memory: Tokens for always-loaded memories
        retrieved_memories: Tokens for search results
        conversation_history: Tokens for chat history
        response_buffer: Reserved for model response
        chars_per_token: Approximate characters per token (fallback)
        use_tiktoken: Whether to use tiktoken for accurate counting
        tiktoken_model: Model name for tiktoken encoding
    """

    total_budget: int = 8000
    system_prompt: int = 500
    core_memory: int = 400
    retrieved_memories: int = 3000
    conversation_history: int = 3000
    response_buffer: int = 1100
    chars_per_token: float = 4.0  # Approximate for English text (fallback)
    use_tiktoken: bool = True  # Use tiktoken for accurate counting
    tiktoken_model: str = "gpt-4o-mini"  # Model for tiktoken encoding

    def __post_init__(self) -> None:
        """Validate budget allocations."""
        total_allocated = (
            self.system_prompt
            + self.core_memory
            + self.retrieved_memories
            + self.conversation_history
            + self.response_buffer
        )
        if total_allocated > self.total_budget:
            raise ValueError(
                f"Total allocated ({total_allocated}) exceeds budget ({self.total_budget}). "
                f"Allocations: system={self.system_prompt}, core={self.core_memory}, "
                f"retrieved={self.retrieved_memories}, conversation={self.conversation_history}, "
                f"response={self.response_buffer}"
            )

    @classmethod
    def for_context_size(cls, context_size: int) -> "TokenBudgetConfig":
        """
        Create a budget config scaled for a specific context size.

        Args:
            context_size: Total context window (e.g., 4096, 8192, 32768)

        Returns:
            TokenBudgetConfig scaled proportionally
        """
        # Scale from default 8K proportionally
        scale = context_size / 8000
        return cls(
            total_budget=context_size,
            system_prompt=int(500 * scale),
            core_memory=int(400 * scale),
            retrieved_memories=int(3000 * scale),
            conversation_history=int(3000 * scale),
            response_buffer=int(1100 * scale),
        )

    @classmethod
    def unlimited(cls) -> "TokenBudgetConfig":
        """Create config with no effective budget limit."""
        large = 1_000_000
        return cls(
            total_budget=large,
            system_prompt=0,
            core_memory=0,
            retrieved_memories=large,
            conversation_history=0,
            response_buffer=0,
        )


@dataclass
class RetrievalOptions:
    """
    Options for retrieval behavior.

    Attributes:
        limit: Maximum results to return
        min_relevance: Minimum relevance score (0-1)
        memory_types: Filter by specific memory types
        include_metadata: Include full metadata in results
        enable_reranking: Use LLM reranking (if available)
        strategy: Ranking strategy to use
        over_fetch_factor: How many extra results to fetch for reranking
        time_window_hours: Only retrieve memories from last N hours (None = all)
    """

    limit: int = 10
    min_relevance: float = 0.3
    memory_types: list[MemoryType] | None = None
    include_metadata: bool = True
    enable_reranking: bool = False
    strategy: RetrievalStrategy = RetrievalStrategy.MULTI_FACTOR
    over_fetch_factor: int = 5
    time_window_hours: float | None = None

    def __post_init__(self) -> None:
        """Validate options."""
        if self.limit < 1:
            raise ValueError("limit must be >= 1")
        if self.limit > 1000:
            raise ValueError("limit must be <= 1000")
        if not 0.0 <= self.min_relevance <= 1.0:
            raise ValueError("min_relevance must be between 0.0 and 1.0")
        if self.over_fetch_factor < 1:
            raise ValueError("over_fetch_factor must be >= 1")


@dataclass
class RetrievalContext:
    """
    Complete context for a retrieval operation.

    Combines query, scope, options, and budget configuration
    into a single context object for the retrieval pipeline.

    Attributes:
        query: The search query text
        scope: Isolation scope for the query
        options: Retrieval behavior options
        token_budget: Token budget configuration
        request_time: When the request was made
        request_id: Unique identifier for this request
        extra: Additional context data
    """

    query: str
    scope: Any  # IsolationScope (avoiding circular import)
    options: RetrievalOptions = field(default_factory=RetrievalOptions)
    token_budget: TokenBudgetConfig = field(default_factory=TokenBudgetConfig)
    request_time: datetime = field(default_factory=lambda: datetime.now(UTC))
    request_id: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Generate request_id if not provided."""
        if not self.request_id:
            import uuid

            self.request_id = str(uuid.uuid4())[:8]

    @property
    def available_tokens(self) -> int:
        """Get tokens available for retrieved memories."""
        return self.token_budget.retrieved_memories

    def with_options(self, **kwargs: Any) -> "RetrievalContext":
        """
        Create new context with modified options.

        Args:
            **kwargs: Option fields to override

        Returns:
            New RetrievalContext with updated options
        """
        new_options = RetrievalOptions(
            limit=kwargs.get("limit", self.options.limit),
            min_relevance=kwargs.get("min_relevance", self.options.min_relevance),
            memory_types=kwargs.get("memory_types", self.options.memory_types),
            include_metadata=kwargs.get("include_metadata", self.options.include_metadata),
            enable_reranking=kwargs.get("enable_reranking", self.options.enable_reranking),
            strategy=kwargs.get("strategy", self.options.strategy),
            over_fetch_factor=kwargs.get("over_fetch_factor", self.options.over_fetch_factor),
            time_window_hours=kwargs.get("time_window_hours", self.options.time_window_hours),
        )
        return RetrievalContext(
            query=self.query,
            scope=self.scope,
            options=new_options,
            token_budget=self.token_budget,
            request_time=self.request_time,
            request_id=self.request_id,
            extra=self.extra.copy(),
        )

    def with_budget(self, budget: TokenBudgetConfig) -> "RetrievalContext":
        """
        Create new context with different token budget.

        Args:
            budget: New token budget config

        Returns:
            New RetrievalContext with updated budget
        """
        return RetrievalContext(
            query=self.query,
            scope=self.scope,
            options=self.options,
            token_budget=budget,
            request_time=self.request_time,
            request_id=self.request_id,
            extra=self.extra.copy(),
        )
