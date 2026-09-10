"""
Token budget management for LLM context windows.

Manages how many tokens are allocated to different components
and prunes results to fit within budgets.
"""

import logging
from collections.abc import Sequence
from dataclasses import dataclass, field

from neo_memory_hub.domain.memory import MemoryResult
from neo_memory_hub.retrieval.context import TokenBudgetConfig


logger = logging.getLogger(__name__)


@dataclass
class TokenStats:
    """Statistics from token budget operations."""

    total_candidates: int = 0
    total_tokens_available: int = 0
    total_tokens_used: int = 0
    tokens_remaining: int = 0
    results_kept: int = 0
    results_pruned: int = 0
    avg_tokens_per_result: float = 0.0


@dataclass
class CategoryAllocation:
    """Token allocation for a specific category of memories."""

    category: str
    budget: int
    used: int = 0
    results: list[MemoryResult] = field(default_factory=list)

    @property
    def remaining(self) -> int:
        """Tokens remaining in this category."""
        return max(0, self.budget - self.used)

    @property
    def is_exhausted(self) -> bool:
        """Whether budget is exhausted."""
        return self.remaining == 0


class TokenBudgetManager:
    """
    Manages token allocation for LLM context windows.

    Ensures retrieved memories fit within token budgets while
    prioritizing higher-scored results.

    Budget Categories (default 8K context):
    - system_prompt: 500 tokens
    - core_memory: 400 tokens (persona, security)
    - retrieved_memories: 3000 tokens
    - conversation_history: 3000 tokens
    - response_buffer: 1100 tokens

    Token Counting:
    - Uses tiktoken for accurate token counting when available
    - Falls back to character-based estimation if tiktoken unavailable

    Example:
        >>> config = TokenBudgetConfig(total_budget=8000, use_tiktoken=True)
        >>> manager = TokenBudgetManager(config)
        >>> pruned = manager.prune_to_budget(results, budget=3000)
    """

    # Approximate characters per token for fallback estimation
    DEFAULT_CHARS_PER_TOKEN = 4.0

    def __init__(self, config: TokenBudgetConfig | None = None):
        """
        Initialize token budget manager.

        Args:
            config: Token budget configuration. Defaults to 8K budget.
        """
        self.config = config or TokenBudgetConfig()
        self._last_stats: TokenStats | None = None
        self._chars_per_token = config.chars_per_token if config else self.DEFAULT_CHARS_PER_TOKEN

        # Initialize tiktoken encoder if enabled
        self._encoder = None
        self._tiktoken_available = False

        if self.config.use_tiktoken:
            self._init_tiktoken()

    def _init_tiktoken(self) -> None:
        """Initialize tiktoken encoder for accurate token counting."""
        try:
            import tiktoken

            # Try to get encoder for specific model
            try:
                self._encoder = tiktoken.encoding_for_model(self.config.tiktoken_model)
                self._tiktoken_available = True
                logger.debug(f"Using tiktoken encoder for model: {self.config.tiktoken_model}")
            except KeyError:
                # Fall back to cl100k_base (used by GPT-4, GPT-3.5-turbo)
                self._encoder = tiktoken.get_encoding("cl100k_base")
                self._tiktoken_available = True
                logger.debug(
                    f"Model {self.config.tiktoken_model} not found, using cl100k_base encoding"
                )

        except ImportError:
            logger.warning(
                "tiktoken not installed, falling back to character-based "
                "token estimation. Install with: pip install tiktoken"
            )
            self._tiktoken_available = False

    @property
    def last_stats(self) -> TokenStats | None:
        """Get statistics from last operation."""
        return self._last_stats

    @property
    def retrieved_budget(self) -> int:
        """Get budget for retrieved memories."""
        return self.config.retrieved_memories

    @property
    def using_tiktoken(self) -> bool:
        """Check if tiktoken is being used for accurate counting."""
        return self._tiktoken_available

    def estimate_tokens(self, text: str) -> int:
        """
        Estimate token count for text.

        Uses tiktoken for accurate counting if available and enabled,
        otherwise falls back to character-based estimation.

        Args:
            text: Text to estimate tokens for

        Returns:
            Token count (accurate with tiktoken, estimated without)
        """
        if not text:
            return 0

        # Use tiktoken if available
        if self._tiktoken_available and self._encoder is not None:
            try:
                return len(self._encoder.encode(text))
            except Exception as e:
                logger.warning(f"tiktoken encoding failed, using fallback: {e}")

        # Fallback to character-based estimation
        return max(1, int(len(text) / self._chars_per_token))

    def estimate_result_tokens(self, result: MemoryResult) -> int:
        """
        Estimate tokens for a memory result.

        Includes content and minimal metadata overhead.

        Args:
            result: Memory result to estimate

        Returns:
            Token count for the result
        """
        content_tokens = self.estimate_tokens(result.entry.content)

        # Add overhead for metadata (type, score info, etc.)
        # Approximate: ~20 tokens for formatting
        overhead = 20

        return content_tokens + overhead

    def update_result_tokens(self, result: MemoryResult) -> MemoryResult:
        """
        Update token_count field on a result.

        Args:
            result: Memory result to update

        Returns:
            Same result with token_count updated
        """
        result.token_count = self.estimate_result_tokens(result)
        return result

    def prune_to_budget(
        self,
        results: Sequence[MemoryResult],
        budget: int | None = None,
    ) -> list[MemoryResult]:
        """
        Prune results to fit within token budget.

        Assumes results are already sorted by priority (final_score).
        Keeps highest-priority results that fit within budget.

        Args:
            results: Memory results sorted by priority (best first)
            budget: Token budget to fit within (defaults to retrieved_memories)

        Returns:
            List of results that fit within budget
        """
        if budget is None:
            budget = self.retrieved_budget

        if not results:
            self._last_stats = TokenStats(
                total_tokens_available=budget,
                tokens_remaining=budget,
            )
            return []

        kept: list[MemoryResult] = []
        tokens_used = 0
        total_tokens_all = 0

        for result in results:
            # Estimate tokens for this result
            result_tokens = self.estimate_result_tokens(result)
            total_tokens_all += result_tokens
            result.token_count = result_tokens

            # Check if it fits
            if tokens_used + result_tokens <= budget:
                kept.append(result)
                tokens_used += result_tokens
            else:
                # Budget exhausted, remaining results are pruned
                logger.debug(
                    f"Token budget exhausted at {len(kept)} results "
                    f"({tokens_used}/{budget} tokens used)"
                )
                break

        # Collect statistics
        self._last_stats = TokenStats(
            total_candidates=len(results),
            total_tokens_available=budget,
            total_tokens_used=tokens_used,
            tokens_remaining=budget - tokens_used,
            results_kept=len(kept),
            results_pruned=len(results) - len(kept),
            avg_tokens_per_result=total_tokens_all / len(results) if results else 0,
        )

        logger.debug(
            f"Token budget: kept {len(kept)}/{len(results)} results, "
            f"used {tokens_used}/{budget} tokens"
        )

        return kept

    def allocate_by_category(
        self,
        results: Sequence[MemoryResult],
        category_budgets: dict[str, int],
    ) -> dict[str, list[MemoryResult]]:
        """
        Allocate results to categories with separate budgets.

        Useful for allocating tokens to different memory types:
        - core: PERSONA, SECURITY (always loaded)
        - retrieved: SEMANTIC, EPISODIC, etc.
        - tools: TOOL memories

        Args:
            results: Memory results to allocate
            category_budgets: Mapping of category name to token budget

        Returns:
            Dictionary mapping category to allocated results

        Example:
            >>> budgets = {"core": 400, "retrieved": 3000}
            >>> allocated = manager.allocate_by_category(results, budgets)
        """
        allocations: dict[str, CategoryAllocation] = {
            cat: CategoryAllocation(category=cat, budget=budget)
            for cat, budget in category_budgets.items()
        }

        # Default category for unmatched results
        if "other" not in allocations:
            allocations["other"] = CategoryAllocation(category="other", budget=0)

        for result in results:
            result_tokens = self.estimate_result_tokens(result)
            result.token_count = result_tokens

            # Determine category based on memory type
            category = self._categorize_result(result, list(category_budgets.keys()))

            allocation = allocations.get(category, allocations["other"])

            # Add if fits in budget
            if allocation.used + result_tokens <= allocation.budget:
                allocation.results.append(result)
                allocation.used += result_tokens

        return {cat: alloc.results for cat, alloc in allocations.items()}

    def _categorize_result(self, result: MemoryResult, categories: list[str]) -> str:
        """
        Determine category for a result based on memory type.

        Override this method for custom categorization logic.

        Args:
            result: Memory result to categorize
            categories: Available category names

        Returns:
            Category name
        """
        memory_type = result.entry.memory_type.value.lower()

        # Core memories (always loaded)
        if memory_type in ("persona", "security"):
            return "core" if "core" in categories else "other"

        # Tool memories
        if memory_type == "tool":
            return "tools" if "tools" in categories else "retrieved"

        # Working/session memories
        if memory_type == "working":
            return "session" if "session" in categories else "retrieved"

        # Default: retrieved memories
        return "retrieved" if "retrieved" in categories else "other"

    def fit_conversation_history(
        self,
        messages: Sequence[dict[str, str]],
        budget: int | None = None,
    ) -> list[dict[str, str]]:
        """
        Fit conversation history within budget.

        Keeps most recent messages that fit within budget.

        Args:
            messages: Conversation messages (oldest first)
            budget: Token budget (defaults to conversation_history)

        Returns:
            List of messages that fit (most recent prioritized)
        """
        if budget is None:
            budget = self.config.conversation_history

        if not messages:
            return []

        # Process from newest to oldest
        reversed_messages = list(reversed(messages))
        kept: list[dict[str, str]] = []
        tokens_used = 0

        for msg in reversed_messages:
            content = msg.get("content", "")
            role = msg.get("role", "")
            msg_tokens = self.estimate_tokens(f"{role}: {content}")

            if tokens_used + msg_tokens <= budget:
                kept.append(msg)
                tokens_used += msg_tokens
            else:
                break

        # Restore chronological order
        return list(reversed(kept))

    def calculate_total_usage(
        self,
        system_tokens: int = 0,
        core_tokens: int = 0,
        retrieved_tokens: int = 0,
        conversation_tokens: int = 0,
    ) -> dict[str, int]:
        """
        Calculate total token usage across categories.

        Args:
            system_tokens: Tokens used for system prompt
            core_tokens: Tokens used for core memories
            retrieved_tokens: Tokens used for retrieved memories
            conversation_tokens: Tokens used for conversation

        Returns:
            Dictionary with usage breakdown and remaining
        """
        total_used = system_tokens + core_tokens + retrieved_tokens + conversation_tokens
        available_for_response = max(0, self.config.total_budget - total_used)

        return {
            "system_prompt": system_tokens,
            "core_memory": core_tokens,
            "retrieved_memories": retrieved_tokens,
            "conversation_history": conversation_tokens,
            "total_used": total_used,
            "response_available": available_for_response,
            "total_budget": self.config.total_budget,
            "utilization_percent": round((total_used / self.config.total_budget) * 100, 1),
        }
