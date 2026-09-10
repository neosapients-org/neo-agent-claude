"""
Consolidated unit tests for retrieval system.

Tests critical functionality for:
- MultiFactorRanker: scoring formula, ranking order
- TokenBudgetManager: pruning, budget management
- RetrievalPipeline: stage execution
"""

import math
from datetime import UTC, datetime, timedelta

import pytest

from neo_memory_hub.domain.memory import MemoryEntry, MemoryResult
from neo_memory_hub.domain.scope import IsolationScope
from neo_memory_hub.domain.types import MemoryType
from neo_memory_hub.retrieval.context import (
    RetrievalContext,
    RetrievalOptions,
    RetrievalStrategy,
    TokenBudgetConfig,
)
from neo_memory_hub.retrieval.ranker import MultiFactorRanker, RankingConfig
from neo_memory_hub.retrieval.token_budget import TokenBudgetManager


# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture
def default_scope() -> IsolationScope:
    """Default scope for tests."""
    return IsolationScope(tenant_id="test_tenant", user_id="test_user")


def create_memory_result(
    content: str,
    scope: IsolationScope,
    relevance_score: float = 0.5,
    importance: float = 5.0,
    accessed_at: datetime | None = None,
) -> MemoryResult:
    """Helper to create memory results."""
    entry = MemoryEntry(
        content=content,
        memory_type=MemoryType.SEMANTIC,
        scope=scope,
        importance=importance,
        accessed_at=accessed_at or datetime.now(UTC),
    )
    return MemoryResult.from_entry(entry=entry, relevance_score=relevance_score)


# ============================================================================
# RankingConfig Tests
# ============================================================================


class TestRankingConfig:
    """Tests for RankingConfig."""

    def test_weights_must_sum_to_one(self) -> None:
        """Weights must sum to 1.0."""
        # Valid
        config = RankingConfig(
            relevance_weight=0.5,
            recency_weight=0.3,
            importance_weight=0.2,
        )
        assert config.relevance_weight == 0.5

        # Invalid - raises
        with pytest.raises(ValueError, match="must sum to 1.0"):
            RankingConfig(
                relevance_weight=0.5,
                recency_weight=0.5,
                importance_weight=0.5,
            )

    def test_presets(self) -> None:
        """Should have working presets."""
        relevance_only = RankingConfig.relevance_only()
        assert relevance_only.relevance_weight == 1.0
        assert relevance_only.recency_weight == 0.0

        balanced = RankingConfig.balanced()
        assert balanced.relevance_weight == 0.5
        assert balanced.recency_weight == 0.3


# ============================================================================
# MultiFactorRanker Tests
# ============================================================================


class TestMultiFactorRanker:
    """Tests for MultiFactorRanker."""

    def test_rank_empty_list(self) -> None:
        """Should handle empty candidate list."""
        ranker = MultiFactorRanker()
        results = ranker.rank([])
        assert results == []

    def test_recency_score_decay(self, default_scope: IsolationScope) -> None:
        """Recency should decay over time."""
        ranker = MultiFactorRanker()
        reference_time = datetime.now(UTC)

        # Recent memory
        recent = create_memory_result(
            "Recent",
            default_scope,
            accessed_at=reference_time,
        )

        # Old memory (100 hours ago)
        old_time = reference_time - timedelta(hours=100)
        old = create_memory_result(
            "Old",
            default_scope,
            accessed_at=old_time,
        )

        ranked = ranker.rank([recent, old], reference_time=reference_time)

        # Recent should have higher recency score
        assert ranked[0].recency_score > ranked[1].recency_score

        # Check decay formula: exp(-0.01 * 100) ≈ 0.368
        expected_old_recency = math.exp(-0.01 * 100)
        assert abs(ranked[1].recency_score - expected_old_recency) < 0.01

    def test_ranking_by_relevance(self, default_scope: IsolationScope) -> None:
        """Higher relevance should rank higher."""
        ranker = MultiFactorRanker()
        now = datetime.now(UTC)

        results = [
            create_memory_result("Low", default_scope, relevance_score=0.3, accessed_at=now),
            create_memory_result("High", default_scope, relevance_score=0.9, accessed_at=now),
        ]

        ranked = ranker.rank(results)
        assert ranked[0].entry.content == "High"
        assert ranked[1].entry.content == "Low"

    def test_ranking_by_importance(self, default_scope: IsolationScope) -> None:
        """Higher importance should boost ranking."""
        ranker = MultiFactorRanker()
        now = datetime.now(UTC)

        results = [
            create_memory_result(
                "Low importance",
                default_scope,
                relevance_score=0.5,
                importance=1.0,
                accessed_at=now,
            ),
            create_memory_result(
                "High importance",
                default_scope,
                relevance_score=0.5,
                importance=10.0,
                accessed_at=now,
            ),
        ]

        ranked = ranker.rank(results)
        assert ranked[0].entry.content == "High importance"

    def test_relevance_only_ignores_other_factors(self, default_scope: IsolationScope) -> None:
        """Relevance-only mode should ignore recency and importance."""
        ranker = MultiFactorRanker(RankingConfig.relevance_only())
        now = datetime.now(UTC)
        old = now - timedelta(hours=1000)

        results = [
            create_memory_result(
                "Recent low relevance",
                default_scope,
                relevance_score=0.3,
                importance=10.0,
                accessed_at=now,
            ),
            create_memory_result(
                "Old high relevance",
                default_scope,
                relevance_score=0.9,
                importance=1.0,
                accessed_at=old,
            ),
        ]

        ranked = ranker.rank(results)
        # High relevance should win despite being old with low importance
        assert ranked[0].entry.content == "Old high relevance"


# ============================================================================
# TokenBudgetManager Tests
# ============================================================================


class TestTokenBudgetManager:
    """Tests for TokenBudgetManager."""

    def test_estimate_tokens(self) -> None:
        """Should estimate tokens from text."""
        manager = TokenBudgetManager()

        # ~4 chars per token
        tokens = manager.estimate_tokens("This is a test.")  # 15 chars
        assert 3 <= tokens <= 5

    def test_prune_to_budget_all_fit(self, default_scope: IsolationScope) -> None:
        """Should keep all results if within budget."""
        manager = TokenBudgetManager()

        results = [create_memory_result(f"Short {i}", default_scope) for i in range(3)]

        pruned = manager.prune_to_budget(results, budget=1000)
        assert len(pruned) == 3

    def test_prune_to_budget_exceeds(self, default_scope: IsolationScope) -> None:
        """Should prune results exceeding budget."""
        manager = TokenBudgetManager()

        # Create results with known sizes
        results = [
            create_memory_result("A" * 200, default_scope),  # ~50 tokens + overhead
            create_memory_result("B" * 200, default_scope),
            create_memory_result("C" * 200, default_scope),
        ]

        # Small budget should prune
        pruned = manager.prune_to_budget(results, budget=100)
        assert len(pruned) < len(results)

    def test_prune_preserves_order(self, default_scope: IsolationScope) -> None:
        """Should preserve input order (assumed sorted by score)."""
        manager = TokenBudgetManager()

        results = [
            create_memory_result("First", default_scope),
            create_memory_result("Second", default_scope),
        ]

        pruned = manager.prune_to_budget(results, budget=1000)
        assert pruned[0].entry.content == "First"
        assert pruned[1].entry.content == "Second"


# ============================================================================
# TokenBudgetConfig Tests
# ============================================================================


class TestTokenBudgetConfig:
    """Tests for TokenBudgetConfig."""

    def test_default_valid(self) -> None:
        """Default config should be valid."""
        config = TokenBudgetConfig()
        assert config.total_budget == 8000
        assert config.retrieved_memories == 3000

    def test_allocations_must_fit(self) -> None:
        """Allocations must not exceed total budget."""
        with pytest.raises(ValueError, match="exceeds budget"):
            TokenBudgetConfig(
                total_budget=1000,
                system_prompt=500,
                core_memory=400,
                retrieved_memories=500,
                conversation_history=500,
                response_buffer=100,
            )

    def test_for_context_size(self) -> None:
        """Should scale for different context sizes."""
        config_4k = TokenBudgetConfig.for_context_size(4096)
        config_32k = TokenBudgetConfig.for_context_size(32768)

        assert config_4k.total_budget == 4096
        assert config_32k.total_budget == 32768
        assert config_32k.retrieved_memories > config_4k.retrieved_memories


# ============================================================================
# RetrievalContext Tests
# ============================================================================


class TestRetrievalOptions:
    """Tests for RetrievalOptions."""

    def test_defaults(self) -> None:
        """Should have sensible defaults."""
        options = RetrievalOptions()
        assert options.limit == 10
        assert options.min_relevance == 0.3
        assert options.strategy == RetrievalStrategy.MULTI_FACTOR

    def test_validation(self) -> None:
        """Should validate option values."""
        with pytest.raises(ValueError, match="limit must be >= 1"):
            RetrievalOptions(limit=0)

        with pytest.raises(ValueError, match="min_relevance must be between"):
            RetrievalOptions(min_relevance=1.5)
