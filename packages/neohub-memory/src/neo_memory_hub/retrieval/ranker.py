"""
Multi-factor ranking for memory retrieval.

Implements the ranking formula:
    Final Score = α × Relevance + β × Recency + γ × Importance

Where:
- Relevance: Vector similarity score (0-1) from Mem0/Milvus
- Recency: exp(-λ × hours_since_access) - exponential decay
- Importance: User-defined score (1-10) normalized to 0-1
"""

import logging
import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime

from neo_memory_hub.domain.memory import MemoryEntry, MemoryResult
from neo_memory_hub.retrieval.context import RetrievalStrategy


logger = logging.getLogger(__name__)


@dataclass
class RankingConfig:
    """
    Configuration for multi-factor ranking.

    Attributes:
        relevance_weight: Weight for relevance score (α)
        recency_weight: Weight for recency score (β)
        importance_weight: Weight for importance score (γ)
        recency_decay_lambda: Decay rate for recency (λ)
        strategy: Ranking strategy to use

    Note:
        Weights must sum to 1.0 for proper normalization.
    """

    relevance_weight: float = 0.5
    recency_weight: float = 0.3
    importance_weight: float = 0.2
    recency_decay_lambda: float = 0.01
    strategy: RetrievalStrategy = RetrievalStrategy.MULTI_FACTOR

    def __post_init__(self) -> None:
        """Validate weights sum to 1.0."""
        total = self.relevance_weight + self.recency_weight + self.importance_weight
        if abs(total - 1.0) > 0.001:
            raise ValueError(
                f"Ranking weights must sum to 1.0, got {total:.3f}. "
                f"(α={self.relevance_weight}, β={self.recency_weight}, γ={self.importance_weight})"
            )
        if self.recency_decay_lambda < 0:
            raise ValueError("recency_decay_lambda must be >= 0")

    @classmethod
    def relevance_only(cls) -> "RankingConfig":
        """Create config that only uses relevance score."""
        return cls(
            relevance_weight=1.0,
            recency_weight=0.0,
            importance_weight=0.0,
            strategy=RetrievalStrategy.RELEVANCE_ONLY,
        )

    @classmethod
    def recency_biased(cls) -> "RankingConfig":
        """Create config with higher recency weight."""
        return cls(
            relevance_weight=0.3,
            recency_weight=0.5,
            importance_weight=0.2,
            strategy=RetrievalStrategy.RECENCY_BIASED,
        )

    @classmethod
    def balanced(cls) -> "RankingConfig":
        """Create balanced config (default)."""
        return cls(
            relevance_weight=0.5,
            recency_weight=0.3,
            importance_weight=0.2,
            strategy=RetrievalStrategy.MULTI_FACTOR,
        )


@dataclass
class RankingStats:
    """Statistics from a ranking operation."""

    total_candidates: int = 0
    ranked_count: int = 0
    avg_relevance: float = 0.0
    avg_recency: float = 0.0
    avg_importance: float = 0.0
    avg_final_score: float = 0.0
    min_final_score: float = 0.0
    max_final_score: float = 0.0
    config_used: RankingConfig = field(default_factory=RankingConfig)


class MultiFactorRanker:
    """
    Multi-factor ranking system for memory retrieval.

    Combines multiple signals to rank memories:
    1. Relevance (vector similarity from search)
    2. Recency (how recently accessed)
    3. Importance (user-defined priority)

    Formula:
        Score = α×Relevance + β×Recency + γ×Importance

    Example:
        >>> config = RankingConfig(
        ...     relevance_weight=0.5,
        ...     recency_weight=0.3,
        ...     importance_weight=0.2
        ... )
        >>> ranker = MultiFactorRanker(config)
        >>> ranked = ranker.rank(candidates)
    """

    def __init__(self, config: RankingConfig | None = None):
        """
        Initialize ranker with configuration.

        Args:
            config: Ranking configuration. Defaults to balanced weights.
        """
        self.config = config or RankingConfig.balanced()
        self._last_stats: RankingStats | None = None

    @property
    def alpha(self) -> float:
        """Relevance weight."""
        return self.config.relevance_weight

    @property
    def beta(self) -> float:
        """Recency weight."""
        return self.config.recency_weight

    @property
    def gamma(self) -> float:
        """Importance weight."""
        return self.config.importance_weight

    @property
    def decay_lambda(self) -> float:
        """Recency decay rate."""
        return self.config.recency_decay_lambda

    @property
    def last_stats(self) -> RankingStats | None:
        """Get statistics from last ranking operation."""
        return self._last_stats

    def rank(
        self,
        candidates: Sequence[MemoryResult],
        reference_time: datetime | None = None,
    ) -> list[MemoryResult]:
        """
        Rank memory results using multi-factor scoring.

        Args:
            candidates: Memory results with relevance scores
            reference_time: Reference time for recency calculation
                           (defaults to now)

        Returns:
            List of MemoryResults sorted by final_score descending
        """
        if not candidates:
            self._last_stats = RankingStats(config_used=self.config)
            return []

        reference_time = reference_time or datetime.now(UTC)
        results: list[MemoryResult] = []

        # Calculate scores for each candidate
        for result in candidates:
            scored = self._calculate_scores(result, reference_time)
            results.append(scored)

        # Sort by final score descending
        results.sort(key=lambda r: r.final_score, reverse=True)

        # Collect statistics
        self._collect_stats(results)

        logger.debug(
            f"Ranked {len(results)} memories. "
            f"Score range: {self._last_stats.min_final_score:.3f} - {self._last_stats.max_final_score:.3f}"
        )

        return results

    def rank_entries(
        self,
        entries: Sequence[MemoryEntry],
        relevance_scores: dict[str, float] | None = None,
        reference_time: datetime | None = None,
    ) -> list[MemoryResult]:
        """
        Rank raw memory entries (without pre-computed relevance).

        Args:
            entries: Memory entries to rank
            relevance_scores: Mapping of entry ID to relevance score
            reference_time: Reference time for recency

        Returns:
            List of MemoryResults sorted by final_score
        """
        relevance_scores = relevance_scores or {}
        candidates = []

        for entry in entries:
            # Get relevance score or default to 0.5
            rel_score = relevance_scores.get(entry.id, 0.5)
            result = MemoryResult.from_entry(
                entry=entry,
                relevance_score=rel_score,
                retrieval_method="ranked",
            )
            candidates.append(result)

        return self.rank(candidates, reference_time)

    def _calculate_scores(
        self,
        result: MemoryResult,
        reference_time: datetime,
    ) -> MemoryResult:
        """
        Calculate all component scores and final score.

        Args:
            result: Memory result with relevance score
            reference_time: Reference time for recency

        Returns:
            Updated MemoryResult with all scores filled
        """
        entry = result.entry

        # Relevance score (already computed from vector search)
        relevance = result.relevance_score

        # Recency score (exponential decay)
        recency = self._calculate_recency_score(entry.accessed_at, reference_time)

        # Importance score (normalize from 1-10 to 0-1)
        importance = self._normalize_importance(entry.importance)

        # Final score using weighted formula
        final_score = self._calculate_final_score(relevance, recency, importance)

        # Update result with scores
        result.recency_score = recency
        result.importance_score = importance
        result.final_score = final_score

        return result

    def _calculate_recency_score(
        self,
        accessed_at: datetime,
        reference_time: datetime,
    ) -> float:
        """
        Calculate recency score using exponential decay.

        Formula: exp(-λ × hours_since_access)

        Args:
            accessed_at: When memory was last accessed
            reference_time: Reference time (usually now)

        Returns:
            Recency score between 0 and 1 (1 = most recent)
        """
        # Handle timezone-naive datetimes
        if accessed_at.tzinfo is None:
            accessed_at = accessed_at.replace(tzinfo=UTC)
        if reference_time.tzinfo is None:
            reference_time = reference_time.replace(tzinfo=UTC)

        # Calculate hours since access
        delta = reference_time - accessed_at
        hours = max(0, delta.total_seconds() / 3600)

        # Exponential decay: score decreases as time increases
        score = math.exp(-self.decay_lambda * hours)

        return min(1.0, max(0.0, score))

    def _normalize_importance(self, importance: float) -> float:
        """
        Normalize importance from 1-10 scale to 0-1.

        Args:
            importance: Importance score (1-10)

        Returns:
            Normalized score (0-1)
        """
        # Clamp to valid range
        importance = min(10.0, max(1.0, importance))
        # Normalize: (value - min) / (max - min)
        return (importance - 1.0) / 9.0

    def _calculate_final_score(
        self,
        relevance: float,
        recency: float,
        importance: float,
    ) -> float:
        """
        Calculate final score using weighted formula.

        Formula: α×Relevance + β×Recency + γ×Importance

        Args:
            relevance: Relevance score (0-1)
            recency: Recency score (0-1)
            importance: Importance score (0-1)

        Returns:
            Final score (0-1)
        """
        if self.config.strategy == RetrievalStrategy.RELEVANCE_ONLY:
            return relevance

        score = self.alpha * relevance + self.beta * recency + self.gamma * importance

        # Ensure score is in valid range
        return min(1.0, max(0.0, score))

    def _collect_stats(self, results: list[MemoryResult]) -> None:
        """Collect statistics from ranking results."""
        if not results:
            self._last_stats = RankingStats(config_used=self.config)
            return

        relevances = [r.relevance_score for r in results]
        recencies = [r.recency_score for r in results]
        importances = [r.importance_score for r in results]
        final_scores = [r.final_score for r in results]

        self._last_stats = RankingStats(
            total_candidates=len(results),
            ranked_count=len(results),
            avg_relevance=sum(relevances) / len(relevances),
            avg_recency=sum(recencies) / len(recencies),
            avg_importance=sum(importances) / len(importances),
            avg_final_score=sum(final_scores) / len(final_scores),
            min_final_score=min(final_scores),
            max_final_score=max(final_scores),
            config_used=self.config,
        )

    def adjust_weights(
        self,
        relevance_weight: float | None = None,
        recency_weight: float | None = None,
        importance_weight: float | None = None,
    ) -> None:
        """
        Adjust ranking weights dynamically.

        Weights are normalized to sum to 1.0.

        Args:
            relevance_weight: New relevance weight
            recency_weight: New recency weight
            importance_weight: New importance weight
        """
        # Get current or new values
        alpha = relevance_weight if relevance_weight is not None else self.alpha
        beta = recency_weight if recency_weight is not None else self.beta
        gamma = importance_weight if importance_weight is not None else self.gamma

        # Normalize to sum to 1.0
        total = alpha + beta + gamma
        if total <= 0:
            raise ValueError("At least one weight must be positive")

        self.config = RankingConfig(
            relevance_weight=alpha / total,
            recency_weight=beta / total,
            importance_weight=gamma / total,
            recency_decay_lambda=self.decay_lambda,
            strategy=self.config.strategy,
        )

        logger.info(
            f"Adjusted ranking weights: α={self.alpha:.2f}, β={self.beta:.2f}, γ={self.gamma:.2f}"
        )
