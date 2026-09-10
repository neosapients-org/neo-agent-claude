"""
Storage Gateway for Neo Memory Hub.

Provides basic content validation and importance scoring for memory storage.

DESIGN PHILOSOPHY:
------------------
This module is intentionally MINIMAL. We rely on Mem0 for:
- Fact extraction (via `infer=True`)
- Deduplication
- Conflict resolution (ADD/UPDATE/DELETE)
- Semantic understanding

Neo's Storage Gateway provides only:
- Basic content validation (not empty, minimum length)
- Optional importance scoring (configurable)
- Storage decision routing

WHY NO KEYWORD/PATTERN FILTERING:
---------------------------------
Hardcoded keyword patterns are NOT production-grade because:
1. They don't generalize across domains
2. They miss context and nuance
3. They require constant maintenance
4. LLMs (via Mem0) do this better

If you need content filtering, use Mem0's `custom_fact_extraction_prompt`
to customize what gets stored.
"""

import logging
from dataclasses import dataclass
from enum import Enum


logger = logging.getLogger(__name__)


class StorageDecision(str, Enum):
    """Decision for whether to store content."""

    STORE = "store"  # Worth storing - send to Mem0
    SKIP = "skip"  # Not worth storing - empty/too short


@dataclass
class WorthinessResult:
    """Result of worthiness evaluation."""

    decision: StorageDecision
    reason: str
    content_length: int = 0


@dataclass
class StorageGatewayConfig:
    """
    Configuration for the storage gateway.

    Note: For semantic content filtering, use Mem0's `custom_fact_extraction_prompt`.
    Pattern-based filtering doesn't generalize across domains and Mem0's LLM-based
    extraction does this better.
    """

    # Basic content validation
    min_content_length: int = 3  # Minimum characters to store
    max_content_length: int = 10000  # Maximum characters (prevent abuse)

    # Importance scoring
    default_importance: float = 5.0  # Default importance if not specified

    # Salience gating (v0.2)
    min_salience_threshold: float = 0.0
    """Minimum salience score to accept for storage.
    Scale: 0.0-1.0 (matches CSV UnifiedMemorySchema salience_score).
    0.0 = accept all (no gating). 0.7 = only high-value insights stored.
    """

    # Feature toggles
    enable_worthiness_check: bool = True


class StorageGateway:
    """
    Lightweight gateway for memory storage decisions.

    This gateway performs basic validation before sending content to Mem0.
    All intelligent processing (fact extraction, deduplication, conflict
    resolution) is handled by Mem0's `add(infer=True)`.

    Features:
    - Basic content validation (non-empty, minimum length)
    - Configurable importance scoring
    - Simple and predictable behavior

    Usage:
        >>> gateway = StorageGateway()
        >>> result = gateway.evaluate("I prefer vegetarian food")
        >>> if result.decision == StorageDecision.STORE:
        ...     await mem0.add(content, importance=5.0)

    For advanced content filtering, configure Mem0's `custom_fact_extraction_prompt`
    instead of adding logic here.
    """

    def __init__(self, config: StorageGatewayConfig | None = None):
        """
        Initialize the storage gateway.

        Args:
            config: Gateway configuration (uses defaults if not provided)
        """
        self.config = config or StorageGatewayConfig()

    def evaluate(self, content: str, *, salience: float | None = None) -> WorthinessResult:
        """
        Evaluate content for storage worthiness.

        This is a basic validation - we check if content is:
        1. Not None or empty
        2. Meets minimum length requirement
        3. Doesn't exceed maximum length
        4. Meets salience threshold (if salience is provided)

        All semantic evaluation is delegated to Mem0.

        Args:
            content: Content to evaluate
            salience: Salience score from LLM extraction (0.0-1.0).
                If below min_salience_threshold, memory is rejected.

        Returns:
            WorthinessResult with decision
        """
        if not self.config.enable_worthiness_check:
            return WorthinessResult(
                decision=StorageDecision.STORE,
                reason="Worthiness check disabled",
                content_length=len(content) if content else 0,
            )

        # Check for empty content
        if not content:
            return WorthinessResult(
                decision=StorageDecision.SKIP,
                reason="Content is empty",
                content_length=0,
            )

        content = content.strip()
        content_length = len(content)

        # Check minimum length
        if content_length < self.config.min_content_length:
            return WorthinessResult(
                decision=StorageDecision.SKIP,
                reason=f"Content too short ({content_length} < {self.config.min_content_length})",
                content_length=content_length,
            )

        # Check maximum length
        if content_length > self.config.max_content_length:
            return WorthinessResult(
                decision=StorageDecision.SKIP,
                reason=f"Content too long ({content_length} > {self.config.max_content_length})",
                content_length=content_length,
            )

        # Salience gate (v0.2)
        if (
            salience is not None
            and self.config.min_salience_threshold > 0.0
            and salience < self.config.min_salience_threshold
        ):
            return WorthinessResult(
                decision=StorageDecision.SKIP,
                reason=(
                    f"Salience too low "
                    f"({salience:.2f} < {self.config.min_salience_threshold})"
                ),
                content_length=content_length,
            )

        return WorthinessResult(
            decision=StorageDecision.STORE,
            reason="Content passed validation",
            content_length=content_length,
        )


__all__ = [
    "StorageDecision",
    "StorageGateway",
    "StorageGatewayConfig",
    "WorthinessResult",
]
