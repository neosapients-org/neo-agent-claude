"""
Salience scoring for memory storage decisions.

Uses a lightweight LLM call to rate the importance of extracted facts.
Inspired by Stanford Generative Agents' "poignancy scoring" pattern.

The scorer is an OPTIONAL utility — apps can:
1. Use SalienceScorer with a custom prompt (recommended)
2. Pass their own pre-computed salience to StorageGateway
3. Skip salience scoring entirely (StorageGateway threshold=0.0)
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# =============================================================================
# Models
# =============================================================================


class ScoredFact(BaseModel):
    """A fact with its salience score from LLM evaluation."""

    text: str = Field(..., description="The fact text")
    salience: float = Field(
        ..., ge=0.0, le=1.0, description="Salience score on 0.0-1.0 scale"
    )
    reasoning: str = Field(
        default="", description="Brief reasoning for the score"
    )


class ScoredFactList(BaseModel):
    """Structured output schema for batch scoring."""

    scored_facts: list[ScoredFact] = Field(
        ..., description="List of facts with salience scores"
    )


# =============================================================================
# Default Prompt
# =============================================================================

DEFAULT_SCORING_PROMPT = """You are a memory importance scorer for an AI assistant's long-term memory system.

Given a list of facts extracted from a conversation, rate each fact's importance
on a scale of 0.0 to 1.0:

- 0.0-0.2: Trivial — greetings, small talk, filler, acknowledgments
- 0.2-0.4: Low — routine updates, common knowledge, temporary info
- 0.4-0.6: Medium — useful preferences, general observations, context
- 0.6-0.8: High — specific goals, important decisions, behavioral patterns
- 0.8-1.0: Critical — identity changes, risk profile shifts, compliance events,
  key relationships, financial decisions

For each fact, provide:
1. The salience score (0.0-1.0)
2. Brief reasoning (one sentence)

{context_section}

Facts to score:
{facts_list}

Respond with a JSON object matching this schema:
{{
  "scored_facts": [
    {{"text": "the fact", "salience": 0.85, "reasoning": "one sentence reason"}}
  ]
}}"""


# =============================================================================
# Config
# =============================================================================


@dataclass
class SalienceScorerConfig:
    """Configuration for the salience scorer."""

    # LLM config for the scoring call (uses a cheap/fast model)
    llm_provider: str = "openai"
    llm_model: str = "gpt-4.1-nano"
    llm_temperature: float = 0.0
    llm_max_tokens: int = 1024
    llm_api_key: str | None = None  # Resolved from env if not set

    # Scoring behavior
    default_score: float = 0.5  # Fallback if LLM call fails
    score_scale: tuple[float, float] = (0.0, 1.0)

    # Custom scoring prompt (the key customization point)
    scoring_prompt: str | None = None  # None = use built-in default

    # Additional LLM kwargs (e.g. base_url, organization, etc.)
    llm_kwargs: dict[str, Any] = field(default_factory=dict)


# =============================================================================
# Scorer
# =============================================================================


class SalienceScorer:
    """
    Lightweight LLM-based salience scorer for memory facts.

    Uses a single LLM call to score one or more facts for importance.
    Designed to be CHEAP and FAST:
    - Uses gpt-4.1-nano by default (cheapest available)
    - Temperature=0 for deterministic output
    - Batch scoring: N facts in one call (not N calls)
    - Structured output for reliable parsing

    Usage:
        >>> scorer = SalienceScorer()
        >>> results = await scorer.score_facts(
        ...     facts=["Client moving to Moderate risk", "User said hello"],
        ...     context="Wealth management RM conversation"
        ... )
        >>> results
        [
            ScoredFact(text="Client moving to Moderate risk", salience=0.95, ...),
            ScoredFact(text="User said hello", salience=0.1, ...),
        ]
    """

    def __init__(self, config: SalienceScorerConfig | None = None) -> None:
        self.config = config or SalienceScorerConfig()
        self._client: Any = None

    async def _get_client(self) -> Any:
        """Lazy-initialize the OpenAI async client."""
        if self._client is not None:
            return self._client

        try:
            from openai import AsyncOpenAI
        except ImportError as e:
            raise ImportError(
                "openai package required for SalienceScorer. "
                "Install with: pip install openai"
            ) from e

        kwargs: dict[str, Any] = {}
        if self.config.llm_api_key:
            kwargs["api_key"] = self.config.llm_api_key
        kwargs.update(self.config.llm_kwargs)

        self._client = AsyncOpenAI(**kwargs)
        return self._client

    def _build_prompt(
        self,
        facts: list[str],
        context: str | None = None,
        prompt_override: str | None = None,
    ) -> str:
        """Build the scoring prompt from template + facts."""
        template = prompt_override or self.config.scoring_prompt or DEFAULT_SCORING_PROMPT

        # Build context section
        context_section = ""
        if context:
            context_section = f"Context about this conversation:\n{context}"

        # Build facts list
        facts_list = "\n".join(f"{i + 1}. {fact}" for i, fact in enumerate(facts))

        return template.replace(
            "{context_section}", context_section
        ).replace(
            "{facts_list}", facts_list
        )

    def _parse_response(self, content: str, original_facts: list[str]) -> list[ScoredFact]:
        """Parse LLM response into ScoredFact list."""
        try:
            data = json.loads(content)
            result = ScoredFactList.model_validate(data)
            return result.scored_facts
        except (json.JSONDecodeError, Exception) as e:
            logger.warning(f"Failed to parse scoring response: {e}. Using default scores.")
            return [
                ScoredFact(
                    text=fact,
                    salience=self.config.default_score,
                    reasoning="Parsing failed — using default score",
                )
                for fact in original_facts
            ]

    async def score_facts(
        self,
        facts: list[str],
        *,
        context: str | None = None,
        prompt_override: str | None = None,
    ) -> list[ScoredFact]:
        """
        Score a batch of facts for salience in a single LLM call.

        Args:
            facts: List of extracted fact strings to score
            context: Optional context about the domain/conversation
            prompt_override: Optional per-call prompt override

        Returns:
            List of ScoredFact with salience scores (0.0-1.0)
        """
        if not facts:
            return []

        prompt = self._build_prompt(facts, context, prompt_override)

        try:
            client = await self._get_client()
            response = await client.chat.completions.create(
                model=self.config.llm_model,
                temperature=self.config.llm_temperature,
                max_tokens=self.config.llm_max_tokens,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a memory importance scorer. Respond only with valid JSON.",
                    },
                    {"role": "user", "content": prompt},
                ],
                response_format={"type": "json_object"},
            )

            content = response.choices[0].message.content or "{}"
            return self._parse_response(content, facts)

        except Exception as e:
            logger.warning(f"Salience scoring LLM call failed: {e}. Using default scores.")
            return [
                ScoredFact(
                    text=fact,
                    salience=self.config.default_score,
                    reasoning=f"LLM call failed: {e}",
                )
                for fact in facts
            ]

    async def score_single(
        self,
        fact: str,
        *,
        context: str | None = None,
    ) -> ScoredFact:
        """
        Score a single fact. Convenience wrapper around score_facts().

        Args:
            fact: The fact string to score
            context: Optional context

        Returns:
            ScoredFact with salience score
        """
        results = await self.score_facts([fact], context=context)
        return results[0]


__all__ = [
    "DEFAULT_SCORING_PROMPT",
    "SalienceScorer",
    "SalienceScorerConfig",
    "ScoredFact",
    "ScoredFactList",
]
