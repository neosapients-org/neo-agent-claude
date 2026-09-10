"""
Tests for SalienceScorer utility (Phase 3.5).

All tests mock the OpenAI client — no real API calls.
"""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from neo_memory_hub.core.salience import (
    DEFAULT_SCORING_PROMPT,
    SalienceScorer,
    SalienceScorerConfig,
    ScoredFact,
    ScoredFactList,
)


@pytest.fixture
def mock_openai_response():
    """Factory to create mock OpenAI responses."""

    def _make(scored_facts: list[dict]):
        response = MagicMock()
        response.choices = [MagicMock()]
        response.choices[0].message.content = json.dumps(
            {"scored_facts": scored_facts}
        )
        return response

    return _make


@pytest.fixture
def mock_client(mock_openai_response):
    """Create a mock AsyncOpenAI client."""
    client = AsyncMock()
    client.chat.completions.create = AsyncMock(
        return_value=mock_openai_response(
            [
                {"text": "Client moving to Moderate risk", "salience": 0.95, "reasoning": "Risk change"},
                {"text": "User said hello", "salience": 0.1, "reasoning": "Greeting"},
            ]
        )
    )
    return client


class TestScoredFact:
    """Tests for the ScoredFact model."""

    def test_valid_scored_fact(self) -> None:
        sf = ScoredFact(text="fact", salience=0.85, reasoning="important")
        assert sf.text == "fact"
        assert sf.salience == 0.85

    def test_salience_clamped_to_range(self) -> None:
        """Salience outside 0-1 should be rejected."""
        with pytest.raises(Exception):
            ScoredFact(text="fact", salience=1.5, reasoning="too high")
        with pytest.raises(Exception):
            ScoredFact(text="fact", salience=-0.1, reasoning="too low")

    def test_default_reasoning(self) -> None:
        sf = ScoredFact(text="fact", salience=0.5)
        assert sf.reasoning == ""


class TestSalienceScorerConfig:
    """Tests for config defaults."""

    def test_defaults(self) -> None:
        config = SalienceScorerConfig()
        assert config.llm_model == "gpt-4.1-nano"
        assert config.llm_temperature == 0.0
        assert config.default_score == 0.5
        assert config.scoring_prompt is None

    def test_custom_prompt(self) -> None:
        config = SalienceScorerConfig(scoring_prompt="Custom: {facts_list} {context_section}")
        assert "Custom" in config.scoring_prompt


class TestSalienceScorer:
    """Tests for the SalienceScorer class."""

    @pytest.mark.asyncio
    async def test_score_empty_list(self) -> None:
        """Empty fact list should return empty list without LLM call."""
        scorer = SalienceScorer()
        results = await scorer.score_facts([])
        assert results == []

    @pytest.mark.asyncio
    async def test_score_facts_batch(self, mock_client, mock_openai_response) -> None:
        """Should score multiple facts in a single LLM call."""
        scorer = SalienceScorer()
        scorer._client = mock_client

        results = await scorer.score_facts(
            ["Client moving to Moderate risk", "User said hello"],
            context="Wealth management RM conversation",
        )

        assert len(results) == 2
        assert results[0].salience == 0.95
        assert results[1].salience == 0.1
        # Verify single LLM call was made
        mock_client.chat.completions.create.assert_called_once()

    @pytest.mark.asyncio
    async def test_score_single(self, mock_openai_response) -> None:
        """score_single should work as a convenience wrapper."""
        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(
            return_value=mock_openai_response(
                [{"text": "Important fact", "salience": 0.9, "reasoning": "Key info"}]
            )
        )

        scorer = SalienceScorer()
        scorer._client = mock_client

        result = await scorer.score_single("Important fact")
        assert result.salience == 0.9
        assert result.text == "Important fact"

    @pytest.mark.asyncio
    async def test_custom_prompt_used(self, mock_client) -> None:
        """Custom scoring_prompt should be included in the LLM call."""
        config = SalienceScorerConfig(
            scoring_prompt="Rate for wealth management: {facts_list} {context_section}"
        )
        scorer = SalienceScorer(config)
        scorer._client = mock_client

        await scorer.score_facts(["fact1"])
        call_args = mock_client.chat.completions.create.call_args
        user_msg = call_args.kwargs["messages"][1]["content"]
        assert "Rate for wealth management" in user_msg

    @pytest.mark.asyncio
    async def test_context_injected(self, mock_client) -> None:
        """Context parameter should appear in the prompt."""
        scorer = SalienceScorer()
        scorer._client = mock_client

        await scorer.score_facts(["fact1"], context="HR department context")
        call_args = mock_client.chat.completions.create.call_args
        user_msg = call_args.kwargs["messages"][1]["content"]
        assert "HR department context" in user_msg

    @pytest.mark.asyncio
    async def test_prompt_override_takes_precedence(self, mock_client) -> None:
        """prompt_override in score_facts should override config prompt."""
        config = SalienceScorerConfig(scoring_prompt="Config prompt: {facts_list} {context_section}")
        scorer = SalienceScorer(config)
        scorer._client = mock_client

        await scorer.score_facts(
            ["fact1"],
            prompt_override="Override prompt: {facts_list} {context_section}",
        )
        call_args = mock_client.chat.completions.create.call_args
        user_msg = call_args.kwargs["messages"][1]["content"]
        assert "Override prompt" in user_msg
        assert "Config prompt" not in user_msg

    @pytest.mark.asyncio
    async def test_llm_failure_returns_default_scores(self) -> None:
        """LLM call failure should return default_score for all facts."""
        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(
            side_effect=RuntimeError("API error")
        )

        config = SalienceScorerConfig(default_score=0.4)
        scorer = SalienceScorer(config)
        scorer._client = mock_client

        results = await scorer.score_facts(["fact1", "fact2"])
        assert len(results) == 2
        assert all(r.salience == 0.4 for r in results)
        assert "LLM call failed" in results[0].reasoning

    @pytest.mark.asyncio
    async def test_malformed_json_returns_default_scores(self, mock_openai_response) -> None:
        """Malformed LLM JSON should return default scores."""
        mock_client = AsyncMock()
        bad_response = MagicMock()
        bad_response.choices = [MagicMock()]
        bad_response.choices[0].message.content = "not valid json"
        mock_client.chat.completions.create = AsyncMock(return_value=bad_response)

        scorer = SalienceScorer(SalienceScorerConfig(default_score=0.3))
        scorer._client = mock_client

        results = await scorer.score_facts(["fact1"])
        assert len(results) == 1
        assert results[0].salience == 0.3

    @pytest.mark.asyncio
    async def test_model_and_temperature_passed(self, mock_client) -> None:
        """Config model and temperature should be passed to LLM call."""
        config = SalienceScorerConfig(
            llm_model="gpt-4o-mini",
            llm_temperature=0.1,
            llm_max_tokens=512,
        )
        scorer = SalienceScorer(config)
        scorer._client = mock_client

        await scorer.score_facts(["fact1"])
        call_args = mock_client.chat.completions.create.call_args
        assert call_args.kwargs["model"] == "gpt-4o-mini"
        assert call_args.kwargs["temperature"] == 0.1
        assert call_args.kwargs["max_tokens"] == 512

    def test_build_prompt_with_context(self) -> None:
        """_build_prompt should inject context when provided."""
        scorer = SalienceScorer()
        prompt = scorer._build_prompt(
            ["fact1", "fact2"], context="Wealth management"
        )
        assert "Wealth management" in prompt
        assert "1. fact1" in prompt
        assert "2. fact2" in prompt

    def test_build_prompt_without_context(self) -> None:
        """_build_prompt without context should have empty context section."""
        scorer = SalienceScorer()
        prompt = scorer._build_prompt(["fact1"])
        assert "1. fact1" in prompt

    def test_parse_response_valid(self) -> None:
        """_parse_response should handle valid JSON."""
        scorer = SalienceScorer()
        content = json.dumps({
            "scored_facts": [
                {"text": "f1", "salience": 0.8, "reasoning": "r1"},
            ]
        })
        results = scorer._parse_response(content, ["f1"])
        assert len(results) == 1
        assert results[0].salience == 0.8

    def test_parse_response_invalid_json(self) -> None:
        """_parse_response should fallback on invalid JSON."""
        scorer = SalienceScorer(SalienceScorerConfig(default_score=0.6))
        results = scorer._parse_response("not json", ["f1", "f2"])
        assert len(results) == 2
        assert all(r.salience == 0.6 for r in results)
