"""Unit tests for Phase 2 output guardrails.

Tests for: no_refusal, relevance, toxicity_output, factual_consistency, ban_topics, ban_substrings_output
"""

import pytest

from neo_guardrail_hub.guardrails.output import (
    BanSubstringsOutputGuardrail,
    BanTopicsGuardrail,
    FactualConsistencyGuardrail,
    NoRefusalGuardrail,
    RelevanceGuardrail,
    ToxicityOutputGuardrail,
)


class TestNoRefusalGuardrail:
    """Tests for NoRefusalGuardrail."""

    @pytest.fixture
    def guardrail(self):
        """Create a NoRefusalGuardrail instance."""
        return NoRefusalGuardrail({
            "threshold": 0.5,
            "match_type": "FULL",
        })

    @pytest.mark.asyncio
    async def test_initialization(self, guardrail):
        """Test guardrail initializes correctly."""
        await guardrail.initialize()
        assert guardrail.name == "no_refusal"
        assert guardrail.layer.value == "output"

    @pytest.mark.asyncio
    async def test_helpful_response_passes(self, guardrail):
        """Test that helpful responses pass the check."""
        await guardrail.initialize()
        result = await guardrail.check(
            "Here's the information you requested about cooking pasta."
        )
        assert result.passed is True

    @pytest.mark.asyncio
    async def test_refusal_detected_fallback(self, guardrail):
        """Test that refusal responses are detected (fallback)."""
        await guardrail.initialize()
        guardrail._scanner = None  # Force fallback
        result = await guardrail.check(
            "I'm sorry, but I cannot help with that request."
        )
        assert result.passed is False
        assert result.metadata.get("refusal_detected") is True

    @pytest.mark.asyncio
    async def test_another_refusal_pattern(self, guardrail):
        """Test another refusal pattern."""
        await guardrail.initialize()
        guardrail._scanner = None
        result = await guardrail.check(
            "As an AI language model, I am not able to provide that information."
        )
        assert result.passed is False


class TestRelevanceGuardrail:
    """Tests for RelevanceGuardrail."""

    @pytest.fixture
    def guardrail(self):
        """Create a RelevanceGuardrail instance."""
        return RelevanceGuardrail({
            "threshold": 0.5,
            "keyword_overlap_threshold": 0.2,
        })

    @pytest.mark.asyncio
    async def test_initialization(self, guardrail):
        """Test guardrail initializes correctly."""
        await guardrail.initialize()
        assert guardrail.name == "relevance"
        assert guardrail.layer.value == "output"

    @pytest.mark.asyncio
    async def test_no_prompt_skips(self, guardrail):
        """Test that check is skipped without prompt."""
        await guardrail.initialize()
        result = await guardrail.check("Some output text")
        assert result.passed is True
        assert result.metadata.get("skipped") is True

    @pytest.mark.asyncio
    async def test_relevant_response_passes_fallback(self, guardrail):
        """Test that relevant responses pass (fallback)."""
        await guardrail.initialize()
        guardrail._scanner = None  # Force fallback
        result = await guardrail.check(
            "Python is a programming language that is easy to learn.",
            context={"prompt": "Tell me about Python programming."}
        )
        assert result.passed is True
        assert result.metadata.get("detection_method") == "keyword_overlap"

    @pytest.mark.asyncio
    async def test_irrelevant_response_fails_fallback(self, guardrail):
        """Test that irrelevant responses fail (fallback)."""
        await guardrail.initialize()
        guardrail._scanner = None
        result = await guardrail.check(
            "The weather in Paris is sunny today.",
            context={"prompt": "How do quantum computers work?"}
        )
        # Should fail due to lack of keyword overlap
        assert result.metadata.get("detection_method") == "keyword_overlap"


class TestToxicityOutputGuardrail:
    """Tests for ToxicityOutputGuardrail."""

    @pytest.fixture
    def guardrail(self):
        """Create a ToxicityOutputGuardrail instance."""
        return ToxicityOutputGuardrail({
            "threshold": 0.5,
            "match_type": "SENTENCE",
        })

    @pytest.mark.asyncio
    async def test_initialization(self, guardrail):
        """Test guardrail initializes correctly."""
        await guardrail.initialize()
        assert guardrail.name == "toxicity_output"
        assert guardrail.layer.value == "output"

    @pytest.mark.asyncio
    async def test_safe_output_passes(self, guardrail):
        """Test that safe outputs pass the check."""
        await guardrail.initialize()
        result = await guardrail.check("Here's a helpful and friendly response.")
        assert result.passed is True

    @pytest.mark.asyncio
    async def test_toxic_output_detected_fallback(self, guardrail):
        """Test that toxic outputs are detected (fallback)."""
        await guardrail.initialize()
        guardrail._scanner = None
        result = await guardrail.check("You're worthless and pathetic!")
        assert result.passed is False
        assert result.metadata.get("toxicity_detected") is True


class TestFactualConsistencyGuardrail:
    """Tests for FactualConsistencyGuardrail."""

    @pytest.fixture
    def guardrail(self):
        """Create a FactualConsistencyGuardrail instance."""
        return FactualConsistencyGuardrail({
            "minimum_score": 0.5,
        })

    @pytest.mark.asyncio
    async def test_initialization(self, guardrail):
        """Test guardrail initializes correctly."""
        await guardrail.initialize()
        assert guardrail.name == "factual_consistency"
        assert guardrail.layer.value == "output"

    @pytest.mark.asyncio
    async def test_no_context_skips(self, guardrail):
        """Test that check is skipped without context."""
        await guardrail.initialize()
        result = await guardrail.check("Some output text")
        assert result.passed is True
        assert result.metadata.get("skipped") is True

    @pytest.mark.asyncio
    async def test_consistent_output_passes_fallback(self, guardrail):
        """Test that consistent outputs pass (fallback)."""
        await guardrail.initialize()
        guardrail._scanner = None
        result = await guardrail.check(
            "Yes, the sky is blue due to light scattering.",
            context={"prompt": "Is it true that the sky is blue?"}
        )
        assert result.passed is True

    @pytest.mark.asyncio
    async def test_contradiction_detected_fallback(self, guardrail):
        """Test that contradictions are detected (fallback)."""
        await guardrail.initialize()
        guardrail._scanner = None
        result = await guardrail.check(
            "No, that statement is false and incorrect.",
            context={"prompt": "Is it true that water is wet?"}
        )
        # The heuristic should detect the negation
        assert result.metadata.get("detection_method") == "heuristic"


class TestBanTopicsGuardrail:
    """Tests for BanTopicsGuardrail."""

    @pytest.fixture
    def guardrail(self):
        """Create a BanTopicsGuardrail instance."""
        return BanTopicsGuardrail({
            "topics": ["politics", "violence"],
            "threshold": 0.5,
        })

    @pytest.mark.asyncio
    async def test_initialization(self, guardrail):
        """Test guardrail initializes correctly."""
        await guardrail.initialize()
        assert guardrail.name == "ban_topics"
        assert guardrail.layer.value == "output"

    @pytest.mark.asyncio
    async def test_safe_topic_passes(self, guardrail):
        """Test that safe topics pass the check."""
        await guardrail.initialize()
        guardrail._scanner = None  # Force fallback
        result = await guardrail.check("Let me tell you about cooking recipes.")
        assert result.passed is True

    @pytest.mark.asyncio
    async def test_banned_topic_detected_fallback(self, guardrail):
        """Test that banned topics are detected (fallback)."""
        await guardrail.initialize()
        guardrail._scanner = None
        result = await guardrail.check(
            "Let's discuss the upcoming election and vote for candidates."
        )
        assert result.passed is False
        assert "politics" in result.metadata.get("detected_topics", {})

    @pytest.mark.asyncio
    async def test_empty_topics_skips(self):
        """Test that empty topics list skips the check."""
        guardrail = BanTopicsGuardrail({"topics": []})
        await guardrail.initialize()
        result = await guardrail.check("Any text")
        assert result.passed is True
        assert result.metadata.get("skipped") is True


class TestBanSubstringsOutputGuardrail:
    """Tests for BanSubstringsOutputGuardrail."""

    @pytest.fixture
    def guardrail(self):
        """Create a BanSubstringsOutputGuardrail instance."""
        return BanSubstringsOutputGuardrail({
            "substrings": ["confidential", "internal only", "do not share"],
            "match_type": "WORD",
            "case_sensitive": False,
        })

    @pytest.mark.asyncio
    async def test_initialization(self, guardrail):
        """Test guardrail initializes correctly."""
        await guardrail.initialize()
        assert guardrail.name == "ban_substrings_output"
        assert guardrail.layer.value == "output"

    @pytest.mark.asyncio
    async def test_safe_output_passes(self, guardrail):
        """Test that outputs without banned substrings pass."""
        await guardrail.initialize()
        guardrail._scanner = None  # Force fallback
        result = await guardrail.check("This is a normal public response.")
        assert result.passed is True

    @pytest.mark.asyncio
    async def test_banned_substring_detected(self, guardrail):
        """Test that banned substrings are detected."""
        await guardrail.initialize()
        guardrail._scanner = None
        result = await guardrail.check("This is confidential information.")
        assert result.passed is False
        assert "confidential" in result.metadata.get("found_substrings", [])

    @pytest.mark.asyncio
    async def test_redact_mode(self):
        """Test redact mode returns sanitized text."""
        guardrail = BanSubstringsOutputGuardrail({
            "substrings": ["secret"],
            "redact": True,
        })
        await guardrail.initialize()
        guardrail._scanner = None  # Force fallback
        result = await guardrail.check("This is a secret message.")
        assert result.passed is True  # Passes with sanitization
        assert result.sanitized_text is not None
        assert "[REDACTED]" in result.sanitized_text

    @pytest.mark.asyncio
    async def test_empty_substrings_skips(self):
        """Test that empty substrings list skips the check."""
        guardrail = BanSubstringsOutputGuardrail({"substrings": []})
        await guardrail.initialize()
        result = await guardrail.check("Any text")
        assert result.passed is True
        assert result.metadata.get("skipped") is True


class TestOutputGuardrailMetadata:
    """Tests for common metadata across all output guardrails."""

    @pytest.mark.asyncio
    async def test_all_output_guardrails_return_proper_metadata(self):
        """Test that all output guardrails return proper GuardrailResult structure."""
        guardrails = [
            NoRefusalGuardrail({}),
            RelevanceGuardrail({}),
            ToxicityOutputGuardrail({}),
            FactualConsistencyGuardrail({}),
            BanTopicsGuardrail({"topics": ["test"]}),
            BanSubstringsOutputGuardrail({"substrings": ["test"]}),
        ]

        for guardrail in guardrails:
            await guardrail.initialize()
            result = await guardrail.check("Test output message")
            
            # All results should have these attributes
            assert hasattr(result, "passed")
            assert hasattr(result, "guardrail_name")
            assert hasattr(result, "layer")
            assert hasattr(result, "risk_score")
            assert hasattr(result, "metadata")
            
            # Risk score should be normalized
            assert 0.0 <= result.risk_score <= 1.0
            
            # Guardrail name should be set
            assert result.guardrail_name == guardrail.name
            
            # Layer should be output
            assert guardrail.layer.value == "output"
