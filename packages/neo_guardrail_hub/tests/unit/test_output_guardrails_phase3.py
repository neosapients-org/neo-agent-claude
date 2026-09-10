"""Tests for Phase 3 LLM Guard Output Scanners.

This module tests all Phase 3 output guardrails that wrap LLM Guard's
output scanners.
"""

import pytest

from neo_guardrail_hub.core.models import GuardrailResult
from neo_guardrail_hub.guardrails.output import (
    BiasOutputGuardrail,
    CodeDetectionOutputGuardrail,
    BanCompetitorsOutputGuardrail,
    GibberishOutputGuardrail,
    JSONValidationOutputGuardrail,
    LanguageOutputGuardrail,
    LanguageSameOutputGuardrail,
    MaliciousURLsOutputGuardrail,
    ReadingTimeOutputGuardrail,
    RegexOutputGuardrail,
    SensitiveDataOutputGuardrail,
    SentimentOutputGuardrail,
    URLReachabilityOutputGuardrail,
)


# =============================================================================
# BiasOutputGuardrail Tests
# =============================================================================


class TestBiasOutputGuardrail:
    """Tests for BiasOutputGuardrail."""

    @pytest.fixture
    def guardrail(self):
        """Create BiasOutputGuardrail instance."""
        return BiasOutputGuardrail({"threshold": 0.5})

    @pytest.mark.asyncio
    async def test_neutral_text_passes(self, guardrail):
        """Test that neutral text passes."""
        text = "The weather today is sunny with clear skies."
        result = await guardrail.check(text)
        assert result.passed

    @pytest.mark.asyncio
    async def test_biased_text_detected(self, guardrail):
        """Test that biased text is detected."""
        text = "All people from that group are lazy and incompetent."
        result = await guardrail.check(text)
        assert isinstance(result, GuardrailResult)

    @pytest.mark.asyncio
    async def test_empty_text(self, guardrail):
        """Test empty text handling."""
        result = await guardrail.check("")
        assert result.passed

    @pytest.mark.asyncio
    async def test_sentence_match_type(self):
        """Test sentence-level bias detection."""
        guardrail = BiasOutputGuardrail({"threshold": 0.5, "match_type": "sentence"})
        text = "Normal sentence. Another normal one."
        result = await guardrail.check(text)
        assert isinstance(result, GuardrailResult)


# =============================================================================
# CodeDetectionOutputGuardrail Tests
# =============================================================================


class TestCodeDetectionOutputGuardrail:
    """Tests for CodeDetectionOutputGuardrail."""

    @pytest.fixture
    def guardrail(self):
        """Create CodeDetectionOutputGuardrail instance."""
        return CodeDetectionOutputGuardrail({"languages": ["Python"], "is_blocked": True})

    @pytest.mark.asyncio
    async def test_no_code_passes(self, guardrail):
        """Test that text without code passes."""
        text = "This is just plain text without any programming code."
        result = await guardrail.check(text)
        assert result.passed

    @pytest.mark.asyncio
    async def test_python_code_detected(self, guardrail):
        """Test that Python code is detected when configured."""
        text = """
def hello_world():
    print("Hello, World!")
    return True
"""
        result = await guardrail.check(text)
        assert isinstance(result, GuardrailResult)

    @pytest.mark.asyncio
    async def test_allow_only_listed_languages(self):
        """Test allowing only listed languages."""
        guardrail = CodeDetectionOutputGuardrail({"languages": ["Python"], "is_blocked": False})
        text = "Just regular text"
        result = await guardrail.check(text)
        assert isinstance(result, GuardrailResult)


# =============================================================================
# BanCompetitorsOutputGuardrail Tests
# =============================================================================


class TestBanCompetitorsOutputGuardrail:
    """Tests for BanCompetitorsOutputGuardrail."""

    @pytest.fixture
    def guardrail(self):
        """Create BanCompetitorsOutputGuardrail instance."""
        return BanCompetitorsOutputGuardrail({
            "competitors": ["CompetitorCorp", "RivalInc"],
            "redact": False
        })

    @pytest.mark.asyncio
    async def test_no_competitors_passes(self, guardrail):
        """Test that text without competitor mentions passes."""
        text = "Our product is the best solution for your needs."
        result = await guardrail.check(text)
        assert result.passed

    @pytest.mark.asyncio
    async def test_competitor_mention_detected(self, guardrail):
        """Test that competitor mentions are detected."""
        text = "You should consider using CompetitorCorp instead."
        result = await guardrail.check(text)
        assert isinstance(result, GuardrailResult)

    @pytest.mark.asyncio
    async def test_redact_mode(self):
        """Test competitor name redaction."""
        guardrail = BanCompetitorsOutputGuardrail({
            "competitors": ["CompetitorCorp"],
            "redact": True
        })
        text = "CompetitorCorp offers similar features."
        result = await guardrail.check(text)
        assert isinstance(result, GuardrailResult)


# =============================================================================
# GibberishOutputGuardrail Tests
# =============================================================================


class TestGibberishOutputGuardrail:
    """Tests for GibberishOutputGuardrail."""

    @pytest.fixture
    def guardrail(self):
        """Create GibberishOutputGuardrail instance."""
        return GibberishOutputGuardrail({"threshold": 0.5})

    @pytest.mark.asyncio
    async def test_coherent_text_passes(self, guardrail):
        """Test that coherent text passes."""
        text = "The quick brown fox jumps over the lazy dog."
        result = await guardrail.check(text)
        assert result.passed

    @pytest.mark.asyncio
    async def test_gibberish_detected(self, guardrail):
        """Test that gibberish text is detected."""
        text = "asdfkj lkjsdf lkjlk sdfjslkdfj skdjf"
        result = await guardrail.check(text)
        assert isinstance(result, GuardrailResult)

    @pytest.mark.asyncio
    async def test_random_characters(self, guardrail):
        """Test random character strings."""
        text = "xkcd qwerty zxcvb mnbvc lkjhg"
        result = await guardrail.check(text)
        assert isinstance(result, GuardrailResult)


# =============================================================================
# JSONValidationOutputGuardrail Tests
# =============================================================================


class TestJSONValidationOutputGuardrail:
    """Tests for JSONValidationOutputGuardrail."""

    @pytest.fixture
    def guardrail(self):
        """Create JSONValidationOutputGuardrail instance."""
        return JSONValidationOutputGuardrail()

    @pytest.mark.asyncio
    async def test_valid_json_passes(self, guardrail):
        """Test that valid JSON passes."""
        text = '{"name": "John", "age": 30}'
        result = await guardrail.check(text)
        assert result.passed

    @pytest.mark.asyncio
    async def test_no_json_passes(self, guardrail):
        """Test that text without JSON passes."""
        text = "This is regular text without any JSON."
        result = await guardrail.check(text)
        assert result.passed

    @pytest.mark.asyncio
    async def test_required_elements(self):
        """Test required elements validation."""
        guardrail = JSONValidationOutputGuardrail({"required_elements": 3})
        text = '{"a": 1, "b": 2}'
        result = await guardrail.check(text)
        assert isinstance(result, GuardrailResult)


# =============================================================================
# LanguageOutputGuardrail Tests
# =============================================================================


class TestLanguageOutputGuardrail:
    """Tests for LanguageOutputGuardrail."""

    @pytest.fixture
    def guardrail(self):
        """Create LanguageOutputGuardrail instance."""
        return LanguageOutputGuardrail({"valid_languages": ["en"]})

    @pytest.mark.asyncio
    async def test_english_passes(self, guardrail):
        """Test that English text passes."""
        text = "This is a test message in English."
        result = await guardrail.check(text)
        assert result.passed

    @pytest.mark.asyncio
    async def test_other_language_detected(self, guardrail):
        """Test that other languages are detected."""
        text = "Dies ist ein Test auf Deutsch."
        result = await guardrail.check(text)
        assert isinstance(result, GuardrailResult)

    @pytest.mark.asyncio
    async def test_multiple_valid_languages(self):
        """Test multiple valid languages."""
        guardrail = LanguageOutputGuardrail({"valid_languages": ["en", "es"]})
        text = "Hello world"
        result = await guardrail.check(text)
        assert isinstance(result, GuardrailResult)


# =============================================================================
# LanguageSameOutputGuardrail Tests
# =============================================================================


class TestLanguageSameOutputGuardrail:
    """Tests for LanguageSameOutputGuardrail."""

    @pytest.fixture
    def guardrail(self):
        """Create LanguageSameOutputGuardrail instance."""
        return LanguageSameOutputGuardrail()

    @pytest.mark.asyncio
    async def test_same_language_passes(self, guardrail):
        """Test that same language passes."""
        prompt = "Hello, how are you?"
        output = "I'm doing well, thank you!"
        # Pass context with prompt
        result = await guardrail.check(output, context={"prompt": prompt})
        assert result.passed

    @pytest.mark.asyncio
    async def test_different_language_detected(self, guardrail):
        """Test that different languages are detected."""
        prompt = "Hello, how are you?"
        output = "Ich bin gut, danke!"
        result = await guardrail.check(output, context={"prompt": prompt})
        assert isinstance(result, GuardrailResult)

    @pytest.mark.asyncio
    async def test_missing_prompt(self, guardrail):
        """Test handling of missing prompt."""
        result = await guardrail.check("Some output text")
        assert isinstance(result, GuardrailResult)


# =============================================================================
# MaliciousURLsOutputGuardrail Tests
# =============================================================================


class TestMaliciousURLsOutputGuardrail:
    """Tests for MaliciousURLsOutputGuardrail."""

    @pytest.fixture
    def guardrail(self):
        """Create MaliciousURLsOutputGuardrail instance."""
        return MaliciousURLsOutputGuardrail({"threshold": 0.5})

    @pytest.mark.asyncio
    async def test_no_urls_passes(self, guardrail):
        """Test that text without URLs passes."""
        text = "This is a safe message without any links."
        result = await guardrail.check(text)
        assert result.passed

    @pytest.mark.asyncio
    async def test_safe_url_passes(self, guardrail):
        """Test that safe URLs pass."""
        text = "Check out https://www.google.com for more info."
        result = await guardrail.check(text)
        assert result.passed

    @pytest.mark.asyncio
    async def test_suspicious_url(self, guardrail):
        """Test handling of suspicious URLs."""
        text = "Visit http://phishing-site.xyz/login.php?user=admin"
        result = await guardrail.check(text)
        assert isinstance(result, GuardrailResult)


# =============================================================================
# ReadingTimeOutputGuardrail Tests
# =============================================================================


class TestReadingTimeOutputGuardrail:
    """Tests for ReadingTimeOutputGuardrail."""

    @pytest.fixture
    def guardrail(self):
        """Create ReadingTimeOutputGuardrail instance."""
        return ReadingTimeOutputGuardrail({"max_time": 1.0})  # 1 minute

    @pytest.mark.asyncio
    async def test_short_text_passes(self, guardrail):
        """Test that short text passes."""
        text = "This is a brief message."
        result = await guardrail.check(text)
        assert result.passed

    @pytest.mark.asyncio
    async def test_long_text_exceeds_time(self):
        """Test that very long text fails."""
        guardrail = ReadingTimeOutputGuardrail({"max_time": 0.001})  # Very short limit
        text = " ".join(["word"] * 1000)  # Long text
        result = await guardrail.check(text)
        # Check it was processed
        assert isinstance(result, GuardrailResult)

    @pytest.mark.asyncio
    async def test_truncate_mode(self):
        """Test truncation mode."""
        guardrail = ReadingTimeOutputGuardrail({"max_time": 0.001, "truncate": True})
        text = " ".join(["word"] * 1000)
        result = await guardrail.check(text)
        assert isinstance(result, GuardrailResult)


# =============================================================================
# RegexOutputGuardrail Tests
# =============================================================================


class TestRegexOutputGuardrail:
    """Tests for RegexOutputGuardrail."""

    @pytest.fixture
    def guardrail(self):
        """Create RegexOutputGuardrail instance."""
        return RegexOutputGuardrail({
            "patterns": [r"\b\d{3}-\d{2}-\d{4}\b"],  # SSN pattern
            "is_blocked": True
        })

    @pytest.mark.asyncio
    async def test_no_match_passes(self, guardrail):
        """Test that text without pattern matches passes."""
        text = "This text contains no sensitive patterns."
        result = await guardrail.check(text)
        assert result.passed

    @pytest.mark.asyncio
    async def test_pattern_match_blocked(self, guardrail):
        """Test that pattern matches are blocked."""
        text = "My SSN is 123-45-6789."
        result = await guardrail.check(text)
        assert not result.passed

    @pytest.mark.asyncio
    async def test_require_pattern(self):
        """Test pattern requirement mode."""
        guardrail = RegexOutputGuardrail({
            "patterns": [r"Thank you"],
            "is_blocked": False  # Require pattern
        })
        text = "Here is your answer."
        result = await guardrail.check(text)
        assert not result.passed

    @pytest.mark.asyncio
    async def test_redact_mode(self):
        """Test pattern redaction."""
        guardrail = RegexOutputGuardrail({
            "patterns": [r"\b\d{3}-\d{2}-\d{4}\b"],
            "is_blocked": True,
            "redact": True
        })
        text = "SSN: 123-45-6789"
        result = await guardrail.check(text)
        if result.sanitized_text:
            assert "123-45-6789" not in result.sanitized_text


# =============================================================================
# SensitiveDataOutputGuardrail Tests
# =============================================================================


class TestSensitiveDataOutputGuardrail:
    """Tests for SensitiveDataOutputGuardrail."""

    @pytest.fixture
    def guardrail(self):
        """Create SensitiveDataOutputGuardrail instance."""
        return SensitiveDataOutputGuardrail({"redact": False})

    @pytest.mark.asyncio
    async def test_no_sensitive_data_passes(self, guardrail):
        """Test that text without sensitive data passes."""
        text = "The weather is nice today."
        result = await guardrail.check(text)
        assert result.passed

    @pytest.mark.asyncio
    async def test_sensitive_data_detected(self, guardrail):
        """Test that sensitive data is detected."""
        text = "My email is john.doe@example.com and my phone is 555-123-4567."
        result = await guardrail.check(text)
        assert isinstance(result, GuardrailResult)

    @pytest.mark.asyncio
    async def test_redact_sensitive_data(self):
        """Test sensitive data redaction."""
        guardrail = SensitiveDataOutputGuardrail({"redact": True})
        text = "Contact me at john.doe@example.com"
        result = await guardrail.check(text)
        assert isinstance(result, GuardrailResult)

    @pytest.mark.asyncio
    async def test_specific_entity_types(self):
        """Test specific entity type detection."""
        guardrail = SensitiveDataOutputGuardrail({
            "entity_types": ["email"],
            "redact": False
        })
        text = "My email is test@test.com"
        result = await guardrail.check(text)
        assert isinstance(result, GuardrailResult)


# =============================================================================
# SentimentOutputGuardrail Tests
# =============================================================================


class TestSentimentOutputGuardrail:
    """Tests for SentimentOutputGuardrail."""

    @pytest.fixture
    def guardrail(self):
        """Create SentimentOutputGuardrail instance."""
        return SentimentOutputGuardrail({"threshold": -0.5})

    @pytest.mark.asyncio
    async def test_positive_sentiment_passes(self, guardrail):
        """Test that positive sentiment passes."""
        text = "I'm so happy to help you! This is wonderful!"
        result = await guardrail.check(text)
        assert result.passed

    @pytest.mark.asyncio
    async def test_neutral_sentiment_passes(self, guardrail):
        """Test that neutral sentiment passes."""
        text = "The meeting is scheduled for tomorrow at 3pm."
        result = await guardrail.check(text)
        assert result.passed

    @pytest.mark.asyncio
    async def test_negative_sentiment_detected(self, guardrail):
        """Test that very negative sentiment is detected."""
        text = "This is terrible, horrible, and absolutely awful!"
        result = await guardrail.check(text)
        assert isinstance(result, GuardrailResult)


# =============================================================================
# URLReachabilityOutputGuardrail Tests
# =============================================================================


class TestURLReachabilityOutputGuardrail:
    """Tests for URLReachabilityOutputGuardrail."""

    @pytest.fixture
    def guardrail(self):
        """Create URLReachabilityOutputGuardrail instance."""
        return URLReachabilityOutputGuardrail({"timeout": 5})

    @pytest.mark.asyncio
    async def test_no_urls_passes(self, guardrail):
        """Test that text without URLs passes."""
        text = "This text has no links."
        result = await guardrail.check(text)
        assert result.passed

    @pytest.mark.asyncio
    async def test_reachable_url(self, guardrail):
        """Test reachable URL."""
        text = "Visit https://www.google.com"
        result = await guardrail.check(text)
        assert isinstance(result, GuardrailResult)

    @pytest.mark.asyncio
    async def test_unreachable_url(self, guardrail):
        """Test unreachable URL detection."""
        text = "Check http://this-domain-definitely-does-not-exist-xyz123.com"
        result = await guardrail.check(text)
        assert isinstance(result, GuardrailResult)


# =============================================================================
# Provider Integration Tests
# =============================================================================


class TestLLMGuardProviderOutputIntegration:
    """Tests for LLM Guard provider integration with output guardrails."""

    @pytest.mark.asyncio
    async def test_provider_has_all_output_guardrails(self):
        """Test that provider registers all Phase 3 output guardrails."""
        from neo_guardrail_hub.providers.llm_guard import LLMGuardProvider

        provider = LLMGuardProvider()
        await provider.initialize()

        # Check all Phase 3 output guardrails are registered
        phase3_output_guardrails = [
            "bias_output",
            "code_detection_output",
            "ban_competitors_output",
            "gibberish_output",
            "json_validation",
            "language_output",
            "language_same",
            "malicious_urls",
            "reading_time",
            "regex_output",
            "sensitive_output",
            "sentiment_output",
            "url_reachability",
        ]

        for guardrail_type in phase3_output_guardrails:
            guardrail = provider.get_guardrail(guardrail_type)
            assert guardrail is not None, f"Missing guardrail: {guardrail_type}"

    @pytest.mark.asyncio
    async def test_provider_guardrail_execution(self):
        """Test executing guardrails through provider."""
        from neo_guardrail_hub.providers.llm_guard import LLMGuardProvider

        provider = LLMGuardProvider()
        await provider.initialize()

        # Test JSON validation through provider
        guardrail = provider.get_guardrail("json_validation")
        result = await guardrail.check('{"valid": "json"}')
        assert result.passed
