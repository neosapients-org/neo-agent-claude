"""Unit tests for Phase 2 input guardrails.

Tests for: secret_detection, input_length, toxicity, ban_substrings, harmful_content
"""

import pytest

from neo_guardrail_hub.guardrails.input import (
    BanSubstringsInputGuardrail,
    HarmfulContentGuardrail,
    InputLengthGuardrail,
    SecretDetectionGuardrail,
    ToxicityInputGuardrail,
)


class TestSecretDetectionGuardrail:
    """Tests for SecretDetectionGuardrail."""

    @pytest.fixture
    def guardrail(self):
        """Create a SecretDetectionGuardrail instance."""
        return SecretDetectionGuardrail({
            "threshold": 0.5,
            "redact_mode": "partial",
        })

    @pytest.mark.asyncio
    async def test_initialization(self, guardrail):
        """Test guardrail initializes correctly."""
        await guardrail.initialize()
        assert guardrail.name == "secret_detection"
        assert guardrail.layer.value == "input"

    @pytest.mark.asyncio
    async def test_safe_text_passes(self, guardrail):
        """Test that normal text passes the check."""
        await guardrail.initialize()
        result = await guardrail.check("This is a normal message without secrets.")
        assert result.passed is True
        assert result.risk_score <= 0.5

    @pytest.mark.asyncio
    async def test_aws_key_detected_fallback(self, guardrail):
        """Test that AWS keys are detected (fallback pattern)."""
        await guardrail.initialize()
        # Clear scanner to force fallback
        guardrail._scanner = None
        text = "My AWS key is AKIAIOSFODNN7EXAMPLE"
        result = await guardrail.check(text)
        # Even if passed (low risk score), check that secret was detected
        assert len(result.metadata.get("detected_secrets", [])) > 0
        assert "AWS" in str(result.metadata.get("detected_secrets", []))

    @pytest.mark.asyncio
    async def test_api_key_detected_fallback(self, guardrail):
        """Test that API keys are detected (fallback pattern)."""
        await guardrail.initialize()
        guardrail._scanner = None
        text = "api_key=sk_live_abcdef1234567890abcdef"
        result = await guardrail.check(text)
        # Check detected_secrets instead of secret_types
        detected = result.metadata.get("detected_secrets", [])
        assert len(detected) > 0
        # Should detect either Stripe or API Key pattern
        types = [s.get("type", "") for s in detected]
        assert any("Key" in t for t in types)


class TestInputLengthGuardrail:
    """Tests for InputLengthGuardrail."""

    @pytest.fixture
    def guardrail(self):
        """Create an InputLengthGuardrail instance."""
        return InputLengthGuardrail({
            "min_length": 10,
            "max_length": 100,
            "count_mode": "chars",
        })

    @pytest.mark.asyncio
    async def test_initialization(self, guardrail):
        """Test guardrail initializes correctly."""
        await guardrail.initialize()
        assert guardrail.name == "input_length"
        assert guardrail.layer.value == "input"

    @pytest.mark.asyncio
    async def test_valid_length_passes(self, guardrail):
        """Test that text within length bounds passes."""
        await guardrail.initialize()
        result = await guardrail.check("This is a valid message that is within bounds.")
        assert result.passed is True
        assert result.risk_score == 0.0

    @pytest.mark.asyncio
    async def test_too_short_fails(self, guardrail):
        """Test that text below minimum length fails."""
        await guardrail.initialize()
        result = await guardrail.check("Short")
        assert result.passed is False
        assert "too short" in result.message.lower()

    @pytest.mark.asyncio
    async def test_too_long_fails(self, guardrail):
        """Test that text exceeding maximum length fails."""
        await guardrail.initialize()
        long_text = "A" * 150
        result = await guardrail.check(long_text)
        assert result.passed is False
        assert "too long" in result.message.lower()

    @pytest.mark.asyncio
    async def test_word_count_mode(self):
        """Test word count mode."""
        guardrail = InputLengthGuardrail({
            "min_length": 5,
            "max_length": 20,
            "count_mode": "words",
        })
        await guardrail.initialize()
        
        # 6 words - should pass
        result = await guardrail.check("This is a six word sentence.")
        assert result.passed is True


class TestToxicityInputGuardrail:
    """Tests for ToxicityInputGuardrail."""

    @pytest.fixture
    def guardrail(self):
        """Create a ToxicityInputGuardrail instance."""
        return ToxicityInputGuardrail({
            "threshold": 0.5,
            "match_type": "SENTENCE",
        })

    @pytest.mark.asyncio
    async def test_initialization(self, guardrail):
        """Test guardrail initializes correctly."""
        await guardrail.initialize()
        assert guardrail.name == "toxicity_input"
        assert guardrail.layer.value == "input"

    @pytest.mark.asyncio
    async def test_safe_text_passes(self, guardrail):
        """Test that normal text passes the check."""
        await guardrail.initialize()
        result = await guardrail.check("Hello, how are you today?")
        assert result.passed is True

    @pytest.mark.asyncio
    async def test_toxic_text_detected_fallback(self, guardrail):
        """Test that toxic text is detected (fallback pattern)."""
        await guardrail.initialize()
        # Force fallback
        guardrail._scanner = None
        # Use text that matches multiple toxic patterns
        text = "I hate you and I will kill you, you worthless pathetic garbage"
        result = await guardrail.check(text)
        assert result.passed is False
        assert result.metadata.get("detection_method") == "regex_fallback"


class TestBanSubstringsInputGuardrail:
    """Tests for BanSubstringsInputGuardrail."""

    @pytest.fixture
    def guardrail(self):
        """Create a BanSubstringsInputGuardrail instance."""
        return BanSubstringsInputGuardrail({
            "substrings": ["forbidden", "banned", "secret_code"],
            "match_type": "WORD",
            "case_sensitive": False,
        })

    @pytest.mark.asyncio
    async def test_initialization(self, guardrail):
        """Test guardrail initializes correctly."""
        await guardrail.initialize()
        assert guardrail.name == "ban_substrings_input"
        assert guardrail.layer.value == "input"

    @pytest.mark.asyncio
    async def test_safe_text_passes(self, guardrail):
        """Test that normal text passes the check."""
        await guardrail.initialize()
        result = await guardrail.check("This is a normal message.")
        assert result.passed is True

    @pytest.mark.asyncio
    async def test_banned_substring_detected(self, guardrail):
        """Test that banned substrings are detected."""
        await guardrail.initialize()
        # Force fallback for consistent testing
        guardrail._scanner = None
        result = await guardrail.check("This message contains forbidden content.")
        assert result.passed is False
        assert "forbidden" in result.metadata.get("found_substrings", [])

    @pytest.mark.asyncio
    async def test_case_insensitive(self, guardrail):
        """Test case insensitive matching."""
        await guardrail.initialize()
        guardrail._scanner = None
        result = await guardrail.check("This contains FORBIDDEN words.")
        assert result.passed is False

    @pytest.mark.asyncio
    async def test_empty_substrings_skips(self):
        """Test that empty substrings list skips the check."""
        guardrail = BanSubstringsInputGuardrail({"substrings": []})
        await guardrail.initialize()
        result = await guardrail.check("Any text")
        assert result.passed is True
        # Empty substrings should have a reason in metadata
        assert "no_substrings_configured" in str(result.metadata)


class TestHarmfulContentGuardrail:
    """Tests for HarmfulContentGuardrail."""

    @pytest.fixture
    def guardrail(self):
        """Create a HarmfulContentGuardrail instance."""
        return HarmfulContentGuardrail({
            "threshold": 0.5,
            "categories": ["violence", "hate_speech", "self_harm"],
        })

    @pytest.mark.asyncio
    async def test_initialization(self, guardrail):
        """Test guardrail initializes correctly."""
        await guardrail.initialize()
        assert guardrail.name == "harmful_content"
        assert guardrail.layer.value == "input"

    @pytest.mark.asyncio
    async def test_safe_text_passes(self, guardrail):
        """Test that normal text passes the check."""
        await guardrail.initialize()
        result = await guardrail.check("Let's have a nice conversation about cooking.")
        assert result.passed is True

    @pytest.mark.asyncio
    async def test_harmful_content_detected_fallback(self, guardrail):
        """Test that harmful content is detected (fallback)."""
        await guardrail.initialize()
        # Force fallback
        guardrail._scanner = None
        # Text that matches violence patterns: "kill", "attack", "stab"
        text = "I want to kill them and stab someone"
        result = await guardrail.check(text)
        assert result.passed is False
        # Check detected_categories (not categories_detected)
        assert "violence" in str(result.metadata.get("detected_categories", []))

    @pytest.mark.asyncio
    async def test_empty_categories_skips(self):
        """Test that empty categories list allows all content."""
        guardrail = HarmfulContentGuardrail({"categories": []})
        await guardrail.initialize()
        result = await guardrail.check("Any text even harmful attack")
        # With no categories to check, it should pass
        assert result.passed is True


class TestGuardrailMetadata:
    """Tests for common metadata across all guardrails."""

    @pytest.mark.asyncio
    async def test_all_guardrails_return_proper_metadata(self):
        """Test that all guardrails return proper GuardrailResult structure."""
        guardrails = [
            SecretDetectionGuardrail({"threshold": 0.5}),
            InputLengthGuardrail({"min_length": 1, "max_length": 10000}),
            ToxicityInputGuardrail({"threshold": 0.5}),
            BanSubstringsInputGuardrail({"substrings": ["test"]}),
            HarmfulContentGuardrail({"categories": ["violence"]}),
        ]

        for guardrail in guardrails:
            await guardrail.initialize()
            # Force fallback for scanners
            if hasattr(guardrail, '_scanner'):
                guardrail._scanner = None
            result = await guardrail.check("Test message here")
            
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
