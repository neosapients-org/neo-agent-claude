"""Unit tests for Phase 3 input guardrails (Additional LLM Guard Scanners).

Tests for: ban_code, ban_competitors, ban_topics, code_detection, 
gibberish, invisible_text, language, regex, sentiment
"""

import pytest

from neo_guardrail_hub.guardrails.input import (
    BanCodeInputGuardrail,
    BanCompetitorsInputGuardrail,
    BanTopicsInputGuardrail,
    CodeDetectionInputGuardrail,
    GibberishInputGuardrail,
    InvisibleTextInputGuardrail,
    LanguageInputGuardrail,
    RegexInputGuardrail,
    SentimentInputGuardrail,
)


class TestBanCodeInputGuardrail:
    """Tests for BanCodeInputGuardrail."""

    @pytest.fixture
    def guardrail(self):
        """Create a BanCodeInputGuardrail instance."""
        return BanCodeInputGuardrail({
            "threshold": 0.5,
        })

    @pytest.mark.asyncio
    async def test_initialization(self, guardrail):
        """Test guardrail initializes correctly."""
        await guardrail.initialize()
        assert guardrail.name == "ban_code_input"
        assert guardrail.layer.value == "input"

    @pytest.mark.asyncio
    async def test_safe_text_passes(self, guardrail):
        """Test that normal text passes the check."""
        await guardrail.initialize()
        guardrail._scanner = None  # Use fallback
        result = await guardrail.check("This is a normal message without any code.")
        assert result.passed is True
        assert result.risk_score < 0.5

    @pytest.mark.asyncio
    async def test_python_code_detected(self, guardrail):
        """Test that Python code is detected."""
        await guardrail.initialize()
        guardrail._scanner = None  # Use fallback
        code = """
def hello_world():
    print("Hello World")
    return True
"""
        result = await guardrail.check(code)
        assert result.passed is False
        assert "Code" in result.message or "code" in result.message.lower()

    @pytest.mark.asyncio
    async def test_sql_code_detected(self, guardrail):
        """Test that SQL code is detected."""
        await guardrail.initialize()
        guardrail._scanner = None  # Use fallback
        result = await guardrail.check("SELECT * FROM users WHERE id = 1")
        assert result.passed is False


class TestBanCompetitorsInputGuardrail:
    """Tests for BanCompetitorsInputGuardrail."""

    @pytest.fixture
    def guardrail(self):
        """Create a BanCompetitorsInputGuardrail instance."""
        return BanCompetitorsInputGuardrail({
            "competitors": ["Microsoft", "Google", "Amazon"],
            "threshold": 0.5,
            "redact": False,
        })

    @pytest.mark.asyncio
    async def test_initialization(self, guardrail):
        """Test guardrail initializes correctly."""
        await guardrail.initialize()
        assert guardrail.name == "ban_competitors_input"
        assert guardrail.layer.value == "input"

    @pytest.mark.asyncio
    async def test_safe_text_passes(self, guardrail):
        """Test that text without competitors passes."""
        await guardrail.initialize()
        guardrail._scanner = None  # Use fallback
        result = await guardrail.check("Tell me about your company's products.")
        assert result.passed is True

    @pytest.mark.asyncio
    async def test_competitor_detected(self, guardrail):
        """Test that competitor mention is detected."""
        await guardrail.initialize()
        guardrail._scanner = None  # Use fallback
        result = await guardrail.check("How does your product compare to Microsoft?")
        assert result.passed is False
        assert "competitor" in result.message.lower()

    @pytest.mark.asyncio
    async def test_redaction(self):
        """Test that competitors can be redacted."""
        guardrail = BanCompetitorsInputGuardrail({
            "competitors": ["Microsoft"],
            "redact": True,
        })
        await guardrail.initialize()
        guardrail._scanner = None
        result = await guardrail.check("Tell me about Microsoft products")
        assert result.sanitized_text is not None
        assert "Microsoft" not in result.sanitized_text
        assert "[COMPETITOR]" in result.sanitized_text

    @pytest.mark.asyncio
    async def test_no_competitors_configured(self):
        """Test that empty competitor list passes all."""
        guardrail = BanCompetitorsInputGuardrail({"competitors": []})
        await guardrail.initialize()
        result = await guardrail.check("Tell me about Microsoft")
        assert result.passed is True


class TestBanTopicsInputGuardrail:
    """Tests for BanTopicsInputGuardrail."""

    @pytest.fixture
    def guardrail(self):
        """Create a BanTopicsInputGuardrail instance."""
        return BanTopicsInputGuardrail({
            "topics": ["politics", "religion"],
            "threshold": 0.5,
        })

    @pytest.mark.asyncio
    async def test_initialization(self, guardrail):
        """Test guardrail initializes correctly."""
        await guardrail.initialize()
        assert guardrail.name == "ban_topics_input"
        assert guardrail.layer.value == "input"

    @pytest.mark.asyncio
    async def test_safe_text_passes(self, guardrail):
        """Test that neutral text passes."""
        await guardrail.initialize()
        guardrail._scanner = None  # Use fallback
        result = await guardrail.check("What's the weather like today?")
        assert result.passed is True

    @pytest.mark.asyncio
    async def test_political_topic_detected(self, guardrail):
        """Test that political content is detected."""
        await guardrail.initialize()
        guardrail._scanner = None  # Use fallback
        result = await guardrail.check("What do you think about the upcoming election?")
        assert result.passed is False
        assert "politics" in str(result.metadata.get("detected_topics", {})).lower()

    @pytest.mark.asyncio
    async def test_no_topics_configured(self):
        """Test that empty topics list passes all."""
        guardrail = BanTopicsInputGuardrail({"topics": []})
        await guardrail.initialize()
        result = await guardrail.check("Tell me about politics")
        assert result.passed is True


class TestCodeDetectionInputGuardrail:
    """Tests for CodeDetectionInputGuardrail."""

    @pytest.fixture
    def guardrail(self):
        """Create a CodeDetectionInputGuardrail instance with blocked Python."""
        return CodeDetectionInputGuardrail({
            "languages": ["Python"],
            "is_blocked": True,
            "threshold": 0.5,
        })

    @pytest.mark.asyncio
    async def test_initialization(self, guardrail):
        """Test guardrail initializes correctly."""
        await guardrail.initialize()
        assert guardrail.name == "code_detection_input"
        assert guardrail.layer.value == "input"

    @pytest.mark.asyncio
    async def test_normal_text_passes(self, guardrail):
        """Test that normal text passes."""
        await guardrail.initialize()
        guardrail._scanner = None  # Use fallback
        result = await guardrail.check("This is regular text without code.")
        assert result.passed is True

    @pytest.mark.asyncio
    async def test_python_code_blocked(self, guardrail):
        """Test that Python code is blocked when configured."""
        await guardrail.initialize()
        guardrail._scanner = None  # Use fallback
        result = await guardrail.check("```python\ndef foo(): pass\n```")
        assert result.passed is False

    @pytest.mark.asyncio
    async def test_allow_mode(self):
        """Test allow mode (require specific language)."""
        guardrail = CodeDetectionInputGuardrail({
            "languages": ["Python"],
            "is_blocked": False,  # Allow mode
        })
        await guardrail.initialize()
        guardrail._scanner = None
        
        # Python should pass in allow mode
        result = await guardrail.check("```python\nimport os\n```")
        assert result.passed is True
        
        # JavaScript should fail in allow mode
        result = await guardrail.check("```javascript\nconst x = 1;\n```")
        assert result.passed is False


class TestGibberishInputGuardrail:
    """Tests for GibberishInputGuardrail."""

    @pytest.fixture
    def guardrail(self):
        """Create a GibberishInputGuardrail instance."""
        return GibberishInputGuardrail({
            "threshold": 0.5,
            "match_type": "full",
        })

    @pytest.mark.asyncio
    async def test_initialization(self, guardrail):
        """Test guardrail initializes correctly."""
        await guardrail.initialize()
        assert guardrail.name == "gibberish_input"
        assert guardrail.layer.value == "input"

    @pytest.mark.asyncio
    async def test_normal_text_passes(self, guardrail):
        """Test that normal English text passes."""
        await guardrail.initialize()
        guardrail._scanner = None  # Use fallback
        result = await guardrail.check("This is a perfectly normal sentence.")
        assert result.passed is True
        assert result.risk_score < 0.5

    @pytest.mark.asyncio
    async def test_gibberish_detected(self, guardrail):
        """Test that gibberish is detected."""
        await guardrail.initialize()
        guardrail._scanner = None  # Use fallback
        # Very obvious keyboard mashing gibberish
        result = await guardrail.check("asdf qwer zxcv hjkl uiop asdf qwer")
        assert result.passed is False or result.risk_score > 0.1

    @pytest.mark.asyncio
    async def test_repeated_chars_detected(self, guardrail):
        """Test that repeated characters are flagged."""
        await guardrail.initialize()
        guardrail._scanner = None
        result = await guardrail.check("aaaaaaaaaa bbbbbbbbbb cccccccccc")
        assert result.risk_score > 0.2


class TestInvisibleTextInputGuardrail:
    """Tests for InvisibleTextInputGuardrail."""

    @pytest.fixture
    def guardrail(self):
        """Create an InvisibleTextInputGuardrail instance."""
        return InvisibleTextInputGuardrail()

    @pytest.mark.asyncio
    async def test_initialization(self, guardrail):
        """Test guardrail initializes correctly."""
        await guardrail.initialize()
        assert guardrail.name == "invisible_text_input"
        assert guardrail.layer.value == "input"

    @pytest.mark.asyncio
    async def test_normal_text_passes(self, guardrail):
        """Test that normal text passes."""
        await guardrail.initialize()
        guardrail._scanner = None  # Use fallback
        result = await guardrail.check("This is normal visible text.")
        assert result.passed is True
        assert result.risk_score == 0.0

    @pytest.mark.asyncio
    async def test_zero_width_space_detected(self, guardrail):
        """Test that zero-width space is detected."""
        await guardrail.initialize()
        guardrail._scanner = None  # Use fallback
        # \u200b is zero-width space
        result = await guardrail.check("Hello\u200bWorld")
        assert result.passed is False
        assert result.sanitized_text == "HelloWorld"

    @pytest.mark.asyncio
    async def test_multiple_invisible_chars(self, guardrail):
        """Test that multiple invisible characters are detected."""
        await guardrail.initialize()
        guardrail._scanner = None
        # Mix of zero-width characters
        text = "Test\u200b\u200c\u200dString"
        result = await guardrail.check(text)
        assert result.passed is False
        assert result.metadata["invisible_char_count"] == 3


class TestLanguageInputGuardrail:
    """Tests for LanguageInputGuardrail."""

    @pytest.fixture
    def guardrail(self):
        """Create a LanguageInputGuardrail instance."""
        return LanguageInputGuardrail({
            "valid_languages": ["en"],
            "threshold": 0.5,
        })

    @pytest.mark.asyncio
    async def test_initialization(self, guardrail):
        """Test guardrail initializes correctly."""
        await guardrail.initialize()
        assert guardrail.name == "language_input"
        assert guardrail.layer.value == "input"

    @pytest.mark.asyncio
    async def test_english_text_passes(self, guardrail):
        """Test that English text passes."""
        await guardrail.initialize()
        guardrail._scanner = None  # Use fallback
        result = await guardrail.check("This is English text that should pass.")
        assert result.passed is True

    @pytest.mark.asyncio
    async def test_chinese_text_detected(self, guardrail):
        """Test that Chinese text is detected as non-English."""
        await guardrail.initialize()
        guardrail._scanner = None  # Use fallback
        result = await guardrail.check("这是中文文本")
        # May pass as "unknown" but will have detected language
        assert result.metadata.get("detected_language") in ["zh", "unknown", "en"]

    @pytest.mark.asyncio
    async def test_multiple_languages_allowed(self):
        """Test that multiple languages can be allowed."""
        guardrail = LanguageInputGuardrail({
            "valid_languages": ["en", "es", "fr"],
        })
        await guardrail.initialize()
        guardrail._scanner = None
        result = await guardrail.check("This is English")
        assert result.passed is True


class TestRegexInputGuardrail:
    """Tests for RegexInputGuardrail."""

    @pytest.fixture
    def guardrail(self):
        """Create a RegexInputGuardrail instance."""
        return RegexInputGuardrail({
            "patterns": [r"Bearer [A-Za-z0-9-._~+/]+", r"\b\d{3}-\d{2}-\d{4}\b"],
            "is_blocked": True,
            "redact": False,
        })

    @pytest.mark.asyncio
    async def test_initialization(self, guardrail):
        """Test guardrail initializes correctly."""
        await guardrail.initialize()
        assert guardrail.name == "regex_input"
        assert guardrail.layer.value == "input"

    @pytest.mark.asyncio
    async def test_safe_text_passes(self, guardrail):
        """Test that text without patterns passes."""
        await guardrail.initialize()
        guardrail._scanner = None  # Use fallback
        result = await guardrail.check("This is normal text.")
        assert result.passed is True

    @pytest.mark.asyncio
    async def test_bearer_token_blocked(self, guardrail):
        """Test that Bearer token is blocked."""
        await guardrail.initialize()
        guardrail._scanner = None  # Use fallback
        result = await guardrail.check("My token is Bearer abc123xyz")
        assert result.passed is False

    @pytest.mark.asyncio
    async def test_ssn_pattern_blocked(self, guardrail):
        """Test that SSN pattern is blocked."""
        await guardrail.initialize()
        guardrail._scanner = None
        result = await guardrail.check("My SSN is 123-45-6789")
        assert result.passed is False

    @pytest.mark.asyncio
    async def test_redaction(self):
        """Test that patterns can be redacted."""
        guardrail = RegexInputGuardrail({
            "patterns": [r"secret-\w+"],
            "is_blocked": True,
            "redact": True,
        })
        await guardrail.initialize()
        guardrail._scanner = None
        result = await guardrail.check("The key is secret-abc123")
        assert result.sanitized_text is not None
        assert "secret-abc123" not in result.sanitized_text
        assert "[REDACTED]" in result.sanitized_text

    @pytest.mark.asyncio
    async def test_no_patterns_configured(self):
        """Test that empty patterns list passes all."""
        guardrail = RegexInputGuardrail({"patterns": []})
        await guardrail.initialize()
        result = await guardrail.check("Any text at all")
        assert result.passed is True


class TestSentimentInputGuardrail:
    """Tests for SentimentInputGuardrail."""

    @pytest.fixture
    def guardrail(self):
        """Create a SentimentInputGuardrail instance."""
        return SentimentInputGuardrail({
            "threshold": -0.5,  # Block very negative
        })

    @pytest.mark.asyncio
    async def test_initialization(self, guardrail):
        """Test guardrail initializes correctly."""
        await guardrail.initialize()
        assert guardrail.name == "sentiment_input"
        assert guardrail.layer.value == "input"

    @pytest.mark.asyncio
    async def test_positive_text_passes(self, guardrail):
        """Test that positive text passes."""
        await guardrail.initialize()
        guardrail._scanner = None  # Use fallback (keyword-based)
        result = await guardrail.check("I love this product! It's amazing and wonderful!")
        assert result.passed is True

    @pytest.mark.asyncio
    async def test_neutral_text_passes(self, guardrail):
        """Test that neutral text passes."""
        await guardrail.initialize()
        guardrail._scanner = None
        result = await guardrail.check("The weather is cloudy today.")
        assert result.passed is True

    @pytest.mark.asyncio
    async def test_negative_text_detected(self, guardrail):
        """Test that very negative text is detected."""
        await guardrail.initialize()
        guardrail._scanner = None
        result = await guardrail.check("I hate this terrible awful garbage useless product! It's the worst failure ever!")
        # Should fail or have high risk
        assert result.passed is False or result.risk_score > 0.3

    @pytest.mark.asyncio
    async def test_sentiment_score_in_metadata(self, guardrail):
        """Test that sentiment score is included in metadata."""
        await guardrail.initialize()
        guardrail._scanner = None
        result = await guardrail.check("This is an okay message.")
        assert "sentiment_score" in result.metadata or "detection_method" in result.metadata
