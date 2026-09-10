"""Tests for FactExtractor with mocked OpenAI client."""
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from neo_memory_hub.extraction.extractor import FactExtractor, _NULL_NAME_STRINGS
from neo_memory_hub.extraction.models import ExtractedFact, ExtractionResult


class TestFactExtractorInit:
    """Test FactExtractor initialization."""

    def test_default_init(self):
        ext = FactExtractor()
        assert ext._llm_model == "gpt-4o-mini"
        assert ext._temperature == 0.1
        assert "persona" in ext._valid_categories
        assert ext._default_salience == 0.5

    def test_custom_init(self):
        ext = FactExtractor(
            llm_model="gpt-4o",
            temperature=0.2,
            valid_categories=["custom_cat"],
            default_salience=0.7,
        )
        assert ext._llm_model == "gpt-4o"
        assert ext._valid_categories == {"custom_cat"}
        assert ext._default_salience == 0.7


class TestCategoryValidation:
    """Test category validation logic."""

    def test_valid_category(self):
        ext = FactExtractor()
        assert ext._validate_category("persona") == "persona"
        assert ext._validate_category("EPISODIC") == "episodic"

    def test_invalid_category_defaults_to_persona(self):
        ext = FactExtractor()
        assert ext._validate_category("invalid_cat") == "persona"
        assert ext._validate_category("") == "persona"
        assert ext._validate_category(None) == "persona"

    def test_custom_categories(self):
        ext = FactExtractor(valid_categories=["alpha", "beta"])
        assert ext._validate_category("alpha") == "alpha"
        assert ext._validate_category("persona") == "persona"  # not in custom set → default


class TestEntityNameCleaning:
    """Test entity name cleanup."""

    def test_valid_name(self):
        assert FactExtractor._clean_entity_name("Senthil Kumar") == "Senthil Kumar"

    def test_null_like_names(self):
        for null_name in ["null", "None", "unknown", "N/A", "", "the client", "User"]:
            assert FactExtractor._clean_entity_name(null_name) is None

    def test_non_string_returns_none(self):
        assert FactExtractor._clean_entity_name(None) is None
        assert FactExtractor._clean_entity_name(123) is None

    def test_strips_whitespace(self):
        assert FactExtractor._clean_entity_name("  John  ") == "John"


class TestBuildPrompt:
    """Test system prompt building."""

    def test_default_prompt_uses_template(self):
        ext = FactExtractor()
        prompt = ext._build_system_prompt()
        assert "persona" in prompt
        assert "preference" in prompt

    def test_custom_prompt_used_directly(self):
        ext = FactExtractor(extraction_prompt="My custom prompt")
        assert ext._build_system_prompt() == "My custom prompt"


class TestExtraction:
    """Test the full extraction pipeline with mocked LLM."""

    @pytest.fixture
    def mock_openai(self):
        """Create a mock OpenAI client."""
        with patch("openai.AsyncOpenAI") as mock_cls:
            mock_client = AsyncMock()
            mock_cls.return_value = mock_client
            yield mock_client

    def _make_completion(self, content: str):
        """Create a mock completion response."""
        msg = MagicMock()
        msg.content = content
        choice = MagicMock()
        choice.message = msg
        response = MagicMock()
        response.choices = [choice]
        return response

    @pytest.mark.asyncio
    async def test_successful_extraction(self, mock_openai):
        llm_output = json.dumps({
            "memories": [
                {
                    "category": "persona",
                    "investor_name": "Senthil Kumar",
                    "key": "risk_behavior",
                    "value": "Conservative investor",
                    "detail": "Prefers low risk",
                    "salience": 0.8,
                    "reasoning": "Core identity fact",
                    "pool": "private",
                },
                {
                    "category": "preference",
                    "investor_name": "Senthil Kumar",
                    "key": "report_format",
                    "value": "Monthly PDF",
                    "salience": 0.6,
                    "reasoning": "Service preference",
                },
            ]
        })
        mock_openai.chat.completions.create = AsyncMock(
            return_value=self._make_completion(llm_output)
        )

        ext = FactExtractor()
        result = await ext.extract(query="Tell me about Senthil", response="He is conservative")

        assert isinstance(result, ExtractionResult)
        assert len(result.facts) == 2
        assert result.facts[0].category == "persona"
        assert result.facts[0].content == "Conservative investor"
        assert result.facts[0].investor_name == "Senthil Kumar"
        assert result.facts[0].salience == 0.8
        assert result.facts[0].pool == "private"
        assert result.facts[0].profile_key == "risk_behavior"
        assert result.facts[0].detail == "Prefers low risk"
        assert "Senthil Kumar" in result.investor_names_found

    @pytest.mark.asyncio
    async def test_extraction_salience_clamping(self, mock_openai):
        llm_output = json.dumps({
            "memories": [
                {"category": "persona", "value": "fact1", "salience": 1.5},
                {"category": "persona", "value": "fact2", "salience": -0.5},
                {"category": "persona", "value": "fact3", "salience": "not_a_number"},
            ]
        })
        mock_openai.chat.completions.create = AsyncMock(
            return_value=self._make_completion(llm_output)
        )

        ext = FactExtractor(default_salience=0.4)
        result = await ext.extract(query="q", response="r")
        assert result.facts[0].salience == 1.0  # clamped to max
        assert result.facts[1].salience == 0.0  # clamped to min
        assert result.facts[2].salience == 0.4  # fallback to default

    @pytest.mark.asyncio
    async def test_extraction_pool_validation(self, mock_openai):
        llm_output = json.dumps({
            "memories": [
                {"category": "persona", "value": "fact1", "pool": "team"},
                {"category": "persona", "value": "fact2", "pool": "INVALID"},
                {"category": "persona", "value": "fact3", "pool": ""},
            ]
        })
        mock_openai.chat.completions.create = AsyncMock(
            return_value=self._make_completion(llm_output)
        )

        ext = FactExtractor()
        result = await ext.extract(query="q", response="r")
        assert result.facts[0].pool == "team"
        assert result.facts[1].pool is None  # invalid
        assert result.facts[2].pool is None  # empty

    @pytest.mark.asyncio
    async def test_extraction_skips_empty_values(self, mock_openai):
        llm_output = json.dumps({
            "memories": [
                {"category": "persona", "value": ""},
                {"category": "persona", "value": "  "},
                {"category": "persona", "value": "valid fact"},
            ]
        })
        mock_openai.chat.completions.create = AsyncMock(
            return_value=self._make_completion(llm_output)
        )

        ext = FactExtractor()
        result = await ext.extract(query="q", response="r")
        assert len(result.facts) == 1
        assert result.facts[0].content == "valid fact"

    @pytest.mark.asyncio
    async def test_extraction_llm_failure(self, mock_openai):
        mock_openai.chat.completions.create = AsyncMock(
            side_effect=Exception("LLM API error")
        )

        ext = FactExtractor()
        result = await ext.extract(query="q", response="r")
        assert result.error is not None
        assert "LLM API error" in result.error
        assert result.facts == []

    @pytest.mark.asyncio
    async def test_extraction_truncates_long_input(self, mock_openai):
        llm_output = json.dumps({"memories": []})
        mock_openai.chat.completions.create = AsyncMock(
            return_value=self._make_completion(llm_output)
        )

        ext = FactExtractor()
        long_query = "x" * 10000
        long_response = "y" * 10000
        result = await ext.extract(query=long_query, response=long_response, max_input_chars=500)
        # Should not error — just truncate
        assert result.error is None

    @pytest.mark.asyncio
    async def test_extraction_non_dict_items_skipped(self, mock_openai):
        llm_output = json.dumps({
            "memories": [
                "not a dict",
                42,
                {"category": "persona", "value": "valid"},
            ]
        })
        mock_openai.chat.completions.create = AsyncMock(
            return_value=self._make_completion(llm_output)
        )

        ext = FactExtractor()
        result = await ext.extract(query="q", response="r")
        assert len(result.facts) == 1

    @pytest.mark.asyncio
    async def test_extraction_behavioral_note_and_trigger(self, mock_openai):
        llm_output = json.dumps({
            "memories": [
                {
                    "category": "episodic",
                    "value": "Sold house in 2024",
                    "behavioral_note": "Shows risk aversion",
                    "salience": 0.8,
                },
                {
                    "category": "procedural",
                    "value": "Send quarterly report",
                    "trigger": "Every quarter end",
                    "salience": 0.9,
                },
            ]
        })
        mock_openai.chat.completions.create = AsyncMock(
            return_value=self._make_completion(llm_output)
        )

        ext = FactExtractor()
        result = await ext.extract(query="q", response="r")
        assert result.facts[0].behavioral_note == "Shows risk aversion"
        assert result.facts[1].trigger == "Every quarter end"


class TestExtractedFactModel:
    """Test the ExtractedFact dataclass."""

    def test_default_values(self):
        fact = ExtractedFact(category="persona", content="test")
        assert fact.salience == 0.5
        assert fact.pool is None
        assert fact.investor_name is None
        assert fact.profile_key == ""

    def test_full_construction(self):
        fact = ExtractedFact(
            category="episodic",
            content="Sold house",
            investor_name="John",
            profile_key="housing_decision",
            detail="In Mumbai",
            behavioral_note="Risk averse",
            trigger="",
            salience=0.9,
            salience_reasoning="Important event",
            pool="private",
        )
        assert fact.content == "Sold house"
        assert fact.salience == 0.9


class TestExtractionResultModel:
    """Test the ExtractionResult dataclass."""

    def test_default_values(self):
        result = ExtractionResult()
        assert result.facts == []
        assert result.investor_names_found == set()
        assert result.raw_llm_response == ""
        assert result.error is None

    def test_with_error(self):
        result = ExtractionResult(error="Something went wrong")
        assert result.error == "Something went wrong"
