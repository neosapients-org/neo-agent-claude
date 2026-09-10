"""Integration tests for Phase 2 guardrails.

Tests running multiple Phase 2 guardrails together.
"""

import asyncio
from typing import List, Tuple
import pytest

from neo_guardrail_hub.core.models import GuardrailResult, GuardrailContext
from neo_guardrail_hub.core.interfaces import BaseGuardrail
from neo_guardrail_hub.guardrails.input import (
    BanSubstringsInputGuardrail,
    HarmfulContentGuardrail,
    InputLengthGuardrail,
    SecretDetectionGuardrail,
    ToxicityInputGuardrail,
)
from neo_guardrail_hub.guardrails.output import (
    BanSubstringsOutputGuardrail,
    BanTopicsGuardrail,
    NoRefusalGuardrail,
    RelevanceGuardrail,
    ToxicityOutputGuardrail,
)


async def run_guardrails(
    guardrails: List[BaseGuardrail],
    text: str,
    context: GuardrailContext = None
) -> Tuple[bool, List[GuardrailResult]]:
    """Run multiple guardrails and collect results.
    
    Returns:
        Tuple of (all_passed, list of results)
    """
    if context is None:
        context = GuardrailContext(
            user_input=text,
            conversation_history=[],
            metadata={}
        )
    
    results = []
    for guardrail in guardrails:
        result = await guardrail.check(text, context)
        results.append(result)
    
    all_passed = all(r.passed for r in results)
    return all_passed, results


class TestPhase2InputGuardrailsIntegration:
    """Integration tests for Phase 2 input guardrails."""

    @pytest.fixture
    def input_guardrails(self) -> List[BaseGuardrail]:
        """Create Phase 2 input guardrails."""
        return [
            SecretDetectionGuardrail({
                "threshold": 0.5,
                "redact_mode": "partial",
            }),
            InputLengthGuardrail({
                "min_length": 5,
                "max_length": 1000,
            }),
            ToxicityInputGuardrail({
                "threshold": 0.3,
            }),
            BanSubstringsInputGuardrail({
                "substrings": ["forbidden", "blocked"],
            }),
            HarmfulContentGuardrail({
                "categories": ["violence", "hate_speech"],
            }),
        ]

    @pytest.mark.asyncio
    async def test_safe_input_passes_all_guardrails(self, input_guardrails):
        """Test that safe input passes all Phase 2 input guardrails."""
        text = "Hello, I have a question about programming in Python."
        all_passed, results = await run_guardrails(input_guardrails, text)
        
        assert all_passed is True
        assert len(results) == 5
        for result in results:
            assert result.passed is True

    @pytest.mark.asyncio
    async def test_secret_blocks_input(self, input_guardrails):
        """Test that secret detection blocks input with secrets."""
        # Use AWS key pattern for fallback detection
        text = "My AWS key is AKIAIOSFODNN7EXAMPLE"
        all_passed, results = await run_guardrails(input_guardrails, text)
        
        # Should fail due to secret detection
        secret_result = results[0]  # SecretDetectionGuardrail is first
        assert secret_result.passed is False

    @pytest.mark.asyncio
    async def test_banned_substring_blocks_input(self, input_guardrails):
        """Test that banned substrings are detected."""
        text = "This contains a forbidden word"
        all_passed, results = await run_guardrails(input_guardrails, text)
        
        assert all_passed is False
        # BanSubstringsInputGuardrail is at index 3
        ban_result = results[3]
        assert ban_result.passed is False

    @pytest.mark.asyncio
    async def test_length_violation_detected(self, input_guardrails):
        """Test that length violations are detected."""
        text = "Hi"  # Too short (min_length is 5)
        all_passed, results = await run_guardrails(input_guardrails, text)
        
        assert all_passed is False
        # InputLengthGuardrail is at index 1
        length_result = results[1]
        assert length_result.passed is False

    @pytest.mark.asyncio
    async def test_multiple_violations_detected(self, input_guardrails):
        """Test that multiple violations are detected."""
        # Input that violates multiple guardrails: banned substring + secret
        text = "forbidden AKIAIOSFODNN7EXAMPLE"
        all_passed, results = await run_guardrails(input_guardrails, text)
        
        assert all_passed is False
        # Count failures
        failures = [r for r in results if not r.passed]
        assert len(failures) >= 2


class TestPhase2OutputGuardrailsIntegration:
    """Integration tests for Phase 2 output guardrails."""

    @pytest.fixture
    def output_guardrails(self) -> List[BaseGuardrail]:
        """Create Phase 2 output guardrails."""
        return [
            NoRefusalGuardrail({
                "threshold": 0.5,
            }),
            RelevanceGuardrail({
                "threshold": 0.3,
            }),
            ToxicityOutputGuardrail({
                "threshold": 0.3,
            }),
            BanTopicsGuardrail({
                "topics": ["politics", "violence"],
            }),
            BanSubstringsOutputGuardrail({
                "substrings": ["confidential", "internal only"],
            }),
        ]

    @pytest.mark.asyncio
    async def test_safe_output_passes_all_guardrails(self, output_guardrails):
        """Test that safe output passes all Phase 2 output guardrails."""
        text = "Here's a helpful response to your programming question."
        context = GuardrailContext(
            user_input="How do I write Python code?",
            conversation_history=[],
            metadata={"original_prompt": "How do I write Python code?"}
        )
        
        all_passed, results = await run_guardrails(output_guardrails, text, context)
        
        assert all_passed is True

    @pytest.mark.asyncio
    async def test_refusal_detected_in_output(self, output_guardrails):
        """Test that refusal responses are detected."""
        text = "I'm sorry, but I cannot help with that request."
        context = GuardrailContext(
            user_input="Tell me about cooking.",
            conversation_history=[],
            metadata={"original_prompt": "Tell me about cooking."}
        )
        
        all_passed, results = await run_guardrails(output_guardrails, text, context)
        
        # Should fail due to no_refusal guardrail
        assert all_passed is False
        refusal_result = results[0]  # NoRefusalGuardrail is first
        assert refusal_result.passed is False

    @pytest.mark.asyncio
    async def test_toxic_output_blocked(self, output_guardrails):
        """Test that toxic outputs are blocked."""
        text = "You're worthless and pathetic!"
        context = GuardrailContext(
            user_input="Hello",
            conversation_history=[],
            metadata={}
        )
        
        all_passed, results = await run_guardrails(output_guardrails, text, context)
        
        assert all_passed is False

    @pytest.mark.asyncio
    async def test_banned_topic_blocked(self, output_guardrails):
        """Test that banned topics are blocked."""
        text = "Let me tell you about the election and political parties. The violence in the debate was concerning."
        context = GuardrailContext(
            user_input="What happened?",
            conversation_history=[],
            metadata={}
        )
        
        all_passed, results = await run_guardrails(output_guardrails, text, context)
        
        assert all_passed is False
        # BanTopicsGuardrail is at index 3
        topic_result = results[3]
        assert topic_result.passed is False

    @pytest.mark.asyncio
    async def test_banned_substring_in_output(self, output_guardrails):
        """Test that banned substrings in output are detected."""
        text = "This document is confidential and for internal only viewing."
        context = GuardrailContext(
            user_input="Show me the document",
            conversation_history=[],
            metadata={}
        )
        
        all_passed, results = await run_guardrails(output_guardrails, text, context)
        
        assert all_passed is False


class TestFullPipelineIntegration:
    """Integration tests for full input-output pipeline with Phase 2 guardrails."""

    @pytest.fixture
    def input_guardrails(self) -> List[BaseGuardrail]:
        """Create input guardrails for pipeline."""
        return [
            InputLengthGuardrail({
                "min_length": 5,
                "max_length": 1000,
            }),
            ToxicityInputGuardrail({
                "threshold": 0.3,
            }),
            BanSubstringsInputGuardrail({
                "substrings": ["forbidden"],
            }),
        ]

    @pytest.fixture
    def output_guardrails(self) -> List[BaseGuardrail]:
        """Create output guardrails for pipeline."""
        return [
            NoRefusalGuardrail({
                "threshold": 0.5,
            }),
            ToxicityOutputGuardrail({
                "threshold": 0.3,
            }),
        ]

    @pytest.mark.asyncio
    async def test_full_pipeline_safe_content(self, input_guardrails, output_guardrails):
        """Test full pipeline with safe input and output."""
        input_text = "Tell me about Python programming."
        output_text = "Python is a versatile programming language."
        
        # Check input
        context = GuardrailContext(
            user_input=input_text,
            conversation_history=[],
            metadata={}
        )
        input_passed, input_results = await run_guardrails(input_guardrails, input_text, context)
        assert input_passed is True
        
        # Check output
        context.metadata["original_prompt"] = input_text
        output_passed, output_results = await run_guardrails(output_guardrails, output_text, context)
        assert output_passed is True

    @pytest.mark.asyncio
    async def test_input_blocked_stops_pipeline(self, input_guardrails, output_guardrails):
        """Test that blocked input would stop the pipeline."""
        input_text = "forbidden content here"
        
        context = GuardrailContext(
            user_input=input_text,
            conversation_history=[],
            metadata={}
        )
        
        # Check input with banned content
        input_passed, input_results = await run_guardrails(input_guardrails, input_text, context)
        
        # Should fail at input stage
        assert input_passed is False

    @pytest.mark.asyncio
    async def test_output_blocked_after_good_input(self, input_guardrails, output_guardrails):
        """Test that bad output is blocked even with good input."""
        input_text = "Tell me a joke."
        output_text = "You're an idiot for asking that!"
        
        context = GuardrailContext(
            user_input=input_text,
            conversation_history=[],
            metadata={}
        )
        
        # Input passes
        input_passed, _ = await run_guardrails(input_guardrails, input_text, context)
        assert input_passed is True
        
        # Output fails (toxic)
        context.metadata["original_prompt"] = input_text
        output_passed, _ = await run_guardrails(output_guardrails, output_text, context)
        assert output_passed is False


class TestGuardrailCombinations:
    """Test various combinations of guardrails."""

    @pytest.mark.asyncio
    async def test_all_phase2_guardrails_together(self):
        """Test all Phase 2 guardrails can be instantiated together."""
        # Create all Phase 2 guardrails
        guardrails = [
            SecretDetectionGuardrail({}),
            InputLengthGuardrail({}),
            ToxicityInputGuardrail({}),
            BanSubstringsInputGuardrail({"substrings": ["test"]}),
            HarmfulContentGuardrail({"categories": ["violence"]}),
            NoRefusalGuardrail({}),
            RelevanceGuardrail({}),
            ToxicityOutputGuardrail({}),
            BanTopicsGuardrail({"topics": ["politics"]}),
            BanSubstringsOutputGuardrail({"substrings": ["secret"]}),
        ]
        
        # Should have all guardrails
        assert len(guardrails) == 10
        
        # All should have names
        for g in guardrails:
            assert g.name is not None

    @pytest.mark.asyncio
    async def test_input_and_output_guardrails_independent(self):
        """Test that input and output guardrails work independently."""
        input_guardrails = [
            SecretDetectionGuardrail({}),
            InputLengthGuardrail({"min_length": 5}),
        ]
        
        output_guardrails = [
            NoRefusalGuardrail({}),
            BanTopicsGuardrail({"topics": ["violence"]}),
        ]
        
        # Safe text for input
        input_text = "Hello, how are you today?"
        context = GuardrailContext(
            user_input=input_text,
            conversation_history=[],
            metadata={}
        )
        
        input_passed, _ = await run_guardrails(input_guardrails, input_text, context)
        assert input_passed is True
        
        # Safe text for output
        output_text = "I'm doing great, thanks for asking!"
        context.metadata["original_prompt"] = input_text
        output_passed, _ = await run_guardrails(output_guardrails, output_text, context)
        assert output_passed is True

    @pytest.mark.asyncio
    async def test_parallel_execution(self):
        """Test that guardrails can be run in parallel."""
        guardrails = [
            InputLengthGuardrail({"min_length": 5}),
            BanSubstringsInputGuardrail({"substrings": ["forbidden"]}),
            SecretDetectionGuardrail({}),
        ]
        
        text = "Hello, how are you today?"
        context = GuardrailContext(
            user_input=text,
            conversation_history=[],
            metadata={}
        )
        
        # Run in parallel
        tasks = [g.check(text, context) for g in guardrails]
        results = await asyncio.gather(*tasks)
        
        # All should pass
        assert len(results) == 3
        assert all(r.passed for r in results)
