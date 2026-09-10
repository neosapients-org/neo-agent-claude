"""Tests for core models."""

import pytest

from neo_guardrail_hub.core.models import (
    ActionOnFail,
    AggregatedResult,
    ExecutionMode,
    GuardrailContext,
    GuardrailLayer,
    GuardrailResult,
)


class TestGuardrailLayer:
    """Tests for GuardrailLayer enum."""

    def test_layer_values(self):
        """Test enum values."""
        assert GuardrailLayer.INPUT.value == "input"
        assert GuardrailLayer.CONTEXT.value == "context"
        assert GuardrailLayer.OUTPUT.value == "output"

    def test_layer_from_string(self):
        """Test creating enum from string."""
        assert GuardrailLayer("input") == GuardrailLayer.INPUT
        assert GuardrailLayer("context") == GuardrailLayer.CONTEXT
        assert GuardrailLayer("output") == GuardrailLayer.OUTPUT


class TestActionOnFail:
    """Tests for ActionOnFail enum."""

    def test_action_values(self):
        """Test enum values."""
        assert ActionOnFail.BLOCK.value == "block"
        assert ActionOnFail.WARN.value == "warn"
        assert ActionOnFail.SANITIZE.value == "sanitize"


class TestGuardrailResult:
    """Tests for GuardrailResult model."""

    def test_create_result(self):
        """Test creating a result."""
        result = GuardrailResult(
            guardrail_name="test",
            layer=GuardrailLayer.INPUT,
            passed=True,
            risk_score=0.1,
            message="Test passed",
        )

        assert result.guardrail_name == "test"
        assert result.layer == GuardrailLayer.INPUT
        assert result.passed is True
        assert result.risk_score == 0.1
        assert result.message == "Test passed"

    def test_result_with_details(self):
        """Test result with additional details."""
        result = GuardrailResult(
            guardrail_name="pii_detection",
            layer=GuardrailLayer.INPUT,
            passed=False,
            risk_score=0.8,
            message="PII detected",
            metadata={"entities": ["email", "phone"]},
            sanitized_text="My email is [REDACTED]",
        )

        assert result.metadata == {"entities": ["email", "phone"]}
        assert result.sanitized_text == "My email is [REDACTED]"

    def test_result_validation(self):
        """Test risk score validation."""
        # Valid scores
        GuardrailResult(
            guardrail_name="test",
            layer=GuardrailLayer.INPUT,
            passed=True,
            risk_score=0.0,
        )
        GuardrailResult(
            guardrail_name="test",
            layer=GuardrailLayer.INPUT,
            passed=True,
            risk_score=1.0,
        )

        # Invalid scores should raise
        with pytest.raises(ValueError):
            GuardrailResult(
                guardrail_name="test",
                layer=GuardrailLayer.INPUT,
                passed=True,
                risk_score=-0.1,
            )

        with pytest.raises(ValueError):
            GuardrailResult(
                guardrail_name="test",
                layer=GuardrailLayer.INPUT,
                passed=True,
                risk_score=1.5,
            )


class TestAggregatedResult:
    """Tests for AggregatedResult model."""

    def test_aggregated_result(self):
        """Test aggregated result creation."""
        results = [
            GuardrailResult(
                guardrail_name="g1",
                layer=GuardrailLayer.INPUT,
                passed=True,
                risk_score=0.2,
            ),
            GuardrailResult(
                guardrail_name="g2",
                layer=GuardrailLayer.INPUT,
                passed=True,
                risk_score=0.3,
            ),
        ]

        agg = AggregatedResult(
            passed=True,
            layer=GuardrailLayer.INPUT,
            results=results,
            total_latency_ms=25.5,
        )

        assert agg.passed is True
        assert len(agg.results) == 2
        assert agg.total_latency_ms == 25.5

    def test_max_risk_score(self):
        """Test max risk score property."""
        results = [
            GuardrailResult(
                guardrail_name="g1",
                layer=GuardrailLayer.INPUT,
                passed=True,
                risk_score=0.2,
            ),
            GuardrailResult(
                guardrail_name="g2",
                layer=GuardrailLayer.INPUT,
                passed=False,
                risk_score=0.8,
            ),
        ]

        agg = AggregatedResult(passed=False, layer=GuardrailLayer.INPUT, results=results)
        assert agg.max_risk_score == 0.8

    def test_failed_checks(self):
        """Test failed_checks property."""
        results = [
            GuardrailResult(
                guardrail_name="g1",
                layer=GuardrailLayer.INPUT,
                passed=True,
                risk_score=0.2,
            ),
            GuardrailResult(
                guardrail_name="g2",
                layer=GuardrailLayer.INPUT,
                passed=False,
                risk_score=0.8,
            ),
        ]

        agg = AggregatedResult(passed=False, layer=GuardrailLayer.INPUT, results=results)
        assert len(agg.failed_checks) == 1
        assert agg.failed_checks[0].guardrail_name == "g2"


class TestGuardrailContext:
    """Tests for GuardrailContext model."""

    def test_context_creation(self):
        """Test context creation."""
        ctx = GuardrailContext(
            agent_id="agent1",
            session_id="session1",
            user_id="user1",
        )

        assert ctx.agent_id == "agent1"
        assert ctx.session_id == "session1"
        assert ctx.user_id == "user1"
        assert ctx.conversation_history == []
        assert ctx.metadata == {}

    def test_add_message(self):
        """Test adding messages to history."""
        ctx = GuardrailContext(agent_id="agent1")
        ctx.add_message("user", "Hello")
        ctx.add_message("assistant", "Hi there!")

        assert len(ctx.conversation_history) == 2
        assert ctx.conversation_history[0] == {"role": "user", "content": "Hello"}
        assert ctx.conversation_history[1] == {"role": "assistant", "content": "Hi there!"}
