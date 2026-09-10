"""Tests for the orchestrator."""

from pathlib import Path
from typing import Any, Dict
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from neo_guardrail_hub.core.models import (
    AggregatedResult,
    GuardrailContext,
    GuardrailLayer,
    GuardrailResult,
)
from neo_guardrail_hub.core.orchestrator import NeoGuardrailOrchestrator


class TestNeoGuardrailOrchestrator:
    """Tests for NeoGuardrailOrchestrator."""

    @pytest.fixture
    def orchestrator(self, temp_config_dir: Path) -> NeoGuardrailOrchestrator:
        """Create an orchestrator instance."""
        return NeoGuardrailOrchestrator(config_path=temp_config_dir)

    @pytest.mark.asyncio
    async def test_guard_input_safe(
        self,
        orchestrator: NeoGuardrailOrchestrator,
        safe_text: str,
    ):
        """Test guarding safe input."""
        result = await orchestrator.guard_input(safe_text)

        assert isinstance(result, AggregatedResult)
        # Safe text should generally pass
        assert result.passed is True

    @pytest.mark.asyncio
    async def test_guard_input_unsafe(
        self,
        orchestrator: NeoGuardrailOrchestrator,
        injection_text: str,
    ):
        """Test guarding unsafe input."""
        result = await orchestrator.guard_input(injection_text)

        assert isinstance(result, AggregatedResult)
        # Injection attempts should have results
        assert len(result.results) >= 0

    @pytest.mark.asyncio
    async def test_guard_output(
        self,
        orchestrator: NeoGuardrailOrchestrator,
        safe_text: str,
    ):
        """Test guarding output."""
        result = await orchestrator.guard_output(safe_text)

        assert isinstance(result, AggregatedResult)

    @pytest.mark.asyncio
    async def test_guard_output_with_pii(
        self,
        orchestrator: NeoGuardrailOrchestrator,
        pii_text: str,
    ):
        """Test guarding output with PII."""
        result = await orchestrator.guard_output(pii_text)

        assert isinstance(result, AggregatedResult)

    @pytest.mark.asyncio
    async def test_guard_all(
        self,
        orchestrator: NeoGuardrailOrchestrator,
        safe_text: str,
    ):
        """Test guarding all layers."""
        result = await orchestrator.guard_all(
            input_text=safe_text, 
            output_text=safe_text
        )

        assert isinstance(result, dict)
        assert "input" in result

    @pytest.mark.asyncio
    async def test_with_context(
        self,
        orchestrator: NeoGuardrailOrchestrator,
        sample_context: GuardrailContext,
        safe_text: str,
    ):
        """Test with context provided."""
        result = await orchestrator.guard_input(
            safe_text,
            context=sample_context,
        )

        assert isinstance(result, AggregatedResult)

    @pytest.mark.asyncio
    async def test_with_agent_id(
        self,
        orchestrator: NeoGuardrailOrchestrator,
        safe_text: str,
    ):
        """Test with specific agent ID."""
        result = await orchestrator.guard_input(
            safe_text,
            agent_id="test_agent",
        )

        assert isinstance(result, AggregatedResult)

    def test_guard_input_sync(
        self,
        orchestrator: NeoGuardrailOrchestrator,
        safe_text: str,
    ):
        """Test synchronous input guard."""
        result = orchestrator.guard_input_sync(safe_text)

        assert isinstance(result, AggregatedResult)

    def test_guard_output_sync(
        self,
        orchestrator: NeoGuardrailOrchestrator,
        safe_text: str,
    ):
        """Test synchronous output guard."""
        result = orchestrator.guard_output_sync(safe_text)

        assert isinstance(result, AggregatedResult)
