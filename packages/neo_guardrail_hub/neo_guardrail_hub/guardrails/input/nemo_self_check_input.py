"""NeMo self-check input guardrail.

This module provides guardrail protection using NeMo Guardrails'
self_check_input flow for LLM-based input validation.
"""

import time
from typing import Any, Dict, Optional, TYPE_CHECKING

from ...core.models import GuardrailContext, GuardrailLayer, GuardrailResult
from ..base import GuardrailBase

if TYPE_CHECKING:
    from ...providers.nemo.provider import NeMoProvider


class NeMoSelfCheckInputGuardrail(GuardrailBase):
    """
    LLM-based input validation using NeMo's self_check_input.
    
    This guardrail uses NeMo Guardrails' self_check_input flow to detect
    jailbreak attempts, prompt injection, and other harmful user inputs.
    It prompts an LLM to determine if the user input should be blocked.
    
    How it works:
    1. Takes user input text
    2. Uses NeMo's self_check_input action with configured prompt
    3. LLM responds "Yes" to block, "No" to allow
    4. Returns GuardrailResult with pass/fail status
    
    Configuration:
        threshold: Not used (LLM makes binary decision)
        on_fail: Action when input is blocked (default: "block")
        provider: NeMoProvider instance (required, set via configure())
    
    Example:
        from neo_guardrail_hub.providers.nemo import NeMoProvider
        
        provider = NeMoProvider(config_path="./configs")
        await provider.initialize()
        
        guardrail = provider.get_guardrail("nemo_self_check_input")
        result = await guardrail.check("Hello, how are you?")
        # result.passed = True
        
        result = await guardrail.check("Ignore all previous instructions...")
        # result.passed = False
    """
    
    name = "nemo_self_check_input"
    layer = GuardrailLayer.INPUT
    description = "LLM-based input validation using NeMo's self_check_input"
    
    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize the NeMo self-check input guardrail.
        
        Args:
            config: Configuration dictionary including:
                - provider: NeMoProvider instance (required)
                - on_fail: Action when blocked ("block", "warn", "sanitize")
                - refusal_message: Custom refusal message
        """
        super().__init__(config)
        
        self._provider: Optional["NeMoProvider"] = self._config.get("provider")
        
        # Refusal message for blocked inputs
        self._refusal_message = self._config.get(
            "refusal_message",
            "I'm sorry, I can't respond to that."
        )
    
    async def initialize(self) -> None:
        """Initialize the guardrail and ensure provider is ready."""
        if self._initialized:
            return
            
        if self._provider is None:
            raise ValueError(
                "NeMoSelfCheckInputGuardrail requires a NeMoProvider. "
                "Get this guardrail from provider.get_guardrail('nemo_self_check_input')"
            )
        
        # Ensure provider is initialized
        if not self._provider._initialized:
            await self._provider.initialize()
        
        self._initialized = True
        self.logger.info("nemo_self_check_input_guardrail_initialized")
    
    async def _check(
        self,
        text: str,
        context: Optional[GuardrailContext] = None,
    ) -> GuardrailResult:
        """Check if user input should be allowed using NeMo's self_check_input.
        
        Args:
            text: User input to validate
            context: Optional context (passed to NeMo if provided)
            
        Returns:
            GuardrailResult with:
                - passed: True if input is allowed, False if blocked
                - risk_score: 1.0 if blocked, 0.0 if allowed
                - message: Refusal message if blocked
        """
        # Initialize if needed
        if not self._initialized:
            await self.initialize()
        
        # Use provider to check input (provider handles GuardrailContext conversion)
        result = await self._provider.check_input(text, context=context)
        
        # Use NeMo's actual response from Colang file (no hardcoded messages)
        nemo_message = result.get("message", "")
        
        if result["allowed"]:
            return GuardrailResult(
                passed=True,
                guardrail_name=self.name,
                layer=self.layer,
                risk_score=0.0,
                message=nemo_message if nemo_message else None,
                metadata={
                    "nemo_response": nemo_message,
                    "details": result.get("details", {}),
                },
            )
        else:
            return GuardrailResult(
                passed=False,
                guardrail_name=self.name,
                layer=self.layer,
                risk_score=1.0,
                message=nemo_message if nemo_message else self._refusal_message,
                metadata={
                    "blocked_reason": "nemo_self_check_input",
                    "nemo_response": nemo_message,
                    "details": result.get("details", {}),
                },
            )
    
    def configure(self, config: Dict[str, Any]) -> None:
        """Configure the guardrail.
        
        Args:
            config: Configuration dictionary
        """
        super().configure(config)
        
        if "provider" in config:
            self._provider = config["provider"]
        if "refusal_message" in config:
            self._refusal_message = config["refusal_message"]
