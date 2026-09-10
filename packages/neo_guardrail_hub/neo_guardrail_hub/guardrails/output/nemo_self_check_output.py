"""NeMo self-check output guardrail.

This module provides guardrail protection using NeMo Guardrails'
self_check_output flow for LLM-based output validation.

The self_check_output flow validates bot responses before they are
returned to users, checking for harmful, inappropriate, or policy-violating
content.

Reference:
- https://docs.nvidia.com/nemo/guardrails/latest/user-guides/guardrails-library.html#self-check-output
- https://docs.nvidia.com/nemo/guardrails/latest/getting-started/5-output-rails/README.html
"""

import time
from typing import Any, Dict, Optional, TYPE_CHECKING

from ...core.models import GuardrailContext, GuardrailLayer, GuardrailResult
from ..base import GuardrailBase

if TYPE_CHECKING:
    from ...providers.nemo.provider import NeMoProvider


class NeMoSelfCheckOutputGuardrail(GuardrailBase):
    """
    LLM-based output validation using NeMo's self_check_output.
    
    This guardrail uses NeMo Guardrails' self_check_output flow to validate
    bot responses before they are returned to users. It prompts an LLM to
    determine if the output should be blocked based on safety policies.
    
    How it works:
    1. Takes bot response text
    2. Uses NeMo's self_check_output action with configured prompt
    3. LLM responds "Yes" to block, "No" to allow
    4. Returns GuardrailResult with pass/fail status
    
    Common reasons for blocking output:
    - Harmful or abusive content
    - Illegal activity instructions
    - Sensitive information disclosure
    - Offensive or discriminatory content
    - Privacy violations
    
    Configuration:
        threshold: Not used (LLM makes binary decision)
        on_fail: Action when output is blocked (default: "block")
        provider: NeMoProvider instance (required, set via configure())
    
    Example:
        from neo_guardrail_hub.providers.nemo import NeMoProvider
        
        provider = NeMoProvider(config_path="./configs")
        await provider.initialize()
        
        guardrail = provider.get_guardrail("nemo_self_check_output")
        result = await guardrail.check("Here is how to hack a computer...")
        # result.passed = False
        
        result = await guardrail.check("The weather today is sunny and warm.")
        # result.passed = True
    
    NeMo Configuration:
        In config.yml, output rails must include:
        
        rails:
          output:
            flows:
              - self check output
        
        In prompts section (or prompts.yml):
        
        prompts:
          - task: self_check_output
            content: |
              Your task is to check if the bot message below complies with...
    """
    
    name = "nemo_self_check_output"
    layer = GuardrailLayer.OUTPUT
    description = "LLM-based output validation using NeMo's self_check_output"
    
    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize the NeMo self-check output guardrail.
        
        Args:
            config: Configuration dictionary including:
                - provider: NeMoProvider instance (required)
                - on_fail: Action when blocked ("block", "warn", "sanitize")
                - refusal_message: Custom refusal message
        """
        super().__init__(config)
        
        self._provider: Optional["NeMoProvider"] = self._config.get("provider")
        
        # Refusal message for blocked outputs
        self._refusal_message = self._config.get(
            "refusal_message",
            "I'm sorry, I can't provide that response."
        )
    
    async def initialize(self) -> None:
        """Initialize the guardrail and ensure provider is ready."""
        if self._initialized:
            return
            
        if self._provider is None:
            raise ValueError(
                "NeMoSelfCheckOutputGuardrail requires a NeMoProvider. "
                "Get this guardrail from provider.get_guardrail('nemo_self_check_output')"
            )
        
        # Ensure provider is initialized
        if not self._provider._initialized:
            await self._provider.initialize()
        
        self._initialized = True
        self.logger.info("nemo_self_check_output_guardrail_initialized")
    
    async def _check(
        self,
        text: str,
        context: Optional[GuardrailContext] = None,
    ) -> GuardrailResult:
        """Check if bot output should be allowed using NeMo's self_check_output.
        
        Args:
            text: Bot response to validate
            context: Optional context (may include user input for better checking)
            
        Returns:
            GuardrailResult with:
                - passed: True if output is allowed, False if blocked
                - risk_score: 1.0 if blocked, 0.0 if allowed
                - message: Refusal message if blocked
        """
        # Initialize if needed
        if not self._initialized:
            await self.initialize()
        
        # Use provider to check output (provider handles GuardrailContext conversion)
        result = await self._provider.check_output(text, context=context)
        
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
                    "blocked_reason": "nemo_self_check_output",
                    "nemo_response": nemo_message,
                    "details": result.get("details", {}),
                    "on_fail": self._config.get("on_fail", "block"),
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
