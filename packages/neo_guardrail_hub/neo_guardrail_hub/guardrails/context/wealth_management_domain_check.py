"""Wealth Management domain check guardrail.

This module provides wealth management domain-specific validation using NeMo Guardrails'
prompt-based classification. It validates that user queries are appropriate for a
wealth management context.
"""

import time
from typing import Any, Dict, Optional, TYPE_CHECKING

from ...core.models import GuardrailContext, GuardrailLayer, GuardrailResult
from ..base import GuardrailBase

if TYPE_CHECKING:
    from ...providers.nemo.provider import NeMoProvider


class WealthManagementDomainCheckGuardrail(GuardrailBase):
    """
    Wealth management domain-specific validation using NeMo's prompt-based classification.
    
    This guardrail uses the wealth_management_domain_check prompt (generated from default.yaml)
    to classify user queries as ALLOWED or BLOCKED based on wealth management domain rules.
    
    How it works:
    1. Takes user message text
    2. Uses task_manager.render_task_prompt(task="wealth_management_domain_check")
    3. LLM classifies the message based on wealth management rules
    4. If response starts with "BLOCKED:", the query is blocked
    5. Returns GuardrailResult with pass/fail status
    
    Domain Rules:
    - ALLOW: General wealth management education, concepts, strategies, planning
    - BLOCK: Specific investment recommendations, stock picks, market timing advice
    
    Configuration:
        provider: NeMoProvider instance (required, set via configure())
        mode: "blocking" or "warning" (default: "blocking")
    
    Example:
        from neo_guardrail_hub.providers.nemo import NeMoProvider
        
        provider = NeMoProvider(config_path="./configs")
        await provider.initialize()
        
        guardrail = provider.get_guardrail("wealth_management_domain_check")
        
        # Educational question - allowed
        result = await guardrail.check("What is asset allocation?")
        # result.passed = True
        
        # Specific investment advice - blocked
        result = await guardrail.check("Should I buy Tesla stock?")
        # result.passed = False
    
    Note:
        The prompt is automatically generated from default.yaml when NeMo configs are created.
        This ensures consistency between YAML config and runtime behavior.
    
    Reference:
        Uses task_manager.render_task_prompt() approach (same as finance_guardrail.ipynb).
    """
    
    name = "wealth_management_domain_check"
    layer = GuardrailLayer.CONTEXT
    description = "Wealth management domain-specific validation using prompt-based classification"
    
    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize the wealth management domain check guardrail.
        
        Args:
            config: Configuration dictionary including:
                - provider: NeMoProvider instance (required)
                - mode: "blocking" or "warning" (default: "blocking")
        """
        super().__init__(config)
        
        self._provider: Optional["NeMoProvider"] = self._config.get("provider")
        
        # Mode: blocking vs warning
        self._mode = self._config.get("mode", "blocking")
        
        # Refusal message for blocked queries
        self._refusal_message = self._config.get(
            "refusal_message",
            "I cannot provide specific investment recommendations or advice on particular assets."
        )
    
    @property
    def mode(self) -> str:
        """Get the guardrail mode (blocking or warning)."""
        return self._mode
    
    async def initialize(self) -> None:
        """Initialize the guardrail and ensure provider is ready."""
        if self._initialized:
            return
            
        if self._provider is None:
            raise ValueError(
                "WealthManagementDomainCheckGuardrail requires a NeMoProvider. "
                "Get this guardrail from provider.get_guardrail('wealth_management_domain_check')"
            )
        
        # Ensure provider is initialized
        if not self._provider._initialized:
            await self._provider.initialize()
        
        self._initialized = True
        self.logger.info(
            "wealth_management_domain_check_guardrail_initialized",
            mode=self._mode,
        )
    
    async def _check(
        self,
        text: str,
        context: Optional[GuardrailContext] = None,
    ) -> GuardrailResult:
        """Check if user message is allowed in wealth management domain.
        
        Args:
            text: User message to check for domain compliance
            context: Optional context with conversation history
            
        Returns:
            GuardrailResult with:
                - passed: True if allowed, False if blocked
                - risk_score: 1.0 if blocked, 0.0 if allowed
                - message: Response from LLM or refusal message
        """
        # Initialize if needed
        if not self._initialized:
            await self.initialize()
        
        start_time = time.time()
        
        # Use provider's wealth management domain check method
        result = await self._provider.check_wealth_management_domain(
            text=text,
            context=context,
        )
        
        execution_time = time.time() - start_time
        
        is_allowed = result["allowed"]
        nemo_response = result.get("response", "")
        
        if is_allowed:
            return GuardrailResult(
                passed=True,
                guardrail_name=self.name,
                layer=self.layer,
                risk_score=0.0,
                message=nemo_response if nemo_response else None,
                metadata={
                    "domain": "Wealth Management",
                    "nemo_response": nemo_response,
                    "execution_time_ms": execution_time * 1000,
                    "details": result.get("details", {}),
                },
            )
        else:
            # Blocked by wealth management domain rules
            if self._mode == "warning":
                # Warning mode - pass but include warning
                return GuardrailResult(
                    passed=True,
                    guardrail_name=self.name,
                    layer=self.layer,
                    risk_score=0.5,
                    message=nemo_response if nemo_response else "Message may not be appropriate for wealth management domain (warning only)",
                    metadata={
                        "warning": "Wealth management domain policy may be violated",
                        "domain": "Wealth Management",
                        "nemo_response": nemo_response,
                        "execution_time_ms": execution_time * 1000,
                        "details": result.get("details", {}),
                    },
                )
            else:
                # Blocking mode - fail - use NeMo's response
                return GuardrailResult(
                    passed=False,
                    guardrail_name=self.name,
                    layer=self.layer,
                    risk_score=1.0,
                    message=nemo_response if nemo_response else self._refusal_message,
                    metadata={
                        "blocked_reason": "wealth_management_domain_policy_violation",
                        "domain": "Wealth Management",
                        "nemo_response": nemo_response,
                        "execution_time_ms": execution_time * 1000,
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
        if "mode" in config:
            self._mode = config["mode"]
        if "refusal_message" in config:
            self._refusal_message = config["refusal_message"]
