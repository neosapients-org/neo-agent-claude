"""NeMo topical rail guardrail.

This module provides guardrail protection using NeMo Guardrails'
dialog rails for topic control - keeping the LLM on-topic.
"""

import time
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from ...core.models import GuardrailContext, GuardrailLayer, GuardrailResult
from ..base import GuardrailBase

if TYPE_CHECKING:
    from ...providers.nemo.provider import NeMoProvider


class NeMoTopicalRailGuardrail(GuardrailBase):
    """
    Dialog-based topic control using NeMo's topical rails.
    
    This guardrail uses NeMo Guardrails' dialog rails to enforce topic control,
    keeping conversations on allowed topics and blocking off-topic discussions.
    This is particularly useful for domain-specific agents (e.g., financial,
    healthcare) that should only discuss certain topics.
    
    How it works:
    1. Takes user message text
    2. Uses NeMo's generate() with dialog rails enabled
    3. NeMo's intent recognition matches against defined topic patterns
    4. If off-topic, NeMo blocks the request via dialog flows
    5. Returns GuardrailResult with pass/fail status
    
    Configuration:
        allowed_topics: List of topics the agent CAN discuss
        blocked_topics: List of topics the agent should NOT discuss
        mode: "blocking" (default) or "warning"
        provider: NeMoProvider instance (required, set via configure())
    
    Example:
        from neo_guardrail_hub.providers.nemo import NeMoProvider
        
        provider = NeMoProvider(config_path="./configs")
        await provider.initialize()
        
        guardrail = provider.get_guardrail("nemo_topical_rail")
        
        # On-topic question for financial agent
        result = await guardrail.check("What's my account balance?")
        # result.passed = True
        
        # Off-topic question
        result = await guardrail.check("Give me a stock tip")
        # result.passed = False (blocked by dialog rail)
    
    Reference:
        https://docs.nvidia.com/nemo/guardrails/latest/getting-started/6-topical-rails/README.html
    """
    
    name = "nemo_topical_rail"
    layer = GuardrailLayer.CONTEXT
    description = "Dialog-based topic control using NeMo's topical rails"
    
    # Default blocked topics for general safety
    DEFAULT_BLOCKED_TOPICS = [
        "stock_tips",
        "investment_advice",
        "guaranteed_returns",
        "medical_advice",
        "legal_advice",
    ]
    
    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize the NeMo topical rail guardrail.
        
        Args:
            config: Configuration dictionary including:
                - provider: NeMoProvider instance (required)
                - allowed_topics: List of allowed topics (optional)
                - blocked_topics: List of blocked topics (optional)
                - mode: "blocking" or "warning" (default: "blocking")
                - refusal_message: Custom refusal message for off-topic
        """
        super().__init__(config)
        
        self._provider: Optional["NeMoProvider"] = self._config.get("provider")
        
        # Topic configuration
        self._allowed_topics: List[str] = self._config.get("allowed_topics", [])
        self._blocked_topics: List[str] = self._config.get(
            "blocked_topics", 
            self.DEFAULT_BLOCKED_TOPICS
        )
        
        # Mode: blocking vs warning
        self._mode = self._config.get("mode", "blocking")
        
        # Refusal message for off-topic
        self._refusal_message = self._config.get(
            "refusal_message",
            "I'm not able to discuss that topic. Is there something else I can help with?"
        )
    
    @property
    def allowed_topics(self) -> List[str]:
        """Get the list of allowed topics."""
        return self._allowed_topics
    
    @property
    def blocked_topics(self) -> List[str]:
        """Get the list of blocked topics."""
        return self._blocked_topics
    
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
                "NeMoTopicalRailGuardrail requires a NeMoProvider. "
                "Get this guardrail from provider.get_guardrail('nemo_topical_rail')"
            )
        
        # Ensure provider is initialized
        if not self._provider._initialized:
            await self._provider.initialize()
        
        self._initialized = True
        self.logger.info("nemo_topical_rail_guardrail_initialized")
    
    async def _check(
        self,
        text: str,
        context: Optional[GuardrailContext] = None,
    ) -> GuardrailResult:
        """Check if user message is on-topic using NeMo's dialog rails.
        
        Args:
            text: User message to check for topic compliance
            context: Optional context with conversation history
            
        Returns:
            GuardrailResult with:
                - passed: True if on-topic, False if off-topic
                - risk_score: 1.0 if off-topic, 0.0 if on-topic
                - message: Refusal message if off-topic
        """
        # Initialize if needed
        if not self._initialized:
            await self.initialize()
        
        # Use provider to check topic
        result = await self._provider.check_topic(
            text=text,
            context=context,
        )
        
        is_on_topic = result["on_topic"]
        nemo_response = result.get("response", "")
        
        if is_on_topic:
            return GuardrailResult(
                passed=True,
                guardrail_name=self.name,
                layer=self.layer,
                risk_score=0.0,
                message="Message is on-topic",
                metadata={
                    "nemo_response": nemo_response,
                    "details": result.get("details", {}),
                },
            )
        else:
            # Off-topic detected
            if self._mode == "warning":
                # Warning mode - pass but include warning
                return GuardrailResult(
                    passed=True,
                    guardrail_name=self.name,
                    layer=self.layer,
                    risk_score=0.5,
                    message="Message is off-topic (warning only)",
                    metadata={
                        "warning": "Off-topic message detected",
                        "nemo_response": nemo_response,
                        "details": result.get("details", {}),
                    },
                )
            else:
                # Blocking mode - fail
                return GuardrailResult(
                    passed=False,
                    guardrail_name=self.name,
                    layer=self.layer,
                    risk_score=1.0,
                    message=nemo_response or self._refusal_message,
                    metadata={
                        "blocked_reason": "off_topic",
                        "nemo_response": nemo_response,
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
        if "allowed_topics" in config:
            self._allowed_topics = config["allowed_topics"]
        if "blocked_topics" in config:
            self._blocked_topics = config["blocked_topics"]
        if "mode" in config:
            self._mode = config["mode"]
        if "refusal_message" in config:
            self._refusal_message = config["refusal_message"]
