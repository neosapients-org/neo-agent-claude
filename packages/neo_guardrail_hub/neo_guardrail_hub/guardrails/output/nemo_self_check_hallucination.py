"""NeMo self-check hallucination guardrail.

This module provides guardrail protection using NeMo Guardrails'
self_check_hallucination flow for LLM-based hallucination detection.

The self_check_hallucination flow detects when the LLM generates false claims
or "hallucinations" by comparing the response against alternative generations.
This is useful when there's no external knowledge base to fact-check against.

Reference:
- https://docs.nvidia.com/nemo/guardrails/latest/user-guides/guardrails-library.html#hallucination-detection
- https://arxiv.org/abs/2303.08896 (SelfCheckGPT paper)
"""

import time
from typing import Any, Dict, List, Optional, TYPE_CHECKING, Union

from ...core.models import GuardrailContext, GuardrailLayer, GuardrailResult
from ..base import GuardrailBase

if TYPE_CHECKING:
    from ...providers.nemo.provider import NeMoProvider


class NeMoSelfCheckHallucinationGuardrail(GuardrailBase):
    """
    LLM-based hallucination detection using NeMo's self_check_hallucination.
    
    This guardrail uses NeMo Guardrails' self_check_hallucination flow to detect
    when the LLM generates false claims (hallucinations). Unlike fact-checking,
    this works without external evidence by comparing against alternative
    generations from the same LLM.
    
    How it works:
    1. Takes bot response
    2. Samples additional responses from the LLM for the same query
    3. Uses NeMo's self_check_hallucination action to check consistency
    4. LLM determines if original response agrees with alternatives
    5. Returns hallucination score between 0.0 and 1.0
    6. Blocks if score < threshold (default 0.5)
    
    Implementation based on SelfCheckGPT paper:
    - Generates multiple alternative responses for the same query
    - Compares original response against alternatives for consistency
    - Inconsistent claims across samples indicate hallucination
    
    Use cases:
    - General LLM outputs - detect fabricated information
    - Question answering - catch made-up answers
    - People/facts queries - particularly prone to hallucination
    
    Modes:
    - Blocking: Block the response if hallucination detected
    - Warning: Allow response but add a warning message
    
    Configuration:
        threshold: Minimum consistency score to pass (default: 0.5)
        on_fail: Action when blocked (default: "block")
        mode: "blocking" or "warning" (default: "blocking")
        num_samples: Number of alternative generations (default: 2)
        provider: NeMoProvider instance (required, set via configure())
    
    Example:
        from neo_guardrail_hub.providers.nemo import NeMoProvider
        
        provider = NeMoProvider(config_path="./configs")
        await provider.initialize()
        
        guardrail = provider.get_guardrail("nemo_self_check_hallucination")
        
        # Check for hallucination in response
        result = await guardrail.check(
            "Albert Einstein was born in 1879 in Germany.",
            context=GuardrailContext(
                metadata={"user_input": "When was Einstein born?"}
            )
        )
        # result.passed = True (consistent across samples)
        
        result = await guardrail.check(
            "Einstein invented the telephone in 1920.",
            context=GuardrailContext(
                metadata={"user_input": "What did Einstein invent?"}
            )
        )
        # result.passed = False (likely hallucination)
    
    NeMo Configuration:
        In config.yml, output rails must include:
        
        rails:
          output:
            flows:
              - self check hallucination
        
        In prompts section (or prompts.yml):
        
        prompts:
          - task: self_check_hallucination
            content: |
              You are given a task to identify if the hypothesis is in agreement
              with the context below...
    """
    
    name = "nemo_self_check_hallucination"
    layer = GuardrailLayer.OUTPUT
    description = "LLM-based hallucination detection using NeMo's self_check_hallucination"
    
    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize the NeMo self-check hallucination guardrail.
        
        Args:
            config: Configuration dictionary including:
                - provider: NeMoProvider instance (required)
                - threshold: Consistency threshold (default: 0.5)
                - mode: "blocking" or "warning" (default: "blocking")
                - num_samples: Number of alternative generations (default: 2)
                - on_fail: Action when blocked ("block", "warn")
                - refusal_message: Custom refusal message
                - warning_message: Custom warning message for warning mode
        """
        super().__init__(config)
        
        self._provider: Optional["NeMoProvider"] = self._config.get("provider")
        
        # Consistency threshold - responses below this are blocked/warned
        self._threshold = self._config.get("threshold", 0.5)
        
        # Mode: "blocking" or "warning"
        self._mode = self._config.get("mode", "blocking")
        
        # Number of alternative samples to generate
        self._num_samples = self._config.get("num_samples", 2)
        
        # Messages for blocked/warned responses
        self._refusal_message = self._config.get(
            "refusal_message",
            "I don't have reliable information to answer that question."
        )
        self._warning_message = self._config.get(
            "warning_message",
            "Note: The previous answer may not be fully accurate and should be verified."
        )
    
    async def initialize(self) -> None:
        """Initialize the guardrail and ensure provider is ready."""
        if self._initialized:
            return
            
        if self._provider is None:
            raise ValueError(
                "NeMoSelfCheckHallucinationGuardrail requires a NeMoProvider. "
                "Get this guardrail from provider.get_guardrail('nemo_self_check_hallucination')"
            )
        
        # Ensure provider is initialized
        if not self._provider._initialized:
            await self._provider.initialize()
        
        self._initialized = True
        self.logger.info("nemo_self_check_hallucination_guardrail_initialized")
    
    async def _check(
        self,
        text: str,
        context: Optional[GuardrailContext] = None,
    ) -> GuardrailResult:
        """Check if bot output contains hallucinations.
        
        Args:
            text: Bot response to validate
            context: Context containing user_input for generating alternatives
            
        Returns:
            GuardrailResult with:
                - passed: True if not hallucinating, False if detected
                - risk_score: 1.0 - consistency (higher = more risk)
                - message: Refusal/warning message if blocked/warned
        """
        # Initialize if needed
        if not self._initialized:
            await self.initialize()
        
        # Extract user input from context
        user_input = self._extract_user_input(context)
        
        if not user_input:
            # If no user input, we can't generate alternatives
            self.logger.warning(
                "nemo_self_check_hallucination_no_user_input",
                message="No user input provided for hallucination check, skipping"
            )
            return GuardrailResult(
                passed=True,
                guardrail_name=self.name,
                layer=self.layer,
                risk_score=0.0,
                message="No user input provided for hallucination check",
                details={
                    "skipped": True,
                    "reason": "no_user_input",
                },
            )
        
        # Use provider to check for hallucination
        result = await self._provider.check_hallucination(
            response=text,
            user_input=user_input,
            num_samples=self._num_samples,
            context=context,
        )
        
        consistency = result.get("consistency", 0.0)
        is_hallucination = consistency < self._threshold
        
        if not is_hallucination:
            return GuardrailResult(
                passed=True,
                guardrail_name=self.name,
                layer=self.layer,
                risk_score=1.0 - consistency,
                message="Response appears consistent and not hallucinated",
                details={
                    "consistency": consistency,
                    "threshold": self._threshold,
                    "mode": self._mode,
                    "num_samples": self._num_samples,
                },
            )
        else:
            # Handle based on mode
            if self._mode == "warning":
                # Warning mode - pass but include warning
                return GuardrailResult(
                    passed=True,  # Still passes in warning mode
                    guardrail_name=self.name,
                    layer=self.layer,
                    risk_score=1.0 - consistency,
                    message=self._warning_message,
                    details={
                        "consistency": consistency,
                        "threshold": self._threshold,
                        "mode": self._mode,
                        "num_samples": self._num_samples,
                        "hallucination_detected": True,
                        "warning": self._warning_message,
                    },
                )
            else:
                # Blocking mode - fail the check
                return GuardrailResult(
                    passed=False,
                    guardrail_name=self.name,
                    layer=self.layer,
                    risk_score=1.0 - consistency,
                    message=self._refusal_message,
                    details={
                        "consistency": consistency,
                        "threshold": self._threshold,
                        "mode": self._mode,
                        "num_samples": self._num_samples,
                        "blocked_reason": "hallucination_detected",
                    },
                )
    
    def _extract_user_input(
        self,
        context: Optional[GuardrailContext],
    ) -> str:
        """Extract user input from context.
        
        Looks for user input in various context fields:
        - context.metadata["user_input"]
        - context.metadata["user_message"]
        - context.metadata["query"]
        - context.metadata["question"]
        
        Args:
            context: Optional guardrail context
            
        Returns:
            User input string or empty string if not found
        """
        if not context:
            return ""
        
        metadata = context.metadata or {}
        
        # Check various possible field names
        input_fields = [
            "user_input",
            "user_message",
            "query",
            "question",
            "prompt",
            "input",
        ]
        
        for field in input_fields:
            if field in metadata:
                value = metadata[field]
                if isinstance(value, str):
                    return value
        
        return ""
    
    @property
    def threshold(self) -> float:
        """Get the consistency threshold."""
        return self._threshold
    
    @threshold.setter
    def threshold(self, value: float) -> None:
        """Set the consistency threshold."""
        if not 0.0 <= value <= 1.0:
            raise ValueError("Threshold must be between 0.0 and 1.0")
        self._threshold = value
    
    @property
    def mode(self) -> str:
        """Get the detection mode."""
        return self._mode
    
    @mode.setter
    def mode(self, value: str) -> None:
        """Set the detection mode."""
        if value not in ("blocking", "warning"):
            raise ValueError("Mode must be 'blocking' or 'warning'")
        self._mode = value
