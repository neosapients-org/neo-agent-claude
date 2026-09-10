"""Base guardrail class with common functionality.

This module provides a concrete base class that guardrail
implementations can inherit from to get common functionality.
"""

import time
from typing import Any, Dict, Optional, Union

from ..core.interfaces import BaseGuardrail
from ..core.models import GuardrailContext, GuardrailLayer, GuardrailResult
from ..utils.logging import get_logger


def extract_prompt_from_context(context: Optional[Union[GuardrailContext, Dict[str, Any]]]) -> str:
    """Extract prompt from context for output guardrails.
    
    LLM Guard output scanners require both the original prompt and 
    the model output. This helper extracts the prompt from various
    context formats.
    
    Args:
        context: Either a GuardrailContext object, a dict, or None
        
    Returns:
        The extracted prompt string, or empty string if not found
    """
    if context is None:
        return ""
    
    # Handle dict context (backwards compatibility)
    if isinstance(context, dict):
        return context.get("prompt", context.get("input", ""))
    
    # Handle GuardrailContext object
    if isinstance(context, GuardrailContext):
        # Check for prompt as extra attribute (extra='allow')
        if hasattr(context, "prompt") and context.prompt:
            return context.prompt
        
        # Check metadata for prompt
        if context.metadata and context.metadata.get("prompt"):
            return context.metadata["prompt"]
        
        # Check metadata for input
        if context.metadata and context.metadata.get("input"):
            return context.metadata["input"]
        
        # Fallback: get last user message from conversation history
        if context.conversation_history:
            for msg in reversed(context.conversation_history):
                if msg.get("role") == "user":
                    return msg.get("content", "")
    
    return ""


class GuardrailBase(BaseGuardrail):
    """Concrete base class for guardrail implementations.

    Provides common functionality like timing, logging, and error handling.
    Guardrail implementations should inherit from this class and
    implement the _check method.

    Example:
        class MyGuardrail(GuardrailBase):
            name = "my_guardrail"
            layer = GuardrailLayer.INPUT

            async def _check(self, text, context):
                # Your check logic here
                return GuardrailResult(...)
    """

    name: str = "base"
    layer: GuardrailLayer = GuardrailLayer.INPUT
    description: str = "Base guardrail implementation"

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize the guardrail.

        Args:
            config: Optional configuration dictionary
        """
        super().__init__(config)
        self.logger = get_logger(self.__class__.__name__)
        self._on_fail = self._config.get("on_fail", "block")

    async def check(
        self,
        text: str,
        context: Optional[GuardrailContext] = None,
    ) -> GuardrailResult:
        """Execute the guardrail check with timing and error handling.

        This method wraps the _check method with:
        - Timing measurement
        - Error handling
        - Logging

        Args:
            text: Text to check
            context: Optional context

        Returns:
            GuardrailResult with timing and metadata
        """
        start_time = time.monotonic()

        try:
            # Initialize if needed
            if not self._initialized:
                await self.initialize()

            # Execute the actual check
            result = await self._check(text, context)

            # Calculate latency
            latency_ms = (time.monotonic() - start_time) * 1000
            result.latency_ms = latency_ms

            # Add on_fail to metadata for aggregation
            result.metadata["on_fail"] = self._on_fail

            # Log the result
            self.logger.info(
                "guardrail_check_complete",
                guardrail=self.name,
                passed=result.passed,
                risk_score=result.risk_score,
                latency_ms=round(latency_ms, 2),
            )

            return result

        except Exception as e:
            latency_ms = (time.monotonic() - start_time) * 1000

            self.logger.error(
                "guardrail_check_error",
                guardrail=self.name,
                error=str(e),
                error_type=type(e).__name__,
                latency_ms=round(latency_ms, 2),
            )

            return GuardrailResult(
                passed=False,
                guardrail_name=self.name,
                layer=self.layer,
                risk_score=1.0,
                message=f"Guardrail error: {str(e)}",
                latency_ms=latency_ms,
                metadata={
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "on_fail": self._on_fail,
                },
            )

    async def _check(
        self,
        text: str,
        context: Optional[GuardrailContext] = None,
    ) -> GuardrailResult:
        """Implement the actual guardrail check logic.

        Override this method in subclasses to implement
        the specific guardrail logic.

        Args:
            text: Text to check
            context: Optional context

        Returns:
            GuardrailResult with check results
        """
        # Default implementation passes everything
        return GuardrailResult(
            passed=True,
            guardrail_name=self.name,
            layer=self.layer,
            risk_score=0.0,
            message="No check implemented",
        )

    def configure(self, config: Dict[str, Any]) -> None:
        """Configure the guardrail with settings.

        Args:
            config: Configuration dictionary
        """
        super().configure(config)
        self._on_fail = config.get("on_fail", self._on_fail)

    async def initialize(self) -> None:
        """Initialize resources needed by the guardrail."""
        self.logger.debug("initializing_guardrail", guardrail=self.name)
        await super().initialize()

    async def cleanup(self) -> None:
        """Clean up resources used by the guardrail."""
        self.logger.debug("cleaning_up_guardrail", guardrail=self.name)
        await super().cleanup()
