"""Input length validation guardrail.

This module provides guardrail protection for validating
input text length to prevent excessively short or long inputs.
"""

from typing import Any, Dict, Optional

from ...core.models import GuardrailContext, GuardrailLayer, GuardrailResult
from ..base import GuardrailBase


class InputLengthGuardrail(GuardrailBase):
    """Validate input text length.

    Ensures user input meets minimum and maximum length requirements.
    Supports counting by characters, words, or estimated tokens.

    Configuration:
        min_length: Minimum allowed length, default 1
        max_length: Maximum allowed length, default 10000
        count_mode: How to count length - "chars", "words", or "tokens"
            - chars: Count characters
            - words: Count words (split by whitespace)
            - tokens: Estimate tokens (chars / 4 approximation)
        on_fail: Action on failure - "block" or "warn", default "block"

    Example:
        guardrail = InputLengthGuardrail({
            "min_length": 5,
            "max_length": 5000,
            "count_mode": "chars"
        })
        result = await guardrail.check("Hello world!")
    """

    name = "input_length"
    layer = GuardrailLayer.INPUT
    description = "Validate input text length constraints"

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize the input length guardrail.

        Args:
            config: Configuration dictionary
        """
        super().__init__(config)
        self._min_length = self._config.get("min_length", 1)
        self._max_length = self._config.get("max_length", 10000)
        self._count_mode = self._config.get("count_mode", "chars")

    async def initialize(self) -> None:
        """Initialize the guardrail."""
        if self._initialized:
            return

        self.logger.info(
            "input_length_guardrail_initialized",
            min_length=self._min_length,
            max_length=self._max_length,
            count_mode=self._count_mode,
        )

        await super().initialize()

    def _count_length(self, text: str) -> int:
        """Count the length of text based on count_mode.

        Args:
            text: Text to measure

        Returns:
            Length count based on configured mode
        """
        if self._count_mode == "words":
            return len(text.split())
        elif self._count_mode == "tokens":
            # Approximate token count (chars / 4 is a common estimate)
            return len(text) // 4
        else:  # chars (default)
            return len(text)

    async def _check(
        self,
        text: str,
        context: Optional[GuardrailContext] = None,
    ) -> GuardrailResult:
        """Check text length against configured limits.

        Args:
            text: Text to check
            context: Optional context (not used)

        Returns:
            GuardrailResult indicating if length is valid
        """
        self.logger.debug("initializing_guardrail", guardrail=self.name)

        text_length = self._count_length(text)
        char_length = len(text)
        word_count = len(text.split())

        # Check minimum length
        if text_length < self._min_length:
            self.logger.warning(
                "input_too_short",
                length=text_length,
                min_length=self._min_length,
                count_mode=self._count_mode,
            )
            return GuardrailResult(
                passed=False,
                guardrail_name=self.name,
                layer=self.layer,
                risk_score=0.5,
                message=f"Input too short: {text_length} {self._count_mode} < {self._min_length} minimum",
                metadata={
                    "length": text_length,
                    "char_length": char_length,
                    "word_count": word_count,
                    "min_length": self._min_length,
                    "max_length": self._max_length,
                    "count_mode": self._count_mode,
                    "violation": "too_short",
                },
            )

        # Check maximum length
        if text_length > self._max_length:
            self.logger.warning(
                "input_too_long",
                length=text_length,
                max_length=self._max_length,
                count_mode=self._count_mode,
            )
            # Calculate risk score based on how much over the limit
            overage_ratio = text_length / self._max_length
            risk_score = min(0.5 + (overage_ratio - 1) * 0.5, 1.0)

            return GuardrailResult(
                passed=False,
                guardrail_name=self.name,
                layer=self.layer,
                risk_score=risk_score,
                message=f"Input too long: {text_length} {self._count_mode} > {self._max_length} maximum",
                metadata={
                    "length": text_length,
                    "char_length": char_length,
                    "word_count": word_count,
                    "min_length": self._min_length,
                    "max_length": self._max_length,
                    "count_mode": self._count_mode,
                    "violation": "too_long",
                    "overage": text_length - self._max_length,
                },
            )

        # Length is valid
        self.logger.debug(
            "input_length_valid",
            length=text_length,
            count_mode=self._count_mode,
        )

        return GuardrailResult(
            passed=True,
            guardrail_name=self.name,
            layer=self.layer,
            risk_score=0.0,
            message=None,
            metadata={
                "length": text_length,
                "char_length": char_length,
                "word_count": word_count,
                "min_length": self._min_length,
                "max_length": self._max_length,
                "count_mode": self._count_mode,
            },
        )
