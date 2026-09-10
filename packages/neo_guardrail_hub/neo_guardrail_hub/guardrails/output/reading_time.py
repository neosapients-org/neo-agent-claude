"""Reading time guardrail for output.

This module provides guardrail protection for controlling the length
of LLM outputs based on estimated reading time.
"""

from typing import Any, Dict, Optional

from ...core.models import GuardrailContext, GuardrailLayer, GuardrailResult
from ..base import GuardrailBase, extract_prompt_from_context


class ReadingTimeOutputGuardrail(GuardrailBase):
    """Control LLM output length based on reading time.

    Uses LLM Guard's ReadingTime output scanner to estimate reading time
    and optionally truncate outputs that exceed the limit.

    Configuration:
        max_time: Maximum reading time in minutes (default 5)
        truncate: Whether to truncate long outputs (default False)
        words_per_minute: Average reading speed (default 200)

    Example:
        guardrail = ReadingTimeOutputGuardrail({
            "max_time": 2,
            "truncate": True
        })
        result = await guardrail.check(
            "Very long output text...",
            context={"prompt": "Give me a brief summary"}
        )
    """

    name = "reading_time"
    layer = GuardrailLayer.OUTPUT
    description = "Control LLM output length based on reading time"

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize the reading time guardrail.

        Args:
            config: Configuration dictionary
        """
        super().__init__(config)
        self._scanner = None
        self._max_time = self._config.get("max_time", 5)
        self._truncate = self._config.get("truncate", False)
        self._words_per_minute = self._config.get("words_per_minute", 200)

    async def initialize(self) -> None:
        """Initialize the LLM Guard scanner."""
        if self._initialized:
            return

        try:
            from llm_guard.output_scanners import ReadingTime

            self._scanner = ReadingTime(
                max_time=self._max_time,
                truncate=self._truncate,
            )

            self.logger.info(
                "reading_time_scanner_initialized",
                max_time=self._max_time,
                truncate=self._truncate,
            )

        except ImportError as e:
            self.logger.warning(
                "llm_guard_not_installed",
                message="Install with: pip install neo-guardrail-hub[llm-guard]",
                error=str(e),
            )
            self._scanner = None
        except Exception as e:
            self.logger.warning(
                "reading_time_scanner_init_failed",
                message="Failed to initialize ReadingTime scanner, will use fallback",
                error=str(e),
            )
            self._scanner = None

        await super().initialize()

    async def _check(
        self,
        text: str,
        context: Optional[GuardrailContext] = None,
    ) -> GuardrailResult:
        """Check the reading time of the output.

        Args:
            text: LLM output to check
            context: Optional context with prompt

        Returns:
            GuardrailResult indicating if reading time is acceptable
        """
        prompt = extract_prompt_from_context(context)

        if self._scanner is None:
            return self._fallback_check(text)

        try:
            sanitized_text, is_valid, risk_score = self._scanner.scan(prompt, text)

            normalized_risk = max(0.0, min(1.0, risk_score))
            passed = is_valid

            if not passed:
                self.logger.warning(
                    "reading_time_exceeded",
                    max_time=self._max_time,
                )

            return GuardrailResult(
                passed=passed,
                guardrail_name=self.name,
                layer=self.layer,
                risk_score=normalized_risk,
                message=(
                    f"Output exceeds {self._max_time} minute reading time"
                    if not passed else None
                ),
                sanitized_text=sanitized_text if self._truncate and sanitized_text != text else None,
                metadata={
                    "within_time_limit": passed,
                    "max_time_minutes": self._max_time,
                    "truncated": self._truncate and sanitized_text != text,
                    "detection_method": "llm_guard_reading_time",
                },
            )

        except Exception as e:
            self.logger.error("reading_time_check_error", error=str(e))
            return self._fallback_check(text)

    def _fallback_check(self, text: str) -> GuardrailResult:
        """Fallback check using word count.

        Args:
            text: Text to check

        Returns:
            GuardrailResult based on estimated reading time
        """
        word_count = len(text.split())
        estimated_minutes = word_count / self._words_per_minute
        
        passed = estimated_minutes <= self._max_time
        
        # Risk increases with how much over the limit
        if passed:
            risk_score = 0.0
        else:
            excess_ratio = (estimated_minutes - self._max_time) / self._max_time
            risk_score = min(excess_ratio, 1.0)

        truncated_text = None
        if not passed and self._truncate:
            max_words = int(self._max_time * self._words_per_minute)
            words = text.split()[:max_words]
            truncated_text = " ".join(words) + "..."

        return GuardrailResult(
            passed=passed,
            guardrail_name=self.name,
            layer=self.layer,
            risk_score=risk_score,
            message=(
                f"Estimated reading time: {estimated_minutes:.1f} min (max: {self._max_time})"
                if not passed else None
            ),
            sanitized_text=truncated_text,
            metadata={
                "within_time_limit": passed,
                "estimated_minutes": round(estimated_minutes, 2),
                "max_time_minutes": self._max_time,
                "word_count": word_count,
                "truncated": truncated_text is not None,
                "detection_method": "word_count_fallback",
            },
        )
