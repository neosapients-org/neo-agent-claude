"""Language consistency guardrail for output.

This module provides guardrail protection for ensuring that LLM outputs
are in the same language as the input prompt.
"""

from typing import Any, Dict, Optional

from ...core.models import GuardrailContext, GuardrailLayer, GuardrailResult
from ..base import GuardrailBase, extract_prompt_from_context


class LanguageSameOutputGuardrail(GuardrailBase):
    """Check that LLM output is in the same language as the input.

    Uses LLM Guard's LanguageSame output scanner with the
    papluca/xlm-roberta-base-language-detection model.

    Configuration:
        threshold: Detection confidence threshold (0.0-1.0, default 0.5)
        use_onnx: Whether to use ONNX runtime (default True)

    Example:
        guardrail = LanguageSameOutputGuardrail({
            "threshold": 0.5
        })
        result = await guardrail.check(
            "Esta es una respuesta en español.",  # Spanish output
            context={"prompt": "Tell me about weather"}  # English prompt
        )
    """

    name = "language_same"
    layer = GuardrailLayer.OUTPUT
    description = "Check that output is in the same language as input"

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize the language consistency guardrail.

        Args:
            config: Configuration dictionary
        """
        super().__init__(config)
        self._scanner = None
        self._use_onnx = self._config.get("use_onnx", True)

    async def initialize(self) -> None:
        """Initialize the LLM Guard scanner."""
        if self._initialized:
            return

        try:
            from llm_guard.output_scanners import LanguageSame

            self._scanner = LanguageSame(
                threshold=self._threshold,
                use_onnx=self._use_onnx,
            )

            self.logger.info(
                "language_same_scanner_initialized",
                threshold=self._threshold,
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
                "language_same_scanner_init_failed",
                message="Failed to initialize LanguageSame scanner, will use fallback",
                error=str(e),
            )
            self._scanner = None

        await super().initialize()

    async def _check(
        self,
        text: str,
        context: Optional[GuardrailContext] = None,
    ) -> GuardrailResult:
        """Check language consistency between input and output.

        Args:
            text: LLM output to check
            context: Context containing the original prompt

        Returns:
            GuardrailResult indicating if languages match
        """
        prompt = extract_prompt_from_context(context)

        if not prompt:
            return GuardrailResult(
                passed=True,
                guardrail_name=self.name,
                layer=self.layer,
                risk_score=0.0,
                message="No prompt provided, skipping language consistency check",
                metadata={"skipped": True, "reason": "no_prompt"},
            )

        if self._scanner is None:
            return self._fallback_check(prompt, text)

        try:
            sanitized_text, is_valid, risk_score = self._scanner.scan(prompt, text)

            normalized_risk = max(0.0, min(1.0, risk_score))
            passed = is_valid

            if not passed:
                self.logger.warning(
                    "language_mismatch_detected",
                    risk_score=normalized_risk,
                )

            return GuardrailResult(
                passed=passed,
                guardrail_name=self.name,
                layer=self.layer,
                risk_score=normalized_risk,
                message="Output language differs from input language" if not passed else None,
                metadata={
                    "languages_match": passed,
                    "detection_method": "xlm_roberta",
                },
            )

        except Exception as e:
            self.logger.error("language_same_check_error", error=str(e))
            return self._fallback_check(prompt, text)

    def _fallback_check(self, prompt: str, output: str) -> GuardrailResult:
        """Fallback check using ASCII ratio comparison.

        Args:
            prompt: Original prompt
            output: LLM output

        Returns:
            GuardrailResult based on character set similarity
        """
        # Calculate ASCII ratio for both
        prompt_ascii = sum(ord(c) < 128 for c in prompt) / max(len(prompt), 1)
        output_ascii = sum(ord(c) < 128 for c in output) / max(len(output), 1)

        # If both have similar ASCII ratios, likely same language family
        ascii_diff = abs(prompt_ascii - output_ascii)
        
        passed = ascii_diff < 0.3  # Allow 30% difference
        risk_score = min(ascii_diff * 2, 1.0)

        return GuardrailResult(
            passed=passed,
            guardrail_name=self.name,
            layer=self.layer,
            risk_score=risk_score,
            message="Possible language mismatch (fallback)" if not passed else None,
            metadata={
                "languages_match": passed,
                "prompt_ascii_ratio": prompt_ascii,
                "output_ascii_ratio": output_ascii,
                "detection_method": "ascii_ratio_fallback",
            },
        )
