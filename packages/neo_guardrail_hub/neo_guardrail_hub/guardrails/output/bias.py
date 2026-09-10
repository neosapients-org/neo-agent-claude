"""Bias detection guardrail for output.

This module provides guardrail protection for detecting biased content
in LLM outputs using the valurank/distilroberta-bias model.
"""

import os
from pathlib import Path
from typing import Any, Dict, Optional

from ...core.models import GuardrailContext, GuardrailLayer, GuardrailResult
from ..base import GuardrailBase, extract_prompt_from_context


class BiasOutputGuardrail(GuardrailBase):
    """Detect biased content in LLM outputs.

    Uses LLM Guard's Bias output scanner with the valurank/distilroberta-bias
    model to identify biased statements in text.

    Configuration:
        threshold: Bias detection threshold (0.0-1.0, default 0.5)
        match_type: Match type - "full" or "sentence" (default "full")
        use_onnx: Whether to use ONNX runtime (default True)

    Example:
        guardrail = BiasOutputGuardrail({
            "threshold": 0.5,
            "match_type": "full"
        })
        result = await guardrail.check(
            "Men are naturally better at math than women.",
            context={"prompt": "Explain gender differences"}
        )
    """

    name = "bias_output"
    layer = GuardrailLayer.OUTPUT
    description = "Detect biased statements in LLM output"

    # Keywords associated with bias
    BIAS_KEYWORDS = [
        "always", "never", "all", "none", "everyone", "nobody",
        "obviously", "clearly", "naturally", "inherently",
        "superior", "inferior", "better than", "worse than",
    ]

    def __init__(self, config: Optional[Dict[str, Any]] = None, models_dir: Optional[str] = None) -> None:
        """Initialize the bias detection guardrail.

        Args:
            config: Configuration dictionary
            models_dir: Optional path to directory containing local models
        """
        super().__init__(config)
        self._scanner = None
        self._match_type = self._config.get("match_type", "full")
        self._use_onnx = self._config.get("use_onnx", True)
        self._models_dir = models_dir or os.getenv("NEO_LLM_GUARD_MODELS_DIR")

    async def initialize(self) -> None:
        """Initialize the LLM Guard scanner."""
        if self._initialized:
            return

        try:
            from llm_guard.output_scanners import Bias
            from llm_guard.output_scanners.bias import MatchType, DEFAULT_MODEL

            match_type = (
                MatchType.FULL if self._match_type.lower() == "full"
                else MatchType.SENTENCE
            )

            # Configure local model if available
            model_config = None
            if self._models_dir:
                local_model_path = Path(self._models_dir) / "distilroberta-bias"
                if local_model_path.exists():
                    self.logger.info(
                        "bias_output_using_local_model",
                        model_path=str(local_model_path)
                    )
                    DEFAULT_MODEL.path = str(local_model_path)
                    DEFAULT_MODEL.kwargs["local_files_only"] = True
                    model_config = DEFAULT_MODEL

            if model_config:
                self._scanner = Bias(
                    threshold=self._threshold,
                    match_type=match_type,
                    use_onnx=self._use_onnx,
                    model=model_config,
                )
            else:
                self._scanner = Bias(
                    threshold=self._threshold,
                    match_type=match_type,
                    use_onnx=self._use_onnx,
                )

            self.logger.info(
                "bias_output_scanner_initialized",
                threshold=self._threshold,
                match_type=self._match_type,
                using_local_model=model_config is not None,
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
                "bias_output_scanner_init_failed",
                message="Failed to initialize Bias scanner, will use fallback",
                error=str(e),
            )
            self._scanner = None

        await super().initialize()

    async def _check(
        self,
        text: str,
        context: Optional[GuardrailContext] = None,
    ) -> GuardrailResult:
        """Check for biased content in the output.

        Args:
            text: LLM output to check
            context: Optional context with prompt

        Returns:
            GuardrailResult indicating if bias was detected
        """
        prompt = extract_prompt_from_context(context)

        if self._scanner is None:
            return self._fallback_check(text)

        try:
            sanitized_text, is_valid, risk_score = self._scanner.scan(prompt, text)

            normalized_risk = max(0.0, min(1.0, risk_score))
            passed = is_valid and normalized_risk < self._threshold

            if not passed:
                self.logger.warning(
                    "bias_detected_in_output",
                    risk_score=normalized_risk,
                    threshold=self._threshold,
                )

            return GuardrailResult(
                passed=passed,
                guardrail_name=self.name,
                layer=self.layer,
                risk_score=normalized_risk,
                message="Biased content detected in output" if not passed else None,
                metadata={
                    "is_biased": not passed,
                    "bias_score": normalized_risk,
                    "threshold": self._threshold,
                    "detection_method": "distilroberta_bias",
                },
            )

        except Exception as e:
            self.logger.error("bias_check_error", error=str(e))
            return self._fallback_check(text)

    def _fallback_check(self, text: str) -> GuardrailResult:
        """Fallback check using keyword matching.

        Args:
            text: Text to check

        Returns:
            GuardrailResult based on keyword presence
        """
        text_lower = text.lower()
        found_keywords = [
            kw for kw in self.BIAS_KEYWORDS if kw in text_lower
        ]

        # Simple heuristic: more keywords = higher risk
        risk_score = min(len(found_keywords) * 0.15, 0.9)
        passed = risk_score < self._threshold

        return GuardrailResult(
            passed=passed,
            guardrail_name=self.name,
            layer=self.layer,
            risk_score=risk_score,
            message="Potential bias detected (fallback)" if not passed else None,
            metadata={
                "is_biased": not passed,
                "found_keywords": found_keywords,
                "detection_method": "keyword_fallback",
            },
        )
