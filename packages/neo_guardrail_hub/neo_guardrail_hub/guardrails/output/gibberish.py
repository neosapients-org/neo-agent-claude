"""Gibberish detection guardrail for output.

This module provides guardrail protection for detecting nonsensical
or gibberish content in LLM outputs.
"""

import os
import re
from pathlib import Path
from typing import Any, Dict, Optional

from ...core.models import GuardrailContext, GuardrailLayer, GuardrailResult
from ..base import GuardrailBase, extract_prompt_from_context


class GibberishOutputGuardrail(GuardrailBase):
    """Detect gibberish or nonsensical content in LLM outputs.

    Uses LLM Guard's Gibberish output scanner with the
    madhurjindal/autonlp-Gibberish-Detector-492513457 model.

    Configuration:
        threshold: Gibberish detection threshold (0.0-1.0, default 0.5)
        match_type: Match type - "full" or "sentence" (default "full")
        use_onnx: Whether to use ONNX runtime (default True)

    Example:
        guardrail = GibberishOutputGuardrail({
            "threshold": 0.5,
            "match_type": "full"
        })
        result = await guardrail.check(
            "asdf jkl qwerty zxcv random nonsense",
            context={"prompt": "Explain something"}
        )
    """

    name = "gibberish_output"
    layer = GuardrailLayer.OUTPUT
    description = "Detect gibberish or nonsensical content in LLM output"

    def __init__(self, config: Optional[Dict[str, Any]] = None, models_dir: Optional[str] = None) -> None:
        """Initialize the gibberish detection guardrail.

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
            from llm_guard.output_scanners import Gibberish
            from llm_guard.output_scanners.gibberish import MatchType
            from llm_guard.input_scanners.gibberish import DEFAULT_MODEL

            match_type = (
                MatchType.FULL if self._match_type.lower() == "full"
                else MatchType.SENTENCE
            )

            # Configure local model if available
            model_config = None
            if self._models_dir:
                local_model_path = Path(self._models_dir) / "autonlp-Gibberish-Detector-492513457"
                if local_model_path.exists():
                    self.logger.info(
                        "gibberish_output_using_local_model",
                        model_path=str(local_model_path)
                    )
                    DEFAULT_MODEL.path = str(local_model_path)
                    DEFAULT_MODEL.kwargs["local_files_only"] = True
                    model_config = DEFAULT_MODEL

            if model_config:
                self._scanner = Gibberish(
                    threshold=self._threshold,
                    match_type=match_type,
                    use_onnx=self._use_onnx,
                    model=model_config,
                )
            else:
                self._scanner = Gibberish(
                    threshold=self._threshold,
                    match_type=match_type,
                    use_onnx=self._use_onnx,
                )

            self.logger.info(
                "gibberish_output_scanner_initialized",
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
                "gibberish_output_scanner_init_failed",
                message="Failed to initialize Gibberish scanner, will use fallback",
                error=str(e),
            )
            self._scanner = None

        await super().initialize()

    async def _check(
        self,
        text: str,
        context: Optional[GuardrailContext] = None,
    ) -> GuardrailResult:
        """Check for gibberish in the output.

        Args:
            text: LLM output to check
            context: Optional context with prompt

        Returns:
            GuardrailResult indicating if gibberish was detected
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
                    "gibberish_detected_in_output",
                    risk_score=normalized_risk,
                    threshold=self._threshold,
                )

            return GuardrailResult(
                passed=passed,
                guardrail_name=self.name,
                layer=self.layer,
                risk_score=normalized_risk,
                message="Gibberish content detected in output" if not passed else None,
                metadata={
                    "is_gibberish": not passed,
                    "gibberish_score": normalized_risk,
                    "threshold": self._threshold,
                    "detection_method": "gibberish_detector_model",
                },
            )

        except Exception as e:
            self.logger.error("gibberish_output_check_error", error=str(e))
            return self._fallback_check(text)

    def _fallback_check(self, text: str) -> GuardrailResult:
        """Fallback check using heuristics.

        Args:
            text: Text to check

        Returns:
            GuardrailResult based on heuristic analysis
        """
        if not text.strip():
            return GuardrailResult(
                passed=True,
                guardrail_name=self.name,
                layer=self.layer,
                risk_score=0.0,
                message="Empty text",
                metadata={"detection_method": "heuristic_fallback"},
            )

        # Heuristics for gibberish detection
        words = text.split()
        
        # Check for excessive non-alphabetic content
        alpha_ratio = sum(c.isalpha() for c in text) / max(len(text), 1)
        
        # Check for vowel ratio (gibberish often lacks proper vowel distribution)
        vowels = sum(c.lower() in 'aeiou' for c in text if c.isalpha())
        letters = sum(c.isalpha() for c in text)
        vowel_ratio = vowels / max(letters, 1)
        
        # Check for repeated characters
        repeated_chars = len(re.findall(r'(.)\1{3,}', text))
        
        # Check for very long "words"
        long_words = sum(1 for w in words if len(w) > 25)
        
        # Calculate risk score
        risk_factors = []
        
        if alpha_ratio < 0.5:
            risk_factors.append(0.3)
        if vowel_ratio < 0.15 or vowel_ratio > 0.7:
            risk_factors.append(0.3)
        if repeated_chars > 2:
            risk_factors.append(0.2)
        if long_words > 2:
            risk_factors.append(0.2)
            
        risk_score = min(sum(risk_factors), 0.9)
        passed = risk_score < self._threshold

        return GuardrailResult(
            passed=passed,
            guardrail_name=self.name,
            layer=self.layer,
            risk_score=risk_score,
            message="Potential gibberish detected (fallback)" if not passed else None,
            metadata={
                "is_gibberish": not passed,
                "alpha_ratio": alpha_ratio,
                "vowel_ratio": vowel_ratio,
                "detection_method": "heuristic_fallback",
            },
        )
