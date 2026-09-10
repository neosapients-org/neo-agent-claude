"""Gibberish detection guardrail for input.

This module provides guardrail protection against nonsensical or
gibberish inputs using LLM Guard's Gibberish scanner.
"""

import os
from pathlib import Path
from typing import Any, Dict, Optional

from ...core.models import GuardrailContext, GuardrailLayer, GuardrailResult
from ..base import GuardrailBase


class GibberishInputGuardrail(GuardrailBase):
    """Detect and block gibberish/nonsensical text in user input.

    Uses LLM Guard's Gibberish scanner to identify text that is
    completely nonsensical or poorly structured. Helps prevent
    confusion attacks and maintain platform integrity.

    Configuration:
        threshold: Detection threshold (0.0-1.0), default 0.5
        match_type: "full" or "sentence", default "full"
            - full: Check entire text at once
            - sentence: Check each sentence separately
        use_onnx: Whether to use ONNX runtime for faster inference, default True

    Example:
        guardrail = GibberishInputGuardrail({
            "threshold": 0.5,
            "match_type": "full"
        })
        result = await guardrail.check("asdf jkl; qwer uiop zxcv")
    """

    name = "gibberish_input"
    layer = GuardrailLayer.INPUT
    description = "Detect and block gibberish/nonsensical input"

    def __init__(self, config: Optional[Dict[str, Any]] = None, models_dir: Optional[str] = None) -> None:
        """Initialize the gibberish input guardrail.

        Args:
            config: Configuration dictionary
            models_dir: Optional directory containing local models
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
            from llm_guard.input_scanners import Gibberish
            from llm_guard.input_scanners.gibberish import DEFAULT_MODEL, MatchType

            model_config = None
            
            # Check if local models are available
            if self._models_dir:
                local_model_path = Path(self._models_dir) / "autonlp-Gibberish-Detector-492513457"
                
                if local_model_path.exists():
                    self.logger.info(
                        "gibberish_using_local_model",
                        model_path=str(local_model_path)
                    )
                    
                    # Configure DEFAULT_MODEL to use local path
                    DEFAULT_MODEL.path = str(local_model_path)
                    DEFAULT_MODEL.kwargs["local_files_only"] = True
                    model_config = DEFAULT_MODEL
                else:
                    self.logger.warning(
                        "gibberish_local_model_not_found",
                        expected_path=str(local_model_path),
                        message="Will download from HuggingFace (slower). Run: neo-guardrail download-models --scanners gibberish"
                    )

            match_type = (
                MatchType.FULL
                if self._match_type == "full"
                else MatchType.SENTENCE
            )

            # Initialize scanner with model config if available
            if model_config:
                self._scanner = Gibberish(
                    threshold=self._threshold,
                    match_type=match_type,
                    use_onnx=self._use_onnx,
                    model=model_config
                )
            else:
                self._scanner = Gibberish(
                    threshold=self._threshold,
                    match_type=match_type,
                    use_onnx=self._use_onnx,
                )

            self.logger.info(
                "gibberish_input_scanner_initialized",
                threshold=self._threshold,
                match_type=self._match_type,
                using_local_model=model_config is not None,
            )

        except ImportError:
            self.logger.warning(
                "llm_guard_not_installed",
                message="Install with: pip install neo-guardrail-hub[llm-guard]",
            )
            self._scanner = None

        await super().initialize()

    async def _check(
        self,
        text: str,
        context: Optional[GuardrailContext] = None,
    ) -> GuardrailResult:
        """Check text for gibberish content.

        Args:
            text: Text to check
            context: Optional context (not used)

        Returns:
            GuardrailResult indicating if gibberish was detected
        """
        if self._scanner is None:
            return self._fallback_check(text)

        try:
            sanitized_prompt, is_valid, risk_score = self._scanner.scan(text)

            # Normalize risk score to 0.0-1.0 range
            normalized_risk_score = max(0.0, min(1.0, risk_score))

            return GuardrailResult(
                passed=is_valid,
                guardrail_name=self.name,
                layer=self.layer,
                risk_score=normalized_risk_score,
                message=(
                    "Gibberish/nonsensical text detected"
                    if not is_valid
                    else None
                ),
                metadata={
                    "detection_method": "llm_guard",
                    "match_type": self._match_type,
                    "raw_risk_score": risk_score,
                },
            )

        except Exception as e:
            self.logger.error(
                "gibberish_detection_error",
                error=str(e),
            )
            return self._fallback_check(text)

    def _fallback_check(self, text: str) -> GuardrailResult:
        """Fallback gibberish detection using heuristics.

        Args:
            text: Text to check

        Returns:
            GuardrailResult with detection results
        """
        import re
        import string

        # Heuristics for detecting gibberish
        gibberish_score = 0.0
        checks_performed = 0

        # Check 1: Character diversity ratio
        if len(text) > 10:
            unique_chars = len(set(text.lower()))
            char_ratio = unique_chars / len(text)
            # Very low diversity (same char repeated) or very high (random chars)
            if char_ratio < 0.1 or char_ratio > 0.9:
                gibberish_score += 0.3
            checks_performed += 1

        # Check 2: Vowel/consonant ratio
        vowels = set('aeiouAEIOU')
        consonants = set('bcdfghjklmnpqrstvwxyzBCDFGHJKLMNPQRSTVWXYZ')
        vowel_count = sum(1 for c in text if c in vowels)
        consonant_count = sum(1 for c in text if c in consonants)

        if consonant_count > 0:
            vc_ratio = vowel_count / consonant_count
            # Normal English has ratio around 0.6-0.8
            if vc_ratio < 0.1 or vc_ratio > 2.0:
                gibberish_score += 0.2
            checks_performed += 1

        # Check 3: Repeated character patterns
        repeated_pattern = re.search(r'(.)\1{4,}', text)  # Same char 5+ times
        if repeated_pattern:
            gibberish_score += 0.3
            checks_performed += 1

        # Check 4: Random keyboard mashing patterns
        keyboard_patterns = [
            r'asdf', r'qwer', r'zxcv', r'hjkl', r'uiop',
            r'jkl;', r'fghj', r'vbnm', r'tyui', r'xcvb'
        ]
        for pattern in keyboard_patterns:
            if pattern in text.lower():
                gibberish_score += 0.2
                break
        checks_performed += 1

        # Check 5: Excessive special characters
        special_char_ratio = sum(1 for c in text if c in string.punctuation) / max(len(text), 1)
        if special_char_ratio > 0.4:
            gibberish_score += 0.2
            checks_performed += 1

        # Check 6: Very short words ratio (potential random text)
        words = text.split()
        if len(words) > 3:
            short_words = sum(1 for w in words if len(w) <= 2)
            short_ratio = short_words / len(words)
            if short_ratio > 0.6:
                gibberish_score += 0.2
            checks_performed += 1

        # Normalize score
        final_score = min(gibberish_score, 1.0)
        is_gibberish = final_score >= self._threshold

        return GuardrailResult(
            passed=not is_gibberish,
            guardrail_name=self.name,
            layer=self.layer,
            risk_score=final_score,
            message=(
                "Potential gibberish detected"
                if is_gibberish
                else None
            ),
            metadata={
                "detection_method": "heuristics",
                "gibberish_score": final_score,
                "checks_performed": checks_performed,
            },
        )

    async def cleanup(self) -> None:
        """Clean up scanner resources."""
        self._scanner = None
        await super().cleanup()
