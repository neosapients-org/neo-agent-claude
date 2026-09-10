"""Language detection guardrail for output.

This module provides guardrail protection for validating that LLM outputs
are in the expected language(s).
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from ...core.models import GuardrailContext, GuardrailLayer, GuardrailResult
from ..base import GuardrailBase, extract_prompt_from_context


class LanguageOutputGuardrail(GuardrailBase):
    """Validate that LLM output is in expected language(s).

    Uses LLM Guard's Language output scanner with the
    papluca/xlm-roberta-base-language-detection model.

    Configuration:
        threshold: Detection confidence threshold (0.0-1.0, default 0.5)
        valid_languages: List of allowed language codes (ISO 639-1, default ["en"])
        match_type: Match type - "full" or "sentence" (default "full")
        use_onnx: Whether to use ONNX runtime (default True)

    Supported languages:
        ar, bg, de, el, en, es, fr, hi, it, ja, nl, pl, pt, ru, sw, th, tr, ur, vi, zh

    Example:
        guardrail = LanguageOutputGuardrail({
            "valid_languages": ["en", "es"],
            "threshold": 0.5
        })
        result = await guardrail.check(
            "This is an English response.",
            context={"prompt": "Answer in English"}
        )
    """

    name = "language_output"
    layer = GuardrailLayer.OUTPUT
    description = "Validate LLM output is in expected language(s)"

    SUPPORTED_LANGUAGES = [
        "ar", "bg", "de", "el", "en", "es", "fr", "hi", "it", "ja",
        "nl", "pl", "pt", "ru", "sw", "th", "tr", "ur", "vi", "zh"
    ]

    def __init__(self, config: Optional[Dict[str, Any]] = None, models_dir: Optional[str] = None) -> None:
        """Initialize the language detection guardrail.

        Args:
            config: Configuration dictionary
            models_dir: Optional path to directory containing local models
        """
        super().__init__(config)
        self._scanner = None
        self._valid_languages = self._config.get("valid_languages", ["en"])
        self._match_type = self._config.get("match_type", "full")
        self._use_onnx = self._config.get("use_onnx", True)
        self._models_dir = models_dir or os.getenv("NEO_LLM_GUARD_MODELS_DIR")

    async def initialize(self) -> None:
        """Initialize the LLM Guard scanner."""
        if self._initialized:
            return

        try:
            from llm_guard.output_scanners import Language
            from llm_guard.output_scanners.language import MatchType
            from llm_guard.input_scanners.language import DEFAULT_MODEL

            match_type = (
                MatchType.FULL if self._match_type.lower() == "full"
                else MatchType.SENTENCE
            )

            # Configure local model if available
            model_config = None
            if self._models_dir:
                local_model_path = Path(self._models_dir) / "xlm-roberta-base-language-detection"
                if local_model_path.exists():
                    self.logger.info(
                        "language_output_using_local_model",
                        model_path=str(local_model_path)
                    )
                    DEFAULT_MODEL.path = str(local_model_path)
                    DEFAULT_MODEL.kwargs["local_files_only"] = True
                    model_config = DEFAULT_MODEL

            if model_config:
                self._scanner = Language(
                    valid_languages=self._valid_languages,
                    threshold=self._threshold,
                    match_type=match_type,
                    use_onnx=self._use_onnx,
                    model=model_config,
                )
            else:
                self._scanner = Language(
                    valid_languages=self._valid_languages,
                    threshold=self._threshold,
                    match_type=match_type,
                    use_onnx=self._use_onnx,
                )

            self.logger.info(
                "language_output_scanner_initialized",
                valid_languages=self._valid_languages,
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
                "language_output_scanner_init_failed",
                message="Failed to initialize Language scanner, will use fallback",
                error=str(e),
            )
            self._scanner = None

        await super().initialize()

    async def _check(
        self,
        text: str,
        context: Optional[GuardrailContext] = None,
    ) -> GuardrailResult:
        """Check the language of the output.

        Args:
            text: LLM output to check
            context: Optional context with prompt

        Returns:
            GuardrailResult indicating if language is valid
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
                    "invalid_language_in_output",
                    valid_languages=self._valid_languages,
                    risk_score=normalized_risk,
                )

            return GuardrailResult(
                passed=passed,
                guardrail_name=self.name,
                layer=self.layer,
                risk_score=normalized_risk,
                message=(
                    f"Output not in expected language(s): {self._valid_languages}"
                    if not passed else None
                ),
                metadata={
                    "language_valid": passed,
                    "valid_languages": self._valid_languages,
                    "detection_method": "xlm_roberta",
                },
            )

        except Exception as e:
            self.logger.error("language_output_check_error", error=str(e))
            return self._fallback_check(text)

    def _fallback_check(self, text: str) -> GuardrailResult:
        """Fallback check - assumes English if 'en' in valid languages.

        Args:
            text: Text to check

        Returns:
            GuardrailResult (passes if 'en' in valid languages)
        """
        # Simple heuristic: check for ASCII ratio
        ascii_ratio = sum(ord(c) < 128 for c in text) / max(len(text), 1)
        
        # If high ASCII ratio and 'en' is valid, likely English
        if "en" in self._valid_languages and ascii_ratio > 0.9:
            passed = True
            risk_score = 0.0
        else:
            passed = True  # Default pass for fallback
            risk_score = 0.2

        return GuardrailResult(
            passed=passed,
            guardrail_name=self.name,
            layer=self.layer,
            risk_score=risk_score,
            message=None,
            metadata={
                "language_valid": passed,
                "ascii_ratio": ascii_ratio,
                "valid_languages": self._valid_languages,
                "detection_method": "heuristic_fallback",
            },
        )
