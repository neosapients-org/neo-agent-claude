"""Language detection guardrail for input.

This module provides guardrail protection for ensuring input
is in supported languages using LLM Guard's Language scanner.
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from ...core.models import GuardrailContext, GuardrailLayer, GuardrailResult
from ..base import GuardrailBase


# ISO 639-1 language codes supported by the model
SUPPORTED_LANGUAGES = [
    "ar",  # Arabic
    "bg",  # Bulgarian
    "de",  # German
    "el",  # Greek
    "en",  # English
    "es",  # Spanish
    "fr",  # French
    "hi",  # Hindi
    "it",  # Italian
    "ja",  # Japanese
    "nl",  # Dutch
    "pl",  # Polish
    "pt",  # Portuguese
    "ru",  # Russian
    "sw",  # Swahili
    "th",  # Thai
    "tr",  # Turkish
    "ur",  # Urdu
    "vi",  # Vietnamese
    "zh",  # Chinese
]


class LanguageInputGuardrail(GuardrailBase):
    """Detect and validate language of user input.

    Uses LLM Guard's Language scanner to identify the language
    of input text and validate it against a list of allowed languages.
    Useful for preventing multilingual jailbreak attacks.

    Configuration:
        valid_languages: List of allowed language codes (ISO 639-1), default ["en"]
        threshold: Detection confidence threshold (0.0-1.0), default 0.5
        match_type: "full" or "sentence", default "full"
        use_onnx: Whether to use ONNX runtime for faster inference, default True

    Example:
        guardrail = LanguageInputGuardrail({
            "valid_languages": ["en", "es", "fr"],
            "threshold": 0.5
        })
        result = await guardrail.check("Bonjour le monde!")
    """

    name = "language_input"
    layer = GuardrailLayer.INPUT
    description = "Detect and validate input language"

    def __init__(self, config: Optional[Dict[str, Any]] = None, models_dir: Optional[str] = None) -> None:
        """Initialize the language input guardrail.

        Args:
            config: Configuration dictionary
            models_dir: Optional directory containing local models
        """
        super().__init__(config)
        self._scanner = None
        self._valid_languages: List[str] = self._config.get("valid_languages", ["en"])
        self._match_type = self._config.get("match_type", "full")
        self._use_onnx = self._config.get("use_onnx", True)
        self._models_dir = models_dir or os.getenv("NEO_LLM_GUARD_MODELS_DIR")

    async def initialize(self) -> None:
        """Initialize the LLM Guard scanner."""
        if self._initialized:
            return

        try:
            from llm_guard.input_scanners import Language
            from llm_guard.input_scanners.language import DEFAULT_MODEL, MatchType

            model_config = None
            
            # Check if local models are available
            if self._models_dir:
                local_model_path = Path(self._models_dir) / "xlm-roberta-base-language-detection"
                
                if local_model_path.exists():
                    self.logger.info(
                        "language_using_local_model",
                        model_path=str(local_model_path)
                    )
                    
                    # Configure DEFAULT_MODEL to use local path
                    DEFAULT_MODEL.path = str(local_model_path)
                    DEFAULT_MODEL.kwargs["local_files_only"] = True
                    model_config = DEFAULT_MODEL
                else:
                    self.logger.warning(
                        "language_local_model_not_found",
                        expected_path=str(local_model_path),
                        message="Will download from HuggingFace (slower). Run: neo-guardrail download-models --scanners language"
                    )

            match_type = (
                MatchType.FULL
                if self._match_type == "full"
                else MatchType.SENTENCE
            )

            # Initialize scanner with model config if available
            if model_config:
                self._scanner = Language(
                    valid_languages=self._valid_languages,
                    threshold=self._threshold,
                    match_type=match_type,
                    use_onnx=self._use_onnx,
                    model=model_config
                )
            else:
                self._scanner = Language(
                    valid_languages=self._valid_languages,
                    threshold=self._threshold,
                    match_type=match_type,
                    use_onnx=self._use_onnx,
                )

            self.logger.info(
                "language_input_scanner_initialized",
                valid_languages=self._valid_languages,
                threshold=self._threshold,
                using_local_model=model_config is not None,
                match_type=self._match_type,
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
        """Check text for language validity.

        Args:
            text: Text to check
            context: Optional context (not used)

        Returns:
            GuardrailResult indicating if language is valid
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
                    f"Input language not in allowed list: {self._valid_languages}"
                    if not is_valid
                    else None
                ),
                metadata={
                    "detection_method": "llm_guard",
                    "valid_languages": self._valid_languages,
                    "raw_risk_score": risk_score,
                },
            )

        except Exception as e:
            self.logger.error(
                "language_detection_error",
                error=str(e),
            )
            return self._fallback_check(text)

    def _fallback_check(self, text: str) -> GuardrailResult:
        """Fallback language detection using heuristics.

        Args:
            text: Text to check

        Returns:
            GuardrailResult with detection results

        Note:
            This is a simplified fallback that only reliably detects
            a few languages. For production use, install llm_guard.
        """
        # Simple heuristic-based language detection
        detected_lang = self._detect_language_heuristic(text)

        is_valid = detected_lang in self._valid_languages or detected_lang == "unknown"

        return GuardrailResult(
            passed=is_valid,
            guardrail_name=self.name,
            layer=self.layer,
            risk_score=0.0 if is_valid else 0.7,
            message=(
                f"Detected language '{detected_lang}' not in allowed list"
                if not is_valid
                else None
            ),
            metadata={
                "detection_method": "heuristics",
                "detected_language": detected_lang,
                "valid_languages": self._valid_languages,
            },
        )

    def _detect_language_heuristic(self, text: str) -> str:
        """Simple heuristic-based language detection.

        Args:
            text: Text to analyze

        Returns:
            Detected language code or "unknown"
        """
        # Character set patterns for different languages
        # Check for specific Unicode ranges

        # Chinese characters (CJK Unified Ideographs)
        chinese_count = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
        if chinese_count > len(text) * 0.1:
            return "zh"

        # Japanese (Hiragana and Katakana)
        japanese_count = sum(1 for c in text if '\u3040' <= c <= '\u30ff')
        if japanese_count > len(text) * 0.1:
            return "ja"

        # Arabic characters
        arabic_count = sum(1 for c in text if '\u0600' <= c <= '\u06ff')
        if arabic_count > len(text) * 0.2:
            return "ar"

        # Russian/Cyrillic
        cyrillic_count = sum(1 for c in text if '\u0400' <= c <= '\u04ff')
        if cyrillic_count > len(text) * 0.2:
            return "ru"

        # Hindi/Devanagari
        hindi_count = sum(1 for c in text if '\u0900' <= c <= '\u097f')
        if hindi_count > len(text) * 0.2:
            return "hi"

        # Thai
        thai_count = sum(1 for c in text if '\u0e00' <= c <= '\u0e7f')
        if thai_count > len(text) * 0.2:
            return "th"

        # Greek
        greek_count = sum(1 for c in text if '\u0370' <= c <= '\u03ff')
        if greek_count > len(text) * 0.2:
            return "el"

        # If mostly Latin characters, assume English (default)
        latin_count = sum(1 for c in text if 'a' <= c.lower() <= 'z')
        if latin_count > len(text) * 0.3:
            # Could be English, Spanish, French, German, etc.
            # Default to "en" for Latin-based scripts
            return "en"

        return "unknown"

    async def cleanup(self) -> None:
        """Clean up scanner resources."""
        self._scanner = None
        await super().cleanup()
