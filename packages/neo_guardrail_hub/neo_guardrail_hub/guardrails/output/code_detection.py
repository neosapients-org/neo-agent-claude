"""Code detection guardrail for output.

This module provides guardrail protection for detecting and filtering
specific programming languages in LLM outputs.
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from ...core.models import GuardrailContext, GuardrailLayer, GuardrailResult
from ..base import GuardrailBase, extract_prompt_from_context


class CodeDetectionOutputGuardrail(GuardrailBase):
    """Detect specific programming languages in LLM outputs.

    Uses LLM Guard's Code output scanner with the 
    philomath-1209/programming-language-identification model.

    Configuration:
        threshold: Detection confidence threshold (0.0-1.0, default 0.5)
        languages: List of programming languages to detect (default ["Python"])
        is_blocked: If True, block listed languages; if False, allow only listed (default True)
        use_onnx: Whether to use ONNX runtime (default True)

    Example:
        guardrail = CodeDetectionOutputGuardrail({
            "languages": ["Python", "JavaScript"],
            "is_blocked": True
        })
        result = await guardrail.check(
            "Here's Python code: def hello(): print('hi')",
            context={"prompt": "Write a greeting function"}
        )
    """

    name = "code_detection_output"
    layer = GuardrailLayer.OUTPUT
    description = "Detect specific programming languages in LLM output"

    # Common code patterns for fallback
    CODE_PATTERNS = {
        "python": ["def ", "import ", "from ", "class ", "print(", "if __name__"],
        "javascript": ["function ", "const ", "let ", "var ", "=>", "console.log"],
        "java": ["public class", "public static", "System.out", "void main"],
        "c": ["#include", "int main", "printf(", "scanf("],
        "cpp": ["#include", "std::", "cout <<", "cin >>", "namespace"],
        "ruby": ["def ", "end", "puts ", "require ", "class "],
        "go": ["func ", "package ", "import ", "fmt."],
        "rust": ["fn ", "let ", "mut ", "impl ", "pub "],
    }

    def __init__(self, config: Optional[Dict[str, Any]] = None, models_dir: Optional[str] = None) -> None:
        """Initialize the code detection guardrail.

        Args:
            config: Configuration dictionary
            models_dir: Optional path to directory containing local models
        """
        super().__init__(config)
        self._scanner = None
        self._languages = self._config.get("languages", ["Python"])
        self._is_blocked = self._config.get("is_blocked", True)
        self._use_onnx = self._config.get("use_onnx", True)
        self._models_dir = models_dir or os.getenv("NEO_LLM_GUARD_MODELS_DIR")

    async def initialize(self) -> None:
        """Initialize the LLM Guard scanner."""
        if self._initialized:
            return

        try:
            from llm_guard.output_scanners import Code
            from llm_guard.input_scanners.code import DEFAULT_MODEL

            # Configure local model if available
            model_config = None
            if self._models_dir:
                local_model_path = Path(self._models_dir) / "programming-language-identification"
                if local_model_path.exists():
                    self.logger.info(
                        "code_detection_output_using_local_model",
                        model_path=str(local_model_path)
                    )
                    DEFAULT_MODEL.path = str(local_model_path)
                    DEFAULT_MODEL.kwargs["local_files_only"] = True
                    model_config = DEFAULT_MODEL

            if model_config:
                self._scanner = Code(
                    languages=self._languages,
                    is_blocked=self._is_blocked,
                    threshold=self._threshold,
                    use_onnx=self._use_onnx,
                    model=model_config,
                )
            else:
                self._scanner = Code(
                    languages=self._languages,
                    is_blocked=self._is_blocked,
                    threshold=self._threshold,
                    use_onnx=self._use_onnx,
                )

            self.logger.info(
                "code_detection_output_scanner_initialized",
                languages=self._languages,
                is_blocked=self._is_blocked,
                threshold=self._threshold,
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
                "code_detection_output_scanner_init_failed",
                message="Failed to initialize Code scanner, will use fallback",
                error=str(e),
            )
            self._scanner = None

        await super().initialize()

    async def _check(
        self,
        text: str,
        context: Optional[GuardrailContext] = None,
    ) -> GuardrailResult:
        """Check for code in the output.

        Args:
            text: LLM output to check
            context: Optional context with prompt

        Returns:
            GuardrailResult indicating if blocked code was detected
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
                    "blocked_code_detected_in_output",
                    languages=self._languages,
                    is_blocked=self._is_blocked,
                    risk_score=normalized_risk,
                )

            return GuardrailResult(
                passed=passed,
                guardrail_name=self.name,
                layer=self.layer,
                risk_score=normalized_risk,
                message=(
                    f"Blocked programming language detected in output"
                    if not passed else None
                ),
                metadata={
                    "code_detected": not passed,
                    "languages_configured": self._languages,
                    "is_blocked": self._is_blocked,
                    "detection_method": "language_identification_model",
                },
            )

        except Exception as e:
            self.logger.error("code_detection_output_error", error=str(e))
            return self._fallback_check(text)

    def _fallback_check(self, text: str) -> GuardrailResult:
        """Fallback check using pattern matching.

        Args:
            text: Text to check

        Returns:
            GuardrailResult based on pattern presence
        """
        detected_languages: List[str] = []

        for lang, patterns in self.CODE_PATTERNS.items():
            # Check if language is in configured list (case insensitive)
            lang_matches = any(
                lang.lower() == configured.lower()
                for configured in self._languages
            )
            
            if lang_matches:
                for pattern in patterns:
                    if pattern in text:
                        detected_languages.append(lang)
                        break

        has_blocked_code = len(detected_languages) > 0

        if self._is_blocked:
            passed = not has_blocked_code
            risk_score = 1.0 if has_blocked_code else 0.0
        else:
            # is_blocked=False means only allow listed languages
            # If no code detected, pass; if code detected, it should be in list
            passed = True  # Simplified fallback
            risk_score = 0.0

        return GuardrailResult(
            passed=passed,
            guardrail_name=self.name,
            layer=self.layer,
            risk_score=risk_score,
            message=(
                f"Blocked code detected: {detected_languages}" if not passed else None
            ),
            metadata={
                "detected_languages": detected_languages,
                "languages_configured": self._languages,
                "detection_method": "pattern_fallback",
            },
        )
