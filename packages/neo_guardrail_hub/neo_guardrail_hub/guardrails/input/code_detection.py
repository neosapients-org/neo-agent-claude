"""Code detection guardrail for input.

This module provides guardrail protection for detecting and validating
code in user inputs using LLM Guard's Code scanner.
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from ...core.models import GuardrailContext, GuardrailLayer, GuardrailResult
from ..base import GuardrailBase


# Supported programming languages
SUPPORTED_LANGUAGES = [
    "ARM Assembly", "AppleScript", "C", "C#", "C++", "COBOL", "Erlang",
    "Fortran", "Go", "Java", "JavaScript", "Kotlin", "Lua",
    "Mathematica/Wolfram Language", "PHP", "Pascal", "Perl", "PowerShell",
    "Python", "R", "Ruby", "Rust", "Scala", "Swift", "Visual Basic .NET", "jq"
]


class CodeDetectionInputGuardrail(GuardrailBase):
    """Detect and validate code in user input.

    Uses LLM Guard's Code scanner to identify programming code
    in specific languages. Can either allow only specific languages
    or block specific languages.

    Configuration:
        languages: List of programming languages to allow/block
        is_blocked: If True, block listed languages; if False, allow only listed languages
        threshold: Detection threshold (0.0-1.0), default 0.5
        use_onnx: Whether to use ONNX runtime for faster inference, default True

    Example:
        # Block Python and JavaScript code
        guardrail = CodeDetectionInputGuardrail({
            "languages": ["Python", "JavaScript"],
            "is_blocked": True,
            "threshold": 0.5
        })
        result = await guardrail.check("```python\\ndef hello(): pass\\n```")
    """

    name = "code_detection_input"
    layer = GuardrailLayer.INPUT
    description = "Detect and validate code in input"

    def __init__(self, config: Optional[Dict[str, Any]] = None, models_dir: Optional[str] = None) -> None:
        """Initialize the code detection input guardrail.

        Args:
            config: Configuration dictionary
            models_dir: Optional directory containing local models
        """
        super().__init__(config)
        self._scanner = None
        self._languages: List[str] = self._config.get("languages", ["Python"])
        self._is_blocked = self._config.get("is_blocked", True)
        self._use_onnx = self._config.get("use_onnx", True)
        self._models_dir = models_dir or os.getenv("NEO_LLM_GUARD_MODELS_DIR")

    async def initialize(self) -> None:
        """Initialize the LLM Guard scanner."""
        if self._initialized:
            return

        try:
            from llm_guard.input_scanners import Code
            from llm_guard.input_scanners.code import DEFAULT_MODEL

            model_config = None
            
            # Check if local models are available
            if self._models_dir:
                local_model_path = Path(self._models_dir) / "programming-language-identification"
                
                if local_model_path.exists():
                    self.logger.info(
                        "code_detection_using_local_model",
                        model_path=str(local_model_path)
                    )
                    
                    # Configure DEFAULT_MODEL to use local path
                    DEFAULT_MODEL.path = str(local_model_path)
                    DEFAULT_MODEL.kwargs["local_files_only"] = True
                    model_config = DEFAULT_MODEL
                else:
                    self.logger.warning(
                        "code_detection_local_model_not_found",
                        expected_path=str(local_model_path),
                        message="Will download from HuggingFace (slower). Run: neo-guardrail download-models --scanners code_detection"
                    )

            # Initialize scanner with model config if available
            if model_config:
                self._scanner = Code(
                    languages=self._languages,
                    is_blocked=self._is_blocked,
                    threshold=self._threshold,
                    use_onnx=self._use_onnx,
                    model=model_config
                )
            else:
                self._scanner = Code(
                    languages=self._languages,
                    is_blocked=self._is_blocked,
                    threshold=self._threshold,
                    use_onnx=self._use_onnx,
                )

            self.logger.info(
                "code_detection_input_scanner_initialized",
                languages=self._languages,
                is_blocked=self._is_blocked,
                threshold=self._threshold,
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
        """Check text for code in specific languages.

        Args:
            text: Text to check
            context: Optional context (not used)

        Returns:
            GuardrailResult indicating if code was detected
        """
        if self._scanner is None:
            return self._fallback_check(text)

        try:
            sanitized_prompt, is_valid, risk_score = self._scanner.scan(text)

            # Normalize risk score to 0.0-1.0 range
            normalized_risk_score = max(0.0, min(1.0, risk_score))

            message = None
            if not is_valid:
                if self._is_blocked:
                    message = f"Blocked code language detected"
                else:
                    message = "Input does not contain required code language"

            return GuardrailResult(
                passed=is_valid,
                guardrail_name=self.name,
                layer=self.layer,
                risk_score=normalized_risk_score,
                message=message,
                metadata={
                    "detection_method": "llm_guard",
                    "languages": self._languages,
                    "is_blocked": self._is_blocked,
                    "raw_risk_score": risk_score,
                },
            )

        except Exception as e:
            self.logger.error(
                "code_detection_error",
                error=str(e),
            )
            return self._fallback_check(text)

    def _fallback_check(self, text: str) -> GuardrailResult:
        """Fallback code detection using pattern matching.

        Args:
            text: Text to check

        Returns:
            GuardrailResult with detection results
        """
        import re

        # Language-specific patterns
        language_patterns = {
            "Python": [
                r'\bdef\s+\w+\s*\(',
                r'\bclass\s+\w+\s*(\(|:)',
                r'\bimport\s+\w+',
                r'\bfrom\s+\w+\s+import',
                r'```python',
            ],
            "JavaScript": [
                r'\bfunction\s+\w+\s*\(',
                r'\bconst\s+\w+\s*=',
                r'\blet\s+\w+\s*=',
                r'\bvar\s+\w+\s*=',
                r'=>\s*{',
                r'```javascript',
                r'```js',
            ],
            "Java": [
                r'\bpublic\s+(class|interface|static|void)',
                r'\bprivate\s+(class|interface|static|void)',
                r'```java',
            ],
            "C": [
                r'#include\s*<\w+\.h>',
                r'\bint\s+main\s*\(',
                r'```c\b',
            ],
            "C++": [
                r'#include\s*<\w+>',
                r'\bstd::',
                r'```cpp',
                r'```c\+\+',
            ],
            "Go": [
                r'\bfunc\s+\w+\s*\(',
                r'\bpackage\s+\w+',
                r'```go',
            ],
            "Ruby": [
                r'\bdef\s+\w+',
                r'\bclass\s+\w+\s*<',
                r'```ruby',
            ],
            "Rust": [
                r'\bfn\s+\w+\s*\(',
                r'\blet\s+mut\s+',
                r'```rust',
            ],
        }

        detected_languages: List[str] = []

        for lang, patterns in language_patterns.items():
            for pattern in patterns:
                if re.search(pattern, text, re.IGNORECASE):
                    if lang not in detected_languages:
                        detected_languages.append(lang)
                    break

        # Determine validity based on is_blocked setting
        if self._is_blocked:
            # Block mode: fail if any blocked language is detected
            blocked_detected = [lang for lang in detected_languages if lang in self._languages]
            is_valid = len(blocked_detected) == 0
            message = f"Blocked code detected: {', '.join(blocked_detected)}" if blocked_detected else None
        else:
            # Allow mode: fail if detected language is not in allowed list
            if detected_languages:
                allowed_detected = [lang for lang in detected_languages if lang in self._languages]
                is_valid = len(allowed_detected) > 0
                message = None if is_valid else f"Code not in allowed languages: {', '.join(detected_languages)}"
            else:
                is_valid = True
                message = None

        risk_score = 0.0 if is_valid else 0.7

        return GuardrailResult(
            passed=is_valid,
            guardrail_name=self.name,
            layer=self.layer,
            risk_score=risk_score,
            message=message,
            metadata={
                "detection_method": "pattern_matching",
                "detected_languages": detected_languages,
                "is_blocked": self._is_blocked,
            },
        )

    async def cleanup(self) -> None:
        """Clean up scanner resources."""
        self._scanner = None
        await super().cleanup()
