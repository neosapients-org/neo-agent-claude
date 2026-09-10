"""Ban code guardrail for input.

This module provides guardrail protection against code snippets
in user inputs using LLM Guard's BanCode scanner.
"""

import os
from pathlib import Path
from typing import Any, Dict, Optional

from ...core.models import GuardrailContext, GuardrailLayer, GuardrailResult
from ..base import GuardrailBase


class BanCodeInputGuardrail(GuardrailBase):
    """Detect and block code snippets in user input.

    Uses LLM Guard's BanCode scanner to identify programming code
    in prompts. Useful for preventing code injection, proprietary
    code sharing, or unintended code execution.

    Configuration:
        threshold: Detection threshold (0.0-1.0), default 0.5
        use_onnx: Whether to use ONNX runtime for faster inference, default True
        model_path: Optional path to custom model

    Example:
        guardrail = BanCodeInputGuardrail({
            "threshold": 0.5
        })
        result = await guardrail.check("def hello(): print('world')")
    """

    name = "ban_code_input"
    layer = GuardrailLayer.INPUT
    description = "Detect and block code snippets in input"

    def __init__(self, config: Optional[Dict[str, Any]] = None, models_dir: Optional[str] = None) -> None:
        """Initialize the ban code input guardrail.

        Args:
            config: Configuration dictionary
            models_dir: Optional directory containing local models
        """
        super().__init__(config)
        self._scanner = None
        self._use_onnx = self._config.get("use_onnx", True)
        self._models_dir = models_dir or os.getenv("NEO_LLM_GUARD_MODELS_DIR")

    async def initialize(self) -> None:
        """Initialize the LLM Guard scanner."""
        if self._initialized:
            return

        try:
            from llm_guard.input_scanners import BanCode
            from llm_guard.input_scanners.ban_code import MODEL_SM

            model_config = None
            
            # Check if local models are available
            if self._models_dir:
                local_model_path = Path(self._models_dir) / "codenlbert-sm"
                
                if local_model_path.exists():
                    self.logger.info(
                        "ban_code_using_local_model",
                        model_path=str(local_model_path)
                    )
                    
                    # Configure MODEL_SM to use local path
                    MODEL_SM.path = str(local_model_path)
                    MODEL_SM.kwargs["local_files_only"] = True
                    model_config = MODEL_SM
                else:
                    self.logger.warning(
                        "ban_code_local_model_not_found",
                        expected_path=str(local_model_path),
                        message="Will download from HuggingFace (slower). Run: neo-guardrail download-models --scanners ban_code"
                    )

            # Initialize scanner with model config if available
            if model_config:
                self._scanner = BanCode(
                    threshold=self._threshold,
                    use_onnx=self._use_onnx,
                    model=model_config
                )
            else:
                self._scanner = BanCode(
                    threshold=self._threshold,
                    use_onnx=self._use_onnx,
                )

            self.logger.info(
                "ban_code_input_scanner_initialized",
                threshold=self._threshold,
                use_onnx=self._use_onnx,
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
        """Check text for code snippets.

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

            return GuardrailResult(
                passed=is_valid,
                guardrail_name=self.name,
                layer=self.layer,
                risk_score=normalized_risk_score,
                message=(
                    "Code detected in input"
                    if not is_valid
                    else None
                ),
                sanitized_text=sanitized_prompt,
                metadata={
                    "detection_method": "llm_guard",
                    "raw_risk_score": risk_score,
                },
            )

        except Exception as e:
            self.logger.error(
                "ban_code_error",
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

        # Common code patterns
        code_patterns = [
            # Python
            r'\bdef\s+\w+\s*\(.*\)\s*:',
            r'\bclass\s+\w+\s*(\(.*\))?\s*:',
            r'\bimport\s+\w+',
            r'\bfrom\s+\w+\s+import\s+',
            r'\bif\s+__name__\s*==\s*["\']__main__["\']\s*:',
            # JavaScript
            r'\bfunction\s+\w+\s*\(.*\)\s*\{',
            r'\bconst\s+\w+\s*=\s*\(.*\)\s*=>',
            r'\bvar\s+\w+\s*=',
            r'\blet\s+\w+\s*=',
            # SQL
            r'\bSELECT\s+.+\s+FROM\s+',
            r'\bINSERT\s+INTO\s+',
            r'\bUPDATE\s+\w+\s+SET\s+',
            r'\bDELETE\s+FROM\s+',
            r'\bCREATE\s+TABLE\s+',
            # General
            r'```[\w]*\n',  # Markdown code blocks
            r'\breturn\s+\w+',
            r'\bfor\s+\w+\s+in\s+',
            r'\bwhile\s+.+\s*:',
            r'\btry\s*:',
            r'\bexcept\s+',
        ]

        detected_patterns = []
        for pattern in code_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                detected_patterns.append(pattern)

        has_code = len(detected_patterns) > 0
        risk_score = min(0.3 * len(detected_patterns), 1.0) if has_code else 0.0

        return GuardrailResult(
            passed=not has_code,
            guardrail_name=self.name,
            layer=self.layer,
            risk_score=risk_score,
            message=(
                "Code patterns detected in input"
                if has_code
                else None
            ),
            metadata={
                "detection_method": "pattern_matching",
                "patterns_found": len(detected_patterns),
            },
        )

    async def cleanup(self) -> None:
        """Clean up scanner resources."""
        self._scanner = None
        await super().cleanup()
