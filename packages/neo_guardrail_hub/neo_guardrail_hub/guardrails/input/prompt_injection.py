"""Prompt injection detection guardrail.

This module provides guardrail protection against prompt injection
attacks using LLM Guard's PromptInjection scanner.
"""

import os
from pathlib import Path
from typing import Any, Dict, Optional

from ...core.models import GuardrailContext, GuardrailLayer, GuardrailResult
from ..base import GuardrailBase


class PromptInjectionGuardrail(GuardrailBase):
    """Detect and block prompt injection attacks.

    Uses LLM Guard's PromptInjection scanner to identify attempts
    to manipulate the LLM through crafted prompts.

    Configuration:
        threshold: Detection threshold (0.0-1.0), default 0.5
        match_type: "full" or "sentence", default "full"
        model: Model to use for detection, default "deberta"

    Example:
        guardrail = PromptInjectionGuardrail({
            "threshold": 0.8,
            "match_type": "full"
        })
        result = await guardrail.check("Ignore previous instructions...")
    """

    name = "prompt_injection"
    layer = GuardrailLayer.INPUT
    description = "Detect and block prompt injection attacks"

    def __init__(self, config: Optional[Dict[str, Any]] = None, models_dir: Optional[str] = None) -> None:
        """Initialize the prompt injection guardrail.

        Args:
            config: Configuration dictionary
            models_dir: Directory containing pre-downloaded models (optional)
        """
        super().__init__(config)
        self._scanner = None
        self._match_type = self._config.get("match_type", "full")
        self._models_dir = models_dir or os.getenv("NEO_LLM_GUARD_MODELS_DIR")

    async def initialize(self) -> None:
        """Initialize the LLM Guard scanner."""
        if self._initialized:
            return

        try:
            from llm_guard.input_scanners import PromptInjection
            from llm_guard.input_scanners.prompt_injection import MatchType, V2_MODEL

            match_type = (
                MatchType.FULL
                if self._match_type == "full"
                else MatchType.SENTENCE
            )

            # Configure local model if models directory is provided
            model_config = None
            if self._models_dir:
                local_model_path = Path(self._models_dir) / "deberta-v3-base-prompt-injection-v2"
                if local_model_path.exists() and (local_model_path / "config.json").exists():
                    # Configure LLM Guard to use local model
                    V2_MODEL.path = str(local_model_path)
                    V2_MODEL.kwargs["local_files_only"] = True
                    model_config = V2_MODEL
                    
                    self.logger.info(
                        "prompt_injection_using_local_model",
                        model_path=str(local_model_path)
                    )
                else:
                    self.logger.warning(
                        "prompt_injection_local_model_not_found",
                        expected_path=str(local_model_path),
                        message="Will download from HuggingFace. Run 'neo-guardrail download-models' to cache models locally."
                    )

            # Initialize scanner with model config if available
            if model_config:
                self._scanner = PromptInjection(
                    threshold=self._threshold,
                    match_type=match_type,
                    model=model_config
                )
            else:
                self._scanner = PromptInjection(
                    threshold=self._threshold,
                    match_type=match_type,
                )

            self.logger.info(
                "prompt_injection_scanner_initialized",
                threshold=self._threshold,
                match_type=self._match_type,
                using_local_model=bool(model_config)
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
        """Check text for prompt injection attempts.

        Args:
            text: Text to check
            context: Optional context (not used)

        Returns:
            GuardrailResult indicating if injection was detected
        """
        if self._scanner is None:
            # Fallback: use simple pattern matching
            return self._fallback_check(text)

        try:
            sanitized_prompt, is_valid, risk_score = self._scanner.scan(text)

            # Normalize risk score - LLM Guard returns -1.0 for safe content
            # We need to ensure it's between 0.0 and 1.0
            normalized_risk_score = max(0.0, min(1.0, risk_score))

            return GuardrailResult(
                passed=is_valid,
                guardrail_name=self.name,
                layer=self.layer,
                risk_score=normalized_risk_score,
                message=(
                    None
                    if is_valid
                    else "Potential prompt injection detected"
                ),
                sanitized_text=sanitized_prompt if not is_valid else None,
                metadata={
                    "original_length": len(text),
                    "sanitized_length": len(sanitized_prompt),
                    "detection_method": "llm_guard",
                    "raw_risk_score": risk_score,
                },
            )

        except Exception as e:
            self.logger.error(
                "prompt_injection_scan_error",
                error=str(e),
            )
            # Fail open or closed based on config
            return GuardrailResult(
                passed=False,
                guardrail_name=self.name,
                layer=self.layer,
                risk_score=1.0,
                message=f"Scan error: {str(e)}",
                metadata={"error": str(e)},
            )

    def _fallback_check(self, text: str) -> GuardrailResult:
        """Simple fallback check when LLM Guard is not available.

        Uses basic pattern matching to detect obvious injection attempts.
        """
        text_lower = text.lower()

        # Common injection patterns
        injection_patterns = [
            "ignore previous instructions",
            "ignore all previous",
            "disregard previous",
            "forget your instructions",
            "new instructions:",
            "override your programming",
            "you are now",
            "pretend you are",
            "act as if you are",
            "ignore the above",
            "ignore everything above",
            "do not follow",
            "system prompt:",
            "###instruction###",
            "```system",
        ]

        for pattern in injection_patterns:
            if pattern in text_lower:
                return GuardrailResult(
                    passed=False,
                    guardrail_name=self.name,
                    layer=self.layer,
                    risk_score=0.9,
                    message=f"Potential prompt injection detected: '{pattern}'",
                    metadata={
                        "matched_pattern": pattern,
                        "detection_method": "pattern_matching",
                    },
                )

        return GuardrailResult(
            passed=True,
            guardrail_name=self.name,
            layer=self.layer,
            risk_score=0.0,
            message=None,
            metadata={"detection_method": "pattern_matching"},
        )

    async def cleanup(self) -> None:
        """Clean up scanner resources."""
        self._scanner = None
        await super().cleanup()
