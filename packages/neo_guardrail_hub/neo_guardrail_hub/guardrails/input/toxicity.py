"""Toxicity detection guardrail for input.

This module provides guardrail protection against toxic, harmful,
or offensive content in user inputs using LLM Guard's Toxicity scanner.
"""

import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from ...core.models import GuardrailContext, GuardrailLayer, GuardrailResult
from ..base import GuardrailBase


class ToxicityInputGuardrail(GuardrailBase):
    """Detect and block toxic content in user input.

    Uses LLM Guard's Toxicity scanner with the unitary/unbiased-toxic-roberta
    model to identify toxic, hateful, or offensive content.

    Configuration:
        threshold: Detection threshold (0.0-1.0), default 0.5
        match_type: "full" or "sentence"
            - full: Scan entire text at once
            - sentence: Scan each sentence separately
        on_fail: Action on failure - "block" or "warn", default "block"

    Example:
        guardrail = ToxicityInputGuardrail({
            "threshold": 0.7,
            "match_type": "sentence"
        })
        result = await guardrail.check("Some potentially toxic text...")
    """

    name = "toxicity_input"
    layer = GuardrailLayer.INPUT
    description = "Detect and block toxic or offensive content in input"

    # Toxic word patterns for fallback detection
    TOXIC_PATTERNS = [
        # Profanity and slurs (censored patterns)
        r"\b(?:f+u+c+k+|sh+i+t+|a+ss+|b+i+t+c+h+|d+a+m+n+)\b",
        # Hate speech indicators
        r"\b(?:hate|kill|murder|attack|destroy)\s+(?:you|them|all|everyone)\b",
        # Threatening language
        r"\b(?:i\s+will|gonna|going\s+to)\s+(?:kill|hurt|destroy|attack)\b",
        # Harassment patterns
        r"\b(?:you\s+are\s+(?:stupid|idiot|moron|dumb|worthless))\b",
        r"\b(?:go\s+die|kys|kill\s+yourself)\b",
    ]

    def __init__(self, config: Optional[Dict[str, Any]] = None, models_dir: Optional[str] = None) -> None:
        """Initialize the toxicity input guardrail.

        Args:
            config: Configuration dictionary
            models_dir: Optional directory containing local models
        """
        super().__init__(config)
        self._scanner = None
        self._match_type = self._config.get("match_type", "full")
        self._models_dir = models_dir or os.getenv("NEO_LLM_GUARD_MODELS_DIR")
        self._compiled_patterns = [
            re.compile(pattern, re.IGNORECASE)
            for pattern in self.TOXIC_PATTERNS
        ]

    async def initialize(self) -> None:
        """Initialize the LLM Guard scanner."""
        if self._initialized:
            return

        try:
            from llm_guard.input_scanners import Toxicity
            from llm_guard.input_scanners.toxicity import DEFAULT_MODEL, MatchType

            model_config = None
            
            # Check if local models are available
            if self._models_dir:
                local_model_path = Path(self._models_dir) / "unbiased-toxic-roberta"
                
                if local_model_path.exists():
                    self.logger.info(
                        "toxicity_using_local_model",
                        model_path=str(local_model_path)
                    )
                    
                    # Configure DEFAULT_MODEL to use local path
                    DEFAULT_MODEL.path = str(local_model_path)
                    DEFAULT_MODEL.kwargs["local_files_only"] = True
                    model_config = DEFAULT_MODEL
                else:
                    self.logger.warning(
                        "toxicity_local_model_not_found",
                        expected_path=str(local_model_path),
                        message="Will download from HuggingFace (slower). Run: neo-guardrail download-models --scanners toxicity"
                    )

            match_type = (
                MatchType.FULL
                if self._match_type == "full"
                else MatchType.SENTENCE
            )

            # Initialize scanner with model config if available
            if model_config:
                self._scanner = Toxicity(
                    threshold=self._threshold,
                    match_type=match_type,
                    model=model_config
                )
            else:
                self._scanner = Toxicity(
                    threshold=self._threshold,
                    match_type=match_type,
                )

            self.logger.info(
                "toxicity_input_scanner_initialized",
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
        """Check text for toxic content.

        Args:
            text: Text to check
            context: Optional context (not used)

        Returns:
            GuardrailResult indicating if toxicity was detected
        """
        if self._scanner is None:
            return self._fallback_check(text)

        try:
            sanitized_prompt, is_valid, risk_score = self._scanner.scan(text)

            # Normalize risk score to 0.0-1.0 range
            normalized_risk_score = max(0.0, min(1.0, risk_score))

            if not is_valid:
                self.logger.warning(
                    "toxicity_detected",
                    risk_score=normalized_risk_score,
                )

            return GuardrailResult(
                passed=is_valid,
                guardrail_name=self.name,
                layer=self.layer,
                risk_score=normalized_risk_score,
                message=(
                    "Toxic content detected"
                    if not is_valid
                    else None
                ),
                sanitized_text=sanitized_prompt if not is_valid else None,
                metadata={
                    "detection_method": "llm_guard",
                    "match_type": self._match_type,
                    "raw_risk_score": risk_score,
                },
            )

        except Exception as e:
            self.logger.error(
                "toxicity_detection_error",
                error=str(e),
            )
            return self._fallback_check(text)

    def _fallback_check(self, text: str) -> GuardrailResult:
        """Fallback toxicity detection using pattern matching.

        Args:
            text: Text to check

        Returns:
            GuardrailResult with detection results
        """
        self.logger.debug("initializing_guardrail", guardrail=self.name)

        detected_patterns: List[str] = []

        for pattern in self._compiled_patterns:
            matches = pattern.findall(text)
            if matches:
                detected_patterns.extend(matches)

        has_toxicity = len(detected_patterns) > 0
        risk_score = min(0.4 * len(detected_patterns), 1.0) if has_toxicity else 0.0
        passed = risk_score < self._threshold

        if has_toxicity:
            self.logger.warning(
                "toxicity_detected_fallback",
                pattern_count=len(detected_patterns),
            )
        else:
            self.logger.debug("no_toxicity_detected")

        return GuardrailResult(
            passed=passed,
            guardrail_name=self.name,
            layer=self.layer,
            risk_score=risk_score,
            message=(
                "Potential toxic content detected"
                if has_toxicity and not passed
                else None
            ),
            metadata={
                "detection_method": "regex_fallback",
                "patterns_matched": len(detected_patterns),
                "match_type": self._match_type,
            },
        )
