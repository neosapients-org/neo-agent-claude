"""Toxicity detection guardrail for output.

This module provides guardrail protection for detecting toxic, harmful,
or offensive content in LLM outputs.
"""

import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from ...core.models import GuardrailContext, GuardrailLayer, GuardrailResult
from ..base import GuardrailBase, extract_prompt_from_context


# Fallback patterns for toxicity detection
DEFAULT_TOXIC_PATTERNS = [
    # Slurs and offensive terms (patterns, not actual words)
    r"(?i)\b(idiot|moron|stupid|dumb)\b",
    r"(?i)\b(hate|kill|murder|destroy)\s+(you|them|him|her)\b",
    r"(?i)\b(shut\s+up|go\s+away|get\s+lost)\b",
    r"(?i)\b(worthless|useless|pathetic)\b",
    r"(?i)you\s+(suck|are\s+terrible|are\s+awful)\b",
    # Threats
    r"(?i)\b(i('ll| will)\s+(hurt|harm|kill|find))\b",
    r"(?i)\b(you('ll| will)\s+(regret|suffer|pay))\b",
    r"(?i)\b(threat|threaten|threatening)\b",
]


class ToxicityOutputGuardrail(GuardrailBase):
    """Detect toxic, harmful, or offensive content in LLM outputs.

    Uses LLM Guard's Toxicity output scanner with the
    unitary/unbiased-toxic-roberta model for classification.

    Configuration:
        threshold: Toxicity detection threshold (0.0-1.0, default 0.5)
        match_type: How to match text - "FULL" or "SENTENCE" (default "SENTENCE")
        use_onnx: Whether to use ONNX runtime for faster inference (default True)

    Example:
        guardrail = ToxicityOutputGuardrail({
            "threshold": 0.5,
            "match_type": "SENTENCE"
        })
        result = await guardrail.check("Here's a helpful response...")
    """

    name = "toxicity_output"
    layer = GuardrailLayer.OUTPUT
    description = "Detect toxic content in LLM outputs"

    def __init__(self, config: Optional[Dict[str, Any]] = None, models_dir: Optional[str] = None) -> None:
        """Initialize the toxicity output guardrail.

        Args:
            config: Configuration dictionary
            models_dir: Optional path to directory containing local models
        """
        super().__init__(config)
        self._scanner = None
        self._match_type = self._config.get("match_type", "SENTENCE")
        self._use_onnx = self._config.get("use_onnx", True)
        self._models_dir = models_dir or os.getenv("NEO_LLM_GUARD_MODELS_DIR")
        self._toxic_patterns: List[str] = self._config.get(
            "toxic_patterns", DEFAULT_TOXIC_PATTERNS
        )
        self._compiled_patterns: List[re.Pattern] = []

    async def initialize(self) -> None:
        """Initialize the LLM Guard scanner."""
        if self._initialized:
            return

        # Compile fallback patterns
        self._compiled_patterns = [
            re.compile(pattern) for pattern in self._toxic_patterns
        ]

        try:
            from llm_guard.output_scanners import Toxicity
            from llm_guard.output_scanners.toxicity import MatchType
            from llm_guard.input_scanners.toxicity import DEFAULT_MODEL

            # Determine match type
            match_type = (
                MatchType.SENTENCE
                if self._match_type.upper() == "SENTENCE"
                else MatchType.FULL
            )

            # Configure local model if available
            model_config = None
            if self._models_dir:
                local_model_path = Path(self._models_dir) / "unbiased-toxic-roberta"
                if local_model_path.exists():
                    self.logger.info(
                        "toxicity_output_using_local_model",
                        model_path=str(local_model_path)
                    )
                    DEFAULT_MODEL.path = str(local_model_path)
                    DEFAULT_MODEL.kwargs["local_files_only"] = True
                    model_config = DEFAULT_MODEL

            if model_config:
                self._scanner = Toxicity(
                    threshold=self._threshold,
                    match_type=match_type,
                    use_onnx=self._use_onnx,
                    model=model_config,
                )
            else:
                self._scanner = Toxicity(
                    threshold=self._threshold,
                    match_type=match_type,
                    use_onnx=self._use_onnx,
                )

            self.logger.info(
                "toxicity_output_scanner_initialized",
                threshold=self._threshold,
                match_type=self._match_type,
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
        """Check if output contains toxic content.

        Args:
            text: LLM output text to check
            context: Optional context

        Returns:
            GuardrailResult indicating if toxicity was detected
        """
        # Get the original prompt from context for the scanner
        prompt = extract_prompt_from_context(context)

        if self._scanner is None:
            # Fallback: use pattern matching
            return self._fallback_check(text)

        try:
            # LLM Guard output scanners take both prompt and output
            sanitized_text, is_valid, risk_score = self._scanner.scan(prompt, text)

            # Normalize risk_score to 0.0-1.0 range
            normalized_risk_score = max(0.0, min(1.0, risk_score))

            # is_valid indicates if the output is NOT toxic
            toxicity_detected = not is_valid or normalized_risk_score >= self._threshold

            return GuardrailResult(
                passed=not toxicity_detected,
                guardrail_name=self.name,
                layer=self.layer,
                risk_score=normalized_risk_score,
                message=(
                    f"Toxic content detected (confidence: {normalized_risk_score:.2f})"
                    if toxicity_detected
                    else None
                ),
                metadata={
                    "toxicity_detected": toxicity_detected,
                    "toxicity_score": normalized_risk_score,
                    "detection_method": "llm_guard",
                    "raw_risk_score": risk_score,
                },
            )

        except Exception as e:
            self.logger.error(
                "toxicity_output_check_error",
                error=str(e),
            )
            # Fall back to pattern matching on error
            return self._fallback_check(text)

    def _fallback_check(self, text: str) -> GuardrailResult:
        """Fallback check using regex patterns.

        Args:
            text: Text to check for toxic patterns

        Returns:
            GuardrailResult based on pattern matching
        """
        matched_patterns = []

        for pattern in self._compiled_patterns:
            matches = pattern.findall(text)
            if matches:
                matched_patterns.append({
                    "pattern": pattern.pattern,
                    "match_count": len(matches),
                })

        if matched_patterns:
            # Calculate risk based on number and diversity of matched patterns
            total_matches = sum(p["match_count"] for p in matched_patterns)
            risk_score = min(0.95, 0.4 + (len(matched_patterns) * 0.1) + (total_matches * 0.05))

            return GuardrailResult(
                passed=False,
                guardrail_name=self.name,
                layer=self.layer,
                risk_score=risk_score,
                message=f"Potentially toxic content detected ({len(matched_patterns)} pattern types)",
                metadata={
                    "toxicity_detected": True,
                    "pattern_matches": matched_patterns[:5],
                    "total_matches": total_matches,
                    "detection_method": "pattern_matching",
                },
            )

        return GuardrailResult(
            passed=True,
            guardrail_name=self.name,
            layer=self.layer,
            risk_score=0.0,
            message=None,
            metadata={
                "toxicity_detected": False,
                "detection_method": "pattern_matching",
            },
        )
