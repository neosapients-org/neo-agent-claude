"""No refusal detection guardrail for output.

This module provides guardrail protection for detecting when an LLM
refuses to answer a legitimate question.
"""

import re
from typing import Any, Dict, List, Optional

from ...core.models import GuardrailContext, GuardrailLayer, GuardrailResult
from ..base import GuardrailBase, extract_prompt_from_context


# Common refusal patterns for fallback detection
DEFAULT_REFUSAL_PATTERNS = [
    r"(?i)i('m| am) (sorry|afraid).*can('t|not)",
    r"(?i)i (can't|cannot|won't|will not) (help|assist|provide|answer)",
    r"(?i)as an ai( language model)?.*i (can't|cannot|am (not able|unable))",
    r"(?i)i('m| am) (not able|unable) to",
    r"(?i)unfortunately.*i (can't|cannot)",
    r"(?i)it('s| is) (not possible|impossible) for me",
    r"(?i)i don't have (the ability|access|permission)",
    r"(?i)i('m| am) not (designed|programmed|allowed) to",
    r"(?i)my (guidelines|policies|rules) (prevent|prohibit|don't allow)",
    r"(?i)i must (decline|refuse)",
    r"(?i)that('s| is) (outside|beyond) (my|the scope)",
    r"(?i)i('m| am) prohibited from",
]


class NoRefusalGuardrail(GuardrailBase):
    """Detect when LLM refuses to answer a legitimate question.

    Uses LLM Guard's NoRefusal output scanner to identify refusal responses.
    Uses ProtectAI/distilroberta-base-rejection-v1 model for classification.

    Configuration:
        threshold: Refusal detection threshold (0.0-1.0, default 0.5)
        match_type: How to match text - "FULL" or "SENTENCE" (default "FULL")
        use_onnx: Whether to use ONNX runtime for faster inference (default True)

    Example:
        guardrail = NoRefusalGuardrail({
            "threshold": 0.5,
            "match_type": "FULL"
        })
        result = await guardrail.check(
            "I cannot help with that request.",
            context={"prompt": "What's the weather?"}
        )
    """

    name = "no_refusal"
    layer = GuardrailLayer.OUTPUT
    description = "Detect when LLM refuses to answer legitimate questions"

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize the no refusal detection guardrail.

        Args:
            config: Configuration dictionary
        """
        super().__init__(config)
        self._scanner = None
        self._match_type = self._config.get("match_type", "FULL")
        self._use_onnx = self._config.get("use_onnx", True)
        self._refusal_patterns: List[str] = self._config.get(
            "refusal_patterns", DEFAULT_REFUSAL_PATTERNS
        )
        self._compiled_patterns: List[re.Pattern] = []

    async def initialize(self) -> None:
        """Initialize the LLM Guard scanner."""
        if self._initialized:
            return

        # Compile fallback patterns
        self._compiled_patterns = [
            re.compile(pattern) for pattern in self._refusal_patterns
        ]

        try:
            from llm_guard.output_scanners import NoRefusal
            from llm_guard.output_scanners.no_refusal import MatchType

            # Determine match type
            match_type = (
                MatchType.SENTENCE
                if self._match_type.upper() == "SENTENCE"
                else MatchType.FULL
            )

            self._scanner = NoRefusal(
                threshold=self._threshold,
                match_type=match_type,
                use_onnx=self._use_onnx,
            )

            self.logger.info(
                "no_refusal_scanner_initialized",
                threshold=self._threshold,
                match_type=self._match_type,
                use_onnx=self._use_onnx,
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
        """Check if output contains a refusal response.

        Args:
            text: LLM output text to check
            context: Optional context containing the original prompt

        Returns:
            GuardrailResult indicating if refusal was detected
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

            # Detection means refusal was found - higher score = more likely refusal
            refusal_detected = not is_valid or normalized_risk_score >= self._threshold

            return GuardrailResult(
                passed=not refusal_detected,  # Pass if NO refusal detected
                guardrail_name=self.name,
                layer=self.layer,
                risk_score=normalized_risk_score,
                message=(
                    f"Refusal response detected (confidence: {normalized_risk_score:.2f})"
                    if refusal_detected
                    else None
                ),
                metadata={
                    "refusal_detected": refusal_detected,
                    "refusal_confidence": normalized_risk_score,
                    "prompt_provided": bool(prompt),
                    "detection_method": "llm_guard",
                    "raw_risk_score": risk_score,
                },
            )

        except Exception as e:
            self.logger.error(
                "no_refusal_check_error",
                error=str(e),
            )
            # Fall back to pattern matching on error
            return self._fallback_check(text)

    def _fallback_check(self, text: str) -> GuardrailResult:
        """Fallback check using regex patterns.

        Args:
            text: Text to check for refusal patterns

        Returns:
            GuardrailResult based on pattern matching
        """
        matched_patterns = []
        text_lower = text.lower()

        for pattern in self._compiled_patterns:
            if pattern.search(text):
                matched_patterns.append(pattern.pattern)

        if matched_patterns:
            # Calculate risk based on number of matched patterns
            risk_score = min(0.9, 0.5 + (len(matched_patterns) * 0.1))

            return GuardrailResult(
                passed=False,
                guardrail_name=self.name,
                layer=self.layer,
                risk_score=risk_score,
                message=f"Refusal response detected (matched {len(matched_patterns)} patterns)",
                metadata={
                    "refusal_detected": True,
                    "matched_patterns": matched_patterns[:5],  # Limit to first 5
                    "pattern_count": len(matched_patterns),
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
                "refusal_detected": False,
                "detection_method": "pattern_matching",
            },
        )
