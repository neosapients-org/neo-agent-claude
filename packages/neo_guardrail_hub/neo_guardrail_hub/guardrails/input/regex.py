"""Regex pattern matching guardrail for input.

This module provides guardrail protection using custom regex
patterns using LLM Guard's Regex scanner.
"""

import re
from typing import Any, Dict, List, Optional

from ...core.models import GuardrailContext, GuardrailLayer, GuardrailResult
from ..base import GuardrailBase


class RegexInputGuardrail(GuardrailBase):
    """Apply custom regex patterns to validate user input.

    Uses LLM Guard's Regex scanner to match text against custom
    patterns. Can be used to block or require specific patterns.

    Configuration:
        patterns: List of regex patterns to check
        is_blocked: If True, patterns are "bad" (block on match);
                   If False, patterns are "good" (require match)
        match_type: "search" or "full_match", default "search"
            - search: Find pattern anywhere in text
            - full_match: Pattern must match entire text
        redact: Whether to redact matched patterns, default False

    Example:
        # Block Bearer tokens
        guardrail = RegexInputGuardrail({
            "patterns": [r"Bearer [A-Za-z0-9-._~+/]+"],
            "is_blocked": True,
            "redact": True
        })
        result = await guardrail.check("My token is Bearer abc123xyz")
    """

    name = "regex_input"
    layer = GuardrailLayer.INPUT
    description = "Apply custom regex patterns to validate input"

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize the regex input guardrail.

        Args:
            config: Configuration dictionary
        """
        super().__init__(config)
        self._scanner = None
        self._patterns: List[str] = self._config.get("patterns", [])
        self._is_blocked = self._config.get("is_blocked", True)
        self._match_type = self._config.get("match_type", "search")
        self._redact = self._config.get("redact", False)
        self._compiled_patterns: List[re.Pattern] = []

    async def initialize(self) -> None:
        """Initialize the LLM Guard scanner."""
        if self._initialized:
            return

        if not self._patterns:
            self.logger.warning(
                "no_patterns_configured",
                message="Regex guardrail has no patterns configured",
            )
            await super().initialize()
            return

        try:
            from llm_guard.input_scanners import Regex
            from llm_guard.input_scanners.regex import MatchType

            match_type = (
                MatchType.SEARCH
                if self._match_type == "search"
                else MatchType.FULL_MATCH
            )

            self._scanner = Regex(
                patterns=self._patterns,
                is_blocked=self._is_blocked,
                match_type=match_type,
                redact=self._redact,
            )

            self.logger.info(
                "regex_input_scanner_initialized",
                pattern_count=len(self._patterns),
                is_blocked=self._is_blocked,
                match_type=self._match_type,
            )

        except ImportError:
            self.logger.warning(
                "llm_guard_not_installed",
                message="Install with: pip install neo-guardrail-hub[llm-guard]",
            )
            self._scanner = None

            # Compile patterns for fallback
            for pattern in self._patterns:
                try:
                    self._compiled_patterns.append(re.compile(pattern))
                except re.error as e:
                    self.logger.error(
                        "invalid_regex_pattern",
                        pattern=pattern,
                        error=str(e),
                    )

        await super().initialize()

    async def _check(
        self,
        text: str,
        context: Optional[GuardrailContext] = None,
    ) -> GuardrailResult:
        """Check text against regex patterns.

        Args:
            text: Text to check
            context: Optional context (not used)

        Returns:
            GuardrailResult indicating if patterns were matched
        """
        if not self._patterns:
            return GuardrailResult(
                passed=True,
                guardrail_name=self.name,
                layer=self.layer,
                risk_score=0.0,
                message=None,
                metadata={"detection_method": "none", "reason": "no_patterns_configured"},
            )

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
                    "Regex pattern matched"
                    if not is_valid
                    else None
                ),
                sanitized_text=sanitized_prompt if self._redact and not is_valid else None,
                metadata={
                    "detection_method": "llm_guard",
                    "is_blocked": self._is_blocked,
                    "raw_risk_score": risk_score,
                },
            )

        except Exception as e:
            self.logger.error(
                "regex_check_error",
                error=str(e),
            )
            return self._fallback_check(text)

    def _fallback_check(self, text: str) -> GuardrailResult:
        """Fallback regex matching.

        Args:
            text: Text to check

        Returns:
            GuardrailResult with match results
        """
        matched_patterns: List[str] = []
        sanitized_text = text

        # Compile patterns if not already done
        if not self._compiled_patterns:
            for pattern in self._patterns:
                try:
                    self._compiled_patterns.append(re.compile(pattern))
                except re.error:
                    pass

        for i, compiled_pattern in enumerate(self._compiled_patterns):
            if self._match_type == "full_match":
                match = compiled_pattern.fullmatch(text)
            else:
                match = compiled_pattern.search(text)

            if match:
                matched_patterns.append(self._patterns[i])
                if self._redact:
                    sanitized_text = compiled_pattern.sub("[REDACTED]", sanitized_text)

        has_match = len(matched_patterns) > 0

        # Determine validity based on is_blocked setting
        if self._is_blocked:
            # Block mode: fail if any pattern matches
            is_valid = not has_match
            message = f"Blocked pattern(s) matched" if has_match else None
        else:
            # Allow mode: fail if no pattern matches
            is_valid = has_match
            message = "Required pattern not matched" if not has_match else None

        risk_score = 0.7 if not is_valid else 0.0

        return GuardrailResult(
            passed=is_valid,
            guardrail_name=self.name,
            layer=self.layer,
            risk_score=risk_score,
            message=message,
            sanitized_text=sanitized_text if self._redact and matched_patterns else None,
            metadata={
                "detection_method": "regex_fallback",
                "is_blocked": self._is_blocked,
                "matched_patterns_count": len(matched_patterns),
            },
        )

    async def cleanup(self) -> None:
        """Clean up scanner resources."""
        self._scanner = None
        self._compiled_patterns = []
        await super().cleanup()
