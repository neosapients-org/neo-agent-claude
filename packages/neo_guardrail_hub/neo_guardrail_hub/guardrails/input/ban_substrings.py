"""Ban substrings guardrail for input.

This module provides guardrail protection against specific banned
substrings in user inputs using LLM Guard's BanSubstrings scanner.
"""

from typing import Any, Dict, List, Optional

from ...core.models import GuardrailContext, GuardrailLayer, GuardrailResult
from ..base import GuardrailBase


class BanSubstringsInputGuardrail(GuardrailBase):
    """Detect and block banned substrings in user input.

    Uses LLM Guard's BanSubstrings scanner to identify and optionally
    redact specific substrings that should not appear in prompts.

    Configuration:
        substrings: List of substrings to ban
        match_type: "str" or "word"
            - str: Match substring anywhere in text
            - word: Match only whole words
        case_sensitive: Whether matching is case-sensitive, default False
        redact: Whether to redact banned substrings, default False
        contains_all: If True, fail only if ALL substrings present, default False
        on_fail: Action on failure - "block" or "sanitize", default "block"

    Example:
        guardrail = BanSubstringsInputGuardrail({
            "substrings": ["competitor1", "competitor2"],
            "match_type": "word",
            "case_sensitive": False
        })
        result = await guardrail.check("Tell me about competitor1")
    """

    name = "ban_substrings_input"
    layer = GuardrailLayer.INPUT
    description = "Detect and block banned substrings in input"

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize the ban substrings input guardrail.

        Args:
            config: Configuration dictionary
        """
        super().__init__(config)
        self._scanner = None
        self._substrings = self._config.get("substrings", [])
        self._match_type = self._config.get("match_type", "str")
        self._case_sensitive = self._config.get("case_sensitive", False)
        self._redact = self._config.get("redact", False)
        self._contains_all = self._config.get("contains_all", False)

    async def initialize(self) -> None:
        """Initialize the LLM Guard scanner."""
        if self._initialized:
            return

        if not self._substrings:
            self.logger.warning(
                "no_substrings_configured",
                message="Ban substrings guardrail has no substrings configured",
            )
            await super().initialize()
            return

        try:
            from llm_guard.input_scanners import BanSubstrings
            from llm_guard.input_scanners.ban_substrings import MatchType

            match_type = (
                MatchType.STR
                if self._match_type == "str"
                else MatchType.WORD
            )

            self._scanner = BanSubstrings(
                substrings=self._substrings,
                match_type=match_type,
                case_sensitive=self._case_sensitive,
                redact=self._redact,
                contains_all=self._contains_all,
            )

            self.logger.info(
                "ban_substrings_input_scanner_initialized",
                substring_count=len(self._substrings),
                match_type=self._match_type,
                case_sensitive=self._case_sensitive,
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
        """Check text for banned substrings.

        Args:
            text: Text to check
            context: Optional context (not used)

        Returns:
            GuardrailResult indicating if banned substrings were found
        """
        if not self._substrings:
            # No substrings configured, pass through
            return GuardrailResult(
                passed=True,
                guardrail_name=self.name,
                layer=self.layer,
                risk_score=0.0,
                message=None,
                metadata={"detection_method": "none", "reason": "no_substrings_configured"},
            )

        if self._scanner is None:
            return self._fallback_check(text)

        try:
            sanitized_prompt, is_valid, risk_score = self._scanner.scan(text)

            # Normalize risk score to 0.0-1.0 range
            normalized_risk_score = max(0.0, min(1.0, risk_score))

            if not is_valid:
                self.logger.warning(
                    "banned_substring_detected",
                    risk_score=normalized_risk_score,
                )

            return GuardrailResult(
                passed=is_valid,
                guardrail_name=self.name,
                layer=self.layer,
                risk_score=normalized_risk_score,
                message=(
                    "Banned substring detected"
                    if not is_valid
                    else None
                ),
                sanitized_text=sanitized_prompt if self._redact and not is_valid else None,
                metadata={
                    "detection_method": "llm_guard",
                    "match_type": self._match_type,
                    "case_sensitive": self._case_sensitive,
                    "raw_risk_score": risk_score,
                },
            )

        except Exception as e:
            self.logger.error(
                "ban_substrings_error",
                error=str(e),
            )
            return self._fallback_check(text)

    def _fallback_check(self, text: str) -> GuardrailResult:
        """Fallback substring detection.

        Args:
            text: Text to check

        Returns:
            GuardrailResult with detection results
        """
        self.logger.debug("initializing_guardrail", guardrail=self.name)

        check_text = text if self._case_sensitive else text.lower()
        found_substrings: List[str] = []

        for substring in self._substrings:
            check_substring = substring if self._case_sensitive else substring.lower()

            if self._match_type == "word":
                # Word boundary matching
                import re
                pattern = rf"\b{re.escape(check_substring)}\b"
                if re.search(pattern, check_text, re.IGNORECASE if not self._case_sensitive else 0):
                    found_substrings.append(substring)
            else:
                # String matching
                if check_substring in check_text:
                    found_substrings.append(substring)

        # Determine if check passes
        if self._contains_all:
            has_violation = len(found_substrings) == len(self._substrings)
        else:
            has_violation = len(found_substrings) > 0

        risk_score = min(0.5 * len(found_substrings), 1.0) if has_violation else 0.0
        passed = not has_violation

        # Optionally redact
        sanitized_text = text
        if self._redact and found_substrings:
            for substring in found_substrings:
                if self._case_sensitive:
                    sanitized_text = sanitized_text.replace(substring, "[REDACTED]")
                else:
                    import re
                    pattern = re.compile(re.escape(substring), re.IGNORECASE)
                    sanitized_text = pattern.sub("[REDACTED]", sanitized_text)

        if has_violation:
            self.logger.warning(
                "banned_substring_detected_fallback",
                found_count=len(found_substrings),
            )
        else:
            self.logger.debug("no_banned_substrings_detected")

        return GuardrailResult(
            passed=passed,
            guardrail_name=self.name,
            layer=self.layer,
            risk_score=risk_score,
            message=(
                f"Found {len(found_substrings)} banned substring(s)"
                if has_violation
                else None
            ),
            sanitized_text=sanitized_text if self._redact and has_violation else None,
            metadata={
                "detection_method": "fallback",
                "found_substrings": found_substrings,
                "match_type": self._match_type,
                "case_sensitive": self._case_sensitive,
            },
        )
