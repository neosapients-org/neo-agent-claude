"""Regex pattern guardrail for output.

This module provides guardrail protection for detecting or blocking
content matching custom regex patterns in LLM outputs.
"""

import re
from typing import Any, Dict, List, Optional

from ...core.models import GuardrailContext, GuardrailLayer, GuardrailResult
from ..base import GuardrailBase, extract_prompt_from_context


class RegexOutputGuardrail(GuardrailBase):
    """Detect content matching regex patterns in LLM outputs.

    Uses LLM Guard's Regex output scanner for pattern-based content filtering.

    Configuration:
        patterns: List of regex patterns to match
        is_blocked: If True, block on match; if False, require match (default True)
        match_type: Match type - "search" or "full_match" (default "search")
        redact: Whether to redact matching content (default False)

    Example:
        guardrail = RegexOutputGuardrail({
            "patterns": [r"Bearer [A-Za-z0-9-._~+/]+"],
            "is_blocked": True,
            "redact": True
        })
        result = await guardrail.check(
            "Use this token: Bearer abc123xyz",
            context={"prompt": "How do I authenticate?"}
        )
    """

    name = "regex_output"
    layer = GuardrailLayer.OUTPUT
    description = "Detect content matching regex patterns in LLM output"

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize the regex guardrail.

        Args:
            config: Configuration dictionary
        """
        super().__init__(config)
        self._scanner = None
        self._patterns = self._config.get("patterns", [])
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

        # Compile patterns for fallback
        for pattern in self._patterns:
            try:
                self._compiled_patterns.append(re.compile(pattern))
            except re.error as e:
                self.logger.warning(
                    "invalid_regex_pattern",
                    pattern=pattern,
                    error=str(e),
                )

        try:
            from llm_guard.output_scanners import Regex
            from llm_guard.output_scanners.regex import MatchType

            if self._patterns:
                match_type = (
                    MatchType.SEARCH if self._match_type.lower() == "search"
                    else MatchType.FULL_MATCH
                )

                self._scanner = Regex(
                    patterns=self._patterns,
                    is_blocked=self._is_blocked,
                    match_type=match_type,
                    redact=self._redact,
                )

                self.logger.info(
                    "regex_output_scanner_initialized",
                    num_patterns=len(self._patterns),
                    is_blocked=self._is_blocked,
                    redact=self._redact,
                )
            else:
                self._scanner = None

        except ImportError as e:
            self.logger.warning(
                "llm_guard_not_installed",
                message="Install with: pip install neo-guardrail-hub[llm-guard]",
                error=str(e),
            )
            self._scanner = None
        except Exception as e:
            self.logger.warning(
                "regex_output_scanner_init_failed",
                message="Failed to initialize Regex scanner, will use fallback",
                error=str(e),
            )
            self._scanner = None

        await super().initialize()

    async def _check(
        self,
        text: str,
        context: Optional[GuardrailContext] = None,
    ) -> GuardrailResult:
        """Check for pattern matches in the output.

        Args:
            text: LLM output to check
            context: Optional context with prompt

        Returns:
            GuardrailResult indicating if patterns were matched
        """
        if not self._patterns:
            return GuardrailResult(
                passed=True,
                guardrail_name=self.name,
                layer=self.layer,
                risk_score=0.0,
                message="No patterns configured, skipping check",
                metadata={"skipped": True, "reason": "no_patterns"},
            )

        prompt = extract_prompt_from_context(context)

        if self._scanner is None:
            return self._fallback_check(text)

        try:
            sanitized_text, is_valid, risk_score = self._scanner.scan(prompt, text)

            normalized_risk = max(0.0, min(1.0, risk_score))
            passed = is_valid

            if not passed:
                self.logger.warning(
                    "regex_pattern_matched_in_output",
                    is_blocked=self._is_blocked,
                )

            return GuardrailResult(
                passed=passed,
                guardrail_name=self.name,
                layer=self.layer,
                risk_score=normalized_risk,
                message="Pattern match violation in output" if not passed else None,
                sanitized_text=sanitized_text if self._redact and sanitized_text != text else None,
                metadata={
                    "pattern_matched": not passed if self._is_blocked else passed,
                    "is_blocked": self._is_blocked,
                    "num_patterns": len(self._patterns),
                    "redacted": self._redact and sanitized_text != text,
                    "detection_method": "llm_guard_regex",
                },
            )

        except Exception as e:
            self.logger.error("regex_output_check_error", error=str(e))
            return self._fallback_check(text)

    def _fallback_check(self, text: str) -> GuardrailResult:
        """Fallback check using Python regex.

        Args:
            text: Text to check

        Returns:
            GuardrailResult based on regex matching
        """
        matches: List[str] = []
        
        for pattern in self._compiled_patterns:
            if self._match_type.lower() == "search":
                match = pattern.search(text)
            else:
                match = pattern.fullmatch(text)
            
            if match:
                matches.append(match.group())

        has_match = len(matches) > 0
        
        if self._is_blocked:
            passed = not has_match
        else:
            passed = has_match

        risk_score = 1.0 if not passed else 0.0

        sanitized_text = None
        if self._redact and has_match:
            sanitized_text = text
            for pattern in self._compiled_patterns:
                sanitized_text = pattern.sub("[REDACTED]", sanitized_text)

        return GuardrailResult(
            passed=passed,
            guardrail_name=self.name,
            layer=self.layer,
            risk_score=risk_score,
            message=(
                f"Pattern matched: {matches[:3]}" if has_match and self._is_blocked else None
            ),
            sanitized_text=sanitized_text,
            metadata={
                "pattern_matched": has_match,
                "matches": matches[:5],
                "is_blocked": self._is_blocked,
                "detection_method": "python_regex_fallback",
            },
        )
