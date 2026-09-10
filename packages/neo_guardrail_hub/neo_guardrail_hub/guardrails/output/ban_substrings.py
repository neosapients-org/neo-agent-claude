"""Ban substrings guardrail for output.

This module provides guardrail protection for detecting and blocking
specific substrings or words in LLM outputs.
"""

import re
from typing import Any, Dict, List, Optional

from ...core.models import GuardrailContext, GuardrailLayer, GuardrailResult
from ..base import GuardrailBase, extract_prompt_from_context


class BanSubstringsOutputGuardrail(GuardrailBase):
    """Detect and block specific substrings in LLM outputs.

    Uses LLM Guard's BanSubstrings output scanner to detect forbidden
    strings, words, or patterns in model outputs.

    Configuration:
        substrings: List of strings to ban (required)
        match_type: How to match - "STR" (substring) or "WORD" (word boundary) (default "STR")
        case_sensitive: Whether matching is case-sensitive (default False)
        redact: Whether to redact found substrings instead of blocking (default False)
        contains_all: If True, all substrings must be present to trigger (default False)

    Example:
        guardrail = BanSubstringsOutputGuardrail({
            "substrings": ["confidential", "internal only", "do not share"],
            "match_type": "WORD",
            "case_sensitive": False
        })
        result = await guardrail.check("This is confidential information.")
    """

    name = "ban_substrings_output"
    layer = GuardrailLayer.OUTPUT
    description = "Block outputs containing banned substrings"

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize the ban substrings output guardrail.

        Args:
            config: Configuration dictionary
        """
        super().__init__(config)
        self._scanner = None
        self._substrings: List[str] = self._config.get("substrings", [])
        self._match_type = self._config.get("match_type", "STR")
        self._case_sensitive = self._config.get("case_sensitive", False)
        self._redact = self._config.get("redact", False)
        self._contains_all = self._config.get("contains_all", False)
        self._compiled_patterns: List[re.Pattern] = []

    async def initialize(self) -> None:
        """Initialize the LLM Guard scanner."""
        if self._initialized:
            return

        # Compile fallback patterns
        flags = 0 if self._case_sensitive else re.IGNORECASE
        for substring in self._substrings:
            if self._match_type.upper() == "WORD":
                # Word boundary matching
                pattern = rf"\b{re.escape(substring)}\b"
            else:
                # Substring matching
                pattern = re.escape(substring)
            self._compiled_patterns.append(re.compile(pattern, flags))

        if not self._substrings:
            self.logger.warning(
                "no_substrings_configured",
                message="No substrings configured for ban_substrings_output guardrail",
            )

        try:
            from llm_guard.output_scanners import BanSubstrings
            from llm_guard.output_scanners.ban_substrings import MatchType

            # Determine match type
            match_type = (
                MatchType.WORD
                if self._match_type.upper() == "WORD"
                else MatchType.STR
            )

            self._scanner = BanSubstrings(
                substrings=self._substrings,
                match_type=match_type,
                case_sensitive=self._case_sensitive,
                redact=self._redact,
                contains_all=self._contains_all,
            )

            self.logger.info(
                "ban_substrings_output_scanner_initialized",
                substring_count=len(self._substrings),
                match_type=self._match_type,
                case_sensitive=self._case_sensitive,
                redact=self._redact,
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
        """Check if output contains banned substrings.

        Args:
            text: LLM output text to check
            context: Optional context

        Returns:
            GuardrailResult indicating if banned substrings were found
        """
        if not self._substrings:
            return GuardrailResult(
                passed=True,
                guardrail_name=self.name,
                layer=self.layer,
                risk_score=0.0,
                message="No substrings configured",
                metadata={
                    "skipped": True,
                    "reason": "no_substrings_configured",
                },
            )

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

            # is_valid indicates if no banned substrings were found
            banned_found = not is_valid

            # Determine what was found/redacted
            found_substrings = []
            if sanitized_text != text:
                found_substrings = self._detect_substrings(text)

            result_kwargs = {
                "passed": not banned_found,
                "guardrail_name": self.name,
                "layer": self.layer,
                "risk_score": normalized_risk_score,
                "message": (
                    f"Banned substring(s) detected: {', '.join(found_substrings[:3])}"
                    if banned_found and found_substrings
                    else ("Banned substring detected" if banned_found else None)
                ),
                "metadata": {
                    "banned_found": banned_found,
                    "found_substrings": found_substrings[:10],
                    "configured_count": len(self._substrings),
                    "detection_method": "llm_guard",
                    "raw_risk_score": risk_score,
                },
            }

            # If redact mode is on, include sanitized text
            if self._redact and sanitized_text != text:
                result_kwargs["sanitized_text"] = sanitized_text
                result_kwargs["passed"] = True  # Allow through with sanitized text

            return GuardrailResult(**result_kwargs)

        except Exception as e:
            self.logger.error(
                "ban_substrings_output_check_error",
                error=str(e),
            )
            # Fall back to pattern matching on error
            return self._fallback_check(text)

    def _detect_substrings(self, text: str) -> List[str]:
        """Detect which banned substrings are present.

        Args:
            text: Text to search

        Returns:
            List of found substrings
        """
        found = []
        for i, pattern in enumerate(self._compiled_patterns):
            if pattern.search(text):
                found.append(self._substrings[i])
        return found

    def _fallback_check(self, text: str) -> GuardrailResult:
        """Fallback check using regex patterns.

        Args:
            text: Text to check for banned substrings

        Returns:
            GuardrailResult based on pattern matching
        """
        found_substrings = []
        match_details = []

        for i, pattern in enumerate(self._compiled_patterns):
            matches = pattern.findall(text)
            if matches:
                found_substrings.append(self._substrings[i])
                match_details.append({
                    "substring": self._substrings[i],
                    "match_count": len(matches),
                })

        # Check contains_all mode
        if self._contains_all:
            all_present = len(found_substrings) == len(self._substrings)
            banned_found = all_present
        else:
            banned_found = len(found_substrings) > 0

        if banned_found:
            # Calculate risk based on number of matches
            total_matches = sum(m["match_count"] for m in match_details)
            risk_score = min(0.95, 0.5 + (len(found_substrings) * 0.1) + (total_matches * 0.02))

            # Handle redact mode
            sanitized_text = None
            if self._redact:
                sanitized_text = text
                for pattern in self._compiled_patterns:
                    sanitized_text = pattern.sub("[REDACTED]", sanitized_text)

            result_kwargs = {
                "passed": False if not self._redact else True,
                "guardrail_name": self.name,
                "layer": self.layer,
                "risk_score": risk_score,
                "message": f"Banned substring(s) detected: {', '.join(found_substrings[:3])}",
                "metadata": {
                    "banned_found": True,
                    "found_substrings": found_substrings,
                    "match_details": match_details[:10],
                    "configured_count": len(self._substrings),
                    "detection_method": "pattern_matching",
                    "contains_all_mode": self._contains_all,
                },
            }

            if sanitized_text:
                result_kwargs["sanitized_text"] = sanitized_text

            return GuardrailResult(**result_kwargs)

        return GuardrailResult(
            passed=True,
            guardrail_name=self.name,
            layer=self.layer,
            risk_score=0.0,
            message=None,
            metadata={
                "banned_found": False,
                "configured_count": len(self._substrings),
                "detection_method": "pattern_matching",
            },
        )
