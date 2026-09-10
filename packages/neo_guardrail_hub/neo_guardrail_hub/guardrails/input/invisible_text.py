"""Invisible text detection guardrail for input.

This module provides guardrail protection against invisible/hidden
Unicode characters in user inputs using LLM Guard's InvisibleText scanner.
"""

import unicodedata
from typing import Any, Dict, Optional

from ...core.models import GuardrailContext, GuardrailLayer, GuardrailResult
from ..base import GuardrailBase


class InvisibleTextInputGuardrail(GuardrailBase):
    """Detect and remove invisible Unicode characters from user input.

    Uses LLM Guard's InvisibleText scanner to identify hidden characters
    that could be used for steganography or prompt injection attacks.
    Detects zero-width characters, Private Use Area characters, and
    other non-printable Unicode.

    Configuration:
        threshold: Detection threshold (0.0-1.0), default 0.0
            (Any invisible text is flagged by default)

    Example:
        guardrail = InvisibleTextInputGuardrail()
        result = await guardrail.check("Hello\\u200bWorld")  # Contains zero-width space
    """

    name = "invisible_text_input"
    layer = GuardrailLayer.INPUT
    description = "Detect and remove invisible Unicode characters"

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize the invisible text input guardrail.

        Args:
            config: Configuration dictionary
        """
        super().__init__(config)
        self._scanner = None
        # Default threshold to 0 since any invisible text is suspicious
        self._threshold = self._config.get("threshold", 0.0)

    async def initialize(self) -> None:
        """Initialize the LLM Guard scanner."""
        if self._initialized:
            return

        try:
            from llm_guard.input_scanners import InvisibleText

            self._scanner = InvisibleText()

            self.logger.info(
                "invisible_text_input_scanner_initialized",
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
        """Check text for invisible characters.

        Args:
            text: Text to check
            context: Optional context (not used)

        Returns:
            GuardrailResult indicating if invisible text was detected
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
                    "Invisible Unicode characters detected"
                    if not is_valid
                    else None
                ),
                sanitized_text=sanitized_prompt if not is_valid else None,
                metadata={
                    "detection_method": "llm_guard",
                    "original_length": len(text),
                    "sanitized_length": len(sanitized_prompt),
                    "raw_risk_score": risk_score,
                },
            )

        except Exception as e:
            self.logger.error(
                "invisible_text_detection_error",
                error=str(e),
            )
            return self._fallback_check(text)

    def _fallback_check(self, text: str) -> GuardrailResult:
        """Fallback invisible text detection using Unicode inspection.

        Args:
            text: Text to check

        Returns:
            GuardrailResult with detection results
        """
        invisible_chars = []
        sanitized_chars = []

        # Categories of invisible/problematic characters
        # Cf: Format characters
        # Cc: Control characters  
        # Co: Private use characters
        # Cn: Unassigned characters
        problematic_categories = {'Cf', 'Cc', 'Co', 'Cn'}

        # Zero-width and invisible character ranges
        invisible_ranges = [
            (0x200B, 0x200F),  # Zero-width space, zero-width joiner, etc.
            (0x2028, 0x202F),  # Line/paragraph separators, special spaces
            (0x2060, 0x206F),  # Word joiner, invisible operators
            (0xFEFF, 0xFEFF),  # Byte order mark
            (0xE000, 0xF8FF),  # Private Use Area (BMP)
            (0xF0000, 0xFFFFD),  # Supplementary Private Use Area-A
            (0x100000, 0x10FFFD),  # Supplementary Private Use Area-B
        ]

        def is_invisible(char: str) -> bool:
            """Check if character is invisible/hidden."""
            code_point = ord(char)
            category = unicodedata.category(char)

            # Check category
            if category in problematic_categories:
                return True

            # Check specific ranges
            for start, end in invisible_ranges:
                if start <= code_point <= end:
                    return True

            return False

        for char in text:
            if is_invisible(char):
                invisible_chars.append((char, f"U+{ord(char):04X}", unicodedata.category(char)))
            else:
                sanitized_chars.append(char)

        has_invisible = len(invisible_chars) > 0
        sanitized_text = ''.join(sanitized_chars)

        # Risk score based on proportion of invisible characters
        risk_score = len(invisible_chars) / max(len(text), 1) if has_invisible else 0.0
        risk_score = min(risk_score * 10, 1.0)  # Amplify since even few chars are suspicious

        return GuardrailResult(
            passed=not has_invisible,
            guardrail_name=self.name,
            layer=self.layer,
            risk_score=risk_score,
            message=(
                f"Found {len(invisible_chars)} invisible character(s)"
                if has_invisible
                else None
            ),
            sanitized_text=sanitized_text if has_invisible else None,
            metadata={
                "detection_method": "unicode_inspection",
                "invisible_char_count": len(invisible_chars),
                "original_length": len(text),
                "sanitized_length": len(sanitized_text),
                "invisible_chars": [
                    {"char": c[1], "category": c[2]}
                    for c in invisible_chars[:10]  # Limit to first 10 for metadata
                ],
            },
        )

    async def cleanup(self) -> None:
        """Clean up scanner resources."""
        self._scanner = None
        await super().cleanup()
