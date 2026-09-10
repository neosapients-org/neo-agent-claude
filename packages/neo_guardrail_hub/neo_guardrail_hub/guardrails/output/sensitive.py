"""Sensitive data detection guardrail for output.

This module provides guardrail protection for detecting and redacting
sensitive or personally identifiable information in LLM outputs.
"""

from typing import Any, Dict, List, Optional

from ...core.models import GuardrailContext, GuardrailLayer, GuardrailResult
from ..base import GuardrailBase, extract_prompt_from_context


class SensitiveDataOutputGuardrail(GuardrailBase):
    """Detect and redact sensitive data in LLM outputs.

    Uses LLM Guard's Sensitive output scanner with NER-based detection
    to identify PII and other sensitive information.

    Configuration:
        threshold: Detection threshold (0.0-1.0, default 0.5)
        entity_types: List of entity types to detect (default common PII types)
        redact: Whether to redact detected entities (default True)
        use_onnx: Whether to use ONNX runtime (default True)

    Supported entity types:
        PERSON, EMAIL, PHONE_NUMBER, CREDIT_CARD, US_SSN, IP_ADDRESS, URL, etc.

    Example:
        guardrail = SensitiveDataOutputGuardrail({
            "entity_types": ["PERSON", "EMAIL", "PHONE_NUMBER"],
            "redact": True
        })
        result = await guardrail.check(
            "Contact John Smith at john@example.com or 555-1234",
            context={"prompt": "Who can I contact?"}
        )
    """

    name = "sensitive_data"
    layer = GuardrailLayer.OUTPUT
    description = "Detect and redact sensitive data in LLM output"

    DEFAULT_ENTITY_TYPES = [
        "PERSON",
        "EMAIL",
        "PHONE_NUMBER",
        "CREDIT_CARD",
        "US_SSN",
        "IP_ADDRESS",
        "URL",
    ]

    # Patterns for fallback detection
    FALLBACK_PATTERNS = {
        "email": r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
        "phone": r'\b(?:\+?1[-.\s]?)?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}\b',
        "ssn": r'\b\d{3}[-]?\d{2}[-]?\d{4}\b',
        "credit_card": r'\b(?:\d{4}[-\s]?){3}\d{4}\b',
        "ip_address": r'\b(?:\d{1,3}\.){3}\d{1,3}\b',
    }

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize the sensitive data guardrail.

        Args:
            config: Configuration dictionary
        """
        super().__init__(config)
        self._scanner = None
        self._entity_types = self._config.get("entity_types", self.DEFAULT_ENTITY_TYPES)
        self._redact = self._config.get("redact", True)
        self._use_onnx = self._config.get("use_onnx", True)

    async def initialize(self) -> None:
        """Initialize the LLM Guard scanner."""
        if self._initialized:
            return

        try:
            from llm_guard.output_scanners import Sensitive

            self._scanner = Sensitive(
                entity_types=self._entity_types,
                redact=self._redact,
                threshold=self._threshold,
                use_onnx=self._use_onnx,
            )

            self.logger.info(
                "sensitive_data_scanner_initialized",
                entity_types=self._entity_types,
                redact=self._redact,
                threshold=self._threshold,
            )

        except ImportError as e:
            self.logger.warning(
                "llm_guard_not_installed",
                message="Install with: pip install neo-guardrail-hub[llm-guard]",
                error=str(e),
            )
            self._scanner = None
        except Exception as e:
            self.logger.warning(
                "sensitive_data_scanner_init_failed",
                message="Failed to initialize Sensitive scanner, will use fallback",
                error=str(e),
            )
            self._scanner = None

        await super().initialize()

    async def _check(
        self,
        text: str,
        context: Optional[GuardrailContext] = None,
    ) -> GuardrailResult:
        """Check for sensitive data in the output.

        Args:
            text: LLM output to check
            context: Optional context with prompt

        Returns:
            GuardrailResult indicating if sensitive data was detected
        """
        prompt = extract_prompt_from_context(context)

        if self._scanner is None:
            return self._fallback_check(text)

        try:
            sanitized_text, is_valid, risk_score = self._scanner.scan(prompt, text)

            normalized_risk = max(0.0, min(1.0, risk_score))
            passed = is_valid

            if not passed:
                self.logger.warning(
                    "sensitive_data_detected_in_output",
                    risk_score=normalized_risk,
                )

            return GuardrailResult(
                passed=passed,
                guardrail_name=self.name,
                layer=self.layer,
                risk_score=normalized_risk,
                message="Sensitive data detected in output" if not passed else None,
                sanitized_text=sanitized_text if self._redact and sanitized_text != text else None,
                metadata={
                    "sensitive_data_detected": not passed,
                    "entity_types": self._entity_types,
                    "redacted": self._redact and sanitized_text != text,
                    "detection_method": "ner_model",
                },
            )

        except Exception as e:
            self.logger.error("sensitive_data_check_error", error=str(e))
            return self._fallback_check(text)

    def _fallback_check(self, text: str) -> GuardrailResult:
        """Fallback check using regex patterns.

        Args:
            text: Text to check

        Returns:
            GuardrailResult based on pattern matching
        """
        import re
        
        found_entities: Dict[str, List[str]] = {}
        
        for entity_type, pattern in self.FALLBACK_PATTERNS.items():
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                found_entities[entity_type] = matches

        has_sensitive = len(found_entities) > 0
        passed = not has_sensitive
        risk_score = min(len(found_entities) * 0.3, 1.0) if has_sensitive else 0.0

        sanitized_text = None
        if self._redact and has_sensitive:
            sanitized_text = text
            for entity_type, pattern in self.FALLBACK_PATTERNS.items():
                sanitized_text = re.sub(
                    pattern, f"[{entity_type.upper()}_REDACTED]", sanitized_text
                )

        return GuardrailResult(
            passed=passed,
            guardrail_name=self.name,
            layer=self.layer,
            risk_score=risk_score,
            message=(
                f"Sensitive data found: {list(found_entities.keys())}" if not passed else None
            ),
            sanitized_text=sanitized_text,
            metadata={
                "sensitive_data_detected": has_sensitive,
                "found_types": list(found_entities.keys()),
                "detection_method": "regex_fallback",
            },
        )
