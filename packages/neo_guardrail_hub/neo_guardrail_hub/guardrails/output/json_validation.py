"""JSON validation guardrail for output.

This module provides guardrail protection for validating JSON structures
in LLM outputs and optionally repairing malformed JSON.
"""

import json
import re
from typing import Any, Dict, Optional

from ...core.models import GuardrailContext, GuardrailLayer, GuardrailResult
from ..base import GuardrailBase, extract_prompt_from_context


class JSONValidationOutputGuardrail(GuardrailBase):
    """Validate JSON structures in LLM outputs.

    Uses LLM Guard's JSON output scanner to detect, validate, and repair
    JSON structures in output text.

    Configuration:
        required_elements: Minimum number of valid JSON objects required (default 0)
        repair: Whether to attempt JSON repair (default True)

    Example:
        guardrail = JSONValidationOutputGuardrail({
            "required_elements": 1,
            "repair": True
        })
        result = await guardrail.check(
            '{"name": "John", "age": 30}',
            context={"prompt": "Return user data as JSON"}
        )
    """

    name = "json_validation"
    layer = GuardrailLayer.OUTPUT
    description = "Validate JSON structures in LLM output"

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize the JSON validation guardrail.

        Args:
            config: Configuration dictionary
        """
        super().__init__(config)
        self._scanner = None
        self._required_elements = self._config.get("required_elements", 0)
        self._repair = self._config.get("repair", True)

    async def initialize(self) -> None:
        """Initialize the LLM Guard scanner."""
        if self._initialized:
            return

        try:
            from llm_guard.output_scanners import JSON

            self._scanner = JSON(
                required_elements=self._required_elements,
                repair=self._repair,
            )

            self.logger.info(
                "json_validation_scanner_initialized",
                required_elements=self._required_elements,
                repair=self._repair,
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
                "json_validation_scanner_init_failed",
                message="Failed to initialize JSON scanner, will use fallback",
                error=str(e),
            )
            self._scanner = None

        await super().initialize()

    async def _check(
        self,
        text: str,
        context: Optional[GuardrailContext] = None,
    ) -> GuardrailResult:
        """Check for valid JSON in the output.

        Args:
            text: LLM output to check
            context: Optional context with prompt

        Returns:
            GuardrailResult indicating JSON validation status
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
                    "json_validation_failed",
                    required_elements=self._required_elements,
                )

            return GuardrailResult(
                passed=passed,
                guardrail_name=self.name,
                layer=self.layer,
                risk_score=normalized_risk,
                message="JSON validation failed" if not passed else None,
                sanitized_text=sanitized_text if sanitized_text != text else None,
                metadata={
                    "json_valid": passed,
                    "required_elements": self._required_elements,
                    "repaired": sanitized_text != text,
                    "detection_method": "llm_guard_json",
                },
            )

        except Exception as e:
            self.logger.error("json_validation_error", error=str(e))
            return self._fallback_check(text)

    def _fallback_check(self, text: str) -> GuardrailResult:
        """Fallback check using Python's json library.

        Args:
            text: Text to check

        Returns:
            GuardrailResult based on JSON parsing
        """
        # Try to find JSON objects in the text
        json_pattern = r'\{[^{}]*\}'
        found_jsons = re.findall(json_pattern, text, re.DOTALL)
        
        valid_jsons = 0
        invalid_jsons = 0
        
        for potential_json in found_jsons:
            try:
                json.loads(potential_json)
                valid_jsons += 1
            except json.JSONDecodeError:
                invalid_jsons += 1

        # Also try to parse the entire text as JSON
        try:
            json.loads(text)
            valid_jsons += 1
        except json.JSONDecodeError:
            pass

        # Check if we have enough valid JSON elements
        has_required = valid_jsons >= self._required_elements
        passed = has_required if self._required_elements > 0 else True
        
        risk_score = 0.0 if passed else 1.0

        return GuardrailResult(
            passed=passed,
            guardrail_name=self.name,
            layer=self.layer,
            risk_score=risk_score,
            message=(
                f"JSON validation: found {valid_jsons} valid, {invalid_jsons} invalid"
                if not passed else None
            ),
            metadata={
                "json_valid": passed,
                "valid_json_count": valid_jsons,
                "invalid_json_count": invalid_jsons,
                "required_elements": self._required_elements,
                "detection_method": "python_json_fallback",
            },
        )
