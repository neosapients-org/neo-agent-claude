"""Ban competitors guardrail for output.

This module provides guardrail protection for detecting and blocking
competitor mentions in LLM outputs.
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from ...core.models import GuardrailContext, GuardrailLayer, GuardrailResult
from ..base import GuardrailBase, extract_prompt_from_context


class BanCompetitorsOutputGuardrail(GuardrailBase):
    """Detect and block competitor mentions in LLM outputs.

    Uses LLM Guard's BanCompetitors output scanner with NER-based detection
    using the guishe/nuner-v1_orgs model.

    Configuration:
        threshold: Detection threshold (0.0-1.0, default 0.5)
        competitors: List of competitor names to block
        redact: Whether to redact competitor names (default False)
        use_onnx: Whether to use ONNX runtime (default True)

    Example:
        guardrail = BanCompetitorsOutputGuardrail({
            "competitors": ["Competitor Inc", "Rival Corp"],
            "redact": False
        })
        result = await guardrail.check(
            "You should consider Competitor Inc for this service.",
            context={"prompt": "Recommend a service provider"}
        )
    """

    name = "ban_competitors_output"
    layer = GuardrailLayer.OUTPUT
    description = "Detect and block competitor mentions in LLM output"

    def __init__(self, config: Optional[Dict[str, Any]] = None, models_dir: Optional[str] = None) -> None:
        """Initialize the ban competitors guardrail.

        Args:
            config: Configuration dictionary
            models_dir: Optional path to directory containing local models
        """
        super().__init__(config)
        self._scanner = None
        self._competitors = self._config.get("competitors", [])
        self._redact = self._config.get("redact", False)
        self._use_onnx = self._config.get("use_onnx", True)
        self._models_dir = models_dir or os.getenv("NEO_LLM_GUARD_MODELS_DIR")

    async def initialize(self) -> None:
        """Initialize the LLM Guard scanner."""
        if self._initialized:
            return

        if not self._competitors:
            self.logger.warning(
                "no_competitors_configured",
                message="Ban competitors guardrail has no competitors configured",
            )

        try:
            from llm_guard.output_scanners import BanCompetitors
            from llm_guard.model import Model as LLMGuardModel

            # Configure local model if available
            model_config = None
            if self._models_dir:
                local_model_path = Path(self._models_dir) / "nuner-v1_orgs"
                if local_model_path.exists():
                    self.logger.info(
                        "ban_competitors_output_using_local_model",
                        model_path=str(local_model_path)
                    )
                    # Use the same MODEL_V1 from input scanner
                    from llm_guard.input_scanners.ban_competitors import MODEL_V1
                    MODEL_V1.path = str(local_model_path)
                    MODEL_V1.kwargs["local_files_only"] = True
                    model_config = MODEL_V1

            if self._competitors:
                if model_config:
                    self._scanner = BanCompetitors(
                        competitors=self._competitors,
                        threshold=self._threshold,
                        redact=self._redact,
                        use_onnx=self._use_onnx,
                        model=model_config,
                    )
                else:
                    self._scanner = BanCompetitors(
                        competitors=self._competitors,
                        threshold=self._threshold,
                        redact=self._redact,
                        use_onnx=self._use_onnx,
                    )

                self.logger.info(
                    "ban_competitors_output_scanner_initialized",
                    num_competitors=len(self._competitors),
                    redact=self._redact,
                    threshold=self._threshold,
                    using_local_model=model_config is not None,
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
                "ban_competitors_output_scanner_init_failed",
                message="Failed to initialize BanCompetitors scanner, will use fallback",
                error=str(e),
            )
            self._scanner = None

        await super().initialize()

    async def _check(
        self,
        text: str,
        context: Optional[GuardrailContext] = None,
    ) -> GuardrailResult:
        """Check for competitor mentions in the output.

        Args:
            text: LLM output to check
            context: Optional context with prompt

        Returns:
            GuardrailResult indicating if competitors were detected
        """
        if not self._competitors:
            return GuardrailResult(
                passed=True,
                guardrail_name=self.name,
                layer=self.layer,
                risk_score=0.0,
                message="No competitors configured, skipping check",
                metadata={"skipped": True, "reason": "no_competitors"},
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
                    "competitor_detected_in_output",
                    risk_score=normalized_risk,
                )

            return GuardrailResult(
                passed=passed,
                guardrail_name=self.name,
                layer=self.layer,
                risk_score=normalized_risk,
                message="Competitor mention detected in output" if not passed else None,
                sanitized_text=sanitized_text if self._redact else None,
                metadata={
                    "competitor_detected": not passed,
                    "num_competitors_configured": len(self._competitors),
                    "redacted": self._redact and not passed,
                    "detection_method": "ner_model",
                },
            )

        except Exception as e:
            self.logger.error("ban_competitors_output_error", error=str(e))
            return self._fallback_check(text)

    def _fallback_check(self, text: str) -> GuardrailResult:
        """Fallback check using string matching.

        Args:
            text: Text to check

        Returns:
            GuardrailResult based on string presence
        """
        text_lower = text.lower()
        found_competitors: List[str] = []

        for competitor in self._competitors:
            if competitor.lower() in text_lower:
                found_competitors.append(competitor)

        passed = len(found_competitors) == 0
        risk_score = 1.0 if found_competitors else 0.0

        return GuardrailResult(
            passed=passed,
            guardrail_name=self.name,
            layer=self.layer,
            risk_score=risk_score,
            message=(
                f"Competitor mentions found: {found_competitors}" if not passed else None
            ),
            metadata={
                "competitor_detected": not passed,
                "found_competitors": found_competitors,
                "detection_method": "string_fallback",
            },
        )
