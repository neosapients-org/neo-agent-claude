"""Ban competitors guardrail for input.

This module provides guardrail protection against competitor mentions
in user inputs using LLM Guard's BanCompetitors scanner.
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from ...core.models import GuardrailContext, GuardrailLayer, GuardrailResult
from ..base import GuardrailBase


class BanCompetitorsInputGuardrail(GuardrailBase):
    """Detect and block competitor mentions in user input.

    Uses LLM Guard's BanCompetitors scanner with NER to identify
    organization entities and check them against a competitor list.

    Configuration:
        competitors: List of competitor names to block
        threshold: Detection threshold (0.0-1.0), default 0.5
        redact: Whether to redact competitor names, default False
        use_onnx: Whether to use ONNX runtime for faster inference, default True

    Example:
        guardrail = BanCompetitorsInputGuardrail({
            "competitors": ["CompetitorA", "CompetitorB", "Rival Inc"],
            "threshold": 0.5,
            "redact": False
        })
        result = await guardrail.check("Tell me about CompetitorA's products")
    """

    name = "ban_competitors_input"
    layer = GuardrailLayer.INPUT
    description = "Detect and block competitor mentions in input"

    def __init__(self, config: Optional[Dict[str, Any]] = None, models_dir: Optional[str] = None) -> None:
        """Initialize the ban competitors input guardrail.

        Args:
            config: Configuration dictionary
            models_dir: Optional directory containing local models
        """
        super().__init__(config)
        self._scanner = None
        self._competitors: List[str] = self._config.get("competitors", [])
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
            await super().initialize()
            return

        try:
            from llm_guard.input_scanners import BanCompetitors
            from llm_guard.input_scanners.ban_competitors import MODEL_V1

            model_config = None
            
            # Check if local models are available
            if self._models_dir:
                local_model_path = Path(self._models_dir) / "nuner-v1_orgs"
                
                if local_model_path.exists():
                    self.logger.info(
                        "ban_competitors_using_local_model",
                        model_path=str(local_model_path)
                    )
                    
                    # Configure MODEL_V1 to use local path
                    MODEL_V1.path = str(local_model_path)
                    MODEL_V1.kwargs["local_files_only"] = True
                    model_config = MODEL_V1
                else:
                    self.logger.warning(
                        "ban_competitors_local_model_not_found",
                        expected_path=str(local_model_path),
                        message="Will download from HuggingFace (slower). Run: neo-guardrail download-models --scanners ban_competitors"
                    )

            # Initialize scanner with model config if available
            if model_config:
                self._scanner = BanCompetitors(
                    competitors=self._competitors,
                    threshold=self._threshold,
                    redact=self._redact,
                    use_onnx=self._use_onnx,
                    model=model_config
                )
            else:
                self._scanner = BanCompetitors(
                    competitors=self._competitors,
                    threshold=self._threshold,
                    redact=self._redact,
                    use_onnx=self._use_onnx,
                )

            self.logger.info(
                "ban_competitors_input_scanner_initialized",
                competitor_count=len(self._competitors),
                threshold=self._threshold,
                redact=self._redact,
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
        """Check text for competitor mentions.

        Args:
            text: Text to check
            context: Optional context (not used)

        Returns:
            GuardrailResult indicating if competitors were detected
        """
        if not self._competitors:
            return GuardrailResult(
                passed=True,
                guardrail_name=self.name,
                layer=self.layer,
                risk_score=0.0,
                message=None,
                metadata={"detection_method": "none", "reason": "no_competitors_configured"},
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
                    "Competitor mention detected"
                    if not is_valid
                    else None
                ),
                sanitized_text=sanitized_prompt if self._redact and not is_valid else None,
                metadata={
                    "detection_method": "llm_guard",
                    "raw_risk_score": risk_score,
                },
            )

        except Exception as e:
            self.logger.error(
                "ban_competitors_error",
                error=str(e),
            )
            return self._fallback_check(text)

    def _fallback_check(self, text: str) -> GuardrailResult:
        """Fallback competitor detection using string matching.

        Args:
            text: Text to check

        Returns:
            GuardrailResult with detection results
        """
        text_lower = text.lower()
        found_competitors: List[str] = []

        for competitor in self._competitors:
            if competitor.lower() in text_lower:
                found_competitors.append(competitor)

        has_competitor = len(found_competitors) > 0
        risk_score = min(0.5 * len(found_competitors), 1.0) if has_competitor else 0.0

        # Optionally redact
        sanitized_text = text
        if self._redact and found_competitors:
            import re
            for competitor in found_competitors:
                pattern = re.compile(re.escape(competitor), re.IGNORECASE)
                sanitized_text = pattern.sub("[COMPETITOR]", sanitized_text)

        return GuardrailResult(
            passed=not has_competitor,
            guardrail_name=self.name,
            layer=self.layer,
            risk_score=risk_score,
            message=(
                f"Competitor mention detected: {', '.join(found_competitors)}"
                if has_competitor
                else None
            ),
            sanitized_text=sanitized_text if self._redact and has_competitor else None,
            metadata={
                "detection_method": "string_matching",
                "found_competitors": found_competitors,
            },
        )

    async def cleanup(self) -> None:
        """Clean up scanner resources."""
        self._scanner = None
        await super().cleanup()
