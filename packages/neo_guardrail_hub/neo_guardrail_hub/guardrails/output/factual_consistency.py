"""Factual consistency guardrail for output.

This module provides guardrail protection for detecting when an LLM output
contradicts or is inconsistent with the given context/prompt.
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from ...core.models import GuardrailContext, GuardrailLayer, GuardrailResult
from ..base import GuardrailBase, extract_prompt_from_context


class FactualConsistencyGuardrail(GuardrailBase):
    """Detect when LLM output is inconsistent with the given context.

    Uses LLM Guard's FactualConsistency scanner with NLI (Natural Language
    Inference) to detect contradictions between the prompt/context and output.
    Uses MoritzLaurer/deberta-v3-base-zeroshot or similar NLI model.

    Configuration:
        minimum_score: Minimum consistency score required (0.0-1.0, default 0.5)
        use_onnx: Whether to use ONNX runtime for faster inference (default True)

    Example:
        guardrail = FactualConsistencyGuardrail({
            "minimum_score": 0.5
        })
        result = await guardrail.check(
            "The capital of France is Berlin.",
            context={"prompt": "What is the capital of France? It is Paris."}
        )
    """

    name = "factual_consistency"
    layer = GuardrailLayer.OUTPUT
    description = "Detect contradictions between output and context"

    def __init__(
        self, config: Optional[Dict[str, Any]] = None, models_dir: Optional[str] = None
    ) -> None:
        """Initialize the factual consistency guardrail.

        Args:
            config: Configuration dictionary
            models_dir: Optional directory containing pre-downloaded models
        """
        super().__init__(config)
        self._scanner = None
        self._minimum_score = self._config.get("minimum_score", 0.5)
        self._use_onnx = self._config.get("use_onnx", True)
        self._models_dir = models_dir or os.getenv("NEO_LLM_GUARD_MODELS_DIR")

    async def initialize(self) -> None:
        """Initialize the LLM Guard scanner."""
        if self._initialized:
            return

        try:
            from llm_guard.output_scanners import FactualConsistency
            from llm_guard.input_scanners.ban_topics import MODEL_DEBERTA_BASE_V2

            # Configure local model if available
            model_config = None
            if self._models_dir:
                # FactualConsistency uses MODEL_DEBERTA_BASE_V2 by default (deberta-v3-base-zeroshot-v2.0)
                local_model_path = (
                    Path(self._models_dir) / "deberta-v3-base-zeroshot-v2.0"
                )
                if local_model_path.exists():
                    self.logger.info(
                        "factual_consistency_output_using_local_model",
                        model_path=str(local_model_path),
                    )
                    MODEL_DEBERTA_BASE_V2.path = str(local_model_path)
                    MODEL_DEBERTA_BASE_V2.kwargs["local_files_only"] = True
                    model_config = MODEL_DEBERTA_BASE_V2

            if model_config:
                self._scanner = FactualConsistency(
                    minimum_score=self._minimum_score,
                    use_onnx=self._use_onnx,
                    model=model_config,
                )
            else:
                self._scanner = FactualConsistency(
                    minimum_score=self._minimum_score,
                    use_onnx=self._use_onnx,
                )

            self.logger.info(
                "factual_consistency_scanner_initialized",
                minimum_score=self._minimum_score,
                use_onnx=self._use_onnx,
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
        """Check if output is consistent with the provided context.

        Args:
            text: LLM output text to check
            context: Optional context containing the original prompt

        Returns:
            GuardrailResult indicating if output is consistent
        """
        # Get the original prompt from context for the scanner
        prompt = extract_prompt_from_context(context)

        if not prompt:
            # Can't check consistency without a prompt/context
            return GuardrailResult(
                passed=True,
                guardrail_name=self.name,
                layer=self.layer,
                risk_score=0.0,
                message="No context provided, skipping consistency check",
                metadata={
                    "skipped": True,
                    "reason": "no_context",
                },
            )

        if self._scanner is None:
            # Fallback: simple heuristic check
            return self._fallback_check(prompt, text)

        try:
            # LLM Guard output scanners take both prompt and output
            sanitized_text, is_valid, risk_score = self._scanner.scan(prompt, text)

            # Normalize risk_score to 0.0-1.0 range
            # For factual consistency, risk_score indicates inconsistency
            normalized_risk_score = max(0.0, min(1.0, risk_score))

            # Calculate consistency score (inverse of risk)
            consistency_score = 1.0 - normalized_risk_score

            # is_valid indicates if the output is consistent
            is_consistent = is_valid and consistency_score >= self._minimum_score

            return GuardrailResult(
                passed=is_consistent,
                guardrail_name=self.name,
                layer=self.layer,
                risk_score=normalized_risk_score,
                message=(
                    f"Output may contradict context (consistency: {consistency_score:.2f})"
                    if not is_consistent
                    else None
                ),
                metadata={
                    "is_consistent": is_consistent,
                    "consistency_score": consistency_score,
                    "minimum_score": self._minimum_score,
                    "detection_method": "nli_model",
                    "raw_risk_score": risk_score,
                },
            )

        except Exception as e:
            self.logger.error(
                "factual_consistency_check_error",
                error=str(e),
            )
            # Fall back to heuristic check on error
            return self._fallback_check(prompt, text)

    def _fallback_check(self, prompt: str, output: str) -> GuardrailResult:
        """Fallback check using simple contradiction heuristics.

        Looks for explicit contradictions like negations of statements
        in the prompt appearing in the output.

        Args:
            prompt: Original input/context
            output: LLM output to check

        Returns:
            GuardrailResult based on heuristic analysis
        """
        # Extract simple factual claims from prompt (very basic)
        prompt_lower = prompt.lower()
        output_lower = output.lower()

        contradictions_found = []

        # Look for basic contradiction patterns
        # Pattern: "X is Y" in prompt but "X is not Y" or "X is Z" in output

        # Check for negation patterns
        negation_words = ["not", "never", "no", "isn't", "aren't", "wasn't", "weren't", "don't", "doesn't", "didn't"]

        # Simple check: if prompt says something positive and output negates it
        # This is a very basic heuristic
        for negation in negation_words:
            # Find cases where the output explicitly contradicts
            if negation in output_lower:
                # Check if key words from prompt appear near negation in output
                prompt_words = set(prompt_lower.split())
                # Look for negation context
                output_words = output_lower.split()
                for i, word in enumerate(output_words):
                    if word == negation:
                        # Get surrounding context
                        start = max(0, i - 3)
                        end = min(len(output_words), i + 4)
                        context_words = set(output_words[start:end])
                        # If prompt words appear near negation, might be contradiction
                        overlap = prompt_words & context_words
                        if len(overlap) >= 2:
                            contradictions_found.append({
                                "negation": negation,
                                "overlapping_words": list(overlap)[:5],
                            })

        # Look for opposite claims (very basic)
        opposite_pairs = [
            ("true", "false"),
            ("yes", "no"),
            ("correct", "incorrect"),
            ("right", "wrong"),
            ("possible", "impossible"),
            ("can", "cannot"),
            ("will", "won't"),
        ]

        for pos, neg in opposite_pairs:
            if pos in prompt_lower and neg in output_lower:
                contradictions_found.append({
                    "positive_in_prompt": pos,
                    "negative_in_output": neg,
                })
            elif neg in prompt_lower and pos in output_lower:
                contradictions_found.append({
                    "negative_in_prompt": neg,
                    "positive_in_output": pos,
                })

        if contradictions_found:
            # Calculate risk based on number of potential contradictions
            risk_score = min(0.8, 0.4 + (len(contradictions_found) * 0.1))

            return GuardrailResult(
                passed=False,
                guardrail_name=self.name,
                layer=self.layer,
                risk_score=risk_score,
                message=f"Potential contradictions detected ({len(contradictions_found)} found)",
                metadata={
                    "is_consistent": False,
                    "potential_contradictions": contradictions_found[:5],
                    "detection_method": "heuristic",
                    "note": "Heuristic analysis - consider installing LLM Guard for accurate NLI-based detection",
                },
            )

        return GuardrailResult(
            passed=True,
            guardrail_name=self.name,
            layer=self.layer,
            risk_score=0.0,
            message=None,
            metadata={
                "is_consistent": True,
                "detection_method": "heuristic",
                "note": "No obvious contradictions found via heuristics",
            },
        )
