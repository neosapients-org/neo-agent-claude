"""Relevance detection guardrail for output.

This module provides guardrail protection for detecting when an LLM output
is not relevant to the original prompt/question.
"""

import os
from pathlib import Path
from typing import Any, Dict, Optional

from ...core.models import GuardrailContext, GuardrailLayer, GuardrailResult
from ..base import GuardrailBase, extract_prompt_from_context


class RelevanceGuardrail(GuardrailBase):
    """Detect when LLM output is not relevant to the input prompt.

    Uses LLM Guard's Relevance output scanner with embedding-based
    similarity comparison. Uses BAAI/bge-base-en-v1.5 for embeddings.

    Configuration:
        threshold: Relevance threshold (0.0-1.0, default 0.5)
                   Higher threshold requires more similarity
        use_onnx: Whether to use ONNX runtime for faster inference (default True)

    Example:
        guardrail = RelevanceGuardrail({
            "threshold": 0.5
        })
        result = await guardrail.check(
            "The sky is blue because of Rayleigh scattering.",
            context={"prompt": "Why is the sky blue?"}
        )
    """

    name = "relevance"
    layer = GuardrailLayer.OUTPUT
    description = "Detect when LLM output is not relevant to the prompt"

    def __init__(
        self, config: Optional[Dict[str, Any]] = None, models_dir: Optional[str] = None
    ) -> None:
        """Initialize the relevance detection guardrail.

        Args:
            config: Configuration dictionary
            models_dir: Optional directory containing pre-downloaded models
        """
        super().__init__(config)
        self._scanner = None
        self._threshold = self._config.get("threshold", 0.5)
        self._use_onnx = self._config.get("use_onnx", True)
        # For fallback: keyword overlap threshold
        self._keyword_overlap_threshold = self._config.get("keyword_overlap_threshold", 0.2)
        self._models_dir = models_dir or os.getenv("NEO_LLM_GUARD_MODELS_DIR")

    async def initialize(self) -> None:
        """Initialize the LLM Guard scanner."""
        if self._initialized:
            return

        try:
            from llm_guard.output_scanners import Relevance
            from llm_guard.output_scanners.relevance import MODEL_EN_BGE_BASE

            # Configure local model if available
            model_config = None
            if self._models_dir:
                local_model_path = Path(self._models_dir) / "bge-base-en-v1.5"
                if local_model_path.exists():
                    self.logger.info(
                        "relevance_output_using_local_model",
                        model_path=str(local_model_path),
                    )
                    MODEL_EN_BGE_BASE.path = str(local_model_path)
                    MODEL_EN_BGE_BASE.kwargs["local_files_only"] = True
                    model_config = MODEL_EN_BGE_BASE

            if model_config:
                self._scanner = Relevance(
                    threshold=self._threshold,
                    use_onnx=self._use_onnx,
                    model=model_config,
                )
            else:
                self._scanner = Relevance(
                    threshold=self._threshold,
                    use_onnx=self._use_onnx,
                )

            self.logger.info(
                "relevance_scanner_initialized",
                threshold=self._threshold,
                use_onnx=self._use_onnx,
                using_local_model=model_config is not None,
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
                "relevance_scanner_init_failed",
                message="Failed to initialize Relevance scanner, will use fallback",
                error=str(e),
                error_type=type(e).__name__,
            )
            self._scanner = None

        await super().initialize()

    async def _check(
        self,
        text: str,
        context: Optional[GuardrailContext] = None,
    ) -> GuardrailResult:
        """Check if output is relevant to the original prompt.

        Args:
            text: LLM output text to check
            context: Optional context containing the original prompt

        Returns:
            GuardrailResult indicating if output is relevant
        """
        # Get the original prompt from context for the scanner
        prompt = extract_prompt_from_context(context)

        if not prompt:
            # Can't check relevance without a prompt
            return GuardrailResult(
                passed=True,
                guardrail_name=self.name,
                layer=self.layer,
                risk_score=0.0,
                message="No prompt provided, skipping relevance check",
                metadata={
                    "skipped": True,
                    "reason": "no_prompt",
                },
            )

        if self._scanner is None:
            # Fallback: use keyword overlap
            return self._fallback_check(prompt, text)

        try:
            # LLM Guard output scanners take both prompt and output
            sanitized_text, is_valid, risk_score = self._scanner.scan(prompt, text)

            # Normalize risk_score to 0.0-1.0 range
            # For relevance, higher score = more irrelevant
            normalized_risk_score = max(0.0, min(1.0, risk_score))

            # is_valid indicates if the output is relevant
            # Risk score indicates how irrelevant (higher = worse)
            is_relevant = is_valid and normalized_risk_score < self._threshold

            # Calculate similarity for metadata (inverse of risk)
            similarity = 1.0 - normalized_risk_score

            return GuardrailResult(
                passed=is_relevant,
                guardrail_name=self.name,
                layer=self.layer,
                risk_score=normalized_risk_score,
                message=(
                    f"Output not relevant to prompt (similarity: {similarity:.2f})"
                    if not is_relevant
                    else None
                ),
                metadata={
                    "is_relevant": is_relevant,
                    "similarity_score": similarity,
                    "threshold": self._threshold,
                    "detection_method": "embedding_similarity",
                    "raw_risk_score": risk_score,
                },
            )

        except Exception as e:
            self.logger.error(
                "relevance_check_error",
                error=str(e),
            )
            # Fall back to keyword overlap on error
            return self._fallback_check(prompt, text)

    def _fallback_check(self, prompt: str, output: str) -> GuardrailResult:
        """Fallback check using keyword overlap.

        Calculates the Jaccard similarity of keywords between prompt and output.

        Args:
            prompt: Original input prompt
            output: LLM output to check

        Returns:
            GuardrailResult based on keyword overlap
        """
        # Extract keywords (simple word tokenization, filter short words)
        prompt_words = set(
            word.lower()
            for word in prompt.split()
            if len(word) > 3 and word.isalnum()
        )
        output_words = set(
            word.lower()
            for word in output.split()
            if len(word) > 3 and word.isalnum()
        )

        # Calculate Jaccard similarity
        if not prompt_words or not output_words:
            # Can't calculate overlap without words
            return GuardrailResult(
                passed=True,
                guardrail_name=self.name,
                layer=self.layer,
                risk_score=0.0,
                message="Insufficient text for relevance check",
                metadata={
                    "skipped": True,
                    "reason": "insufficient_text",
                    "detection_method": "keyword_overlap",
                },
            )

        intersection = prompt_words & output_words
        union = prompt_words | output_words
        jaccard_similarity = len(intersection) / len(union) if union else 0.0

        # Calculate overlap ratio (what portion of prompt words appear in output)
        overlap_ratio = (
            len(intersection) / len(prompt_words) if prompt_words else 0.0
        )

        # Combine both metrics
        combined_similarity = (jaccard_similarity + overlap_ratio) / 2

        # Determine relevance
        is_relevant = combined_similarity >= self._keyword_overlap_threshold

        # Risk is inverse of similarity
        risk_score = 1.0 - combined_similarity

        return GuardrailResult(
            passed=is_relevant,
            guardrail_name=self.name,
            layer=self.layer,
            risk_score=risk_score,
            message=(
                f"Output may not be relevant (similarity: {combined_similarity:.2f})"
                if not is_relevant
                else None
            ),
            metadata={
                "is_relevant": is_relevant,
                "jaccard_similarity": jaccard_similarity,
                "overlap_ratio": overlap_ratio,
                "combined_similarity": combined_similarity,
                "threshold": self._keyword_overlap_threshold,
                "common_keywords": list(intersection)[:10],
                "detection_method": "keyword_overlap",
            },
        )
