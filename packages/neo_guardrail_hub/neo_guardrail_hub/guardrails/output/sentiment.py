"""Sentiment analysis guardrail for output.

This module provides guardrail protection for detecting negative or
inappropriate sentiment in LLM outputs.
"""

from typing import Any, Dict, Optional

from ...core.models import GuardrailContext, GuardrailLayer, GuardrailResult
from ..base import GuardrailBase, extract_prompt_from_context


class SentimentOutputGuardrail(GuardrailBase):
    """Analyze sentiment in LLM outputs.

    Uses LLM Guard's Sentiment output scanner with NLTK's VADER
    sentiment analyzer to detect negative sentiment.

    Configuration:
        threshold: Sentiment threshold (-1.0 to 1.0, default 0.0)
                   Outputs below this threshold are flagged as too negative.
                   -1.0 = completely negative
                    0.0 = neutral
                    1.0 = completely positive

    Example:
        guardrail = SentimentOutputGuardrail({
            "threshold": -0.3
        })
        result = await guardrail.check(
            "This is terrible and will never work.",
            context={"prompt": "Give me feedback on my idea"}
        )
    """

    name = "sentiment_output"
    layer = GuardrailLayer.OUTPUT
    description = "Analyze sentiment in LLM output"

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize the sentiment guardrail.

        Args:
            config: Configuration dictionary
        """
        super().__init__(config)
        self._scanner = None
        # Override default threshold - sentiment uses -1 to 1 scale
        self._threshold = self._config.get("threshold", 0.0)
        self._vader = None

    async def initialize(self) -> None:
        """Initialize the LLM Guard scanner."""
        if self._initialized:
            return

        try:
            from llm_guard.output_scanners import Sentiment

            self._scanner = Sentiment(
                threshold=self._threshold,
            )

            self.logger.info(
                "sentiment_output_scanner_initialized",
                threshold=self._threshold,
            )

        except ImportError as e:
            self.logger.warning(
                "llm_guard_not_installed",
                message="Install with: pip install neo-guardrail-hub[llm-guard]",
                error=str(e),
            )
            self._scanner = None
            
            # Try to initialize NLTK VADER for fallback
            try:
                from nltk.sentiment import SentimentIntensityAnalyzer
                self._vader = SentimentIntensityAnalyzer()
            except Exception:
                self._vader = None

        except Exception as e:
            self.logger.warning(
                "sentiment_output_scanner_init_failed",
                message="Failed to initialize Sentiment scanner, will use fallback",
                error=str(e),
            )
            self._scanner = None

        await super().initialize()

    async def _check(
        self,
        text: str,
        context: Optional[GuardrailContext] = None,
    ) -> GuardrailResult:
        """Check the sentiment of the output.

        Args:
            text: LLM output to check
            context: Optional context with prompt

        Returns:
            GuardrailResult indicating if sentiment is acceptable
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
                    "negative_sentiment_in_output",
                    threshold=self._threshold,
                )

            return GuardrailResult(
                passed=passed,
                guardrail_name=self.name,
                layer=self.layer,
                risk_score=normalized_risk,
                message="Negative sentiment detected in output" if not passed else None,
                metadata={
                    "sentiment_acceptable": passed,
                    "threshold": self._threshold,
                    "detection_method": "vader",
                },
            )

        except Exception as e:
            self.logger.error("sentiment_output_check_error", error=str(e))
            return self._fallback_check(text)

    def _fallback_check(self, text: str) -> GuardrailResult:
        """Fallback check using VADER or keyword heuristics.

        Args:
            text: Text to check

        Returns:
            GuardrailResult based on sentiment analysis
        """
        sentiment_score = 0.0
        detection_method = "keyword_fallback"

        if self._vader:
            try:
                scores = self._vader.polarity_scores(text)
                sentiment_score = scores.get("compound", 0.0)
                detection_method = "vader_fallback"
            except Exception:
                pass

        if detection_method == "keyword_fallback":
            # Simple keyword-based fallback
            negative_words = [
                "terrible", "awful", "horrible", "bad", "wrong", "fail",
                "hate", "stupid", "idiotic", "useless", "worthless",
            ]
            positive_words = [
                "great", "good", "excellent", "wonderful", "amazing",
                "helpful", "useful", "correct", "right", "perfect",
            ]
            
            text_lower = text.lower()
            neg_count = sum(1 for w in negative_words if w in text_lower)
            pos_count = sum(1 for w in positive_words if w in text_lower)
            
            total = neg_count + pos_count
            if total > 0:
                sentiment_score = (pos_count - neg_count) / total
            else:
                sentiment_score = 0.0

        passed = sentiment_score >= self._threshold
        
        # Convert sentiment to risk score (more negative = higher risk)
        risk_score = max(0.0, min(1.0, (1.0 - sentiment_score) / 2))

        return GuardrailResult(
            passed=passed,
            guardrail_name=self.name,
            layer=self.layer,
            risk_score=risk_score,
            message=(
                f"Negative sentiment (score: {sentiment_score:.2f})" if not passed else None
            ),
            metadata={
                "sentiment_acceptable": passed,
                "sentiment_score": sentiment_score,
                "threshold": self._threshold,
                "detection_method": detection_method,
            },
        )
