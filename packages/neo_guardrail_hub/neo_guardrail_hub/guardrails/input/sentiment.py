"""Sentiment analysis guardrail for input.

This module provides guardrail protection for detecting negative
sentiment in user inputs using LLM Guard's Sentiment scanner.
"""

from typing import Any, Dict, Optional

from ...core.models import GuardrailContext, GuardrailLayer, GuardrailResult
from ..base import GuardrailBase


class SentimentInputGuardrail(GuardrailBase):
    """Detect and filter negative sentiment in user input.

    Uses NLTK's VADER sentiment analyzer to evaluate the sentiment
    of input text. Useful for monitoring and moderating user sentiment
    to prevent aggressive or negative interactions.

    Configuration:
        threshold: Minimum acceptable sentiment score (-1.0 to 1.0), default 0.0
            - -1.0: Completely negative
            - 0.0: Neutral
            - 1.0: Completely positive
        Any input with sentiment below threshold is flagged.

    Example:
        guardrail = SentimentInputGuardrail({
            "threshold": -0.5  # Block very negative content
        })
        result = await guardrail.check("I hate this terrible service!")
    """

    name = "sentiment_input"
    layer = GuardrailLayer.INPUT
    description = "Detect and filter negative sentiment in input"

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize the sentiment input guardrail.

        Args:
            config: Configuration dictionary
        """
        super().__init__(config)
        self._scanner = None
        # Sentiment threshold: -1.0 (negative) to 1.0 (positive)
        # Default 0.0 means neutral is acceptable
        self._sentiment_threshold = self._config.get("threshold", 0.0)

    async def initialize(self) -> None:
        """Initialize the LLM Guard scanner."""
        if self._initialized:
            return

        try:
            from llm_guard.input_scanners import Sentiment

            self._scanner = Sentiment(
                threshold=self._sentiment_threshold,
            )

            self.logger.info(
                "sentiment_input_scanner_initialized",
                threshold=self._sentiment_threshold,
            )

        except ImportError:
            self.logger.warning(
                "llm_guard_not_installed",
                message="Install with: pip install neo-guardrail-hub[llm-guard]",
            )
            self._scanner = None
        except LookupError as e:
            # NLTK data not downloaded
            self.logger.warning(
                "nltk_data_not_found",
                message="NLTK vader_lexicon not found. Run: import nltk; nltk.download('vader_lexicon')",
                error=str(e),
            )
            self._scanner = None
        except Exception as e:
            self.logger.warning(
                "sentiment_scanner_init_error",
                message=f"Failed to initialize sentiment scanner: {str(e)}",
            )
            self._scanner = None

        await super().initialize()

    async def _check(
        self,
        text: str,
        context: Optional[GuardrailContext] = None,
    ) -> GuardrailResult:
        """Check text for sentiment.

        Args:
            text: Text to check
            context: Optional context (not used)

        Returns:
            GuardrailResult indicating if sentiment is acceptable
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
                    f"Negative sentiment detected (below threshold: {self._sentiment_threshold})"
                    if not is_valid
                    else None
                ),
                metadata={
                    "detection_method": "llm_guard",
                    "threshold": self._sentiment_threshold,
                    "raw_risk_score": risk_score,
                },
            )

        except Exception as e:
            self.logger.error(
                "sentiment_detection_error",
                error=str(e),
            )
            return self._fallback_check(text)

    def _fallback_check(self, text: str) -> GuardrailResult:
        """Fallback sentiment analysis using NLTK VADER or simple heuristics.

        Args:
            text: Text to check

        Returns:
            GuardrailResult with sentiment analysis
        """
        try:
            # Try to use NLTK VADER
            from nltk.sentiment.vader import SentimentIntensityAnalyzer

            analyzer = SentimentIntensityAnalyzer()
            scores = analyzer.polarity_scores(text)
            compound_score = scores['compound']

            is_valid = compound_score >= self._sentiment_threshold

            # Convert sentiment to risk score (negative sentiment = higher risk)
            # Map [-1, 1] to [1, 0]
            risk_score = (1 - compound_score) / 2

            return GuardrailResult(
                passed=is_valid,
                guardrail_name=self.name,
                layer=self.layer,
                risk_score=risk_score,
                message=(
                    f"Negative sentiment detected (score: {compound_score:.2f})"
                    if not is_valid
                    else None
                ),
                metadata={
                    "detection_method": "nltk_vader",
                    "sentiment_score": compound_score,
                    "sentiment_details": scores,
                    "threshold": self._sentiment_threshold,
                },
            )

        except (ImportError, LookupError, Exception) as e:
            # Fall back to simple keyword-based detection
            # LookupError occurs when NLTK is installed but vader_lexicon not downloaded
            self.logger.debug(f"NLTK VADER unavailable: {e}, using keyword fallback")
            return self._simple_sentiment_check(text)

    def _simple_sentiment_check(self, text: str) -> GuardrailResult:
        """Very simple sentiment check using keyword lists.

        Args:
            text: Text to check

        Returns:
            GuardrailResult with basic sentiment analysis
        """
        text_lower = text.lower()

        # Negative sentiment indicators
        negative_words = [
            "hate", "terrible", "awful", "horrible", "worst", "disgusting",
            "pathetic", "useless", "stupid", "idiot", "dumb", "garbage",
            "trash", "fail", "failure", "sucks", "suck", "angry", "furious",
            "frustrated", "annoyed", "disappointed", "never", "nothing works"
        ]

        # Positive sentiment indicators
        positive_words = [
            "love", "great", "amazing", "wonderful", "excellent", "fantastic",
            "awesome", "perfect", "best", "good", "happy", "pleased",
            "satisfied", "helpful", "thank", "thanks", "appreciate"
        ]

        negative_count = sum(1 for word in negative_words if word in text_lower)
        positive_count = sum(1 for word in positive_words if word in text_lower)

        # Simple scoring: range approximately -1 to 1
        total = negative_count + positive_count
        if total == 0:
            sentiment_score = 0.0  # Neutral
        else:
            sentiment_score = (positive_count - negative_count) / total

        is_valid = sentiment_score >= self._sentiment_threshold
        risk_score = max(0.0, (1 - sentiment_score) / 2)

        return GuardrailResult(
            passed=is_valid,
            guardrail_name=self.name,
            layer=self.layer,
            risk_score=risk_score,
            message=(
                f"Negative sentiment detected (score: {sentiment_score:.2f})"
                if not is_valid
                else None
            ),
            metadata={
                "detection_method": "keyword_heuristics",
                "sentiment_score": sentiment_score,
                "negative_count": negative_count,
                "positive_count": positive_count,
                "threshold": self._sentiment_threshold,
            },
        )

    async def cleanup(self) -> None:
        """Clean up scanner resources."""
        self._scanner = None
        await super().cleanup()
