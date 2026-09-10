"""Ban topics guardrail for input.

This module provides guardrail protection for detecting and blocking
inputs that discuss banned/sensitive topics using zero-shot classification.
"""

import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from ...core.models import GuardrailContext, GuardrailLayer, GuardrailResult
from ..base import GuardrailBase


# Default topic keywords for fallback detection
DEFAULT_TOPIC_KEYWORDS: Dict[str, List[str]] = {
    "politics": [
        "election", "vote", "political", "democrat", "republican",
        "liberal", "conservative", "congress", "parliament", "legislation",
        "president", "senator", "governor", "campaign", "ballot",
    ],
    "religion": [
        "religious", "church", "mosque", "temple", "prayer",
        "christian", "muslim", "jewish", "hindu", "buddhist", "faith",
        "god", "allah", "jesus", "bible", "quran", "torah",
    ],
    "violence": [
        "violent", "attack", "weapon", "murder", "assault",
        "fight", "war", "combat", "kill", "harm", "shoot",
        "bomb", "explode", "stab", "threat",
    ],
    "drugs": [
        "drug", "narcotic", "cocaine", "heroin", "marijuana",
        "meth", "opioid", "substance abuse", "overdose", "cannabis",
        "lsd", "ecstasy", "amphetamine",
    ],
    "gambling": [
        "gamble", "gambling", "casino", "bet", "betting",
        "poker", "slot", "wager", "odds", "blackjack", "roulette",
    ],
    "adult_content": [
        "explicit", "adult", "sexual", "pornography", "nsfw",
        "nude", "erotic", "xxx",
    ],
    "medical_advice": [
        "diagnose", "diagnosis", "treatment", "medication", "prescription",
        "symptoms", "cure", "disease", "medical condition",
    ],
    "legal_advice": [
        "legal advice", "lawsuit", "sue", "court case", "attorney",
        "lawyer", "legal matter", "litigation",
    ],
    "financial_advice": [
        "invest", "stock tips", "financial advice", "trading strategy",
        "portfolio", "buy stocks", "sell stocks",
    ],
}


class BanTopicsInputGuardrail(GuardrailBase):
    """Detect and block inputs discussing banned topics.

    Uses LLM Guard's BanTopics scanner with zero-shot classification
    to detect inputs on sensitive topics. Uses MoritzLaurer/deberta-v3-base-zeroshot
    or similar model.

    Configuration:
        topics: List of topic names to ban (default: [])
        threshold: Topic detection threshold (0.0-1.0, default 0.5)
        use_onnx: Whether to use ONNX runtime for faster inference (default True)

    Example:
        guardrail = BanTopicsInputGuardrail({
            "topics": ["politics", "religion", "violence"],
            "threshold": 0.5
        })
        result = await guardrail.check("Let me tell you about the election...")
    """

    name = "ban_topics_input"
    layer = GuardrailLayer.INPUT
    description = "Block inputs discussing banned topics"

    def __init__(self, config: Optional[Dict[str, Any]] = None, models_dir: Optional[str] = None) -> None:
        """Initialize the ban topics input guardrail.

        Args:
            config: Configuration dictionary
            models_dir: Optional directory containing local models
        """
        super().__init__(config)
        self._scanner = None
        self._topics: List[str] = self._config.get("topics", [])
        self._use_onnx = self._config.get("use_onnx", True)
        self._models_dir = models_dir or os.getenv("NEO_LLM_GUARD_MODELS_DIR")

    async def initialize(self) -> None:
        """Initialize the LLM Guard scanner."""
        if self._initialized:
            return

        if not self._topics:
            self.logger.warning(
                "no_topics_configured",
                message="No topics configured for ban_topics_input guardrail",
            )
            await super().initialize()
            return

        try:
            from llm_guard.input_scanners import BanTopics
            from llm_guard.input_scanners.ban_topics import MODEL_DEBERTA_BASE_V2

            model_config = None
            
            # Check if local models are available
            if self._models_dir:
                local_model_path = Path(self._models_dir) / "deberta-v3-base-zeroshot-v2.0"
                
                if local_model_path.exists():
                    self.logger.info(
                        "ban_topics_using_local_model",
                        model_path=str(local_model_path)
                    )
                    
                    # Configure MODEL_DEBERTA_BASE_V2 to use local path
                    MODEL_DEBERTA_BASE_V2.path = str(local_model_path)
                    MODEL_DEBERTA_BASE_V2.kwargs["local_files_only"] = True
                    model_config = MODEL_DEBERTA_BASE_V2
                else:
                    self.logger.warning(
                        "ban_topics_local_model_not_found",
                        expected_path=str(local_model_path),
                        message="Will download from HuggingFace (slower). Run: neo-guardrail download-models --scanners ban_topics"
                    )

            # Initialize scanner with model config if available
            if model_config:
                self._scanner = BanTopics(
                    topics=self._topics,
                    threshold=self._threshold,
                    use_onnx=self._use_onnx,
                    model=model_config
                )
            else:
                self._scanner = BanTopics(
                    topics=self._topics,
                    threshold=self._threshold,
                    use_onnx=self._use_onnx,
                )

            self.logger.info(
                "ban_topics_input_scanner_initialized",
                topics=self._topics,
                threshold=self._threshold,
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
        """Check text for banned topics.

        Args:
            text: Text to check
            context: Optional context (not used)

        Returns:
            GuardrailResult indicating if banned topics were detected
        """
        if not self._topics:
            return GuardrailResult(
                passed=True,
                guardrail_name=self.name,
                layer=self.layer,
                risk_score=0.0,
                message=None,
                metadata={
                    "detection_method": "none",
                    "reason": "no_topics_configured",
                },
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
                    "Input discusses banned topic"
                    if not is_valid
                    else None
                ),
                metadata={
                    "detection_method": "llm_guard",
                    "topics": self._topics,
                    "raw_risk_score": risk_score,
                },
            )

        except Exception as e:
            self.logger.error(
                "ban_topics_input_check_error",
                error=str(e),
            )
            return self._fallback_check(text)

    def _fallback_check(self, text: str) -> GuardrailResult:
        """Fallback topic detection using keyword matching.

        Args:
            text: Text to check

        Returns:
            GuardrailResult with topic detection results
        """
        text_lower = text.lower()
        detected_topics: Dict[str, List[str]] = {}

        for topic in self._topics:
            # Get keywords for this topic
            keywords = DEFAULT_TOPIC_KEYWORDS.get(topic.lower(), [])
            if not keywords:
                # If topic not in default list, use topic name as keyword
                keywords = [topic.lower()]

            matched_keywords: List[str] = []
            for keyword in keywords:
                # Use word boundary matching for better accuracy
                pattern = rf'\b{re.escape(keyword)}\b'
                if re.search(pattern, text_lower, re.IGNORECASE):
                    matched_keywords.append(keyword)

            if matched_keywords:
                detected_topics[topic] = matched_keywords

        has_banned_topic = len(detected_topics) > 0

        # Calculate risk score based on number of keyword matches
        total_matches = sum(len(kw) for kw in detected_topics.values())
        risk_score = min(0.3 * total_matches, 1.0) if has_banned_topic else 0.0

        return GuardrailResult(
            passed=not has_banned_topic,
            guardrail_name=self.name,
            layer=self.layer,
            risk_score=risk_score,
            message=(
                f"Detected banned topic(s): {', '.join(detected_topics.keys())}"
                if has_banned_topic
                else None
            ),
            metadata={
                "detection_method": "keyword_matching",
                "detected_topics": detected_topics,
                "configured_topics": self._topics,
            },
        )

    async def cleanup(self) -> None:
        """Clean up scanner resources."""
        self._scanner = None
        await super().cleanup()
