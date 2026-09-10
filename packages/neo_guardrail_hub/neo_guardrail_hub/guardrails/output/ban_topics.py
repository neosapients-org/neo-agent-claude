"""Ban topics guardrail for output.

This module provides guardrail protection for detecting and blocking
outputs that discuss banned/sensitive topics.
"""

import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from ...core.models import GuardrailContext, GuardrailLayer, GuardrailResult
from ..base import GuardrailBase, extract_prompt_from_context


# Default topic keywords for fallback detection
DEFAULT_TOPIC_KEYWORDS: Dict[str, List[str]] = {
    "politics": [
        "election", "vote", "political", "democrat", "republican",
        "liberal", "conservative", "congress", "parliament", "legislation",
    ],
    "religion": [
        "religious", "church", "mosque", "temple", "prayer",
        "christian", "muslim", "jewish", "hindu", "buddhist", "faith",
    ],
    "violence": [
        "violent", "attack", "weapon", "murder", "assault",
        "fight", "war", "combat", "kill", "harm",
    ],
    "drugs": [
        "drug", "narcotic", "cocaine", "heroin", "marijuana",
        "meth", "opioid", "substance abuse", "overdose",
    ],
    "gambling": [
        "gamble", "gambling", "casino", "bet", "betting",
        "poker", "slot", "wager", "odds",
    ],
    "adult_content": [
        "explicit", "adult", "sexual", "pornography", "nsfw",
    ],
}


class BanTopicsGuardrail(GuardrailBase):
    """Detect and block outputs discussing banned topics.

    Uses LLM Guard's BanTopics scanner with zero-shot classification
    to detect outputs on sensitive topics. Uses MoritzLaurer/deberta-v3-base-zeroshot
    or similar model.

    Configuration:
        topics: List of topic names to ban (default: [])
        threshold: Topic detection threshold (0.0-1.0, default 0.5)
        use_onnx: Whether to use ONNX runtime for faster inference (default True)

    Example:
        guardrail = BanTopicsGuardrail({
            "topics": ["politics", "religion", "violence"],
            "threshold": 0.5
        })
        result = await guardrail.check("Let me tell you about the election...")
    """

    name = "ban_topics"
    layer = GuardrailLayer.OUTPUT
    description = "Block outputs discussing banned topics"

    def __init__(self, config: Optional[Dict[str, Any]] = None, models_dir: Optional[str] = None) -> None:
        """Initialize the ban topics guardrail.

        Args:
            config: Configuration dictionary
            models_dir: Optional path to directory containing local models
        """
        super().__init__(config)
        self._scanner = None
        self._topics: List[str] = self._config.get("topics", [])
        self._use_onnx = self._config.get("use_onnx", True)
        self._models_dir = models_dir or os.getenv("NEO_LLM_GUARD_MODELS_DIR")
        # Custom keywords for fallback (can extend defaults)
        self._topic_keywords: Dict[str, List[str]] = self._config.get(
            "topic_keywords", DEFAULT_TOPIC_KEYWORDS
        )
        self._compiled_patterns: Dict[str, List[re.Pattern]] = {}

    async def initialize(self) -> None:
        """Initialize the LLM Guard scanner."""
        if self._initialized:
            return

        # Compile fallback patterns for configured topics
        self._compiled_patterns = {}
        for topic in self._topics:
            topic_lower = topic.lower()
            if topic_lower in self._topic_keywords:
                keywords = self._topic_keywords[topic_lower]
                # Create word-boundary patterns for keywords
                self._compiled_patterns[topic_lower] = [
                    re.compile(rf"\b{re.escape(kw)}\b", re.IGNORECASE)
                    for kw in keywords
                ]

        if not self._topics:
            self.logger.warning(
                "no_topics_configured",
                message="No topics configured for ban_topics guardrail",
            )

        try:
            from llm_guard.output_scanners import BanTopics
            from llm_guard.input_scanners.ban_topics import MODEL_DEBERTA_BASE_V2

            # Configure local model if available
            model_config = None
            if self._models_dir:
                local_model_path = Path(self._models_dir) / "deberta-v3-base-zeroshot-v2.0"
                if local_model_path.exists():
                    self.logger.info(
                        "ban_topics_output_using_local_model",
                        model_path=str(local_model_path)
                    )
                    MODEL_DEBERTA_BASE_V2.path = str(local_model_path)
                    MODEL_DEBERTA_BASE_V2.kwargs["local_files_only"] = True
                    model_config = MODEL_DEBERTA_BASE_V2

            if model_config:
                self._scanner = BanTopics(
                    topics=self._topics,
                    threshold=self._threshold,
                    use_onnx=self._use_onnx,
                    model=model_config,
                )
            else:
                self._scanner = BanTopics(
                    topics=self._topics,
                    threshold=self._threshold,
                    use_onnx=self._use_onnx,
                )

            self.logger.info(
                "ban_topics_scanner_initialized",
                topics=self._topics,
                threshold=self._threshold,
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
        """Check if output discusses banned topics.

        Args:
            text: LLM output text to check
            context: Optional context

        Returns:
            GuardrailResult indicating if banned topics were detected
        """
        if not self._topics:
            return GuardrailResult(
                passed=True,
                guardrail_name=self.name,
                layer=self.layer,
                risk_score=0.0,
                message="No topics configured",
                metadata={
                    "skipped": True,
                    "reason": "no_topics_configured",
                },
            )

        # Get the original prompt from context for the scanner
        prompt = extract_prompt_from_context(context)

        if self._scanner is None:
            # Fallback: use keyword matching
            return self._fallback_check(text)

        try:
            # LLM Guard output scanners take both prompt and output
            sanitized_text, is_valid, risk_score = self._scanner.scan(prompt, text)

            # Normalize risk_score to 0.0-1.0 range
            normalized_risk_score = max(0.0, min(1.0, risk_score))

            # is_valid indicates if no banned topics were found
            banned_topics_found = not is_valid or normalized_risk_score >= self._threshold

            return GuardrailResult(
                passed=not banned_topics_found,
                guardrail_name=self.name,
                layer=self.layer,
                risk_score=normalized_risk_score,
                message=(
                    f"Banned topic detected (confidence: {normalized_risk_score:.2f})"
                    if banned_topics_found
                    else None
                ),
                metadata={
                    "banned_topics_found": banned_topics_found,
                    "configured_topics": self._topics,
                    "detection_confidence": normalized_risk_score,
                    "detection_method": "zero_shot_classification",
                    "raw_risk_score": risk_score,
                },
            )

        except Exception as e:
            self.logger.error(
                "ban_topics_check_error",
                error=str(e),
            )
            # Fall back to keyword matching on error
            return self._fallback_check(text)

    def _fallback_check(self, text: str) -> GuardrailResult:
        """Fallback check using keyword matching.

        Args:
            text: Text to check for banned topic keywords

        Returns:
            GuardrailResult based on keyword matching
        """
        detected_topics = {}

        for topic, patterns in self._compiled_patterns.items():
            matched_keywords = []
            for pattern in patterns:
                matches = pattern.findall(text)
                if matches:
                    matched_keywords.extend(matches)

            if matched_keywords:
                detected_topics[topic] = {
                    "keyword_count": len(matched_keywords),
                    "keywords": list(set(matched_keywords))[:5],
                }

        if detected_topics:
            # Calculate risk based on number and diversity of topic matches
            total_keywords = sum(t["keyword_count"] for t in detected_topics.values())
            topic_count = len(detected_topics)
            risk_score = min(0.9, 0.4 + (topic_count * 0.15) + (total_keywords * 0.02))

            return GuardrailResult(
                passed=False,
                guardrail_name=self.name,
                layer=self.layer,
                risk_score=risk_score,
                message=f"Banned topic(s) detected: {', '.join(detected_topics.keys())}",
                metadata={
                    "banned_topics_found": True,
                    "detected_topics": detected_topics,
                    "topic_count": topic_count,
                    "configured_topics": self._topics,
                    "detection_method": "keyword_matching",
                },
            )

        return GuardrailResult(
            passed=True,
            guardrail_name=self.name,
            layer=self.layer,
            risk_score=0.0,
            message=None,
            metadata={
                "banned_topics_found": False,
                "configured_topics": self._topics,
                "detection_method": "keyword_matching",
            },
        )
