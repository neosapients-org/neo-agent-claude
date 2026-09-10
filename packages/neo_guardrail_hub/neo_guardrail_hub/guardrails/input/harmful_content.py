"""Harmful content detection guardrail.

This module provides guardrail protection against harmful content
including violence, hate speech, self-harm, and other dangerous content.
"""

import re
from typing import Any, Dict, List, Optional, Set

from ...core.models import GuardrailContext, GuardrailLayer, GuardrailResult
from ..base import GuardrailBase


class HarmfulContentGuardrail(GuardrailBase):
    """Detect and block harmful content in user input.

    Uses pattern matching and optional ML models to identify content
    related to violence, hate, self-harm, illegal activities, and more.

    Configuration:
        threshold: Detection threshold (0.0-1.0), default 0.5
        categories: List of harm categories to detect
            - violence: Content promoting or describing violence
            - hate: Hate speech and discrimination
            - self_harm: Self-harm or suicide related content
            - illegal: Illegal activities and instructions
            - sexual: Explicit sexual content
            - harassment: Bullying and harassment
            - weapons: Weapons and explosives instructions
            - drugs: Drug use and manufacturing
        on_fail: Action on failure - "block" or "warn", default "block"

    Example:
        guardrail = HarmfulContentGuardrail({
            "threshold": 0.5,
            "categories": ["violence", "hate", "self_harm"]
        })
        result = await guardrail.check("Some text to check...")
    """

    name = "harmful_content"
    layer = GuardrailLayer.INPUT
    description = "Detect and block harmful content"

    # Default categories to detect
    DEFAULT_CATEGORIES = [
        "violence",
        "hate",
        "self_harm",
        "illegal",
        "harassment",
    ]

    # Patterns for each harm category
    HARM_PATTERNS: Dict[str, List[str]] = {
        "violence": [
            r"\b(?:kill|murder|attack|shoot|stab|beat|assault)\s+(?:someone|people|them|him|her|you)\b",
            r"\b(?:how\s+to\s+(?:kill|hurt|harm|attack))\b",
            r"\b(?:want\s+to\s+(?:kill|hurt|harm|attack))\b",
            r"\b(?:planning\s+(?:to\s+)?(?:kill|hurt|harm|attack))\b",
            r"\b(?:bomb|explosive|weapon)\s+(?:make|create|build)\b",
        ],
        "hate": [
            r"\b(?:all\s+)?(?:blacks?|whites?|jews?|muslims?|christians?)\s+(?:should|must|need\s+to)\s+(?:die|be\s+killed)\b",
            r"\b(?:hate|despise)\s+(?:all\s+)?(?:blacks?|whites?|jews?|muslims?|asians?|latinos?)\b",
            r"\b(?:racial|ethnic)\s+(?:cleansing|purge)\b",
            r"\b(?:inferior|subhuman)\s+(?:race|people)\b",
        ],
        "self_harm": [
            r"\b(?:want\s+to|going\s+to|how\s+to)\s+(?:kill\s+myself|commit\s+suicide|end\s+my\s+life)\b",
            r"\b(?:suicide|self-harm|cut\s+myself|hurt\s+myself)\s+(?:methods?|ways?|instructions?)\b",
            r"\bkys\b",
            r"\b(?:better\s+off\s+dead|no\s+reason\s+to\s+live)\b",
        ],
        "illegal": [
            r"\b(?:how\s+to\s+(?:hack|steal|rob|fraud|scam))\b",
            r"\b(?:counterfeit|forge|fake)\s+(?:money|documents?|id|passport)\b",
            r"\b(?:launder|laundering)\s+money\b",
            r"\b(?:traffic|trafficking)\s+(?:humans?|people|drugs?)\b",
        ],
        "sexual": [
            r"\b(?:child|minor|underage)\s+(?:porn|sex|nude|naked)\b",
            r"\b(?:rape|molest|abuse)\s+(?:child|minor|kid)\b",
            r"\bcsam\b",
        ],
        "harassment": [
            r"\b(?:i\s+will|gonna)\s+(?:find|track|hunt)\s+you\b",
            r"\b(?:doxx|dox|expose)\s+(?:you|them|someone)\b",
            r"\b(?:stalk|stalking)\s+(?:you|them|someone)\b",
            r"\b(?:send\s+(?:threats?|death\s+threats?))\b",
        ],
        "weapons": [
            r"\b(?:how\s+to\s+(?:make|build|create))\s+(?:a\s+)?(?:bomb|explosive|weapon|gun)\b",
            r"\b(?:3d\s+print|print)\s+(?:a\s+)?(?:gun|weapon|firearm)\b",
            r"\b(?:poison|chemical\s+weapon)\s+(?:make|create|recipe)\b",
        ],
        "drugs": [
            r"\b(?:how\s+to\s+(?:make|cook|synthesize))\s+(?:meth|cocaine|heroin|drugs?)\b",
            r"\b(?:drug)\s+(?:recipe|synthesis|manufacturing)\b",
            r"\b(?:buy|sell|deal)\s+(?:meth|cocaine|heroin|fentanyl)\b",
        ],
    }

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize the harmful content guardrail.

        Args:
            config: Configuration dictionary
        """
        super().__init__(config)
        self._categories: Set[str] = set(
            self._config.get("categories", self.DEFAULT_CATEGORIES)
        )
        self._compiled_patterns: Dict[str, List[re.Pattern]] = {}

        # Compile patterns for selected categories
        for category in self._categories:
            if category in self.HARM_PATTERNS:
                self._compiled_patterns[category] = [
                    re.compile(pattern, re.IGNORECASE)
                    for pattern in self.HARM_PATTERNS[category]
                ]

    async def initialize(self) -> None:
        """Initialize the guardrail."""
        if self._initialized:
            return

        self.logger.info(
            "harmful_content_guardrail_initialized",
            categories=list(self._categories),
            threshold=self._threshold,
        )

        await super().initialize()

    async def _check(
        self,
        text: str,
        context: Optional[GuardrailContext] = None,
    ) -> GuardrailResult:
        """Check text for harmful content.

        Args:
            text: Text to check
            context: Optional context (not used)

        Returns:
            GuardrailResult indicating if harmful content was detected
        """
        self.logger.debug("initializing_guardrail", guardrail=self.name)

        detected_categories: Dict[str, List[str]] = {}
        total_matches = 0

        for category, patterns in self._compiled_patterns.items():
            matches = []
            for pattern in patterns:
                found = pattern.findall(text)
                if found:
                    matches.extend(found)
                    total_matches += len(found)

            if matches:
                detected_categories[category] = matches

        has_harmful_content = len(detected_categories) > 0

        # Calculate risk score based on severity and count
        if has_harmful_content:
            # Weight certain categories higher
            severity_weights = {
                "violence": 1.0,
                "hate": 1.0,
                "self_harm": 1.0,
                "sexual": 1.0,
                "weapons": 0.9,
                "illegal": 0.8,
                "harassment": 0.7,
                "drugs": 0.7,
            }

            max_severity = max(
                severity_weights.get(cat, 0.5) for cat in detected_categories.keys()
            )
            risk_score = min(max_severity * (1 + 0.1 * (total_matches - 1)), 1.0)
        else:
            risk_score = 0.0

        passed = risk_score < self._threshold

        if has_harmful_content:
            self.logger.warning(
                "harmful_content_detected",
                categories=list(detected_categories.keys()),
                total_matches=total_matches,
                risk_score=risk_score,
            )
        else:
            self.logger.debug("no_harmful_content_detected")

        return GuardrailResult(
            passed=passed,
            guardrail_name=self.name,
            layer=self.layer,
            risk_score=risk_score,
            message=(
                f"Harmful content detected in categories: {', '.join(detected_categories.keys())}"
                if has_harmful_content and not passed
                else None
            ),
            metadata={
                "detection_method": "pattern_matching",
                "detected_categories": list(detected_categories.keys()),
                "total_matches": total_matches,
                "enabled_categories": list(self._categories),
            },
        )
