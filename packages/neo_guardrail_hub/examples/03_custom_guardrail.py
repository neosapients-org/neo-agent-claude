#!/usr/bin/env python
"""
Example 3: Creating Custom Guardrails

This example demonstrates how to create your own custom guardrails
by extending the GuardrailBase class.
"""

import asyncio
import re
from typing import Any, Dict, Optional

from neo_guardrail_hub import (
    GuardrailBase,
    GuardrailContext,
    GuardrailLayer,
    GuardrailResult,
    NeoGuardrailOrchestrator,
)


class ProfanityGuardrail(GuardrailBase):
    """Custom guardrail to detect profanity in text.

    This is a simple example - in production, you'd want to use
    a more sophisticated profanity detection library.
    """

    name = "profanity_filter"
    layer = GuardrailLayer.INPUT

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize with custom word list."""
        super().__init__(config)

        # Default profanity list (simplified for example)
        self.blocked_words = self._config.get(
            "blocked_words",
            ["badword1", "badword2", "offensive"],  # Add real words as needed
        )
        self.case_sensitive = self._config.get("case_sensitive", False)

    async def check(
        self,
        text: str,
        context: Optional[GuardrailContext] = None,
    ) -> GuardrailResult:
        """Check text for profanity."""
        check_text = text if self.case_sensitive else text.lower()

        found_words = []
        for word in self.blocked_words:
            check_word = word if self.case_sensitive else word.lower()
            if check_word in check_text:
                found_words.append(word)

        if found_words:
            return GuardrailResult(
                guardrail_name=self.name,
                layer=self.layer,
                passed=False,
                risk_score=min(0.3 * len(found_words), 1.0),
                message=f"Profanity detected: {len(found_words)} word(s) found",
                metadata={"found_words": found_words},
            )

        return GuardrailResult(
            guardrail_name=self.name,
            layer=self.layer,
            passed=True,
            risk_score=0.0,
            message="No profanity detected",
        )


class URLGuardrail(GuardrailBase):
    """Custom guardrail to detect and optionally block URLs in text."""

    name = "url_filter"
    layer = GuardrailLayer.INPUT

    # Common URL pattern
    URL_PATTERN = re.compile(
        r'https?://[^\s<>"{}|\\^`\[\]]+|'
        r'www\.[^\s<>"{}|\\^`\[\]]+'
    )

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize URL guardrail."""
        super().__init__(config)

        self.block_urls = self._config.get("block_urls", True)
        self.allowed_domains = self._config.get("allowed_domains", [])

    async def check(
        self,
        text: str,
        context: Optional[GuardrailContext] = None,
    ) -> GuardrailResult:
        """Check text for URLs."""
        urls = self.URL_PATTERN.findall(text)

        if not urls:
            return GuardrailResult(
                guardrail_name=self.name,
                layer=self.layer,
                passed=True,
                risk_score=0.0,
                message="No URLs detected",
            )

        # Check if URLs are from allowed domains
        blocked_urls = []
        for url in urls:
            is_allowed = any(
                domain in url.lower()
                for domain in self.allowed_domains
            )
            if not is_allowed:
                blocked_urls.append(url)

        if blocked_urls and self.block_urls:
            return GuardrailResult(
                guardrail_name=self.name,
                layer=self.layer,
                passed=False,
                risk_score=0.7,
                message=f"Blocked URLs detected: {len(blocked_urls)}",
                metadata={"blocked_urls": blocked_urls},
            )

        return GuardrailResult(
            guardrail_name=self.name,
            layer=self.layer,
            passed=True,
            risk_score=0.2 if urls else 0.0,
            message=f"URLs detected but allowed: {len(urls)}",
            metadata={"urls": urls},
        )


class LengthGuardrail(GuardrailBase):
    """Custom guardrail to enforce text length limits."""

    name = "length_limit"
    layer = GuardrailLayer.INPUT

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize length guardrail."""
        super().__init__(config)

        self.min_length = self._config.get("min_length", 1)
        self.max_length = self._config.get("max_length", 10000)

    async def check(
        self,
        text: str,
        context: Optional[GuardrailContext] = None,
    ) -> GuardrailResult:
        """Check text length."""
        text_length = len(text)

        if text_length < self.min_length:
            return GuardrailResult(
                guardrail_name=self.name,
                layer=self.layer,
                passed=False,
                risk_score=0.5,
                message=f"Text too short: {text_length} < {self.min_length}",
                metadata={"length": text_length, "min": self.min_length},
            )

        if text_length > self.max_length:
            return GuardrailResult(
                guardrail_name=self.name,
                layer=self.layer,
                passed=False,
                risk_score=0.6,
                message=f"Text too long: {text_length} > {self.max_length}",
                metadata={"length": text_length, "max": self.max_length},
            )

        return GuardrailResult(
            guardrail_name=self.name,
            layer=self.layer,
            passed=True,
            risk_score=0.0,
            message=f"Text length OK: {text_length}",
            metadata={"length": text_length},
        )


async def main():
    """Main example function."""
    print("=" * 60)
    print("Neo Guardrail Hub - Custom Guardrails Example")
    print("=" * 60)

    # Create instances of custom guardrails
    profanity_guard = ProfanityGuardrail({
        "blocked_words": ["badword", "spam", "offensive"],
    })

    url_guard = URLGuardrail({
        "block_urls": True,
        "allowed_domains": ["github.com", "python.org"],
    })

    length_guard = LengthGuardrail({
        "min_length": 5,
        "max_length": 1000,
    })

    # Test profanity guardrail
    print("\n1. Testing Profanity Guardrail...")
    
    clean_text = "Hello, this is a friendly message."
    result = await profanity_guard.check(clean_text)
    print(f"   Text: '{clean_text}'")
    print(f"   Passed: {result.passed}")
    print(f"   Message: {result.message}")

    bad_text = "This message contains badword and spam content."
    result = await profanity_guard.check(bad_text)
    print(f"\n   Text: '{bad_text}'")
    print(f"   Passed: {result.passed}")
    print(f"   Message: {result.message}")
    print(f"   Metadata: {result.metadata}")

    # Test URL guardrail
    print("\n2. Testing URL Guardrail...")

    no_url = "This message has no links."
    result = await url_guard.check(no_url)
    print(f"   Text: '{no_url}'")
    print(f"   Passed: {result.passed}")

    allowed_url = "Check out https://github.com/myproject"
    result = await url_guard.check(allowed_url)
    print(f"\n   Text: '{allowed_url}'")
    print(f"   Passed: {result.passed}")
    print(f"   Message: {result.message}")

    blocked_url = "Visit https://malicious-site.com for deals"
    result = await url_guard.check(blocked_url)
    print(f"\n   Text: '{blocked_url}'")
    print(f"   Passed: {result.passed}")
    print(f"   Message: {result.message}")

    # Test length guardrail
    print("\n3. Testing Length Guardrail...")

    short_text = "Hi"
    result = await length_guard.check(short_text)
    print(f"   Text: '{short_text}' (length: {len(short_text)})")
    print(f"   Passed: {result.passed}")
    print(f"   Message: {result.message}")

    good_text = "This is a properly sized message."
    result = await length_guard.check(good_text)
    print(f"\n   Text: '{good_text}' (length: {len(good_text)})")
    print(f"   Passed: {result.passed}")
    print(f"   Message: {result.message}")

    # Register custom guardrails with orchestrator
    print("\n4. Registering Custom Guardrails with Orchestrator...")
    
    orchestrator = NeoGuardrailOrchestrator(config_path="./configs")
    orchestrator.registry.register("profanity_filter", profanity_guard)
    orchestrator.registry.register("url_filter", url_guard)
    orchestrator.registry.register("length_limit", length_guard)

    print("   Custom guardrails registered successfully!")
    print(f"   Available guardrails: {list(orchestrator.registry._guardrails.keys())}")

    print("\n" + "=" * 60)
    print("Custom guardrails example completed!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
