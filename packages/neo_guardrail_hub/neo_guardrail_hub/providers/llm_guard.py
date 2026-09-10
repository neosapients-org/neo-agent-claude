"""LLM Guard provider for Neo Guardrail Hub.

This module provides integration with Protect AI's LLM Guard library
for input and output scanning.
"""

import os
from typing import Any, Dict, Optional

from ..core.interfaces import BaseGuardrail

# Phase 1 Input Guardrails
from ..guardrails.input.pii_detection import PIIDetectionGuardrail
from ..guardrails.input.prompt_injection import PromptInjectionGuardrail

# Phase 2 Input Guardrails
from ..guardrails.input.ban_substrings import BanSubstringsInputGuardrail
from ..guardrails.input.harmful_content import HarmfulContentGuardrail
from ..guardrails.input.input_length import InputLengthGuardrail
from ..guardrails.input.secret_detection import SecretDetectionGuardrail
from ..guardrails.input.toxicity import ToxicityInputGuardrail

# Phase 3 Input Guardrails (Additional LLM Guard Scanners)
from ..guardrails.input.ban_code import BanCodeInputGuardrail
from ..guardrails.input.ban_competitors import BanCompetitorsInputGuardrail
from ..guardrails.input.ban_topics import BanTopicsInputGuardrail
from ..guardrails.input.code_detection import CodeDetectionInputGuardrail
from ..guardrails.input.gibberish import GibberishInputGuardrail
from ..guardrails.input.invisible_text import InvisibleTextInputGuardrail
from ..guardrails.input.language import LanguageInputGuardrail
from ..guardrails.input.regex import RegexInputGuardrail
from ..guardrails.input.sentiment import SentimentInputGuardrail

# Phase 1 Output Guardrails
from ..guardrails.output.pii_redaction import PIIRedactionGuardrail

# Phase 2 Output Guardrails
from ..guardrails.output.ban_substrings import BanSubstringsOutputGuardrail
from ..guardrails.output.ban_topics import BanTopicsGuardrail
from ..guardrails.output.factual_consistency import FactualConsistencyGuardrail
from ..guardrails.output.no_refusal import NoRefusalGuardrail
from ..guardrails.output.relevance import RelevanceGuardrail
from ..guardrails.output.toxicity import ToxicityOutputGuardrail

# Phase 3 Output Guardrails (Additional LLM Guard Scanners)
from ..guardrails.output.bias import BiasOutputGuardrail
from ..guardrails.output.code_detection import CodeDetectionOutputGuardrail
from ..guardrails.output.ban_competitors import BanCompetitorsOutputGuardrail
from ..guardrails.output.gibberish import GibberishOutputGuardrail
from ..guardrails.output.json_validation import JSONValidationOutputGuardrail
from ..guardrails.output.language import LanguageOutputGuardrail
from ..guardrails.output.language_same import LanguageSameOutputGuardrail
from ..guardrails.output.malicious_urls import MaliciousURLsOutputGuardrail
from ..guardrails.output.reading_time import ReadingTimeOutputGuardrail
from ..guardrails.output.regex import RegexOutputGuardrail
from ..guardrails.output.sensitive import SensitiveDataOutputGuardrail
from ..guardrails.output.sentiment import SentimentOutputGuardrail
from ..guardrails.output.url_reachability import URLReachabilityOutputGuardrail

from .base import ProviderBase


class LLMGuardProvider(ProviderBase):
    """Provider for LLM Guard scanners.

    Wraps LLM Guard's input and output scanners as Neo Guardrail Hub
    guardrails with a unified interface.

    Supported Input Guardrails:
        - prompt_injection: Detect prompt injection attacks
        - pii_detection: Detect PII in input
        - secret_detection: Detect API keys, tokens, secrets
        - input_length: Validate input length constraints
        - toxicity_input: Detect toxic/offensive input
        - ban_substrings_input: Block specific substrings in input
        - harmful_content: Detect harmful content categories
        - ban_code_input: Detect and block code snippets
        - ban_competitors_input: Block competitor mentions
        - ban_topics_input: Block inputs on sensitive topics
        - code_detection_input: Detect specific programming languages
        - gibberish_input: Detect nonsensical/gibberish text
        - invisible_text_input: Detect hidden Unicode characters
        - language_input: Validate input language
        - regex_input: Custom regex pattern matching
        - sentiment_input: Detect negative sentiment

    Supported Output Guardrails:
        - pii_redaction: Redact PII from output
        - no_refusal: Detect refusal responses
        - relevance: Check output relevance to prompt
        - toxicity_output: Detect toxic content in output
        - factual_consistency: Detect contradictions
        - ban_topics: Block outputs on sensitive topics
        - ban_substrings_output: Block specific substrings in output
        - bias_output: Detect biased statements in output
        - code_detection_output: Detect programming languages in output
        - ban_competitors_output: Block competitor mentions in output
        - gibberish_output: Detect nonsensical/incoherent output
        - json_validation: Validate JSON structure in output
        - language_output: Validate output language
        - language_same: Check input/output language consistency
        - malicious_urls: Detect harmful URLs in output
        - reading_time: Enforce reading time limits
        - regex_output: Custom regex pattern matching for output
        - sensitive_output: Detect/redact sensitive data in output
        - sentiment_output: Analyze output sentiment
        - url_reachability: Check URL accessibility

    Example:
        provider = LLMGuardProvider()
        guardrail = provider.get_guardrail("prompt_injection", {"threshold": 0.8})
    """

    name = "llm_guard"

    def __init__(self, models_dir: Optional[str] = None) -> None:
        """Initialize the LLM Guard provider.
        
        Args:
            models_dir: Directory containing pre-downloaded models (optional).
                       Can also be set via NEO_LLM_GUARD_MODELS_DIR environment variable.
        """
        super().__init__()
        self._models_dir = models_dir or os.getenv("NEO_LLM_GUARD_MODELS_DIR")
        self._register_guardrails()

    def _register_guardrails(self) -> None:
        """Register all LLM Guard guardrails."""
        self._guardrails = {
            # Phase 1 Input Guardrails
            "prompt_injection": PromptInjectionGuardrail,
            "pii_detection": PIIDetectionGuardrail,
            # Phase 2 Input Guardrails
            "secret_detection": SecretDetectionGuardrail,
            "input_length": InputLengthGuardrail,
            "toxicity_input": ToxicityInputGuardrail,
            "ban_substrings_input": BanSubstringsInputGuardrail,
            "harmful_content": HarmfulContentGuardrail,
            # Phase 3 Input Guardrails (Additional LLM Guard Scanners)
            "ban_code_input": BanCodeInputGuardrail,
            "ban_competitors_input": BanCompetitorsInputGuardrail,
            "ban_topics_input": BanTopicsInputGuardrail,
            "code_detection_input": CodeDetectionInputGuardrail,
            "gibberish_input": GibberishInputGuardrail,
            "invisible_text_input": InvisibleTextInputGuardrail,
            "language_input": LanguageInputGuardrail,
            "regex_input": RegexInputGuardrail,
            "sentiment_input": SentimentInputGuardrail,
            # Phase 1 Output Guardrails
            "pii_redaction": PIIRedactionGuardrail,
            # Phase 2 Output Guardrails
            "no_refusal": NoRefusalGuardrail,
            "relevance": RelevanceGuardrail,
            "toxicity_output": ToxicityOutputGuardrail,
            "factual_consistency": FactualConsistencyGuardrail,
            "ban_topics": BanTopicsGuardrail,
            "ban_substrings_output": BanSubstringsOutputGuardrail,
            # Phase 3 Output Guardrails (Additional LLM Guard Scanners)
            "bias_output": BiasOutputGuardrail,
            "code_detection_output": CodeDetectionOutputGuardrail,
            "ban_competitors_output": BanCompetitorsOutputGuardrail,
            "gibberish_output": GibberishOutputGuardrail,
            "json_validation": JSONValidationOutputGuardrail,
            "language_output": LanguageOutputGuardrail,
            "language_same": LanguageSameOutputGuardrail,
            "malicious_urls": MaliciousURLsOutputGuardrail,
            "reading_time": ReadingTimeOutputGuardrail,
            "regex_output": RegexOutputGuardrail,
            "sensitive_output": SensitiveDataOutputGuardrail,
            "sentiment_output": SentimentOutputGuardrail,
            "url_reachability": URLReachabilityOutputGuardrail,
        }

    async def initialize(self) -> None:
        """Initialize the provider.

        Checks if LLM Guard is installed and available.
        """
        try:
            import llm_guard  # noqa: F401

            self._initialized = True
        except ImportError:
            # LLM Guard not installed, guardrails will use fallback
            self._initialized = True

        await super().initialize()

    def get_guardrail(
        self,
        guardrail_type: str,
        config: Optional[Dict[str, Any]] = None,
    ) -> BaseGuardrail:
        """Get a guardrail implementation.

        Args:
            guardrail_type: Type of guardrail
            config: Configuration for the guardrail

        Returns:
            Configured guardrail instance
        """
        guardrail_class = self._guardrails.get(guardrail_type)
        
        if guardrail_class is None:
            raise ValueError(f"Unknown guardrail type: {guardrail_type}")
        
        # Check if the guardrail supports models_dir parameter
        # (currently only PromptInjectionGuardrail does, others will be updated later)
        try:
            import inspect
            sig = inspect.signature(guardrail_class.__init__)
            if 'models_dir' in sig.parameters:
                return guardrail_class(config, models_dir=self._models_dir)
        except Exception:
            pass
        
        # Fallback to original instantiation
        return guardrail_class(config)
