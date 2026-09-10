"""Malicious URLs detection guardrail for output.

This module provides guardrail protection for detecting malicious URLs
in LLM outputs using ML-based classification.
"""

import re
from typing import Any, Dict, List, Optional

from ...core.models import GuardrailContext, GuardrailLayer, GuardrailResult
from ..base import GuardrailBase, extract_prompt_from_context


class MaliciousURLsOutputGuardrail(GuardrailBase):
    """Detect malicious URLs in LLM outputs.

    Uses LLM Guard's MaliciousURLs output scanner with the
    DunnBC22/codebert-base-Malicious_URLs model.

    Configuration:
        threshold: Malware detection threshold (0.0-1.0, default 0.7)
        use_onnx: Whether to use ONNX runtime (default True)

    Example:
        guardrail = MaliciousURLsOutputGuardrail({
            "threshold": 0.7
        })
        result = await guardrail.check(
            "Visit https://malicious-site.com for more info",
            context={"prompt": "Where can I find resources?"}
        )
    """

    name = "malicious_urls"
    layer = GuardrailLayer.OUTPUT
    description = "Detect malicious URLs in LLM output"

    # URL pattern for fallback
    URL_PATTERN = re.compile(
        r'https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+'
        r'(?:/[-\w./?%&=]*)?'
    )

    # Suspicious patterns for fallback
    SUSPICIOUS_PATTERNS = [
        r'\.ru/', r'\.cn/', r'\.tk/', r'\.top/',
        r'bit\.ly', r'tinyurl',
        r'free-.*\.com', r'.*-free\.com',
        r'login.*\.', r'.*-login\.',
        r'secure.*\.', r'account.*\.',
        r'\.exe', r'\.zip', r'\.scr',
    ]

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize the malicious URLs guardrail.

        Args:
            config: Configuration dictionary
        """
        super().__init__(config)
        self._scanner = None
        self._threshold = self._config.get("threshold", 0.7)
        self._use_onnx = self._config.get("use_onnx", True)

    async def initialize(self) -> None:
        """Initialize the LLM Guard scanner."""
        if self._initialized:
            return

        try:
            from llm_guard.output_scanners import MaliciousURLs

            self._scanner = MaliciousURLs(
                threshold=self._threshold,
                use_onnx=self._use_onnx,
            )

            self.logger.info(
                "malicious_urls_scanner_initialized",
                threshold=self._threshold,
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
                "malicious_urls_scanner_init_failed",
                message="Failed to initialize MaliciousURLs scanner, will use fallback",
                error=str(e),
            )
            self._scanner = None

        await super().initialize()

    async def _check(
        self,
        text: str,
        context: Optional[GuardrailContext] = None,
    ) -> GuardrailResult:
        """Check for malicious URLs in the output.

        Args:
            text: LLM output to check
            context: Optional context with prompt

        Returns:
            GuardrailResult indicating if malicious URLs were detected
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
                    "malicious_url_detected",
                    risk_score=normalized_risk,
                )

            return GuardrailResult(
                passed=passed,
                guardrail_name=self.name,
                layer=self.layer,
                risk_score=normalized_risk,
                message="Malicious URL detected in output" if not passed else None,
                metadata={
                    "malicious_url_detected": not passed,
                    "threshold": self._threshold,
                    "detection_method": "codebert_malicious_urls",
                },
            )

        except Exception as e:
            self.logger.error("malicious_urls_check_error", error=str(e))
            return self._fallback_check(text)

    def _fallback_check(self, text: str) -> GuardrailResult:
        """Fallback check using pattern matching.

        Args:
            text: Text to check

        Returns:
            GuardrailResult based on pattern matching
        """
        # Find all URLs
        urls = self.URL_PATTERN.findall(text)
        
        if not urls:
            return GuardrailResult(
                passed=True,
                guardrail_name=self.name,
                layer=self.layer,
                risk_score=0.0,
                message=None,
                metadata={
                    "urls_found": 0,
                    "detection_method": "pattern_fallback",
                },
            )

        # Check for suspicious patterns
        suspicious_urls: List[str] = []
        for url in urls:
            for pattern in self.SUSPICIOUS_PATTERNS:
                if re.search(pattern, url, re.IGNORECASE):
                    suspicious_urls.append(url)
                    break

        passed = len(suspicious_urls) == 0
        risk_score = min(len(suspicious_urls) * 0.3, 1.0)

        return GuardrailResult(
            passed=passed,
            guardrail_name=self.name,
            layer=self.layer,
            risk_score=risk_score,
            message=(
                f"Suspicious URLs detected: {suspicious_urls[:3]}" if not passed else None
            ),
            metadata={
                "malicious_url_detected": not passed,
                "urls_found": len(urls),
                "suspicious_urls": suspicious_urls,
                "detection_method": "pattern_fallback",
            },
        )
