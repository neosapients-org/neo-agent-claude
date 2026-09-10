"""Secret detection guardrail.

This module provides guardrail protection against leaking secrets
such as API keys, passwords, private keys, and other sensitive data
using LLM Guard's Secrets scanner.
"""

import re
from typing import Any, Dict, List, Optional

from ...core.models import GuardrailContext, GuardrailLayer, GuardrailResult
from ..base import GuardrailBase


class SecretDetectionGuardrail(GuardrailBase):
    """Detect and redact secrets from user input.

    Uses LLM Guard's Secrets scanner to identify API tokens, private keys,
    passwords, and high-entropy strings that may be secrets.

    Configuration:
        threshold: Detection threshold (0.0-1.0), default 0.5
        redact_mode: How to redact secrets - "partial", "full", or "hash"
            - partial: Show first/last few chars (e.g., "sk-...xyz")
            - full: Replace entirely with [REDACTED]
            - hash: Replace with hash of the secret

    Example:
        guardrail = SecretDetectionGuardrail({
            "threshold": 0.5,
            "redact_mode": "partial"
        })
        result = await guardrail.check("My API key is sk-1234567890abcdef")
    """

    name = "secret_detection"
    layer = GuardrailLayer.INPUT
    description = "Detect and redact secrets like API keys and passwords"

    # Common secret patterns for fallback detection
    SECRET_PATTERNS = [
        # API Keys
        (r"sk-[a-zA-Z0-9]{20,}", "OpenAI API Key"),
        (r"sk_live_[a-zA-Z0-9]{20,}", "Stripe Live Key"),
        (r"sk_test_[a-zA-Z0-9]{20,}", "Stripe Test Key"),
        (r"AKIA[0-9A-Z]{16}", "AWS Access Key ID"),
        (r"ghp_[a-zA-Z0-9]{36}", "GitHub Personal Access Token"),
        (r"gho_[a-zA-Z0-9]{36}", "GitHub OAuth Token"),
        (r"github_pat_[a-zA-Z0-9]{22}_[a-zA-Z0-9]{59}", "GitHub PAT"),
        (r"xox[baprs]-[a-zA-Z0-9-]+", "Slack Token"),
        # Private Keys
        (r"-----BEGIN (?:RSA |EC |DSA |OPENSSH )?PRIVATE KEY-----", "Private Key"),
        (r"-----BEGIN PGP PRIVATE KEY BLOCK-----", "PGP Private Key"),
        # Passwords in common formats
        (r"(?i)password\s*[=:]\s*['\"]?[^\s'\"]{8,}['\"]?", "Password"),
        (r"(?i)passwd\s*[=:]\s*['\"]?[^\s'\"]{8,}['\"]?", "Password"),
        (r"(?i)pwd\s*[=:]\s*['\"]?[^\s'\"]{8,}['\"]?", "Password"),
        (r"(?i)secret\s*[=:]\s*['\"]?[^\s'\"]{8,}['\"]?", "Secret"),
        (r"(?i)api_key\s*[=:]\s*['\"]?[^\s'\"]{8,}['\"]?", "API Key"),
        (r"(?i)apikey\s*[=:]\s*['\"]?[^\s'\"]{8,}['\"]?", "API Key"),
        (r"(?i)auth_token\s*[=:]\s*['\"]?[^\s'\"]{8,}['\"]?", "Auth Token"),
        (r"(?i)access_token\s*[=:]\s*['\"]?[^\s'\"]{8,}['\"]?", "Access Token"),
        # Database connection strings
        (r"(?i)mongodb\+srv://[^\s]+", "MongoDB Connection String"),
        (r"(?i)postgres://[^\s]+", "PostgreSQL Connection String"),
        (r"(?i)mysql://[^\s]+", "MySQL Connection String"),
        (r"(?i)redis://[^\s]+", "Redis Connection String"),
        # JWT tokens
        (r"eyJ[a-zA-Z0-9_-]*\.eyJ[a-zA-Z0-9_-]*\.[a-zA-Z0-9_-]*", "JWT Token"),
        # High entropy base64 strings (potential secrets)
        (r"[A-Za-z0-9+/]{40,}={0,2}", "Base64 String"),
    ]

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize the secret detection guardrail.

        Args:
            config: Configuration dictionary
        """
        super().__init__(config)
        self._scanner = None
        self._redact_mode = self._config.get("redact_mode", "partial")
        self._compiled_patterns = [
            (re.compile(pattern), name)
            for pattern, name in self.SECRET_PATTERNS
        ]

    async def initialize(self) -> None:
        """Initialize the LLM Guard scanner."""
        if self._initialized:
            return

        try:
            from llm_guard.input_scanners import Secrets

            # Map redact_mode - Secrets scanner accepts string values
            redact_mode = self._redact_mode
            if redact_mode not in ("partial", "all", "hash"):
                redact_mode = "partial"

            self._scanner = Secrets(redact_mode=redact_mode)

            self.logger.info(
                "secret_detection_scanner_initialized",
                redact_mode=self._redact_mode,
            )

        except ImportError:
            self.logger.warning(
                "llm_guard_not_installed",
                message="Install with: pip install neo-guardrail-hub[llm-guard]",
            )
            self._scanner = None
        except Exception as e:
            self.logger.warning(
                "secret_scanner_init_failed",
                error=str(e),
            )
            self._scanner = None

        await super().initialize()

    async def _check(
        self,
        text: str,
        context: Optional[GuardrailContext] = None,
    ) -> GuardrailResult:
        """Check text for secrets.

        Args:
            text: Text to check
            context: Optional context (not used)

        Returns:
            GuardrailResult indicating if secrets were detected
        """
        if self._scanner is None:
            return self._fallback_check(text)

        try:
            sanitized_prompt, is_valid, risk_score = self._scanner.scan(text)

            # Normalize risk score to 0.0-1.0 range
            normalized_risk_score = max(0.0, min(1.0, risk_score))

            # Determine if secrets were found
            secrets_found = sanitized_prompt != text

            return GuardrailResult(
                passed=is_valid,
                guardrail_name=self.name,
                layer=self.layer,
                risk_score=normalized_risk_score,
                message=(
                    "Secrets detected and redacted"
                    if secrets_found
                    else None
                ),
                sanitized_text=sanitized_prompt if secrets_found else None,
                metadata={
                    "original_length": len(text),
                    "sanitized_length": len(sanitized_prompt),
                    "secrets_found": secrets_found,
                    "detection_method": "llm_guard",
                    "redact_mode": self._redact_mode,
                    "raw_risk_score": risk_score,
                },
            )

        except Exception as e:
            self.logger.error(
                "secret_detection_error",
                error=str(e),
            )
            return self._fallback_check(text)

    def _fallback_check(self, text: str) -> GuardrailResult:
        """Fallback secret detection using regex patterns.

        Args:
            text: Text to check

        Returns:
            GuardrailResult with detection results
        """
        self.logger.debug("initializing_guardrail", guardrail=self.name)

        detected_secrets: List[Dict[str, Any]] = []
        sanitized_text = text

        for pattern, secret_type in self._compiled_patterns:
            matches = pattern.findall(text)
            if matches:
                for match in matches:
                    detected_secrets.append({
                        "type": secret_type,
                        "length": len(match) if isinstance(match, str) else 0,
                    })
                    # Redact the secret
                    if self._redact_mode == "partial" and isinstance(match, str):
                        if len(match) > 8:
                            redacted = f"{match[:4]}...{match[-4:]}"
                        else:
                            redacted = "[REDACTED]"
                    else:
                        redacted = "[REDACTED]"
                    sanitized_text = sanitized_text.replace(
                        match if isinstance(match, str) else str(match),
                        redacted
                    )

        has_secrets = len(detected_secrets) > 0
        risk_score = min(0.3 * len(detected_secrets), 1.0) if has_secrets else 0.0

        # Determine pass/fail based on threshold
        passed = risk_score < self._threshold

        if has_secrets:
            self.logger.warning(
                "secrets_detected",
                count=len(detected_secrets),
                types=[s["type"] for s in detected_secrets],
            )
        else:
            self.logger.debug("no_secrets_detected")

        return GuardrailResult(
            passed=passed,
            guardrail_name=self.name,
            layer=self.layer,
            risk_score=risk_score,
            message=(
                f"Detected {len(detected_secrets)} potential secret(s)"
                if has_secrets
                else None
            ),
            sanitized_text=sanitized_text if has_secrets else None,
            metadata={
                "detected_secrets": detected_secrets,
                "detection_method": "regex_fallback",
                "redact_mode": self._redact_mode,
            },
        )
