"""URL reachability guardrail for output.

This module provides guardrail protection for verifying that URLs
in LLM outputs are reachable and not broken.
"""

import re
from typing import Any, Dict, List, Optional

from ...core.models import GuardrailContext, GuardrailLayer, GuardrailResult
from ..base import GuardrailBase, extract_prompt_from_context


class URLReachabilityOutputGuardrail(GuardrailBase):
    """Verify that URLs in LLM outputs are reachable.

    Uses LLM Guard's URLReachability output scanner to check if URLs
    in the output are accessible.

    Configuration:
        success_status_codes: List of HTTP status codes considered successful
                             (default [200, 201, 202, 301, 302])
        timeout: Request timeout in seconds (default 5)

    Example:
        guardrail = URLReachabilityOutputGuardrail({
            "success_status_codes": [200, 201, 301, 302],
            "timeout": 3
        })
        result = await guardrail.check(
            "Check out https://example.com for more info",
            context={"prompt": "Where can I learn more?"}
        )
    """

    name = "url_reachability"
    layer = GuardrailLayer.OUTPUT
    description = "Verify URLs in LLM output are reachable"

    # URL pattern for extraction
    URL_PATTERN = re.compile(
        r'https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+'
        r'(?:/[-\w./?%&=]*)?'
    )

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize the URL reachability guardrail.

        Args:
            config: Configuration dictionary
        """
        super().__init__(config)
        self._scanner = None
        self._success_status_codes = self._config.get(
            "success_status_codes", [200, 201, 202, 301, 302]
        )
        self._timeout = self._config.get("timeout", 5)

    async def initialize(self) -> None:
        """Initialize the LLM Guard scanner."""
        if self._initialized:
            return

        try:
            from llm_guard.output_scanners import URLReachability

            self._scanner = URLReachability(
                success_status_codes=self._success_status_codes,
                timeout=self._timeout,
            )

            self.logger.info(
                "url_reachability_scanner_initialized",
                success_codes=self._success_status_codes,
                timeout=self._timeout,
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
                "url_reachability_scanner_init_failed",
                message="Failed to initialize URLReachability scanner, will use fallback",
                error=str(e),
            )
            self._scanner = None

        await super().initialize()

    async def _check(
        self,
        text: str,
        context: Optional[GuardrailContext] = None,
    ) -> GuardrailResult:
        """Check URL reachability in the output.

        Args:
            text: LLM output to check
            context: Optional context with prompt

        Returns:
            GuardrailResult indicating if all URLs are reachable
        """
        # First check if there are any URLs
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
                    "all_reachable": True,
                },
            )

        prompt = extract_prompt_from_context(context)

        if self._scanner is None:
            return self._fallback_check(text, urls)

        try:
            sanitized_text, is_valid, risk_score = self._scanner.scan(prompt, text)

            normalized_risk = max(0.0, min(1.0, risk_score))
            passed = is_valid

            if not passed:
                self.logger.warning(
                    "unreachable_urls_in_output",
                    num_urls=len(urls),
                )

            return GuardrailResult(
                passed=passed,
                guardrail_name=self.name,
                layer=self.layer,
                risk_score=normalized_risk,
                message="Unreachable URL(s) detected in output" if not passed else None,
                metadata={
                    "urls_found": len(urls),
                    "all_reachable": passed,
                    "detection_method": "llm_guard_url_reachability",
                },
            )

        except Exception as e:
            self.logger.error("url_reachability_check_error", error=str(e))
            return self._fallback_check(text, urls)

    def _fallback_check(self, text: str, urls: List[str]) -> GuardrailResult:
        """Fallback check - just reports URLs found without checking.

        Note: The fallback doesn't actually check reachability as it would
        require making HTTP requests which could be slow and have side effects.

        Args:
            text: Text that was checked
            urls: List of URLs found

        Returns:
            GuardrailResult (passes by default in fallback mode)
        """
        return GuardrailResult(
            passed=True,
            guardrail_name=self.name,
            layer=self.layer,
            risk_score=0.0,
            message="URL reachability not verified (fallback mode)",
            metadata={
                "urls_found": len(urls),
                "urls": urls[:10],  # Limit to first 10
                "verified": False,
                "detection_method": "fallback_no_check",
            },
        )

    async def _check_url_reachable(self, url: str) -> bool:
        """Check if a single URL is reachable.

        This is an async method that could be used for actual URL checking
        in a more advanced fallback implementation.

        Args:
            url: URL to check

        Returns:
            True if reachable, False otherwise
        """
        try:
            import aiohttp
            
            async with aiohttp.ClientSession() as session:
                async with session.head(
                    url, 
                    timeout=aiohttp.ClientTimeout(total=self._timeout),
                    allow_redirects=True
                ) as response:
                    return response.status in self._success_status_codes
        except Exception:
            return False
