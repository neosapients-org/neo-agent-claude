"""Base provider class for external frameworks.

This module provides a concrete base class that provider
implementations can inherit from.
"""

from typing import Any, Dict, List, Optional, Type

from ..core.exceptions import ProviderError
from ..core.interfaces import BaseGuardrail, BaseProvider


class ProviderBase(BaseProvider):
    """Concrete base class for provider implementations.

    Provides common functionality for registering and
    retrieving guardrail implementations.

    Example:
        class MyProvider(ProviderBase):
            name = "my_provider"

            def __init__(self):
                super().__init__()
                self._register_guardrails()

            def _register_guardrails(self):
                self._guardrails["my_guardrail"] = MyGuardrail
    """

    name: str = "base_provider"

    def __init__(self) -> None:
        """Initialize the provider."""
        super().__init__()
        self._guardrails: Dict[str, Type[BaseGuardrail]] = {}

    def get_guardrail(
        self,
        guardrail_type: str,
        config: Optional[Dict[str, Any]] = None,
    ) -> BaseGuardrail:
        """Get a guardrail implementation for the given type.

        Args:
            guardrail_type: The type of guardrail to get
            config: Optional configuration for the guardrail

        Returns:
            An instance of the requested guardrail

        Raises:
            ProviderError: If the guardrail type is not supported
        """
        if guardrail_type not in self._guardrails:
            raise ProviderError(
                f"Guardrail type '{guardrail_type}' not supported",
                provider_name=self.name,
                guardrail_type=guardrail_type,
            )

        guardrail_class = self._guardrails[guardrail_type]
        return guardrail_class(config)

    def supports(self, guardrail_type: str) -> bool:
        """Check if this provider supports a guardrail type.

        Args:
            guardrail_type: The type of guardrail to check

        Returns:
            True if the provider supports this guardrail type
        """
        return guardrail_type in self._guardrails

    def list_supported_guardrails(self) -> List[str]:
        """List all guardrail types supported by this provider.

        Returns:
            List of supported guardrail type names
        """
        return list(self._guardrails.keys())

    def register_guardrail(
        self,
        name: str,
        guardrail_class: Type[BaseGuardrail],
    ) -> None:
        """Register a guardrail class with this provider.

        Args:
            name: Name for the guardrail type
            guardrail_class: The guardrail class to register
        """
        self._guardrails[name] = guardrail_class
