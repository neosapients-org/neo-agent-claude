"""Abstract interfaces for Neo Guardrail Hub.

This module defines the abstract base classes that all guardrails
and providers must implement to integrate with the framework.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Type

from .models import GuardrailContext, GuardrailLayer, GuardrailResult


class BaseGuardrail(ABC):
    """Abstract base class for all guardrails.

    All guardrail implementations must inherit from this class
    and implement the required methods.

    Attributes:
        name: Unique identifier for this guardrail type
        layer: The layer this guardrail operates at (input/context/output)
        description: Human-readable description of what this guardrail does
    """

    name: str = "base_guardrail"
    layer: GuardrailLayer = GuardrailLayer.INPUT
    description: str = "Base guardrail class"

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize the guardrail with optional configuration.

        Args:
            config: Optional dictionary of guardrail-specific settings
        """
        self._config: Dict[str, Any] = config or {}
        self._threshold: float = self._config.get("threshold", 0.5)
        self._initialized: bool = False

    @abstractmethod
    async def check(
        self,
        text: str,
        context: Optional[GuardrailContext] = None,
    ) -> GuardrailResult:
        """Execute the guardrail check on the given text.

        This is the main method that performs the actual guardrail logic.
        It must be implemented by all concrete guardrail classes.

        Args:
            text: The text to check (user input or LLM output)
            context: Optional context with session/conversation info

        Returns:
            GuardrailResult with pass/fail status and details
        """
        pass

    def configure(self, config: Dict[str, Any]) -> None:
        """Configure the guardrail with settings from YAML config.

        Override this method to handle custom configuration options.

        Args:
            config: Dictionary of configuration settings
        """
        self._config.update(config)
        self._threshold = config.get("threshold", self._threshold)

    def get_config(self) -> Dict[str, Any]:
        """Get the current configuration.

        Returns:
            Dictionary of current configuration settings
        """
        return self._config.copy()

    @property
    def threshold(self) -> float:
        """Get the current detection threshold."""
        return self._threshold

    @threshold.setter
    def threshold(self, value: float) -> None:
        """Set the detection threshold."""
        if not 0.0 <= value <= 1.0:
            raise ValueError("Threshold must be between 0.0 and 1.0")
        self._threshold = value

    async def initialize(self) -> None:
        """Initialize any resources needed by the guardrail.

        Override this method if your guardrail needs to load models
        or establish connections before first use.
        """
        self._initialized = True

    async def cleanup(self) -> None:
        """Clean up any resources used by the guardrail.

        Override this method if your guardrail needs to release
        resources when shutting down.
        """
        self._initialized = False

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r}, layer={self.layer.value!r})"


class BaseProvider(ABC):
    """Abstract base class for external framework providers.

    Providers wrap external guardrail libraries (like LLM Guard or NeMo)
    and expose their functionality through the unified interface.

    Attributes:
        name: Unique identifier for this provider (e.g., "llm_guard", "nemo")
    """

    name: str = "base_provider"

    def __init__(self) -> None:
        """Initialize the provider."""
        self._guardrails: Dict[str, Type[BaseGuardrail]] = {}
        self._initialized: bool = False

    @abstractmethod
    def get_guardrail(
        self, guardrail_type: str, config: Optional[Dict[str, Any]] = None
    ) -> BaseGuardrail:
        """Get a guardrail implementation for the given type.

        Args:
            guardrail_type: The type of guardrail to get (e.g., "prompt_injection")
            config: Optional configuration for the guardrail

        Returns:
            An instance of the requested guardrail

        Raises:
            ProviderError: If the guardrail type is not supported
        """
        pass

    @abstractmethod
    def supports(self, guardrail_type: str) -> bool:
        """Check if this provider supports a guardrail type.

        Args:
            guardrail_type: The type of guardrail to check

        Returns:
            True if the provider supports this guardrail type
        """
        pass

    def list_supported_guardrails(self) -> List[str]:
        """List all guardrail types supported by this provider.

        Returns:
            List of supported guardrail type names
        """
        return list(self._guardrails.keys())

    async def initialize(self) -> None:
        """Initialize the provider and load any required resources.

        Override this method if the provider needs setup before use.
        """
        self._initialized = True

    async def cleanup(self) -> None:
        """Clean up provider resources.

        Override this method if the provider needs to release resources.
        """
        self._initialized = False

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r})"
