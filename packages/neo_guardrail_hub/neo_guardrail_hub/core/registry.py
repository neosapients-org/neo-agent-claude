"""Guardrail registry for Neo Guardrail Hub.

This module provides a registry for discovering, registering,
and retrieving guardrail implementations.
"""

import importlib
import pkgutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Type

from .exceptions import GuardrailNotFoundError, ProviderError
from .interfaces import BaseGuardrail, BaseProvider
from .models import GuardrailLayer


class GuardrailRegistry:
    """Registry for guardrail types and their implementations.

    The registry manages all available guardrails and providers,
    allowing for dynamic discovery and instantiation.

    Example:
        registry = GuardrailRegistry()
        registry.register("prompt_injection", PromptInjectionGuardrail)
        guardrail = registry.get("prompt_injection", {"threshold": 0.8})
    """

    def __init__(self, auto_discover: bool = True) -> None:
        """Initialize the registry.

        Args:
            auto_discover: Whether to automatically discover guardrails
        """
        self._guardrails: Dict[str, Type[BaseGuardrail]] = {}
        self._providers: Dict[str, BaseProvider] = {}  # Registered provider instances
        self._provider_classes: Dict[str, Type[BaseProvider]] = {}  # Discovered provider classes
        self._provider_mappings: Dict[str, str] = {}  # guardrail_type -> provider_name

        if auto_discover:
            self._auto_discover()

    def register(
        self,
        name: str,
        guardrail_class: Type[BaseGuardrail],
        override: bool = False,
    ) -> None:
        """Register a guardrail implementation.

        Args:
            name: Unique name for this guardrail type
            guardrail_class: The guardrail class to register
            override: Whether to override existing registration

        Raises:
            ValueError: If name already registered and override=False
        """
        if name in self._guardrails and not override:
            raise ValueError(
                f"Guardrail '{name}' is already registered. "
                f"Use override=True to replace it."
            )

        self._guardrails[name] = guardrail_class

    def unregister(self, name: str) -> bool:
        """Unregister a guardrail.

        Args:
            name: Name of the guardrail to unregister

        Returns:
            True if guardrail was unregistered, False if not found
        """
        if name in self._guardrails:
            del self._guardrails[name]
            return True
        return False

    def get(
        self,
        name: str,
        config: Optional[Dict[str, Any]] = None,
    ) -> BaseGuardrail:
        """Get a configured guardrail instance.

        Args:
            name: Name of the guardrail type
            config: Optional configuration for the guardrail.
                    May include 'models_dir' key which will be extracted
                    and passed separately to guardrails that support it.

        Returns:
            Configured guardrail instance

        Raises:
            GuardrailNotFoundError: If guardrail type not found
        """
        config = config or {}
        
        # Extract models_dir from config (passed by orchestrator)
        models_dir = config.pop("models_dir", None)

        # First check direct registrations
        if name in self._guardrails:
            guardrail_class = self._guardrails[name]
            # Check if guardrail accepts models_dir parameter
            import inspect
            sig = inspect.signature(guardrail_class.__init__)
            if 'models_dir' in sig.parameters and models_dir:
                guardrail = guardrail_class(config, models_dir=models_dir)
            else:
                guardrail = guardrail_class(config)
            return guardrail

        # Then check provider mappings
        if name in self._provider_mappings:
            provider_name = self._provider_mappings[name]
            provider = self._providers.get(provider_name)
            if provider:
                return provider.get_guardrail(name, config)

        # Finally, try each provider
        for provider in self._providers.values():
            if provider.supports(name):
                self._provider_mappings[name] = provider.name
                return provider.get_guardrail(name, config)

        raise GuardrailNotFoundError(
            guardrail_type=name,
            available_types=self.list_available(),
        )

    def has(self, name: str) -> bool:
        """Check if a guardrail type is available.

        Args:
            name: Name of the guardrail type

        Returns:
            True if available, False otherwise
        """
        if name in self._guardrails:
            return True

        for provider in self._providers.values():
            if provider.supports(name):
                return True

        return False

    def register_provider(self, provider: BaseProvider) -> None:
        """Register an external provider.

        Args:
            provider: Provider instance to register

        Raises:
            ProviderError: If provider with same name already registered
        """
        if provider.name in self._providers:
            raise ProviderError(
                f"Provider '{provider.name}' is already registered",
                provider_name=provider.name,
            )

        self._providers[provider.name] = provider

    def unregister_provider(self, name: str) -> bool:
        """Unregister a provider.

        Args:
            name: Name of the provider to unregister

        Returns:
            True if provider was unregistered, False if not found
        """
        if name in self._providers:
            del self._providers[name]
            # Clear any mappings to this provider
            self._provider_mappings = {
                k: v for k, v in self._provider_mappings.items() if v != name
            }
            return True
        return False

    def get_provider(self, name: str) -> Optional[BaseProvider]:
        """Get a registered provider by name.

        Args:
            name: Name of the provider

        Returns:
            Provider instance or None if not found
        """
        return self._providers.get(name)

    def list_available(self) -> List[str]:
        """List all available guardrail types.

        Returns:
            List of guardrail type names
        """
        available = set(self._guardrails.keys())

        for provider in self._providers.values():
            available.update(provider.list_supported_guardrails())

        return sorted(available)

    def list_providers(self) -> List[str]:
        """List all registered providers.

        Returns:
            List of provider names
        """
        return list(self._providers.keys())

    def list_by_layer(self, layer: GuardrailLayer) -> List[str]:
        """List guardrails available for a specific layer.

        Args:
            layer: The layer to filter by

        Returns:
            List of guardrail names for that layer
        """
        result = []

        for name, guardrail_class in self._guardrails.items():
            if guardrail_class.layer == layer:
                result.append(name)

        return sorted(result)

    def _auto_discover(self) -> None:
        """Auto-discover guardrails from the guardrails package.

        Scans the guardrails directory and imports any modules
        that contain guardrail implementations.
        """
        try:
            # Import the guardrails package
            import neo_guardrail_hub.guardrails as guardrails_pkg

            package_path = Path(guardrails_pkg.__file__).parent

            # Discover modules in subdirectories (input, context, output)
            for subdir in ["input", "context", "output"]:
                subdir_path = package_path / subdir
                if subdir_path.exists():
                    self._discover_in_directory(subdir_path, f"neo_guardrail_hub.guardrails.{subdir}")

        except ImportError:
            # Package not fully installed yet, skip auto-discovery
            pass

        # Also auto-discover and register providers
        self._auto_discover_providers()

    def _discover_in_directory(self, path: Path, package_name: str) -> None:
        """Discover guardrails in a specific directory.

        Args:
            path: Path to the directory
            package_name: Python package name for the directory
        """
        for module_info in pkgutil.iter_modules([str(path)]):
            if module_info.name.startswith("_"):
                continue

            try:
                module = importlib.import_module(f"{package_name}.{module_info.name}")

                # Look for guardrail classes in the module
                for attr_name in dir(module):
                    attr = getattr(module, attr_name)
                    if (
                        isinstance(attr, type)
                        and issubclass(attr, BaseGuardrail)
                        and attr is not BaseGuardrail
                        and hasattr(attr, "name")
                    ):
                        # Register the guardrail
                        self._guardrails[attr.name] = attr

            except ImportError:
                # Skip modules that can't be imported
                pass

    def _auto_discover_providers(self) -> None:
        """Auto-discover and register provider classes from the providers package.

        This method discovers available provider classes (like NeMoProvider)
        and stores them for later instantiation when needed.
        Note: This stores CLASSES, not instances. Use create_provider() to instantiate.
        """
        try:
            # Try to import NeMoProvider if available
            from neo_guardrail_hub.providers.nemo import NeMoProvider
            self._provider_classes["nemo"] = NeMoProvider
        except ImportError:
            # NeMo not installed, skip
            pass

        try:
            # Import LLMGuardProvider
            from neo_guardrail_hub.providers import LLMGuardProvider
            self._provider_classes["llm_guard"] = LLMGuardProvider
        except ImportError:
            pass

    def get_provider_class(self, name: str) -> Optional[Type[BaseProvider]]:
        """Get a provider class by name.

        Args:
            name: Name of the provider (e.g., 'nemo', 'llm_guard')

        Returns:
            The provider class if found, None otherwise
        """
        return self._provider_classes.get(name)

    def create_provider(self, name: str, config: Optional[Dict[str, Any]] = None) -> Optional[BaseProvider]:
        """Create and register a provider instance.

        Args:
            name: Name of the provider (e.g., 'nemo', 'llm_guard')
            config: Configuration to pass to the provider

        Returns:
            The provider instance if created, None if provider class not found
        """
        # Check if already instantiated
        if name in self._providers:
            return self._providers[name]

        # Get the class and instantiate
        provider_class = self._provider_classes.get(name)
        if provider_class:
            # For LLM Guard provider, pass models_dir directly
            if name == "llm_guard" and config:
                models_dir = config.get("models_dir")
                provider = provider_class(models_dir=models_dir) if models_dir else provider_class()
            else:
                # For other providers, pass the config dict
                provider = provider_class(config or {})
            
            self._providers[name] = provider
            return provider
        return None

    def list_provider_classes(self) -> List[str]:
        """List all discovered provider class names.

        Returns:
            List of provider class names
        """
        return list(self._provider_classes.keys())

    def clear(self) -> None:
        """Clear all registrations."""
        self._guardrails.clear()
        self._providers.clear()
        self._provider_classes.clear()
        self._provider_mappings.clear()

    def __len__(self) -> int:
        """Get the number of registered guardrails."""
        return len(self.list_available())

    def __contains__(self, name: str) -> bool:
        """Check if a guardrail is registered."""
        return self.has(name)
