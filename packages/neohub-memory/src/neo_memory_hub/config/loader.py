"""Configuration loader for YAML files."""

import os
import re
from pathlib import Path
from typing import Any

import yaml

from neo_memory_hub.core.exceptions import ConfigurationError


class ConfigLoader:
    """
    Loads configuration from YAML files with environment variable substitution.

    Features:
    - Load from multiple YAML files
    - Environment variable substitution (${VAR} or ${VAR:-default})
    - Merge configurations (later files override earlier)
    - Validate against Settings schema

    Example:
        >>> loader = ConfigLoader()
        >>> config = loader.load("config/defaults.yaml", "config/development.yaml")
    """

    def __init__(self, base_path: Path | str | None = None):
        """
        Initialize the config loader.

        Args:
            base_path: Base path for resolving relative config file paths
        """
        if base_path is None:
            # Try to find project root (where pyproject.toml is)
            base_path = self._find_project_root()
        self.base_path = Path(base_path) if base_path else Path.cwd()

    def _find_project_root(self) -> Path:
        """Find project root by looking for pyproject.toml."""
        current = Path.cwd()
        for parent in [current, *current.parents]:
            if (parent / "pyproject.toml").exists():
                return parent
        return current

    def load(self, *config_paths: str | Path) -> dict[str, Any]:
        """
        Load and merge configuration from multiple YAML files.

        Args:
            *config_paths: Paths to YAML config files (relative or absolute)

        Returns:
            Merged configuration dictionary

        Raises:
            ConfigurationError: If config file not found or invalid
        """
        merged_config: dict[str, Any] = {}

        for path in config_paths:
            config = self._load_single(path)
            merged_config = self._deep_merge(merged_config, config)

        return merged_config

    def _load_single(self, config_path: str | Path) -> dict[str, Any]:
        """Load a single YAML config file."""
        path = Path(config_path)

        # Resolve relative paths
        if not path.is_absolute():
            path = self.base_path / path

        if not path.exists():
            raise ConfigurationError(
                f"Configuration file not found: {path}", details={"path": str(path)}
            )

        try:
            with path.open() as f:
                content = f.read()

            # Substitute environment variables
            content = self._substitute_env_vars(content)

            # Parse YAML
            config = yaml.safe_load(content) or {}

            if not isinstance(config, dict):
                raise ConfigurationError(
                    f"Configuration file must contain a mapping: {path}",
                    details={"path": str(path), "type": type(config).__name__},
                )

            return config

        except yaml.YAMLError as e:
            raise ConfigurationError(
                f"Invalid YAML in configuration file: {path}",
                details={"path": str(path), "error": str(e)},
            ) from e

    def _substitute_env_vars(self, content: str) -> str:
        """
        Substitute environment variables in config content.

        Supports:
        - ${VAR} - Required variable
        - ${VAR:-default} - Variable with default value
        """
        # Pattern: ${VAR} or ${VAR:-default}
        pattern = r"\$\{([^}:]+)(?::-([^}]*))?\}"

        def replace(match: re.Match[str]) -> str:
            var_name = match.group(1)
            default = match.group(2)

            value = os.environ.get(var_name)

            if value is not None:
                return value
            elif default is not None:
                return default
            else:
                # Return the original string if no value and no default
                # This allows for optional env vars
                return match.group(0)

        return re.sub(pattern, replace, content)

    def _deep_merge(self, base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
        """
        Deep merge two dictionaries.

        Override values take precedence. Nested dicts are merged recursively.
        """
        result = base.copy()

        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value

        return result

    def load_for_environment(self, environment: str | None = None) -> dict[str, Any]:
        """
        Load configuration for a specific environment.

        Loads defaults.yaml first, then environment-specific config.

        Args:
            environment: Environment name (development, staging, production, test)
                        If None, uses NEO_MEMORY_ENVIRONMENT env var

        Returns:
            Merged configuration dictionary
        """
        if environment is None:
            environment = os.environ.get("NEO_MEMORY_ENVIRONMENT", "development")

        configs_to_load = ["config/defaults.yaml"]

        env_config = f"config/{environment}.yaml"
        if (self.base_path / env_config).exists():
            configs_to_load.append(env_config)

        return self.load(*configs_to_load)
