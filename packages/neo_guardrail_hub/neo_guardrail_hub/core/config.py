"""Configuration loader for Neo Guardrail Hub.

This module handles loading, merging, and validating configuration
files from YAML sources with inheritance support.
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml

from .exceptions import ConfigurationError
from .models import ActionOnFail, ExecutionMode, GuardrailConfig, LayerConfig


class ConfigLoader:
    """Load and merge configuration from multiple sources.

    Supports configuration inheritance, where agent-specific configs
    can extend and override a base configuration.

    Example:
        loader = ConfigLoader("./configs")
        config = loader.load(agent_id="wealth_advisor")
    """

    DEFAULT_CONFIG_FILENAME = "default.yaml"
    AGENTS_DIR = "agents"

    def __init__(self, config_path: Union[str, Path] = "./configs") -> None:
        """Initialize the config loader.

        Args:
            config_path: Path to the configuration directory
        """
        self.config_path = Path(config_path)
        self._cache: Dict[str, Dict[str, Any]] = {}

    def load(
        self,
        agent_id: Optional[str] = None,
        config_override: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Load configuration with inheritance resolution.

        Args:
            agent_id: Optional agent ID to load specific config
            config_override: Optional runtime overrides

        Returns:
            Merged configuration dictionary

        Raises:
            ConfigurationError: If config files are invalid or missing
        """
        # Start with default config
        config = self._load_default_config()

        # Load agent-specific config if provided
        if agent_id and agent_id != "default":
            agent_config = self._load_agent_config(agent_id)
            if agent_config:
                config = self._merge_configs(config, agent_config)

        # Apply runtime overrides
        if config_override:
            config = self._merge_configs(config, config_override)

        # Validate the final config
        errors = self.validate(config)
        if errors:
            raise ConfigurationError(
                f"Configuration validation failed: {'; '.join(errors)}"
            )

        return config

    def _load_default_config(self) -> Dict[str, Any]:
        """Load the default configuration file."""
        default_path = self.config_path / self.DEFAULT_CONFIG_FILENAME

        if not default_path.exists():
            # Return sensible defaults if no config file exists
            return self._get_builtin_defaults()

        return self._load_yaml_file(default_path)

    def _load_agent_config(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """Load agent-specific configuration.

        Looks for config in: configs/agents/{agent_id}.yaml
        or configs/agents/{agent_id}/guardrails.yaml
        """
        # Try direct file first
        agent_file = self.config_path / self.AGENTS_DIR / f"{agent_id}.yaml"
        if agent_file.exists():
            return self._load_yaml_file(agent_file)

        # Try directory structure
        agent_dir = self.config_path / self.AGENTS_DIR / agent_id / "guardrails.yaml"
        if agent_dir.exists():
            return self._load_yaml_file(agent_dir)

        return None

    def _load_yaml_file(self, path: Path) -> Dict[str, Any]:
        """Load and parse a YAML configuration file."""
        cache_key = str(path)
        if cache_key in self._cache:
            return self._cache[cache_key].copy()

        try:
            with open(path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f) or {}
                self._cache[cache_key] = config
                return config.copy()
        except yaml.YAMLError as e:
            raise ConfigurationError(
                f"Invalid YAML syntax: {e}", config_path=str(path)
            )
        except OSError as e:
            raise ConfigurationError(
                f"Could not read config file: {e}", config_path=str(path)
            )

    def _merge_configs(
        self, base: Dict[str, Any], override: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Deep merge two configuration dictionaries.

        Override values take precedence. Lists are replaced, not merged.
        """
        result = base.copy()

        for key, value in override.items():
            if (
                key in result
                and isinstance(result[key], dict)
                and isinstance(value, dict)
            ):
                result[key] = self._merge_configs(result[key], value)
            else:
                result[key] = value

        return result

    def _get_builtin_defaults(self) -> Dict[str, Any]:
        """Get built-in default configuration."""
        return {
            "version": "1.0",
            "enabled": True,
            "execution": {
                "input": {"mode": "parallel", "timeout_ms": 5000},
                "context": {"mode": "sequential", "timeout_ms": 10000},
                "output": {"mode": "parallel", "timeout_ms": 5000},
            },
            "defaults": {"on_fail": "block", "log_level": "INFO"},
            "guardrails": {
                "input": {"enabled": True, "checks": []},
                "context": {"enabled": True, "checks": []},
                "output": {"enabled": True, "checks": []},
            },
        }

    def validate(self, config: Dict[str, Any]) -> List[str]:
        """Validate configuration against expected schema.

        Args:
            config: Configuration dictionary to validate

        Returns:
            List of validation error messages (empty if valid)
        """
        errors: List[str] = []

        # Check required top-level fields
        if "guardrails" not in config:
            errors.append("Missing required field 'guardrails'")
            return errors

        guardrails = config.get("guardrails", {})

        # Validate each layer
        for layer in ["input", "context", "output"]:
            if layer in guardrails:
                layer_config = guardrails[layer]
                layer_errors = self._validate_layer(layer, layer_config)
                errors.extend(layer_errors)

        # Validate execution settings
        if "execution" in config:
            for layer, settings in config["execution"].items():
                if "mode" in settings:
                    mode = settings["mode"]
                    if mode not in ["parallel", "sequential"]:
                        errors.append(
                            f"Invalid execution mode '{mode}' for {layer}. "
                            f"Must be 'parallel' or 'sequential'"
                        )

        return errors

    def _validate_layer(self, layer: str, layer_config: Dict[str, Any]) -> List[str]:
        """Validate a single layer configuration."""
        errors: List[str] = []

        if not isinstance(layer_config, dict):
            errors.append(f"Layer '{layer}' config must be a dictionary")
            return errors

        checks = layer_config.get("checks", [])
        if not isinstance(checks, list):
            errors.append(f"Layer '{layer}' checks must be a list")
            return errors

        for i, check in enumerate(checks):
            if not isinstance(check, dict):
                errors.append(f"Layer '{layer}' check {i} must be a dictionary")
                continue

            if "type" not in check:
                errors.append(f"Layer '{layer}' check {i} missing required 'type' field")

            if "threshold" in check:
                threshold = check["threshold"]
                if not isinstance(threshold, (int, float)) or not 0 <= threshold <= 1:
                    errors.append(
                        f"Layer '{layer}' check {i} threshold must be between 0 and 1"
                    )

            if "on_fail" in check:
                on_fail = check["on_fail"]
                if on_fail not in ["block", "warn", "sanitize"]:
                    errors.append(
                        f"Layer '{layer}' check {i} on_fail must be "
                        f"'block', 'warn', or 'sanitize'"
                    )

        return errors

    def get_enabled_guardrails(
        self, config: Dict[str, Any], layer: str
    ) -> List[GuardrailConfig]:
        """Get list of enabled guardrails for a specific layer.

        Args:
            config: The loaded configuration
            layer: Layer name (input, context, output)

        Returns:
            List of GuardrailConfig objects for enabled guardrails
        """
        guardrails_config = config.get("guardrails", {})
        layer_config = guardrails_config.get(layer, {})

        # Check if layer is enabled
        if not layer_config.get("enabled", True):
            return []

        checks = layer_config.get("checks", [])
        enabled_guardrails: List[GuardrailConfig] = []

        for check in checks:
            if check.get("enabled", True):
                enabled_guardrails.append(
                    GuardrailConfig(
                        type=check["type"],
                        enabled=True,
                        provider=check.get("provider", "llm_guard"),
                        priority=check.get("priority", 10),
                        threshold=check.get("threshold", 0.5),
                        on_fail=ActionOnFail(check.get("on_fail", "block")),
                        config=check.get("config", {}),
                    )
                )

        return enabled_guardrails

    def get_layer_config(self, config: Dict[str, Any], layer: str) -> LayerConfig:
        """Get the configuration for a specific layer.

        Args:
            config: The loaded configuration
            layer: Layer name (input, context, output)

        Returns:
            LayerConfig object for the specified layer
        """
        guardrails_config = config.get("guardrails", {})
        layer_config = guardrails_config.get(layer, {})
        execution_config = config.get("execution", {}).get(layer, {})

        return LayerConfig(
            enabled=layer_config.get("enabled", True),
            execution=ExecutionMode(execution_config.get("mode", "parallel")),
            timeout_ms=execution_config.get("timeout_ms", 5000),
            checks=self.get_enabled_guardrails(config, layer),
        )

    def clear_cache(self) -> None:
        """Clear the configuration cache."""
        self._cache.clear()

    def reload(self, agent_id: Optional[str] = None) -> Dict[str, Any]:
        """Reload configuration from disk, clearing cache.

        Args:
            agent_id: Optional agent ID to reload

        Returns:
            Fresh configuration dictionary
        """
        self.clear_cache()
        return self.load(agent_id=agent_id)
