"""Tests for config loading."""

from pathlib import Path
from typing import Any, Dict

import pytest

from neo_guardrail_hub.core.config import ConfigLoader
from neo_guardrail_hub.core.exceptions import ConfigurationError
from neo_guardrail_hub.core.models import GuardrailLayer


class TestConfigLoader:
    """Tests for ConfigLoader class."""

    def test_load_config(self, temp_config_dir: Path):
        """Test loading configuration from directory."""
        loader = ConfigLoader(config_path=temp_config_dir)
        config = loader.load()

        assert config is not None
        assert config.get("version") == "1.0"
        assert "guardrails" in config

    def test_load_missing_config(self, tmp_path: Path):
        """Test loading from non-existent directory returns defaults."""
        loader = ConfigLoader(config_path=tmp_path / "nonexistent")
        config = loader.load()

        # Should return default config
        assert config is not None
        assert "guardrails" in config

    def test_validate_valid_config(self, sample_config: Dict[str, Any]):
        """Test validating a valid config."""
        loader = ConfigLoader()
        # Should not raise
        errors = loader.validate(sample_config)
        assert errors == []

    def test_get_enabled_guardrails(self, temp_config_dir: Path):
        """Test getting enabled guardrails."""
        loader = ConfigLoader(config_path=temp_config_dir)
        config = loader.load()

        input_guardrails = loader.get_enabled_guardrails(config, "input")
        assert len(input_guardrails) >= 1

        context_guardrails = loader.get_enabled_guardrails(config, "context")
        assert len(context_guardrails) == 0

    def test_get_layer_config(self, temp_config_dir: Path):
        """Test getting layer config."""
        loader = ConfigLoader(config_path=temp_config_dir)
        config = loader.load()

        input_config = loader.get_layer_config(config, "input")
        assert input_config is not None
        assert input_config.enabled is True

    def test_merge_configs(self):
        """Test config merging with inheritance."""
        loader = ConfigLoader()

        base = {
            "version": "1.0",
            "execution": {"input": {"timeout_ms": 5000}},
            "guardrails": {
                "input": {"enabled": True, "checks": []},
            },
        }

        override = {
            "execution": {"input": {"timeout_ms": 3000}},
        }

        merged = loader._merge_configs(base, override)

        assert merged["version"] == "1.0"  # From base
        assert merged["execution"]["input"]["timeout_ms"] == 3000  # Overridden


class TestConfigDefaults:
    """Tests for configuration defaults."""

    def test_default_config_has_guardrails(self):
        """Test default config has guardrails key."""
        loader = ConfigLoader()
        config = loader.load()

        assert "guardrails" in config
        assert "input" in config.get("guardrails", {})
        assert "output" in config.get("guardrails", {})
