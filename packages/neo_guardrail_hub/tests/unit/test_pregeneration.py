"""
Unit tests for the Pre-generation API (Phase 4).

Tests cover:
1. generate_configs() function
2. initialize() function
3. PreGenerationResult dataclass
4. Error handling and edge cases
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from neo_guardrail_hub.api.pregeneration import (
    generate_configs,
    generate_configs_sync,
    initialize,
    initialize_sync,
    PreGenerationResult,
    _build_config_summary,
    _validate_generated_config,
)


class TestPreGenerationResult:
    """Tests for PreGenerationResult dataclass."""
    
    def test_success_result(self):
        """Test successful result."""
        result = PreGenerationResult(
            success=True,
            output_path=Path("/tmp/test"),
            agent_id="test_agent",
        )
        
        assert result.success is True
        assert result.output_path == Path("/tmp/test")
        assert result.agent_id == "test_agent"
        assert bool(result) is True
    
    def test_failure_result(self):
        """Test failed result."""
        result = PreGenerationResult(
            success=False,
            errors=["Error 1", "Error 2"],
        )
        
        assert result.success is False
        assert len(result.errors) == 2
        assert bool(result) is False
    
    def test_warnings(self):
        """Test result with warnings."""
        result = PreGenerationResult(
            success=True,
            warnings=["Warning 1"],
        )
        
        assert result.success is True
        assert len(result.warnings) == 1


class TestGenerateConfigs:
    """Tests for generate_configs function."""
    
    @pytest.fixture
    def temp_config_dir(self, tmp_path):
        """Create a temporary config directory."""
        config_dir = tmp_path / "configs"
        config_dir.mkdir()
        
        # Create default.yaml
        default_yaml = config_dir / "default.yaml"
        default_yaml.write_text("""
version: "1.0"
enabled: true

nemo:
  enabled: true
  llm:
    engine: openai
    model: gpt-4o-mini
    temperature: 0.0
  rails:
    input:
      enabled: true
      self_check_input: true
    output:
      enabled: true
      self_check_output: true
    dialog:
      enabled: true
      allowed_topics:
        - greeting
        - help
      disallowed_topics:
        - violence

guardrails:
  input:
    enabled: false
  output:
    enabled: false
""")
        
        return config_dir
    
    @pytest.fixture
    def temp_config_dir_with_agent(self, temp_config_dir):
        """Create config with agent-specific config."""
        agents_dir = temp_config_dir / "agents"
        agents_dir.mkdir()
        
        agent_yaml = agents_dir / "test_agent.yaml"
        agent_yaml.write_text("""
agent_id: test_agent
inherits: default

nemo:
  enabled: true
  rails:
    dialog:
      allowed_topics:
        - custom_topic
""")
        
        return temp_config_dir
    
    def test_generate_configs_success(self, temp_config_dir, tmp_path):
        """Test successful config generation."""
        output_dir = tmp_path / "output"
        
        result = generate_configs(
            config_path=temp_config_dir,
            output_path=output_dir,
        )
        
        assert result.success is True
        assert result.output_path is not None
        assert result.output_path.exists()
    
    def test_generate_configs_with_agent(self, temp_config_dir_with_agent, tmp_path):
        """Test config generation for specific agent."""
        output_dir = tmp_path / "output"
        
        result = generate_configs(
            config_path=temp_config_dir_with_agent,
            agent_id="test_agent",
            output_path=output_dir,
        )
        
        assert result.success is True
        assert result.agent_id == "test_agent"
    
    def test_generate_configs_nemo_disabled(self, tmp_path):
        """Test behavior when NeMo is disabled."""
        config_dir = tmp_path / "configs"
        config_dir.mkdir()
        
        default_yaml = config_dir / "default.yaml"
        default_yaml.write_text("""
version: "1.0"
nemo:
  enabled: false

guardrails:
  input:
    enabled: false
  output:
    enabled: false
""")
        
        result = generate_configs(config_path=config_dir)
        
        assert result.success is True
        assert "NeMo is not enabled" in result.warnings[0]
    
    def test_generate_configs_existing_files_no_force(self, temp_config_dir, tmp_path):
        """Test that existing files are not overwritten without force."""
        output_dir = tmp_path / "output"
        output_dir.mkdir(parents=True)
        
        # Create existing config
        (output_dir / "config.yml").write_text("existing: true")
        
        result = generate_configs(
            config_path=temp_config_dir,
            output_path=output_dir,
            force=False,
        )
        
        assert result.success is True
        assert any("already exist" in w for w in result.warnings)
    
    def test_generate_configs_force_overwrite(self, temp_config_dir, tmp_path):
        """Test force overwrite of existing files."""
        output_dir = tmp_path / "output"
        output_dir.mkdir(parents=True)
        
        # Create existing config
        (output_dir / "config.yml").write_text("existing: true")
        
        result = generate_configs(
            config_path=temp_config_dir,
            output_path=output_dir,
            force=True,
        )
        
        assert result.success is True
    
    def test_generate_configs_missing_config(self, tmp_path):
        """Test behavior with missing config file."""
        config_dir = tmp_path / "nonexistent"
        
        result = generate_configs(config_path=config_dir)
        
        # Should handle gracefully - either success with defaults or error
        # Depends on implementation behavior
        assert isinstance(result, PreGenerationResult)
    
    def test_config_summary(self, temp_config_dir, tmp_path):
        """Test that config summary is generated."""
        result = generate_configs(
            config_path=temp_config_dir,
            output_path=tmp_path / "output",
        )
        
        assert result.success is True
        assert "nemo_enabled" in result.config_summary
        assert result.config_summary["nemo_enabled"] is True


class TestInitialize:
    """Tests for initialize function."""
    
    @pytest.fixture
    def temp_config_dir(self, tmp_path):
        """Create a temporary config directory."""
        config_dir = tmp_path / "configs"
        config_dir.mkdir()
        
        default_yaml = config_dir / "default.yaml"
        default_yaml.write_text("""
version: "1.0"
enabled: true

nemo:
  enabled: true
  llm:
    engine: openai
    model: gpt-4o-mini
  rails:
    input:
      enabled: true
    output:
      enabled: true
    dialog:
      enabled: true
      allowed_topics: [greeting]
      disallowed_topics: [violence]

guardrails:
  input:
    enabled: false
  output:
    enabled: false
""")
        
        return config_dir
    
    @pytest.mark.asyncio
    async def test_initialize_generates_configs(self, temp_config_dir, tmp_path):
        """Test that initialize generates config files."""
        output_dir = tmp_path / "output"
        
        result = await initialize(
            config_path=temp_config_dir,
            output_path=output_dir,
            preload_rails=False,  # Skip Rails loading in test
        )
        
        assert result.success is True
        assert result.output_path is not None


class TestBuildConfigSummary:
    """Tests for _build_config_summary helper."""
    
    def test_full_config(self):
        """Test summary of full config."""
        config = {
            "nemo": {
                "enabled": True,
                "llm": {"model": "gpt-4"},
                "preset": "financial",
                "rails": {
                    "input": {"enabled": True},
                    "output": {"enabled": True},
                    "dialog": {"enabled": True},
                },
            },
            "guardrails": {
                "input": {"enabled": True},
                "output": {"enabled": False},
            },
        }
        
        summary = _build_config_summary(config)
        
        assert summary["nemo_enabled"] is True
        assert summary["nemo_llm"] == "gpt-4"
        assert summary["nemo_preset"] == "financial"
        assert summary["input_rails"] is True
        assert summary["output_rails"] is True
        assert summary["dialog_rails"] is True
        assert summary["llm_guard_input"] is True
        assert summary["llm_guard_output"] is False
    
    def test_minimal_config(self):
        """Test summary of minimal config."""
        config = {}
        
        summary = _build_config_summary(config)
        
        assert summary["nemo_enabled"] is False


class TestValidateGeneratedConfig:
    """Tests for _validate_generated_config helper."""
    
    def test_valid_config(self, tmp_path):
        """Test validation of valid config."""
        config_dir = tmp_path / "config"
        config_dir.mkdir()
        
        config_yml = config_dir / "config.yml"
        config_yml.write_text("""
models:
  - type: main
    engine: openai
    model: gpt-4

rails:
  input:
    flows: []
""")
        
        warnings = _validate_generated_config(config_dir)
        
        assert len(warnings) == 0
    
    def test_missing_config_yml(self, tmp_path):
        """Test validation when config.yml is missing."""
        config_dir = tmp_path / "config"
        config_dir.mkdir()
        
        warnings = _validate_generated_config(config_dir)
        
        assert len(warnings) > 0
        assert "Missing config.yml" in warnings[0]
    
    def test_invalid_yaml(self, tmp_path):
        """Test validation of invalid YAML."""
        config_dir = tmp_path / "config"
        config_dir.mkdir()
        
        config_yml = config_dir / "config.yml"
        config_yml.write_text("invalid: yaml: content: [")
        
        warnings = _validate_generated_config(config_dir)
        
        assert len(warnings) > 0


class TestSyncWrappers:
    """Tests for synchronous wrapper functions."""
    
    def test_generate_configs_sync(self, tmp_path):
        """Test synchronous generate_configs wrapper."""
        config_dir = tmp_path / "configs"
        config_dir.mkdir()
        
        default_yaml = config_dir / "default.yaml"
        default_yaml.write_text("""
version: "1.0"
nemo:
  enabled: false
""")
        
        result = generate_configs_sync(config_path=config_dir)
        
        assert isinstance(result, PreGenerationResult)
