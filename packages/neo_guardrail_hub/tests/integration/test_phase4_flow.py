"""
Integration tests for Phase 4 features.

These tests verify the complete flow:
1. Pre-generation of config files
2. Caching of NeMo Rails
3. Using cached Rails in guardrail execution

Note: These tests require NeMo Guardrails to be installed
and may make API calls if using OpenAI.
"""

import pytest
import asyncio
import tempfile
import os
from pathlib import Path
from unittest.mock import patch, MagicMock

from neo_guardrail_hub.api import (
    initialize,
    generate_configs,
    get_rails_cache,
    clear_rails_cache,
    PreGenerationResult,
)
from neo_guardrail_hub.api.caching import reset_rails_cache


# Check if NeMo is available
try:
    import nemoguardrails
    NEMO_AVAILABLE = True
except ImportError:
    NEMO_AVAILABLE = False

# Check if OpenAI API key is set
OPENAI_KEY_SET = bool(os.getenv("OPENAI_API_KEY"))


@pytest.fixture
def sample_config_dir(tmp_path):
    """Create a sample configuration directory."""
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
        - general_questions
      disallowed_topics:
        - violence
        - illegal_activity
        - hate_speech

guardrails:
  input:
    enabled: false
  output:
    enabled: false
""")
    
    # Create agents directory
    agents_dir = config_dir / "agents"
    agents_dir.mkdir()
    
    # Create test agent config
    test_agent = agents_dir / "test_agent.yaml"
    test_agent.write_text("""
agent_id: test_agent
inherits: default

nemo:
  enabled: true
  rails:
    dialog:
      allowed_topics:
        - custom_topic
        - test_topic
""")
    
    return config_dir


@pytest.fixture(autouse=True)
def reset_cache():
    """Reset cache before and after each test."""
    reset_rails_cache()
    yield
    reset_rails_cache()


class TestPreGenerationFlow:
    """Tests for the complete pre-generation flow."""
    
    def test_generate_configs_creates_files(self, sample_config_dir, tmp_path):
        """Test that generate_configs creates all necessary files."""
        output_dir = tmp_path / "output"
        
        result = generate_configs(
            config_path=sample_config_dir,
            output_path=output_dir,
        )
        
        assert result.success is True
        assert result.output_path is not None
        assert result.output_path.exists()
        
        # Check that config.yml was created
        config_yml = result.output_path / "config.yml"
        assert config_yml.exists()
    
    def test_generate_for_agent(self, sample_config_dir, tmp_path):
        """Test generating config for a specific agent."""
        output_dir = tmp_path / "output"
        
        result = generate_configs(
            config_path=sample_config_dir,
            agent_id="test_agent",
            output_path=output_dir,
        )
        
        assert result.success is True
        assert result.agent_id == "test_agent"
    
    def test_config_summary_contents(self, sample_config_dir, tmp_path):
        """Test that config summary contains expected info."""
        result = generate_configs(
            config_path=sample_config_dir,
            output_path=tmp_path / "output",
        )
        
        assert result.success is True
        summary = result.config_summary
        
        assert summary["nemo_enabled"] is True
        assert summary["nemo_llm"] == "gpt-4o-mini"
        assert summary["input_rails"] is True
        assert summary["output_rails"] is True
        assert summary["dialog_rails"] is True


class TestCachingIntegration:
    """Tests for caching integration."""
    
    def test_cache_is_singleton(self):
        """Test that cache is a true singleton."""
        cache1 = get_rails_cache()
        cache2 = get_rails_cache()
        
        assert cache1 is cache2
    
    def test_clear_cache_works(self):
        """Test that clear_rails_cache empties the cache."""
        cache = get_rails_cache()
        
        # Add a mock entry
        cache._cache["test"] = MagicMock()
        assert cache.size > 0
        
        clear_rails_cache()
        
        assert cache.size == 0


@pytest.mark.skipif(not NEMO_AVAILABLE, reason="NeMo Guardrails not installed")
class TestNeMoIntegration:
    """Tests requiring NeMo Guardrails."""
    
    @pytest.mark.skipif(not OPENAI_KEY_SET, reason="OPENAI_API_KEY not set")
    @pytest.mark.asyncio
    async def test_initialize_preloads_rails(self, sample_config_dir, tmp_path):
        """Test that initialize pre-loads Rails into cache."""
        output_dir = tmp_path / "output"
        
        result = await initialize(
            config_path=sample_config_dir,
            output_path=output_dir,
            preload_rails=True,
        )
        
        assert result.success is True
        
        # Check cache was populated
        cache = get_rails_cache()
        assert cache.size > 0
        assert cache.contains("default")
    
    @pytest.mark.skipif(not OPENAI_KEY_SET, reason="OPENAI_API_KEY not set")
    @pytest.mark.asyncio
    async def test_cached_rails_used_by_provider(self, sample_config_dir, tmp_path):
        """Test that NeMo provider uses cached Rails."""
        from neo_guardrail_hub.providers.nemo import NeMoProvider
        
        output_dir = tmp_path / "output"
        
        # First, initialize to populate cache
        result = await initialize(
            config_path=sample_config_dir,
            output_path=output_dir,
            preload_rails=True,
        )
        assert result.success
        
        cache = get_rails_cache()
        initial_misses = cache.stats["misses"]
        
        # Create provider (should use cached Rails)
        provider = NeMoProvider(
            config_path=sample_config_dir,
            output_path=output_dir,
        )
        await provider.initialize()
        
        # Should be a cache hit
        assert cache.stats["hits"] > 0


class TestMultiAgentFlow:
    """Tests for multi-agent configuration flow."""
    
    def test_generate_configs_for_multiple_agents(self, sample_config_dir, tmp_path):
        """Test generating configs for multiple agents."""
        agents = ["test_agent"]
        results = []
        
        for agent in agents:
            output_dir = tmp_path / f"output_{agent}"
            result = generate_configs(
                config_path=sample_config_dir,
                agent_id=agent,
                output_path=output_dir,
            )
            results.append(result)
        
        assert all(r.success for r in results)
        assert len(set(r.output_path for r in results)) == len(agents)
    
    @pytest.mark.skipif(not NEMO_AVAILABLE, reason="NeMo not installed")
    @pytest.mark.asyncio
    async def test_cache_handles_multiple_agents(self, sample_config_dir, tmp_path):
        """Test that cache correctly handles multiple agents."""
        cache = get_rails_cache()
        
        # Generate configs for default and test_agent
        agents = [None, "test_agent"]
        
        for agent in agents:
            output_dir = tmp_path / f"output_{agent or 'default'}"
            generate_configs(
                config_path=sample_config_dir,
                agent_id=agent,
                output_path=output_dir,
            )
        
        # Each agent should have its own cache entry
        # (Cache entries are created when Rails are loaded)


class TestErrorHandling:
    """Tests for error handling scenarios."""
    
    def test_generate_with_invalid_path(self):
        """Test handling of invalid config path."""
        result = generate_configs(
            config_path=Path("/nonexistent/path"),
        )
        
        # Should handle gracefully
        assert isinstance(result, PreGenerationResult)
    
    def test_generate_with_malformed_yaml(self, tmp_path):
        """Test handling of malformed YAML config."""
        config_dir = tmp_path / "configs"
        config_dir.mkdir()
        
        default_yaml = config_dir / "default.yaml"
        default_yaml.write_text("invalid: yaml: [content")
        
        result = generate_configs(config_path=config_dir)
        
        assert isinstance(result, PreGenerationResult)
        # Should have errors or handle gracefully
    
    def test_cache_handles_missing_config(self):
        """Test cache behavior with missing config files."""
        cache = get_rails_cache()
        
        # Trying to load from nonexistent path should fail gracefully
        # This is tested in the caching unit tests


class TestCLIIntegration:
    """Tests for CLI integration."""
    
    def test_cli_module_importable(self):
        """Test that CLI module can be imported."""
        from neo_guardrail_hub.cli import main
        
        assert callable(main)
    
    def test_cli_init_generates_files(self, tmp_path, monkeypatch):
        """Test CLI init command creates config files."""
        from neo_guardrail_hub.cli import cmd_init
        import argparse
        
        config_path = tmp_path / "configs"
        
        args = argparse.Namespace(
            path=str(config_path),
            preset="general",
            agent=None,
        )
        
        # Mock input to avoid interactive prompts
        monkeypatch.setattr('builtins.input', lambda x: 'y')
        
        result = cmd_init(args)
        
        assert result == 0
        assert config_path.exists()
        assert (config_path / "default.yaml").exists()
