"""
Unit tests for ConfigGenerator.

Tests the generation of NeMo config.yml files from Neo Guardrail Hub configuration.
"""

import pytest
from pathlib import Path

from neo_guardrail_hub.providers.nemo.generators import ConfigGenerator
from neo_guardrail_hub.providers.nemo.generators.models import (
    NeMoGuardrailsConfig,
    ColangConfig,
    LLMConfig,
    InputRailConfig,
    OutputRailConfig,
    TopicalRailConfig,
)


class TestConfigGeneratorInit:
    """Test ConfigGenerator initialization."""
    
    def test_init_with_minimal_config(self):
        """Test initialization with minimal configuration."""
        nemo_config = NeMoGuardrailsConfig()
        generator = ConfigGenerator(nemo_config)
        
        assert generator.nemo_config == nemo_config
        assert generator.presets_path is None
        
    def test_init_with_presets_path(self, tmp_path):
        """Test initialization with presets path."""
        nemo_config = NeMoGuardrailsConfig()
        presets_path = tmp_path / "presets"
        
        generator = ConfigGenerator(nemo_config, presets_path=presets_path)
        
        assert generator.presets_path == presets_path
        
    def test_init_with_custom_llm(self):
        """Test initialization with custom LLM configuration."""
        llm_config = LLMConfig(
            engine="azure",
            model="gpt-4-turbo",
            temperature=0.7,
            max_tokens=2000,
        )
        nemo_config = NeMoGuardrailsConfig(llm=llm_config)
        generator = ConfigGenerator(nemo_config)
        
        assert generator.nemo_config.llm.engine == "azure"
        assert generator.nemo_config.llm.model == "gpt-4-turbo"


class TestConfigGeneratorGenerate:
    """Test ConfigGenerator.generate() method."""
    
    def test_generate_default_config(self):
        """Test generating default configuration."""
        nemo_config = NeMoGuardrailsConfig()
        generator = ConfigGenerator(nemo_config)
        
        config = generator.generate()
        
        assert "models" in config
        assert len(config["models"]) == 1
        assert config["models"][0]["type"] == "main"
        
    def test_generate_models_section(self):
        """Test models section generation."""
        llm_config = LLMConfig(
            engine="openai",
            model="gpt-4",
            temperature=0.5,
            max_tokens=1500,
        )
        nemo_config = NeMoGuardrailsConfig(llm=llm_config)
        generator = ConfigGenerator(nemo_config)
        
        config = generator.generate()
        
        model = config["models"][0]
        assert model["engine"] == "openai"
        assert model["model"] == "gpt-4"
        assert model["temperature"] == 0.5
        assert model["max_tokens"] == 1500
        
    def test_generate_without_temperature_when_zero(self):
        """Test that temperature is omitted when zero."""
        llm_config = LLMConfig(engine="openai", model="gpt-4", temperature=0.0)
        nemo_config = NeMoGuardrailsConfig(llm=llm_config)
        generator = ConfigGenerator(nemo_config)
        
        config = generator.generate()
        
        model = config["models"][0]
        assert "temperature" not in model
        
    def test_generate_with_streaming_enabled(self):
        """Test streaming configuration."""
        nemo_config = NeMoGuardrailsConfig(streaming=True)
        generator = ConfigGenerator(nemo_config)
        
        config = generator.generate()
        
        assert config.get("streaming") is True
        
    def test_generate_with_streaming_disabled(self):
        """Test that streaming is not included when disabled."""
        nemo_config = NeMoGuardrailsConfig(streaming=False)
        generator = ConfigGenerator(nemo_config)
        
        config = generator.generate()
        
        assert "streaming" not in config


class TestConfigGeneratorInputRails:
    """Test input rails configuration generation."""
    
    def test_generate_with_input_rails_enabled(self):
        """Test config with input rails enabled."""
        colang = ColangConfig(
            input_rails=InputRailConfig(self_check_input=True)
        )
        nemo_config = NeMoGuardrailsConfig(
            enable_input_rails=True,
            colang=colang,
        )
        generator = ConfigGenerator(nemo_config)
        
        config = generator.generate()
        
        assert "rails" in config
        assert "input" in config["rails"]
        assert "flows" in config["rails"]["input"]
        assert "self check input" in config["rails"]["input"]["flows"]
        
    def test_generate_without_input_rails(self):
        """Test config without input rails."""
        nemo_config = NeMoGuardrailsConfig(enable_input_rails=False)
        generator = ConfigGenerator(nemo_config)
        
        config = generator.generate()
        
        # Rails section should not have input if disabled
        if "rails" in config:
            assert "input" not in config.get("rails", {})


class TestConfigGeneratorOutputRails:
    """Test output rails configuration generation."""
    
    def test_generate_with_output_rails(self):
        """Test config with output rails enabled."""
        colang = ColangConfig(
            output_rails=OutputRailConfig(self_check_output=True)
        )
        nemo_config = NeMoGuardrailsConfig(
            enable_output_rails=True,
            colang=colang,
        )
        generator = ConfigGenerator(nemo_config)
        
        config = generator.generate()
        
        assert "rails" in config
        assert "output" in config["rails"]
        assert "self check output" in config["rails"]["output"]["flows"]
        
    def test_generate_with_fact_checking(self):
        """Test config with fact checking enabled."""
        colang = ColangConfig(
            output_rails=OutputRailConfig(
                self_check_output=True,
                self_check_facts=True,
            )
        )
        nemo_config = NeMoGuardrailsConfig(
            enable_output_rails=True,
            colang=colang,
        )
        generator = ConfigGenerator(nemo_config)
        
        config = generator.generate()
        
        flows = config["rails"]["output"]["flows"]
        assert "self check facts" in flows
        
    def test_generate_with_hallucination_check(self):
        """Test config with hallucination check enabled."""
        colang = ColangConfig(
            output_rails=OutputRailConfig(
                self_check_hallucination=True,
            )
        )
        nemo_config = NeMoGuardrailsConfig(
            enable_output_rails=True,
            colang=colang,
        )
        generator = ConfigGenerator(nemo_config)
        
        config = generator.generate()
        
        flows = config["rails"]["output"]["flows"]
        assert "self check hallucination" in flows


class TestConfigGeneratorDialogRails:
    """Test dialog/topical rails configuration generation."""
    
    def test_generate_with_disallowed_topics(self):
        """Test config with disallowed topics enables dialog rails.
        
        Note: The actual topic blocking flows are defined in .co files,
        not in config.yml. The config.yml just enables dialog rails.
        """
        colang = ColangConfig(
            topical_rails=TopicalRailConfig(
                disallowed_topics=["politics", "religion"]
            )
        )
        nemo_config = NeMoGuardrailsConfig(
            enable_dialog_rails=True,  # Dialog rails must be enabled
            colang=colang,
        )
        generator = ConfigGenerator(nemo_config)
        
        config = generator.generate()
        
        # Dialog rails are enabled via config.yml settings
        assert "rails" in config
        assert "dialog" in config["rails"]
        # Dialog settings (flows are in .co files, not config.yml)
        assert "single_call" in config["rails"]["dialog"]
        assert config["rails"]["dialog"]["single_call"]["enabled"] is True
        
    def test_generate_with_allowed_topics(self):
        """Test config with topic whitelist."""
        colang = ColangConfig(
            topical_rails=TopicalRailConfig(
                allowed_topics=["customer_support", "billing"],
                disallowed_topics=["off_topic"]  # Need disallowed to generate flows
            )
        )
        nemo_config = NeMoGuardrailsConfig(
            enable_dialog_rails=True,  # Dialog rails must be enabled
            colang=colang,
        )
        generator = ConfigGenerator(nemo_config)
        
        config = generator.generate()
        
        assert "rails" in config
        assert "dialog" in config["rails"]


class TestConfigGeneratorYamlOutput:
    """Test YAML output generation."""
    
    def test_generate_yaml_has_header(self):
        """Test that YAML output has auto-generation header."""
        nemo_config = NeMoGuardrailsConfig()
        generator = ConfigGenerator(nemo_config)
        
        yaml_content = generator.generate_yaml()
        
        assert "Auto-generated by Neo Guardrail Hub" in yaml_content
        assert "NeMo Guardrails configuration" in yaml_content
        
    def test_generate_yaml_is_valid_yaml(self):
        """Test that output is valid YAML."""
        import yaml as pyyaml
        
        nemo_config = NeMoGuardrailsConfig(
            enable_input_rails=True,
            enable_output_rails=True,
            colang=ColangConfig(
                input_rails=InputRailConfig(self_check_input=True),
                output_rails=OutputRailConfig(self_check_output=True),
            ),
        )
        generator = ConfigGenerator(nemo_config)
        
        yaml_content = generator.generate_yaml()
        
        # Should not raise an exception
        parsed = pyyaml.safe_load(yaml_content)
        assert parsed is not None
        assert "models" in parsed


class TestConfigGeneratorWrite:
    """Test file writing functionality."""
    
    def test_write_creates_file(self, tmp_path):
        """Test that write creates the config file."""
        nemo_config = NeMoGuardrailsConfig()
        generator = ConfigGenerator(nemo_config)
        
        output_file = tmp_path / "config.yml"
        generator.write(output_file)
        
        assert output_file.exists()
        
    def test_write_creates_parent_directories(self, tmp_path):
        """Test that write creates parent directories."""
        nemo_config = NeMoGuardrailsConfig()
        generator = ConfigGenerator(nemo_config)
        
        output_file = tmp_path / "nested" / "deep" / "config.yml"
        generator.write(output_file)
        
        assert output_file.exists()
        assert (tmp_path / "nested" / "deep").is_dir()
        
    def test_write_content_is_valid(self, tmp_path):
        """Test that written content is valid YAML."""
        import yaml as pyyaml
        
        nemo_config = NeMoGuardrailsConfig(
            enable_input_rails=True,
            colang=ColangConfig(
                input_rails=InputRailConfig(self_check_input=True)
            ),
        )
        generator = ConfigGenerator(nemo_config)
        
        output_file = tmp_path / "config.yml"
        generator.write(output_file)
        
        content = output_file.read_text()
        parsed = pyyaml.safe_load(content)
        
        assert parsed is not None
        assert "models" in parsed
        assert "rails" in parsed


class TestConfigGeneratorFullConfiguration:
    """Test full configuration scenarios."""
    
    def test_complete_configuration(self):
        """Test generating a complete configuration."""
        llm_config = LLMConfig(
            engine="openai",
            model="gpt-4",
            temperature=0.3,
            max_tokens=2048,
        )
        colang = ColangConfig(
            input_rails=InputRailConfig(
                self_check_input=True,
                input_length_check=True,
                max_input_length=4000,
            ),
            output_rails=OutputRailConfig(
                self_check_output=True,
                self_check_facts=True,
            ),
            topical_rails=TopicalRailConfig(
                disallowed_topics=["violence", "illegal_activities"]
            ),
        )
        nemo_config = NeMoGuardrailsConfig(
            llm=llm_config,
            colang=colang,
            enable_input_rails=True,
            enable_output_rails=True,
            streaming=True,
        )
        generator = ConfigGenerator(nemo_config)
        
        config = generator.generate()
        
        # Verify all sections
        assert config["models"][0]["model"] == "gpt-4"
        assert config["streaming"] is True
        assert "rails" in config
        assert "input" in config["rails"]
        assert "output" in config["rails"]
        
    def test_minimal_configuration(self):
        """Test generating minimal configuration."""
        nemo_config = NeMoGuardrailsConfig()
        generator = ConfigGenerator(nemo_config)
        
        config = generator.generate()
        
        # Should have only models section
        assert "models" in config
        assert len(config["models"]) == 1
