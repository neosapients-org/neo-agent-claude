"""
Unit tests for NeMo Config Mapper and Generator.

Tests the mapping from Neo Guardrail Hub YAML config to NeMo Colang config
and the full generation pipeline.
"""

import pytest
from pathlib import Path
from tempfile import TemporaryDirectory

from neo_guardrail_hub.providers.nemo import (
    ConfigToNeMoMapper,
    NeMoConfigGenerator,
    generate_nemo_from_yaml,
    map_yaml_to_colang,
    ColangConfig,
    InputRailConfig,
    OutputRailConfig,
)


# Get the real configs directory
CONFIGS_PATH = Path(__file__).parent.parent.parent.parent.parent / "configs"


class TestConfigToNeMoMapper:
    """Tests for ConfigToNeMoMapper."""
    
    def test_mapper_initialization(self):
        """Test mapper initialization."""
        mapper = ConfigToNeMoMapper(config_path=CONFIGS_PATH)
        assert mapper.config_path == CONFIGS_PATH
        
    def test_map_default_config(self):
        """Test mapping the default.yaml configuration."""
        mapper = ConfigToNeMoMapper(config_path=CONFIGS_PATH)
        colang_config = mapper.map_to_colang()
        
        assert isinstance(colang_config, ColangConfig)
        # Default config has prompt_injection enabled, which maps to self_check_input
        assert colang_config.input_rails is not None
        assert colang_config.input_rails.self_check_input is True
        
    def test_map_to_nemo_config(self):
        """Test mapping to full NeMo config."""
        mapper = ConfigToNeMoMapper(config_path=CONFIGS_PATH)
        nemo_config = mapper.map_to_nemo_config()
        
        assert nemo_config.name == "default"
        assert nemo_config.enable_input_rails is True
        # Output rails enabled as of Sprint 3C.1 for self_check_output
        assert nemo_config.enable_output_rails is True
        # Dialog rails enabled in default config
        assert nemo_config.enable_dialog_rails is True
        
    def test_map_nemo_llm_config(self):
        """Test that LLM config is extracted from nemo section."""
        mapper = ConfigToNeMoMapper(config_path=CONFIGS_PATH)
        nemo_config = mapper.map_to_nemo_config()
        
        # Check LLM config from default.yaml nemo section
        assert nemo_config.llm.engine == "openai"
        # Model changed to gpt-4o-mini in Sprint 3B.1
        assert nemo_config.llm.model == "gpt-4o-mini"
        assert nemo_config.llm.temperature == 0.0
        
    def test_map_with_config_override(self):
        """Test mapping with configuration override."""
        mapper = ConfigToNeMoMapper(config_path=CONFIGS_PATH)
        
        # Override nemo.rails section (NeMo-specific config)
        override = {
            "nemo": {
                "rails": {
                    "input": {
                        "enabled": True,
                        "self_check_input": True
                    }
                }
            }
        }
        
        colang_config = mapper.map_to_colang(config_override=override)
        
        assert colang_config.input_rails is not None
        assert colang_config.input_rails.self_check_input is True
        
    def test_map_output_rails(self):
        """Test mapping output rails."""
        mapper = ConfigToNeMoMapper(config_path=CONFIGS_PATH)
        
        # Override nemo.rails.output section
        override = {
            "nemo": {
                "rails": {
                    "output": {
                        "enabled": True,
                        "self_check_output": True,
                        "self_check_facts": True,
                    }
                }
            }
        }
        
        colang_config = mapper.map_to_colang(config_override=override)
        
        assert colang_config.output_rails is not None
        assert colang_config.output_rails.self_check_output is True
        assert colang_config.output_rails.self_check_facts is True
        
    def test_map_dialog_rails(self):
        """Test mapping context rails (topical rails - NeMo-specific)."""
        mapper = ConfigToNeMoMapper(config_path=CONFIGS_PATH)
        
        # Override guardrails.context section with new structure
        override = {
            "guardrails": {
                "context": {
                    "enabled": True,
                    "checks": [
                        {
                            "type": "nemo_topical_rail",
                            "enabled": True,
                            "provider": "nemo",
                            "priority": 1,
                            "config": {
                                "allowed_topics": ["support"],
                                "disallowed_topics": ["politics"],
                            }
                        }
                    ]
                }
            }
        }
        
        colang_config = mapper.map_to_colang(config_override=override)
        
        assert colang_config.topical_rails is not None
        assert "support" in colang_config.topical_rails.allowed_topics
        assert "politics" in colang_config.topical_rails.disallowed_topics
        
    def test_disabled_layer_returns_none(self):
        """Test that disabled layers return None for rails config."""
        mapper = ConfigToNeMoMapper(config_path=CONFIGS_PATH)
        
        # Override guardrails sections with all NeMo checks disabled
        override = {
            "guardrails": {
                "input": {
                    "checks": [
                        {
                            "type": "nemo_self_check_input",
                            "enabled": False,
                            "provider": "nemo"
                        }
                    ]
                },
                "output": {
                    "checks": [
                        {
                            "type": "nemo_self_check_output",
                            "enabled": False,
                            "provider": "nemo"
                        },
                        {
                            "type": "nemo_self_check_facts",
                            "enabled": False,
                            "provider": "nemo"
                        }
                    ]
                },
                "context": {
                    "checks": [
                        {
                            "type": "nemo_topical_rail",
                            "enabled": False,
                            "provider": "nemo"
                        }
                    ]
                }
            }
        }
        
        colang_config = mapper.map_to_colang(config_override=override)
        
        assert colang_config.input_rails is None
        assert colang_config.output_rails is None
        assert colang_config.topical_rails is None
        
    def test_get_output_path_default(self):
        """Test getting output path from default config."""
        mapper = ConfigToNeMoMapper(config_path=CONFIGS_PATH)
        output_path = mapper.get_output_path()
        
        # Default config has output_path: null
        assert output_path is None
        
    def test_get_output_path_with_override(self):
        """Test getting output path with override."""
        mapper = ConfigToNeMoMapper(config_path=CONFIGS_PATH)
        
        override = {
            "nemo": {
                "output_path": "./custom/path"
            }
        }
        
        output_path = mapper.get_output_path(config_override=override)
        assert output_path == "./custom/path"


class TestConfigToNeMoMapperWithAgents:
    """Tests for ConfigToNeMoMapper with agent-specific configs."""
    
    def test_map_wealth_advisor_config(self):
        """Test mapping the wealth_advisor agent config."""
        mapper = ConfigToNeMoMapper(config_path=CONFIGS_PATH)
        colang_config = mapper.map_to_colang(agent_id="wealth_advisor")
        
        assert isinstance(colang_config, ColangConfig)
        # Wealth advisor has input rails enabled
        assert colang_config.input_rails is not None
        assert colang_config.input_rails.self_check_input is True
        
    def test_map_wealth_advisor_context_rails(self):
        """Test that wealth_advisor has dialog rails with topics."""
        mapper = ConfigToNeMoMapper(config_path=CONFIGS_PATH)
        colang_config = mapper.map_to_colang(agent_id="wealth_advisor")
        
        # Wealth advisor has dialog rails enabled with topics
        assert colang_config.topical_rails is not None
        assert "portfolio_overview" in colang_config.topical_rails.allowed_topics
        assert "stock_tips" in colang_config.topical_rails.disallowed_topics
        
    def test_map_wealth_advisor_nemo_config(self):
        """Test full NeMo config for wealth_advisor."""
        mapper = ConfigToNeMoMapper(config_path=CONFIGS_PATH)
        nemo_config = mapper.map_to_nemo_config(agent_id="wealth_advisor")
        
        assert nemo_config.name == "wealth_advisor"
        # Wealth advisor uses gpt-4
        assert nemo_config.llm.model == "gpt-4"
        # Dialog rails enabled
        assert nemo_config.enable_dialog_rails is True
        
    def test_map_test_agent_config(self):
        """Test mapping the test_agent config."""
        mapper = ConfigToNeMoMapper(config_path=CONFIGS_PATH)
        colang_config = mapper.map_to_colang(agent_id="test_agent")
        
        assert isinstance(colang_config, ColangConfig)
        assert colang_config.input_rails is not None
        assert colang_config.topical_rails is not None


class TestNeMoConfigGenerator:
    """Tests for NeMoConfigGenerator."""
    
    def test_generator_initialization(self):
        """Test generator initialization."""
        generator = NeMoConfigGenerator(config_path=CONFIGS_PATH)
        assert generator.config_path == CONFIGS_PATH
        
    def test_generate_creates_files(self):
        """Test that generate creates expected files."""
        generator = NeMoConfigGenerator(config_path=CONFIGS_PATH)
        
        with TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "nemo"
            result_path = generator.generate(output_path=output_path)
            
            assert result_path.exists()
            assert (result_path / "config.yml").exists()
            # Note: rails.co is no longer generated separately as of Sprint 3C.1
            # NeMo's built-in flows are used via config.yml rails section
            
    def test_generate_config_yml_content(self):
        """Test that config.yml has expected content."""
        generator = NeMoConfigGenerator(config_path=CONFIGS_PATH)
        
        with TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "nemo"
            result_path = generator.generate(output_path=output_path)
            
            config_content = (result_path / "config.yml").read_text()
            
            assert "models:" in config_content
            assert "rails:" in config_content
            
    def test_generate_rails_co_content(self):
        """Test that config.yml has expected rails content (rails.co is not generated separately)."""
        generator = NeMoConfigGenerator(config_path=CONFIGS_PATH)
        
        with TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "nemo"
            result_path = generator.generate(output_path=output_path)
            
            # Rails are now included in config.yml, not a separate rails.co file
            config_content = (result_path / "config.yml").read_text()
            
            assert "rails:" in config_content
            assert "self check input" in config_content
            
    def test_generate_for_wealth_advisor(self):
        """Test generation for wealth_advisor agent."""
        generator = NeMoConfigGenerator(config_path=CONFIGS_PATH)
        
        with TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "wealth_advisor"
            result_path = generator.generate(
                agent_id="wealth_advisor",
                output_path=output_path,
            )
            
            assert result_path.exists()
            
            # Check config.yml exists and has proper rails section
            config_content = (result_path / "config.yml").read_text()
            assert "rails:" in config_content
            
    def test_generate_for_test_agent(self):
        """Test generation for test_agent."""
        generator = NeMoConfigGenerator(config_path=CONFIGS_PATH)
        
        with TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test_agent"
            result_path = generator.generate(
                agent_id="test_agent",
                output_path=output_path,
            )
            
            assert result_path.exists()
            assert (result_path / "config.yml").exists()
            # Note: rails.co is no longer generated separately
            
    def test_generate_with_agent_id_creates_subdir(self):
        """Test generation with agent ID creates proper path."""
        generator = NeMoConfigGenerator(config_path=CONFIGS_PATH)
        
        with TemporaryDirectory() as tmpdir:
            # Don't specify output path, let it create agent-specific dir
            generator.DEFAULT_OUTPUT_DIR = tmpdir
            result_path = generator.generate(agent_id="test_agent")
            
            assert "test_agent" in str(result_path)
            
    def test_generate_colang_only(self):
        """Test generating only Colang content."""
        generator = NeMoConfigGenerator(config_path=CONFIGS_PATH)
        
        content = generator.generate_colang_only()
        
        assert isinstance(content, str)
        assert "define flow" in content
        
    def test_generate_colang_only_for_agent(self):
        """Test generating Colang for specific agent."""
        generator = NeMoConfigGenerator(config_path=CONFIGS_PATH)
        
        content = generator.generate_colang_only(agent_id="wealth_advisor")
        
        assert "define flow" in content
        # Should have topic blocking for wealth_advisor
        assert "block" in content and "topic" in content
        
    def test_generate_colang_only_to_file(self):
        """Test generating Colang files to a specific directory.
        
        Note: generate_colang_only now creates individual .co files in rails/ subdirectory.
        """
        generator = NeMoConfigGenerator(config_path=CONFIGS_PATH)
        
        with TemporaryDirectory() as tmpdir:
            base_path = Path(tmpdir) / "custom_output"
            content = generator.generate_colang_only(output_path=base_path)
            
            # Should create rails/ subdirectory with individual files
            rails_dir = base_path / "rails"
            assert rails_dir.exists()
            assert rails_dir.is_dir()
            
            # Should have created input.co and output.co at minimum
            assert (rails_dir / "input.co").exists()
            assert (rails_dir / "output.co").exists()
            
            # Content should be returned (combined for display)
            assert "define flow" in content
            
    def test_generate_with_override(self):
        """Test generation with config override."""
        generator = NeMoConfigGenerator(config_path=CONFIGS_PATH)
        
        # Override nemo.rails section (NeMo-specific)
        override = {
            "nemo": {
                "rails": {
                    "output": {
                        "enabled": True,
                        "self_check_facts": True
                    }
                }
            }
        }
        
        with TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "nemo"
            result_path = generator.generate(
                output_path=output_path,
                config_override=override,
            )
            
            # Config override is applied in generation
            config_content = (result_path / "config.yml").read_text()
            assert "self check output" in config_content


class TestConvenienceFunctions:
    """Tests for convenience functions."""
    
    def test_generate_nemo_from_yaml(self):
        """Test the generate_nemo_from_yaml convenience function."""
        with TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "nemo"
            result_path = generate_nemo_from_yaml(
                config_path=CONFIGS_PATH,
                output_path=output_path,
            )
            
            assert result_path.exists()
            assert (result_path / "config.yml").exists()
            # Note: rails.co is no longer generated separately
            
    def test_generate_nemo_from_yaml_for_agent(self):
        """Test generating NeMo from YAML for specific agent."""
        with TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "wealth_advisor"
            result_path = generate_nemo_from_yaml(
                config_path=CONFIGS_PATH,
                agent_id="wealth_advisor",
                output_path=output_path,
            )
            
            assert result_path.exists()
            
    def test_map_yaml_to_colang(self):
        """Test the map_yaml_to_colang convenience function."""
        colang_config = map_yaml_to_colang(config_path=CONFIGS_PATH)
        
        assert isinstance(colang_config, ColangConfig)
        
    def test_map_yaml_to_colang_for_agent(self):
        """Test mapping YAML to Colang for specific agent."""
        colang_config = map_yaml_to_colang(
            config_path=CONFIGS_PATH,
            agent_id="wealth_advisor",
        )
        
        assert isinstance(colang_config, ColangConfig)
        assert colang_config.topical_rails is not None


class TestRealDefaultYaml:
    """Tests that verify correct mapping of real default.yaml content.
    
    NOTE: NeMo and LLM Guard are SEPARATE providers.
    - NeMo reads from nemo.rails section (uses LLM-based self-checks)
    - LLM Guard reads from guardrails section (uses scanner-based checks)
    """
    
    def test_nemo_self_check_input_from_nemo_section(self):
        """Test that self_check_input is read from nemo.rails.input section."""
        mapper = ConfigToNeMoMapper(config_path=CONFIGS_PATH)
        colang_config = mapper.map_to_colang()
        
        # default.yaml has nemo.rails.input.self_check_input: true
        assert colang_config.input_rails is not None
        assert colang_config.input_rails.self_check_input is True
        
    def test_nemo_self_check_output_from_nemo_section(self):
        """Test that self_check_output is read from nemo.rails.output section.
        
        As of Sprint 3C.1, output rails are enabled in default.yaml
        for self_check_output functionality.
        """
        mapper = ConfigToNeMoMapper(config_path=CONFIGS_PATH)
        colang_config = mapper.map_to_colang()
        
        # default.yaml has nemo.rails.output.enabled: true, self_check_output: true
        assert colang_config.output_rails is not None
        assert colang_config.output_rails.self_check_output is True
        
    def test_generated_colang_has_expected_flows(self):
        """Test that generated Colang has flows matching nemo.rails config.
        
        As of Sprint 3C.1, both input and output rails are enabled in default.yaml.
        """
        generator = NeMoConfigGenerator(config_path=CONFIGS_PATH)
        content = generator.generate_colang_only()
        
        # Should have self check input (from nemo.rails.input.self_check_input)
        assert "define flow self check input" in content
        
        # Output rails are now enabled with self_check_output
        assert "define flow self check output" in content
        
    def test_default_yaml_has_nemo_section(self):
        """Test that default.yaml has nemo configuration section."""
        mapper = ConfigToNeMoMapper(config_path=CONFIGS_PATH)
        nemo_config = mapper.map_to_nemo_config()
        
        # Should have LLM config from nemo section
        assert nemo_config.llm is not None
        assert nemo_config.llm.engine == "openai"
