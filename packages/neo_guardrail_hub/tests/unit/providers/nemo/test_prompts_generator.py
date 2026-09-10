"""
Unit tests for PromptsGenerator.

Tests the generation of NeMo prompts.yml files from Neo Guardrail Hub configuration.
"""

import pytest
from pathlib import Path

from neo_guardrail_hub.providers.nemo.generators import PromptsGenerator
from neo_guardrail_hub.providers.nemo.generators.prompts_generator import PromptStyle
from neo_guardrail_hub.providers.nemo.generators.models import (
    ColangConfig,
    InputRailConfig,
    OutputRailConfig,
    TopicalRailConfig,
)


class TestPromptsGeneratorInit:
    """Test PromptsGenerator initialization."""
    
    def test_init_with_minimal_config(self):
        """Test initialization with minimal configuration."""
        colang_config = ColangConfig()
        generator = PromptsGenerator(colang_config)
        
        assert generator.colang_config == colang_config
        assert generator.style == "simple"
        
    def test_init_with_simple_style(self):
        """Test initialization with simple style."""
        colang_config = ColangConfig()
        generator = PromptsGenerator(colang_config, style="simple")
        
        assert generator.style == "simple"
        
    def test_init_with_complex_style(self):
        """Test initialization with complex style."""
        colang_config = ColangConfig()
        generator = PromptsGenerator(colang_config, style="complex")
        
        assert generator.style == "complex"
        
    def test_init_with_templates_path(self, tmp_path):
        """Test initialization with custom templates path."""
        colang_config = ColangConfig()
        templates_path = tmp_path / "templates"
        
        generator = PromptsGenerator(colang_config, templates_path=templates_path)
        
        assert generator.templates_path == templates_path


class TestPromptsGeneratorGenerate:
    """Test PromptsGenerator.generate() method."""
    
    def test_generate_empty_config(self):
        """Test generating with empty configuration."""
        colang_config = ColangConfig()
        generator = PromptsGenerator(colang_config)
        
        prompts = generator.generate()
        
        assert "prompts" in prompts
        assert isinstance(prompts["prompts"], list)
        
    def test_generate_with_self_check_input(self):
        """Test generating self_check_input prompt."""
        colang_config = ColangConfig(
            input_rails=InputRailConfig(self_check_input=True)
        )
        generator = PromptsGenerator(colang_config)
        
        prompts = generator.generate()
        
        prompt_tasks = [p.get("task") for p in prompts["prompts"]]
        assert "self_check_input" in prompt_tasks
        
    def test_generate_with_self_check_output(self):
        """Test generating self_check_output prompt."""
        colang_config = ColangConfig(
            output_rails=OutputRailConfig(self_check_output=True)
        )
        generator = PromptsGenerator(colang_config)
        
        prompts = generator.generate()
        
        prompt_tasks = [p.get("task") for p in prompts["prompts"]]
        assert "self_check_output" in prompt_tasks
        
    def test_generate_with_self_check_facts(self):
        """Test generating self_check_facts prompt."""
        colang_config = ColangConfig(
            output_rails=OutputRailConfig(self_check_facts=True)
        )
        generator = PromptsGenerator(colang_config)
        
        prompts = generator.generate()
        
        prompt_tasks = [p.get("task") for p in prompts["prompts"]]
        assert "self_check_facts" in prompt_tasks
        
    def test_generate_with_hallucination_check(self):
        """Test generating self_check_hallucination prompt."""
        colang_config = ColangConfig(
            output_rails=OutputRailConfig(self_check_hallucination=True)
        )
        generator = PromptsGenerator(colang_config)
        
        prompts = generator.generate()
        
        prompt_tasks = [p.get("task") for p in prompts["prompts"]]
        assert "self_check_hallucination" in prompt_tasks


class TestPromptsGeneratorStyles:
    """Test different prompt styles."""
    
    def test_simple_style_input_prompt(self):
        """Test simple style input prompt."""
        colang_config = ColangConfig(
            input_rails=InputRailConfig(self_check_input=True)
        )
        generator = PromptsGenerator(colang_config, style="simple")
        
        prompts = generator.generate()
        
        input_prompt = next(
            p for p in prompts["prompts"] if p["task"] == "self_check_input"
        )
        content = input_prompt["content"]
        
        # Simple prompts should be more concise
        assert "{{ user_input }}" in content
        assert "safety policy" in content.lower()
        
    def test_complex_style_input_prompt(self):
        """Test complex style input prompt."""
        colang_config = ColangConfig(
            input_rails=InputRailConfig(self_check_input=True)
        )
        generator = PromptsGenerator(colang_config, style="complex")
        
        prompts = generator.generate()
        
        input_prompt = next(
            p for p in prompts["prompts"] if p["task"] == "self_check_input"
        )
        content = input_prompt["content"]
        
        # Complex prompts should be more detailed
        assert "{{ user_input }}" in content
        # Complex prompts have more detailed conditions
        assert len(content) > 200  # Complex prompts are longer
        
    def test_simple_style_output_prompt(self):
        """Test simple style output prompt."""
        colang_config = ColangConfig(
            output_rails=OutputRailConfig(self_check_output=True)
        )
        generator = PromptsGenerator(colang_config, style="simple")
        
        prompts = generator.generate()
        
        output_prompt = next(
            p for p in prompts["prompts"] if p["task"] == "self_check_output"
        )
        content = output_prompt["content"]
        
        assert "{{ bot_response }}" in content
        
    def test_complex_style_output_prompt(self):
        """Test complex style output prompt."""
        colang_config = ColangConfig(
            output_rails=OutputRailConfig(self_check_output=True)
        )
        generator = PromptsGenerator(colang_config, style="complex")
        
        prompts = generator.generate()
        
        output_prompt = next(
            p for p in prompts["prompts"] if p["task"] == "self_check_output"
        )
        content = output_prompt["content"]
        
        assert "{{ bot_response }}" in content
        assert len(content) > 200  # Complex prompts are longer


class TestPromptsGeneratorFactsPrompts:
    """Test fact-checking prompt generation."""
    
    def test_simple_facts_prompt(self):
        """Test simple style facts prompt."""
        colang_config = ColangConfig(
            output_rails=OutputRailConfig(self_check_facts=True)
        )
        generator = PromptsGenerator(colang_config, style="simple")
        
        prompts = generator.generate()
        
        facts_prompt = next(
            p for p in prompts["prompts"] if p["task"] == "self_check_facts"
        )
        content = facts_prompt["content"]
        
        # Simple style uses {{ evidence }} instead of {{ context }}
        assert "{{ evidence }}" in content or "{{ context }}" in content
        assert "{{ response }}" in content or "{{ bot_response }}" in content
        
    def test_complex_facts_prompt(self):
        """Test complex style facts prompt."""
        colang_config = ColangConfig(
            output_rails=OutputRailConfig(self_check_facts=True)
        )
        generator = PromptsGenerator(colang_config, style="complex")
        
        prompts = generator.generate()
        
        facts_prompt = next(
            p for p in prompts["prompts"] if p["task"] == "self_check_facts"
        )
        content = facts_prompt["content"]
        
        assert "{{ context }}" in content or "evidence" in content.lower()


class TestPromptsGeneratorHallucinationPrompts:
    """Test hallucination check prompt generation."""
    
    def test_simple_hallucination_prompt(self):
        """Test simple style hallucination prompt."""
        colang_config = ColangConfig(
            output_rails=OutputRailConfig(self_check_hallucination=True)
        )
        generator = PromptsGenerator(colang_config, style="simple")
        
        prompts = generator.generate()
        
        hall_prompt = next(
            p for p in prompts["prompts"] if p["task"] == "self_check_hallucination"
        )
        content = hall_prompt["content"]
        
        # Should have the expected placeholders
        assert "{{ paragraph }}" in content or "{{ statement }}" in content or "context" in content.lower()
        
    def test_complex_hallucination_prompt(self):
        """Test complex style hallucination prompt."""
        colang_config = ColangConfig(
            output_rails=OutputRailConfig(self_check_hallucination=True)
        )
        generator = PromptsGenerator(colang_config, style="complex")
        
        prompts = generator.generate()
        
        hall_prompt = next(
            p for p in prompts["prompts"] if p["task"] == "self_check_hallucination"
        )
        content = hall_prompt["content"]
        
        assert len(content) > 100  # Complex prompts are longer


class TestPromptsGeneratorYamlOutput:
    """Test YAML output generation."""
    
    def test_generate_yaml_has_header(self):
        """Test that YAML output has auto-generation header."""
        colang_config = ColangConfig(
            input_rails=InputRailConfig(self_check_input=True)
        )
        generator = PromptsGenerator(colang_config)
        
        yaml_content = generator.generate_yaml()
        
        assert "Auto-generated by Neo Guardrail Hub" in yaml_content
        
    def test_generate_yaml_is_valid_yaml(self):
        """Test that output is valid YAML."""
        import yaml as pyyaml
        
        colang_config = ColangConfig(
            input_rails=InputRailConfig(self_check_input=True),
            output_rails=OutputRailConfig(self_check_output=True),
        )
        generator = PromptsGenerator(colang_config)
        
        yaml_content = generator.generate_yaml()
        
        # Should not raise an exception
        parsed = pyyaml.safe_load(yaml_content)
        assert parsed is not None
        assert "prompts" in parsed
        
    def test_yaml_prompts_structure(self):
        """Test that prompts have correct structure."""
        import yaml as pyyaml
        
        colang_config = ColangConfig(
            input_rails=InputRailConfig(self_check_input=True),
        )
        generator = PromptsGenerator(colang_config)
        
        yaml_content = generator.generate_yaml()
        parsed = pyyaml.safe_load(yaml_content)
        
        assert len(parsed["prompts"]) > 0
        for prompt in parsed["prompts"]:
            assert "task" in prompt
            assert "content" in prompt


class TestPromptsGeneratorWrite:
    """Test file writing functionality."""
    
    def test_write_creates_file(self, tmp_path):
        """Test that write creates the prompts file."""
        colang_config = ColangConfig(
            input_rails=InputRailConfig(self_check_input=True)
        )
        generator = PromptsGenerator(colang_config)
        
        output_file = tmp_path / "prompts.yml"
        generator.write(output_file)
        
        assert output_file.exists()
        
    def test_write_creates_parent_directories(self, tmp_path):
        """Test that write creates parent directories."""
        colang_config = ColangConfig(
            input_rails=InputRailConfig(self_check_input=True)
        )
        generator = PromptsGenerator(colang_config)
        
        output_file = tmp_path / "nested" / "deep" / "prompts.yml"
        generator.write(output_file)
        
        assert output_file.exists()
        assert (tmp_path / "nested" / "deep").is_dir()
        
    def test_write_content_is_valid(self, tmp_path):
        """Test that written content is valid YAML."""
        import yaml as pyyaml
        
        colang_config = ColangConfig(
            input_rails=InputRailConfig(self_check_input=True),
            output_rails=OutputRailConfig(self_check_output=True),
        )
        generator = PromptsGenerator(colang_config)
        
        output_file = tmp_path / "prompts.yml"
        generator.write(output_file)
        
        content = output_file.read_text()
        parsed = pyyaml.safe_load(content)
        
        assert parsed is not None
        assert "prompts" in parsed


class TestPromptsGeneratorMultiplePrompts:
    """Test generation of multiple prompts."""
    
    def test_generate_all_prompts(self):
        """Test generating all possible prompts."""
        colang_config = ColangConfig(
            input_rails=InputRailConfig(self_check_input=True),
            output_rails=OutputRailConfig(
                self_check_output=True,
                self_check_facts=True,
                self_check_hallucination=True,
            ),
        )
        generator = PromptsGenerator(colang_config)
        
        prompts = generator.generate()
        
        prompt_tasks = [p.get("task") for p in prompts["prompts"]]
        assert "self_check_input" in prompt_tasks
        assert "self_check_output" in prompt_tasks
        assert "self_check_facts" in prompt_tasks
        assert "self_check_hallucination" in prompt_tasks
        
    def test_prompt_order_is_consistent(self):
        """Test that prompts are generated in consistent order."""
        colang_config = ColangConfig(
            input_rails=InputRailConfig(self_check_input=True),
            output_rails=OutputRailConfig(
                self_check_output=True,
                self_check_facts=True,
            ),
        )
        
        # Generate twice and compare
        generator1 = PromptsGenerator(colang_config)
        generator2 = PromptsGenerator(colang_config)
        
        prompts1 = generator1.generate()
        prompts2 = generator2.generate()
        
        tasks1 = [p["task"] for p in prompts1["prompts"]]
        tasks2 = [p["task"] for p in prompts2["prompts"]]
        
        assert tasks1 == tasks2


class TestPromptsGeneratorNoPrompts:
    """Test behavior when no prompts are needed."""
    
    def test_generate_no_prompts_needed(self):
        """Test when no self-check features are enabled."""
        colang_config = ColangConfig(
            input_rails=InputRailConfig(self_check_input=False),
            output_rails=OutputRailConfig(
                self_check_output=False,
                self_check_facts=False,
            ),
        )
        generator = PromptsGenerator(colang_config)
        
        prompts = generator.generate()
        
        assert prompts["prompts"] == []
        
    def test_generate_with_only_input_length(self):
        """Test when only input length check is enabled (no prompt needed)."""
        colang_config = ColangConfig(
            input_rails=InputRailConfig(
                self_check_input=False,
                input_length_check=True,
            )
        )
        generator = PromptsGenerator(colang_config)
        
        prompts = generator.generate()
        
        # Input length check doesn't need a prompt
        assert not any(p["task"] == "input_length" for p in prompts["prompts"])


class TestPromptStyleType:
    """Test PromptStyle type."""
    
    def test_prompt_style_values(self):
        """Test that PromptStyle accepts expected values."""
        # These should work without error
        colang_config = ColangConfig()
        
        simple_gen = PromptsGenerator(colang_config, style="simple")
        assert simple_gen.style == "simple"
        
        complex_gen = PromptsGenerator(colang_config, style="complex")
        assert complex_gen.style == "complex"
