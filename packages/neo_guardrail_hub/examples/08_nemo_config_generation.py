"""
Example 08: NeMo config.yml Generation

This example demonstrates how to use the ConfigGenerator to create
NeMo Guardrails config.yml files from Neo Guardrail Hub configuration.

Sprint 3A.2 Feature: Config Generator

The ConfigGenerator handles:
- Model configuration (LLM settings)
- Rails configuration (input, output, dialog flows)
- Preset merging for reusable configurations
- Streaming configuration

Two approaches are demonstrated:
1. HIGH-LEVEL: Using NeMoConfigGenerator with default.yaml (recommended)
2. LOW-LEVEL: Using ConfigGenerator directly with programmatic config

Features demonstrated:
1. Generate config from default.yaml
2. Generate config for specific agent
3. Generate with configuration overrides
4. Custom LLM configuration (programmatic)
5. Input/Output rails configuration
6. Dialog rails for topic blocking
7. Full configuration generation

Output Directory:
- By default, files are saved to ./neo_configs/
- You can customize this with output_base_path or output_path parameters
"""

from pathlib import Path

# High-level imports (uses default.yaml)
from neo_guardrail_hub.providers.nemo import (
    NeMoConfigGenerator,
    generate_nemo_from_yaml,
)

# Low-level imports (programmatic configuration)
from neo_guardrail_hub.providers.nemo.generators import (
    ConfigGenerator,
)
from neo_guardrail_hub.providers.nemo.generators.models import (
    NeMoGuardrailsConfig,
    ColangConfig,
    LLMConfig,
    InputRailConfig,
    OutputRailConfig,
    TopicalRailConfig,
)


# Get the configs directory (relative to this example)
CONFIGS_PATH = Path(__file__).parent.parent / "configs"


# =============================================================================
# HIGH-LEVEL EXAMPLES: Using default.yaml
# =============================================================================

def example_config_from_default_yaml():
    """Example 1: Generate config.yml from default.yaml."""
    print("\n" + "=" * 60)
    print("Example 1: Generate config.yml from default.yaml")
    print("=" * 60)
    
    print(f"\nUsing config from: {CONFIGS_PATH / 'default.yaml'}")
    
    # Create generator pointing to configs directory
    # Files will be saved to ./neo_configs/ by default
    generator = NeMoConfigGenerator(config_path=CONFIGS_PATH)
    
    # Generate config - automatically saves to neo_configs/config.yml
    config_dict = generator.generate_config_only()
    
    # Show where the file was saved
    output_path = Path.cwd() / "neo_configs" / "config.yml"
    print(f"\nConfig saved to: {output_path}")
    print(f"File exists: {output_path.exists()}")
    
    print("\nGenerated config (from nemo.rails section):")
    print(f"  Models: {config_dict.get('models', [])}")
    
    if "rails" in config_dict:
        print(f"  Rails configured:")
        for rail_type in ["input", "output", "dialog"]:
            if rail_type in config_dict["rails"]:
                flows = config_dict["rails"][rail_type].get("flows", [])
                print(f"    - {rail_type}: {flows}")
    
    # Generate as YAML string
    yaml_content = generator.generate_config_yaml()
    print("\nGenerated config.yml:")
    print("-" * 40)
    print(yaml_content)


def example_config_for_agent():
    """Example 2: Generate config.yml for specific agent."""
    print("\n" + "=" * 60)
    print("Example 2: Generate config.yml for Specific Agent")
    print("=" * 60)
    
    generator = NeMoConfigGenerator(config_path=CONFIGS_PATH)
    
    # Check if wealth_advisor agent exists
    agent_path = CONFIGS_PATH / "agents" / "wealth_advisor.yaml"
    if agent_path.exists():
        print(f"\nUsing agent config: {agent_path}")
        
        # Generate for specific agent
        config_dict = generator.generate_config_only(agent_id="wealth_advisor")
        
        print("\nWealth Advisor config:")
        print(f"  Models: {config_dict.get('models', [])}")
        if "rails" in config_dict:
            for rail_type, rail_config in config_dict["rails"].items():
                print(f"  {rail_type} flows: {rail_config.get('flows', [])}")
    else:
        print(f"\nAgent 'wealth_advisor' not found at {agent_path}")
        print("Using default.yaml config instead...")
        config_dict = generator.generate_config_only()
        print(f"  Models: {config_dict.get('models', [])}")


def example_config_with_override():
    """Example 3: Generate config with runtime overrides."""
    print("\n" + "=" * 60)
    print("Example 3: Generate config.yml with Override")
    print("=" * 60)
    
    generator = NeMoConfigGenerator(config_path=CONFIGS_PATH)
    
    # Override the nemo section from default.yaml
    config_override = {
        "nemo": {
            "llm": {
                "engine": "azure",
                "model": "gpt-4-deployment",
                "temperature": 0.2,
            },
            "streaming": True,
            "rails": {
                "input": {
                    "enabled": True,
                    "self_check_input": True,
                    "max_length": 5000,
                },
                "output": {
                    "enabled": True,
                    "self_check_output": True,
                    "self_check_facts": True,
                },
            }
        }
    }
    
    config_dict = generator.generate_config_only(config_override=config_override)
    
    print("\nConfig with overrides:")
    model = config_dict["models"][0]
    print(f"  Engine: {model.get('engine')}")
    print(f"  Model: {model.get('model')}")
    print(f"  Streaming: {config_dict.get('streaming', False)}")
    
    if "rails" in config_dict:
        print(f"  Input flows: {config_dict['rails'].get('input', {}).get('flows', [])}")
        print(f"  Output flows: {config_dict['rails'].get('output', {}).get('flows', [])}")


def example_save_config_to_file():
    """Example 4: Show persistent file output."""
    print("\n" + "=" * 60)
    print("Example 4: Persistent File Output")
    print("=" * 60)
    
    generator = NeMoConfigGenerator(config_path=CONFIGS_PATH)
    
    # Files are automatically saved to neo_configs/ folder
    # No need for temp directories - files persist!
    output_dir = Path.cwd() / "neo_configs"
    
    print(f"\nDefault output directory: {output_dir}")
    print("\nGenerated files will be saved to:")
    print(f"  - {output_dir / 'config.yml'}")
    print(f"  - {output_dir / 'prompts.yml'}")
    print(f"  - {output_dir / 'rails.co'}")
    
    # Generate all files at once
    result_path = generator.generate()
    
    print(f"\nFiles generated at: {result_path}")
    print("\nGenerated files:")
    for file in result_path.iterdir():
        print(f"  - {file.name} ({file.stat().st_size} bytes)")


# =============================================================================
# LOW-LEVEL EXAMPLES: Programmatic Configuration
# =============================================================================

def example_programmatic_basic_config():
    """Example 5: Generate basic config programmatically."""
    print("\n" + "=" * 60)
    print("Example 5: Programmatic Basic Config")
    print("=" * 60)
    
    # Create a minimal NeMo configuration
    nemo_config = NeMoGuardrailsConfig()
    
    # Generate config
    generator = ConfigGenerator(nemo_config)
    config = generator.generate()
    
    print("\nGenerated config (dictionary):")
    print(f"  Models: {config['models']}")
    
    # Generate as YAML
    yaml_content = generator.generate_yaml()
    print("\nGenerated YAML:")
    print(yaml_content)


def example_programmatic_custom_llm():
    """Example 6: Programmatic config with custom LLM."""
    print("\n" + "=" * 60)
    print("Example 6: Programmatic Custom LLM")
    print("=" * 60)
    
    # Configure a custom LLM
    llm_config = LLMConfig(
        engine="openai",
        model="gpt-4-turbo",
        temperature=0.3,
        max_tokens=2048,
    )
    
    nemo_config = NeMoGuardrailsConfig(
        llm=llm_config,
        streaming=True,
    )
    
    generator = ConfigGenerator(nemo_config)
    config = generator.generate()
    
    print("\nLLM Configuration:")
    model = config["models"][0]
    print(f"  Engine: {model['engine']}")
    print(f"  Model: {model['model']}")
    print(f"  Temperature: {model.get('temperature', 'default')}")
    print(f"  Max Tokens: {model.get('max_tokens', 'default')}")
    print(f"\nStreaming: {config.get('streaming', False)}")


def example_programmatic_full_config():
    """Example 7: Programmatic full configuration."""
    print("\n" + "=" * 60)
    print("Example 7: Programmatic Full Configuration")
    print("=" * 60)
    
    # Create a complete configuration
    llm_config = LLMConfig(
        engine="openai",
        model="gpt-4",
        temperature=0.2,
        max_tokens=1500,
    )
    
    input_rails = InputRailConfig(
        self_check_input=True,
        max_length=5000,
    )
    
    output_rails = OutputRailConfig(
        self_check_output=True,
        self_check_facts=True,
    )
    
    topical_rails = TopicalRailConfig(
        disallowed_topics=["competitor_info", "off_topic"],
    )
    
    colang = ColangConfig(
        input_rails=input_rails,
        output_rails=output_rails,
        topical_rails=topical_rails,
    )
    
    nemo_config = NeMoGuardrailsConfig(
        llm=llm_config,
        colang=colang,
        enable_input_rails=True,
        enable_output_rails=True,
        enable_dialog_rails=True,
        streaming=True,
    )
    
    generator = ConfigGenerator(nemo_config)
    
    print("\nFull NeMo config.yml:")
    print(generator.generate_yaml())


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("NeMo Config Generator Examples")
    print("=" * 60)
    print("\nThis example demonstrates two approaches:")
    print("1. HIGH-LEVEL: Using NeMoConfigGenerator with default.yaml")
    print("2. LOW-LEVEL: Using ConfigGenerator with programmatic config")
    
    # High-level examples (using default.yaml)
    print("\n" + "#" * 60)
    print("# HIGH-LEVEL EXAMPLES (using default.yaml)")
    print("#" * 60)
    
    example_config_from_default_yaml()
    example_config_for_agent()
    example_config_with_override()
    example_save_config_to_file()
    
    # Low-level examples (programmatic)
    print("\n" + "#" * 60)
    print("# LOW-LEVEL EXAMPLES (programmatic configuration)")
    print("#" * 60)
    
    example_programmatic_basic_config()
    example_programmatic_custom_llm()
    example_programmatic_full_config()
    
    print("\n" + "=" * 60)
    print("All examples completed!")
