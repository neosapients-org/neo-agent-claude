"""
Example 07: NeMo Colang Generation from YAML Configuration

This example demonstrates how to use the Neo Guardrail Hub's NeMo provider
to generate NeMo Guardrails configuration files from the existing YAML
configuration (default.yaml or agent-specific configs).

Sprint 3A.1 Feature: Base Templates & Colang Generator

The generation follows the architecture hierarchy:
1. configs/default.yaml - Base configuration (used if no agent specified)
2. configs/agents/{agent_id}.yaml - Agent-specific overrides

Generated files:
- config.yml - NeMo main configuration
- rails.co - Colang 2.x flow definitions
- prompts.yml - LLM prompts for self-checking

Features demonstrated:
1. Generate NeMo config from default.yaml
2. Generate for specific agent (with config override)
3. Using the topic library for pre-built topics
4. Custom output directory specification
"""

from pathlib import Path
from tempfile import TemporaryDirectory
import shutil

# Import NeMo provider components
from neo_guardrail_hub.providers.nemo import (
    # High-level generators (main entry points)
    NeMoConfigGenerator,
    generate_nemo_from_yaml,
    # Config mapping
    ConfigToNeMoMapper,
    map_yaml_to_colang,
    # Low-level generators
    ColangGenerator,
    generate_colang_from_dict,
    # Configuration models
    ColangConfig,
    InputRailConfig,
    OutputRailConfig,
    TopicalRailConfig,
    # Topic library
    load_topic_library,
    get_topic_by_name,
    get_sensitive_topics,
    get_harmful_topics,
    get_all_disallowed_topics,
)


# Get the configs directory (relative to this example)
CONFIGS_PATH = Path(__file__).parent.parent / "configs"


def example_generate_from_default_yaml():
    """Example 1: Generate NeMo config from default.yaml."""
    print("\n" + "=" * 60)
    print("Example 1: Generate from default.yaml")
    print("=" * 60)
    
    print(f"\nUsing config from: {CONFIGS_PATH / 'default.yaml'}")
    
    # Create a temporary directory for output
    with TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "nemo_config"
        
        # Generate NeMo configuration from default.yaml
        result_path = generate_nemo_from_yaml(
            config_path=CONFIGS_PATH,  # Directory containing default.yaml
            agent_id=None,              # Use default config
            output_path=output_path,    # Where to save generated files
        )
        
        print(f"\nGenerated files in: {result_path}")
        print("\nGenerated files:")
        for file in result_path.iterdir():
            print(f"  • {file.name} ({file.stat().st_size} bytes)")
            
        # Show config.yml content
        config_yml = result_path / "config.yml"
        if config_yml.exists():
            print("\n--- config.yml ---")
            print(config_yml.read_text())
            
        # Show rails.co content
        rails_co = result_path / "rails.co"
        if rails_co.exists():
            print("\n--- rails.co (first 40 lines) ---")
            lines = rails_co.read_text().split("\n")[:40]
            print("\n".join(lines))
            if len(rails_co.read_text().split("\n")) > 40:
                print("...")


def example_generate_with_agent_override():
    """Example 2: Generate with agent-specific configuration override."""
    print("\n" + "=" * 60)
    print("Example 2: Generate with Configuration Override")
    print("=" * 60)
    
    # Create generator
    generator = NeMoConfigGenerator(config_path=CONFIGS_PATH)
    
    # Override default config with custom settings
    config_override = {
        "guardrails": {
            "input": {
                "enabled": True,
                "checks": [
                    {
                        "type": "prompt_injection",
                        "enabled": True,
                    },
                    {
                        "type": "input_length",
                        "enabled": True,
                        "max_length": 5000,
                    },
                ]
            },
            "output": {
                "enabled": True,
                "checks": [
                    {
                        "type": "pii_redaction",
                        "enabled": True,
                    },
                    {
                        "type": "factual_consistency",
                        "enabled": True,
                    },
                    {
                        "type": "ban_topics",
                        "enabled": True,
                        "topics": ["politics", "religion"],
                    },
                ]
            },
            "context": {
                "enabled": True,
                "checks": [
                    {
                        "type": "dialog_flow",
                        "enabled": True,
                        "config": {
                            "allowed_topics": ["customer_support", "products"],
                            "disallowed_topics": ["violence", "illegal_activities"],
                        }
                    }
                ]
            }
        }
    }
    
    with TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "custom_agent"
        
        # Generate with overrides
        result_path = generator.generate(
            agent_id="custom_agent",
            output_path=output_path,
            config_override=config_override,
        )
        
        print(f"\nGenerated files in: {result_path}")
        
        # Show rails.co content
        rails_co = result_path / "rails.co"
        if rails_co.exists():
            print("\n--- rails.co (with topic blocking) ---")
            print(rails_co.read_text())


def example_map_yaml_to_colang():
    """Example 3: Map YAML config to Colang config (intermediate step)."""
    print("\n" + "=" * 60)
    print("Example 3: Map YAML to ColangConfig")
    print("=" * 60)
    
    # Create mapper
    mapper = ConfigToNeMoMapper(config_path=CONFIGS_PATH)
    
    # Map default.yaml to ColangConfig
    colang_config = mapper.map_to_colang()
    
    print("\nMapped configuration from default.yaml:")
    print(f"  Input Rails: {colang_config.input_rails}")
    print(f"  Output Rails: {colang_config.output_rails}")
    print(f"  Topical Rails: {colang_config.topical_rails}")
    
    # You can also get the full NeMo config
    nemo_config = mapper.map_to_nemo_config()
    print(f"\nNeMo Config:")
    print(f"  Name: {nemo_config.name}")
    print(f"  Enable Input Rails: {nemo_config.enable_input_rails}")
    print(f"  Enable Output Rails: {nemo_config.enable_output_rails}")
    print(f"  Enable Dialog Rails: {nemo_config.enable_dialog_rails}")


def example_generate_colang_only():
    """Example 4: Generate only Colang content without other files."""
    print("\n" + "=" * 60)
    print("Example 4: Generate Colang Only")
    print("=" * 60)
    
    generator = NeMoConfigGenerator(config_path=CONFIGS_PATH)
    
    # Generate Colang content (returns string, optionally saves to file)
    colang_content = generator.generate_colang_only()
    
    print("\nGenerated Colang from default.yaml:")
    print("-" * 40)
    print(colang_content)


def example_topic_library():
    """Example 5: Using the pre-built topic library."""
    print("\n" + "=" * 60)
    print("Example 5: Using the Topic Library")
    print("=" * 60)
    
    # Load the topic library
    library = load_topic_library()
    
    print(f"\nTopic library version: {library.version}")
    print(f"Total topics: {len(library.topics)}")
    
    # List topics by category
    print("\n--- Sensitive Topics ---")
    for name in get_sensitive_topics():
        print(f"  • {name}")
        
    print("\n--- Harmful Topics ---")
    for name in get_harmful_topics():
        print(f"  • {name}")
    
    # Get details for a specific topic
    topic = get_topic_by_name("politics")
    if topic:
        print(f"\n--- Topic Details: {topic.name} ---")
        print(f"Category: {topic.category}")
        print(f"Description: {topic.description}")
        print(f"Example utterances:")
        for ex in topic.user_examples[:3]:
            print(f"  - \"{ex}\"")
        print(f"Keywords: {', '.join(topic.keywords[:5])}")


def example_save_to_custom_path():
    """Example 6: Save generated files to a custom path."""
    print("\n" + "=" * 60)
    print("Example 6: Save to Custom Path")
    print("=" * 60)
    
    with TemporaryDirectory() as tmpdir:
        # Create a custom output structure
        custom_path = Path(tmpdir) / "my_agent" / "nemo_config"
        
        # Generate with custom path
        result_path = generate_nemo_from_yaml(
            config_path=CONFIGS_PATH,
            output_path=custom_path,
        )
        
        print(f"\nGenerated to custom path: {result_path}")
        print("\nDirectory structure:")
        for item in result_path.iterdir():
            print(f"  {result_path.name}/{item.name}")


def example_generate_for_agent():
    """Example 7: Generate NeMo config for a specific agent (wealth_advisor)."""
    print("\n" + "=" * 60)
    print("Example 7: Generate for Specific Agent (wealth_advisor)")
    print("=" * 60)
    
    # Check if wealth_advisor config exists
    agent_config_path = CONFIGS_PATH / "agents" / "wealth_advisor.yaml"
    print(f"\nAgent config: {agent_config_path}")
    print(f"Config exists: {agent_config_path.exists()}")
    
    with TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "agents"
        
        # Generate using agent-specific config
        result_path = generate_nemo_from_yaml(
            config_path=CONFIGS_PATH,
            agent_id="wealth_advisor",  # Uses configs/agents/wealth_advisor.yaml
            output_path=output_path,
        )
        
        print(f"\nGenerated files in: {result_path}")
        
        # Show generated rails.co - should have finance-specific topics
        rails_co = result_path / "rails.co"
        if rails_co.exists():
            print("\n--- rails.co (wealth_advisor with financial topics) ---")
            content = rails_co.read_text()
            lines = content.split("\n")[:60]
            print("\n".join(lines))
            if len(content.split("\n")) > 60:
                print("\n... (truncated)")
                
        # Show config.yml with LLM settings from agent config
        config_yml = result_path / "config.yml"
        if config_yml.exists():
            print("\n--- config.yml ---")
            print(config_yml.read_text())


def example_low_level_generation():
    """Example 8: Low-level generation using ColangGenerator directly."""
    print("\n" + "=" * 60)
    print("Example 8: Low-level ColangGenerator (for advanced use)")
    print("=" * 60)
    
    # For advanced users who want direct control
    config = ColangConfig(
        input_rails=InputRailConfig(
            self_check_input=True,
            max_length=10000,
        ),
        output_rails=OutputRailConfig(
            self_check_output=True,
            self_check_facts=True,
        ),
        topical_rails=TopicalRailConfig(
            disallowed_topics=get_all_disallowed_topics(),  # Use library topics
        ),
        refusal_message="I cannot help with that request.",
    )
    
    generator = ColangGenerator(config)
    colang_content = generator.generate()
    
    print("\nGenerated Colang (with library topics):")
    print(f"  Total length: {len(colang_content)} characters")
    print(f"  Flows defined: {colang_content.count('define flow')}")
    
    # Show first few lines
    lines = colang_content.split("\n")[:20]
    print("\nFirst 20 lines:")
    print("-" * 40)
    print("\n".join(lines))
    print("...")


if __name__ == "__main__":
    print("Neo Guardrail Hub - NeMo Configuration Generation Examples")
    print("=" * 60)
    print(f"Config path: {CONFIGS_PATH}")
    
    # Run all examples
    example_generate_from_default_yaml()
    example_generate_with_agent_override()
    example_map_yaml_to_colang()
    example_generate_colang_only()
    example_topic_library()
    example_save_to_custom_path()
    example_generate_for_agent()
    example_low_level_generation()
    
    print("\n" + "=" * 60)
    print("All examples completed!")
    print("=" * 60)
