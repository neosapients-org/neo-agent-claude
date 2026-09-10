"""
Example 09: NeMo prompts.yml Generation

This example demonstrates how to use the PromptsGenerator to create
NeMo Guardrails prompts.yml files for LLM self-checking tasks.

Sprint 3A.2 Feature: Prompts Generator

The PromptsGenerator handles:
- Standard prompts for all self-check tasks
- Simple vs Complex prompt styles
- Custom prompt support
- Template-based prompt generation

Prompt Styles:
- "simple": Concise prompts for faster/cheaper LLM evaluation
- "complex": Detailed prompts with comprehensive checking rules

Two approaches are demonstrated:
1. HIGH-LEVEL: Using NeMoConfigGenerator with default.yaml (recommended)
2. LOW-LEVEL: Using PromptsGenerator directly with programmatic config

Features demonstrated:
1. Generate prompts from default.yaml
2. Generate prompts for specific agent
3. Simple vs Complex prompt styles
4. Generate with configuration overrides
5. Fact-checking and hallucination prompts
6. Writing prompts to file

Output Directory:
- By default, files are saved to ./neo_configs/
- You can customize this with output_base_path or output_path parameters
"""

from pathlib import Path

# High-level imports (uses default.yaml)
from neo_guardrail_hub.providers.nemo import (
    NeMoConfigGenerator,
)

# Low-level imports (programmatic configuration)
from neo_guardrail_hub.providers.nemo.generators import (
    PromptsGenerator,
)
from neo_guardrail_hub.providers.nemo.generators.prompts_generator import PromptStyle
from neo_guardrail_hub.providers.nemo.generators.models import (
    ColangConfig,
    InputRailConfig,
    OutputRailConfig,
)


# Get the configs directory (relative to this example)
CONFIGS_PATH = Path(__file__).parent.parent / "configs"


# =============================================================================
# HIGH-LEVEL EXAMPLES: Using default.yaml
# =============================================================================

def example_prompts_from_default_yaml():
    """Example 1: Generate prompts.yml from default.yaml."""
    print("\n" + "=" * 60)
    print("Example 1: Generate prompts.yml from default.yaml")
    print("=" * 60)
    
    print(f"\nUsing config from: {CONFIGS_PATH / 'default.yaml'}")
    
    # Create generator pointing to configs directory
    # Files will be saved to ./neo_configs/ by default
    generator = NeMoConfigGenerator(config_path=CONFIGS_PATH)
    
    # Generate prompts - automatically saves to neo_configs/prompts.yml
    prompts_dict = generator.generate_prompts_only(prompt_style="simple")
    
    # Show where the file was saved
    output_path = Path.cwd() / "neo_configs" / "prompts.yml"
    print(f"\nPrompts saved to: {output_path}")
    print(f"File exists: {output_path.exists()}")
    
    print("\nGenerated prompts (from nemo.rails section):")
    print(f"  Total prompts: {len(prompts_dict.get('prompts', []))}")
    
    for prompt in prompts_dict.get("prompts", []):
        print(f"\n  Task: {prompt['task']}")
        # Show first 80 chars of content
        content_preview = prompt["content"][:80].replace("\n", " ")
        print(f"  Preview: {content_preview}...")


def example_prompts_with_style():
    """Example 2: Generate prompts with different styles."""
    print("\n" + "=" * 60)
    print("Example 2: Simple vs Complex Prompt Styles")
    print("=" * 60)
    
    generator = NeMoConfigGenerator(config_path=CONFIGS_PATH)
    
    # Generate with simple style
    simple_prompts = generator.generate_prompts_only(prompt_style="simple")
    
    # Generate with complex style
    complex_prompts = generator.generate_prompts_only(prompt_style="complex")
    
    print("\nComparing styles for self_check_input prompt:")
    
    # Find self_check_input in both
    simple_input = next(
        (p for p in simple_prompts.get("prompts", []) if p["task"] == "self_check_input"),
        None
    )
    complex_input = next(
        (p for p in complex_prompts.get("prompts", []) if p["task"] == "self_check_input"),
        None
    )
    
    if simple_input and complex_input:
        print(f"\n  SIMPLE style length: {len(simple_input['content'])} chars")
        print(f"  COMPLEX style length: {len(complex_input['content'])} chars")
        print(f"\n  Ratio: Complex is {len(complex_input['content']) / len(simple_input['content']):.1f}x longer")
    else:
        print("\n  (self_check_input not enabled in default.yaml)")


def example_prompts_yaml_output():
    """Example 3: Generate prompts as YAML string."""
    print("\n" + "=" * 60)
    print("Example 3: Generate prompts.yml as YAML")
    print("=" * 60)
    
    generator = NeMoConfigGenerator(config_path=CONFIGS_PATH)
    
    # Generate as YAML string
    yaml_content = generator.generate_prompts_yaml(prompt_style="simple")
    
    print("\nGenerated prompts.yml:")
    print("-" * 40)
    # Show first 1000 chars
    if len(yaml_content) > 1000:
        print(yaml_content[:1000])
        print(f"\n... ({len(yaml_content) - 1000} more characters)")
    else:
        print(yaml_content)


def example_prompts_with_override():
    """Example 4: Generate prompts with configuration override."""
    print("\n" + "=" * 60)
    print("Example 4: Generate prompts.yml with Override")
    print("=" * 60)
    
    generator = NeMoConfigGenerator(config_path=CONFIGS_PATH)
    
    # Override to enable more checks
    config_override = {
        "nemo": {
            "rails": {
                "input": {
                    "enabled": True,
                    "self_check_input": True,
                },
                "output": {
                    "enabled": True,
                    "self_check_output": True,
                    "self_check_facts": True,
                    "self_check_hallucination": True,
                },
            }
        }
    }
    
    prompts_dict = generator.generate_prompts_only(
        config_override=config_override,
        prompt_style="complex"
    )
    
    print("\nGenerated prompts with override (all checks enabled):")
    for prompt in prompts_dict.get("prompts", []):
        print(f"  - {prompt['task']} ({len(prompt['content'])} chars)")


def example_save_prompts_to_file():
    """Example 5: Generate and save prompts.yml to file.
    
    By default, files are saved to ./neo_configs/prompts.yml
    You can also specify a custom output path.
    """
    print("\n" + "=" * 60)
    print("Example 5: Save prompts.yml to File")
    print("=" * 60)
    
    generator = NeMoConfigGenerator(config_path=CONFIGS_PATH)
    
    # Option 1: Use default path (./neo_configs/prompts.yml)
    # prompts_dict = generator.generate_prompts_only(prompt_style="simple")
    # This automatically saves to ./neo_configs/prompts.yml
    
    # Option 2: Custom output path
    custom_output = Path("./neo_configs/custom_prompts.yml")
    
    prompts_dict = generator.generate_prompts_only(
        output_path=custom_output,
        prompt_style="simple"
    )
    
    # Check default output location
    default_output = Path("./neo_configs/prompts.yml")
    print(f"\nDefault output location: {default_output}")
    print(f"Custom output location: {custom_output}")
    
    if custom_output.exists():
        print(f"\nCustom file created successfully!")
        print(f"File size: {custom_output.stat().st_size} bytes")
        
        # Show preview
        content = custom_output.read_text()
        lines = content.split("\n")
        print(f"Total lines: {len(lines)}")
        
        print("\nFile preview (first 20 lines):")
        print("-" * 40)
        print("\n".join(lines[:20]))
    else:
        print("(No prompts to write - check if self_check is enabled)")


# =============================================================================
# LOW-LEVEL EXAMPLES: Programmatic Configuration
# =============================================================================

def example_programmatic_basic_prompts():
    """Example 6: Generate basic prompts programmatically."""
    print("\n" + "=" * 60)
    print("Example 6: Programmatic Basic Prompts")
    print("=" * 60)
    
    # Configure with self_check_input enabled
    colang_config = ColangConfig(
        input_rails=InputRailConfig(self_check_input=True)
    )
    
    generator = PromptsGenerator(colang_config)
    prompts = generator.generate()
    
    print("\nGenerated prompts:")
    for prompt in prompts["prompts"]:
        print(f"  Task: {prompt['task']}")
        content_preview = prompt["content"][:80].replace("\n", " ")
        print(f"  Preview: {content_preview}...")


def example_programmatic_all_prompts():
    """Example 7: Generate all prompts programmatically."""
    print("\n" + "=" * 60)
    print("Example 7: Programmatic All Prompts")
    print("=" * 60)
    
    # Enable all self-check features
    colang_config = ColangConfig(
        input_rails=InputRailConfig(self_check_input=True),
        output_rails=OutputRailConfig(
            self_check_output=True,
            self_check_facts=True,
            self_check_hallucination=True,
        ),
    )
    
    # Show both styles
    for style in ["simple", "complex"]:
        generator = PromptsGenerator(colang_config, style=style)
        prompts = generator.generate()
        
        print(f"\n{style.upper()} style prompts:")
        for prompt in prompts["prompts"]:
            print(f"  - {prompt['task']}: {len(prompt['content'])} chars")


def example_programmatic_style_comparison():
    """Example 8: Compare simple vs complex styles."""
    print("\n" + "=" * 60)
    print("Example 8: Simple vs Complex Style Comparison")
    print("=" * 60)
    
    colang_config = ColangConfig(
        input_rails=InputRailConfig(self_check_input=True)
    )
    
    # Generate with simple style
    simple_gen = PromptsGenerator(colang_config, style="simple")
    simple_prompts = simple_gen.generate()
    
    # Generate with complex style
    complex_gen = PromptsGenerator(colang_config, style="complex")
    complex_prompts = complex_gen.generate()
    
    simple_input = next(p for p in simple_prompts["prompts"] if p["task"] == "self_check_input")
    complex_input = next(p for p in complex_prompts["prompts"] if p["task"] == "self_check_input")
    
    print("\nSIMPLE Style self_check_input:")
    print("-" * 40)
    print(simple_input["content"])
    
    print("\n\nCOMPLEX Style self_check_input:")
    print("-" * 40)
    print(complex_input["content"])


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("NeMo Prompts Generator Examples")
    print("=" * 60)
    print("\nThis example demonstrates two approaches:")
    print("1. HIGH-LEVEL: Using NeMoConfigGenerator with default.yaml")
    print("2. LOW-LEVEL: Using PromptsGenerator with programmatic config")
    print("\nPrompt styles available:")
    print("  - 'simple': Concise prompts for faster evaluation")
    print("  - 'complex': Detailed prompts for thorough checking")
    
    # High-level examples (using default.yaml)
    print("\n" + "#" * 60)
    print("# HIGH-LEVEL EXAMPLES (using default.yaml)")
    print("#" * 60)
    
    example_prompts_from_default_yaml()
    example_prompts_with_style()
    example_prompts_yaml_output()
    example_prompts_with_override()
    example_save_prompts_to_file()
    
    # Low-level examples (programmatic)
    print("\n" + "#" * 60)
    print("# LOW-LEVEL EXAMPLES (programmatic configuration)")
    print("#" * 60)
    
    example_programmatic_basic_prompts()
    example_programmatic_all_prompts()
    example_programmatic_style_comparison()
    
    print("\n" + "=" * 60)
    print("All examples completed!")
