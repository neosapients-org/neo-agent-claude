"""
Example 16: Pre-generation API

This example demonstrates how to pre-generate NeMo configuration files
before using Neo Guardrail Hub. Pre-generation is essential for:

1. Faster startup - configs are already generated
2. Validation - catch config errors early  
3. Review - inspect generated files before deployment

Usage:
    python examples/16_pregeneration_api.py
    
Requirements:
    - pip install neo-guardrail-hub[nemo]
    - OPENAI_API_KEY environment variable (for NeMo rails)
"""

import asyncio
from pathlib import Path
from dotenv import load_dotenv

from neo_guardrail_hub import (
    generate_configs,
    generate_configs_sync,
    initialize,
    initialize_sync,
    PreGenerationResult,
)
import os

# Load environment variables from .env file
env_path = Path(__file__).parent.parent / '.env'
load_dotenv(dotenv_path=env_path)


def example_sync_generation():
    """
    Example 1: Synchronous config generation
    
    Use this pattern when you want to generate configs at startup
    before any async code runs.
    """
    print("\n" + "=" * 60)
    print("Example 1: Synchronous Config Generation")
    print("=" * 60)
    
    # Generate configs from default.yaml
    result = generate_configs_sync(
        config_path="./configs",
        output_path="./generated_configs",
        force=True,  # Overwrite existing files
    )
    
    if result.success:
        print(f"✓ Configs generated at: {result.output_path}")
        print(f"  Files: {len(result.generated_files)}")
        
        # Show config summary
        print("\n  Configuration Summary:")
        for key, value in result.config_summary.items():
            print(f"    {key}: {value}")
    else:
        print(f"✗ Generation failed:")
        for error in result.errors:
            print(f"    Error: {error}")
    
    if result.warnings:
        print("\n  Warnings:")
        for warning in result.warnings:
            print(f"    ⚠ {warning}")
    
    return result


def example_agent_specific():
    """
    Example 2: Generate configs for a specific agent
    
    Each agent can have its own configuration that inherits from
    default.yaml but overrides specific settings.
    """
    print("\n" + "=" * 60)
    print("Example 2: Agent-Specific Config Generation")
    print("=" * 60)
    
    # Generate for wealth_advisor agent
    result = generate_configs_sync(
        config_path="./configs",
        agent_id="wealth_advisor",
        output_path="./generated_configs/wealth_advisor",
        force=True,
    )
    
    if result.success:
        print(f"✓ Generated configs for agent: {result.agent_id}")
        print(f"  Output: {result.output_path}")
    else:
        print(f"✗ Failed to generate for agent")
        for error in result.errors:
            print(f"    {error}")
    
    return result


async def example_async_initialize():
    """
    Example 3: Async initialization with Rails preloading
    
    Use initialize() for async applications to both generate configs
    AND preload NeMo Rails into cache for faster first use.
    """
    print("\n" + "=" * 60)
    print("Example 3: Async Initialization with Preloading")
    print("=" * 60)
    
    result = await initialize(
        config_path="./configs",
        output_path="./generated_configs/preloaded",
        preload_rails=True,  # Pre-load Rails into cache
        force=True,
    )
    
    if result.success:
        print(f"✓ Initialization complete!")
        print(f"  Output: {result.output_path}")
        
        if result.config_summary.get("rails_cached"):
            print("  Rails pre-loaded into cache ✓")
    else:
        print(f"✗ Initialization failed")
        for error in result.errors:
            print(f"    {error}")
    
    return result


def example_validation_only():
    """
    Example 4: Validate configs without regenerating
    
    When configs already exist, you can check if they're valid
    without overwriting them.
    """
    print("\n" + "=" * 60)
    print("Example 4: Validation Without Regeneration")
    print("=" * 60)
    
    # First, ensure configs exist
    generate_configs_sync(
        config_path="./configs",
        output_path="./generated_configs/for_validation",
    )
    
    # Now try to generate again without force
    result = generate_configs_sync(
        config_path="./configs",
        output_path="./generated_configs/for_validation",
        force=False,
        validate=True,
    )
    
    if result.success and any("already exist" in w for w in result.warnings):
        print("✓ Existing configs are valid, not regenerated")
    
    return result


def example_custom_workflow():
    """
    Example 5: Custom workflow with result inspection
    
    The PreGenerationResult provides detailed information about
    the generation process that you can use for custom workflows.
    """
    print("\n" + "=" * 60)
    print("Example 5: Custom Workflow with Result Inspection")
    print("=" * 60)
    
    result = generate_configs_sync(
        config_path="./configs",
        output_path="./generated_configs/custom",
        force=True,
    )
    
    # Use result in boolean context
    if result:  # Same as result.success
        print("✓ Generation successful!")
        
        # Inspect generated files
        print("\n  Generated files:")
        for file_path in result.generated_files:
            size = file_path.stat().st_size
            print(f"    - {file_path.name} ({size} bytes)")
        
        # Check what was enabled
        summary = result.config_summary
        print("\n  Enabled features:")
        if summary.get("nemo_enabled"):
            print(f"    - NeMo Guardrails (model: {summary.get('nemo_llm')})")
        if summary.get("input_rails"):
            print("    - Input Rails")
        if summary.get("output_rails"):
            print("    - Output Rails")
        if summary.get("dialog_rails"):
            print("    - Dialog Rails (topical control)")
        if summary.get("llm_guard_input"):
            print("    - LLM Guard Input Checks")
        if summary.get("llm_guard_output"):
            print("    - LLM Guard Output Checks")
    
    return result


async def main():
    """Run all examples."""
    print("\n" + "=" * 60)
    print("Neo Guardrail Hub - Pre-generation API Examples")
    print("=" * 60)
    
    # Example 1: Basic sync generation
    example_sync_generation()
    
    # Example 2: Agent-specific generation
    example_agent_specific()
    
    # Example 3: Async with preloading (may fail without NeMo installed)
    try:
        await example_async_initialize()
    except ImportError as e:
        print(f"\n  Skipped: NeMo Guardrails not installed ({e})")
    except Exception as e:
        print(f"\n  Skipped: {e}")
    
    # Example 4: Validation
    example_validation_only()
    
    # Example 5: Custom workflow
    example_custom_workflow()
    
    print("\n" + "=" * 60)
    print("All examples completed!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
