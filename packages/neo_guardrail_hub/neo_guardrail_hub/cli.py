"""
Command Line Interface for Neo Guardrail Hub.

This module provides CLI commands for:
1. Initializing configuration files
2. Generating NeMo configs
3. Validating configurations
4. Managing cache

Usage:
    # Initialize a new project
    neo-guardrail init
    
    # Generate NeMo configs
    neo-guardrail generate
    neo-guardrail generate --agent wealth_advisor
    
    # Validate configurations
    neo-guardrail validate
    
    # Cache management
    neo-guardrail cache --clear
    neo-guardrail cache --stats
"""

import argparse
import asyncio
import sys
from pathlib import Path
from typing import Optional

# Check if we're running directly
def main() -> int:
    """Main entry point for CLI."""
    parser = argparse.ArgumentParser(
        prog="neo-guardrail",
        description="Neo Guardrail Hub - Universal AI Agent Security",
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # init command
    init_parser = subparsers.add_parser(
        "init",
        help="Initialize Neo Guardrail Hub in current directory",
    )
    init_parser.add_argument(
        "--path",
        type=str,
        default="./configs",
        help="Path for configuration directory (default: ./configs)",
    )
    init_parser.add_argument(
        "--preset",
        type=str,
        choices=["general", "financial", "healthcare", "customer_service"],
        default="general",
        help="Industry preset to use (default: general)",
    )
    init_parser.add_argument(
        "--agent",
        type=str,
        help="Create agent-specific config",
    )
    
    # generate command
    generate_parser = subparsers.add_parser(
        "generate",
        help="Generate NeMo configuration files",
    )
    generate_parser.add_argument(
        "--config",
        type=str,
        default="./configs",
        help="Path to configs directory (default: ./configs)",
    )
    generate_parser.add_argument(
        "--agent",
        type=str,
        help="Agent ID for agent-specific config",
    )
    generate_parser.add_argument(
        "--output",
        type=str,
        help="Output directory for generated files",
    )
    generate_parser.add_argument(
        "--force",
        action="store_true",
        help="Force regeneration even if files exist",
    )
    generate_parser.add_argument(
        "--all-agents",
        action="store_true",
        help="Generate configs for all agents in configs/agents/",
    )
    
    # validate command
    validate_parser = subparsers.add_parser(
        "validate",
        help="Validate configuration files",
    )
    validate_parser.add_argument(
        "--config",
        type=str,
        default="./configs",
        help="Path to configs directory (default: ./configs)",
    )
    validate_parser.add_argument(
        "--agent",
        type=str,
        help="Agent ID to validate",
    )
    
    # cache command
    cache_parser = subparsers.add_parser(
        "cache",
        help="Manage Rails cache",
    )
    cache_parser.add_argument(
        "--clear",
        action="store_true",
        help="Clear the Rails cache",
    )
    cache_parser.add_argument(
        "--stats",
        action="store_true",
        help="Show cache statistics",
    )
    
    # download-models command
    download_parser = subparsers.add_parser(
        "download-models",
        help="Download LLM Guard models for offline use",
    )
    download_parser.add_argument(
        "--config",
        type=str,
        default="./configs",
        help="Path to configs directory (default: ./configs)",
    )
    download_parser.add_argument(
        "--models-dir",
        type=str,
        default="./models/llm_guard",
        help="Directory to save models (default: ./models/llm_guard)",
    )
    download_parser.add_argument(
        "--scanners",
        type=str,
        nargs="+",
        help="Specific scanners to download (default: all enabled in config)",
    )
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        return 0
    
    try:
        if args.command == "init":
            return cmd_init(args)
        elif args.command == "generate":
            return cmd_generate(args)
        elif args.command == "validate":
            return cmd_validate(args)
        elif args.command == "cache":
            return cmd_cache(args)
        elif args.command == "download-models":
            return cmd_download_models(args)
        else:
            parser.print_help()
            return 1
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


def cmd_init(args) -> int:
    """Initialize Neo Guardrail Hub configuration."""
    config_path = Path(args.path)
    
    print(f"Initializing Neo Guardrail Hub at {config_path}")
    
    # Create configs directory
    config_path.mkdir(parents=True, exist_ok=True)
    
    # Create default.yaml from template
    default_yaml = _get_default_yaml(preset=args.preset)
    default_path = config_path / "default.yaml"
    
    if default_path.exists():
        response = input(f"{default_path} exists. Overwrite? [y/N] ")
        if response.lower() != "y":
            print("Skipping default.yaml")
        else:
            default_path.write_text(default_yaml)
            print(f"Created {default_path}")
    else:
        default_path.write_text(default_yaml)
        print(f"Created {default_path}")
    
    # Create agent-specific config if requested
    if args.agent:
        agents_path = config_path / "agents"
        agents_path.mkdir(exist_ok=True)
        
        agent_yaml = _get_agent_yaml(args.agent, preset=args.preset)
        agent_path = agents_path / f"{args.agent}.yaml"
        
        if agent_path.exists():
            response = input(f"{agent_path} exists. Overwrite? [y/N] ")
            if response.lower() != "y":
                print(f"Skipping {agent_path}")
            else:
                agent_path.write_text(agent_yaml)
                print(f"Created {agent_path}")
        else:
            agent_path.write_text(agent_yaml)
            print(f"Created {agent_path}")
    
    print("\n✓ Initialization complete!")
    print("\nNext steps:")
    print("  1. Edit configs/default.yaml to customize settings")
    print("  2. Run: neo-guardrail generate")
    print("  3. Use in your code:")
    print("     from neo_guardrail_hub import NeoGuardrailOrchestrator")
    print("     orchestrator = NeoGuardrailOrchestrator(config_path='./configs')")
    
    return 0


def cmd_generate(args) -> int:
    """Generate NeMo configuration files."""
    from neo_guardrail_hub.api import generate_configs
    
    config_path = Path(args.config)
    
    if args.all_agents:
        # Generate for all agents
        agents_dir = config_path / "agents"
        if not agents_dir.exists():
            print(f"No agents directory found at {agents_dir}")
            return 1
        
        agent_files = list(agents_dir.glob("*.yaml")) + list(agents_dir.glob("*.yml"))
        if not agent_files:
            print(f"No agent configs found in {agents_dir}")
            return 1
        
        success_count = 0
        for agent_file in agent_files:
            agent_id = agent_file.stem
            print(f"\nGenerating for agent: {agent_id}")
            
            result = generate_configs(
                config_path=config_path,
                agent_id=agent_id,
                force=args.force,
            )
            
            if result.success:
                print(f"  ✓ Generated at: {result.output_path}")
                success_count += 1
            else:
                print(f"  ✗ Failed: {result.errors}")
        
        print(f"\n{success_count}/{len(agent_files)} agents generated successfully")
        return 0 if success_count == len(agent_files) else 1
    
    else:
        # Generate for single config
        print(f"Generating NeMo configs from {config_path}")
        if args.agent:
            print(f"Agent: {args.agent}")
        
        result = generate_configs(
            config_path=config_path,
            agent_id=args.agent,
            output_path=args.output,
            force=args.force,
        )
        
        if result.success:
            print(f"\n✓ Configuration generated successfully!")
            print(f"  Output: {result.output_path}")
            print(f"  Files: {len(result.generated_files)}")
            
            if result.warnings:
                print("\nWarnings:")
                for warning in result.warnings:
                    print(f"  ⚠ {warning}")
            
            print("\nConfiguration summary:")
            for key, value in result.config_summary.items():
                print(f"  {key}: {value}")
            
            return 0
        else:
            print("\n✗ Generation failed!")
            for error in result.errors:
                print(f"  Error: {error}")
            return 1


def cmd_validate(args) -> int:
    """Validate configuration files."""
    from neo_guardrail_hub.core.config import ConfigLoader
    
    config_path = Path(args.config)
    
    print(f"Validating configs at {config_path}")
    
    try:
        loader = ConfigLoader(config_path)
        config = loader.load_config(agent_id=args.agent)
        
        # Validation checks
        errors = []
        warnings = []
        
        # Check version
        if "version" not in config:
            warnings.append("Missing 'version' field")
        
        # Check NeMo config
        nemo_config = config.get("nemo", {})
        if nemo_config.get("enabled", False):
            llm_config = nemo_config.get("llm", {})
            if not llm_config.get("model"):
                warnings.append("NeMo enabled but no LLM model specified")
        
        # Check guardrails config
        guardrails_config = config.get("guardrails", {})
        for layer in ["input", "output"]:
            layer_config = guardrails_config.get(layer, {})
            if layer_config.get("enabled", False):
                checks = layer_config.get("checks", [])
                if not checks:
                    warnings.append(f"'{layer}' layer enabled but no checks defined")
        
        if errors:
            print("\n✗ Validation failed!")
            for error in errors:
                print(f"  Error: {error}")
            return 1
        
        print("\n✓ Configuration is valid!")
        
        if warnings:
            print("\nWarnings:")
            for warning in warnings:
                print(f"  ⚠ {warning}")
        
        return 0
        
    except Exception as e:
        print(f"\n✗ Validation failed: {e}")
        return 1


def cmd_cache(args) -> int:
    """Manage Rails cache."""
    from neo_guardrail_hub.api import get_rails_cache, clear_rails_cache
    
    if args.clear:
        clear_rails_cache()
        print("✓ Cache cleared")
        return 0
    
    if args.stats:
        cache = get_rails_cache()
        stats = cache.stats
        
        print("Rails Cache Statistics:")
        print(f"  Size: {stats['size']} / {stats['max_size']}")
        print(f"  Hits: {stats['hits']}")
        print(f"  Misses: {stats['misses']}")
        print(f"  Hit Rate: {stats['hit_rate']:.2%}")
        print(f"  TTL: {stats['ttl_seconds'] or 'None'}")
        
        if cache.size > 0:
            print("\nCached entries:")
            for key in cache.list_keys():
                print(f"  - {key}")
        
        return 0
    
    # Default: show stats
    return cmd_cache(argparse.Namespace(clear=False, stats=True))


def cmd_download_models(args) -> int:
    """Download LLM Guard models for offline use."""
    from neo_guardrail_hub.utils.model_downloader import LLMGuardModelDownloader
    
    print("=" * 70)
    print("LLM Guard Model Downloader")
    print("=" * 70)
    print()
    
    # Create downloader
    downloader = LLMGuardModelDownloader(
        config_path=args.config,
        models_dir=args.models_dir
    )
    
    # Download specific scanners or all enabled
    if args.scanners:
        print(f"Downloading models for specific scanners: {', '.join(args.scanners)}")
        results = downloader.download_specific_models(args.scanners)
    else:
        print("Downloading models for all enabled scanners from config...")
        results = downloader.download_all_enabled()
    
    # Return success if any models were downloaded
    if results.get("success", 0) > 0:
        print("\n✓ Download complete!")
        print(f"\nTo use local models, set environment variable:")
        print(f"  export NEO_LLM_GUARD_MODELS_DIR={Path(args.models_dir).absolute()}")
        return 0
    else:
        print("\n✗ No models were downloaded successfully")
        return 1


def _get_default_yaml(preset: str = "general") -> str:
    """Get default.yaml template."""
    presets = {
        "general": {
            "allowed_topics": ["greeting", "help", "general_questions"],
            "disallowed_topics": ["violence", "illegal_activity", "hate_speech"],
        },
        "financial": {
            "allowed_topics": ["portfolio", "account_balance", "fee_structure", "market_info"],
            "disallowed_topics": ["stock_tips", "investment_advice", "guaranteed_returns", "competitor_products"],
        },
        "healthcare": {
            "allowed_topics": ["appointment", "general_health", "medications", "symptoms"],
            "disallowed_topics": ["diagnosis", "prescriptions", "medical_advice", "treatment_plans"],
        },
        "customer_service": {
            "allowed_topics": ["order_status", "returns", "product_info", "account_help"],
            "disallowed_topics": ["competitor_products", "internal_processes", "employee_info"],
        },
    }
    
    preset_config = presets.get(preset, presets["general"])
    
    return f'''# Neo Guardrail Hub Configuration
# Generated by: neo-guardrail init
# Preset: {preset}

version: "1.0"
enabled: true

# Execution settings
execution:
  input:
    mode: parallel
    timeout_ms: 120000
  context:
    mode: sequential
    timeout_ms: 120000
  output:
    mode: parallel
    timeout_ms: 120000

# Default behavior
defaults:
  on_fail: block
  log_level: INFO

# NeMo Guardrails configuration
nemo:
  enabled: true
  llm:
    engine: openai
    model: gpt-4o-mini
    temperature: 0.0
  output_path: null
  preset: {preset}
  streaming: false
  
  rails:
    input:
      enabled: true
      self_check_input: true
      
    output:
      enabled: true
      self_check_output: true
      self_check_facts: true
      self_check_hallucination: true
      
    dialog:
      enabled: true
      topical_rail: true
      allowed_topics:
{_format_list(preset_config["allowed_topics"], indent=8)}
      disallowed_topics:
{_format_list(preset_config["disallowed_topics"], indent=8)}
      allow_only_listed_topics: false

# LLM Guard guardrails
guardrails:
  input:
    enabled: false
    checks:
      - type: prompt_injection
        enabled: true
        provider: llm_guard
        priority: 1
        threshold: 0.5
        on_fail: block

      - type: pii_detection
        enabled: true
        provider: llm_guard
        priority: 2
        threshold: 0.5
        on_fail: warn

  output:
    enabled: false
    checks:
      - type: pii_redaction
        enabled: true
        provider: llm_guard
        priority: 1
        on_fail: sanitize
'''


def _get_agent_yaml(agent_id: str, preset: str = "general") -> str:
    """Get agent-specific YAML template."""
    return f'''# {agent_id.replace("_", " ").title()} Agent Configuration
# Inherits from default.yaml and overrides specific settings

agent_id: {agent_id}
inherits: default

# Agent-specific instructions
instructions: |
  You are a helpful {agent_id.replace("_", " ")} assistant.
  Follow your designated topic boundaries.

# NeMo configuration overrides
nemo:
  enabled: true
  preset: {preset}
  
  rails:
    dialog:
      enabled: true
      # Customize allowed/disallowed topics for this agent
      allowed_topics:
        - greeting
        - help
      disallowed_topics:
        - violence
        - illegal_activity

# Guardrails configuration overrides
guardrails:
  input:
    enabled: true
    checks:
      - type: prompt_injection
        enabled: true
        provider: llm_guard
        priority: 1
        threshold: 0.5
        on_fail: block
'''


def _format_list(items: list, indent: int = 8) -> str:
    """Format a list as YAML."""
    prefix = " " * indent
    return "\n".join(f"{prefix}- {item}" for item in items)


if __name__ == "__main__":
    sys.exit(main())
