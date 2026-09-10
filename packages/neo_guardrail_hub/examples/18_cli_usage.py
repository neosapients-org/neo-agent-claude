"""
Example 18: CLI Usage

This example demonstrates how to use the Neo Guardrail Hub CLI
for configuration management.

CLI Commands:
    neo-guardrail init      - Initialize configuration files
    neo-guardrail generate  - Generate NeMo config files
    neo-guardrail validate  - Validate configuration
    neo-guardrail cache     - Manage Rails cache

Installation:
    pip install neo-guardrail-hub[nemo]
    
Usage:
    # Run this script to see CLI examples
    python examples/18_cli_usage.py
    
    # Or use the CLI directly:
    neo-guardrail --help
    neo-guardrail init --path ./my_project/configs
    neo-guardrail generate --config ./configs
    neo-guardrail validate --config ./configs
"""

import subprocess
import sys
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
env_path = Path(__file__).parent.parent / '.env'
load_dotenv(dotenv_path=env_path)

def run_command(cmd: str, description: str):
    """Run a CLI command and show output."""
    print(f"\n{'=' * 60}")
    print(f"Command: {cmd}")
    print(f"Description: {description}")
    print("=" * 60)
    
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
        
        if result.stdout:
            print("\nOutput:")
            print(result.stdout)
        
        if result.stderr:
            print("\nStderr:")
            print(result.stderr)
            
        print(f"\nExit code: {result.returncode}")
        
    except subprocess.TimeoutExpired:
        print("\nCommand timed out")
    except Exception as e:
        print(f"\nError running command: {e}")


def main():
    """Demonstrate CLI usage."""
    print("\n" + "=" * 60)
    print("Neo Guardrail Hub - CLI Usage Examples")
    print("=" * 60)
    
    # Check if CLI is available
    print("\nChecking if neo-guardrail CLI is available...")
    
    # Example 1: Show help
    print("\n\n" + "=" * 60)
    print("1. Show CLI Help")
    print("=" * 60)
    print("""
Command: neo-guardrail --help

This shows all available commands and options.

Available commands:
  init       Initialize Neo Guardrail Hub in current directory
  generate   Generate NeMo configuration files  
  validate   Validate configuration files
  cache      Manage Rails cache
""")
    
    # Example 2: Initialize
    print("\n\n" + "=" * 60)
    print("2. Initialize Configuration")
    print("=" * 60)
    print("""
Command: neo-guardrail init --path ./configs

Options:
  --path     Path for configuration directory (default: ./configs)
  --preset   Industry preset: general, financial, healthcare, customer_service
  --agent    Create agent-specific config

Examples:
  # Basic initialization with general preset
  neo-guardrail init
  
  # Financial industry preset
  neo-guardrail init --preset financial
  
  # Create with agent-specific config
  neo-guardrail init --agent wealth_advisor --preset financial

This creates:
  ./configs/default.yaml           - Main configuration file
  ./configs/agents/agent_id.yaml   - Agent-specific config (if --agent used)
""")
    
    # Example 3: Generate configs
    print("\n\n" + "=" * 60)
    print("3. Generate NeMo Configs")
    print("=" * 60)
    print("""
Command: neo-guardrail generate --config ./configs

Options:
  --config       Path to configs directory (default: ./configs)
  --agent        Agent ID for agent-specific config
  --output       Output directory for generated files
  --force        Force regeneration even if files exist
  --all-agents   Generate configs for all agents

Examples:
  # Generate default config
  neo-guardrail generate
  
  # Generate for specific agent
  neo-guardrail generate --agent wealth_advisor
  
  # Generate for all agents in configs/agents/
  neo-guardrail generate --all-agents
  
  # Force regeneration
  neo-guardrail generate --force

Generated files:
  ./neo_configs/config.yml    - NeMo main configuration
  ./neo_configs/rails/        - Colang flow definitions (if used)
""")
    
    # Example 4: Validate configs
    print("\n\n" + "=" * 60)
    print("4. Validate Configuration")
    print("=" * 60)
    print("""
Command: neo-guardrail validate --config ./configs

Options:
  --config   Path to configs directory (default: ./configs)
  --agent    Agent ID to validate

Examples:
  # Validate default config
  neo-guardrail validate
  
  # Validate specific agent
  neo-guardrail validate --agent wealth_advisor

Validation checks:
  - YAML syntax
  - Required fields present
  - NeMo configuration
  - Guardrails configuration
""")
    
    # Example 5: Cache management
    print("\n\n" + "=" * 60)
    print("5. Cache Management")
    print("=" * 60)
    print("""
Command: neo-guardrail cache --stats

Options:
  --clear   Clear the Rails cache
  --stats   Show cache statistics

Examples:
  # Show cache stats
  neo-guardrail cache --stats
  
  # Clear the cache
  neo-guardrail cache --clear

Cache info:
  - Size: Number of cached Rails instances
  - Hits: Cache hits (reused instances)
  - Misses: Cache misses (new loads)
  - Hit Rate: Percentage of requests served from cache
""")
    
    # Example 6: Workflow example
    print("\n\n" + "=" * 60)
    print("6. Complete Workflow Example")
    print("=" * 60)
    print("""
A typical workflow for setting up Neo Guardrail Hub:

1. Initialize configuration:
   $ neo-guardrail init --preset financial --agent wealth_advisor
   
2. Edit the generated configs:
   $ vim configs/default.yaml
   $ vim configs/agents/wealth_advisor.yaml
   
3. Validate your configuration:
   $ neo-guardrail validate
   
4. Generate NeMo config files:
   $ neo-guardrail generate --all-agents
   
5. Use in your application:
   
   from neo_guardrail_hub import NeoGuardrailOrchestrator
   
   orchestrator = NeoGuardrailOrchestrator(config_path="./configs")
   result = await orchestrator.guard_input("user message")
   
6. Monitor cache (optional):
   $ neo-guardrail cache --stats
""")
    
    # Example 7: Programmatic CLI usage
    print("\n\n" + "=" * 60)
    print("7. Programmatic CLI Usage")
    print("=" * 60)
    print("""
You can also use CLI functions programmatically:

from neo_guardrail_hub.cli import cmd_init, cmd_generate, cmd_validate
import argparse

# Create args namespace
args = argparse.Namespace(
    path="./configs",
    preset="financial",
    agent="my_agent",
)

# Run init command
result = cmd_init(args)
if result == 0:
    print("Init successful!")
""")
    
    print("\n" + "=" * 60)
    print("CLI Examples Complete!")
    print("=" * 60)
    print("\nTo use the CLI, run: neo-guardrail --help")


if __name__ == "__main__":
    main()
