"""
Example 19: Complete Phase 4 Workflow

This example demonstrates the complete Phase 4 workflow:
1. Initialize config files using pre-generation API
2. Use caching for efficient Rails reuse
3. Execute guardrails with cached instances

This is the recommended production workflow for Neo Guardrail Hub.

Usage:
    python examples/19_complete_workflow.py
    
Requirements:
    - pip install neo-guardrail-hub[nemo]
    - OPENAI_API_KEY environment variable
"""

import asyncio
import time
from pathlib import Path
import os
from dotenv import load_dotenv

from neo_guardrail_hub import (
    # Core
    NeoGuardrailOrchestrator,
    
    # Phase 4 API
    initialize,
    generate_configs,
    get_rails_cache,
    clear_rails_cache,
)
from neo_guardrail_hub.api.caching import reset_rails_cache

# Load environment variables from .env file
env_path = Path(__file__).parent.parent / '.env'
load_dotenv(dotenv_path=env_path)

async def production_workflow():
    """
    Complete production workflow for Neo Guardrail Hub.
    
    This demonstrates the recommended setup for production:
    1. Pre-generate configs at startup
    2. Pre-load Rails into cache
    3. Use orchestrator with cached Rails
    4. Handle multiple agents
    """
    print("\n" + "=" * 60)
    print("Neo Guardrail Hub - Complete Production Workflow")
    print("=" * 60)
    
    # -------------------------------------------------------------------------
    # STEP 1: Application Startup - Pre-generate and cache
    # -------------------------------------------------------------------------
    print("\n--- STEP 1: Application Startup ---")
    
    # Reset cache for clean demo
    reset_rails_cache()
    
    # Pre-generate configs and optionally preload Rails
    # This should happen once at application startup
    result = await initialize(
        config_path="./configs",
        output_path="./neo_configs",
        preload_rails=True,  # Pre-load into cache for instant access
        force=True,          # Regenerate to ensure fresh config
    )
    
    if not result.success:
        print(f"Initialization failed: {result.errors}")
        return
    
    print(f"✓ Configs generated at: {result.output_path}")
    print(f"✓ Rails pre-loaded into cache")
    
    # Show what's cached
    cache = get_rails_cache()
    print(f"\nCache status:")
    print(f"  Entries: {cache.list_keys()}")
    print(f"  Size: {cache.size}")
    
    # -------------------------------------------------------------------------
    # STEP 2: Create Orchestrator (uses cached Rails)
    # -------------------------------------------------------------------------
    print("\n--- STEP 2: Create Orchestrator ---")
    
    orchestrator = NeoGuardrailOrchestrator(
        config_path="./configs",
    )
    
    print("✓ Orchestrator created")
    
    # Initialize the orchestrator
    await orchestrator.initialize()
    print("✓ Orchestrator initialized")
    
    # -------------------------------------------------------------------------
    # STEP 3: Process Requests with Cached Rails
    # -------------------------------------------------------------------------
    print("\n--- STEP 3: Process Requests ---")
    
    # Simulate multiple requests
    test_messages = [
        "Hello, I need help with my account",
        "What's my portfolio balance?",
        "Can you give me stock tips?",  # Should be blocked by topical rails
        "How do I transfer funds?",
    ]
    
    for message in test_messages:
        print(f"\nProcessing: '{message[:40]}...'")
        
        start_time = time.time()
        
        try:
            result = await orchestrator.guard_input(message)
            elapsed = (time.time() - start_time) * 1000
            
            status = "✓ PASSED" if result.passed else "✗ BLOCKED"
            print(f"  {status} ({elapsed:.0f}ms)")
            
            if not result.passed and result.results:
                print(f"  Reason: {result.results[0].message}")
                
        except Exception as e:
            print(f"  Error: {e}")
    
    # -------------------------------------------------------------------------
    # STEP 4: Multi-Agent Support
    # -------------------------------------------------------------------------
    print("\n--- STEP 4: Multi-Agent Support ---")
    
    # Generate configs for a specific agent
    wealth_result = generate_configs(
        config_path="./configs",
        agent_id="wealth_advisor",
        output_path="./neo_configs/wealth_advisor",
        force=True,
    )
    
    if wealth_result.success:
        print(f"✓ Wealth advisor configs generated")
        
        # Process with agent-specific config
        agent_result = await orchestrator.guard_input(
            "Show me my investment portfolio",
            agent_id="wealth_advisor",
        )
        
        status = "✓ PASSED" if agent_result.passed else "✗ BLOCKED"
        print(f"  Agent request: {status}")
    
    # -------------------------------------------------------------------------
    # STEP 5: Monitoring and Cleanup
    # -------------------------------------------------------------------------
    print("\n--- STEP 5: Monitoring ---")
    
    # Check cache statistics
    stats = cache.stats
    print(f"\nFinal Cache Statistics:")
    print(f"  Total entries: {stats['size']}")
    print(f"  Cache hits: {stats['hits']}")
    print(f"  Cache misses: {stats['misses']}")
    print(f"  Hit rate: {stats['hit_rate']:.0%}")
    
    # Cleanup when shutting down
    await orchestrator.cleanup()
    print("\n✓ Orchestrator cleaned up")
    
    # Optional: Clear cache on shutdown
    # clear_rails_cache()
    # print("✓ Cache cleared")
    
    print("\n" + "=" * 60)
    print("Workflow Complete!")
    print("=" * 60)


async def development_workflow():
    """
    Development workflow with more verbose output.
    
    Use this during development to understand what's happening.
    """
    print("\n" + "=" * 60)
    print("Neo Guardrail Hub - Development Workflow")
    print("=" * 60)
    
    reset_rails_cache()
    
    # Step 1: Generate and inspect configs
    print("\n--- Generating Configs ---")
    
    result = generate_configs(
        config_path="./configs",
        output_path="./neo_configs/dev",
        force=True,
    )
    
    if result.success:
        print(f"✓ Generated at: {result.output_path}")
        
        print("\nGenerated files:")
        for f in result.generated_files:
            print(f"  - {f.name}")
        
        print("\nConfig summary:")
        for key, value in result.config_summary.items():
            print(f"  {key}: {value}")
        
        if result.warnings:
            print("\nWarnings:")
            for w in result.warnings:
                print(f"  ⚠ {w}")
    else:
        print(f"✗ Failed: {result.errors}")
        return
    
    # Step 2: Load Rails and inspect
    print("\n--- Loading Rails ---")
    
    cache = get_rails_cache()
    
    try:
        start = time.time()
        await cache.get_or_create(
            key="dev",
            config_path=result.output_path,
        )
        elapsed = time.time() - start
        
        print(f"✓ Rails loaded in {elapsed:.2f}s")
        
    except ImportError:
        print("⚠ NeMo Guardrails not installed")
        print("  Install with: pip install neo-guardrail-hub[nemo]")
    except Exception as e:
        print(f"✗ Error loading Rails: {e}")
    
    print("\n" + "=" * 60)


async def main():
    """Run example workflows."""
    
    # Show production workflow
    await production_workflow()
    
    # Optionally show development workflow
    # await development_workflow()


if __name__ == "__main__":
    asyncio.run(main())
