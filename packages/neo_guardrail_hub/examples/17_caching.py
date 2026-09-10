"""
Example 17: Rails Caching

This example demonstrates the Rails caching system that avoids
expensive re-initialization of NeMo LLMRails instances.

Benefits of caching:
1. Faster repeated calls - Rails are loaded once
2. Reduced memory - Single instance shared across requests
3. Automatic staleness detection - Cache refreshes when configs change

Usage:
    python examples/17_caching.py
    
Requirements:
    - pip install neo-guardrail-hub[nemo]
    - OPENAI_API_KEY environment variable
"""

import asyncio
import time
from pathlib import Path
import os
from dotenv import load_dotenv

# Load environment variables from .env file
env_path = Path(__file__).parent.parent / '.env'
load_dotenv(dotenv_path=env_path)

from neo_guardrail_hub import (
    generate_configs,
    get_rails_cache,
    clear_rails_cache,
    NeoGuardrailOrchestrator,
)
from neo_guardrail_hub.api.caching import reset_rails_cache


async def example_basic_caching():
    """
    Example 1: Basic cache usage
    
    Get or create Rails from the cache, demonstrating cache hits.
    """
    print("\n" + "=" * 60)
    print("Example 1: Basic Cache Usage")
    print("=" * 60)
    
    # Reset cache for clean demo
    reset_rails_cache()
    
    # Get the global cache
    cache = get_rails_cache()
    print(f"Cache size: {cache.size}")
    
    # First, generate configs
    result = generate_configs(
        config_path="./configs",
        output_path="./neo_configs",
        force=True,
    )
    
    if not result.success:
        print(f"Config generation failed: {result.errors}")
        return
    
    # Get or create Rails (first call = cache miss)
    start = time.time()
    try:
        rails1 = await cache.get_or_create(
            key="default",
            config_path=result.output_path,
        )
        first_time = time.time() - start
        print(f"\nFirst load (cache miss): {first_time:.2f}s")
        
        # Second call = cache hit
        start = time.time()
        rails2 = await cache.get_or_create(
            key="default",
            config_path=result.output_path,
        )
        second_time = time.time() - start
        print(f"Second load (cache hit): {second_time:.4f}s")
        
        # Verify same instance
        print(f"\nSame instance: {rails1 is rails2}")
        
        # Show cache stats
        stats = cache.stats
        print(f"\nCache Stats:")
        print(f"  Hits: {stats['hits']}")
        print(f"  Misses: {stats['misses']}")
        print(f"  Hit Rate: {stats['hit_rate']:.0%}")
        
    except ImportError:
        print("NeMo Guardrails not installed. Skipping actual Rails loading.")


async def example_multi_agent_caching():
    """
    Example 2: Caching for multiple agents
    
    Each agent has its own cached Rails instance.
    """
    print("\n" + "=" * 60)
    print("Example 2: Multi-Agent Caching")
    print("=" * 60)
    
    reset_rails_cache()
    cache = get_rails_cache()
    
    agents = ["default", "wealth_advisor"]
    
    for agent_id in agents:
        # Generate config for each agent
        result = generate_configs(
            config_path="./configs",
            agent_id=agent_id if agent_id != "default" else None,
            output_path=f"./neo_configs/{agent_id}",
            force=True,
        )
        
        if result.success:
            print(f"\n✓ Config generated for: {agent_id}")
            print(f"  Output: {result.output_path}")
        else:
            print(f"✗ Failed for {agent_id}: {result.errors}")
    
    print(f"\nCache contains {cache.size} entries")
    print(f"Cached agents: {cache.list_keys()}")


def example_cache_management():
    """
    Example 3: Cache management operations
    
    Shows how to clear, reset, and inspect the cache.
    """
    print("\n" + "=" * 60)
    print("Example 3: Cache Management")
    print("=" * 60)
    
    # Get cache with custom settings
    cache = get_rails_cache(
        ttl_seconds=3600,  # 1 hour TTL
        max_size=50,       # Max 50 entries
        check_staleness=True,  # Check for config changes
    )
    
    print("Cache Configuration:")
    stats = cache.stats
    print(f"  TTL: {stats['ttl_seconds']}s")
    print(f"  Max Size: {stats['max_size']}")
    print(f"  Current Size: {stats['size']}")
    
    # Check if specific agent is cached
    agent = "wealth_advisor"
    if cache.contains(agent):
        print(f"\n{agent} is in cache")
    else:
        print(f"\n{agent} is NOT in cache")
    
    # Clear the cache
    clear_rails_cache()
    print("\n✓ Cache cleared")
    print(f"  New size: {cache.size}")
    
    # Reset creates new cache instance on next get
    reset_rails_cache()
    print("✓ Cache reset (new instance will be created)")


async def example_staleness_detection():
    """
    Example 4: Automatic staleness detection
    
    The cache detects when config files have changed and reloads.
    """
    print("\n" + "=" * 60)
    print("Example 4: Staleness Detection")
    print("=" * 60)
    
    reset_rails_cache()
    cache = get_rails_cache(check_staleness=True)
    
    # Generate initial config
    result = generate_configs(
        config_path="./configs",
        output_path="./neo_configs/staleness_test",
        force=True,
    )
    
    if not result.success:
        print(f"Config generation failed: {result.errors}")
        return
    
    print("Initial config generated")
    
    # The cache tracks file hashes to detect changes
    # When files change, get_or_create will reload
    
    print("\nHow staleness detection works:")
    print("1. Cache computes hash of config files")
    print("2. On cache hit, current hash is compared")
    print("3. If hash changed, Rails are reloaded")
    print("4. This ensures fresh config is always used")


async def example_with_orchestrator():
    """
    Example 5: Caching with NeoGuardrailOrchestrator
    
    Shows how caching integrates with the main orchestrator.
    """
    print("\n" + "=" * 60)
    print("Example 5: Caching with Orchestrator")
    print("=" * 60)
    
    reset_rails_cache()
    
    # Pre-generate configs
    result = generate_configs(
        config_path="./configs",
        output_path="./neo_configs",
        force=True,
    )
    
    if not result.success:
        print(f"Config generation failed: {result.errors}")
        return
    
    print("Configs pre-generated")
    
    # Create orchestrator
    # The NeMo provider will use the cached Rails
    orchestrator = NeoGuardrailOrchestrator(
        config_path="./configs",
    )
    
    print("\nOrchestrator created")
    print("NeMo provider will use cached Rails when initialized")
    
    # Show cache stats
    cache = get_rails_cache()
    print(f"\nFinal cache stats:")
    print(f"  Size: {cache.size}")
    print(f"  Entries: {cache.list_keys()}")


async def main():
    """Run all caching examples."""
    print("\n" + "=" * 60)
    print("Neo Guardrail Hub - Rails Caching Examples")
    print("=" * 60)
    
    # Example 1: Basic caching
    await example_basic_caching()
    
    # Example 2: Multi-agent
    await example_multi_agent_caching()
    
    # Example 3: Management
    example_cache_management()
    
    # Example 4: Staleness
    await example_staleness_detection()
    
    # Example 5: With orchestrator
    await example_with_orchestrator()
    
    print("\n" + "=" * 60)
    print("All caching examples completed!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
