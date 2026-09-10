"""
Example 21: NeMo Wealth Management Domain Check

This example demonstrates:
1. Automated generation of wealth_management_domain_check prompt from default.yaml
2. Using task_manager to render the prompt (like in finance_guardrail.ipynb)
3. Testing ALLOWED and BLOCKED scenarios for wealth management domain
4. Both direct provider usage and orchestrator pattern

The wealth_management_domain_check is configured in default.yaml under:
guardrails:
  context:
    checks:
      - type: wealth_management_domain_check
        provider: nemo
        enabled: true

This generates a prompt in config.yml that can be used via task_manager.

Prerequisites:
- Install NeMo Guardrails: pip install neo-guardrail-hub[nemo]
- Set OPENAI_API_KEY environment variable
"""

import asyncio
import os
import logging
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
env_path = Path(__file__).parent.parent / '.env'
load_dotenv(dotenv_path=env_path)

# Configure logging to reduce verbosity
logging.getLogger("nemoguardrails").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)

# Get the configs directory (relative to this example)
CONFIGS_PATH = Path(__file__).parent.parent / "configs"

# Get the configs directory (relative to this example)
CONFIGS_PATH = Path(__file__).parent.parent / "configs"


# =============================================================================
# Example 1: Direct Provider Usage - Wealth Management Domain Check
# =============================================================================

async def example_direct_provider():
    """Example 1: Direct NeMo Provider with wealth management domain check."""
    print("\n" + "=" * 80)
    print("Example 1: Direct Provider - Wealth Management Domain Check")
    print("=" * 80)
    
    from neo_guardrail_hub.providers.nemo import NeMoProvider
    
    print(f"\nUsing config from: {CONFIGS_PATH}")
    
    # Create the NeMo provider
    provider = NeMoProvider(
        config_path=CONFIGS_PATH,
        prompt_style="simple",  # Use simple prompts (faster, cheaper)
    )
    
    print("\n🔄 Initializing NeMo Provider...")
    print("   (This generates NeMo configs with wealth_management_domain_check)")
    
    # Initialize the provider
    await provider.initialize()
    
    print(f"✅ Provider initialized!")
    print(f"   Generated configs at: {provider.generated_config_path}")
    
    return provider


async def example_test_allowed_queries(provider):
    """Test ALLOWED queries - valid wealth management questions."""
    print("\n" + "=" * 80)
    print("Example 2: Testing ALLOWED Queries")
    print("=" * 80)
    
    allowed_queries = [
        "What is asset allocation and why is it important?",
        "Can you explain diversification strategies?",
        "How do I create a retirement plan?",
        "What are the differences between stocks and bonds?",
        "Tell me about risk management in investing",
    ]
    
    for query in allowed_queries:
        print(f"\n📝 Query: '{query}'")
        result = await provider.check_wealth_management_domain(query)
        status = "✅ ALLOWED" if result["allowed"] else "❌ BLOCKED"
        print(f"   Result: {status}")
        if not result["allowed"]:
            print(f"   Response: {result['response'][:100]}...")


async def example_test_blocked_queries(provider):
    """Test BLOCKED queries - specific stock recommendations."""
    print("\n" + "=" * 80)
    print("Example 3: Testing BLOCKED Queries")
    print("=" * 80)
    
    blocked_queries = [
        "Should I buy Tesla stock right now?",
        "Is this the right time to sell my Apple shares?",
        "Can you predict if Bitcoin will go up tomorrow?",
        "Tell me which penny stock to invest in",
        "Should I move all my money to gold?",
    ]
    
    for query in blocked_queries:
        print(f"\n📝 Query: '{query}'")
        result = await provider.check_wealth_management_domain(query)
        status = "✅ ALLOWED" if result["allowed"] else "❌ BLOCKED"
        print(f"   Result: {status}")
        if not result["allowed"]:
            # Show blocked reason
            blocked_reason = result['response'].split('BLOCKED:')[1].strip() if 'BLOCKED:' in result['response'] else result['response'][:100]
            print(f"   Reason: {blocked_reason[:150]}...")


async def example_with_context(provider):
    """Test with conversation context."""
    print("\n" + "=" * 80)
    print("Example 4: Testing with Conversation Context")
    print("=" * 80)
    
    from neo_guardrail_hub.core.models import GuardrailContext
    
    # Simulate a conversation context
    context = GuardrailContext(
        agent_id="wealth_advisor",
        session_id="session_001",
        user_id="demo_user",
        metadata={
            "conversation_history": [
                {"role": "user", "content": "I'm planning for retirement"},
                {"role": "assistant", "content": "Great! Let me help you understand retirement planning strategies."},
            ]
        }
    )
    
    query_with_context = "What should I do next?"
    print(f"\n📝 Previous context: User is planning for retirement")
    print(f"📝 Query: '{query_with_context}'")
    
    result = await provider.check_wealth_management_domain(query_with_context, context=context)
    status = "✅ ALLOWED" if result["allowed"] else "❌ BLOCKED"
    print(f"   Result: {status}")
    print(f"   Note: Context helps understand 'what should I do next' is about retirement planning")


# =============================================================================
# Example 5: Orchestrator Pattern
# =============================================================================

async def example_orchestrator_pattern():
    """Example 5: Use with NeoGuardrailOrchestrator."""
    print("\n" + "=" * 80)
    print("Example 5: Orchestrator Pattern with Wealth Management Check")
    print("=" * 80)
    
    from neo_guardrail_hub import NeoGuardrailOrchestrator
    
    # Initialize orchestrator
    orchestrator = NeoGuardrailOrchestrator(config_path=str(CONFIGS_PATH))
    
    print("\n🔄 Initializing orchestrator...")
    await orchestrator.initialize()
    print("✅ Orchestrator initialized!")
    
    # Test with orchestrator
    test_cases = [
        ("Safe query", "What is portfolio rebalancing?"),
        ("Blocked query", "Should I buy Tesla stock now?"),
        ("Education query", "Explain compound interest"),
    ]
    
    for name, query in test_cases:
        print(f"\n📝 {name}: '{query}'")
        result = await orchestrator.guard_context(query)
        
        print(f"   Passed: {result.passed}")
        print(f"   Risk Score: {result.max_risk_score:.2f}")
        if not result.passed:
            print(f"   Failed Checks: {result.failed_checks}")
    
    await orchestrator.cleanup()
    print("\n✓ Orchestrator pattern demo complete!")


# =============================================================================
# Example 6: GuardSession Pattern
# =============================================================================

async def example_guard_session():
    """Example 6: Use GuardSession with wealth management domain check."""
    print("\n" + "=" * 80)
    print("Example 6: GuardSession Pattern")
    print("=" * 80)
    
    from neo_guardrail_hub import GuardSession
    
    async with GuardSession(
        agent_id="wealth_advisor",
        session_id="wm_session_001",
        user_id="demo_user",
        config_path=str(CONFIGS_PATH),
        metadata={
            "demo": "wealth_management",
            "user_tier": "premium",
        },
    ) as session:
        
        print("\n1. Starting wealth management conversation...")
        
        # Turn 1: Safe educational query
        query1 = "What is asset allocation?"
        print(f"\n   User: {query1}")
        
        result1 = await session.guard_context(query1)
        if result1.passed:
            print(f"   ✅ ALLOWED - Educational content")
            print(f"   Risk Score: {result1.max_risk_score:.2f}")
        
        # Turn 2: Blocked specific recommendation
        query2 = "Should I buy Apple stock?"
        print(f"\n   User: {query2}")
        
        result2 = await session.guard_context(query2)
        if not result2.passed:
            print(f"   ❌ BLOCKED - Specific stock recommendation")
            print(f"   Risk Score: {result2.max_risk_score:.2f}")
        
        # Session summary
        print("\n2. Session Statistics:")
        print(f"   Total checks: {len(session.get_results())}")
        print(f"   Failed checks: {len(session.get_failed_checks())}")
    
    print("\n✓ GuardSession pattern demo complete!")


# =============================================================================
# Main: Run all examples
# =============================================================================

async def main():
    """Run all wealth management domain check examples."""
    print("=" * 80)
    print("Neo Guardrail Hub - Wealth Management Domain Check")
    print("=" * 80)
    
    print("\nThis example demonstrates wealth_management_domain_check feature:")
    print("  - Automated config generation from default.yaml")
    print("  - Task manager integration (like finance_guardrail.ipynb)")
    print("  - Distinguishes education vs specific recommendations")
    print("\nPrerequisites:")
    print("  - pip install neo-guardrail-hub[nemo]")
    print("  - export OPENAI_API_KEY=your-key-here")
    
    # Check OpenAI API key
    if not os.getenv("OPENAI_API_KEY"):
        print("\n⚠️  Warning: OPENAI_API_KEY not set!")
        print("   NeMo guardrails require an LLM API key.")
        return
    
    try:
        # Example 1: Initialize provider
        provider = await example_direct_provider()
        
        # Example 2: Test allowed queries
        # await example_test_allowed_queries(provider)
        
        # Example 3: Test blocked queries
        await example_test_blocked_queries(provider)
        
        # Example 4: Test with context
        # await example_with_context(provider)
        
        # Cleanup provider
        await provider.cleanup()
        
        # Example 5: Orchestrator pattern
        await example_orchestrator_pattern()
        
        # Example 6: GuardSession pattern
        await example_guard_session()
        
        print("All examples completed! 🎉")
        print("=" * 80)
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nMake sure you have:")
        print("  1. NeMo Guardrails installed: pip install neo-guardrail-hub[nemo]")
        print("  2. OpenAI API key set: export OPENAI_API_KEY=your-key")
        raise


if __name__ == "__main__":
    asyncio.run(main())

