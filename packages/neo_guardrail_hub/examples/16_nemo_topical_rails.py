#!/usr/bin/env python
"""
Example 16: NeMo Topical Rails (Sprint 3D.1)

This example demonstrates how to use NeMo Guardrails' topical/dialog rails
to keep a Financial Agent on-topic and block off-topic discussions.

Topical rails use Colang dialog flows to:
- Define user intents for different topics
- Block off-topic requests (e.g., stock tips, medical advice)
- Allow on-topic requests (e.g., account balance, transaction history)

Prerequisites:
- Install NeMo Guardrails: pip install neo-guardrail-hub[nemo]
- Set OPENAI_API_KEY environment variable

Reference:
- https://docs.nvidia.com/nemo/guardrails/latest/getting-started/6-topical-rails/README.html
"""

import asyncio
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
env_path = Path(__file__).parent.parent / '.env'
load_dotenv(dotenv_path=env_path)

# Ensure OPENAI_API_KEY is set
if not os.getenv("OPENAI_API_KEY"):
    print("⚠️  Warning: OPENAI_API_KEY not set. NeMo guardrails require an LLM.")
    print("   Set your API key: export OPENAI_API_KEY=your-key-here\n")
    print("   Or create a .env file in the project root with OPENAI_API_KEY=your-key\n")

# Get the configs directory
CONFIGS_PATH = Path(__file__).parent.parent / "configs"


# =============================================================================
# Example 1: Basic Topical Rail with Provider
# =============================================================================

async def example_basic_topical_rail():
    """Demonstrate basic topical rail usage with NeMo Provider."""
    print("\n" + "=" * 60)
    print("Example 1: Basic Topical Rail with Provider")
    print("=" * 60)
    
    from neo_guardrail_hub.providers.nemo import NeMoProvider
    
    # Suppress verbose logging
    import logging
    logging.getLogger("nemoguardrails").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    
    # Create and initialize provider
    provider = NeMoProvider(
        config_path=CONFIGS_PATH,
        output_path=Path(__file__).parent.parent / "neo_configs",
    )
    
    print("\n🔄 Initializing NeMo Provider with dialog rails...")
    await provider.initialize()
    print("✅ Provider initialized!")
    
    # Test cases: (message, expected_on_topic)
    test_cases = [
        # ON-TOPIC - Financial domain questions
        ("What's my account balance?", True),
        ("Show me my transaction history", True),
        ("What are your fees?", True),
        ("How is the market doing?", True),
        
        # OFF-TOPIC - Should be blocked
        ("Give me a stock tip", False),
        ("Should I invest in Bitcoin?", False),
        ("Can you guarantee returns?", False),
        ("How do I make pasta?", False),
        ("What's your political opinion?", False),
        ("Give me medical advice", False),
    ]
    
    print("\n📋 Testing topical rails for Financial Agent:")
    print("-" * 60)
    
    for message, expected_on_topic in test_cases:
        result = await provider.check_topic(text=message)
        
        actual_on_topic = result["on_topic"]
        is_correct = actual_on_topic == expected_on_topic
        
        status = "✅" if is_correct else "❌"
        topic_status = "ON-TOPIC" if actual_on_topic else "OFF-TOPIC"
        expected_status = "on-topic" if expected_on_topic else "off-topic"
        
        print(f"\n{status} \"{message}\"")
        print(f"   Result: {topic_status}")
        print(f"   Expected: {expected_status}")
        if result.get("response") and not actual_on_topic:
            print(f"   Response: {result['response'][:60]}...")
    
    await provider.cleanup()
    print("\n✅ Example 1 complete!")


# =============================================================================
# Example 2: Topical Rail via Guardrail Interface
# =============================================================================

async def example_guardrail_interface():
    """Use NeMoTopicalRailGuardrail with standard guardrail interface."""
    print("\n" + "=" * 60)
    print("Example 2: Guardrail Interface")
    print("=" * 60)
    
    from neo_guardrail_hub.providers.nemo import NeMoProvider
    
    import logging
    logging.getLogger("nemoguardrails").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    
    provider = NeMoProvider(config_path=CONFIGS_PATH)
    await provider.initialize()
    
    # Get the topical rail guardrail
    guardrail = provider.get_guardrail("nemo_topical_rail")
    
    print(f"\nGuardrail: {guardrail.name}")
    print(f"Layer: {guardrail.layer}")
    print(f"Mode: {guardrail.mode}")
    print(f"Blocked Topics: {guardrail.blocked_topics[:3]}...")
    
    # Test messages
    test_messages = [
        "What's my portfolio value?",
        "Recommend me a stock to buy",
        "What are the trading fees?",
        "Can you help me cook dinner?",
    ]
    
    print("\n📋 Testing via guardrail interface:")
    print("-" * 60)
    
    for message in test_messages:
        result = await guardrail.check(message)
        
        status = "✅ PASSED" if result.passed else "❌ BLOCKED"
        print(f"\n{status}: \"{message}\"")
        print(f"   Risk Score: {result.risk_score:.2f}")
        if not result.passed:
            print(f"   Reason: {result.message[:50]}...")
    
    await provider.cleanup()
    print("\n✅ Example 2 complete!")


# =============================================================================
# Example 3: Orchestrator Integration
# =============================================================================

async def example_orchestrator_integration():
    """Use topical rails via the NeoGuardrailOrchestrator.guard_context()."""
    print("\n" + "=" * 60)
    print("Example 3: Orchestrator Integration (guard_context)")
    print("=" * 60)
    
    from neo_guardrail_hub import NeoGuardrailOrchestrator
    from neo_guardrail_hub.core.models import GuardrailContext
    
    import logging
    logging.getLogger("nemoguardrails").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    
    # Initialize orchestrator with default config
    # The config has dialog.topical_rail: true
    orchestrator = NeoGuardrailOrchestrator(config_path=str(CONFIGS_PATH))
    
    print("\n🔄 Initializing orchestrator with dialog rails...")
    await orchestrator.initialize()
    print("✅ Orchestrator initialized!")
    
    # Test cases for a financial assistant
    test_cases = [
        # On-topic - should pass
        ("What's my account balance?", "Financial query - on-topic"),
        ("Show me my recent transactions", "Transaction history - on-topic"),
        ("What are your commission fees?", "Fee inquiry - on-topic"),
        
        # Off-topic - should be blocked
        ("Give me a hot stock tip", "Stock tips - blocked"),
        ("Should I invest in crypto?", "Investment advice - blocked"),
        ("How do I make lasagna?", "Cooking - blocked"),
        ("Who should I vote for?", "Politics - blocked"),
    ]
    
    print("\n📋 Testing via orchestrator.guard_context():")
    print("-" * 60)
    
    for message, description in test_cases:
        print(f"\n👤 User: \"{message}\"")
        print(f"   ({description})")
        
        # Use guard_context for dialog/topical rails
        result = await orchestrator.guard_context(
            text=message,
        )
        
        # Overall result
        status = "✅ ON-TOPIC" if result.passed else "❌ OFF-TOPIC"
        print(f"   {status} (risk: {result.max_risk_score:.2f})")
        
        # Individual guardrail results
        for r in result.results:
            check_status = "✓" if r.passed else "✗"
            print(f"     [{check_status}] {r.guardrail_name}: risk={r.risk_score:.2f}")
    
    await orchestrator.cleanup()
    print("\n✅ Orchestrator example complete!")


# =============================================================================
# Example 4: Financial Agent Simulation
# =============================================================================

async def example_financial_agent():
    """Simulate a complete financial agent with topic guardrails."""
    print("\n" + "=" * 60)
    print("Example 4: Financial Agent Simulation")
    print("=" * 60)
    
    from neo_guardrail_hub import NeoGuardrailOrchestrator
    
    import logging
    logging.getLogger("nemoguardrails").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    
    orchestrator = NeoGuardrailOrchestrator(config_path=str(CONFIGS_PATH))
    await orchestrator.initialize()
    
    # Simulated LLM responses
    def simulate_llm(message: str) -> str:
        """Simulate LLM generating a response."""
        if "balance" in message.lower():
            return "Your current account balance is $12,345.67."
        elif "transaction" in message.lower():
            return "Your last 5 transactions: AAPL +$500, GOOGL -$250..."
        elif "fee" in message.lower():
            return "Our standard commission is $4.95 per trade."
        else:
            return "I'd be happy to help with your financial questions!"
    
    async def chat(user_message: str) -> str:
        """Process a chat message through the guardrails."""
        # Step 1: Check if message is on-topic using guard_context
        context_result = await orchestrator.guard_context(text=user_message)
        
        if not context_result.passed:
            # Off-topic - return refusal
            return "I'm a financial assistant and can only help with account-related questions. How can I assist with your finances today?"
        
        # Step 2: Check input safety
        input_result = await orchestrator.guard_input(text=user_message)
        
        if not input_result.passed:
            return "I'm sorry, I cannot process that request."
        
        # Step 3: Generate response (simulated)
        response = simulate_llm(user_message)
        
        return response
    
    # Conversation simulation
    print("\n💬 Financial Agent Chat Simulation:")
    print("-" * 60)
    
    conversations = [
        "What's my account balance?",
        "Show me my recent transactions",
        "Give me a stock tip to make money fast",
        "What are your trading fees?",
        "How do I cook spaghetti?",
        "Can you guarantee I'll make money?",
    ]
    
    for user_msg in conversations:
        print(f"\n👤 User: {user_msg}")
        response = await chat(user_msg)
        print(f"🤖 Agent: {response}")
    
    await orchestrator.cleanup()
    print("\n✅ Financial agent simulation complete!")


# =============================================================================
# Main
# =============================================================================

async def main():
    """Run all topical rails examples."""
    print("\n" + "=" * 60)
    print("NeMo Topical Rails Examples (Sprint 3D.1)")
    print("=" * 60)
    
    try:
        # Check if NeMo is installed
        try:
            import nemoguardrails
            print(f"\n✅ NeMo Guardrails version: {nemoguardrails.__version__}")
        except ImportError:
            print("\n❌ NeMo Guardrails not installed!")
            print("   Install with: pip install neo-guardrail-hub[nemo]")
            return
        
        # Run examples
        await example_basic_topical_rail()
        await example_guardrail_interface()
        await example_orchestrator_integration()
        await example_financial_agent()
        
        print("\n" + "=" * 60)
        print("All examples completed!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
