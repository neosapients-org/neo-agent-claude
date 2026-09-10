#!/usr/bin/env python
"""
Example 15: NeMo Self-Check Hallucination (Sprint 3C.2)

This example demonstrates how to use the NeMo Provider for LLM-based
hallucination detection using NeMo Guardrails' self_check_hallucination.

The hallucination detection compares the original response against 
alternative LLM generations to detect inconsistencies (SelfCheckGPT approach).

Prerequisites:
- Install NeMo Guardrails: pip install neo-guardrail-hub[nemo]
- Set OPENAI_API_KEY environment variable

Reference:
- https://docs.nvidia.com/nemo/guardrails/latest/user-guides/guardrails-library.html#hallucination-detection
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
# Example 1: Basic Hallucination Detection
# =============================================================================

async def example_basic_hallucination():
    """Demonstrate basic hallucination detection with check_hallucination method."""
    print("\n" + "=" * 60)
    print("Example 1: Basic Hallucination Detection")
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
    
    print("\n🔄 Initializing NeMo Provider...")
    await provider.initialize()
    print("✅ Provider initialized!")
    
    # Test cases: (user_input, response, is_likely_hallucination)
    # Note: Hallucination detection is probabilistic - results may vary
    test_cases = [
        # Likely NOT hallucination (common knowledge, LLM should be consistent)
        ("What is the capital of France?", "The capital of France is Paris.", False),
        ("What is 2 + 2?", "2 + 2 equals 4.", False),
        
        # Likely hallucination (made up or inconsistent facts)
        ("Who invented the telephone?", "Einstein invented the telephone in 1920.", True),
        ("What is the population of Mars?", "Mars has a population of 5 million people.", True),
    ]
    
    print("\n⚠️  Note: Hallucination detection generates alternative responses")
    print("   and checks for consistency. Results may vary between runs.")
    print("\n" + "-" * 60)
    
    for user_input, response, expected_hallucination in test_cases:
        print(f"\n👤 Query: \"{user_input}\"")
        print(f"🤖 Response: \"{response}\"")
        
        result = await provider.check_hallucination(
            response=response,
            user_input=user_input,
            num_samples=2,  # Generate 2 alternative responses
        )
        
        is_hallucination = result["is_hallucination"]
        consistency = result["consistency"]
        
        status = "🔴 HALLUCINATION" if is_hallucination else "🟢 CONSISTENT"
        print(f"   {status}")
        print(f"   Consistency: {consistency:.2f}")
        print(f"   LLM Answer: {result['details'].get('llm_answer', 'N/A')[:30]}...")
    
    await provider.cleanup()
    return provider


# =============================================================================
# Example 2: Using the Guardrail Interface
# =============================================================================

async def example_guardrail_interface():
    """Use NeMoSelfCheckHallucinationGuardrail with standard interface."""
    print("\n" + "=" * 60)
    print("Example 2: Guardrail Interface")
    print("=" * 60)
    
    from neo_guardrail_hub.providers.nemo import NeMoProvider
    from neo_guardrail_hub.core.models import GuardrailContext
    
    import logging
    logging.getLogger("nemoguardrails").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    
    provider = NeMoProvider(config_path=CONFIGS_PATH)
    await provider.initialize()
    
    # Get the hallucination guardrail
    guardrail = provider.get_guardrail("nemo_self_check_hallucination")
    
    print(f"\nGuardrail: {guardrail.name}")
    print(f"Threshold: {guardrail.threshold}")
    print(f"Mode: {guardrail.mode}")
    
    # Test cases - user_input must be passed via context
    test_cases = [
        ("What is the largest planet?", "Jupiter is the largest planet in our solar system."),
        ("Who wrote Romeo and Juliet?", "Shakespeare wrote Romeo and Juliet."),
        ("When was the internet invented?", "The internet was invented by Albert Einstein in 1850."),
    ]
    
    for user_input, response in test_cases:
        # Pass user_input in context metadata
        context = GuardrailContext(metadata={"user_input": user_input})
        
        result = await guardrail.check(response, context=context)
        
        status = "✅ Passed" if result.passed else "❌ Blocked"
        print(f"\n{status}: \"{response[:40]}...\"")
        print(f"   Risk Score: {result.risk_score:.2f}")
        if not result.passed:
            print(f"   Reason: Potential hallucination detected")
    
    await provider.cleanup()


# =============================================================================
# Example 3: Warning Mode vs Blocking Mode
# =============================================================================

async def example_warning_mode():
    """Demonstrate warning mode vs blocking mode."""
    print("\n" + "=" * 60)
    print("Example 3: Warning Mode vs Blocking Mode")
    print("=" * 60)
    
    from neo_guardrail_hub.providers.nemo import NeMoProvider
    from neo_guardrail_hub.core.models import GuardrailContext
    
    import logging
    logging.getLogger("nemoguardrails").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    
    provider = NeMoProvider(config_path=CONFIGS_PATH)
    await provider.initialize()
    
    user_input = "Who invented the light bulb?"
    response = "Thomas Edison invented the first practical light bulb."
    
    context = GuardrailContext(metadata={"user_input": user_input})
    
    # Test in blocking mode
    print("\n📛 BLOCKING MODE:")
    guardrail_blocking = provider.get_guardrail(
        "nemo_self_check_hallucination",
        config={"mode": "blocking"}
    )
    result = await guardrail_blocking.check(response, context=context)
    print(f"   Response: \"{response[:40]}...\"")
    print(f"   Passed: {result.passed}")
    print(f"   Message: {result.message[:50]}...")
    
    # Test in warning mode
    print("\n⚠️  WARNING MODE:")
    guardrail_warning = provider.get_guardrail(
        "nemo_self_check_hallucination", 
        config={"mode": "warning"}
    )
    result = await guardrail_warning.check(response, context=context)
    print(f"   Response: \"{response[:40]}...\"")
    print(f"   Passed: {result.passed}")  # Always passes in warning mode
    print(f"   Warning: {result.details.get('warning', 'None')[:50] if result.details.get('warning') else 'None'}")
    
    await provider.cleanup()


# =============================================================================
# Example 4: Orchestrator Integration
# =============================================================================

async def example_orchestrator_integration():
    """Use hallucination detection via the NeoGuardrailOrchestrator.
    
    The orchestrator runs all enabled output guardrails including:
    - nemo_self_check_output (policy compliance)
    - nemo_self_check_facts (fact-checking)
    - nemo_self_check_hallucination (consistency)
    
    User input is passed via GuardrailContext.metadata['user_input'].
    """
    print("\n" + "=" * 60)
    print("Example 4: Orchestrator Integration")
    print("=" * 60)
    
    from neo_guardrail_hub import NeoGuardrailOrchestrator
    from neo_guardrail_hub.core.models import GuardrailContext
    
    import logging
    logging.getLogger("nemoguardrails").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    
    # Initialize orchestrator with default config
    # The config has self_check_hallucination: true under nemo.rails.output
    orchestrator = NeoGuardrailOrchestrator(config_path=str(CONFIGS_PATH))
    
    print("\n🔄 Initializing orchestrator with NeMo output rails...")
    await orchestrator.initialize()
    print("✅ Orchestrator initialized!")
    
    # Test cases: (user_input, response, description)
    test_cases = [
        # Likely consistent (common knowledge)
        (
            "What is the capital of Japan?",
            "The capital of Japan is Tokyo.",
            "Common knowledge - should be consistent",
        ),
        # Likely consistent
        (
            "Who wrote Hamlet?",
            "William Shakespeare wrote Hamlet.",
            "Well-known fact - should be consistent",
        ),
        # Potentially inconsistent (made up facts)
        (
            "What is the population of Mars?",
            "Mars has a population of 10 million humans living in domed cities.",
            "Fabricated - likely inconsistent",
        ),
        # Mixed accuracy
        (
            "When was the first iPhone released?",
            "The first iPhone was released in 2007 by Steve Jobs.",
            "Mostly accurate - should be consistent",
        ),
    ]
    
    print("\n⚠️  Note: Hallucination detection generates alternative responses")
    print("   and checks for consistency. Results may vary between runs.")
    print("\n" + "-" * 60)
    
    for user_input, response, description in test_cases:
        print(f"\n👤 User: \"{user_input}\"")
        print(f"🤖 Response: \"{response[:50]}...\"")
        print(f"   ({description})")
        
        # Pass user_input in context metadata
        # Also provide evidence for fact-checking if available
        context = GuardrailContext(
            metadata={
                "user_input": user_input,
                "evidence": "",  # No evidence for hallucination-focused test
            }
        )
        
        # Run all output guardrails via orchestrator
        result = await orchestrator.guard_output(
            text=response,
            context=context,
        )
        
        # Overall result
        status = "✅ PASSED" if result.passed else "❌ FAILED"
        print(f"   {status} (risk: {result.max_risk_score:.2f})")
        
        # Individual guardrail results
        for r in result.results:
            check_status = "✓" if r.passed else "✗"
            # Shorten guardrail name for display
            short_name = r.guardrail_name.replace("nemo_self_check_", "")
            print(f"     [{check_status}] {short_name}: risk={r.risk_score:.2f}")
    
    await orchestrator.cleanup()
    print("\n✅ Orchestrator example complete!")


# =============================================================================
# Main
# =============================================================================

async def main():
    """Run all hallucination detection examples."""
    print("\n" + "=" * 60)
    print("NeMo Hallucination Detection Examples")
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
        await example_basic_hallucination()
        # await example_guardrail_interface()
        # await example_warning_mode()
        # await example_orchestrator_integration()
        
        print("\n" + "=" * 60)
        print("All examples completed!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
