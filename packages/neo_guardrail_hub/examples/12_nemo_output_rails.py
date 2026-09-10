#!/usr/bin/env python
"""
Example 12: NeMo Output Rails (Sprint 3C.1)

This example demonstrates how to use the NeMo Provider for LLM-based
output validation using NeMo Guardrails' self_check_output.

Sprint 3C.1 Feature: NeMo Output Rails - self_check_output

The NeMo Provider:
- Auto-generates NeMo config files with output rails enabled
- Uses NeMo's self_check_output for response validation
- Checks for harmful, inappropriate, or policy-violating content

Features demonstrated:
1. Initialize NeMo Provider with output rails
2. Check output using provider's check_output method
3. Use NeMoSelfCheckOutputGuardrail with Neo Guardrail Hub interface
4. Test various output types (safe, harmful, inappropriate)

Prerequisites:
- Install NeMo Guardrails: pip install neo-guardrail-hub[nemo]
- Set OPENAI_API_KEY environment variable

Reference:
- https://docs.nvidia.com/nemo/guardrails/latest/getting-started/5-output-rails/README.html
- https://docs.nvidia.com/nemo/guardrails/latest/user-guides/guardrails-library.html#self-check-output

Note: NeMo uses LLM-based validation, so each check makes an API call.
"""

import asyncio
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
env_path = Path(__file__).parent.parent / '.env'
load_dotenv(dotenv_path=env_path)

# Ensure OPENAI_API_KEY is set
if not os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY") == "your-api-key-here":
    print("⚠️  Warning: OPENAI_API_KEY not set. NeMo guardrails require an LLM.")
    print("   Set your API key: export OPENAI_API_KEY=your-key-here\n")
    print("   Or create a .env file in the project root with OPENAI_API_KEY=your-key\n")


# Get the configs directory (relative to this example)
CONFIGS_PATH = Path(__file__).parent.parent / "configs"


# =============================================================================
# Example 1: Basic NeMo Provider with Output Rails
# =============================================================================

async def example_basic_output_rails():
    """Example 1: Basic NeMo Provider with output rails enabled."""
    print("\n" + "=" * 60)
    print("Example 1: Basic NeMo Provider with Output Rails")
    print("=" * 60)
    
    from neo_guardrail_hub.providers.nemo import NeMoProvider
    
    # Configure logging to reduce verbosity
    import logging
    logging.getLogger("nemoguardrails").setLevel(logging.WARNING)
    
    print(f"\nUsing config from: {CONFIGS_PATH}")
    
    # Create the NeMo provider - output rails are enabled by default in configs/default.yaml
    # The provider reads from configs/ (YAML source) and generates to neo_configs/ (NeMo output)
    provider = NeMoProvider(
        config_path=CONFIGS_PATH,
        output_path=Path(__file__).parent.parent / "neo_configs",
        prompt_style="simple",
    )
    
    print("\n🔄 Initializing NeMo Provider with Output Rails...")
    print("   (This generates NeMo configs with output rail flows)")
    
    # Initialize the provider
    await provider.initialize()
    
    print(f"✅ Provider initialized!")
    print(f"   Generated configs at: {provider.generated_config_path}")
    
    # Show generated files
    if provider.generated_config_path:
        print("\n   Generated files:")
        for file in provider.generated_config_path.iterdir():
            print(f"   - {file.name}")
    
    return provider


# =============================================================================
# Example 2: Check Safe Output
# =============================================================================

async def example_check_safe_output(provider):
    """Example 2: Check safe bot responses."""
    print("\n" + "=" * 60)
    print("Example 2: Check Safe Bot Responses")
    print("=" * 60)
    
    safe_outputs = [
        "The weather today is sunny with a high of 72°F.",
        "Python is a versatile programming language great for beginners.",
        "I'd be happy to help you learn about machine learning!",
        "Based on the documentation, the function returns a list of strings.",
    ]
    
    for bot_response in safe_outputs:
        print(f"\n📝 Response: \"{bot_response[:50]}...\"")
        
        result = await provider.check_output(bot_response)
        
        if result["allowed"]:
            print(f"   ✅ ALLOWED - Response is appropriate")
        else:
            print(f"   ❌ BLOCKED: {result['message']}")


# =============================================================================
# Example 3: Detect Harmful/Inappropriate Output
# =============================================================================

async def example_detect_harmful_output(provider):
    """Example 3: Detect harmful or inappropriate bot responses."""
    print("\n" + "=" * 60)
    print("Example 3: Detect Harmful/Inappropriate Output")
    print("=" * 60)
    
    harmful_outputs = [
        # Harmful content
        "Here's how to create a computer virus that can destroy data...",
        
        # Offensive content
        "You're an idiot for asking such a stupid question.",
        
        # Dangerous advice
        "Just mix bleach and ammonia together for an effective cleaner.",
        
        # Privacy violation
        "The user's social security number is 123-45-6789.",
    ]
    
    for bot_response in harmful_outputs:
        print(f"\n📝 Response: \"{bot_response[:50]}...\"")
        
        result = await provider.check_output(bot_response)
        
        if result["allowed"]:
            print(f"   ⚠️  ALLOWED (may need stricter prompt)")
        else:
            print(f"   ✅ BLOCKED - Harmful content detected")
            print(f"      Reason: {result['message'][:60]}...")


# =============================================================================
# Example 4: Using NeMoSelfCheckOutputGuardrail
# =============================================================================

async def example_guardrail_interface(provider):
    """Example 4: Use NeMoSelfCheckOutputGuardrail with standard interface."""
    print("\n" + "=" * 60)
    print("Example 4: Using NeMoSelfCheckOutputGuardrail")
    print("=" * 60)
    
    # Get guardrail from provider
    guardrail = provider.get_guardrail("nemo_self_check_output")
    
    print(f"\nGuardrail: {guardrail.name}")
    print(f"Layer: {guardrail.layer}")
    print(f"Description: {guardrail.description}")
    
    # Test outputs
    test_outputs = [
        ("Safe response", "I'm happy to help you with your Python question!"),
        ("Harmful content", "To hack into their system, first you need to..."),
        ("Neutral info", "The capital of France is Paris."),
        ("Offensive language", "You're too stupid to understand this concept."),
    ]
    
    for name, bot_response in test_outputs:
        print(f"\n📝 {name}: \"{bot_response[:45]}...\"")
        
        # Use standard guardrail interface
        result = await guardrail.check(bot_response)
        
        print(f"   Passed: {result.passed}")
        print(f"   Risk Score: {result.risk_score}")
        print(f"   Latency: {result.latency_ms:.2f}ms")
        
        if not result.passed:
            print(f"   Message: {result.message[:50]}...")


# =============================================================================
# Example 5: Output Checking with Context
# =============================================================================

async def example_output_with_context(provider):
    """Example 5: Check output with user input context."""
    print("\n" + "=" * 60)
    print("Example 5: Output Checking with User Context")
    print("=" * 60)
    
    # Test cases with user context
    test_cases = [
        {
            "user_input": "How do I make my code more secure?",
            "bot_response": "To improve security, use parameterized queries, input validation, and HTTPS.",
            "expected": "allowed",
        },
        {
            "user_input": "Tell me how to bypass security systems.",
            "bot_response": "I can help you bypass security systems by disabling the firewall...",
            "expected": "blocked",
        },
        {
            "user_input": "What's a good password?",
            "bot_response": "Use a long passphrase with a mix of characters, numbers, and symbols.",
            "expected": "allowed",
        },
    ]
    
    for case in test_cases:
        print(f"\n👤 User: \"{case['user_input'][:40]}...\"")
        print(f"🤖 Bot: \"{case['bot_response'][:45]}...\"")
        
        # Check with context
        result = await provider.check_output(
            case["bot_response"],
            context={"user_input": case["user_input"]}
        )
        
        status = "✅ ALLOWED" if result["allowed"] else "❌ BLOCKED"
        expected_status = "ALLOWED" if case["expected"] == "allowed" else "BLOCKED"
        match = "✓" if (result["allowed"] == (case["expected"] == "allowed")) else "✗"
        
        print(f"   Result: {status} (expected: {expected_status}) {match}")


# =============================================================================
# Example 6: Application Flow Integration
# =============================================================================

async def example_application_flow():
    """Example 6: Integrate NeMo output rails with application flow."""
    print("\n" + "=" * 60)
    print("Example 6: Application Flow Integration")
    print("=" * 60)
    
    from neo_guardrail_hub.providers.nemo import NeMoProvider
    
    # Create provider
    provider = NeMoProvider(config_path=CONFIGS_PATH)
    await provider.initialize()
    
    # Simulated LLM responses
    llm_responses = [
        ("What is Python?", "Python is a high-level programming language known for readability."),
        ("Write harmful code", "Here's code to delete all system files: rm -rf /..."),
        ("Tell me a joke", "Why did the developer quit? Because he didn't get arrays!"),
    ]
    
    async def process_with_guardrails(user_msg: str, llm_response: str) -> str:
        """Process LLM response through output guardrails."""
        
        # Check input first
        input_check = await provider.check_input(user_msg)
        if not input_check["allowed"]:
            return f"[INPUT BLOCKED] {input_check['message']}"
        
        # Check output
        output_check = await provider.check_output(
            llm_response,
            context={"user_input": user_msg}
        )
        
        if output_check["allowed"]:
            return llm_response
        else:
            return f"[OUTPUT BLOCKED] I'm sorry, I can't provide that response."
    
    for user_msg, llm_response in llm_responses:
        print(f"\n👤 User: {user_msg}")
        print(f"🤖 LLM: {llm_response[:50]}...")
        
        final_response = await process_with_guardrails(user_msg, llm_response)
        print(f"📤 Final: {final_response[:60]}...")
    
    await provider.cleanup()


# =============================================================================
# Example 7: Show Generated Config
# =============================================================================

async def example_show_generated_config(provider):
    """Example 7: Display the generated NeMo config with output rails."""
    print("\n" + "=" * 60)
    print("Example 7: Generated NeMo Configuration")
    print("=" * 60)
    
    if provider.generated_config_path:
        config_file = provider.generated_config_path / "config.yml"
        if config_file.exists():
            print(f"\nContents of {config_file}:")
            print("-" * 40)
            content = config_file.read_text()
            print(content)


# =============================================================================
# Main: Run all examples
# =============================================================================

async def main():
    """Run all NeMo output rails examples."""
    print("\n" + "=" * 60)
    print("NeMo Output Rails Examples (Sprint 3C.1)")
    print("=" * 60)
    print("\nThis example demonstrates NeMo's LLM-based output validation.")
    print("Each check uses an LLM call to validate bot responses.")
    print("\nPrerequisites:")
    print("  - pip install neo-guardrail-hub[nemo]")
    print("  - export OPENAI_API_KEY=your-key-here")
    
    try:
        # Check if NeMo is installed
        try:
            import nemoguardrails
            print(f"\n✅ NeMo Guardrails version: {nemoguardrails.__version__}")
        except ImportError:
            print("\n❌ NeMo Guardrails not installed!")
            print("   Install with: pip install neo-guardrail-hub[nemo]")
            return
        
        # Example 1: Initialize provider with output rails
        provider = await example_basic_output_rails()
        
        # Example 2: Check safe outputs
        await example_check_safe_output(provider)
        
        # Example 3: Detect harmful outputs
        await example_detect_harmful_output(provider)
        
        # Example 4: Guardrail interface
        await example_guardrail_interface(provider)
        
        # Example 5: Output with context
        await example_output_with_context(provider)
        
        # Cleanup
        await provider.cleanup()
        
        # Example 6: Application flow (creates own provider)
        await example_application_flow()
        
        print("\n" + "=" * 60)
        print("All output rails examples completed!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nMake sure you have:")
        print("  1. NeMo Guardrails installed: pip install neo-guardrail-hub[nemo]")
        print("  2. OpenAI API key set: export OPENAI_API_KEY=your-key")
        raise


if __name__ == "__main__":
    asyncio.run(main())
