#!/usr/bin/env python3
"""
Example 23: LLM Guard Output Scanners - Complete Usage Patterns
================================================================

This example demonstrates all LLM Guard-based output scanners with four usage patterns:
- Direct usage: Using guardrails directly
- Orchestrator: Using NeoGuardrailOrchestrator with config
- Decorator: Using @guarded decorator
- Context Manager: Using GuardSession

Demonstrates Phase 3 LLM Guard output scanners:
- BiasOutputGuardrail: Detect biased statements
- CodeDetectionOutputGuardrail: Detect programming languages in output
- BanCompetitorsOutputGuardrail: Block competitor mentions in output
- GibberishOutputGuardrail: Detect nonsensical/incoherent output
- JSONValidationOutputGuardrail: Validate JSON structure in output
- LanguageOutputGuardrail: Validate output language
- LanguageSameOutputGuardrail: Check input/output language consistency
- MaliciousURLsOutputGuardrail: Detect harmful URLs in output
- ReadingTimeOutputGuardrail: Enforce reading time limits
- RegexOutputGuardrail: Custom regex pattern matching for output
- SensitiveDataOutputGuardrail: Detect/redact sensitive data in output
- SentimentOutputGuardrail: Analyze output sentiment
- URLReachabilityOutputGuardrail: Check URL accessibility

Run this example:
    cd /path/to/neo_guardrail_hub
    source .venv312/bin/activate
    python examples/23_llm_guard_output_scanners.py
"""

import asyncio
import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import guardrails for direct usage
from neo_guardrail_hub.guardrails.output import (
    BiasOutputGuardrail,
    JSONValidationOutputGuardrail,
    RegexOutputGuardrail,
    SentimentOutputGuardrail,
    ReadingTimeOutputGuardrail,
    GibberishOutputGuardrail,
    LanguageSameOutputGuardrail,
)

# Import models for context
from neo_guardrail_hub.core.models import GuardrailContext

# Import orchestrator, decorator, and context manager
from neo_guardrail_hub import (
    NeoGuardrailOrchestrator,
    guarded,
    GuardSession,
    GuardrailError,
)


def print_separator(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}\n")


def print_result(result, prefix=""):
    status = "PASSED" if result.passed else "BLOCKED"
    emoji = "✓" if result.passed else "✗"
    print(f"{prefix}[{emoji}] {status} - Risk: {result.risk_score:.2f}")
    if result.message:
        print(f"{prefix}  Msg: {result.message}")
    print()


async def simulate_llm_response(prompt, response_type="normal"):
    """Simulate an LLM generating different types of responses."""
    responses = {
        "normal": "Based on your question, here's my recommendation: Consider diversifying your portfolio with a mix of stocks and bonds.",
        "biased": "All people from certain regions are bad at financial planning and shouldn't be trusted with investments.",
        "json": '{"recommendation": "Buy index funds", "risk_level": "low", "expected_return": 0.08}',
        "invalid_json": '{"recommendation": "Buy",, "risk_level": }',
        "gibberish": "asdfkj lkjsdf investment lkjlk money sdfjslkdfj dollars skdjf",
        "negative": "Your portfolio is terrible, horrible, and absolutely worthless!",
        "positive": "I'm so happy to help you! This is a wonderful investment opportunity!",
        "with_ssn": "Your account details show SSN: 123-45-6789 and we recommend investing.",
        "long": " ".join(["This is detailed financial analysis."] * 100),
        "with_code": "Here's how to calculate returns:\ndef calc_returns(principal, rate):\n    return principal * rate",
    }
    return responses.get(response_type, responses["normal"])


# ============================================================================
# Pattern 1: DIRECT USAGE
# ============================================================================

async def demo_direct_usage():
    print_separator("PATTERN 1: DIRECT USAGE")
    print("Using output guardrails directly - full control over initialization.\n")
    
    # Create a reusable context with prompt
    # LLM Guard output scanners need both prompt and output
    def make_context(prompt: str) -> GuardrailContext:
        """Create context with prompt for output guardrails."""
        ctx = GuardrailContext(
            agent_id="example_agent",
            session_id="direct_usage_session",
        )
        # Set prompt as extra attribute (extra='allow' in model)
        ctx.prompt = prompt
        return ctx
    
    # 1. Sentiment Output Guardrail
    print("1. Sentiment Output Guardrail")
    sentiment = SentimentOutputGuardrail({"threshold": -0.5})
    await sentiment.initialize()
    
    prompt = "Give me feedback on my investment idea"
    ctx = make_context(prompt)
    
    tests = [
        ("Positive", await simulate_llm_response("", "positive")),
        ("Negative", await simulate_llm_response("", "negative")),
    ]
    
    for name, text in tests:
        result = await sentiment.check(text, context=ctx)
        print(f"   {name}: {text[:50]}...")
        print_result(result, "   ")
    
    # 2. JSON Validation Guardrail
    print("2. JSON Validation Guardrail")
    json_guard = JSONValidationOutputGuardrail({"required_elements": 1})
    await json_guard.initialize()
    
    prompt = "Return user data as JSON"
    ctx = make_context(prompt)
    
    tests = [
        ("Valid JSON", await simulate_llm_response("", "json")),
        ("Normal Text", await simulate_llm_response("", "normal")),
    ]
    
    for name, text in tests:
        result = await json_guard.check(text, context=ctx)
        print(f"   {name}: {text[:50]}...")
        print_result(result, "   ")
    
    # 3. Regex Output Guardrail (SSN detection)
    print("3. Regex Output Guardrail (SSN Detection)")
    regex_guard = RegexOutputGuardrail({
        "patterns": [r"\b\d{3}-\d{2}-\d{4}\b"],
        "is_blocked": True,
        "redact": True
    })
    await regex_guard.initialize()
    
    prompt = "Show me my account details"
    ctx = make_context(prompt)
    
    tests = [
        ("Clean", await simulate_llm_response("", "normal")),
        ("With SSN", await simulate_llm_response("", "with_ssn")),
    ]
    
    for name, text in tests:
        result = await regex_guard.check(text, context=ctx)
        print(f"   {name}: {text[:50]}...")
        print_result(result, "   ")
        if result.sanitized_text:
            print(f"   Sanitized: {result.sanitized_text[:50]}...")
    
    # 4. Reading Time Guardrail
    print("4. Reading Time Guardrail")
    reading_guard = ReadingTimeOutputGuardrail({"max_time": 0.5, "truncate": False})
    await reading_guard.initialize()
    
    prompt = "Give me a brief summary"
    ctx = make_context(prompt)
    
    tests = [
        ("Short", await simulate_llm_response("", "normal")),
        ("Long", await simulate_llm_response("", "long")),
    ]
    
    for name, text in tests:
        result = await reading_guard.check(text, context=ctx)
        word_count = len(text.split())
        print(f"   {name}: {word_count} words")
        print_result(result, "   ")
    
    # 5. Gibberish Detection
    print("5. Gibberish Detection Guardrail")
    gibberish_guard = GibberishOutputGuardrail({"threshold": 0.5})
    await gibberish_guard.initialize()
    
    prompt = "Explain investment strategies"
    ctx = make_context(prompt)
    
    tests = [
        ("Coherent", await simulate_llm_response("", "normal")),
        ("Gibberish", await simulate_llm_response("", "gibberish")),
    ]
    
    for name, text in tests:
        result = await gibberish_guard.check(text, context=ctx)
        print(f"   {name}: {text[:50]}...")
        print_result(result, "   ")
    
    # 6. Language Same Guardrail (NEW - demonstrates prompt-dependent check)
    print("6. Language Same Guardrail (Prompt-dependent)")
    lang_same = LanguageSameOutputGuardrail({"threshold": 0.5})
    await lang_same.initialize()
    
    # English prompt with English response
    prompt_en = "What are the best investment strategies?"
    ctx_en = make_context(prompt_en)
    response_en = "Consider diversifying your portfolio with stocks and bonds."
    
    result = await lang_same.check(response_en, context=ctx_en)
    print(f"   English prompt + English response:")
    print(f"     Prompt: {prompt_en[:40]}...")
    print(f"     Response: {response_en[:40]}...")
    print_result(result, "   ")
    
    # English prompt with Spanish response (should detect mismatch)
    prompt_en2 = "What are the best investment strategies?"
    ctx_en2 = make_context(prompt_en2)
    response_es = "Considere diversificar su cartera con acciones y bonos."
    
    result = await lang_same.check(response_es, context=ctx_en2)
    print(f"   English prompt + Spanish response:")
    print(f"     Prompt: {prompt_en2[:40]}...")
    print(f"     Response: {response_es[:40]}...")
    print_result(result, "   ")


# ============================================================================
# Pattern 2: ORCHESTRATOR
# ============================================================================

async def demo_orchestrator_usage():
    print_separator("PATTERN 2: ORCHESTRATOR")
    print("Using NeoGuardrailOrchestrator - configuration-based.\n")
    print("NOTE: Output scanners require prompt context for some checks.\n")
    
    orchestrator = NeoGuardrailOrchestrator(config_path="./configs")
    print("Initializing orchestrator...")
    await orchestrator.initialize()
    print("✓ Orchestrator ready!\n")
    
    prompt = "What are the best investment strategies?"
    
    # Create context with prompt for output guardrails
    context = GuardrailContext(
        agent_id="orchestrator_demo",
        session_id="orch_session_001",
    )
    # Set prompt as extra attribute (extra='allow' in model)
    context.prompt = prompt
    
    tests = [
        ("Clean Response", await simulate_llm_response("", "normal")),
        ("Positive Response", await simulate_llm_response("", "positive")),
        ("JSON Response", await simulate_llm_response("", "json")),
    ]
    
    for i, (name, response) in enumerate(tests, 1):
        print(f"{i}. {name}")
        print(f"   Prompt: '{prompt[:40]}...'")
        print(f"   Output: '{response[:60]}...'")
        
        # Pass context with prompt to guard_output
        result = await orchestrator.guard_output(response, context=context)
        
        status = "✓ PASSED" if result.passed else "✗ BLOCKED"
        print(f"   {status}")
        print(f"   Risk: {result.max_risk_score:.2f}")
        
        if not result.passed:
            print(f"   Failed: {result.failed_checks[:2]}")
        print()
    
    await orchestrator.cleanup()


# ============================================================================
# Pattern 3: DECORATOR
# ============================================================================

@guarded(agent_id="example_agent", guard_input=False, guard_output=True)
async def generate_with_guardrails(prompt):
    """Generate response with automatic output guardrails."""
    # Simulate LLM response
    response = await simulate_llm_response(prompt, "normal")
    return response


@guarded(
    agent_id="example_agent",
    guard_input=False,
    guard_output=True,
    raise_on_fail=False
)
async def safe_generate(prompt, response_type="normal"):
    """Generate response safely without raising exceptions."""
    response = await simulate_llm_response(prompt, response_type)
    return {"response": response}


async def demo_decorator_usage():
    print_separator("PATTERN 3: DECORATOR")
    print("Using @guarded decorator - automatic output validation.\n")
    
    tests = [
        ("Normal", "normal", True),
        ("Positive", "positive", True),
        ("JSON", "json", True),
    ]
    
    print("1. Decorator with automatic output validation:\n")
    
    for name, response_type, should_pass in tests:
        print(f"   {name} Response:")
        
        try:
            result = await generate_with_guardrails("investment advice")
            print(f"   ✓ PASSED - Output validated: {result[:40]}...")
        except GuardrailError as e:
            print(f"   ✗ BLOCKED by {e.guardrail_name}")
        print()
    
    print("2. Decorator without exceptions:\n")
    
    for name, response_type, should_pass in tests[:2]:
        print(f"   {name} Response:")
        
        result = await safe_generate("investment advice", response_type)
        
        if isinstance(result, dict) and "error" in result:
            print(f"   ✗ BLOCKED")
        else:
            print(f"   ✓ PASSED - {result.get('response', '')[:40]}...")
        print()


# ============================================================================
# Pattern 4: CONTEXT MANAGER
# ============================================================================

async def demo_context_manager_usage():
    print_separator("PATTERN 4: CONTEXT MANAGER")
    print("Using GuardSession - stateful conversations with output validation.\n")
    print("NOTE: GuardSession context carries prompt for output scanners.\n")
    
    async with GuardSession(
        agent_id="example_agent",
        session_id="demo_456",
        user_id="user_789",
        config_path="./configs",
        metadata={"source": "example_23"},
    ) as session:
        
        print("Conversation with Output Guardrails:\n")
        
        # Turn 1: Normal response
        print("Turn 1: Normal Response")
        prompt1 = "What's the best way to invest?"
        response1 = await simulate_llm_response(prompt1, "normal")
        print(f"   User: {prompt1}")
        print(f"   Bot: {response1[:60]}...")
        
        # Set prompt in session context for output guardrails
        session.context.prompt = prompt1
        
        result = await session.guard_output(response1)
        
        if result.passed:
            print(f"   ✓ Output Validated - Risk: {result.max_risk_score:.2f}")
        else:
            print(f"   ✗ BLOCKED: {result.failed_checks}")
        print()
        
        # Turn 2: JSON response
        print("Turn 2: JSON Response")
        prompt2 = "Give me a structured recommendation"
        response2 = await simulate_llm_response(prompt2, "json")
        print(f"   User: {prompt2}")
        print(f"   Bot: {response2[:60]}...")
        
        # Update prompt in session context
        session.context.prompt = prompt2
        
        result = await session.guard_output(response2)
        
        if result.passed:
            print(f"   ✓ Output Validated")
        else:
            print(f"   ✗ BLOCKED")
        print()
        
        # Turn 3: Positive response
        print("Turn 3: Positive Response")
        prompt3 = "Tell me something encouraging"
        response3 = await simulate_llm_response(prompt3, "positive")
        print(f"   User: {prompt3}")
        print(f"   Bot: {response3[:60]}...")
        
        # Update prompt in session context
        session.context.prompt = prompt3
        
        result = await session.guard_output(response3)
        
        if result.passed:
            print(f"   ✓ Output Validated")
        else:
            print(f"   ✗ BLOCKED")
        print()
        
        # Stats
        print("Session Stats:")
        print(f"   Messages: {len(session.get_history())}")
        print(f"   Checks: {len(session.get_results())}")
        print(f"   Failed: {len(session.get_failed_checks())}")


# ============================================================================
# Comparison Summary
# ============================================================================

def print_comparison():
    print_separator("PATTERN COMPARISON")

async def main():
    print("\n" + "="*60)
    print("  NEO GUARDRAIL HUB - LLM Guard Output Scanners")
    print("  Complete Usage Patterns Demo")
    print("="*60)
    
    # Run all demos
    await demo_direct_usage()
    await demo_orchestrator_usage()
    await demo_decorator_usage()
    await demo_context_manager_usage()
    
    print_comparison()
    
    print_separator("DEMO COMPLETE")
    print("All Phase 3 output scanners demonstrated successfully!")


if __name__ == "__main__":
    asyncio.run(main())
