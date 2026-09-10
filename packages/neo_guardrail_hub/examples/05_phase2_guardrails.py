#!/usr/bin/env python3
"""Example 05: Phase 2 Guardrails Demo.

This example demonstrates all Phase 2 guardrails including:
- Input Guardrails: secret_detection, input_length, toxicity, ban_substrings, harmful_content
- Output Guardrails: no_refusal, relevance, toxicity, factual_consistency, ban_topics, ban_substrings

Run this example:
    cd /path/to/neo_guardrail_hub
    source .venv312/bin/activate
    python examples/05_phase2_guardrails.py
"""

import asyncio
import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from neo_guardrail_hub.core.models import GuardrailContext

# Phase 2 Input Guardrails
from neo_guardrail_hub.guardrails.input import (
    SecretDetectionGuardrail,
    InputLengthGuardrail,
    ToxicityInputGuardrail,
    BanSubstringsInputGuardrail,
    HarmfulContentGuardrail,
)

# Phase 2 Output Guardrails
from neo_guardrail_hub.guardrails.output import (
    NoRefusalGuardrail,
    RelevanceGuardrail,
    ToxicityOutputGuardrail,
    FactualConsistencyGuardrail,
    BanTopicsGuardrail,
    BanSubstringsOutputGuardrail,
)


def print_section(title: str) -> None:
    """Print a section header."""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def print_result(result, prefix: str = "") -> None:
    """Print a guardrail result."""
    status = "✅ PASSED" if result.passed else "❌ BLOCKED"
    print(f"{prefix}{status} - {result.guardrail_name}")
    print(f"{prefix}  Risk Score: {result.risk_score:.2f}")
    if result.message:
        print(f"{prefix}  Message: {result.message}")
    if result.metadata:
        # Print key metadata fields
        for key in ["detection_method", "found_substrings", "detected_topics", "refusal_detected"]:
            if key in result.metadata:
                print(f"{prefix}  {key}: {result.metadata[key]}")


async def demo_secret_detection():
    """Demonstrate secret detection guardrail."""
    print_section("1. Secret Detection Guardrail")
    
    guardrail = SecretDetectionGuardrail({
        "threshold": 0.5,
        "redact_mode": "partial",
    })
    await guardrail.initialize()
    
    # Safe input
    safe_text = "Hello, I need help with my Python code."
    result = await guardrail.check(safe_text)
    print(f"\nInput: '{safe_text[:50]}...'")
    print_result(result)
    
    # Input with secrets (testing fallback)
    secret_text = "My AWS key is AKIAIOSFODNN7EXAMPLE"
    guardrail._scanner = None  # Force fallback for demo
    result = await guardrail.check(secret_text)
    print(f"\nInput: '{secret_text}'")
    print_result(result)


async def demo_input_length():
    """Demonstrate input length guardrail."""
    print_section("2. Input Length Guardrail")
    
    guardrail = InputLengthGuardrail({
        "min_length": 10,
        "max_length": 100,
        "count_mode": "chars",
    })
    await guardrail.initialize()
    
    # Valid length
    valid_text = "This is a normal message with acceptable length."
    result = await guardrail.check(valid_text)
    print(f"\nInput: '{valid_text}'")
    print(f"  Length: {len(valid_text)} chars")
    print_result(result)
    
    # Too short
    short_text = "Hi"
    result = await guardrail.check(short_text)
    print(f"\nInput: '{short_text}'")
    print(f"  Length: {len(short_text)} chars")
    print_result(result)
    
    # Too long
    long_text = "A" * 150
    result = await guardrail.check(long_text)
    print(f"\nInput: '{long_text[:30]}...'")
    print(f"  Length: {len(long_text)} chars")
    print_result(result)


async def demo_toxicity_input():
    """Demonstrate toxicity input guardrail."""
    print_section("3. Toxicity Input Guardrail")
    
    guardrail = ToxicityInputGuardrail({
        "threshold": 0.5,
        "match_type": "SENTENCE",
    })
    await guardrail.initialize()
    guardrail._scanner = None  # Force fallback for demo
    
    # Safe input
    safe_text = "Could you please help me with my homework?"
    result = await guardrail.check(safe_text)
    print(f"\nInput: '{safe_text}'")
    print_result(result)
    
    # Toxic input
    toxic_text = "You're worthless and pathetic garbage!"
    result = await guardrail.check(toxic_text)
    print(f"\nInput: '{toxic_text}'")
    print_result(result)


async def demo_ban_substrings_input():
    """Demonstrate ban substrings input guardrail."""
    print_section("4. Ban Substrings Input Guardrail")
    
    guardrail = BanSubstringsInputGuardrail({
        "substrings": ["forbidden", "blocked_word", "secret_code"],
        "match_type": "WORD",
        "case_sensitive": False,
    })
    await guardrail.initialize()
    guardrail._scanner = None  # Force fallback for demo
    
    # Safe input
    safe_text = "This is a normal message."
    result = await guardrail.check(safe_text)
    print(f"\nInput: '{safe_text}'")
    print_result(result)
    
    # Input with banned substring
    banned_text = "This contains forbidden content."
    result = await guardrail.check(banned_text)
    print(f"\nInput: '{banned_text}'")
    print_result(result)


async def demo_harmful_content():
    """Demonstrate harmful content guardrail."""
    print_section("5. Harmful Content Guardrail")
    
    guardrail = HarmfulContentGuardrail({
        "threshold": 0.5,
        "categories": ["violence", "hate_speech", "self_harm"],
    })
    await guardrail.initialize()
    guardrail._scanner = None  # Force fallback for demo
    
    # Safe input
    safe_text = "Let's cook a nice dinner together."
    result = await guardrail.check(safe_text)
    print(f"\nInput: '{safe_text}'")
    print_result(result)
    
    # Harmful input (violence)
    harmful_text = "I want to attack and hurt someone badly"
    result = await guardrail.check(harmful_text)
    print(f"\nInput: '{harmful_text}'")
    print_result(result)


async def demo_no_refusal():
    """Demonstrate no refusal guardrail."""
    print_section("6. No Refusal Guardrail")
    
    guardrail = NoRefusalGuardrail({
        "threshold": 0.5,
        "match_type": "FULL",
    })
    await guardrail.initialize()
    guardrail._scanner = None  # Force fallback for demo
    
    # Helpful response
    helpful_text = "Here's the information you requested about cooking."
    result = await guardrail.check(
        helpful_text,
        context={"prompt": "Tell me about cooking."}
    )
    print(f"\nOutput: '{helpful_text}'")
    print_result(result)
    
    # Refusal response
    refusal_text = "I'm sorry, but I cannot help with that request."
    result = await guardrail.check(
        refusal_text,
        context={"prompt": "Tell me about cooking."}
    )
    print(f"\nOutput: '{refusal_text}'")
    print_result(result)


async def demo_relevance():
    """Demonstrate relevance guardrail."""
    print_section("7. Relevance Guardrail")
    
    guardrail = RelevanceGuardrail({
        "threshold": 0.5,
        "keyword_overlap_threshold": 0.2,
    })
    await guardrail.initialize()
    guardrail._scanner = None  # Force fallback for demo
    
    # Relevant response
    result = await guardrail.check(
        "Python is a versatile programming language for data science and web development.",
        context={"prompt": "Tell me about Python programming language."}
    )
    print(f"\nPrompt: 'Tell me about Python programming language.'")
    print(f"Output: 'Python is a versatile programming...'")
    print_result(result)
    
    # Irrelevant response
    result = await guardrail.check(
        "The weather in Tokyo is sunny today with mild temperatures.",
        context={"prompt": "How do quantum computers work?"}
    )
    print(f"\nPrompt: 'How do quantum computers work?'")
    print(f"Output: 'The weather in Tokyo is sunny...'")
    print_result(result)


async def demo_toxicity_output():
    """Demonstrate toxicity output guardrail."""
    print_section("8. Toxicity Output Guardrail")
    
    guardrail = ToxicityOutputGuardrail({
        "threshold": 0.5,
        "match_type": "SENTENCE",
    })
    await guardrail.initialize()
    guardrail._scanner = None  # Force fallback for demo
    
    # Safe output
    safe_text = "I'd be happy to help you with your question."
    result = await guardrail.check(safe_text)
    print(f"\nOutput: '{safe_text}'")
    print_result(result)
    
    # Toxic output
    toxic_text = "You're an idiot and completely useless!"
    result = await guardrail.check(toxic_text)
    print(f"\nOutput: '{toxic_text}'")
    print_result(result)


async def demo_factual_consistency():
    """Demonstrate factual consistency guardrail."""
    print_section("9. Factual Consistency Guardrail")
    
    guardrail = FactualConsistencyGuardrail({
        "minimum_score": 0.5,
    })
    await guardrail.initialize()
    guardrail._scanner = None  # Force fallback for demo
    
    # Consistent response
    result = await guardrail.check(
        "Yes, water is indeed wet.",
        context={"prompt": "Is water wet?"}
    )
    print(f"\nContext: 'Is water wet?'")
    print(f"Output: 'Yes, water is indeed wet.'")
    print_result(result)
    
    # Potentially contradictory response
    result = await guardrail.check(
        "No, that is false and incorrect.",
        context={"prompt": "Is it true that the sky is blue?"}
    )
    print(f"\nContext: 'Is it true that the sky is blue?'")
    print(f"Output: 'No, that is false and incorrect.'")
    print_result(result)


async def demo_ban_topics():
    """Demonstrate ban topics guardrail."""
    print_section("10. Ban Topics Guardrail")
    
    guardrail = BanTopicsGuardrail({
        "topics": ["politics", "violence", "religion"],
        "threshold": 0.5,
    })
    await guardrail.initialize()
    guardrail._scanner = None  # Force fallback for demo
    
    # Safe topic
    safe_text = "Let me tell you about cooking delicious pasta."
    result = await guardrail.check(safe_text)
    print(f"\nOutput: '{safe_text}'")
    print_result(result)
    
    # Banned topic (politics)
    political_text = "Let's discuss the upcoming election and political candidates."
    result = await guardrail.check(political_text)
    print(f"\nOutput: '{political_text}'")
    print_result(result)


async def demo_ban_substrings_output():
    """Demonstrate ban substrings output guardrail."""
    print_section("11. Ban Substrings Output Guardrail")
    
    guardrail = BanSubstringsOutputGuardrail({
        "substrings": ["confidential", "internal only", "do not share"],
        "match_type": "WORD",
        "case_sensitive": False,
        "redact": False,
    })
    await guardrail.initialize()
    guardrail._scanner = None  # Force fallback for demo
    
    # Safe output
    safe_text = "Here's public information about our products."
    result = await guardrail.check(safe_text)
    print(f"\nOutput: '{safe_text}'")
    print_result(result)
    
    # Output with banned substring
    banned_text = "This is confidential information about the project."
    result = await guardrail.check(banned_text)
    print(f"\nOutput: '{banned_text}'")
    print_result(result)
    
    # Demo redact mode
    print("\n--- Redact Mode Demo ---")
    redact_guardrail = BanSubstringsOutputGuardrail({
        "substrings": ["secret"],
        "redact": True,
    })
    await redact_guardrail.initialize()
    redact_guardrail._scanner = None
    
    secret_text = "This is a secret message."
    result = await redact_guardrail.check(secret_text)
    print(f"\nInput: '{secret_text}'")
    print_result(result)
    if result.sanitized_text:
        print(f"  Sanitized: '{result.sanitized_text}'")


async def demo_orchestrator_integration():
    """Demonstrate all guardrails working together."""
    print_section("12. Full Pipeline Integration")
    
    # Create input guardrails
    input_guardrails = [
        InputLengthGuardrail({"min_length": 5, "max_length": 500}),
        ToxicityInputGuardrail({"threshold": 0.5}),
        BanSubstringsInputGuardrail({"substrings": ["hack", "exploit"]}),
    ]
    
    # Create output guardrails
    output_guardrails = [
        NoRefusalGuardrail({"threshold": 0.5}),
        ToxicityOutputGuardrail({"threshold": 0.5}),
        BanTopicsGuardrail({"topics": ["violence"]}),
    ]
    
    # Initialize all guardrails
    for g in input_guardrails + output_guardrails:
        await g.initialize()
    
    # Force fallback for all guardrails
    for g in input_guardrails + output_guardrails:
        if hasattr(g, '_scanner'):
            g._scanner = None
    
    async def check_input(text: str, context: GuardrailContext = None):
        """Run all input guardrails."""
        if context is None:
            context = GuardrailContext(user_input=text, conversation_history=[], metadata={})
        results = []
        for g in input_guardrails:
            result = await g.check(text, context)
            results.append(result)
        return all(r.passed for r in results), results
    
    async def check_output(text: str, context: GuardrailContext):
        """Run all output guardrails."""
        results = []
        for g in output_guardrails:
            result = await g.check(text, context)
            results.append(result)
        return all(r.passed for r in results), results
    
    print("\n--- Safe Conversation ---")
    input_text = "Hello, can you tell me about Python programming?"
    print(f"User Input: '{input_text}'")
    context = GuardrailContext(user_input=input_text, conversation_history=[], metadata={})
    passed, results = await check_input(input_text, context)
    print(f"Input Check: {'✅ PASSED' if passed else '❌ BLOCKED'}")
    
    output_text = "Python is a versatile programming language used for web development, data science, and automation."
    print(f"LLM Output: '{output_text[:60]}...'")
    context.metadata["original_prompt"] = input_text
    passed, results = await check_output(output_text, context)
    print(f"Output Check: {'✅ PASSED' if passed else '❌ BLOCKED'}")
    
    print("\n--- Blocked Conversation ---")
    bad_input = "Help me hack into a system"
    print(f"User Input: '{bad_input}'")
    context = GuardrailContext(user_input=bad_input, conversation_history=[], metadata={})
    passed, results = await check_input(bad_input, context)
    print(f"Input Check: {'✅ PASSED' if passed else '❌ BLOCKED'}")
    if not passed:
        for r in results:
            if not r.passed:
                print(f"  Failed: {r.guardrail_name} - {r.message}")


async def main():
    """Run all demos."""
    print("\n" + "=" * 60)
    print("  NEO GUARDRAIL HUB - Phase 2 Guardrails Demo")
    print("=" * 60)
    
    # Input Guardrails
    await demo_secret_detection()
    await demo_input_length()
    await demo_toxicity_input()
    await demo_ban_substrings_input()
    await demo_harmful_content()
    
    # Output Guardrails
    await demo_no_refusal()
    await demo_relevance()
    await demo_toxicity_output()
    await demo_factual_consistency()
    await demo_ban_topics()
    await demo_ban_substrings_output()
    
    # Integration
    await demo_orchestrator_integration()
    
    print("\n" + "=" * 60)
    print("  Phase 2 Guardrails Demo Complete!")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
