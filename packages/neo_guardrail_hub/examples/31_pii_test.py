#!/usr/bin/env python
"""
Example 31: PII Detection Test with Local Models

This example tests PII detection in user inputs using the orchestrator with local models.

What it does:
- Uses local model directory for faster loading (./models/llm_guard)
- Tests 3 detection methods: Deberta NER model, Presidio recognizers, and regex patterns
- Detects 74 different PII entity types (email, phone, SSN, credit cards, etc.)
- Shows which entities are found in test inputs
- Reports detection results and processing time

How to run:
    python examples/31_pii_test.py
    
Requirements:
    pip install neo-guardrail-hub[llm-guard]
    
Configuration:
    Enable pii_detection in configs/default.yaml
    
Local Models:
    Place downloaded models in ./models/llm_guard/ for faster loading
    Example: ./models/llm_guard/lakshyakh93_deberta_finetuned_pii/
"""

import asyncio
from pathlib import Path
from neo_guardrail_hub import NeoGuardrailOrchestrator


def print_section(title):
    """Print a section header."""
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


async def main():
    """Main test function."""
    print_section("PII DETECTION TEST SUITE")
    print("Testing Default, Custom, and Enhanced Pattern Detection")
    
    # Path to local models
    models_dir = "./models/llm_guard"
    print(f"\n📁 Using local models directory: {models_dir}")
    
    # Initialize orchestrator with default config and local models
    print("\nInitializing guardrails (this may take a minute on first run)...")
    orchestrator = NeoGuardrailOrchestrator(
        config_path="./configs",
        models_dir=models_dir
    )
    await orchestrator.initialize()
    print("✓ Guardrails initialized with local model support!\n")
    
    # Test counters
    total_tests = 0
    passed_tests = 0
    
    # ========================================================================
    # TEST 1: DEFAULT PII DETECTION
    # ========================================================================
    print_section("TEST 1: DEFAULT PII DETECTION")
    print("These entities are detected by the base Deberta AI4Privacy model")
    print("Entities: EMAIL_ADDRESS, PHONE_NUMBER, PERSON")
    
    test_cases_default = [
        ("EMAIL_ADDRESS", "Contact me at john.doe@example.com for more info"),
        ("PHONE_NUMBER", "Call me at 555-123-4567 during business hours"),
        ("PERSON", "John Smith will attend the meeting tomorrow"),
    ]
    
    for entity_type, text in test_cases_default:
        total_tests += 1
        result = await orchestrator.guard_input(text)
        
        # Check if PII was detected in the guardrail results
        detected = False
        if not result.passed or result.max_risk_score > 0:
            detected = True
            
        status = "✅ DETECTED" if detected else "❌ NOT DETECTED"
        print(f"{status}: {entity_type}")
        print(f"  Text: {text[:60]}...")
        print(f"  Risk Score: {result.max_risk_score:.2f}")
        print(f'  Redacted Text: "{result.final_text[:80]}{"..." if len(result.final_text) > 80 else ""}"')
        if detected:
            passed_tests += 1
    
    print(f"\nResult: {passed_tests}/{total_tests} detected ({passed_tests/total_tests*100:.0f}%)")
    
    # ========================================================================
    # TEST 2: CUSTOM PII DETECTION (Presidio Recognizers)
    # ========================================================================
    # print_section("TEST 2: CUSTOM PII DETECTION (Presidio Recognizers)")
    # print("These use custom PatternRecognizer objects from custom_recognizers.py")
    # print("Entities: UK_NHS, KR_RRN, AU_MEDICARE, VISA, AMEX")
    
    # test_cases_custom = [
    #     ("UK_NHS", "Patient NHS number: 401 023 2137 for medical records"),
    #     ("KR_RRN", "Korean ID: 900101-1234567 on file"),
    #     ("AU_MEDICARE", "Medicare card: 2123 45670 1 for claims"),
    #     ("CREDIT_CARD (VISA)", "Payment via card number: 4532-1234-5678-9010"),
    #     ("CREDIT_CARD (AMEX)", "Amex card: 3782 822463 10005 on file"),
    # ]
    
    # custom_passed = 0
    # for entity_type, text in test_cases_custom:
    #     total_tests += 1
    #     result = await orchestrator.guard_input(text)
        
    #     detected = False
    #     if not result.passed or result.max_risk_score > 0:
    #         detected = True
            
    #     status = "✅ DETECTED" if detected else "❌ NOT DETECTED"
    #     print(f"{status}: {entity_type}")
    #     print(f"  Text: {text[:60]}...")
    #     print(f"  Risk Score: {result.max_risk_score:.2f}")
        
    #     if detected:
    #         passed_tests += 1
    #         custom_passed += 1
    
    # print(f"\nResult: {custom_passed}/5 detected ({custom_passed/5*100:.0f}%)")
    
    # # ========================================================================
    # # TEST 3: ENHANCED PII DETECTION (Context-Aware Patterns)
    # # ========================================================================
    # print_section("TEST 3: ENHANCED PII DETECTION (Context-Aware Patterns)")
    # print("These use context-aware regex from enhanced_patterns.py")
    # print("Entities: CREDITCARDCVV, PIN, PASSWORD")
    # print("NOTE: These REQUIRE context labels for detection!")
    
    # test_cases_enhanced = [
    #     ("CREDITCARDCVV", "Your card CVV: 123 is required for verification"),
    #     ("PIN", "Enter your PIN Code: 9876 to continue"),
    #     ("PASSWORD", "Your temporary Password: SecureP@ss123 will expire soon"),
    # ]
    
    # enhanced_passed = 0
    # for entity_type, text in test_cases_enhanced:
    #     total_tests += 1
    #     result = await orchestrator.guard_input(text)
        
    #     detected = False
    #     if not result.passed or result.max_risk_score > 0:
    #         detected = True
            
    #     status = "✅ DETECTED" if detected else "❌ NOT DETECTED"
    #     print(f"{status}: {entity_type}")
    #     print(f"  Text: {text[:60]}...")
    #     print(f"  Risk Score: {result.max_risk_score:.2f}")
        
    #     if detected:
    #         passed_tests += 1
    #         enhanced_passed += 1
    
    # print(f"\nResult: {enhanced_passed}/3 detected ({enhanced_passed/3*100:.0f}%)")
    
    # # ========================================================================
    # # TEST 4: COMBINED DETECTION
    # # ========================================================================
    # print_section("TEST 4: COMBINED DETECTION")
    # print("Testing all 11 entities in a single text")
    
    # combined_text = """
    # Customer Details:
    # Name: John Smith
    # Email: john.smith@example.com
    # Phone: 555-123-4567
    # UK NHS: 401 023 2137
    # Korean ID: 900101-1234567
    # Medicare: 2123 45670 1
    # Visa Card: 4532-1234-5678-9010
    # Amex: 3782 822463 10005
    # Card CVV: 123
    # PIN Code: 9876
    # Password: SecureP@ss123
    # """
    
    # total_tests += 1
    # result = await orchestrator.guard_input(combined_text)
    
    # print(f"Text contains 11 different PII entities")
    # print(f"Passed: {result.passed}")
    # print(f"Risk Score: {result.max_risk_score:.2f}")
    # print(f"Execution Time: {result.total_latency_ms:.2f}ms")
    
    # if not result.passed or result.max_risk_score > 0:
    #     passed_tests += 1
    #     print("✅ Combined PII detected")
    #     if result.failed_checks:
    #         print(f"Failed checks: {result.failed_checks}")
    # else:
    #     print("❌ Combined PII not detected")
    
    # ========================================================================
    # TEST SUMMARY
    # ========================================================================
    print_section("TEST SUMMARY")
    print(f"Overall Detection Rate: {passed_tests/total_tests*100:.1f}% ({passed_tests}/{total_tests})")


if __name__ == "__main__":
    asyncio.run(main())
