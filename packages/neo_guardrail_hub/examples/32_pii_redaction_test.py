#!/usr/bin/env python
"""
Example 32: PII Output Redaction Test

This example tests PII redaction on LLM outputs using the orchestrator.

What it does:
- Detects PII in simulated LLM responses
- Replaces sensitive data with [REDACTED] placeholders
- Shows before/after comparison of sanitized text
- Tests 10 real-world scenarios (customer service, medical, payments, etc.)
- Allows responses to pass through (doesn't block, just sanitizes)

How to run:
    python examples/32_pii_redaction_test.py
    
Requirements:
    pip install neo-guardrail-hub[llm-guard]
    
Configuration:
    Enable pii_redaction in configs/default.yaml with on_fail: sanitize
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
    print_section("PII OUTPUT REDACTION TEST SUITE")
    print("Testing End-to-End: LLM Output → PII Detection → Sanitization → User")
    
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
    
    # ========================================================================
    # REDACTION TEST SCENARIOS
    # ========================================================================
    print_section("REDACTION TEST SCENARIOS")
    print("Simulating LLM outputs with PII → Redaction → Sanitized output")
    
    redaction_tests = [
        {
            "name": "Customer Service Response",
            "output": "Hi John Smith! I've found your account. Your email john.doe@example.com is verified and phone 555-123-4567 is on file.",
            "expected_redactions": ["PERSON", "EMAIL_ADDRESS", "PHONE_NUMBER"]
        },
        {
            "name": "Medical Record Summary",
            "output": "Patient SSN 123-45-6789 has NHS number 401 023 2137. Contact at patient@hospital.com",
            "expected_redactions": ["US_SSN", "UK_NHS", "EMAIL_ADDRESS"]
        },
        {
            "name": "Payment Confirmation",
            "output": "Payment processed via Visa card 4532-1234-5678-9010 with CVV: 123. Receipt sent to billing@company.com",
            "expected_redactions": ["CREDIT_CARD", "CREDITCARDCVV", "EMAIL_ADDRESS"]
        },
        {
            "name": "Account Setup",
            "output": "Account created for user@example.com. Your temporary Password: SecureP@ss123 and PIN Code: 9876",
            "expected_redactions": ["EMAIL_ADDRESS", "PASSWORD", "PIN"]
        },
        {
            "name": "International Customer",
            "output": "Korean customer ID: 900101-1234567, Medicare: 2123 45670 1, Amex: 3782 822463 10005",
            "expected_redactions": ["KR_RRN", "AU_MEDICARE", "CREDIT_CARD"]
        },
        {
            "name": "Financial Advisory",
            "output": "Your bank account ending in 1234-5678-9012 has a balance of $5000. Contact advisor@bank.com",
            "expected_redactions": ["ACCOUNTNUMBER", "EMAIL_ADDRESS"]
        },
        {
            "name": "Healthcare Portal",
            "output": "Dr. Jane Wilson reviewed your medical license #MD123456. Call 555-987-6543 for results.",
            "expected_redactions": ["PERSON", "MEDICAL_LICENSE", "PHONE_NUMBER"]
        },
        {
            "name": "Vehicle Registration",
            "output": "Your vehicle VIN: 1HGBH41JXMN109186 and plate number ABC-1234 are registered.",
            "expected_redactions": ["VEHICLEVIN", "VEHICLEVRM"]
        },
        {
            "name": "Employee Onboarding",
            "output": "Welcome! Your employee ID is EMP-98765 and temporary password is TempP@ss456. Email: hr@company.com",
            "expected_redactions": ["ACCOUNTNUMBER", "PASSWORD", "EMAIL_ADDRESS"]
        },
        {
            "name": "Multi-National Data",
            "output": "Italian client fiscal code: RSSMRA85T10A562S, Spanish NIF: 12345678Z, UK NHS: 401 023 2137",
            "expected_redactions": ["IT_FISCAL_CODE", "ES_NIF", "UK_NHS"]
        }
    ]
    
    redaction_passed = 0
    total_redaction_tests = len(redaction_tests)
    total_latency = 0
    
    for idx, test in enumerate(redaction_tests, 1):
        print(f"\n{'─' * 80}")
        print(f"Test {idx}/{total_redaction_tests}: {test['name']}")
        print(f"{'─' * 80}")
        print(f"Original Output ({len(test['output'])} chars):")
        print(f'  "{test["output"][:80]}{"..." if len(test["output"]) > 80 else ""}"')
        
        # Run output guardrail
        result = await orchestrator.guard_output(test["output"])
        
        print(f"\n📊 Redaction Result:")
        print(f"  • Passed: {result.passed} (always True for redaction)")
        print(f"  • Risk Score: {result.max_risk_score:.2f}")
        print(f"  • Text Modified: {result.final_text != test['output']}")
        print(f"  • Latency: {result.total_latency_ms:.2f}ms")
        
        total_latency += result.total_latency_ms
        
        if result.final_text != test["output"]:
            # Count redactions
            redacted_count = result.final_text.count('[REDACTED')
            
            print(f"\n✅ SUCCESS - PII Sanitized")
            print(f"  • Entities Redacted: {redacted_count}")
            print(f"  • Expected Types: {', '.join(test['expected_redactions'])}")
            print(f"\nSanitized Output ({len(result.final_text)} chars):")
            print(f'  "{result.final_text[:80]}{"..." if len(result.final_text) > 80 else ""}"')
            
            # Show a sample of what changed
            if redacted_count > 0:
                print(f"\n  💡 Protection Applied: {redacted_count} sensitive values replaced with placeholders")
            
            redaction_passed += 1
        else:
            print(f"\n❌ FAILED - No redaction occurred")
            print(f"  • Expected to redact: {', '.join(test['expected_redactions'])}")
            print(f"  • Output unchanged - PII may be exposed")
    
    # ========================================================================
    # TEST SUMMARY
    # ========================================================================
    print_section("REDACTION TEST SUMMARY")
    
    success_rate = (redaction_passed / total_redaction_tests) * 100
    avg_latency = total_latency / total_redaction_tests if total_redaction_tests > 0 else 0
    
    print(f"✅ Success Rate: {success_rate:.0f}% ({redaction_passed}/{total_redaction_tests} tests)")
    print(f"⚡ Average Latency: {avg_latency:.2f}ms per redaction")
    print(f"📊 Total Processing Time: {total_latency:.2f}ms")
    print()
    
    if success_rate == 100:
        print("🎉 PERFECT! All LLM outputs successfully sanitized")
    elif success_rate >= 80:
        print("✓ GOOD - Most outputs protected, review failures")
    else:
        print("⚠️  WARNING - Low success rate, check configuration")
    
    print()
    print("KEY FEATURES:")
    print("• Detection: Same 3-layer system as input (Deberta + Presidio + Regex)")
    print("• Action: Sanitize (not block) - responses still flow to users")
    print("• Coverage: All 74 entity types supported")
    print("• Executor Fix: Properly returns sanitized_text for passed=True results")
    print()
    print("CONFIGURATION REQUIRED:")
    print("  pii_redaction:")
    print("    enabled: true")
    print("    on_fail: sanitize  # ← Critical: must be 'sanitize' not 'block'")
    print()
    print("USE CASE:")
    print("Prevents PII leakage in LLM responses while maintaining user experience.")
    print("Ideal for customer service, healthcare, financial applications.")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())
