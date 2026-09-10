#!/usr/bin/env python
"""Example 30: Testing Output Guardrails with Local Models

This example demonstrates all model-based output guardrails loading models
from local disk instead of downloading from HuggingFace. Shows:
- Automatic model downloading based on enabled guardrails in config
- All model-based output guardrails with local loading support
- Orchestrator pattern with config file and models_dir
- Complete output validation workflow

Models used by output guardrails:
1. ban_competitors - guishe/nuner-v1_orgs
2. ban_topics - MoritzLaurer/deberta-v3-base-zeroshot-v2.0 (shared with factual_consistency)
3. bias - valurank/distilroberta-bias
4. code_detection - philomath-1209/programming-language-identification
5. factual_consistency - MoritzLaurer/deberta-v3-base-zeroshot-v2.0 (shared with ban_topics)
6. gibberish - madhurjindal/autonlp-Gibberish-Detector-492513457
7. language - papluca/xlm-roberta-base-language-detection
8. relevance - BAAI/bge-base-en-v1.5
9. toxicity - unitary/unbiased-toxic-roberta

Usage:
    # Step 1: Download models for enabled output guardrails
    python examples/30_output_guardrails_local_models.py --download
    
    # Step 2: Test with local models using config file
    python examples/30_output_guardrails_local_models.py --test
    
    # Step 3: Test individual guardrails (legacy)
    python examples/30_output_guardrails_local_models.py --test-individual
"""

import argparse
import asyncio
import os
import time
from pathlib import Path

from neo_guardrail_hub.core.orchestrator import NeoGuardrailOrchestrator

# Get the project root
PROJECT_ROOT = Path(__file__).parent.parent
CONFIGS_PATH = PROJECT_ROOT / "configs"
MODELS_DIR = PROJECT_ROOT / "models" / "llm_guard"
DEFAULT_CONFIG = CONFIGS_PATH / "default.yaml"


# Example outputs to test
TEST_OUTPUTS = {
    "competitor_mention": "You should try Amazon AWS instead, it's much better than our service.",
    "banned_topic": "Here's detailed information about illegal drug manufacturing and distribution.",
    "biased_content": "Women are naturally worse at mathematics than men.",
    "code_snippet": "import os\nos.system('rm -rf /')",
    "factually_inconsistent": "The capital of France is Berlin, and it has a population of 50 million.",
    "gibberish": "asdfkj aslkdfj laskjdf oiuwoei laksdjf",
    "wrong_language": "Je ne parle pas anglais, seulement français.",
    "irrelevant": "The sky is blue because of Rayleigh scattering.",
    "toxic": "You're an idiot and nobody likes you.",
    "clean": "Thank you for your question. Here's a helpful answer to your inquiry.",
}

# Input prompts for context
TEST_PROMPTS = {
    "competitor_mention": "What cloud service should I use?",
    "banned_topic": "Tell me about chemistry experiments.",
    "biased_content": "Are there differences in mathematical abilities?",
    "code_snippet": "Write a Python script to clean up temporary files.",
    "factually_inconsistent": "What is the capital of France?",
    "gibberish": "Explain quantum physics.",
    "wrong_language": "Please respond in English only.",
    "irrelevant": "What are investment strategies?",
    "toxic": "Can you help me with my homework?",
    "clean": "How does photosynthesis work?",
}


def download_models():
    """Download models for enabled output guardrails from config."""
    print("\n" + "=" * 80)
    print("Downloading Output Guardrail Models")
    print("=" * 80)
    
    from neo_guardrail_hub.utils.model_downloader import LLMGuardModelDownloader
    
    # Check if config file exists
    if not DEFAULT_CONFIG.exists():
        print(f"\n✗ Config file not found: {DEFAULT_CONFIG}")
        print("  Please ensure configs/default.yaml exists")
        return 1
    
    print(f"\n📝 Reading config from: {DEFAULT_CONFIG}")
    print(f"📂 Models will be saved to: {MODELS_DIR}")
    
    downloader = LLMGuardModelDownloader(
        config_path=str(CONFIGS_PATH),
        models_dir=str(MODELS_DIR)
    )
    
    # Get enabled output guardrails from config
    import yaml
    with open(DEFAULT_CONFIG, 'r') as f:
        config = yaml.safe_load(f)
    
    enabled_output_guardrails = []
    if config.get('guardrails', {}).get('output', {}).get('enabled'):
        for check in config['guardrails']['output'].get('checks', []):
            if check.get('enabled') and check.get('provider') == 'llm_guard':
                guardrail_type = check.get('type')
                if guardrail_type:
                    enabled_output_guardrails.append(guardrail_type)
    
    if not enabled_output_guardrails:
        print("\n⚠️  No output guardrails enabled in config")
        print("   Enable some guardrails in configs/default.yaml")
        return 1
    
    print(f"\n✓ Found {len(enabled_output_guardrails)} enabled output guardrails:")
    for gr in enabled_output_guardrails:
        print(f"  - {gr}")
    
    # Map guardrail types to model registry scanner types
    # Some output guardrails have different names than their scanner types
    guardrail_to_scanner = {
        'ban_competitors_output': 'ban_competitors_output',
        'ban_topics': 'ban_topics_output',
        'bias_output': 'bias_output',
        'code_detection_output': 'code_detection_output',
        'factual_consistency': 'factual_consistency_output',
        'gibberish_output': 'gibberish_output',
        'language_output': 'language_output',
        'relevance': 'relevance_output',
        'toxicity_output': 'toxicity_output',
    }
    
    # Get scanner types for download
    scanner_types = []
    for gr in enabled_output_guardrails:
        scanner_type = guardrail_to_scanner.get(gr)
        if scanner_type:
            scanner_types.append(scanner_type)
    
    if not scanner_types:
        print("\n⚠️  No model-based guardrails found")
        return 0
    
    print(f"\n📥 Downloading models for {len(scanner_types)} guardrails...")
    results = downloader.download_specific_models(scanner_types)
    
    print(f"\n" + "=" * 80)
    print("Download Summary")
    print("=" * 80)
    print(f"✓ Successfully downloaded: {results['success']}")
    print(f"⊘ Skipped (already exist): {results['skipped']}")
    print(f"✗ Failed: {results['failed']}")
    
    if results['failed'] > 0:
        print(f"\n⚠️  Some downloads failed. Check logs above.")
        return 1
    
    if results['success'] > 0 or results['skipped'] > 0:
        print(f"\n✓ Models ready at: {MODELS_DIR}")
        print(f"\nTo use these models, set environment variable:")
        print(f"  export NEO_LLM_GUARD_MODELS_DIR={MODELS_DIR.absolute()}")
        return 0
    
    return 1


async def test_with_config():
    """Test output guardrails using config file and orchestrator."""
    print("\n" + "=" * 80)
    print("Testing Output Guardrails with Config File")
    print("=" * 80)
    
    # Check if models directory is configured
    models_dir = os.getenv("NEO_LLM_GUARD_MODELS_DIR") or str(MODELS_DIR)
    if Path(models_dir).exists():
        print(f"\n✓ Using local models from: {models_dir}")
        model_count = sum(1 for p in Path(models_dir).iterdir() if p.is_dir())
        print(f"  Found {model_count} model directories")
    else:
        print(f"\n⚠️  Models directory not found: {models_dir}")
        print("  Run with --download first to download models")
        print("  Will attempt to download on-demand (slower)")
    
    # Check if config file exists
    if not DEFAULT_CONFIG.exists():
        print(f"\n✗ Config file not found: {DEFAULT_CONFIG}")
        return
    
    print(f"\n📝 Loading config from: {DEFAULT_CONFIG}")
    
    # Create orchestrator with config path (directory) and models_dir
    # Following the pattern from example 27
    orchestrator = NeoGuardrailOrchestrator(
        config_path=str(CONFIGS_PATH),  # Directory, not file
        models_dir=models_dir
    )
    
    print(f"\n⏱️  Initializing orchestrator...")
    start_time = time.time()
    # Initialize without specific agent_id - will use 'default' agent
    await orchestrator.initialize()
    init_time = time.time() - start_time
    print(f"✓ Initialized in {init_time:.2f}s")
    
    # Get enabled output guardrails
    print(f"\n✓ Orchestrator initialized with local model support")
    
    # Test each output with the orchestrator
    print("\n" + "=" * 80)
    print("Running Tests")
    print("=" * 80)
    
    test_cases = [
        ("Clean Output", TEST_OUTPUTS["clean"], TEST_PROMPTS["clean"]),
        ("Competitor Mention", TEST_OUTPUTS["competitor_mention"], TEST_PROMPTS["competitor_mention"]),
        ("Banned Topic", TEST_OUTPUTS["banned_topic"], TEST_PROMPTS["banned_topic"]),
        ("Biased Content", TEST_OUTPUTS["biased_content"], TEST_PROMPTS["biased_content"]),
        ("Code Snippet", TEST_OUTPUTS["code_snippet"], TEST_PROMPTS["code_snippet"]),
        ("Factually Inconsistent", TEST_OUTPUTS["factually_inconsistent"], TEST_PROMPTS["factually_inconsistent"]),
        ("Gibberish", TEST_OUTPUTS["gibberish"], TEST_PROMPTS["gibberish"]),
        ("Wrong Language", TEST_OUTPUTS["wrong_language"], TEST_PROMPTS["wrong_language"]),
        ("Irrelevant", TEST_OUTPUTS["irrelevant"], TEST_PROMPTS["irrelevant"]),
        ("Toxic", TEST_OUTPUTS["toxic"], TEST_PROMPTS["toxic"]),
    ]
    
    results = []
    total_time = 0
    
    for name, output_text, prompt_text in test_cases:
        print(f"\n{'─' * 80}")
        print(f"Test: {name}")
        print(f"{'─' * 80}")
        print(f"Prompt: {prompt_text[:80]}{'...' if len(prompt_text) > 80 else ''}")
        print(f"Output: {output_text[:80]}{'...' if len(output_text) > 80 else ''}")
        
        start_time = time.time()
        # Use guard_output method - agent_id defaults to "default"
        result = await orchestrator.guard_output(
            text=output_text,
            context={"prompt": prompt_text}
        )
        elapsed = time.time() - start_time
        total_time += elapsed
        
        status = "✓ PASSED" if result.passed else "✗ BLOCKED"
        print(f"\n{status}")
        if hasattr(result, 'score'):
            print(f"  Overall Score: {result.score:.3f}")
        print(f"  Time: {elapsed:.3f}s")
        
        if not result.passed:
            # Get blocked reason from failed checks (following example 27 pattern)
            if hasattr(result, 'failed_checks') and result.failed_checks:
                print(f"  Failed Guardrails:")
                for check in result.failed_checks:
                    check_name = check.guardrail_name if hasattr(check, 'guardrail_name') else str(check)
                    check_msg = check.message if hasattr(check, 'message') else ''
                    print(f"    - {check_name}: {check_msg}")
        
        results.append({
            "name": name,
            "passed": result.passed,
            "score": result.score if hasattr(result, 'score') else 0.0,
            "time": elapsed
        })
    
    # Summary
    print("\n" + "=" * 80)
    print("Test Summary")
    print("=" * 80)
    
    print(f"\n{'Test Case':<30} {'Status':<10} {'Score':<8} {'Time (s)':<10}")
    print("-" * 80)
    
    passed_count = sum(1 for r in results if r["passed"])
    blocked_count = sum(1 for r in results if not r["passed"])
    
    for result in results:
        status = "✓ PASSED" if result["passed"] else "✗ BLOCKED"
        print(
            f"{result['name']:<30} {status:<10} "
            f"{result['score']:<8.3f} {result['time']:<10.3f}"
        )
    
    print("-" * 80)
    print(f"\nTotal Tests: {len(results)}")
    print(f"Passed: {passed_count}")
    print(f"Blocked: {blocked_count}")
    print(f"Total Time: {total_time:.3f}s")
    print(f"Average Time: {total_time/len(results):.3f}s per test")
    
    if Path(models_dir).exists():
        print(f"\n✓ Successfully used local models from: {models_dir}")
        print("  Typical performance: 2-3s per test with local models")
    else:
        print("\n⚠️  Models downloaded from HuggingFace during runtime")
        print("  Typical performance: 30-60s per test on first run")
    
    print("\n" + "=" * 80)
    print("Testing Complete")
    print("=" * 80)


async def test_single_guardrail(
    orchestrator: NeoGuardrailOrchestrator,
    guardrail_name: str,
    output_text: str,
    prompt_text: str,
) -> dict:
    """Test a single output guardrail."""
    print(f"\n{'='*80}")
    print(f"Testing {guardrail_name}")
    print(f"{'='*80}")
    print(f"Prompt: {prompt_text[:100]}...")
    print(f"Output: {output_text[:100]}...")
    
    start_time = time.time()
    
    result = await orchestrator.check(
        text=output_text,
        layer="output",
        context={"prompt": prompt_text}
    )
    
    elapsed = time.time() - start_time
    
    print(f"\nResult:")
    print(f"  Passed: {result.passed}")
    print(f"  Score: {result.score:.3f}")
    print(f"  Time: {elapsed:.3f}s")
    
    if not result.passed:
        print(f"  Failed Guardrails:")
        for guard_result in result.guardrail_results:
            if not guard_result.passed:
                print(f"    - {guard_result.guardrail_name}: {guard_result.message}")
    
    return {
        "guardrail": guardrail_name,
        "passed": result.passed,
        "score": result.score,
        "time": elapsed,
        "output": output_text[:50] + "..." if len(output_text) > 50 else output_text,
    }


async def test_individual_guardrails():
    """Test individual output guardrails one by one (legacy mode)."""
    print("Output Guardrails Individual Testing")
    print("=" * 80)
    
    # Check if models directory is configured
    models_dir = os.getenv("NEO_LLM_GUARD_MODELS_DIR")
    if models_dir:
        models_path = Path(models_dir)
        if models_path.exists():
            print(f"\n✓ Using local models from: {models_dir}")
            model_count = sum(1 for p in models_path.iterdir() if p.is_dir())
            print(f"  Found {model_count} model directories")
        else:
            print(f"\n⚠ Models directory not found: {models_dir}")
            print("  Will download models from HuggingFace (may take 30-60s per model)")
    else:
        print("\n⚠ NEO_LLM_GUARD_MODELS_DIR not set")
        print("  Will download models from HuggingFace (may take 30-60s per model)")
        print("\nTo use local models:")
        print("  export NEO_LLM_GUARD_MODELS_DIR=/path/to/models")
    
    # Test 1: ban_competitors_output
    print("\n" + "=" * 80)
    print("TEST 1: Ban Competitors Output")
    print("=" * 80)
    
    orchestrator = NeoGuardrailOrchestrator(
        guardrails=[
            {
                "name": "ban_competitors_output",
                "config": {
                    "competitors": ["Amazon", "AWS", "Microsoft Azure", "Google Cloud"],
                    "threshold": 0.5,
                },
            }
        ],
        models_dir=models_dir,
    )
    
    await orchestrator.initialize()
    
    result1 = await test_single_guardrail(
        orchestrator,
        "ban_competitors_output",
        TEST_OUTPUTS["competitor_mention"],
        TEST_PROMPTS["competitor_mention"],
    )
    
    # Test 2: ban_topics output
    print("\n" + "=" * 80)
    print("TEST 2: Ban Topics Output")
    print("=" * 80)
    
    orchestrator2 = NeoGuardrailOrchestrator(
        guardrails=[
            {
                "name": "ban_topics",
                "config": {
                    "topics": ["drugs", "illegal activities", "violence"],
                    "threshold": 0.5,
                },
            }
        ],
        models_dir=models_dir,
    )
    
    await orchestrator2.initialize()
    
    result2 = await test_single_guardrail(
        orchestrator2,
        "ban_topics",
        TEST_OUTPUTS["banned_topic"],
        TEST_PROMPTS["banned_topic"],
    )
    
    # Test 3: bias_output
    print("\n" + "=" * 80)
    print("TEST 3: Bias Detection Output")
    print("=" * 80)
    
    orchestrator3 = NeoGuardrailOrchestrator(
        guardrails=[
            {
                "name": "bias_output",
                "config": {
                    "threshold": 0.5,
                },
            }
        ],
        models_dir=models_dir,
    )
    
    await orchestrator3.initialize()
    
    result3 = await test_single_guardrail(
        orchestrator3,
        "bias_output",
        TEST_OUTPUTS["biased_content"],
        TEST_PROMPTS["biased_content"],
    )
    
    # Test 4: code_detection_output
    print("\n" + "=" * 80)
    print("TEST 4: Code Detection Output")
    print("=" * 80)
    
    orchestrator4 = NeoGuardrailOrchestrator(
        guardrails=[
            {
                "name": "code_detection_output",
                "config": {
                    "denied": ["python", "javascript"],
                },
            }
        ],
        models_dir=models_dir,
    )
    
    await orchestrator4.initialize()
    
    result4 = await test_single_guardrail(
        orchestrator4,
        "code_detection_output",
        TEST_OUTPUTS["code_snippet"],
        TEST_PROMPTS["code_snippet"],
    )
    
    # Test 5: factual_consistency
    print("\n" + "=" * 80)
    print("TEST 5: Factual Consistency Output")
    print("=" * 80)
    
    orchestrator5 = NeoGuardrailOrchestrator(
        guardrails=[
            {
                "name": "factual_consistency",
                "config": {
                    "minimum_score": 0.5,
                },
            }
        ],
        models_dir=models_dir,
    )
    
    await orchestrator5.initialize()
    
    result5 = await test_single_guardrail(
        orchestrator5,
        "factual_consistency",
        TEST_OUTPUTS["factually_inconsistent"],
        TEST_PROMPTS["factually_inconsistent"],
    )
    
    # Test 6: gibberish_output
    print("\n" + "=" * 80)
    print("TEST 6: Gibberish Detection Output")
    print("=" * 80)
    
    orchestrator6 = NeoGuardrailOrchestrator(
        guardrails=[
            {
                "name": "gibberish_output",
                "config": {
                    "threshold": 0.5,
                },
            }
        ],
        models_dir=models_dir,
    )
    
    await orchestrator6.initialize()
    
    result6 = await test_single_guardrail(
        orchestrator6,
        "gibberish_output",
        TEST_OUTPUTS["gibberish"],
        TEST_PROMPTS["gibberish"],
    )
    
    # Test 7: language_output
    print("\n" + "=" * 80)
    print("TEST 7: Language Detection Output")
    print("=" * 80)
    
    orchestrator7 = NeoGuardrailOrchestrator(
        guardrails=[
            {
                "name": "language_output",
                "config": {
                    "valid_languages": ["en"],
                },
            }
        ],
        models_dir=models_dir,
    )
    
    await orchestrator7.initialize()
    
    result7 = await test_single_guardrail(
        orchestrator7,
        "language_output",
        TEST_OUTPUTS["wrong_language"],
        TEST_PROMPTS["wrong_language"],
    )
    
    # Test 8: relevance
    print("\n" + "=" * 80)
    print("TEST 8: Relevance Output")
    print("=" * 80)
    
    orchestrator8 = NeoGuardrailOrchestrator(
        guardrails=[
            {
                "name": "relevance",
                "config": {
                    "threshold": 0.5,
                },
            }
        ],
        models_dir=models_dir,
    )
    
    await orchestrator8.initialize()
    
    result8 = await test_single_guardrail(
        orchestrator8,
        "relevance",
        TEST_OUTPUTS["irrelevant"],
        TEST_PROMPTS["wrong_language"],  # Intentionally mismatched prompt
    )
    
    # Test 9: toxicity_output
    print("\n" + "=" * 80)
    print("TEST 9: Toxicity Detection Output")
    print("=" * 80)
    
    orchestrator9 = NeoGuardrailOrchestrator(
        guardrails=[
            {
                "name": "toxicity_output",
                "config": {
                    "threshold": 0.5,
                },
            }
        ],
        models_dir=models_dir,
    )
    
    await orchestrator9.initialize()
    
    result9 = await test_single_guardrail(
        orchestrator9,
        "toxicity_output",
        TEST_OUTPUTS["toxic"],
        TEST_PROMPTS["toxic"],
    )
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY - Individual Guardrails Test Results")
    print("=" * 80)
    
    results = [result1, result2, result3, result4, result5, result6, result7, result8, result9]
    
    print(f"\n{'Guardrail':<30} {'Status':<10} {'Score':<8} {'Time (s)':<10}")
    print("-" * 80)
    
    total_time = 0
    passed_count = 0
    
    for result in results:
        status = "✓ PASSED" if result["passed"] else "✗ FAILED"
        print(
            f"{result['guardrail']:<30} {status:<10} "
            f"{result['score']:<8.3f} {result['time']:<10.3f}"
        )
        total_time += result["time"]
        if not result["passed"]:
            passed_count += 1
    
    print("-" * 80)
    print(f"\nTotal Tests: {len(results)}")
    print(f"Correctly Blocked: {passed_count} (expected blocks)")
    print(f"Total Time: {total_time:.3f}s")
    print(f"Average Time: {total_time/len(results):.3f}s per guardrail")
    
    if models_dir and Path(models_dir).exists():
        print(f"\n✓ Successfully used local models from: {models_dir}")
        print("  Typical performance: 2-3s per guardrail with local models")
    else:
        print("\n⚠ Models downloaded from HuggingFace during runtime")
        print("  Typical performance: 30-60s per guardrail on first run")
    
    print("\n" + "=" * 80)
    print("Individual Testing Complete")
    print("=" * 80)


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Test Output Guardrails with local models"
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download models for enabled output guardrails from config"
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Test output guardrails using config file (recommended)"
    )
    parser.add_argument(
        "--test-individual",
        action="store_true",
        help="Test individual guardrails one by one (legacy mode)"
    )
    
    args = parser.parse_args()
    
    # If no flags, show help
    if not any([args.download, args.test, args.test_individual]):
        print("Output Guardrails Local Models Test")
        print("\nUsage:")
        print("  1. Download models:      python examples/30_output_guardrails_local_models.py --download")
        print("  2. Test with config:     python examples/30_output_guardrails_local_models.py --test")
        print("  3. Test individual:      python examples/30_output_guardrails_local_models.py --test-individual")
        print("\nRecommended workflow:")
        print("  Step 1: python examples/30_output_guardrails_local_models.py --download")
        print("  Step 2: export NEO_LLM_GUARD_MODELS_DIR=$(pwd)/models/llm_guard")
        print("  Step 3: python examples/30_output_guardrails_local_models.py --test")
        return 0
    
    if args.download:
        return download_models()
    
    if args.test:
        await test_with_config()
        return 0
    
    if args.test_individual:
        await test_individual_guardrails()
        return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    exit(exit_code if exit_code else 0)

