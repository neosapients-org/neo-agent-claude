#!/usr/bin/env python
"""
Example 14: NeMo Self-Check Facts (Fact Checking / Grounding)

This example demonstrates how to use the NeMo Provider for LLM-based
fact-checking using NeMo Guardrails' self_check_facts.

The fact-checking validates if a bot response is grounded in provided evidence.
This is essential for RAG (Retrieval Augmented Generation) systems.

Key concepts:
- Evidence is passed via role="context" with relevant_chunks key
- The self_check_facts flow returns an accuracy score (0.0 to 1.0)
- If accuracy < 0.5, the response is considered NOT grounded

Prerequisites:
- Install NeMo Guardrails: pip install neo-guardrail-hub[nemo]
- Set OPENAI_API_KEY environment variable

Reference:
- https://docs.nvidia.com/nemo/guardrails/latest/user-guides/guardrails-library.html#fact-checking
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

# Get the configs directory
CONFIGS_PATH = Path(__file__).parent.parent / "configs"


# =============================================================================
# Example 1: Basic Fact-Checking with check_facts()
# =============================================================================

async def example_basic_fact_checking():
    """Demonstrate basic fact-checking with check_facts method.
    
    The check_facts method uses NeMo's self_check_facts flow to verify
    if a response is grounded in the provided evidence.
    
    It returns:
    - accuracy: Score from 0.0 to 1.0
    - grounded: True if accuracy >= 0.5
    """
    print("\n" + "=" * 60)
    print("Example 1: Basic Fact-Checking with check_facts()")
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
    
    # Evidence from knowledge base (this will be passed as relevant_chunks)
    evidence = """
    Rome is the capital city of Italy.
    Paris is the capital city of France.
    The Eiffel Tower was built in 1889 and is located in Paris.
    The population of France is approximately 67 million people.
    """
    
    # Test cases: response and whether it should be grounded
    test_cases = [
        # Should be GROUNDED (in evidence)
        ("Rome is the capital of Italy.", True),
        ("The Eiffel Tower was built in 1889.", True),
        ("Paris is the capital of France.", True),
        
        # Should NOT be grounded (contradicts or not in evidence)
        ("France is the capital of Italy.", False),  # Contradicts evidence
        ("The Eiffel Tower was built in 2000.", False),  # Wrong date
        ("London is the capital of France.", False),  # Wrong fact
    ]
    
    print(f"\n📚 Evidence (relevant_chunks):")
    print(f"   {evidence.strip()[:80]}...")
    print("\n" + "-" * 60)
    
    for response, expected_grounded in test_cases:
        result = await provider.check_facts(
            response=response,
            evidence=evidence,  # Can be string or list of chunks
        )
        
        # Check if result matches expectation
        is_correct = result["grounded"] == expected_grounded
        status = "✅" if is_correct else "⚠️"
        grounded_str = "GROUNDED" if result["grounded"] else "NOT GROUNDED"
        
        print(f"\n{status} Response: \"{response}\"")
        print(f"   Result: {grounded_str}")
        print(f"   Accuracy Score: {result['accuracy']:.2f}")
        print(f"   Expected Grounded: {expected_grounded}")
    
    await provider.cleanup()
    return provider


# =============================================================================
# Example 2: Evidence as List of Chunks (RAG Pattern)
# =============================================================================

async def example_rag_pipeline():
    """Integrate fact-checking into a RAG pipeline.
    
    This example shows how to pass evidence as a list of chunks,
    simulating a typical RAG scenario where relevant documents
    are retrieved and used as evidence.
    """
    print("\n" + "=" * 60)
    print("Example 2: Evidence as List of Chunks (RAG Pipeline)")
    print("=" * 60)
    
    from neo_guardrail_hub.providers.nemo import NeMoProvider
    
    import logging
    logging.getLogger("nemoguardrails").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    
    provider = NeMoProvider(config_path=CONFIGS_PATH)
    await provider.initialize()
    
    # Simulated knowledge base as list of chunks (like from vector DB)
    knowledge_chunks = [
        "Tesla, Inc. is an American electric vehicle and clean energy company.",
        "The company was founded in 2003 by Martin Eberhard and Marc Tarpenning.",
        "Elon Musk joined the company in 2004 as chairman of the board.",
        "Tesla is headquartered in Austin, Texas.",
        "The Tesla Model S was first delivered in 2012.",
        "The Model S has a range of over 400 miles on a single charge.",
    ]
    
    async def rag_with_fact_check(query: str, llm_response: str) -> dict:
        """RAG pipeline: check if LLM response is grounded in knowledge chunks."""
        # Pass chunks as list - they will be joined with \n
        result = await provider.check_facts(
            response=llm_response,
            evidence=knowledge_chunks,  # List of chunks!
        )
        
        return {
            "response": llm_response if result["grounded"] else 
                       "I cannot verify that information based on available sources.",
            "verified": result["grounded"],
            "accuracy": result["accuracy"],
            "original": llm_response,
        }
    
    # Test queries
    test_queries = [
        ("When was Tesla founded?", "Tesla was founded in 2003."),
        ("Who founded Tesla?", "Tesla was founded by Martin Eberhard and Marc Tarpenning."),
        ("Who is the CEO of Tesla?", "Steve Jobs is the CEO of Tesla."),  # Wrong!
        ("What is the Model S range?", "The Tesla Model S has a range of over 400 miles."),
        ("When did Tesla go public?", "Tesla went public in 2010."),  # Not in KB
    ]
    
    print(f"\n📚 Knowledge Chunks: {len(knowledge_chunks)} documents")
    for chunk in knowledge_chunks[:2]:
        print(f"   • {chunk[:50]}...")
    print(f"   • ... and {len(knowledge_chunks) - 2} more")
    print("\n" + "-" * 60)
    
    for query, llm_response in test_queries:
        print(f"\n👤 Query: {query}")
        print(f"🤖 LLM: {llm_response}")
        
        result = await rag_with_fact_check(query, llm_response)
        
        status = "✅ Verified" if result["verified"] else "❌ Not Verified"
        print(f"   {status} (accuracy: {result['accuracy']:.2f})")
        print(f"📤 Final: {result['response'][:60]}...")
    
    await provider.cleanup()


# =============================================================================
# Example 3: Using the Guardrail Interface
# =============================================================================

async def example_guardrail_interface():
    """Use NeMoSelfCheckFactsGuardrail with standard guardrail interface.
    
    The guardrail uses the same underlying fact-checking mechanism
    but exposes it via the standard GuardrailResult interface.
    
    Evidence is passed via context.metadata["relevant_chunks"] or context.metadata["evidence"].
    """
    print("\n" + "=" * 60)
    print("Example 3: Guardrail Interface")
    print("=" * 60)
    
    from neo_guardrail_hub.providers.nemo import NeMoProvider
    from neo_guardrail_hub.core.models import GuardrailContext
    
    import logging
    logging.getLogger("nemoguardrails").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    
    provider = NeMoProvider(config_path=CONFIGS_PATH)
    await provider.initialize()
    
    # Get the fact-checking guardrail
    guardrail = provider.get_guardrail("nemo_self_check_facts")
    
    print(f"\n📋 Guardrail: {guardrail.name}")
    print(f"   Threshold: {guardrail.threshold}")
    
    # Evidence can be passed via relevant_chunks (preferred) or evidence
    evidence = "Python was created by Guido van Rossum and first released in 1991."
    
    test_cases = [
        ("Python was released in 1991.", True),
        ("Python was created by Guido van Rossum.", True),
        ("Python was invented in 2020.", False),
    ]
    
    print(f"\n📚 Evidence: \"{evidence}\"")
    print("-" * 60)
    
    for response, expected in test_cases:
        # Pass evidence in context metadata - use "relevant_chunks" key
        context = GuardrailContext(metadata={
            "relevant_chunks": evidence,  # This matches NeMo's expected format
        })
        
        result = await guardrail.check(response, context=context)
        
        status = "✅" if result.passed == expected else "⚠️"
        print(f"\n{status} \"{response}\"")
        print(f"   Passed: {result.passed} (expected: {expected})")
        print(f"   Risk Score: {result.risk_score:.2f}")
    
    await provider.cleanup()


# =============================================================================
# Example 4: Orchestrator Integration
# =============================================================================

async def example_orchestrator_integration():
    """Use fact-checking via the NeoGuardrailOrchestrator.
    
    The orchestrator runs all enabled output guardrails including:
    - nemo_self_check_output (policy compliance)
    - nemo_self_check_facts (fact-checking)
    - nemo_self_check_hallucination (consistency)
    
    Evidence is passed via GuardrailContext.metadata['relevant_chunks'].
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
    # The config has self_check_facts: true under nemo.rails.output
    orchestrator = NeoGuardrailOrchestrator(config_path=str(CONFIGS_PATH))
    
    print("\n🔄 Initializing orchestrator with NeMo output rails...")
    await orchestrator.initialize()
    print("✅ Orchestrator initialized!")
    
    # Knowledge base (evidence) for fact-checking - as list of chunks
    evidence_chunks = [
        "Apple Inc. is an American multinational technology company.",
        "Apple was founded by Steve Jobs, Steve Wozniak, and Ronald Wayne in 1976.",
        "The company is headquartered in Cupertino, California.",
        "Tim Cook became CEO in 2011 after Steve Jobs resigned.",
        "The iPhone was first released in 2007.",
    ]
    
    # Test cases: (response, description)
    test_cases = [
        ("Apple was founded in 1976.", "Correct fact"),
        ("The iPhone was released in 2007.", "Correct fact"),
        ("Apple was founded in 1990.", "Incorrect year"),
        ("Microsoft is headquartered in Cupertino.", "Wrong company"),
    ]
    
    print("\n📚 Knowledge Chunks:")
    for chunk in evidence_chunks[:2]:
        print(f"   • {chunk[:50]}...")
    print(f"   • ... and {len(evidence_chunks) - 2} more")
    print("\n" + "-" * 60)
    
    for response, description in test_cases:
        print(f"\n🤖 Response: \"{response}\"")
        print(f"   ({description})")
        
        # Pass evidence in context metadata using relevant_chunks
        context = GuardrailContext(
            metadata={
                "relevant_chunks": "\n".join(evidence_chunks),  # Join chunks with newline
                "user_input": "Tell me about Apple Inc.",  # For hallucination check
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
            print(f"     [{check_status}] {r.guardrail_name}: risk={r.risk_score:.2f}")
    
    await orchestrator.cleanup()
    print("\n✅ Orchestrator example complete!")


# =============================================================================
# Main
# =============================================================================

async def main():
    """Run all fact-checking examples."""
    print("\n" + "=" * 60)
    print("NeMo Fact-Checking Examples (Self Check Facts)")
    print("=" * 60)
    print("\nThis example demonstrates how to verify if LLM responses")
    print("are grounded in provided evidence (relevant_chunks).")
    print("\nKey concepts:")
    print("  • Evidence is passed via messages with role='context'")
    print("  • NeMo returns an accuracy score (0.0 to 1.0)")
    print("  • If accuracy < 0.5, response is NOT grounded")
    
    try:
        # Check if NeMo is installed
        try:
            import nemoguardrails
            print(f"\n✅ NeMo Guardrails version: {nemoguardrails.__version__}")
        except ImportError:
            print("\n❌ NeMo Guardrails not installed!")
            print("   Install with: pip install neo-guardrail-hub[nemo]")
            return
        
        # Run examples - uncomment to run specific examples
        await example_basic_fact_checking()
        await example_rag_pipeline()
        await example_guardrail_interface()
        await example_orchestrator_integration()
        
        print("\n" + "=" * 60)
        print("All fact-checking examples completed!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
