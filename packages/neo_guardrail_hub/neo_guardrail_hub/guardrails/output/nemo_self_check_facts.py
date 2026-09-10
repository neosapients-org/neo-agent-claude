"""NeMo self-check facts guardrail.

This module provides guardrail protection using NeMo Guardrails'
self_check_facts flow for LLM-based fact-checking validation.

The self_check_facts flow validates bot responses against provided evidence
(relevant_chunks) to ensure the response is grounded in the provided context.
This is particularly useful for RAG (Retrieval Augmented Generation) systems.

Reference:
- https://docs.nvidia.com/nemo/guardrails/latest/user-guides/guardrails-library.html#fact-checking
- https://docs.nvidia.com/nemo/guardrails/latest/getting-started/7-rag/README.html
"""

import time
from typing import Any, Dict, List, Optional, TYPE_CHECKING, Union

from ...core.models import GuardrailContext, GuardrailLayer, GuardrailResult
from ..base import GuardrailBase

if TYPE_CHECKING:
    from ...providers.nemo.provider import NeMoProvider


class NeMoSelfCheckFactsGuardrail(GuardrailBase):
    """
    LLM-based fact-checking using NeMo's self_check_facts.
    
    This guardrail uses NeMo Guardrails' self_check_facts flow to validate
    that bot responses are grounded in provided evidence/context. It's
    designed for RAG systems where you want to ensure responses don't
    include information not present in the retrieved documents.
    
    How it works:
    1. Takes bot response and evidence (relevant_chunks)
    2. Uses NeMo's self_check_facts action with configured prompt
    3. LLM checks if response is entailed by the evidence
    4. Returns accuracy score between 0.0 and 1.0
    5. Blocks if accuracy < threshold (default 0.5)
    
    Use cases:
    - RAG systems - verify answers match retrieved documents
    - Document QA - ensure responses cite provided sources
    - Knowledge base queries - prevent fabricated information
    
    Configuration:
        threshold: Minimum accuracy score to pass (default: 0.5)
        on_fail: Action when blocked (default: "block")
        provider: NeMoProvider instance (required, set via configure())
    
    Example:
        from neo_guardrail_hub.providers.nemo import NeMoProvider
        
        provider = NeMoProvider(config_path="./configs")
        await provider.initialize()
        
        guardrail = provider.get_guardrail("nemo_self_check_facts")
        
        # Provide evidence via context
        result = await guardrail.check(
            "The capital of France is Paris.",
            context=GuardrailContext(
                metadata={
                    "evidence": "France is a country in Western Europe. Paris is its capital city.",
                    # Or use "relevant_chunks" for NeMo compatibility
                    "relevant_chunks": "France is a country in Western Europe. Paris is its capital city."
                }
            )
        )
        # result.passed = True (response matches evidence)
        
        result = await guardrail.check(
            "The capital of France is London.",
            context=GuardrailContext(
                metadata={"evidence": "Paris is the capital of France."}
            )
        )
        # result.passed = False (response contradicts evidence)
    
    NeMo Configuration:
        In config.yml, output rails must include:
        
        rails:
          output:
            flows:
              - self check facts
        
        In prompts section (or prompts.yml):
        
        prompts:
          - task: self_check_facts
            content: |
              You are given a task to identify if the hypothesis is grounded
              and entailed to the evidence...
    """
    
    name = "nemo_self_check_facts"
    layer = GuardrailLayer.OUTPUT
    description = "LLM-based fact-checking using NeMo's self_check_facts"
    
    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize the NeMo self-check facts guardrail.
        
        Args:
            config: Configuration dictionary including:
                - provider: NeMoProvider instance (required)
                - threshold: Accuracy threshold (default: 0.5)
                - on_fail: Action when blocked ("block", "warn")
                - refusal_message: Custom refusal message
        """
        super().__init__(config)
        
        self._provider: Optional["NeMoProvider"] = self._config.get("provider")
        
        # Accuracy threshold - responses below this are blocked
        self._threshold = self._config.get("threshold", 0.5)
        
        # Refusal message for blocked outputs
        self._refusal_message = self._config.get(
            "refusal_message",
            "I cannot verify that information based on the available sources."
        )
    
    async def initialize(self) -> None:
        """Initialize the guardrail and ensure provider is ready."""
        if self._initialized:
            return
            
        if self._provider is None:
            raise ValueError(
                "NeMoSelfCheckFactsGuardrail requires a NeMoProvider. "
                "Get this guardrail from provider.get_guardrail('nemo_self_check_facts')"
            )
        
        # Ensure provider is initialized
        if not self._provider._initialized:
            await self._provider.initialize()
        
        self._initialized = True
        self.logger.info("nemo_self_check_facts_guardrail_initialized")
    
    async def _check(
        self,
        text: str,
        context: Optional[GuardrailContext] = None,
    ) -> GuardrailResult:
        """Check if bot output is factually grounded in evidence.
        
        Args:
            text: Bot response to validate
            context: Context containing evidence/relevant_chunks
            
        Returns:
            GuardrailResult with:
                - passed: True if factually accurate, False if not grounded
                - risk_score: 1.0 - accuracy (higher = more risk)
                - message: Refusal message if blocked
        """
        # Initialize if needed
        if not self._initialized:
            await self.initialize()
        
        # Extract evidence from context
        evidence = self._extract_evidence(context)
        
        if not evidence:
            # If no evidence provided, we can't fact-check
            # Default to passing with a warning
            self.logger.warning(
                "nemo_self_check_facts_no_evidence",
                message="No evidence provided for fact-checking, skipping check"
            )
            return GuardrailResult(
                passed=True,
                guardrail_name=self.name,
                layer=self.layer,
                risk_score=0.0,
                message="No evidence provided for fact-checking",
                details={
                    "skipped": True,
                    "reason": "no_evidence",
                },
            )
        
        # Use provider to check facts
        result = await self._provider.check_facts(
            response=text,
            evidence=evidence,
            context=context,
        )
        
        accuracy = result.get("accuracy", 0.0)
        passed = accuracy >= self._threshold
        nemo_message = result.get("message", "")
        
        if passed:
            return GuardrailResult(
                passed=True,
                guardrail_name=self.name,
                layer=self.layer,
                risk_score=1.0 - accuracy,
                message=nemo_message if nemo_message else None,
                details={
                    "accuracy": accuracy,
                    "threshold": self._threshold,
                    "evidence_length": len(evidence),
                    "nemo_response": nemo_message,
                },
            )
        else:
            return GuardrailResult(
                passed=False,
                guardrail_name=self.name,
                layer=self.layer,
                risk_score=1.0 - accuracy,
                message=nemo_message if nemo_message else self._refusal_message,
                details={
                    "accuracy": accuracy,
                    "threshold": self._threshold,
                    "evidence_length": len(evidence),
                    "blocked_reason": "factual_inaccuracy",
                    "nemo_response": nemo_message,
                },
            )
    
    def _extract_evidence(
        self,
        context: Optional[GuardrailContext],
    ) -> str:
        """Extract evidence from context.
        
        Looks for evidence in various context fields:
        - context.metadata["relevant_chunks"] (preferred - matches NeMo format)
        - context.metadata["evidence"]
        - context.metadata["context"]
        - context.metadata["documents"]
        
        If evidence is a list, joins with newline separator.
        
        Args:
            context: Optional guardrail context
            
        Returns:
            Evidence string or empty string if not found
        """
        if not context:
            return ""
        
        metadata = context.metadata or {}
        
        # Check various possible field names (relevant_chunks first as it matches NeMo format)
        evidence_fields = [
            "relevant_chunks",
            "evidence",
            "context",
            "documents",
            "retrieved_documents",
            "source_documents",
        ]
        
        for field in evidence_fields:
            if field in metadata:
                value = metadata[field]
                if isinstance(value, str):
                    return value
                elif isinstance(value, list):
                    # Join list of chunks/documents with newline
                    return "\n".join(str(v) for v in value)
                elif isinstance(value, dict):
                    # Handle structured documents
                    return str(value)
        
        return ""
    
    @property
    def threshold(self) -> float:
        """Get the accuracy threshold."""
        return self._threshold
    
    @threshold.setter
    def threshold(self, value: float) -> None:
        """Set the accuracy threshold."""
        if not 0.0 <= value <= 1.0:
            raise ValueError("Threshold must be between 0.0 and 1.0")
        self._threshold = value
