"""Data models for Neo Guardrail Hub.

This module contains all Pydantic models used throughout the framework,
including result types, context objects, and configuration enums.
"""

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class GuardrailLayer(str, Enum):
    """Layer at which a guardrail operates."""

    INPUT = "input"
    CONTEXT = "context"
    OUTPUT = "output"


class ActionOnFail(str, Enum):
    """Action to take when a guardrail check fails."""

    BLOCK = "block"  # Block the request/response entirely
    WARN = "warn"  # Log warning but allow through
    SANITIZE = "sanitize"  # Modify content to remove issues


class ExecutionMode(str, Enum):
    """Execution mode for guardrail checks."""

    PARALLEL = "parallel"  # Run all checks concurrently
    SEQUENTIAL = "sequential"  # Run checks one after another


class GuardrailResult(BaseModel):
    """Standardized result from any guardrail check.

    This is the output format that all guardrails must return,
    ensuring consistent handling across different implementations.
    """

    passed: bool = Field(
        description="Whether the guardrail check passed (True) or failed (False)"
    )
    guardrail_name: str = Field(description="Unique identifier for the guardrail")
    layer: GuardrailLayer = Field(
        description="The layer this guardrail operates at (input/context/output)"
    )
    risk_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Risk score from 0.0 (safe) to 1.0 (high risk)",
    )
    message: Optional[str] = Field(
        default=None, description="Human-readable message explaining the result"
    )
    sanitized_text: Optional[str] = Field(
        default=None,
        description="Modified text if action=sanitize was applied",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata from the guardrail check",
    )
    latency_ms: float = Field(
        default=0.0, description="Time taken for this check in milliseconds"
    )

    model_config = {"frozen": False, "extra": "allow"}


class AggregatedResult(BaseModel):
    """Combined result from all guardrails in a layer.

    Aggregates multiple GuardrailResult objects into a single
    result that represents the overall outcome for a layer.
    """

    passed: bool = Field(
        description="Overall pass/fail status (False if any check failed with BLOCK)"
    )
    layer: GuardrailLayer = Field(description="The layer these results are from")
    results: List[GuardrailResult] = Field(
        default_factory=list, description="Individual results from each guardrail"
    )
    action_taken: ActionOnFail = Field(
        default=ActionOnFail.BLOCK,
        description="The action that was taken based on failures",
    )
    final_text: Optional[str] = Field(
        default=None,
        description="Final text after any sanitization, or original if unchanged",
    )
    total_latency_ms: float = Field(
        default=0.0, description="Total time for all checks in this layer"
    )

    model_config = {"frozen": False, "extra": "allow"}

    @property
    def failed_checks(self) -> List[GuardrailResult]:
        """Get list of failed guardrail checks."""
        return [r for r in self.results if not r.passed]

    @property
    def passed_checks(self) -> List[GuardrailResult]:
        """Get list of passed guardrail checks."""
        return [r for r in self.results if r.passed]

    @property
    def max_risk_score(self) -> float:
        """Get the maximum risk score from all checks."""
        if not self.results:
            return 0.0
        return max(r.risk_score for r in self.results)


class GuardrailContext(BaseModel):
    """Context passed to guardrails for stateful checks.

    Contains information about the current session, user,
    and conversation history that guardrails may need.
    """

    agent_id: str = Field(
        default="default", description="Identifier for the agent being guarded"
    )
    session_id: Optional[str] = Field(
        default=None, description="Unique session identifier for multi-turn tracking"
    )
    user_id: Optional[str] = Field(
        default=None, description="User identifier for user-specific policies"
    )
    conversation_history: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Previous messages in the conversation",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional context data for guardrails",
    )

    model_config = {"frozen": False, "extra": "allow"}

    def add_message(self, role: str, content: str) -> None:
        """Add a message to the conversation history."""
        self.conversation_history.append({"role": role, "content": content})

    def get_last_message(self) -> Optional[Dict[str, Any]]:
        """Get the most recent message from history."""
        if self.conversation_history:
            return self.conversation_history[-1]
        return None


class GuardrailConfig(BaseModel):
    """Configuration for a single guardrail check."""

    type: str = Field(description="The type/name of the guardrail")
    enabled: bool = Field(default=True, description="Whether this guardrail is active")
    provider: str = Field(
        default="llm_guard",
        description="Provider name (nemo, llm_guard, custom)"
    )
    priority: int = Field(
        default=10,
        description="Execution priority (lower = earlier)"
    )
    threshold: float = Field(
        default=0.5, ge=0.0, le=1.0, description="Detection threshold"
    )
    on_fail: ActionOnFail = Field(
        default=ActionOnFail.BLOCK, description="Action to take on failure"
    )
    config: Dict[str, Any] = Field(
        default_factory=dict, description="Guardrail-specific configuration"
    )

    model_config = {"frozen": False, "extra": "allow"}


class LayerConfig(BaseModel):
    """Configuration for a guardrail layer (input/context/output)."""

    enabled: bool = Field(default=True, description="Whether this layer is active")
    execution: ExecutionMode = Field(
        default=ExecutionMode.PARALLEL, description="How to run checks in this layer"
    )
    timeout_ms: int = Field(
        default=5000, description="Timeout for all checks in this layer"
    )
    checks: List[GuardrailConfig] = Field(
        default_factory=list, description="List of guardrail checks for this layer"
    )

    model_config = {"frozen": False, "extra": "allow"}
