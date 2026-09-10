"""Conditional edge logic for the agent graph."""

from ..graph.state import AgentState


def after_guardrail(state: AgentState) -> str:
    """Route based on guardrail result."""
    if not state.get("guardrail_passed", True):
        return "blocked"
    return "passed"


def after_intent(state: AgentState) -> str:
    """Route based on classified intent."""
    intent = state.get("intent", "general")
    if intent == "clarification":
        return "clarification"
    elif intent == "fetch_data":
        return "fetch_data"
    else:
        return "general"


def after_parallel_prep(state: AgentState) -> str:
    """Route based on guardrail result from parallel_prep.

    Intent classification now happens inside intent_enrichment,
    so we always proceed to enrichment unless blocked.
    """
    if not state.get("guardrail_passed", True):
        return "blocked"
    return "enrich"


def after_enrichment(state: AgentState) -> str:
    """Route based on intent enrichment result.

    Handles all intent outcomes including general (conversational) queries.
    """
    intent = state.get("intent")
    if intent == "clarification":
        return "clarification"
    if intent == "from_memory":
        return "from_memory"
    if intent == "general":
        return "general"
    if intent == "acknowledge":
        return "acknowledge"
    return "ready"
