"""LangGraph StateGraph definition for the agent agent."""

from langgraph.graph import StateGraph, END

from .state import AgentState
from .nodes.guardrail import guardrail_check
from .nodes.memory import memory_retrieval, memory_storage
from .nodes.intent import route_intent
from .nodes.intent_enrichment import intent_enrichment
from .nodes.mcp_fetch import mcp_fetch
from .nodes.generate import generate_response, generate_blocked_response
from .edges import after_guardrail, after_intent, after_enrichment


def build_graph() -> StateGraph:
    """Build and compile the agent graph."""
    graph = StateGraph(AgentState)

    # Add nodes
    graph.add_node("guardrail_check", guardrail_check)
    graph.add_node("memory_retrieval", memory_retrieval)
    graph.add_node("route_intent", route_intent)
    graph.add_node("intent_enrichment", intent_enrichment)
    graph.add_node("mcp_fetch", mcp_fetch)
    graph.add_node("generate_response", generate_response)
    graph.add_node("generate_blocked", generate_blocked_response)
    graph.add_node("memory_storage", memory_storage)

    # Entry point
    graph.set_entry_point("guardrail_check")

    # Edges from guardrail
    graph.add_conditional_edges(
        "guardrail_check",
        after_guardrail,
        {
            "blocked": "generate_blocked",
            "passed": "memory_retrieval",
        },
    )

    # Memory -> Intent
    graph.add_edge("memory_retrieval", "route_intent")

    # Intent routing
    graph.add_conditional_edges(
        "route_intent",
        after_intent,
        {
            "fetch_data": "intent_enrichment",
            "clarification": "intent_enrichment",
            "general": "generate_response",
        },
    )

    # Enrichment routing: ready → mcp_fetch, clarification → END
    graph.add_conditional_edges(
        "intent_enrichment",
        after_enrichment,
        {
            "ready": "mcp_fetch",
            "clarification": END,
        },
    )

    # MCP -> Generate
    graph.add_edge("mcp_fetch", "generate_response")

    # Generate -> Memory Storage -> END
    graph.add_edge("generate_response", "memory_storage")
    graph.add_edge("memory_storage", END)
    graph.add_edge("generate_blocked", END)

    return graph.compile()


# Compiled graph instance
agent_graph = build_graph()
