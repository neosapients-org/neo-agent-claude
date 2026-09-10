"""LangGraph StateGraph definition for the agent agent.

Architecture (two concurrent tracks, gated on the guardrail):
  orchestrate
    ├─ Track A (gate)   : guardrail + memory
    └─ Track B (answer) : enrichment → mcp → generate   (runs speculatively)

  The answer track runs CONCURRENTLY with the gate. The moment the guardrail
  verdict is known: if it BLOCKED, Track B is cancelled and the refusal is
  returned; if it PASSED, Track B's buffered answer is returned. Implemented as a
  single node (manual asyncio) so cancellation works and the checkpointer still
  persists memory_loaded + accumulates messages.
"""

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from .state import AgentState
from .nodes.parallel import orchestrate


def build_graph() -> StateGraph:
    """Build and compile the agent graph with in-memory checkpointer."""
    graph = StateGraph(AgentState)

    graph.add_node("orchestrate", orchestrate)
    graph.set_entry_point("orchestrate")
    graph.add_edge("orchestrate", END)

    # Use MemorySaver for last-N conversation history (session-scoped by thread_id).
    checkpointer = MemorySaver()
    return graph.compile(checkpointer=checkpointer)


# Compiled graph instance
agent_graph = build_graph()
