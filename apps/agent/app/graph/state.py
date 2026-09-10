"""Agent state definition for LangGraph."""

from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    query: str
    user_id: str
    intent: str | None
    mcp_tool_calls: list[dict]
    guardrail_result: dict
    guardrail_passed: bool
    latency_breakdown: dict
    investor_name: str | None
    memory_context: str
    session_preferences: str  # RM_SESSION_PREFERENCES — ephemeral preference-box text for
                              # this session. Highest-priority HOW-to-respond signal; applied
                              # at answer time, never auto-persisted to long-term memory.
    memory_loaded: bool  # True once long-term memory has been fetched this session
                         # (even if it came back empty) — gates the once-per-session fetch
    long_term_memories: list  # raw memory items from the once-per-session fetch, surfaced
                              # to the debug panel (avoids a 2nd redundant retrieve in main)
    enriched_query: str
    # The PARENT (top-level) enriched query that hit MCP last turn, post scope-lock — fed to
    # the resolver next turn as the prior exchange's resolved form, so a follow-up inherits a
    # self-contained antecedent. Set to "" on non-data turns so it never goes stale (always
    # reflects the immediately prior turn). Decomposition sub-queries are excluded (sub-states).
    last_enriched_query: str | None
    enrichment_entities: dict
    chat_history: list  # frontend session turns [{role, content}], replaces per-turn
    chart_spec: dict | None  # Chart.js-shaped spec emitted by generate_response (issue #2)
    enrichment_prepared: dict | None  # enrichment result precomputed concurrently in
                                      # parallel_prep; applied by the intent_enrichment node
    query_plan: dict | None  # Phase 3: the planner's decision (strategy + steps + synthesis)
    execution_trace: list  # Phase 3: per sub-question record (question, success) for tracing
    mode: str | None  # M1: chat mode from the frontend (quick / client / deep)
    mode_config: dict  # M1: effective per-mode settings resolved by app/modes.mode_config
    refresh: bool  # request flag, kept for API compatibility (Quick Facts answer cache removed)
    session_id: str | None  # conversation/session identifier from the frontend
    selected_client: str | None  # M7: client selected in Client Insights mode (scope lock)
    active_skills: list  # M3: skill brief dicts selected for this turn (read by generate)
    skill_plan: dict | None  # M3: plan compiled from a methodology skill (Option B)
