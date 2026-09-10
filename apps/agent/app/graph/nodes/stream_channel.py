"""Per-request streaming channel shared between the orchestrator and the generate
node.

main.py sets `event_queue` to an asyncio.Queue before running the graph; the
orchestrator pushes thinking-step markers and the generate node pushes answer
tokens AS they're produced, so main.py can relay them to the SSE stream live.
Optional — if `event_queue` is unset (e.g. the non-streaming /chat endpoint), the
emits are no-ops and the answer is simply returned in the final state instead.
"""

import contextvars

event_queue: contextvars.ContextVar = contextvars.ContextVar("agent_event_queue", default=None)

# Per-request log of EVERY platform (MCP) call — set to a fresh list by main.py before the
# graph runs. mcp_client.call_tool appends one entry per call (incl. retries / sub-questions /
# fund-resolution probes), so the UI can show everything that hit the platform, not just the
# final per-sub-question results.
platform_calls: contextvars.ContextVar = contextvars.ContextVar("agent_platform_calls", default=None)

# When True, record_platform_call is a no-op. Used to keep SYSTEM / front-load MCP calls (the
# Cortex context-map + skills registry refresh: roster, RM map, dimensions, list_skills) OUT of
# the user-facing "queries sent to the platform" panel — those are internal bootstrap queries, not
# answers to the user's question. Set only around the registry refresh (see context/skills registry).
suppress_platform_log: contextvars.ContextVar = contextvars.ContextVar(
    "agent_suppress_platform_log", default=False)


def record_platform_call(entry: dict) -> None:
    """Append one {tool, q, ok, ms} record to the per-request platform-call log (no-op if unset,
    or while suppressed for a system/front-load refresh)."""
    if suppress_platform_log.get():
        return
    calls = platform_calls.get()
    if calls is not None:
        try:
            calls.append(entry)
        except Exception:
            pass


def emit_step(tool: str) -> None:
    """Mark a thinking-step (e.g. 'mcp_fetch') as reached, for progressive ticking."""
    q = event_queue.get()
    if q is not None:
        try:
            q.put_nowait({"type": "step", "tool": tool})
        except Exception:
            pass


def emit_detail(tool: str, text: str) -> None:
    """Attach a live sub-status line to a thinking-step — the human-readable 'what it's
    doing right now / what it found' detail shown under the step in the UI."""
    q = event_queue.get()
    if q is not None and text:
        try:
            q.put_nowait({"type": "detail", "tool": tool, "text": text})
        except Exception:
            pass


def emit_token(content: str) -> None:
    """Emit one answer-token chunk live (used by generate's streaming LLM call)."""
    q = event_queue.get()
    if q is not None and content:
        try:
            q.put_nowait({"type": "token", "content": content})
        except Exception:
            pass


def streaming_active() -> bool:
    """True when a live SSE consumer is attached (so generate should astream)."""
    return event_queue.get() is not None
