"""Guardrail check node using Neo Guardrail Hub."""

import os
import time

# --- ns_probe at the top ---
try:
    from ns_probe import observe
except ImportError:
    def observe(*a, **kw):
        def _id(fn): return fn
        return _id

from neo_guardrail_hub import GuardSession

try:
    from ns_probe import current_span
except ImportError:
    def current_span(): return None

from ..state import AgentState

# Resolve configs path relative to the agent directory
_AGENT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
_CONFIGS_PATH = os.path.join(_AGENT_DIR, "configs")


async def guardrail_check(state: AgentState) -> dict:
    """Validate user input through guardrails. Short-circuits on failure."""
    query = state["query"]
    start = time.perf_counter()

    try:
        async with GuardSession(
            agent_id="wealth_advisor",
            config_path=_CONFIGS_PATH,
        ) as session:
            result = await session.guard_input(query)

        elapsed_ms = round((time.perf_counter() - start) * 1000)

        span = current_span()
        if span:
            span.set_attributes({
                "guardrail.passed": result.passed,
                "guardrail.risk_score": float(result.max_risk_score),
                "guardrail.latency_ms": elapsed_ms,
            })

        if not result.passed:
            return {
                "guardrail_passed": False,
                "guardrail_result": {
                    "passed": False,
                    "reason": result.results[0].message if result.results else "Blocked by guardrail",
                    "risk_score": result.max_risk_score,
                },
                "latency_breakdown": {**state.get("latency_breakdown", {}), "guardrail_ms": elapsed_ms},
            }

        return {
            "guardrail_passed": True,
            "guardrail_result": {"passed": True, "reason": None, "risk_score": 0.0},
            "latency_breakdown": {**state.get("latency_breakdown", {}), "guardrail_ms": elapsed_ms},
        }

    except Exception as e:
        elapsed_ms = round((time.perf_counter() - start) * 1000)
        # On error, allow through but log
        return {
            "guardrail_passed": True,
            "guardrail_result": {"passed": True, "reason": f"Guardrail error (allowing): {e}", "risk_score": 0.0},
            "latency_breakdown": {**state.get("latency_breakdown", {}), "guardrail_ms": elapsed_ms},
        }


# --- Instrument at the bottom of the file ---
_obs = dict(capture_args=False, capture_result=False)
guardrail_check = observe(name="Input Guardrail", agent_name="agent", **_obs)(guardrail_check)
