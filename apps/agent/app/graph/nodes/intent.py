"""Intent routing node — classifies user intent via LLM."""

import time

# --- ns_probe at the top ---
try:
    from ns_probe import observe
except ImportError:
    def observe(*a, **kw):
        def _id(fn): return fn
        return _id

from langchain_core.messages import SystemMessage, HumanMessage

from app.llm import make_llm
from ...config import config
from ..state import AgentState

INTENT_SYSTEM_PROMPT = """Classify the user query into one intent. Respond with ONLY the intent name.

- "fetch_data" — User wants financial data, portfolio info, account details, investor profile, or any platform data. If they mention an investor name OR a data type, classify as fetch_data. Timeframe is optional.
- "clarification" — Query is missing BOTH the investor name AND what data they want. Only use this if the query is truly incomplete (e.g. "show me portfolio for" with no name).
- "general" — Greetings, capabilities questions, or non-data conversation.

Respond with ONLY: fetch_data, clarification, or general"""


async def route_intent(state: AgentState) -> dict:
    """Classify the user's intent using LLM."""
    query = state["query"]
    start = time.perf_counter()

    llm = make_llm("default", temperature=0.0, max_tokens=20)

    messages = [
        SystemMessage(content=INTENT_SYSTEM_PROMPT),
        HumanMessage(content=query),
    ]

    response = await llm.ainvoke(messages)
    intent = response.content.strip().lower()

    # Normalize
    if intent not in ("fetch_data", "clarification", "general"):
        intent = "fetch_data"  # Default to fetching data if unclear

    elapsed_ms = round((time.perf_counter() - start) * 1000)

    return {
        "intent": intent,
        "latency_breakdown": {**state.get("latency_breakdown", {}), "intent_ms": elapsed_ms},
    }


# --- Instrument at the bottom of the file ---
_obs = dict(capture_args=False, capture_result=False)
route_intent = observe(name="Intent Classification", agent_name="agent", **_obs)(route_intent)
