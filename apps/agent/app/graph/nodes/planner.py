"""Query Planner — decides HOW to attack a question (Phase 3).

Most questions are simple: one subject, one ask → answer with a single platform call (DIRECT).
But a Relationship Manager often asks layered questions the stateless platform cannot answer
in one shot, e.g. "for my top 3 clients by AUM, show each one's biggest losing holding". For
those the planner returns a small to-do list of self-contained sub-questions (DECOMPOSE),
noting which sub-answers feed into which, so the Execution Engine can run them one-by-one /
in parallel and the answer step can combine them.

Cost control: obvious single-clause questions skip the planner LLM entirely via the
deterministic ``_looks_simple`` pre-gate, so the common case adds zero latency. The planner
runs on the cheap/fast model and its output is validated + capped before use.
"""

import json
import re
import time
from typing import Optional

try:
    from typing import Literal
except ImportError:  # pragma: no cover
    Literal = None  # type: ignore

try:
    from ns_probe import observe
except ImportError:
    def observe(*a, **kw):
        def _id(fn): return fn
        return _id

from langchain_core.messages import SystemMessage, HumanMessage
from pydantic import BaseModel, Field

from app.llm import make_llm
from ...config import config


# --- structured plan shapes ------------------------------------------------------

class PlanStep(BaseModel):
    """One sub-question in a decomposition plan."""
    question: str = Field(description="A self-contained plain-English question the stateless platform can answer alone.")
    depends_on: list[int] = Field(default_factory=list, description="Indexes of EARLIER steps whose ANSWERS this step needs.")
    fills: Optional[str] = Field(default=None, description="Short name for the value this step's answer provides, referenced by a later step as {name}.")
    for_each: bool = Field(default=False, description="Run this step once PER item from the (single) dependency's list, substituting each item for {item}.")


class QueryPlan(BaseModel):
    """How to attack the question."""
    strategy: str = Field(description="'direct' (one platform call) or 'decompose' (several sub-questions).")
    reason: str = Field(default="", description="One short clause on why (for logs).")
    steps: list[PlanStep] = Field(default_factory=list, description="One step for direct; several for decompose, in logical order.")
    synthesis: str = Field(default="", description="Instruction for how to combine the sub-answers into the final answer.")


PLANNER_SYSTEM_PROMPT = """You plan how a wealth-management agent should answer a Relationship Manager's question \
against a STATELESS data platform (it answers ONE self-contained natural-language question at a time and cannot \
see other questions).

Decide a strategy:
- "direct": the question has one subject and one ask, or is a single lookup/list/aggregate the platform can answer \
in one call. Return exactly ONE step whose question is the user's question (lightly cleaned). A superlative \
selection that returns ONE answer is DIRECT, not decompose — e.g. "Find the client with the worst-performing \
portfolio" or "Which client has the highest AUM?" is a single lookup (do NOT add a follow-up step).
- "decompose": the question is layered and needs several platform calls. Return an ordered list of self-contained \
sub-questions.

Use "decompose" when the question:
- asks for several different metrics at once (e.g. "AUM, XIRR and top sector for Ram") → one step per metric;
- needs an intermediate answer first (e.g. "the RM who manages the client with the highest AUM") → a discovery step \
then a step that uses it;
- says "for each" / "per" a set you must first find (e.g. "for my top 3 clients by AUM, show their biggest loss").

Dependencies & placeholders:
- If a later step needs an earlier step's answer, set its depends_on to the earlier step index and reference the \
value with {name}, where the earlier step set fills:"name".
- If a step must run once per item of a discovered list, set for_each:true, depend on that list step, and write the \
step using {item} (e.g. "List the holdings with unrealized loss for {item}.").
- Keep every name, filter and timeframe. NEVER invent client/fund names. Keep the original ask type (count/list/value).

Keep plans SMALL (prefer 2-4 steps). Plain English only — no SQL or operators.

Examples:
Q: "What is Ram Krishnan's AUM?" -> direct, 1 step.
Q: "Show Ram Krishnan's AUM, XIRR and largest holding." -> decompose: 3 independent steps (no deps).
Q: "Which RM manages the client with the highest AUM?" -> decompose:
   step0 {question:"Which client has the highest AUM? Give the client name.", fills:"client"}
   step1 {question:"Who is the relationship manager of {client}?", depends_on:[0]}
Q: "For my top 3 clients by AUM, what is each one's biggest losing holding?" -> decompose:
   step0 {question:"Who are the top 3 clients by AUM? List their names.", fills:"clients"}
   step1 {question:"List all holdings with unrealized gain/loss for {item}.", depends_on:[0], for_each:true}
   synthesis:"For each client, pick the holding with the most negative unrealized gain."
"""


# Signals that a question is layered enough to be worth the planner LLM. If NONE are present we
# treat it as simple and skip planning (the common, fast case). Missing a signal only costs a
# possible decomposition (we answer directly), never correctness.
_MULTIPART_SIGNALS = (
    " for each ", " per ", " each of ", " respectively", " and also ", " as well as ",
    " along with ", " then ", " after that ", " broken down by ", " for every ",
    " compared to their ", " who also ", " that also ", " and their ", " and the ",
)
_METRIC_WORDS = (
    "aum", "xirr", "cagr", "returns", "return", "nav", "expense ratio", "sector",
    "holding", "holdings", "gain", "loss", "allocation", "revenue", "rating",
    "maturity", "sip", "transaction", "pan", "risk", "value",
)


# Distinct ENTITY-TYPE nouns — two or more alongside a superlative usually means a multi-hop
# question ("the RM of the client with the highest AUM" -> find client, then find their RM).
# NOTE: attribute words like "portfolio"/"holding"/"sector" are deliberately NOT here — "the
# client with the worst portfolio" is ONE entity (client) + its attribute, a single lookup, and
# must not be mistaken for a two-entity multi-hop.
_ROLE_WORDS = (
    "client", "investor", " rm ", "relationship manager", "banker", " amc ",
    "fund", "scheme", "family", "manager",
)
_SUPERLATIVE_RE = re.compile(
    r"\b(highest|lowest|largest|smallest|biggest|top|bottom|worst|best|most|least|maximum|minimum)\b", re.I)
_ITERATION_RE = re.compile(
    r"\b(each|every|per|both|respectively|compare|comparison|versus|vs|between|one'?s)\b", re.I)
# A FILTER CONDITION = a threshold comparator or a negation/absence marker. Two or more of
# these means the question selects a set by SEVERAL independent predicates (e.g. "no purchases
# last month" + "AUM above ₹20 Cr"). The stateless platform can't evaluate them jointly, so it
# needs one lookup per predicate + an intersection — a decompose. We count the CONDITIONS, not
# the joining word, so any conjunction (and / but / with / a comma) is handled the same way.
_FILTER_CONDITION_RE = re.compile(
    r"\b(above|below|more than|greater than|less than|fewer than|at least|at most|"
    r"exceeding|higher than|lower than|no|without|none|not|nil)\b", re.I)


def _looks_simple(query: str) -> bool:
    """Deterministic pre-gate: True ONLY for an obviously atomic single-shot question, so we can
    skip the planner LLM. Anything with a sign of layering returns False and lets the planner
    LLM decide (which may still answer 'direct'). Biased toward consulting the planner — a
    false 'complex' only costs one cheap planner call, but a false 'simple' silently loses
    decomposition (the bug we just hit)."""
    q = " " + (query or "").lower().strip() + " "

    if q.count("?") > 1:
        return False
    if any(sig in q for sig in _MULTIPART_SIGNALS):
        return False
    # Per-item / comparison language ("for each", "each one's", "compare", "vs", "both").
    if _ITERATION_RE.search(q):
        return False
    # "top/bottom N ..." selects a SET you then ask about -> multi-hop / fan-out.
    if re.search(r"\b(top|bottom)\s+\d+\b", q):
        return False
    # Two+ filter conditions (thresholds / negations) -> a multi-predicate set selection the
    # stateless platform can't compute in one shot (intersection). Let the planner split it.
    if len(_FILTER_CONDITION_RE.findall(q)) >= 2:
        return False

    metrics = sum(1 for m in _METRIC_WORDS if m in q)
    roles = sum(1 for r in _ROLE_WORDS if r in q)

    # A superlative that picks an entity referenced by a SECOND entity/relation -> multi-hop.
    if _SUPERLATIVE_RE.search(q) and (roles >= 2 or re.search(r"\b(and|their|whose|then)\b", q)):
        return False
    # "A and B" or a comma list joining two+ distinct metrics -> multi-part.
    if (" and " in q or q.count(",") >= 1) and metrics >= 2:
        return False
    return True


def _direct_plan(query: str, reason: str = "single-shot") -> dict:
    return {"strategy": "direct", "reason": reason,
            "steps": [{"question": query, "depends_on": [], "fills": None, "for_each": False}],
            "synthesis": ""}


# --- XIRR -> always pull the benchmark alongside -------------------------------------------------
# Whenever an RM asks for a client's XIRR, the answer should show the Nifty 50 benchmark XIRR next
# to it. The benchmark is client-independent, so it's a no-dependency parallel sub-step. It is
# rendered ONLY when the platform returns it (synthesis says "when available"), so this is safe to
# ship before the MCP supports the benchmark — if it comes back empty it is simply omitted and the
# client's XIRR is shown alone, and it lights up automatically once the MCP starts returning it.
_XIRR_RE = re.compile(r"\bxirr\b", re.I)
# Benchmark already covered by the plan (e.g. a methodology skill) — don't add a second one.
_BENCH_RE = re.compile(r"\b(benchmark|nifty|index\s+return)\b", re.I)
# List / aggregate / cross-client XIRR questions where a single benchmark line would be noise —
# leave those alone (a value question about ONE client's XIRR is the target).
_XIRR_MULTI_RE = re.compile(
    r"\b(all\s+clients?|every\s+client|each\s+client|clients'?|list\b|top\s+\d+|bottom\s+\d+|"
    r"average\s+xirr|avg\s+xirr|compare|comparison|across|breakdown|rank)\b", re.I)

_BENCHMARK_STEP_Q = "What is the XIRR of the Nifty 50 benchmark index?"
_BENCHMARK_SYNTHESIS = (
    "Show the client's XIRR with the Nifty 50 benchmark XIRR beside it WHEN AVAILABLE; if the "
    "benchmark came back empty, show ONLY the client's XIRR — do not fabricate or estimate the "
    "benchmark figure."
)


def _attach_benchmark(plan: dict, query: str) -> dict:
    """If the question is about a single client's XIRR value, ensure the plan also fetches the
    Nifty 50 1Y/3Y benchmark (client-independent, parallel) so the answer can show it alongside.
    No-op when the question is a list/aggregate of XIRR, or a benchmark step already exists. Caller
    must only invoke this when decomposition is enabled (this turns a direct plan into decompose)."""
    q = query or ""
    steps = plan.get("steps") or []
    mentions_xirr = bool(_XIRR_RE.search(q)) or any(_XIRR_RE.search(s.get("question", "")) for s in steps)
    if not mentions_xirr or _XIRR_MULTI_RE.search(q):
        return plan
    if any(_BENCH_RE.search(s.get("question", "")) for s in steps):
        return plan  # benchmark already covered
    bench_step = {"question": _BENCHMARK_STEP_Q, "depends_on": [], "fills": None, "for_each": False}
    synth = (plan.get("synthesis") or "").strip()
    synth = f"{synth} {_BENCHMARK_SYNTHESIS}".strip() if synth else _BENCHMARK_SYNTHESIS
    return {**plan, "strategy": "decompose",
            "reason": ((plan.get("reason") or "") + "+xirr-benchmark")[:200],
            "steps": list(steps) + [bench_step], "synthesis": synth}


# --- Solution A: client-level performance + holdings/details -> split ----------------------------
# The stateless platform returns ONE aspect per call, so a single fetch for "investment details
# including XIRR, holdings, AUM..." comes back as only one slice (observed: XIRR alone, dropping the
# holdings). When a query asks for a CLIENT-LEVEL performance metric (XIRR/CAGR/overall return)
# TOGETHER with a holdings/portfolio/investment-details ask, split it into a holdings step + a
# performance step so each is fetched cleanly and the synthesis merges them.
#
# Deliberately NARROW so it can't hamper other queries:
#   - needs BOTH a client-level performance term AND a holdings-family term — a lone "total AUM",
#     a lone "her XIRR", or a lone "her holdings" has only one group and never triggers;
#   - "AUM"/"profile"/"transactions" are NOT in the holdings-family set, so "AUM and XIRR" is left
#     untouched (direct);
#   - a per-holding XIRR COLUMN request (platform returns it inline with the holdings) is excluded;
#   - only ever turns a DIRECT plan into decompose, only when the mode enables decomposition, and
#     only for a single named client.
_CLIENT_PERF_RE = re.compile(
    r"\b(xirr|cagr|annuali[sz]ed\s+return|overall\s+return|portfolio\s+return|performance)\b", re.I)
_PER_HOLDING_PERF_RE = re.compile(
    r"\b(per[-\s]holding|for\s+each\s+holding|each\s+holding'?s|by\s+holding|holding[-\s]level|"
    r"per\s+instrument|for\s+each\s+instrument)\b", re.I)
_HOLDINGS_FAMILY_RE = re.compile(
    r"\b(holdings?|portfolio|investment\s+detail|investments|positions?|instruments?)\b", re.I)


def _split_performance_and_details(plan: dict, query: str, mc: dict | None, client: str) -> dict:
    """Solution A (see module note above). Turn a DIRECT plan into a 2-step decompose — a holdings
    overview step + a client-level performance step — when the query bundles both. No-op unless the
    plan is direct, decomposition is enabled, a single client is named, and both a client-level
    performance term and a holdings-family term are present (and performance isn't just a
    per-holding column)."""
    who = (client or "").strip()
    if not who:
        return plan  # only split a single named-client details+performance query
    if not plan or (plan.get("strategy") or "direct").lower() != "direct":
        return plan
    if not (mc or {}).get("decomposition_enabled", config.decomposition_enabled):
        return plan
    q = query or ""
    if not _CLIENT_PERF_RE.search(q) or not _HOLDINGS_FAMILY_RE.search(q):
        return plan
    if _PER_HOLDING_PERF_RE.search(q):
        return plan  # XIRR as a per-holding column -> platform returns it inline; don't split
    steps = [
        {"question": (f"Provide {who}'s portfolio / holdings overview by product with invested "
                      f"value, current value and unrealized gain."),
         "depends_on": [], "fills": None, "for_each": False},
        {"question": f"Provide {who}'s overall performance — XIRR and CAGR.",
         "depends_on": [], "fills": None, "for_each": False},
    ]
    synth = ("Combine the holdings/portfolio overview and the performance (XIRR/CAGR) into ONE "
             "answer — present the holdings breakdown and the performance figures together.")
    return {"strategy": "decompose",
            "reason": ((plan.get("reason") or "") + "+perf-details-split")[:200],
            "steps": steps, "synthesis": synth}


def _validate_plan(plan: QueryPlan, query: str, mc: dict | None = None) -> dict:
    """Convert + sanity-check the LLM plan: cap steps, drop empties, fix dangling deps, and fall
    back to DIRECT if a decompose plan doesn't actually have 2+ usable steps. ``mc`` is the Mode
    Profile's effective config (decomposition_enabled / max_subquestions); falls back to global."""
    mc = mc or {}
    decomposition_enabled = mc.get("decomposition_enabled", config.decomposition_enabled)
    max_subq = mc.get("max_subquestions", config.max_subquestions)
    steps_in = plan.steps or []
    steps: list[dict] = []
    for s in steps_in:
        q = (s.question or "").strip()
        if not q:
            continue
        # Only keep dependency indexes that point to an EARLIER, already-kept step.
        deps = [d for d in (s.depends_on or []) if isinstance(d, int) and 0 <= d < len(steps)]
        steps.append({"question": q, "depends_on": deps,
                      "fills": (s.fills or None), "for_each": bool(s.for_each)})
        if len(steps) >= max_subq:
            break

    strategy = (plan.strategy or "direct").strip().lower()
    if strategy != "decompose" or len(steps) < 2 or not decomposition_enabled:
        return _direct_plan(query, reason=plan.reason or "validated->direct")
    return {"strategy": "decompose", "reason": (plan.reason or "")[:200],
            "steps": steps, "synthesis": (plan.synthesis or "").strip()}


async def plan_query(state: dict) -> dict:
    """Return a validated plan dict for the (already enriched) question. Cheap path: planner
    disabled or an obviously-simple question -> DIRECT with no LLM call."""
    query = state.get("enriched_query") or state.get("query", "")
    start = time.perf_counter()

    # M1: honour the Mode Profile. Quick Facts disables decomposition, so skip the planner LLM
    # entirely and answer direct (no point planning a split we won't run).
    mc = state.get("mode_config") or {}
    planner_enabled = mc.get("planner_enabled", config.planner_enabled)
    decomposition_enabled = mc.get("decomposition_enabled", config.decomposition_enabled)
    # Solution A: the resolved client for a possible performance+details split (see below).
    _client = ((state.get("enrichment_entities") or {}).get("investor_name") or "").strip()

    if not planner_enabled or not decomposition_enabled or _looks_simple(query):
        reason = ("pre-gate-simple" if (planner_enabled and decomposition_enabled)
                  else ("decompose-disabled" if planner_enabled else "planner-disabled"))
        plan = _direct_plan(query, reason=reason)
        # A simple client-XIRR question still gets the benchmark pulled alongside it, but only when
        # the mode allows decomposition (Quick Facts disables it and stays single-shot).
        if decomposition_enabled:
            plan = _split_performance_and_details(plan, query, mc, _client)
            plan = _attach_benchmark(plan, query)
        plan["_planner_ms"] = round((time.perf_counter() - start) * 1000)
        return plan

    try:
        llm = make_llm("fast", temperature=0.0, max_tokens=600)
        structured = llm.with_structured_output(QueryPlan)
        plan_obj = await structured.ainvoke([
            SystemMessage(content=PLANNER_SYSTEM_PROMPT),
            HumanMessage(content=f"Question: {query}\n\nReturn the plan."),
        ])
        plan = _validate_plan(plan_obj, query, mc)
    except Exception:
        # Any planner failure -> safe fallback to the direct path (today's behaviour).
        plan = _direct_plan(query, reason="planner-error->direct")

    # Decomposition is enabled on this path (we passed the early gate). Split a bundled
    # performance+holdings ask (Solution A), then always pull the benchmark for a client-XIRR
    # question — whether the plan came back direct or decompose.
    plan = _split_performance_and_details(plan, query, mc, _client)
    plan = _attach_benchmark(plan, query)
    plan["_planner_ms"] = round((time.perf_counter() - start) * 1000)
    return plan


# --- Instrument at the bottom of the file ---
from ...observe_utils import observed_node as _observed_node


def _planner_detail(state, result):
    r = result if isinstance(result, dict) else {}
    steps = r.get("steps") or []
    return {
        "input": state.get("enriched_query") or state.get("query", ""),
        "output": {
            "strategy": r.get("strategy"),
            "reason": r.get("reason"),
            "steps": [s.get("question") if isinstance(s, dict) else s for s in steps],
        },
        "ns_strategy": str(r.get("strategy", "")),
        "ns_step_count": len(steps),
    }


plan_query = _observed_node(plan_query, name="Query Planner", detail=_planner_detail)
