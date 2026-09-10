"""Parallel init node — runs guardrail + memory retrieval concurrently.

Saves latency by overlapping the two independent operations.
Memory retrieval (~200-400ms) is fully hidden by guardrail (~3s).
Intent classification is handled downstream by intent_enrichment.
If guardrail blocks, the memory result is discarded.
"""

import asyncio
import os
import re
import threading
import time

# ── Dedicated guardrail event loop ────────────────────────────────────────────
# The guardrail's scanners (toxicity / NER / ONNX via transformers) run SYNCHRONOUS,
# CPU-bound inference. On the main request loop that blocks the event loop, which
# starves the concurrent answer track (enrichment's async LLM call gets no CPU to
# process its response → its measured time balloons from ~1.7s to ~3.5s). We run ALL
# guardrail work on its OWN loop in a daemon thread, so the main loop stays free for
# enrichment/generate/streaming. The GuardSession (and its async clients) is created
# on this loop and every guard_* call is dispatched to it, so there's no cross-loop
# binding issue.
_guard_loop: asyncio.AbstractEventLoop | None = None
_guard_loop_lock = threading.Lock()


def _ensure_guard_loop() -> asyncio.AbstractEventLoop:
    global _guard_loop
    if _guard_loop is None:
        with _guard_loop_lock:
            if _guard_loop is None:
                loop = asyncio.new_event_loop()
                threading.Thread(target=loop.run_forever, name="guardrail-loop",
                                 daemon=True).start()
                _guard_loop = loop
    return _guard_loop


async def _on_guard_loop(coro):
    """Run `coro` on the dedicated guardrail loop and await the result from the
    current loop (keeps the guardrail's blocking inference off the main loop)."""
    loop = _ensure_guard_loop()
    return await asyncio.wrap_future(asyncio.run_coroutine_threadsafe(coro, loop))


# ── Singleton guardrail session ───────────────────────────────────────────────
# GuardSession.__aenter__ loads ~500MB of ML models from disk (NER + toxicity).
# __aexit__ destroys them.  Creating a new session per request means a ~30s cold
# load on every request.  Instead we enter the context manager ONCE at startup
# and keep the session alive for the process lifetime; add_to_history=False keeps
# the call stateless so concurrent requests are safe.
_guardrail_session = None
_guardrail_lock: asyncio.Lock | None = None


def _get_guardrail_lock() -> asyncio.Lock:
    global _guardrail_lock
    if _guardrail_lock is None:
        _guardrail_lock = asyncio.Lock()
    return _guardrail_lock


async def init_guardrail_session():
    """Enter GuardSession once (ON THE GUARDRAIL LOOP) and cache it for the process
    lifetime. Called during lifespan startup so ML models are pre-loaded before the
    first user request. Subsequent calls are no-ops.
    """
    global _guardrail_session
    if _guardrail_session is not None:
        return _guardrail_session
    async with _get_guardrail_lock():
        if _guardrail_session is not None:
            return _guardrail_session
        from neo_guardrail_hub import GuardSession
        _AGENT_DIR = os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        )
        _CONFIGS_PATH = os.path.join(_AGENT_DIR, "configs")
        ctx = GuardSession(agent_id="wealth_advisor", config_path=_CONFIGS_PATH)
        # Enter on the guardrail loop so the session + its async clients are bound there.
        _guardrail_session = await _on_guard_loop(ctx.__aenter__())
    return _guardrail_session


async def run_guard_input(query: str):
    """Run guard_input on the guardrail loop (used by warmup + checks)."""
    session = await init_guardrail_session()
    return await _on_guard_loop(session.guard_input(query, add_to_history=False))


async def run_guard_output(text: str):
    """Run guard_output on the guardrail loop (used by warmup so the OUTPUT scanners —
    toxicity_output etc. — aren't cold-loaded on the first real response, which would
    otherwise hog the single guard loop for ~40s)."""
    session = await init_guardrail_session()
    return await _on_guard_loop(session.guard_output(text, add_to_history=False))

try:
    from ns_probe import observe
except ImportError:
    def observe(*a, **kw):
        def _id(fn): return fn
        return _id

import logging

from ...config import config
from ..state import AgentState
from .memory import memory_manager
# The answer track runs the enrichment node directly (intent_enrichment.py — NOT the
# legacy intent.py, which is only referenced by the unused graph.py).
from .intent_enrichment import intent_enrichment as _enrich_impl
from .stream_channel import emit_step as _emit_step
from .stream_channel import emit_detail as _emit_detail

logger = logging.getLogger(__name__)


# Analysis-pane "Context:" labels — one is shown per turn under "Analyzing your question",
# telling the RM where the answer/query is drawn from (in-session conversation, saved long-term
# memory, or a fresh platform fetch). Informational only; safe to remove if it feels noisy.
_CONTEXT_SOURCE_LABELS = {
    "new": "Context: New question — answered live from the platform (no prior turn used)",
    "followup": "Context: Resolved from this conversation (previous turn used) — fetched live",
    "recall_session": "Context: From this conversation — using the previous answer (no live fetch)",
    "recall_ltm": "Context: From saved memory — recalled a stored note/preference (no live fetch)",
    "general": "Context: General question — no client data used",
    "clarification": "Context: Needs clarification before fetching",
    "acknowledge": "Context: Noted — saving this to memory",
}


# Recall phrases that force a fresh memory fetch even after it's been loaded once.
_RECALL_PHRASES = (
    "what do you know about me", "remind me", "based on what you know",
    "my preferences", "my main financial concern", "what do you remember", "recall",
)


def _should_recall_memory(state) -> bool:
    """Fetch long-term memory only until it's been loaded once this session (tracked by
    the persisted `memory_loaded` flag), or when the user explicitly asks to recall. An
    empty-but-loaded result no longer re-triggers a fetch every turn."""
    already_loaded = state.get("memory_loaded", False)
    query = state.get("query", "").lower()
    explicit_recall = any(p in query for p in _RECALL_PHRASES)
    return (not already_loaded) or explicit_recall



# --- how the data step describes itself -------------------------------------------
# Both variants run this same node; only what sits beneath MCPClient.call_tool differs.
# The narration therefore has to be chosen at runtime, or the direct-SQL build tells the
# user it is "querying the platform" while it is in fact running SQL — which is exactly
# the kind of mislabelling that makes a trace untrustworthy.
def _direct_data() -> bool:
    import os
    return os.getenv("DATA_RESOLVER", "platform").strip().lower() == "direct"


def _fetch_labels() -> dict:
    if _direct_data():
        return {
            "querying": "Querying the database directly (model-written SQL)",
            "sending": "Asking the database: “{q}”",
            "succeeded": "{ok}/{n} database quer{y} succeeded",
        }
    return {
        "querying": "Querying the Cortex platform for live data",
        "sending": "Sending to platform: “{q}”",
        "succeeded": "{ok}/{n} platform quer{y} succeeded",
    }


async def _track_a(state: AgentState) -> dict:
    """GATE track: guardrail (+ first-turn memory), run concurrently. The guardrail
    here is authoritative — if it blocks, the answer track (B) is cancelled."""
    if _should_recall_memory(state):
        guard, mem = await asyncio.gather(_do_guardrail(state), _do_memory(state))
    else:
        guard = await _do_guardrail(state)
        mem = {"memory_context": state.get("memory_context", ""), "latency_breakdown": {}}
    return {"guard": guard, "mem": mem}


async def _track_b(state: AgentState) -> dict:
    """ANSWER-PREP track: enrichment → (mcp if data). Runs concurrently with the gate.
    Does NOT run generate — that happens AFTER the guardrail passes (in orchestrate),
    so the answer LLM streams live without ever leaking tokens for a blocked query.
    Imports are local to avoid an import cycle at module load."""
    from .mcp_fetch import mcp_fetch

    from ...modes import mode_config, QUICK_FACTS

    mc = mode_config(state.get("mode"))

    enr = await _enrich_impl(state)
    intent = enr.get("intent", "fetch_data")
    # Surface what the analysis actually resolved — the intent and any entities (client,
    # fund, metric, period) the enricher pulled out — so the live step shows real work.
    _ents = enr.get("enrichment_entities") or {}
    # Skip internal underscore-prefixed keys (e.g. _context_source, _note_scope) in the display.
    _ent_bits = [f"{k}: {v}" for k, v in _ents.items() if v and not k.startswith("_")]
    _emit_detail("intent_enrichment", f"Intent classified as “{intent}”")
    if _ent_bits:
        _emit_detail("intent_enrichment", "Resolved " + " · ".join(_ent_bits[:4]))
    # Context-source line (Analysis pane): where THIS turn is drawn from — a fresh platform
    # fetch, the in-session conversation, or saved long-term memory. Explicit branches
    # (follow-up / recall) set _context_source in enrichment; the rest fall back to the intent.
    _csrc = _ents.get("_context_source") or ""
    if not _csrc:
        if intent == "general":
            _csrc = "general"
        elif intent == "clarification":
            _csrc = "clarification"
        elif intent == "acknowledge":
            _csrc = "acknowledge"
        elif intent == "from_memory":
            _csrc = "recall_ltm" if (enr.get("memory_context") or "").strip() else "recall_session"
        else:
            _csrc = "new"
    _ctx_label = _CONTEXT_SOURCE_LABELS.get(_csrc)
    if _ctx_label:
        _emit_detail("intent_enrichment", _ctx_label)
    _emit_step("intent_enrichment")  # "Analyzing your question" — done
    # M1: thread the Mode Profile through — planner/execution/verify read their effective flags
    # from here (unknown/None mode → global defaults).
    sb = {**state, **{k: v for k, v in enr.items() if k != "latency_breakdown"}, "mode_config": mc}

    # M7: Client Insights scope lock — pin every question to the selected client. Inject the
    # client into the enriched query (so the platform scopes to them) and set it as the entity
    # (so name resolution + skill compilation target them), unless the query already names them.
    _sel_client = (state.get("selected_client") or "").strip()
    if mc.get("mode") == "client_insights" and _sel_client:
        _eq = sb.get("enriched_query") or state.get("query", "")
        # The enrichment step sometimes broadens "all <X>" into a book-wide / platform-wide scope —
        # "across all client portfolios", "across all internal investors" — which CONTRADICTS the
        # client-scope lock and, with the "internal investors" wording, intermittently makes the
        # platform return 0/empty. Strip that book-wide phrasing (clients OR investors, with or
        # without the internal brand) before pinning the question to the selected client/family
        # (e.g. "...transactions for the Krishnan family across all internal investors" ->
        # "...transactions for the Krishnan family"; "split across all client portfolios" -> "split").
        _eq = re.sub(
            r"\s*\b(across|for|of|over|among|in)\s+all\s+(of\s+)?(the\s+)?(neo\s*sapients?\s+)?"
            r"(client|clients|client's|clients'|investor|investors|investor's|investors')\s*"
            r"(portfolios?|accounts?|holdings?)?\b", " ", _eq, flags=re.IGNORECASE)
        # Collapse the whitespace / dangling punctuation the strip may leave behind.
        _eq = re.sub(r"\s+([.,?!])", r"\1", re.sub(r"\s{2,}", " ", _eq)).strip()
        # A FAMILY selection is a GROUP of clients, not one client — phrase it as "the Krishnan
        # family" (which the platform understands) and do NOT pin a single investor_name to it.
        if re.search(r"\bfamil(y|ies)\b", _sel_client, re.IGNORECASE):
            _base = re.sub(r"\s*\bfamil(y|ies)\b\.?\s*$", "", _sel_client, flags=re.IGNORECASE).strip()
            if _base and _base.lower() not in _eq.lower():
                _eq = f"{_eq.rstrip(' ?.')} (for the {_base} family)."
            # A FAMILY scope must NOT be pinned to a single member. The request still carries a
            # per-request investor_name fallback (the UI passes the previously-selected individual),
            # which enrichment injects when the question names no one — so the answer ends up
            # attributed to "Ram Krishnan" even though the whole family was selected. Clear it so
            # generate scopes the answer to the family, matching the platform query.
            _ents = dict(sb.get("enrichment_entities") or {})
            _ents["investor_name"] = None
            sb = {**sb, "enriched_query": _eq, "enrichment_entities": _ents}
        else:
            # If the question refers to the client by pronoun ("her portfolio"), splice the
            # selected client's name INLINE ("Sita Krishnan's portfolio") rather than tacking
            # on a "(for client …)" parenthetical and leaving the dangling pronoun — cleaner
            # and unambiguous for the stateless platform. Only fall back to the parenthetical
            # when there is no pronoun (and no name) to anchor the scope on.
            from .intent_enrichment import _substitute_pronoun, _PRONOUN_RE
            if _sel_client.lower() not in _eq.lower() and _PRONOUN_RE.search(_eq):
                _eq = _substitute_pronoun(_eq, _sel_client)
            if _sel_client.lower() not in _eq.lower():
                _eq = f"{_eq.rstrip(' ?.')} (for client {_sel_client})."
            _ents = dict(sb.get("enrichment_entities") or {})
            _ents["investor_name"] = _sel_client
            sb = {**sb, "enriched_query": _eq, "enrichment_entities": _ents}

        # orchestrate forwards `enr` (NOT sb) to generate, so mirror the scope-locked query +
        # entities back onto enr — otherwise generate attributes the answer using the pre-scope
        # entities (e.g. the per-request investor_name fallback), which is how a family-scoped
        # question got answered "For Ram Krishnan …".
        enr = {**enr, "enriched_query": sb.get("enriched_query", _eq),
               "enrichment_entities": sb.get("enrichment_entities") or {}}

    # M3: select the skill(s) for this turn (gated by mode + master flag). Brief dicts go to
    # generate; the Skill objects drive plan compilation below.
    from .skill_selection import select_skills
    active_skills = await select_skills(sb)
    sb = {**sb, "active_skills": [s.brief() for s in active_skills]}
    if active_skills:
        _emit_detail("intent_enrichment",
                     "Applying skill: " + ", ".join(getattr(s, "name", "skill") for s in active_skills[:3]))

    # Skill-over-smalltalk: a matched skill means the user explicitly wants that skill's
    # analysis, which needs a data fetch. The classifier sometimes mislabels a skill trigger
    # as acknowledge/general (e.g. "prepare me for my meeting with X") and would short-circuit
    # to a generic chit-chat reply, never running the skill. Give the skill priority:
    #   - a methodology skill (plan_steps, no client required) -> always force a fetch;
    #   - a client-scoped skill (requires_client) -> force a fetch ONLY when a client is
    #     actually resolved, so we never trigger a fetch we cannot fulfil.
    if active_skills and intent in ("acknowledge", "general"):
        from .skill_compiler import _resolve_client
        _sk = active_skills[0]
        _needs_client = getattr(_sk, "requires_client", False)
        if (_needs_client and _resolve_client(sb)) or (getattr(_sk, "plan_steps", None) and not _needs_client):
            intent = "fetch_data"
            sb = {**sb, "intent": "fetch_data"}

    # Analysis modes (Client/Deep): a question about a REAL client's portfolio that is phrased
    # causally/hypothetically ("how might rising rates affect their holdings?", "because of the
    # war what's most exposed?") gets mis-classified as 'general' (concept) or 'clarification',
    # which then skips the platform fetch and answers generically (or punts). When enrichment HAS
    # resolved a client AND a fetchable data_type, force a real data fetch instead.
    from ...modes import CLIENT_INSIGHTS, DEEP_INSIGHT
    if intent in ("general", "clarification") and mc.get("mode") in (CLIENT_INSIGHTS, DEEP_INSIGHT):
        _ent = sb.get("enrichment_entities") or {}
        _client = (_ent.get("investor_name") or "").strip() or (state.get("selected_client") or "").strip()
        _dtype = (_ent.get("data_type") or "").strip().lower()
        _FETCHABLE = ("portfolio", "holding", "holdings", "allocation", "exposure", "position",
                      "positions", "investment", "investments", "fund", "funds", "performance",
                      "returns", "value", "gain", "asset")
        if _client and any(t in _dtype for t in _FETCHABLE):
            intent = "fetch_data"
            sb = {**sb, "intent": "fetch_data"}

    if intent == "clarification":
        # Enrichment already produced the clarification message; no mcp/generate.
        return {"intent": intent, "enr": enr, "mcp": None, "sb": sb,
                "clarification": enr.get("messages", []), "active_skills": sb["active_skills"]}

    # Preference-aware fetch shaping: fold the RM's standing field/filter preferences
    # into the query BEFORE it hits MCP. The platform is stateless (it never sees the
    # preferences), so the only way a preference can change what data comes back is to
    # bake the required fields into the query string. Dedicated single-purpose pass —
    # the main enrichment LLM resolves references but won't reliably add fields. Runs
    # only for data turns AND only when the user actually has a stored summary, so
    # preference-less users pay no extra latency.
    # Quick Facts is a standalone fact lookup — never reshape its query with stored
    # preferences (the question is answered exactly as asked); Client/Deep keep this.
    # INTENT/SCOPE GATE: only reshape the query when the user has a stored preference AND it is
    # relevant to THIS query's data_type (deterministic match — no LLM call). This stops simple
    # lookups (e.g. "NAV of Fund X") and any data type the preferences don't cover from being
    # over-enriched: they reach MCP exactly as asked. Preference-less users were already untouched.
    if (intent not in ("general", "from_memory", "acknowledge", "clarification")
            and mc.get("mode") != QUICK_FACTS):
        try:
            from .memory import memory_manager
            _pref_summary = await memory_manager.get_summary(state["user_id"])
            _data_type = (sb.get("enrichment_entities") or {}).get("data_type")
            from .intent_enrichment import (
                augment_query_with_preferences, preference_augmentation_applies,
            )
            if _pref_summary and preference_augmentation_applies(_data_type, _pref_summary):
                _eq = sb.get("enriched_query") or state.get("query", "")
                _aug = await augment_query_with_preferences(_eq, _pref_summary)
                if _aug and _aug.strip() and _aug.strip() != (_eq or "").strip():
                    sb = {**sb, "enriched_query": _aug.strip()}
                    _emit_detail("intent_enrichment",
                                 "Context: your saved preferences applied to this query")
        except Exception:
            pass

    mcp = None
    plan = None
    if intent not in ("general", "from_memory", "acknowledge", "clarification"):
        # Phase 3 + M3: pick the plan. A matching methodology skill compiles straight into a
        # decomposition plan (Option B); otherwise the planner decides direct vs decompose.
        from .skill_compiler import compile_skill_plan
        plan = await compile_skill_plan(active_skills, sb)
        if not plan:
            # A plain breakdown / holdings / fund question is a SINGLE presentation fetch — keep
            # it direct (don't decompose) so generate can render the table in code. Only
            # interpretive/comparison variants (handled by _INTERPRETIVE_RE) fall through to the
            # planner. Holdings/fund are included only while their frames are enabled.
            from ...templates import classify_question, _INTERPRETIVE_RE, _HF_TEMPLATES_ENABLED
            _tq = sb.get("enriched_query") or state.get("query", "")
            _direct_kinds = ("asset_allocation",) + (("holdings", "fund") if _HF_TEMPLATES_ENABLED else ())
            if classify_question(_tq) in _direct_kinds and not _INTERPRETIVE_RE.search(_tq):
                from .planner import _direct_plan
                plan = _direct_plan(_tq)
            else:
                from .planner import plan_query
                plan = await plan_query(sb)
        sb = {**sb, "query_plan": plan}
        if plan is None:
            # Macro skill found no clear sector signal — skip the platform fetch; generate answers
            # against the live web context alone ("no clear signal yet").
            _emit_detail("data_fetch", "No clear sector signal — answering from live web context only")
        elif plan.get("strategy") == "decompose":
            _subs = plan.get("sub_questions") or plan.get("steps") or []
            # The destination is named from the resolver actually in use, not a literal.
            # This line previously said "cortex" in a build that queries Postgres directly,
            # which made the step trail look like the two variants shared a data path.
            _dest = ("the database directly"
                     if os.getenv("DATA_RESOLVER", "platform").strip().lower() == "direct"
                     else "the platform")
            _emit_detail("data_fetch",
                         f"Breaking into {len(_subs)} sub-question(s), querying {_dest}")
            from .execution import execute_plan
            mcp = await execute_plan(plan, sb)
            # Phase 4: coverage gate — one bounded retry of any sub-question that came back
            # empty, done here (pre-generate) so the answer is written exactly once.
            if mc.get("verify_enabled", config.verify_enabled) and mcp.get("mcp_tool_calls"):
                from .verify import backfill
                new_calls, retried = await backfill(mcp["mcp_tool_calls"], sb)
                if retried:
                    mcp = {**mcp, "mcp_tool_calls": new_calls}
        else:
            _emit_detail("data_fetch", _fetch_labels()["querying"])
            mcp = await mcp_fetch(sb)
        # Report what came back — count the ACTUAL platform queries sent (incl. retries /
        # refinements / sub-questions), matching the expandable query list, rather than just
        # the logical result count (which is always 1 for a direct fetch). Skipped when there was
        # no platform fetch (macro skill with no sector signal → mcp is None).
        if mcp is not None:
            _calls = mcp.get("mcp_tool_calls") or []
            from .stream_channel import platform_calls as _pcalls
            _hits = _pcalls.get() or []
            if _hits:
                _ok = sum(1 for c in _hits if c.get("ok"))
                _n = len(_hits)
                _emit_detail("data_fetch", _fetch_labels()["succeeded"].format(
                    ok=_ok, n=_n, y='y' if _n == 1 else 'ies'))
            elif _calls:
                _ok = sum(1 for c in _calls if c.get("success"))
                _emit_detail("data_fetch", f"{_ok}/{len(_calls)} platform call(s) succeeded")
        _emit_step("data_fetch")  # "Fetching live data" — done
        if mcp is not None:
            sb = {**sb, **{k: v for k, v in mcp.items() if k != "latency_breakdown"}}
            # Fund entity recovery (Phase 1) couldn't confidently resolve the fund and produced a
            # "did you mean ...?" question. Route it through the clarification short-circuit so the
            # message is returned DIRECTLY (generate is skipped and can't overwrite it).
            if mcp.get("fund_clarification"):
                return {"intent": "clarification", "enr": enr, "mcp": mcp, "sb": sb,
                        "clarification": mcp.get("messages", [])}
    return {"intent": intent, "enr": enr, "mcp": mcp, "sb": sb,
            "clarification": None, "query_plan": plan,
            "active_skills": sb.get("active_skills", [])}


def _collect_latency(base: dict, *parts) -> dict:
    """Merge the per-stage latency keys (guardrail_ms, memory_ms, enrichment_ms,
    mcp_exec_ms, llm_ms) from each sub-result's latency_breakdown into one dict."""
    out = dict(base)
    for p in parts:
        if p:
            out.update(p.get("latency_breakdown", {}))
    return out


async def orchestrate(state: AgentState) -> dict:
    """Two concurrent tracks, with generate gated on (and AFTER) the guardrail:

      Track A (gate)     : guardrail + memory
      Track B (prep)     : enrichment → mcp          (runs concurrently with A)
      then  generate     : streams the answer LIVE — only once the gate has PASSED

    enrichment+mcp overlap the guardrail, so the gate is hidden. generate runs after
    both, and because the guardrail verdict is already known by then it streams tokens
    live (no buffering) with no risk of leaking a blocked answer. cancel-on-block: if
    the guardrail blocks, Track B is cancelled and generate never runs. Single node so
    the checkpointer persists memory_loaded + accumulates messages.
    """
    from .generate import generate_response, generate_blocked_response

    start = time.perf_counter()
    base_latency = state.get("latency_breakdown", {})

    _emit_step("parallel_prep")  # "Checking content safety" — underway
    _emit_detail("parallel_prep", "Running input safety scan + loading memory (in parallel)")
    task_a = asyncio.create_task(_track_a(state))
    task_b = asyncio.create_task(_track_b(state))

    a = await task_a
    guard, mem = a["guard"], a["mem"]
    mem_ctx = mem.get("memory_context", state.get("memory_context", ""))
    # Raw memory items for the debug panel — fetched once here (track A) and surfaced
    # in the result, so main.py needn't run a second redundant retrieve.
    raw_mems = mem.get("long_term_memories", [])
    if guard.get("guardrail_passed", True):
        _emit_detail("parallel_prep", "Safety check passed")

    if not guard.get("guardrail_passed", True):
        # Guardrail blocked → cancel the prep track (aborts in-flight MCP) and return
        # the refusal. generate never ran, so no answer tokens were streamed.
        task_b.cancel()
        try:
            await task_b
        except (asyncio.CancelledError, Exception):
            pass
        blocked = await generate_blocked_response({**state, **guard})
        latency = _collect_latency(base_latency, guard, mem)
        latency["parallel_init_ms"] = round((time.perf_counter() - start) * 1000)
        return {
            **guard,
            "messages": blocked.get("messages", []),
            "memory_context": mem_ctx,
            "memory_loaded": True,
            "long_term_memories": raw_mems,
            "last_enriched_query": "",  # blocked turn produced no MCP query — don't leave a stale one
            "latency_breakdown": latency,
        }

    # Gate passed — wait for prep (enrichment+mcp), then run generate.
    b = await task_b
    enr = b.get("enr") or {}
    mcp = b.get("mcp")

    # Recall paths (recall_client / recall_personal → intent "from_memory") fetch a
    # query-SCOPED memory context inside enrichment and return it as enr["memory_context"].
    # Prefer it over Track A's generic, unscoped preload — otherwise the scoped recall
    # (e.g. "what do you remember about <client>") gets silently overwritten and generate
    # answers "I don't have a note" despite the memory existing.
    recall_ctx = enr.get("memory_context")
    effective_ctx = recall_ctx if (recall_ctx and recall_ctx.strip()) else mem_ctx

    # Carry the PARENT enriched query that actually hit MCP (post scope-lock, from sb — NOT the
    # pre-scope-lock enr value) so next turn's resolver sees the prior exchange's resolved form.
    # Only on a data turn; "" otherwise so it always reflects the IMMEDIATELY prior turn (no stale
    # value bleeding from an older data turn through an intervening general/clarify turn).
    _final_eq = (b.get("sb") or {}).get("enriched_query") or enr.get("enriched_query", "")
    _last_eq = _final_eq if b.get("intent") == "fetch_data" else ""

    base = {
        **guard,  # guardrail_passed=True, guardrail_result
        "intent": b.get("intent", "fetch_data"),
        "memory_context": effective_ctx,
        "memory_loaded": True,
        "long_term_memories": raw_mems,
        "enriched_query": enr.get("enriched_query", ""),
        "last_enriched_query": _last_eq,
        "enrichment_entities": enr.get("enrichment_entities", {}),
    }
    if mcp and mcp.get("mcp_tool_calls") is not None:
        base["mcp_tool_calls"] = mcp["mcp_tool_calls"]
    # Phase 3: pass the plan to generate so it can use the synthesis instruction when combining
    # the sub-answers of a decomposed question. Set it UNCONDITIONALLY (even to None/{}) so a
    # direct-fetch turn CLEARS any decomposition plan the previous turn left in the persisted
    # checkpointer state, instead of inheriting it.
    base["query_plan"] = b.get("query_plan")
    # M3: pass the active skill(s) so generate can apply the skill's methodology / output format.
    # Write UNCONDITIONALLY (even []) — like query_plan above — so a turn that selects NO skill
    # CLEARS the previous turn's skill from the persisted checkpointer state instead of inheriting
    # it. The old conditional write let a stale skill (e.g. the pre-meeting brief) bleed onto later
    # turns: "provide his aum" rendered as the brief skeleton with every section "Not available".
    base["active_skills"] = b.get("active_skills") or []

    # Clarification: enrichment already produced the question — no generate.
    if b.get("clarification"):
        latency = _collect_latency(base_latency, guard, mem, enr)
        latency["parallel_init_ms"] = round((time.perf_counter() - start) * 1000)
        return {**base, "messages": b["clarification"], "latency_breakdown": latency}

    # Run generate — it streams answer tokens live (gate already passed).
    _emit_step("generate_response")  # "Generating response" — underway
    _n_calls = len((mcp or {}).get("mcp_tool_calls") or []) if mcp else 0
    _emit_detail("generate_response",
                 f"Composing answer from {_n_calls} data source(s)" if _n_calls
                 else "Composing answer")
    sb = {**state, **mem, **{k: v for k, v in base.items() if k != "latency_breakdown"}}
    sb["memory_context"] = effective_ctx
    gen = await generate_response(sb)

    latency = _collect_latency(base_latency, guard, mem, enr, mcp, gen)
    latency["parallel_init_ms"] = round((time.perf_counter() - start) * 1000)

    merged = {**base, "messages": gen.get("messages", []), "latency_breakdown": latency}
    if gen.get("chart_spec"):
        merged["chart_spec"] = gen["chart_spec"]

    return merged


# When several guardrails fire at once (e.g. an SSN-laden off-topic query trips
# both pii_detection and nemo_self_check_input), surface the most actionable one.
_FAILURE_PRIORITY = ("pii", "injection", "toxic", "length", "domain", "topical", "self_check")


def _first_failure(result):
    """Return the most informative GuardrailResult that failed. Not results[0] —
    in parallel mode that may be a check that passed, and when multiple fail we
    prefer the most actionable category (PII > injection > toxicity > ...)."""
    failed = [r for r in (result.results or []) if not getattr(r, "passed", True)]
    if not failed:
        return (result.results or [None])[0]

    def rank(r):
        name = (getattr(r, "guardrail_name", "") or "").lower()
        for i, key in enumerate(_FAILURE_PRIORITY):
            if key in name:
                return i
        return len(_FAILURE_PRIORITY)

    return min(failed, key=rank)


async def _do_guardrail(state: AgentState) -> dict:
    """Internal: run input + context guardrails using the singleton session.

    Two layers gate the request BEFORE any generation work:
      1. guard_input   — prompt injection, PII, length, toxicity, nemo self-check
      2. guard_context — nemo topical/domain rail (keeps queries in the wealth domain)
    Either failure short-circuits to the blocked path. Both calls are stateless
    (add_to_history / no history override) so the shared singleton is concurrency-safe.
    """
    query = state["query"]
    start = time.perf_counter()

    # Kill-switch: bypass all guardrail layers without touching the ML models or the
    # self-check LLM. Used when the guardrail LLM provider is down (e.g. OpenAI 429).
    if not config.guardrails_enabled:
        return {
            "guardrail_passed": True,
            "guardrail_result": {"passed": True, "reason": "Guardrails disabled", "risk_score": 0.0},
            "latency_breakdown": {"guardrail_ms": 0},
        }

    try:
        session = await init_guardrail_session()

        # Run both guardrail layers CONCURRENTLY rather than sequentially:
        #   Layer 1 (guard_input)   — prompt injection, PII, length, toxicity, self-check
        #   Layer 2 (guard_context) — topical/domain rail, run statelessly (empty history)
        # They're independent and both stateless (add_to_history=False / history=[]), so
        # they overlap safely on the shared singleton. Dispatched onto the dedicated
        # guardrail loop so their blocking ML inference doesn't starve the main loop's
        # concurrent answer track. return_exceptions=True lets each layer apply its own
        # failure policy below (input fails closed; context fails open).
        async def _run_both():
            return await asyncio.gather(
                session.guard_input(query, add_to_history=False),
                session.guard_context(query, history=[]),
                return_exceptions=True,
            )
        result, ctx_result = await _on_guard_loop(_run_both())

        # Layer 1: input scanners — fail CLOSED. A raised exception is handled by the
        # outer except (session-level fail-open), so re-raise it.
        if isinstance(result, BaseException):
            raise result
        if not result.passed:
            elapsed_ms = round((time.perf_counter() - start) * 1000)
            failed = _first_failure(result)
            return {
                "guardrail_passed": False,
                "guardrail_result": {
                    "passed": False,
                    "blocked_by": getattr(failed, "guardrail_name", None),
                    "reason": getattr(failed, "message", None) or "Blocked by guardrail",
                    "risk_score": result.max_risk_score,
                },
                "latency_breakdown": {"guardrail_ms": elapsed_ms},
            }

        # Layer 2: context / topical-domain rail — fail OPEN. A provider/LLM error must
        # not block legitimate queries, so a raised exception here is ignored.
        if not isinstance(ctx_result, BaseException) and not ctx_result.passed:
            elapsed_ms = round((time.perf_counter() - start) * 1000)
            failed = _first_failure(ctx_result)
            return {
                "guardrail_passed": False,
                "guardrail_result": {
                    "passed": False,
                    "blocked_by": getattr(failed, "guardrail_name", None),
                    "reason": getattr(failed, "message", None) or "Off-topic for a wealth advisor",
                    "risk_score": ctx_result.max_risk_score,
                },
                "latency_breakdown": {"guardrail_ms": elapsed_ms},
            }

        elapsed_ms = round((time.perf_counter() - start) * 1000)
        return {
            "guardrail_passed": True,
            "guardrail_result": {"passed": True, "reason": None, "risk_score": 0.0},
            "latency_breakdown": {"guardrail_ms": elapsed_ms},
        }

    except Exception as e:
        elapsed_ms = round((time.perf_counter() - start) * 1000)
        return {
            "guardrail_passed": True,
            "guardrail_result": {"passed": True, "reason": f"Guardrail error (allowing): {e}", "risk_score": 0.0},
            "latency_breakdown": {"guardrail_ms": elapsed_ms},
        }


async def guard_output_text(text: str) -> dict:
    """Run output-layer guardrails on a generated response.

    Used as a post-stream check (tokens are already emitted live), so this can
    only flag/append a notice — it cannot un-send content. Returns
    {'passed', 'final_text', 'reason'}. Fails open on infrastructure errors.
    """
    if not config.guardrails_enabled:
        return {"passed": True, "final_text": text, "reason": "Guardrails disabled"}

    try:
        session = await init_guardrail_session()
        # add_to_history=False — never mutate the shared singleton's context.
        # Dispatched onto the guardrail loop (keeps its inference off the main loop).
        result = await _on_guard_loop(session.guard_output(text, add_to_history=False))
        return {
            "passed": result.passed,
            "final_text": result.final_text or text,
            "reason": None if result.passed else (
                result.results[0].message if result.results else "Blocked by output guardrail"
            ),
        }
    except Exception as e:
        return {"passed": True, "final_text": text, "reason": f"Output guardrail error (allowing): {e}"}


async def _do_memory(state: AgentState) -> dict:
    """Internal: retrieve relevant memories from Qdrant."""
    query = state["query"]
    user_id = state.get("user_id", "default_user")
    start = time.perf_counter()

    # One fetch returns both the context (for generate) AND the raw items (for the
    # debug panel) — so we don't run a second, redundant retrieve outside the graph.
    memory_context, raw_memories = await memory_manager.retrieve_with_raw(query, user_id)

    elapsed_ms = round((time.perf_counter() - start) * 1000)
    return {
        "memory_context": memory_context,
        "long_term_memories": raw_memories,
        "latency_breakdown": {"memory_ms": elapsed_ms},
    }


# --- Instrument at the bottom of the file ---
from ...observe_utils import observed_node as _observed_node


def _orchestrate_detail(state, result):
    r = result if isinstance(result, dict) else {}
    msgs = r.get("messages") or []
    answer = ""
    for m in reversed(msgs):
        answer = getattr(m, "content", None) or (m.get("content") if isinstance(m, dict) else None) or ""
        if answer:
            break
    return {
        "input": state.get("query", ""),
        "output": {
            "intent": r.get("intent") or state.get("intent"),
            "enriched_query": r.get("last_enriched_query"),
            "answer": answer,
        },
        "ns_intent": str(r.get("intent") or state.get("intent") or ""),
    }


orchestrate = _observed_node(orchestrate, name="Orchestrate", detail=_orchestrate_detail)
