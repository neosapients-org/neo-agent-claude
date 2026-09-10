"""
agent Agent — FastAPI server with SSE streaming.

ns_probe: configure() + instrument_all() at the top,
observe() wrappers applied at the bottom of the file.
"""

import os
import sys
import time
import uuid
import json
from pathlib import Path
import logging
import asyncio
from contextlib import asynccontextmanager, nullcontext

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from pydantic import BaseModel

from langchain_core.messages import HumanMessage

# --- .env must load BEFORE ns_probe is configured ---------------------------------
# configure() reads NS_PROBE_ENDPOINT / NS_PROBE_SERVICE_NAME from the ENVIRONMENT. Until
# now the only load_dotenv() call lived in app.config, which is imported ~40 lines below
# this point — so on a local run those values were still unset when configure() ran, and
# the agent silently used ns_probe's built-in defaults instead of the .env ones.
#
# It went unnoticed because in Docker compose puts them in the real environment, so the
# bug only appears outside containers — where it sends traces to the wrong collector under
# the wrong service name, with nothing to indicate it.
try:
    from dotenv import load_dotenv as _load_dotenv
    _load_dotenv()
except Exception:
    pass

# --- ns_probe: configure at the top, before other imports ---
# Observability ENABLED — traces export to NS_PROBE_ENDPOINT (set in docker-compose env,
# with OTEL_SERVICE_NAME). instrument_all() auto-instruments LangGraph/LangChain/OpenAI.
try:
    from ns_probe import (
        configure, observe, force_flush, get_tracer, SpanKind, StatusCode,
        # Agent-level surface: `turn` is the trace root for one exchange, `set_metrics`
        # attaches computed measurements, `outcome_ledger` writes the one-row-per-turn
        # summary the cost dashboards read. `clip` bounds free text before it becomes a
        # span attribute.
        turn, step, set_metrics, clip, as_bool, outcome_ledger, SpanContext,
        OUTCOME_OK, OUTCOME_BLOCKED_INPUT, OUTCOME_BLOCKED_OUTPUT, OUTCOME_ERROR,
    )
    from ns_probe.instrumentors import instrument_all
    configure()  # enabled by default; reads endpoint + service name from env
    instrument_all()
    NS_PROBE_ENABLED = True
    try:
        from ns_probe import get_config as _ns_get_config
        print(f"[ns_probe] tracing enabled — enabled={_ns_get_config().enabled}", flush=True)
    except Exception as _e:  # pragma: no cover
        print(f"[ns_probe] enable-check failed: {_e}", flush=True)
except ImportError:
    NS_PROBE_ENABLED = False
    from contextlib import contextmanager as _contextmanager

    def observe(*a, **kw):
        def _id(fn): return fn
        return _id
    def force_flush(): pass

    @_contextmanager
    def turn(*a, **kw): yield None

    @_contextmanager
    def step(*a, **kw): yield None

    def set_metrics(metrics): return False
    def clip(text, limit=2000): return (text or "")[:limit]
    def as_bool(value): return "true" if value else "false"

    class _NoLedger:
        @staticmethod
        def record(**kw): return False
    outcome_ledger = _NoLedger()
    SpanContext = None
    OUTCOME_OK, OUTCOME_BLOCKED_INPUT = "ok", "blocked_input"
    OUTCOME_BLOCKED_OUTPUT, OUTCOME_ERROR = "blocked_output", "error"
    get_tracer = None
    SpanKind = None
    StatusCode = None

from app.config import config
from app.llm import get_turn_usage, reset_turn_usage
from app.graph.graph_v2 import agent_graph
from app.graph.nodes import stream_channel as _stream
from app.graph.nodes.memory import memory_manager
from app.metrics.collector import metrics_collector, QueryMetrics


# --- Request/Response Models ---
class ChatRequest(BaseModel):
    message: str
    user_id: str = "default_user"
    investor_name: str | None = None
    # Human-readable label for the account running the request. Set on the root
    # trace span (user.name) so traces can be differentiated by user.
    user_name: str | None = None
    chat_history: list = []  # frontend session turns [{role, content}]
    # RM_SESSION_PREFERENCES — ephemeral text the RM typed into the preference box for
    # THIS session. Applied to the answer (highest-priority HOW-to-respond signal), but
    # NEVER auto-persisted to long-term memory. (Explicit "save this preference" → durable
    # storage is handled separately in Gap 3.)
    session_preferences: str | None = None
    # Per-session id. Used as the LangGraph checkpointer thread_id so conversation
    # memory is STRICTLY session-scoped — a new session starts with empty history (and
    # therefore re-loads Qdrant long-term memory on its first message). Long-term memory
    # itself stays namespaced by user_id. Falls back to user_id when absent.
    session_id: str | None = None
    # M1: chat mode chosen in the UI (quick / client / deep). Drives the Mode Profile —
    # absent/unknown falls back to today's default behaviour.
    mode: str | None = None
    # M2: when true, bypass the Quick Facts answer cache and force a live fetch.
    refresh: bool = False
    # M7: the client selected in Client Insights mode — every question is scoped to them.
    client: str | None = None


# Maximum conversation turns to keep in memory (last 5 exchanges = 10 messages)
MAX_MEMORY_MESSAGES = 10

# Fire-and-forget background tasks (e.g. post-response memory store). We hold a
# reference so they aren't garbage-collected before completing.
_bg_tasks: set = set()


def _run_in_background(coro):
    """Schedule a coroutine without blocking, keeping a strong reference to it."""
    task = asyncio.create_task(coro)
    _bg_tasks.add(task)
    task.add_done_callback(_bg_tasks.discard)
    return task


async def _audit_output_guardrails(text: str, message_id: str):
    """Run output-layer guardrails AFTER the stream's `done` event (audit only).

    The LLM-Guard scanners (pii_redaction / toxicity_output / ban_topics) are
    CPU-bound and run synchronously, so doing this inline before `done` froze the
    UI for ~5-8s with the answer already on screen. Tokens are streamed live and
    can't be un-sent, so the post-check is best-effort: we just log a flag.
    """
    try:
        from app.graph.nodes.parallel import guard_output_text
        res = await guard_output_text(text)
        if not res.get("passed", True):
            logging.getLogger("agent.guardrail").warning(
                "output_guardrail_flagged message_id=%s reason=%s",
                message_id, res.get("reason"),
            )
    except Exception:
        pass


# --- Lifespan ---
async def _warmup_guardrails():
    """Warm up guardrail models in background — prevents long startup delay."""
    try:
        from neo_guardrail_hub import GuardSession
        _configs = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "configs")
        async with GuardSession(agent_id="wealth_advisor", config_path=_configs) as session:
            await session.guard_input("warmup test")
        print("[INFO] Guardrail models warmed up")
    except Exception as e:
        print(f"[WARN] Guardrail warmup failed: {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize services on startup — memory init + front-loads + guardrail warmup.

    All startup work runs under ONE ``Startup front-load`` root span so the otherwise-orphan
    child spans it spawns (skill methodology-compile llm.calls, the Cortex context-map MCP
    calls, etc.) stitch into a single startup trace instead of scattering as loose roots.
    ns_probe propagates the span via a ContextVar, so everything awaited inside auto-parents.
    """
    print("[INFO] agent Agent starting up...")

    startup_span = nullcontext()
    if NS_PROBE_ENABLED:
        startup_span = get_tracer("agent.startup").start_span(
            "Startup front-load", kind=SpanKind.INTERNAL,
            attributes={"service.phase": "startup"},
        )

    with startup_span:
        # Initialize Qdrant memory connection (no-op when MEMORY_ENABLED=false)
        try:
            await memory_manager.initialize()
            if config.memory_enabled:
                print("[INFO] Qdrant memory initialized")
            else:
                print("[INFO] Memory DISABLED (MEMORY_ENABLED=false)")
        except Exception as e:
            print(f"[WARN] Memory init failed (will retry on first use): {e}")

        # Front-load domain skills from the Cortex platform (MCP list_skills) into the RAM
        # registry so the first turn already has the full skill catalog. A failure here is
        # non-fatal — the registry keeps its bundled seeds and retries on the next TTL refresh.
        if config.skills_enabled and config.skills_mcp_sync:
            # The platform MCP key can transiently 401 on a cold call and recover immediately,
            # so retry a few times before giving up — otherwise the registry sits empty until
            # the next TTL refresh. A persistent failure is non-fatal: no skill applies until then.
            from app.skills.registry import registry
            info = {"count": 0, "source": "empty"}
            for attempt in range(1, 4):
                try:
                    info = await registry.force_refresh()
                    if info["count"] > 0:
                        break
                except Exception as e:
                    print(f"[WARN] Skill front-load attempt {attempt}/3 failed: {e}")
                await asyncio.sleep(1.5)
            print(f"[INFO] Skills front-loaded: {info['count']} skill(s) from {info['source']}")

        # Front-load the Cortex context map (capabilities + client roster + dimensions) so the
        # first turn already resolves names / scopes clarifications from LIVE platform data.
        # Non-fatal: a failure leaves the seed roster in place and retries on the TTL refresh.
        if config.cortex_context_sync:
            from app.context import cortex_context
            ctx_info = {}
            for attempt in range(1, 4):
                try:
                    ctx_info = await cortex_context.force_refresh()
                    if ctx_info.get("clients") and cortex_context.source == "mcp":
                        break
                except Exception as e:
                    print(f"[WARN] Context front-load attempt {attempt}/3 failed: {e}")
                await asyncio.sleep(1.5)
            print(f"[INFO] Cortex context front-loaded: {ctx_info}")

        # Warm up guardrail models — enters the singleton session so ML models are
        # loaded into memory before the first user request arrives.
        # We run warmup with a generous timeout since model loading can take 2+ min.
        # Skipped entirely when guardrails are disabled (no point loading the models).
        if not config.guardrails_enabled:
            print("[INFO] Guardrails DISABLED (GUARDRAILS_ENABLED=false) — skipping warmup")
        else:
            try:
                from app.graph.nodes.parallel import init_guardrail_session, run_guard_input, run_guard_output
                await init_guardrail_session()
                # Run warmup without timeout constraint — let all models fully load.
                # run_guard_input dispatches onto the dedicated guardrail loop (where the
                # session lives) and triggers lazy init of ALL scanners (PII, toxicity, etc).
                warmup_inputs = ["warmup test", "hello world check", "financial portfolio review"]
                for w in warmup_inputs:
                    try:
                        await asyncio.wait_for(run_guard_input(w), timeout=180.0)
                    except asyncio.TimeoutError:
                        print(f"[WARN] Guardrail warmup timed out for '{w}' — continuing")
                # Also warm the OUTPUT scanners (toxicity_output etc.) — otherwise the first
                # real response cold-loads them (~40s) and hogs the single guard loop.
                try:
                    await asyncio.wait_for(run_guard_output("Your portfolio summary is ready."), timeout=180.0)
                except asyncio.TimeoutError:
                    print("[WARN] Output-guardrail warmup timed out — continuing")
                print("[INFO] Guardrail models warmed up")
            except Exception as e:
                print(f"[WARN] Guardrail warmup failed: {e}")

    yield


# --- App ---
app = FastAPI(
    title="agent Agent",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- SSE Streaming Endpoint ---
def _store_attribution(result: dict, default_investor: str | None) -> tuple[str | None, bool]:
    """Decide whose long-term memory a turn's facts belong to, from the routing result.

    - client_note (_note_scope='investor'): the user stated facts ABOUT a named client, so
      attribute even preference/persona facts to that resolved client (force_investor_scope).
    - advisor preference (_note_scope='advisor'): the advisor's own — advisor-scoped (no client).
    - otherwise: the request-level investor (unchanged behavior for data turns)."""
    ents = result.get("enrichment_entities") or {}
    scope = ents.get("_note_scope")
    if scope == "investor":
        return (ents.get("investor_name") or default_investor), True
    if scope == "advisor":
        return None, False
    return default_investor, False


def _is_skill_turn(result: dict) -> bool:
    """True when a skill drove this turn. Used to keep the skill's generated output out of
    long-term memory extraction (the RM's own message is still mined)."""
    return bool(result.get("active_skills") or result.get("skill_plan"))


def _write_turn_ledger_row(span, question, answer, latency_ms, ttft_ms,
                           session_id, user_id, outcome) -> None:
    """One Outcome Ledger row per turn — the summary the cost dashboards read.

    `parent` is passed explicitly, and it matters: this row is written from inside an
    async generator, where ContextVar-based parenting does not survive a `yield`. Without
    it the row starts a trace of its own and loses the join to the very spans it
    summarises, so a turn's cost can no longer be tied to the turn.

    Scope is passed explicitly for the same underlying reason. Attributes are not
    inherited through a parent SpanContext — only through the ambient ContextVar — so in
    exactly the case `parent` exists for, `tenant.id` and `workspace.id` go missing.

    Token and cost columns are deliberately left UNSET. They are summed downstream from
    the `llm.call` spans beneath this turn, which is the only place that knows what each
    model call actually cost. A second, agent-side estimate written here would give the
    dashboard two numbers that disagree, and no way to tell which is wrong.

    `outcome_ledger.record` never raises — a turn that answered the user correctly must
    not be reported as failed because its bookkeeping could not be written.
    """
    parent = None
    if span is not None and SpanContext is not None:
        try:
            parent = SpanContext(trace_id=span.trace_id, span_id=span.span_id)
        except Exception:
            parent = None

    outcome_ledger.record(
        parent=parent,
        question=clip(question),
        answer=clip(answer) if answer else None,
        outcome=outcome,
        session_id=session_id or "",
        user_id=user_id,
        surface="http",
        latency_ms=latency_ms,
        ttft_ms=ttft_ms,
        llm_model=os.getenv("LLM_MODEL", ""),
        llm_provider=os.getenv("LLM_PROVIDER", ""),
        tenant_id=os.getenv("NS_PROBE_TENANT_ID", ""),
        workspace_id=os.getenv("NS_PROBE_WORKSPACE_ID", ""),
        environment=os.getenv("NS_PROBE_ENVIRONMENT", ""),
    )


async def run_agent_stream(message: str, user_id: str, investor_name: str | None, chat_history: list = [], session_id: str | None = None, session_preferences: str | None = None, mode: str | None = None, refresh: bool = False, client: str | None = None, user_name: str | None = None):
    """Run the agent graph and yield SSE events."""
    # instrument_all() auto-instruments LangGraph/LangChain and the provider SDK. The
    # root span is created here (rather than by a decorator) so it stays open for the
    # whole SSE stream — the context manager is entered inside this async generator.
    #
    # `turn()` rather than a bare start_span, for two reasons it documents itself:
    #   - it always starts a NEW trace. The span stack lives in a mutable list inside a
    #     ContextVar, and a ContextVar isolates rebinds but not mutations — so without
    #     root=True two concurrent requests can parent to each other and fuse two users'
    #     conversations into one trace.
    #   - it publishes the turn's identity as ambient, so session.id / user.id land on
    #     every span opened while the turn is live, including spans in libraries we
    #     cannot pass anything to.

    message_id = str(uuid.uuid4())
    start_time = time.perf_counter()

    # Zero the per-turn token counters before any model call runs. The comparison UI reads
    # these back on the `usage` event below; without the reset a turn would inherit the
    # previous turn's totals and every cost shown would drift upward.
    reset_turn_usage()

    span_ctx = turn(
        "agent.turn",
        agent_name=os.getenv("NS_PROBE_SERVICE_NAME", "agent"),
        session_id=session_id or "",
        user_id=user_id,
        surface="http",
        attributes={
            "user.name": user_name or user_id,
            "request.message_id": message_id,
            "request.mode": mode or "",
            "request.client": client or "",
            "input.value": clip(message),
            "turn.message_chars": len(message or ""),
        },
    )

    initial_state = {
        "messages": [HumanMessage(content=message)],
        "query": message,
        "user_id": user_id,
        "intent": None,
        "mcp_tool_calls": [],
        "guardrail_result": {},
        "guardrail_passed": True,
        "latency_breakdown": {},
        "investor_name": investor_name,
        "chat_history": chat_history,
        "session_preferences": session_preferences or "",
        "mode": mode,
        "refresh": refresh,
        "selected_client": client,
        # memory_context intentionally omitted — MemorySaver preserves
        # the value fetched on turn-1 across all subsequent turns.
    }

    # LangGraph checkpointer config — thread_id groups conversation history per SESSION
    # (falls back to user_id if the caller didn't supply a session id).
    graph_config = {"configurable": {"thread_id": session_id or user_id}}

    with span_ctx as span:
        try:
            # The debug-panel memory items now ride out of the graph itself (track A's
            # once-per-session fetch returns them as result["long_term_memories"]). We no
            # longer kick off a SECOND retrieve here — on the first turn that duplicate
            # embed+search ran concurrently with the graph, contended for the same memory
            # backend, and delayed the graph's start by ~5-8s (inflating total/ttft far
            # past parallel_init). Sourcing from the graph removes that gap.

            # The graph orchestrates two concurrent tracks (guardrail+memory gate ||
            # enrichment→mcp). generate runs only AFTER the gate passes and streams the
            # answer LIVE. The orchestrator/generate push events to this queue — step
            # markers (progressive ticking) and answer tokens (live streaming) — which
            # we relay to SSE as they arrive.
            event_q: asyncio.Queue = asyncio.Queue()
            _stream.event_queue.set(event_q)
            # Fresh per-request platform-call log — the graph task (created below) inherits
            # this contextvar, and mcp_client.call_tool appends every call to it.
            platform_call_log: list = []
            _stream.platform_calls.set(platform_call_log)

            graph_task = asyncio.create_task(agent_graph.ainvoke(initial_state, config=graph_config))

            first_token_ms: int | None = None
            streamed_chunks = []

            # Accumulate the live thinking trace (ordered steps + their detail lines) so it
            # can be persisted alongside the answer and replayed when a session is reloaded
            # — otherwise only the latency keys survive and the sub-status lines are lost.
            thinking_trace: list = []

            def _trace_step(tool):
                s = next((x for x in thinking_trace if x["tool"] == tool), None)
                if s is None:
                    s = {"tool": tool, "detail": []}
                    thinking_trace.append(s)
                return s

            async def _relay(ev):
                nonlocal first_token_ms
                if ev.get("type") == "step":
                    _trace_step(ev["tool"])
                    return f"event: tool_start\ndata: {json.dumps({'tool': ev['tool']})}\n\n"
                if ev.get("type") == "detail":
                    if ev.get("text"):
                        _trace_step(ev["tool"])["detail"].append(ev["text"])
                    return f"event: step_detail\ndata: {json.dumps({'tool': ev['tool'], 'text': ev['text']})}\n\n"
                if ev.get("type") == "token":
                    if first_token_ms is None:
                        first_token_ms = round((time.perf_counter() - start_time) * 1000)
                    streamed_chunks.append(ev["content"])
                    return f"event: token\ndata: {json.dumps({'content': ev['content']})}\n\n"
                return None

            while not graph_task.done():
                try:
                    ev = await asyncio.wait_for(event_q.get(), timeout=0.1)
                    sse = await _relay(ev)
                    if sse:
                        yield sse
                except asyncio.TimeoutError:
                    pass
            while not event_q.empty():
                sse = await _relay(event_q.get_nowait())
                if sse:
                    yield sse

            result = await graph_task

            latency_breakdown = dict(result.get("latency_breakdown", {}) or {})
            mcp_tool_calls = result.get("mcp_tool_calls", []) or []
            blocked_by_guardrail = result.get("guardrail_passed", True) is False

            # Answer text: prefer the live-streamed chunks; otherwise (blocked or
            # clarification — no generate stream) pull the message from the final state
            # and emit it as one token so the UI still shows it.
            output_text = "".join(streamed_chunks)
            if not output_text:
                for msg in reversed(result.get("messages", []) or []):
                    if getattr(msg, "type", None) == "ai" and getattr(msg, "content", None):
                        output_text = msg.content
                        break
                if output_text:
                    if first_token_ms is None:
                        first_token_ms = round((time.perf_counter() - start_time) * 1000)
                    yield f"event: token\ndata: {json.dumps({'content': output_text})}\n\n"

            if first_token_ms is None:
                first_token_ms = round((time.perf_counter() - start_time) * 1000)

            # Emit a chart spec (issue #2) after the answer, before latency/done.
            if result.get("chart_spec"):
                yield f"event: chart\ndata: {json.dumps({'spec': result['chart_spec']})}\n\n"

            total_ms = round((time.perf_counter() - start_time) * 1000)
            latency_breakdown["total_ms"] = total_ms
            latency_breakdown["ttft_ms"] = first_token_ms

            # Attach EVERY platform query (incl. retries / sub-questions / fund-resolution
            # probes) to the mcp_fetch step, so the UI can expand it to show everything that
            # actually hit the platform — not just the final per-sub-question results.
            queries = [
                {"q": c.get("q"), "tool": c.get("tool"), "ok": bool(c.get("ok")), "ms": c.get("ms")}
                for c in platform_call_log
                if isinstance(c, dict) and c.get("q")
            ]
            # Fallback: nothing captured but a single logical call ran — show its resolved query.
            if not queries and mcp_tool_calls:
                q = result.get("enriched_query") or message
                if q:
                    queries = [{"q": q, "tool": (mcp_tool_calls[0] or {}).get("tool"),
                                "ok": bool((mcp_tool_calls[0] or {}).get("success")), "ms": None}]
            if queries:
                _trace_step("data_fetch")["queries"] = queries

            # Persist the thinking trace by riding it inside the latency JSON (reserved
            # `_steps` key) — the backend stores latency verbatim, so no schema change is
            # needed and a reloaded session replays the full step + detail + query view.
            if thinking_trace:
                latency_breakdown["_steps"] = thinking_trace

            # Emit latency event
            yield f"event: latency\ndata: {json.dumps(latency_breakdown)}\n\n"

            if span is not None:
                span.set_attributes({
                    "traceloop.entity.input": message,
                    "traceloop.entity.output": output_text,
                })

            # Emit long-term memories for the debug panel — sourced from the graph's
            # once-per-session memory fetch (result["long_term_memories"]), so no extra
            # retrieve and no added wait. Populated only on the turn the fetch runs (first
            # turn / explicit recall); empty otherwise, matching the prior behaviour.
            raw_mems = result.get("long_term_memories") or []
            if raw_mems:
                yield f"event: long_term_memories\ndata: {json.dumps({'results': raw_mems})}\n\n"

            # Emit done — the stream closes right after this, so the UI re-enables the
            # input immediately instead of waiting on the guardrail/memory work below.
            # Token totals for this turn, straight from the SDK's usage fields. Emitted
            # before `done` so the comparison UI can show cost beside the answer instead
            # of waiting on the observability pipeline, which lags 20-36s from span to
            # queryable row. Pricing is applied in the UI, deliberately: the agent must
            # not publish a second cost number that can disagree with the dashboard's.
            _usage = get_turn_usage()
            _usage["model"] = os.getenv("LLM_MODEL") or ""
            _usage["ttft_ms"] = first_token_ms
            _usage["total_ms"] = round((time.perf_counter() - start_time) * 1000)
            yield f"event: usage\ndata: {json.dumps(_usage)}\n\n"

            yield f"event: done\ndata: {json.dumps({'message_id': message_id})}\n\n"

            # Output-layer guardrails run in the BACKGROUND, AFTER `done`. The scanners
            # are CPU-bound (~5-8s) and synchronous; running them before `done` froze the
            # UI with the answer already on screen. Post-`done` they're audit-only.
            # Skip entirely when the input was refused (the refusal text isn't a "response").
            if output_text and not blocked_by_guardrail:
                _run_in_background(_audit_output_guardrails(output_text, message_id))

            # Store memory in the BACKGROUND — do NOT await it here (the embed + Qdrant
            # upsert takes ~1-3s and would keep the stream open / input disabled).
            if output_text and not blocked_by_guardrail:
                _store_inv, _force_scope = _store_attribution(result, investor_name)
                _run_in_background(memory_manager.store(message, output_text, user_id, _store_inv, force_investor_scope=_force_scope, exclude_response=_is_skill_turn(result)))

            # Record metrics
            metrics_collector.record(QueryMetrics(
                query=message,
                total_latency_ms=total_ms,
                latency_breakdown=latency_breakdown,
                success=True,
                timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                tool_calls=mcp_tool_calls,
            ))

            # The same numbers onto the turn span, where a trace reader can see them
            # next to the spans that produced them. metrics_collector is in-process
            # only and dies with the container.
            set_metrics({
                "latency_ms": total_ms,
                "ttft_ms": first_token_ms,
                "platform_calls": len(mcp_tool_calls or []),
                "blocked": as_bool(blocked_by_guardrail),
            })

            _write_turn_ledger_row(
                span, message, output_text, total_ms, first_token_ms,
                session_id, user_id,
                OUTCOME_BLOCKED_INPUT if blocked_by_guardrail else OUTCOME_OK,
            )

        except Exception as e:
            if span is not None:
                span.record_exception(e)
                span.set_status(StatusCode.ERROR, str(e))
                span.set_attribute("traceloop.entity.input", message)
            yield f"event: error\ndata: {json.dumps({'message': str(e)})}\n\n"
            total_ms = round((time.perf_counter() - start_time) * 1000)
            metrics_collector.record(QueryMetrics(
                query=message,
                total_latency_ms=total_ms,
                latency_breakdown={},
                success=False,
                timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            ))
            # A failed turn still gets a ledger row — a turn that errored and a turn
            # that never happened look identical otherwise, and error rate is one of
            # the things the comparison is for.
            _write_turn_ledger_row(
                span, message, None, total_ms, None,
                session_id, user_id, OUTCOME_ERROR,
            )
        finally:
            # Flush traces off the critical path — force_flush() blocks on the OTLP
            # export (~1s), which would otherwise keep the stream open and the UI input
            # disabled. The periodic exporter still flushes on its own interval too.
            try:
                _run_in_background(asyncio.to_thread(force_flush))
            except Exception:
                pass


@app.post("/chat/stream")
async def chat_stream(req: ChatRequest):
    """SSE streaming endpoint."""
    return StreamingResponse(
        run_agent_stream(req.message, req.user_id, req.investor_name, req.chat_history, req.session_id, req.session_preferences, req.mode, req.refresh, req.client, req.user_name),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/chat")
async def chat(req: ChatRequest):
    """Non-streaming fallback endpoint."""
    start_time = time.perf_counter()

    initial_state = {
        "messages": [HumanMessage(content=req.message)],
        "query": req.message,
        "user_id": req.user_id,
        "intent": None,
        "mcp_tool_calls": [],
        "guardrail_result": {},
        "guardrail_passed": True,
        "latency_breakdown": {},
        "investor_name": req.investor_name,
        "chat_history": req.chat_history,
        "session_preferences": req.session_preferences or "",
        "mode": req.mode,
        "refresh": req.refresh,
        "selected_client": req.client,
        # memory_context intentionally omitted — MemorySaver preserves
        # the value fetched on turn-1 across all subsequent turns.
    }

    graph_config = {"configurable": {"thread_id": req.session_id or req.user_id}}

    try:
        # Note: instrument_all() auto-instruments LangGraph via OpenLLMetry.
        # No manual span needed — it would create a duplicate root trace.
        result = await agent_graph.ainvoke(initial_state, config=graph_config)
        total_ms = round((time.perf_counter() - start_time) * 1000)
        latency_breakdown = result.get("latency_breakdown", {})
        latency_breakdown["total_ms"] = total_ms

        # Get the last AI message
        messages = result.get("messages", [])
        response_text = ""
        for msg in reversed(messages):
            if hasattr(msg, "content") and msg.type == "ai":
                response_text = msg.content
                break

        metrics_collector.record(QueryMetrics(
            query=req.message,
            total_latency_ms=total_ms,
            latency_breakdown=latency_breakdown,
            success=True,
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            tool_calls=result.get("mcp_tool_calls", []),
        ))

        # Output-layer guardrails. Non-streaming, so we can apply the sanitized
        # text (e.g. PII redaction) or replace blocked content before returning.
        # Detect an input-block via the structured guardrail flag (not a brittle
        # string match) so the refusal message isn't re-checked/overwritten.
        guardrail_blocked = result.get("guardrail_passed", True) is False
        if response_text and not guardrail_blocked:
            from app.graph.nodes.parallel import guard_output_text
            out_guard = await guard_output_text(response_text)
            if not out_guard["passed"]:
                response_text = (
                    "I'm sorry, I can't share that response. "
                    f"It was flagged by an output guardrail ({out_guard['reason']})."
                )
                guardrail_blocked = True
            else:
                # use sanitized text (PII redaction may have rewritten it)
                response_text = out_guard["final_text"]

        # Store memory — runs after response is built, before returning
        if response_text and not guardrail_blocked:
            _store_inv, _force_scope = _store_attribution(result, req.investor_name)
            await memory_manager.store(req.message, response_text, req.user_id, _store_inv, force_investor_scope=_force_scope, exclude_response=_is_skill_turn(result))

        # Retrieve long-term memories for debug panel — only on the first turn of a
        # session (empty chat_history); the set is stable within a session, so re-running
        # the full embed+search every turn just duplicates work.
        raw_mems = []
        if not req.chat_history:
            try:
                raw_mems = await memory_manager.retrieve_raw(req.message, req.user_id)
            except Exception:
                pass

        return JSONResponse({
            "message_id": str(uuid.uuid4()),
            "response": response_text,
            "latency_breakdown": latency_breakdown,
            "guardrail_result": result.get("guardrail_result", {}),
            "mcp_tool_calls": result.get("mcp_tool_calls", []),
            "chart_spec": result.get("chart_spec"),
            "long_term_memories": raw_mems,
        })

    except Exception as e:
        return JSONResponse(
            {"error": str(e)},
            status_code=500,
        )
    finally:
        force_flush()


@app.delete("/memory/clear/{user_id}")
async def clear_memory(user_id: str):
    """Delete all long-term (Qdrant) memories for a user namespace.

    Accepts any non-empty user namespace (the backend already derives this from the
    authenticated user, so a caller can only ever clear their own memories). The old
    hard-coded user_1/user_2 allowlist silently rejected every other account, which made
    the debug-panel "Clear Long-Term Memory" button fail for real users.
    """
    user_id = (user_id or "").strip()
    if not user_id:
        return JSONResponse({"error": "Missing user_id"}, status_code=400)
    success = await memory_manager.clear_user_memories(user_id)
    if success:
        return JSONResponse({"success": True, "user_id": user_id})
    return JSONResponse({"error": "Failed to clear memories"}, status_code=500)


class SavePreferenceRequest(BaseModel):
    user_id: str
    text: str


@app.post("/memory/preference")
async def save_preference(req: SavePreferenceRequest):
    """Gap 3 — persist RM-typed preference text as DURABLE advisor preferences in Qdrant.

    Applies conservative UPSERT-by-slot in the store pipeline (an updated value supersedes
    the same-slot preference instead of duplicating)."""
    user_id = (req.user_id or "").strip()
    text = (req.text or "").strip()
    if not user_id or not text:
        return JSONResponse({"error": "Missing user_id or text"}, status_code=400)
    result = await memory_manager.save_preference(text, user_id)
    return JSONResponse({"success": True, "user_id": user_id, **result})


class DeleteMemoryRequest(BaseModel):
    user_id: str
    memory_id: str


@app.post("/memory/delete")
async def delete_memory_item(req: DeleteMemoryRequest):
    """Delete a single stored memory, scoped to the requesting user's namespace."""
    user_id = (req.user_id or "").strip()
    memory_id = (req.memory_id or "").strip()
    if not user_id or not memory_id:
        return JSONResponse({"error": "Missing user_id or memory_id"}, status_code=400)
    ok = await memory_manager.delete_memory(memory_id, user_id)
    if ok:
        return JSONResponse({"success": True, "memory_id": memory_id})
    return JSONResponse({"error": "Memory not found or not owned by user"}, status_code=404)


@app.get("/memory/list/{user_id}")
async def list_memory(user_id: str):
    """List ALL stored long-term memories for a user (debug panel 'Stored (all)').

    Unlike the per-message retrieval shown during chat, this returns the full
    contents of Qdrant for the user so the panel reflects what is actually stored.
    """
    user_id = (user_id or "").strip()
    if not user_id:
        return JSONResponse({"error": "Missing user_id"}, status_code=400)
    mems = await memory_manager.list_all_memories(user_id)
    return JSONResponse({"results": mems, "user_id": user_id})


@app.get("/metrics/summary")
async def metrics_summary():
    """Aggregated latency/success stats."""
    return JSONResponse(metrics_collector.get_summary())


@app.post("/skills/reload")
async def skills_reload():
    """Force the agent to re-pull skills from the Cortex platform (MCP) now, bypassing the TTL."""
    from app.skills.registry import registry
    return JSONResponse(await registry.force_refresh())


@app.get("/skills/list")
async def skills_list():
    """The skills currently front-loaded in the agent's RAM registry, shaped for the Skills UI."""
    from app.skills.registry import registry
    await registry.ensure_fresh()
    return JSONResponse(registry.cards())


@app.get("/context/clients")
async def context_clients():
    """The live client roster front-loaded from the Cortex platform — consumed by the
    backend proxy for the Client Insights picker. Serves the RAM cache (no MCP call)."""
    from app.context import cortex_context
    await cortex_context.ensure_fresh()
    return JSONResponse({
        "clients": cortex_context.roster(),
        "source": cortex_context.source,
        "loaded_at": cortex_context.loaded_at,
    })


@app.post("/context/reload")
async def context_reload():
    """Force a context-map re-pull from the platform now, bypassing the TTL."""
    from app.context import cortex_context
    return JSONResponse(await cortex_context.force_refresh())


@app.get("/health")
async def health():
    """Health check."""
    return JSONResponse({
        "status": "healthy",
        "ns_probe": NS_PROBE_ENABLED,
        "model": config.openai_model,
    })


@app.get("/debug/ns_probe")
async def debug_ns_probe():
    """Live ns_probe exporter stats — quick check that traces are flowing to the endpoint."""
    if not NS_PROBE_ENABLED:
        return JSONResponse({"enabled": False})
    out = {"enabled": True, "endpoint": os.getenv("NS_PROBE_ENDPOINT")}
    try:
        from ns_probe.tracer import TracerProvider
        prov = TracerProvider.get_instance()
        out["flushed_now"] = force_flush()
        exp = getattr(prov, "_exporter", None)
        out["has_exporter"] = exp is not None
        if exp is not None:
            out["stats"] = getattr(exp, "stats", {})
    except Exception as e:
        out["error"] = f"{type(e).__name__}: {e}"
    return JSONResponse(out)


@app.get("/", response_class=HTMLResponse)
async def chat_ui():
    """Serve the chat UI.

    Served by the agent itself rather than as a separate frontend app: this is an
    experiment harness, and a build step plus a second process per variant would be
    more moving parts than the thing being measured. One file, no bundler.

    The variant's identity is injected server-side so a single template serves both —
    the accent colour differs so two side-by-side windows cannot be confused.
    """
    service = os.getenv("NS_PROBE_SERVICE_NAME", "agent")
    direct = os.getenv("DATA_RESOLVER", "platform").strip().lower() == "direct"
    cfg = {
        "name": service,
        "variant": "claude" if direct else "cortex",
        "service": service,
        "model": os.getenv("LLM_MODEL", "") or os.getenv("OPENAI_MODEL", ""),
        "dataPath": "direct SQL" if direct else "cortex platform",
        "tagline": ("Claude writes the SQL and queries the database directly"
                    if direct else
                    "Questions are answered by the Cortex platform"),
    }
    html = (Path(__file__).parent / "static" / "index.html").read_text(encoding="utf-8")
    html = html.replace("__AGENT_NAME__", service)
    html = html.replace(
        "<script>", f"<script>window.__AGENT_CONFIG__={json.dumps(cfg)};</script>\n<script>", 1
    )
    return HTMLResponse(html)




# --- Memory Debug Endpoints --- (REMOVED: using LangGraph checkpointer now)


# --- Chat UI --- (served by this app from app/static/index.html)


# --- Instrument at the bottom of the file ---
# NOTE: instrument_all() handles tracing via LangGraph/OpenAI instrumentors.
# We also add a manual root span around SSE streaming to keep context alive.
