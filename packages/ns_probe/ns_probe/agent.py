"""Agent-level observability: the turn, and the measurements hung off it.

`observe()` covers the common case — wrap a function, get a span with its
arguments and result. What it cannot do is record a number the function WORKED
OUT rather than received: retrieval quality, a memory conflict index, a guardrail
risk score. Those are the measurements an agent exists to be judged on, and they
only exist part-way through the call.

This module is the other half. It is deliberately agent-agnostic — nothing here
names a pillar, a metric or a product. Every agent emitting into neo-observe
needs the same four things:

    turn()            one conversation turn, as a trace root
    set_metrics()     attach computed measurements to the current span
    as_bool()         write a boolean the pipeline can actually read
    clip()            truncate free text before it becomes an attribute

    with turn("ava.turn", session_id=sid, user_id=uid, agent_name="ava_agent"):
        ...
        set_metrics({"knowledge.top_chunk_score": 0.83})

Everything is best-effort: an observability failure must never take down the
agent it is observing, so every function here swallows its own errors.
"""

from __future__ import annotations

import base64
import logging
import os
from contextlib import contextmanager
from typing import Any, Dict, Iterator, Optional

from .tracer import (
    get_tracer,
    reset_context_attributes,
    set_context_attributes,
)

logger = logging.getLogger(__name__)

#: Free text on a span is truncated to this many characters. Spans cross Kafka
#: and land in a ClickHouse Map column; an unbounded document chunk would bloat
#: every row for no analytical gain.
SPAN_TEXT_LIMIT = 2000

#: Marks the span that begins one turn. ns_api's coverage metrics divide by the
#: number of turns, and they count them with this attribute rather than a span
#: name — which is what lets one metric definition serve every agent.
TURN_ROOT_ATTR = "turn.root"

#: Identity that belongs on every span underneath a turn, including the ones the
#: agent never creates itself (llm.call, httpx, framework internals). ns_processor
#: promotes session.id to a real ClickHouse column, so a span without it is
#: invisible to every per-session query.
#:
#: An allowlist on purpose. Inheriting the whole turn span would copy the prompt
#: onto every child, and would copy TURN_ROOT_ATTR — which would make every span
#: look like a turn and multiply every per-turn rate by the number of spans.
#:
#: tenant.id and workspace.id are here because scope in this platform is a triple
#: (tenant → workspace → subgraph), not a tenant alone. ns_processor promotes both
#: to real columns on `traces` and `outcome_ledger`, and a durable row that
#: carries only the tenant cannot be workspace-scoped by a reader afterwards —
#: the identity is not recoverable once the span is gone.
TURN_IDENTITY_KEYS = (
    "agent.name",
    "session.id",
    "user.id",
    "turn.surface",
    "tenant.id",
    "workspace.id",
)

#: Deployment-wide scope defaults. Scope is usually a property of where the agent
#: RUNS rather than of each turn, so reading it from the environment means an
#: operator can supply it without every call site passing it — and an agent that
#: has not been updated to pass scope still emits correctly scoped spans. An
#: explicit argument to `turn()` always wins.
TENANT_ID_ENV = "NS_PROBE_TENANT_ID"
WORKSPACE_ID_ENV = "NS_PROBE_WORKSPACE_ID"


def as_bool(value: Any) -> str:
    """Render a boolean as the STRING 'true' / 'false'.

    Not cosmetic. ns_processor decodes an attribute as
    `string_value or str(int_value)`. A genuine OTLP bool has an empty (falsy)
    string_value, so it falls through to `str(int_value)` == "0" — a TRUTHY
    string — and bool_value is never read. Every real boolean therefore lands in
    ClickHouse as "0", whatever it was. Same trap for an empty string.

    Until the processor's decoder is fixed, every boolean measurement has to
    travel as text.
    """
    return "true" if value else "false"


def clip(text: Optional[str], limit: int = SPAN_TEXT_LIMIT) -> str:
    """Truncate free text destined for a span attribute."""
    text = text or ""
    return text if len(text) <= limit else text[:limit] + "…"


def set_metrics(metrics: Dict[str, Any]) -> bool:
    """Attach computed measurements to the span that is currently open.

    This is the companion to `observe()`. `observe(capture_args=True)` records
    what a function was GIVEN; this records what it WORKED OUT — and the second
    is what an agent is actually judged on.

        @observe("Retrieve Knowledge")          # applied at the bottom of the file
        def retrieve(query):
            chunks = search(query)
            set_metrics({"knowledge.top_chunk_score": chunks[0].score})
            return chunks

    Returns True if the metrics landed. False means no span was open — the
    caller is outside any instrumented function — which is worth neither an
    exception nor a log line at anything above debug.

    Booleans must already be strings; run them through `as_bool()` first. This
    does not convert them, because a dict of measurements is usually assembled in
    one literal and converting silently here would hide the rule rather than
    teach it.
    """
    try:
        from .tracer import current_span

        span = current_span()
        if span is None:
            logger.debug("set_metrics outside a span — %d metrics dropped", len(metrics))
            return False
        span.set_attributes(metrics)
        return True
    except Exception:  # pragma: no cover — observability is best-effort
        logger.debug("set_metrics failed", exc_info=True)
        return False


@contextmanager
def turn(
    name: str = "agent.turn",
    *,
    agent_name: str = "",
    session_id: str = "",
    user_id: str = "",
    surface: str = "",
    tenant_id: str = "",
    workspace_id: str = "",
    attributes: Optional[Dict[str, Any]] = None,
) -> Iterator[Any]:
    """One conversation turn, as the root of its own trace.

    Two things this does that a plain `start_span` does not:

    1. STARTS A NEW TRACE, ALWAYS. A turn must never attach itself to whatever
       span happens to be open. The span stack lives in a ContextVar as a mutable
       list, and a ContextVar isolates rebinds, not mutations — so once anything
       has recorded a span at start-up, every request inherits THE SAME list.
       Two concurrent turns then see each other on the stack and the second
       parents to the first, fusing two users' conversations into one trace.
       `root=True` is what prevents that. Do not remove it.

    2. PUBLISHES THE TURN'S IDENTITY AS AMBIENT. session.id and friends are then
       merged into every span created while the turn is open, including spans the
       agent never creates and cannot pass anything to.

    `tenant_id` / `workspace_id` fall back to NS_PROBE_TENANT_ID /
    NS_PROBE_WORKSPACE_ID, because scope is normally a property of the deployment
    rather than of the turn. Both reach `outcome_ledger` through this ambient
    path — the ledger span is created inside the turn and inherits them — so
    scope needs supplying in exactly one place.

    Yields the span, or None if it could not be created — callers must not have
    to branch on tracing being available. Exceptions from the body propagate
    unchanged.
    """
    attrs: Dict[str, Any] = dict(attributes or {})
    for key, value in (
        ("agent.name", agent_name),
        ("session.id", session_id),
        ("user.id", user_id),
        ("turn.surface", surface),
        ("tenant.id", tenant_id or os.getenv(TENANT_ID_ENV, "")),
        ("workspace.id", workspace_id or os.getenv(WORKSPACE_ID_ENV, "")),
    ):
        if value:
            attrs[key] = value
    attrs[TURN_ROOT_ATTR] = "true"

    # Opening the span and running the body must NOT share an exception handler.
    # If they did, an error raised by the caller would be thrown into this
    # generator at the yield, caught here, and followed by a second yield —
    # which contextlib reports as "generator didn't stop after throw()",
    # destroying the original traceback.
    cm = span = None
    try:
        cm = get_tracer("ns_probe.agent").start_span(name, attributes=attrs, root=True)
        span = cm.__enter__()
    except Exception:  # pragma: no cover — observability is best-effort
        logger.debug("turn span %s not started", name, exc_info=True)
        cm = None

    if cm is None:
        yield None
        return

    # Absent beats empty: a span must not claim a session it does not have.
    ambient = {k: attrs[k] for k in TURN_IDENTITY_KEYS if attrs.get(k)}
    token = None
    try:
        token = set_context_attributes(ambient) if ambient else None
    except Exception:  # pragma: no cover
        logger.debug("ambient turn identity unavailable", exc_info=True)

    def _release() -> None:
        if token is None:
            return
        try:
            reset_context_attributes(token)
        except Exception:  # pragma: no cover
            logger.debug("ambient turn identity not reset", exc_info=True)

    try:
        yield span
    except BaseException as exc:
        _release()
        if not cm.__exit__(type(exc), exc, exc.__traceback__):
            raise
        return
    _release()
    cm.__exit__(None, None, None)


def storage_trace_id(trace_id: Optional[str]) -> str:
    """The trace id AS STORED downstream, which is not the one that was emitted.

    WORKAROUND FOR A COLLECTOR BUG. Delete this once ns_collector is fixed.

    ns_probe emits `"traceId": "<32 hex chars>"`, which is correct: the OTLP/JSON
    spec overrides protobuf-JSON's default base64 encoding for `bytes` fields and
    mandates hex for trace and span ids. ns_collector parses the envelope with a
    generic `google.protobuf.json_format.Parse`, which knows nothing of that
    override and base64-decodes the field. Hex characters are all valid base64,
    so the decode silently succeeds and yields 24 bytes of unrelated data — not
    even a legal 16-byte trace id.

        emitted   03edcd6a4ab4469782f6fb6eabc5da80                  (16 bytes)
        stored    d3779d71de9ae1a6f8e3af7bf367fa7dbe9e69b73975af34  (24 bytes)

    A UI that shows a trace id should show THIS one, so that pasting it into
    Grafana or ClickHouse actually finds the turn. Showing the
    correct-but-unfindable id would satisfy the spec and fail the user.

    Total over well-formed ids: hex is a subset of the base64 alphabet and 32
    characters is already a multiple of 4, so the decode never needs padding.
    """
    if not trace_id:
        return ""
    try:
        return base64.b64decode(trace_id).hex()
    except Exception:
        # An id that cannot be transformed is better shown as emitted than not at all.
        logger.debug("trace id %r not convertible to storage form", trace_id)
        return trace_id
