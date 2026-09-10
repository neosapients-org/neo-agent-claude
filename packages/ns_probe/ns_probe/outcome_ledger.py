"""Outcome Ledger — one durable row per conversation turn, for every agent.

WHY IT EXISTS
-------------
`observability.traces` answers "what happened inside this turn". It cannot
cheaply answer "show me the turn the user is complaining about", because a turn
is eight rows whose facts must be re-joined by hand every time, and an agent's
own verdict on the turn — its pillar scores — is an aggregate that lives in no
single span. The ledger is the flat, one-row-per-turn table those questions want.

It is deliberately NOT agent-specific. Ava is the first caller, not the owner:
identity, cost, latency and outcome are typed columns because every agent has
them, while anything an agent judges for itself travels in `scores` / `metrics`
maps so a new agent adds its own without a schema migration.

HOW A ROW GETS THERE
--------------------
    agent → record() → span "outcome.ledger" → collector → Kafka
                                                             │
                              ClickHouse ← ns_processor ←────┘
                              observability.outcome_ledger

The row rides the span pipeline rather than going to ClickHouse directly, so an
agent still has exactly one egress and needs no database credentials. The
processor recognises the span by its LEDGER_MARKER attribute, writes the ledger
row, and keeps it out of `traces` — the turn-root span already records the same
prompt and reply, and storing them twice would double the largest column in the
store for no new information.

Because `record()` opens a span inside the current trace, the ledger row and the
spans it summarises share a trace_id for free. That is the join.

EVERY VALUE IS SENT AS A STRING, deliberately
---------------------------------------------
ns_processor decodes an attribute as
    `string_value or str(int_value) or str(double_value) or str(bool_value)`
An OTLP bool or double has an EMPTY string_value, so evaluation falls through to
`str(int_value)` == "0" — a truthy string — and the real value is never read.
Booleans and floats both land as "0". Stringifying here is what makes a cost of
0.00012 and an outcome of False survive the trip; callers pass native Python
types and never have to know.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Mapping, Optional

logger = logging.getLogger("ns_probe.outcome_ledger")

# ── Wire contract with ns_processor ───────────────────────────────────────────
# ns_processor mirrors these four constants. They are the whole contract: change
# one here and the processor stops recognising ledger spans, which fails quietly
# (rows keep flowing into `traces` and the ledger simply stays empty), so treat
# them as versioned API rather than internal names.

#: Span name carrying a ledger row. The processor keys off LEDGER_MARKER, not
#: this, so the name is free to change for readability without breaking ingest.
LEDGER_SPAN_NAME = "outcome.ledger"

#: Presence of this attribute (as the string "true") is what marks a span as a
#: ledger row.
LEDGER_MARKER = "ledger.record"

#: Namespace for the ledger's own fields, keeping them clear of span attributes.
LEDGER_PREFIX = "ledger."

#: An agent's self-assessment. `ledger.score.knowledge` → scores['knowledge'].
LEDGER_SCORE_PREFIX = "ledger.score."

#: Individual metrics. `ledger.metric.knowledge.noise_ratio` → metrics[...].
LEDGER_METRIC_PREFIX = "ledger.metric."

# ── Outcomes ──────────────────────────────────────────────────────────────────
# Conventional values for the `outcome` column. The column is a free string so an
# agent can record something these do not cover; these exist so the ones every
# agent DOES share are spelled identically and remain comparable across agents.
OUTCOME_OK = "ok"
OUTCOME_BLOCKED_INPUT = "blocked_input"
OUTCOME_BLOCKED_OUTPUT = "blocked_output"
OUTCOME_ERROR = "error"

#: Question and answer are clipped before they become span attributes: they cross
#: Kafka and land in a Map column, and an unbounded reply would bloat every row.
TEXT_LIMIT = 4000

#: Deployment-wide scope defaults, the same names `ns_probe.agent.turn` reads.
#:
#: Named here rather than imported from `agent`, to keep this module free of that import — it is the
#: one an agent may use alone. They are ONE contract even so: a row written through this path and a
#: span written through `turn()` must not disagree about which tenant they belong to.
TENANT_ID_ENV = "NS_PROBE_TENANT_ID"
WORKSPACE_ID_ENV = "NS_PROBE_WORKSPACE_ID"

#: Whether the "this row has no workspace" warning has already been emitted.
#:
#: Once per process, not once per turn: an agent that is misconfigured is misconfigured for every
#: turn it will ever serve, and a per-turn warning would bury the one line that matters under
#: thousands of copies. A plain bool rather than a lock — the worst a race can do is log twice.
_warned_unscoped = False


def _clip(text: Optional[str], limit: int) -> str:
    text = "" if text is None else str(text)
    return text if len(text) <= limit else text[:limit] + "…"


def _as_wire_value(value: Any) -> str:
    """Render one value the way ns_processor can actually read it back.

    See the module docstring: bools and floats are destroyed unless they travel
    as text. `repr` on a float round-trips exactly, so a cost of 0.00012 stays
    0.00012 instead of being rounded away.
    """
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return repr(value)
    return str(value)


def _put(attrs: dict, key: str, value: Any) -> None:
    """Set `key` unless the value is absent.

    Omission and zero must stay distinguishable: a turn that genuinely cost
    nothing is not the same as a turn whose cost was never measured, and the
    ledger's numeric columns default to 0 for the second case.
    """
    if value is None or value == "":
        return
    attrs[key] = _as_wire_value(value)


def _numeric_map(prefix: str, values: Optional[Mapping[str, Any]]) -> dict:
    """Flatten a name→number mapping into prefixed span attributes.

    Unset and non-numeric entries are DROPPED rather than coerced: the ledger's
    map columns are Float64, and a metric that had no value recorded as 0.0 would
    read as a genuine measurement of zero. Anything dropped is still recoverable
    from `metrics_json`, which keeps the caller's payload verbatim.

    Booleans are kept, as 1.0 / 0.0. A False here is a real measurement — the
    source had no validated owner — not a missing one.
    """
    flattened: dict[str, str] = {}
    for name, value in (values or {}).items():
        if value is None:
            continue
        try:
            flattened[f"{prefix}{name}"] = repr(float(value))
        except (TypeError, ValueError):
            continue
    return flattened


def build_attributes(
    *,
    question: Optional[str] = None,
    answer: Optional[str] = None,
    outcome: str = OUTCOME_OK,
    session_id: Optional[str] = None,
    user_id: Optional[str] = None,
    surface: Optional[str] = None,
    turn_index: Optional[int] = None,
    latency_ms: Optional[float] = None,
    ttft_ms: Optional[float] = None,
    llm_provider: Optional[str] = None,
    llm_model: Optional[str] = None,
    tokens_in: Optional[int] = None,
    tokens_out: Optional[int] = None,
    cost_usd: Optional[float] = None,
    scores: Optional[Mapping[str, Any]] = None,
    metrics: Optional[Mapping[str, Any]] = None,
    metrics_json: Any = None,
    agent_version: Optional[str] = None,
    environment: Optional[str] = None,
    tenant_id: Optional[str] = None,
    workspace_id: Optional[str] = None,
    text_limit: int = TEXT_LIMIT,
    extra: Optional[Mapping[str, Any]] = None,
) -> dict:
    """The span attributes for one ledger row. Pure — useful on its own in tests."""
    attrs: dict[str, Any] = {LEDGER_MARKER: "true"}

    # Scope, resolved here rather than left entirely to the ambient turn identity.
    #
    # ⚠️ **A row emitted outside its turn's context loses its scope silently.** The ambient path
    # (`agent.turn` -> set_context_attributes -> _merge_context_attributes) only reaches spans
    # created while that ContextVar is still set. `record()` already carries a `parent=` argument
    # for exactly the case where it is not — an async generator, where ContextVar parenting does not
    # survive a `yield` — and `parent` restores the TRACE link, not the attributes. So in the one
    # situation this SDK already anticipates, the row lands with no tenant and no workspace, and the
    # symptom is not an error: the Outcome Ledger's reader scopes strictly, so the turn is simply
    # never visible to anyone. Measured in dev: 13 of 78 rows unreadable for this reason.
    _put_scope(attrs, tenant_id, workspace_id)

    # Clipped unconditionally, including when empty, so a blocked turn records an
    # empty answer rather than no answer at all.
    attrs[f"{LEDGER_PREFIX}question"] = _clip(question, text_limit)
    attrs[f"{LEDGER_PREFIX}answer"] = _clip(answer, text_limit)
    attrs[f"{LEDGER_PREFIX}outcome"] = outcome

    _put(attrs, f"{LEDGER_PREFIX}user_id", user_id)
    _put(attrs, f"{LEDGER_PREFIX}surface", surface)
    _put(attrs, f"{LEDGER_PREFIX}turn_index", turn_index)
    _put(attrs, f"{LEDGER_PREFIX}latency_ms", latency_ms)
    _put(attrs, f"{LEDGER_PREFIX}ttft_ms", ttft_ms)
    _put(attrs, f"{LEDGER_PREFIX}llm_provider", llm_provider)
    _put(attrs, f"{LEDGER_PREFIX}llm_model", llm_model)
    _put(attrs, f"{LEDGER_PREFIX}tokens_in", tokens_in)
    _put(attrs, f"{LEDGER_PREFIX}tokens_out", tokens_out)
    _put(attrs, f"{LEDGER_PREFIX}cost_usd", cost_usd)
    _put(attrs, f"{LEDGER_PREFIX}agent_version", agent_version)
    _put(attrs, f"{LEDGER_PREFIX}environment", environment)

    # session.id is unprefixed on purpose: it is the same attribute the processor
    # already promotes to a column on every span. Usually ambient — passed here
    # only when the caller has it and the turn did not publish it.
    _put(attrs, "session.id", session_id)

    attrs.update(_numeric_map(LEDGER_SCORE_PREFIX, scores))
    attrs.update(_numeric_map(LEDGER_METRIC_PREFIX, metrics))

    if metrics_json is not None:
        payload = (
            metrics_json if isinstance(metrics_json, str) else json.dumps(metrics_json, default=str)
        )
        # BOUNDED like question/answer above, and for the same two reasons — this
        # was the only unbounded field left, and unlike those two it is the one
        # that gets PERSISTED (they are dropped by the processor).
        #
        # Size: it crosses Kafka and lands in a String column, so a large blob
        # risks the broker's max.message.bytes and takes the whole poll's spans
        # with it, not just this row.
        #
        # Privacy: the ledger's contract is that it stores no free text. Whatever
        # an agent puts in a debug payload would otherwise land verbatim, outside
        # the review that removed question/answer.
        #
        # This is for NUMBERS — metric bands and counts — not prose.
        #
        # DROPPED over the limit rather than clipped, unlike question/answer:
        # those are prose, and a reader of a clipped sentence still learns
        # something. This one is a JSON document the Outcome Ledger's drawer
        # parses, so a clipped payload does not parse at all — the reader sees
        # "this row reported nothing", which is the same thing it sees for an
        # absent field, except a whole record's worth of bytes was stored to say
        # it. The TS writer already refuses to emit a truncated document for
        # exactly this reason (`buildLedgerMetricsJson` returns undefined);
        # this is the Python half of the same contract.
        if len(payload) <= text_limit:
            attrs[f"{LEDGER_PREFIX}metrics_json"] = payload

    for key, value in (extra or {}).items():
        _put(attrs, key, value)

    return attrs


def _ambient(key: str) -> str:
    """One ambient turn-identity attribute, or ''. Never raises — tracing is best-effort."""
    try:
        from .tracer import get_context_attributes

        value = get_context_attributes().get(key)
        return value if isinstance(value, str) and value else ""
    except Exception:  # pragma: no cover — the probe must not fail a turn
        return ""


def _put_scope(attrs: dict, tenant_id: Optional[str], workspace_id: Optional[str]) -> None:
    """Resolve `tenant.id` / `workspace.id` for a ledger row, most specific first.

    Order, and each step is load-bearing:

    1. **The explicit argument.** A worker that serves several workspaces — one MCP endpoint per
       workspace — knows which one a turn served at the moment it answers, and nothing else does.
       This is the only input that can say so per turn.
    2. **The ambient turn identity**, left ALONE rather than copied. When a turn published scope,
       `_merge_context_attributes` already folds it under the span's own attributes, and writing it
       here from the environment instead would OVERRIDE the turn's answer with a deployment-wide
       constant — turning a correct per-turn scope into a wrong one. So an ambient value means
       "write nothing and let the fold do it".
    3. **The environment** (`NS_PROBE_TENANT_ID` / `NS_PROBE_WORKSPACE_ID`), for a row emitted
       outside any turn. Same defaults `agent.turn` uses, so the two paths cannot disagree.

    Nothing is written when none of the three has a value: an absent attribute is honest, and
    `''` would land as a real empty-string column that no scoped read can ever match.
    """
    resolved = {}
    for key, explicit, env in (
        ("tenant.id", tenant_id, TENANT_ID_ENV),
        ("workspace.id", workspace_id, WORKSPACE_ID_ENV),
    ):
        value = (explicit or "").strip()
        ambient = "" if value else _ambient(key)
        if ambient:
            # Present already — the fold will supply it. Recorded here only so the warning below
            # can tell "the turn published it" apart from "nobody has it".
            resolved[key] = ambient
            continue
        if not value:
            value = os.getenv(env, "").strip()
        if value:
            attrs[key] = resolved[key] = value

    _warn_if_unscoped(resolved)


def _warn_if_unscoped(resolved: dict) -> None:
    """Say once, loudly, that these rows will not be readable by anyone.

    ⚠️ **This is the only place the misconfiguration is visible.** A ledger row with no workspace is
    written, ingested and stored exactly like any other — nothing errors. It is the READER that
    scopes strictly, so the turn is simply never shown, and the symptom presented to a person is an
    empty Outcome Ledger page for an agent that is plainly serving traffic. Diagnosing it from that
    end takes a ClickHouse query; from this end it takes one line at start-up.

    Measured in dev before this existed: 30 of 78 rows unreadable, across two separate causes, both
    reported as "nothing shows up".
    """
    global _warned_unscoped
    if _warned_unscoped:
        return
    missing = [key for key in ("tenant.id", "workspace.id") if not resolved.get(key)]
    if not missing:
        return
    _warned_unscoped = True
    logger.warning(
        "Outcome Ledger rows are being written with no %s. They will be stored but will NOT be "
        "readable — the ledger scopes every read, so an unscoped row is invisible to every caller. "
        "Pass tenant_id/workspace_id to ns_probe.agent.turn() for per-turn scope (a worker serving "
        "several workspaces needs this), or set %s / %s for a single-workspace deployment.",
        " or ".join(missing),
        TENANT_ID_ENV,
        WORKSPACE_ID_ENV,
    )


def record(**fields: Any) -> bool:
    """Write one turn to the Outcome Ledger. Returns True if the span was emitted.

    Takes the same keyword arguments as `build_attributes`, plus `parent`: an
    explicit SpanContext to hang the ledger span from. Pass it when the turn span
    lives in an async generator, where ContextVar-based parenting does not survive
    a `yield` and the row would otherwise start a trace of its own — losing the
    join to the spans it summarises.

    ⚠️ **`parent` restores the trace link, not the scope.** Attributes are not inherited from a
    parent SpanContext — only from the ambient ContextVar — so in exactly the case `parent` exists
    for, `tenant.id` and `workspace.id` are gone. Pass them explicitly there:
    `record(parent=ctx, tenant_id=..., workspace_id=..., ...)`. See `_put_scope`.

    Never raises. A turn that answered the user correctly must not be reported as
    failed because its bookkeeping could not be written.
    """
    parent = fields.pop("parent", None)
    try:
        from .tracer import get_tracer

        attributes = build_attributes(**fields)
        with get_tracer("ns_probe.outcome_ledger").start_span(
            LEDGER_SPAN_NAME, attributes=attributes, parent=parent
        ):
            pass
        return True
    except Exception:
        logger.debug("outcome ledger row not emitted", exc_info=True)
        return False
