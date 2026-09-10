"""Tests for ns_probe.agent — the turn, and the measurements hung off it.

These moved out of Ava when the API did. They are not Ava-specific: every agent
emitting into neo-observe depends on this behaviour, and two of the rules below
encode pipeline bugs that silently corrupt data if broken.

Everything runs against the real tracer with the ring buffer swapped for a list,
so nothing here reimplements parenting — which is the whole thing being tested.
"""

import asyncio

import pytest

import ns_probe
from ns_probe import as_bool, clip, observe, set_metrics, storage_trace_id, turn
from ns_probe.tracer import Tracer, get_context_attributes, get_tracer

# ---------------------------------------------------------------------------
# Booleans must travel as the STRINGS 'true' / 'false'.
#
# ns_processor decodes attributes as `string_value or str(int_value)`. A real
# OTLP bool has an empty (falsy) string_value, so it falls through to
# str(int_value) == "0" — a TRUTHY string — and bool_value is never read. Every
# genuine boolean therefore lands in ClickHouse as "0", whatever it was.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "value, expected",
    [
        (True, "true"),
        (False, "false"),
        (1, "true"),
        (0, "false"),
        ([], "false"),
        ([1, 2], "true"),
        ("", "false"),
        ("x", "true"),
        (None, "false"),
    ],
)
def test_as_bool_always_returns_a_string(value, expected):
    result = as_bool(value)
    assert result == expected
    assert isinstance(result, str), "a real bool is decoded as '0' by the processor"


def test_as_bool_never_returns_a_python_bool():
    for value in (True, False):
        assert not isinstance(as_bool(value), bool)


def test_clip_leaves_short_text_alone():
    assert clip("hello") == "hello"


def test_clip_truncates_long_text():
    clipped = clip("x" * 5000)
    assert len(clipped) == ns_probe.SPAN_TEXT_LIMIT + 1  # + the ellipsis
    assert clipped.endswith("…")


def test_clip_handles_none():
    assert clip(None) == ""


# ---------------------------------------------------------------------------
# Span capture
# ---------------------------------------------------------------------------


class _CaptureBuffer:
    def __init__(self, inner, sink):
        self._inner = inner
        self._sink = sink

    def push(self, span):
        self._sink.append(span)

    def __getattr__(self, name):
        return getattr(self._inner, name)


@pytest.fixture
def spans(monkeypatch):
    """Collect emitted spans instead of exporting them.

    A span is created here first, on purpose. That is the precondition that used
    to fuse concurrent turns together: it forces the span-stack list to exist in
    this context, so any test that passes below passes under the conditions a
    real server actually runs in.
    """
    with get_tracer("ns_probe.agent").start_span("precondition.warmup"):
        pass

    sink: list = []
    for name in (
        "ns_probe.agent",
        "ns_probe.observe",
        "ns_probe.step",
        "ns_probe.outcome_ledger",
        "test",
    ):
        tracer = get_tracer(name)
        monkeypatch.setattr(tracer, "_buffer", _CaptureBuffer(tracer._buffer, sink))
    return sink


@pytest.fixture(autouse=True)
def ambient_is_left_clean():
    """Fail the test that leaks ambient state, not the innocent one after it."""
    yield
    assert get_context_attributes() == {}, "ambient attributes leaked out of a turn"


def _named(sink, name):
    return next(s for s in sink if s.name == name)


# ---------------------------------------------------------------------------
# turn() — the trace root
# ---------------------------------------------------------------------------


def test_turn_marks_itself_as_a_turn_root(spans):
    """ns_api's coverage metrics count turns with this attribute rather than a
    span name, which is what lets one metric definition serve every agent."""
    with turn("ava.turn", agent_name="ava_agent", session_id="s1"):
        pass

    span = _named(spans, "ava.turn")
    assert span.attributes[ns_probe.TURN_ROOT_ATTR] == "true"
    assert span.attributes["agent.name"] == "ava_agent"
    assert span.attributes["session.id"] == "s1"


def test_turn_carries_caller_attributes(spans):
    with turn("ava.turn", attributes={"input.value": "hello", "turn.message_chars": 5}):
        pass
    span = _named(spans, "ava.turn")
    assert span.attributes["input.value"] == "hello"
    assert span.attributes["turn.message_chars"] == 5


def test_a_turn_is_always_a_new_trace(spans):
    """A turn must never attach itself to whatever span happens to be open."""
    with get_tracer("test").start_span("something.else") as outer:
        with turn("ava.turn") as t:
            inner_trace, parent = t.trace_id, t.parent_span_id

    assert parent is None, "the turn inherited a parent"
    assert inner_trace != outer.trace_id


def test_a_second_turn_does_not_parent_to_the_first(spans):
    with turn("ava.turn"):
        pass
    with turn("ava.turn"):
        pass

    first_span, second_span = [s for s in spans if s.name == "ava.turn"]
    assert second_span.parent_span_id is None
    assert second_span.trace_id != first_span.trace_id


def test_two_concurrent_turns_stay_in_separate_traces(spans):
    """The failure this rules out is worse than a missing span: one user's
    question appearing inside another user's trace.

    Regression test. The span stack is a mutable list in a ContextVar, and a
    ContextVar isolates rebinds, not mutations — so once the list exists in a
    parent context every task shares it, and turn B parents to turn A.
    """

    async def one(tag, delay):
        with turn("ava.turn", session_id=tag) as t:
            await asyncio.sleep(delay)
            with ns_probe.step("knowledge.retrieve", attributes={"tag": tag}):
                pass
            await asyncio.sleep(delay)
            return t.trace_id

    async def both():
        return await asyncio.gather(one("A", 0.01), one("B", 0.02))

    trace_a, trace_b = asyncio.run(both())
    assert trace_a != trace_b, "two concurrent turns fused into one trace"

    by_tag = {s.attributes["tag"]: s for s in spans if s.name == "knowledge.retrieve"}
    assert by_tag["A"].trace_id == trace_a
    assert by_tag["B"].trace_id == trace_b


# ---------------------------------------------------------------------------
# Parenting under an SSE-style async generator
# ---------------------------------------------------------------------------


def test_children_created_after_a_yield_still_parent_to_the_turn(spans):
    """The shape of a streaming HTTP handler: the turn lives in an async
    generator, and the work happens between yields."""

    async def event_stream():
        with turn("ava.turn", session_id="s1"):
            yield "token"
            with ns_probe.step("knowledge.retrieve"):
                pass
            yield "token"
            with ns_probe.step("memory.write_back"):
                pass
            yield "done"

    async def drive():
        async for _ in event_stream():
            pass

    asyncio.run(drive())

    root = _named(spans, "ava.turn")
    for name in ("knowledge.retrieve", "memory.write_back"):
        child = _named(spans, name)
        assert child.parent_span_id == root.span_id, f"{name} orphaned after a yield"
        assert child.trace_id == root.trace_id


def test_a_child_created_in_a_worker_thread_still_parents_to_the_turn(spans):
    """Agents routinely push blocking work through asyncio.to_thread."""

    def work():
        with ns_probe.step("memory.write_back"):
            pass

    async def drive():
        with turn("ava.turn"):
            await asyncio.to_thread(work)

    asyncio.run(drive())
    root = _named(spans, "ava.turn")
    assert _named(spans, "memory.write_back").parent_span_id == root.span_id


# ---------------------------------------------------------------------------
# Ambient identity
# ---------------------------------------------------------------------------


def test_turn_publishes_its_identity_to_spans_created_by_anyone(spans):
    """`llm.call` stands in for an auto-instrumented span: the agent never
    creates it and cannot pass it anything, and it must still carry the session."""
    with turn("ava.turn", agent_name="ava_agent", session_id="s1", user_id="mohan", surface="http"):
        with Tracer("test").start_span("llm.call") as span:
            emitted = dict(span.attributes)

    assert emitted["session.id"] == "s1"
    assert emitted["user.id"] == "mohan"
    assert emitted["turn.surface"] == "http"
    assert emitted["agent.name"] == "ava_agent"


def test_turn_publishes_the_scope_triple_to_every_span(spans):
    """Scope is (tenant → workspace → subgraph), not a tenant alone. ns_processor
    promotes both to columns; a durable row with only the tenant cannot be
    workspace-scoped by a reader afterwards."""
    with turn("ava.turn", session_id="s1", tenant_id="acme", workspace_id="ws-42"):
        with Tracer("test").start_span("llm.call") as span:
            emitted = dict(span.attributes)

    assert emitted["tenant.id"] == "acme"
    assert emitted["workspace.id"] == "ws-42"


def test_scope_reaches_the_ledger_span(spans):
    """The row the ledger stores is scoped through this ambient path — it is the
    only place scope is supplied, so this is the assertion that matters."""
    from ns_probe import outcome_ledger

    with turn("ava.turn", session_id="s1", tenant_id="acme", workspace_id="ws-42"):
        outcome_ledger.record(question="q", answer="a")

    ledger = _named(spans, outcome_ledger.LEDGER_SPAN_NAME)
    assert ledger.attributes["tenant.id"] == "acme"
    assert ledger.attributes["workspace.id"] == "ws-42"


def test_scope_falls_back_to_the_environment(spans, monkeypatch):
    """Scope is a property of the deployment, so an agent that was never updated
    to pass it still emits correctly scoped spans."""
    monkeypatch.setenv("NS_PROBE_TENANT_ID", "env-tenant")
    monkeypatch.setenv("NS_PROBE_WORKSPACE_ID", "env-ws")

    with turn("ava.turn", session_id="s1"):
        with Tracer("test").start_span("llm.call") as span:
            emitted = dict(span.attributes)

    assert emitted["tenant.id"] == "env-tenant"
    assert emitted["workspace.id"] == "env-ws"


def test_an_explicit_scope_beats_the_environment(spans, monkeypatch):
    monkeypatch.setenv("NS_PROBE_WORKSPACE_ID", "env-ws")

    with turn("ava.turn", session_id="s1", workspace_id="explicit-ws"):
        with Tracer("test").start_span("llm.call") as span:
            emitted = dict(span.attributes)

    assert emitted["workspace.id"] == "explicit-ws"


def test_a_turn_with_no_scope_publishes_nothing_rather_than_empty(spans, monkeypatch):
    """Absent beats empty, same rule as session.id. An empty workspace attribute
    would look like a real scope to the processor."""
    monkeypatch.delenv("NS_PROBE_TENANT_ID", raising=False)
    monkeypatch.delenv("NS_PROBE_WORKSPACE_ID", raising=False)

    with turn("ava.turn", session_id="s1"):
        with Tracer("test").start_span("llm.call") as span:
            emitted = dict(span.attributes)

    assert "tenant.id" not in emitted
    assert "workspace.id" not in emitted


def test_the_prompt_is_not_copied_onto_every_span(spans):
    """A prompt is ~2 KB. Inheriting it would repeat it on every child."""
    with turn("ava.turn", session_id="s1", attributes={"input.value": "x" * 2000}):
        with ns_probe.step("memory.search"):
            pass
    assert "input.value" not in _named(spans, "memory.search").attributes


def test_the_turn_root_marker_is_not_inherited(spans):
    """turn.root is what identifies a span AS a turn — inheriting it would make
    every span a turn and multiply every per-turn rate by the number of spans."""
    with turn("ava.turn", session_id="s1"):
        with ns_probe.step("memory.search"):
            pass
    assert ns_probe.TURN_ROOT_ATTR not in _named(spans, "memory.search").attributes


def test_a_turn_without_a_session_does_not_publish_an_empty_one(spans):
    """Absent beats empty: a span must not claim a session it does not have."""
    with turn("ava.turn", session_id=""):
        with ns_probe.step("memory.search"):
            pass
    assert not _named(spans, "memory.search").attributes.get("session.id")


def test_the_session_does_not_outlive_the_turn(spans):
    with turn("ava.turn", session_id="s1"):
        pass
    assert get_context_attributes() == {}
    with Tracer("test").start_span("llm.call") as span:
        assert "session.id" not in span.attributes


def test_the_session_is_cleared_even_when_the_turn_raises(spans):
    with pytest.raises(ValueError):
        with turn("ava.turn", session_id="s1"):
            raise ValueError("boom")
    assert get_context_attributes() == {}


def test_an_exception_in_a_turn_propagates_unchanged(spans):
    """The original error must survive — tracing must not eat it.

    When span setup and the `yield` shared one try/except, an exception from the
    body was thrown into the generator, caught, and followed by a second yield;
    contextlib then raised "generator didn't stop after throw()" and the real
    traceback was lost.
    """
    with pytest.raises(ValueError, match="the real error"):
        with turn("ava.turn"):
            raise ValueError("the real error")


# ---------------------------------------------------------------------------
# set_metrics() — computed measurements
# ---------------------------------------------------------------------------


def test_set_metrics_lands_on_the_observe_span(spans):
    """The companion to observe(): what the function WORKED OUT, not what it
    was given."""

    def retrieve(query):
        set_metrics({"knowledge.top_chunk_score": 0.83, "knowledge.multi_source": as_bool(False)})
        return ["chunk"]

    retrieve = observe(name="knowledge.retrieve", agent_name="ava_agent")(retrieve)

    with turn("ava.turn", session_id="s1"):
        retrieve("what is my deductible")

    span = _named(spans, "knowledge.retrieve")
    assert span.attributes["knowledge.top_chunk_score"] == 0.83
    assert span.attributes["knowledge.multi_source"] == "false"
    assert span.attributes["session.id"] == "s1", "identity should still be inherited"


def test_set_metrics_outside_a_span_is_a_no_op_not_an_error(monkeypatch):
    """An agent calling an instrumented helper directly must not crash.

    The current span is cleared explicitly rather than assumed: `current_span()`
    is a ContextVar that earlier work in this context may still be holding, and
    the point of this test is the no-span path, not test ordering.
    """
    from ns_probe import tracer as tracer_mod

    # current_span() reads the contextvar AND falls back to the span stack, so
    # both have to be empty for this to be the no-span path.
    monkeypatch.setattr(
        tracer_mod,
        "_current_span_var",
        tracer_mod.contextvars.ContextVar("span_obj_test", default=None),
    )
    monkeypatch.setattr(
        tracer_mod,
        "_span_stack_var",
        tracer_mod.contextvars.ContextVar("span_stack_test", default=None),
    )
    assert set_metrics({"a": 1}) is False


def test_set_metrics_never_raises_on_a_bad_span(monkeypatch):
    """Tracing is best-effort: a broken span must not fail the caller."""

    def _boom():
        raise RuntimeError("tracer is down")

    monkeypatch.setattr("ns_probe.tracer.current_span", _boom)
    assert set_metrics({"a": 1}) is False


# ---------------------------------------------------------------------------
# The trace id as stored downstream — a collector workaround.
# ---------------------------------------------------------------------------


def test_storage_trace_id_reproduces_what_the_collector_stores():
    """Both halves of this pair were read from a real turn: the id on the span,
    and the id that turn actually has in observability.traces."""
    assert storage_trace_id("03edcd6a4ab4469782f6fb6eabc5da80") == (
        "d3779d71de9ae1a6f8e3af7bf367fa7dbe9e69b73975af34"
    )


def test_the_stored_form_is_longer_than_a_legal_trace_id():
    """24 bytes, not 16 — the stored value is not a valid trace id at all.
    Documents the severity: this is corruption, not a different encoding."""
    stored = storage_trace_id("03edcd6a4ab4469782f6fb6eabc5da80")
    assert len(bytes.fromhex(stored)) == 24


def test_storage_trace_id_is_total_over_well_formed_ids():
    for emitted in ("0" * 32, "f" * 32, "0123456789abcdef" * 2):
        assert len(storage_trace_id(emitted)) == 48


def test_storage_trace_id_passes_through_an_untransformable_id():
    assert storage_trace_id("not-a-trace-id!") == "not-a-trace-id!"


def test_storage_trace_id_of_nothing_is_empty():
    assert storage_trace_id("") == ""
    assert storage_trace_id(None) == ""
