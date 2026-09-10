"""The three promises ns_probe makes to the agent that imports it.

Everything else in the SDK is a feature. These are the reasons it is safe to put
in a request path at all, and none of them had a test:

1. It never blocks the agent. The buffer drops spans rather than making the
   caller wait, because losing telemetry is acceptable and slowing the agent is
   not.
2. It never breaks the agent. Every entry point swallows its own errors — an
   unreachable collector, a value that will not serialise, a missing optional
   dependency — and the agent answers normally regardless.
3. A turn is isolated. Two turns running at once never land in each other's
   trace, however they are scheduled.

Promise 3 is the one with teeth. `agent.turn()` carries a long comment about why
`root=True` cannot be removed: the span stack is a mutable list inside a
ContextVar, and a ContextVar isolates rebinds rather than mutations, so once
anything has recorded a span at start-up every request inherits the same list.
Without `root=True` the second concurrent turn parents itself to the first and
two users' conversations fuse into one trace. That is a privacy problem wearing
a performance problem's clothes, and it is worth a regression test.
"""

from __future__ import annotations

import asyncio
import threading

import pytest

from ns_probe import clip, configure, force_flush, observe, set_metrics, step, turn
from ns_probe.buffer import SPSCRingBuffer


@pytest.fixture(autouse=True)
def _pointed_nowhere():
    """Port 9 is 'discard'. Nothing listens, which is the point of most of these."""
    configure(service_name="guarantee-tests", endpoint="http://127.0.0.1:9/v1/traces")
    yield


# ---------------------------------------------------------------------------
# 1. It never blocks the agent.
# ---------------------------------------------------------------------------


class TestTheBufferDropsRatherThanBlocks:
    def test_a_full_buffer_reports_the_drop_instead_of_waiting(self):
        # The capacity rounds up to a power of two, so ask the buffer what it
        # took rather than assuming the number passed in.
        buf: SPSCRingBuffer = SPSCRingBuffer(capacity=8)

        accepted = sum(1 for i in range(buf.capacity * 4) if buf.push(i))

        assert accepted < buf.capacity * 4, "nothing was dropped, so something blocked"
        assert buf.dropped_count > 0
        assert buf.is_full()

    def test_push_returns_false_rather_than_raising(self):
        """The caller is agent code in a request path. It gets a boolean it is
        free to ignore, never an exception it has to catch."""
        buf: SPSCRingBuffer = SPSCRingBuffer(capacity=2)
        for i in range(buf.capacity):
            buf.push(i)

        assert buf.push("one too many") is False

    def test_the_oldest_data_is_still_readable_after_an_overflow(self):
        """Overflow drops the new span. It must not corrupt or discard what is
        already queued — a dropped tail is a gap, a corrupted buffer is a lie."""
        buf: SPSCRingBuffer = SPSCRingBuffer(capacity=4)
        for i in range(buf.capacity):
            buf.push(i)
        buf.push("dropped")

        assert buf.pop() == 0

    def test_dropped_count_is_resettable_so_it_can_be_reported_periodically(self):
        buf: SPSCRingBuffer = SPSCRingBuffer(capacity=2)
        for i in range(10):
            buf.push(i)

        assert buf.reset_dropped_count() > 0
        assert buf.dropped_count == 0


# ---------------------------------------------------------------------------
# 2. It never breaks the agent.
# ---------------------------------------------------------------------------


class TestFailOpen:
    def test_a_traced_function_still_returns_with_no_collector_listening(self):
        @observe(name="work")
        def work(x):
            return x * 2

        assert work(21) == 42

    def test_a_traced_function_still_raises_ITS_OWN_error(self):
        """Fail-open means tracing does not add failures. It must not swallow
        the agent's either, or a real bug becomes invisible."""

        @observe(name="explodes")
        def explodes():
            raise ValueError("the real error")

        with pytest.raises(ValueError, match="the real error"):
            explodes()

    def test_an_unserialisable_attribute_does_not_reach_the_caller(self):
        """Whatever the agent hands us, the answer is still the agent's answer."""

        class Unserialisable:
            def __repr__(self):
                raise RuntimeError("not even repr works")

        with turn("agent.turn", attributes={"input.value": Unserialisable()}):
            pass  # reaching here at all is the assertion

    def test_set_metrics_outside_a_span_is_a_no_op_not_a_crash(self):
        """Called from a function that happens not to be traced this time."""
        assert set_metrics({"score": "0.9"}) is False

    def test_force_flush_against_a_dead_collector_returns(self):
        with turn("agent.turn"):
            pass

        force_flush()  # must not raise, must not hang

    def test_clip_handles_none_rather_than_making_the_caller_check(self):
        assert clip(None) == ""

    def test_clip_marks_what_it_truncated(self):
        # limit + 1: the ellipsis is appended AFTER the cut, deliberately, so a
        # reader can tell a truncated value from one that merely ended there.
        # Worth knowing when sizing anything downstream against the same number —
        # clip(text, 2000) yields 2001 characters, not 2000.
        clipped = clip("x" * 10_000, limit=100)

        assert len(clipped) == 101
        assert clipped.endswith("…")
        assert clipped[:100] == "x" * 100


# ---------------------------------------------------------------------------
# 3. A turn is isolated.
# ---------------------------------------------------------------------------


class TestTurnIsolation:
    def test_two_sequential_turns_are_two_traces(self):
        with turn("agent.turn", session_id="s1") as a:
            trace_a = a.trace_id
        with turn("agent.turn", session_id="s2") as b:
            trace_b = b.trace_id

        assert trace_a != trace_b

    def test_a_turn_never_parents_to_a_span_left_open_around_it(self):
        """The start-up case from the agent.turn() comment: something records a
        span before the first request, and every turn afterwards inherits it."""
        with step("startup.warmup"):
            with turn("agent.turn") as t:
                assert t.parent_span_id is None, "the turn adopted an ambient parent"

    def test_work_inside_a_turn_joins_that_turn(self):
        """Isolation must not be achieved by orphaning everything."""

        @observe(name="child")
        def child():
            from ns_probe import current_span

            return current_span()

        with turn("agent.turn") as t:
            inner = child()

        assert inner is not None
        assert inner.trace_id == t.trace_id

    @pytest.mark.asyncio
    async def test_concurrent_turns_under_gather_do_not_fuse(self):
        """The failure this guards: two users answered at once, one trace, each
        able to read the other's conversation in the trace explorer."""

        async def one_turn(n):
            with turn("agent.turn", session_id=f"s{n}") as t:
                await asyncio.sleep(0)  # force interleaving
                return t.trace_id, t.parent_span_id

        results = await asyncio.gather(*(one_turn(n) for n in range(8)))
        trace_ids = [r[0] for r in results]
        parents = [r[1] for r in results]

        assert len(set(trace_ids)) == len(trace_ids), "concurrent turns shared a trace"
        assert parents == [None] * len(parents), "a concurrent turn found a parent"

    def test_concurrent_turns_across_threads_do_not_fuse(self):
        """Threads start with a fresh context, which is the other half of the
        same problem — and the shape CrewAI's threaded task execution has."""
        results = []
        lock = threading.Lock()

        def one_turn(n):
            with turn("agent.turn", session_id=f"s{n}") as t:
                with lock:
                    results.append((t.trace_id, t.parent_span_id))

        threads = [threading.Thread(target=one_turn, args=(n,)) for n in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(set(r[0] for r in results)) == 8, "threaded turns shared a trace"
        assert all(r[1] is None for r in results)

    def test_the_turn_publishes_identity_for_spans_it_did_not_create(self):
        """The reason turn() exists rather than a plain span: session and user
        have to reach spans the agent never touches, such as the model call."""
        with turn("agent.turn", session_id="s-1", user_id="u-1", surface="http") as t:
            assert t.attributes["session.id"] == "s-1"
            assert t.attributes["user.id"] == "u-1"
            assert t.attributes["turn.surface"] == "http"
            assert t.attributes["turn.root"] == "true"
