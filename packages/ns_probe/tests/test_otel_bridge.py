"""The bridge that lets OpenTelemetry-shaped instrumentors write into our trace.

This is the piece that makes the whole OpenLLMetry catalogue usable — LlamaIndex,
CrewAI's own structure, Haystack, Bedrock, Vertex, Cohere — without writing an
instrumentor per framework. Every one of them emits through
`opentelemetry.trace`, so all that was ever missing is a provider on the other end
of that call that belongs to us.

The two assertions that matter are the first two: a bridged span must land in the
SAME trace as the surrounding `turn()`, and it must inherit that turn's identity.
Get either wrong and the spans still appear — orphaned, or anonymous — which is
the failure mode that made `_try_openllmetry()` unusable in the first place and
is exactly the thing nobody notices until they go looking for a turn.
"""

from __future__ import annotations

import pytest

import ns_probe
from ns_probe import otel_bridge

otel_trace = pytest.importorskip("opentelemetry.trace", reason="OpenTelemetry API not installed")


@pytest.fixture(autouse=True)
def bridged(monkeypatch):
    """A fresh bridge per test; the OTel global is process-wide."""
    ns_probe.configure(service_name="bridge-tests", endpoint="http://127.0.0.1:9/v1/traces")
    monkeypatch.setattr(otel_trace, "_TRACER_PROVIDER", None, raising=False)
    monkeypatch.setattr(
        otel_trace,
        "_TRACER_PROVIDER_SET_ONCE",
        type(otel_trace._TRACER_PROVIDER_SET_ONCE)(),
        raising=False,
    )
    assert otel_bridge.install_otel_bridge() is True
    yield otel_trace.get_tracer("test.instrumentor")


class TestItJoinsOurTrace:
    def test_a_bridged_span_lands_in_the_surrounding_turn(self, bridged):
        """Orphaned spans are the exact failure that made OpenLLMetry unusable."""
        with ns_probe.turn("agent.turn") as t:
            with bridged.start_as_current_span("llamaindex.query") as span:
                child = span._ns

        assert child.trace_id == t.trace_id
        assert child.parent_span_id == t.span_id

    def test_the_turn_identity_reaches_a_span_we_did_not_create(self, bridged):
        """The whole point of turn() publishing ambient identity: it has to reach
        spans the agent never touches, and an instrumentor's span is precisely
        that."""
        with ns_probe.turn("agent.turn", session_id="s-42", user_id="u-7", tenant_id="acme"):
            with bridged.start_as_current_span("llamaindex.retrieve") as span:
                attrs = span._ns.attributes

        assert attrs["session.id"] == "s-42"
        assert attrs["user.id"] == "u-7"
        assert attrs["tenant.id"] == "acme"

    def test_nested_bridged_spans_parent_to_each_other(self, bridged):
        with ns_probe.turn("agent.turn"):
            with bridged.start_as_current_span("outer") as outer:
                with bridged.start_as_current_span("inner") as inner:
                    pass

        assert inner._ns.parent_span_id == outer._ns.span_id
        assert inner._ns.trace_id == outer._ns.trace_id

    def test_the_span_context_reports_our_ids(self, bridged):
        """OTel identifies spans by int, we do by hex. An instrumentor that
        propagates context downstream reads this."""
        with ns_probe.turn("agent.turn") as t:
            with bridged.start_as_current_span("s") as span:
                ctx = span.get_span_context()

        assert format(ctx.trace_id, "032x") == t.trace_id
        assert ctx.is_remote is False


class TestTheRecordingSurface:
    def test_attributes_land_on_the_span(self, bridged):
        with bridged.start_as_current_span("s") as span:
            span.set_attribute("gen_ai.system", "openai")
            span.set_attributes({"gen_ai.usage.input_tokens": 42, "a": "b"})
            attrs = span._ns.attributes

        assert attrs["gen_ai.system"] == "openai"
        assert attrs["gen_ai.usage.input_tokens"] == 42
        assert attrs["a"] == "b"

    def test_a_renamed_span_keeps_the_new_name(self, bridged):
        """Instrumentors routinely open a span then rename it once they know what
        it turned out to be."""
        with bridged.start_as_current_span("placeholder") as span:
            span.update_name("llamaindex.query")

        assert span._ns.name == "llamaindex.query"

    def test_an_error_status_maps_to_ours(self, bridged):
        from opentelemetry.trace import Status, StatusCode

        with bridged.start_as_current_span("s") as span:
            span.set_status(Status(StatusCode.ERROR, "upstream refused"))
            code = span._ns.status_code

        from ns_probe.span import StatusCode as NsStatusCode

        assert code == NsStatusCode.ERROR

    def test_events_are_recorded(self, bridged):
        with bridged.start_as_current_span("s") as span:
            span.add_event("cache.miss", {"key": "abc"})

        assert any(e.name == "cache.miss" for e in span._ns.events)

    def test_an_exception_propagates_and_is_marked(self, bridged):
        """Fail-open must not mean swallowing the caller's error."""
        from ns_probe.span import StatusCode as NsStatusCode

        with pytest.raises(ValueError, match="boom"):
            with bridged.start_as_current_span("s") as span:
                captured = span
                raise ValueError("boom")

        assert captured._ns.status_code == NsStatusCode.ERROR

    def test_is_recording_flips_when_the_span_ends(self, bridged):
        span = bridged.start_span("manual")
        assert span.is_recording() is True
        span.end()
        assert span.is_recording() is False

    def test_ending_twice_is_harmless(self, bridged):
        """Some instrumentors end a span in both a finally and an exit hook."""
        span = bridged.start_span("manual")
        span.end()
        span.end()


class TestItDoesNotTakeSomethingItWasNotGiven:
    def test_it_refuses_to_replace_a_provider_someone_else_installed(self, monkeypatch):
        """A service running its own OTel SDK for its HTTP layer must not find its
        spans quietly redirected here because it imported ns_probe."""

        class _SomeoneElses(otel_trace.TracerProvider):
            def get_tracer(self, *a, **kw):
                raise AssertionError("should not be called")

        monkeypatch.setattr(otel_trace, "_TRACER_PROVIDER", _SomeoneElses(), raising=False)

        assert otel_bridge.install_otel_bridge() is False

    def test_force_takes_it_over_deliberately(self, monkeypatch):
        class _SomeoneElses(otel_trace.TracerProvider):
            def get_tracer(self, *a, **kw):
                raise AssertionError("should not be called")

        monkeypatch.setattr(otel_trace, "_TRACER_PROVIDER", _SomeoneElses(), raising=False)

        assert otel_bridge.install_otel_bridge(force=True) is True

    def test_the_env_switch_is_off_unless_asked(self, monkeypatch):
        monkeypatch.delenv(otel_bridge.BRIDGE_ENV, raising=False)
        assert otel_bridge.install_from_env() is False

    def test_the_env_switch_installs_when_set(self, monkeypatch):
        monkeypatch.setenv(otel_bridge.BRIDGE_ENV, "true")
        monkeypatch.setattr(otel_trace, "_TRACER_PROVIDER", None, raising=False)
        monkeypatch.setattr(
            otel_trace,
            "_TRACER_PROVIDER_SET_ONCE",
            type(otel_trace._TRACER_PROVIDER_SET_ONCE)(),
            raising=False,
        )
        assert otel_bridge.install_from_env() is True

    def test_importing_the_module_never_requires_opentelemetry(self):
        """The provider classes are built inside a function precisely so that
        `import ns_probe` does not drag in an optional dependency."""
        import ast
        import inspect

        tree = ast.parse(inspect.getsource(otel_bridge))
        top_level = [
            n
            for n in tree.body
            if isinstance(n, (ast.Import, ast.ImportFrom)) and "opentelemetry" in ast.dump(n)
        ]
        assert top_level == [], "opentelemetry is imported at module scope"


class TestAgainstARealInstrumentor:
    """The tests above use our own tracer, which is how a real bug got through.

    `int(kind)` on OTel's SpanKind looked correct and raises TypeError, because
    SpanKind is a plain Enum rather than an IntEnum. Nothing in this file caught
    it — every case passed None or a bare int. A genuine instrumentor passes the
    enum, and the resulting TypeError came out of the caller's HTTP request
    rather than merely losing a span.
    """

    def test_a_real_otel_spankind_is_accepted(self, bridged):
        """The regression. Every OTel instrumentor passes this type."""
        for kind in otel_trace.SpanKind:
            with bridged.start_as_current_span("s", kind=kind) as span:
                assert span._ns is not None

    def test_span_kinds_map_across_rather_than_defaulting(self, bridged):
        from ns_probe.span import SpanKind as NsSpanKind

        with bridged.start_as_current_span("s", kind=otel_trace.SpanKind.CLIENT) as span:
            assert span._ns.kind == NsSpanKind.CLIENT
        with bridged.start_as_current_span("s", kind=otel_trace.SpanKind.SERVER) as span:
            assert span._ns.kind == NsSpanKind.SERVER

    def test_an_unmodified_third_party_instrumentor_lands_in_our_trace(self, bridged):
        """End to end, with somebody else's code. This is the claim the whole
        module rests on: that the OpenLLMetry / OTel-contrib catalogue can be
        used as-is. Proving it with one real package is the difference between
        'satisfies the API' and 'works'."""
        instr = pytest.importorskip(
            "opentelemetry.instrumentation.urllib",
            reason="opentelemetry-instrumentation-urllib not installed",
        )
        import http.server
        import socketserver
        import threading
        import urllib.request

        from ns_probe.tracer import TracerProvider

        class _H(http.server.BaseHTTPRequestHandler):
            def do_GET(self):
                self.send_response(200)
                self.end_headers()
                self.wfile.write(b"ok")

            def log_message(self, *a):
                pass

        srv = socketserver.TCPServer(("127.0.0.1", 0), _H)
        port = srv.server_address[1]
        threading.Thread(target=srv.serve_forever, daemon=True).start()

        instrumentor = instr.URLLibInstrumentor()
        instrumentor.instrument()
        try:
            buf = TracerProvider.get_instance()._buffer
            buf.get_batch()  # drain anything already queued
            with ns_probe.turn("agent.turn", session_id="s-1") as t:
                urllib.request.urlopen(f"http://127.0.0.1:{port}/hello", timeout=2).read()

            spans = {s.name: s for s in buf.get_batch()}
            assert "GET" in spans, "the third-party instrumentor produced no span"
            emitted = spans["GET"]
            assert emitted.trace_id == t.trace_id, "third-party span orphaned"
            assert emitted.attributes["session.id"] == "s-1", "identity did not reach it"
            assert emitted.attributes.get("http.status_code") == 200
        finally:
            instrumentor.uninstrument()
            srv.shutdown()
