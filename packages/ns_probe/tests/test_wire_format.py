"""What actually goes on the wire, and the import-order trap.

Two things nothing covered.

**The wire format is a contract with a reader we do not control.** ns_processor
populates the ClickHouse attributes map from this JSON, and it reads only some of
OTLP's value kinds. Every encoding choice in `_attr_to_otlp` is a decision about
that reader, not about OTLP — which means a well-meaning "let's emit proper
doubleValue, it's the spec" would silently zero every latency and every cost, and
no test would have objected. These pin the choices to their reasons.

**Import order.** `instrument_all()` wraps library functions by rebinding them on
their module, so it can only affect names looked up *after* it runs. A caller who
did `from litellm import completion` first holds the original and is never traced.
The SOP calls this out twice and it is the top entry in its troubleshooting table,
because the symptom is a trace with no model calls in it and nothing raises.
"""

from __future__ import annotations

import sys
import types

import pytest

from ns_probe.exporter import OTLPExporter
from ns_probe.span import Span, SpanKind


def _attr(value, key="k"):
    """The OTLP value object a single attribute becomes."""
    return OTLPExporter(endpoint="http://127.0.0.1:9/v1/traces")._attr_to_otlp(key, value)["value"]


class TestAttributeEncoding:
    def test_a_float_is_sent_as_text_not_as_a_double(self):
        """ns_processor reads intValue and stringValue and drops doubleValue, so a
        spec-correct doubleValue lands in ClickHouse as 0. Latency and cost are
        floats, so this one choice is the difference between real numbers and
        zeroes."""
        v = _attr(12.5)

        assert "doubleValue" not in v
        assert v["stringValue"] == "12.5"

    def test_a_small_float_keeps_its_magnitude(self):
        """repr(), not repr(round(x, 4)). Rounding turned a cost of 0.000021 into
        "0.0" and flattened every rate below 0.00005 to zero."""
        assert _attr(0.000021)["stringValue"] == "2.1e-05"
        assert float(_attr(0.000021)["stringValue"]) == 0.000021

    def test_a_float_round_trips_exactly(self):
        """outcome_ledger relies on this too, so both wire paths agree."""
        for f in (0.1, 1 / 3, 1e-9, 123456.789):
            assert float(_attr(f)["stringValue"]) == f

    def test_an_int_is_text_in_an_int_field(self):
        """OTLP models a 64-bit int as a JSON string — larger than JSON numbers
        safely carry."""
        v = _attr(42)
        assert v["intValue"] == "42" and isinstance(v["intValue"], str)

    def test_a_real_bool_is_emitted_as_a_bool(self):
        """Which is exactly why as_bool() exists. The exporter is correct here;
        the reader downstream is what mis-decodes boolValue, so agent code sends
        booleans as the STRINGS "true"/"false" instead. If this ever starts
        emitting text, as_bool's whole reason disappears — check the reader first."""
        assert _attr(True) == {"boolValue": True}

    def test_as_bool_output_travels_as_a_plain_string(self):
        from ns_probe import as_bool

        assert _attr(as_bool(True)) == {"stringValue": "true"}
        assert _attr(as_bool(False)) == {"stringValue": "false"}

    def test_a_bool_is_checked_before_int(self):
        """bool is a subclass of int in Python. Order the isinstance chain the
        other way and every True becomes intValue "1"."""
        assert "boolValue" in _attr(False)
        assert "intValue" not in _attr(False)

    def test_a_string_list_becomes_an_array(self):
        assert _attr(["a", "b"])["arrayValue"] == {
            "values": [{"stringValue": "a"}, {"stringValue": "b"}]
        }

    def test_something_unserialisable_degrades_to_text(self):
        class Odd:
            def __str__(self):
                return "odd"

        assert _attr(Odd())["stringValue"] == "odd"


class TestSpanEnvelope:
    def _otlp(self, span):
        return OTLPExporter(endpoint="http://127.0.0.1:9/v1/traces")._span_to_otlp(span)

    def test_span_kind_is_shifted_because_otlp_is_one_indexed(self):
        """Off by one here mislabels every span's kind, and nothing errors."""
        span = Span(name="s", trace_id="a" * 32, span_id="b" * 16, kind=SpanKind.CLIENT)
        span.end()

        assert self._otlp(span)["kind"] == SpanKind.CLIENT + 1

    def test_timestamps_are_nanosecond_strings(self):
        span = Span(name="s", trace_id="a" * 32, span_id="b" * 16)
        span.end()
        out = self._otlp(span)

        assert isinstance(out["startTimeUnixNano"], str)
        assert isinstance(out["endTimeUnixNano"], str)
        assert int(out["endTimeUnixNano"]) >= int(out["startTimeUnixNano"])

    def test_an_unended_span_still_carries_an_end_time(self):
        """A span that never ended must not export a null the reader will choke on."""
        span = Span(name="s", trace_id="a" * 32, span_id="b" * 16)

        assert self._otlp(span)["endTimeUnixNano"] == str(span.start_time_ns)

    def test_ids_travel_as_the_hex_they_were_created_as(self):
        span = Span(name="s", trace_id="a" * 32, span_id="b" * 16)
        span.end()
        out = self._otlp(span)

        assert out["traceId"] == "a" * 32 and len(out["traceId"]) == 32
        assert out["spanId"] == "b" * 16 and len(out["spanId"]) == 16

    def test_the_payload_is_json_the_collector_can_parse(self):
        import json

        span = Span(name="s", trace_id="a" * 32, span_id="b" * 16, attributes={"n": 1.5})
        span.end()
        payload = OTLPExporter(endpoint="http://127.0.0.1:9/v1/traces")._spans_to_otlp([span])

        decoded = json.loads(payload)
        assert "resourceSpans" in decoded


class TestTheImportOrderTrap:
    """`configure()` and `instrument_all()` must run above the other imports."""

    @pytest.fixture
    def fake_lib(self, monkeypatch):
        mod = types.ModuleType("litellm")
        mod.completion = lambda **kw: "original"
        monkeypatch.setitem(sys.modules, "litellm", mod)
        return mod

    def test_a_name_imported_before_instrumentation_is_never_traced(self, fake_lib):
        """The trap, demonstrated. `from litellm import completion` binds the
        function object into the caller's namespace; rebinding it on the module
        afterwards cannot reach that binding. The agent runs perfectly and its
        trace contains no model calls — the SOP's top troubleshooting entry."""
        from litellm import completion as bound_early

        from ns_probe.instrumentors import _instrument_litellm

        assert _instrument_litellm() is True

        assert bound_early is not sys.modules["litellm"].completion
        assert not hasattr(bound_early, "__wrapped__"), "the early binding was somehow patched"

    def test_a_name_looked_up_after_instrumentation_is_traced(self, fake_lib):
        """The same import written the supported way."""
        from ns_probe.instrumentors import _instrument_litellm

        assert _instrument_litellm() is True

        import litellm

        assert hasattr(litellm.completion, "__wrapped__"), "module attribute was not wrapped"

    def test_instrumenting_a_module_that_is_not_imported_yet_is_not_an_error(self, monkeypatch):
        """Absence is the normal case for most of the libraries we try."""
        monkeypatch.setitem(sys.modules, "litellm", None)
        from ns_probe.instrumentors import _instrument_litellm

        assert _instrument_litellm() is False
