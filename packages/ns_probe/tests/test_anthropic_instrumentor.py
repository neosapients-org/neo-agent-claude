"""Anthropic, and the three ways its instrumentor lost every token it should have counted.

The processor computes `cost_usd` FROM tokens. A span with no tokens is not a gap you
notice — it is a zero, and a zero looks like a cheap model rather than a broken probe.
The Anthropic instrumentor shipped with three holes, each of which produces exactly that:

  1. It wrapped `Messages.create` and not `AsyncMessages.create`. Any async agent — which
     is to say any agent serving HTTP — recorded nothing at all.
  2. It had no streaming path. The OpenAI side has `_StreamWrapper` for precisely this
     reason: a streamed call returns an iterator, not an object with `.usage`, so reading
     `.usage` off it captures nothing. Anything streaming a reply to a user hit this.
  3. It ignored `cache_read_input_tokens` and `cache_creation_input_tokens`. With prompt
     caching on, most of the input moves into those fields, so the recorded input token
     count collapses to a fraction of what was actually billed.

A note on cache semantics, because it differs from OpenAI and getting it backwards
double-counts. OpenAI reports `prompt_tokens` as the total and `cached_tokens` as the
portion of it served from cache, so the OpenAI path *subtracts* to avoid counting twice.
Anthropic reports three non-overlapping buckets — `input_tokens` already EXCLUDES both
cache figures — so nothing is subtracted here. `gen_ai.usage.cached_tokens` means the same
thing on both sides ("input served from cache"), which is what lets one dashboard read both.

`anthropic` is not a dependency of this package, so these build a stub into sys.modules,
the same way the litellm tests do. The contract under test is "whatever the anthropic SDK
hands back, read usage off it the Anthropic way" — a stub tests that without pinning a
version of somebody else's library.
"""

from __future__ import annotations

import sys
import types

import pytest

from ns_probe import configure
from ns_probe import instrumentors


# --- stub anthropic ----------------------------------------------------------------

class _Usage:
    """Anthropic's usage block. The two cache fields are absent on an uncached call."""

    def __init__(self, input_tokens=11, output_tokens=7, cache_read=None, cache_creation=None):
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        if cache_read is not None:
            self.cache_read_input_tokens = cache_read
        if cache_creation is not None:
            self.cache_creation_input_tokens = cache_creation


class _Message:
    def __init__(self, model="claude-sonnet-5", usage=None, stop_reason="end_turn"):
        self.model = model
        self.usage = usage or _Usage()
        self.stop_reason = stop_reason
        self.content = [types.SimpleNamespace(type="text", text="hi")]


class _Event:
    """One server-sent event off a streamed message."""

    def __init__(self, type_, **kw):
        self.type = type_
        for k, v in kw.items():
            setattr(self, k, v)


def _stream_events(input_tokens=11, output_tokens=7, cache_read=None):
    """The event sequence Anthropic streams: input usage up front, output usage at the end."""
    return [
        _Event("message_start",
               message=_Message(usage=_Usage(input_tokens, 0, cache_read=cache_read))),
        _Event("content_block_delta", delta=types.SimpleNamespace(text="hi")),
        _Event("message_delta", usage=_Usage(0, output_tokens)),
        _Event("message_stop"),
    ]


@pytest.fixture
def fake_anthropic(monkeypatch):
    """An anthropic module shaped like the real one: sync and async Messages resources."""
    mod = types.ModuleType("anthropic")
    resources = types.ModuleType("anthropic.resources")
    calls = {"create": 0, "acreate": 0}

    class Messages:
        def create(self, **kwargs):
            calls["create"] += 1
            if kwargs.get("stream"):
                return iter(_stream_events(**kwargs.pop("_usage", {})))
            return _Message(usage=_Usage(**kwargs.pop("_usage", {})))

    class AsyncMessages:
        async def create(self, **kwargs):
            calls["acreate"] += 1
            if kwargs.get("stream"):
                async def _agen():
                    for e in _stream_events(**kwargs.pop("_usage", {})):
                        yield e
                return _agen()
            return _Message(usage=_Usage(**kwargs.pop("_usage", {})))

    resources.Messages = Messages
    resources.AsyncMessages = AsyncMessages
    mod.resources = resources
    mod._calls = calls

    monkeypatch.setitem(sys.modules, "anthropic", mod)
    monkeypatch.setitem(sys.modules, "anthropic.resources", resources)
    configure(service_name="anthropic-tests", endpoint="http://127.0.0.1:9/v1/traces")
    instrumentors._llm_span_active.set(False)
    yield mod


@pytest.fixture
def captured(monkeypatch):
    """Every span the instrumentor opens, in order.

    Records `start_span_no_context` as well as `start_span`. A streamed call cannot use a
    `with` block — the span has to outlive the function that returns the iterator — so the
    streaming path opens a detached span and ends it when the stream is exhausted. A fixture
    watching only `start_span` sees nothing on those calls and would report the streaming
    holes as fixed while they were still wide open.
    """
    spans = []
    real = instrumentors.get_tracer

    class _Recorder:
        def __init__(self, inner):
            self._inner = inner

        def __getattr__(self, item):
            return getattr(self._inner, item)

        def start_span(self, name, **kw):
            return _Tracked(self._inner.start_span(name, **kw), spans, name)

        def start_span_no_context(self, name, **kw):
            span = self._inner.start_span_no_context(name, **kw)
            spans.append((name, span))
            return span

    class _Tracked:
        def __init__(self, cm, sink, name):
            self._cm, self._sink, self._name = cm, sink, name

        def __enter__(self):
            span = self._cm.__enter__()
            self._sink.append((self._name, span))
            return span

        def __exit__(self, *exc):
            return self._cm.__exit__(*exc)

    monkeypatch.setattr(instrumentors, "get_tracer", lambda n: _Recorder(real(n)))
    return spans


MSGS = [{"role": "user", "content": "hello"}]


# --- 1. the async hole -------------------------------------------------------------

class TestAsyncCallsAreRecorded:
    """Hole 1: an async agent recorded nothing, because only the sync class was wrapped."""

    @pytest.mark.asyncio
    async def test_an_async_message_produces_an_llm_call_span_with_tokens(
        self, fake_anthropic, captured
    ):
        assert instrumentors._instrument_anthropic() is True
        import anthropic

        await anthropic.resources.AsyncMessages().create(
            model="claude-sonnet-5", messages=MSGS
        )

        names = [n for n, _ in captured]
        assert "llm.call" in names, "no llm.call span — this is the async-blind state again"
        span = dict(captured)["llm.call"]
        assert span.attributes["gen_ai.usage.input_tokens"] == 11
        assert span.attributes["gen_ai.usage.output_tokens"] == 7
        assert span.attributes["gen_ai.system"] == "anthropic"

    @pytest.mark.asyncio
    async def test_the_async_caller_still_gets_its_message(self, fake_anthropic):
        assert instrumentors._instrument_anthropic() is True
        import anthropic

        result = await anthropic.resources.AsyncMessages().create(
            model="claude-sonnet-5", messages=MSGS
        )

        assert result.model == "claude-sonnet-5"
        assert anthropic._calls["acreate"] == 1


# --- 2. the streaming hole ---------------------------------------------------------

class TestStreamedCallsAreRecorded:
    """Hole 2: a streamed call returns an iterator, so reading .usage off it found nothing."""

    def test_a_streamed_message_records_input_and_output_tokens(
        self, fake_anthropic, captured
    ):
        assert instrumentors._instrument_anthropic() is True
        import anthropic

        stream = anthropic.resources.Messages().create(
            model="claude-sonnet-5", messages=MSGS, stream=True
        )
        events = list(stream)

        assert [e.type for e in events] == [
            "message_start", "content_block_delta", "message_delta", "message_stop"
        ], "the caller must still see every event, unchanged"
        span = dict(captured)["llm.call"]
        assert span.attributes["gen_ai.usage.input_tokens"] == 11, "input usage rides on message_start"
        assert span.attributes["gen_ai.usage.output_tokens"] == 7, "output usage rides on message_delta"

    @pytest.mark.asyncio
    async def test_an_async_streamed_message_records_tokens(self, fake_anthropic, captured):
        assert instrumentors._instrument_anthropic() is True
        import anthropic

        stream = await anthropic.resources.AsyncMessages().create(
            model="claude-sonnet-5", messages=MSGS, stream=True
        )
        events = [e async for e in stream]

        assert len(events) == 4
        span = dict(captured)["llm.call"]
        assert span.attributes["gen_ai.usage.input_tokens"] == 11
        assert span.attributes["gen_ai.usage.output_tokens"] == 7


# --- 3. the cache hole -------------------------------------------------------------

class TestCacheTokensAreRecorded:
    """Hole 3: with caching on, most of the input lives in fields nobody read."""

    def test_cache_read_tokens_land_on_the_parity_attribute(self, fake_anthropic, captured):
        assert instrumentors._instrument_anthropic() is True
        import anthropic

        anthropic.resources.Messages().create(
            model="claude-sonnet-5", messages=MSGS,
            _usage={"input_tokens": 11, "output_tokens": 7, "cache_read": 900},
        )

        span = dict(captured)["llm.call"]
        assert span.attributes["gen_ai.usage.cached_tokens"] == 900
        assert span.attributes["gen_ai.usage.input_tokens"] == 11, (
            "Anthropic's input_tokens already excludes cache reads — subtracting "
            "here would under-report the bill"
        )

    def test_cache_creation_tokens_are_recorded_separately(self, fake_anthropic, captured):
        assert instrumentors._instrument_anthropic() is True
        import anthropic

        anthropic.resources.Messages().create(
            model="claude-sonnet-5", messages=MSGS,
            _usage={"input_tokens": 11, "output_tokens": 7, "cache_creation": 400},
        )

        span = dict(captured)["llm.call"]
        assert span.attributes["gen_ai.usage.cache_creation_tokens"] == 400

    def test_an_uncached_call_does_not_invent_cache_attributes(self, fake_anthropic, captured):
        assert instrumentors._instrument_anthropic() is True
        import anthropic

        anthropic.resources.Messages().create(model="claude-sonnet-5", messages=MSGS)

        attrs = dict(captured)["llm.call"].attributes
        assert "gen_ai.usage.cached_tokens" not in attrs
        assert "gen_ai.usage.cache_creation_tokens" not in attrs

    def test_a_streamed_call_records_cache_reads_too(self, fake_anthropic, captured):
        assert instrumentors._instrument_anthropic() is True
        import anthropic

        list(anthropic.resources.Messages().create(
            model="claude-sonnet-5", messages=MSGS, stream=True,
            _usage={"input_tokens": 11, "output_tokens": 7, "cache_read": 900},
        ))

        assert dict(captured)["llm.call"].attributes["gen_ai.usage.cached_tokens"] == 900


# --- guards ------------------------------------------------------------------------

class TestOnlyOneSpanPerModelCall:
    def test_a_nested_instrumented_call_does_not_open_a_second_span(
        self, fake_anthropic, captured
    ):
        assert instrumentors._instrument_anthropic() is True
        import anthropic

        with instrumentors._llm_span(instrumentors.get_tracer("outer")) as outer:
            anthropic.resources.Messages().create(model="claude-sonnet-5", messages=MSGS)

        llm_spans = [n for n, _ in captured if n == "llm.call"]
        assert len(llm_spans) == 1, "one model call produced two spans — cost doubles"
        assert "gen_ai.usage.input_tokens" not in outer.attributes


class TestItStaysOutOfTheWay:
    def test_instrumenting_twice_is_a_no_op(self, fake_anthropic):
        assert instrumentors._instrument_anthropic() is True
        assert instrumentors._instrument_anthropic() is True
        import anthropic

        anthropic.resources.Messages().create(model="claude-sonnet-5", messages=MSGS)
        assert anthropic._calls["create"] == 1, "double-wrapped: called twice"

    def test_a_missing_anthropic_is_not_an_error(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "anthropic", None)
        assert instrumentors._instrument_anthropic() is False

    def test_a_build_without_the_async_resource_still_wraps_the_sync_one(
        self, fake_anthropic, captured
    ):
        import anthropic

        del anthropic.resources.AsyncMessages

        assert instrumentors._instrument_anthropic() is True
        anthropic.resources.Messages().create(model="claude-sonnet-5", messages=MSGS)
        assert "llm.call" in [n for n, _ in captured]
