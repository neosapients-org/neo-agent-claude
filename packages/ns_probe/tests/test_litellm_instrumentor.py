"""LiteLLM, and the reason it is worth instrumenting at all.

CrewAI does not talk to a provider SDK. Neither do a growing number of other
frameworks — they hand the call to LiteLLM, which picks a provider and calls it.
Before this instrumentor those calls arrived as bare httpx spans with no token
counts, and the processor computes `cost_usd` FROM tokens, so an entire CrewAI
agent read as costing nothing. A missing span is a visible gap. A span with no
tokens is a wrong number, and nothing about it looks wrong.

The second half of these tests is the part that is easy to get wrong. LiteLLM
reaches OpenAI-compatible providers by calling the same `openai` client ns_probe
also patches, so the naive version of this instrumentor produces TWO llm.call
spans for one completion, each carrying the same usage — and the bill doubles.
`_llm_span` is what stops that, and `TestOnlyOneSpanPerModelCall` is what stops
someone removing it.

litellm is not a dependency of this package and is not installed in CI, so these
build a stub into sys.modules. That is deliberate: the instrumentor's contract is
"whatever litellm returns, read usage off it the OpenAI way", and a stub tests
that contract without pinning a version of somebody else's library.
"""

from __future__ import annotations

import sys
import types

import pytest

from ns_probe import configure
from ns_probe import instrumentors


class _Usage:
    def __init__(self, prompt_tokens: int, completion_tokens: int) -> None:
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens


class _Message:
    def __init__(self, content: str) -> None:
        self.content = content


class _Choice:
    def __init__(self, content: str, finish_reason: str = "stop") -> None:
        self.message = _Message(content)
        self.finish_reason = finish_reason


class _Response:
    """LiteLLM normalises every provider onto this shape, whoever answered."""

    def __init__(self, model="gpt-4o-mini", content="hi", pt=11, ct=7) -> None:
        self.model = model
        self.choices = [_Choice(content)]
        self.usage = _Usage(pt, ct)


@pytest.fixture
def fake_litellm(monkeypatch):
    """A litellm module with the four entry points the instrumentor wraps."""
    mod = types.ModuleType("litellm")
    calls = {"completion": 0, "acompletion": 0, "embedding": 0, "aembedding": 0}

    def completion(**kwargs):
        calls["completion"] += 1
        return _Response()

    async def acompletion(**kwargs):
        calls["acompletion"] += 1
        return _Response()

    def embedding(**kwargs):
        calls["embedding"] += 1
        return _Response()

    async def aembedding(**kwargs):
        calls["aembedding"] += 1
        return _Response()

    mod.completion = completion
    mod.acompletion = acompletion
    mod.embedding = embedding
    mod.aembedding = aembedding
    mod._calls = calls

    monkeypatch.setitem(sys.modules, "litellm", mod)
    configure(service_name="litellm-tests", endpoint="http://127.0.0.1:9/v1/traces")
    instrumentors._llm_span_active.set(False)
    yield mod


@pytest.fixture
def captured(monkeypatch):
    """Every span the instrumentor opens, in order."""
    spans = []
    real = instrumentors.get_tracer

    class _Recorder:
        def __init__(self, inner):
            self._inner = inner

        def start_span(self, name, **kw):
            return _Tracked(self._inner.start_span(name, **kw), spans, name)

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


class TestItRecordsWhatTheHttpSpanCouldNot:
    def test_a_completion_produces_an_llm_call_span_with_tokens(self, fake_litellm, captured):
        assert instrumentors._instrument_litellm() is True
        import litellm

        litellm.completion(model="gpt-4o-mini", messages=[{"role": "user", "content": "hi"}])

        names = [n for n, _ in captured]
        assert "llm.call" in names, "no llm.call span — this is the httpx-only state again"
        span = dict((n, s) for n, s in captured)["llm.call"]
        assert span.attributes["gen_ai.usage.input_tokens"] == 11
        assert span.attributes["gen_ai.usage.output_tokens"] == 7

    def test_the_provider_is_read_from_the_model_prefix(self, fake_litellm, captured):
        """ "anthropic/claude-…" is a routing decision worth seeing in the trace."""
        instrumentors._instrument_litellm()
        import litellm

        litellm.completion(model="anthropic/claude-sonnet-4", messages=[])

        span = dict((n, s) for n, s in captured)["llm.call"]
        assert span.attributes["gen_ai.system"] == "anthropic"
        assert span.attributes["gen_ai.request.model"] == "anthropic/claude-sonnet-4"

    def test_a_bare_model_name_is_attributed_to_openai(self, fake_litellm, captured):
        instrumentors._instrument_litellm()
        import litellm

        litellm.completion(model="gpt-4o-mini", messages=[])

        assert dict(captured)["llm.call"].attributes["gen_ai.system"] == "openai"

    def test_the_span_says_which_layer_produced_it(self, fake_litellm, captured):
        """Two frameworks reaching one provider are worth telling apart when a
        cost figure looks wrong."""
        instrumentors._instrument_litellm()
        import litellm

        litellm.completion(model="gpt-4o-mini", messages=[])

        assert dict(captured)["llm.call"].attributes["gen_ai.route"] == "litellm"

    @pytest.mark.asyncio
    async def test_acompletion_is_wrapped_too(self, fake_litellm, captured):
        """CrewAI's async paths would otherwise be the uninstrumented half."""
        instrumentors._instrument_litellm()
        import litellm

        await litellm.acompletion(model="gpt-4o-mini", messages=[])

        assert dict(captured)["llm.call"].attributes["gen_ai.usage.input_tokens"] == 11

    def test_the_caller_still_gets_its_result(self, fake_litellm):
        instrumentors._instrument_litellm()
        import litellm

        result = litellm.completion(model="gpt-4o-mini", messages=[])

        assert result.choices[0].message.content == "hi"
        assert litellm._calls["completion"] == 1


class TestOnlyOneSpanPerModelCall:
    """The double-count guard. LiteLLM calls the same openai client we patch."""

    def test_a_nested_instrumented_call_does_not_open_a_second_span(self, fake_litellm, captured):
        instrumentors._instrument_litellm()
        tracer = instrumentors.get_tracer("test")

        with instrumentors._llm_span(tracer) as outer:
            with instrumentors._llm_span(tracer) as inner:
                inner.set_attribute("gen_ai.usage.input_tokens", 999)

        llm_spans = [n for n, _ in captured if n == "llm.call"]
        assert len(llm_spans) == 1, "one model call produced two spans — cost doubles"
        assert isinstance(inner, instrumentors._NoSpan)
        assert "gen_ai.usage.input_tokens" not in outer.attributes

    def test_the_suppression_lifts_after_the_outer_call_finishes(self, fake_litellm, captured):
        """A flag that leaks would silence every model call for the rest of the
        process — a far worse failure than the duplicate it prevents."""
        instrumentors._instrument_litellm()
        import litellm

        litellm.completion(model="gpt-4o-mini", messages=[])
        litellm.completion(model="gpt-4o-mini", messages=[])

        assert len([n for n, _ in captured if n == "llm.call"]) == 2

    def test_the_flag_is_reset_even_when_the_call_raises(self, fake_litellm, captured):
        tracer = instrumentors.get_tracer("test")

        with pytest.raises(ValueError):
            with instrumentors._llm_span(tracer):
                raise ValueError("boom")

        assert instrumentors._llm_span_active.get() is False


class TestItStaysOutOfTheWay:
    def test_instrumenting_twice_is_a_no_op(self, fake_litellm):
        assert instrumentors._instrument_litellm() is True
        assert instrumentors._instrument_litellm() is True
        import litellm

        litellm.completion(model="gpt-4o-mini", messages=[])
        assert litellm._calls["completion"] == 1, "double-wrapped: called twice"

    def test_a_missing_litellm_is_not_an_error(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "litellm", None)
        assert instrumentors._instrument_litellm() is False

    def test_a_build_without_every_entry_point_still_wraps_what_it_has(self, monkeypatch):
        """Older and trimmed builds do not ship all four."""
        mod = types.ModuleType("litellm")
        mod.completion = lambda **kw: _Response()
        monkeypatch.setitem(sys.modules, "litellm", mod)

        assert instrumentors._instrument_litellm() is True

    def test_a_module_with_no_entry_points_reports_failure(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "litellm", types.ModuleType("litellm"))
        assert instrumentors._instrument_litellm() is False

    def test_the_agent_still_gets_its_error(self, fake_litellm):
        """Fail-open cuts both ways: we must not swallow the caller's exception."""
        import litellm

        def boom(**kwargs):
            raise RuntimeError("provider is down")

        litellm.completion = boom
        instrumentors._instrument_litellm()

        with pytest.raises(RuntimeError, match="provider is down"):
            litellm.completion(model="gpt-4o-mini", messages=[])
