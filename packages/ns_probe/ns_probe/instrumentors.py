"""
Auto-instrumentors for common libraries.

What instrument_all() actually covers:
- OpenAI (chat completions, embeddings)
- Anthropic (messages)
- LangGraph (see langgraph_instrumentor: node spans and context propagation)
- HTTPX / Requests (outgoing HTTP)
- asyncpg / psycopg2 (database calls)

What it does NOT cover, despite what you may have read elsewhere: LangChain
outside LangGraph, LlamaIndex, CrewAI, LiteLLM, and the Bedrock / Vertex /
Gemini SDKs. Those were to arrive through OpenLLMetry, and _try_openllmetry()
is disabled — Traceloop.init() installs its own TracerProvider, so its spans
orphan instead of parenting under our root. Anything reaching a model through
the openai or anthropic SDK is still captured, whatever framework wraps it;
anything reaching one another way currently arrives as a bare HTTP span with
no token counts, and therefore no cost.

The goal is out-of-the-box visibility into LLM calls and external HTTP
requests without requiring code changes.
"""

from __future__ import annotations

import contextlib
import contextvars
import logging
import functools
import os
import time
from typing import Any, Callable, Optional, TypeVar
from urllib.parse import urlparse

from .config import get_config
from .tracer import get_tracer, get_current_span_context, TracerProvider
from .span import SpanKind, StatusCode

logger = logging.getLogger("ns_probe.instrumentors")

F = TypeVar("F", bound=Callable[..., Any])


# One llm.call span per logical model call, however many instrumented layers it
# passes through.
#
# LiteLLM reaches most providers by calling the same openai/anthropic client we
# also patch, so without this a single completion() produces two llm.call spans,
# each carrying the same tokens. The processor prices cost FROM tokens, so that
# is not a cosmetic duplicate — it doubles the reported spend. The outermost
# instrumented layer wins, because it is the one that knows the model string the
# caller actually asked for (litellm's "anthropic/claude-..." rather than the
# provider-native name it rewrites to).
_llm_span_active: contextvars.ContextVar = contextvars.ContextVar(
    "ns_probe_llm_span_active", default=False
)


class _NoSpan:
    """A span-shaped object that records nothing.

    Lets a suppressed call site keep its `with` block and every set_attribute
    inside it exactly as written, rather than growing a branch per attribute.
    """

    __slots__ = ()

    def set_attribute(self, *a: Any, **kw: Any) -> None:
        return None

    def set_attributes(self, *a: Any, **kw: Any) -> None:
        return None

    def set_status(self, *a: Any, **kw: Any) -> None:
        return None

    def add_event(self, *a: Any, **kw: Any) -> None:
        return None

    def record_exception(self, *a: Any, **kw: Any) -> None:
        return None

    def end(self, *a: Any, **kw: Any) -> None:
        return None

    @property
    def status_code(self) -> Any:
        return StatusCode.UNSET


@contextlib.contextmanager
def _llm_span(tracer: Any):
    """Open an llm.call span, or hand back a no-op if one is already open."""
    if _llm_span_active.get():
        yield _NoSpan()
        return
    token = _llm_span_active.set(True)
    try:
        with tracer.start_span("llm.call", kind=SpanKind.CLIENT) as span:
            yield span
    finally:
        _llm_span_active.reset(token)


class _UsageStreamProxy:
    """Wraps an OpenAI streaming response so token usage + completion text are captured.

    A streamed `chat.completions.create(stream=True)` returns an async iterator, not a
    response with `.usage`/`.choices` — so the normal (non-streaming) capture records no
    tokens/cost. This proxy transparently forwards iteration to the underlying stream
    while accumulating the delta content and reading the final usage chunk (emitted when
    `stream_options={"include_usage": True}`), then finalizes and exports the llm.call
    span when the stream is exhausted or closed. Non-iteration attribute access and the
    async-context-manager protocol are delegated to the underlying stream so callers
    (e.g. langchain) behave exactly as before.
    """

    def __init__(self, stream: Any, span: Any, start_time: float) -> None:
        self._stream = stream
        self._span = span
        self._start = start_time
        self._parts: list = []
        self._finalized = False
        self._iter = None

    def __aiter__(self):
        self._iter = self._stream.__aiter__()
        return self

    async def __anext__(self):
        if self._iter is None:
            self._iter = self._stream.__aiter__()
        try:
            chunk = await self._iter.__anext__()
        except StopAsyncIteration:
            self._finalize()
            raise
        self._observe_chunk(chunk)
        return chunk

    def _observe_chunk(self, chunk: Any) -> None:
        try:
            usage = getattr(chunk, "usage", None)
            if usage is not None:
                pt = getattr(usage, "prompt_tokens", None)
                ct = getattr(usage, "completion_tokens", None)
                if pt is not None:
                    self._span.set_attribute("gen_ai.usage.input_tokens", pt)
                if ct is not None:
                    self._span.set_attribute("gen_ai.usage.output_tokens", ct)
            model = getattr(chunk, "model", None)
            if model:
                self._span.set_attribute("gen_ai.response.model", model)
            choices = getattr(chunk, "choices", None)
            if choices:
                delta = getattr(choices[0], "delta", None)
                piece = getattr(delta, "content", None) if delta is not None else None
                if piece:
                    self._parts.append(piece)
                fr = getattr(choices[0], "finish_reason", None)
                if fr:
                    self._span.set_attribute("gen_ai.response.finish_reason", fr)
        except Exception:
            pass

    def _finalize(self) -> None:
        if self._finalized:
            return
        self._finalized = True
        if isinstance(self._span, _NoSpan):
            # Suppressed by an outer instrumented layer. Nothing to end, and
            # nothing that may be pushed to the exporter.
            return
        try:
            self._span.set_attribute("gen_ai.latency_ms", (time.time() - self._start) * 1000)
            text = "".join(self._parts)
            if text:
                self._span.set_attribute("gen_ai.completion", _truncate(text, 1000))
                self._span.set_attribute("traceloop.entity.output", text)
            if self._span.status_code == StatusCode.UNSET:
                self._span.set_status(StatusCode.OK)
        except Exception:
            pass
        finally:
            self._span.end()
            try:
                provider = TracerProvider.get_instance()
                if provider is not None and provider._buffer is not None:
                    provider._buffer.push(self._span)
            except Exception:
                pass

    async def __aenter__(self):
        if hasattr(self._stream, "__aenter__"):
            await self._stream.__aenter__()
        return self

    async def __aexit__(self, *exc):
        try:
            if hasattr(self._stream, "__aexit__"):
                return await self._stream.__aexit__(*exc)
        finally:
            self._finalize()

    async def close(self):
        try:
            if hasattr(self._stream, "close"):
                await self._stream.close()
        finally:
            self._finalize()

    def parse(self, *args, **kwargs):
        # Raw-response streaming: langchain calls response.parse() to get the real
        # AsyncStream. Re-wrap it so we keep observing chunks (usage/content). If parse
        # is async or returns a non-iterable, pass it through untouched (no capture,
        # but no breakage).
        try:
            parsed = self._stream.parse(*args, **kwargs)
        except Exception:
            raise
        if hasattr(parsed, "__aiter__"):
            self._stream = parsed
            self._iter = None
            return self
        return parsed

    def __getattr__(self, name):
        # Delegate any other attribute/method to the wrapped stream.
        return getattr(self._stream, name)


async def _aparse_openai(result: Any) -> Any:
    """Return the parsed OpenAI response.

    langchain often calls the client with raw-response mode, so the instrumentor
    receives a LegacyAPIResponse / AsyncAPIResponse whose real ChatCompletion (with
    usage/choices) is behind ``.parse()``. If the object already exposes usage/choices
    it's returned as-is. ``.parse()`` caches, so calling it here doesn't disturb the
    caller parsing it later. Fail-open: returns the original on any error.
    """
    if hasattr(result, "usage") or hasattr(result, "choices"):
        return result
    parse = getattr(result, "parse", None)
    if parse is None:
        return result
    try:
        import inspect

        out = parse()
        if inspect.isawaitable(out):
            out = await out
        return out
    except Exception:
        return result


def _parse_openai(result: Any) -> Any:
    """Sync variant of :func:`_aparse_openai` for the sync instrumentor path."""
    if hasattr(result, "usage") or hasattr(result, "choices"):
        return result
    parse = getattr(result, "parse", None)
    if parse is None:
        return result
    try:
        return parse()
    except Exception:
        return result


# URLs that should NOT be traced by the HTTP instrumentors.
#
# Only protocol-level facts belong in this tuple. Anything naming a particular
# deployment does not: it is meaningless to every other installation, and the
# collector each installation actually uses is derived below instead.
_EXCLUDED_URL_SUBSTRINGS = (
    # The OTLP paths and the two standard OTLP ports. A span describing the
    # request that ships spans is a feedback loop, not a trace.
    "/v1/traces",
    "/v1/logs",
    ":4317",
    ":4318",
    # Already covered by a dedicated LLM instrumentor, which records the model,
    # the tokens and therefore the cost. Tracing these again as plain HTTP would
    # put every model call in the trace twice, once without any of that.
    "api.openai.com",
    "api.anthropic.com",
)

# Extra substrings, comma-separated. For collectors that share a host with
# application traffic, or an internal observability API that should stay out of
# the traces it serves.
EXCLUDED_URLS_ENV = "NS_PROBE_EXCLUDED_URLS"

# (endpoint, exclusions) — recomputed when configure() changes the endpoint.
# _should_skip_url runs on every outbound HTTP call, so this must not re-parse.
_exclusion_cache: Optional[tuple] = None


def _exclusions() -> tuple:
    """The static exclusions, plus the configured collector, plus env extras.

    The configured endpoint is added automatically so that pointing ns_probe at
    a collector is enough to keep that collector out of the traces. Requiring a
    second setting to stop the SDK tracing itself would be a trap, and the
    failure it produces — spans about sending spans, growing with the traffic
    they describe — is an unpleasant one to diagnose.
    """
    global _exclusion_cache

    try:
        endpoint = (get_config().endpoint or "").lower()
    except Exception:  # pragma: no cover — observability is best-effort
        endpoint = ""

    if _exclusion_cache is not None and _exclusion_cache[0] == endpoint:
        return _exclusion_cache[1]

    extra = []

    # host[:port] out of the endpoint, so the scheme and path do not narrow it.
    if endpoint:
        try:
            netloc = urlparse(endpoint).netloc
            if netloc:
                extra.append(netloc.split("@")[-1])
        except Exception:  # pragma: no cover
            pass

    for item in os.getenv(EXCLUDED_URLS_ENV, "").split(","):
        item = item.strip().lower()
        if item:
            extra.append(item)

    resolved = _EXCLUDED_URL_SUBSTRINGS + tuple(extra)
    _exclusion_cache = (endpoint, resolved)
    return resolved


def _should_skip_url(url: str) -> bool:
    """Return True if the URL should NOT produce an HTTP span."""
    url_lower = url.lower()
    return any(sub in url_lower for sub in _exclusions())


def instrument_all() -> None:
    """
    Instrument all supported libraries.

    This is called automatically by ns-probe-run unless --no-auto-instrument
    is specified.
    """
    # Track what was instrumented
    instrumented = []
    failed = []

    # Try OpenLLMetry instrumentors first (they're well-maintained)
    if _try_openllmetry():
        instrumented.append("OpenLLMetry (OpenAI, Anthropic, LangChain, etc.)")
    else:
        # Fallback to our own instrumentors
        if _instrument_openai():
            instrumented.append("OpenAI")
        else:
            failed.append("OpenAI")

        if _instrument_anthropic():
            instrumented.append("Anthropic")
        else:
            failed.append("Anthropic")

    # LangGraph instrumentation (context propagation through StateGraph)
    # This is critical - LangGraph breaks contextvars, we fix it
    from .langgraph_instrumentor import instrument_langgraph

    if instrument_langgraph():
        instrumented.append("LangGraph")

    # LiteLLM. Must be attempted BEFORE the HTTP clients so that when it calls a
    # provider SDK underneath, litellm's span is the one already open and the
    # inner one goes quiet — see _llm_span.
    if _instrument_litellm():
        instrumented.append("LiteLLM")

    # HTTP clients (usually not covered by OpenLLMetry)
    if _instrument_httpx():
        instrumented.append("httpx")

    if _instrument_requests():
        instrumented.append("requests")

    # Database instrumentation (asyncpg, psycopg2)
    if _instrument_asyncpg():
        instrumented.append("asyncpg")

    if _instrument_psycopg2():
        instrumented.append("psycopg2")

    if instrumented:
        logger.info(f"Instrumented: {', '.join(instrumented)}")

    if failed:
        logger.debug(f"Could not instrument (not installed?): {', '.join(failed)}")


def _try_openllmetry() -> bool:
    """
    Try to use OpenLLMetry's instrumentors.

    Disabled: Traceloop.init() creates a separate TracerProvider that
    produces orphan spans instead of child spans under ns_probe's root.
    ns_probe's built-in instrumentors (below) use get_tracer() which
    shares the same trace context, keeping LLM spans as children.
    """
    logger.debug(
        "Skipping OpenLLMetry (uses separate TracerProvider); using built-in instrumentors"
    )
    return False


def _instrument_openai() -> bool:
    """
    Instrument OpenAI library.

    Wraps:
        - client.chat.completions.create()
        - client.embeddings.create()

    Returns:
        True if instrumentation succeeded
    """
    try:
        import openai

        # Check if already instrumented
        if hasattr(openai, "_ns_probe_instrumented"):
            return True

        tracer = get_tracer("ns_probe.openai")

        # Instrument the completions create method
        original_create = openai.resources.chat.completions.Completions.create

        @functools.wraps(original_create)
        def traced_create(self: Any, *args: Any, **kwargs: Any) -> Any:
            model = kwargs.get("model", "unknown")

            with _llm_span(tracer) as span:
                span.set_attributes(
                    {
                        "gen_ai.system": "openai",
                        "gen_ai.request.model": model,
                    }
                )

                # Capture messages if present (compact + full for the trace UI)
                if "messages" in kwargs:
                    messages = kwargs["messages"]
                    span.set_attribute("gen_ai.prompt", _format_messages(messages))
                    span.set_attribute("traceloop.entity.input", _format_messages_full(messages))

                start_time = time.time()
                result = original_create(self, *args, **kwargs)
                latency_ms = (time.time() - start_time) * 1000

                span.set_attribute("gen_ai.latency_ms", latency_ms)

                # Unwrap raw HTTP response (langchain raw-response mode) for extraction.
                parsed = _parse_openai(result)

                # Extract response metadata
                if hasattr(parsed, "usage") and parsed.usage is not None:
                    span.set_attributes(
                        {
                            "gen_ai.usage.input_tokens": parsed.usage.prompt_tokens,
                            "gen_ai.usage.output_tokens": parsed.usage.completion_tokens,
                        }
                    )

                    # Extract cached tokens if available
                    if (
                        hasattr(parsed.usage, "prompt_tokens_details")
                        and parsed.usage.prompt_tokens_details
                    ):
                        cached = getattr(parsed.usage.prompt_tokens_details, "cached_tokens", 0)
                        if cached:
                            span.set_attribute("gen_ai.usage.cached_tokens", cached)
                            # Actual non-cached input is total - cached
                            span.set_attribute(
                                "gen_ai.usage.input_tokens", parsed.usage.prompt_tokens - cached
                            )

                    # GPT-5/o-series: Extract reasoning tokens from completion_tokens_details
                    if hasattr(parsed.usage, "completion_tokens_details"):
                        details = parsed.usage.completion_tokens_details
                        if details and hasattr(details, "reasoning_tokens"):
                            reasoning_tokens = details.reasoning_tokens or 0
                            span.set_attribute("gen_ai.usage.reasoning_tokens", reasoning_tokens)

                if hasattr(parsed, "model"):
                    span.set_attribute("gen_ai.response.model", parsed.model)

                if hasattr(parsed, "choices") and parsed.choices:
                    choice = parsed.choices[0]
                    if hasattr(choice, "finish_reason"):
                        span.set_attribute("gen_ai.response.finish_reason", choice.finish_reason)
                    if hasattr(choice, "message") and hasattr(choice.message, "content"):
                        _content = choice.message.content or ""
                        span.set_attribute("gen_ai.completion", _truncate(_content, 1000))
                        span.set_attribute("traceloop.entity.output", _content)

                return result

        openai.resources.chat.completions.Completions.create = traced_create

        # Instrument Responses API (GPT-5 visible reasoning)
        try:
            original_responses_create = openai.resources.responses.Responses.create

            @functools.wraps(original_responses_create)
            def traced_responses_create(self: Any, *args: Any, **kwargs: Any) -> Any:
                model = kwargs.get("model", "unknown")

                with _llm_span(tracer) as span:
                    span.set_attributes(
                        {
                            "gen_ai.system": "openai",
                            "gen_ai.request.model": model,
                        }
                    )

                    if "messages" in kwargs:
                        span.set_attribute("gen_ai.prompt", _format_messages(kwargs["messages"]))
                        span.set_attribute(
                            "traceloop.entity.input", _format_messages_full(kwargs["messages"])
                        )
                    elif "input" in kwargs:
                        # Capture input, handling list of messages or string
                        inp = kwargs["input"]
                        if isinstance(inp, list):
                            span.set_attribute("gen_ai.prompt", _format_messages(inp))
                            span.set_attribute("traceloop.entity.input", _format_messages_full(inp))
                        else:
                            span.set_attribute("gen_ai.prompt", _truncate(str(inp), 1000))
                            span.set_attribute("traceloop.entity.input", str(inp))

                    start_time = time.time()
                    result = original_responses_create(self, *args, **kwargs)
                    latency_ms = (time.time() - start_time) * 1000

                    span.set_attribute("gen_ai.latency_ms", latency_ms)

                    # Capture visible reasoning if available
                    if hasattr(result, "reasoning_summary"):
                        span.set_attribute("gen_ai.usage.reasoning_text", result.reasoning_summary)

                    if hasattr(result, "output_text"):
                        _out = result.output_text or ""
                        span.set_attribute("gen_ai.completion", _truncate(_out, 1000))
                        span.set_attribute("traceloop.entity.output", _out)

                    if hasattr(result, "usage"):
                        span.set_attributes(
                            {
                                "gen_ai.usage.input_tokens": getattr(
                                    result.usage, "input_tokens", 0
                                ),
                                "gen_ai.usage.output_tokens": getattr(
                                    result.usage, "output_tokens", 0
                                ),
                            }
                        )

                    return result

            openai.resources.responses.Responses.create = traced_responses_create
            logger.debug("Instrumented OpenAI Responses API")

        except AttributeError:
            # Responses API not available in this SDK version
            pass

        # Also instrument async version
        try:
            original_acreate = openai.resources.chat.completions.AsyncCompletions.create

            @functools.wraps(original_acreate)
            async def traced_acreate(self: Any, *args: Any, **kwargs: Any) -> Any:
                model = kwargs.get("model", "unknown")

                # STREAMING: the call returns an async iterator (no .usage/.choices).
                # Ask for a usage chunk, then wrap the stream so tokens + completion are
                # captured as it's consumed. Uses a manual span (ended by the proxy) since
                # a context manager would close before the stream is iterated.
                if kwargs.get("stream"):
                    so = kwargs.get("stream_options") or {}
                    if isinstance(so, dict) and "include_usage" not in so:
                        kwargs["stream_options"] = {**so, "include_usage": True}
                    parent = get_current_span_context()
                    span = tracer.start_span_no_context(
                        "llm.call",
                        kind=SpanKind.CLIENT,
                        trace_id=parent.trace_id if parent else None,
                        parent_span_id=parent.span_id if parent else None,
                    )
                    span.set_attribute("gen_ai.system", "openai")
                    span.set_attribute("gen_ai.request.model", model)
                    if "messages" in kwargs:
                        span.set_attribute("gen_ai.prompt", _format_messages(kwargs["messages"]))
                        span.set_attribute(
                            "traceloop.entity.input", _format_messages_full(kwargs["messages"])
                        )
                    start_time = time.time()
                    try:
                        result = await original_acreate(self, *args, **kwargs)
                    except Exception as e:
                        span.set_status(StatusCode.ERROR, str(e))
                        span.record_exception(e)
                        span.end()
                        prov = TracerProvider.get_instance()
                        if prov is not None and prov._buffer is not None:
                            prov._buffer.push(span)
                        raise
                    return _UsageStreamProxy(result, span, start_time)

                with _llm_span(tracer) as span:
                    span.set_attributes(
                        {
                            "gen_ai.system": "openai",
                            "gen_ai.request.model": model,
                        }
                    )

                    if "messages" in kwargs:
                        span.set_attribute("gen_ai.prompt", _format_messages(kwargs["messages"]))
                        span.set_attribute(
                            "traceloop.entity.input", _format_messages_full(kwargs["messages"])
                        )

                    start_time = time.time()
                    result = await original_acreate(self, *args, **kwargs)
                    latency_ms = (time.time() - start_time) * 1000

                    span.set_attribute("gen_ai.latency_ms", latency_ms)

                    # langchain requests the raw HTTP response (LegacyAPIResponse/
                    # AsyncAPIResponse); the parsed ChatCompletion with usage/choices is
                    # behind .parse(). Unwrap for attribute extraction but return the
                    # ORIGINAL object so the caller behaves exactly as before.
                    parsed = await _aparse_openai(result)

                    if hasattr(parsed, "usage") and parsed.usage is not None:
                        span.set_attributes(
                            {
                                "gen_ai.usage.input_tokens": parsed.usage.prompt_tokens,
                                "gen_ai.usage.output_tokens": parsed.usage.completion_tokens,
                            }
                        )

                        # Extract cached tokens if available
                        if (
                            hasattr(parsed.usage, "prompt_tokens_details")
                            and parsed.usage.prompt_tokens_details
                        ):
                            cached = getattr(parsed.usage.prompt_tokens_details, "cached_tokens", 0)
                            if cached:
                                span.set_attribute("gen_ai.usage.cached_tokens", cached)
                                # Actual non-cached input is total - cached
                                span.set_attribute(
                                    "gen_ai.usage.input_tokens", parsed.usage.prompt_tokens - cached
                                )

                    if hasattr(parsed, "model"):
                        span.set_attribute("gen_ai.response.model", parsed.model)

                    if hasattr(parsed, "choices") and parsed.choices:
                        choice = parsed.choices[0]
                        if hasattr(choice, "finish_reason"):
                            span.set_attribute(
                                "gen_ai.response.finish_reason", choice.finish_reason
                            )
                        if hasattr(choice, "message") and hasattr(choice.message, "content"):
                            _content = choice.message.content or ""
                            span.set_attribute("gen_ai.completion", _truncate(_content, 1000))
                            span.set_attribute("traceloop.entity.output", _content)

                    return result

            openai.resources.chat.completions.AsyncCompletions.create = traced_acreate
        except AttributeError:
            pass  # Async not available in this version

        openai._ns_probe_instrumented = True  # type: ignore
        logger.debug("Instrumented OpenAI")
        return True

    except ImportError:
        return False
    except Exception as e:
        logger.debug(f"Failed to instrument OpenAI: {e}")
        return False


def _anthropic_usage_attrs(usage: Any) -> dict:
    """Token attributes for one Anthropic usage block.

    Anthropic reports three NON-OVERLAPPING input buckets: `input_tokens` (billed at the
    normal rate), `cache_read_input_tokens` (billed at a fraction) and
    `cache_creation_input_tokens` (billed at a premium). OpenAI, by contrast, reports a
    single `prompt_tokens` total with `cached_tokens` as a subset of it — which is why the
    OpenAI path subtracts and this one must not. Subtracting here would silently
    under-report every cached call.

    `gen_ai.usage.cached_tokens` carries the same meaning on both sides ("input served
    from cache"), so one dashboard reads both providers without a special case.
    """
    attrs: dict = {}
    if usage is None:
        return attrs
    for attr, key in (
        ("input_tokens", "gen_ai.usage.input_tokens"),
        ("output_tokens", "gen_ai.usage.output_tokens"),
        ("cache_read_input_tokens", "gen_ai.usage.cached_tokens"),
        ("cache_creation_input_tokens", "gen_ai.usage.cache_creation_tokens"),
    ):
        value = getattr(usage, attr, None)
        if value is not None:
            attrs[key] = value
    return attrs


class _AnthropicStreamProxy:
    """Wraps a streamed Anthropic message so token usage is captured as it is consumed.

    A streamed `messages.create(stream=True)` returns an iterator of server-sent events,
    not an object with `.usage` — so the non-streaming capture reads nothing and the span
    records zero tokens. Usage arrives in two places: `message_start` carries the input
    counts (including both cache figures) and `message_delta` carries the output count.

    Handles sync and async iteration, because the SDK returns `Stream` or `AsyncStream`
    depending on which client made the call, and delegates everything else to the
    underlying stream so callers behave exactly as before.
    """

    def __init__(self, stream: Any, span: Any, start_time: float) -> None:
        self._stream = stream
        self._span = span
        self._start = start_time
        self._parts: list = []
        self._finalized = False
        self._iter = None

    def __getattr__(self, item):
        return getattr(self._stream, item)

    # --- sync iteration ---
    def __iter__(self):
        self._iter = iter(self._stream)
        return self

    def __next__(self):
        if self._iter is None:
            self._iter = iter(self._stream)
        try:
            event = next(self._iter)
        except StopIteration:
            self._finalize()
            raise
        self._observe_event(event)
        return event

    # --- async iteration ---
    def __aiter__(self):
        self._iter = self._stream.__aiter__()
        return self

    async def __anext__(self):
        if self._iter is None:
            self._iter = self._stream.__aiter__()
        try:
            event = await self._iter.__anext__()
        except StopAsyncIteration:
            self._finalize()
            raise
        self._observe_event(event)
        return event

    def _observe_event(self, event: Any) -> None:
        try:
            etype = getattr(event, "type", None)
            if etype == "message_start":
                message = getattr(event, "message", None)
                for key, value in _anthropic_usage_attrs(
                    getattr(message, "usage", None)
                ).items():
                    self._span.set_attribute(key, value)
                model = getattr(message, "model", None)
                if model:
                    self._span.set_attribute("gen_ai.response.model", model)
            elif etype == "message_delta":
                # Output tokens are cumulative on message_delta; the last one wins.
                for key, value in _anthropic_usage_attrs(
                    getattr(event, "usage", None)
                ).items():
                    if key == "gen_ai.usage.output_tokens":
                        self._span.set_attribute(key, value)
                stop_reason = getattr(getattr(event, "delta", None), "stop_reason", None)
                if stop_reason:
                    self._span.set_attribute("gen_ai.response.finish_reason", stop_reason)
            elif etype == "content_block_delta":
                piece = getattr(getattr(event, "delta", None), "text", None)
                if piece:
                    self._parts.append(piece)
        except Exception:
            pass

    def _finalize(self) -> None:
        if self._finalized:
            return
        self._finalized = True
        if isinstance(self._span, _NoSpan):
            return
        try:
            self._span.set_attribute("gen_ai.latency_ms", (time.time() - self._start) * 1000)
            text = "".join(self._parts)
            if text:
                self._span.set_attribute("gen_ai.completion", _truncate(text, 1000))
                self._span.set_attribute("traceloop.entity.output", text)
            if self._span.status_code == StatusCode.UNSET:
                self._span.set_status(StatusCode.OK)
        except Exception:
            pass
        finally:
            self._span.end()
            try:
                provider = TracerProvider.get_instance()
                if provider is not None and provider._buffer is not None:
                    provider._buffer.push(self._span)
            except Exception:
                pass

    def __enter__(self):
        if hasattr(self._stream, "__enter__"):
            self._stream.__enter__()
        return self

    def __exit__(self, *exc):
        try:
            if hasattr(self._stream, "__exit__"):
                self._stream.__exit__(*exc)
        finally:
            self._finalize()
        return False

    async def __aenter__(self):
        if hasattr(self._stream, "__aenter__"):
            await self._stream.__aenter__()
        return self

    async def __aexit__(self, *exc):
        try:
            if hasattr(self._stream, "__aexit__"):
                await self._stream.__aexit__(*exc)
        finally:
            self._finalize()
        return False


def _instrument_anthropic() -> bool:
    """
    Instrument Anthropic library.

    Wraps:
        - Messages.create()       (sync, blocking and streaming)
        - AsyncMessages.create()  (async, blocking and streaming)

    Both resource classes are shared by the first-party client AND the Bedrock/Vertex
    clients — `AnthropicBedrockMantle().messages` IS `anthropic.resources.Messages` — so
    wrapping them here covers every platform in one place. A boto3-based client would not
    be covered, which is why callers should reach Bedrock through the Anthropic SDK.

    Returns:
        True if instrumentation succeeded
    """
    try:
        import anthropic

        if hasattr(anthropic, "_ns_probe_instrumented"):
            return True

        tracer = get_tracer("ns_probe.anthropic")

        def _begin_stream_span(model: str, kwargs: dict) -> Any:
            """A detached span for a streamed call, ended by the proxy.

            A context manager would close before the caller has iterated a single event,
            so the span is opened manually and parented explicitly.
            """
            parent = get_current_span_context()
            span = tracer.start_span_no_context(
                "llm.call",
                kind=SpanKind.CLIENT,
                trace_id=parent.trace_id if parent else None,
                parent_span_id=parent.span_id if parent else None,
            )
            span.set_attribute("gen_ai.system", "anthropic")
            span.set_attribute("gen_ai.request.model", model)
            if "messages" in kwargs:
                span.set_attribute("gen_ai.prompt", _format_messages(kwargs["messages"]))
            return span

        def _record_message(span: Any, result: Any, start_time: float) -> None:
            span.set_attribute("gen_ai.latency_ms", (time.time() - start_time) * 1000)
            usage_attrs = _anthropic_usage_attrs(getattr(result, "usage", None))
            if usage_attrs:
                span.set_attributes(usage_attrs)
            if hasattr(result, "model"):
                span.set_attribute("gen_ai.response.model", result.model)
            if hasattr(result, "stop_reason"):
                span.set_attribute("gen_ai.response.finish_reason", result.stop_reason)

        original_create = anthropic.resources.Messages.create

        @functools.wraps(original_create)
        def traced_create(self: Any, *args: Any, **kwargs: Any) -> Any:
            model = kwargs.get("model", "unknown")

            if kwargs.get("stream"):
                span = _begin_stream_span(model, kwargs)
                start_time = time.time()
                try:
                    result = original_create(self, *args, **kwargs)
                except Exception as e:
                    _fail_detached_span(span, e)
                    raise
                return _AnthropicStreamProxy(result, span, start_time)

            with _llm_span(tracer) as span:
                span.set_attributes(
                    {
                        "gen_ai.system": "anthropic",
                        "gen_ai.request.model": model,
                    }
                )
                if "messages" in kwargs:
                    span.set_attribute("gen_ai.prompt", _format_messages(kwargs["messages"]))

                start_time = time.time()
                result = original_create(self, *args, **kwargs)
                _record_message(span, result, start_time)
                return result

        anthropic.resources.Messages.create = traced_create

        # The async resource is what an async agent actually calls. Wrapping only the sync
        # class meant every async call went unrecorded, and an unrecorded LLM call reads as
        # a free one. Guarded with getattr because older builds may not expose it.
        async_messages = getattr(anthropic.resources, "AsyncMessages", None)
        if async_messages is not None:
            original_acreate = async_messages.create

            @functools.wraps(original_acreate)
            async def traced_acreate(self: Any, *args: Any, **kwargs: Any) -> Any:
                model = kwargs.get("model", "unknown")

                if kwargs.get("stream"):
                    span = _begin_stream_span(model, kwargs)
                    start_time = time.time()
                    try:
                        result = await original_acreate(self, *args, **kwargs)
                    except Exception as e:
                        _fail_detached_span(span, e)
                        raise
                    return _AnthropicStreamProxy(result, span, start_time)

                with _llm_span(tracer) as span:
                    span.set_attributes(
                        {
                            "gen_ai.system": "anthropic",
                            "gen_ai.request.model": model,
                        }
                    )
                    if "messages" in kwargs:
                        span.set_attribute(
                            "gen_ai.prompt", _format_messages(kwargs["messages"])
                        )

                    start_time = time.time()
                    result = await original_acreate(self, *args, **kwargs)
                    _record_message(span, result, start_time)
                    return result

            async_messages.create = traced_acreate

        anthropic._ns_probe_instrumented = True  # type: ignore

        logger.debug("Instrumented Anthropic")
        return True

    except ImportError:
        return False
    except Exception as e:
        logger.debug(f"Failed to instrument Anthropic: {e}")
        return False


def _fail_detached_span(span: Any, exc: Exception) -> None:
    """End a detached (streaming) span that never produced a stream."""
    try:
        span.set_status(StatusCode.ERROR, str(exc))
        span.record_exception(exc)
    except Exception:
        pass
    finally:
        try:
            span.end()
            provider = TracerProvider.get_instance()
            if provider is not None and provider._buffer is not None:
                provider._buffer.push(span)
        except Exception:
            pass


def _instrument_litellm() -> bool:
    """
    Instrument LiteLLM.

    Wraps:
        - litellm.completion() / litellm.acompletion()
        - litellm.embedding()  / litellm.aembedding()

    Why this one matters more than it looks. CrewAI — and a growing number of
    frameworks — do not talk to a provider SDK at all; they hand everything to
    LiteLLM, which picks a provider and calls it. Without this, those calls
    arrive as bare httpx spans carrying no token counts, and the processor
    computes cost FROM tokens, so an entire agent reads as free. A missing span
    is a visible gap; a span with no tokens is a wrong number, which is worse.

    LiteLLM normalises every provider onto the OpenAI response shape, so usage is
    always `prompt_tokens` / `completion_tokens` regardless of who answered.

    The model string is kept exactly as the caller wrote it ("anthropic/claude-…",
    "bedrock/…"), because that is the routing decision worth seeing in a trace.
    The provider is the part before the slash when there is one; LiteLLM defaults
    to OpenAI when there is not.

    Returns:
        True if instrumentation succeeded
    """
    try:
        import litellm

        if hasattr(litellm, "_ns_probe_instrumented"):
            return True

        tracer = get_tracer("ns_probe.litellm")

        def _record_request(span: Any, kwargs: Any) -> None:
            model = kwargs.get("model", "unknown")
            provider = model.split("/", 1)[0] if "/" in str(model) else "openai"
            span.set_attributes(
                {
                    "gen_ai.system": provider,
                    "gen_ai.request.model": model,
                    # Which layer produced this span. Two frameworks reaching the
                    # same provider are worth telling apart when a cost figure
                    # looks wrong.
                    "gen_ai.route": "litellm",
                }
            )
            if kwargs.get("messages"):
                span.set_attribute("gen_ai.prompt", _format_messages(kwargs["messages"]))
                span.set_attribute(
                    "traceloop.entity.input", _format_messages_full(kwargs["messages"])
                )

        def _record_response(span: Any, result: Any, started: float) -> None:
            span.set_attribute("gen_ai.latency_ms", (time.time() - started) * 1000)

            usage = getattr(result, "usage", None)
            if usage is not None:
                # Dict on some providers, object on others — LiteLLM's own
                # Usage type supports both accesses, third-party mocks may not.
                def _u(name: str) -> Any:
                    if isinstance(usage, dict):
                        return usage.get(name)
                    return getattr(usage, name, None)

                pt, ct = _u("prompt_tokens"), _u("completion_tokens")
                if pt is not None:
                    span.set_attribute("gen_ai.usage.input_tokens", pt)
                if ct is not None:
                    span.set_attribute("gen_ai.usage.output_tokens", ct)

            if getattr(result, "model", None):
                span.set_attribute("gen_ai.response.model", result.model)

            choices = getattr(result, "choices", None)
            if choices:
                finish = getattr(choices[0], "finish_reason", None)
                if finish:
                    span.set_attribute("gen_ai.response.finish_reason", finish)
                message = getattr(choices[0], "message", None)
                content = getattr(message, "content", None) if message is not None else None
                if content:
                    span.set_attribute("gen_ai.completion", _truncate(str(content), 1000))
                    span.set_attribute("traceloop.entity.output", str(content))

        def _wrap_sync(original: Any, span_name: str) -> Any:
            @functools.wraps(original)
            def traced(*args: Any, **kwargs: Any) -> Any:
                with _llm_span(tracer) as span:
                    _record_request(span, kwargs)
                    started = time.time()
                    result = original(*args, **kwargs)
                    _record_response(span, result, started)
                    return result

            return traced

        def _wrap_async(original: Any, span_name: str) -> Any:
            @functools.wraps(original)
            async def traced(*args: Any, **kwargs: Any) -> Any:
                with _llm_span(tracer) as span:
                    _record_request(span, kwargs)
                    started = time.time()
                    result = await original(*args, **kwargs)
                    _record_response(span, result, started)
                    return result

            return traced

        wrapped_any = False
        for name, wrapper in (
            ("completion", _wrap_sync),
            ("embedding", _wrap_sync),
            ("acompletion", _wrap_async),
            ("aembedding", _wrap_async),
        ):
            original = getattr(litellm, name, None)
            if original is None:
                continue  # older or trimmed builds do not ship every entry point
            setattr(litellm, name, wrapper(original, name))
            wrapped_any = True

        if not wrapped_any:
            return False

        litellm._ns_probe_instrumented = True  # type: ignore

        logger.debug("Instrumented LiteLLM")
        return True

    except ImportError:
        return False
    except Exception as e:
        logger.debug(f"Failed to instrument LiteLLM: {e}")
        return False


def _instrument_httpx() -> bool:
    """
    Instrument httpx library.

    Wraps:
        - client.request()
        - client.get/post/put/delete/etc.

    Returns:
        True if instrumentation succeeded
    """
    try:
        import httpx

        if hasattr(httpx, "_ns_probe_instrumented"):
            return True

        tracer = get_tracer("ns_probe.httpx")

        original_send = httpx.Client.send

        @functools.wraps(original_send)
        def traced_send(self: Any, request: Any, *args: Any, **kwargs: Any) -> Any:
            if _should_skip_url(str(request.url)):
                return original_send(self, request, *args, **kwargs)

            span_name = f"HTTP {request.method}"

            with tracer.start_span(span_name, kind=SpanKind.CLIENT) as span:
                span.set_attributes(
                    {
                        "http.method": request.method,
                        "http.url": str(request.url),
                        "http.host": request.url.host,
                    }
                )

                start_time = time.time()
                response = original_send(self, request, *args, **kwargs)
                latency_ms = (time.time() - start_time) * 1000

                span.set_attributes(
                    {
                        "http.status_code": response.status_code,
                        "http.latency_ms": latency_ms,
                    }
                )

                return response

        httpx.Client.send = traced_send

        # Also instrument async client
        try:
            original_async_send = httpx.AsyncClient.send

            @functools.wraps(original_async_send)
            async def traced_async_send(self: Any, request: Any, *args: Any, **kwargs: Any) -> Any:
                if _should_skip_url(str(request.url)):
                    return await original_async_send(self, request, *args, **kwargs)

                span_name = f"HTTP {request.method}"

                with tracer.start_span(span_name, kind=SpanKind.CLIENT) as span:
                    span.set_attributes(
                        {
                            "http.method": request.method,
                            "http.url": str(request.url),
                            "http.host": request.url.host,
                        }
                    )

                    start_time = time.time()
                    response = await original_async_send(self, request, *args, **kwargs)
                    latency_ms = (time.time() - start_time) * 1000

                    span.set_attributes(
                        {
                            "http.status_code": response.status_code,
                            "http.latency_ms": latency_ms,
                        }
                    )

                    return response

            httpx.AsyncClient.send = traced_async_send
        except AttributeError:
            pass

        httpx._ns_probe_instrumented = True  # type: ignore
        logger.debug("Instrumented httpx")
        return True

    except ImportError:
        return False
    except Exception as e:
        logger.debug(f"Failed to instrument httpx: {e}")
        return False


def _instrument_requests() -> bool:
    """
    Instrument requests library.

    Wraps:
        - requests.Session.request()

    Returns:
        True if instrumentation succeeded
    """
    try:
        import requests

        if hasattr(requests, "_ns_probe_instrumented"):
            return True

        tracer = get_tracer("ns_probe.requests")

        original_request = requests.Session.request

        @functools.wraps(original_request)
        def traced_request(self: Any, method: str, url: str, *args: Any, **kwargs: Any) -> Any:
            if _should_skip_url(url):
                return original_request(self, method, url, *args, **kwargs)

            span_name = f"HTTP {method}"

            with tracer.start_span(span_name, kind=SpanKind.CLIENT) as span:
                span.set_attributes(
                    {
                        "http.method": method,
                        "http.url": url,
                    }
                )

                start_time = time.time()
                response = original_request(self, method, url, *args, **kwargs)
                latency_ms = (time.time() - start_time) * 1000

                span.set_attributes(
                    {
                        "http.status_code": response.status_code,
                        "http.latency_ms": latency_ms,
                    }
                )

                return response

        requests.Session.request = traced_request
        requests._ns_probe_instrumented = True  # type: ignore

        logger.debug("Instrumented requests")
        return True

    except ImportError:
        return False
    except Exception as e:
        logger.debug(f"Failed to instrument requests: {e}")
        return False


def _format_messages(messages: list) -> str:
    """Format chat messages for span attribute."""
    try:
        formatted = []
        for msg in messages[:5]:  # Limit to first 5 messages
            if isinstance(msg, dict):
                role = msg.get("role", "unknown")
                content = msg.get("content", "")
            else:
                role = getattr(msg, "role", "unknown")
                content = getattr(msg, "content", "")

            formatted.append(f"[{role}]: {_truncate(str(content), 200)}")

        if len(messages) > 5:
            formatted.append(f"... and {len(messages) - 5} more messages")

        return "\n".join(formatted)
    except Exception:
        return str(messages)[:500]


def _truncate(s: str, max_len: int) -> str:
    """Truncate string to max length."""
    if len(s) <= max_len:
        return s
    return s[: max_len - 3] + "..."


def _format_messages_full(messages: list) -> str:
    """
    Format chat messages for a span attribute WITHOUT truncation.

    Used for `traceloop.entity.input` so the trace explorer shows the complete
    prompt (every message, full content). `_format_messages` remains the
    truncated variant used for the compact `gen_ai.prompt` attribute.
    """
    try:
        formatted = []
        for msg in messages:
            if isinstance(msg, dict):
                role = msg.get("role", "unknown")
                content = msg.get("content", "")
            else:
                role = getattr(msg, "role", "unknown")
                content = getattr(msg, "content", "")
            formatted.append(f"[{role}]: {content}")
        return "\n".join(formatted)
    except Exception:
        return str(messages)


def _instrument_asyncpg() -> bool:
    """
    Instrument asyncpg library for PostgreSQL.

    Wraps:
        - asyncpg.connect()
        - asyncpg.create_pool()
        - Connection.execute(), fetch(), fetchrow(), fetchval()

    Returns:
        True if instrumentation succeeded
    """
    try:
        import asyncpg

        if hasattr(asyncpg, "_ns_probe_instrumented"):
            return True

        tracer = get_tracer("ns_probe.asyncpg")
        from .span import StatusCode

        # Instrument connect()
        original_connect = asyncpg.connect

        @functools.wraps(original_connect)
        async def traced_connect(*args: Any, **kwargs: Any) -> Any:
            # Extract connection info for span attributes
            host = kwargs.get("host", args[0] if args else "localhost")
            port = kwargs.get("port", 5432)
            database = kwargs.get("database", kwargs.get("dbname", "unknown"))
            user = kwargs.get("user", "unknown")

            with tracer.start_span("db.connect", kind=SpanKind.CLIENT) as span:
                span.set_attributes(
                    {
                        "db.system": "postgresql",
                        "db.name": str(database),
                        "db.user": str(user),
                        "net.peer.name": str(host),
                        "net.peer.port": port,
                        "db.operation": "connect",
                    }
                )

                start_time = time.time()
                try:
                    result = await original_connect(*args, **kwargs)
                    latency_ms = (time.time() - start_time) * 1000
                    span.set_attribute("db.latency_ms", latency_ms)
                    span.set_attribute("db.connection.status", "success")
                    return result
                except Exception as e:
                    latency_ms = (time.time() - start_time) * 1000
                    span.set_attribute("db.latency_ms", latency_ms)
                    span.set_attribute("db.connection.status", "error")
                    span.set_attribute("db.error.message", str(e))
                    span.set_attribute("db.error.type", type(e).__name__)
                    span.set_status(StatusCode.ERROR, str(e))
                    raise

        asyncpg.connect = traced_connect

        # Instrument create_pool()
        original_create_pool = asyncpg.create_pool

        @functools.wraps(original_create_pool)
        async def traced_create_pool(*args: Any, **kwargs: Any) -> Any:
            host = kwargs.get("host", args[0] if args else "localhost")
            database = kwargs.get("database", kwargs.get("dsn", "unknown"))

            with tracer.start_span("db.pool.create", kind=SpanKind.CLIENT) as span:
                span.set_attributes(
                    {
                        "db.system": "postgresql",
                        "db.name": str(database),
                        "net.peer.name": str(host),
                        "db.operation": "create_pool",
                    }
                )

                start_time = time.time()
                try:
                    result = await original_create_pool(*args, **kwargs)
                    latency_ms = (time.time() - start_time) * 1000
                    span.set_attribute("db.latency_ms", latency_ms)
                    span.set_attribute("db.pool.status", "success")
                    return result
                except Exception as e:
                    latency_ms = (time.time() - start_time) * 1000
                    span.set_attribute("db.latency_ms", latency_ms)
                    span.set_attribute("db.pool.status", "error")
                    span.set_attribute("db.error.message", str(e))
                    span.set_attribute("db.error.type", type(e).__name__)
                    span.set_status(StatusCode.ERROR, str(e))
                    raise

        asyncpg.create_pool = traced_create_pool

        # Instrument Connection methods
        try:
            original_execute = asyncpg.Connection.execute
            original_fetch = asyncpg.Connection.fetch
            original_fetchrow = asyncpg.Connection.fetchrow
            original_fetchval = asyncpg.Connection.fetchval

            @functools.wraps(original_execute)
            async def traced_execute(self: Any, query: str, *args: Any, **kwargs: Any) -> Any:
                with tracer.start_span("db.execute", kind=SpanKind.CLIENT) as span:
                    span.set_attributes(
                        {
                            "db.system": "postgresql",
                            "db.statement": _truncate(query, 500),
                            "db.operation": "execute",
                        }
                    )
                    start_time = time.time()
                    try:
                        result = await original_execute(self, query, *args, **kwargs)
                        span.set_attribute("db.latency_ms", (time.time() - start_time) * 1000)
                        return result
                    except Exception as e:
                        span.set_attribute("db.error.message", str(e))
                        span.set_status(StatusCode.ERROR, str(e))
                        raise

            @functools.wraps(original_fetch)
            async def traced_fetch(self: Any, query: str, *args: Any, **kwargs: Any) -> Any:
                with tracer.start_span("db.fetch", kind=SpanKind.CLIENT) as span:
                    span.set_attributes(
                        {
                            "db.system": "postgresql",
                            "db.statement": _truncate(query, 500),
                            "db.operation": "fetch",
                        }
                    )
                    start_time = time.time()
                    try:
                        result = await original_fetch(self, query, *args, **kwargs)
                        span.set_attribute("db.latency_ms", (time.time() - start_time) * 1000)
                        span.set_attribute("db.row_count", len(result) if result else 0)
                        return result
                    except Exception as e:
                        span.set_attribute("db.error.message", str(e))
                        span.set_status(StatusCode.ERROR, str(e))
                        raise

            @functools.wraps(original_fetchrow)
            async def traced_fetchrow(self: Any, query: str, *args: Any, **kwargs: Any) -> Any:
                with tracer.start_span("db.fetchrow", kind=SpanKind.CLIENT) as span:
                    span.set_attributes(
                        {
                            "db.system": "postgresql",
                            "db.statement": _truncate(query, 500),
                            "db.operation": "fetchrow",
                        }
                    )
                    start_time = time.time()
                    try:
                        result = await original_fetchrow(self, query, *args, **kwargs)
                        span.set_attribute("db.latency_ms", (time.time() - start_time) * 1000)
                        return result
                    except Exception as e:
                        span.set_attribute("db.error.message", str(e))
                        span.set_status(StatusCode.ERROR, str(e))
                        raise

            @functools.wraps(original_fetchval)
            async def traced_fetchval(self: Any, query: str, *args: Any, **kwargs: Any) -> Any:
                with tracer.start_span("db.fetchval", kind=SpanKind.CLIENT) as span:
                    span.set_attributes(
                        {
                            "db.system": "postgresql",
                            "db.statement": _truncate(query, 500),
                            "db.operation": "fetchval",
                        }
                    )
                    start_time = time.time()
                    try:
                        result = await original_fetchval(self, query, *args, **kwargs)
                        span.set_attribute("db.latency_ms", (time.time() - start_time) * 1000)
                        return result
                    except Exception as e:
                        span.set_attribute("db.error.message", str(e))
                        span.set_status(StatusCode.ERROR, str(e))
                        raise

            asyncpg.Connection.execute = traced_execute
            asyncpg.Connection.fetch = traced_fetch
            asyncpg.Connection.fetchrow = traced_fetchrow
            asyncpg.Connection.fetchval = traced_fetchval

        except AttributeError:
            pass  # Connection class not available

        asyncpg._ns_probe_instrumented = True  # type: ignore
        logger.debug("Instrumented asyncpg")
        return True

    except ImportError:
        return False
    except Exception as e:
        logger.debug(f"Failed to instrument asyncpg: {e}")
        return False


def _instrument_psycopg2() -> bool:
    """
    Instrument psycopg2 library for PostgreSQL.

    Wraps:
        - psycopg2.connect()
        - Cursor.execute(), executemany(), fetchone(), fetchall(), fetchmany()

    Returns:
        True if instrumentation succeeded
    """
    try:
        import psycopg2

        if hasattr(psycopg2, "_ns_probe_instrumented"):
            return True

        tracer = get_tracer("ns_probe.psycopg2")
        from .span import StatusCode

        # Instrument connect()
        original_connect = psycopg2.connect

        @functools.wraps(original_connect)
        def traced_connect(*args: Any, **kwargs: Any) -> Any:
            # Extract connection info
            host = kwargs.get("host", "localhost")
            port = kwargs.get("port", 5432)
            database = kwargs.get("database", kwargs.get("dbname", "unknown"))
            user = kwargs.get("user", "unknown")

            # Also check DSN string
            if args and isinstance(args[0], str):
                dsn = args[0]
                database = dsn  # Use DSN as database identifier

            with tracer.start_span("db.connect", kind=SpanKind.CLIENT) as span:
                span.set_attributes(
                    {
                        "db.system": "postgresql",
                        "db.name": str(database),
                        "db.user": str(user),
                        "net.peer.name": str(host),
                        "net.peer.port": port,
                        "db.operation": "connect",
                    }
                )

                start_time = time.time()
                try:
                    result = original_connect(*args, **kwargs)
                    latency_ms = (time.time() - start_time) * 1000
                    span.set_attribute("db.latency_ms", latency_ms)
                    span.set_attribute("db.connection.status", "success")
                    return result
                except Exception as e:
                    latency_ms = (time.time() - start_time) * 1000
                    span.set_attribute("db.latency_ms", latency_ms)
                    span.set_attribute("db.connection.status", "error")
                    span.set_attribute("db.error.message", str(e))
                    span.set_attribute("db.error.type", type(e).__name__)
                    span.set_status(StatusCode.ERROR, str(e))
                    raise

        psycopg2.connect = traced_connect

        # Instrument Cursor methods
        try:
            from psycopg2.extensions import cursor as CursorClass

            original_execute = CursorClass.execute
            original_executemany = CursorClass.executemany

            @functools.wraps(original_execute)
            def traced_cursor_execute(self: Any, query: Any, vars: Any = None) -> Any:
                query_str = str(query) if query else ""

                with tracer.start_span("db.execute", kind=SpanKind.CLIENT) as span:
                    span.set_attributes(
                        {
                            "db.system": "postgresql",
                            "db.statement": _truncate(query_str, 500),
                            "db.operation": "execute",
                        }
                    )
                    start_time = time.time()
                    try:
                        result = original_execute(self, query, vars)
                        span.set_attribute("db.latency_ms", (time.time() - start_time) * 1000)
                        if hasattr(self, "rowcount") and self.rowcount >= 0:
                            span.set_attribute("db.row_count", self.rowcount)
                        return result
                    except Exception as e:
                        span.set_attribute("db.error.message", str(e))
                        span.set_status(StatusCode.ERROR, str(e))
                        raise

            @functools.wraps(original_executemany)
            def traced_cursor_executemany(self: Any, query: Any, vars_list: Any) -> Any:
                query_str = str(query) if query else ""

                with tracer.start_span("db.executemany", kind=SpanKind.CLIENT) as span:
                    span.set_attributes(
                        {
                            "db.system": "postgresql",
                            "db.statement": _truncate(query_str, 500),
                            "db.operation": "executemany",
                            "db.batch_size": len(vars_list) if vars_list else 0,
                        }
                    )
                    start_time = time.time()
                    try:
                        result = original_executemany(self, query, vars_list)
                        span.set_attribute("db.latency_ms", (time.time() - start_time) * 1000)
                        return result
                    except Exception as e:
                        span.set_attribute("db.error.message", str(e))
                        span.set_status(StatusCode.ERROR, str(e))
                        raise

            CursorClass.execute = traced_cursor_execute
            CursorClass.executemany = traced_cursor_executemany

        except (ImportError, AttributeError):
            pass  # Cursor class not available

        psycopg2._ns_probe_instrumented = True  # type: ignore
        logger.debug("Instrumented psycopg2")
        return True

    except ImportError:
        return False
    except Exception as e:
        logger.debug(f"Failed to instrument psycopg2: {e}")
        return False
