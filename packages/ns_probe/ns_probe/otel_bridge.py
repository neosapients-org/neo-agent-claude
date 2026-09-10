"""Let OpenTelemetry-shaped instrumentors write into ns_probe's trace.

The problem this solves
-----------------------
ns_probe implements its own span engine rather than using the OpenTelemetry SDK —
deliberately, for weight and for control over blocking. The cost of that decision
is that the entire ecosystem of ready-made instrumentors (OpenLLMetry's LlamaIndex,
CrewAI, Haystack, Bedrock, Vertex, Cohere; OTel-contrib's FastAPI, Celery, Kafka)
cannot be used: they emit through `opentelemetry.trace`, whose global provider is
either absent — in which case every span they produce is silently discarded into
the API's no-op implementation — or a real OTel SDK provider, in which case their
spans go somewhere else entirely and orphan instead of parenting under our root.
That is exactly why `_try_openllmetry()` in instrumentors.py is switched off.

This module is the third option: a TracerProvider that satisfies the OTel *API*
but is backed by ns_probe's engine. Register it and those instrumentors write into
our tree, inherit `turn()`'s ambient identity, and leave through our ring buffer,
with no OTel SDK anywhere in the process.

Why it is opt-in
----------------
`install_otel_bridge()` claims a global that an application may already own. A
service that runs its own OTel SDK for its HTTP layer would find its spans
redirected here, which is a decision nobody should make on someone's behalf as a
side effect of importing a library. So `instrument_all()` does not call this —
you ask for it, by calling it or by setting NS_PROBE_OTEL_BRIDGE=true.

What it does not do
-------------------
It does not implement the OTel *SDK*: no samplers, no span processors, no
exporters, no `Resource`. Those are ns_probe's own concern and configuring them
here would create two answers to the same question. It satisfies the API surface
that instrumentors actually call, which is small: two methods on Tracer and ten
on Span.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, Optional

from .span import Span as NsSpan
from .span import SpanKind as NsSpanKind
from .span import StatusCode as NsStatusCode
from .tracer import TracerProvider as NsTracerProvider
from .tracer import get_current_span_context, get_tracer

logger = logging.getLogger("ns_probe.otel_bridge")

BRIDGE_ENV = "NS_PROBE_OTEL_BRIDGE"

# OTel's SpanKind is an IntEnum: INTERNAL=0, SERVER=1, CLIENT=2, PRODUCER=3,
# CONSUMER=4. ns_probe's happens to use the same ordering, but relying on that
# silently would make a reordering in either project mislabel every span, so the
# mapping is written out.
_KIND_FROM_OTEL = {
    0: NsSpanKind.INTERNAL,
    1: NsSpanKind.SERVER,
    2: NsSpanKind.CLIENT,
    3: NsSpanKind.PRODUCER,
    4: NsSpanKind.CONSUMER,
}


def _ns_kind(kind: Any) -> Any:
    """Map an OTel SpanKind onto ours.

    `int(kind)` looks right and is wrong: OTel's SpanKind is a plain `Enum`, not
    an `IntEnum`, so int() raises TypeError on it. Every real instrumentor passes
    the enum, which is how this got past a test suite that only ever passed None
    and bare ints — and the failure is a TypeError thrown out of the caller's
    HTTP request, not a dropped span.
    """
    if kind is None:
        return NsSpanKind.INTERNAL
    raw = getattr(kind, "value", kind)
    try:
        return _KIND_FROM_OTEL.get(int(raw), NsSpanKind.INTERNAL)
    except (TypeError, ValueError):
        return NsSpanKind.INTERNAL


def _hex_to_int(value: Optional[str], width: int) -> int:
    """OTel identifies spans by int, ns_probe by hex string."""
    try:
        return int(value, 16) if value else 0
    except (TypeError, ValueError):
        return 0


def install_otel_bridge(force: bool = False) -> bool:
    """Point the global OpenTelemetry tracer provider at ns_probe.

    Args:
        force: Replace a provider that is already set. OTel itself refuses the
            second `set_tracer_provider` call and only logs about it, so without
            this an app that configured OTel first would keep its own provider
            and the bridge would appear to succeed while doing nothing.

    Returns:
        True if the bridge is installed. False if the OpenTelemetry API is not
        installed, or a provider was already set and *force* was not given.
    """
    try:
        from opentelemetry import trace as otel_trace
    except ImportError:
        logger.debug("OpenTelemetry API not installed; bridge unavailable")
        return False

    try:
        existing = otel_trace.get_tracer_provider()
        already_real = not isinstance(
            existing, (otel_trace.NoOpTracerProvider, otel_trace.ProxyTracerProvider)
        )
        if already_real and not force:
            logger.warning(
                "An OpenTelemetry TracerProvider is already installed (%s). "
                "ns_probe left it alone — pass force=True to take it over, but "
                "note that redirects that application's spans here too.",
                type(existing).__name__,
            )
            return False

        provider = _build_provider(otel_trace)
        if force:
            # set_tracer_provider refuses to replace, and only logs about it.
            otel_trace._TRACER_PROVIDER = provider  # type: ignore[attr-defined]
        else:
            otel_trace.set_tracer_provider(provider)

        logger.info("OpenTelemetry bridge installed; OTel spans now flow into ns_probe")
        return True

    except Exception:  # pragma: no cover — observability is best-effort
        logger.debug("Could not install the OpenTelemetry bridge", exc_info=True)
        return False


def install_from_env() -> bool:
    """Install the bridge if NS_PROBE_OTEL_BRIDGE says so."""
    if os.getenv(BRIDGE_ENV, "").strip().lower() not in ("1", "true", "yes", "on"):
        return False
    return install_otel_bridge()


def _build_provider(otel_trace: Any) -> Any:
    """Built inside a function so importing this module never requires OTel."""

    class _BridgeSpan(otel_trace.Span):
        """An OTel Span whose recording lands on an ns_probe span."""

        def __init__(self, ns_span: NsSpan, on_end: Any = None) -> None:
            self._ns = ns_span
            self._on_end = on_end
            self._ended = False

        # -- recording ---------------------------------------------------
        def set_attribute(self, key: str, value: Any) -> None:
            self._ns.set_attribute(key, value)

        def set_attributes(self, attributes: Dict[str, Any]) -> None:
            for k, v in (attributes or {}).items():
                self._ns.set_attribute(k, v)

        def add_event(
            self, name: str, attributes: Any = None, timestamp: Optional[int] = None
        ) -> None:
            self._ns.add_event(name, dict(attributes or {}))

        def add_link(self, context: Any, attributes: Any = None) -> None:
            # Links have no representation in our span shape. Recorded as an
            # event so the relationship is at least visible, rather than dropped.
            try:
                self._ns.add_event(
                    "link",
                    {"linked.trace_id": format(context.trace_id, "032x")},
                )
            except Exception:
                pass

        def set_status(self, status: Any, description: Optional[str] = None) -> None:
            code = getattr(status, "status_code", status)
            name = getattr(code, "name", str(code)).upper()
            self._ns.set_status(
                NsStatusCode.ERROR if name == "ERROR" else NsStatusCode.OK,
                description or getattr(status, "description", None),
            )

        def record_exception(
            self,
            exception: BaseException,
            attributes: Any = None,
            timestamp: Optional[int] = None,
            escaped: bool = False,
        ) -> None:
            self._ns.record_exception(exception)

        def update_name(self, name: str) -> None:
            self._ns.name = name

        # -- lifecycle ---------------------------------------------------
        def end(self, end_time: Optional[int] = None) -> None:
            if self._ended:
                return
            self._ended = True
            self._ns.end(end_time)
            if self._on_end is not None:
                self._on_end(self._ns)

        def is_recording(self) -> bool:
            return not self._ended

        def get_span_context(self) -> Any:
            return otel_trace.SpanContext(
                trace_id=_hex_to_int(self._ns.trace_id, 32),
                span_id=_hex_to_int(self._ns.span_id, 16),
                is_remote=False,
                trace_flags=otel_trace.TraceFlags(otel_trace.TraceFlags.SAMPLED),
            )

        # Instrumentors routinely use the span as a context manager.
        def __enter__(self) -> "_BridgeSpan":
            return self

        def __exit__(self, exc_type, exc, tb) -> None:
            if exc is not None:
                self.record_exception(exc)
                self._ns.set_status(NsStatusCode.ERROR, str(exc))
            self.end()

    class _BridgeTracer(otel_trace.Tracer):
        def __init__(self, name: str) -> None:
            self._name = name
            self._ns_tracer = get_tracer(name)

        def _push(self, ns_span: NsSpan) -> None:
            """A span made outside our context manager still has to be exported."""
            try:
                provider = NsTracerProvider.get_instance()
                if provider is not None and provider._buffer is not None:
                    provider._buffer.push(ns_span)
            except Exception:  # pragma: no cover
                pass

        def start_span(
            self,
            name: str,
            context: Any = None,
            kind: Any = None,
            attributes: Any = None,
            links: Any = None,
            start_time: Optional[int] = None,
            record_exception: bool = True,
            set_status_on_exception: bool = True,
        ) -> Any:
            """A span that is NOT made current — OTel's semantics, not ours."""
            ns_kind = _ns_kind(kind)
            parent = get_current_span_context()
            ns_span = self._ns_tracer.start_span_no_context(
                name=name,
                kind=ns_kind,
                attributes=dict(attributes or {}),
                trace_id=parent.trace_id if parent else None,
                parent_span_id=parent.span_id if parent else None,
            )
            return _BridgeSpan(ns_span, on_end=self._push)

        def start_as_current_span(
            self,
            name: str,
            context: Any = None,
            kind: Any = None,
            attributes: Any = None,
            links: Any = None,
            start_time: Optional[int] = None,
            record_exception: bool = True,
            set_status_on_exception: bool = True,
            end_on_exit: bool = True,
        ) -> Any:
            """The one instrumentors actually use, and the one that has to get
            parenting right: it delegates to ns_probe's own context manager, so
            the span joins our stack and anything opened underneath it — by us or
            by another OTel instrumentor — parents correctly."""
            import contextlib

            ns_kind = _ns_kind(kind)
            tracer = self._ns_tracer

            @contextlib.contextmanager
            def _cm():
                with tracer.start_span(
                    name, kind=ns_kind, attributes=dict(attributes or {})
                ) as ns_span:
                    bridged = _BridgeSpan(ns_span)
                    try:
                        yield bridged
                    except Exception as exc:
                        if record_exception:
                            bridged.record_exception(exc)
                        if set_status_on_exception:
                            ns_span.set_status(NsStatusCode.ERROR, str(exc))
                        raise
                    finally:
                        # ns_probe's context manager ends and exports the span;
                        # marking it stops _BridgeSpan.end() double-ending it.
                        bridged._ended = True

            return _cm()

    class _BridgeTracerProvider(otel_trace.TracerProvider):
        def get_tracer(
            self,
            instrumenting_module_name: str,
            instrumenting_library_version: Optional[str] = None,
            schema_url: Optional[str] = None,
            attributes: Any = None,
        ) -> Any:
            return _BridgeTracer(instrumenting_module_name or "ns_probe.otel")

    return _BridgeTracerProvider()


# Third-party instrumentors reachable once the bridge is installed. Each entry is
# (import path, class name, what it covers). The list is not exhaustive and is not
# meant to be — it is the set worth trying automatically, chosen because each
# closes a gap ns_probe has no native instrumentor for.
#
# Deliberately absent: OpenAI, Anthropic, httpx, requests, asyncpg and psycopg2.
# ns_probe instruments those itself, and enabling both would produce two spans
# per call — the same double-count `_llm_span` exists to prevent, except across
# a boundary that guard cannot see.
_KNOWN_INSTRUMENTORS = (
    ("opentelemetry.instrumentation.langchain", "LangchainInstrumentor", "LangChain (LCEL)"),
    ("opentelemetry.instrumentation.llamaindex", "LlamaIndexInstrumentor", "LlamaIndex"),
    ("opentelemetry.instrumentation.crewai", "CrewAIInstrumentor", "CrewAI"),
    ("opentelemetry.instrumentation.haystack", "HaystackInstrumentor", "Haystack"),
    ("opentelemetry.instrumentation.bedrock", "BedrockInstrumentor", "AWS Bedrock"),
    ("opentelemetry.instrumentation.vertexai", "VertexAIInstrumentor", "Google Vertex AI"),
    (
        "opentelemetry.instrumentation.google_generativeai",
        "GoogleGenerativeAiInstrumentor",
        "Gemini",
    ),
    ("opentelemetry.instrumentation.cohere", "CohereInstrumentor", "Cohere"),
    ("opentelemetry.instrumentation.together", "TogetherAiInstrumentor", "Together"),
    ("opentelemetry.instrumentation.groq", "GroqInstrumentor", "Groq"),
    ("opentelemetry.instrumentation.ollama", "OllamaInstrumentor", "Ollama"),
    ("opentelemetry.instrumentation.replicate", "ReplicateInstrumentor", "Replicate"),
)


def instrument_via_otel(force: bool = False) -> list:
    """Install the bridge, then switch on every known instrumentor that is present.

    This is how LangChain outside LangGraph, LlamaIndex, CrewAI's own crew/task
    structure, Bedrock, Vertex and Gemini become visible — none of which ns_probe
    instruments natively, and each of which would otherwise arrive as a bare HTTP
    span with no tokens and therefore no cost.

    Nothing is installed for you: an instrumentor that is not already a dependency
    of the agent is skipped. Add the ones you want, e.g.
    ``pip install opentelemetry-instrumentation-bedrock``.

    Args:
        force: Passed to :func:`install_otel_bridge`.

    Returns:
        The human-readable names of what was switched on. Empty if the bridge
        could not be installed or nothing was available.
    """
    if not install_otel_bridge(force=force):
        return []

    enabled = []
    for module_path, class_name, label in _KNOWN_INSTRUMENTORS:
        try:
            module = __import__(module_path, fromlist=[class_name])
            instrumentor = getattr(module, class_name)()
            if getattr(instrumentor, "is_instrumented_by_opentelemetry", False):
                continue
            instrumentor.instrument()
            enabled.append(label)
        except ImportError:
            continue
        except Exception:  # pragma: no cover — one bad instrumentor must not stop the rest
            logger.debug("Could not enable %s", label, exc_info=True)

    if enabled:
        logger.info("Instrumented via the OTel bridge: %s", ", ".join(enabled))
    return enabled
