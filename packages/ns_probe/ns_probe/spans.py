"""
Span ergonomics for ns_probe.

`observe()` covers the common case — one span around one function — but real
agents need more than that, and until now every agent hand-rolled the rest:

* spans that **open in one method and close in another** (a call spans a whole
  session; a conversation turn spans speak -> listen -> reply), which cannot use
  a `with` block at all;
* spans for work you only learn about **after it finished** (a provider reports
  latency once the operation is over), which need explicit start/end timestamps
  or they collapse to the instant you were notified;
* **point-in-time events** (an interruption, an escalation) that would be
  misleading as spans because they have no duration;
* recording a span's **input and output**, which is what makes a trace readable
  — a span with only a duration cannot be debugged from a UI;
* making a span the **current parent** for work started elsewhere, e.g. so a
  framework's own root span nests underneath yours.

Everything here is a no-op when tracing is disabled and fail-open on error:
telemetry must never break the agent it observes.

    from ns_probe.spans import traced, set_span_io

    with traced("Fetch the customer record", inputs={"id": cid}) as span:
        record = await fetch(cid)
        set_span_io(span, outputs={"found": record is not None})
"""

from __future__ import annotations

import contextlib
import json
from typing import Any, Dict, Iterator, Optional

from .span import Span, SpanKind, StatusCode, SpanContext

__all__ = [
    "start_span",
    "end_span",
    "traced",
    "record_span",
    "add_event",
    "set_span_io",
    "set_span_attributes",
    "set_current_span_attributes",
    "span_scope",
    "rename_current_span",
]

# Values are truncated before export so one runaway payload cannot blow up a span.
_MAX_IO_CHARS = 8000


def _enabled() -> bool:
    try:
        from .config import get_config

        return bool(get_config().enabled)
    except Exception:
        return False


def _clean(attributes: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Drop None values — an absent attribute is clearer than a null one."""
    return {k: v for k, v in (attributes or {}).items() if v is not None}


def start_span(
    name: str,
    attributes: Optional[Dict[str, Any]] = None,
    parent: Optional[Span] = None,
    kind: SpanKind = SpanKind.INTERNAL,
) -> Optional[Span]:
    """Open a span to be closed later by :func:`end_span`.

    Use when the span cannot be a `with` block because it ends in another
    method. `parent` is another span from this function; when omitted the span
    parents to whatever is currently in scope, or becomes a root.

    Returns None when tracing is disabled — every helper here accepts None, so
    callers never need to branch on it.
    """
    if not _enabled():
        return None
    try:
        from .tracer import get_tracer, get_current_span_context

        if parent is not None:
            trace_id, parent_span_id = parent.trace_id, parent.span_id
        else:
            ctx = get_current_span_context()
            trace_id, parent_span_id = (ctx.trace_id, ctx.span_id) if ctx else (None, None)

        return get_tracer("ns_probe.spans").start_span_no_context(
            name,
            kind=kind,
            attributes=_clean(attributes),
            trace_id=trace_id,
            parent_span_id=parent_span_id,
        )
    except Exception:
        return None


def end_span(
    span: Optional[Span],
    attributes: Optional[Dict[str, Any]] = None,
    error: Optional[BaseException] = None,
) -> None:
    """Close a span from :func:`start_span` and hand it to the exporter."""
    if span is None:
        return
    try:
        from .tracer import TracerProvider

        for key, value in _clean(attributes).items():
            span.set_attribute(key, value)
        if error is not None:
            span.set_status(StatusCode.ERROR, str(error))
            span.record_exception(error)
        else:
            span.set_status(StatusCode.OK)
        span.end()
        provider = TracerProvider.get_instance()
        if provider is not None and provider._buffer is not None:
            provider._buffer.push(span)
    except Exception:
        pass


@contextlib.contextmanager
def traced(
    name: str,
    parent: Optional[Span] = None,
    inputs: Any = None,
    attributes: Optional[Dict[str, Any]] = None,
    kind: SpanKind = SpanKind.INTERNAL,
) -> Iterator[Optional[Span]]:
    """Trace a block of work, yielding the span so outputs can be added.

    Records an escaping exception on the span and re-raises it. Works across
    `await` — the span simply spans the block.
    """
    span = start_span(name, attributes=attributes, parent=parent, kind=kind)
    set_span_io(span, inputs=inputs)
    error: Optional[BaseException] = None
    try:
        yield span
    except BaseException as exc:  # noqa: BLE001 - recorded, then re-raised
        error = exc
        raise
    finally:
        end_span(span, error=error)


def record_span(
    name: str,
    parent: Optional[Span] = None,
    inputs: Any = None,
    outputs: Any = None,
    attributes: Optional[Dict[str, Any]] = None,
    start_time_ns: Optional[int] = None,
    end_time_ns: Optional[int] = None,
    kind: SpanKind = SpanKind.INTERNAL,
) -> None:
    """Record an already-finished operation as a span, in one call.

    For work reported after the fact — a provider that hands you STT or TTS
    timings once the operation is over. Pass the real start/end so the span
    lands where it belongs in the waterfall instead of collapsing to the moment
    you were notified.
    """
    span = start_span(name, attributes=attributes, parent=parent, kind=kind)
    if span is None:
        return
    if start_time_ns is not None:
        try:
            span.start_time_ns = start_time_ns
        except Exception:
            pass
    set_span_io(span, inputs=inputs, outputs=outputs)
    try:
        from .tracer import TracerProvider

        span.set_status(StatusCode.OK)
        span.end(end_time_ns=end_time_ns)
        provider = TracerProvider.get_instance()
        if provider is not None and provider._buffer is not None:
            provider._buffer.push(span)
    except Exception:
        pass


def add_event(span: Optional[Span], name: str, attributes: Optional[Dict[str, Any]] = None) -> None:
    """Mark something that *happened* on a span, rather than took time.

    Event attributes ARE covered by PHI sanitization — `_span_to_otlp` runs them
    through the sanitizer alongside span and link attributes. The warning that
    used to be here said the opposite, which told agent authors a protected
    surface was unprotected.

    The invariant worth stating instead: any NEW attribute-bearing surface added
    to `Span` must be routed through `sanitize_attributes` in `_span_to_otlp`, or
    it exports raw.
    """
    if span is None:
        return
    try:
        span.add_event(name, attributes=_clean(attributes))
    except Exception:
        pass


def set_span_io(span: Optional[Span], inputs: Any = None, outputs: Any = None) -> None:
    """Record what a span received and produced.

    Written as `traceloop.entity.input` / `.output`, the same keys
    ``observe(capture_args=True, capture_result=True)`` produces, so manually
    created spans render identically to decorated functions.
    """
    if span is None:
        return
    try:
        if inputs is not None:
            span.set_attribute(
                "traceloop.entity.input",
                json.dumps({"inputs": inputs}, default=str)[:_MAX_IO_CHARS],
            )
        if outputs is not None:
            span.set_attribute(
                "traceloop.entity.output",
                json.dumps({"outputs": outputs}, default=str)[:_MAX_IO_CHARS],
            )
    except Exception:
        pass


def set_span_attributes(span: Optional[Span], **attributes: Any) -> None:
    """Attach attributes to a specific span."""
    if span is None:
        return
    try:
        for key, value in attributes.items():
            if value is not None:
                span.set_attribute(key, value)
    except Exception:
        pass


def set_current_span_attributes(**attributes: Any) -> None:
    """Attach attributes to whichever span is currently in scope.

    Useful inside a function wrapped by ``observe()`` when the values only exist
    at runtime — e.g. the ``gen_ai.*`` conventions read off a provider response.
    """
    if not _enabled():
        return
    try:
        from .tracer import current_span

        set_span_attributes(current_span(), **attributes)
    except Exception:
        pass


@contextlib.contextmanager
def span_scope(span: Optional[Span]) -> Iterator[None]:
    """Make `span` the current parent for spans created inside the block.

    Needed when a framework starts its own root span and you want it nested
    under yours — the LangGraph instrumentor, for instance, parents itself off
    the current span context. `asyncio.create_task` copies the context at
    creation, so entering this scope before starting a task covers the task too.
    """
    if span is None:
        yield
        return
    token = None
    try:
        from .tracer import _current_span_context

        token = _current_span_context.set(SpanContext(trace_id=span.trace_id, span_id=span.span_id))
    except Exception:
        token = None
    try:
        yield
    finally:
        if token is not None:
            try:
                from .tracer import _current_span_context

                _current_span_context.reset(token)
            except Exception:
                pass


def rename_current_span(name: str) -> None:
    """Rename the span currently in scope.

    ``observe(name=...)`` fixes the name at decoration time, so every call
    through a shared choke point — a single LLM helper, say — produces
    identically named spans that cannot be told apart in a waterfall. Renaming
    at call time restores the distinction.
    """
    if not _enabled():
        return
    try:
        from .tracer import current_span

        span = current_span()
        if span is not None:
            span.name = name
    except Exception:
        pass
