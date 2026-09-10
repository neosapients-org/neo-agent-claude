"""
W3C Trace Context Propagation for ns_probe.

This module implements the W3C Trace Context specification for distributed
tracing across service boundaries. It enables trace continuity when:

- Agent A calls Agent B via HTTP
- Your agent calls an external MCP server
- Cross-service orchestration (L1 → L2 → L3 agent hierarchy)

W3C Trace Context Headers:
    traceparent: 00-{trace_id}-{parent_span_id}-{flags}
    tracestate: vendor-specific key-value pairs

Example traceparent:
    00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01
    │   │                               │                  │
    │   │                               │                  └── flags (01 = sampled)
    │   │                               └── parent span ID (16 hex)
    │   └── trace ID (32 hex)
    └── version (always 00)

Example tracestate:
    ns=tenant:acme,env:prod,priority:high

Usage:
    # Inject context into outbound HTTP headers
    headers = {}
    inject_context(headers)
    response = httpx.get(url, headers=headers)

    # Extract context from inbound HTTP headers
    ctx = extract_context(request.headers)
    with tracer.start_span("handle_request", parent=ctx) as span:
        ...

    # Auto-instrumentation for httpx/requests
    instrument_http_client()

Architecture:
    This is a critical requirement for multi-agent systems. Without trace
    propagation, each agent starts a new trace, making it impossible to
    track a request across the L1 → L2 → L3 hierarchy.

    With propagation:
    ┌─────────────┐     traceparent     ┌─────────────┐     traceparent     ┌─────────────┐
    │ L1 Concierge│ ───────────────────►│ L2 Analyst  │ ───────────────────►│ L3 Primitive│
    │ trace_id: A │                     │ trace_id: A │                     │ trace_id: A │
    │ span_id: 1  │                     │ span_id: 2  │                     │ span_id: 3  │
    └─────────────┘                     └─────────────┘                     └─────────────┘

    All three agents share trace_id A with proper parent-child linkage.
"""

from __future__ import annotations

import logging
from typing import Dict, Optional, Any, Mapping, MutableMapping, TypeVar

from .span import SpanContext

logger = logging.getLogger("ns_probe.propagation")

# Type for headers (supports dict, httpx.Headers, etc.)
HeadersT = TypeVar("HeadersT", bound=MutableMapping[str, str])

# W3C Trace Context header names (case-insensitive per HTTP spec)
TRACEPARENT_HEADER = "traceparent"
TRACESTATE_HEADER = "tracestate"

# internal vendor key for tracestate
NS_TRACESTATE_KEY = "ns"


# =============================================================================
# INJECT: Add trace context to outbound requests
# =============================================================================


def inject_context(
    headers: MutableMapping[str, str],
    context: Optional[SpanContext] = None,
    tenant_id: Optional[str] = None,
    workspace_id: Optional[str] = None,
    additional_state: Optional[Dict[str, str]] = None,
) -> None:
    """
    Inject W3C Trace Context headers into outbound HTTP request headers.

    This should be called before making HTTP requests to downstream services
    to propagate the trace context and maintain trace continuity.

    Args:
        headers: Mutable mapping to inject headers into (dict, httpx.Headers, etc.)
        context: SpanContext to inject. If None, uses current span context.
        tenant_id: Optional tenant ID to include in tracestate
        workspace_id: Optional workspace ID to include in tracestate. Scope in
            this platform is a TRIPLE (tenant → workspace → subgraph), and
            workspace is the enforced boundary — a downstream service handed only
            the tenant cannot scope to it, and carrying it through
            `additional_state` left every caller to spell the key differently.
        additional_state: Additional key-value pairs for tracestate

    Usage:
        # With explicit context
        ctx = SpanContext(trace_id="abc123...", span_id="def456...")
        headers = {}
        inject_context(headers, ctx)

        # With current span context (most common)
        headers = {}
        inject_context(headers)  # Uses get_current_span_context()
        response = httpx.get(url, headers=headers)

    Example headers after injection:
        {
            "traceparent": "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01",
            "tracestate": "ns=tenant:acme,env:prod"
        }
    """
    # Get context from current span if not provided
    if context is None:
        from .tracer import get_current_span_context

        context = get_current_span_context()

    if context is None or not context.is_valid:
        # No active span context, nothing to inject
        return

    # Inject traceparent
    headers[TRACEPARENT_HEADER] = context.to_traceparent()

    # Build tracestate
    tracestate_parts = []

    # Add existing tracestate if present
    if context.trace_state:
        tracestate_parts.append(context.trace_state)

    # Build ns vendor state
    ns_state_parts = []
    if tenant_id:
        ns_state_parts.append(f"tenant:{tenant_id}")
    if workspace_id:
        ns_state_parts.append(f"workspace:{workspace_id}")
    if additional_state:
        for key, value in additional_state.items():
            # Sanitize key/value (no commas, equals, spaces)
            key = key.replace(",", "_").replace("=", "_").replace(" ", "_")
            value = value.replace(",", "_").replace("=", "_").replace(" ", "_")
            ns_state_parts.append(f"{key}:{value}")

    if ns_state_parts:
        ns_state = f"{NS_TRACESTATE_KEY}=" + ",".join(ns_state_parts)
        tracestate_parts.insert(0, ns_state)  # Our vendor state goes first

    if tracestate_parts:
        headers[TRACESTATE_HEADER] = ",".join(tracestate_parts)

    logger.debug(f"Injected trace context: {headers.get(TRACEPARENT_HEADER)}")


def get_injection_headers(
    context: Optional[SpanContext] = None,
    tenant_id: Optional[str] = None,
    additional_state: Optional[Dict[str, str]] = None,
) -> Dict[str, str]:
    """
    Get W3C Trace Context headers as a dictionary.

    Convenience function that returns headers instead of mutating.

    Args:
        context: SpanContext to use. If None, uses current span context.
        tenant_id: Optional tenant ID to include in tracestate
        additional_state: Additional key-value pairs for tracestate

    Returns:
        Dictionary with traceparent and optionally tracestate headers

    Usage:
        headers = get_injection_headers(tenant_id="acme")
        response = requests.get(url, headers={**existing_headers, **headers})
    """
    headers: Dict[str, str] = {}
    inject_context(headers, context, tenant_id, additional_state)
    return headers


# =============================================================================
# EXTRACT: Parse trace context from inbound requests
# =============================================================================


def extract_context(
    headers: Mapping[str, str],
) -> Optional[SpanContext]:
    """
    Extract W3C Trace Context from inbound HTTP request headers.

    This should be called when receiving HTTP requests to continue
    an existing trace started by an upstream service.

    Args:
        headers: HTTP headers mapping (case-insensitive lookup attempted)

    Returns:
        SpanContext if valid traceparent found, None otherwise

    Usage:
        # In a FastAPI endpoint
        @app.post("/agent/execute")
        async def execute_agent(request: Request):
            parent_ctx = extract_context(request.headers)
            with tracer.start_span("handle_request", parent=parent_ctx) as span:
                # This span continues the trace from the caller
                result = await agent.execute(...)
                return result

        # In a Flask endpoint
        @app.route("/agent/execute", methods=["POST"])
        def execute_agent():
            parent_ctx = extract_context(request.headers)
            with tracer.start_span("handle_request", parent=parent_ctx) as span:
                ...
    """
    # Try to get traceparent (case-insensitive)
    traceparent = _get_header_case_insensitive(headers, TRACEPARENT_HEADER)

    if not traceparent:
        return None

    # Parse traceparent
    context = SpanContext.from_traceparent(traceparent)

    if context is None:
        logger.warning(f"Invalid traceparent header: {traceparent}")
        return None

    # Try to get tracestate
    tracestate = _get_header_case_insensitive(headers, TRACESTATE_HEADER)
    if tracestate:
        # Create new context with tracestate (SpanContext is immutable-ish)
        context = SpanContext(
            trace_id=context.trace_id,
            span_id=context.span_id,
            trace_flags=context.trace_flags,
            trace_state=tracestate,
            is_remote=True,
        )

    logger.debug(f"Extracted trace context: trace_id={context.trace_id[:8]}...")
    return context


def _get_header_case_insensitive(headers: Mapping[str, str], name: str) -> Optional[str]:
    """Get header value with case-insensitive lookup."""
    # Try exact match first (most common)
    if name in headers:
        return headers[name]

    # Try lowercase (HTTP/2 uses lowercase)
    name_lower = name.lower()
    if name_lower in headers:
        return headers[name_lower]

    # Try case-insensitive search
    for key, value in headers.items():
        if key.lower() == name_lower:
            return value

    return None


def parse_tracestate(tracestate: str) -> Dict[str, Dict[str, str]]:
    """
    Parse tracestate header into vendor-keyed dictionary.

    Args:
        tracestate: tracestate header value

    Returns:
        Dictionary mapping vendor keys to their key-value pairs

    Example:
        >>> parse_tracestate("ns=tenant:acme,env:prod,rojo=t:00f067")
        {
            'ns': {'tenant': 'acme', 'env': 'prod'},
            'rojo': {'t': '00f067'}
        }
    """
    result: Dict[str, Dict[str, str]] = {}

    if not tracestate:
        return result

    # Split by comma (vendor entries)
    for entry in tracestate.split(","):
        entry = entry.strip()
        if "=" not in entry:
            continue

        vendor_key, vendor_value = entry.split("=", 1)
        vendor_key = vendor_key.strip()

        # Parse vendor value (colon-separated key:value pairs)
        vendor_data: Dict[str, str] = {}
        for kv in vendor_value.split(","):
            if ":" in kv:
                k, v = kv.split(":", 1)
                vendor_data[k.strip()] = v.strip()
            else:
                # Single value without key
                vendor_data["value"] = kv.strip()

        result[vendor_key] = vendor_data

    return result


def get_ns_tracestate(headers: Mapping[str, str]) -> Dict[str, str]:
    """
    Get internal-specific tracestate values.

    Args:
        headers: HTTP headers

    Returns:
        Dictionary of ns-specific key-value pairs

    Usage:
        ns_state = get_ns_tracestate(request.headers)
        tenant_id = ns_state.get('tenant')
        priority = ns_state.get('priority')
    """
    tracestate = _get_header_case_insensitive(headers, TRACESTATE_HEADER)
    if not tracestate:
        return {}

    parsed = parse_tracestate(tracestate)
    return parsed.get(NS_TRACESTATE_KEY, {})


# =============================================================================
# HTTP CLIENT INSTRUMENTATION
# =============================================================================

_httpx_instrumented = False
_requests_instrumented = False


def instrument_http_clients() -> None:
    """
    Auto-instrument HTTP clients (httpx, requests) to inject trace context.

    After calling this, all outbound HTTP requests will automatically
    include traceparent/tracestate headers.

    Supported libraries:
        - httpx (sync and async)
        - requests

    Usage:
        from ns_probe.propagation import instrument_http_clients
        instrument_http_clients()

        # Now all HTTP requests automatically propagate trace context
        response = httpx.get("http://downstream-service/api")
    """
    _instrument_httpx()
    _instrument_requests()


def _instrument_httpx() -> None:
    """Instrument httpx library for automatic trace propagation."""
    global _httpx_instrumented
    if _httpx_instrumented:
        return

    try:
        import httpx

        original_send = httpx.Client.send
        original_async_send = httpx.AsyncClient.send

        def patched_send(self: Any, request: Any, **kwargs: Any) -> Any:
            inject_context(request.headers)
            return original_send(self, request, **kwargs)

        async def patched_async_send(self: Any, request: Any, **kwargs: Any) -> Any:
            inject_context(request.headers)
            return await original_async_send(self, request, **kwargs)

        httpx.Client.send = patched_send  # type: ignore
        httpx.AsyncClient.send = patched_async_send  # type: ignore

        _httpx_instrumented = True
        logger.debug("Instrumented httpx for trace propagation")

    except ImportError:
        pass  # httpx not installed


def _instrument_requests() -> None:
    """Instrument requests library for automatic trace propagation."""
    global _requests_instrumented
    if _requests_instrumented:
        return

    try:
        import requests  # noqa: F401 — availability probe
        from requests import Session

        original_request = Session.request

        def patched_request(self: Any, method: str, url: str, **kwargs: Any) -> Any:
            headers = kwargs.get("headers", {})
            if headers is None:
                headers = {}
            inject_context(headers)
            kwargs["headers"] = headers
            return original_request(self, method, url, **kwargs)

        Session.request = patched_request  # type: ignore

        _requests_instrumented = True
        logger.debug("Instrumented requests for trace propagation")

    except ImportError:
        pass  # requests not installed


# =============================================================================
# CONVENIENCE: Context manager for incoming requests
# =============================================================================

from contextlib import contextmanager
from typing import Iterator


@contextmanager
def trace_from_headers(
    headers: Mapping[str, str],
    span_name: str = "handle_request",
) -> Iterator[Any]:
    """
    Context manager to start a span from incoming HTTP headers.

    This is the recommended way to handle incoming traced requests.
    It extracts the context and starts a properly parented span.

    Args:
        headers: HTTP headers from incoming request
        span_name: Name for the new span

    Yields:
        The created span

    Usage:
        @app.post("/agent/execute")
        async def execute_agent(request: Request):
            with trace_from_headers(request.headers, "execute_agent") as span:
                span.set_attribute("agent.type", "portfolio")
                result = await agent.execute(...)
                return result
    """
    from .tracer import get_tracer
    from .span import SpanKind

    parent_ctx = extract_context(headers)
    tracer = get_tracer("ns_probe.http")

    with tracer.start_span(
        name=span_name,
        kind=SpanKind.SERVER,
        parent=parent_ctx,
    ) as span:
        # Add standard HTTP attributes
        span.set_attribute("http.request.header.traceparent_present", parent_ctx is not None)
        if parent_ctx:
            span.set_attribute("trace.parent.is_remote", parent_ctx.is_remote)

        yield span
