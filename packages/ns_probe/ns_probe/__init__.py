"""
ns_probe - Lightweight Observability SDK for Agents.

This package provides zero-overhead, non-blocking instrumentation for
AI agents built on the platform.

Quick Start:
    # Option 1: CLI wrapper (recommended, zero code changes)
    $ ns-probe-run my_agent.py

    # Option 2: Programmatic initialization
    from ns_probe import configure
    configure(
        endpoint="http://ns-collector:4318/v1/traces",
        service_name="my-agent",
    )

    # Option 3: Manual tracing
    from ns_probe import get_tracer
    tracer = get_tracer("my_module")
    with tracer.start_span("my.operation") as span:
        span.set_attribute("key", "value")
        # ... do work ...

Architecture:
    ┌─────────────────┐      ┌─────────────────┐      ┌─────────────────┐
    │  Agent Thread   │      │  Ring Buffer    │      │ Exporter Thread │
    │  (Main Thread)  │─────►│  (SPSC Queue)   │─────►│  (Daemon)       │
    └─────────────────┘      └─────────────────┘      └────────┬────────┘
                                                              │ OTLP/HTTP
                                                              ▼
                                                    ┌─────────────────┐
                                                    │  ns_collector   │
                                                    └─────────────────┘

Design Principles:
    1. Agent's main thread must NEVER block for telemetry
    2. Memory allocated once at init (zero GC during runtime)
    3. Fail-open: telemetry failures don't affect agent
    4. Zero code changes with import hook patching

Bundled Dependencies:
    - orjson: 5-10x faster JSON serialization (falls back to stdlib json)
    - protobuf: Efficient OTLP wire format (falls back to OTLP/JSON)
    - requests: HTTP transport to collector (falls back to urllib)

    All three are reached through ns_probe._compat, which degrades to the
    stdlib when they are absent. None is load-bearing.
"""

__version__ = "0.1.0"

# =============================================================================
# CAPABILITY DETECTION: Runtime inspection of available dependencies
# =============================================================================
import os

from ._compat import (
    JSON_ENCODER,
    PROTOBUF_AVAILABLE,
    HTTP_CLIENT,
    json_dumps,
    json_loads,
)

# Store capabilities for runtime inspection
_CAPABILITIES = {
    "json_backend": JSON_ENCODER,
    "protobuf_available": PROTOBUF_AVAILABLE,
    "http_backend": HTTP_CLIENT,
}

# Optional: Print capabilities on first import (controlled by env var)
if os.getenv("NS_PROBE_VERBOSE") == "1":
    print("[ns_probe] Capabilities:")
    print(f"  JSON: {JSON_ENCODER} {'✓' if JSON_ENCODER == 'orjson' else '(slower, stdlib)'}")
    print(f"  Protobuf: {'✓' if PROTOBUF_AVAILABLE else '✗ (using JSON)'}")
    print(f"  HTTP: {HTTP_CLIENT} {'✓' if HTTP_CLIENT == 'requests' else '(stdlib urllib)'}")


def get_capabilities():
    """
    Get the current runtime capabilities of ns_probe.

    Returns:
        dict: Capability flags showing which optional dependencies are active

    Example:
        >>> from ns_probe import get_capabilities
        >>> caps = get_capabilities()
        >>> if caps['json_backend'] == 'stdlib':
        ...     print("Consider installing orjson for better performance")
    """
    return _CAPABILITIES.copy()


# =============================================================================
# PUBLIC API IMPORTS
# =============================================================================

# Public API
from .config import configure, get_config, ProbeConfig
from .tracer import (
    get_tracer,
    shutdown,
    force_flush,
    get_current_span_context,
    # Ambient attributes — merged into every span created in this context
    context_attributes,
    set_context_attributes,
    get_context_attributes,
    reset_context_attributes,
    # High-performance decorators and helpers
    observe,
    trace_span,
    # Micro-instrumentation
    step,
    track,
    atrack,
    event,
    current_span,
    # UNIQUE: Agent-Native Observability (not in Langfuse/Datadog/New Relic)
    thought,
    ThoughtContext,  # Chain-of-Thought capture
    guard,
    Guard,
    CostBudgetExceeded,
    LoopBudgetExceeded,  # Cost/loop budgets
    snapshot,  # State capture
    branch,
    BranchContext,  # Decision tree tracking
)
from .span import Span, SpanKind, StatusCode, SpanContext

# Agent-level observability — the turn, and the measurements hung off it.
# `observe` records what a function was GIVEN; `set_metrics` records what it
# WORKED OUT, which is what an agent is actually judged on.
from .agent import (
    turn,
    set_metrics,
    as_bool,
    clip,
    storage_trace_id,
    SPAN_TEXT_LIMIT,
    TURN_ROOT_ATTR,
    TURN_IDENTITY_KEYS,
)
from . import outcome_ledger
from .outcome_ledger import (
    OUTCOME_BLOCKED_INPUT,
    OUTCOME_BLOCKED_OUTPUT,
    OUTCOME_ERROR,
    OUTCOME_OK,
)
from .otel_bridge import (
    install_otel_bridge,
    instrument_via_otel,
)
from .patcher import install_import_hook, patch_existing_base_agent
from .instrumentors import instrument_all

# PHI Sanitization
from .phi_sanitizer import (
    configure_phi,
    sanitize_attributes,
    sanitize_free_text,
    get_phi_config,
    PHIEngine,
    SanitizationStrategy,
    REDACTED_PLACEHOLDER,
)

# W3C Trace Context Propagation (Distributed Tracing)
from .propagation import (
    inject_context,
    extract_context,
    get_injection_headers,
    trace_from_headers,
    instrument_http_clients,
    get_ns_tracestate,
)

# Span ergonomics: manual lifecycle, blocks, after-the-fact spans, events, I/O
from .spans import (
    start_span,
    end_span,
    traced,
    record_span,
    add_event,
    set_span_io,
    set_span_attributes,
    set_current_span_attributes,
    span_scope,
    rename_current_span,
)

# Voice agents (LiveKit AgentSession): STT / end-of-turn / TTS spans + envelope
from .voice import instrument_livekit, voice_call, voice_turn

# Inline PHI masking that keeps text readable
from .phi_sanitizer import mask_text

# Diagnostics
from .exporter import get_json_encoder

__all__ = [
    # Configuration
    "configure",
    "get_config",
    "ProbeConfig",
    # PHI Sanitization
    "configure_phi",
    "sanitize_attributes",
    "sanitize_free_text",
    "get_phi_config",
    "PHIEngine",
    "REDACTED_PLACEHOLDER",
    # Capabilities
    "get_capabilities",  # Inspect available optional dependencies
    # Tracing
    "get_tracer",
    "shutdown",
    "force_flush",
    "get_current_span_context",
    # Ambient span attributes (session.id and friends, on every span)
    "context_attributes",
    "set_context_attributes",
    "get_context_attributes",
    "reset_context_attributes",
    # Outcome Ledger — one row per conversation turn, shared by every agent
    "outcome_ledger",
    "OUTCOME_OK",
    "OUTCOME_BLOCKED_INPUT",
    "OUTCOME_BLOCKED_OUTPUT",
    "OUTCOME_ERROR",
    # Decorators (High-Performance)
    "observe",
    "trace_span",  # Alias for observe
    # Agent-level observability (turn + computed measurements)
    "turn",  # One conversation turn, as its own trace root
    "set_metrics",  # Attach computed measurements to the current span
    "as_bool",  # Booleans must travel as 'true'/'false' strings
    "clip",  # Truncate free text before it becomes an attribute
    "storage_trace_id",  # The id as stored downstream (collector workaround)
    "SPAN_TEXT_LIMIT",
    "TURN_ROOT_ATTR",
    "TURN_IDENTITY_KEYS",
    # Micro-Instrumentation (Granular Tracking)
    "step",  # Context manager for code blocks
    "track",  # One-liner for sync function calls
    "atrack",  # One-liner for async function calls
    "event",  # Point-in-time marker (no duration)
    "current_span",  # Access current span directly
    # ═══════════════════════════════════════════════════════════════════════
    # UNIQUE DIFFERENTIATORS - NOT IN LANGFUSE/DATADOG/NEW RELIC/OPENLLMETRY
    # ═══════════════════════════════════════════════════════════════════════
    # Chain-of-Thought Capture
    "thought",  # 🧠 Capture reasoning, decision, confidence, alternatives
    "ThoughtContext",
    # Cost & Loop Budgets
    "guard",  # 💰 Enforce cost limits, prevent infinite loops
    "Guard",
    "CostBudgetExceeded",
    "LoopBudgetExceeded",
    # State Snapshots
    "snapshot",  # 📸 Capture agent state, compute diffs
    # Decision Trees
    "branch",  # 🌳 Track decisions and alternatives
    "BranchContext",
    # ═══════════════════════════════════════════════════════════════════════
    # W3C TRACE CONTEXT PROPAGATION (Distributed Multi-Agent Tracing)
    # ═══════════════════════════════════════════════════════════════════════
    "inject_context",  # Add traceparent/tracestate to outbound HTTP
    "extract_context",  # Parse traceparent/tracestate from inbound HTTP
    "get_injection_headers",  # Get headers as dict (convenience)
    "trace_from_headers",  # Context manager for incoming requests
    "instrument_http_clients",  # Auto-instrument httpx/requests
    "get_ns_tracestate",  # Get internal-specific tracestate values
    # Span types
    "Span",
    "SpanKind",
    "StatusCode",
    "SpanContext",
    # Instrumentation
    "install_import_hook",
    "patch_existing_base_agent",
    "instrument_all",
    "install_otel_bridge",
    "instrument_via_otel",
    # Span ergonomics
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
    # Voice agents
    "instrument_livekit",
    "voice_call",
    "voice_turn",
    # PHI
    "mask_text",
    # Diagnostics
    "get_json_encoder",  # Returns 'orjson' (fast) or 'json' (stdlib)
]
