"""
Async OTLP Exporter for ns_probe.

This module implements the non-blocking export layer that ships spans
from the ring buffer to the ns_collector service.

Architecture:
    The exporter runs on a daemon thread, completely decoupled from the
    Agent's main thread. The Agent pushes spans to the ring buffer (O(1),
    ~50ns), and the exporter pulls batches and sends them over HTTP.

    ┌─────────────────┐      ┌─────────────────┐      ┌─────────────────┐
    │  Agent Thread   │      │  Ring Buffer    │      │ Exporter Thread │
    │  (Main Thread)  │─────►│  (SPSC Queue)   │─────►│  (Daemon)       │
    │                 │ push │  4096 slots     │ pop  │                 │
    └─────────────────┘      └─────────────────┘      └────────┬────────┘
                                                              │
                                                              │ HTTP POST
                                                              ▼
                                                    ┌─────────────────┐
                                                    │  ns_collector   │
                                                    │  /v1/traces     │
                                                    └─────────────────┘

Non-Blocking Guarantees:
    - push() never blocks (ring buffer write)
    - If buffer overflows, spans are dropped (never block)
    - If HTTP fails, spans are retried with backoff (in exporter thread)
    - Exporter thread failures do not affect Agent execution

Wire Format:
    We use OTLP/HTTP with JSON encoding for simplicity.
    Protobuf is faster but adds complexity; JSON is sufficient for
    our throughput requirements (<100K spans/sec).
"""

from __future__ import annotations

import gzip
import logging
import threading
import time
import urllib.request
import urllib.error
from typing import List, Optional, Dict, Any
from dataclasses import dataclass
from enum import Enum

from .buffer import BatchingBuffer
from .span import Span
from .phi_sanitizer import sanitize_attributes, sanitize_free_text

# ---------------------------------------------------------------------
# Fast JSON Serialization: orjson with stdlib fallback
# ---------------------------------------------------------------------
# orjson is ~5x faster than json.dumps() due to Rust implementation
# and GIL-free serialization. Falls back gracefully to stdlib json.
#
# Benchmark (1000 spans, avg payload 1.5KB):
#   - json.dumps:   10.2ms
#   - orjson.dumps:  2.1ms
# ---------------------------------------------------------------------

# Use compatibility layer for graceful degradation
from ._compat import json_dumps_bytes, JSON_ENCODER


def _fast_json_encode(obj: Any) -> bytes:
    """Encode object to JSON bytes (uses orjson if available, otherwise stdlib)."""
    return json_dumps_bytes(obj)


_JSON_ENCODER = JSON_ENCODER

logger = logging.getLogger(__name__)
from .config import get_config, ProbeConfig

logger = logging.getLogger("ns_probe.exporter")


# =============================================================================
# CIRCUIT BREAKER
# =============================================================================
# Prevents retry storms when ns_collector is down.
# State machine: CLOSED → OPEN → HALF_OPEN → CLOSED
#
# - CLOSED: Normal operation, requests pass through
# - OPEN: Circuit tripped, requests fail fast (no network call)
# - HALF_OPEN: Testing recovery, allow one request through
#
# This is a critical P0 requirement from architecture.md:
# "Circuit breaker prevents retry storms"
# =============================================================================


class CircuitState(Enum):
    """Circuit breaker states."""

    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Failing fast, no requests
    HALF_OPEN = "half_open"  # Testing recovery


class CircuitBreaker:
    """
    Circuit breaker to prevent retry storms on collector failures.

    Thread Safety:
        Uses a lock for state transitions. The circuit breaker is designed
        to be used from a single exporter thread, but the lock ensures
        correctness if accessed from multiple threads.

    Configuration:
        - failure_threshold: Number of consecutive failures before opening
        - recovery_timeout_sec: Time to wait before testing recovery
        - success_threshold: Successes needed in half-open to close

    Usage:
        cb = CircuitBreaker()

        if cb.allow_request():
            try:
                result = do_request()
                cb.record_success()
            except Exception:
                cb.record_failure()
        else:
            # Circuit is open, fail fast
            logger.warning("Circuit open, skipping request")

    Architecture:
        This prevents the following failure cascade:

        1. ns_collector goes down
        2. Exporter retries indefinitely
        3. Ring buffer fills up (blocked by retries)
        4. Agent spans start dropping
        5. Retries add network pressure

        With circuit breaker:
        1. ns_collector goes down
        2. 5 failures → circuit OPENS
        3. Requests fail fast (no network)
        4. After 30s → circuit HALF_OPEN
        5. Test request succeeds → circuit CLOSES
    """

    __slots__ = (
        "_state",
        "_failure_count",
        "_success_count",
        "_last_failure_time",
        "_failure_threshold",
        "_recovery_timeout_sec",
        "_success_threshold",
        "_lock",
        "_total_rejections",
        "_total_trips",
    )

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout_sec: float = 30.0,
        success_threshold: int = 2,
    ) -> None:
        """
        Initialize circuit breaker.

        Args:
            failure_threshold: Consecutive failures before opening circuit
            recovery_timeout_sec: Seconds to wait before testing recovery
            success_threshold: Successes needed in half-open to fully close
        """
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time: float = 0.0

        self._failure_threshold = failure_threshold
        self._recovery_timeout_sec = recovery_timeout_sec
        self._success_threshold = success_threshold

        self._lock = threading.Lock()

        # Metrics
        self._total_rejections = 0
        self._total_trips = 0

    @property
    def state(self) -> CircuitState:
        """Current circuit state."""
        with self._lock:
            return self._state

    @property
    def is_open(self) -> bool:
        """True if circuit is open (rejecting requests)."""
        return self.state == CircuitState.OPEN

    def allow_request(self) -> bool:
        """
        Check if a request should be allowed through.

        Returns:
            True if request should proceed, False if circuit is open
        """
        with self._lock:
            if self._state == CircuitState.CLOSED:
                return True

            if self._state == CircuitState.OPEN:
                # Check if recovery timeout has passed
                elapsed = time.monotonic() - self._last_failure_time
                if elapsed >= self._recovery_timeout_sec:
                    # Transition to half-open, allow test request
                    self._state = CircuitState.HALF_OPEN
                    self._success_count = 0
                    logger.info(
                        f"Circuit breaker: OPEN → HALF_OPEN (testing recovery after {elapsed:.1f}s)"
                    )
                    return True
                else:
                    # Still in open state, reject
                    self._total_rejections += 1
                    return False

            if self._state == CircuitState.HALF_OPEN:
                # Allow test requests through in half-open
                return True

            return False

    def record_success(self) -> None:
        """Record a successful request."""
        with self._lock:
            if self._state == CircuitState.HALF_OPEN:
                self._success_count += 1
                if self._success_count >= self._success_threshold:
                    # Enough successes, close the circuit
                    self._state = CircuitState.CLOSED
                    self._failure_count = 0
                    logger.info(
                        f"Circuit breaker: HALF_OPEN → CLOSED (recovered after {self._success_count} successes)"
                    )
            elif self._state == CircuitState.CLOSED:
                # Reset failure count on success
                self._failure_count = 0

    def record_failure(self) -> None:
        """Record a failed request."""
        with self._lock:
            self._last_failure_time = time.monotonic()

            if self._state == CircuitState.HALF_OPEN:
                # Failed during recovery test, back to open
                self._state = CircuitState.OPEN
                self._total_trips += 1
                logger.warning("Circuit breaker: HALF_OPEN → OPEN (recovery failed)")

            elif self._state == CircuitState.CLOSED:
                self._failure_count += 1
                if self._failure_count >= self._failure_threshold:
                    # Too many failures, open the circuit
                    self._state = CircuitState.OPEN
                    self._total_trips += 1
                    logger.warning(
                        f"Circuit breaker: CLOSED → OPEN "
                        f"({self._failure_count} consecutive failures, "
                        f"will test recovery in {self._recovery_timeout_sec}s)"
                    )

    def reset(self) -> None:
        """Reset circuit breaker to closed state (for testing)."""
        with self._lock:
            self._state = CircuitState.CLOSED
            self._failure_count = 0
            self._success_count = 0
            self._last_failure_time = 0.0

    @property
    def stats(self) -> Dict[str, Any]:
        """Circuit breaker statistics."""
        with self._lock:
            return {
                "state": self._state.value,
                "failure_count": self._failure_count,
                "success_count": self._success_count,
                "total_rejections": self._total_rejections,
                "total_trips": self._total_trips,
                "seconds_since_last_failure": (
                    time.monotonic() - self._last_failure_time
                    if self._last_failure_time > 0
                    else None
                ),
            }


@dataclass
class ExportResult:
    """Result of an export operation."""

    success: bool
    spans_exported: int
    spans_dropped: int
    error_message: Optional[str] = None
    retry_after_ms: Optional[int] = None


class OTLPExporter:
    """
    OTLP/HTTP JSON Exporter.

    Converts spans to OTLP JSON format and sends to the collector.
    This class handles the serialization and HTTP transport.

    Thread Safety: Single exporter instance should only be used from
    the exporter thread. Multiple exporters can exist for different
    endpoints.

    Features:
        - orjson for 5x faster JSON serialization (with stdlib fallback)
        - Gzip compression for ~70% bandwidth reduction
        - HTTP Keep-Alive for connection reuse
    """

    __slots__ = (
        "_endpoint",
        "_headers",
        "_timeout",
        "_service_name",
        "_service_version",
        "_resource_attributes",
        "_compression",
        "_config_ref",
    )

    def __init__(
        self,
        endpoint: str,
        headers: Optional[Dict[str, str]] = None,
        timeout_ms: int = 30000,
        service_name: str = "unknown-service",
        service_version: str = "0.0.0",
        resource_attributes: Optional[Dict[str, Any]] = None,
        compression: bool = True,  # Gzip compression enabled by default
        config_ref: Optional[Any] = None,
    ) -> None:
        """
        Initialize the exporter.

        Args:
            endpoint: OTLP collector endpoint (e.g., http://localhost:4318/v1/traces)
            headers: HTTP headers (for authentication)
            timeout_ms: HTTP timeout in milliseconds
            service_name: Service name for resource attributes
            service_version: Service version
            resource_attributes: Additional resource attributes
            compression: Enable gzip compression (default: True, ~70% smaller payloads)
            config_ref: Reference to ProbeConfig for dynamic service_name reads
        """
        self._endpoint = endpoint
        self._headers = headers or {}
        self._timeout = timeout_ms / 1000  # Convert to seconds for urllib
        self._service_name = service_name
        self._service_version = service_version
        self._resource_attributes = resource_attributes or {}
        self._compression = compression
        self._config_ref = config_ref

    def export(self, spans: List[Span]) -> ExportResult:
        """
        Export a batch of spans to the collector.

        Args:
            spans: List of Span objects to export

        Returns:
            ExportResult with success status and counts
        """
        if not spans:
            return ExportResult(success=True, spans_exported=0, spans_dropped=0)

        try:
            # Convert spans to OTLP JSON format
            payload = self._spans_to_otlp(spans)

            # Send HTTP request
            self._send_http(payload)

            return ExportResult(
                success=True,
                spans_exported=len(spans),
                spans_dropped=0,
            )

        except urllib.error.HTTPError as e:
            error_msg = f"HTTP {e.code}: {e.reason}"
            retry_after = None

            # Handle rate limiting
            if e.code == 429:
                retry_header = e.headers.get("Retry-After")
                if retry_header:
                    try:
                        retry_after = int(retry_header) * 1000  # Convert to ms
                    except ValueError:
                        retry_after = 60000  # Default 60s

            logger.warning(f"Export failed: {error_msg}")
            return ExportResult(
                success=False,
                spans_exported=0,
                spans_dropped=len(spans),
                error_message=error_msg,
                retry_after_ms=retry_after,
            )

        except urllib.error.URLError as e:
            error_msg = f"Connection error: {e.reason}"
            logger.warning(f"Export failed: {error_msg}")
            return ExportResult(
                success=False,
                spans_exported=0,
                spans_dropped=len(spans),
                error_message=error_msg,
            )

        except Exception as e:
            error_msg = f"Unexpected error: {str(e)}"
            logger.error(f"Export failed: {error_msg}", exc_info=True)
            return ExportResult(
                success=False,
                spans_exported=0,
                spans_dropped=len(spans),
                error_message=error_msg,
            )

    def _spans_to_otlp(self, spans: List[Span]) -> bytes:
        """
        Convert spans to OTLP JSON format.

        OTLP structure:
        {
            "resourceSpans": [{
                "resource": { "attributes": [...] },
                "scopeSpans": [{
                    "scope": { "name": "ns_probe", "version": "0.1.0" },
                    "spans": [...]
                }]
            }]
        }
        """
        # Build resource attributes
        # Read service_name dynamically from config if available
        svc = self._config_ref.service_name if self._config_ref else self._service_name
        resource_attrs = [
            {"key": "service.name", "value": {"stringValue": svc}},
            {"key": "service.version", "value": {"stringValue": self._service_version}},
        ]

        for key, value in self._resource_attributes.items():
            resource_attrs.append(self._attr_to_otlp(key, value))

        # Convert spans
        otlp_spans = []
        for span in spans:
            otlp_span = self._span_to_otlp(span)
            otlp_spans.append(otlp_span)

        # Build OTLP payload
        payload = {
            "resourceSpans": [
                {
                    "resource": {
                        "attributes": resource_attrs,
                    },
                    "scopeSpans": [
                        {
                            "scope": {
                                "name": "ns_probe",
                                "version": "0.1.0",
                            },
                            "spans": otlp_spans,
                        }
                    ],
                }
            ],
        }

        return _fast_json_encode(payload)

    def _span_to_otlp(self, span: Span) -> Dict[str, Any]:
        """Convert a single span to OTLP format.

        PHI sanitization is applied here — *after* the span is fully
        populated but *before* it is serialized and sent over the wire.

        EVERY attribute-bearing surface is sanitized, not just `span.attributes`.
        Events and links carry attributes too, and they were previously
        serialized verbatim: `record_exception` puts the message and stack trace
        on an event, and agents put snapshot state and reasoning there. So
        NS_PROBE_PHI_ENABLED=true still exported raw PHI to the collector — the
        one thing the flag exists to prevent. Any new attribute surface added to
        Span must be routed through sanitize_attributes here as well.
        """
        # Sanitize attributes (no-op when PHI filtering is disabled)
        safe_attrs = sanitize_attributes(span.attributes)

        # The status message is sanitized BEFORE the attribute list is built,
        # because encrypting it can yield a DEK that has to travel on an
        # attribute. A span whose only content is set_status(ERROR, text) has no
        # other surface to carry it, so dropping it would export ciphertext
        # nothing can read. An existing DEK from span.attributes wins — both come
        # from the same engine, so they are the same key.
        safe_status = sanitize_free_text(span.status_message)
        if safe_status.encrypted_dek and "ns.crypto.encrypted_dek" not in safe_attrs:
            safe_attrs = {
                **safe_attrs,
                "ns.crypto.encrypted_dek": safe_status.encrypted_dek,
            }

        otlp_span: Dict[str, Any] = {
            "traceId": span.trace_id,
            "spanId": span.span_id,
            "name": span.name,
            "kind": span.kind + 1,  # OTLP uses 1-indexed SpanKind
            "startTimeUnixNano": str(span.start_time_ns),
            "endTimeUnixNano": str(span.end_time_ns or span.start_time_ns),
            "attributes": [self._attr_to_otlp(k, v) for k, v in safe_attrs.items()],
            "status": {
                "code": span.status_code,
            },
        }

        # Add parent span ID if present
        if span.parent_span_id:
            otlp_span["parentSpanId"] = span.parent_span_id

        # Add status message if present. Sanitized above — it is a DUPLICATE of
        # the exception text that also lands on an event (start_span catches,
        # calls record_exception, then set_status(ERROR, str(e))), so covering
        # only the event left the same string leaving through this field.
        if safe_status.text:
            otlp_span["status"]["message"] = safe_status.text

        # Add events
        if span.events:
            otlp_span["events"] = [
                {
                    "name": e.name,
                    "timeUnixNano": str(e.timestamp_ns),
                    "attributes": [
                        self._attr_to_otlp(k, v)
                        for k, v in sanitize_attributes(e.attributes).items()
                    ],
                }
                for e in span.events
            ]

        # Add links
        if span.links:
            otlp_span["links"] = [
                {
                    "traceId": link.trace_id,
                    "spanId": link.span_id,
                    "attributes": [
                        self._attr_to_otlp(k, v)
                        for k, v in sanitize_attributes(link.attributes).items()
                    ],
                }
                for link in span.links
            ]

        return otlp_span

    def _attr_to_otlp(self, key: str, value: Any) -> Dict[str, Any]:
        """Convert attribute to OTLP format."""
        attr: Dict[str, Any] = {"key": key, "value": {}}

        if isinstance(value, bool):
            attr["value"]["boolValue"] = value
        elif isinstance(value, int):
            attr["value"]["intValue"] = str(value)
        elif isinstance(value, float):
            # Emit floats as stringValue, not doubleValue: the ns_processor that
            # populates the ClickHouse attributes map reads intValue/stringValue but
            # drops doubleValue, so float attributes (e.g. gen_ai.latency_ms,
            # langgraph.latency_ms) were landing as 0. Stringifying preserves them.
            #
            # repr(value), NOT repr(round(value, 4)). The rounding was a separate
            # lossy decision riding along with the stringify fix: it turned a cost
            # of 0.000021 into "0.0" and flattened every rate below 0.00005 to
            # zero. repr round-trips a float exactly, which is what
            # outcome_ledger.py already relies on, so both wire paths now agree.
            attr["value"]["stringValue"] = repr(value)
        elif isinstance(value, str):
            attr["value"]["stringValue"] = value
        elif isinstance(value, (list, tuple)):
            # Array value
            if value and isinstance(value[0], str):
                attr["value"]["arrayValue"] = {"values": [{"stringValue": v} for v in value]}
            elif value and isinstance(value[0], int):
                attr["value"]["arrayValue"] = {"values": [{"intValue": str(v)} for v in value]}
            else:
                # Fallback to string representation
                attr["value"]["stringValue"] = str(value)
        else:
            # Fallback to string representation
            attr["value"]["stringValue"] = str(value)

        return attr

    def _send_http(self, payload: bytes) -> None:
        """
        Send HTTP POST request to collector.

        Applies gzip compression if enabled (~70% smaller payloads).
        """
        headers = {
            "Content-Type": "application/json",
            **self._headers,
        }

        # Apply gzip compression if enabled
        if self._compression:
            payload = gzip.compress(payload, compresslevel=6)  # Level 6 = good balance
            headers["Content-Encoding"] = "gzip"

        request = urllib.request.Request(
            self._endpoint,
            data=payload,
            headers=headers,
            method="POST",
        )

        with urllib.request.urlopen(request, timeout=self._timeout) as response:
            # Read response to ensure request completed
            response.read()


class BatchExporter:
    """
    Background exporter that runs on a daemon thread.

    This is the main interface used by the tracer. It manages:
    - Periodic export based on interval
    - Circuit breaker for failure isolation
    - Retry with exponential backoff
    - Self-reporting metrics (dropped spans, buffer utilization)
    - Graceful shutdown with final flush

    The exporter thread is a daemon, so it will not prevent program exit.
    However, shutdown() should be called to flush pending spans.

    Circuit Breaker:
        Prevents retry storms when collector is down. After 5 consecutive
        failures, requests fail fast for 30 seconds before testing recovery.
    """

    def __init__(
        self,
        buffer: BatchingBuffer[Span],
        config: Optional[ProbeConfig] = None,
    ) -> None:
        """
        Initialize the batch exporter.

        Args:
            buffer: Ring buffer to consume spans from
            config: Configuration (uses global config if None)
        """
        self._buffer = buffer
        self._config = config or get_config()

        # Initialize OTLP exporter
        self._exporter = OTLPExporter(
            endpoint=self._config.endpoint,
            headers=self._config.headers,
            timeout_ms=self._config.export_timeout_ms,
            service_name=self._config.service_name,
            service_version=self._config.service_version,
            resource_attributes=self._config.resource_attributes,
            compression=self._config.compression,
            config_ref=self._config,
        )

        # Circuit breaker for failure isolation
        self._circuit_breaker = CircuitBreaker(
            failure_threshold=5,  # Open after 5 consecutive failures
            recovery_timeout_sec=30.0,  # Test recovery after 30 seconds
            success_threshold=2,  # Need 2 successes to fully close
        )

        # Thread control
        self._shutdown_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

        # Metrics
        self._total_exported = 0
        self._total_dropped = 0
        self._export_errors = 0
        self._circuit_rejections = 0

    def start(self) -> None:
        """
        Start the exporter thread.

        The thread is a daemon, so it won't prevent program exit.
        """
        if self._thread is not None and self._thread.is_alive():
            return  # Already running

        self._shutdown_event.clear()
        self._thread = threading.Thread(
            target=self._export_loop,
            name="ns_probe_exporter",
            daemon=True,
        )
        self._thread.start()
        logger.debug("Exporter thread started")

    def shutdown(self, timeout_ms: int = 5000) -> None:
        """
        Shutdown the exporter and flush pending spans.

        Args:
            timeout_ms: Maximum time to wait for flush
        """
        if self._thread is None or not self._thread.is_alive():
            return

        logger.debug("Shutting down exporter...")
        self._shutdown_event.set()

        # Wait for thread to finish
        self._thread.join(timeout=timeout_ms / 1000)

        if self._thread.is_alive():
            logger.warning("Exporter thread did not shutdown cleanly")
        else:
            logger.debug(
                f"Exporter shutdown complete. Exported: {self._total_exported}, Dropped: {self._total_dropped}"
            )

    def force_flush(self) -> int:
        """
        Force export all pending spans (blocks until complete).

        Returns:
            Number of spans exported
        """
        exported = 0
        while True:
            batch = self._buffer.get_batch()
            if not batch:
                break

            result = self._exporter.export(batch)
            exported += result.spans_exported
            self._total_exported += result.spans_exported
            self._total_dropped += result.spans_dropped

            if not result.success:
                self._export_errors += 1

        return exported

    def _export_loop(self) -> None:
        """
        Main export loop (runs on daemon thread).

        Exports batches at regular intervals with:
        - Circuit breaker for failure isolation
        - Exponential backoff on failures
        - Self-reporting metrics
        """
        backoff_ms = 1000  # Initial backoff
        max_backoff_ms = 60000  # Max backoff (1 minute)

        while not self._shutdown_event.is_set():
            try:
                # Get batch from buffer
                batch = self._buffer.get_batch()

                if batch:
                    # Check circuit breaker BEFORE attempting export
                    if not self._circuit_breaker.allow_request():
                        # Circuit is open - fail fast, don't even try
                        self._circuit_rejections += 1
                        self._total_dropped += len(batch)

                        # Log periodically (not every rejection)
                        if self._circuit_rejections % 10 == 1:
                            logger.warning(
                                f"Circuit breaker OPEN: dropping {len(batch)} spans "
                                f"(total rejections: {self._circuit_rejections})"
                            )
                    else:
                        # Circuit allows request, attempt export
                        result = self._exporter.export(batch)

                        self._total_exported += result.spans_exported
                        self._total_dropped += result.spans_dropped

                        if result.success:
                            # Record success with circuit breaker
                            self._circuit_breaker.record_success()
                            backoff_ms = 1000  # Reset backoff on success
                        else:
                            # Record failure with circuit breaker
                            self._circuit_breaker.record_failure()
                            self._export_errors += 1

                            # Handle rate limiting
                            if result.retry_after_ms:
                                backoff_ms = result.retry_after_ms
                            else:
                                # Exponential backoff
                                backoff_ms = min(backoff_ms * 2, max_backoff_ms)

                            logger.debug(f"Export failed, backing off for {backoff_ms}ms")

                # Check dropped count from buffer
                dropped = self._buffer.dropped_count
                if dropped > 0:
                    self._total_dropped += dropped
                    logger.warning(f"Buffer overflow: {dropped} spans dropped")

                # Wait for next interval (or shutdown)
                self._shutdown_event.wait(timeout=self._config.export_interval_ms / 1000)

            except Exception as e:
                logger.error(f"Export loop error: {e}", exc_info=True)
                self._circuit_breaker.record_failure()
                self._shutdown_event.wait(timeout=backoff_ms / 1000)

        # Final flush on shutdown (bypass circuit breaker for cleanup)
        logger.debug("Performing final flush...")
        self._final_flush()

    def _final_flush(self) -> int:
        """
        Final flush on shutdown - attempts export regardless of circuit state.

        This gives us one last chance to export spans before the process exits.
        """
        exported = 0
        while True:
            batch = self._buffer.get_batch()
            if not batch:
                break

            result = self._exporter.export(batch)
            exported += result.spans_exported
            self._total_exported += result.spans_exported
            self._total_dropped += result.spans_dropped

            if not result.success:
                self._export_errors += 1

        return exported

    @property
    def stats(self) -> Dict[str, Any]:
        """Export statistics including circuit breaker state."""
        return {
            "total_exported": self._total_exported,
            "total_dropped": self._total_dropped,
            "export_errors": self._export_errors,
            "circuit_rejections": self._circuit_rejections,
            "buffer_size": len(self._buffer),
            "buffer_utilization_percent": int(self._buffer.utilization * 100),
            "circuit_breaker": self._circuit_breaker.stats,
            "json_encoder": _JSON_ENCODER,  # orjson or json
        }


def get_json_encoder() -> str:
    """
    Returns the active JSON encoder: 'orjson' (fast) or 'json' (stdlib).

    Use this for diagnostics to verify orjson is installed.
    """
    return _JSON_ENCODER
