"""
OTLP Log Exporter for ns_probe.

Ships Python logging records to ns_collector via HTTP POST /v1/logs.
Uses the same gzip + JSON pattern as the trace exporter.

Architecture:
    Python logging.Handler → queue → background thread → HTTP POST /v1/logs
"""

from __future__ import annotations

import gzip
import logging
import queue
import threading
import urllib.request
import urllib.error
from typing import List, Optional

from ._compat import json_dumps_bytes
from .phi_sanitizer import SanitizedText, sanitize_free_text

logger = logging.getLogger("ns_probe.log_exporter")

# Severity mapping: Python logging level → OTLP severity number
_SEVERITY_MAP = {
    logging.DEBUG: 5,
    logging.INFO: 9,
    logging.WARNING: 13,
    logging.ERROR: 17,
    logging.CRITICAL: 21,
}


class OTLPLogHandler(logging.Handler):
    """
    Python logging.Handler that batches log records and ships them
    to ns_collector /v1/logs as OTLP JSON.
    """

    def __init__(
        self,
        endpoint: str,
        service_name: str = "unknown-service",
        flush_interval_sec: float = 2.0,
        max_batch_size: int = 100,
        compression: bool = True,
    ) -> None:
        super().__init__()
        # Replace /v1/traces with /v1/logs if needed
        self._endpoint = endpoint.replace("/v1/traces", "/v1/logs")
        self._service_name = service_name  # fallback only; _current_service_name() takes priority
        self._flush_interval = flush_interval_sec
        self._max_batch_size = max_batch_size
        self._compression = compression
        self._queue: queue.Queue = queue.Queue(maxsize=4096)
        self._shutdown = threading.Event()

        self._thread = threading.Thread(
            target=self._export_loop,
            name="ns_probe_log_exporter",
            daemon=True,
        )
        self._thread.start()

    def emit(self, record: logging.LogRecord) -> None:
        """Called by the logging framework for every log record."""
        # Skip our own logs to avoid recursion
        if record.name.startswith("ns_probe"):
            return
        try:
            self._queue.put_nowait(record)
        except queue.Full:
            pass  # drop silently, never block the caller

    def _export_loop(self) -> None:
        while not self._shutdown.is_set():
            self._shutdown.wait(timeout=self._flush_interval)
            self._flush()

    def _flush(self) -> None:
        records: List[logging.LogRecord] = []
        while len(records) < self._max_batch_size:
            try:
                records.append(self._queue.get_nowait())
            except queue.Empty:
                break
        if not records:
            return
        try:
            payload = self._build_payload(records)
            self._send(payload)
        except Exception:
            pass  # fail-open: never crash the agent

    def _current_service_name(self) -> str:
        """Read service_name from the live ns_probe config so it updates when configure() is re-called."""
        try:
            from .config import get_config

            return get_config().service_name or self._service_name
        except Exception:
            return self._service_name

    def _sanitize_body(self, record: logging.LogRecord) -> SanitizedText:
        """Format the log record body and apply PHI sanitization.

        FREE TEXT, so this uses sanitize_free_text rather than the attribute API —
        the same reason Span.status_message does.

        Routing it through sanitize_attributes failed OPEN under the keys engine:
        that engine matches on the attribute KEY, the synthetic "_body" key
        matched nothing, and the fallback for an unmatched string is the
        JSON-embedded scan, which returns immediately for anything not starting
        with { or [. A prose log line is not JSON, so
        `logger.error("lookup failed for patient %s", name)` exported verbatim
        with NS_PROBE_PHI_ENABLED=true.

        Returns the DEK alongside the text, because encrypting the body under
        presidio produces one and nothing else in an OTLP log record carries it.

        Under presidio the body is detected and redacted, so it stays readable.
        Under the `keys` engine it is suppressed, because there is no way to find
        a name in prose — masking leaves "patient Jane Doe" intact, which is the
        leak this replaced. An operator who wants partial masking instead can set
        NS_PROBE_PHI_MASK_FREE_TEXT=true; `sanitize_free_text` reads that.
        """
        body = self.format(record) if self.formatter else record.getMessage()
        return sanitize_free_text(body)

    def _build_payload(self, records: List[logging.LogRecord]) -> bytes:
        # Try to get current trace/span IDs from ns_probe context
        trace_id = ""
        span_id = ""
        try:
            from .tracer import get_current_span_context

            ctx = get_current_span_context()
            if ctx:
                trace_id = ctx.trace_id or ""
                span_id = ctx.span_id or ""
        except Exception:
            pass

        log_records = []
        for r in records:
            safe_body = self._sanitize_body(r)

            attributes = [
                {"key": "logger.name", "value": {"stringValue": r.name}},
                {"key": "code.filepath", "value": {"stringValue": r.pathname or ""}},
                {"key": "code.lineno", "value": {"intValue": str(r.lineno)}},
            ]
            # Ciphertext whose key was dropped is unreadable, and a log record has
            # no other surface to hang it on — same defect as the span status
            # message, so it gets the same treatment.
            if safe_body.encrypted_dek:
                attributes.append(
                    {
                        "key": "ns.crypto.encrypted_dek",
                        "value": {"stringValue": safe_body.encrypted_dek},
                    }
                )

            log_records.append(
                {
                    "timeUnixNano": str(int(r.created * 1e9)),
                    "observedTimeUnixNano": str(int(r.created * 1e9)),
                    "severityText": r.levelname,
                    "severityNumber": _SEVERITY_MAP.get(r.levelno, 9),
                    "body": {"stringValue": safe_body.text or ""},  # PHI-safe
                    "traceId": trace_id,
                    "spanId": span_id,
                    "attributes": attributes,
                }
            )

        payload = {
            "resourceLogs": [
                {
                    "resource": {
                        "attributes": [
                            {
                                "key": "service.name",
                                "value": {"stringValue": self._current_service_name()},
                            },
                        ]
                    },
                    "scopeLogs": [
                        {
                            "scope": {"name": "ns_probe.log_exporter", "version": "0.1.0"},
                            "logRecords": log_records,
                        }
                    ],
                }
            ]
        }
        return json_dumps_bytes(payload)

    def _send(self, payload: bytes) -> None:
        headers = {"Content-Type": "application/json"}
        if self._compression:
            payload = gzip.compress(payload, compresslevel=6)
            headers["Content-Encoding"] = "gzip"

        request = urllib.request.Request(
            self._endpoint,
            data=payload,
            headers=headers,
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=5) as response:
            response.read()

    def shutdown(self) -> None:
        self._shutdown.set()
        self._thread.join(timeout=3)
        self._flush()  # final drain


# Module-level handle to the active handler so we can clean up
_active_handler: Optional[OTLPLogHandler] = None


class _NoiseFilter(logging.Filter):
    """Drop log records from known-noisy third-party libraries."""

    _SKIP = (
        "openai",
        "httpx",
        "urllib3",
        "httpcore",
        "aiokafka",
        "asyncio",
        "uvicorn",
        "fastapi",
        "starlette",
        "streamlit",
    )

    def filter(self, record: logging.LogRecord) -> bool:
        return not record.name.startswith(self._SKIP)


def install_log_handler(endpoint: str, service_name: str) -> None:
    """Install the OTLP log handler on the root logger and capture print()."""
    global _active_handler

    # If already installed, just update the fallback service_name — the handler
    # reads the live config at export time so no restart needed.
    if _active_handler is not None:
        _active_handler._service_name = service_name
        _install_print_capture()
        return

    handler = OTLPLogHandler(
        endpoint=endpoint,
        service_name=service_name,
    )
    handler.setLevel(logging.INFO)
    handler.addFilter(_NoiseFilter())

    root = logging.getLogger()
    # Ensure the root logger passes INFO records to handlers
    if root.level == logging.NOTSET or root.level > logging.INFO:
        root.setLevel(logging.INFO)
    root.addHandler(handler)
    _active_handler = handler

    # Capture print() output as log records
    _install_print_capture()


class _PrintCapture:
    """Wraps sys.stdout/sys.stderr so print() also emits log records."""

    def __init__(self, original, level: int = logging.INFO):
        self._original = original
        self._level = level
        self._logger = logging.getLogger("print")
        self._buf = ""

    def write(self, text: str) -> int:
        n = self._original.write(text)
        # Buffer lines and emit complete lines only
        self._buf += text
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            line = line.strip()
            if line:
                self._logger.log(self._level, line)
        return n

    def flush(self):
        self._original.flush()
        # Emit any remaining buffered text
        if self._buf.strip():
            self._logger.log(self._level, self._buf.strip())
            self._buf = ""

    def __getattr__(self, name):
        return getattr(self._original, name)


_print_capture_installed = False


def _install_print_capture():
    """Replace sys.stdout/sys.stderr with wrappers that feed the log handler.

    Uses isinstance check instead of a boolean flag so that Streamlit's per-run
    stdout rotation (Streamlit replaces sys.stdout on every script re-run) is
    detected and re-wrapped correctly.
    """
    global _print_capture_installed
    import sys

    if not isinstance(sys.stdout, _PrintCapture):
        sys.stdout = _PrintCapture(sys.stdout, logging.INFO)
    if not isinstance(sys.stderr, _PrintCapture):
        sys.stderr = _PrintCapture(sys.stderr, logging.ERROR)
    _print_capture_installed = True


def shutdown_log_handler() -> None:
    """Shutdown and remove the log handler, restore original stdout/stderr."""
    global _active_handler, _print_capture_installed
    if _active_handler:
        logging.getLogger().removeHandler(_active_handler)
        _active_handler.shutdown()
        _active_handler = None
    if _print_capture_installed:
        import sys

        if isinstance(sys.stdout, _PrintCapture):
            sys.stdout = sys.stdout._original
        if isinstance(sys.stderr, _PrintCapture):
            sys.stderr = sys.stderr._original
        _print_capture_installed = False
