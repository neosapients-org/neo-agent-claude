"""
Metrics and observability for Neo Memory Hub.

Provides:
- Prometheus-compatible metrics (counters, histograms, gauges)
- OpenTelemetry tracing integration
- Structured operation logging

Usage:
    >>> from neo_memory_hub.core.metrics import get_metrics_registry
    >>> registry = get_metrics_registry()
    >>> registry.inc_counter("memory_operations", operation="store", status="success")
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Generator, Protocol


# =============================================================================
# Metric Data Classes
# =============================================================================


@dataclass(frozen=True)
class OperationMetric:
    """Structured metric for a memory operation."""

    name: str
    duration_ms: float
    success: bool
    error: str | None = None
    metadata: dict[str, Any] | None = None


@dataclass
class Counter:
    """Simple counter metric."""

    name: str
    description: str
    labels: list[str] = field(default_factory=list)
    _values: dict[tuple[str, ...], float] = field(default_factory=dict)

    def inc(self, value: float = 1.0, **labels: str) -> None:
        """Increment counter."""
        key = tuple(labels.get(l, "") for l in self.labels)
        self._values[key] = self._values.get(key, 0) + value

    def get(self, **labels: str) -> float:
        """Get current value."""
        key = tuple(labels.get(l, "") for l in self.labels)
        return self._values.get(key, 0)

    def to_prometheus(self) -> str:
        """Export as Prometheus format."""
        lines = [f"# HELP {self.name} {self.description}", f"# TYPE {self.name} counter"]
        for key, value in self._values.items():
            label_str = ",".join(f'{l}="{v}"' for l, v in zip(self.labels, key) if v)
            if label_str:
                lines.append(f"{self.name}{{{label_str}}} {value}")
            else:
                lines.append(f"{self.name} {value}")
        return "\n".join(lines)


@dataclass
class Histogram:
    """Simple histogram metric for latency tracking."""

    name: str
    description: str
    labels: list[str] = field(default_factory=list)
    buckets: list[float] = field(
        default_factory=lambda: [0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]
    )
    _observations: dict[tuple[str, ...], list[float]] = field(default_factory=dict)

    def observe(self, value: float, **labels: str) -> None:
        """Record an observation."""
        key = tuple(labels.get(l, "") for l in self.labels)
        if key not in self._observations:
            self._observations[key] = []
        self._observations[key].append(value)

    def get_count(self, **labels: str) -> int:
        """Get number of observations."""
        key = tuple(labels.get(l, "") for l in self.labels)
        return len(self._observations.get(key, []))

    def get_sum(self, **labels: str) -> float:
        """Get sum of observations."""
        key = tuple(labels.get(l, "") for l in self.labels)
        return sum(self._observations.get(key, []))

    def to_prometheus(self) -> str:
        """Export as Prometheus format."""
        lines = [f"# HELP {self.name} {self.description}", f"# TYPE {self.name} histogram"]
        for key, observations in self._observations.items():
            label_str = ",".join(f'{l}="{v}"' for l, v in zip(self.labels, key) if v)
            base_labels = f"{{{label_str}}}" if label_str else ""

            # Bucket counts
            for bucket in self.buckets:
                count = sum(1 for o in observations if o <= bucket)
                bucket_labels = f'{label_str},le="{bucket}"' if label_str else f'le="{bucket}"'
                lines.append(f"{self.name}_bucket{{{bucket_labels}}} {count}")

            # +Inf bucket
            inf_labels = f'{label_str},le="+Inf"' if label_str else 'le="+Inf"'
            lines.append(f"{self.name}_bucket{{{inf_labels}}} {len(observations)}")

            # Sum and count
            lines.append(f"{self.name}_sum{base_labels} {sum(observations)}")
            lines.append(f"{self.name}_count{base_labels} {len(observations)}")

        return "\n".join(lines)


@dataclass
class Gauge:
    """Simple gauge metric."""

    name: str
    description: str
    labels: list[str] = field(default_factory=list)
    _values: dict[tuple[str, ...], float] = field(default_factory=dict)

    def set(self, value: float, **labels: str) -> None:
        """Set gauge value."""
        key = tuple(labels.get(l, "") for l in self.labels)
        self._values[key] = value

    def inc(self, value: float = 1.0, **labels: str) -> None:
        """Increment gauge."""
        key = tuple(labels.get(l, "") for l in self.labels)
        self._values[key] = self._values.get(key, 0) + value

    def dec(self, value: float = 1.0, **labels: str) -> None:
        """Decrement gauge."""
        key = tuple(labels.get(l, "") for l in self.labels)
        self._values[key] = self._values.get(key, 0) - value

    def get(self, **labels: str) -> float:
        """Get current value."""
        key = tuple(labels.get(l, "") for l in self.labels)
        return self._values.get(key, 0)

    def to_prometheus(self) -> str:
        """Export as Prometheus format."""
        lines = [f"# HELP {self.name} {self.description}", f"# TYPE {self.name} gauge"]
        for key, value in self._values.items():
            label_str = ",".join(f'{l}="{v}"' for l, v in zip(self.labels, key) if v)
            if label_str:
                lines.append(f"{self.name}{{{label_str}}} {value}")
            else:
                lines.append(f"{self.name} {value}")
        return "\n".join(lines)


# =============================================================================
# Metrics Registry
# =============================================================================


class MetricsRegistry:
    """
    Central registry for all Neo Memory Hub metrics.

    Provides Prometheus-compatible metrics collection and export.

    Metrics tracked:
    - neohub_memory_operations_total: Counter of memory operations
    - neohub_memory_operation_duration_seconds: Histogram of operation latencies
    - neohub_memory_store_filtered_total: Counter of filtered/skipped stores
    - neohub_memory_retrieval_results: Histogram of retrieval result counts
    - neohub_circuit_breaker_state: Gauge for circuit breaker state
    - neohub_active_connections: Gauge for active connections
    """

    def __init__(self) -> None:
        # Operation counters
        self.operations_total = Counter(
            name="neohub_memory_operations_total",
            description="Total number of memory operations",
            labels=["operation", "status", "tenant_id"],
        )

        # Operation duration histogram
        self.operation_duration = Histogram(
            name="neohub_memory_operation_duration_seconds",
            description="Duration of memory operations in seconds",
            labels=["operation", "tenant_id"],
            buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0],
        )

        # Store filtering counter
        self.store_filtered = Counter(
            name="neohub_memory_store_filtered_total",
            description="Number of store operations filtered/skipped",
            labels=["reason", "tenant_id"],
        )

        # Retrieval results histogram
        self.retrieval_results = Histogram(
            name="neohub_memory_retrieval_results",
            description="Number of results returned per retrieval",
            labels=["tenant_id"],
            buckets=[0, 1, 5, 10, 20, 50, 100],
        )

        # Circuit breaker state gauge
        self.circuit_breaker_state = Gauge(
            name="neohub_circuit_breaker_state",
            description="Circuit breaker state (0=closed, 1=open, 2=half-open)",
            labels=["backend"],
        )

        # Active connections gauge
        self.active_connections = Gauge(
            name="neohub_active_connections",
            description="Number of active connections",
            labels=["backend"],
        )

        # Error counter
        self.errors_total = Counter(
            name="neohub_errors_total",
            description="Total number of errors",
            labels=["operation", "error_type", "tenant_id"],
        )

    def inc_counter(
        self,
        name: str,
        value: float = 1.0,
        **labels: str,
    ) -> None:
        """Increment a counter by name."""
        counter = getattr(self, name, None)
        if isinstance(counter, Counter):
            counter.inc(value, **labels)

    def observe_histogram(
        self,
        name: str,
        value: float,
        **labels: str,
    ) -> None:
        """Record a histogram observation."""
        histogram = getattr(self, name, None)
        if isinstance(histogram, Histogram):
            histogram.observe(value, **labels)

    def set_gauge(
        self,
        name: str,
        value: float,
        **labels: str,
    ) -> None:
        """Set a gauge value."""
        gauge = getattr(self, name, None)
        if isinstance(gauge, Gauge):
            gauge.set(value, **labels)

    @contextmanager
    def track_operation(
        self,
        operation: str,
        tenant_id: str = "unknown",
    ) -> Generator[dict[str, Any], None, None]:
        """
        Context manager to track operation metrics.

        Usage:
            with registry.track_operation("store", tenant_id="acme") as ctx:
                # do operation
                ctx["result_count"] = 10
        """
        start_time = time.perf_counter()
        context: dict[str, Any] = {"success": True, "error": None}

        try:
            yield context
        except Exception as e:
            context["success"] = False
            context["error"] = type(e).__name__
            self.errors_total.inc(
                operation=operation,
                error_type=type(e).__name__,
                tenant_id=tenant_id,
            )
            raise
        finally:
            duration = time.perf_counter() - start_time
            status = "success" if context["success"] else "error"

            self.operations_total.inc(
                operation=operation,
                status=status,
                tenant_id=tenant_id,
            )
            self.operation_duration.observe(
                duration,
                operation=operation,
                tenant_id=tenant_id,
            )

    def record_operation(self, metric: OperationMetric) -> None:
        """Record a completed operation metric."""
        status = "success" if metric.success else "error"
        tenant_id = (metric.metadata or {}).get("tenant_id", "unknown")

        self.operations_total.inc(
            operation=metric.name,
            status=status,
            tenant_id=tenant_id,
        )
        self.operation_duration.observe(
            metric.duration_ms / 1000.0,  # Convert to seconds
            operation=metric.name,
            tenant_id=tenant_id,
        )

        if not metric.success and metric.error:
            self.errors_total.inc(
                operation=metric.name,
                error_type=metric.error,
                tenant_id=tenant_id,
            )

    def export_prometheus(self) -> str:
        """Export all metrics in Prometheus format."""
        parts = [
            self.operations_total.to_prometheus(),
            self.operation_duration.to_prometheus(),
            self.store_filtered.to_prometheus(),
            self.retrieval_results.to_prometheus(),
            self.circuit_breaker_state.to_prometheus(),
            self.active_connections.to_prometheus(),
            self.errors_total.to_prometheus(),
        ]
        return "\n\n".join(p for p in parts if p)

    def reset(self) -> None:
        """Reset all metrics (for testing)."""
        self.operations_total._values.clear()
        self.operation_duration._observations.clear()
        self.store_filtered._values.clear()
        self.retrieval_results._observations.clear()
        self.circuit_breaker_state._values.clear()
        self.active_connections._values.clear()
        self.errors_total._values.clear()


# =============================================================================
# Metrics Recorder Protocol
# =============================================================================


class MetricsRecorder(Protocol):
    """Protocol for recording metrics."""

    def record_operation(self, metric: OperationMetric) -> None:
        """Record a metric."""


class NoopMetricsRecorder:
    """Default no-op metrics recorder."""

    def record_operation(self, metric: OperationMetric) -> None:
        return None


# =============================================================================
# Global Registry
# =============================================================================


@lru_cache(maxsize=1)
def get_metrics_registry() -> MetricsRegistry:
    """Get the global metrics registry (singleton)."""
    return MetricsRegistry()


__all__ = [
    "Counter",
    "Gauge",
    "Histogram",
    "MetricsRecorder",
    "MetricsRegistry",
    "NoopMetricsRecorder",
    "OperationMetric",
    "get_metrics_registry",
]
