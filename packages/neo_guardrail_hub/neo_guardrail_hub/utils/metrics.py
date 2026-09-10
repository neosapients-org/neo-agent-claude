"""Metrics collection utilities for Neo Guardrail Hub.

This module provides utilities for collecting and tracking
performance metrics from guardrail executions.
"""

import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class MetricPoint:
    """A single metric data point."""

    name: str
    value: float
    timestamp: float
    tags: Dict[str, str] = field(default_factory=dict)


@dataclass
class MetricsSummary:
    """Summary statistics for a metric."""

    name: str
    count: int
    total: float
    min: float
    max: float
    avg: float

    @property
    def p50(self) -> float:
        """Placeholder for 50th percentile."""
        return self.avg

    @property
    def p99(self) -> float:
        """Placeholder for 99th percentile."""
        return self.max


class MetricsCollector:
    """Collect and aggregate metrics from guardrail executions.

    Provides methods for recording metrics and generating
    summary statistics.

    Example:
        collector = MetricsCollector()
        collector.record("guardrail.latency", 45.2, {"guardrail": "prompt_injection"})
        summary = collector.get_summary("guardrail.latency")
    """

    def __init__(self, max_points: int = 10000) -> None:
        """Initialize the collector.

        Args:
            max_points: Maximum points to keep per metric
        """
        self.max_points = max_points
        self._metrics: Dict[str, List[MetricPoint]] = defaultdict(list)

    def record(
        self,
        name: str,
        value: float,
        tags: Optional[Dict[str, str]] = None,
    ) -> None:
        """Record a metric value.

        Args:
            name: Metric name
            value: Metric value
            tags: Optional tags for filtering
        """
        point = MetricPoint(
            name=name,
            value=value,
            timestamp=time.time(),
            tags=tags or {},
        )

        points = self._metrics[name]
        points.append(point)

        # Trim if over limit
        if len(points) > self.max_points:
            self._metrics[name] = points[-self.max_points :]

    def record_latency(
        self,
        guardrail_name: str,
        latency_ms: float,
        passed: bool,
    ) -> None:
        """Record guardrail latency metric.

        Args:
            guardrail_name: Name of the guardrail
            latency_ms: Latency in milliseconds
            passed: Whether the check passed
        """
        self.record(
            "guardrail.latency",
            latency_ms,
            {"guardrail": guardrail_name, "passed": str(passed)},
        )

    def record_check(
        self,
        guardrail_name: str,
        passed: bool,
        risk_score: float,
    ) -> None:
        """Record a guardrail check result.

        Args:
            guardrail_name: Name of the guardrail
            passed: Whether the check passed
            risk_score: Risk score from 0-1
        """
        self.record(
            "guardrail.check",
            1.0 if passed else 0.0,
            {"guardrail": guardrail_name},
        )
        self.record(
            "guardrail.risk_score",
            risk_score,
            {"guardrail": guardrail_name},
        )

    def get_summary(self, name: str) -> Optional[MetricsSummary]:
        """Get summary statistics for a metric.

        Args:
            name: Metric name

        Returns:
            MetricsSummary or None if no data
        """
        points = self._metrics.get(name, [])
        if not points:
            return None

        values = [p.value for p in points]
        return MetricsSummary(
            name=name,
            count=len(values),
            total=sum(values),
            min=min(values),
            max=max(values),
            avg=sum(values) / len(values),
        )

    def get_all_summaries(self) -> Dict[str, MetricsSummary]:
        """Get summaries for all metrics.

        Returns:
            Dictionary of metric name to summary
        """
        return {
            name: summary
            for name in self._metrics
            if (summary := self.get_summary(name)) is not None
        }

    def clear(self) -> None:
        """Clear all collected metrics."""
        self._metrics.clear()


class Timer:
    """Context manager for timing code blocks.

    Example:
        with Timer() as t:
            # code to time
        print(f"Took {t.elapsed_ms}ms")
    """

    def __init__(self) -> None:
        """Initialize the timer."""
        self.start_time: float = 0.0
        self.end_time: float = 0.0

    def __enter__(self) -> "Timer":
        """Start the timer."""
        self.start_time = time.monotonic()
        return self

    def __exit__(self, *args: Any) -> None:
        """Stop the timer."""
        self.end_time = time.monotonic()

    @property
    def elapsed(self) -> float:
        """Get elapsed time in seconds."""
        if self.end_time:
            return self.end_time - self.start_time
        return time.monotonic() - self.start_time

    @property
    def elapsed_ms(self) -> float:
        """Get elapsed time in milliseconds."""
        return self.elapsed * 1000


# Global metrics collector
_global_collector: Optional[MetricsCollector] = None


def get_metrics_collector() -> MetricsCollector:
    """Get the global metrics collector."""
    global _global_collector
    if _global_collector is None:
        _global_collector = MetricsCollector()
    return _global_collector
