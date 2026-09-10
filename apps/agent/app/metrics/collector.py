"""In-memory metrics collection."""

import time
from dataclasses import dataclass, field
from collections import deque
from threading import Lock


@dataclass
class QueryMetrics:
    query: str
    total_latency_ms: float
    latency_breakdown: dict
    success: bool
    timestamp: str
    tool_calls: list[dict] = field(default_factory=list)


class MetricsCollector:
    """Thread-safe in-memory metrics collector."""

    def __init__(self, max_entries: int = 1000):
        self._metrics: deque[QueryMetrics] = deque(maxlen=max_entries)
        self._lock = Lock()

    def record(self, metrics: QueryMetrics):
        with self._lock:
            self._metrics.append(metrics)

    def get_summary(self) -> dict:
        with self._lock:
            if not self._metrics:
                return {
                    "total_queries": 0,
                    "avg_latency_ms": 0,
                    "success_rate": 0,
                    "recent_queries": [],
                }

            total = len(self._metrics)
            successful = sum(1 for m in self._metrics if m.success)
            avg_latency = sum(m.total_latency_ms for m in self._metrics) / total

            recent = [
                {
                    "query": m.query[:100],
                    "total_latency_ms": m.total_latency_ms,
                    "success": m.success,
                    "timestamp": m.timestamp,
                }
                for m in list(self._metrics)[-10:]
            ]

            return {
                "total_queries": total,
                "avg_latency_ms": round(avg_latency, 1),
                "success_rate": round(successful / total, 3),
                "recent_queries": recent,
            }


# Singleton
metrics_collector = MetricsCollector()
