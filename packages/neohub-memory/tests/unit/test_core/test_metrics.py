"""Tests for metrics and observability."""

import pytest

from neo_memory_hub.core.metrics import (
    Counter,
    Gauge,
    Histogram,
    MetricsRegistry,
    OperationMetric,
    get_metrics_registry,
)


class TestCounter:
    """Tests for Counter metric."""

    def test_counter_increment(self):
        """Test counter increment."""
        counter = Counter(
            name="test_counter",
            description="Test counter",
            labels=["operation"],
        )
        counter.inc(operation="store")
        counter.inc(operation="store")
        counter.inc(operation="retrieve")

        assert counter.get(operation="store") == 2
        assert counter.get(operation="retrieve") == 1
        assert counter.get(operation="unknown") == 0

    def test_counter_increment_by_value(self):
        """Test counter increment by specific value."""
        counter = Counter(name="test", description="Test", labels=[])
        counter.inc(5)
        counter.inc(3)
        assert counter.get() == 8

    def test_counter_prometheus_format(self):
        """Test Prometheus export format."""
        counter = Counter(
            name="requests_total",
            description="Total requests",
            labels=["status"],
        )
        counter.inc(status="success")
        counter.inc(status="success")
        counter.inc(status="error")

        output = counter.to_prometheus()
        assert "# HELP requests_total Total requests" in output
        assert "# TYPE requests_total counter" in output
        assert 'requests_total{status="success"}' in output


class TestHistogram:
    """Tests for Histogram metric."""

    def test_histogram_observe(self):
        """Test histogram observations."""
        histogram = Histogram(
            name="latency",
            description="Request latency",
            labels=["endpoint"],
            buckets=[0.1, 0.5, 1.0],
        )
        histogram.observe(0.05, endpoint="store")
        histogram.observe(0.3, endpoint="store")
        histogram.observe(0.8, endpoint="store")

        assert histogram.get_count(endpoint="store") == 3
        assert histogram.get_sum(endpoint="store") == pytest.approx(1.15)

    def test_histogram_prometheus_format(self):
        """Test Prometheus export format."""
        histogram = Histogram(
            name="duration_seconds",
            description="Duration",
            labels=[],
            buckets=[0.1, 0.5, 1.0],
        )
        histogram.observe(0.05)
        histogram.observe(0.3)
        histogram.observe(2.0)

        output = histogram.to_prometheus()
        assert "# TYPE duration_seconds histogram" in output
        assert "duration_seconds_bucket" in output
        assert 'le="+Inf"' in output
        assert "duration_seconds_sum" in output
        assert "duration_seconds_count" in output


class TestGauge:
    """Tests for Gauge metric."""

    def test_gauge_set(self):
        """Test gauge set."""
        gauge = Gauge(
            name="connections",
            description="Active connections",
            labels=["backend"],
        )
        gauge.set(10, backend="postgres")
        assert gauge.get(backend="postgres") == 10

        gauge.set(5, backend="postgres")
        assert gauge.get(backend="postgres") == 5

    def test_gauge_inc_dec(self):
        """Test gauge increment and decrement."""
        gauge = Gauge(name="active", description="Active", labels=[])
        gauge.inc()
        gauge.inc()
        gauge.inc(5)
        assert gauge.get() == 7

        gauge.dec(3)
        assert gauge.get() == 4

    def test_gauge_prometheus_format(self):
        """Test Prometheus export format."""
        gauge = Gauge(
            name="circuit_state",
            description="Circuit breaker state",
            labels=["backend"],
        )
        gauge.set(0, backend="mem0")

        output = gauge.to_prometheus()
        assert "# TYPE circuit_state gauge" in output
        assert 'circuit_state{backend="mem0"}' in output


class TestMetricsRegistry:
    """Tests for MetricsRegistry."""

    def test_registry_track_operation(self):
        """Test operation tracking context manager."""
        registry = MetricsRegistry()

        with registry.track_operation("store", tenant_id="acme"):
            pass  # Simulate operation

        assert (
            registry.operations_total.get(operation="store", status="success", tenant_id="acme")
            == 1
        )
        assert registry.operation_duration.get_count(operation="store", tenant_id="acme") == 1

    def test_registry_track_operation_error(self):
        """Test operation tracking with error."""
        registry = MetricsRegistry()

        with pytest.raises(ValueError):
            with registry.track_operation("store", tenant_id="acme"):
                raise ValueError("Test error")

        assert (
            registry.operations_total.get(operation="store", status="error", tenant_id="acme") == 1
        )
        assert (
            registry.errors_total.get(operation="store", error_type="ValueError", tenant_id="acme")
            == 1
        )

    def test_registry_record_operation(self):
        """Test recording operation metric."""
        registry = MetricsRegistry()

        metric = OperationMetric(
            name="retrieve",
            duration_ms=150.0,
            success=True,
            metadata={"tenant_id": "corp"},
        )
        registry.record_operation(metric)

        assert (
            registry.operations_total.get(operation="retrieve", status="success", tenant_id="corp")
            == 1
        )
        # Duration is converted from ms to seconds
        assert registry.operation_duration.get_count(operation="retrieve", tenant_id="corp") == 1

    def test_registry_prometheus_export(self):
        """Test Prometheus export."""
        registry = MetricsRegistry()
        registry.operations_total.inc(operation="store", status="success", tenant_id="test")
        registry.store_filtered.inc(reason="too_short", tenant_id="test")

        output = registry.export_prometheus()
        assert "neohub_memory_operations_total" in output
        assert "neohub_memory_store_filtered_total" in output

    def test_registry_reset(self):
        """Test registry reset."""
        registry = MetricsRegistry()
        registry.operations_total.inc(operation="store", status="success", tenant_id="test")

        registry.reset()

        assert (
            registry.operations_total.get(operation="store", status="success", tenant_id="test")
            == 0
        )


class TestGlobalRegistry:
    """Tests for global registry singleton."""

    def test_singleton(self):
        """Test that get_metrics_registry returns singleton."""
        registry1 = get_metrics_registry()
        registry2 = get_metrics_registry()
        assert registry1 is registry2

    def test_global_registry_usage(self):
        """Test using global registry."""
        registry = get_metrics_registry()
        registry.reset()  # Clear any previous state

        registry.operations_total.inc(operation="test", status="success", tenant_id="global")
        assert (
            registry.operations_total.get(operation="test", status="success", tenant_id="global")
            == 1
        )
