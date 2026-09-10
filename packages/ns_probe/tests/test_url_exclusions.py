"""Which outgoing HTTP calls the instrumentors decline to trace.

The list used to name one deployment's collector directly, which meant every
other installation traced its own telemetry traffic: a span describing the
request that ships spans, produced once per batch, growing with the traffic it
describes. The collector in use is now derived from the configured endpoint, so
pointing ns_probe somewhere is enough to keep that somewhere out of the traces.
"""

from __future__ import annotations

import pytest

from ns_probe import instrumentors
from ns_probe.config import configure


@pytest.fixture(autouse=True)
def _clear_cache(monkeypatch):
    monkeypatch.delenv(instrumentors.EXCLUDED_URLS_ENV, raising=False)
    instrumentors._exclusion_cache = None
    yield
    instrumentors._exclusion_cache = None


def test_the_configured_collector_is_not_traced():
    """The whole point: configuring an endpoint excludes it, with no second setting."""
    configure(endpoint="https://collector.example.com/v1/traces")

    assert instrumentors._should_skip_url("https://collector.example.com/v1/traces")


def test_the_collector_is_excluded_by_host_not_by_path():
    """A collector on a non-standard path is still the collector."""
    configure(endpoint="https://otel.example.com:9999/ingest")

    assert instrumentors._should_skip_url("https://otel.example.com:9999/anything")


def test_ordinary_traffic_is_still_traced():
    configure(endpoint="https://collector.example.com/v1/traces")

    assert not instrumentors._should_skip_url("https://api.stripe.com/v1/charges")
    assert not instrumentors._should_skip_url("https://example.com/health")


def test_otlp_ports_are_excluded_on_any_host():
    """The old list enumerated hosts one at a time and missed every other one."""
    configure(endpoint="https://collector.example.com/v1/traces")

    assert instrumentors._should_skip_url("http://some-sidecar:4318/v1/traces")
    assert instrumentors._should_skip_url("http://10.0.0.5:4317/")


def test_model_apis_are_left_to_their_own_instrumentors():
    """Tracing these as plain HTTP would record every model call twice, once
    without the tokens that make it cost anything."""
    configure(endpoint="https://collector.example.com/v1/traces")

    assert instrumentors._should_skip_url("https://api.openai.com/v1/chat/completions")
    assert instrumentors._should_skip_url("https://api.anthropic.com/v1/messages")


def test_extra_exclusions_come_from_the_environment(monkeypatch):
    """For an internal observability API that should stay out of what it serves."""
    configure(endpoint="https://collector.example.com/v1/traces")
    monkeypatch.setenv(instrumentors.EXCLUDED_URLS_ENV, "internal-api, neo_observe")
    instrumentors._exclusion_cache = None

    assert instrumentors._should_skip_url("https://internal-api.svc/query")
    assert instrumentors._should_skip_url("http://neo_observe:8080/traces")
    assert not instrumentors._should_skip_url("https://example.com/query")


def test_the_cache_follows_a_reconfigured_endpoint():
    """_should_skip_url runs per request and caches, so it has to notice
    configure() being called again rather than pinning the first endpoint."""
    configure(endpoint="https://first.example.com/v1/traces")
    assert instrumentors._should_skip_url("https://first.example.com/x")

    configure(endpoint="https://second.example.com/v1/traces")
    assert instrumentors._should_skip_url("https://second.example.com/x")
    assert not instrumentors._should_skip_url("https://first.example.com/x")


def test_no_deployment_is_named_in_the_source():
    """A company hostname hardcoded here is meaningless to every other
    installation, and this package is going out publicly."""
    for entry in instrumentors._EXCLUDED_URL_SUBSTRINGS:
        assert "neosapient" not in entry.lower()
        assert "neo_observe" not in entry.lower()
