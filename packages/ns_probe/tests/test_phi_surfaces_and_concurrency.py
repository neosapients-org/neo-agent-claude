"""The three findings from PR #865's review of the agent SDK.

Two are about a sanitizer that did not cover everything it was believed to, and
one is about a span stack that was not as task-local as it looked. All three fail
SILENTLY — no exception, no log — which is why each gets a test that would have
caught it rather than a comment saying it is fixed.

1. `NS_PROBE_PHI_ENABLED=true` sanitized `span.attributes` only. Event and link
   attributes were serialized verbatim, and events are where `record_exception`
   puts messages and stack traces.
2. `configure_phi(engine="keys", strategy="encrypt")` was accepted, built a
   CryptoEngine, then redacted — destroying the data it was asked to encrypt.
3. The span stack was a mutable list in a ContextVar. ContextVar isolates
   rebinds, not mutations, so sibling tasks shared one list.
"""

from __future__ import annotations

import asyncio

import pytest

from ns_probe.exporter import OTLPExporter
from ns_probe.phi_sanitizer import (
    configure_phi,
    reset_phi_config,
    sanitize_attributes,
)
from ns_probe.span import Span, SpanLink
from ns_probe.tracer import Tracer


@pytest.fixture(autouse=True)
def phi_off():
    """Never let one test's PHI config leak into the next."""
    reset_phi_config()
    yield
    reset_phi_config()


@pytest.fixture
def exporter():
    return OTLPExporter(endpoint="http://localhost:4318/v1/traces")


def _keys_redaction(strategy="redact"):
    configure_phi(
        enabled=True,
        engine="keys",
        sensitive_keys=["patient", "ssn", "exception.message", "exception.stacktrace"],
        strategy=strategy,
    )


def _attr(otlp_attrs, key):
    for a in otlp_attrs:
        if a["key"] == key:
            return a["value"]
    raise AssertionError(f"{key} not in {[a['key'] for a in otlp_attrs]}")


# ---------------------------------------------------------------------------
# 1. Every attribute-bearing surface is sanitized, not just span.attributes.
# ---------------------------------------------------------------------------


def test_event_attributes_are_sanitized(exporter):
    """The reported leak: events were serialized straight from e.attributes."""
    _keys_redaction()

    span = Span(name="agent.turn")
    span.add_event("snapshot", {"patient": "Jane Doe", "step": "retrieve"})
    span.end()

    otlp = exporter._span_to_otlp(span)
    attrs = otlp["events"][0]["attributes"]

    assert _attr(attrs, "patient")["stringValue"] == "[REDACTED]"
    # A non-sensitive key on the same event must survive untouched.
    assert _attr(attrs, "step")["stringValue"] == "retrieve"


def test_link_attributes_are_sanitized(exporter):
    """Links bypassed the sanitizer too."""
    _keys_redaction()

    span = Span(name="agent.turn")
    span.links.append(
        SpanLink(trace_id="a" * 32, span_id="b" * 16, attributes={"ssn": "123-45-6789"})
    )
    span.end()

    otlp = exporter._span_to_otlp(span)
    assert _attr(otlp["links"][0]["attributes"], "ssn")["stringValue"] == "[REDACTED]"


def test_an_exception_event_cannot_carry_phi_to_the_collector(exporter):
    """record_exception puts the message and stack trace on an event, so an
    exception carrying patient data was a leak on the default path."""
    _keys_redaction()

    span = Span(name="agent.turn")
    try:
        raise ValueError("patient Jane Doe not found")
    except ValueError as exc:
        span.record_exception(exc)
    span.end()

    payload = str(exporter._span_to_otlp(span))
    assert "Jane Doe" not in payload, payload


def test_no_raw_phi_survives_anywhere_in_the_serialized_span(exporter):
    """The property that actually matters, asserted over the whole payload
    rather than per surface — a new attribute surface added to Span later should
    fail this even if nobody updates the per-surface tests above."""
    _keys_redaction()

    span = Span(name="agent.turn", attributes={"patient": "Jane Doe"})
    span.add_event("snapshot", {"patient": "Jane Doe"})
    span.links.append(
        SpanLink(trace_id="a" * 32, span_id="b" * 16, attributes={"patient": "Jane Doe"})
    )
    span.set_status(2, "patient Jane Doe not found")
    span.end()

    assert "Jane Doe" not in str(exporter._span_to_otlp(span))


def test_an_exception_raised_through_start_span_leaks_nothing(exporter):
    """The REAL exception path, not a hand-built span.

    `start_span` catches, calls record_exception, then set_status(ERROR, str(e)).
    That second call duplicates the message into `status_message`, which the
    exporter copied verbatim — so covering only the event left the same text
    leaving through the status. Building the span by hand never sets that field,
    which is exactly why the earlier version of this test passed while the leak
    was open.
    """
    _keys_redaction()
    tracer = Tracer("test")

    captured = {}
    with pytest.raises(ValueError):
        with tracer.start_span("agent.turn", root=True) as span:
            captured["span"] = span
            raise ValueError("patient Jane Doe not found")

    payload = str(exporter._span_to_otlp(captured["span"]))
    assert "Jane Doe" not in payload, payload


def test_the_status_message_is_suppressed_under_the_keys_engine(exporter):
    """The keys engine matches attribute KEYS; a bare message has none, so there
    is no way to tell a safe message from one quoting a record. Fail closed."""
    _keys_redaction()

    span = Span(name="agent.turn")
    span.set_status(2, "patient Jane Doe not found")
    span.end()

    assert exporter._span_to_otlp(span)["status"]["message"] == "[REDACTED]"


#
# Presidio is an optional dependency, so these stub `_sanitize_attrs_presidio`
# rather than installing the NLP stack. The bug is that the caller DROPPED the
# DEK the presidio path returns, which is testable without running detection.


def _presidio_encrypting(monkeypatch):
    """Make the presidio path behave as it does under strategy="encrypt"."""
    import ns_probe.phi_sanitizer as ps

    monkeypatch.setattr(
        ps,
        "get_phi_config",
        lambda: ps.PHISanitizerConfig(
            enabled=True,
            engine=ps.PHIEngine.PRESIDIO,
        ),
    )
    monkeypatch.setattr(
        ps,
        "_sanitize_attrs_presidio",
        lambda attrs: {
            **{k: f"ns_enc_v1:stub:{v}" for k, v in attrs.items()},
            "ns.crypto.encrypted_dek": "stub-dek",
        },
    )


def test_a_status_only_span_keeps_the_dek_for_its_ciphertext(exporter, monkeypatch):
    """The reported bug. `sanitize_free_text` took the message out of the
    presidio result with `.get("message")` and discarded the DEK beside it, so a
    span carrying nothing but a status exported ciphertext no consumer could
    decrypt — there is no other attribute to hold the key."""
    _presidio_encrypting(monkeypatch)

    span = Span(name="agent.turn")
    span.set_status(2, "patient Jane Doe not found")
    span.end()

    otlp = exporter._span_to_otlp(span)

    assert otlp["status"]["message"].startswith("ns_enc_v1:")
    assert _attr(otlp["attributes"], "ns.crypto.encrypted_dek")["stringValue"] == (
        "stub-dek"
    ), "ciphertext exported without the key that decrypts it"


def test_the_dek_is_not_duplicated_when_attributes_already_carry_one(exporter, monkeypatch):
    """Both come from the same engine, so they are the same key — one is enough."""
    _presidio_encrypting(monkeypatch)

    span = Span(name="agent.turn", attributes={"patient": "Jane Doe"})
    span.set_status(2, "patient Jane Doe not found")
    span.end()

    keys = [a["key"] for a in exporter._span_to_otlp(span)["attributes"]]
    assert keys.count("ns.crypto.encrypted_dek") == 1


def test_no_dek_is_added_when_the_status_was_not_encrypted(exporter):
    """The keys engine suppresses the message instead of encrypting it, so there
    is no key to advertise."""
    _keys_redaction()

    span = Span(name="agent.turn")
    span.set_status(2, "patient Jane Doe not found")
    span.end()

    keys = [a["key"] for a in exporter._span_to_otlp(span)["attributes"]]
    assert "ns.crypto.encrypted_dek" not in keys


def test_the_status_message_survives_when_phi_is_disabled(exporter):
    """Suppression must be scoped to the policy — debuggability is the default."""
    span = Span(name="agent.turn")
    span.set_status(2, "boom")
    span.end()

    assert exporter._span_to_otlp(span)["status"]["message"] == "boom"


def test_sanitization_is_still_a_no_op_when_disabled(exporter):
    """Cost of the fix must be zero when the flag is off."""
    span = Span(name="agent.turn")
    span.add_event("snapshot", {"patient": "Jane Doe"})
    span.end()

    attrs = exporter._span_to_otlp(span)["events"][0]["attributes"]
    assert _attr(attrs, "patient")["stringValue"] == "Jane Doe"


# ---------------------------------------------------------------------------
# 2. The keys engine honours ENCRYPT instead of quietly redacting.
# ---------------------------------------------------------------------------


#
# These drive `_sanitize_attrs_keys` with a stub engine rather than going through
# `configure_phi`, which raises ImportError for the encrypt strategy unless the
# optional `cryptography` package is installed. The bug was in the keys path
# ignoring the engine, not in key management, so testing that seam directly keeps
# the coverage unconditional — a skip here would be the same
# success-by-absence that hid the roundtrip contract test.


class _StubCrypto:
    """Stands in for CryptoEngine. AES-GCM is not what is under test here."""

    def encrypt(self, plaintext: str) -> str:
        return f"ns_enc_v1:stub:{plaintext[::-1]}"

    def get_encrypted_dek(self) -> str:
        return "stub-dek"


def _encrypt_cfg():
    from ns_probe.phi_sanitizer import PHISanitizerConfig, PHIEngine, SanitizationStrategy

    return PHISanitizerConfig(
        enabled=True,
        engine=PHIEngine.KEYS,
        sensitive_keys=frozenset({"ssn"}),
        strategy=SanitizationStrategy.ENCRYPT,
    )


def test_the_keys_engine_encrypts_rather_than_redacting(monkeypatch):
    """The reported bug: this path treated every non-hash strategy as redaction,
    so the documented encryption mode destroyed the data instead of protecting
    it, and emitted no ciphertext and no DEK."""
    import ns_probe.crypto as crypto
    from ns_probe.phi_sanitizer import _sanitize_attrs_keys

    monkeypatch.setattr(crypto, "get_crypto_engine", lambda: _StubCrypto())

    out = _sanitize_attrs_keys({"ssn": "123-45-6789", "step": "retrieve"}, _encrypt_cfg())

    assert out["ssn"] != "[REDACTED]", "encrypt silently fell back to redaction"
    assert out["ssn"] != "123-45-6789", "plaintext survived"
    assert out["ssn"].startswith("ns_enc_v1:")
    assert (
        out["ns.crypto.encrypted_dek"] == "stub-dek"
    ), "ciphertext without a DEK is unreadable — data loss wearing a different hat"
    assert out["step"] == "retrieve"


def test_encrypt_without_an_engine_redacts_rather_than_leaking(monkeypatch):
    """Degrading to redaction is safe and the caller is warned. Leaking plaintext
    because the engine is missing would not be."""
    import ns_probe.crypto as crypto
    from ns_probe.phi_sanitizer import _sanitize_attrs_keys

    monkeypatch.setattr(crypto, "get_crypto_engine", lambda: None)

    out = _sanitize_attrs_keys({"ssn": "123-45-6789"}, _encrypt_cfg())

    assert out["ssn"] == "[REDACTED]"
    assert "ns.crypto.encrypted_dek" not in out


def test_json_embedded_matches_are_encrypted_too(monkeypatch):
    """A match nested in a JSON string is the same match as a top-level one.
    `_sanitize_nested` dropped the engine, so ENCRYPT silently redacted nested
    values while top-level values encrypted — destroying the nested data."""
    import json as _json

    import ns_probe.crypto as crypto
    from ns_probe.phi_sanitizer import _sanitize_attrs_keys

    monkeypatch.setattr(crypto, "get_crypto_engine", lambda: _StubCrypto())

    out = _sanitize_attrs_keys(
        {"payload": _json.dumps({"ssn": "123-45-6789", "step": "retrieve"})},
        _encrypt_cfg(),
    )
    inner = _json.loads(out["payload"])

    assert inner["ssn"] != "[REDACTED]", "nested match silently redacted"
    assert inner["ssn"] != "123-45-6789", "nested plaintext survived"
    assert inner["ssn"].startswith("ns_enc_v1:")
    assert inner["step"] == "retrieve"
    assert out["ns.crypto.encrypted_dek"] == "stub-dek"


def test_matches_nested_in_lists_and_deeper_objects_are_encrypted(monkeypatch):
    """`_sanitize_nested` recurses through lists as well as dicts, so the list
    branch needs the engine too — a match two levels down inside an array is the
    same match as one at the top."""
    import json as _json

    import ns_probe.crypto as crypto
    from ns_probe.phi_sanitizer import _sanitize_attrs_keys

    monkeypatch.setattr(crypto, "get_crypto_engine", lambda: _StubCrypto())

    payload = {
        "patients": [
            {"ssn": "111-11-1111", "step": "a"},
            {"nested": {"ssn": "222-22-2222"}},
        ]
    }
    out = _sanitize_attrs_keys({"payload": _json.dumps(payload)}, _encrypt_cfg())
    inner = _json.loads(out["payload"])

    first = inner["patients"][0]["ssn"]
    deep = inner["patients"][1]["nested"]["ssn"]

    for value, where in ((first, "inside a list"), (deep, "two levels down")):
        assert value.startswith("ns_enc_v1:"), f"{where}: not encrypted ({value})"
        assert "-" not in value.split(":", 2)[-1] or value != "111-11-1111"

    assert "111-11-1111" not in out["payload"]
    assert "222-22-2222" not in out["payload"]
    assert inner["patients"][0]["step"] == "a"
    assert out["ns.crypto.encrypted_dek"] == "stub-dek"


def test_the_dek_is_attached_only_when_ciphertext_was_produced(monkeypatch):
    """A DEK on a span that encrypted nothing decrypts nothing — it just implies
    there is something to decrypt."""
    import ns_probe.crypto as crypto
    from ns_probe.phi_sanitizer import _sanitize_attrs_keys

    monkeypatch.setattr(crypto, "get_crypto_engine", lambda: _StubCrypto())

    out = _sanitize_attrs_keys({"step": "retrieve"}, _encrypt_cfg())

    assert out["step"] == "retrieve"
    assert "ns.crypto.encrypted_dek" not in out


def test_an_encryption_failure_redacts_instead_of_leaking(monkeypatch):
    """If encrypt() raises, the plaintext must not fall through to the wire."""
    import ns_probe.crypto as crypto
    from ns_probe.phi_sanitizer import _sanitize_attrs_keys

    class _Broken(_StubCrypto):
        def encrypt(self, plaintext: str) -> str:
            raise RuntimeError("KMS unreachable")

    monkeypatch.setattr(crypto, "get_crypto_engine", lambda: _Broken())

    out = _sanitize_attrs_keys({"ssn": "123-45-6789"}, _encrypt_cfg())
    assert out["ssn"] == "[REDACTED]"


def test_hash_strategy_is_unaffected():
    _keys_redaction(strategy="hash")
    out = sanitize_attributes({"ssn": "123-45-6789"})
    assert out["ssn"].startswith("sha256:")


# ---------------------------------------------------------------------------
# 3. Concurrent roots, completing out of order.
# ---------------------------------------------------------------------------


def _warm_the_stack():
    """Force the span-stack contextvar to exist before the tasks fork.

    This is the precondition for the whole bug: with a mutable list, the list
    object is created once and then inherited by reference. A test that skips
    this can pass while the bug is fully present.
    """
    with Tracer("warmup").start_span("precondition.warmup"):
        pass


@pytest.mark.asyncio
async def test_a_span_in_one_task_does_not_parent_into_another_tasks_trace():
    """The fusion itself.

    A non-root span consults the stack when no span context is set in its own
    task. With one shared list, a task that has opened nothing still saw a
    sibling's frame there and parented to it — joining two unrelated requests
    into a single trace. This is the assertion that fails on the mutable-list
    implementation.
    """
    _warm_the_stack()
    tracer = Tracer("test")
    holder_open = asyncio.Event()
    observed: dict = {}

    async def holder():
        with tracer.start_span("root.holder", root=True) as root:
            observed["holder_trace"] = root.trace_id
            holder_open.set()
            await asyncio.sleep(0.02)

    async def stranger():
        await holder_open.wait()
        # Not root: it is entitled to a parent if its OWN context has one. It
        # does not, so it must start its own trace rather than adopt holder's.
        with tracer.start_span("stranger.work") as span:
            observed["stranger_trace"] = span.trace_id
            observed["stranger_parent"] = span.parent_span_id

    await asyncio.gather(holder(), stranger())

    assert observed["stranger_parent"] is None, "span adopted a parent from another task's stack"
    assert (
        observed["stranger_trace"] != observed["holder_trace"]
    ), "two unrelated requests were fused into one trace"


@pytest.mark.asyncio
async def test_concurrent_roots_completing_out_of_order_stay_separate():
    """A exits BEFORE B, the case the identity-guarded pop mishandled: A was not
    on top, so the pop was skipped and A stayed on the stack under B."""
    _warm_the_stack()
    tracer = Tracer("test")
    started = asyncio.Event()
    traces: dict = {}

    async def slow_root():
        with tracer.start_span("root.b", root=True) as b:
            started.set()
            await asyncio.sleep(0.02)
            with tracer.start_span("child.b") as cb:
                traces["b"] = (b.trace_id, cb.trace_id, cb.parent_span_id, b.span_id)

    async def fast_root():
        await started.wait()
        with tracer.start_span("root.a", root=True) as a:
            with tracer.start_span("child.a") as ca:
                traces["a"] = (a.trace_id, ca.trace_id, ca.parent_span_id, a.span_id)

    await asyncio.gather(slow_root(), fast_root())

    for name in ("a", "b"):
        root_trace, child_trace, child_parent, root_span = traces[name]
        assert child_trace == root_trace, f"{name}: child joined another trace"
        assert child_parent == root_span, f"{name}: child parented outside its root"

    assert traces["a"][0] != traces["b"][0], "two roots shared one trace"


@pytest.mark.asyncio
async def test_the_stack_is_empty_again_after_concurrent_turns():
    """A leftover frame is what later spans attach to. Nothing may survive."""
    from ns_probe.tracer import _get_span_stack

    _warm_the_stack()
    tracer = Tracer("test")

    async def root(delay):
        with tracer.start_span(f"root.{delay}", root=True):
            await asyncio.sleep(delay)

    await asyncio.gather(root(0.02), root(0.0), root(0.01))
    assert _get_span_stack() == ()


@pytest.mark.asyncio
async def test_a_sibling_cannot_see_another_tasks_span():
    """The direct assertion that the stack is task-local. With a shared mutable
    list each task saw the other's frame."""
    from ns_probe.tracer import _get_span_stack

    _warm_the_stack()
    tracer = Tracer("test")
    depths: dict = {}
    inside = asyncio.Event()

    async def holder():
        with tracer.start_span("root.holder", root=True):
            inside.set()
            await asyncio.sleep(0.02)

    async def observer():
        await inside.wait()
        depths["observer"] = len(_get_span_stack())

    await asyncio.gather(holder(), observer())
    assert depths["observer"] == 0


def test_the_stack_is_immutable():
    """A tuple is what makes the isolation structural rather than a convention —
    a future caller cannot reintroduce the bug with an .append()."""
    from ns_probe.tracer import _get_span_stack

    tracer = Tracer("test")
    with tracer.start_span("root", root=True):
        assert isinstance(_get_span_stack(), tuple)


def test_nested_spans_still_parent_correctly():
    """The fix must not break the ordinary sequential case."""
    tracer = Tracer("test")
    with tracer.start_span("outer", root=True) as outer:
        with tracer.start_span("middle") as middle:
            with tracer.start_span("inner") as inner:
                assert middle.parent_span_id == outer.span_id
                assert inner.parent_span_id == middle.span_id
                assert inner.trace_id == outer.trace_id
