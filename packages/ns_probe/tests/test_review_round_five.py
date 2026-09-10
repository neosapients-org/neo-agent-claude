"""The eight findings from the 19 Aug review.

Grouped here because they are unrelated to each other but share one cause: each
is a place where a fix landed on the surface named in the review and not on the
others of its kind.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

if sys.version_info >= (3, 11):
    import tomllib
else:  # tomllib landed in 3.11; the package floor is 3.10
    import tomli as tomllib


import pytest

from ns_probe.exporter import OTLPExporter
from ns_probe.phi_sanitizer import (
    PHIEngine,
    PHISanitizerConfig,
    SanitizationStrategy,
    _sanitize_attrs_presidio,
    reset_phi_config,
)


@pytest.fixture(autouse=True)
def phi_off():
    reset_phi_config()
    yield
    reset_phi_config()


# ---------------------------------------------------------------------------
# 1. The `phi` extra has to exist, or every install instruction is a dead end.
# ---------------------------------------------------------------------------


def _pyproject():
    path = Path(__file__).resolve().parents[1] / "pyproject.toml"
    return tomllib.loads(path.read_text())


def test_the_phi_extra_is_declared():
    """Three error messages tell operators to run `pip install 'ns_probe[phi]'`.
    Without the extra that installs the base package and warns about an unknown
    name, so presidio stays unimportable and the whole encryption half of the
    subsystem is unreachable."""
    extras = _pyproject()["project"]["optional-dependencies"]
    assert "phi" in extras, "the install command in three error messages is a no-op"


@pytest.mark.parametrize("package", ["presidio-analyzer", "presidio-anonymizer", "cryptography"])
def test_the_phi_extra_pulls_what_the_code_imports(package):
    declared = " ".join(_pyproject()["project"]["optional-dependencies"]["phi"])
    assert package in declared


def test_the_error_messages_point_at_an_extra_that_exists():
    """Keeps the message and the packaging honest about each other."""
    src = (Path(__file__).resolve().parents[1] / "ns_probe").glob("*.py")
    referencing = [p.name for p in src if "ns_probe[phi]" in p.read_text()]
    assert referencing, "expected the install hint to appear in the source"
    assert "phi" in _pyproject()["project"]["optional-dependencies"]


# ---------------------------------------------------------------------------
# 2. repr(value), not repr(round(value, 4)).
# ---------------------------------------------------------------------------


@pytest.fixture
def exporter():
    return OTLPExporter(endpoint="http://localhost:4318/v1/traces")


@pytest.mark.parametrize("value", [0.000021, 0.00001234, 1e-9, 0.00004999])
def test_a_small_float_is_not_flattened_to_zero(exporter, value):
    """round(value, 4) exported 0.000021 as "0.0" and zeroed every rate below
    0.00005. Stringifying was the fix for the processor dropping doubleValue; the
    rounding was a separate lossy decision riding along with it."""
    attr = exporter._attr_to_otlp("cost.total_usd", value)
    assert float(attr["value"]["stringValue"]) == value


def test_a_float_round_trips_exactly(exporter):
    """The property outcome_ledger.py already relies on, now shared."""
    for value in (0.00012, 12345.6789, 0.1 + 0.2):
        attr = exporter._attr_to_otlp("m", value)
        assert float(attr["value"]["stringValue"]) == value


def test_floats_still_travel_as_strings(exporter):
    """The original reason for this branch: the processor reads intValue and
    stringValue and drops doubleValue."""
    attr = exporter._attr_to_otlp("m", 1.5)
    assert "stringValue" in attr["value"]
    assert "doubleValue" not in attr["value"]


# ---------------------------------------------------------------------------
# 4. metrics_json is bounded — and DROPPED over the bound, never clipped.
# ---------------------------------------------------------------------------


def test_metrics_json_over_the_limit_is_dropped_not_clipped():
    """It was the only unbounded ledger field, and the only one that is actually
    persisted — question/answer are clipped AND then dropped by the processor.
    An oversized blob risks the broker's max.message.bytes, which takes the whole
    poll's spans with it.

    Bounded, then: but DROPPED rather than clipped, because the value is a JSON
    document the Outcome Ledger's drawer parses. A clipped one does not parse, so
    the reader sees exactly what it sees for an absent field — except a whole
    record's worth of bytes was stored to say it. The TS writer already refuses
    to emit a truncated document (`buildLedgerMetricsJson` returns undefined)."""
    from ns_probe import outcome_ledger as ol

    attrs = ol.build_attributes(metrics_json={"blob": "x" * 20_000})

    assert f"{ol.LEDGER_PREFIX}metrics_json" not in attrs


def test_metrics_json_honours_an_explicit_limit():
    from ns_probe import outcome_ledger as ol

    attrs = ol.build_attributes(metrics_json={"blob": "x" * 5000}, text_limit=100)

    assert f"{ol.LEDGER_PREFIX}metrics_json" not in attrs


def test_a_metrics_json_that_fits_is_stored_whole_and_parseable():
    """The point of dropping rather than clipping: whatever IS stored parses."""
    from ns_probe import outcome_ledger as ol

    payload = {"blob": "x" * (ol.TEXT_LIMIT - 200)}
    attrs = ol.build_attributes(metrics_json=payload)

    assert json.loads(attrs[f"{ol.LEDGER_PREFIX}metrics_json"]) == payload


def test_a_small_metrics_json_is_untouched():
    from ns_probe import outcome_ledger as ol

    payload = {"knowledge": {"band": "High"}}
    attrs = ol.build_attributes(metrics_json=payload)
    assert json.loads(attrs[f"{ol.LEDGER_PREFIX}metrics_json"]) == payload


# ---------------------------------------------------------------------------
# 6. Presidio attaches the DEK only when it produced ciphertext.
# ---------------------------------------------------------------------------


class _StubKey:
    def encrypt(self, plaintext: str) -> str:
        return f"ns_enc_v1:stub:{plaintext[::-1]}"

    def get_encrypted_dek(self) -> str:
        return "stub-dek"


def _presidio_cfg():
    return PHISanitizerConfig(
        enabled=True, engine=PHIEngine.PRESIDIO, strategy=SanitizationStrategy.ENCRYPT
    )


def _fake_presidio(monkeypatch, transform):
    """Stand in for ns_probe.presidio_redactor.

    _sanitize_attrs_presidio imports it inside the function, so the module has to
    be replaced in sys.modules rather than patched on phi_sanitizer. presidio
    itself is not installed here, and it is not what these two assert.
    """
    import sys
    import types

    import ns_probe.phi_sanitizer as ps

    fake = types.ModuleType("ns_probe.presidio_redactor")
    fake.sanitize_payload = lambda value, redactor: transform(value)
    fake.PresidioRedactor = type(
        "PresidioRedactor", (), {"get_instance": staticmethod(lambda **kw: object())}
    )
    fake.EncryptOperator = type(
        "EncryptOperator", (), {"set_crypto_engine": classmethod(lambda cls, e: None)}
    )
    monkeypatch.setitem(sys.modules, "ns_probe.presidio_redactor", fake)
    monkeypatch.setattr(ps, "get_phi_config", _presidio_cfg)
    monkeypatch.setattr(ps, "_resolve_crypto_engine", lambda cfg: _StubKey())


def test_presidio_attaches_no_dek_when_nothing_was_encrypted(monkeypatch):
    """Matches the keys engine. A DEK on a span that encrypted nothing decrypts
    nothing and makes the attribute useless as a signal for "has ciphertext"."""
    _fake_presidio(monkeypatch, lambda value: value)

    out = _sanitize_attrs_presidio({"step": "retrieve"})
    assert "ns.crypto.encrypted_dek" not in out


def test_presidio_attaches_the_dek_when_it_did_encrypt(monkeypatch):
    _fake_presidio(monkeypatch, lambda value: f"ns_enc_v1:stub:{value}")

    out = _sanitize_attrs_presidio({"patient": "Jane Doe"})
    assert out["ns.crypto.encrypted_dek"] == "stub-dek"


# ---------------------------------------------------------------------------
# 8. The ciphertext and the DEK come from the same key.
# ---------------------------------------------------------------------------


def test_a_pinned_key_holds_its_dek_still():
    """The pinning property itself, with no dependency on the cryptography
    package — PinnedKey is a value object, and the bug was that the DEK was
    re-read from the engine instead of coming from the key that encrypted."""
    from ns_probe.crypto import PinnedKey

    class _FakeAESGCM:
        def encrypt(self, iv, data, aad):
            return b"ct" + data

    key = PinnedKey(aesgcm=_FakeAESGCM(), encrypted_dek="dek-1")

    assert key.encrypt("secret").startswith("ns_enc_v1:")
    assert key.encrypted_dek == "dek-1"
    assert key.get_encrypted_dek() == "dek-1"
    # Same object, same key, however many values it encrypts.
    key.encrypt("another")
    assert key.encrypted_dek == "dek-1"


def test_a_pinned_key_survives_a_rotation():
    """encrypt() and get_encrypted_dek() each called _ensure_dek(), so a rotation
    between them paired new-key DEK with old-key ciphertext — unrecoverable.
    snapshot() takes the key once and answers both from it."""
    crypto = pytest.importorskip("ns_probe.crypto", reason="cryptography not installed")
    if not getattr(crypto, "_CRYPTO_AVAILABLE", False):
        pytest.skip("cryptography not installed")

    crypto.reset_crypto_engine()
    engine = crypto.init_crypto_engine()
    try:
        key = engine.snapshot()
        ciphertext = key.encrypt("secret")
        dek_before = key.encrypted_dek

        # Force the engine to rotate underneath the pinned key.
        engine._dek_created_at = 0.0
        engine.get_encrypted_dek()

        assert key.encrypted_dek == dek_before, "the pinned key moved"
        assert key.encrypt("secret") != ciphertext  # fresh IV, same key
        assert engine.get_encrypted_dek() != dek_before, "engine did not rotate"
    finally:
        crypto.reset_crypto_engine()


def test_a_pinned_key_reports_the_dek_that_matches_its_ciphertext():
    crypto = pytest.importorskip("ns_probe.crypto")
    if not getattr(crypto, "_CRYPTO_AVAILABLE", False):
        pytest.skip("cryptography not installed")

    crypto.reset_crypto_engine()
    engine = crypto.init_crypto_engine()
    try:
        key = engine.snapshot()
        assert key.encrypt("x").startswith("ns_enc_v1:")
        assert key.get_encrypted_dek() == key.encrypted_dek
    finally:
        crypto.reset_crypto_engine()
