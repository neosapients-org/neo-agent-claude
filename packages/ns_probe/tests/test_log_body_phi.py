"""PHI on the log body — the third free-text surface.

A log body is prose with no attribute key, and it was routed through
`sanitize_attributes`, which is the attribute-keyed half of the API. Under the
keys engine that FAILS OPEN: the synthetic "_body" key matches nothing, and the
fallback for an unmatched string is the JSON-embedded scan, which returns
immediately for anything that does not start with { or [. So an ordinary log line
was exported verbatim with PHI filtering on.

Same shape as `Span.status_message`, so it gets the same helper —
`sanitize_free_text`, which fails closed and hands back any DEK.
"""

from __future__ import annotations

import json
import logging

import pytest

from ns_probe.log_exporter import OTLPLogHandler
from ns_probe.phi_sanitizer import configure_phi, reset_phi_config


@pytest.fixture(autouse=True)
def phi_off():
    reset_phi_config()
    yield
    reset_phi_config()


@pytest.fixture
def handler():
    """A handler with no background export thread — __init__ starts one."""
    h = OTLPLogHandler.__new__(OTLPLogHandler)
    logging.Handler.__init__(h)
    h._service_name = "test-svc"
    h._compression = False
    return h


def _record(msg="lookup failed for patient %s", args=("Jane Doe",), level=logging.ERROR):
    return logging.LogRecord("app.lookup", level, "/app/x.py", 10, msg, args, None)


def _first(handler, record):
    payload = json.loads(handler._build_payload([record]))
    return payload["resourceLogs"][0]["scopeLogs"][0]["logRecords"][0]


def _keys_redaction(strategy="redact"):
    configure_phi(
        enabled=True,
        engine="keys",
        sensitive_keys=["patient", "ssn"],
        strategy=strategy,
    )


def _presidio_encrypting(monkeypatch):
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


def _attr(record, key):
    for a in record["attributes"]:
        if a["key"] == key:
            return a["value"]
    return None


# ---------------------------------------------------------------------------
# Fails closed, not open.
# ---------------------------------------------------------------------------


def test_a_prose_log_body_is_not_exported_verbatim(handler):
    """The reported leak. Not JSON, no attribute key — the old path returned it
    untouched."""
    _keys_redaction()

    body = _first(handler, _record())["body"]["stringValue"]

    assert "Jane Doe" not in body, f"PHI exported verbatim: {body!r}"
    assert body == "[REDACTED]"


@pytest.mark.parametrize(
    "msg",
    [
        "patient Jane Doe not found",
        "ssn 123-45-6789 rejected",
        "plain message with no sensitive key at all",
    ],
)
def test_every_body_is_suppressed_under_the_keys_engine(handler, msg):
    """The keys engine cannot inspect free text, so it cannot tell a safe line
    from one quoting a record. All of them are suppressed."""
    _keys_redaction()
    record = logging.LogRecord("app", logging.ERROR, "/app/x.py", 10, msg, None, None)
    assert _first(handler, record)["body"]["stringValue"] == "[REDACTED]"


# ---------------------------------------------------------------------------
# Opt-in masking. Kept OPT-IN because mask_text cannot remove a name, so
# defaulting to it would reopen the leak this file exists to close.
# ---------------------------------------------------------------------------


def test_masking_is_opt_in_and_keeps_the_sentence(handler):
    _keys_redaction()
    configure_phi(mask_free_text=True)

    record = logging.LogRecord(
        "app", logging.ERROR, "/app/x.py", 10, "call failed id 987654321", None, None
    )
    body = _first(handler, record)["body"]["stringValue"]

    assert body == "call failed id [ID]"


def test_masking_does_not_pretend_to_remove_names(handler):
    """Documents the limit rather than hiding it. mask_text matches patterns, and
    a name is not a pattern — which is exactly why suppression is the default."""
    _keys_redaction()
    configure_phi(mask_free_text=True)

    body = _first(handler, _record())["body"]["stringValue"]

    assert (
        "Jane Doe" in body
    ), "if masking ever does remove names, make it the default and delete this test"


def test_a_json_log_body_is_sanitized_by_key_even_when_masking(handler):
    """A JSON body has real keys, so the key scan runs before mask_text. It has
    to: mask_text needs 6+ consecutive digits and the hyphens in an SSN break
    that, so masking alone would have exported this verbatim."""
    _keys_redaction()
    configure_phi(mask_free_text=True)

    record = logging.LogRecord(
        "app", logging.ERROR, "/app/x.py", 10, '{"ssn": "123-45-6789"}', None, None
    )
    assert "123-45-6789" not in _first(handler, record)["body"]["stringValue"]


def test_a_json_log_body_is_suppressed_by_default(handler):
    _keys_redaction()
    record = logging.LogRecord(
        "app", logging.ERROR, "/app/x.py", 10, '{"ssn": "123-45-6789"}', None, None
    )
    assert "123-45-6789" not in _first(handler, record)["body"]["stringValue"]


def test_the_body_is_untouched_when_phi_is_disabled(handler):
    """Suppression is scoped to the policy — logs stay readable by default."""
    body = _first(handler, _record())["body"]["stringValue"]
    assert body == "lookup failed for patient Jane Doe"


def test_an_empty_body_does_not_become_none(handler):
    """`body.stringValue` must stay a string for the OTLP encoder."""
    record = logging.LogRecord("app", logging.ERROR, "/app/x.py", 10, "", None, None)
    assert _first(handler, record)["body"]["stringValue"] == ""


# ---------------------------------------------------------------------------
# The DEK travels with the ciphertext.
# ---------------------------------------------------------------------------


def test_an_encrypted_body_carries_its_dek(handler, monkeypatch):
    """`.get("_body", body)` discarded the DEK that the presidio path returns.
    A log record has no other surface to hold it, so the body exported as
    ciphertext nothing could read."""
    _presidio_encrypting(monkeypatch)

    record = _first(handler, _record())

    assert record["body"]["stringValue"].startswith("ns_enc_v1:")
    dek = _attr(record, "ns.crypto.encrypted_dek")
    assert dek is not None, "ciphertext exported without the key that decrypts it"
    assert dek["stringValue"] == "stub-dek"


def test_no_dek_is_attached_when_the_body_was_not_encrypted(handler):
    """The keys engine suppresses rather than encrypting — no key to advertise."""
    _keys_redaction()
    assert _attr(_first(handler, _record()), "ns.crypto.encrypted_dek") is None


def test_the_standard_log_attributes_are_preserved(handler, monkeypatch):
    """Adding the DEK must not displace logger.name / code.filepath / code.lineno."""
    _presidio_encrypting(monkeypatch)

    keys = [a["key"] for a in _first(handler, _record())["attributes"]]

    assert keys[:3] == ["logger.name", "code.filepath", "code.lineno"]
    assert keys.count("ns.crypto.encrypted_dek") == 1
