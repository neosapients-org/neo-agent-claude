"""
Presidio-based PHI redaction engine for ns_probe.

This module provides NLP-powered detection and anonymisation of Protected
Health Information (PHI) in arbitrary text.  It wraps Microsoft Presidio's
``AnalyzerEngine`` and ``AnonymizerEngine`` behind a **lazy singleton**
(``PresidioRedactor``) so the heavy spaCy model is loaded at most once per
process and the per-span overhead is limited to Presidio's inference time.

Supported entity types (configurable):
    PERSON, PHONE_NUMBER, US_SSN, EMAIL_ADDRESS, DATE_TIME,
    CREDIT_CARD, US_DRIVER_LICENSE, IP_ADDRESS, MEDICAL_LICENSE,
    NRP, LOCATION, URL

Thread Safety:
    Presidio engines are thread-safe for read-only analysis/anonymise calls.
    The singleton is protected by a ``threading.Lock`` during first init.

Performance:
    - First call:  ~1-2 s (spaCy model load)
    - Subsequent:  ~0.5-5 ms per string depending on length
    - Non-string attributes are passed through with zero cost
"""

from __future__ import annotations

import json
import logging
import os
import threading
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .crypto import CryptoEngine

logger = logging.getLogger("ns_probe.presidio_redactor")

# ── Default entities to detect ────────────────────────────────────────
DEFAULT_ENTITIES: List[str] = [
    "PERSON",
    "PHONE_NUMBER",
    "US_SSN",
    "EMAIL_ADDRESS",
    "DATE_TIME",
    "CREDIT_CARD",
    "US_DRIVER_LICENSE",
    "IP_ADDRESS",
    "LOCATION",
    "MEDICAL_LICENSE",
    # NOTE: URL is intentionally excluded from defaults because Presidio
    # false-positives on Python repr strings like ``datetime.date(...)``.
    # Enable it via NS_PROBE_PHI_ENTITIES if needed.
]

# Env-var override: comma-separated entity list
# e.g. NS_PROBE_PHI_ENTITIES=PERSON,US_SSN,PHONE_NUMBER
_ENV_ENTITIES = "NS_PROBE_PHI_ENTITIES"

# Env-var for spaCy model name (default: en_core_web_sm)
_ENV_SPACY_MODEL = "NS_PROBE_PHI_SPACY_MODEL"


def _get_entity_list() -> List[str]:
    """Return the entity list from env-var or defaults."""
    raw = os.getenv(_ENV_ENTITIES)
    if raw:
        return [e.strip().upper() for e in raw.split(",") if e.strip()]
    return list(DEFAULT_ENTITIES)


# ── Custom Presidio Operator: Encrypt ─────────────────────────────────


class EncryptOperator:
    """
    Presidio-compatible operator that encrypts detected PHI via
    ``CryptoEngine`` (AES-256-GCM envelope encryption).

    Registered with Presidio under the name ``encrypt_phi``.  When the
    anonymizer is configured with this operator, detected PHI values are
    replaced with the wire format ``ns_enc_v1:<iv>:<ciphertext>`` instead
    of being destructively redacted.

    Inherits from ``presidio_anonymizer.operators.Operator`` so it can
    be registered via the ``OperatorsFactory``.
    """

    # Module-level reference so operate() can access the engine without
    # needing it passed per-call (Presidio instantiates operators via
    # the class, not the existing instance).
    _crypto_engine: Optional["CryptoEngine"] = None

    @classmethod
    def set_crypto_engine(cls, engine: "CryptoEngine") -> None:
        cls._crypto_engine = engine

    def operate(self, text: str, params: Optional[Dict] = None) -> str:
        """Encrypt the detected PHI text."""
        engine = self._crypto_engine
        if engine is None:
            return "<REDACTED>"
        return engine.encrypt(text)

    def validate(self, params: Optional[Dict] = None) -> None:
        """No per-call params required; engine set via class method."""
        pass

    def operator_name(self) -> str:
        return "encrypt_phi"

    def operator_type(self):
        from presidio_anonymizer.operators import OperatorType

        return OperatorType.Anonymize


class PresidioRedactor:
    """
    Lazy-initialised, thread-safe singleton that wraps Presidio's
    AnalyzerEngine + AnonymizerEngine.

    Supports two strategies:
    - ``redact``:  replace PHI with entity-type placeholders (``<PERSON>``)
    - ``encrypt``: replace PHI with AES-256-GCM ciphertext via ``CryptoEngine``

    Usage::

        redactor = PresidioRedactor.get_instance()
        safe = redactor.sanitize_text("Call Jane Doe at 555-123-4567")
        # → "Call <PERSON> at <PHONE_NUMBER>"
    """

    _instance: Optional["PresidioRedactor"] = None
    _lock = threading.Lock()

    def __init__(
        self,
        strategy: str = "redact",
        crypto_engine: Optional["CryptoEngine"] = None,
    ) -> None:
        # Deferred imports — these are heavy; only pay the cost when PHI
        # sanitisation is actually enabled.
        try:
            from presidio_analyzer import AnalyzerEngine
            from presidio_anonymizer import AnonymizerEngine
            from presidio_anonymizer.entities import OperatorConfig
        except ImportError as exc:
            raise ImportError(
                "Presidio is required for NLP-based PHI sanitisation. "
                "Install with:  pip install 'ns_probe[phi]'\n"
                "Then download the spaCy model:  python -m spacy download en_core_web_sm"
            ) from exc

        self._strategy = strategy
        self._crypto_engine = crypto_engine

        spacy_model = os.getenv(_ENV_SPACY_MODEL, "en_core_web_sm")
        logger.info(
            "Initialising Presidio (spaCy model: %s, strategy: %s) …",
            spacy_model,
            strategy,
        )

        from presidio_analyzer.nlp_engine import NlpEngineProvider

        provider = NlpEngineProvider(
            nlp_configuration={
                "nlp_engine_name": "spacy",
                "models": [{"lang_code": "en", "model_name": spacy_model}],
            }
        )

        self._analyzer = AnalyzerEngine(
            nlp_engine=provider.create_engine(),
            supported_languages=["en"],
        )

        # Register custom recognisers for Python-repr patterns that
        # standard Presidio NER won't catch (e.g. key='value' syntax).
        self._register_custom_recognizers()

        self._anonymizer = AnonymizerEngine()

        # Build the operator config based on strategy
        if strategy == "encrypt" and crypto_engine is not None:
            self._operator_config = self._build_encrypt_operators(crypto_engine)
        else:
            self._operator_config = self._build_redact_operators(OperatorConfig)

        self._entities = _get_entity_list() + ["CUSTOM_PHI"]
        self._score_threshold = float(os.getenv("NS_PROBE_PHI_SCORE_THRESHOLD", "0.35"))
        logger.info(
            "Presidio ready — entities=%s, threshold=%.2f, strategy=%s",
            self._entities,
            self._score_threshold,
            strategy,
        )

    @staticmethod
    def _build_redact_operators(OperatorConfig):
        """Build operator config for the default redaction strategy."""
        return {
            "DEFAULT": OperatorConfig("replace", {"new_value": "<REDACTED>"}),
            "PERSON": OperatorConfig("replace", {"new_value": "<PERSON>"}),
            "PHONE_NUMBER": OperatorConfig("replace", {"new_value": "<PHONE_NUMBER>"}),
            "US_SSN": OperatorConfig("replace", {"new_value": "<US_SSN>"}),
            "EMAIL_ADDRESS": OperatorConfig("replace", {"new_value": "<EMAIL_ADDRESS>"}),
            "DATE_TIME": OperatorConfig("replace", {"new_value": "<DATE_TIME>"}),
            "CREDIT_CARD": OperatorConfig("replace", {"new_value": "<CREDIT_CARD>"}),
            "LOCATION": OperatorConfig("replace", {"new_value": "<LOCATION>"}),
            "IP_ADDRESS": OperatorConfig("replace", {"new_value": "<IP_ADDRESS>"}),
            "CUSTOM_PHI": OperatorConfig("replace", {"new_value": "<REDACTED>"}),
        }

    def _build_encrypt_operators(self, crypto_engine: "CryptoEngine"):
        """Build operator config that encrypts detected PHI via CryptoEngine."""
        from presidio_anonymizer.entities import OperatorConfig

        # Set the engine on the class so Presidio-instantiated instances can access it
        EncryptOperator.set_crypto_engine(crypto_engine)

        # Register with Presidio's operator factory
        self._anonymizer.operators_factory.add_anonymize_operator(EncryptOperator)

        config = {}
        for entity in [
            "DEFAULT",
            "PERSON",
            "PHONE_NUMBER",
            "US_SSN",
            "EMAIL_ADDRESS",
            "DATE_TIME",
            "CREDIT_CARD",
            "LOCATION",
            "IP_ADDRESS",
            "CUSTOM_PHI",
        ]:
            config[entity] = OperatorConfig("encrypt_phi", {})
        return config

    def _register_custom_recognizers(self) -> None:
        """
        Add regex-based recognizers for PHI patterns that spaCy NER
        misses — primarily values embedded in Python repr strings.

        Patterns cover:
        - ``patient_name='Any Name Here'``  (captures the full match)
        - ``ssn='###-##-####'``
        - ``phone='##########'``
        - ``dob=datetime.date(YYYY, M, D)``
        """
        from presidio_analyzer import PatternRecognizer, Pattern

        # Match values after known PHI keys in repr-style strings
        name_repr_pattern = Pattern(
            name="name_repr",
            regex=r"""(?:patient_name|first_name|last_name|full_name|name)\s*=\s*['"][^'"]+['"]""",
            score=0.85,
        )
        # Match datetime.date(YYYY, M, D) patterns
        date_repr_pattern = Pattern(
            name="date_repr",
            regex=r"""datetime\.date\(\d{4},\s*\d{1,2},\s*\d{1,2}\)""",
            score=0.9,
        )
        # Match phone number values in repr syntax
        phone_repr_pattern = Pattern(
            name="phone_repr",
            regex=r"""(?:phone|phone_number)\s*=\s*['"][\d\-\.\s\(\)]{7,15}['"]""",
            score=0.85,
        )
        # Match SSN values in repr syntax: ssn='123-45-6789' or ssn="123456789"
        ssn_repr_pattern = Pattern(
            name="ssn_repr",
            regex=r"""(?:ssn|social_security|social_security_number)\s*=\s*['"][\d\-]{9,11}['"]""",
            score=0.95,
        )
        # Match DOB values in repr syntax: dob='2003-05-29' or dob="1990-01-01"
        dob_repr_pattern = Pattern(
            name="dob_repr",
            regex=r"""(?:dob|date_of_birth|birth_date)\s*=\s*['"][^'"]+['"]""",
            score=0.85,
        )

        custom_recognizer = PatternRecognizer(
            supported_entity="CUSTOM_PHI",
            patterns=[
                name_repr_pattern,
                date_repr_pattern,
                phone_repr_pattern,
                ssn_repr_pattern,
                dob_repr_pattern,
            ],
            supported_language="en",
        )
        self._analyzer.registry.add_recognizer(custom_recognizer)

    # ── Singleton accessor ────────────────────────────────────────────

    @classmethod
    def get_instance(
        cls,
        strategy: str = "redact",
        crypto_engine: Optional["CryptoEngine"] = None,
    ) -> "PresidioRedactor":
        """Return (or create) the singleton redactor."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls(
                        strategy=strategy,
                        crypto_engine=crypto_engine,
                    )
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """Tear down the singleton (for tests)."""
        with cls._lock:
            cls._instance = None

    # ── Core API ──────────────────────────────────────────────────────

    def sanitize_text(self, text: str) -> str:
        """
        Run *text* through Presidio and return the anonymised version.

        Short strings (≤2 chars) are returned as-is to avoid false positives
        and unnecessary overhead.
        """
        if len(text) <= 2:
            return text

        try:
            results = self._analyzer.analyze(
                text=text,
                entities=self._entities,
                language="en",
                score_threshold=self._score_threshold,
            )
            if not results:
                return text
            anonymized = self._anonymizer.anonymize(
                text=text,
                analyzer_results=results,
                operators=self._operator_config,
            )
            return anonymized.text
        except Exception:
            # Fail-open: if Presidio errors, return original text
            # rather than crashing the export pipeline.
            logger.debug("Presidio analysis failed, returning original text", exc_info=True)
            return text


# ── Recursive payload traversal ───────────────────────────────────────


def sanitize_payload(payload: Any, redactor: Optional[PresidioRedactor] = None) -> Any:
    """
    Recursively traverse *payload* and sanitise every string leaf via
    Presidio.

    Handles:
    - ``str``  – try JSON-parse first; if valid JSON, traverse the parsed
      structure then re-serialise.  Otherwise run ``sanitize_text()``.
    - ``dict`` – recurse into values (keys are left untouched).
    - ``list`` / ``tuple`` – recurse into elements.
    - Everything else (int, float, bool, None) – pass through unchanged.

    Args:
        payload: Arbitrary value (typically a span attribute value).
        redactor: ``PresidioRedactor`` instance (uses singleton if ``None``).

    Returns:
        A sanitised copy; the original is never mutated.
    """
    if redactor is None:
        redactor = PresidioRedactor.get_instance()

    if isinstance(payload, str):
        return _sanitize_string(payload, redactor)
    if isinstance(payload, dict):
        return {k: sanitize_payload(v, redactor) for k, v in payload.items()}
    if isinstance(payload, (list, tuple)):
        sanitized = [sanitize_payload(item, redactor) for item in payload]
        return type(payload)(sanitized) if isinstance(payload, tuple) else sanitized
    # int, float, bool, None, etc. — pass through
    return payload


def _sanitize_string(value: str, redactor: PresidioRedactor) -> str:
    """
    Sanitise a single string value.

    If the string looks like JSON (starts with `{` or `[`), parse → recurse
    → re-serialise so that embedded PHI in serialised dicts/lists is caught.
    Otherwise, run the raw string through ``sanitize_text()``.
    """
    if not value:
        return value

    # Fast-path: try JSON parse for attribute values like
    # traceloop.entity.input: '{"inputs": {"patient_name": "Jane Doe"}}'
    if value[0] in ("{", "["):
        try:
            parsed = json.loads(value)
            sanitized = sanitize_payload(parsed, redactor)
            return json.dumps(sanitized, default=str)
        except (json.JSONDecodeError, ValueError):
            pass  # Not valid JSON — fall through to plain-text path

    return redactor.sanitize_text(value)
