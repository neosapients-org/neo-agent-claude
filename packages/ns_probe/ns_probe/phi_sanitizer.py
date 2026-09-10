"""
PHI (Protected Health Information) Sanitization for ns_probe.

This module is the **single entry-point** for all PHI redaction inside the
SDK.  It sits between the span/log data and the export layer.

Two engines are supported (auto-selected):

1. **Presidio (NLP)** – Microsoft Presidio + spaCy.  Detects PHI in *any*
   string (names, SSNs, phone numbers, dates …) regardless of which
   attribute key holds the value.  This is the **recommended** engine for
   healthcare workloads.

2. **Key-matching (regex)** – lightweight fallback that replaces values
   whose *attribute key* matches a configurable set of sensitive patterns.
   Zero external dependencies.

Engine selection:
    * If Presidio libs are importable **and** PHI is enabled →  Presidio.
    * Otherwise → key-matching.
    * ``NS_PROBE_PHI_ENGINE=keys`` forces key-matching even when Presidio
      is installed (useful for latency-sensitive batch pipelines).

Configuration priority (highest → lowest):
    1. ``configure_phi()`` programmatic call
    2. Environment variables
    3. Built-in defaults

Environment Variables:
    NS_PROBE_PHI_ENABLED:
        Master switch (default: ``false``).
    NS_PROBE_PHI_ENGINE:
        ``presidio`` (default if libs present) or ``keys``.
    NS_PROBE_PHI_SENSITIVE_KEYS:
        Comma-separated list of attribute key patterns (key-matching engine).
    NS_PROBE_PHI_STRATEGY:
        ``redact``  (default) – replace with ``[REDACTED]``
        ``hash``              – replace with ``sha256:<hex>`` (key-matching only)
        ``encrypt``           – AES-256-GCM envelope encryption
    NS_PROBE_PHI_SENSITIVE_KEYS_FILE:
        Path to a newline-delimited file of sensitive key patterns.
    NS_PROBE_PHI_ENTITIES:
        Comma-separated Presidio entity list override.
    NS_PROBE_PHI_SPACY_MODEL:
        spaCy model override (default ``en_core_web_sm``).
    NS_PROBE_PHI_SCORE_THRESHOLD:
        Minimum Presidio confidence score (default ``0.4``).

Thread Safety:
    The sanitizer is called from the exporter daemon thread.
    Configuration is read-only after ``configure_phi()`` completes.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, FrozenSet, Iterable, List, NamedTuple, Optional, Set

logger = logging.getLogger("ns_probe.phi_sanitizer")

# ── Presidio availability probe (import-time, zero cost if absent) ────
_PRESIDIO_AVAILABLE = False
try:
    import presidio_analyzer  # noqa: F401
    import presidio_anonymizer  # noqa: F401

    _PRESIDIO_AVAILABLE = True
except ImportError:
    pass

# ── Default sensitive keys ────────────────────────────────────────────
# This list covers common HIPAA identifiers.  It is intentionally broad
# so that a new deployment is safe by default; production teams can
# narrow it via configuration.
DEFAULT_SENSITIVE_KEYS: FrozenSet[str] = frozenset(
    {
        # Patient identifiers
        "patient_name",
        "patient_id",
        "first_name",
        "last_name",
        "full_name",
        "name",
        # Government IDs
        "ssn",
        "social_security",
        "social_security_number",
        # Dates
        "dob",
        "date_of_birth",
        "birth_date",
        # Contact information
        "phone",
        "phone_number",
        "email",
        "email_address",
        "address",
        "street_address",
        "zip_code",
        "postal_code",
        # Medical record identifiers
        "mrn",
        "medical_record_number",
        "insurance_id",
        "policy_number",
        # Biometric / clinical (attribute-level)
        "blood_type",
        "diagnosis",
        "prescription",
    }
)


class SanitizationStrategy(str, Enum):
    """How to replace sensitive values."""

    REDACT = "redact"  # → "[REDACTED]" / "<PERSON>" etc.
    HASH = "hash"  # → "sha256:<hex>" (key-matching only)
    ENCRYPT = "encrypt"  # → "ns_enc_v1:<iv>:<ct>" (envelope encryption)


class PHIEngine(str, Enum):
    """Which engine performs the actual detection."""

    PRESIDIO = "presidio"  # NLP-based (Microsoft Presidio + spaCy)
    KEYS = "keys"  # Regex key-matching (lightweight fallback)


REDACTED_PLACEHOLDER = "[REDACTED]"

#: Wire prefix of an encrypted value — mirrors crypto._PREFIX. Used to tell
#: "ciphertext was produced" from "the engine existed", which is what decides
#: whether a DEK is attached.
CIPHERTEXT_PREFIX = "ns_enc_v1:"


@dataclass
class PHISanitizerConfig:
    """Runtime-immutable configuration for PHI sanitization."""

    enabled: bool = False
    engine: PHIEngine = PHIEngine.PRESIDIO
    # Key-matching settings (used when engine == KEYS)
    sensitive_keys: FrozenSet[str] = field(default_factory=lambda: DEFAULT_SENSITIVE_KEYS)
    strategy: SanitizationStrategy = SanitizationStrategy.REDACT
    # Free text (log bodies) under the KEYS engine: suppress, or mask and keep
    # the sentence. Defaults to suppress because masking cannot remove names —
    # see sanitize_free_text.
    mask_free_text: bool = False
    # Compiled regex patterns derived from sensitive_keys (built once)
    _patterns: Optional[re.Pattern] = field(default=None, repr=False)

    def __post_init__(self) -> None:
        # Build a single compiled regex that matches any sensitive key
        # as a *substring* of the attribute key (case-insensitive).
        if self.sensitive_keys:
            escaped = [re.escape(k) for k in sorted(self.sensitive_keys)]
            self._patterns = re.compile("|".join(escaped), re.IGNORECASE)
        else:
            self._patterns = None


# ── Module-level singleton ────────────────────────────────────────────

_config: Optional[PHISanitizerConfig] = None


def _load_keys_from_file(path: str) -> Set[str]:
    """Load sensitive keys from a newline-delimited text file."""
    keys: Set[str] = set()
    try:
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                stripped = line.strip()
                if stripped and not stripped.startswith("#"):
                    keys.add(stripped.lower())
    except OSError as exc:
        logger.warning("Could not read PHI keys file %s: %s", path, exc)
    return keys


def get_phi_config() -> PHISanitizerConfig:
    """Return the current PHI config, creating from env if needed."""
    global _config
    if _config is None:
        _config = _build_config_from_env()
    return _config


def _build_config_from_env() -> PHISanitizerConfig:
    """Build PHI config from environment variables + defaults."""
    enabled = os.getenv("NS_PROBE_PHI_ENABLED", "false").lower() in ("true", "1", "yes")

    # Engine selection
    engine_str = os.getenv("NS_PROBE_PHI_ENGINE", "").lower()
    if engine_str == "keys":
        engine = PHIEngine.KEYS
    elif engine_str == "presidio":
        engine = PHIEngine.PRESIDIO
    else:
        # Auto-detect: use Presidio if available, else fall back to keys
        engine = PHIEngine.PRESIDIO if _PRESIDIO_AVAILABLE else PHIEngine.KEYS

    # Merge sources: defaults ← file ← env-var list
    keys: Set[str] = set(DEFAULT_SENSITIVE_KEYS)

    keys_file = os.getenv("NS_PROBE_PHI_SENSITIVE_KEYS_FILE")
    if keys_file:
        keys |= _load_keys_from_file(keys_file)

    keys_env = os.getenv("NS_PROBE_PHI_SENSITIVE_KEYS")
    if keys_env:
        keys |= {k.strip().lower() for k in keys_env.split(",") if k.strip()}

    mask_free_text = os.getenv("NS_PROBE_PHI_MASK_FREE_TEXT", "false").lower() in (
        "true",
        "1",
        "yes",
    )

    strategy_str = os.getenv("NS_PROBE_PHI_STRATEGY", "redact").lower()
    try:
        strategy = SanitizationStrategy(strategy_str)
    except ValueError:
        logger.warning("Unknown PHI strategy '%s', falling back to 'redact'", strategy_str)
        strategy = SanitizationStrategy.REDACT

    return PHISanitizerConfig(
        enabled=enabled,
        engine=engine,
        sensitive_keys=frozenset(keys),
        strategy=strategy,
        mask_free_text=mask_free_text,
    )


def configure_phi(
    *,
    enabled: Optional[bool] = None,
    engine: Optional[str] = None,
    sensitive_keys: Optional[List[str]] = None,
    strategy: Optional[str] = None,
    mask_free_text: Optional[bool] = None,
) -> PHISanitizerConfig:
    """
    Programmatically configure PHI sanitization.

    Call this **before** any spans are created (typically right after
    ``ns_probe.configure()``).

    Args:
        enabled:  Master on/off switch.
        engine:   ``"presidio"`` (NLP) or ``"keys"`` (regex key-matching).
                  Defaults to ``"presidio"`` when libs are installed.
        sensitive_keys:  Full *replacement* list of sensitive attribute keys
                         (only used by the ``keys`` engine).  Pass ``None``
                         to keep the current list.
        strategy: ``"redact"``, ``"hash"``, or ``"encrypt"``.
                  ``"encrypt"`` uses AES-256-GCM envelope encryption.

    Returns:
        The active PHISanitizerConfig.
    """
    global _config
    cfg = get_phi_config()

    new_enabled = enabled if enabled is not None else cfg.enabled

    if engine is not None:
        try:
            new_engine = PHIEngine(engine.lower())
        except ValueError:
            raise ValueError(f"Invalid PHI engine: {engine!r}.  Use 'presidio' or 'keys'.")
        if new_engine is PHIEngine.PRESIDIO and not _PRESIDIO_AVAILABLE:
            raise ImportError(
                "Presidio engine requested but presidio-analyzer / presidio-anonymizer "
                "are not installed.  Install with: pip install 'ns_probe[phi]'"
            )
    else:
        new_engine = cfg.engine

    new_keys = (
        frozenset(k.lower() for k in sensitive_keys)
        if sensitive_keys is not None
        else cfg.sensitive_keys
    )

    if strategy is not None:
        try:
            new_strategy = SanitizationStrategy(strategy.lower())
        except ValueError:
            raise ValueError(
                f"Invalid PHI strategy: {strategy!r}.  Use 'redact', 'hash', or 'encrypt'."
            )
    else:
        new_strategy = cfg.strategy

    _config = PHISanitizerConfig(
        enabled=new_enabled,
        engine=new_engine,
        sensitive_keys=new_keys,
        strategy=new_strategy,
        mask_free_text=cfg.mask_free_text if mask_free_text is None else mask_free_text,
    )

    # If encrypt strategy is selected, initialise the CryptoEngine
    if new_enabled and new_strategy is SanitizationStrategy.ENCRYPT:
        from .crypto import get_crypto_engine, init_crypto_engine, _CRYPTO_AVAILABLE

        if not _CRYPTO_AVAILABLE:
            raise ImportError(
                "encrypt strategy requires the 'cryptography' package. "
                "Install with:  pip install 'ns_probe[phi]'"
            )
        if get_crypto_engine() is None:
            init_crypto_engine()

    logger.info(
        "PHI sanitizer configured: enabled=%s, engine=%s, strategy=%s, keys=%d",
        _config.enabled,
        _config.engine.value,
        _config.strategy.value,
        len(_config.sensitive_keys),
    )
    return _config


def reset_phi_config() -> None:
    """Reset to defaults (primarily for tests)."""
    global _config
    _config = None
    # Also reset singletons so next configure_phi starts fresh
    try:
        from .presidio_redactor import PresidioRedactor

        PresidioRedactor.reset()
    except ImportError:
        pass
    try:
        from .crypto import reset_crypto_engine

        reset_crypto_engine()
    except ImportError:
        pass


# ── Core sanitization helpers ─────────────────────────────────────────


def _hash_value(value: Any) -> str:
    """Deterministic SHA-256 hash prefixed with ``sha256:``."""
    raw = str(value).encode("utf-8")
    return f"sha256:{hashlib.sha256(raw).hexdigest()}"


def _is_sensitive_key(key: str, pattern: Optional[re.Pattern]) -> bool:
    """Return True if *key* matches any sensitive pattern."""
    if pattern is None:
        return False
    return pattern.search(key) is not None


def sanitize_value(
    value: Any,
    strategy: SanitizationStrategy,
    crypto_engine: Optional[Any] = None,
) -> Any:
    """Replace a single value according to *strategy*.

    ENCRYPT needs `crypto_engine`. Without one it degrades to redaction, which is
    safe but lossy — so the caller resolves the engine once and warns, rather
    than letting each value silently downgrade. Redaction is the fallback for
    every unrecognised strategy because failing closed is the only safe default
    for a sanitizer.
    """
    if strategy is SanitizationStrategy.HASH:
        return _hash_value(value)
    if strategy is SanitizationStrategy.ENCRYPT and crypto_engine is not None:
        try:
            return crypto_engine.encrypt(str(value))
        except Exception:  # pragma: no cover — never leak plaintext on failure
            logger.warning("encryption failed for a value; redacting", exc_info=True)
            return REDACTED_PLACEHOLDER
    return REDACTED_PLACEHOLDER


# ── Public API: sanitize a dict of attributes ─────────────────────────


def sanitize_attributes(attrs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Return a *new* dict with sensitive values replaced.

    If PHI sanitization is disabled the original dict is returned
    unchanged (zero overhead).

    Dispatches to the configured engine:
    - **presidio**: every string value (including JSON-embedded) is run
      through NLP-based entity detection.
    - **keys**: only values whose attribute key matches the sensitive-key
      regex are replaced.
    """
    cfg = get_phi_config()
    if not cfg.enabled or not attrs:
        return attrs

    if cfg.engine is PHIEngine.PRESIDIO:
        return _sanitize_attrs_presidio(attrs)
    return _sanitize_attrs_keys(attrs, cfg)


class SanitizedText(NamedTuple):
    """Sanitized free text, plus any DEK the caller must keep with it.

    `encrypted_dek` is set only when sanitizing produced ciphertext. The caller
    is responsible for putting it on an exported attribute surface — ciphertext
    whose DEK was dropped is unreadable, and a status message can be the ONLY
    thing on a span, so there may be no other attribute carrying it.
    """

    text: Optional[str]
    encrypted_dek: Optional[str] = None


def sanitize_free_text(text: Optional[str], *, suppress: Optional[bool] = None) -> SanitizedText:
    """Sanitize a bare string that has no attribute key to match on.

    `span.status_message` is the case that forced this: `Tracer.start_span`
    catches an exception and calls `set_status(ERROR, str(e))`, and the exporter
    copied that into the OTLP status verbatim — so an exception message carrying
    patient text left the process with PHI filtering on.

    The two engines can do different amounts here:

    - **presidio** detects entities in free text, so the message is redacted and
      whatever is left is still useful for debugging. Under ENCRYPT it returns
      ciphertext and a DEK, which the caller must carry — see SanitizedText.
    - **keys** matches on ATTRIBUTE KEYS. A bare message has none, so there is
      nothing to match and no way to tell a safe message from one quoting a
      record.

    `suppress=True` (the default) replaces the whole string. `suppress=False`
    runs the JSON key scan and then `mask_text`, keeping the sentence.

    ⚠️ MASKING DOES NOT REMOVE NAMES, and that is why suppression is the default.
    `mask_text` works from patterns — email, phone, dates, runs of 6+ digits —
    plus any `known_values` it is handed. A name is not a pattern, so
    "lookup failed for patient Jane Doe" masks to itself: the exact leak this
    function was written to close. Masking is therefore opt-in
    (`NS_PROBE_PHI_MASK_FREE_TEXT=true`) for operators who want partial coverage
    knowingly, not a default.

    The real fix for "logs go dark under PHI mode" is installing the `phi` extra
    so presidio is importable. Presidio DOES detect names, so the presidio branch
    above gives a log body that is both safe and readable. Under `keys` there is
    no way to find a name in prose, so the only fail-closed answer is to drop it.
    """
    cfg = get_phi_config()
    if not cfg.enabled or not text:
        return SanitizedText(text)

    if suppress is None:
        suppress = not cfg.mask_free_text

    if cfg.engine is PHIEngine.PRESIDIO:
        out = _sanitize_attrs_presidio({"message": text})
        return SanitizedText(
            out.get("message", REDACTED_PLACEHOLDER),
            out.get("ns.crypto.encrypted_dek"),
        )
    if suppress:
        return SanitizedText(REDACTED_PLACEHOLDER)

    # Masking alone is NOT enough. A JSON body has real keys to match, and the
    # key scan catches {"ssn": "123-45-6789"} where mask_text does not — its
    # digit-run pattern needs 6+ consecutive digits and the hyphens break that.
    # So the key scan runs first (it returns the text untouched when it is not
    # JSON) and mask_text covers the prose case afterwards.
    crypto_engine = _resolve_crypto_engine(cfg)
    scanned = _sanitize_json_string(text, cfg, crypto_engine)

    if _is_ciphertext(scanned):
        # Already protected, and masking would corrupt it: base64 contains long
        # digit runs that mask_text would rewrite to [ID], making the ciphertext
        # undecryptable.
        dek = crypto_engine.get_encrypted_dek() if crypto_engine is not None else None
        return SanitizedText(scanned, dek)

    return SanitizedText(mask_text(scanned))


# ── Presidio engine ───────────────────────────────────────────────────


def _sanitize_attrs_presidio(attrs: Dict[str, Any]) -> Dict[str, Any]:
    """Sanitize every attribute value using Presidio NLP."""
    from .presidio_redactor import sanitize_payload, PresidioRedactor

    cfg = get_phi_config()
    strategy = cfg.strategy.value  # "redact" or "encrypt"

    # Shared with the keys engine so both paths get the same PINNED key. This
    # used to resolve the live engine inline, which meant presidio could encrypt
    # with one key and then report a rotated one — the pairing bug this round
    # fixed for the keys path only.
    crypto_engine = _resolve_crypto_engine(cfg)
    if cfg.strategy is SanitizationStrategy.ENCRYPT and crypto_engine is None:
        strategy = "redact"

    redactor = PresidioRedactor.get_instance(
        strategy=strategy,
        crypto_engine=crypto_engine,
    )

    # get_instance is a singleton and ignores crypto_engine after the first call,
    # so the pinned key has to be pushed onto the operator explicitly or every
    # later pass would encrypt with whatever key the FIRST pass captured.
    if crypto_engine is not None:
        try:
            from .presidio_redactor import EncryptOperator

            EncryptOperator.set_crypto_engine(crypto_engine)
        except Exception:  # pragma: no cover — best-effort
            logger.debug("could not pin the crypto key for presidio", exc_info=True)

    sanitized: Dict[str, Any] = {}
    for key, value in attrs.items():
        sanitized[key] = sanitize_payload(value, redactor)

    # Only when ciphertext was actually produced, matching the keys engine. A DEK
    # on a span that encrypted nothing decrypts nothing and makes the attribute
    # useless as a signal for "this span has ciphertext".
    if crypto_engine is not None and any(_is_ciphertext(v) for v in sanitized.values()):
        sanitized["ns.crypto.encrypted_dek"] = crypto_engine.get_encrypted_dek()

    return sanitized


# ── Key-matching engine (original, lightweight) ───────────────────────


def _sanitize_attrs_keys(attrs: Dict[str, Any], cfg: PHISanitizerConfig) -> Dict[str, Any]:
    """Sanitize based on attribute-key regex matching.

    ENCRYPT is honoured here, not silently downgraded. `configure_phi` accepts
    `engine="keys", strategy="encrypt"` and initialises a CryptoEngine for it, but
    this path used to treat every non-hash strategy as redaction — so the
    documented encryption mode destroyed the data it was asked to protect, and
    emitted neither ciphertext nor `ns.crypto.encrypted_dek`. Same shape as the
    presidio engine now, including the DEK the reader needs to decrypt.
    """
    crypto_engine = _resolve_crypto_engine(cfg)

    sanitized = {}
    encrypted_any = False
    for key, value in attrs.items():
        if _is_sensitive_key(key, cfg._patterns):
            sanitized[key] = sanitize_value(value, cfg.strategy, crypto_engine)
            encrypted_any = encrypted_any or _is_ciphertext(sanitized[key])
        elif isinstance(value, str):
            sanitized[key] = _sanitize_json_string(value, cfg, crypto_engine)
            encrypted_any = encrypted_any or _is_ciphertext(sanitized[key])
        else:
            sanitized[key] = value

    # Without this the ciphertext is undecryptable, which is data loss wearing a
    # different hat. Keyed off ciphertext actually being produced, not off the
    # engine existing: a span whose values all redacted (or that matched nothing)
    # would otherwise carry a DEK that decrypts nothing.
    if encrypted_any:
        try:
            sanitized["ns.crypto.encrypted_dek"] = crypto_engine.get_encrypted_dek()
        except Exception:  # pragma: no cover
            logger.warning("encrypted DEK unavailable", exc_info=True)

    return sanitized


def _is_ciphertext(value: Any) -> bool:
    """True when `value` carries an encrypted payload, nested JSON included."""
    return isinstance(value, str) and CIPHERTEXT_PREFIX in value


def _resolve_crypto_engine(cfg: PHISanitizerConfig) -> Optional[Any]:
    """The CryptoEngine for an ENCRYPT config, or None.

    Returns None for every other strategy, and warns once per call when ENCRYPT
    was asked for but no engine is initialised — the values then redact, which is
    safe but is not what the caller configured.
    """
    if cfg.strategy is not SanitizationStrategy.ENCRYPT:
        return None
    from .crypto import get_crypto_engine

    engine = get_crypto_engine()
    if engine is None:
        logger.warning(
            "encrypt strategy requested but CryptoEngine not initialised; "
            "sensitive values will be redacted instead of encrypted"
        )
        return None

    # PINNED for the whole pass. Encrypting values and then asking for the DEK
    # are two calls, and the key rotates on a timer — a rotation between them
    # exported ciphertext from the old key beside the new wrapped one, which is
    # unrecoverable. `snapshot()` returns one key that answers both.
    try:
        return engine.snapshot()
    except AttributeError:
        # A stand-in engine without snapshot(); it still encrypts.
        return engine


def _sanitize_json_string(
    value: str, cfg: PHISanitizerConfig, crypto_engine: Optional[Any] = None
) -> str:
    """If *value* looks like a JSON object, sanitize keys inside it."""
    if not value or value[0] not in ("{", "["):
        return value
    try:
        parsed = json.loads(value)
    except (json.JSONDecodeError, ValueError):
        return value

    changed = _sanitize_nested(parsed, cfg, crypto_engine)
    return json.dumps(changed, default=str)


def _sanitize_nested(obj: Any, cfg: PHISanitizerConfig, crypto_engine: Optional[Any] = None) -> Any:
    """Recursively sanitize dicts/lists.

    `crypto_engine` is threaded through because a match nested inside a JSON
    string is the same match as a top-level one and must get the same treatment.
    Without it, ENCRYPT fell back to redaction here while top-level values
    encrypted — so `{"ssn": "..."}` inside an attribute was destroyed rather
    than protected, silently and only for nested data.
    """
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            if _is_sensitive_key(k, cfg._patterns):
                out[k] = sanitize_value(v, cfg.strategy, crypto_engine)
            else:
                out[k] = _sanitize_nested(v, cfg, crypto_engine)
        return out
    if isinstance(obj, list):
        return [_sanitize_nested(item, cfg, crypto_engine) for item in obj]
    return obj


# ── Inline text masking ───────────────────────────────────────────────
# The two engines above are all-or-nothing per attribute: `keys` replaces a
# whole value, `presidio` needs spaCy and several hundred MB. Neither suits the
# common case of a *conversation* — replacing the whole exchange loses the thing
# you wanted to see, but shipping it raw is not an option either.
#
# mask_text() sits between them: it removes identifiers *inside* the sentence so
# a reviewer still sees what was asked and answered.
#
#     "Confirm your date of birth, Alex Sample?"
#     -> "Confirm your date of birth, [NAME]?"
#
# It is NOT a general-purpose de-identifier. It covers the categories most
# conversations elicit, plus values the caller already knows (a name from a
# record beats inferring one). Free text can always carry something these
# patterns miss — use the presidio engine when you need a real guarantee.

_TEXT_PHI_PATTERNS = [
    (re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.]+\b"), "[EMAIL]"),
    (re.compile(r"\b(?:\+?\d{1,2}[\s.-]?)?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}\b"), "[PHONE]"),
    # Numeric dates: 2015-05-15, 05/15/2015
    (re.compile(r"\b\d{4}-\d{1,2}-\d{1,2}\b|\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b"), "[DATE]"),
    # Spoken/written dates: "May 15, 2015", "15 May 2015"
    (
        re.compile(
            r"\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s+\d{1,2}(?:st|nd|rd|th)?,?\s*\d{2,4}\b"
            r"|\b\d{1,2}(?:st|nd|rd|th)?\s+(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?,?\s*\d{2,4}\b",
            re.I,
        ),
        "[DATE]",
    ),
    # Long digit runs: record numbers, SSN, insurance, account ids
    (re.compile(r"\b\d{6,}\b"), "[ID]"),
]


def mask_text(text: str, known_values: Iterable[str] = ()) -> str:
    """Mask identifiers inside free text, keeping the sentence readable.

    Args:
        text: the sentence to mask.
        known_values: exact strings to remove — e.g. the name and date of birth
            already on file. Matching a known value is far more reliable than
            inferring it: a name is not detectable by pattern, but you have it.

    Returns the masked text; on any internal error returns ``"[MASKED]"`` rather
    than risk returning the original.
    """
    if not text or not isinstance(text, str):
        return text
    try:
        out = text
        # Patterns first: an address like a.b@example.com must be masked as one
        # unit, before name-part matching can chew it into [NAME].[NAME]@…
        for pattern, replacement in _TEXT_PHI_PATTERNS:
            out = pattern.sub(replacement, out)
        for value in known_values or ():
            if value and isinstance(value, str) and len(value) > 2:
                out = re.sub(re.escape(value), "[NAME]", out, flags=re.I)
                # Also mask the parts of a full name said on their own.
                for part in value.split():
                    if len(part) > 2:
                        out = re.sub(rf"\b{re.escape(part)}\b", "[NAME]", out, flags=re.I)
        return out
    except Exception:
        # Masking must never fail open onto raw text.
        return "[MASKED]"
