"""
Envelope Encryption engine for ns_probe PHI sanitization.

Implements HIPAA-compliant envelope encryption where PHI values are
encrypted with a Data Encryption Key (DEK) that is itself encrypted
by a Customer Master Key (CMK).  The encrypted DEK travels alongside
the spans so that only authorised downstream systems with ``kms:Decrypt``
can recover the plaintext.

Architecture::

    ┌──────────────┐  plaintext DEK   ┌───────────────┐
    │ KeyProvider   │─────────────────►│ CryptoEngine  │
    │ (CMK holder) │                  │ (AES-256-GCM) │
    └──────┬───────┘                  └───────┬───────┘
           │ encrypted DEK                    │ ciphertext
           ▼                                  ▼
    ┌──────────────────────────────────────────────┐
    │            Span Attributes                   │
    │  ns.crypto.encrypted_dek = <b64>             │
    │  patient_name = ns_enc_v1:<iv>:<ct>          │
    └──────────────────────────────────────────────┘

Wire format:  ``ns_enc_v1:<base64_iv>:<base64_ciphertext>``

Key Providers:
    * ``LocalKMSProvider``  – AES-256-GCM wrap/unwrap using a local CMK
      read from ``NS_PROBE_LOCAL_CMK`` (base64 string).  Suitable for
      local development and testing.
    * ``AWSKMSProvider``    – (future) calls ``kms:GenerateDataKey`` to
      obtain a DEK from AWS KMS.  See ``docs/PHI_SANITIZATION.md``.

Thread Safety:
    ``CryptoEngine`` is thread-safe.  DEK rotation is protected by a lock.
"""

from __future__ import annotations

import base64
import logging
import os
import secrets
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, NamedTuple, Optional

logger = logging.getLogger("ns_probe.crypto")

# ── Availability probe ────────────────────────────────────────────────
_CRYPTO_AVAILABLE = False
try:
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM

    _CRYPTO_AVAILABLE = True
except ImportError:
    AESGCM = None  # type: ignore[misc,assignment]

# ── Constants ─────────────────────────────────────────────────────────
_PREFIX = "ns_enc_v1"
_DEK_SIZE = 32  # AES-256
_IV_SIZE = 12  # 96-bit nonce for GCM
_DEK_TTL_SECONDS = 300  # 5 minutes


# ── Data structures ───────────────────────────────────────────────────


@dataclass(frozen=True)
class DataKeyPair:
    """A DEK in both plaintext (for encryption) and wrapped (for export)."""

    plaintext_dek: bytes  # 32 bytes — used by CryptoEngine
    encrypted_dek: bytes  # wrapped by CMK — attached to spans


# ── Key Provider interface ────────────────────────────────────────────


class KeyProvider(ABC):
    """
    Abstract base class for envelope-encryption key providers.

    Sub-classes supply Data Encryption Keys (DEKs) that are already
    wrapped by a Customer Master Key (CMK).
    """

    @abstractmethod
    def get_data_key(self) -> DataKeyPair:
        """
        Generate or retrieve a DEK.

        Returns:
            A ``DataKeyPair`` containing the plaintext DEK and the
            CMK-encrypted DEK.
        """
        ...


# ── Local KMS Provider ────────────────────────────────────────────────


class LocalKMSProvider(KeyProvider):
    """
    Local envelope-encryption key provider for development/testing.

    Reads a base64-encoded 256-bit CMK from ``NS_PROBE_LOCAL_CMK`` and
    uses AES-256-GCM to wrap freshly generated DEKs.

    **NOT** for production — use ``AWSKMSProvider`` (or equivalent)
    with proper key management in deployed environments.
    """

    def __init__(self, cmk_b64: Optional[str] = None) -> None:
        if not _CRYPTO_AVAILABLE:
            raise ImportError(
                "The 'cryptography' package is required for envelope encryption. "
                "Install with:  pip install 'ns_probe[phi]'"
            )
        raw = cmk_b64 or os.getenv("NS_PROBE_LOCAL_CMK")
        if not raw:
            raise ValueError(
                "No CMK provided.  Set NS_PROBE_LOCAL_CMK (base64 string) "
                "or pass cmk_b64= to LocalKMSProvider()."
            )
        self._cmk = base64.b64decode(raw)
        if len(self._cmk) != _DEK_SIZE:
            raise ValueError(
                f"CMK must be exactly {_DEK_SIZE} bytes (got {len(self._cmk)}). "
                f'Generate one with:  python -c "import secrets,base64; '
                f'print(base64.b64encode(secrets.token_bytes(32)).decode())"'
            )

    def get_data_key(self) -> DataKeyPair:
        """Generate a random DEK and wrap it with the local CMK."""
        plaintext_dek = secrets.token_bytes(_DEK_SIZE)
        # Wrap the DEK: AES-256-GCM(CMK, DEK)
        iv = secrets.token_bytes(_IV_SIZE)
        aesgcm = AESGCM(self._cmk)
        ciphertext = aesgcm.encrypt(iv, plaintext_dek, None)
        # encrypted_dek = iv || ciphertext (easy to split for unwrap)
        encrypted_dek = iv + ciphertext
        return DataKeyPair(plaintext_dek=plaintext_dek, encrypted_dek=encrypted_dek)


# ── Crypto Engine ─────────────────────────────────────────────────────


class CryptoEngine:
    """
    AES-256-GCM encryption engine with automatic DEK rotation.

    The engine caches a DEK for up to ``_DEK_TTL_SECONDS`` (default 5 min)
    to amortise key-generation cost across many spans.

    Thread-safe: DEK rotation is serialised by a lock; concurrent
    ``encrypt()`` calls on the *same* DEK share the cached key without
    contention (AESGCM is safe for concurrent use with unique nonces,
    and ``secrets.token_bytes`` provides unique IVs).

    Usage::

        provider = LocalKMSProvider()
        engine = CryptoEngine(provider)
        ct = engine.encrypt("Jane Doe")
        # → "ns_enc_v1:rAnDoMiV...:cIpHeRtExT..."
        dek = engine.get_encrypted_dek()
        # → base64 string to attach as span attribute
    """

    def __init__(self, provider: KeyProvider) -> None:
        if not _CRYPTO_AVAILABLE:
            raise ImportError(
                "The 'cryptography' package is required for envelope encryption. "
                "Install with:  pip install 'ns_probe[phi]'"
            )
        self._provider = provider
        self._lock = threading.Lock()
        self._aesgcm: Optional[AESGCM] = None
        self._encrypted_dek: Optional[bytes] = None
        self._dek_created_at: float = 0.0

    def _ensure_dek(self) -> None:
        """Rotate the DEK if expired or missing."""
        now = time.monotonic()
        if self._aesgcm is not None and (now - self._dek_created_at) < _DEK_TTL_SECONDS:
            return
        with self._lock:
            # Double-check under lock
            now = time.monotonic()
            if self._aesgcm is not None and (now - self._dek_created_at) < _DEK_TTL_SECONDS:
                return
            pair = self._provider.get_data_key()
            self._aesgcm = AESGCM(pair.plaintext_dek)
            self._encrypted_dek = pair.encrypted_dek
            self._dek_created_at = now
            logger.debug("DEK rotated (TTL=%ds)", _DEK_TTL_SECONDS)

    def snapshot(self) -> "PinnedKey":
        """Pin one DEK for a whole sanitization pass.

        `encrypt()` and `get_encrypted_dek()` each call `_ensure_dek()`
        independently, so nothing tied them to the same key. Sanitizers encrypt
        values first and ask for the DEK afterwards, so a rotation landing
        between the two steps exported ciphertext from the OLD key beside the
        NEW wrapped one — permanently unrecoverable, on a %d-second timer.

        Taking the key once and using that object for both steps removes the
        window. Callers that encrypt anything should use this, not the methods
        below.
        """ % _DEK_TTL_SECONDS
        self._ensure_dek()
        with self._lock:
            return PinnedKey(
                aesgcm=self._aesgcm,
                encrypted_dek=base64.b64encode(self._encrypted_dek).decode("ascii"),  # type: ignore[arg-type]
            )

    def encrypt(self, plaintext: str) -> str:
        """
        Encrypt *plaintext* with AES-256-GCM.

        Prefer `snapshot()` when the wrapped DEK is also needed — this method
        re-reads the current key, which may already have rotated.

        Returns:
            Wire format string: ``ns_enc_v1:<base64_iv>:<base64_ciphertext>``
        """
        return self.snapshot().encrypt(plaintext)

    def get_encrypted_dek(self) -> str:
        """Return the current wrapped DEK as a base64 string.

        Prefer `snapshot()`: this returns whatever key is current NOW, which is
        not necessarily the one a previous `encrypt()` call used.
        """
        self._ensure_dek()
        return base64.b64encode(self._encrypted_dek).decode("ascii")  # type: ignore[arg-type]


class PinnedKey(NamedTuple):
    """One DEK, held still.

    Exposes the same `encrypt()` the engine does, so anything that duck-types on
    a crypto engine accepts a PinnedKey unchanged, plus the wrapped DEK that
    belongs to the ciphertext it produces.
    """

    aesgcm: Any
    encrypted_dek: str

    def encrypt(self, plaintext: str) -> str:
        """Encrypt with THIS key. Wire format ``ns_enc_v1:<iv>:<ciphertext>``."""
        iv = secrets.token_bytes(_IV_SIZE)
        ct = self.aesgcm.encrypt(iv, plaintext.encode("utf-8"), None)
        return (
            f"{_PREFIX}:"
            f"{base64.b64encode(iv).decode('ascii')}:"
            f"{base64.b64encode(ct).decode('ascii')}"
        )

    def get_encrypted_dek(self) -> str:
        """Present the same surface as CryptoEngine for drop-in use."""
        return self.encrypted_dek


# ── Module-level singleton ────────────────────────────────────────────

_engine: Optional[CryptoEngine] = None
_engine_lock = threading.Lock()


def get_crypto_engine() -> Optional[CryptoEngine]:
    """Return the module-level CryptoEngine singleton (or None)."""
    return _engine


def init_crypto_engine(provider: Optional[KeyProvider] = None) -> CryptoEngine:
    """
    Initialise or replace the module-level CryptoEngine.

    If *provider* is ``None``, creates a ``LocalKMSProvider`` from
    the ``NS_PROBE_LOCAL_CMK`` environment variable.
    """
    global _engine
    with _engine_lock:
        if provider is None:
            provider = LocalKMSProvider()
        _engine = CryptoEngine(provider)
        logger.info("CryptoEngine initialised (provider=%s)", type(provider).__name__)
        return _engine


def reset_crypto_engine() -> None:
    """Tear down the singleton (for tests)."""
    global _engine
    with _engine_lock:
        _engine = None
