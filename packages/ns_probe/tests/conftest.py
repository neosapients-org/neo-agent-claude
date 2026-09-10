"""Fixtures shared by the whole suite."""

from __future__ import annotations

import base64
import os
import secrets

import pytest


@pytest.fixture(scope="session", autouse=True)
def local_cmk():
    """Give the crypto engine a CMK for the length of the run.

    `LocalKMSProvider` reads a base64 256-bit key from NS_PROBE_LOCAL_CMK and
    raises without one, so every test that reaches `init_crypto_engine()` fails
    on a clean checkout unless the developer happens to have exported a key.
    That made two of the round-five tests fail for anyone who had not, which is
    the same thing as the suite not passing on a fresh clone.

    Generated per run rather than committed: a fixed key in the repository is a
    checked-in secret, and these tests need *a* key, never a particular one.

    An existing value is left alone, so a developer pointed at a real CMK keeps it.
    """
    if os.environ.get("NS_PROBE_LOCAL_CMK"):
        yield
        return

    os.environ["NS_PROBE_LOCAL_CMK"] = base64.b64encode(secrets.token_bytes(32)).decode("ascii")
    try:
        yield
    finally:
        os.environ.pop("NS_PROBE_LOCAL_CMK", None)
