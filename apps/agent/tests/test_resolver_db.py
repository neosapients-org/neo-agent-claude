"""Connecting to Postgres safely, and the one Neon setting that silently breaks things.

Neon offers two endpoints for the same database. The pooled one (`-pooler` in the host)
runs PgBouncer in transaction mode, which does NOT support prepared statements — and asyncpg
prepares statements by default. Point asyncpg at a pooled endpoint without disabling its
statement cache and queries fail intermittently with errors that read like network faults or
server bugs, not configuration.

Rather than ask a human to remember which endpoint they copied, the host is inspected and
the setting derived. Getting this wrong is the single most likely reason a correct query
appears to be broken.

The other half is the read guarantee, which is layered on purpose:
  - the role itself cannot write (granted SELECT only)
  - the transaction is opened READ ONLY, so a write is refused even if the role could
  - a statement timeout bounds a legal-but-expensive query
The parser in sql_guard is the layer above these; none of them is trusted alone.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.resolver.db import connect_kwargs, is_pooled  # noqa: E402

POOLED = "postgresql://u:p@ep-cool-boat-123-pooler.us-east-2.aws.neon.tech/db?sslmode=require"
DIRECT = "postgresql://u:p@ep-cool-boat-123.us-east-2.aws.neon.tech/db?sslmode=require"


class TestItDetectsThePooledEndpoint:
    def test_a_pooler_host_is_recognised(self):
        assert is_pooled(POOLED) is True

    def test_a_direct_host_is_not(self):
        assert is_pooled(DIRECT) is False

    def test_prepared_statements_are_disabled_for_the_pooler(self):
        """PgBouncer in transaction mode cannot hold a prepared statement across
        checkouts; leaving the cache on produces intermittent, misleading errors."""
        assert connect_kwargs(POOLED)["statement_cache_size"] == 0

    def test_prepared_statements_stay_on_for_a_direct_connection(self):
        assert connect_kwargs(DIRECT)["statement_cache_size"] != 0


class TestItAlwaysUsesTLSAndATimeout:
    @pytest.mark.parametrize("dsn", [POOLED, DIRECT])
    def test_tls_is_required(self, dsn):
        assert connect_kwargs(dsn)["ssl"]

    @pytest.mark.parametrize("dsn", [POOLED, DIRECT])
    def test_a_statement_timeout_is_set_on_the_server_side(self, dsn):
        """Server-side, not client-side: a client-side cancel leaves the query running
        on the database, which is exactly what a timeout is supposed to prevent."""
        settings = connect_kwargs(dsn, timeout_ms=9000)["server_settings"]
        assert settings["statement_timeout"] == "9000"

    def test_the_connection_is_labelled(self, ):
        """A readable application_name makes it obvious in pg_stat_activity who is
        running a query, which matters on a shared database."""
        assert "agent" in connect_kwargs(DIRECT)["server_settings"]["application_name"]


class TestItRefusesToGuess:
    def test_an_empty_dsn_is_rejected(self):
        with pytest.raises(ValueError):
            connect_kwargs("")
