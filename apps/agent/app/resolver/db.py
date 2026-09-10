"""Read-only Postgres access for the direct resolver.

Three things this module is responsible for, in order of how easily they go wrong:

1. **Pooled vs direct endpoint.** Neon serves the same database on two hosts. The pooled one
   (`-pooler` in the name) runs PgBouncer in transaction mode, which cannot carry a prepared
   statement across connection checkouts — and asyncpg prepares statements by default. The
   symptom is intermittent failures that look like network or server problems rather than
   configuration, so the endpoint is detected from the host and the setting derived, instead
   of relying on whoever pasted the connection string to remember which one it was.

2. **The read guarantee.** Layered deliberately: the role is granted SELECT only, the
   transaction is opened READ ONLY so a write is refused even if the role could do it, and a
   server-side statement timeout bounds a legal-but-expensive query. `sql_guard` sits above
   all three. No single layer is trusted on its own.

3. **Schema introspection**, so the model can discover tables and columns rather than being
   handed a hardcoded schema that silently goes stale.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Optional
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT_MS = int(os.getenv("SQL_TIMEOUT_MS", "15000"))
APPLICATION_NAME = "agent-direct-resolver"


def is_pooled(dsn: str) -> bool:
    """True if this DSN points at a PgBouncer-fronted endpoint."""
    host = urlparse(dsn).hostname or ""
    return "-pooler" in host


def connect_kwargs(dsn: str, *, timeout_ms: int = DEFAULT_TIMEOUT_MS) -> dict[str, Any]:
    """Driver settings derived from the DSN. Pure, so the tricky parts are testable."""
    if not dsn or not dsn.strip():
        raise ValueError("no database DSN configured (NEON_DATABASE_URL is unset)")

    return {
        "dsn": dsn,
        # TLS always. Neon refuses plaintext, and a silent downgrade is not something to
        # leave to a query-string parameter that drivers interpret inconsistently.
        "ssl": "require",
        # 0 disables asyncpg's prepared-statement cache, which PgBouncer's transaction
        # mode cannot support. Harmless on a direct connection, essential on a pooled one.
        "statement_cache_size": 0 if is_pooled(dsn) else 100,
        "server_settings": {
            # Server-side: a client-side cancellation leaves the query running on the
            # database, which defeats the point of having a timeout at all.
            "statement_timeout": str(timeout_ms),
            "application_name": APPLICATION_NAME,
        },
    }


class PostgresDatabase:
    """The read-only surface `resolve_direct` expects, backed by asyncpg."""

    def __init__(self, dsn: Optional[str] = None, *, timeout_ms: int = DEFAULT_TIMEOUT_MS):
        self._dsn = dsn if dsn is not None else os.getenv("NEON_DATABASE_URL", "")
        self._timeout_ms = timeout_ms
        self._pool = None

    @classmethod
    def from_env(cls) -> Optional["PostgresDatabase"]:
        """None when unconfigured, so the caller reports it rather than pretending."""
        return cls() if os.getenv("NEON_DATABASE_URL", "").strip() else None

    async def _ensure_pool(self):
        if self._pool is None:
            import asyncpg

            kwargs = connect_kwargs(self._dsn, timeout_ms=self._timeout_ms)
            dsn = kwargs.pop("dsn")
            self._pool = await asyncpg.create_pool(dsn, min_size=1, max_size=4, **kwargs)
        return self._pool

    async def _fetch(self, sql: str, *args) -> list[dict]:
        pool = await self._ensure_pool()
        async with pool.acquire() as conn:
            # readonly=True issues BEGIN ... READ ONLY. Postgres then refuses any write
            # regardless of what the statement turned out to be.
            async with conn.transaction(readonly=True):
                rows = await conn.fetch(sql, *args)
        return [dict(r) for r in rows]

    async def list_tables(self) -> list[str]:
        rows = await self._fetch(
            """
            SELECT table_schema, table_name
            FROM information_schema.tables
            WHERE table_type = 'BASE TABLE'
              AND table_schema NOT IN ('pg_catalog', 'information_schema')
            ORDER BY table_schema, table_name
            """
        )
        # Unqualified inside the default schema, qualified elsewhere — the model writes
        # shorter SQL for the common case without losing the ability to reach the rest.
        return [
            r["table_name"] if r["table_schema"] == "public"
            else f"{r['table_schema']}.{r['table_name']}"
            for r in rows
        ]

    async def describe_table(self, table: str) -> list[dict]:
        schema, _, name = table.rpartition(".")
        return await self._fetch(
            """
            SELECT column_name AS column, data_type AS type, is_nullable AS nullable
            FROM information_schema.columns
            WHERE table_name = $1 AND table_schema = COALESCE(NULLIF($2, ''), 'public')
            ORDER BY ordinal_position
            """,
            name, schema,
        )

    async def run_query(self, sql: str) -> list[dict]:
        """Run already-guarded SQL. `sql_guard` must have approved it before this point."""
        return await self._fetch(sql)

    async def fingerprint(self) -> dict[str, int]:
        """Row counts per table — the cheap check that both variants read the same store.

        The two variants reach data by different routes and nothing structurally guarantees
        those routes end at the same database. Comparing counts catches that in seconds,
        instead of leaving it to be puzzled over after the numbers disagree.
        """
        rows = await self._fetch(
            """
            SELECT relname AS table, n_live_tup AS rows
            FROM pg_stat_user_tables
            ORDER BY n_live_tup DESC
            """
        )
        return {r["table"]: r["rows"] for r in rows}

    async def close(self) -> None:
        if self._pool is not None:
            await self._pool.close()
            self._pool = None
