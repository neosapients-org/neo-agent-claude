"""Deterministic gate between model-written SQL and a real database.

The model is asked for read-only SQL. This module is what makes that true. Prompt
instructions are not a security control — they are a request, and a model under pressure to
answer will occasionally write something else. So every statement is parsed and checked
before the database sees it.

What this is NOT: a sandbox. It is the first of three layers. The database role is read-only
and the transaction is opened READ ONLY, so a write that somehow got past this parser would
still be refused; a statement timeout and a row cap bound the cost of an expensive read. Each
layer fails in a different way, which is the point of having all three.

Fails closed: anything unparseable, empty, or unrecognised is rejected. A wrong rejection
costs one retry. A wrong acceptance costs the database.
"""

from __future__ import annotations

import sqlglot
from sqlglot import expressions as exp

DIALECT = "postgres"

# Only these node types may be the root of a statement. Everything else — every DML, DDL,
# and permission statement — is rejected by omission rather than by blacklist, so a SQL
# feature nobody thought of is refused rather than waved through.
_ALLOWED_ROOTS = (exp.Select, exp.Union, exp.Except, exp.Intersect, exp.Subquery)

# Node types that mutate, wherever they appear — including nested inside a CTE, which is how
# `WITH d AS (DELETE ... RETURNING id) SELECT * FROM d` disguises a write as a read.
_FORBIDDEN_ANYWHERE = (
    exp.Insert, exp.Update, exp.Delete, exp.Drop, exp.Create, exp.Alter,
    exp.TruncateTable, exp.Grant, exp.Merge, exp.Command,
)


class GuardRejection(Exception):
    """Rejected before execution. The message is fed back to the model to retry."""


def guard_sql(sql: str, *, row_limit: int = 200) -> str:
    """Return safe, bounded SQL, or raise GuardRejection explaining what was wrong.

    The returned string is re-generated from the parsed tree rather than string-patched, so
    what runs is what was inspected — there is no window for a comment or quoting trick to
    mean one thing to the checker and another to the database.
    """
    if not sql or not sql.strip():
        raise GuardRejection("empty statement")

    try:
        statements = [s for s in sqlglot.parse(sql, dialect=DIALECT) if s is not None]
    except Exception as e:
        raise GuardRejection(f"could not parse the SQL: {e}") from e

    if not statements:
        raise GuardRejection("no statement found")
    if len(statements) > 1:
        # A trailing semicolon parses to one statement, so this is genuine stacking.
        raise GuardRejection(
            f"{len(statements)} statements sent; exactly one read is allowed"
        )

    tree = statements[0]

    for node in tree.walk():
        if isinstance(node, _FORBIDDEN_ANYWHERE):
            raise GuardRejection(
                f"{type(node).__name__.upper()} is not permitted; this connection is read-only"
            )

    if not isinstance(tree, _ALLOWED_ROOTS):
        raise GuardRejection(
            f"only SELECT statements are permitted, got {type(tree).__name__.upper()}"
        )

    _apply_row_limit(tree, row_limit)
    return tree.sql(dialect=DIALECT)


def _apply_row_limit(tree: exp.Expression, row_limit: int) -> None:
    """Force a row ceiling. An existing smaller limit is respected; a larger one is capped.

    Without this a single unintended cross join can pull a table's worth of rows into a
    model prompt — slow, expensive, and it drowns the answer.
    """
    existing = tree.args.get("limit")
    if existing is not None:
        try:
            asked = int(existing.expression.this)
        except (AttributeError, TypeError, ValueError):
            asked = None
        if asked is not None and asked <= row_limit:
            return
    tree.set("limit", exp.Limit(expression=exp.Literal.number(row_limit)))
