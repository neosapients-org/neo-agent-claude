"""The SQL guard, and why it is code rather than a sentence in a prompt.

This variant hands SQL-writing to a language model and then runs the result against a real
database. "We told the model to only read" is not a control — it is a hope. Every request
therefore goes through a parser that rejects anything that is not a single, bounded read,
before the database ever sees it.

Defence in depth, because each layer fails differently:
  - this guard    — rejects the statement outright (a bug here is a wrong rejection)
  - a read-only role + read-only transaction — the database refuses writes (a bug here is
    a privilege misconfiguration)
  - a row cap and a statement timeout — bounds the damage of an accidental cross join

The tests below are adversarial on purpose. A guard that only handles the polite cases is
worse than none, because it invites trust it has not earned.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.resolver.sql_guard import GuardRejection, guard_sql  # noqa: E402


class TestItOnlyAllowsReads:
    @pytest.mark.parametrize("sql", [
        "INSERT INTO clients (name) VALUES ('x')",
        "UPDATE clients SET name = 'x'",
        "DELETE FROM clients",
        "DROP TABLE clients",
        "TRUNCATE clients",
        "ALTER TABLE clients ADD COLUMN x int",
        "CREATE TABLE t (a int)",
        "GRANT SELECT ON clients TO PUBLIC",
    ])
    def test_a_write_is_rejected(self, sql):
        with pytest.raises(GuardRejection):
            guard_sql(sql)

    def test_a_plain_select_is_allowed(self):
        assert "SELECT" in guard_sql("SELECT name FROM clients").upper()

    def test_a_cte_read_is_allowed(self):
        out = guard_sql("WITH t AS (SELECT 1 AS a) SELECT a FROM t")
        assert "SELECT" in out.upper()

    def test_a_write_hidden_inside_a_cte_is_rejected(self):
        """`WITH ... AS (DELETE ... RETURNING ...)` is a write that reads like a read."""
        with pytest.raises(GuardRejection):
            guard_sql("WITH d AS (DELETE FROM clients RETURNING id) SELECT id FROM d")


class TestItRejectsStatementStacking:
    def test_a_second_statement_is_rejected(self):
        """The classic escape: a benign read, a semicolon, then anything at all."""
        with pytest.raises(GuardRejection):
            guard_sql("SELECT 1; DROP TABLE clients")

    def test_a_trailing_semicolon_alone_is_fine(self):
        assert guard_sql("SELECT name FROM clients;")

    def test_a_commented_out_second_statement_is_still_one_statement(self):
        assert guard_sql("SELECT name FROM clients -- ; DROP TABLE clients")


class TestItBoundsTheResult:
    def test_a_missing_limit_is_added(self):
        out = guard_sql("SELECT name FROM clients", row_limit=200)
        assert "LIMIT 200" in out.upper()

    def test_a_smaller_limit_is_left_alone(self):
        out = guard_sql("SELECT name FROM clients LIMIT 5", row_limit=200)
        assert "LIMIT 5" in out.upper()
        assert "LIMIT 200" not in out.upper()

    def test_a_larger_limit_is_capped(self):
        """A model asking for 100k rows must not get them just because it asked."""
        out = guard_sql("SELECT name FROM clients LIMIT 100000", row_limit=200)
        assert "LIMIT 200" in out.upper()
        assert "100000" not in out


class TestItFailsClosed:
    def test_unparseable_sql_is_rejected(self):
        with pytest.raises(GuardRejection):
            guard_sql("SELECT FROM WHERE ((((")

    def test_empty_input_is_rejected(self):
        with pytest.raises(GuardRejection):
            guard_sql("   ")

    def test_the_rejection_says_why(self):
        """The reason is fed back to the model so it can correct itself."""
        with pytest.raises(GuardRejection) as e:
            guard_sql("DELETE FROM clients")
        assert str(e.value)
