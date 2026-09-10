"""The direct resolver: the one thing that differs between the two variants.

Variant A asks the platform a question in English and gets rows back. This variant hands the
model three read-only tools and lets it work the schema out itself, looping until it has an
answer.

The contract is the important part. `_parse_mcp_text` normalises the platform's reply into a
JSON array of row dicts, and everything downstream — enrichment, generation, verification,
charting — consumes that shape. Returning the SAME shape is what keeps the rest of the agent
untouched, so the comparison measures two ways of fetching data rather than two different
agents.

The model and the database are both injected. That is not only for testability: the loop is
written against the LangChain interface, so it runs on whichever provider `LLM_PROVIDER`
selects and does not have to be rewritten when the model changes.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.resolver.direct import MAX_ROUNDTRIPS_DEFAULT, resolve_direct  # noqa: E402


class FakeDB:
    """Stands in for Neon. Records the SQL it was asked to run."""

    def __init__(self, rows=None, error=None):
        self.rows = rows if rows is not None else [{"name": "Ram Krishnan", "aum": 120}]
        self.error = error
        self.executed: list[str] = []

    async def list_tables(self):
        return ["clients", "holdings"]

    async def describe_table(self, table):
        return [{"column": "name", "type": "text"}, {"column": "aum", "type": "numeric"}]

    async def run_query(self, sql):
        self.executed.append(sql)
        if self.error:
            raise RuntimeError(self.error)
        return self.rows


class FakeLLM:
    """Replays a scripted sequence of assistant turns, like a model would produce."""

    def __init__(self, turns):
        self._turns = list(turns)
        self.calls = 0
        self.bound_tools = None
        self.seen_tool_results: list[str] = []

    def bind_tools(self, tools, **kw):
        self.bound_tools = [getattr(t, "name", None) or t["name"] for t in tools]
        return self

    async def ainvoke(self, messages, **kw):
        self.calls += 1
        for m in messages:
            if m.__class__.__name__ == "ToolMessage":
                self.seen_tool_results.append(str(m.content))
        return self._turns.pop(0) if self._turns else AIMessage(content="done")


def _query_turn(sql, call_id="c1"):
    return AIMessage(
        content="",
        tool_calls=[{"name": "run_query", "args": {"sql": sql}, "id": call_id,
                     "type": "tool_call"}],
    )


class TestItReturnsTheSameShapeTheAgentAlreadyConsumes:
    async def test_rows_come_back_as_a_json_array(self):
        db = FakeDB()
        llm = FakeLLM([_query_turn("SELECT name, aum FROM clients"),
                       AIMessage(content="Ram Krishnan has 120.")])

        result = await resolve_direct("Who has the most AUM?", llm=llm, db=db)

        assert result["success"] is True
        parsed = json.loads(result["_parsed_text"])
        assert parsed == [{"name": "Ram Krishnan", "aum": 120}], (
            "downstream nodes expect a JSON array of row dicts, the same shape the "
            "platform path produces"
        )

    async def test_the_sql_actually_run_is_reported(self):
        """The UI shows what was asked of the data platform; this variant must too."""
        db = FakeDB()
        llm = FakeLLM([_query_turn("SELECT name FROM clients"), AIMessage(content="ok")])

        result = await resolve_direct("names?", llm=llm, db=db)

        assert any("SELECT" in q.upper() for q in result["queries"])

    async def test_the_model_is_given_the_three_read_only_tools(self):
        db = FakeDB()
        llm = FakeLLM([AIMessage(content="no tools needed")])

        await resolve_direct("hi", llm=llm, db=db)

        assert set(llm.bound_tools) == {"list_tables", "describe_table", "run_query"}


class TestTheGuardIsWiredIn:
    async def test_a_write_never_reaches_the_database(self):
        db = FakeDB()
        llm = FakeLLM([_query_turn("DELETE FROM clients"), AIMessage(content="sorry")])

        await resolve_direct("delete everything", llm=llm, db=db)

        assert db.executed == [], "a rejected statement must not be executed"

    async def test_the_rejection_reason_goes_back_to_the_model(self):
        """A silent failure wastes the turn; the model can fix a stated problem."""
        db = FakeDB()
        llm = FakeLLM([_query_turn("DELETE FROM clients"),
                       _query_turn("SELECT name FROM clients"),
                       AIMessage(content="ok")])

        await resolve_direct("remove them", llm=llm, db=db)

        assert any("read-only" in r.lower() or "not permitted" in r.lower()
                   for r in llm.seen_tool_results)

    async def test_a_row_limit_is_forced_onto_the_query(self):
        db = FakeDB()
        llm = FakeLLM([_query_turn("SELECT name FROM clients"), AIMessage(content="ok")])

        await resolve_direct("names", llm=llm, db=db, row_limit=25)

        assert "LIMIT 25" in db.executed[0].upper()


class TestItCannotRunAway:
    async def test_the_loop_stops_at_the_cap(self):
        """A confused model that keeps querying must not bill unboundedly — the cost of
        this variant is the thing being measured."""
        db = FakeDB()
        llm = FakeLLM([_query_turn(f"SELECT {i} FROM clients", f"c{i}") for i in range(50)])

        result = await resolve_direct("loop", llm=llm, db=db, max_roundtrips=3)

        assert llm.calls <= 3 + 1
        assert len(db.executed) <= 3
        assert result["roundtrips"] <= 3

    async def test_the_default_cap_is_finite(self):
        assert 0 < MAX_ROUNDTRIPS_DEFAULT < 20


class TestItFailsLoudly:
    async def test_a_database_error_is_reported_not_swallowed(self):
        db = FakeDB(error="connection refused")
        llm = FakeLLM([_query_turn("SELECT name FROM clients"), AIMessage(content="ok")])

        result = await resolve_direct("names", llm=llm, db=db)

        assert any("connection refused" in r for r in llm.seen_tool_results)

    async def test_no_database_configured_is_an_explicit_failure(self):
        llm = FakeLLM([AIMessage(content="x")])

        result = await resolve_direct("names", llm=llm, db=None)

        assert result["success"] is False
        assert "not configured" in (result["error"] or "").lower()


class TestTruncationIsNeverSilent:
    """The row cap can turn a correct query into a confidently wrong answer.

    The cap exists so one bad join cannot pull a table into a prompt. But if the model
    fetches rows intending to total them itself, the cap removes rows and says nothing —
    and summing 200 of 505 holdings understates the real figure by nearly half while
    looking entirely plausible. A wrong number that looks right is worse than an error.

    So a result that exactly fills the cap is reported as possibly incomplete, and the
    model is told to aggregate in SQL instead.
    """

    async def test_a_full_page_is_flagged_as_possibly_truncated(self):
        db = FakeDB(rows=[{"v": i} for i in range(5)])
        llm = FakeLLM([_query_turn("SELECT v FROM t"), AIMessage(content="ok")])

        await resolve_direct("total?", llm=llm, db=db, row_limit=5)

        assert any("truncat" in r.lower() or "incomplete" in r.lower()
                   for r in llm.seen_tool_results), (
            "the model was handed a capped result set with no indication it was capped"
        )

    async def test_a_short_result_is_not_flagged(self):
        """A genuine 3-row answer must not be second-guessed."""
        db = FakeDB(rows=[{"v": i} for i in range(3)])
        llm = FakeLLM([_query_turn("SELECT v FROM t"), AIMessage(content="ok")])

        await resolve_direct("total?", llm=llm, db=db, row_limit=5)

        assert not any("truncat" in r.lower() for r in llm.seen_tool_results)

    async def test_the_prompt_tells_the_model_to_aggregate_in_sql(self):
        from app.resolver.direct import SYSTEM_PROMPT
        low = SYSTEM_PROMPT.lower()
        assert "sum(" in low or "aggregate" in low, (
            "nothing instructs the model to let the database do the arithmetic"
        )
