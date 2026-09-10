"""Direct data access: the model writes the SQL, we run it read-only.

This is the ONE thing that differs between the two variants. The platform variant sends a
plain-English question to `resolve_context` and gets rows back, with the platform doing its
own schema mapping (and spending its own model tokens doing it). Here the model gets three
read-only tools and works the schema out itself.

TWO DESIGN CHOICES WORTH KNOWING
--------------------------------
1. It returns the SAME shape as the platform path — a JSON array of row dicts on
   `_parsed_text`. `_parse_mcp_text` normalises the platform's reply to exactly that, and
   every downstream node (enrichment, generation, verification, charting) consumes it. Match
   the contract and the rest of the agent needs no changes, which is what keeps the
   comparison honest: two ways of fetching data, not two different agents.

2. The loop is written against the LangChain interface, not against one provider's SDK. So
   it runs on whatever `LLM_PROVIDER` selects and does not need rewriting when the model
   changes — which matters, because it is running on OpenAI today and Claude later.

A tool loop rather than one-shot text-to-SQL is a deliberate cost/robustness trade: it
handles an unfamiliar schema and corrects its own SQL errors, at the price of a variable
number of model calls per turn. `max_roundtrips` is the ceiling on that, because the token
cost of this variant is the very thing being measured.
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Any, Optional, Protocol

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from app.resolver.sql_guard import GuardRejection, guard_sql

logger = logging.getLogger(__name__)

MAX_ROUNDTRIPS_DEFAULT = int(os.getenv("SQL_MAX_TOOL_ROUNDTRIPS", "6"))
ROW_LIMIT_DEFAULT = int(os.getenv("SQL_ROW_LIMIT", "200"))

SYSTEM_PROMPT = """You answer questions about a wealth-management database by querying it.

You have three tools. Prefer to answer in as few queries as possible.
- list_tables: the tables available
- describe_table: the columns of one table
- run_query: run ONE read-only SELECT and get rows back

Rules that are enforced in code, not just here — breaking them wastes a turn:
- SELECT only. No INSERT, UPDATE, DELETE, DDL, or multiple statements.
- One statement per call. A semicolon-separated second statement is rejected.
- A row limit is applied automatically; you do not need to add one.

DO THE ARITHMETIC IN SQL, NOT IN YOUR HEAD. Totals, counts, averages, rankings and
comparisons must be written as SUM/COUNT/AVG/ORDER BY so the database computes them.
Never fetch rows in order to add them up yourself: the automatic row limit may have
removed rows you never saw, so your total would be wrong with nothing to indicate it.
If a result is reported as possibly truncated, do not compute over it — rewrite the query
to aggregate in SQL.

If a query is rejected or errors, read the reason and correct the SQL. When you have the
rows you need, stop calling tools and state the answer plainly."""


class Database(Protocol):
    """The read-only surface the resolver needs. Injected so it can be faked in tests."""

    async def list_tables(self) -> list[str]: ...
    async def describe_table(self, table: str) -> list[dict]: ...
    async def run_query(self, sql: str) -> list[dict]: ...


def _tool_specs() -> list[dict]:
    """Tool schemas in the neutral JSON-schema form both providers accept."""
    return [
        {
            "name": "list_tables",
            "description": "List the tables in the database.",
            "input_schema": {"type": "object", "properties": {}, "additionalProperties": False},
        },
        {
            "name": "describe_table",
            "description": "List the columns and types of one table.",
            "input_schema": {
                "type": "object",
                "properties": {"table": {"type": "string"}},
                "required": ["table"],
                "additionalProperties": False,
            },
        },
        {
            "name": "run_query",
            "description": "Run one read-only SELECT and return the rows.",
            "input_schema": {
                "type": "object",
                "properties": {"sql": {"type": "string"}},
                "required": ["sql"],
                "additionalProperties": False,
            },
        },
    ]


def _empty(question: str, error: str, latency_ms: int) -> dict:
    return {
        "tool": "direct_sql",
        "args": {"query": question},
        "success": False,
        "error": error,
        "_parsed_text": "",
        "queries": [],
        "roundtrips": 0,
        "latency_ms": latency_ms,
    }


async def resolve_direct(
    question: str,
    *,
    llm: Any = None,
    db: Optional[Database] = None,
    max_roundtrips: Optional[int] = None,
    row_limit: Optional[int] = None,
) -> dict:
    """Answer `question` from the database, returning the platform path's result shape."""
    started = time.perf_counter()
    cap = max_roundtrips if max_roundtrips is not None else MAX_ROUNDTRIPS_DEFAULT
    limit = row_limit if row_limit is not None else ROW_LIMIT_DEFAULT

    if db is None:
        # Explicit, not silent. An unconfigured database that returns "no rows" is
        # indistinguishable from a question with no answer, and the agent would
        # confidently report emptiness as fact.
        return _empty(question, "the database is not configured (NEON_DATABASE_URL is unset)",
                      round((time.perf_counter() - started) * 1000))

    if llm is None:
        from app.llm import make_llm

        llm = make_llm("fast", temperature=0.0, max_tokens=1500)

    model = llm.bind_tools(_tool_specs())
    messages: list = [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=question)]

    rows: list[dict] = []
    queries: list[str] = []
    roundtrips = 0

    while roundtrips < cap:
        reply: AIMessage = await model.ainvoke(messages)
        messages.append(reply)

        calls = getattr(reply, "tool_calls", None) or []
        if not calls:
            break  # the model is done asking and has stated its answer

        roundtrips += 1
        for call in calls:
            content = await _run_tool(call, db, limit, queries, rows)
            messages.append(
                ToolMessage(content=content, tool_call_id=call.get("id") or "call")
            )

    latency_ms = round((time.perf_counter() - started) * 1000)
    return {
        "tool": "direct_sql",
        "args": {"query": question},
        "success": bool(rows),
        "error": None if rows else "no rows returned",
        # The contract: a JSON array of row dicts, exactly what _parse_mcp_text produces
        # for the platform path.
        "_parsed_text": json.dumps(rows, default=str) if rows else "",
        "queries": queries,
        "roundtrips": roundtrips,
        "latency_ms": latency_ms,
    }


async def _run_tool(call: dict, db: Database, row_limit: int,
                    queries: list[str], rows: list[dict]) -> str:
    """Execute one tool call, returning the text handed back to the model.

    Every failure is returned AS TEXT rather than raised: the model can correct a stated
    problem, and a raised exception ends the turn with nothing to show for the tokens
    already spent.
    """
    name = call.get("name")
    args = call.get("args") or {}

    try:
        if name == "list_tables":
            return json.dumps(await db.list_tables())

        if name == "describe_table":
            return json.dumps(await db.describe_table(args.get("table", "")), default=str)

        if name == "run_query":
            try:
                safe_sql = guard_sql(args.get("sql", ""), row_limit=row_limit)
            except GuardRejection as e:
                logger.warning("SQL rejected by guard: %s", e)
                return f"REJECTED: {e}. Rewrite it as a single read-only SELECT."
            queries.append(safe_sql)
            result = await db.run_query(safe_sql)
            rows.clear()
            rows.extend(result)
            payload = json.dumps(result, default=str)
            if len(result) >= row_limit:
                # A result that exactly fills the cap was probably cut short. Saying so is
                # the difference between a visible limitation and a confidently wrong
                # total — summing 200 of 505 rows understates the answer by half and looks
                # entirely plausible. Mechanical, not a prompt rule: it fires on the count.
                return (
                    f"{payload}\n\nWARNING: exactly {row_limit} rows returned — the row "
                    f"limit was reached, so this result is probably INCOMPLETE. Do not "
                    f"total or rank these rows yourself. Rewrite the query to aggregate "
                    f"in SQL (SUM/COUNT/AVG/ORDER BY) so the database computes it."
                )
            return payload

        return f"unknown tool {name!r}"
    except Exception as e:  # noqa: BLE001 — surfaced to the model, not swallowed
        logger.warning("tool %s failed: %s", name, e)
        return f"ERROR: {e}"
