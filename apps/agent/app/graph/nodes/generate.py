"""Response generation node — streams LLM response."""

import json
import re
import time

try:
    from ns_probe import observe
except ImportError:
    def observe(*a, **kw):
        def _id(fn): return fn
        return _id

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

from app.llm import make_llm
from ...config import config
from ..state import AgentState
from ...chainlog import log_chain, make_cid
from ...money import annotate_money
from ...aggregate import coerce_rows, build_aggregate_block, summarize_holdings, summarize_by_category
from ...templates import apply_template, render_table_answer

SYSTEM_PROMPT = """You are a wealth management assistant. Format the data below into a clear, readable response.

RULE PRECEDENCE (read first):
- There are two kinds of rules below. ACCURACY rules — exact counts, monetary figures and crore/lakh
  formatting, using the AUTHORITATIVE AGGREGATES verbatim, showing every row, and never fabricating
  any value — are ABSOLUTE. Never break them, not even to satisfy a preference.
- PRESENTATION rules (table vs prose, bullets vs numbered list, column layout, tone, length) are
  DEFAULTS. If the "User context" section or session preferences specify a different format or tone,
  FOLLOW the preference — provided no ACCURACY rule is broken (e.g. a brevity preference may tighten
  wording but must never drop a row, a figure, or change a number).
- "User context" may also carry CONSTRAINTS to honor (e.g. exclusions, restricted product types), not
  just suitability inputs. Treat any such constraint as binding.

Rules:
- By DEFAULT, present multi-record data using MARKDOWN TABLES (with | header | separators |) — tables for portfolio holdings, investment lists, investor comparisons, and ANY data with 2+ records — UNLESS the User context or a session preference asks for a different format (e.g. plain prose), in which case follow that while still including every row.
- By DEFAULT do NOT use bullet points for lists of investors or multi-row data — prefer a table unless a preference specifies otherwise.
- Use bullet points ONLY for single-value answers (e.g. one investor's health score) or brief narrative summaries.
- Show monetary values with the ₹ (Indian Rupee) symbol. Never use the $ symbol. Use the COMPACT crore/lakh form (see below), not the full digit-grouped number.
- ₹ APPLIES ONLY TO ACTUAL MONEY. A Rank, Ranking, Position, S.No, Sr. No, "#", serial, sequence, index, or any ordinal/counter column holds plain integers (1, 2, 3, …) — NEVER prefix these with ₹ or any currency symbol, and never crore/lakh-format them. Likewise counts, quantities/units, percentages, ratios, years/dates and interest rates are NOT money. Apply ₹ only to columns that are genuinely monetary amounts (those carrying a "(₹)" companion value, or clearly money such as value/AUM/cost/invested/gain/loss/NAV/price).
- PRE-FORMATTED AMOUNTS: monetary columns in the Data may include a companion column named "<col> (₹)" that already holds the exact ₹ amount with the correct crore/lakh equivalent. When a companion "(₹)" value is present, use it VERBATIM for display and do NOT re-group or re-scale the raw number yourself (re-grouping causes 10x errors). The raw numeric column is only for arithmetic you must perform. Show only ONE column per amount in your table — the formatted "(₹)" value — do NOT also output the raw numeric "<col>" column next to it.
- AUTHORITATIVE AGGREGATES: the Data may be followed by an "AUTHORITATIVE AGGREGATES" block holding sums/subtotals that were computed in code (they are EXACT). Whenever you report a total, sum, or combined figure, copy the matching number from that block VERBATIM. NEVER add the rows up yourself — your mental arithmetic is unreliable and produces wrong totals. If the block is absent and a total is genuinely required, you may sum, but a provided figure always wins.
- State amounts COMPACTLY in Indian terms, choosing the RIGHT unit by magnitude (do NOT also print the full digit-grouped number):
  • ₹1 crore and above → crore, e.g. "₹16.63 crore", "₹77.23 crore".
  • below ₹1 crore but ₹1 lakh and above → LAKH, NOT crore, e.g. "₹63.24 lakh", "₹40.42 lakh". Do NOT write "₹0.63 crore" — use lakh for anything under a crore.
  • below ₹1 lakh → the exact ₹ figure, e.g. "₹45,230.00".
- Be concise but COMPLETE.
- Do NOT include record counts, row numbers, or metadata headers as separate bullet points.
- If a tool call failed, say so briefly and suggest the user rephrase as a more specific query.
- NEVER ask the user to send, provide, or share data with you — you are a read-only assistant.
- ALWAYS present ALL data that IS available. Only say "could not be retrieved" if the data section is completely empty or contains an explicit error message like "Context resolution failed".
- If data has columns with values, present those values — do NOT claim the data is incomplete or unavailable.
- The data returned is ALREADY filtered/scoped by the platform to the client, fund, or entity named in the question. TRUST that scoping: if the question names a client and the data contains values (e.g. current values, amounts, returns), those values ARE that client's — present and aggregate them. Do NOT say "no records for <client>" or "does not have data" merely because the rows do not repeat the client's name or the "Client Name"/"Id" column is blank. Blank label columns are normal; the values are still valid.
- When the question asks for a single total/value (e.g. "portfolio value", "net worth", "total investment", "combined AUM") and the data is a LIST of values/rows for that client or group, report ONE total — taken from the AUTHORITATIVE AGGREGATES block when present — do NOT report just the first row's value (e.g. "Ram Krishnan's portfolio is currently worth ₹X" = the provided sum of ALL the Current Value rows, not one holding).
- DIRECT-ANSWER for aggregate questions: when the user asks for a combined / total / overall figure (e.g. "combined AUM of the Krishnan family"), give the TOTAL as a single headline sentence (with the ₹ amount and its crore/lakh equivalent). Do NOT enumerate every underlying client/account/snapshot row — at most add one short clarifying line. Only show a full table when the user explicitly asks to "list", "show all", "break down", or "by <dimension>".
- BREAKDOWN / ALLOCATION questions: present EACH category exactly as it appears in the data (e.g. one row per product type / asset class the data returned, with its own value). Do NOT merge several returned categories into one different bucket, and do NOT add or sum across categories that the data did not already combine. If the user names categories (e.g. "Equity, Debt, Mutual Funds") that do not exactly match the data's categories, show the data's ACTUAL categories and values as given — never re-bucket or re-total them yourself (that produces wrong subtotals).
- PARTIAL data: if the data covers SOME of what was asked (e.g. one of two clients/funds in a comparison), LEAD with the data you DO have (as a table), then add ONE short line noting what part was unavailable. NEVER open with "could not retrieve" when any relevant data is present.
- NEGATIVE answers are valid answers, not failures: if the data shows zero matching records (e.g. no ETFs, no bonds maturing, a count of 0), state that directly and confidently ("Ram Krishnan holds no ETFs.") — do NOT ask the user to rephrase.
- A value of "-" or blank in a column means that field is not populated in the source data; report the rest normally and, only if the user specifically asked for that field, note briefly that it isn't available.
- Ambiguous names: If data has multiple matching investors, list the full names and ask which one.
- NEVER invent or hallucinate investor names. Only mention names in the data.
- NO FABRICATION OF ANY VALUE: never invent or alter fund/scheme names, product categories, instrument names, numbers, dates, or rows. Use ONLY values present in the Data section, exactly as given (do not rename "HDFC Top 100" to "Mutual Fund A", do not add a category like "Upfront Commission" that is not in the data). If the data does not contain something the question asks for, say it is not available — never substitute a plausible-looking or placeholder value.
- Present EVERY row from the Data section and ONLY those rows. Do not add rows that are not in the data, and do not drop rows that are.
- If data shows exactly one match, respond directly without clarifying.
- For portfolio/holdings data: use a table with columns like Investment Name, Type, Sector, Segment, Purchase Date, Cost, Current Value.
- GAIN/LOSS FILTERING (you must compute this yourself — the data is the FULL holdings list, already containing each holding's gain/loss): when the user asks which holdings are in a LOSS position / underwater / losing / "in the red", select ONLY the rows whose unrealized gain (or current value − invested/cost) is NEGATIVE. When they ask which are in PROFIT / gaining / "in the green", select ONLY the rows with POSITIVE gain. Present just the matching rows as a table (Investment Name, Invested/Cost, Current Value, Unrealized Gain/Loss, Gain/Loss %). If the user asks for "best/worst" or "top/bottom" performers, sort by gain % and show the relevant end. If NO holding matches (e.g. none are in loss), say so directly ("All of <name>'s holdings are currently in profit; none are in a loss position.") — do NOT say the data could not be retrieved.
- For summary metrics (total cost, returns, etc.): present as a separate small table with Metric and Value columns.

CLIENT / DATA NOTES:
- The data comes from the agent wealth-management platform. Clients belong to families:
  Krishnan, Mehra, Patel, Choudhary, and Fernandes.
- The data already contains human-readable client names (e.g. "Ram Krishnan"). Use the names as they appear.
- If a row only has a numeric client id (e.g. 7001) and a separate "Client Name" column is present,
  pair them up by position. Do NOT invent a name for an id you cannot resolve from the data — show the id.

COUNTING & AGGREGATION RULES (critical):
- When data contains a list of records: ALWAYS state the EXACT total count (e.g. "Found 12 investors matching...")
- Count EVERY record in the data — do NOT estimate or round.
- If asked "how many" and data shows a count value, use that EXACT number as your answer.
- ALWAYS list ALL records in the data as a markdown table. NEVER truncate, summarize, or skip rows.
- Even if there are 50, 80, or 100 records — put EVERY SINGLE ONE in the table.
- Do NOT say "and N more" or "remaining records omitted". Show the complete table.
- NEVER say a number unless it matches the data exactly. 73 means 73, not "about 70" or "15".
- For monetary values: use exact amounts from data, don't round unless explicitly asked.
- When data has a "count" field with a number, that IS the answer — report it directly.
- SUITABILITY / PREFERENCE MATCH: if the question asks whether something is "a good fit for me",
  "suits my style", or whether a client's holdings "match what I prefer", use the FACTS in the data
  section together with the user's profile/preferences in the "User context" section to judge it.
  Give a clear verdict (good fit / partial / not a fit, or which holdings match vs don't) with a brief
  reason grounded in their stated preferences. NEVER treat any number in User context as live data —
  all figures come from the data section only. If User context has no relevant preference, say what
  you'd need to know rather than guessing."""

SYSTEM_PROMPT_FROM_MEMORY = """You are a wealth management assistant recalling saved notes.
Answer using TWO sources that may be provided:
  1. The conversation history above (data shown earlier this session), and
  2. The "User context" section — long-term saved notes about the advisor and their clients.

CONTENT RULES — ABSOLUTE. These govern WHAT you may say and can never be relaxed,
not even by a session/user preference:
- Quote ONLY what is explicitly written in the User context or conversation history. Word-for-word.
- DO NOT infer, deduce, elaborate, or expand on what is written. If the note says "conservative investor who prefers government bonds", do NOT add "therefore excludes other asset classes" — that is inference, not a note.
- DO NOT rephrase facts into implications. Report exactly what was noted, nothing more.
- List EVERY relevant stored fact. NEVER drop, merge, or omit a fact in order to be shorter — completeness of recall always wins over brevity.
- If the user asks for something that is NOT in the notes, say "I don't have a note on that" — never fill the gap with a logical conclusion.
- NEVER hallucinate or invent names, preferences, numbers, or categories.
- If NEITHER source has anything about the client/topic, say you'll need to look it up live.

FILTERING / RANKING A LIST SHOWN EARLIER — when the user selects over rows just shown
("which of those are X", "of those, which have the highest Y", "rank those by Z"):
- Operate ONLY on the rows shown earlier this session — never invent rows, and never fetch new ones.
- Apply the user's predicate to those exact rows: filter, sort, or select as asked.
- You MAY classify an item from what is plainly shown (e.g. a listed company/stock is an equity,
  NOT a mutual fund) — but NEVER invent a numeric value or an attribute that wasn't shown.
- If NONE of the shown rows match, say so plainly (e.g. "None of those are mutual funds — they're
  equity stocks").
- If the predicate needs a figure/attribute that is NOT in the shown rows (e.g. XIRR when only
  market value was shown), do NOT guess it — say it isn't in what you have for those rows and ask
  whether to pull it fresh (e.g. "XIRR isn't in what I have for those — want me to pull it?").

PRESENTATION — governs only HOW you format the answer; session preferences (if any,
shown below) take priority here:
- DEFAULT to bullet points, one bullet per stored fact, verbatim.
- If session preferences specify a different format or tone (e.g. a numbered list, plain prose, a particular style), follow them — but keep every fact (see content rules).
- A length/brevity preference may tighten the wording of each fact, but must NOT cause any fact to be dropped or combined away."""

# Max chars of MCP data to send to LLM (prevents token bloat)
# 60K allows ~100 records with many columns to pass through untruncated
_MAX_MCP_DATA_CHARS = 60000


def _table_start_index(text: str) -> int:
    """Character index where the first markdown-table line begins (a line whose first non-space
    char is '|'), or -1 if none yet. Used to stop live-streaming once a table starts."""
    idx = 0
    for line in text.split("\n"):
        if line.lstrip().startswith("|"):
            return idx
        idx += len(line) + 1  # +1 for the '\n' that split() removed
    return -1


def _sources_footer(results: list, answer_text: str) -> str:
    """A deterministic, numbered "Sources" block for the live web results used this turn.

    The LLM is told to cite sources inline as [n], but it does so inconsistently (sometimes
    none, sometimes a subset). So we ALWAYS append the full list in code — numbered [1..n] to
    match format_web_context's ordering, so any inline [n] the model DID emit lines up. Returns
    "" when there are no results or the answer already ends with its own Sources section."""
    if not results:
        return ""
    # Don't double up if the model already produced a Sources/References heading.
    if re.search(r"(^|\n)\s*\**\s*(sources?|references?)\b\s*:?\**\s*(\n|$)", answer_text or "", re.IGNORECASE):
        return ""
    lines = ["\n\n**Sources**"]
    seen = set()
    i = 0
    for r in results:
        url = (r.get("url") or "").strip()
        title = (r.get("title") or "").strip() or url or "Source"
        if not url or url in seen:
            continue
        seen.add(url)
        i += 1
        lines.append(f"{i}. [{title}]({url})")
    return "\n".join(lines) if i else ""


def _data_is_empty(text: str) -> bool:
    """True when an MCP payload carries no usable data (empty list, `rowCount: 0` field
    dump, all-null rows). Used to decide whether to offer a name clarification (issue #1A)."""
    s = re.sub(r'^\[\d+\]', '', (text or "").strip()).strip()
    if s in ("", "[]", "{}", "[ ]", "null", "None"):
        return True
    if re.search(r'rowcount:\s*0\b', s, re.IGNORECASE):
        return True
    rows = coerce_rows(s)
    if rows is not None:
        if not rows:
            return True
        return all(
            isinstance(r, dict) and all(v in (None, "", "null", "-") for v in r.values())
            for r in rows
        )
    return False


def _smart_truncate(data_text: str, max_chars: int = _MAX_MCP_DATA_CHARS) -> str:
    """Truncate MCP data intelligently — preserve record boundaries and add count."""
    if len(data_text) <= max_chars:
        return data_text

    # Count total records for context
    total_records = data_text.count('"investor_id"') or data_text.count('\n')

    # For JSON arrays, find the last complete object before limit
    truncation_point = max_chars
    if data_text.strip().startswith("["):
        last_boundary = data_text.rfind("},", 0, max_chars)
        if last_boundary > max_chars * 0.5:
            truncation_point = last_boundary + 1
    else:
        last_newline = data_text.rfind("\n", 0, max_chars)
        if last_newline > max_chars * 0.7:
            truncation_point = last_newline

    shown_records = data_text[:truncation_point].count('"investor_id"') or data_text[:truncation_point].count('\n')

    suffix = f"\n...(showing {shown_records} of {total_records} total records — data truncated)"
    return data_text[:truncation_point] + suffix


def _name_clarification_if_empty(state: AgentState, query: str, all_empty: bool) -> str | None:
    """Issue #1A, Part 1: if the platform returned NOTHING for a question about a specific
    person whose name we could NOT confidently resolve to a known client, ask for the correct
    name instead of confidently reporting "no holdings". A name that DID resolve to a real
    client and still came back empty is a genuine negative — we leave that to the answer path.
    """
    if not all_empty:
        return None
    # Lazy import avoids any import-time cycle with intent_enrichment.
    from .intent_enrichment import (
        _resolve_client_name, _query_is_multi_investor, _INVESTOR_ID_PATTERN,
        _known_investors_lower,
    )
    entities = state.get("enrichment_entities") or {}
    named = (entities.get("investor_name") or "").strip()
    if not named or _query_is_multi_investor(query) or _INVESTOR_ID_PATTERN.match(named):
        return None
    if named.lower() in _known_investors_lower():
        return None  # a real client with genuinely no matching rows -> normal negative answer
    canonical, candidates = _resolve_client_name(named)
    if candidates:
        listed = ", ".join(candidates[:-1]) + " or " + candidates[-1] if len(candidates) > 1 else candidates[0]
        return f"I couldn't find a client matching “{named}”. Did you mean {listed}? Please confirm the full name."
    if canonical is None:
        return (f"I couldn't find a client named “{named}” in the platform. "
                f"Could you share the client's full name as it appears in agent?")
    return None  # canonical resolved -> the empty result is a real negative


def _session_pref_block(state: AgentState) -> str:
    """Render RM_SESSION_PREFERENCES (the preference-box text) as a high-priority prompt
    block. These are ephemeral, session-only delivery preferences typed by the RM; they
    OVERRIDE conflicting stored advisor preferences for this turn, but must NEVER override
    a client's hard constraint. Returns "" when the box is empty."""
    sp = (state.get("session_preferences") or "").strip()
    if not sp:
        return ""
    return (
        "\n\n## SESSION PREFERENCES (RM — highest priority, apply this turn)\n"
        "The RM set these for THIS session via the preference box. They govern HOW you "
        "respond (tone, length, format, jargon, detail). They OVERRIDE any conflicting "
        "advisor preference in 'User context'. EXCEPTION: a client's hard constraint or "
        "restriction (e.g. exclusions, prohibited/leveraged products, areas to avoid) "
        "ALWAYS wins — never relax a client constraint to satisfy these. Apply them "
        "silently; do not list them back unless asked.\n"
        f"{sp}"
    )


async def generate_response(state: AgentState) -> dict:
    """Generate the final LLM response using context from MCP tools or conversation history."""
    query = state["query"]
    mcp_results = state.get("mcp_tool_calls", [])
    memory_context = state.get("memory_context", "")
    # Quick Facts ignores STORED LTM preferences/advisor memories on a data answer — the
    # question is answered standalone. (Explicit recall still uses memory_context via the
    # from_memory path, and the session preference box still applies in every mode.)
    from ...modes import normalize_mode, QUICK_FACTS
    _is_quick_facts = normalize_mode(state.get("mode")) == QUICK_FACTS
    intent = state.get("intent", "")
    all_messages = state.get("messages", [])
    start = time.perf_counter()
    chart_spec = None  # populated in the MCP/data path when a chart fits
    _web_results: list = []  # live web sources used this turn (Deep Insight) — rendered as a
    # deterministic "Sources" footer so they ALWAYS show, regardless of whether the LLM cited them

    llm = make_llm("answer", temperature=0.0, max_tokens=16000)

    # ── acknowledge path: user stated a preference/persona fact ─────────────
    # No MCP, no data formatting — just confirm it's noted. The async store
    # pipeline persists it to long-term memory after this reply.
    if intent == "acknowledge":
        ack_sys = (
            "The user is telling you a preference or fact about themselves "
            "(e.g. risk tolerance, goals, how they want reports or to be contacted). "
            "Acknowledge it warmly in 1-2 sentences and confirm you'll remember it "
            "and apply it going forward. Do NOT fetch, mention, or ask for any "
            "financial data. Do NOT ask follow-up questions."
        )
        ack_sys += _session_pref_block(state)
        ack_resp = await llm.ainvoke(
            [SystemMessage(content=ack_sys), HumanMessage(content=query)]
        )
        latency_ms = round((time.perf_counter() - start) * 1000)
        return {
            "messages": [AIMessage(content=ack_resp.content)],
            "latency_breakdown": {**state.get("latency_breakdown", {}), "llm_ms": latency_ms},
        }

    # ── from_memory path: answer from conversation history ──────────────────
    # ONLY enter this path when intent_enrichment explicitly routed here.
    # A fetch_data intent with empty MCP results must go through the data path
    # (which handles errors via SYSTEM_PROMPT) — never silently fall back to
    # conversation history for investors not yet in session.
    if intent == "from_memory":
        sys_prompt = SYSTEM_PROMPT_FROM_MEMORY
        if memory_context:
            sys_prompt += f"\nUser context: {memory_context}"
        sys_prompt += _session_pref_block(state)

        # Pass prior messages (Human+AI) so LLM has the full context to answer from
        # Exclude the current human message (last in list) — we add it explicitly
        prior_messages = [
            m for m in all_messages[:-1]
            if hasattr(m, "type") and m.type in ("human", "ai") and m.content
        ]
        # Deduplicate consecutive AIMessages: mcp_fetch adds a raw-data AIMessage
        # and generate_response adds the formatted one — keep only the formatted (last) one
        deduped = []
        for i, m in enumerate(prior_messages):
            if (
                m.type == "ai"
                and i + 1 < len(prior_messages)
                and prior_messages[i + 1].type == "ai"
            ):
                continue  # skip raw MCP message, keep the formatted response
            deduped.append(m)
        # Keep last 4 messages (2 exchanges) to stay within token budget
        recent = deduped[-4:]

        messages = [SystemMessage(content=sys_prompt)] + recent + [HumanMessage(content=query)]

    # ── MCP / general path: answer from fetched data ─────────────────────────
    else:
        sys_prompt = SYSTEM_PROMPT
        if memory_context and not _is_quick_facts:
            sys_prompt += f"\nUser context: {memory_context}"
        sys_prompt += _session_pref_block(state)

        user_content = query
        # If the user typed a partial/misspelled client name that we resolved to a real
        # client (e.g. "Rama Krishna" -> "Ram Krishnan"), tell the answer step the canonical
        # name so the prose uses it — otherwise the model echoes the user's spelling.
        _ent = state.get("enrichment_entities") or {}
        _canon = (_ent.get("investor_name") or "").strip()
        # Only steer the answer toward a specific client on DATA questions. For general
        # chat / greetings (intent == "general") there is no "client in question" — the
        # investor_name here is just the logged-in fallback, so injecting it makes the
        # assistant awkwardly greet "How can I help with <client>'s portfolio?".
        if _canon and intent != "general" and _canon.lower() not in query.lower():
            user_content += (
                f"\n\nNote: the client in question is \"{_canon}\". Use this EXACT name in your "
                f"answer — the user may have typed it differently or misspelled it."
            )
        # Phase 3: a decomposed question was answered in parts — tell the model the blocks
        # below are sub-answers and how to combine them (the planner's synthesis instruction).
        _plan = state.get("query_plan") or {}
        if _plan.get("strategy") == "decompose":
            user_content += (
                "\n\nThis question was answered in PARTS — each \"Data for ...\" block below "
                "answers one sub-question. Combine them into one coherent answer; use the "
                "matching data for each part and do not confuse one block's figures for another's."
            )
            if _plan.get("synthesis"):
                user_content += f"\nHow to combine: {_plan['synthesis']}"

        # M3: apply the selected skill — inject its methodology + output format (Option A). For a
        # methodology skill that compiled to a plan, this reinforces the structure on top of the
        # per-block synthesis above.
        _skills = state.get("active_skills") or []
        if _skills:
            sk = _skills[0]
            user_content += (
                f"\n\nApply this skill — \"{sk.get('name', '')}\". Follow its methodology and "
                f"produce the answer in the structure it describes:\n{sk.get('content', '')}"
            )
            if sk.get("synthesis"):
                user_content += f"\n\nFinal assembly: {sk['synthesis']}"
        chart_spec = None
        det_answer = None  # a table built in CODE (no LLM) for plain holdings/fund/asset questions
        empties: list[bool] = []  # one per call; all_empty = every successful call was empty
        if mcp_results:
            for call in mcp_results:
                if call.get("success"):
                    # Use pre-parsed text if available (from mcp_fetch)
                    data_text = call.get("_parsed_text", "")
                    if not data_text:
                        response_data = call.get("response", {})
                        content_list = response_data.get("content", []) if isinstance(response_data, dict) else []
                        for item in content_list:
                            if isinstance(item, dict) and item.get("type") == "text":
                                data_text += item["text"]
                        if not data_text:
                            data_text = json.dumps(response_data, default=str)
                    # Strip leading record-count prefix like "[15]" emitted by resolve_context
                    count_match = re.match(r'^\[(\d+)\]', data_text)
                    data_text = re.sub(r'^\[\d+\]', '', data_text).strip()
                    empties.append(_data_is_empty(data_text))
                    # Normalise to rows so money-formatting + aggregation also work on the
                    # platform's no-braces field format ("Total AUM": 12345) — previously left
                    # unparsed, which let the LLM mis-group the digits (the Q1 10x bug).
                    rows = coerce_rows(data_text)
                    # Judge each result by the SUB-QUESTION that fetched it, not the overall
                    # query. In a decomposed skill the overall query (e.g. a meeting-prep prompt)
                    # lacks the per-step intent, so query-driven guards misfire — notably a
                    # "monthly … trend" sub-question would be collapsed to its latest snapshot
                    # because the trend keyword lives in the sub-question, not the overall query.
                    # Falls back to `query` for a single, non-decomposed call (sub_question=None).
                    _q = call.get("sub_question") or query
                    # Holdings come back as a per-holding × per-month time series (e.g. 174 rows
                    # for ~29 holdings). Collapse to the LATEST snapshot per holding so the answer
                    # is a concise current-positions table that gets summed into portfolio totals,
                    # instead of dumping every historical row. No-op for non-holdings data.
                    holdings_note = ""
                    if rows:
                        _summary = summarize_holdings(rows, _q)
                        if _summary:
                            rows = _summary["rows"]
                            count_match = None  # the original [174] count no longer describes the rows
                            holdings_note = (
                                f"\n\nNOTE: the platform returned {_summary['original_count']} historical holding "
                                f"snapshots; these have been collapsed to the LATEST position per holding (as of "
                                f"{_summary['as_of']}). Present these {_summary['kept']} current holdings as ONE "
                                f"table grouped by holding, and report the portfolio totals from the AUTHORITATIVE "
                                f"AGGREGATES block. Do NOT list historical monthly rows."
                            )
                        else:
                            # Raw rows that repeat a category (e.g. 30 holdings tagged with ~5
                            # sectors, each carrying a Risk Metric): collapse to ONE row per
                            # category with a count + per-metric totals, so the answer is a clean
                            # grouped summary instead of a long list of near-identical rows.
                            _grouped = summarize_by_category(rows, _q)
                            if _grouped:
                                rows = _grouped["rows"]
                                count_match = None  # the original row count no longer describes the rows
                                _vc = ", ".join(f'"Total {c}"' for c in _grouped["value_cols"])
                                holdings_note = (
                                    f"\n\nNOTE: the platform returned {_grouped['original_count']} individual rows "
                                    f"that repeat the same \"{_grouped['category']}\" values. They have been GROUPED "
                                    f"into {_grouped['groups']} categories, each with its row Count and totals "
                                    f"({_vc}). Present this GROUPED SUMMARY as ONE table (one row per "
                                    f"\"{_grouped['category']}\"), sorted by the total — do NOT list the individual "
                                    f"underlying rows."
                                )
                    # Domain template: holdings / fund / asset-allocation questions get a
                    # standard column frame, and share-% columns (% Current Value, % of Total)
                    # computed in code so they sum to 100%.
                    template_note = ""
                    if rows:
                        rows, template_note = apply_template(rows, _q)
                        # Try to render the table in CODE (no LLM) — reliable + fast — for a
                        # single-call plain holdings/fund/asset question. Falls back to the LLM
                        # (det_answer stays None) when columns can't be mapped or the question
                        # needs interpretation. Decided/applied after the loop with full gating.
                        if len(mcp_results) == 1:
                            _meta = {"as_of": _summary["as_of"]} if _summary else {}
                            det_answer = render_table_answer(rows, _q, _meta)
                    if rows is not None:
                        data_text = json.dumps(rows)
                    # Compute exact totals (issue #1) from the rows. (Charts are now rendered
                    # on-demand client-side from the answer's table — no server chart spec.)
                    agg_block = build_aggregate_block(rows, _q) if rows else ""
                    # Pre-format monetary columns deterministically so the LLM copies the
                    # correct ₹ string instead of mis-grouping digits (the 10x bug).
                    data_text = annotate_money(data_text)
                    if len(data_text) > _MAX_MCP_DATA_CHARS:
                        data_text = _smart_truncate(data_text)
                    # Add record count annotation so LLM doesn't have to count manually
                    record_count = ""
                    if count_match:
                        record_count = f" [{count_match.group(1)} records]"
                    elif data_text.startswith("["):
                        try:
                            record_count = f" [{len(json.loads(data_text))} records]"
                        except (json.JSONDecodeError, TypeError):
                            pass
                    # Label with the sub-question when present (decomposed answer) so the model
                    # knows which part each block answers; otherwise keep the original label.
                    _sub_q = call.get("sub_question")
                    if _sub_q:
                        user_content += f"\n\nData for \"{_sub_q}\"{record_count}:\n{data_text}"
                    else:
                        user_content += f"\n\nData ({call.get('tool', 'api')}){record_count}:\n{data_text}"
                    if agg_block:
                        user_content += f"\n\n{agg_block}"
                    if holdings_note:
                        user_content += holdings_note
                    if template_note:
                        user_content += template_note
                else:
                    empties.append(False)  # an error is not "empty data"
                    error_msg = call.get('error', 'failed')
                    _sub_q = call.get("sub_question")
                    _src = f'"{_sub_q}"' if _sub_q else call.get('tool', 'api')
                    user_content += f"\n\nError from {_src}: {error_msg}"

        # ── Decomposed question with NO usable data anywhere (every sub-question was empty or
        # failed — e.g. the platform can't compute "the worst-performing portfolio" and returns
        # a null Client Name, so the dependent steps were skipped). Give a clean, honest answer
        # deterministically rather than letting the LLM fabricate or echo placeholder errors.
        # (Also avoids an LLM call — helpful under provider rate limits.)
        if _plan.get("strategy") == "decompose" and mcp_results:
            def _call_has_no_data(c):
                if not c.get("success"):
                    return True
                txt = re.sub(r'^\[\d+\]', '', (c.get("_parsed_text") or "")).strip()
                return _data_is_empty(txt)
            if all(_call_has_no_data(c) for c in mcp_results):
                elapsed_ms = round((time.perf_counter() - start) * 1000)
                msg = (
                    "I couldn't find the data needed to answer that. The platform didn't return "
                    "a result for the key part of your question (for example, identifying the "
                    "specific client or value to look up), so I can't complete the rest. "
                    "Could you rephrase it or give the specific client/fund name?"
                )
                return {
                    "messages": [AIMessage(content=msg)],
                    "latency_breakdown": {**state.get("latency_breakdown", {}), "llm_ms": elapsed_ms},
                }

        # ── Name safety-net (issue #1A): when a specific PERSON was asked about and the
        # platform returned NOTHING, do not assert "no holdings" if we can't even be sure we
        # had the right name — ask instead. A resolvable name that returns empty is a real
        # negative and is left to the normal answer path.
        all_empty = bool(empties) and all(empties)
        clarify = _name_clarification_if_empty(state, query, all_empty)
        if clarify:
            elapsed_ms = round((time.perf_counter() - start) * 1000)
            return {
                "messages": [AIMessage(content=clarify)],
                "latency_breakdown": {**state.get("latency_breakdown", {}), "llm_ms": elapsed_ms},
            }

        # Does this turn warrant a live web lookup? (Deep Insight + enabled + a macro /
        # current-events question.) Check the ENRICHED query, not the raw one: after a
        # clarification the raw message is just "yes", but the reconstructed standalone question
        # still carries the macro signal ("...based on latest market news..."). When web context
        # is warranted we must NOT short-circuit to the deterministic table below — an advice /
        # macro question needs the LLM path (portfolio data + web context), not a bare snapshot.
        from ...modes import normalize_mode, DEEP_INSIGHT
        from ...web_search import needs_web_search
        # A macro-impact question has its macro words ("war") stripped from enriched_query (so the
        # platform gets a clean portfolio lookup) — intent_enrichment stashes the ORIGINAL macro
        # question on the entities. Prefer it so the macro signal survives for the web lookup.
        _macro_q = (state.get("enrichment_entities") or {}).get("_macro_query")
        # Pick the phrasing that still carries the event/news signal. Enrichment may strip the
        # macro words out of enriched_query (to hand the platform a clean lookup), so also fall
        # back to the raw query. This lets the live-web pre-step fire for ANY Deep-Insight event
        # question — whether a skill was selected (it combines web + platform data) or not
        # (web-only answer) — without depending on the old macro executor's stashes.
        _web_candidates = [_macro_q, state.get("enriched_query"), query]
        _web_query = next(
            (q.strip() for q in _web_candidates if q and needs_web_search(q)),
            (state.get("enriched_query") or query or "").strip(),
        )
        _wants_web = (
            config.web_search_enabled
            and normalize_mode(state.get("mode")) == DEEP_INSIGHT
            and needs_web_search(_web_query)
        )

        # ── Deterministic table answer (NO LLM): a plain holdings / fund / asset-allocation
        # question whose columns we could map → build the table in code (reliable, no dropped
        # rows, no mis-formatting, no LLM latency). Skipped when a skill or session preference
        # needs the LLM, the question was decomposed (handled above/below by the LLM path), or
        # the turn warrants live web context (an advice/macro question, not a plain snapshot).
        if (det_answer and not _skills and not _wants_web
                and not (state.get("session_preferences") or "").strip()
                and _plan.get("strategy") != "decompose"):
            from .stream_channel import streaming_active as _sa, emit_token as _emit
            if _sa():
                head, sep, body = det_answer.partition("\n\n")
                _emit(head + sep)   # intro line as prose
                _emit(body)         # the table (one block — not dripped)
            elapsed_ms = round((time.perf_counter() - start) * 1000)
            log_chain("answer", make_cid(query, state.get("user_id", "")),
                      intent=intent or "fetch_data", user_query=query,
                      enriched=state.get("enriched_query", ""), answer=det_answer)
            return {
                "messages": [AIMessage(content=det_answer)],
                "latency_breakdown": {**state.get("latency_breakdown", {}), "llm_ms": elapsed_ms},
            }

        # ── Deep Insight: live web context. For macro / current-events questions ("because of
        # the war, what's hit this portfolio?") fetch real-world context and hand it to the LLM
        # alongside the platform data. Gated to Deep Insight + enabled + a configured key; inert
        # otherwise. Treated as UNTRUSTED background (see format_web_context). Searches the
        # ENRICHED query so the macro signal survives a clarification turn (see _wants_web above).
        if _wants_web:
            from ...web_search import web_search, format_web_context, _strip_scope, strip_entity_names
            from .stream_channel import emit_step, emit_detail
            # Search the macro/news topic, not the client: strip the resolved client/investor
            # name so results are about the market (not a same-named stranger on the web).
            _search_q = strip_entity_names(
                _strip_scope(_web_query),
                [
                    (state.get("enrichment_entities") or {}).get("investor_name"),
                    state.get("investor_name"),
                    state.get("selected_client"),
                ],
            ) or _strip_scope(_web_query)
            # Surface a dedicated "live web search" step with its own detail lines
            # (the search query + the sources found), shown alongside the platform steps.
            emit_step("web_search")
            emit_detail("web_search", f"Searching the web for live context: “{_search_q}”")
            try:
                _web_data = await web_search(_search_q)
            except Exception:
                _web_data = {"answer": None, "results": []}
            _results = _web_data.get("results") or []
            _web_results = _results  # remembered for the deterministic Sources footer below
            if _results:
                emit_detail("web_search", f"{len(_results)} source(s) found")
                for _r in _results[:5]:
                    _t = (_r.get("title") or "").strip()
                    _u = (_r.get("url") or "").strip()
                    if _t or _u:
                        emit_detail("web_search", f"{_t} — {_u}" if _u else _t)
            else:
                emit_detail("web_search", "No live web results (search returned nothing)")
            _web_block = format_web_context(_web_data)
            if _web_block:
                user_content += _web_block

        messages = [
            SystemMessage(content=sys_prompt),
            HumanMessage(content=user_content),
        ]

    # Stream the answer LIVE when an SSE consumer is attached (the orchestrator only
    # reaches here once the guardrail has PASSED, so live tokens never leak a blocked
    # answer). Otherwise (e.g. non-streaming /chat) fall back to a single ainvoke.
    from .stream_channel import streaming_active, emit_token
    if streaming_active():
        # Stream PROSE live, but DON'T drip markdown tables token-by-token — once the answer
        # enters a table, hold it back and emit the whole block at once (tables render instantly
        # instead of materialising cell-by-cell). The emitted slices always reconstruct the full
        # text exactly, so the final rendered answer is identical to a plain stream.
        full = ""
        emitted = 0
        buffering = False
        async for chunk in llm.astream(messages):
            piece = getattr(chunk, "content", "") or ""
            if not piece:
                continue
            full += piece
            if buffering:
                continue  # inside a table — accumulate, flush at the end
            tbl_start = _table_start_index(full)
            if tbl_start >= 0 and tbl_start >= emitted:
                if tbl_start > emitted:
                    emit_token(full[emitted:tbl_start])
                    emitted = tbl_start
                buffering = True  # from here on, buffer until the stream ends
            else:
                # Emit only up to the last completed line, so a partial line that may yet turn
                # into a table header isn't streamed prematurely.
                last_nl = full.rfind("\n")
                if last_nl + 1 > emitted:
                    emit_token(full[emitted:last_nl + 1])
                    emitted = last_nl + 1
        if len(full) > emitted:
            emit_token(full[emitted:])  # flush the held table (and anything after it)
        answer_text = full
        # Deterministic Sources footer — emit it live so it renders at the end of the stream.
        _footer = _sources_footer(_web_results, answer_text)
        if _footer:
            emit_token(_footer)
            answer_text += _footer
    else:
        response = await llm.ainvoke(messages)
        answer_text = response.content
        _footer = _sources_footer(_web_results, answer_text)
        if _footer:
            answer_text += _footer
        # Phase 4 (opt-in, non-streaming only): flag a coverage gap. We can't amend a live
        # stream, so this annotates the single-shot answer when ANSWER_VERIFY_ENABLED is on.
        if intent != "from_memory":
            from .verify import verify_answer
            _v = await verify_answer(query, answer_text)
            if not _v["complete"] and _v["missing"]:
                answer_text += f"\n\n_I wasn't able to fully cover: {_v['missing']}. Ask me again for that part and I'll look it up._"
    elapsed_ms = round((time.perf_counter() - start) * 1000)

    log_chain("answer", make_cid(query, state.get("user_id", "")),
              intent=intent or "fetch_data", user_query=query,
              enriched=state.get("enriched_query", ""), answer=answer_text)

    out = {
        "messages": [AIMessage(content=answer_text)],
        "latency_breakdown": {**state.get("latency_breakdown", {}), "llm_ms": elapsed_ms},
    }
    if chart_spec:
        out["chart_spec"] = chart_spec
    return out


def _blocked_message(guardrail_result: dict) -> str:
    """Build a category-specific refusal based on WHICH guardrail fired.

    NeMo returns the same generic refusal for every block, so keying off the
    reason text is useless — we key off `blocked_by` (the guardrail name) instead.
    """
    blocked_by = (guardrail_result.get("blocked_by") or "").lower()
    reason = (guardrail_result.get("reason") or "").strip()

    # Map the firing guardrail to a purpose-built, single message.
    if "pii" in blocked_by:
        return ("I can't process that — please don't include sensitive identifiers "
                "like an SSN, PIN, card number, or Aadhaar in your message.")
    if "injection" in blocked_by:
        return "I can't process that request for security reasons."
    if "toxic" in blocked_by:
        return "Let's keep it professional — please rephrase and I'll help."
    if "length" in blocked_by:
        return "Your message is too long. Please shorten it and try again."
    if "topical" in blocked_by or "domain" in blocked_by or "self_check" in blocked_by:
        return ("I'm a wealth management assistant — I can only help with portfolios, "
                "accounts, holdings, fees, transactions, and financial education. I can't "
                "give specific buy/sell recommendations or help with off-topic requests.")

    # Fallback: unknown guardrail — use its reason if it's meaningful.
    if reason.lower() in ("", "none", "blocked by guardrail", "content policy violation",
                          "i'm sorry, i can't respond to that.", "off-topic for a wealth advisor"):
        return ("I can't process that request. It may be outside what I can help with, "
                "or contain sensitive information.")
    return f"I can't process that request. {reason}"


async def generate_blocked_response(state: AgentState) -> dict:
    """Generate a polite, category-specific refusal when a guardrail blocks the input."""
    guardrail_result = state.get("guardrail_result", {}) or {}
    return {
        "messages": [AIMessage(content=_blocked_message(guardrail_result))],
    }


# --- Instrument at the bottom of the file ---
from ...observe_utils import observed_node as _observed_node


def _answer_text(result):
    msgs = (result or {}).get("messages") or []
    for m in reversed(msgs):
        content = getattr(m, "content", None) or (m.get("content") if isinstance(m, dict) else None)
        if content:
            return content
    return ""


def _generate_detail(state, result):
    return {
        "input": state.get("enriched_query") or state.get("query", ""),
        "output": _answer_text(result),
    }


generate_response = _observed_node(generate_response, name="Generate Response", detail=_generate_detail)
generate_blocked_response = _observed_node(generate_blocked_response, name="Generate Blocked Response", detail=_generate_detail)
