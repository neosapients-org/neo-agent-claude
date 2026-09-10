"""MCP fetch node — discovers and executes MCP tools.

Split into two measured phases:
1. Tool Selection (LLM call) — picks the right tool + args
2. MCP Execution — actual network call to the MCP server
"""

import asyncio
import csv
import io
import json
import random
import re
import time
import logging

import httpx

try:
    from ns_probe import observe
except ImportError:
    def observe(*a, **kw):
        def _id(fn): return fn
        return _id

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

from ...config import config
from ..state import AgentState
from ...mcp.client import mcp_client
from ...chainlog import log_chain, make_cid
from ...money import fmt_inr as _fmt_inr, inr_group as _inr_group

logger = logging.getLogger(__name__)


def _norm_key(k: str) -> str:
    return re.sub(r"[\s_]+", " ", str(k).strip().lower())


def _parse_mcp_text(raw_text: str) -> str:
    """Parse MCP response text into structured JSON.
    
    The MCP resolve_context tool returns data in a custom CSV-like format:
      [count]{"Col1","Col2",...}:
      val1,val2,...
      val1,val2,...
    
    This function converts it to a JSON array for easier LLM consumption.
    Also handles NDJSON (newline-delimited JSON) and plain JSON.
    """
    raw_text = raw_text.strip()
    if not raw_text:
        return raw_text

    # If it's already valid JSON, return as-is
    try:
        json.loads(raw_text)
        return raw_text
    except json.JSONDecodeError:
        pass

    # Handle the MCP custom CSV format: [count]{headers}:\nrow1\nrow2
    csv_match = re.match(r'^\[(\d+)\]\{(.+?)\}:\s*\n?(.+)', raw_text, re.DOTALL)
    if csv_match:
        count = int(csv_match.group(1))
        headers_raw = csv_match.group(2)
        data_section = csv_match.group(3)
        
        # Parse headers using csv.reader for proper handling of mixed quoted/unquoted
        try:
            reader = csv.reader(io.StringIO(headers_raw))
            headers = [h.strip() for h in next(reader)]
        except (StopIteration, csv.Error):
            headers = [h.strip().strip('"') for h in headers_raw.split(',')]
        
        # Parse data rows using csv.reader
        rows = []
        for line in data_section.strip().split('\n'):
            line = line.strip()
            if not line:
                continue
            try:
                reader = csv.reader(io.StringIO(line))
                values = [v.strip() for v in next(reader)]
            except (StopIteration, csv.Error):
                values = [v.strip() for v in line.split(',')]
            row = {}
            for i, h in enumerate(headers):
                row[h] = values[i] if i < len(values) else ""
            rows.append(row)
        
        return json.dumps(rows)
    
    # Handle multi-field MCP format: "Field"[count]: val1,val2,...\n"Field2"[count]: ...
    # Also handles: "Field"[count]{nullValue}:\n  val1\n  val2
    # And unquoted: Field[count]{nullValue}:\n  val1
    if re.match(r'^"?[^"[\n]+"?\[\d+\]', raw_text):
        fields = {}
        count = 0
        lines = raw_text.strip().split('\n')
        i = 0
        while i < len(lines):
            line = lines[i]
            # Match: "Field"[count]: values  OR  "Field"[count]{annotation}:  OR  Field[count]:
            field_match = re.match(r'^"?([^"[\]]+)"?\[(\d+)\](?:\{[^}]*\})?:\s*(.*)', line)
            if field_match:
                field_name = field_match.group(1)
                count = int(field_match.group(2))
                values_str = field_match.group(3).strip()
                if values_str:
                    # Values on the same line, comma-separated
                    fields[field_name] = [v.strip().strip('"') for v in values_str.split(',')]
                else:
                    # Values on subsequent indented lines
                    values = []
                    i += 1
                    while i < len(lines) and (lines[i].startswith('  ') or lines[i].startswith('\t')):
                        val = lines[i].strip().rstrip(',')
                        if val.startswith('- '):
                            val = val[2:]
                        values.append(val.strip('"'))
                        i += 1
                    fields[field_name] = values
                    continue  # don't increment i again
            i += 1
        
        if fields:
            # Transpose field lists into array of objects
            # Use max count across all fields to ensure no records are lost
            max_count = max(len(v) for v in fields.values()) if fields else count
            rows = []
            all_keys = list(fields.keys())
            for idx in range(max_count):
                row = {}
                for key in all_keys:
                    row[key] = fields[key][idx] if idx < len(fields[key]) else ""
                rows.append(row)
            return json.dumps(rows)

    # Handle NDJSON: multiple JSON objects separated by newlines
    lines = raw_text.split("\n")
    parsed_objects = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
            parsed_objects.append(obj)
        except json.JSONDecodeError:
            continue

    if parsed_objects:
        if len(parsed_objects) == 1:
            return json.dumps(parsed_objects[0])
        return json.dumps(parsed_objects)

    return raw_text


# Pure-chitchat openers that should NOT trigger a data fetch.
_CHITCHAT_ONLY = re.compile(
    r"^\s*(hi|hello|hey|thanks|thank you|thank u|ok|okay|cool|great|bye|goodbye|"
    r"good (morning|afternoon|evening)|how are you|who are you|what can you do|"
    r"what do you do|help)\b[\s!.?]*$",
    re.IGNORECASE,
)


# FAST transient platform errors — these fail quickly (~a few seconds) and a
# retry usually succeeds. We deliberately EXCLUDE gateway timeouts here because
# mcp_client.call_tool already retries those internally (and each timeout attempt is
# slow); retrying them again here would multiply latency badly.
# NOTE: "api key is invalid or revoked" is included on purpose — on this platform it is
# an INTERMITTENT/spurious 401 (the same key succeeds on the very next call and on direct
# probes), so we treat it as a transient and retry rather than failing the user's query.
_FAST_TRANSIENT_PAT = re.compile(
    r"context resolution failed|service unavailable|no t1/t2/t3|"
    r"internal server error|temporarily|api key is invalid or revoked|invalid or revoked",
    re.IGNORECASE,
)


def _is_transient_platform_error(msg: str) -> bool:
    return bool(msg) and bool(_FAST_TRANSIENT_PAT.search(msg))


# Refinement of empty/failed retries lives in refine.py (Phase 2): `cheap_variant` is the
# deterministic cache-bust, `next_refinement` escalates to a genuinely different LLM rewording.
from .refine import next_refinement, cheap_variant


def _extract_mcp_text(tool_result: dict) -> str:
    """Return the DATA text from an MCP tool result.

    Cortex returns a single raw-data block today. If it ever returns multiple blocks
    (e.g. raw data + an LLM prose summary), we keep only the raw/data block — letting a
    narrative summary into our own synthesis is exactly the poisoning we want to avoid.
    """
    resp = tool_result.get("response")
    if not (tool_result.get("success") and isinstance(resp, dict)):
        return ""
    blocks = [
        item.get("text", "")
        for item in resp.get("content", [])
        if isinstance(item, dict) and item.get("type") == "text" and item.get("text")
    ]
    if len(blocks) <= 1:
        return blocks[0] if blocks else ""
    # Multiple blocks: prefer the structured data block over any prose summary block.
    data_like = [b for b in blocks if re.match(r'^\s*[\[{]|^"[^"]+"\[\d+\]', b.strip())]
    return max(data_like or blocks, key=len)


def _is_empty_or_null_text(text: str) -> bool:
    """True if the MCP payload carries no usable data (empty list, all-null rows…)."""
    s = (text or "").strip()
    if not s:
        return True
    s = re.sub(r'^\[\d+\]', '', s).strip()  # drop a leading [count] prefix
    if s in ("", "[]", "{}", "[ ]", "null"):
        return True
    try:
        v = json.loads(s)
    except (json.JSONDecodeError, TypeError):
        return False
    if isinstance(v, list):
        if not v:
            return True
        return all(
            isinstance(r, dict) and all(x in (None, "", "null") for x in r.values())
            for r in v
        )
    if isinstance(v, dict):
        return all(x in (None, "", "null") for x in v.values())
    return False


def _is_platform_error_payload(text: str) -> bool:
    """True if the payload is a platform ERROR envelope returned inside an otherwise
    'successful' response, e.g. {"error":{"code":"EXECUTION_ERROR","retryable":false,...}}.

    The platform surfaces some failures this way (HTTP/JSON-RPC success, but the body is an
    error object). Without recognising it we'd treat the error object as valid non-empty data —
    which is exactly what broke fund recovery: the retry loop stopped early thinking it had data,
    recovery was skipped, and the raw error JSON reached the user.
    """
    s = re.sub(r'^\[\d+\]', '', (text or "").strip()).strip()
    if not s or s[0] not in "[{":
        return False
    try:
        v = json.loads(s)
    except (json.JSONDecodeError, TypeError):
        return False
    if isinstance(v, list):
        v = v[0] if (len(v) == 1 and isinstance(v[0], dict)) else None
    return isinstance(v, dict) and "error" in v and bool(v.get("error"))


async def _resolve_with_retries(tool_name: str, arguments: dict, max_attempts: int = 4,
                                record: list | None = None, entities: dict | None = None,
                                smart_refine: bool | None = None):
    """Call resolve_context, retrying while the result is UNUSABLE — i.e. a fast
    intermittent platform failure ("Context resolution failed" / "Service Unavailable")
    or an empty / all-null data payload.

    The platform resolver is non-deterministic: the same query that returns nothing on
    one call frequently returns the data on the next. We retry (bounded + backoff) so the
    user doesn't have to manually "try again". Slow gateway timeouts are NOT retried here
    (mcp_client already retries those internally), which keeps total latency bounded.

    Each retry reframes the question via the Phase-2 refinement ladder (cheap cache-bust
    first, then a genuinely different LLM rewording). Pass ``smart_refine=False`` to force the
    deterministic cache-bust only (used by recovery sub-calls, which must stay cheap/exact).

    Returns (tool_result, mcp_text).
    """
    if smart_refine is None:
        smart_refine = config.smart_refine_enabled
    base_query = arguments.get("query", "") if isinstance(arguments, dict) else ""
    tried: list[str] = [base_query] if base_query else []
    if record is not None and base_query:
        record.append(base_query)
    result = await mcp_client.call_tool(tool_name, arguments)
    raw = _extract_mcp_text(result)
    attempt = 1
    while attempt < max_attempts:
        err = str(result.get("error", ""))
        # The raw payload uses a custom CSV/field format — parse it before judging
        # emptiness (an empty `"Field"[0]:` block is not valid JSON on its own).
        parsed = _parse_mcp_text(raw) if raw else ""
        plat_err = _is_platform_error_payload(parsed)
        empty = _is_empty_or_null_text(parsed) or plat_err
        ctx_failed = "context resolution failed" in raw.lower()
        usable = result.get("success") and not empty and not ctx_failed
        if usable:
            break
        # Note: a platform error ENVELOPE ({"error":{...}}) is folded into `empty` above, so it
        # is treated as retryable — the platform is non-deterministic and re-firing the SAME
        # (correct) query frequently returns data on a later attempt.
        retryable = (
            (not result.get("success") and _is_transient_platform_error(err))
            or (result.get("success") and (empty or ctx_failed))
        )
        if not retryable:
            break
        # Jittered backoff — also helps a later attempt land outside the platform's
        # ~120s exact-string cache window.
        await asyncio.sleep(min(0.4 * attempt, 1.5) + random.uniform(0.0, 0.3))
        # Reframe the query for this retry. Smart refinement escalates from a cheap
        # cache-bust (attempt 1) to a genuinely different LLM rewording (attempt >=2); it
        # falls back to the cheap variant whenever it has nothing safe/better to offer.
        retry_args = arguments
        if base_query:
            new_q = None
            if smart_refine:
                new_q = await next_refinement(base_query, attempt, entities=entities, history=tried)
            if not new_q:
                new_q = cheap_variant(base_query, attempt)
            tried.append(new_q)
            retry_args = {**arguments, "query": new_q}
        if record is not None and isinstance(retry_args, dict):
            record.append(retry_args.get("query", ""))
        logger.info(
            "[mcp_fetch] unusable (empty=%s ctx_failed=%s err=%.40s) — retry %s/%s | variant=%r",
            empty, ctx_failed, err, attempt, max_attempts - 1,
            (retry_args.get("query", "")[:70] if isinstance(retry_args, dict) else ""),
        )
        result = await mcp_client.call_tool(tool_name, retry_args)
        raw = _extract_mcp_text(result)
        attempt += 1
    return result, raw


def _names_from_payload(parsed_text: str) -> list[str]:
    """Pull candidate fund/scheme names out of a parsed DISTINCT payload (JSON produced by
    ``_parse_mcp_text``). Takes the column whose header looks like a fund/scheme/name field;
    falls back to a sole column or bare string rows."""
    try:
        data = json.loads(parsed_text)
    except (json.JSONDecodeError, TypeError):
        return []

    names: list[str] = []
    if isinstance(data, list):
        for row in data:
            if isinstance(row, dict):
                keys = list(row.keys())
                # Prefer an explicit NAME column; never an ID column (the platform returns a
                # "Fund ID" column too, which would otherwise grab numeric IDs, not names).
                key = next((k for k in keys if re.search(r"name", str(k), re.I)), None)
                if key is None:
                    key = next((k for k in keys
                                if re.search(r"fund|scheme", str(k), re.I)
                                and not re.search(r"id", str(k), re.I)), None)
                if key is None and len(keys) == 1:
                    key = keys[0]
                val = row.get(key) if key else None
                # Skip blanks and pure-numeric IDs — we want human-readable scheme names.
                if val not in (None, "", "null") and not str(val).strip().isdigit():
                    names.append(str(val))
            elif isinstance(row, str) and row.strip() and not row.strip().isdigit():
                names.append(row.strip())
    elif isinstance(data, dict):
        for k, v in data.items():
            if re.search(r"fund|scheme|name", str(k), re.I):
                if isinstance(v, list):
                    names.extend(str(x) for x in v if x not in (None, "", "null"))
                elif v not in (None, "", "null"):
                    names.append(str(v))

    # De-dupe, preserving order.
    seen: set[str] = set()
    out: list[str] = []
    for n in names:
        if n not in seen:
            seen.add(n)
            out.append(n)
    return out


async def _recover_fund_entity(tool_name: str, original_query: str, user_fund: str,
                               record: list | None = None) -> dict | None:
    """Distinct-lookup + similarity recovery for a fund/scheme question that came back EMPTY
    (Phase 1 — fixes exact-match misses like "HDFC Midcap opportunities").

    Steps: pick a broad anchor (AMC/first word) → ask the platform for that anchor's distinct
    fund names → similarity-match the user's term → re-fetch the ORIGINAL question with the
    canonical name substituted in.

    Returns one of:
        {"status": "resolved", "tool_result": ..., "mcp_text": ..., "meta": ...}
        {"status": "clarify", "question": "...", "meta": ...}
        {"status": "none", "meta": ...}
        None  — recovery couldn't even start (no usable anchor).
    The sub-calls use a small ``max_attempts`` to keep the extra latency bounded.
    """
    from .fund_resolver import anchor_token, best_match

    anchor = anchor_token(user_fund)
    if not anchor:
        return None

    # Phrase the lookup so the platform returns scheme NAMES (an earlier "distinct fund and
    # scheme names" phrasing made it return Fund IDs instead).
    distinct_q = f"List the names of all mutual fund schemes from {anchor} AMC."
    dist_result, dist_raw = await _resolve_with_retries(
        tool_name, {"query": distinct_q}, max_attempts=2, record=record, smart_refine=False,
    )
    names = _names_from_payload(_parse_mcp_text(dist_raw)) if dist_raw else []
    meta = {"user_fund": user_fund, "anchor": anchor, "distinct_count": len(names)}
    if not names:
        return {"status": "none", "meta": meta}

    canonical, candidates = best_match(user_fund, names)
    meta["resolved"] = canonical
    meta["candidates"] = candidates

    if canonical and canonical.strip().lower() != user_fund.strip().lower():
        # Substitute the platform's exact name into the original question (or append it if the
        # user's wording wasn't literally present — e.g. it came from entity extraction).
        fixed_q = re.sub(re.escape(user_fund), canonical, original_query, flags=re.IGNORECASE)
        if fixed_q == original_query:
            fixed_q = f"{original_query.rstrip(' ?.')} (the fund is {canonical})."
        # Re-fire the CORRECT name a few times — even the right name is only ~75% reliable per
        # call, so we retry through the platform's non-determinism before concluding "no data".
        for _ in range(3):
            re_result, re_raw = await _resolve_with_retries(
                tool_name, {"query": fixed_q}, max_attempts=1, record=record, smart_refine=False,
            )
            re_text = _parse_mcp_text(re_raw) if re_raw else ""
            if not _is_empty_or_null_text(re_text) and not _is_platform_error_payload(re_text):
                return {"status": "resolved", "tool_result": re_result, "mcp_text": re_text, "meta": meta}
        return {"status": "none", "meta": meta}

    if candidates:
        listed = (", ".join(candidates[:-1]) + " or " + candidates[-1]) if len(candidates) > 1 else candidates[0]
        return {"status": "clarify",
                "question": f"I couldn't find “{user_fund}” exactly. Did you mean {listed}?",
                "meta": meta}

    return {"status": "none", "meta": meta}


# --- Proactive fund-name canonicalization -------------------------------------------------
# The whole fund universe is small (~30 schemes), so we fetch the catalog ONCE per process and
# resolve any user-mentioned fund to its exact stored name BEFORE the first fetch. This is far
# more reliable than depending on the platform to resolve a slightly-off name (which it does
# very inconsistently). Reactive recovery (_recover_fund_entity) stays as a fallback.
_FUND_NAMES_CACHE: list[str] | None = None
_FUND_CACHE_LOCK = asyncio.Lock()


async def _all_fund_names() -> list[str]:
    """All mutual-fund scheme names from the platform catalog, fetched once and cached."""
    global _FUND_NAMES_CACHE
    if _FUND_NAMES_CACHE is not None:
        return _FUND_NAMES_CACHE
    async with _FUND_CACHE_LOCK:
        if _FUND_NAMES_CACHE is not None:
            return _FUND_NAMES_CACHE
        try:
            _, raw = await _resolve_with_retries(
                "resolve_context",
                {"query": "List the names of all mutual fund schemes in the product catalog."},
                max_attempts=3, smart_refine=False,
            )
            names = _names_from_payload(_parse_mcp_text(raw)) if raw else []
        except Exception:
            names = []
        if names:  # only cache a non-empty result, so a transient miss can be retried next time
            _FUND_NAMES_CACHE = names
        if names:
            return names
        # Live fetch failed (platform degraded) — fall back to the bundled static catalog so
        # fund-name canonicalization still works. Not cached, so we re-try live next time.
        from .fund_resolver import _STATIC_FUND_CATALOG
        return list(_STATIC_FUND_CATALOG)


async def _canonical_fund(user_fund: str) -> str | None:
    """The confident exact catalog name for a user's fund mention, or None (no/uncertain match)."""
    from .fund_resolver import best_match
    names = await _all_fund_names()
    if not names:
        return None
    canon, _candidates = best_match(user_fund, names)
    return canon


def _looks_like_data_query(query: str) -> bool:
    """Heuristic: should this query hit resolve_context? Almost everything that is
    not a bare greeting/thanks is a data question for this wealth platform."""
    q = (query or "").strip()
    if not q:
        return False
    if _CHITCHAT_ONLY.match(q):
        return False
    return True


def _sanitize_tool_arguments(arguments: dict) -> dict:
    """Sanitize tool arguments before sending to MCP — fix common LLM mistakes."""
    sanitized = {k: v for k, v in arguments.items() if v is not None and v != ""}

    # The resolve_context tool resolves natural language itself — strip any raw
    # SQL operators the LLM may have leaked into the query so it stays plain English.
    for key in ("query", "filter", "sql", "question"):
        if key in sanitized and isinstance(sanitized[key], str):
            val = sanitized[key]
            # Replace ILIKE with = for exact value matches
            val = re.sub(r'\bILIKE\b', '=', val, flags=re.IGNORECASE)
            # UNIVERSAL SAFETY NET: never send an unfilled {placeholder} to the platform. The
            # decomposition engine already skips steps it can't fill, but if a templated query
            # ever reaches here with a leftover {x}, strip it (and tidy the stray punctuation /
            # whitespace it leaves) rather than asking the platform about a literal "{x}".
            if "{" in val and "}" in val:
                val = re.sub(r'\{[^}]*\}', '', val)
                # Drop a now-dangling preposition that introduced the removed placeholder.
                val = re.sub(r'\b(?:for|of|by|to|held by)\s*(?=[,.]|$)', '', val, flags=re.IGNORECASE)
                val = re.sub(r'\s+([,.])', r'\1', val)        # " ," -> ","
                val = re.sub(r'\s{2,}', ' ', val).strip()
            sanitized[key] = val

    return sanitized

SCHEMA_CONTEXT = """
The platform is a wealth-management data platform. It holds data about clients (investors),
their families, relationship managers (servicing bankers), holdings, valuations, transactions, AUM,
revenue, and the product catalog. You do NOT need to know exact table names — the resolve_context tool
maps natural-language questions to the underlying data itself.

Domain vocabulary you may reference naturally in the query:
- client / investor (e.g. Ram Krishnan), family (Krishnan, Mehra, Patel, Choudhary, Fernandes families)
- relationship manager / servicing banker, LOB / sub-LOB / SBU, client tier (HNI / Ultra HNI)
- holdings, asset class (Equity / Fixed Income / Hybrid / Alternatives), product type
  (Mutual Funds / Direct Equity / Bonds / ETF / AIF / PMS / REIT / InvIT), sector, AMC, fund/scheme, NAV
- current value, invested value, AUM, realized gain, unrealized gain, expense ratio, XIRR, returns
- transactions (SIP / Purchase / Redemption / Switch), maturity date, credit rating
- cash inflows / outflows / net new money, revenue
"""

TOOL_SELECTION_PROMPT = """You are a tool-selection assistant for The platform (internal).
The platform has ONE tool: resolve_context — it answers natural-language questions against the
wealth-management data platform. It performs its own schema mapping, so you do NOT need table or column names.

{schema_context}

IMPORTANT: The tool takes a SINGLE "query" argument. Pass a clear, self-contained question in PLAIN ENGLISH.
- Do NOT write SQL, do NOT use operators like =, <, >, LIKE, ILIKE.
- Keep the user's intent and any specific names, filters, timeframes, or comparisons.
- Preserve count / list / ranking / "for each" / aggregate intent exactly.
- If the user named a specific client or fund, keep that exact name in the query.
- For questions about all/multiple clients, do NOT narrow to one client.

CORRECT query examples (plain English, names and filters preserved):
- "What is the risk tolerance and time horizon of client Ram Krishnan?"
- "Show all Equity holdings of Ram Krishnan with their current value and returns."
- "Which client has the highest AUM?"
- "How many clients have invested in AIF?"
- "List all bond holdings with credit rating above AA."
- "What is the total invested value in IT sector stocks for the Krishnan family?"
- "Give the AUM for each relationship manager."
- "Top 5 clients by total portfolio value."
- "Compare 3-year returns of HDFC Mid-Cap vs SBI Small Cap."

Product / market / instrument lookups (NO client needed — still use resolve_context):
- "What is the PAN number of client Ram Krishnan?"
- "What is the expense ratio of HDFC Mid-Cap Opportunities?"
- "What is the credit rating of the NHAI 7.25% 2030 bond?"
- "What sector does Reliance Industries belong to?"
- "What is the current price of Infosys?"
- "What is the 1-year return of Axis Bluechip Fund?"
- "What is the Nifty 50 closing value?"

Advisor / organisation lookups (NO client needed — still use resolve_context):
- "How many clients does banker Priya Sharma manage?"
- "What is the total AUM in the Neo Wealth Advisory LOB?"
- "Which AMC has the highest AUM across all clients?"
- "Show AUM breakdown by LOB."

WRONG examples (never do this):
- "SELECT * FROM client_holdings WHERE client_name = 'Ram'" — raw SQL WRONG
- "show holdings where asset_class = 'Equity'" — SQL operator WRONG
- narrowing "how many clients have AIF" down to a single client WRONG
- returning null tool for a PAN / credit-rating / price / sector / expense-ratio lookup WRONG

CRITICAL ROUTING RULE: ANY question about clients, families, relationship managers/bankers, portfolios,
holdings, allocations, performance, returns, gains, AUM, revenue, transactions, funds, schemes, NAV,
expense ratios, PAN/KYC/profile details, credit ratings, prices, sectors, indices, LOB/SBU, products, or
any financial/wealth/market data MUST use resolve_context. When unsure, USE resolve_context.
Only return null tool for pure chitchat — greetings ("hi", "thanks"), or asking what you can do.

Available tools:
{tools_json}

User query: {query}

Respond ONLY in this exact JSON format (no markdown, no extra text):
{{"tool_name": "<name>", "arguments": {{"query": "<clear plain-English question>"}}}}

For pure chitchat only: {{"tool_name": null, "arguments": {{}}}}"""




async def mcp_fetch(state: AgentState) -> dict:
    """Execute MCP tool based on user intent and available tools.

    Uses the enriched query (from intent_enrichment) if available,
    otherwise falls back to the raw user query.

    Spans:
    - "Tool Selection" — LLM picks the tool
    - "MCP Execution" — actual MCP server call
    """
    # Prefer enriched query from intent enrichment step
    query = state.get("enriched_query") or state["query"]
    start = time.perf_counter()
    _cid = make_cid(state.get("query", ""), state.get("user_id", ""))

    # NOTE: AUM questions are NOT intercepted. They flow through the generic path below and
    # are sent to resolve_context VERBATIM, so the platform returns the payload for the exact
    # question asked (no canned "list of clients and their AUM" rewrite, no per-client join).

    try:
        # --- Phase 1: Tool selection (deterministic, no LLM) ---
        # resolve_context is the ONLY data tool on this platform. The old LLM
        # "tool selection" hop cost a full round-trip and added a failure mode while
        # almost always resolving to "call resolve_context with this query". So we
        # route to it directly. Pure chitchat is already filtered out upstream by
        # intent_enrichment; we keep one cheap guard so a stray greeting that reaches
        # here can't trigger a needless platform call.
        selection_start = time.perf_counter()

        if not _looks_like_data_query(query):
            elapsed_ms = round((time.perf_counter() - start) * 1000)
            return {
                "messages": [AIMessage(content="No relevant tool found for your query. Please try rephrasing.")],
                "mcp_tool_calls": [{"tool": "no_match", "error": "No relevant tool found", "success": False, "latency_ms": elapsed_ms}],
                "latency_breakdown": {
                    **state.get("latency_breakdown", {}),
                    "mcp_exec_ms": 0,
                },
            }

        tool_name = "resolve_context"

        # Proactive fund-name canonicalization (most reliable fix for the HDFC-style misses):
        # resolve a named fund to the platform's EXACT catalog name BEFORE fetching, so the
        # query never depends on the user's spelling. After this, an empty result is a GENUINE
        # null (the fund simply has no such value), which the answer step states plainly.
        proactively_resolved = False
        if config.fund_recovery_enabled:
            _ent0 = state.get("enrichment_entities") or {}
            _uf0 = (_ent0.get("fund_name") or "").strip()
            if _uf0:
                _canon0 = await _canonical_fund(_uf0)
                if _canon0 and _canon0.strip().lower() != _uf0.strip().lower():
                    _nq = re.sub(re.escape(_uf0), _canon0, query, flags=re.IGNORECASE)
                    if _nq == query:
                        _nq = f"{query.rstrip(' ?.')} (the fund is {_canon0})."
                    logger.info("[mcp_fetch] proactive fund canonicalization %r -> %r", _uf0, _canon0)
                    query = _nq
                    proactively_resolved = True

        # NOTE: query augmentation is intentionally DISABLED — the user's question is sent to the
        # platform as-is. The breakdown share-% is computed in code by generate from whatever the
        # platform returns, so there is no need to reshape the question (which previously forced an
        # "by asset class" framing onto unrelated breakdowns).

        arguments = _sanitize_tool_arguments({"query": query})
        selection_ms = round((time.perf_counter() - selection_start) * 1000)

        # --- Phase 2: MCP Execution (network call) ---
        exec_start = time.perf_counter()

        # Show the question being sent LIVE in the analysis pane. Suppressed when this runs as
        # a decomposed sub-question — execute_plan emits the (possibly parallel) batch instead.
        if not state.get("_suppress_send_emit"):
            from .parallel import _fetch_labels
            from .stream_channel import emit_detail as _emit_send
            _emit_send("data_fetch", _fetch_labels()["sending"].format(q=query))

        # Resolve with bounded retries. The platform resolver is non-deterministic and
        # often returns the data only on a later attempt after an empty / "Context
        # resolution failed" first try — so retry (capped + backoff) until we get usable
        # data, instead of making the user manually re-ask.
        queries_sent: list = []
        tool_result, _ = await _resolve_with_retries(
            tool_name, arguments, record=queries_sent,
            entities=state.get("enrichment_entities"),
        )

        exec_ms = round((time.perf_counter() - exec_start) * 1000)

        elapsed_ms = round((time.perf_counter() - start) * 1000)

        # Handle MCP errors (HTTP 500, ILIKE errors, etc.)
        if not tool_result.get("success"):
            error_msg = tool_result.get("error", "Unknown MCP error")
            # Extract useful info from platform error messages
            if "ILIKE" in error_msg and "not allowed" in error_msg:
                error_msg = "Platform filter error — the query used an unsupported operator. Retrying with exact match."
                # Retry with explicit equality phrasing
                retry_query = arguments.get("query", query) + " (use exact equality = operator only, never ILIKE)"
                retry_result = await mcp_client.call_tool(tool_name, {"query": retry_query})
                if retry_result.get("success"):
                    tool_result = retry_result
                else:
                    return {
                        "messages": [AIMessage(content=f"Error: {error_msg}")],
                        "mcp_tool_calls": [tool_result],
                        "latency_breakdown": {
                            **state.get("latency_breakdown", {}),
                            "mcp_exec_ms": exec_ms,
                        },
                    }
            else:
                return {
                    "messages": [AIMessage(content=f"Error: {error_msg}")],
                    "mcp_tool_calls": [tool_result],
                    "latency_breakdown": {
                        **state.get("latency_breakdown", {}),
                        "mcp_exec_ms": exec_ms,
                    },
                }

        # Extract the MCP response text and return as AIMessage directly
        mcp_text = ""
        if tool_result.get("success") and tool_result.get("response"):
            content_list = tool_result["response"].get("content", [])
            for item in content_list:
                if item.get("type") == "text":
                    mcp_text += item.get("text", "")

        # Detect platform error messages returned as content text
        if mcp_text and len(mcp_text) < 200 and any(err in mcp_text for err in [
            "Extra data:", "JSONDecodeError", "Traceback", "Internal Server Error",
            "TypeError:", "ValueError:", "KeyError:", "AttributeError:",
        ]):
            logger.warning("[mcp_fetch] MCP returned error as content: %s", mcp_text[:200])
            tool_result["success"] = False
            tool_result["error"] = mcp_text

        # Retry on "Context resolution failed" responses
        if mcp_text and "Context resolution failed" in mcp_text:
            logger.info("[mcp_fetch] Context resolution failed, retrying...")
            retry_result = await mcp_client.call_tool(tool_name, arguments)
            retry_text = ""
            if retry_result.get("success"):
                for item in retry_result.get("response", {}).get("content", []):
                    if item.get("type") == "text":
                        retry_text += item.get("text", "")

            if retry_text and "Context resolution failed" not in retry_text:
                # Retry succeeded — update data and continue processing
                mcp_text = retry_text
                tool_result = retry_result
                # Fall through to continue processing below
            else:
                # Both original and retry failed — return error
                tool_result["success"] = False
                tool_result["error"] = mcp_text
                return {
                    "messages": [AIMessage(content=f"Error: {mcp_text}")],
                    "mcp_tool_calls": [tool_result],
                    "latency_breakdown": {
                        **state.get("latency_breakdown", {}),
                        "mcp_exec_ms": exec_ms,
                    },
                }

        if mcp_text:
            # Parse custom CSV/NDJSON format into JSON
            mcp_text = _parse_mcp_text(mcp_text)

        # Detect all-null responses and retry once
        if mcp_text:
            try:
                parsed_check = json.loads(mcp_text)
                if isinstance(parsed_check, list) and len(parsed_check) <= 2:
                    all_null = all(
                        all(v in (None, "", "null", "0") for v in row.values())
                        for row in parsed_check if isinstance(row, dict)
                    )
                    if all_null:
                        logger.info("[mcp_fetch] All-null response detected, retrying...")
                        retry_result = await mcp_client.call_tool(tool_name, arguments)
                        if retry_result.get("success"):
                            retry_text = ""
                            for item in retry_result.get("response", {}).get("content", []):
                                if item.get("type") == "text":
                                    retry_text += item.get("text", "")
                            if retry_text:
                                retry_parsed = _parse_mcp_text(retry_text)
                                # Only use retry if it's better (not all null)
                                try:
                                    retry_check = json.loads(retry_parsed)
                                    retry_null = isinstance(retry_check, list) and all(
                                        all(v in (None, "", "null", "0") for v in row.values())
                                        for row in retry_check if isinstance(row, dict)
                                    )
                                    if not retry_null:
                                        mcp_text = retry_parsed
                                        tool_result = retry_result
                                except (json.JSONDecodeError, TypeError):
                                    mcp_text = retry_parsed
                                    tool_result = retry_result
            except (json.JSONDecodeError, TypeError):
                pass

        # The platform sometimes returns an error envelope ({"error":{...}}) as a 'successful'
        # result. Treat it as no-data: blank it so entity recovery + the empty path engage
        # below, instead of surfacing raw error JSON to the user.
        _platform_errored = _is_platform_error_payload(mcp_text)
        if _platform_errored:
            mcp_text = ""

        # --- Phase 1: fund entity recovery on an empty result ---------------------------
        # A fund/scheme question that came back with NOTHING is usually an exact-match miss on
        # the fund name (the platform matches names exactly, e.g. "HDFC Midcap opportunities"
        # != stored "HDFC Mid-Cap Opportunities"). Look up the real names from the platform and
        # similarity-match before giving up. Gated so it ONLY fires on an empty result that
        # actually named a fund — the happy path never pays for this.
        recovery_meta = None
        # Skip reactive recovery if we already proactively canonicalized the fund — in that case
        # an empty result is a genuine null, not a name mismatch, so we let it through as such.
        if config.fund_recovery_enabled and not proactively_resolved:
            _ent = state.get("enrichment_entities") or {}
            _user_fund = (_ent.get("fund_name") or "").strip()
            if _user_fund and _is_empty_or_null_text(mcp_text):
                logger.info("[mcp_fetch] empty result for fund %r — attempting recovery", _user_fund)
                rec = await _recover_fund_entity(tool_name, query, _user_fund, record=queries_sent)
                if rec:
                    recovery_meta = rec.get("meta")
                    if rec["status"] == "resolved":
                        tool_result = rec["tool_result"]
                        mcp_text = rec["mcp_text"]
                        logger.info("[mcp_fetch] fund recovery resolved %r -> %r",
                                    _user_fund, (recovery_meta or {}).get("resolved"))
                    elif rec["status"] == "clarify":
                        exec_ms = round((time.perf_counter() - exec_start) * 1000)
                        # Signal a clarification to _track_b via `fund_clarification`. The
                        # orchestrator short-circuits on it and uses this message DIRECTLY —
                        # generate is skipped, so the "did you mean?" question isn't
                        # overwritten by an LLM answer built on the empty payload.
                        return {
                            "messages": [AIMessage(content=rec["question"])],
                            "fund_clarification": rec["question"],
                            "mcp_tool_calls": [{**tool_result, "fund_recovery": recovery_meta,
                                                "queries_sent": queries_sent}],
                            "latency_breakdown": {
                                **state.get("latency_breakdown", {}),
                                "mcp_exec_ms": exec_ms,
                            },
                        }
        # Recovery couldn't fix a platform error envelope -> surface a clean failure (not raw
        # error JSON, and not a misleading "no data" negative).
        if _platform_errored and _is_empty_or_null_text(mcp_text):
            tool_result = {**tool_result, "success": False,
                           "error": "The data platform returned an internal error for this request. Please try again."}

        # Recovery may have issued extra platform calls — refresh the exec timer so the
        # reported latency reflects the real wall-clock cost.
        exec_ms = round((time.perf_counter() - exec_start) * 1000)

        if not mcp_text:
            mcp_text = json.dumps(tool_result, indent=2, default=str)

        # Store parsed text in tool_result for generate_response to use
        tool_result["_parsed_text"] = mcp_text
        # For reporting/observability: the exact query the agent sent to Cortex (after any
        # enrichment) plus every retry rephrasing it tried.
        tool_result["enriched_query"] = query
        tool_result["queries_sent"] = queries_sent
        if recovery_meta is not None:
            tool_result["fund_recovery"] = recovery_meta

        log_chain("fetch", _cid, path="generic", user_query=state.get("query"), enriched=query,
                  mcp_query=arguments.get("query"), success=tool_result.get("success"),
                  exec_ms=exec_ms, payload=mcp_text)

        return {
            "messages": [AIMessage(content=mcp_text)],
            "mcp_tool_calls": [tool_result],
            "latency_breakdown": {
                **state.get("latency_breakdown", {}),
                "mcp_exec_ms": exec_ms,
            },
        }

    except httpx.HTTPStatusError as e:
        elapsed_ms = round((time.perf_counter() - start) * 1000)
        logger.error(
            "[mcp_fetch] HTTP error calling tool | status=%s | tool=%s | query=%s",
            e.response.status_code,
            "unknown",
            query[:200],
        )
        error_msg = f"MCP request failed with HTTP {e.response.status_code}: {e.response.reason_phrase}"
        return {
            "messages": [AIMessage(content=f"Error fetching data: {error_msg}")],
            "mcp_tool_calls": [{"tool": "error", "error": error_msg, "status_code": e.response.status_code, "latency_ms": elapsed_ms, "success": False}],
            "latency_breakdown": {**state.get("latency_breakdown", {}), "mcp_exec_ms": elapsed_ms},
        }

    except Exception as e:
        elapsed_ms = round((time.perf_counter() - start) * 1000)
        logger.error("[mcp_fetch] Unexpected error | error=%s | query=%s", str(e), query[:200])
        return {
            "messages": [AIMessage(content=f"Error fetching data: {e}")],
            "mcp_tool_calls": [{"tool": "error", "error": str(e), "latency_ms": elapsed_ms, "success": False}],
            "latency_breakdown": {**state.get("latency_breakdown", {}), "mcp_total_ms": elapsed_ms},
        }


async def fetch_one(question: str, state: dict) -> dict:
    """Run ONE plain-English question through the full robust fetch path (retries, smart
    refinement, fund recovery, parsing) and return its single tool_result dict (carrying
    ``_parsed_text``). Reuses the mcp_fetch node so there is exactly one fetch implementation;
    used by the Execution Engine for each sub-question of a decomposed plan.

    The node's AIMessage is ignored here — the caller (Generate) builds the real answer. A fund
    clarification, if any, is surfaced on the returned dict as ``fund_clarification``.
    """
    # execute_plan emits the per-wave "sending" lines (with parallel grouping), so the
    # inner fetch must not also emit its own single "Sending: …" line.
    sub_state = {**state, "enriched_query": question, "_suppress_send_emit": True}
    out = await mcp_fetch(sub_state)
    calls = out.get("mcp_tool_calls") or []
    tr = dict(calls[0]) if calls else {
        "tool": "resolve_context", "success": False, "error": "no result", "_parsed_text": "",
    }
    if out.get("fund_clarification"):
        tr["fund_clarification"] = out["fund_clarification"]
    return tr


# --- Instrument at the bottom of the file ---
from ...observe_utils import observed_node as _observed_node


def _cortex_detail(state, result):
    r = result if isinstance(result, dict) else {}
    calls = r.get("mcp_tool_calls") or []

    def _call_summary(c):
        if not isinstance(c, dict):
            return c
        resp = c.get("response") or {}
        text = ""
        if isinstance(resp, dict):
            for item in resp.get("content", []) or []:
                if isinstance(item, dict):
                    text += item.get("text", "")
        return {
            "tool": c.get("tool") or c.get("tool_name"),
            "arguments": c.get("arguments"),
            "success": c.get("success"),
            "latency_ms": c.get("latency_ms"),
            "error": c.get("error"),
            "result": text or resp or c.get("mcp_text"),
        }

    return {
        # What the platform was asked
        "input": {
            "query": state.get("enriched_query") or state.get("query", ""),
            "tools_called": [(_call_summary(c) or {}).get("tool") for c in calls],
        },
        # The full tool calls: name, args, and what came back
        "output": [_call_summary(c) for c in calls],
        "ns_tools": ", ".join(str((_call_summary(c) or {}).get("tool")) for c in calls) or "none",
        "ns_tool_count": len(calls),
    }


mcp_fetch = _observed_node(mcp_fetch, name="data_fetch", detail=_cortex_detail)
