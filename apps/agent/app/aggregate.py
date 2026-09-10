"""Deterministic aggregation over MCP rows.

Issue #1 in the response-quality plan: the LLM was asked to sum a list of numbers in its
head and got it wrong (e.g. ₹35.55 lakh instead of the correct ₹39.04 lakh). LLMs are
unreliable at multi-row arithmetic. So we compute the sums (and per-category subtotals) in
plain Python and hand the model an AUTHORITATIVE block it copies verbatim — the same
philosophy as money.annotate_money().

Two public helpers:
  - coerce_rows(text)            -> list[dict] | None  (normalise whatever the LLM would see)
  - build_aggregate_block(rows, query) -> str          (the text injected next to the data)
"""
import json
import re

from .money import fmt_inr


# Columns whose values are genuinely ADDITIVE (summing them is meaningful). Deliberately
# stricter than money._MONEY_KEY: we must NOT sum per-unit quantities like price / NAV /
# rate / ratio (their sum is meaningless), even though those are "money" for display.
_ADDITIVE_KEY = re.compile(
    r"\b(value|amount|aum|gain|loss|pnl|p&l|invested|cost|worth|investment|redemption|"
    r"inflow|outflow|net[_ ]?new|net[_ ]?money|revenue|commission|corpus|exposure|"
    r"principal|balance|fee|proceeds)\b",
    re.IGNORECASE,
)
_NOT_ADDITIVE = re.compile(
    r"\b(price|nav|rate|ratio|percent|pct|id|date|count|number|qty|quantit|units?|"
    r"per[_ ]|average|avg|mean|expense|xirr|cagr|return|yield)\b",
    re.IGNORECASE,
)
# Short text columns that name a category we can subtotal by.
_CATEGORY_KEY = re.compile(
    r"\b(asset[_ ]?class|product[_ ]?type|investment[_ ]?type|category|sector|"
    r"segment|amc|scheme[_ ]?type|fund[_ ]?type|lob|sbu|family|tier|type)\b",
    re.IGNORECASE,
)
_DATE_NAME = re.compile(r"\b(month|date|period|quarter|year|as[_ ]?on|day|week)\b", re.IGNORECASE)
_ISO_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}")


def _is_additive(col: str) -> bool:
    # Percent / share columns must never be summed.
    if "%" in str(col) or "percent" in str(col).lower():
        return False
    # Underscores -> spaces so \b boundaries fire on snake_case columns (market_value,
    # cost_value, unrealized_pnl). See money._looks_money_col for the same rationale.
    c = str(col).replace("_", " ")
    return bool(_ADDITIVE_KEY.search(c)) and not _NOT_ADDITIVE.search(c)


def _is_timeseries(rows: list) -> bool:
    """True when the rows are a TIME SERIES (one value per distinct date/period). Summing the
    value column across time is meaningless (e.g. monthly AUM snapshots), so we must NOT emit
    a total for these. A constant snapshot date (a normal holdings table) is NOT a time series.
    """
    if len(rows) < 3:
        return False
    for col in rows[0].keys():
        name_date = bool(_DATE_NAME.search(str(col)))
        vals = [str(r.get(col, "")).strip() for r in rows if isinstance(r, dict)]
        iso_hits = sum(1 for v in vals if _ISO_DATE.match(v))
        looks_date = name_date or iso_hits >= max(2, len(rows) // 2)
        if not looks_date:
            continue
        distinct = len({v for v in vals if v})
        if distinct >= max(3, int(len(rows) * 0.7)):  # mostly-unique dates => a time axis
            return True
    return False


def _as_float(v):
    if v in (None, "", "null", "-"):
        return None
    try:
        return float(str(v).replace(",", "").replace("₹", "").strip())
    except (TypeError, ValueError):
        return None


def coerce_rows(data_text: str):
    """Best-effort: turn the data the LLM would see into a list[dict], or None.

    Handles three shapes:
      1. a JSON array of objects (the common mcp_fetch output),
      2. a single JSON object,
      3. the platform's no-braces field format, one "Key": value per line
         (e.g. `"Total AUM": 13455063632.72`) — this was previously left UNPARSED, so
         money.annotate_money() never ran on it and the LLM mis-grouped the digits (the Q1
         10x bug). Coercing it to a row lets annotate_money + aggregation work.
    """
    if not data_text:
        return None
    s = re.sub(r"^\[\d+\]", "", str(data_text).strip()).strip()  # drop a leading [count]
    if not s:
        return None
    try:
        v = json.loads(s)
        if isinstance(v, list) and v and all(isinstance(r, dict) for r in v):
            return v
        if isinstance(v, dict) and v:
            return [v]
        return None
    except (json.JSONDecodeError, TypeError):
        pass

    # Field format: every non-empty line must be `"Key": value` for us to treat it as one row.
    line_re = re.compile(r'^\s*"?([^"\n:][^"\n:]*?)"?\s*:\s*(.+?),?\s*$')
    row: dict = {}
    nonblank = 0
    for line in s.split("\n"):
        if not line.strip():
            continue
        nonblank += 1
        m = line_re.match(line)
        if not m:
            return None  # a line that isn't key:value -> not the field format
        key = m.group(1).strip()
        val = m.group(2).strip().strip('"')
        row[key] = val
    if nonblank and len(row) == nonblank:
        return [row]
    return None


def compute_sums(rows: list) -> dict:
    """Return {column: (sum, n_values)} for additive columns where EVERY non-empty cell is
    numeric and at least 2 values are present (a single value needs no sum — annotate_money
    already formats it)."""
    if not rows:
        return {}
    out: dict = {}
    keys = list(rows[0].keys())
    for col in keys:
        if not _is_additive(col):
            continue
        vals = []
        bad = False
        for r in rows:
            if not isinstance(r, dict):
                continue
            raw = r.get(col)
            if raw in (None, "", "null", "-"):
                continue
            f = _as_float(raw)
            if f is None:
                bad = True
                break
            vals.append(f)
        if bad or len(vals) < 2:
            continue
        out[col] = (sum(vals), len(vals))
    return out


def _subtotals(rows: list, sum_cols: list) -> list:
    """One per-category subtotal block when there's exactly one category column and at most
    two additive value columns. Conservative on purpose — wrong subtotals are worse than none."""
    if not rows or not sum_cols:
        return []
    cat_cols = [c for c in rows[0].keys() if _CATEGORY_KEY.search(str(c))]
    if len(cat_cols) != 1:
        return []
    cat = cat_cols[0]
    distinct = {str(r.get(cat, "")).strip() for r in rows if isinstance(r, dict)}
    distinct.discard("")
    if not (2 <= len(distinct) <= 15):
        return []
    value_col = sum_cols[0]  # subtotal the first additive value column
    groups: dict = {}
    for r in rows:
        if not isinstance(r, dict):
            continue
        key = str(r.get(cat, "")).strip() or "(blank)"
        f = _as_float(r.get(value_col))
        if f is not None:
            groups[key] = groups.get(key, 0.0) + f
    if len(groups) < 2:
        return []
    lines = [f'Subtotals of "{value_col}" by "{cat}":']
    for key, total in sorted(groups.items(), key=lambda kv: kv[1], reverse=True):
        lines.append(f"  - {key}: {fmt_inr(total)}")
    return lines


# Columns we never want to show in a collapsed holdings summary (internal keys).
_DROP_COLS = {"id", "client_id", "investor_id", "account_id", "row_id", "holding_id"}
# A query that genuinely wants the time axis — do NOT collapse it to the latest snapshot.
_TREND_RE = re.compile(
    r"\b(over time|trend|monthly|month[- ]?on[- ]?month|history|historical|by month|each month|"
    r"every month|movement|progression|growth over|time series|all snapshots?|each snapshot|"
    r"month by month|across months|since|timeline)\b",
    re.IGNORECASE,
)


def _table_value_cols(cols) -> list:
    return [c for c in cols if _is_additive(c)]


def summarize_holdings(rows, query: str = ""):
    """Collapse a holdings TIME SERIES (one row per holding per monthly snapshot, e.g. 174 rows
    = ~29 holdings × 6 months) down to the CURRENT positions, so the answer shows a concise
    one-row-per-holding table instead of every historical row.

    Two strategies, chosen by what the platform returned (its column set is non-deterministic):
      • a VARYING per-holding identifier exists (e.g. Product ID / ISIN) → keep each holding's
        own LATEST snapshot (so a holding sold earlier still appears at its last value);
      • no usable identifier (only a constant Client Name + date + values) → keep just the rows
        on the LATEST valuation date (the current snapshot). This avoids the bug where grouping
        by a constant column collapsed all 174 rows into a single meaningless row.

    Returns {"rows": [...current per holding...], "as_of": <date>, "original_count": N,
    "kept": M} or None when the data isn't a collapsible holdings series.
    """
    if not rows or not isinstance(rows, list) or len(rows) < 6:
        return None
    if _TREND_RE.search(query or ""):
        return None  # the user explicitly asked for the history — keep every snapshot
    cols = list(rows[0].keys())
    date_col = next((c for c in cols if _DATE_NAME.search(str(c).replace("_", " "))), None)
    if not date_col:
        return None
    value_cols = _table_value_cols(cols)
    if not value_cols:
        return None
    dates = {str(r.get(date_col, "")).strip() for r in rows if isinstance(r, dict)}
    dates.discard("")
    if len(dates) < 2:
        return None  # a single snapshot — already current, no collapse needed
    as_of = max(dates)

    def _is_drop(c):
        return str(c).strip().lower().replace(" ", "_") in _DROP_COLS

    # Candidate per-holding identity = columns that are neither the date, a value, nor an id.
    id_cols = [c for c in cols if c != date_col and c not in value_cols and not _is_drop(c)]
    identities = {
        tuple(str(r.get(c, "")).strip() for c in id_cols)
        for r in rows if isinstance(r, dict)
    } if id_cols else set()

    if id_cols and len(identities) > 1:
        # Real per-holding identifier — collapse each holding to its own latest snapshot.
        latest: dict = {}
        for r in rows:
            if not isinstance(r, dict):
                continue
            key = tuple(str(r.get(c, "")).strip() for c in id_cols)
            d = str(r.get(date_col, "")).strip()
            cur = latest.get(key)
            if cur is None or d > cur[0]:
                latest[key] = (d, r)
        picked = [r for _, r in latest.values()]
    else:
        # No varying identifier (or none at all) — the current snapshot is the latest date's rows.
        picked = [r for r in rows if isinstance(r, dict) and str(r.get(date_col, "")).strip() == as_of]

    if not picked or len(picked) >= len(rows):
        return None  # nothing meaningfully collapsed
    out_rows = [{k: v for k, v in r.items() if not _is_drop(k)} for r in picked]
    return {"rows": out_rows, "as_of": as_of, "original_count": len(rows), "kept": len(out_rows)}


# A query that explicitly wants every individual row — do NOT collapse it to grouped totals.
_LIST_EACH_RE = re.compile(
    r"\b(list|show|display|give me)\s+(me\s+)?(all|each|every|the individual|the full list)|"
    r"\b(each|every|individual|line[- ]?by[- ]?line|row[- ]?by[- ]?row|itemi[sz]e|itemi[sz]ed|"
    r"one by one|in detail|detailed list|full list|every single|all the rows?|all records?)\b",
    re.IGNORECASE,
)

# An INSTRUMENT-LEVEL request wants one row per holding/instrument — never a category roll-up.
# This keeps the "by instrument" holdings table (Client Holdings) from being collapsed onto a
# product_type/category column. "by product type" etc. deliberately does NOT match, so the
# "Holdings by Product" table still groups as intended.
_INSTRUMENT_LEVEL_RE = re.compile(
    r"\bby\s+(instrument|holding|scheme|fund)\b|\bper\s+(instrument|holding|scheme|fund)\b|"
    r"\b(instrument|scheme|fund)\s+name\b|\beach\s+(instrument|holding|scheme|fund)\b",
    re.IGNORECASE,
)


def _numeric_value_cols(rows: list, exclude: set) -> list:
    """Columns (excluding `exclude`) whose non-empty cells are ALL numeric and at least
    half the rows carry a value — i.e. genuine metric/value columns we can aggregate."""
    if not rows:
        return []
    out = []
    for col in rows[0].keys():
        if col in exclude:
            continue
        c = str(col).replace("_", " ")
        if _DATE_NAME.search(c) or "%" in str(col):
            continue
        if re.search(r"\b(id|count|number|year|qty|quantit|units?)\b", c, re.IGNORECASE):
            continue
        vals, bad = 0, False
        for r in rows:
            if not isinstance(r, dict):
                continue
            raw = r.get(col)
            if raw in (None, "", "null", "-"):
                continue
            if _as_float(raw) is None:
                bad = True
                break
            vals += 1
        if not bad and vals >= max(2, len(rows) // 2):
            out.append(col)
    return out


def _best_category_col(rows: list, value_cols: set) -> str | None:
    """Pick the text column to GROUP BY: the one whose values genuinely REPEAT (so listing
    every row is noise). Prefers a known category name (sector, asset class, …); otherwise
    any text column with 2..20 distinct values that each repeat. None when nothing fits."""
    cols = [c for c in rows[0].keys() if c not in value_cols]
    candidates = []
    for col in cols:
        c = str(col).replace("_", " ")
        if _DATE_NAME.search(c) or "%" in str(col):
            continue
        if re.search(r"\b(id|count|number)\b", c, re.IGNORECASE):
            continue
        vals = [str(r.get(col, "")).strip() for r in rows if isinstance(r, dict)]
        nonblank = [v for v in vals if v]
        if len(nonblank) < len(rows) * 0.6:
            continue  # mostly blank -> not a grouping key
        # A grouping column must hold short LABELS, not free-form numbers.
        if any(_as_float(v) is not None for v in nonblank[:5]):
            continue
        distinct = len(set(nonblank))
        if not (2 <= distinct <= 20):
            continue
        if distinct > len(nonblank) * 0.6:
            continue  # values barely repeat -> grouping wouldn't help
        named = bool(_CATEGORY_KEY.search(str(col)))
        # score: prefer named categories, then the most-repeated (fewest distinct) column
        candidates.append((named, -distinct, col))
    if not candidates:
        return None
    candidates.sort(reverse=True)
    return candidates[0][2]


def summarize_by_category(rows, query: str = ""):
    """Collapse RAW rows that repeat a category (e.g. 30 holdings tagged with ~5 sectors, each
    with a Risk Metric) into ONE row per category with a count and per-metric totals — so the
    answer is a clean grouped summary, not a long list of near-identical rows.

    Returns {"rows": [...one per category...], "category": <col>, "value_cols": [...],
    "original_count": N, "groups": M} or None when the data isn't a collapsible repeated-category
    list (too few rows, no repeated category, no numeric column, or the user asked to list each).
    """
    if not rows or not isinstance(rows, list) or len(rows) < 8:
        return None
    if _LIST_EACH_RE.search(query or "") or _TREND_RE.search(query or ""):
        return None
    if _INSTRUMENT_LEVEL_RE.search(query or ""):
        return None  # "by instrument" wants one row per holding — never roll up to product_type
    if _is_timeseries(rows):
        return None  # a time axis is handled by the snapshot collapsers, not category grouping
    value_cols = _numeric_value_cols(rows, exclude=set())
    if not value_cols:
        return None
    cat = _best_category_col(rows, set(value_cols))
    if not cat:
        return None
    groups: dict = {}
    order: list = []
    for r in rows:
        if not isinstance(r, dict):
            continue
        key = str(r.get(cat, "")).strip() or "(blank)"
        g = groups.get(key)
        if g is None:
            g = {"_count": 0, **{vc: 0.0 for vc in value_cols}}
            groups[key] = g
            order.append(key)
        g["_count"] += 1
        for vc in value_cols:
            f = _as_float(r.get(vc))
            if f is not None:
                g[vc] += f
    if len(groups) < 2 or len(groups) >= len(rows):
        return None  # nothing meaningfully collapsed
    out_rows = []
    for key in sorted(order, key=lambda k: groups[k][value_cols[0]], reverse=True):
        g = groups[key]
        row = {cat: key, "Count": g["_count"]}
        for vc in value_cols:
            row[f"Total {vc}"] = round(g[vc], 2)
        out_rows.append(row)
    return {
        "rows": out_rows,
        "category": cat,
        "value_cols": value_cols,
        "original_count": len(rows),
        "groups": len(out_rows),
    }


def build_aggregate_block(rows, query: str = "") -> str:
    """The authoritative-totals text injected next to the data. Empty string when there is
    nothing meaningful to sum."""
    if not rows or not isinstance(rows, list):
        return ""
    if _is_timeseries(rows):
        return ""  # summing a time series (e.g. monthly AUM) is meaningless
    sums = compute_sums(rows)
    if not sums:
        return ""
    lines = [
        "AUTHORITATIVE AGGREGATES (computed in code from the rows above — these are EXACT). "
        "Whenever you state a total/sum, copy the matching figure from here VERBATIM and do NOT "
        "add the rows up yourself. Do not invent a totals row for a plain list unless the user "
        "asked for a total.",
    ]
    for col, (total, n) in sums.items():
        lines.append(f'- Sum of "{col}": {fmt_inr(total)}  [{n} values]')
    sub = _subtotals(rows, list(sums.keys()))
    if sub:
        lines.append("")
        lines.extend(sub)
    return "\n".join(lines)
