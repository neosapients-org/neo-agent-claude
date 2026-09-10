"""Deterministic Indian-rupee formatting + money annotation.

Shared by mcp_fetch (AUM handler) and generate (pre-formatting data before the LLM sees
it). Formatting is done in CODE so the model can never mis-group digits — the cause of the
10x errors we saw (e.g. ₹16,33,02,516 shown as "₹163 crore").
"""
import json
import re


def inr_group(digits: str) -> str:
    """Group an integer-digit string in the Indian system (16629584 -> 1,66,29,584)."""
    if len(digits) <= 3:
        return digits
    head, last3 = digits[:-3], digits[-3:]
    parts = []
    while len(head) > 2:
        parts.insert(0, head[-2:])
        head = head[:-2]
    if head:
        parts.insert(0, head)
    return ",".join(parts) + "," + last3


def _grouped_decimal(x: float) -> str:
    """Indian-grouped number with exactly 2 decimals, e.g. 1345.5 -> '1,345.50'."""
    total_cents = int(round(abs(x) * 100))
    intp, cents = divmod(total_cents, 100)
    return f"{inr_group(str(intp))}.{cents:02d}"


def fmt_inr(value) -> str:
    """Format a number as a COMPACT ₹ amount.

    >=1 crore -> "₹77.23 crore"; >=1 lakh (and <1 crore) -> "₹15.61 lakh";
    below ₹1 lakh -> the exact grouped figure "₹45,230.00" (no suffix to round to).
    """
    try:
        v = float(str(value).replace(",", "").replace("₹", "").strip())
    except (TypeError, ValueError):
        return str(value)
    neg = v < 0
    av = abs(v)
    if av >= 1_00_00_000:
        body = f"₹{_grouped_decimal(av / 1_00_00_000)} crore"
    elif av >= 1_00_000:
        body = f"₹{_grouped_decimal(av / 1_00_000)} lakh"
    else:
        total_cents = int(round(av * 100))
        intp, cents = divmod(total_cents, 100)
        body = f"₹{inr_group(str(intp))}.{cents:02d}"
    return ("-" + body) if neg else body


# Column names that hold monetary amounts (so we can pre-format their values).
_MONEY_KEY = re.compile(
    r"\b(value|amount|aum|gain|loss|pnl|p&l|revenue|commission|price|invested|current|inflow|"
    r"outflow|net[_ ]?new|net[_ ]?money|cost|worth|nav|sum|investment|redemption|"
    r"portfolio|fee|corpus)\b",
    re.IGNORECASE,
)
# Columns that are NOT money even though they may match above (avoid mangling ids/ratios/ranks).
_NOT_MONEY_KEY = re.compile(
    r"\b(id|date|count|number|pct|percent|ratio|units?|quantity|rate|"
    r"rank|ranking|position|serial|sr|sno|seq|sequence|index|ordinal)\b", re.IGNORECASE)


def _looks_money_col(key: str) -> bool:
    # Percent / share columns (e.g. "% Current Value", "% of Total") are NOT money — never
    # format them as ₹.
    if "%" in str(key) or "percent" in str(key).lower():
        return False
    # Normalise underscores to spaces so the \b word boundaries fire on the platform's
    # snake_case column names (e.g. "total_aum", "market_value", "cost_value"). Without this,
    # "\baum\b" never matches inside "total_aum" — the silent cause of the 10x AUM bug, since
    # un-annotated billions were then mis-scaled by the LLM.
    k = str(key).replace("_", " ")
    return bool(_MONEY_KEY.search(k)) and not _NOT_MONEY_KEY.search(k)


def annotate_money(data_text: str) -> str:
    """Given the parsed MCP JSON (a list of row dicts), add a pre-formatted ₹ companion
    field for every monetary column, so the LLM copies the correct ₹ string instead of
    grouping digits itself. The raw numeric value is kept for any arithmetic.

    Returns the original text unchanged if it isn't a JSON list of objects.
    """
    try:
        data = json.loads(data_text)
    except (json.JSONDecodeError, TypeError):
        return data_text
    if not isinstance(data, list) or not data or not all(isinstance(r, dict) for r in data):
        return data_text

    changed = False
    for row in data:
        for k in list(row.keys()):
            if not _looks_money_col(k):
                continue
            v = row[k]
            if v in (None, "", "null", "-"):
                continue
            try:
                f = float(str(v).replace(",", ""))
            except (TypeError, ValueError):
                continue
            companion = f"{k} (₹)"
            if companion not in row:
                row[companion] = fmt_inr(f)
                changed = True
    return json.dumps(data) if changed else data_text
