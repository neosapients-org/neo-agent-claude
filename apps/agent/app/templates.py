"""Domain presentation templates.

Detect the KIND of wealth question (holdings / fund / asset allocation) and:
  (a) nudge the platform query to include the standard fields advisors expect for that kind,
  (b) compute the derived columns (share %) deterministically in code, and
  (c) hand `generate` a STANDARD COLUMN FRAME so the answer is consistent even when the
      platform omits some fields (those show as "—" rather than being silently dropped).

This is intentionally lightweight and heuristic — the classifier errs toward the generic
path (returns None) when unsure, so a misclassification can only ADD a column hint, never
break a normal answer.
"""

import re

from .aggregate import _is_additive, _DATE_NAME
from .money import fmt_inr

# Holdings & Fund standard-frame templates. DISABLED — these forced a fixed standard column
# frame (Holding Name | Investment Name | … | XIRR | Benchmark …) onto every holdings/fund
# question, instructing the LLM (and the code table builders) to KEEP every column and show "—"
# for fields the platform never returned. That produced a "fixed" table padded with "—"
# placeholders even when the platform sent little or no data. With this off, holdings/fund
# questions are answered from whatever the platform actually returns (no synthetic frame).
# Asset-allocation templates are unaffected and stay enabled.
_HF_TEMPLATES_ENABLED = False

# --- question classification --------------------------------------------------------------
# Broadened to also catch "split of all investment types", "by product type", etc. — any
# breakdown/allocation question should get the share-% column.
_ASSET_ALLOC_RE = re.compile(
    r"\b(asset[- ]?(class|allocation|mix|split|breakdown|type)|"
    r"investment[- ]?type|product[- ]?type|"
    r"allocation|breakdown|composition|distribution|"
    r"split\s+(of|by|across|between)|break[- ]?up|diversification|how is .* allocated|"
    r"(mix|exposure)\s+(of|by|across)|"
    r"by\s+(asset|product|investment|instrument|fund|scheme|sector|amc)[- ]?(class|type|category)?)\b",
    re.IGNORECASE,
)
_FUND_PERF_RE = re.compile(
    r"\b(returns?|cagr|trailing|annuali[sz]ed|performance|nav|expense ratio|"
    r"\d+[- ]?(month|months|mo|year|years|yr|yrs)\s*(return|performance)?|"
    r"how (has|did) .* (perform|return)|best[- ]?performing|top[- ]?performing) (fund|scheme)?",
    re.IGNORECASE,
)
_FUND_NOUN_RE = re.compile(r"\b(fund|funds|scheme|schemes|mutual fund|amc)\b", re.IGNORECASE)
_HOLDINGS_RE = re.compile(
    r"\b(holding|holdings|positions?|portfolio|securities|invested in|"
    r"current (positions|portfolio))\b"
    r"|what .{0,40}\b(hold|holds|holding|own|owns)\b",
    re.IGNORECASE,
)


def classify_question(query: str) -> str | None:
    """Return 'asset_allocation' | 'holdings' | 'fund' | None."""
    q = query or ""
    if _ASSET_ALLOC_RE.search(q):
        return "asset_allocation"
    # A fund-PERFORMANCE question (returns / NAV / expense ratio) takes the fund frame even if
    # it mentions a portfolio; a plain "holdings/positions" question is holdings.
    if _FUND_PERF_RE.search(q) and _FUND_NOUN_RE.search(q):
        return "fund"
    if _HOLDINGS_RE.search(q):
        return "holdings"
    if _FUND_PERF_RE.search(q):
        return "fund"
    return None


# --- platform-query augmentation ----------------------------------------------------------
_AUGMENT = {
    "holdings": (
        " For each holding include if available: holding/scheme name, current market value, "
        "invested (cost) value, unrealized gain, XIRR, benchmark XIRR, benchmark name."
    ),
    "fund": (
        " For each fund include if available: fund name and trailing returns for 1 month, "
        "6 months, 1 year and 3 years."
    ),
    "asset_allocation": (
        " Break the result down by asset class with each asset class's current value."
    ),
}


def augment_query(query: str, kind: str | None) -> str:
    """Append a hint asking the platform for the standard fields of this question kind."""
    if not kind:
        return query
    if kind in ("holdings", "fund") and not _HF_TEMPLATES_ENABLED:
        return query  # holdings/fund templates disabled — don't reshape the query
    add = _AUGMENT.get(kind, "")
    if not add or add.strip().lower() in (query or "").lower():
        return query
    return f"{query.rstrip()}.{add}" if not query.rstrip().endswith((".", "?")) else f"{query.rstrip()}{add}"


# --- standard column frames (presentation guidance for generate) --------------------------
_HOLDINGS_FRAME = (
    "\n\nHOLDINGS QUESTION — present the holdings as ONE table using this STANDARD column "
    "frame, in this order:\n"
    "Holding Name | Investment Name | Present Value (Invested Value) | Unrealized Gain | XIRR | "
    "Benchmark XIRR | Benchmark Name | % Current Value\n"
    "- Use the data's values for every column it provides. Map obvious synonyms (product id → "
    "Holding Name; scheme/security name or ISIN → Investment Name; cost/invested value → Present "
    "Value (Invested Value); unrealized pnl → Unrealized Gain).\n"
    "- '% Current Value' has ALREADY been computed for you in the data (it sums to 100%); copy it "
    "verbatim — do not recompute.\n"
    "- If the platform did not return a column (e.g. Investment Name, XIRR, Benchmark XIRR, "
    "Benchmark Name), keep the column and show '—' for it. Do NOT drop the column and do NOT "
    "invent a value."
)
_FUND_FRAME = (
    "\n\nFUND QUESTION — present the fund(s) as a table using this STANDARD column frame:\n"
    "Fund Name | 1M Return | 6M Return | 1Y Return | 3Y Return\n"
    "- Add any extra return period or metric the data provides as further columns.\n"
    "- If a return period was not returned by the platform, keep the column and show '—'. "
    "Do NOT invent returns."
)
_ASSET_FRAME = (
    "\n\nASSET ALLOCATION — present each asset class as a row with its Current Value and the "
    "'% of Total' column (already computed; copy verbatim — the percentages sum to 100%). Order "
    "rows by value, largest first, and add a final Total row (100%)."
)


# --- derived columns ----------------------------------------------------------------------
def _to_float(v):
    if v in (None, "", "null", "-", "—"):
        return None
    try:
        return float(str(v).replace(",", "").replace("₹", "").replace("%", "").strip())
    except (TypeError, ValueError):
        return None


def _pick_value_col(cols) -> str | None:
    """The column that best represents 'current value' for share-of-total maths."""
    for pat in (r"current.*value|market.*value", r"\bcurrent[_ ]?value\b",
                r"\bmarket[_ ]?value\b", r"\bcurrent\b", r"\bvalue\b", r"\bamount\b", r"\baum\b"):
        for c in cols:
            if "%" not in str(c) and re.search(pat, str(c).replace("_", " "), re.IGNORECASE):
                return c
    for c in cols:
        if "%" not in str(c) and _is_additive(c):
            return c
    return None


def add_share_percent(rows: list, value_col: str, pct_label: str) -> list:
    """Append `pct_label` to every row = that row's value / total * 100 (1 dp), adjusted so the
    column sums to exactly 100.0. Rows with no numeric value get '—'."""
    if not rows or not value_col:
        return rows
    vals = [_to_float(r.get(value_col)) for r in rows]
    total = sum(v for v in vals if v is not None)
    if not total:
        return rows
    pcts = [round(v / total * 100, 1) if v is not None else None for v in vals]
    known = [(i, p) for i, p in enumerate(pcts) if p is not None]
    drift = round(100.0 - sum(p for _, p in known), 1)
    if known and abs(drift) >= 0.1:
        imax = max(known, key=lambda ip: ip[1])[0]
        pcts[imax] = round(pcts[imax] + drift, 1)
    return [
        {**r, pct_label: (f"{pcts[i]:.1f}%" if pcts[i] is not None else "—")}
        for i, r in enumerate(rows)
    ]


def _collapse_to_current(rows: list) -> list:
    """If the rows are a monthly time series (a date column with >1 distinct values), keep only
    the LATEST date's rows — the current snapshot. Fixes fund-return tables that repeat one fund
    across 12 months, and asset-allocation values inflated by summing every month."""
    if not rows:
        return rows
    cols = list(rows[0].keys())
    date_col = next((c for c in cols if _DATE_NAME.search(str(c).replace("_", " "))), None)
    if not date_col:
        return rows
    dates = {str(r.get(date_col, "")).strip() for r in rows if isinstance(r, dict)}
    dates.discard("")
    if len(dates) < 2:
        return rows
    as_of = max(dates)
    latest = [r for r in rows if isinstance(r, dict) and str(r.get(date_col, "")).strip() == as_of]
    return latest or rows


def apply_template(rows: list, query: str):
    """Compute derived columns and pick the standard column frame for this question kind.

    Returns (rows, instruction). `rows` may have an extra share-% column added; `instruction`
    is the presentation guidance appended to the generate prompt (empty when no template fits).
    """
    kind = classify_question(query)
    if not kind or not rows or not isinstance(rows, list):
        return rows, ""
    if kind == "asset_allocation":
        # Breakdown / allocation question (asset class, investment type, product type, sector…):
        # collapse to the current snapshot and add the share-% column (sums to 100%).
        rows = _collapse_to_current(rows)  # current snapshot only — don't sum across months
        vcol = _pick_value_col(list(rows[0].keys()))
        if vcol:
            rows = add_share_percent(rows, vcol, "% of Total")
        return rows, _ASSET_FRAME
    # --- Holdings & Fund standard frames are commented out (disabled) for now -----------------
    if kind == "holdings" and _HF_TEMPLATES_ENABLED:
        # holdings were already collapsed per-holding by aggregate.summarize_holdings
        vcol = _pick_value_col(list(rows[0].keys()))
        if vcol:
            rows = add_share_percent(rows, vcol, "% Current Value")
        return rows, _HOLDINGS_FRAME
    if kind == "fund" and _HF_TEMPLATES_ENABLED:
        rows = _collapse_to_current(rows)  # one row per fund (latest snapshot), not 12 months
        # Funds often come back as N monthly observations with NO date column — collapse can't
        # help, so dedupe by fund name keeping the LAST row (the platform orders oldest→newest).
        name_col = _find_col(list(rows[0].keys()), r"fund\s*name|scheme\s*name|\bfund\b|\bscheme\b") \
            or _find_col(list(rows[0].keys()), r"\bname\b")
        if name_col:
            seen = {}
            for r in rows:
                seen[str(r.get(name_col, "")).strip()] = r
            if 0 < len(seen) < len(rows):
                rows = list(seen.values())
        return rows, _FUND_FRAME
    return rows, ""


# --- deterministic table rendering (no LLM) -----------------------------------------------
# Questions that need INTERPRETATION (filtering / ranking / comparison) or ANALYSIS (causal /
# "how might X affect…" / impact) — leave those to the LLM (which gets the data + web context
# and writes the analysis). A plain "show / list" question is rendered in code.
_INTERPRETIVE_RE = re.compile(
    r"\b(loss|losing|underwater|in the red|profit|profitable|gaining|in the green|"
    r"top|bottom|best|worst|highest|lowest|largest|smallest|more than|less than|greater|"
    r"above|below|over \d|under \d|compare|comparison|versus|vs\b|rank|sorted?|only show|"
    r"affect|affects|affected|impact|impacts|impacted|exposed|at risk|why\b|"
    r"how (might|would|will|could|does|do)|because of|due to|in light of|what if|"
    r"hedge|mitigate|protect|scenario|implication)\b",
    re.IGNORECASE,
)


def _find_col(cols, pattern):
    rx = re.compile(pattern, re.IGNORECASE)
    for c in cols:
        if "%" in str(c):
            continue
        if rx.search(str(c).replace("_", " ")):
            return c
    return None


def _val(row, col):
    if not col:
        return None
    v = row.get(col)
    return None if v in (None, "", "null", "-", "—") else v


def _money(row, col):
    v = _val(row, col)
    return fmt_inr(v) if v is not None else "—"


def _pct_cell(row, col):
    v = _val(row, col)
    if v is None:
        return "—"
    s = str(v).strip()
    return s if s.endswith("%") else f"{s}%"


def _md_table(headers, rows_cells):
    aligns = ["---" if i == 0 else "---:" for i in range(len(headers))]
    out = ["| " + " | ".join(headers) + " |", "| " + " | ".join(aligns) + " |"]
    for r in rows_cells:
        out.append("| " + " | ".join(r) + " |")
    return "\n".join(out)


def _holdings_table(rows):
    cols = list(rows[0].keys())
    holding = (_find_col(cols, r"holding\s*name") or _find_col(cols, r"\bproduct\s*id\b|product\s*name")
               or _find_col(cols, r"\bname\b"))
    invname = (_find_col(cols, r"(scheme|fund|investment|instrument|security)\s*name")
               or _find_col(cols, r"investment\s*name") or _find_col(cols, r"\bisin\b"))
    if invname == holding:  # don't show the same column twice
        invname = _find_col(cols, r"\bisin\b")
        if invname == holding:
            invname = None
    inv = _find_col(cols, r"present\s*value|invested\s*value|cost\s*value|\bcost\b|\binvested\b")
    gain = _find_col(cols, r"unrealized.*(gain|pnl|p&l|profit)|\bgain\b")
    xirr = _find_col(cols, r"\bxirr\b(?!.*bench)")
    bxirr = _find_col(cols, r"benchmark.*xirr|bench.*xirr")
    bname = _find_col(cols, r"benchmark.*name|benchmark$|index\s*name")
    pct = "% Current Value" if "% Current Value" in cols else None
    if not holding or not (inv or gain or pct):
        return None  # can't map confidently -> let the LLM render
    headers = ["Holding Name", "Investment Name", "Present Value (Invested Value)",
               "Unrealized Gain", "XIRR", "Benchmark XIRR", "Benchmark Name", "% Current Value"]
    cells = []
    for r in rows:
        cells.append([
            str(_val(r, holding) or "—"),
            str(_val(r, invname) or "—"),
            _money(r, inv), _money(r, gain),
            (str(_val(r, xirr)) if _val(r, xirr) is not None else "—"),
            (str(_val(r, bxirr)) if _val(r, bxirr) is not None else "—"),
            str(_val(r, bname) or "—"), _pct_cell(r, pct),
        ])
    return _md_table(headers, cells)


def _asset_table(rows):
    cols = list(rows[0].keys())
    cls = (_find_col(cols, r"asset\s*class|asset\s*type|investment\s*type|product\s*type|"
                           r"\bclass\b|category|sector|segment|instrument\s*type|\btype\b")
           or _find_col(cols, r"\bname\b"))
    val = _pick_value_col(cols)
    if not cls or not val:
        return None
    label = str(cls)  # keep the platform's own label (e.g. "Investment Type")
    pct = "% of Total" if "% of Total" in cols else None
    headers = [label, "Current Value", "% of Total"]
    cells = [[str(_val(r, cls) or "—"), _money(r, val), _pct_cell(r, pct)] for r in rows]
    # total row
    total = 0.0
    for r in rows:
        v = _val(r, val)
        try:
            total += float(str(v).replace(",", "").replace("₹", ""))
        except (TypeError, ValueError):
            pass
    cells.append(["**Total**", f"**{fmt_inr(total)}**", "**100.0%**"])
    return _md_table(headers, cells)


def _fund_table(rows):
    cols = list(rows[0].keys())
    name = _find_col(cols, r"fund\s*name|scheme\s*name|\bfund\b|\bscheme\b") or _find_col(cols, r"\bname\b")
    periods = [("1M Return", r"\b1\s*m(onth)?\b"), ("6M Return", r"\b6\s*m(onth)?\b"),
               ("1Y Return", r"\b1\s*y(ear|r)?\b"), ("3Y Return", r"\b3\s*y(ear|r)?\b")]
    pcols = [(label, _find_col(cols, pat)) for label, pat in periods]
    if not name or not any(c for _, c in pcols):
        return None
    headers = ["Fund Name"] + [label for label, _ in pcols]
    cells = []
    for r in rows:
        row_cells = [str(_val(r, name) or "—")]
        for _, col in pcols:
            v = _val(r, col)
            row_cells.append(f"{v}%" if (v is not None and not str(v).strip().endswith("%")) else (str(v) if v is not None else "—"))
        cells.append(row_cells)
    return _md_table(headers, cells)


def render_table_answer(rows, query: str, meta: dict | None = None):
    """Build the answer table in CODE (no LLM) for a plain breakdown/allocation question (and,
    when _HF_TEMPLATES_ENABLED, holdings/fund). Returns the full markdown answer string, or None
    to fall back to the LLM (interpretive question, columns can't be mapped, or kind disabled)."""
    if not rows or not isinstance(rows, list):
        return None
    if _INTERPRETIVE_RE.search(query or ""):
        return None
    kind = classify_question(query)
    _builders = {"asset_allocation": _asset_table}
    if _HF_TEMPLATES_ENABLED:
        _builders["holdings"] = _holdings_table
        _builders["fund"] = _fund_table
    if kind not in _builders:
        return None
    meta = meta or {}
    table = _builders[kind](rows)
    if not table:
        return None
    as_of = meta.get("as_of")
    asof_txt = f" (as of {as_of})" if as_of else ""
    if kind == "holdings":
        head = f"**Current holdings** — {len(rows)} position{'s' if len(rows) != 1 else ''}{asof_txt}:"
        note = "\n\n_Values are the latest snapshot per holding. “—” means the platform did not return that field._"
    elif kind == "asset_allocation":
        # use the breakdown dimension's own label (e.g. "Investment Type") for the headline
        _dim = table.split("|", 2)[1].strip() if "|" in table else "category"
        head = f"**Breakdown by {_dim}**{asof_txt} — with each line's share of the total:"
        note = ""
    else:
        head = "**Fund returns:**"
        note = "\n\n_“—” means the platform did not return that period._"
    return f"{head}\n\n{table}{note}"
