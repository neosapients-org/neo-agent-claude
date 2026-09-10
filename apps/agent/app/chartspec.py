"""Conservative chart-spec builder (issue #2).

Decides — IN CODE, never via the LLM — whether a chart helps for a given query + payload,
and emits a small Chart.js-shaped spec ({type, labels, datasets}). The data points come
straight from the already-parsed rows, so a chart can never disagree with the table.

Golden rule: when unsure, return None. A missing chart is invisible; a wrong chart is a
bug the user sees. So every branch is intentionally narrow.
"""
import re
from datetime import datetime

_DATE_NAME = re.compile(r"\b(month|date|period|quarter|year|as[_ ]?on|day|week)\b", re.IGNORECASE)
_ISO_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}")

_TREND_WORDS = ("trend", "over time", "over the last", "over the past", "monthly", "quarterly",
                "month over month", "history", "historical", "movement", "growth", "yoy",
                "year on year", "by month", "per month", "timeline")
_BREAKDOWN_WORDS = ("split", "breakdown", "break down", "allocation", "distribution",
                    "composition", "mix", "by asset class", "by product", "by sector",
                    "by type", "exposure across", "spread across")
_RANK_WORDS = ("top ", "highest", "lowest", "rank", "ranked", "most ", "largest", "biggest",
               "per ", "by amc", "by rm", "by banker", "by lob", "by client", "which ")

_MONTHS = ["", "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def _as_float(v):
    if v in (None, "", "null", "-"):
        return None
    try:
        return float(str(v).replace(",", "").replace("₹", "").strip())
    except (TypeError, ValueError):
        return None


def _numeric_cols(rows: list) -> list:
    """Columns where at least 2 cells are present and EVERY non-empty cell is numeric."""
    cols = []
    for col in rows[0].keys():
        n = 0
        ok = True
        for r in rows:
            raw = r.get(col) if isinstance(r, dict) else None
            if raw in (None, "", "null", "-"):
                continue
            if _as_float(raw) is None:
                ok = False
                break
            n += 1
        if ok and n >= 2:
            cols.append(col)
    return cols


def _is_date_col(col: str, rows: list) -> bool:
    if _DATE_NAME.search(str(col)):
        # confirm at least some values look date-like, else trust the name
        return True
    hit = sum(1 for r in rows if isinstance(r, dict) and _ISO_DATE.match(str(r.get(col, "")).strip()))
    return hit >= max(2, len(rows) // 2)


def _fmt_date(v: str) -> str:
    s = str(v).strip()
    m = re.match(r"^(\d{4})-(\d{2})-(\d{2})", s)
    if m:
        y, mo = int(m.group(1)), int(m.group(2))
        return f"{_MONTHS[mo]} {y}" if 1 <= mo <= 12 else f"{m.group(1)}-{m.group(2)}"
    return s[:16]


def _label_cols(rows: list, numeric: list) -> list:
    """Short text columns usable as category labels (not the numeric ones, not ids)."""
    out = []
    for col in rows[0].keys():
        if col in numeric:
            continue
        if re.search(r"\b(id|isin|product id|code)\b", str(col), re.IGNORECASE):
            continue
        vals = [str(r.get(col, "")).strip() for r in rows if isinstance(r, dict)]
        nonblank = [v for v in vals if v]
        if len(nonblank) < 2:
            continue
        if len(set(nonblank)) < 2:  # all the same -> useless as a category
            continue
        out.append(col)
    return out


def _scale(values: list):
    """Pick a readable unit for big rupee numbers. Returns (scaled_values, unit_label)."""
    mx = max((abs(v) for v in values), default=0)
    if mx >= 1_00_00_000:
        return [round(v / 1_00_00_000, 2) for v in values], "₹ crore"
    if mx >= 1_00_000:
        return [round(v / 1_00_000, 2) for v in values], "₹ lakh"
    return [round(v, 2) for v in values], ""


def build_chart_spec(query: str, rows, aggregates: dict | None = None) -> dict | None:
    """Return a Chart.js-shaped dict {type, title, unit, labels, datasets} or None."""
    if not rows or not isinstance(rows, list) or len(rows) < 2 or len(rows) > 60:
        return None
    if not all(isinstance(r, dict) for r in rows):
        return None
    q = (query or "").lower()
    numeric = _numeric_cols(rows)
    if not numeric:
        return None

    # --- 1. Time series -> line ------------------------------------------------------
    date_cols = [c for c in rows[0].keys() if _is_date_col(c, rows)]
    wants_trend = any(w in q for w in _TREND_WORDS)
    if date_cols and (wants_trend or len(date_cols) == 1):
        date_col = date_cols[0]
        value_col = next((c for c in numeric if c != date_col), None)
        if value_col:
            pairs = [(r.get(date_col), _as_float(r.get(value_col))) for r in rows]
            pairs = [(d, v) for d, v in pairs if d not in (None, "", "null") and v is not None]
            if len(pairs) >= 2:
                labels = [_fmt_date(d) for d, _ in pairs]
                vals, unit = _scale([v for _, v in pairs])
                lbl = value_col + (f" ({unit})" if unit else "")
                return {
                    "type": "line", "title": f"{value_col} over time", "unit": unit,
                    "labels": labels, "datasets": [{"label": lbl, "data": vals}],
                }

    # --- 2. Category breakdown / ranking -> pie or bar -------------------------------
    labels_cols = _label_cols(rows, numeric)
    if labels_cols and len(rows) <= 30:
        label_col = labels_cols[0]
        value_col = numeric[0]
        pairs = [(str(r.get(label_col, "")).strip(), _as_float(r.get(value_col))) for r in rows]
        pairs = [(k, v) for k, v in pairs if k and v is not None]
        # collapse duplicate labels (sum) so a per-holding list becomes a clean breakdown
        agg: dict = {}
        for k, v in pairs:
            agg[k] = agg.get(k, 0.0) + v
        if 2 <= len(agg) <= 15:
            wants_breakdown = any(w in q for w in _BREAKDOWN_WORDS)
            wants_rank = any(w in q for w in _RANK_WORDS)
            if not (wants_breakdown or wants_rank):
                return None  # don't chart an arbitrary list nobody asked to visualise
            items = sorted(agg.items(), key=lambda kv: kv[1], reverse=True)
            labels = [k for k, _ in items]
            vals, unit = _scale([v for _, v in items])
            lbl = value_col + (f" ({unit})" if unit else "")
            # pie for a small breakdown; bar for ranking or many slices
            ctype = "pie" if (wants_breakdown and not wants_rank and len(items) <= 8) else "bar"
            title = f"{value_col} by {label_col}"
            return {
                "type": ctype, "title": title, "unit": unit,
                "labels": labels, "datasets": [{"label": lbl, "data": vals}],
            }

    return None
