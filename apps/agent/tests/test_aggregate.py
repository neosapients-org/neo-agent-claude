"""Tests for deterministic aggregation (issue #1) — the Q13 sum + Q1 grouping fixes."""
import json

from app.aggregate import (
    coerce_rows, compute_sums, build_aggregate_block, _is_timeseries,
    summarize_by_category,
)


# --- coerce_rows -----------------------------------------------------------------

def test_coerce_json_list():
    rows = coerce_rows('[{"a": "1"}, {"a": "2"}]')
    assert rows == [{"a": "1"}, {"a": "2"}]


def test_coerce_field_format_single_value_q1():
    # The platform's no-braces field format that was previously left unparsed (Q1 10x bug).
    rows = coerce_rows('"Total AUM": 13455063632.72')
    assert rows == [{"Total AUM": "13455063632.72"}]


def test_coerce_multi_field_format():
    rows = coerce_rows('AMC: null\nAUM: 624718834.68')
    assert rows == [{"AMC": "null", "AUM": "624718834.68"}]


def test_coerce_non_tabular_returns_none():
    assert coerce_rows("Here is some prose answer with no structure.") is None
    assert coerce_rows("[]") is None
    assert coerce_rows("") is None


# --- compute_sums / build_aggregate_block ----------------------------------------

def test_q13_sum_is_exact():
    vals = [85292, 137821.65, 132504.97, 311203.62, 1585092.87, 788600.85, 162645.67, 700606.78]
    rows = [{"Total Unrealized Gain": str(v)} for v in vals]
    sums = compute_sums(rows)
    assert "Total Unrealized Gain" in sums
    total, n = sums["Total Unrealized Gain"]
    assert n == 8
    assert round(total, 2) == 3903768.41
    block = build_aggregate_block(rows, "unrealized gain")
    assert "₹39.04 lakh" in block  # compact form


def test_single_value_no_sum_emitted():
    # n < 2 -> no aggregate line (annotate_money handles the single value's display).
    assert build_aggregate_block([{"Total AUM": "13455063632.72"}], "total aum") == ""


def test_non_additive_columns_skipped():
    # Price / NAV / quantity / count must never be summed.
    rows = [{"Current Price": "100", "NAV": "12.5", "Quantity": "5", "Count": "3"} for _ in range(4)]
    assert compute_sums(rows) == {}


def test_mixed_non_numeric_column_skipped():
    rows = [{"Current Value": "100"}, {"Current Value": "n/a"}, {"Current Value": "200"}]
    assert compute_sums(rows) == {}  # a non-numeric cell present -> not summable


def test_timeseries_is_not_summed():
    # Monthly AUM snapshots: summing across time is meaningless (Q2 bogus-total fix).
    rows = [
        {"Month": "2025-12-01T00:00:00.000Z", "AUM": "1375295173.37"},
        {"Month": "2026-01-01T00:00:00.000Z", "AUM": "1402498814.15"},
        {"Month": "2026-02-01T00:00:00.000Z", "AUM": "1429702454.94"},
    ]
    assert _is_timeseries(rows) is True
    assert build_aggregate_block(rows, "aum trend monthly") == ""


def test_snapshot_table_is_not_timeseries():
    # Same As On Date across rows -> a normal holdings snapshot, sums allowed.
    rows = [
        {"As On Date": "2026-05-19", "Current Value": "100"},
        {"As On Date": "2026-05-19", "Current Value": "200"},
        {"As On Date": "2026-05-19", "Current Value": "300"},
    ]
    assert _is_timeseries(rows) is False
    block = build_aggregate_block(rows, "holdings")
    assert 'Sum of "Current Value"' in block


# --- summarize_by_category (grouping repeated-category raw rows) ------------------

def _sector_risk_rows():
    raw = [
        ("Defence & Aerospace", 72140.84), ("Defence & Aerospace", 56957.44),
        ("Defence & Aerospace", 51486.76), ("Defence & Aerospace", 54657.83),
        ("Defence & Aerospace", 43077.46), ("Defence & Aerospace", 38704.23),
        ("Automobiles & Ancillaries", 218446.87), ("Automobiles & Ancillaries", 211079.37),
        ("Automobiles & Ancillaries", 143940.74), ("Automobiles & Ancillaries", 99048.16),
        ("Banking & Financial Services", 680720.94), ("Banking & Financial Services", 120000.0),
    ]
    return [{"Sector": s, "Risk Metric": v} for s, v in raw]


def test_group_repeated_category_collapses_and_totals():
    out = summarize_by_category(_sector_risk_rows(), "what sectors are at risk")
    assert out is not None
    assert out["category"] == "Sector"
    assert out["value_cols"] == ["Risk Metric"]
    assert out["original_count"] == 12
    assert out["groups"] == 3
    rows = out["rows"]
    assert [r["Sector"] for r in rows][0] == "Banking & Financial Services"  # largest total first
    auto = next(r for r in rows if r["Sector"] == "Automobiles & Ancillaries")
    assert auto["Count"] == 4
    assert auto["Total Risk Metric"] == round(218446.87 + 211079.37 + 143940.74 + 99048.16, 2)


def test_group_skips_when_user_asks_to_list_each():
    assert summarize_by_category(_sector_risk_rows(), "list every individual holding by sector") is None


def test_group_skips_unique_rows():
    rows = [{"Holding": f"Fund {i}", "Current Value": float(i) * 1000} for i in range(10)]
    assert summarize_by_category(rows, "show holdings") is None


def test_group_skips_short_lists():
    rows = [{"Sector": "Tech", "Risk Metric": 1.0}, {"Sector": "Tech", "Risk Metric": 2.0}]
    assert summarize_by_category(rows, "sectors at risk") is None


def test_group_skips_timeseries():
    rows = [
        {"Month": f"2026-0{i}-01", "Sector": "Tech", "AUM": str(i * 100)}
        for i in range(1, 9)
    ]
    assert summarize_by_category(rows, "aum over time") is None
