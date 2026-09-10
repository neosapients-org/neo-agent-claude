"""Tests for the conservative chart-spec builder (issue #2)."""
from app.chartspec import build_chart_spec


def test_timeseries_line():
    rows = [
        {"Month": "2025-12-01T00:00:00.000Z", "AUM": "1375295173.37"},
        {"Month": "2026-01-01T00:00:00.000Z", "AUM": "1402498814.15"},
        {"Month": "2026-02-01T00:00:00.000Z", "AUM": "1429702454.94"},
    ]
    spec = build_chart_spec("Show AUM trend monthly", rows)
    assert spec and spec["type"] == "line"
    assert spec["unit"] == "₹ crore"
    assert spec["labels"][0] == "Dec 2025"
    assert spec["datasets"][0]["data"][0] == 137.53


def test_breakdown_pie():
    rows = [
        {"Investment Type": "Mutual Funds", "Count": "30"},
        {"Investment Type": "Bonds", "Count": "10"},
        {"Investment Type": "AIF", "Count": "8"},
    ]
    spec = build_chart_spec("What is the split of investment types?", rows)
    assert spec and spec["type"] == "pie"
    assert spec["labels"] == ["Mutual Funds", "Bonds", "AIF"]
    assert spec["datasets"][0]["data"] == [30.0, 10.0, 8.0]


def test_ranking_bar():
    rows = [
        {"AMC": "HDFC", "AUM": "5000000"},
        {"AMC": "ICICI", "AUM": "3000000"},
        {"AMC": "Axis", "AUM": "1000000"},
    ]
    spec = build_chart_spec("Top AMCs by AUM, which has the highest", rows)
    assert spec and spec["type"] == "bar"
    # sorted descending by value
    assert spec["labels"][0] == "HDFC"


def test_no_chart_for_single_value():
    assert build_chart_spec("total aum", [{"Total AUM": "13455063632.72"}]) is None


def test_no_chart_for_plain_list_not_requested():
    # A holdings list the user did not ask to visualise -> no chart.
    rows = [
        {"Instrument": "Fund A", "Current Value": "100"},
        {"Instrument": "Fund B", "Current Value": "200"},
        {"Instrument": "Fund C", "Current Value": "300"},
    ]
    assert build_chart_spec("show all holdings for Ram", rows) is None


def test_no_chart_for_freetext():
    assert build_chart_spec("what is his PAN number", [{"PAN": "ABCDE1234F"}]) is None
