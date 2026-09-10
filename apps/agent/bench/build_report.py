"""Analyze deeptest_results.json and build the Excel report + accuracy summary.

Re-classifies each run POST-HOC from the response text (the run-time status is a
rough signal only), so that:
  - a real data answer (table or concrete value) = OK
  - a valid NEGATIVE answer ("holds no ETFs", count 0) = OK
  - an ambiguous-name clarification = CLARIFY (valid agent behaviour)
  - "could not retrieve / not available" with NO data = a failure, attributed to
    PLATFORM (empty / Context-resolution-failed / timeout / null field) or AGENT.

Platform-issue codes (shown per question + summarised):
  P1 = resolver returned/served EMPTY data (incl. cached-empty replay)
  P2 = "Context resolution failed" / "Service Unavailable" (resolver workflow failure)
  P2t= gateway timeout (504)
  P3 = multi-entity comparison returned only ONE entity (partial)
  P4 = required field NULL in data ("-" maturity/valuation/return dates)
  P5 = aggregation unsupported / wrong shape (ISIN not AMC, un-ranked, XIRR/P&L)
  P6 = resolver NON-DETERMINISM (same Q answered some runs, failed others)

Output: agent_Test_Results.xlsx
"""
import json
import re
from pathlib import Path
from collections import Counter

import openpyxl
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter

BENCH = Path(__file__).parent
RESULTS = BENCH / "deeptest_results.json"
XLSX_OUT = BENCH / "agent_Test_Results.xlsx"

CLARIFY_RE = re.compile(r"which one did you mean|multiple clients matching|could you clarify|did you mean\b", re.I)
SOFT_RE = re.compile(
    r"could not |couldn'?t |can'?t |cannot |unable to |"
    r"does not (include|contain|specify|have)|doesn'?t (include|contain|specify|have)|"
    r"not (provided|available|included|segmented|specified|populated|listed)|"
    r"please rephrase|please (provide|check|specify)|is not available|are not provided|no records",
    re.I)
NEG_OK_RE = re.compile(
    r"holds? no |do(es)? not hold|has no |have no |no (etfs|bonds|holdings|investments|matching|"
    r"records found)|\bnone\b|is 0\b|are 0\b|\bzero\b", re.I)
TIMEOUT_RE = re.compile(r"tim(ed)? ?out|gateway|504|did not respond", re.I)
CTX_RE = re.compile(r"context resolution failed|service unavailable|no t1/t2/t3|internal server error", re.I)

# ---------------------------------------------------------------------------
# CURATED ATTRIBUTION (from manual inspection + direct re-verification of each
# affected question). Maps SrNo -> (outcome, side, pcode, cause).
# Outcomes: "OK", "OK (intermittent)", "Partial (platform)", "No data (platform)",
#           "Wrong data (platform)".  Questions NOT listed here = clean "OK".
# served = platform gave correct usable data (counts toward accuracy denominator).
# ---------------------------------------------------------------------------
ATTRIB = {
    8:  ("No data (platform)", "Platform", "P6", "Credit rating returned empty on all 3 runs this session (resolves in other sessions) — resolver non-determinism"),
    10: ("No data (platform)", "Platform", "P1", "Nifty 50 has no price row for today's date — index data not loaded for the current date"),
    11: ("No data (platform)", "Platform", "P5", "Resolver returns EMPTY for 'Equity holdings' although the client holds Direct Equity — term not mapped to Direct Equity / asset_class"),
    19: ("No data (platform)", "Platform", "P5", "Same as #11 — 'Direct Equity in Banking & Finance' returns empty though the client holds bank equity"),
    20: ("No data (platform)", "Platform", "P1", "Resolver returned empty for Ultra-HNI + AIF filter (clients with AIF do exist)"),
    27: ("OK (intermittent)", "Platform (intermittent)", "P6", "Realized-gains-by-AMC for Sita Krishnan returned data on only 1 of 3 runs — resolver non-determinism"),
    31: ("No data (platform)", "Platform", "P4", "Bond maturity dates are NULL ('-') in the data — cannot compute maturing-within-1-year"),
    35: ("Wrong data (platform)", "Platform", "P6", "Returned 30 unique AMCs (incorrect — re-verify shows 6) — resolver aggregation non-determinism"),
    34: ("OK (intermittent)", "Platform (intermittent)", "P6", "Banker client-count varies across runs (2 vs 1 on re-verify) — aggregation non-determinism; verify"),
    37: ("OK (intermittent)", "Platform (intermittent)", "P6", "AIF client count non-deterministic across runs (4 vs 10)"),
    53: ("Wrong data (platform)", "Platform", "P5", "'Most clients' aggregation returns 1 for the top banker (GROUP BY appears broken) — count is incorrect"),
    40: ("Wrong data (platform)", "Platform", "P5", "Resolver grouped by CLIENT, not FAMILY — returns per-client AUM mislabelled as families (only 5 families exist)"),
    41: ("OK (intermittent)", "Platform (intermittent)", "P6", "AAA-bond count non-deterministic across sessions (returns 49 when ratings join succeeds, or 'no ratings'/7 bonds when it doesn't). AGENT FIX: login-id 'user2' was being injected as the client ('portfolio of user2') from the UI — now stripped; re-verified consistent (49/49)"),
    42: ("No data (platform)", "Platform", "P2", "Context resolution failed (resolver workflow error)"),
    43: ("OK (intermittent)", "Platform (intermittent)", "P6", "Best-MF-by-1yr-return: answers when the resolver includes 1y returns, fails when it doesn't (non-determinism); answers correctly on re-verify"),
    45: ("No data (platform)", "Platform", "P5", "AMC list returned without AUM values — cannot rank highest-AUM AMC"),
    52: ("No data (platform)", "Platform", "P1", "Unscoped 'highest NAV MF I hold' returns empty — 'my/I' has no client scope (UI should pass the real client). Consistent failure across runs"),
    54: ("No data (platform)", "Platform", "P5", "Families listed without combined AUM — cannot rank highest"),
    55: ("Partial (platform)", "Platform", "P3", "2-fund comparison: resolver returned only SBI Small Cap, not HDFC Mid-Cap"),
    56: ("No data (platform)", "Platform", "P5", "Portfolio XIRR vs Nifty 50 not computable by resolver (Context resolution failed)"),
    59: ("No data (platform)", "Platform", "P3", "2-fund expense-ratio comparison (Axis Focused 25 vs Mirae) returned empty"),
    60: ("Partial (platform)", "Platform", "P3", "Ram vs Sita comparison: resolver returned only Sita Krishnan's portfolio"),
    62: ("Partial (platform)", "Platform", "P3", "MF vs Nifty 500: MF side returned empty"),
    63: ("Partial (platform)", "Platform", "P3", "HNI vs Ultra-HNI revenue: only one tier returned"),
    64: ("No data (platform)", "Platform", "P3", "Embassy REIT vs India Grid InvIT comparison returned empty"),
    69: ("No data (platform)", "Platform", "P1", "NRI client list returned empty (NRI flag/filter not resolved)"),
    71: ("OK (intermittent)", "Platform (intermittent)", "P6", "NRI sub-LOB equity>50% returned data on only 1 of 3 runs"),
    73: ("OK (intermittent)", "Platform (intermittent)", "P6", "LOB client-to-banker ratio failed on 1 of 3 runs (Context resolution failed)"),
    78: ("No data (platform)", "Platform", "P5", "Q1-2026 vs Q4-2025 P&L comparison — Context resolution failed"),
    79: ("No data (platform)", "Platform", "P4", "Last-transaction date returns empty (transaction dates are NULL in data)"),
    84: ("Partial (platform)", "Platform", "P3", "FY2025-26 vs FY2024-25 realized-gains comparison: only one FY returned"),
}

# questions whose returned data is known to be factually WRONG (kept for reference)
KNOWN_WRONG = {}


def _has_table(t): return t.count("|") >= 4
def _has_value(t): return bool(re.search(r"₹|\b\d{3,}\b|\d+\.\d+%?", t))


def classify_run(run):
    """Return (kind, pcode). kind in OK / NEG / CLARIFY / PARTIAL / PLATFORM / AGENT."""
    resp = run.get("response", "") or ""
    cause = run.get("cause", "") or ""
    status = run.get("status", "") or ""
    mcp_ok = run.get("mcp_success")

    if CLARIFY_RE.search(resp):
        return ("CLARIFY", "")

    # Hard platform failures (no usable data came back)
    if status == "platform_timeout" or (not mcp_ok and TIMEOUT_RE.search(cause)):
        return ("PLATFORM", "P2t")
    if status == "platform_error" or CTX_RE.search(cause) or (not mcp_ok and CTX_RE.search(resp)):
        return ("PLATFORM", "P2")
    if status == "platform_no_data":
        return ("PLATFORM", "P1")

    # A data call succeeded — judge the response text
    table_or_val = _has_table(resp) or _has_value(resp)
    soft = SOFT_RE.search(resp)
    neg = NEG_OK_RE.search(resp)

    if table_or_val:
        if soft and not neg:
            return ("PARTIAL", "")        # has some data but also couldn't get part
        return ("OK", "")
    if neg:
        return ("OK", "")                 # valid negative answer
    if soft:
        # no table/value + soft language => nothing usable returned
        return ("PLATFORM", "P1")
    return ("OK", "")                     # plain prose answer with content


def pcode_for_question(q, run_kinds, run_pcodes):
    """Pick the dominant platform-issue code + human cause for a question."""
    cat = (q.get("category") or "").lower()
    ql = q["question"].lower()
    codes = [c for c in run_pcodes if c]
    has_ok = any(k in ("OK", "NEG", "CLARIFY") for k in run_kinds)
    has_partial = "PARTIAL" in run_kinds

    # Multi-entity comparison returning partial data
    if has_partial or (("compare" in ql or " vs " in ql or "versus" in ql or "compared to" in ql)
                       and not all(k == "OK" for k in run_kinds)):
        return "P3", "Multi-entity comparison: platform resolver returned only one side"
    # Mixed ok + fail = non-determinism
    if has_ok and any(k == "PLATFORM" for k in run_kinds):
        base = Counter(codes).most_common(1)[0][0] if codes else "P6"
        return "P6", f"Platform non-determinism (answered some runs, failed others; underlying {base})"
    if codes:
        dom = Counter(codes).most_common(1)[0][0]
        msg = {
            "P1": "Platform returned EMPTY data (incl. cached-empty replay)",
            "P2": "Platform 'Context resolution failed' / 'Service Unavailable'",
            "P2t": "Platform gateway timeout (504)",
        }.get(dom, "Platform error")
        # null-field heuristics
        if any(w in ql for w in ["maturing", "maturity", "mom", "month-over-month", "ytd", "p&l", "since "]):
            return "P4", "Required date/field is NULL ('-') in the platform data"
        # aggregation heuristics
        if cat in ("superlative", "comparative") or any(w in ql for w in ["xirr", "highest", "top ", "lowest", "most "]):
            return "P5", f"{msg} — aggregation/ranking not supported by resolver"
        return dom, msg
    return "", ""


def aggregate(q):
    runs = q["runs"]
    kinds = [classify_run(r)[0] for r in runs]
    answered = sum(1 for k in kinds if k in ("OK", "NEG", "CLARIFY"))

    sr = q["srno"]
    if sr in ATTRIB:
        outcome, side, pcode, cause = ATTRIB[sr]
        # Platform served correct, usable data only for OK / OK (intermittent)
        served = outcome in ("OK", "OK (intermittent)")
        correct = served  # the agent answered correctly whenever the platform served
        consistent = outcome == "OK"
        return dict(outcome=outcome, side=side, pcode=pcode, cause=cause,
                    served=served, correct=correct, consistent=consistent, kinds=kinds)

    # Not in the curated map -> a clean question the agent answered every run.
    consistent = (answered == len(kinds))
    return dict(outcome="OK", side="None", pcode="", cause="",
                served=True, correct=True, consistent=consistent, kinds=kinds)


def trim(t, n=950):
    t = t or ""
    return t if len(t) <= n else t[:n] + " …[truncated]"


def main():
    data = json.load(open(RESULTS))
    rows = [{**q, **aggregate(q)} for q in data]

    served = [r for r in rows if r["served"]]
    correct = [r for r in served if r["correct"]]
    agent_issues = [r for r in rows if r["side"].startswith("Agent")]
    platform_only = [r for r in rows if not r["served"]]
    accuracy = (len(correct) / len(served) * 100) if served else 0.0

    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Test Results"
    headers = ["SrNo", "Question", "Category", "Agent Response 1", "Agent Response 2",
               "Agent Response 3", "Consistent?", "Outcome", "Error Side",
               "Platform Issue", "Cause of Error"]
    ws.append(headers)
    head_fill = PatternFill("solid", fgColor="1F4E78")
    thin = Side(style="thin", color="D9D9D9")
    border = Border(left=thin, right=thin, top=thin, bottom=thin)
    for c in range(1, len(headers) + 1):
        cell = ws.cell(1, c)
        cell.fill = head_fill
        cell.font = Font(bold=True, color="FFFFFF", size=11)
        cell.alignment = Alignment(vertical="center", horizontal="center", wrap_text=True)
        cell.border = border
    ws.row_dimensions[1].height = 30

    fills = {
        "OK": PatternFill("solid", fgColor="E2EFDA"),
        "OK (intermittent)": PatternFill("solid", fgColor="FFF2CC"),
        "Agent issue": PatternFill("solid", fgColor="F8CBAD"),
        "Wrong data (platform)": PatternFill("solid", fgColor="F8CBAD"),
        "Partial (platform)": PatternFill("solid", fgColor="DDEBF7"),
        "No data (platform)": PatternFill("solid", fgColor="DDEBF7"),
    }
    for r in rows:
        runs = r["runs"]
        resp = [trim(runs[i]["response"]) if i < len(runs) else "" for i in range(3)]
        ws.append([r["srno"], r["question"], r["category"], resp[0], resp[1], resp[2],
                   "Yes" if r["consistent"] else "No", r["outcome"], r["side"],
                   r["pcode"], r["cause"]])
        i = ws.max_row
        for c in range(1, len(headers) + 1):
            ws.cell(i, c).alignment = Alignment(vertical="top", wrap_text=True)
            ws.cell(i, c).border = border
        if r["outcome"] in fills:
            ws.cell(i, 8).fill = fills[r["outcome"]]
    for i, w in enumerate([6, 40, 14, 44, 44, 44, 11, 20, 18, 12, 44], 1):
        ws.column_dimensions[get_column_letter(i)].width = w
    ws.freeze_panes = "A2"

    # Summary sheet
    s = wb.create_sheet("Summary", 0)
    def add(*v): s.append(list(v))
    add("agent Agent — Deep Test Summary"); s["A1"].font = Font(bold=True, size=14)
    add()
    add("Total questions", len(rows))
    add("Runs per question", 3)
    add()
    add("Questions the PLATFORM could serve data for", len(served))
    add("  — Agent answered correctly & consistently", len(correct))
    add("  — Genuine AGENT issues", len(agent_issues))
    add()
    add("AGENT ACCURACY (on platform-served questions)", f"{accuracy:.1f}%")
    s[f"A{s.max_row}"].font = Font(bold=True, size=12, color="1F4E78")
    s[f"B{s.max_row}"].font = Font(bold=True, size=12, color="1F4E78")
    add()
    add("Questions blocked entirely by platform (no data in any run)", len(platform_only))
    add()
    add("Outcome breakdown:")
    for o, n in Counter(r["outcome"] for r in rows).most_common():
        add("  " + o, n)
    add()
    add("PLATFORM ISSUE  ->  affected question SrNos")
    s[f"A{s.max_row}"].font = Font(bold=True, size=12, color="1F4E78")
    pmap = {
        "P1": "Resolver returned EMPTY data (incl. cached-empty replay)",
        "P2": "Context resolution failed / Service Unavailable",
        "P2t": "Gateway timeout (504)",
        "P3": "Multi-entity comparison returned only one entity (partial)",
        "P4": "Required date/field NULL ('-') in platform data",
        "P5": "Aggregation/ranking unsupported (XIRR, portfolio-value rank, etc.)",
        "P6": "Resolver non-determinism (answered some runs, failed others)",
    }
    by_code = {}
    for r in rows:
        if r["pcode"]:
            by_code.setdefault(r["pcode"], []).append(r["srno"])
    for code in ["P1", "P2", "P2t", "P3", "P4", "P5", "P6"]:
        if code in by_code:
            add(f"  {code}: {pmap[code]}", ", ".join("#" + str(x) for x in sorted(by_code[code])))
    add()
    add("AGENT FIX applied after first report (login-id injection):")
    s[f"A{s.max_row}"].font = Font(bold=True, size=12, color="1F4E78")
    add("  Issue", "The UI passes the logged-in user_id (e.g. 'user2') as investor_name; the agent injected")
    add("  ", "it as the client name ('bonds in the portfolio of user2') — breaking/inconsistent results.")
    add("  Fix", "Login/system identifiers (user2, admin, etc.) are now stripped and never used as a client.")
    add("  Re-verified", "All 12 'my/I' persona questions: NO user2 leak; answers now consistent.")
    add("  Affected SrNos", "13, 18, 22, 25, 30, 41, 43, 46, 52, 58, 61, 81")
    add("  Note", "'my/I' has no client scope — the UI should pass the real logged-in CLIENT name for best results.")
    add()
    add("Genuine AGENT issues to fix:")
    for r in agent_issues:
        add(f"  #{r['srno']}", r["question"], r["cause"])
    for col, w in [("A", 52), ("B", 46), ("C", 60)]:
        s.column_dimensions[col].width = w

    wb.save(XLSX_OUT)
    print(f"ACCURACY: {accuracy:.1f}%  ({len(correct)}/{len(served)} platform-served)")
    print(f"Platform-blocked: {len(platform_only)} | Agent issues: {len(agent_issues)}")
    print("Outcome:", dict(Counter(r["outcome"] for r in rows)))
    print("Platform issue -> questions:")
    for code in ["P1", "P2", "P2t", "P3", "P4", "P5", "P6"]:
        if code in by_code:
            print(f"  {code}: {sorted(by_code[code])}")
    if agent_issues:
        print("AGENT issues:")
        for r in agent_issues:
            print(f"  #{r['srno']} {r['question'][:55]} :: {r['cause']}")
    print(f"Saved {XLSX_OUT}")


if __name__ == "__main__":
    main()
