"""Flow inconsistency test — asks each question N times against the live agent.

Goal:
  * Measure response INCONSISTENCY (do repeated asks of the same question agree?)
  * FLAG questions for which the platform returns no usable data on ANY run.

Constraints:
  * STRICTLY SEQUENTIAL (concurrency = 1). Parallel requests overload the MCP
    gateway and produce false 504s.
  * Fresh user_id per request so conversation/session memory never reuses a prior
    answer — every ask is an independent platform fetch.

Output: flow_test_results.json (incremental, safe to inspect mid-run).
"""

import json
import re
import time
import uuid
from pathlib import Path

import requests

AGENT_URL = "http://localhost:8000/chat"
RUNS_PER_Q = 3
TIMEOUT = 200  # seconds per request (heavy queries retry on the platform side)
OUT = Path(__file__).with_name("flow_test_results.json")

QUESTIONS = [
    "What is the total AUM across all my client portfolios?",
    "Show quarterly AUM trend for all UHNI clients.",
    "How many of my clients have direct or indirect exposure to Information Technology",
    "What is the split of investment types across all client portfolios — Equities, Fixed Income, Alternatives?",
    "List all sectoral and thematic funds held across my client portfolios.",
    "How many clients have invested in AIF? What is the aggregate AIF exposure?",
    "Which AMC has the highest AUM concentration across all my clients?",
    "Which clients have more than 30% concentration in a single asset class?",
    "Show me everything Rama Krishna holds — full portfolio snapshot with current values.",
    "Show all equity holdings for Rama Krishna.",
    "List all direct equity holdings in Banking & Finance sector for Rama Krishna's portfolio.",
    "List all large cap mutual funds in Ram's portfolio.",
    "What is the unrealized gain on Ram's mutual fund holdings?",
    "How does Rama Krishna's portfolio XIRR compare to Nifty 50 over 1 year?",
    "Compare Rama Krishna's equity vs. fixed income allocation percentages.",
    "Is Rama Krishna's MF portfolio outperforming Nifty 500 over 1 year?",
    "Which client family has the highest combined AUM?",
    "How many families have total AUM above ₹10 crores?",
    "What ETFs are Ram and his family holding across all accounts?",
    "Show net inflows by product type for Krishna's family members.",
    "Show realized gains by AMC for client Sita Krishnan.",
    "Which of Arjun Mehra's bonds are maturing within 1 year?",
    "Who is my top client by total portfolio value?",
    "Top 5 clients by total new investments in the last 3 years.",
    "How much did all my clients invest in April 2026?",
    "What is the total number of transactions Rama Krishna made this year?",
    "Show the 1-year return of key funds in my clients' portfolios — Axis Bluechip and HDFC Mid-Cap Opportunities.",
    "Compare expense ratios of Axis Focused 25 vs. Mirae Asset Large Cap.",
    "Show me clients whose portfolio XIRR has underperformed their assigned benchmark over the last 1 year.",
]

# Phrases that signal the platform gave the agent NOTHING usable.
_FAIL_PAT = re.compile(
    r"could not (be )?retriev|couldn't (find|retriev)|unable to (find|retriev|provide)|"
    r"no (relevant|data|information|records?) (found|available|was returned)|"
    r"context resolution failed|please (try again|rephrase)|i (don't|do not) have (access|the data)|"
    r"there (is|are) no data|no results|encountered an error|error fetching data",
    re.IGNORECASE,
)


def _classify(resp_text: str, lat: dict) -> str:
    """data | empty | no_mcp — coarse per-run status."""
    mcp_hit = "mcp_exec_ms" in (lat or {})
    if not resp_text or not resp_text.strip():
        return "empty"
    if _FAIL_PAT.search(resp_text):
        return "empty"
    if not mcp_hit:
        return "no_mcp"
    return "data"


def ask(question: str) -> dict:
    uid = f"flow_{uuid.uuid4().hex[:12]}"
    t0 = time.time()
    try:
        r = requests.post(
            AGENT_URL,
            json={"message": question, "user_id": uid},
            timeout=TIMEOUT,
        )
        wall_ms = round((time.time() - t0) * 1000)
        if r.status_code != 200:
            return {"ok": False, "status": r.status_code, "response": f"HTTP {r.status_code}",
                    "latency": {}, "wall_ms": wall_ms, "class": "empty", "mcp_hit": False}
        d = r.json()
        resp = d.get("response", "") or ""
        lat = d.get("latency_breakdown", {}) or {}
        cls = _classify(resp, lat)
        return {
            "ok": True,
            "status": 200,
            "response": resp,
            "latency": lat,
            "wall_ms": wall_ms,
            "class": cls,
            "mcp_hit": "mcp_exec_ms" in lat,
            "mcp_exec_ms": lat.get("mcp_exec_ms"),
            "total_ms": lat.get("total_ms"),
        }
    except Exception as e:  # noqa: BLE001
        wall_ms = round((time.time() - t0) * 1000)
        return {"ok": False, "status": 0, "response": f"EXC: {e}", "latency": {},
                "wall_ms": wall_ms, "class": "empty", "mcp_hit": False}


def main():
    results = []
    total = len(QUESTIONS) * RUNS_PER_Q
    n = 0
    for qi, q in enumerate(QUESTIONS, 1):
        runs = []
        for run in range(1, RUNS_PER_Q + 1):
            n += 1
            print(f"[{n}/{total}] Q{qi} run{run}: {q[:60]}", flush=True)
            res = ask(q)
            res["run"] = run
            runs.append(res)
            print(f"    -> {res['class']} | mcp={res['mcp_hit']} | "
                  f"{res['wall_ms']}ms | {len(res['response'])} chars", flush=True)
            # incremental save after every request
            entry = {"srno": qi, "question": q, "runs": runs}
            tmp = [r for r in results if r["srno"] != qi] + [entry]
            tmp.sort(key=lambda x: x["srno"])
            OUT.write_text(json.dumps(tmp, indent=2, ensure_ascii=False))
        results = [r for r in results if r["srno"] != qi] + [{"srno": qi, "question": q, "runs": runs}]
        results.sort(key=lambda x: x["srno"])
        OUT.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"DONE. {total} requests. Results -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
