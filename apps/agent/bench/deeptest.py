"""Deep test harness for the agent agent.

Runs every question from the question bank 3x against the live /chat endpoint,
each run in a FRESH session (unique user_id => no memory carryover), then
classifies platform-side vs agent-side errors and detects inconsistency.

Outputs: deeptest_results.json (incremental).
"""
import json
import time
import asyncio
import argparse
import re
from pathlib import Path

import httpx

BENCH = Path(__file__).parent
QUESTIONS = BENCH / "question_bank.json"
RESULTS = BENCH / "deeptest_results.json"
BASE_URL = "http://localhost:8000"
RUNS = 3
CONCURRENCY = 1  # sequential — mirrors real single-user UI load; avoids overloading MCP

_TIMEOUT_PAT = re.compile(r"tim(ed )?out|gateway|504|read ?timeout|did not respond", re.I)
_PLATFORM_ERR_PAT = re.compile(
    r"context resolution failed|jsondecode|internal server error|http 5\d\d|"
    r"traceback|not allowed|invalid or revoked|unauthorized|500", re.I)


def _parsed_data_text(call: dict) -> str:
    t = call.get("_parsed_text", "")
    if not t and isinstance(call.get("response"), dict):
        for it in call["response"].get("content", []):
            if isinstance(it, dict) and it.get("type") == "text":
                t += it.get("text", "")
    return t or ""


def _is_empty_data(text: str) -> bool:
    s = (text or "").strip()
    if not s or s in ("[]", "{}", "null"):
        return True
    # All-null JSON array?
    try:
        v = json.loads(s)
        if isinstance(v, list):
            if len(v) == 0:
                return True
            return all(
                isinstance(r, dict) and all(x in (None, "", "null") for x in r.values())
                for r in v
            )
    except (json.JSONDecodeError, TypeError):
        pass
    return False


def classify_run(resp_text: str, calls: list) -> tuple[str, str]:
    """Return (status, cause). status in: ok, platform_timeout, platform_error,
    platform_no_data, agent_clarification, agent_no_tool, agent_error."""
    resp = resp_text or ""
    # Examine MCP calls
    failed = [c for c in calls if not c.get("success")]
    ok_calls = [c for c in calls if c.get("success")]

    if failed:
        err = str(failed[0].get("error", "")) or resp
        if _TIMEOUT_PAT.search(err):
            return "platform_timeout", err[:300]
        if _PLATFORM_ERR_PAT.search(err):
            return "platform_error", err[:300]
        return "platform_error", err[:300]

    if ok_calls:
        # Got a successful tool call — check if data came back
        if all(_is_empty_data(_parsed_data_text(c)) for c in ok_calls):
            return "platform_no_data", "MCP returned no/empty/all-null data"
        # Data present — did the agent still claim failure?
        if _TIMEOUT_PAT.search(resp) or re.search(r"could not be (retrieved|completed|processed)|no data", resp, re.I):
            return "agent_error", "Agent reported failure despite data present: " + resp[:200]
        return "ok", ""

    # No MCP call was made at all
    if re.search(r"no relevant tool", resp, re.I):
        return "agent_no_tool", resp[:200]
    # Heuristic: a short response that is a question => clarification
    if resp.strip().endswith("?") and len(resp) < 320:
        return "agent_clarification", resp[:200]
    if _TIMEOUT_PAT.search(resp):
        return "platform_timeout", resp[:200]
    # Otherwise the agent answered from general knowledge / memory without a tool
    return "agent_no_tool", "No MCP tool call made; answered without data: " + resp[:160]


async def call_once(client, question, qid, run):
    uid = f"dt_{qid}_{run}_{int(time.time()*1000)%100000}"
    payload = {"message": question, "user_id": uid}
    t0 = time.perf_counter()
    try:
        r = await client.post(f"{BASE_URL}/chat", json=payload, timeout=220.0)
        ms = round((time.perf_counter() - t0) * 1000)
        if r.status_code != 200:
            return {"run": run, "response": f"HTTP {r.status_code}: {r.text[:200]}",
                    "calls": [], "status": "agent_error",
                    "cause": f"agent endpoint HTTP {r.status_code}", "ms": ms}
        d = r.json()
        resp = d.get("response", "")
        calls = d.get("mcp_tool_calls", [])
        status, cause = classify_run(resp, calls)
        tool = calls[0].get("tool") if calls else None
        mcp_ms = calls[0].get("latency_ms") if calls else None
        return {"run": run, "response": resp, "status": status, "cause": cause,
                "tool": tool, "mcp_ms": mcp_ms, "ms": ms,
                "mcp_success": bool(calls and calls[0].get("success"))}
    except Exception as e:
        ms = round((time.perf_counter() - t0) * 1000)
        return {"run": run, "response": f"EXC: {e}", "calls": [],
                "status": "agent_error", "cause": f"client exception: {e}", "ms": ms}


async def run_question(client, sem, q, idx, total, results, done):
    # Question-level concurrency: this question's 3 runs execute together (sequentially),
    # so each question completes and is saved incrementally.
    async with sem:
        runs = []
        for run in range(1, RUNS + 1):
            res = await call_once(client, q["question"], q["srno"], run)
            runs.append(res)
    statuses = [r["status"] for r in runs]
    done[0] += 1
    print(f"[{done[0]}/{total}] srno={q['srno']:>3} {q['question'][:52]!r:54} -> {statuses}", flush=True)
    results.append({"srno": q["srno"], "question": q["question"], "category": q["category"],
                    "persona": q.get("persona", ""), "runs": runs})
    results.sort(key=lambda x: x["srno"])
    json.dump(results, open(RESULTS, "w"), indent=2, ensure_ascii=False)


async def main(only=None):
    questions = json.load(open(QUESTIONS))
    # Renumber sequentially to avoid duplicate srno collisions
    for i, q in enumerate(questions, 1):
        q["srno"] = i
    if only:
        only_set = set(only)
        questions = [q for q in questions if q["srno"] in only_set]
    total = len(questions)
    sem = asyncio.Semaphore(CONCURRENCY)
    results, done = [], [0]
    async with httpx.AsyncClient() as client:
        await asyncio.gather(*[
            run_question(client, sem, q, i, total, results, done)
            for i, q in enumerate(questions, 1)
        ])
    print(f"\nDONE. {len(results)} questions saved to {RESULTS}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", type=str, default="", help="comma-separated srno list")
    args = ap.parse_args()
    only = [int(x) for x in args.only.split(",") if x.strip()] if args.only else None
    asyncio.run(main(only))
