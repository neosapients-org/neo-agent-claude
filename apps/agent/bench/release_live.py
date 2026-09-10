"""LIVE deep stress-test — hits the running agent over HTTP (full flow: guardrails +
session memory + retries), captures the response, the platform payload, and latency, then
scores each answer against the raw-DB truth with the two-layer judge (agent-faithful vs
payload-correct).

Each question runs in its OWN fresh session_id, so every ask is an independent platform
fetch (and exercises the new session-scoped memory / Qdrant reload path).

Run INSIDE the agent container (so localhost:8000 is the live server, and OpenAI/app deps
are available):
    docker cp release_ground_truth.json <agent>:/tmp/gt.json
    docker cp release_eval.py            <agent>:/tmp/release_eval.py   # for the judge
    docker cp release_live.py            <agent>:/tmp/release_live.py
    docker exec -w /workspace/agent -e PYTHONPATH=/workspace/agent:/tmp \
      -e GT=/tmp/gt.json -e OUT=/tmp/res_live.json <agent> python /tmp/release_live.py
"""
import asyncio
import json
import os
import time
import uuid

import httpx

from release_eval import judge  # reuse the two-layer judge

AGENT = os.getenv("AGENT_URL", "http://localhost:8000/chat")
GT = os.getenv("GT", "/tmp/gt.json")
OUT = os.getenv("OUT", "/tmp/res_live.json")
ONLY = {int(x) for x in os.getenv("ONLY", "").split(",") if x.strip().isdigit()}


async def ask(session_id: str, message: str):
    async with httpx.AsyncClient(timeout=240) as c:
        r = await c.post(AGENT, json={"message": message, "user_id": "user_1", "session_id": session_id})
        d = r.json()
    resp = d.get("response", "") or ""
    payload, enriched, queries_sent = "", "", []
    for tc in d.get("mcp_tool_calls", []) or []:
        if tc.get("_parsed_text") and not payload:
            payload = tc["_parsed_text"]
        if tc.get("enriched_query") and not enriched:
            enriched = tc["enriched_query"]
        if tc.get("queries_sent") and not queries_sent:
            queries_sent = tc["queries_sent"]
    return resp, payload, d.get("latency_breakdown", {}) or {}, d.get("guardrail_result", {}), enriched, queries_sent


def _n(x):
    return x if isinstance(x, (int, float)) else 0


async def main():
    gt = json.load(open(GT))
    if ONLY:
        gt = [x for x in gt if x["id"] in ONLY]
    print(f"LIVE deep test: {len(gt)} questions -> {AGENT}")
    results = []
    for i, item in enumerate(gt, 1):
        sid = f"deeptest_{item['id']}_{uuid.uuid4().hex[:6]}"  # fresh session per question
        t0 = time.time()
        try:
            resp, payload, lat, guard, enriched, queries_sent = await ask(sid, item["question"])
        except Exception as e:
            resp, payload, lat, guard, enriched, queries_sent = f"ERROR: {type(e).__name__}: {e}", "", {}, {}, "", []
        wall = round((time.time() - t0) * 1000)
        try:
            sc = await judge(item["question"], item["raw"], resp, payload)
        except Exception as e:
            sc = {"_judge_error": str(e)}
        fa, hl, ce, cm = _n(sc.get("factual_accuracy")), _n(sc.get("hallucination")), _n(sc.get("correct_entity", 10)), _n(sc.get("completeness"))
        af, pc = _n(sc.get("agent_faithful")), _n(sc.get("payload_correct"))
        passed = fa >= 9 and hl >= 9 and ce >= 9 and cm >= 7
        agent_ok = af >= 8 and hl >= 8
        platform_ok = pc >= 8
        results.append({
            "id": item["id"], "question": item["question"], "raw": item["raw"],
            "answer": resp, "payload": payload[:4000], "score": sc,
            "passed": passed, "agent_ok": agent_ok, "platform_ok": platform_ok,
            "wall_ms": wall, "latency": lat, "guardrail": guard,
            "enriched_query": enriched, "queries_sent": queries_sent,
            "noted_error": item.get("noted_error", ""),
        })
        print(f"[{i:2}/{len(gt)}] Q{item['id']:<2} e2e={'PASS' if passed else 'fail':<4} "
              f"agent={'ok' if agent_ok else 'BUG':<3} platform={'ok' if platform_ok else 'BAD':<3} "
              f"fa={fa} hl={hl} af={af} pc={pc} | {wall}ms | {item['question'][:38]}", flush=True)
        json.dump(results, open(OUT, "w"), indent=2, ensure_ascii=False)
        await asyncio.sleep(0.4)

    n = len(results)
    npass = sum(1 for r in results if r["passed"])
    nagent = sum(1 for r in results if r["agent_ok"])
    nplat = sum(1 for r in results if r["platform_ok"])
    good = [r for r in results if r["platform_ok"]]
    nag = sum(1 for r in good if r["agent_ok"])
    lat = sorted(r["wall_ms"] for r in results if r.get("wall_ms"))
    print("\n" + "=" * 64)
    print(f"END-TO-END (vs raw DB):                 {npass}/{n} = {100*npass/n:.1f}%")
    print(f"AGENT-FAITHFUL (vs its payload):        {nagent}/{n} = {100*nagent/n:.1f}%")
    print(f"PLATFORM-OK (payload had right data):   {nplat}/{n} = {100*nplat/n:.1f}%")
    if good:
        print(f"AGENT ACCURACY given good platform data: {nag}/{len(good)} = {100*nag/len(good):.1f}%")
    if lat:
        print(f"LATENCY: median={lat[len(lat)//2]}ms p90={lat[int(len(lat)*0.9)]}ms max={lat[-1]}ms")
    print(f"results -> {OUT}")


if __name__ == "__main__":
    asyncio.run(main())
