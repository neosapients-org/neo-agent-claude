"""Release accuracy harness — runs each ground-truth question through the agent's
DATA PATH directly (intent_enrichment -> mcp_fetch -> generate_response), then scores
the answer against the raw-DB reference with an LLM judge.

Runs INSIDE the agent container (has app deps + .env + OpenAI/MCP reachability):
    docker cp release_ground_truth.json <agent>:/tmp/gt.json
    docker cp release_eval.py          <agent>:/tmp/release_eval.py
    docker exec -w /workspace/agent -e GT=/tmp/gt.json -e OUT=/tmp/res.json <agent> \
        python /tmp/release_eval.py

Why node-level (not the HTTP server): the server is hung on a guardrail model
download in this sandbox. The guardrail is orthogonal to routing/fetch/generate, so
driving the nodes directly is a faithful and faster measurement of what we change.

Env:
  GT    path to ground-truth json (default ./release_ground_truth.json)
  OUT   path to write results json (default ./release_results.json)
  JUDGE_MODEL  override judge model (default = config.openai_model_answer)
  ONLY  comma-separated question ids to run (default all)
"""
import asyncio
import json
import os
import re
import time

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from app.config import config
from app.graph.nodes.intent_enrichment import intent_enrichment
from app.graph.nodes.mcp_fetch import mcp_fetch
from app.graph.nodes.generate import generate_response

GT = os.getenv("GT", os.path.join(os.path.dirname(__file__), "release_ground_truth.json"))
OUT = os.getenv("OUT", os.path.join(os.path.dirname(__file__), "release_results.json"))
JUDGE_MODEL = os.getenv("JUDGE_MODEL", config.openai_model_answer)
ONLY = {int(x) for x in os.getenv("ONLY", "").split(",") if x.strip().isdigit()}

JUDGE_SYS = """You grade a wealth-management agent. You see the QUESTION, the RAW database result
(ground truth), the PLATFORM PAYLOAD the agent actually received from the data tool, and the
agent's ANSWER. The agent can ONLY synthesize from the PAYLOAD — it never sees the RAW DB.

Score 0-10 each:
- factual_accuracy: do numbers/facts in the ANSWER match the RAW data? 10=match (rounding ok). A 10x scaling error or wrong figure is <=3.
- completeness: are all rows/fields the question asked for present in the ANSWER? proportional.
- hallucination: 10=invents nothing; any number/category/row fabricated (not in PAYLOAD) = <=3.
- correct_entity: right client/security/category? 10=yes, 0=wrong entity.
- agent_faithful: does the ANSWER correctly & completely reflect the PAYLOAD it was given (regardless of whether the payload itself is right)? 10=faithful, 0=misread/garbled the payload.
- payload_correct: does the PLATFORM PAYLOAD itself contain data matching the RAW truth? 10=yes, 0=payload wrong/empty/missing it. (Isolates platform errors from agent errors.)
- answered: false only if the agent said it could not retrieve / no data / asked to rephrase WHILE the PAYLOAD actually had usable data.

IMPORTANT: the agent may legitimately add human-readable client NAMES resolved from the platform
(e.g. "Ram Krishnan" for client_id 7001) even when the RAW shows only an id — do NOT treat correct
names as hallucination. Judge correctness on the NUMBERS and the id/entity mapping.

Return ONLY JSON: {"factual_accuracy":N,"completeness":N,"hallucination":N,"correct_entity":N,"agent_faithful":N,"payload_correct":N,"answered":true/false,"reason":"<=20 words"}"""


_judge_llm = ChatOpenAI(
    api_key=config.openai_api_key, base_url=config.openai_base_url,
    model=JUDGE_MODEL, temperature=0.0, max_tokens=300,
)


async def judge(question, raw, answer, payload=""):
    resp = await _judge_llm.ainvoke([
        SystemMessage(content=JUDGE_SYS),
        HumanMessage(content=f"QUESTION:\n{question}\n\nRAW (truth):\n{raw[:5000]}\n\nPLATFORM PAYLOAD (what the agent got):\n{(payload or '')[:5000]}\n\nAGENT ANSWER:\n{answer[:5000]}"),
    ])
    txt = re.sub(r"^```\w*|```$", "", resp.content.strip()).strip()
    try:
        return json.loads(txt)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", txt, re.DOTALL)
        return json.loads(m.group()) if m else {"_unparsed": txt[:200]}


def _state(q):
    return {
        "messages": [HumanMessage(content=q)], "query": q, "user_id": f"eval_{abs(hash(q))%99999}",
        "intent": None, "mcp_tool_calls": [], "guardrail_result": {}, "guardrail_passed": True,
        "latency_breakdown": {}, "investor_name": None, "memory_context": "",
        "enriched_query": "", "enrichment_entities": {}, "chat_history": [],
    }


async def run_one(item):
    q = item["question"]
    st = _state(q)
    t0 = time.perf_counter()
    enr = await intent_enrichment(st)
    st.update(enr)
    intent = st.get("intent")
    enriched = st.get("enriched_query") or ""
    payload = ""
    mcp_ok = None
    if intent == "fetch_data":
        m = await mcp_fetch(st)
        st["mcp_tool_calls"] = m.get("mcp_tool_calls", [])
        st["latency_breakdown"].update(m.get("latency_breakdown", {}))
        for msg in m.get("messages", []):
            st["messages"].append(msg)
        tc = (st["mcp_tool_calls"] or [{}])[0]
        mcp_ok = tc.get("success")
        payload = tc.get("_parsed_text", "") or ""
        g = await generate_response(st)
        answer = g["messages"][-1].content
    elif intent in ("general", "from_memory"):
        g = await generate_response(st)
        answer = g["messages"][-1].content
    else:  # clarification
        answer = st["messages"][-1].content if st["messages"] else ""
    wall_ms = round((time.perf_counter() - t0) * 1000)

    sc = await judge(q, item["raw"], answer, payload)

    def _n(x):
        return x if isinstance(x, (int, float)) else 0
    fa, hl, ce, cm = _n(sc.get("factual_accuracy")), _n(sc.get("hallucination")), _n(sc.get("correct_entity", 10)), _n(sc.get("completeness"))
    af, pc = _n(sc.get("agent_faithful")), _n(sc.get("payload_correct"))
    passed = fa >= 9 and hl >= 9 and ce >= 9 and cm >= 7      # end-to-end vs raw DB
    agent_ok = af >= 8 and hl >= 8                            # agent faithfully synthesized its payload
    platform_ok = pc >= 8                                     # platform payload had the right data
    return {
        "id": item["id"], "question": q, "intent": intent, "enriched": enriched,
        "mcp_ok": mcp_ok, "answer": answer, "score": sc,
        "passed": passed, "agent_ok": agent_ok, "platform_ok": platform_ok,
        "wall_ms": wall_ms, "latency": st["latency_breakdown"],
        "payload_preview": (payload or "")[:400], "noted_error": item.get("noted_error", ""),
    }


async def main():
    gt = json.load(open(GT))
    if ONLY:
        gt = [x for x in gt if x["id"] in ONLY]
    print(f"running {len(gt)} questions | judge={JUDGE_MODEL}")
    results = []
    for i, item in enumerate(gt, 1):
        try:
            res = await asyncio.wait_for(run_one(item), timeout=150)
        except Exception as e:
            res = {"id": item["id"], "question": item["question"], "error": f"{type(e).__name__}: {e}",
                   "passed": False, "score": {}, "noted_error": item.get("noted_error", "")}
        results.append(res)
        sc = res.get("score", {})
        e2e = "PASS" if res.get("passed") else "fail"
        agent = "ok" if res.get("agent_ok") else "BUG"
        plat = "ok" if res.get("platform_ok") else "BAD"
        print(f"[{i:2}/{len(gt)}] Q{res['id']:<2} e2e={e2e:<4} agent={agent:<3} platform={plat:<3} "
              f"fa={sc.get('factual_accuracy','-')} hl={sc.get('hallucination','-')} "
              f"af={sc.get('agent_faithful','-')} pc={sc.get('payload_correct','-')} "
              f"| {res.get('wall_ms','-')}ms | {res['question'][:40]}", flush=True)
        json.dump(results, open(OUT, "w"), indent=2, ensure_ascii=False)
        await asyncio.sleep(0.3)

    n = len(results)
    npass = sum(1 for r in results if r.get("passed"))
    nagent = sum(1 for r in results if r.get("agent_ok"))
    nplat = sum(1 for r in results if r.get("platform_ok"))
    # agent accuracy GIVEN good platform data = faithful among the platform-ok subset
    good = [r for r in results if r.get("platform_ok")]
    nagent_given_good = sum(1 for r in good if r.get("agent_ok"))
    print("\n" + "=" * 64)
    print(f"END-TO-END (answer vs raw DB):        {npass}/{n} = {100*npass/n:.1f}%")
    print(f"AGENT-FAITHFUL (answer vs its payload): {nagent}/{n} = {100*nagent/n:.1f}%")
    print(f"PLATFORM-OK (payload had right data):   {nplat}/{n} = {100*nplat/n:.1f}%")
    if good:
        print(f"AGENT ACCURACY given good platform data: {nagent_given_good}/{len(good)} = {100*nagent_given_good/len(good):.1f}%")
    print(f"e2e FAILED: " + ", ".join(f"Q{r['id']}" for r in results if not r.get("passed")))
    print(f"AGENT BUGS (platform ok but agent wrong): " + ", ".join(f"Q{r['id']}" for r in results if r.get("platform_ok") and not r.get("agent_ok")))
    print(f"results -> {OUT}")


if __name__ == "__main__":
    asyncio.run(main())
