"""Execution Engine — runs a query plan (Phase 3).

For a DIRECT plan it does a single robust platform call. For a DECOMPOSE plan it runs the
sub-questions one-by-one / in parallel, respecting dependencies: independent steps run together
(asyncio.gather), and a dependent step's {placeholders} are filled with the resolved values of
the steps it depends on (or fanned out once per item via {item}). Every sub-question goes
through the SAME single-hop primitive (``mcp_fetch.fetch_one``), so retries, smart refinement
and fund recovery apply uniformly.

The result is a LIST of tool_result dicts in ``mcp_tool_calls`` — exactly the shape the Generate
node already iterates — each tagged with the sub-question it answered so the final answer can be
synthesised. The whole question's platform-call count (sub-questions + fan-out) is capped by
``config.max_subquestions``.
"""

import asyncio
import json
import re
import time

try:
    from ns_probe import observe
except ImportError:
    def observe(*a, **kw):
        def _id(fn): return fn
        return _id

from langchain_core.messages import AIMessage

from app.llm import make_llm
from ...config import config
from .parallel import _fetch_labels  # noqa: E402
from .stream_channel import emit_detail as _emit_detail


def _topological_order(steps: list[dict]) -> list[int]:
    """Kahn's algorithm over the depends_on graph. Returns step indexes in a safe run order.
    On a cycle/unsatisfiable graph, appends the leftovers in index order (the wave runner also
    degrades gracefully, so this never blocks)."""
    n = len(steps)
    indeg = [0] * n
    children: list[list[int]] = [[] for _ in range(n)]
    for i, s in enumerate(steps):
        for d in s.get("depends_on", []):
            if 0 <= d < n:
                indeg[i] += 1
                children[d].append(i)
    ready = [i for i in range(n) if indeg[i] == 0]
    order: list[int] = []
    while ready:
        i = ready.pop(0)
        order.append(i)
        for c in children[i]:
            indeg[c] -= 1
            if indeg[c] == 0:
                ready.append(c)
    if len(order) < n:  # cycle — append the rest deterministically
        order += [i for i in range(n) if i not in order]
    return order


def _primary_values(parsed_text: str, limit: int = 20) -> list[str]:
    """Pull the primary entities (names/ids) out of a parsed step payload — used to fill a
    dependent step's placeholder or to fan a for_each step out over items. Takes a name-ish
    column, else the sole column, else bare string rows."""
    try:
        data = json.loads(parsed_text)
    except (json.JSONDecodeError, TypeError):
        return []
    vals: list[str] = []
    if isinstance(data, list):
        for row in data:
            if isinstance(row, dict):
                key = next((k for k in row if re.search(r"name|client|investor|fund|scheme|rm|manager", str(k), re.I)), None)
                if key is None and len(row) == 1:
                    key = next(iter(row))
                if key is None and row:
                    key = next(iter(row))
                v = row.get(key) if key else None
                if v not in (None, "", "null"):
                    vals.append(str(v))
            elif isinstance(row, str) and row.strip():
                vals.append(row.strip())
    elif isinstance(data, dict):
        for v in data.values():
            if v not in (None, "", "null"):
                vals.append(str(v))
    # de-dupe, preserve order
    seen, out = set(), []
    for v in vals:
        if v not in seen:
            seen.add(v)
            out.append(v)
        if len(out) >= limit:
            break
    return out


def _fill(question: str, fill_map: dict, item: str | None = None) -> str:
    """Substitute {item} (fan-out) and {name} (dependency values) into a step's question, then
    strip any leftover placeholder so a stray {x} never reaches the platform."""
    out = question
    if item is not None:
        out = out.replace("{item}", item)
    for name, val in (fill_map or {}).items():
        if name:
            out = out.replace("{" + name + "}", val)
    out = re.sub(r"\{[^}]*\}", "", out)
    return re.sub(r"\s{2,}", " ", out).strip()


def _fill_value(trs: list[dict]) -> str:
    """A short string standing in for a step's answer (joined primary values), used when a later
    step references it as {name}."""
    vals: list[str] = []
    for tr in trs:
        vals.extend(_primary_values(tr.get("_parsed_text", "")))
    # de-dupe
    seen, out = set(), []
    for v in vals:
        if v not in seen:
            seen.add(v)
            out.append(v)
    return ", ".join(out[:10])


_PLACEHOLDER_RE = re.compile(r"\{([^}]+)\}")


def _required_placeholders(question: str) -> set[str]:
    """Placeholder names a question needs filled (excluding {item}, which a for_each supplies)."""
    return {m.strip() for m in _PLACEHOLDER_RE.findall(question or "") if m.strip().lower() != "item"}


def _has_placeholder(question: str) -> bool:
    return bool(_PLACEHOLDER_RE.search(question or ""))


def _fail_tr(question: str, reason: str) -> dict:
    """A failed tool_result for a sub-question we deliberately did NOT run (so generate reports
    the gap honestly instead of us fetching a broken/blank query)."""
    return {"tool": "resolve_context", "success": False, "_parsed_text": "",
            "error": reason, "sub_question": question}


_TRAILING_METRIC_RE = re.compile(r"\s*[,;:\-–—]\s*-?\d[\d.,%]*\s*$")


def _strip_trailing_metric(val: str) -> str:
    """Remove a metric the model may append to an identifier ("Sunita Choudhary, -1.48" ->
    "Sunita Choudhary"). Applied per comma-separated item so a real multi-entity list survives."""
    parts = [p.strip() for p in (val or "").split(",")]
    cleaned = [_TRAILING_METRIC_RE.sub("", p).strip() for p in parts]
    # Drop items that became empty or are pure numbers (stray metric fragments).
    cleaned = [c for c in cleaned if c and not re.fullmatch(r"-?\d[\d.,%]*", c)]
    return ", ".join(cleaned)


async def _resolve_fill(question: str, trs: list[dict], fill_name: str | None = None) -> str:
    """Extract the IDENTIFIER a later step needs from this step's data — a name or id only, never
    a metric/number.

    Column extraction alone is wrong for interpretive steps (e.g. "which client has the WORST
    portfolio" returns a list, not the answer), so we ask the fast model for just the identifier,
    keyed to the step's own question. Falls back to column extraction if the LLM is unavailable.
    """
    data = "\n".join(tr.get("_parsed_text", "") for tr in trs if tr.get("_parsed_text"))
    if not data.strip():
        return ""
    try:
        from langchain_core.messages import SystemMessage, HumanMessage
        from ...config import config
        llm = make_llm("fast", temperature=0.0, max_tokens=60)
        hint = f" The value represents: {fill_name}." if fill_name else ""
        resp = await llm.ainvoke([
            SystemMessage(content=(
                "From the DATA, extract the ONE identifying value that answers the QUESTION, to be "
                "plugged into a later question." + hint + " Return ONLY the identifier (a name or "
                "ID) — NO metric, number, score, percentage, date, or any other column. If several "
                "DISTINCT entities answer it, comma-separate just their identifiers. If the data "
                "has no answer, reply with an empty line.")),
            HumanMessage(content=f"QUESTION: {question}\n\nDATA:\n{data[:4000]}\n\nIDENTIFIER:"),
        ])
        val = (resp.content or "").strip().strip('".').strip()
        val = _strip_trailing_metric(val)  # safety net if the model still appended a metric
        if val and len(val) < 200:
            return val
    except Exception:
        pass
    return _fill_value(trs)


def _wrap(outputs: list[dict], state: dict, start: float, plan: dict, trace: list) -> dict:
    """Shape the engine's result like the mcp_fetch node return (so orchestrate/generate consume
    it unchanged). Bubbles up a fund clarification if any sub-question produced one."""
    exec_ms = round((time.perf_counter() - start) * 1000)
    combined = "\n\n".join(tr.get("_parsed_text", "") for tr in outputs if tr.get("_parsed_text"))
    out = {
        "messages": [AIMessage(content=combined)],  # raw; generate produces the real answer
        "mcp_tool_calls": outputs,
        "query_plan": plan,
        "execution_trace": trace,
        "latency_breakdown": {**state.get("latency_breakdown", {}), "mcp_exec_ms": exec_ms},
    }
    clar = next((tr.get("fund_clarification") for tr in outputs if tr.get("fund_clarification")), None)
    if clar:
        out["fund_clarification"] = clar
        out["messages"] = [AIMessage(content=clar)]

    # Annotate the enclosing "Execution Engine" observe span with what it ran.
    try:
        from ...observe_utils import annotate as _annotate
        _annotate(
            input={
                "query": state.get("enriched_query") or state.get("query", ""),
                "strategy": plan.get("strategy"),
                "steps": [s.get("question") for s in (plan.get("steps") or [])],
            },
            output={"tool_calls": len(outputs), "combined_text": combined},
            ns_step_count=len(outputs),
            ns_strategy=str(plan.get("strategy", "")),
        )
    except Exception:
        pass

    return out


async def execute_plan(plan: dict, state: dict) -> dict:
    """Run a validated plan and return a tool_calls-shaped dict for orchestrate/generate."""
    from .mcp_fetch import fetch_one

    start = time.perf_counter()
    steps = plan.get("steps") or []
    trace: list[dict] = []

    # DIRECT (or a degenerate single-step decompose): one robust hop.
    if plan.get("strategy") != "decompose" or len(steps) <= 1:
        q = steps[0]["question"] if steps else (state.get("enriched_query") or state.get("query", ""))
        _emit_detail("data_fetch", _fetch_labels()["sending"].format(q=q))
        tr = await fetch_one(q, state)
        trace.append({"step": 0, "question": q, "success": tr.get("success")})
        return _wrap([tr], state, start, plan, trace)

    # DECOMPOSE: run in dependency waves. Independent steps in a wave run concurrently.
    step_outputs: dict[int, list[dict]] = {}   # step idx -> its tool_result(s)
    fills: dict[int, str] = {}                  # step idx -> value for {fills-name}
    budget = (state.get("mode_config") or {}).get("max_subquestions", config.max_subquestions)
    spent = 0
    remaining = set(range(len(steps)))
    depended = {d for s in steps for d in s.get("depends_on", [])}  # steps whose value is reused

    while remaining:
        ready = [i for i in remaining if all(d in step_outputs for d in steps[i].get("depends_on", []))]
        if not ready:  # unsatisfiable (cycle) — run the rest as-is
            ready = sorted(remaining)

        coros, meta = [], []        # runnable: (step_idx, question)
        skipped: list[tuple] = []   # deliberately not run: (step_idx, fail_tr)
        for i in ready:
            step = steps[i]
            fill_map = {steps[d].get("fills"): fills.get(d, "")
                        for d in step.get("depends_on", []) if steps[d].get("fills")}
            reqs = _required_placeholders(step["question"])

            if step.get("for_each") and step.get("depends_on"):
                items = [it for it in _primary_values(
                    "\n".join(tr.get("_parsed_text", "") for tr in step_outputs.get(step["depends_on"][0], []))
                ) if it]
                if not items:
                    # Nothing to iterate over — DON'T run a blank query (that's how a wrong
                    # client got hallucinated in). Record the gap instead.
                    skipped.append((i, _fail_tr(step["question"], "could not resolve the list from the previous step")))
                    continue
                for it in items:
                    if spent >= budget:
                        break
                    q = _fill(step["question"], fill_map, item=it)
                    if _has_placeholder(q):  # some other {name} still unfilled
                        continue
                    coros.append(fetch_one(q, state)); meta.append((i, q)); spent += 1
            else:
                # Values this step needs from its dependencies (name -> resolved value).
                dep_fills = [(steps[d].get("fills"), fills.get(d, ""))
                             for d in step.get("depends_on", []) if steps[d].get("fills")]
                # Skip if the dependency that should supply the subject came back empty — a
                # blank/subjectless query ("...holdings for ,") is meaningless and gets reworded
                # into a hallucinated entity. Report the gap honestly instead.
                if dep_fills and all(not str(v).strip() for _, v in dep_fills):
                    skipped.append((i, _fail_tr(step["question"], "could not determine the subject from the previous step")))
                    continue
                missing = [n for n in reqs if not str(fill_map.get(n, "")).strip()]
                if missing:
                    skipped.append((i, _fail_tr(step["question"], f"could not determine {', '.join(missing)} from the previous step")))
                    continue
                if spent >= budget:
                    skipped.append((i, _fail_tr(step["question"], "sub-question budget reached")))
                    continue
                q = _fill(step["question"], fill_map)
                # The planner sometimes omits the {placeholder} and relies on implicit context
                # ("show the full holdings") — which would send a subjectless query. Make sure
                # every resolved dependency value actually appears in the query.
                extras = [v for _, v in dep_fills if str(v).strip() and v.lower() not in q.lower()]
                if extras:
                    q = f"{q.rstrip(' .?!')} for {', '.join(extras)}."
                coros.append(fetch_one(q, state)); meta.append((i, q)); spent += 1

        # Show the wave being dispatched LIVE — one line if a single question, or a grouped
        # "N questions in parallel" block when the wave fans out concurrently.
        if meta:
            _wave_qs = [q for (_, q) in meta]
            if len(_wave_qs) == 1:
                _emit_detail("data_fetch", f"Sending to platform: “{_wave_qs[0]}”")
            else:
                _emit_detail("data_fetch", f"Sending {len(_wave_qs)} questions in parallel:")
                for _wq in _wave_qs:
                    _emit_detail("data_fetch", f"• {_wq}")

        results = await asyncio.gather(*coros) if coros else []
        for (i, q), tr in zip(meta, results):
            tr = dict(tr)
            tr["sub_question"] = q
            step_outputs.setdefault(i, []).append(tr)
            trace.append({"step": i, "question": q, "success": tr.get("success")})
        for i, tr in skipped:
            step_outputs.setdefault(i, []).append(tr)
            trace.append({"step": i, "question": tr["sub_question"], "success": False})

        for i in ready:
            # Resolve the fill value (for steps a later step depends on) by INTERPRETING this
            # step's data via the LLM — handles "which is worst/highest" where a plain column
            # dump is the wrong answer. Cheap fallback to column extraction.
            if i in depended and steps[i].get("fills") and step_outputs.get(i):
                fills[i] = await _resolve_fill(steps[i]["question"], step_outputs.get(i, []), steps[i].get("fills"))
            else:
                fills[i] = ""
            remaining.discard(i)

    # Flatten outputs in original step order (and sub-order within a fanned-out step).
    outputs: list[dict] = []
    for i in range(len(steps)):
        outputs.extend(step_outputs.get(i, []))
    return _wrap(outputs, state, start, plan, trace)


# --- Instrument at the bottom of the file ---
_obs = dict(capture_args=False, capture_result=False)
execute_plan = observe(name="Execution Engine", agent_name="agent", **_obs)(execute_plan)
