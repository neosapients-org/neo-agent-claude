"""Answer Verifier — confirm the gathered data covers the question (Phase 4).

Two layers, both bounded:

1. coverage_check + backfill (deterministic, no LLM, ON by default): after a DECOMPOSED
   question runs, check that every sub-question actually returned usable data. Any that came
   back empty get ONE more concurrent retry. This happens at the DATA level — before the answer
   is written — so a gap is filled without ever double-streaming an answer to the RM. Bounded by
   ``config.max_subquestions`` so the worst case stays small.

2. verify_answer (fast-model, OFF by default): a final check that the WRITTEN answer addresses
   each part of the question. Because the answer is streamed live it can only annotate, so it is
   opt-in via ``ANSWER_VERIFY_ENABLED``.

Like every other phase this fails safe: any error or disabled flag leaves today's behaviour.
"""

import asyncio

from langchain_core.messages import SystemMessage, HumanMessage

from app.llm import make_llm
from ...config import config


def _is_empty(text: str) -> bool:
    # Reuse the single emptiness judge from mcp_fetch so "no usable data" means the same thing
    # everywhere (empty list, all-null rows, [count]-only payloads…).
    from .mcp_fetch import _is_empty_or_null_text
    return _is_empty_or_null_text(text or "")


def coverage_check(tool_calls: list[dict]) -> dict:
    """Which sub-questions came back with NO usable data. Deterministic, no LLM.

    Returns {"complete": bool, "missing": [sub_question, ...], "missing_idx": [int, ...]}.
    """
    missing: list[str] = []
    idxs: list[int] = []
    for i, tr in enumerate(tool_calls or []):
        q = tr.get("sub_question") or tr.get("enriched_query") or ""
        ok = bool(tr.get("success")) and not _is_empty(tr.get("_parsed_text", ""))
        if not ok:
            missing.append(q)
            idxs.append(i)
    return {"complete": not missing, "missing": missing, "missing_idx": idxs}


async def backfill(tool_calls: list[dict], state: dict) -> tuple[list[dict], int]:
    """One bounded reflection pass: re-fetch each empty/failed sub-question ONCE (concurrently)
    and replace its result only if the retry yields real data. Returns (updated_calls, n_retried).
    The platform is non-deterministic, so a second independent attempt frequently lands data the
    first pass missed — and because this runs before Generate, the answer is written just once."""
    cov = coverage_check(tool_calls)
    if cov["complete"]:
        return tool_calls, 0

    from .mcp_fetch import fetch_one

    out = list(tool_calls)
    budget = (state.get("mode_config") or {}).get("max_subquestions", config.max_subquestions)
    targets = [(i, out[i].get("sub_question") or out[i].get("enriched_query"))
               for i in cov["missing_idx"]]
    targets = [(i, q) for i, q in targets if q][: budget]
    if not targets:
        return tool_calls, 0

    results = await asyncio.gather(*[fetch_one(q, state) for _, q in targets])
    for (i, q), new in zip(targets, results):
        if new.get("success") and not _is_empty(new.get("_parsed_text", "")):
            new = dict(new)
            new["sub_question"] = q
            out[i] = new
    return out, len(targets)


_VERIFY_PROMPT = """You check whether an ANSWER fully addresses a wealth-management QUESTION. \
Reply on ONE line:
- "COMPLETE" if the answer addresses every part the question asked.
- "MISSING: <short phrase>" naming only the part(s) the answer did not address.
Judge coverage of the ASK only — do not second-guess the figures, and treat a clearly stated \
"no records / none" as a valid, complete answer."""


async def verify_answer(query: str, answer: str) -> dict:
    """Fast-model coverage check of the WRITTEN answer. Returns {"complete": bool, "missing": str}.
    No-op (complete) when disabled or on any error — never blocks a reply."""
    if not config.answer_verify_enabled or not (answer or "").strip():
        return {"complete": True, "missing": ""}
    try:
        llm = make_llm("fast", temperature=0.0, max_tokens=60)
        resp = await llm.ainvoke([
            SystemMessage(content=_VERIFY_PROMPT),
            HumanMessage(content=f"QUESTION:\n{query}\n\nANSWER:\n{answer}\n\nVerdict:"),
        ])
        text = (resp.content or "").strip()
    except Exception:
        return {"complete": True, "missing": ""}

    if text.upper().startswith("COMPLETE") or not text:
        return {"complete": True, "missing": ""}
    missing = text.split(":", 1)[1].strip() if ":" in text else text
    return {"complete": False, "missing": missing}
