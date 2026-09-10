"""Skill compiler (M3, Option B) — turn a step-by-step methodology skill into a plan.

A reasoning_methodology skill with ``plan_steps`` is compiled into the same decomposition plan
shape the Execution Engine already runs (Phase 3): its steps become parallel sub-questions with
the target client substituted in. This is deterministic — no planner LLM — and reuses execution,
verification and generate unchanged.

Approach A: platform skills ship only a markdown methodology (no ``plan_steps``). When that's the
case we derive the steps from the methodology once (LLM, cached) so the same multi-query fan-out
still happens — see ``skills/methodology_compiler.py``.
"""

import re

from ...config import config
from ...skills.methodology_compiler import derive_plan_steps

# Book-wide phrasings: the question is about the advisor's clients as a GROUP, not one person.
_BOOK_RE = re.compile(
    r"\b(my clients|my book|my portfolios|all (my |the )?clients|which of my clients|"
    r"which clients|who (among|in|of) [\w\s']*clients|each (of my )?clients|every client|"
    r"across (my|the) (clients|book|portfolios))\b",
    re.IGNORECASE,
)


def _resolve_client(state: dict) -> str | None:
    """The client the skill is about: the Client-Insights selected client first, then the
    extracted client name, then the session investor."""
    sel = (state.get("selected_client") or "").strip()
    if sel:
        return sel
    ents = state.get("enrichment_entities") or {}
    name = (ents.get("investor_name") or state.get("investor_name") or "").strip()
    return name or None


def _resolve_scope(state: dict, skill) -> str | None:
    """What the {client} placeholder should be filled with this turn, or None if the skill has no
    valid target:
      • "each client"          → book-wide scope, for a book_capable skill on a book-wide query,
      • a specific client name → single-client scope (precise fetch),
      • None                   → no target (a client/book skill is then skipped).

    Book-wide phrasing WINS for a book_capable skill: "all the clients" / "my clients" is about the
    whole book, and any single client that enrichment resolved for such a query (e.g. a fallback id
    like ``user1``) is a phantom — we must not fetch one client's data for a book-wide question.
    """
    if getattr(skill, "book_capable", False):
        q = f"{state.get('query') or ''} {state.get('enriched_query') or ''}"
        if _BOOK_RE.search(q):
            return "each client"
    client = _resolve_client(state)
    if client:
        return client
    return None


async def compile_skill_plan(skills: list, state: dict) -> dict | None:
    """Compile the first compilable methodology skill into a decomposition plan, or None (caller
    falls back to the normal planner). Returns None if the skill needs a client and none is known
    — a client-scoped brief can't run without a client."""
    for s in skills or []:
        if getattr(s, "skill_type", None) != "reasoning_methodology":
            continue
        plan_steps = getattr(s, "plan_steps", None)
        # Approach A: no structured steps, but a markdown methodology → derive the steps from it
        # (cached, so at most one LLM call per skill version). Also lets the methodology declare
        # whether a client is required.
        if not plan_steps and config.skill_methodology_compile and getattr(s, "content", ""):
            derived = await derive_plan_steps(s)
            plan_steps = derived["steps"]
            if derived.get("requires_client"):
                s.requires_client = True
            if derived.get("book_capable"):
                s.book_capable = True
        if not plan_steps:
            continue
        # Scope-aware target: a named client → the name (precise single-client fetch); a book-wide
        # query on a book_capable skill → "each client" (the SAME {client} questions then fan out
        # per client across the book).
        target = _resolve_scope(state, s)
        if target is None and (s.requires_client or getattr(s, "book_capable", False)):
            # A client/book skill with no resolvable target (no named client, not a book query)
            # can't run — fall through to the normal planner.
            continue
        steps = []
        for st in plan_steps:
            q = st.get("question", "")
            if target:
                q = q.replace("{client}", target)
            q = re.sub(r"\{[^}]*\}", "", q).strip()  # drop any leftover placeholder
            if q:
                steps.append({"question": q, "depends_on": [], "fills": None, "for_each": False})
        if len(steps) >= 2:
            return {"strategy": "decompose", "reason": f"skill:{s.skill_id}",
                    "steps": steps, "synthesis": s.synthesis, "skill_id": s.skill_id}
    return None
