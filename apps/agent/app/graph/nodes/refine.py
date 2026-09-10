"""Smart refinement ladder for empty / failed platform results (Phase 2).

The old approach (``_retry_variant``) re-asked the SAME question up to four times with only
cosmetic changes — a full stop, a "Please provide:" prefix, a "give the exact values" suffix.
None of those change the actual content, so if the platform couldn't answer the question the
first time (too complex, an odd phrasing), re-asking it with a full stop never helped.

This module escalates like a person would:

    attempt 1   -> cheap_variant: a deterministic cache-busting reframe. The platform caches a
                   result for the EXACT query string (~120s) and is non-deterministic, so a
                   surface change frequently lands fresh, usable data. Cheap, no LLM.
    attempt >=2 -> _llm_reword: a genuinely DIFFERENT, meaning-preserving rewording produced by
                   the fast model, told which phrasings already failed so it tries a new angle
                   each time (simplify a compound filter, drop a constraint the platform itself
                   computes, rephrase the metric, use the canonical entity name).

Entity resolution (canonical fund/AMC names) is the complementary Phase-1 step handled by
``fund_resolver`` + ``_recover_fund_entity`` in mcp_fetch. Decomposition is Phase 3.

``next_refinement`` returns the next query string to try, or None when it has nothing better —
in which case the caller falls back to ``cheap_variant``. Failing safe like this means the
worst case is exactly today's behaviour.
"""

import re

from langchain_core.messages import SystemMessage, HumanMessage

from app.llm import make_llm
from ...config import config


# Cache-busting, MEANING-PRESERVING surface variants. Each only changes the string so the
# platform can't replay its cached result for the identical text — it never adds a filter,
# scope, or entity (that would be meaning drift).
def cheap_variant(query: str, attempt: int) -> str:
    q = query.strip().rstrip(" ?.")
    variants = [
        query,                                  # attempt 0: original
        f"{q}.",                                # punctuation change
        f"Please provide: {q}.",                # neutral prefix
        f"{q} — give the exact values.",        # neutral suffix (no filter/scope change)
        f"Answer this precisely: {q}.",
    ]
    return variants[min(attempt, len(variants) - 1)]


_REWORD_PROMPT = """You rewrite a wealth-management data question that a stateless data platform just \
returned NOTHING for. Produce ONE genuinely DIFFERENT phrasing that the platform is more likely to \
answer, while keeping the EXACT same meaning and scope.

Hard rules:
- Plain English only. No SQL, no operators (=, <, >, LIKE, ILIKE), no table/column names.
- Keep every specific name, filter, timeframe and the ask type (count vs list vs single value).
- Do NOT add a new filter, entity, client, or aggregate scope that wasn't in the original.
- Do NOT just add punctuation or a prefix — change the WORDING (this is a real rephrase).
- Allowed moves: simplify a compound condition, ask for the underlying list instead of a computed
  filter the platform may not apply, rephrase the metric in plainer terms, or use a fuller name.
- Output ONLY the rewritten question — no quotes, no explanation."""


def _normkey(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", (s or "").lower())


def _safe_reword(original: str, candidate: str, history: list[str], entities: dict | None) -> bool:
    """Accept a reword only if it is non-empty, genuinely different from the original and from
    everything already tried, not a runaway expansion, and (when entities are known) free of the
    scope/filter drift the existing rewrite guard catches."""
    if not candidate:
        return False
    ck = _normkey(candidate)
    if ck == _normkey(original):
        return False
    if any(ck == _normkey(h) for h in (history or [])):
        return False
    if len(candidate) > len(original) * 2.5 + 80:
        return False
    # Never let a reword INVENT a client name that wasn't in the original (this is how a blank
    # decomposed sub-question — "...holdings for ," — got reworded into a wrong client).
    try:
        from .intent_enrichment import _known_investors
        o, c = original.lower(), candidate.lower()
        for nm in _known_investors():
            nl = nm.lower()
            if nl in c and nl not in o:
                return False
    except Exception:
        pass
    if entities:
        try:
            from .intent_enrichment import _is_safe_rewrite
            if not _is_safe_rewrite(original, candidate, entities):
                return False
        except Exception:
            pass
    return True


async def _llm_reword(query: str, history: list[str], entities: dict | None) -> str | None:
    """One fast-model rewrite into a genuinely different phrasing. Returns None on any failure
    or if the result doesn't pass the safety guard (caller then uses cheap_variant)."""
    try:
        llm = make_llm("fast", temperature=0.3, max_tokens=120)
        tried = "\n".join(f"- {h}" for h in (history or []) if h)
        resp = await llm.ainvoke([
            SystemMessage(content=_REWORD_PROMPT),
            HumanMessage(content=(
                f"Original question:\n{query}\n\n"
                f"Phrasings already tried (do NOT repeat these):\n{tried or '- (none)'}\n\nRewrite:"
            )),
        ])
        candidate = (resp.content or "").strip().strip('"').strip()
    except Exception:
        return None
    return candidate if _safe_reword(query, candidate, history, entities) else None


async def next_refinement(query: str, attempt: int, *, entities: dict | None = None,
                          history: list[str] | None = None) -> str | None:
    """Return the next query string to try for this retry attempt.

    Smart refinement leads with a GENUINELY different LLM rewording from the very first retry
    (it also cache-busts, so we don't lose the non-determinism benefit of the old cheap retry).
    Only if the reword is unavailable/unsafe do we fall back to the cheap deterministic variant
    — so the refined query the user sees is a real rephrase, not a cosmetic "{q}." tweak.
    """
    if not query:
        return None
    reworded = await _llm_reword(query, history or [], entities or {})
    return reworded or cheap_variant(query, attempt)
