"""Skill selection (M3) — pick the skill that applies to this turn.

Gated by the master ``SKILLS_ENABLED`` flag AND the Mode Profile (Quick Facts off, Client
Insights on, Deep optional). Selection is LLM-over-descriptions: the platform skills are
front-loaded into the RAM registry at startup, and an LLM (openai_model_answer, gpt-5.4-mini)
reads their descriptions and picks the single best skill for the query (or none). Capped at
one skill.

Market-event / current-affairs questions are handled WITHOUT a special executor: the platform
authors an ordinary methodology skill (e.g. holdings + sector exposure), and generate's generic
Deep-Insight web pre-step supplies the live external context that the answer combines with the
platform data.
"""

import json
import logging

from langchain_core.messages import HumanMessage, SystemMessage

from app.llm import ClaudeChat, make_llm
from ...config import config
from ...skills.registry import registry

logger = logging.getLogger(__name__)

_DOMAIN = "wealth_management"

_SELECT_SYSTEM = (
    "You are a skill router for a wealth-management assistant. You are given the user's request "
    "and a numbered catalog of available skills. Each skill has a name, a description of when it "
    "applies, and some example phrasings that typically indicate it. Judge by MEANING — reason "
    "about whether the request matches a skill's intent; the example phrasings are hints, not "
    "keywords to string-match. "
    "Select a skill ONLY when the user is asking for the ENTIRE task/workflow that skill performs. "
    "A request for a single data point or one metric (e.g. one holding, the AUM, the XIRR, the top "
    "position, a single number) is NOT a match, even if the skill would fetch that data as part of "
    "its larger workflow — that is a plain lookup, so choose none. Match on the scope of what is "
    "asked, not on topical overlap with the data a skill happens to touch. "
    "Choose the SINGLE best-matching skill. If none clearly applies, choose none — do not force a "
    "match. "
    'Reply with ONLY a JSON object: {"skill_id": "<the id>"} or {"skill_id": null}.'
)


def _extract_json(text: str) -> dict:
    """Best-effort JSON parse — tolerate a code fence or surrounding prose the model may add."""
    t = (text or "").strip()
    if t.startswith("```"):
        t = t.strip("`")
        t = t[t.find("{"):] if "{" in t else t
    start, end = t.find("{"), t.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(t[start:end + 1])
        except json.JSONDecodeError:
            return {}
    return {}


def _selector_llm(max_tokens: int) -> ClaudeChat:
    # Skill routing runs on the mid tier (openai_model_answer, gpt-5.4-mini) rather than the
    # nano fast tier — picking the right skill from descriptions benefits from the stronger model.
    return make_llm("answer", temperature=0, max_tokens=max_tokens)


async def _llm_select(query: str, candidates: list) -> object | None:
    """Ask the LLM to pick the best skill from the candidates' descriptions + example phrasings,
    or None. Triggers are passed as example phrasings for the model to REASON over — not
    string-matched — so they sharpen precision without acting as brittle keyword rules."""
    if not candidates:
        return None
    catalog_lines = []
    for i, s in enumerate(candidates, 1):
        line = f'{i}. id="{s.skill_id}" — {s.name}: {s.description}'
        if s.triggers:
            # Cap the phrasings so a skill with a long trigger list can't dominate the prompt.
            examples = "; ".join(s.triggers[:8])
            line += f'\n   example phrasings: {examples}'
        catalog_lines.append(line)
    catalog = "\n".join(catalog_lines)
    user = f"User request:\n{query}\n\nAvailable skills:\n{catalog}"
    try:
        llm = _selector_llm(40)
        resp = await llm.ainvoke([SystemMessage(content=_SELECT_SYSTEM), HumanMessage(content=user)])
        picked = (_extract_json(resp.content) or {}).get("skill_id")
    except Exception as e:  # LLM/transport failure must never crash the turn
        logger.warning("[skills] LLM selection failed (%s) — no skill applied", e)
        return None
    if not picked:
        return None
    return next((s for s in candidates if s.skill_id == picked), None)


async def select_skills(state: dict) -> list:
    """Active skill(s) for this turn (Skill objects). Empty when skills are off for the mode/
    master flag or nothing matches."""
    if not config.skills_enabled:
        return []
    mc = state.get("mode_config") or {}
    if not mc.get("skills_enabled", config.skills_enabled):
        return []
    # Keep the front-loaded catalog fresh (cheap no-op within the TTL).
    await registry.ensure_fresh()

    # --- LLM-over-descriptions selection over the platform catalog --------------------------
    # Trigger phrases live in how the USER phrases the request, so route on the raw query first;
    # fall back to the enriched query (pronoun follow-ups whose intent only appears after the
    # client/intent is resolved).
    raw = (state.get("query") or "").strip()
    enriched = (state.get("enriched_query") or "").strip()
    # Candidates: every active platform skill. (agentic_flow skills are never in the registry.)
    candidates = [s for s in registry.all() if s.status == "active"]

    from .skill_compiler import _resolve_client
    client_resolved = bool(_resolve_client(state))

    skill = await _llm_select(raw, candidates)
    if not skill and enriched and enriched != raw:
        skill = await _llm_select(enriched, candidates)
    # requires_client gate: a client-scoped skill can't run without a resolved client.
    if skill and skill.requires_client and not client_resolved:
        skill = None
    return [skill] if skill else []
