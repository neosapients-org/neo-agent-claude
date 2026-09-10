"""Approach A — derive Cortex questions from a skill's markdown methodology.

Platform skills (from ``list_skills``) ship a human-readable methodology (``content``) but no
structured ``plan_steps``. Rather than ask the platform for a new field, we read the methodology
the way Claude reads a SKILL.md: an LLM extracts the ordered list of data questions the skill
needs, phrased the way ``resolve_context`` expects, with a ``{client}`` placeholder. Those become
the decomposition plan (one platform call per question) — restoring the multi-query fan-out.

The extraction is cached per ``(skill_id, methodology-hash)`` so it runs at most once per skill
version, not per turn. It is deterministic (temperature 0) and never raises — on any failure the
caller falls back to applying the skill as content-only guidance.
"""

from app.llm import ClaudeChat, make_llm
import hashlib
import json
import logging

from langchain_core.messages import HumanMessage, SystemMessage

from ..config import config

logger = logging.getLogger(__name__)

# (skill_id, sha1(content)) -> {"steps": [{"question": str}], "requires_client": bool}
_CACHE: dict[tuple[str, str], dict] = {}

_SYSTEM = (
    "You convert a wealth-management skill methodology into the exact data questions to ask the "
    "analytics platform. Read the methodology and output the ORDERED list of natural-language "
    "questions needed to gather its data. Rules: "
    "(1) phrase each as a standalone question the platform can answer, using a {client} "
    "placeholder wherever the client's name belongs; "
    "(2) include ONLY questions answerable from portfolio/analytics data — OMIT any step that "
    "needs email, documents, calendars, files, or tools; "
    "(3) preserve the order and granularity the methodology describes (if it asks for holdings by "
    "product type AND by instrument, that is two separate questions); "
    "(4) do not invent data the methodology does not ask for. "
    "Also decide two flags: requires_client — does the skill need a specific named client to run; "
    "and book_capable — does the methodology explicitly describe handling the WHOLE BOOK / "
    "\"my clients\" / all clients as a group, not only one named client. "
    'Reply with ONLY JSON: {"requires_client": <bool>, "book_capable": <bool>, '
    '"steps": ["question 1", "question 2", ...]}.'
)


def _extract_json(text: str) -> dict:
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


def _llm() -> ClaudeChat:
    return make_llm("answer", temperature=0, max_tokens=600)


async def derive_plan_steps(skill) -> dict:
    """Return ``{"steps": [{"question": str}], "requires_client": bool}`` derived from the skill's
    methodology, or empty steps on failure. Cached per (skill_id, methodology-hash)."""
    content = getattr(skill, "content", "") or ""
    if not content.strip():
        return {"steps": [], "requires_client": bool(getattr(skill, "requires_client", False)),
                "book_capable": bool(getattr(skill, "book_capable", False))}

    key = (skill.skill_id, hashlib.sha1(content.encode("utf-8")).hexdigest())
    if key in _CACHE:
        return _CACHE[key]

    user = f"Skill: {skill.name}\nDescription: {skill.description}\n\nMethodology:\n{content}"
    try:
        resp = await _llm().ainvoke([SystemMessage(content=_SYSTEM), HumanMessage(content=user)])
        data = _extract_json(resp.content) or {}
        raw_steps = data.get("steps") or []
        steps = [{"question": q.strip()} for q in raw_steps if isinstance(q, str) and q.strip()]
        result = {"steps": steps, "requires_client": bool(data.get("requires_client")),
                  "book_capable": bool(data.get("book_capable"))}
    except Exception as e:  # never crash the turn — fall back to content-only
        logger.warning("[skills] methodology compile failed for %s (%s)", skill.skill_id, e)
        result = {"steps": [], "requires_client": bool(getattr(skill, "requires_client", False)),
                  "book_capable": bool(getattr(skill, "book_capable", False))}

    _CACHE[key] = result
    logger.info("[skills] methodology-compiled %s → %d step(s)", skill.skill_id, len(result["steps"]))
    return result
