"""Skill registry — holds the active skills and matches a query to one.

Skills are FRONT-LOADED from the Cortex platform MCP server (the ``list_skills`` tool) at
startup and held in RAM, refreshed on a TTL so platform edits take effect without a redeploy.
If the MCP server is unreachable or returns nothing, the registry keeps the last-good set,
falling back to the bundled seeds. Matching is deterministic and cheap: trigger-phrase hits
(strong), tag hits, and sample-question token overlap.

Platform skills are applied CONTENT-driven (Option A): the ``list_skills`` payload carries the
methodology as ``markdownText`` (YAML frontmatter + body). We map the structured top-level row
fields onto the Skill model and use the frontmatter-stripped body as ``content``; the platform
does not supply plan_steps/synthesis/requires_client, so those stay empty and the skill is
applied as guidance the planner/Generate follows rather than a compiled decomposition plan.
"""

import asyncio
import logging
import re
import time

from ..config import config
from .models import Skill

logger = logging.getLogger(__name__)


def _tokens(text: str) -> set[str]:
    return {t for t in re.findall(r"[a-z0-9]+", (text or "").lower()) if len(t) > 2}


def _strip_frontmatter(md: str) -> str:
    """Return the markdown body with a leading ``---...---`` YAML frontmatter block removed.

    ``list_skills`` renders each skill as ``---\\n<yaml>\\n---\\n\\n<body>``; the structured
    fields we need are already on the top-level row, so we only want the human-readable body
    as the skill's ``content``. If there is no frontmatter, the text is returned unchanged."""
    text = (md or "").lstrip()
    if not text.startswith("---"):
        return md or ""
    # Split off the first fenced block: ---\n ... \n---
    m = re.match(r"^---\s*\n.*?\n---\s*\n?", text, re.DOTALL)
    if not m:
        return md or ""
    return text[m.end():].lstrip("\n")


def _extract_skills_payload(result: dict) -> list[dict]:
    """Pull the ``skills`` list out of an MCP ``list_skills`` tool result.

    ``mcp_client.call_tool`` returns ``{success, response, ...}`` where ``response`` is the
    JSON-RPC result: ``{"content": [{"type": "text", "text": "<json>"}], "isError": false}``.
    The ``text`` is the JSON string ``{"total": N, "skills": [...]}``. Returns ``[]`` on any
    shape we don't recognise (never raises) so a bad response just keeps the last-good set."""
    import json as _json
    if not result or not result.get("success"):
        return []
    response = result.get("response") or {}
    content = response.get("content") or []
    for block in content:
        if (block or {}).get("type") != "text":
            continue
        try:
            payload = _json.loads(block.get("text") or "{}")
        except (_json.JSONDecodeError, TypeError):
            continue
        skills = payload.get("skills")
        if isinstance(skills, list):
            return skills
    return []


def _mcp_skill_to_skill(d: dict) -> Skill:
    """Map a ``list_skills`` row (structured top-level fields + markdownText) to the Skill model.

    The row already carries name/displayName/description/skillType/triggers/tags as structured
    values, so no YAML parsing is needed — we only strip the frontmatter off ``markdownText`` to
    recover the methodology body as ``content``."""
    name = d.get("name", "") or ""
    return Skill(
        skill_id=name,
        name=d.get("displayName") or name,
        domain=d.get("domain") or "wealth_management",
        skill_type=d.get("skillType") or d.get("skill_type") or "reasoning_methodology",
        sensitivity=d.get("sensitivity", "internal"),
        description=d.get("description", ""),
        tags=d.get("tags") or [],
        triggers=d.get("triggers") or [],
        sample_questions=d.get("sample_questions") or [],
        content=_strip_frontmatter(d.get("markdownText", "")),
        # The platform does not supply these — Option A (content-driven) skills.
        plan_steps=[],
        synthesis="",
        requires_client=bool(d.get("requires_client")),
        exclusions=d.get("exclusions") or [],
        version=d.get("version", "v1"),
        status=d.get("status", "active"),
        created_by=d.get("created_by", "platform"),
    )


class SkillRegistry:
    def __init__(self, skills):
        self._skills = [s for s in skills if s.status == "active"]
        self._loaded_at = 0.0            # 0 => never synced from the platform yet
        self._source = "empty"
        self._lock = asyncio.Lock()

    def all(self) -> list:
        return list(self._skills)

    def cards(self) -> dict:
        """The skills currently held in RAM, shaped for the Skills UI card view."""
        skills = [
            {
                "id": s.skill_id,
                "skill_id": s.skill_id,
                "name": s.name,
                "domain": s.domain,
                "skill_type": s.skill_type,
                "description": s.description,
                "tags": list(s.tags),
                "triggers": list(s.triggers),
                "version": s.version,
                "status": s.status,
                "content": s.content,
            }
            for s in self._skills
        ]
        return {
            "skills": skills,
            "active": sum(1 for s in self._skills if s.status == "active"),
            "draft": sum(1 for s in self._skills if s.status == "draft"),
            "source": self._source,
            "loaded_at": self._loaded_at,
        }

    def get(self, skill_id: str):
        return next((s for s in self._skills if s.skill_id == skill_id), None)

    # --- live sync from the backend -------------------------------------------------
    async def ensure_fresh(self) -> None:
        """Refresh from the MCP server if the cache is stale. Cheap no-op within the TTL."""
        if not config.skills_mcp_sync:
            return
        if (time.time() - self._loaded_at) < config.skills_refresh_ttl:
            return
        async with self._lock:
            if (time.time() - self._loaded_at) < config.skills_refresh_ttl:
                return
            await self._refresh()

    async def force_refresh(self) -> dict:
        """Re-pull from the backend right now, ignoring the TTL (used by the manual reload)."""
        async with self._lock:
            await self._refresh()
        return {"count": len(self._skills), "source": self._source,
                "skills": [s.skill_id for s in self._skills]}

    async def _refresh(self) -> None:
        """Suppress per-request platform-call logging around the refresh: the ``list_skills``
        bootstrap call is internal and must not show up in the UI's "queries sent to the
        platform" panel (which shows what answered the user's question)."""
        from ..graph.nodes.stream_channel import suppress_platform_log
        _tok = suppress_platform_log.set(True)
        try:
            await self._refresh_impl()
        finally:
            suppress_platform_log.reset(_tok)

    async def _refresh_impl(self) -> None:
        # Pull the tenant/workspace skills from the Cortex platform via the MCP ``list_skills``
        # tool. The MCP client is generic and never raises — it returns a structured dict.
        from ..mcp.client import mcp_client
        try:
            result = await mcp_client.call_tool("list_skills", {"maxResults": 200})
            rows = _extract_skills_payload(result)
            active = [
                _mcp_skill_to_skill(r) for r in rows
                if (r.get("status") or "active") == "active"
            ]
            # Only adopt a non-empty platform set; an empty/failed response keeps the last-good skills.
            # The registry is platform-only — every skill is authored on the Cortex platform.
            if active:
                # Approach A: derive plan_steps from each methodology's markdown NOW (at load,
                # cached per content-hash) so the steps + requires_client are known before any
                # downstream gating/compilation looks at the skill — not lazily mid-turn.
                if config.skill_methodology_compile:
                    await self._compile_methodologies(active)
                self._skills = active
                self._source = "mcp"
            self._loaded_at = time.time()
            logger.info("[skills] front-loaded %d active skills from MCP list_skills", len(self._skills))
        except Exception as e:
            # Keep the current (seed / last-good) set; back off so a down server isn't hammered.
            self._loaded_at = time.time()
            logger.warning("[skills] MCP list_skills sync failed (%s) — using %s skills", e, self._source)

    @staticmethod
    async def _compile_methodologies(skills: list) -> None:
        """Populate plan_steps (+ requires_client, book_capable) for reasoning_methodology skills
        that ship only a markdown methodology, by deriving the Cortex questions from it. Best-effort
        per skill — a failure leaves the skill content-only. Cached, so unchanged skills are free."""
        from .methodology_compiler import derive_plan_steps
        for s in skills:
            if s.skill_type != "reasoning_methodology" or s.plan_steps or not (s.content or "").strip():
                continue
            try:
                derived = await derive_plan_steps(s)
            except Exception:
                continue
            if derived.get("steps"):
                s.plan_steps = derived["steps"]
                if derived.get("requires_client"):
                    s.requires_client = True
                if derived.get("book_capable"):
                    s.book_capable = True

    # --- matching (Option B: rule-based) --------------------------------------------
    @staticmethod
    def _phrase_hit(query_low: str, phrase: str) -> bool:
        """Whole-word phrase match — '\bmeet with\b' style. Stops a trigger from matching
        inside a larger word ('meet' in 'meet targets', 'meeting' in 'meetings')."""
        p = (phrase or "").strip().lower()
        if not p:
            return False
        return re.search(r"\b" + re.escape(p) + r"\b", query_low) is not None

    def _excluded(self, query_low: str, skill) -> bool:
        """A VETO rule: True when any of the skill's exclusion patterns matches the query.
        Exclusions are regexes (e.g. logistics/interrogative uses of a trigger word)."""
        for pat in getattr(skill, "exclusions", None) or []:
            try:
                if re.search(pat, query_low, re.IGNORECASE):
                    return True
            except re.error:
                # A malformed pattern must never crash matching — treat it as a plain phrase.
                if self._phrase_hit(query_low, pat):
                    return True
        return False

    def _passes_rules(self, query_low: str, skill, client_resolved: bool) -> bool:
        """Activation GATE evaluated before scoring. A skill is eligible only when:
          • no exclusion pattern vetoes it, AND
          • if it requires a client, a client has actually been resolved this turn."""
        if self._excluded(query_low, skill):
            return False
        if skill.requires_client and not client_resolved:
            return False
        return True

    def _score(self, query_low: str, query_tokens: set[str], skill) -> int:
        score = 0
        for trig in skill.triggers:
            if self._phrase_hit(query_low, trig):
                score += 3  # explicit trigger phrase — strong signal
        for tag in skill.tags:
            if self._phrase_hit(query_low, tag.replace("_", " ")):
                score += 1
        best_overlap = 0
        for sq in skill.sample_questions:
            best_overlap = max(best_overlap, len(query_tokens & _tokens(sq)))
        score += min(best_overlap, 2)
        return score

    def match(self, query: str, domain: str | None = None, min_score: int = 3,
              *, client_resolved: bool = False, intent: str | None = None):
        """Best-matching active skill for the query (optionally within a domain), or None.

        Rule-based: each candidate must first PASS its activation rules (exclusion vetoes +
        requires_client gate); only then is it scored on trigger/tag/sample-question signal.
        ``client_resolved`` / ``intent`` are the turn signals the rules evaluate.
        """
        q = (query or "").lower()
        qt = _tokens(query)
        best, best_score = None, 0
        for s in self._skills:
            if domain and s.domain != domain:
                continue
            # ``agentic_flow`` skills are driven by their own executor and activated by an LLM
            # intent gate, not keywords (e.g. Macro Portfolio Impact). They deliberately carry no
            # triggers, so they must never be selected by the deterministic scorer — a stray tag/
            # token overlap could otherwise nudge one in.
            if getattr(s, "skill_type", "") == "agentic_flow":
                continue
            if not self._passes_rules(q, s, client_resolved):
                continue
            sc = self._score(q, qt, s)
            if sc > best_score:
                best, best_score = s, sc
        return best if best_score >= min_score else None


# Platform-only: the registry starts EMPTY and is front-loaded from MCP list_skills at startup
# (see main.lifespan) and refreshed on a TTL. If the platform returns nothing, no skill applies.
registry = SkillRegistry([])
