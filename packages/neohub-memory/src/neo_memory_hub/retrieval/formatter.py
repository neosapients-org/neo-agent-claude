"""Context formatting and profile assembly for LLM prompt injection.

Assembles retrieved memories into structured profiles and renders them
as markdown sections for LLM consumption.
"""
from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class ProfileAssembler:
    """Assemble retrieved memories into a structured entity profile.

    Groups memories by category and profile_key, producing a rich profile.

    Assembly rules (dynamic, no hardcoded keys):
    - persona/preference: group by profile_key → scalar or list
    - episodic: array of event objects with behavioral_note
    - procedural: array of action objects with trigger
    - team/org: array of directive objects with pool
    """

    def assemble(
        self,
        memories: List[Dict[str, Any]],
        entity_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Assemble memories into a structured profile dict.

        Args:
            memories: List of memory dicts with _retrieval_reason tag.
            entity_name: Primary entity name (investor, client, user).

        Returns:
            Dict with structure:
            {
                "entity_name": "...",
                "persona": {"key": "value", ...},
                "preference": {...},
                "episodic": [{"event": "...", "behavioral_note": "..."}, ...],
                "procedural": [{"action": "...", "trigger": "..."}, ...],
                "team": [...],
                "org": [...],
            }
        """
        profile: Dict[str, Any] = {"entity_name": entity_name}

        by_category: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for mem in memories:
            meta = mem.get("metadata", {}) or {}
            reason = mem.get(
                "_retrieval_reason", meta.get("memory_type", "persona")
            )
            by_category[reason].append(mem)

        def _group_by_key(mems: List[Dict[str, Any]]) -> Dict[str, Any]:
            """Group memories by profile_key into the assembled value shape (single value or
            list, with optional detail), sorted by salience."""
            key_groups: Dict[str, List[tuple]] = defaultdict(list)
            for mem in mems:
                meta = mem.get("metadata", {}) or {}
                pkey = meta.get("profile_key") or "general"
                content = (
                    mem.get("memory")
                    or mem.get("data")
                    or mem.get("content")
                    or ""
                )
                key_groups[pkey].append(
                    (content, meta.get("salience_score", 0.5), meta.get("detail", ""))
                )
            out: Dict[str, Any] = {}
            for pkey, entries in key_groups.items():
                entries.sort(key=lambda x: x[1], reverse=True)
                if len(entries) == 1:
                    content, salience, detail = entries[0]
                    out[pkey] = (
                        {"value": content, "detail": detail, "salience": salience}
                        if detail else content
                    )
                else:
                    out[pkey] = [
                        ({"value": content, "detail": detail, "salience": salience}
                         if detail else content)
                        for content, salience, detail in entries
                    ]
            return out

        # --- persona: group by profile_key ---
        persona_memories = by_category.get("persona", [])
        if persona_memories:
            assembled = _group_by_key(persona_memories)
            if assembled:
                profile["persona"] = assembled

        # --- preference: split ADVISOR vs CLIENT so a client recall never shows the
        # advisor's own working-style prefs as if they belonged to the client (and vice
        # versa). Each scope is grouped and rendered under its own labelled section. ---
        pref_memories = by_category.get("preference", [])
        if pref_memories:
            advisor_prefs = [
                m for m in pref_memories
                if (m.get("metadata", {}) or {}).get("subject_type") != "investor"
            ]
            client_prefs = [
                m for m in pref_memories
                if (m.get("metadata", {}) or {}).get("subject_type") == "investor"
            ]
            if advisor_prefs:
                profile["preference"] = _group_by_key(advisor_prefs)
            if client_prefs:
                profile["preference_client"] = _group_by_key(client_prefs)
                names = sorted({
                    (m.get("metadata", {}) or {}).get("investor_name")
                    for m in client_prefs
                    if (m.get("metadata", {}) or {}).get("investor_name")
                })
                profile["_preference_client_name"] = ", ".join(names)

        # --- episodic: event objects ---
        episodic = by_category.get("episodic", [])
        if episodic:
            events = []
            for mem in episodic:
                meta = mem.get("metadata", {}) or {}
                content = (
                    mem.get("memory")
                    or mem.get("data")
                    or mem.get("content")
                    or ""
                )
                if not content:
                    continue
                event_obj: Dict[str, Any] = {
                    "event": content,
                    "salience": meta.get("salience_score", 0.5),
                }
                if meta.get("profile_key"):
                    event_obj["key"] = meta["profile_key"]
                if meta.get("behavioral_note"):
                    event_obj["behavioral_note"] = meta["behavioral_note"]
                stored_at = meta.get("stored_at", "")
                if stored_at:
                    event_obj["date"] = (
                        stored_at[:10] if len(stored_at) >= 10 else stored_at
                    )
                events.append(event_obj)
            events.sort(key=lambda x: x.get("salience", 0), reverse=True)
            profile["episodic"] = events

        # --- procedural: action objects ---
        procedural = by_category.get("procedural", [])
        if procedural:
            actions = []
            for mem in procedural:
                meta = mem.get("metadata", {}) or {}
                content = (
                    mem.get("memory")
                    or mem.get("data")
                    or mem.get("content")
                    or ""
                )
                if not content:
                    continue
                action_obj: Dict[str, Any] = {
                    "action": content,
                    "salience": meta.get("salience_score", 0.5),
                }
                if meta.get("profile_key"):
                    action_obj["key"] = meta["profile_key"]
                if meta.get("trigger"):
                    action_obj["trigger"] = meta["trigger"]
                stored_at = meta.get("stored_at", "")
                if stored_at:
                    action_obj["date"] = (
                        stored_at[:10] if len(stored_at) >= 10 else stored_at
                    )
                actions.append(action_obj)
            actions.sort(key=lambda x: x.get("salience", 0), reverse=True)
            profile["procedural"] = actions

        # --- team/org/shared pool memories ---
        for pool_key in ("team", "org", "shared"):
            pool_mems = by_category.get(pool_key, [])
            if pool_mems:
                items = []
                for mem in pool_mems:
                    meta = mem.get("metadata", {}) or {}
                    content = (
                        mem.get("memory")
                        or mem.get("data")
                        or mem.get("content")
                        or ""
                    )
                    if not content:
                        continue
                    item: Dict[str, Any] = {
                        "directive": content,
                        "pool": pool_key,
                        "salience": meta.get("salience_score", 0.5),
                    }
                    if meta.get("profile_key"):
                        item["key"] = meta["profile_key"]
                    stored_at = meta.get("stored_at", "")
                    if stored_at:
                        item["date"] = (
                            stored_at[:10]
                            if len(stored_at) >= 10
                            else stored_at
                        )
                    items.append(item)
                items.sort(key=lambda x: x.get("salience", 0), reverse=True)
                profile[pool_key] = items

        return profile


class ContextFormatter:
    """Render a profile dict as markdown sections for LLM prompt injection.

    Usage:
        formatter = ContextFormatter(header="---CLIENT MEMORY CONTEXT---")
        text = formatter.format(profile, entity_name="Senthil Kumar")
    """

    def __init__(
        self,
        header: str = "---MEMORY CONTEXT---",
        max_chars: int = 8000,
    ) -> None:
        self._header = header
        self._max_chars = max_chars

    def format(
        self,
        profile: Dict[str, Any],
        entity_name: Optional[str] = None,
    ) -> str:
        """Format assembled profile as markdown for LLM prompt injection."""
        entity_name = entity_name or profile.get("entity_name", "Unknown")
        sections = []

        # --- Persona ---
        persona = profile.get("persona")
        if persona:
            lines = [f"## CLIENT PROFILE ({entity_name})"]
            for key, val in persona.items():
                display_key = key.replace("_", " ").title()
                if isinstance(val, list):
                    items = [
                        (
                            f"{v.get('value', '')} ({v.get('detail', '')})"
                            if isinstance(v, dict) and v.get("detail")
                            else (
                                v.get("value", "") if isinstance(v, dict) else str(v)
                            )
                        )
                        for v in val
                    ]
                    lines.append(f"- **{display_key}:** {', '.join(items)}")
                elif isinstance(val, dict):
                    text = val.get("value", "")
                    detail = val.get("detail", "")
                    lines.append(
                        f"- **{display_key}:** {text}"
                        + (f" — {detail}" if detail else "")
                    )
                else:
                    lines.append(f"- **{display_key}:** {val}")
            sections.append("\n".join(lines))

        # --- Preferences: rendered as two separate, clearly-attributed sections so a client
        # recall never surfaces the advisor's own working-style prefs as the client's. ---
        def _render_prefs(pref_dict: dict, header: str) -> None:
            lines = [header]
            for key, val in pref_dict.items():
                display_key = key.replace("_", " ").title()
                if isinstance(val, list):
                    items = [
                        (v.get("value", "") if isinstance(v, dict) else str(v))
                        for v in val
                    ]
                    lines.append(f"- **{display_key}:** {', '.join(items)}")
                elif isinstance(val, dict):
                    text = val.get("value", "")
                    detail = val.get("detail", "")
                    lines.append(
                        f"- **{display_key}:** {text}"
                        + (f" — {detail}" if detail else "")
                    )
                else:
                    lines.append(f"- **{display_key}:** {val}")
            sections.append("\n".join(lines))

        client_pref = profile.get("preference_client")
        if client_pref:
            name = profile.get("_preference_client_name") or (entity_name or "Client")
            _render_prefs(client_pref, f"## CLIENT PREFERENCES ({name})")

        advisor_pref = profile.get("preference")
        if advisor_pref:
            _render_prefs(advisor_pref, "## SERVICE PREFERENCES (Advisor)")

        # --- Episodic ---
        episodic = profile.get("episodic")
        if episodic:
            lines = [
                "## CLIENT EVENTS & DECISION CONTEXT",
                "Significant events, emotional moments, and decision reasoning.",
                "",
            ]
            for event in episodic:
                text = event.get("event", "")
                note = event.get("behavioral_note", "")
                date = event.get("date", "")
                lines.append(f"- {text}" + (f" ({date})" if date else ""))
                if note:
                    lines.append(f"  → *Behavioral insight:* {note}")
            sections.append("\n".join(lines))

        # --- Procedural ---
        procedural = profile.get("procedural")
        if procedural:
            lines = [
                "## ACTIVE REMINDERS & PROCEDURES",
                "Active reminders and standing instructions.",
                "ALWAYS mention relevant ones when they relate to the current topic.",
                "",
            ]
            for action in procedural:
                text = action.get("action", "")
                trigger = action.get("trigger", "")
                date = action.get("date", "")
                if trigger:
                    lines.append(
                        f"- **[when: {trigger}]** {text}"
                        + (f" (set on {date})" if date else "")
                    )
                else:
                    lines.append(
                        f"- {text}" + (f" (set on {date})" if date else "")
                    )
            sections.append("\n".join(lines))

        # --- Team ---
        team = profile.get("team")
        if team:
            lines = [
                "## DEPARTMENT DIRECTIVES (TEAM)",
                "Department-wide procedures shared across all users.",
                "",
            ]
            for item in team:
                text = item.get("directive", "")
                date = item.get("date", "")
                lines.append(
                    f"- {text}" + (f" (issued {date})" if date else "")
                )
            sections.append("\n".join(lines))

        # --- Org ---
        org = profile.get("org")
        if org:
            lines = [
                "## FIRM-WIDE DIRECTIVES (ORG)",
                "Organisation-wide mandates. Factor these into recommendations.",
                "",
            ]
            for item in org:
                text = item.get("directive", "")
                date = item.get("date", "")
                lines.append(
                    f"- {text}" + (f" (issued {date})" if date else "")
                )
            sections.append("\n".join(lines))

        if not sections:
            return ""

        formatted = f"{self._header}\n\n" + "\n\n".join(sections)
        if self._max_chars and len(formatted) > self._max_chars:
            cutoff = max(0, self._max_chars - 40)
            formatted = (
                formatted[:cutoff].rstrip()
                + "\n[...memory context truncated...]"
            )

        return formatted
