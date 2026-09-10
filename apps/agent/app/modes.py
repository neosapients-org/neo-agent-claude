"""Mode Profile (M1) — map the chat mode to HOW the agent runs.

The frontend offers three modes — Quick Facts, Client Insights, Deep Insight — and sends the
chosen one with each message. This module turns that mode into concrete overrides for the
existing pipeline flags, so ONE agent core behaves three ways:

  • Quick Facts    — fast, direct facts; no decomposition, single platform call.
  • Client Insights— one client, in depth; decomposition allowed, verification on.
  • Deep Insight   — complex, multi-hop; decomposition on, higher budget, verification on.

An unknown / missing mode falls back to the global config defaults (today's behaviour), so
nothing breaks if the frontend doesn't send a mode.

Skills, long-term memory, client-scope locking and the Quick-Facts answer cache are LATER
phases (M2+); this module already reserves their keys so wiring them in is additive.
"""

from .config import config

# Canonical mode names used internally.
QUICK_FACTS = "quick_facts"
CLIENT_INSIGHTS = "client_insights"
DEEP_INSIGHT = "deep_insight"

# Accept the frontend ids ("quick"/"client"/"deep") and a few spellings.
_ALIASES = {
    "quick": QUICK_FACTS, "quick_facts": QUICK_FACTS, "quick-facts": QUICK_FACTS,
    "quickfacts": QUICK_FACTS, "quick facts": QUICK_FACTS,
    "client": CLIENT_INSIGHTS, "client_insights": CLIENT_INSIGHTS, "client-insights": CLIENT_INSIGHTS,
    "clientinsights": CLIENT_INSIGHTS, "client insights": CLIENT_INSIGHTS,
    "deep": DEEP_INSIGHT, "deep_insight": DEEP_INSIGHT, "deep-insight": DEEP_INSIGHT,
    "deepinsight": DEEP_INSIGHT, "deep insight": DEEP_INSIGHT,
}

# Per-mode overrides. Only the keys present here override the global config default; everything
# else inherits config. (skills / long-term memory / client_scope / cache are reserved for M2+.)
_MODE_PROFILES: dict[str, dict] = {
    QUICK_FACTS: {
        "decomposition_enabled": False,   # direct facts only — never fan out
        "max_subquestions": 1,
        "verify_enabled": False,          # nothing to backfill on a single call
        "skills_enabled": False,          # facts mode doesn't apply skills
        # reserved (later phases): "long_term_memory": False, "client_scope": False, "answer_cache": True,
    },
    CLIENT_INSIGHTS: {
        "decomposition_enabled": True,
        "max_subquestions": 6,            # a client brief needs ~5-6 parallel fetches
        "verify_enabled": True,
        "skills_enabled": True,           # skills (Pre-Meeting Brief, Recommendation) live here
        # reserved: "long_term_memory": True, "client_scope": True,
    },
    DEEP_INSIGHT: {
        "decomposition_enabled": True,
        "max_subquestions": 8,
        "verify_enabled": True,
        "skills_enabled": True,           # skills apply here too (e.g. Pre-Meeting Brief, Recommendation)
        # reserved: "long_term_memory": True,
    },
}

# The config flags a mode may override, with their global defaults.
_OVERRIDABLE = (
    "planner_enabled", "decomposition_enabled", "max_subquestions",
    "verify_enabled", "smart_refine_enabled", "fund_recovery_enabled",
    "skills_enabled",
)


def normalize_mode(mode: str | None) -> str | None:
    """Resolve any frontend/spelling variant to a canonical mode, or None if unrecognised."""
    if not mode:
        return None
    return _ALIASES.get(str(mode).strip().lower())


def mode_config(mode: str | None) -> dict:
    """Effective settings for this turn: global config defaults with the mode's overrides applied.

    Always returns every overridable key (so callers can read them uniformly), plus the resolved
    canonical ``mode``. Unknown/None mode → pure global defaults (today's behaviour).
    """
    eff = {k: getattr(config, k) for k in _OVERRIDABLE}
    canonical = normalize_mode(mode)
    if canonical:
        eff.update(_MODE_PROFILES.get(canonical, {}))
    eff["mode"] = canonical
    return eff


def eff(state: dict, key: str):
    """Read an effective flag for the current turn: the mode_config value if present, else the
    global config default. Lets nodes honour the mode without each one re-resolving it."""
    mc = state.get("mode_config") if isinstance(state, dict) else None
    if mc and key in mc:
        return mc[key]
    return getattr(config, key)
