"""Configuration loaded from environment variables."""

import os
from dataclasses import dataclass, field
from dotenv import load_dotenv

load_dotenv()


@dataclass
class Config:
    # LLM
    openai_api_key: str = field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))
    openai_base_url: str = field(default_factory=lambda: os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"))
    # Legacy single-model knob (kept for backward compat / any stray references).
    openai_model: str = field(default_factory=lambda: os.getenv("OPENAI_MODEL", "gpt-5.4-nano"))
    # Per-call model split (cost/accuracy): the final answer step needs the stronger
    # model; every other call (entity extraction, memory fact-extraction, etc.) is a
    # trivial structured task and runs on the cheap "fast" model.
    #   answer  -> gpt-5.4-mini  ($0.75/$4.50 per 1M)
    #   fast    -> gpt-5.4-nano  ($0.20/$1.25 per 1M)
    openai_model_answer: str = field(default_factory=lambda: os.getenv("OPENAI_MODEL_ANSWER", "gpt-5.4-mini"))
    openai_model_fast: str = field(default_factory=lambda: os.getenv("OPENAI_MODEL_FAST", "gpt-5.4-nano"))

    # MCP
    # NO in-code default, deliberately. This is a real external endpoint, so a
    # placeholder here would either leak the deployment's naming into the repo or
    # (worse) ship a plausible-looking hostname that does not resolve. It is supplied
    # by the gitignored .env; an empty value means "not configured" and the platform
    # calls fail loudly rather than silently pointing somewhere wrong.
    mcp_url: str = field(default_factory=lambda: os.getenv("MCP_URL", ""))
    mcp_api_key: str = field(default_factory=lambda: os.getenv("MCP_API_KEY", ""))

    # Server
    port: int = field(default_factory=lambda: int(os.getenv("PORT", "8000")))

    # Guardrail
    guardrail_config_path: str = field(default_factory=lambda: os.getenv("GUARDRAIL_CONFIG_PATH", "./configs"))
    # Master kill-switch. Set GUARDRAILS_ENABLED=false to bypass all guardrail layers
    # (input/context/output) — they then pass everything without calling the ML models
    # or the self-check LLM. Use temporarily, e.g. when the guardrail LLM provider is
    # unavailable (OpenAI 429/quota). Defaults to enabled.
    guardrails_enabled: bool = field(
        default_factory=lambda: os.getenv("GUARDRAILS_ENABLED", "true").strip().lower() not in ("false", "0", "no")
    )

    # Skill sync — the agent front-loads live skills from the Cortex platform MCP server
    # (the ``list_skills`` tool) at startup and holds them in RAM, refreshing on a TTL so
    # platform edits take effect without a redeploy. Falls back to the bundled seeds if the
    # MCP server is unreachable or returns nothing.
    skills_mcp_sync: bool = field(
        default_factory=lambda: os.getenv("SKILLS_MCP_SYNC", "true").strip().lower() not in ("false", "0", "no")
    )
    backend_url: str = field(default_factory=lambda: os.getenv("BACKEND_URL", "http://backend:8080/api"))
    skills_refresh_ttl: int = field(default_factory=lambda: int(os.getenv("SKILLS_REFRESH_TTL", "300")))

    # Cortex context map sync — front-loads the tenant context (get_cortex_capabilities),
    # the live client roster, and reference dimensions from the platform at startup and on
    # the skills TTL. Off = the agent falls back to its bundled seed roster / static prompts.
    cortex_context_sync: bool = field(
        default_factory=lambda: os.getenv("CORTEX_CONTEXT_SYNC", "true").strip().lower() not in ("false", "0", "no")
    )

    # Skills framework (M3). Master switch for selecting + applying domain skills. Per-mode
    # enablement is set in app/modes.py (Quick Facts off, Client Insights on, Deep optional).
    # Off = no skill is ever selected or applied.
    skills_enabled: bool = field(
        default_factory=lambda: os.getenv("SKILLS_ENABLED", "true").strip().lower() not in ("false", "0", "no")
    )

    # Approach A — methodology-driven multi-query. When a platform skill ships only its markdown
    # methodology (no structured plan_steps), an LLM reads the methodology and derives the ordered
    # Cortex questions to fetch, which then run as a decomposition plan (one resolve_context per
    # question). Derived steps are cached per (skill_id, methodology-hash). Off = content-only.
    skill_methodology_compile: bool = field(
        default_factory=lambda: os.getenv("SKILL_METHODOLOGY_COMPILE", "true").strip().lower() not in ("false", "0", "no")
    )

    # Quick Facts: treat every query as a standalone lookup — ignore in-session conversation
    # history so no pronoun/follow-up carries across turns (Client/Deep Insight keep history).
    quickfacts_standalone_only: bool = field(
        default_factory=lambda: os.getenv("QUICKFACTS_STANDALONE_ONLY", "true").strip().lower() not in ("false", "0", "no")
    )

    # Query planning + decomposition (Phase 3). The planner decides whether a question is
    # DIRECT (one platform call) or needs DECOMPOSE (several sub-questions run one-by-one /
    # in parallel, with earlier answers feeding later ones). Obvious single-clause questions
    # skip the planner LLM via a deterministic pre-gate, so the common case adds no latency.
    planner_enabled: bool = field(
        default_factory=lambda: os.getenv("PLANNER_ENABLED", "true").strip().lower() not in ("false", "0", "no")
    )
    decomposition_enabled: bool = field(
        default_factory=lambda: os.getenv("DECOMPOSITION_ENABLED", "true").strip().lower() not in ("false", "0", "no")
    )
    # Hard cap on platform calls per question (sub-questions + any per-item fan-out) — bounds
    # the worst-case latency/cost of a decomposed answer.
    max_subquestions: int = field(default_factory=lambda: int(os.getenv("MAX_SUBQUESTIONS", "6")))

    # Answer verification (Phase 4). VERIFY_ENABLED turns on the deterministic coverage gate:
    # after a DECOMPOSED question runs, any sub-question that came back empty gets ONE more
    # bounded retry (done at the DATA level, before the answer is written, so nothing is
    # double-streamed). ANSWER_VERIFY_ENABLED additionally runs a fast-model check that the
    # written answer actually addresses each part (off by default — it can only annotate, since
    # the answer is already streamed live).
    verify_enabled: bool = field(
        default_factory=lambda: os.getenv("VERIFY_ENABLED", "true").strip().lower() not in ("false", "0", "no")
    )
    answer_verify_enabled: bool = field(
        default_factory=lambda: os.getenv("ANSWER_VERIFY_ENABLED", "false").strip().lower() in ("true", "1", "yes")
    )

    # Smart refinement (Phase 2). Replaces the old cosmetic retry variants (which only added a
    # full stop / a "Please provide:" prefix). On an empty or failed platform result, retries
    # now escalate: a cheap deterministic cache-bust first, then a genuinely DIFFERENT,
    # meaning-preserving LLM rewording (told which phrasings already failed). Set
    # SMART_REFINE_ENABLED=false to fall back to the deterministic cache-bust variants only.
    smart_refine_enabled: bool = field(
        default_factory=lambda: os.getenv("SMART_REFINE_ENABLED", "true").strip().lower() not in ("false", "0", "no")
    )

    # Entity recovery (Phase 1). When a fund/scheme question returns NOTHING, the cause is
    # usually an exact-match miss on the fund name (the platform matches names exactly, e.g.
    # "HDFC Midcap opportunities" != stored "HDFC Mid-Cap Opportunities"). With this on, the
    # agent looks up the real names from the platform (a broad DISTINCT query) and
    # similarity-matches to recover the correct name, then re-fetches. It only fires on an
    # EMPTY result that actually named a fund, so the happy path is untouched. Set
    # FUND_RECOVERY_ENABLED=false to disable and fall back to today's behaviour.
    fund_recovery_enabled: bool = field(
        default_factory=lambda: os.getenv("FUND_RECOVERY_ENABLED", "true").strip().lower() not in ("false", "0", "no")
    )

    # Memory
    # Master kill-switch for long-term (mem0/Qdrant) memory. Set MEMORY_ENABLED=false to
    # bypass retrieval + storage entirely — no embedding calls, no fact-extraction LLM
    # calls. Use temporarily, e.g. when the LLM provider is unavailable. Defaults to on.
    memory_enabled: bool = field(
        default_factory=lambda: os.getenv("MEMORY_ENABLED", "true").strip().lower() not in ("false", "0", "no")
    )
    # Advisor-only memory: when on, ALL stored facts are attributed to the advisor
    # (the logged-in RM) — never to a client/investor — and retrieval surfaces only
    # advisor-scoped memories. Client (investor) memories are never written or read.
    # Defaults to off (mixed advisor + investor mode).
    memory_advisor_only: bool = field(
        default_factory=lambda: os.getenv("MEMORY_ADVISOR_ONLY", "false").strip().lower() in ("true", "1", "yes")
    )

    # Web search (Deep Insight only) — live real-world/event context. Inert until an API key is
    # set. Master switch defaults on, but with no key it simply returns no results.
    web_search_enabled: bool = field(
        default_factory=lambda: os.getenv("WEB_SEARCH_ENABLED", "true").strip().lower() not in ("false", "0", "no")
    )
    web_search_provider: str = field(default_factory=lambda: os.getenv("WEB_SEARCH_PROVIDER", ""))
    ceramic_api_key: str = field(default_factory=lambda: os.getenv("CERAMIC_API_KEY", ""))
    serper_api_key: str = field(default_factory=lambda: os.getenv("SERPER_API_KEY", ""))
    tavily_api_key: str = field(default_factory=lambda: os.getenv("TAVILY_API_KEY", ""))
    web_search_max_results: int = field(default_factory=lambda: int(os.getenv("WEB_SEARCH_MAX_RESULTS", "5")))
    web_search_timeout: float = field(default_factory=lambda: float(os.getenv("WEB_SEARCH_TIMEOUT", "8")))
    # India-localised results by default (Serper geo `gl` + an India query bias; Ceramic has no
    # geo param, so only the query bias applies there). Set WEB_SEARCH_COUNTRY="" to disable
    # localisation, or another ISO country code.
    web_search_country: str = field(default_factory=lambda: os.getenv("WEB_SEARCH_COUNTRY", "in"))

    # ns_probe
    # Hosted collector — the pipeline behind it (Kafka -> processor -> ClickHouse) is run
    # centrally, so an agent only has to send. A per-product alias also answers, but it is
    # a pre-rename leftover; this neutral host is the current one.
    ns_probe_endpoint: str = field(default_factory=lambda: os.getenv("NS_PROBE_ENDPOINT", "https://observability.neosapientai.com/v1/traces"))
    # The one field that separates the two variants on a shared dashboard. Each variant
    # pins it in its own compose file: `agent-cortex` / `agent-claude`.
    ns_probe_service_name: str = field(default_factory=lambda: os.getenv("NS_PROBE_SERVICE_NAME", "agent"))
    ns_probe_environment: str = field(default_factory=lambda: os.getenv("NS_PROBE_ENVIRONMENT", "development"))


config = Config()
