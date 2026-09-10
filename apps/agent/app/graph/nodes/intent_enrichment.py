"""Intent Enrichment node — extracts structured entities and refines the query.

Runs AFTER intent classification (only for fetch_data intent).
Produces either:
  - An enriched, unambiguous query for MCP tool selection
  - A clarification question if critical entities are missing

This ensures MCP gets a precise, well-formed question instead of raw user input.
"""

import json
import logging
import re
import time
from difflib import get_close_matches, SequenceMatcher
from typing import Optional

logger = logging.getLogger(__name__)

try:
    from ns_probe import observe
except ImportError:
    def observe(*a, **kw):
        def _id(fn): return fn
        return _id

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from pydantic import BaseModel, Field

from app.llm import ClaudeChat, make_llm
from ...config import config
from ...data_dictionary import find_metric_issue
from ..state import AgentState


# --- Structured output models ---

class WealthEntities(BaseModel):
    """Structured entities extracted from a wealth management query."""

    investor_name: Optional[str] = Field(
        default=None,
        description="Name of the investor/client being asked about.",
    )
    data_type: Optional[str] = Field(
        default=None,
        description="Type of data requested: profile, portfolio, holdings, transactions, performance, aum, nav, risk.",
    )
    specific_metric: Optional[str] = Field(
        default=None,
        description="Specific metric if mentioned: returns, XIRR, CAGR, allocation %, NAV, AUM, etc.",
    )
    timeframe: Optional[str] = Field(
        default=None,
        description="Time scope: last month, YTD, last 1 year, since inception, etc.",
    )
    filters: list[str] = Field(
        default_factory=list,
        description="Any filters: asset class, fund name, scheme type, etc.",
    )
    fund_name: Optional[str] = Field(
        default=None,
        description=(
            "The EXACT fund / scheme / instrument name the user named, copied verbatim, if "
            "any (e.g. 'HDFC Midcap opportunities', 'Axis Bluechip', 'SBI Small Cap'). Leave "
            "null when no specific fund/scheme/instrument is mentioned. NEVER put a client or "
            "person name here."
        ),
    )
    comparison: Optional[str] = Field(
        default=None,
        description="Comparison intent: vs benchmark, vs previous period, across clients.",
    )
    confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Confidence in entity extraction quality (0-1).",
    )
    route: str = Field(
        default="fetch_data",
        description=(
            "How to handle the query. One of: "
            "'fetch_data' (wants live platform/financial data — DEFAULT when unsure), "
            "'recall_personal' (asking about THEMSELVES or their own saved preferences/profile, "
            "e.g. 'how should I be contacted', 'what do I prefer', 'what do you know about me'), "
            "'recall_client' (asking to RECALL a NOTE/preference the user previously RECORDED about "
            "a NAMED CLIENT — qualitative, from memory — e.g. 'what did I note about Ram Krishnan', "
            "'what do you remember about the Mehra family', 'remind me of Ram's recorded preferences'. "
            "This is NOT a request for the client's live financial data — if the user asks to "
            "show/list/get the client's holdings/value/transactions, use fetch_data instead), "
            "'preference_statement' (TELLING the assistant a preference/fact about themselves, "
            "e.g. 'I am risk-averse', 'always send PDF', 'only email me'), "
            "'suitability' (asking whether a product/asset/fund fits THEM personally, with no "
            "client named — e.g. 'is a small-cap fund a good fit for me', 'would a high-growth "
            "equity fund suit my risk appetite'), "
            "'preference_match' (asking whether a NAMED CLIENT's data/holdings match the user's "
            "own preferences — e.g. 'do Ram Krishnan's holdings match what I prefer'), "
            "'client_note' (the user is STATING a durable fact/preference ABOUT a NAMED CLIENT — "
            "a declarative sentence, NOT a question or data request — e.g. 'Ram Krishnan is "
            "risk-averse', 'the Mehra family prefers quarterly reviews', 'Venkat wants to move to "
            "debt'. Choose this ONLY when you are CERTAIN it asserts a fact and requests no data; "
            "if it asks to show/list/get/value/holdings or is a question, use fetch_data instead), "
            "'chitchat' (greetings, thanks, or asking what the assistant can do).\n"
            "NOTE: this field is SECONDARY — 'intent_kind' (the speech act) is the primary routing "
            "signal. Set this consistently with intent_kind. Do NOT pick fetch_data merely because "
            "finance words appear: a future-facing directive or a concept question is never fetch_data."
        ),
    )
    intent_kind: str = Field(
        default="request_data",
        description=(
            "The SPEECH ACT — what the user is DOING with this message, independent of the "
            "topic or any financial words it contains. Choose exactly one: "
            "'request_data' (wants specific values/records returned NOW — holdings, value, AUM, "
            "transactions, performance, a list/count); "
            "'request_fit' (asks whether something SUITS them or a client — a judgement against "
            "preferences, e.g. 'is a small-cap fund good for me', 'do Ram's holdings match what I prefer'); "
            "'instruct' (tells the assistant HOW to behave/present GOING FORWARD — a directive about "
            "future output/format/inclusion/exclusion, e.g. 'from now on always include the expense "
            "ratio', 'always send PDF', 'going forward exclude tobacco'. A directive is 'instruct' "
            "EVEN IF it mentions data words like holdings/expense ratio/AUM); "
            "'inform' (STATES a durable fact, not a question — about themselves or a named client, "
            "e.g. 'I prefer mutual funds', 'Ram Krishnan is risk-averse'); "
            "'recall' (asks to RETRIEVE something from saved memory or an earlier turn, e.g. 'what "
            "do you know about me', 'what did I note about Ram', 'repeat that'); "
            "'concept' (a generic/educational finance question with NO client and NO request for "
            "platform data, e.g. 'what is an index fund', 'explain how an SIP works'. A request to "
            "list employees/RMs, the clients they service, or their cities is request_data, NOT "
            "concept); "
            "'smalltalk' (greeting, thanks, or asking what you can do). "
            "Decide by the VERB/MOOD (asking vs instructing vs informing vs recalling), NOT by whether "
            "finance terms appear. If genuinely unsure, choose 'request_data'."
        ),
    )
    message_kind: str = Field(
        default="data",
        description=(
            "Classify the user's CURRENT message using the conversation context: "
            "'data' = a self-contained data question that already names its own subject; "
            "'followup' = a data request that only makes sense via the previous turn "
            "(refers to 'those'/'them'/'that list', or continues/elaborates the last "
            "question, e.g. 'list those clients', 'what about the Mehra family'); "
            "'smalltalk' = greeting/acknowledgement/conversational filler with no data "
            "intent (e.g. 'hey', 'lets begin', 'ok go on', 'thanks'); "
            "'recall' = asking about what was ALREADY said/shown (e.g. 'repeat that', "
            "'summarize the list above'). When unsure between data and followup, pick 'data'."
        ),
    )
    standalone_query: Optional[str] = Field(
        default=None,
        description=(
            "For ANY data request (message_kind 'data' OR 'followup'), rewrite the current "
            "message into ONE self-contained question the data platform can answer WITHOUT the "
            "conversation (the platform is stateless and cannot see prior turns). "
            "PRIORITY — the CURRENT message's OWN data intent wins: ALWAYS resolve the SUBJECT "
            "(pronouns / who) from context, but only inherit a METRIC or TIMEFRAME from a prior "
            "turn when the current message names NONE of its own (e.g. 'and last quarter?'). If the "
            "current message already says what it wants (investment details, holdings, "
            "transactions, AUM, NAV, performance, profile), KEEP THAT — never let the prior turn's "
            "metric override it. Resolve EVERY "
            "back-reference using the conversation, not just obvious pronouns: "
            "pronouns (his/her/their/them) -> the exact client name; referential phrases "
            "('the same fund', 'that family', 'those clients', 'the ones above'); an "
            "implied/omitted subject ('and last quarter?' -> carry the prior client + metric); "
            "and carried-over filters/timeframes from the relevant prior turn. Keep the "
            "message's OWN intent type (list vs count vs single value), e.g. prior 'how many "
            "clients invested in AIF' + 'list those clients' -> 'List all clients who have "
            "invested in AIF, with their AIF exposure.'. Use EXACT client/family/fund names "
            "from context; NEVER invent a name. If the Memory context states a standing "
            "preference for which fields to show for this data type (e.g. holdings -> cost, "
            "current value, unrealized P&L), add those fields to the query so the platform "
            "returns them, without changing the list/count/aggregate intent. If the message is "
            "ALREADY self-contained, echo it UNCHANGED. Leave null ONLY for non-data messages "
            "(smalltalk / recall)."
        ),
    )
    own_data_type: Optional[str] = Field(
        default=None,
        description=(
            "The data type THIS message names ON ITS OWN — one of: profile | portfolio | "
            "holdings | transactions | performance | aum | nav | risk. Judge ONLY the latest "
            "user message, IGNORING the conversation and any prior turn. If the message does "
            "NOT name its own data type and relies on the previous turn (e.g. 'and last "
            "quarter?', 'worst one too', 'her too?'), return null. Examples: 'her investment "
            "detail' -> holdings; 'his info' -> profile; 'what she owns' -> holdings; 'provide "
            "his total aum' -> aum; 'and last quarter?' -> null."
        ),
    )


EXTRACTION_SYSTEM_PROMPT = """You extract structured entities from wealth management queries.
The user is asking about investor data on the Neo Platform (internal).

Extract factual entities only. Do not invent values.
If something is not mentioned or cannot be inferred, leave it null.
Use conversation or memory context to resolve pronouns (e.g., "his" -> a known investor name).

CRITICAL: System/platform identifiers like "user2", "user1", "admin", or any "user<N>" string
are login IDs — they are NEVER investor names. If only such an identifier appears, set investor_name to null.

IMPORTANT: Investor IDs like "INV-001", "INV-002", "INV001" etc. are VALID investor identifiers.
If the user mentions an investor ID (e.g. "investor INV-001"), set investor_name to that ID (e.g. "INV-001").
These are legitimate platform identifiers that the data system can resolve.

Common data types: profile, portfolio, holdings, transactions, performance, aum, nav, risk_assessment
Common metrics: returns, XIRR, CAGR, allocation_percentage, NAV, AUM, total_investment, current_value

ORGANIZATIONAL DATA — the platform ALSO serves org/RM-structure data, not just investor finance:
employees / relationship managers (RMs) / bankers, the clients each one services, their branches
and cities, and team structure. A request to list employees/RMs, who they service, or their cities
is LIVE platform data → intent_kind "request_data" and route "fetch_data". Do NOT treat these as
"concept"/general just because no investor name or finance metric appears.

ROUTING — also set the "route" field based on what the user wants:
- "fetch_data": the user wants WEALTH MANAGEMENT FINANCIAL data — holdings, portfolio value, AUM, clients, employees, RMs,
  transactions, performance, allocations. This is the DEFAULT — when unsure, choose fetch_data.
- "client_note": the RM is STATING a durable fact, goal, or preference ABOUT a NAMED CLIENT.
  Structural rule: the sentence is declarative (no "?"), begins with or names the client, and
  asserts something about WHO they are, WHAT they want, HOW they invest, or WHAT their goals are —
  with no instruction to show, list, fetch, or retrieve any live data. The RM is INFORMING you,
  not ASKING you. Use this whenever the sentence structure is [Client] + [state/preference/goal]
  and there is no data retrieval intent.
- "recall_client": the RM is asking to RECALL a note or preference they previously recorded about
  a NAMED CLIENT — qualitative, from memory. Examples: "what did I note about Joseph Fernandes?",
  "remind me of Arjun Mehra's risk profile I recorded", "what do you know about the Mehra family?".
  NOT a request for live financial data — if they ask for holdings/value/transactions, use fetch_data.
- "recall_personal": the user is asking about THEMSELVES or their own saved preferences/profile.
  Examples: "what do you know about me?", "what are my investment preferences?", "how should I be contacted?".
- "preference_statement": the user is TELLING you a fact about THEMSELVES (not a client). Examples:
  "I am risk-averse", "I prefer mutual funds", "always send me PDF reports".
- "chitchat": greetings, thanks, or asking what you can do.

SPEECH ACT — also set "intent_kind" (this is the PRIMARY routing signal; the route above is
secondary). Classify what the user is DOING, by the verb/mood, NOT by whether finance words appear:
- "request_data": wants specific values/records returned NOW.
- "request_fit": asks whether something SUITS them or a client (a judgement vs preferences).
- "instruct": tells you HOW to behave/present GOING FORWARD (a directive about future output —
  include/show/format/exclude). This is "instruct" EVEN IF it mentions holdings/expense ratio/AUM.
- "inform": STATES a durable fact (not a question) about themselves or a named client.
- "recall": asks to RETRIEVE from saved memory or an earlier turn.
- "concept": a generic/educational finance question with NO client and NO data request.
- "smalltalk": greeting/thanks/what-can-you-do.
DISCRIMINATOR: "show Ram's expense ratio" = request_data; "always include the expense ratio" =
instruct; "what is an expense ratio" = concept. Same words, different speech act.
Do NOT default to request_data just because finance terms appear — a future-facing directive or a
concept question is NEVER request_data. Only choose request_data when the user is genuinely asking
for data to be returned now. If truly unsure, request_data.
CLIENT NOTE RULE: "<Client name> is/prefers/wants/avoids <qualitative fact>" with no "?" = inform (client).
FIRST-PERSON RULE: "me/my/I" referring to the logged-in user asking about their own preferences = recall (self).

CONVERSATION CONTEXT & CONTINUITY (very important — the data platform is STATELESS and
cannot see previous turns, so EVERY data request must be made self-contained here):
- Set message_kind to 'data', 'followup', 'smalltalk', or 'recall' (see the field guide).
- Judge it from the conversation context, NOT just the words. After a greeting, "lets begin"
  / "ok" / "go on" are 'smalltalk'. A bare "list them" / "who are they" / "what about the
  Mehra family" right after a data answer is 'followup'.
- A reply that ANSWERS a clarifying question the assistant just asked is a 'followup'
  (e.g. assistant "Did you mean AUM?" -> user "yes i meant AUM" / "AUM" / "the first one").
  Apply the answer to the ORIGINAL question (-> "What is the AUM across all my clients?").
- For EVERY data request (message_kind 'data' OR 'followup'), produce standalone_query: a
  single self-contained question the stateless platform can answer alone. Resolve ALL
  references from the conversation, NOT just obvious pronouns:
    * pronouns — "his/her/their/them" -> the exact client name from context;
    * referential phrases — "the same fund", "that family", "those clients", "the ones above";
    * an implied/omitted subject — "and last quarter?" -> carry the prior client + metric;
    * carried-over filters/timeframes from the relevant prior turn.
  Keep the message's OWN intent type (list vs count vs single value). Use the EXACT
  client/family/fund names from the conversation; never invent one. If the message is ALREADY
  fully self-contained, echo it UNCHANGED.
- STANDING FIELD PREFERENCES: if the Memory context states a durable preference for WHICH
  metrics/fields to show for a given data type (e.g. "for holdings always show cost, current
  value and unrealized P&L per holding"), and the current request is for that data type,
  INCLUDE those exact fields in standalone_query so the platform returns them (e.g.
  "List Ram Krishnan's holdings with instrument name, cost, current value and unrealized P&L
  for each."). This only adds the preferred fields to what is FETCHED — it does NOT change the
  list/count/aggregate intent, and never invents fields the preference didn't ask for. Ignore
  preferences about pure presentation (tone, table vs prose, grouping) here — those are applied
  when the answer is written, not when the data is fetched.
- For 'smalltalk' / 'recall' (and any non-data message), leave standalone_query null."""


ENRICHMENT_SYSTEM_PROMPT = """You rewrite wealth management queries into precise, unambiguous natural language questions for a data retrieval system.

Rules:
- Output ONLY a plain English question — no SQL, no "field = value" syntax, no table/column names
- Include the investor name/ID, data intent, and any relevant filters or timeframe
- ALWAYS use the EXACT full client name from the extracted entities' investor_name verbatim (e.g. if
  investor_name is "Ram Krishnan", write "Ram Krishnan" — never shorten it to "Ram"). If the original
  query used a short or misspelled name, replace it with this exact full name.
- Make implicit context explicit (e.g., if "his portfolio" and investor is Ram Krishnan → "Ram Krishnan's portfolio holdings")
- Be concise but complete
- NEVER include system/login identifiers (e.g. "user2", "user1", "admin") — these are not investor names
- Investor IDs like INV-001 are valid identifiers — keep them in the enriched query as-is
- If no investor name was specified and the query is about all/multiple investors, keep it as a query across all investors — do NOT add a specific investor name

CRITICAL — whenever a query asks to LIST, NAME, or IDENTIFY clients/investors (any "list any
client", "which clients", "who are the clients", "for each investor" style query), explicitly ask
for the client NAME, not the client ID. The platform returns investor IDs by default, which are not
useful to the user. Always phrase these as "... with their client/investor NAME" (plus any other
requested fields) so the result shows the actual client names rather than IDs.

CRITICAL for aggregate/multi-investor queries (NEVER reduce these to single-investor lookups):
- "How many investors have X" → keep as "How many investors have X?" (preserve count intent)
- "List all investors with X" → keep as "List all investors who have X, with their name" (preserve listing intent; show NAMES not IDs)
- "Which investors prefer X" → keep as "Which investors prefer X? (show their name)" (preserve multi-investor scope; names not IDs)
- "Top N investors by X" → keep as "Top N investors ranked by X, with their name" (preserve ranking intent; names not IDs)
- "For each investor, show X" → keep as "For each investor, show their name and X" (preserve per-investor scope; names not IDs)
- "Average/total/count across all investors" → keep as aggregate question across all investors
- NEVER add a specific investor name to an aggregate query that didn't originally mention one

CRITICAL for single-value "worth/value" questions (must request the AGGREGATED total, otherwise the
data system returns a raw list of individual holdings and only the first is used):
- "<name>'s portfolio value" / "value of <name>'s portfolio" / "how much is <name>'s portfolio worth"
  → "What is the TOTAL portfolio value of <name> (the sum of the current value of all their holdings)?"
- "<name>'s net worth" / "total investment of <name>" / "AUM of <name>"
  → phrase as the TOTAL / SUM for that client, never a per-holding list.
- Always include the word "total" (and "sum of all holdings" where natural) for these value questions.

CRITICAL for gain/loss & performance-filter questions (the data platform returns the FULL holdings
list WITH each holding's gain/loss but does NOT itself filter to "only losers" / "only winners" — if you
ask it to filter, it returns nothing). So NEVER phrase these as a filter the platform must compute.
Instead ask for the COMPLETE holdings list with the figures needed, and the answer step will pick the matches:
- "which holdings are in a loss/loss position / underwater / losing money / in the red"
  → "List ALL holdings of <name> with their invested value, current value and unrealized gain/loss."
- "which holdings are in profit / gaining / in the green / making money"
  → "List ALL holdings of <name> with their invested value, current value and unrealized gain/loss."
- "best / worst / top / bottom performing holdings", "biggest gainers/losers"
  → "List ALL holdings of <name> with their current value and unrealized gain/loss percentage."
- Do the same for any per-holding metric filter (e.g. "holdings up more than 20%") — fetch the full list
  with the metric, never ask the platform to apply the threshold.

Use exact category names when relevant (e.g., "High risk tolerance", "Retirement Planning goal", "Market Crash scenario", "Long-term investment horizon")."""


def _clarification_system_prompt() -> str:
    """Clarification prompt built from the LIVE platform context: the client roster (grouped
    by family) and the capability scope from the Cortex context map. Both come from the
    context registry, so a tenant change (new client, new data domain) flows into the
    clarification behaviour without a code edit."""
    from ...context import cortex_context
    lines = [
        "You ask one concise clarification question to the user.",
        "The user asked a wealth management query but critical information is missing.",
        "Ask naturally — don't list field names, ask like a helpful assistant would.",
        "A complete data question names WHO (client / family / whole book), WHAT (the metric "
        "or data), and — when the data is time-based — WHEN (the period). Ask ONLY for the "
        "missing part.",
        "",
        "IMPORTANT: When disambiguating client names, ONLY suggest names from this known "
        "client list. NEVER invent or hallucinate client names. If you don't know the exact "
        "matches, ask the user to provide the full name without guessing.",
        "",
        "Known clients grouped by family:",
    ]
    for family, members in cortex_context.families().items():
        lines.append(f"- {family}: {', '.join(members)}")
    scope = cortex_context.scope_block(max_examples=12)
    if scope:
        lines += ["", "Platform scope (use it to suggest what CAN be asked):", scope]
    return "\n".join(lines)

def _has_data_vocabulary(q_low: str) -> bool:
    """Does the query use the tenant's data vocabulary? Grounded in the Cortex context map:
    any meaningful word from a primitive description or a verified example question counts
    (see build_capability_vocabulary). Falls back to the seed keyword list below when the
    capability map hasn't loaded — so behaviour degrades to exactly the old check."""
    from ...context import cortex_context
    vocab = cortex_context.capability_vocabulary()
    if vocab:
        for w in re.findall(r"[a-z][a-z&'-]{2,}", q_low):
            if w in vocab:
                return True
        # multi-word seed phrases (e.g. "net new") still checked — vocab is single words
    return any(kw in q_low for kw in _FINANCIAL_KEYWORDS)


# Seed keywords indicating a financial data request — FALLBACK ONLY, used when the platform
# capability map is unavailable (see _has_data_vocabulary above).
_FINANCIAL_KEYWORDS = [
    "portfolio", "holdings", "profile", "investor", "investment", "returns",
    "performance", "risk", "aum", "nav", "transaction", "fund", "scheme",
    "allocation", "dividend", "xirr", "cagr", "gain", "loss", "equity",
    "debt", "mutual fund", "etf", "reits", "epf", "bond", "sip",
    # money flow / inflow / outflow queries (e.g. "net new money inflow") — these
    # are LIVE data and must always re-fetch from MCP, never answer from session memory
    "inflow", "outflow", "net new", "new money", "cash flow", "net flow",
    "money inflow", "money outflow", "net inflow", "net outflow",
    # profile / KYC detail follow-ups (e.g. "what about his pan card number") — fetch
    # live so the platform value is presented, instead of an LLM memory refusal
    "pan", "kyc", "aadhaar", "passport", "nominee", "email", "phone", "mobile",
    "date of birth", "dob", "marital", "address",
    # organizational / RM-structure data (employees, who they service, branches, cities) —
    # LIVE platform data, so these must route to MCP even with no investor name or finance
    # metric present. " rm " is space-padded to avoid matching inside "farmer"/"information".
    "employee", "relationship manager", " rm ", "banker", "branch",
    "city", "cities", "servicing", "team",
]

# Phrases that are clearly general/conversational with no data intent
_GENERAL_PATTERNS = [
    "hello", "hi ", "hey ", "good morning", "good afternoon", "good evening",
    "how are you", "what can you", "what do you do", "help me understand",
    "tell me about yourself", "who are you", "thank", "thanks", "bye",
    "goodbye", "what is agent", "what is neo",
]

# Honorifics and patterns that signal a person name reference (not general chat)
_NAME_INDICATORS = [
    "mr.", "mrs.", "ms.", "mr ", "mrs ", "ms ", "shri", "smt",
    "sir", "madam",
]

# Known client roster — sourced LIVE from the Cortex platform via the context registry
# (front-loaded at startup, TTL-refreshed; see app/context/registry.py). The registry falls
# back to its bundled seed roster when the platform is unreachable, so these accessors never
# come back empty. Derived indexes (lowercase list, first-name map, surname map) are memoized
# per registry version — rebuilt only when a refresh actually adopted a new roster.
_roster_memo: dict = {"version": None, "names": [], "lower": [], "first": {}, "sur": {}}


def _roster_indexes() -> dict:
    from ...context import cortex_context
    if _roster_memo["version"] != cortex_context.version:
        names = cortex_context.roster_names()
        first: dict[str, list[str]] = {}
        sur: dict[str, list[str]] = {}
        for full in names:
            parts = full.split()
            first.setdefault(parts[0].lower(), []).append(full)
            sur.setdefault(parts[-1].lower(), []).append(full)
        _roster_memo.update(version=cortex_context.version, names=names,
                            lower=[n.lower() for n in names], first=first, sur=sur)
    return _roster_memo


def _known_investors() -> list[str]:
    """Full canonical client names for fuzzy matching (avoids MCP calls on misspellings)."""
    return _roster_indexes()["names"]


def _known_investors_lower() -> list[str]:
    return _roster_indexes()["lower"]


def _first_name_map() -> dict[str, list[str]]:
    """first name -> [full canonical names]"""
    return _roster_indexes()["first"]


def _surname_map() -> dict[str, list[str]]:
    """surname/family -> [full canonical names]"""
    return _roster_indexes()["sur"]


def _known_surnames() -> list[str]:
    """Client surnames / family names (for fast name detection)."""
    return list(_roster_indexes()["sur"].keys())

_HONORIFICS = ("mr.", "mrs.", "ms.", "dr.", "mr ", "mrs ", "ms ", "dr ",
               "shri ", "smt ", "smt. ", "sir ", "madam ")

# Login / system identifiers that must NEVER be treated as a client name.
_LOGIN_ID_PATTERN = re.compile(
    r"^\s*(user\s*\d+|user|usr\d*|admin|administrator|guest|test\s*user|testuser|"
    r"test|root|system|null|none|anonymous|account\s*\d+)\s*$", re.I)


def _is_login_id(name: str) -> bool:
    return bool(name) and bool(_LOGIN_ID_PATTERN.match(name.strip()))


def _resolve_client_name(raw: str) -> tuple[str | None, list[str] | None]:
    """Resolve a (possibly partial / misspelled) client reference to a canonical
    full name from the known roster.

    Returns (canonical_name, None)   -> resolved uniquely
            (None, [candidates])     -> ambiguous; caller should clarify
            (None, None)             -> no roster match; pass the name through as-is
    """
    if not raw or _is_login_id(raw):
        return (None, None)
    low = raw.strip().lower()
    for h in _HONORIFICS:
        if low.startswith(h):
            low = low[len(h):].strip()
    # strip a trailing possessive ("ram's" -> "ram")
    if low.endswith("'s"):
        low = low[:-2].strip()
    elif low.endswith("’s"):
        low = low[:-2].strip()
    if not low:
        return (None, None)

    # exact full name
    if low in _known_investors_lower():
        return (_known_investors()[_known_investors_lower().index(low)], None)

    tokens = low.split()

    # single token: first-name, then surname/family, then typo-fuzzy on first names
    if len(tokens) == 1:
        tok = tokens[0]
        if tok in _first_name_map():
            names = _first_name_map()[tok]
            return (names[0], None) if len(names) == 1 else (None, names)
        if tok in _surname_map():
            # a bare surname/family refers to multiple people -> ambiguous
            members = _surname_map()[tok]
            return (None, members) if len(members) > 1 else (members[0], None)
        fm = get_close_matches(tok, list(_first_name_map().keys()), n=2, cutoff=0.82)
        if len(fm) == 1:
            names = _first_name_map()[fm[0]]
            return (names[0], None) if len(names) == 1 else (None, names)
        return (None, None)

    # multi-token: fuzzy against full names (handles typos like "Rama Krishana")
    matches = get_close_matches(low, _known_investors_lower(), n=3, cutoff=0.6)
    if matches:
        best = _known_investors()[_known_investors_lower().index(matches[0])]
        if len(matches) > 1:
            s0 = SequenceMatcher(None, low, matches[0]).ratio()
            s1 = SequenceMatcher(None, low, matches[1]).ratio()
            if abs(s0 - s1) < 0.08:
                cands = [_known_investors()[_known_investors_lower().index(m)] for m in matches[:3]]
                cands = list(dict.fromkeys(cands))
                if len(cands) > 1:
                    return (None, cands)
        # Surname-agreement guard: a whole-name fuzzy match can collapse a DIFFERENT person
        # who merely shares a first name (e.g. "Kavya Reddy" -> "Kavya Patel"). Only auto-apply
        # when the surnames are also close (a genuine typo of the same person, e.g.
        # "krishana" ~ "krishnan"); otherwise DON'T silently rewrite — surface it as a
        # clarification candidate so we ask "did you mean <X>?" instead of acting on the
        # wrong client (which would also leak the wrong client's data on MCP queries).
        in_surname = tokens[-1]
        best_surname = matches[0].split()[-1] if matches[0].split() else ""
        if in_surname != best_surname and SequenceMatcher(
            None, in_surname, best_surname
        ).ratio() < 0.8:
            return (None, [best])
        return (best, None)

    # fall back: known first name as the first token (e.g. "Sita Devi" -> Sita Krishnan)
    if tokens[0] in _first_name_map():
        names = _first_name_map()[tokens[0]]
        return (names[0], None) if len(names) == 1 else (None, names)

    return (None, None)


def _fuzzy_match_investor(name: str) -> str | None:
    """Fuzzy-match an investor name against the known list.

    Returns the corrected name if a close match is found (cutoff=0.6),
    or None if no match. If the name already matches exactly, returns it as-is.
    """
    if not name:
        return None
    name_lower = name.lower().strip()
    # Exact match (case-insensitive)
    if name_lower in _known_investors_lower():
        idx = _known_investors_lower().index(name_lower)
        return _known_investors()[idx]
    # Fuzzy match
    matches = get_close_matches(name_lower, _known_investors_lower(), n=1, cutoff=0.6)
    if matches:
        idx = _known_investors_lower().index(matches[0])
        return _known_investors()[idx]
    return None


def _has_name_reference(query: str) -> bool:
    """Return True if query likely references an investor name."""
    q = query.lower().strip()
    # Check for honorifics
    if any(h in q for h in _NAME_INDICATORS):
        return True
    # Check for known surnames
    if any(s in q for s in _known_surnames()):
        return True
    # Check for possessive patterns suggesting a follow-up about a person
    if any(p in q for p in ["his ", "her ", "their ", "same investor", "same person", "'s"]):
        return True
    return False


def _is_general_query(query: str) -> bool:
    """Fast rule-based check — no LLM needed.

    Returns True if the query is clearly conversational with no wealth management and financial data intent.
    """
    q = query.lower().strip()
    has_financial = _has_data_vocabulary(q)
    if has_financial:
        return False
    # If the query references a person/investor name, it's NOT general
    if _has_name_reference(q):
        return False
    # Check for investor IDs
    if _INVESTOR_ID_PATTERN.search(q):
        return False
    is_general = any(p in q for p in _GENERAL_PATTERNS)
    return is_general


# MCP-forcing backstop for client_note: any sign the message wants DATA (a question
# or a request verb) sends it to MCP instead of acknowledging. One-directional — it
# can only push toward fetch_data, never cause a wrong acknowledgment — so being
# rule-based here is safe and matches the "bias toward MCP" requirement.
_DATA_REQUEST_SIGNALS = (
    "show", "list", "pull up", "get me", "fetch", "display", "give me",
    "what ", "which ", "how much", "how many", "who ", "where ", "when ",
    "value", "holdings", "portfolio", "transactions", "returns", "aum", "nav",
)


def _looks_like_data_request(query: str) -> bool:
    q = (query or "").lower().strip()
    return q.endswith("?") or any(s in q for s in _DATA_REQUEST_SIGNALS)


# NOTE: preference-statement detection is now handled by the LLM `route` field
# (route == "preference_statement") rather than keyword lists — see WealthEntities
# and the routing block in intent_enrichment().


# --- from_memory gating ---------------------------------------------------------
# The agent answers from session/conversation memory ONLY when the user is explicitly
# asking ABOUT the prior conversation or the data already shown. Everything else — even
# a follow-up that happens to lack a data keyword — is RE-FETCHED live. This is the
# anti-hallucination default: never let the model build a financial answer out of an
# earlier turn it half-remembers.
_RECALL_INTENT_PATTERNS = [
    # referring to the previous answer / the data just shown
    "you just", "you said", "you mentioned", "you listed", "you showed", "you told",
    "just told me", "just showed me", "said earlier", "mentioned earlier", "earlier you",
    "repeat that", "say that again", "repeat the", "what was that",
    "your last answer", "your previous answer", "the previous answer", "previous response",
    "last response", "above answer", "answer above",
    "summarize that", "summarise that", "summarize the above", "summarise the above",
    "summarize what", "summarise what", "in summary", "to summarize", "to summarise",
    "recap", "tldr", "tl;dr",
    "from that list", "from the list above", "from the above", "of those", "of these",
    "among those", "among these", "from those", "out of those", "out of these",
    "in that list", "the list you", "that you just",
    # recall ABOUT the user (long-term memory)
    "what do you know about me", "what do you remember", "remind me", "recall",
    "based on what you know", "my preferences", "my main financial concern",
    "do you remember",
]


def _is_recall_intent(query: str) -> bool:
    """True when the user is asking about the prior conversation / shown data / their own
    profile — the only cases allowed to answer from memory instead of a live fetch."""
    q = query.lower()
    return any(p in q for p in _RECALL_INTENT_PATTERNS)


# A prior assistant turn that failed (empty / error / refusal) must NEVER be reused as
# though it were data — otherwise a one-off platform miss gets frozen into the session.
_NONANSWER_PAT = re.compile(
    r"could not (be )?retriev|couldn't (find|retriev)|unable to (find|retriev|provide)|"
    r"no (relevant |)data (found|available|was returned)|please (try again|rephrase)|"
    r"i'?m sorry|i can'?t process|i cannot process|context resolution failed|"
    r"error fetching data",
    re.IGNORECASE,
)


def _looks_like_nonanswer(text: str) -> bool:
    return bool(_NONANSWER_PAT.search(text or ""))


async def _fetch_personal_context(query: str, state: AgentState, investor_name: str | None = None, client_recall_only: bool = False) -> str:
    """Fetch memories from long-term memory, on demand.

    Pass investor_name to scope retrieval to a specific client (recall_client).
    Leave None for advisor-scoped queries (suitability, preference_match, recall_personal).
    Returns "" on any failure.
    """
    try:
        from .memory import memory_manager
        ctx = await memory_manager.retrieve(query, state["user_id"], investor_name=investor_name, client_recall_only=client_recall_only)
        return ctx if (ctx and ctx.strip()) else ""
    except Exception:
        return ""


_PREF_QUERY_AUGMENT_PROMPT = (
    "You refine a single data-retrieval query for a STATELESS wealth-management platform "
    "(it answers only the query string you give it — it never sees the user's preferences).\n\n"
    "You are given the QUERY and the user's STANDING PREFERENCES. Your ONLY job: if a preference "
    "specifies which FIELDS / METRICS / FILTERS to include for this kind of data, rewrite the "
    "query to EXPLICITLY request them, so the platform returns that data.\n\n"
    "STRICT rules:\n"
    "- Change ONLY what data is fetched. Never change the subject, client name, timeframe, or the "
    "list/count/aggregate intent.\n"
    "- Only add fields/filters that a preference actually states AND that fit this query's data type "
    "(e.g. a holdings-field preference applies to a holdings query, not to an AUM-count query).\n"
    "- IGNORE pure presentation preferences (formatting, grouping, table vs prose, tone, 'no "
    "transactions', benchmarks for display) — those are applied when the answer is written, not here.\n"
    "- Never invent fields the preferences didn't mention. If NO preference is relevant to this "
    "query, return the query UNCHANGED.\n"
    "- KEEP IT SHORT — ONE plain sentence, ideally under ~18 words, the way a person would ask. "
    "If the input query is long or padded, SHORTEN it. Append the fields as a single brief clause "
    "('… with <field>, <field> and <field>'). Do NOT pad with 'list every holding and for each "
    "one provide…', 'List all of …', 'include, for each one,', or trailing notes like 'give the "
    "exact values'; do NOT repeat the noun ('the transaction type, the transaction amount' → "
    "'type, amount'). Add ONLY the fields the preference/query explicitly asks for — never extra "
    "ones.\n"
    "- Use PLAIN-ENGLISH field names only. NEVER put a technical/snake_case identifier or "
    "parentheses in the query (e.g. write 'revenue by product type', NOT "
    "'(revenue_by_product_type)') — the platform parses plain English and breaks on raw column ids.\n"
    "- Output ONLY the final query string — no quotes, no explanation.\n\n"
    "Examples (note how short the output stays):\n"
    "  QUERY: 'show Ram Krishnan's holdings' ; PREFERENCE: 'for holdings show cost, current value and "
    "unrealized P&L' -> 'Show Ram Krishnan's holdings with cost, current value and unrealized P&L.'\n"
    "  QUERY: 'show Ram Krishnan's recent transactions' ; PREFERENCE: 'for transactions include type, "
    "amount and date' -> 'Show Ram Krishnan's recent transactions with type, amount and date.'"
)


# Which query data_types are eligible for preference-driven FIELD/FILTER augmentation, and the
# anchor tokens that must appear in the stored preferences for a preference to be relevant to that
# kind of data. This is the deterministic INTENT/SCOPE GATE: a preference only reshapes the MCP
# query when (a) the query is for a list/record type that can carry extra fields AND (b) the user
# actually stored a preference about that data type. Scalar single-value lookups (nav, aum,
# performance, risk) are NOT here — they are never field-augmented, so e.g. "NAV of Fund X" always
# reaches the platform clean. Missing/unknown data_type also falls through to "no augmentation",
# so normal queries are never touched unless a matching preference exists.
_PREF_DOMAIN_ANCHORS: dict[str, tuple[str, ...]] = {
    "holdings": ("holding",),
    "portfolio": ("holding", "portfolio", "allocation"),
    "transactions": ("transaction",),
    "profile": ("profile", "kyc", "contact", "pan", "email", "phone", "nominee", "aadhaar"),
}


def preference_augmentation_applies(data_type: str | None, preferences: str | None) -> bool:
    """Deterministic gate: should stored preferences reshape the MCP query for THIS query?

    True only when the query's extracted data_type belongs to a field-eligible domain AND the
    stored preference summary actually mentions that domain. No LLM call — pure string match on
    values already extracted upstream. Returns False for scalar lookups (nav/aum/performance),
    unknown data_type, or empty preferences, so simple/normal queries reach MCP un-enriched.
    """
    dt = (data_type or "").strip().lower()
    summary = (preferences or "").strip().lower()
    if not dt or not summary:
        return False
    for domain, anchors in _PREF_DOMAIN_ANCHORS.items():
        if domain in dt or dt in domain:
            return any(a in summary for a in anchors)
    return False


async def augment_query_with_preferences(query: str, preferences: str) -> str:
    """Fold any FETCH-relevant standing preferences into a data query.

    Dedicated single-purpose step: unlike the main enrichment LLM (which resolves
    references and tends to echo a self-contained query unchanged), this one's sole
    job is to add preferred fields/filters so the stateless platform returns the data
    the preference needs. Returns the original query unchanged on any failure or when
    no preference is relevant. Generic — it reads whatever preferences exist, nothing
    is hardcoded per preference.
    """
    query = (query or "").strip()
    preferences = (preferences or "").strip()
    if not query or not preferences:
        return query
    try:
        llm = make_llm("fast", temperature=0.0, max_tokens=200)
        human = f"QUERY:\n{query}\n\nSTANDING PREFERENCES:\n{preferences}"
        # NOTE: the platform field menu (schema_digest) is deliberately NOT given to the
        # augmenter. It was being treated as a checklist — the LLM pulled columns off it
        # (e.g. "Weight in Portfolio (%)") and appended them even when NO preference asked
        # for them. The augmenter must add ONLY fields a preference explicitly names; the
        # answer then presents whatever MCP returns (see _HF_TEMPLATES_ENABLED=False).
        resp = await llm.ainvoke([
            SystemMessage(content=_PREF_QUERY_AUGMENT_PROMPT),
            HumanMessage(content=human),
        ])
        out = (resp.content or "").strip().strip('"').strip()
        return out or query
    except Exception as e:
        logger.warning("[enrich] preference query augmentation failed: %s", e)
        return query


async def _build_data_query_from_suitability(llm, query: str) -> Optional[str]:
    """Rewrite a suitability question into a clean, MCP-friendly factual data query.

    Strips personal-judgment framing ('for me', 'suit my', 'a good fit') and keeps
    only the product/asset/category the platform should look up. Returns None on failure.
    """
    prompt = (
        "Rewrite the user's question into a SHORT factual data query for a financial "
        "data platform. Remove ALL personal-judgment language ('for me', 'suit me', "
        "'a good fit', 'should I'). Keep only the product/asset/fund/category being "
        "asked about, as a plain lookup. Output ONLY the rewritten query.\n"
        "Example: 'Is a volatile small-cap fund a good fit for me?' -> "
        "'small-cap mutual fund details and risk characteristics'."
    )
    try:
        resp = await llm.ainvoke([
            SystemMessage(content=prompt),
            HumanMessage(content=query),
        ])
        out = (resp.content or "").strip().strip('"')
        return out or None
    except Exception:
        return None



# --- Follow-up / context-continuity helpers --------------------------------------
# The (context-aware) LLM owns the follow-up decision (message_kind + standalone_query);
# the rewrite is then validated by semantic guards below (_is_safe_followup_rewrite +
# _has_prior_data_question + _is_actionable_query), so no brittle keyword cue list is needed.

# Closed grammatical set of third-person personal pronouns. Unlike an open-ended phrase
# list, this is complete: a message that refers to a person ONLY by a pronoun (with no
# client name extracted) is, by definition, anaphoric — its referent lives in a PRIOR
# turn and it cannot be answered statelessly. We use this only to FORCE the follow-up
# resolver to run when the LLM under-labels such a message as plain "data"; the rewrite
# is still validated by the same downstream guards, so it can never cause a bad fetch.
_PERSONAL_PRONOUNS = re.compile(
    r"\b(he|she|him|her|his|hers|they|them|their|theirs)\b", re.IGNORECASE
)

# A PLURAL / collective PERSON-group noun that a possessive pronoun in the SAME query binds to.
# "show clients and THEIR aum" — "their" refers to "clients", which is named right here, so the
# pronoun is NOT anaphoric (it does not point at a prior turn / an unnamed person). Restricted to
# person-group nouns only: NOT holdings/funds/portfolios (those are the OBJECTS a pronoun
# possesses, e.g. "her holdings" still needs the client resolved), so cross-turn follow-ups like
# "compare her holdings with Martin's" are left to resolve normally.
_GROUP_ANTECEDENT_RE = re.compile(
    r"\b(clients|investors|customers|families|members|people|households|everyone)\b"
    r"|\b(each|every|all|both|several|multiple)\s+(client|investor|customer|family|member)s?\b",
    re.IGNORECASE,
)


def _has_group_antecedent(query: str) -> bool:
    """True when the query itself contains a plural/collective person-group noun that a possessive
    pronoun in the same sentence binds to — so the pronoun is bound in-sentence, not a reference to
    a prior turn / an unnamed person."""
    return bool(_GROUP_ANTECEDENT_RE.search(query or ""))


def _has_unresolved_reference(query: str, messages_history: list | None) -> bool:
    """True when the message contains a personal pronoun AND there is prior conversation it
    can lean on. We trigger even when an investor_name WAS extracted, because a query can name
    one client and refer to ANOTHER by pronoun (e.g. "compare her holdings with Martin's") —
    the pronoun still needs context resolution, which the single investor_name can't provide.
    The prior-context requirement avoids firing on a self-contained first-turn query whose
    pronoun refers to a name in the same message. Downstream guards validate the rewrite."""
    if not messages_history or len(messages_history) <= 1:
        return False
    # A possessive bound to a plural person-group named in the SAME query ("clients … their")
    # is not anaphoric — don't force a follow-up resolution / clarification on it.
    if _has_group_antecedent(query):
        return False
    return bool(_PERSONAL_PRONOUNS.search(query or ""))


# A STRONG referential pronoun that clearly points at an entity from a previous turn (used to
# resolve cross-turn follow-ups even when the classifier mislabels the message as 'data').
_STRONG_REFERENCE = re.compile(
    r"\b(their|theirs|his|her|hers|them|they|those|these|the same|the ones|"
    r"that (list|one|client|fund|family|portfolio))\b", re.IGNORECASE)


def _has_strong_reference(query: str) -> bool:
    return bool(_STRONG_REFERENCE.search(query or ""))


# A follow-up that SELECTS / FILTERS / RANKS over the rows JUST shown ("which of those are
# large cap mutual funds", "of those, which have the highest value", "rank those by current
# value", "which of the above..."). These operate on the prior answer's list, not on fresh
# platform data — sending them to the stateless platform is meaningless ("those" has no
# referent there) and returns an unrelated semantic match. So they're answered in-session.
_SELECT_FROM_PRIOR_LIST = re.compile(
    r"\b("
    r"which\s+(?:of\s+)?(?:those|these|them|the\s+above|the\s+ones?)"   # which of those / the above / the ones
    r"|which\s+ones?\b"                                                  # which ones
    r"|(?:of|from|among|out\s+of)\s+(?:those|these|them|the\s+above)"    # of those / from these / among them
    r"|(?:rank|sort|order|arrange)\s+(?:those|these|them)"              # rank/sort those
    r"|filter\s+(?:those|these|them)"                                    # filter those
    r")\b",
    re.IGNORECASE,
)


def _is_select_from_prior_list(query: str) -> bool:
    """True for a follow-up that selects/filters/ranks over the rows just shown (e.g.
    "which of those are large cap", "rank those by value"). Answered by operating on the prior
    answer in-session — NOT by re-querying the stateless platform, which has no notion of
    "those" and would return an unrelated match."""
    return bool(_SELECT_FROM_PRIOR_LIST.search(query or ""))


def _has_prior_data_question(messages: list | None) -> bool:
    """True if an earlier turn is a real anchor for a follow-up — either a substantive
    (non-greeting) user question, or an assistant turn that actually returned data."""
    if not messages:
        return False
    for m in messages[:-1]:  # exclude the current message
        content = getattr(m, "content", None)
        if not content:
            continue
        kind = getattr(m, "type", None)
        if kind == "human":
            low = content.lower()
            # A non-greeting user turn with some substance is a valid anchor. We can't
            # rely on a fixed keyword list (e.g. "How many clients invested in AIF?" has
            # none), so use "not smalltalk + has substance" instead.
            if not _is_general_query(low) and len(low.split()) >= 3:
                return True
        elif kind == "ai":
            # A prior assistant turn that actually answered (not a refusal/greeting).
            if not _looks_like_nonanswer(content) and len(content) > 40:
                return True
    return False


def _has_prior_ai_answer(messages: list | None) -> bool:
    """True if the session holds at least one real assistant answer (not a refusal,
    greeting, or other non-answer) that a recall question can be answered from.

    Used as the generic recall fallback for non-investor-scoped recalls (aggregate /
    list replies), which _conversation_has_data can't satisfy because it requires an
    investor_name."""
    if not messages:
        return False
    for m in messages[:-1]:  # exclude the current (recall) message
        content = getattr(m, "content", None)
        if (getattr(m, "type", None) == "ai" and content
                and not _looks_like_nonanswer(content) and len(content) > 40):
            return True
    return False


# Client-fact language: words that assert a preference, trait, or goal ABOUT someone.
# Used to re-route a "store his preference: ..." instruction (which the LLM may label as an
# advisor directive) to a client note when a client is already resolved — so the fact attaches
# to the client, not the advisor. Closed, meaning-bearing set, not an open phrase list.
_CLIENT_FACT_RE = re.compile(
    r"\b(prefer|prefers|preference|preferences|prefers?|want|wants|only|avoid|avoids|"
    r"likes?|dislikes?|invests?|holds?|risk[- ]averse|conservative|aggressive|moderate|"
    r"goal|goals|objective|nearing retirement|retiring)\b",
    re.IGNORECASE,
)


def _is_client_fact_statement(query: str) -> bool:
    """True if the statement asserts a preference/trait/goal (client-fact language). Combined
    with a resolved client, this re-routes a 'store his preference: debt only' instruction to a
    client note so EVERY extracted fact attaches to the client instead of defaulting to advisor."""
    return bool(_CLIENT_FACT_RE.search(query or ""))


def _derive_route(intent_kind: str | None, investor_name: str | None) -> str:
    """Map the speech act (intent_kind) + subject (client named or not) onto a concrete route.

    Decomposes the routing decision: the LLM classifies WHAT the user is doing (intent_kind),
    and this deterministically combines it with WHO it's about (investor_name) to pick the route.
    Returns "" for an unknown/missing intent_kind so the caller can fall back to the LLM route."""
    ik = (intent_kind or "").strip().lower()
    has_client = bool(investor_name)
    if ik == "request_data":
        return "fetch_data"
    if ik == "request_fit":
        return "preference_match" if has_client else "suitability"
    if ik == "instruct":
        return "preference_statement"
    if ik == "inform":
        return "client_note" if has_client else "preference_statement"
    if ik == "recall":
        return "recall_client" if has_client else "recall_personal"
    if ik in ("concept", "smalltalk"):
        return "general_knowledge"
    return ""


def _is_safe_followup_rewrite(standalone: str, original: str, messages: list | None) -> bool:
    """Guard the LLM's follow-up rewrite: reject empties, runaway expansions, and any
    rewrite that injects a specific client name that never appeared in the conversation
    (the model hallucinating a subject)."""
    s = (standalone or "").strip()
    if not s:
        return False
    if len(s) > len(original) * 8 + 240:
        return False
    convo = " ".join(
        (getattr(m, "content", "") or "") for m in (messages or [])
    ).lower()
    s_low = s.lower()
    for name in _known_investors():
        nl = name.lower()
        if nl in s_low and nl not in convo:
            return False
    return True


# Canonical data-intent buckets. Used to detect when a follow-up rewrite silently swaps WHAT is
# being asked (e.g. "investment details" -> "total AUM") — conversation-memory bleed from a prior
# turn, NOT legitimate reference resolution. `_is_safe_followup_rewrite` guards only the client
# NAME; this is the SECOND guard, on the METRIC. "value/worth" phrasings deliberately map to BOTH
# the holdings and aum buckets so an intended "portfolio value -> total AUM" rewrite still overlaps
# and is allowed through (see the ENRICHMENT_SYSTEM_PROMPT rule for value/worth questions).
_METRIC_BUCKETS: dict[str, tuple[str, ...]] = {
    "aum": ("aum", "net worth", "total investment", "assets under management",
            "portfolio value", "value of", "worth", "how much is"),
    "nav": ("nav",),
    "performance": ("xirr", "cagr", "returns", "performance", "annualized"),
    # "investment detail" (singular) is a substring of both "investment detail" and
    # "investment details", so it matches either form — plural-only "investment details"
    # let "provide her investment detail" slip through with an empty bucket, defeating Layer 1.
    "holdings": ("investment detail", "investments", "holdings", "portfolio",
                 "instruments", "positions", "portfolio value"),
    "transactions": ("transactions", "purchases", "redemptions", "buys", "sells"),
    "profile": ("profile", "kyc", "pan", "nominee", "contact"),
}


def _metric_buckets(text: str) -> set[str]:
    t = (text or "").lower()
    return {b for b, terms in _METRIC_BUCKETS.items() if any(x in t for x in terms)}


def _rewrite_changes_metric(query: str, rewrite: str) -> bool:
    """True when the CURRENT message names its OWN data intent and the rewrite replaces it with a
    DIFFERENT one that shares nothing with it — i.e. the metric was inherited from a prior turn
    (in-session memory bleed), not resolved from this message. Fails safe: if the message carries
    no data noun of its own (a genuine implied-subject follow-up like "and last quarter?"), returns
    False so the intended 'carry the prior metric' behaviour is preserved."""
    q_b = _metric_buckets(query)
    if not q_b:                       # message has no data intent of its own -> nothing to protect
        return False
    r_b = _metric_buckets(rewrite)
    return bool(r_b) and not (q_b & r_b)


# Markers of a clarifying QUESTION the assistant asked (our own phrasings + the generic
# LLM-built ones). Used to recognise that the user's next turn is ANSWERING us, so it must
# be merged back into the original question before fetching.
_CLARIFY_MARKERS = re.compile(
    r"did you mean|could you clarify|which one did you mean|there are multiple|"
    r"i couldn't find|which client|which investor|as of which|could you tell me which|"
    r"what data (do|would)|which (metric|list|fund|family)",
    re.IGNORECASE,
)


def _prev_turn_was_clarification(messages: list | None) -> bool:
    """True if the most recent assistant turn was a clarifying question (so the current
    user message is answering it and needs merging into the original question)."""
    if not messages:
        return False
    for m in reversed(messages[:-1]):  # most recent first, excluding current
        if getattr(m, "type", None) != "ai" or not getattr(m, "content", None):
            continue
        text = m.content.strip()
        if _CLARIFY_MARKERS.search(text):
            return True
        # Generic fallback for LLM-built clarifications: a short question that doesn't look
        # like a data answer (no currency/large numbers, which a real answer would carry).
        return (
            text.endswith("?")
            and len(text) < 200
            and not re.search(r"[₹$]|\d{3,}", text)
        )
    return False


def _is_actionable_query(q: str) -> bool:
    """True if the (rewritten) text reads like a real data question/command rather than a
    bare confirmation like "yes i meant AUM" — i.e. it starts with an interrogative or
    imperative. Stops an un-merged answer from being sent to the platform."""
    s = (q or "").strip()
    # Strip a polite/auxiliary lead-in ("can you", "could you please", "would you", "please
    # kindly") so the REAL verb is what's evaluated. The follow-up resolver frequently phrases a
    # perfectly valid data request politely ("Can you present the full ranking table?") — without
    # this it was judged non-actionable and the turn fell back to a spurious clarification.
    s = re.sub(r"^\s*(can|could|would|will|may|please|kindly)\s+(you\s+)?(please\s+|kindly\s+)?",
               "", s, flags=re.IGNORECASE)
    return bool(re.match(
        r"\s*(what|which|how|who|when|where|whose|list|show|give|provide|tell|compare|"
        r"find|get|display|name|count|calculate|fetch|pull|present|rank|sort|order|"
        r"summari[sz]e|break|group|export|return|sum|average|total)\b",
        s, re.IGNORECASE,
    ))


FOLLOWUP_RESOLVE_PROMPT = """Rewrite the user's latest message into ONE self-contained question for a STATELESS data platform that cannot see the conversation.

Rules:
- Use the conversation to resolve what the latest message refers to (pronouns, "those"/"them"/"that list", or a short answer to a question the assistant just asked).
- PRIORITY — the CURRENT message's OWN data intent wins: ALWAYS resolve the SUBJECT (pronouns / who) from context, but only inherit a METRIC or TIMEFRAME from a prior turn when the current message names NONE of its own (e.g. "and last quarter?"). If the current message already says what it wants (investment details, holdings, transactions, AUM, NAV, performance, profile), KEEP THAT — never let the prior turn's metric override it (e.g. after "provide his aum", "provide his investment details" -> "Provide <Name>'s investment details", NOT their AUM).
- If the assistant just asked a clarifying question, APPLY the user's answer to the ORIGINAL question and output that full question. Example: original "what is the AUE across all my clients" + assistant "Did you mean AUM?" + user "yes i meant AUM" -> "What is the AUM across all my clients?"
- Preserve the original ask type (list vs count vs single value) and any filters/subjects.
- Use EXACT client/family/fund names from the conversation; never invent one.
- Output ONLY the rewritten question, nothing else. If the latest message is already self-contained, output it unchanged."""


def _last_n_exchanges(messages: list | None, n: int = 2) -> list:
    """The last `n` preceding exchanges (each a USER question and the ASSISTANT answer that
    followed it), flattened in chronological order, excluding the current (last) message.
    In-session memory is intentionally limited to the last `n` Q&A pairs, so a follow-up resolves
    against the recent turns WITHOUT feeding the whole transcript — which lets us pass each in FULL
    (no truncation) while keeping the prompt bounded. Returns [] when there is no prior turn."""
    if not messages or len(messages) <= 1:
        return []
    prior = [m for m in messages[:-1]  # exclude the current message
             if getattr(m, "type", None) in ("human", "ai") and getattr(m, "content", None)]
    # Drop raw MCP data dumps: when two AI messages are consecutive, mcp_fetch added the raw one
    # and generate added the formatted one — keep only the later (formatted) message.
    deduped = [m for i, m in enumerate(prior)
               if not (m.type == "ai" and i + 1 < len(prior) and prior[i + 1].type == "ai")]
    if not deduped:
        return []
    # Start at the n-th-from-last USER question so we include that question and everything after
    # it (its answer plus any later turns), preserving chronological order.
    human_idx = [i for i, m in enumerate(deduped) if m.type == "human"]
    if not human_idx:
        return deduped[-n:]  # no prior user turns — fall back to the last n messages
    start = human_idx[-n] if len(human_idx) >= n else 0
    return deduped[start:]


def _strip_answer_tables(text: str) -> str:
    """Layer 4 (answer-column bleed guard): keep a prior ANSWER's prose for reference
    resolution, but drop any rendered markdown/tab-delimited table. Its column headers
    (e.g. "Investment Name | Type | Quantity | Current Value | As On Date") were being copied
    by the extractor into the NEXT fetch query as "fields to include" — padding an unrelated
    follow-up with the previous answer's columns. Reference resolution only needs the prose +
    entity names, not the table, so table rows are removed and everything else is kept verbatim.

    NOTE: this only sanitises the resolver's background context (see _format_exchange). It does
    NOT touch the query being built, so ENRICHMENT_SYSTEM_PROMPT / preference-driven columns are
    unaffected; and generate builds its own message context, so answers are unaffected."""
    kept: list[str] = []
    for ln in (text or "").splitlines():
        st = ln.strip()
        if not st:
            continue
        # A rendered table row: markdown pipes, a "---|---" separator row, or tab-delimited cols.
        if st.count("|") >= 2 or st.count("\t") >= 2 or re.match(r"^[\s|:\-]+$", st):
            continue
        kept.append(st)
    return "\n".join(kept)


def _format_exchange(messages: list | None, prior_enriched: str = "",
                     current_query: str = "") -> str:
    """Render the prior exchange for the resolver: the raw user question, its RESOLVED
    (MCP-bound) form when available and different, then the assistant answer. `prior_enriched`
    is last turn's parent enriched query (post scope-lock) — giving the resolver an already
    self-contained antecedent. Returns "" when there's no prior exchange, which also keeps this
    OFF for Quick Facts (history is blanked upstream, so _last_n_exchanges is empty).

    Layer 1 (metric-bleed guard): the RESOLVED (metric-laden) antecedent line is exposed ONLY
    when the CURRENT message is elliptical — i.e. it names NO metric of its own. When the current
    message already states its own metric (e.g. "provide his investment details" -> holdings),
    showing the prior turn's fully-enriched query (e.g. "...total AUM across all holdings...")
    just hands the model a copy-template it echoes verbatim, overriding the current metric. The
    raw prior User/Assistant lines are still emitted, so the pronoun/subject ("his" -> the client)
    still resolves — only the copyable resolved query is withheld."""
    pair = _last_n_exchanges(messages, 2)
    if not pair:
        return ""
    pe = (prior_enriched or "").strip()
    # Withhold the resolved antecedent when the current message carries its own metric intent.
    if pe and current_query and _metric_buckets(current_query):
        pe = ""
    # `prior_enriched` is only the MOST RECENT prior turn's resolved query, so attach the
    # "User (resolved):" hint to the last user question only — never to older ones.
    last_user_i = max((i for i, m in enumerate(pair)
                       if getattr(m, "type", None) == "human"), default=-1)
    lines: list[str] = []
    for i, m in enumerate(pair):
        role = "User" if getattr(m, "type", None) == "human" else "Assistant"
        # Layer 4: strip the prior ANSWER's rendered table (keep its prose) so the extractor
        # can't copy the answer's column headers into the next fetch query. User turns are
        # passed through verbatim — only the Assistant answer is sanitised.
        content = m.content if role == "User" else _strip_answer_tables(m.content)
        lines.append(f"{role}: {content}")
        # After the most recent prior USER question, add its resolved form (skip if identical).
        if role == "User" and i == last_user_i and pe and pe.lower() != (m.content or "").strip().lower():
            lines.append(f"User (resolved): {pe}")
    return "\n".join(lines)


async def _resolve_followup_query(llm: ClaudeChat, messages: list, query: str,
                                  prior_enriched: str = "") -> str:
    """LLM rewrite of a follow-up / clarification-answer into a standalone question, using
    the prior exchange. One fast-model call, made only on follow-up turns."""
    convo = _format_exchange(messages, prior_enriched, query)
    resp = await llm.ainvoke([
        SystemMessage(content=FOLLOWUP_RESOLVE_PROMPT),
        HumanMessage(content=f"Conversation:\n{convo}\n\nLatest user message: {query}\n\nRewrite:"),
    ])
    return resp.content.strip()


def _from_memory_result(state: AgentState, entities_dict: dict, start: float) -> dict:
    elapsed_ms = round((time.perf_counter() - start) * 1000)
    return {
        "intent": "from_memory",
        "enriched_query": "",
        "enrichment_entities": entities_dict,
        "latency_breakdown": {
            **state.get("latency_breakdown", {}),
            "enrichment_ms": elapsed_ms,
        },
    }


async def intent_enrichment(state: AgentState) -> dict:
    """Classify intent, extract entities, check completeness, enrich or clarify.

    Routing outcomes:
    - general   → answered directly by generate_response (no MCP)
    - from_memory → answer from conversation history (no MCP)
    - clarification → ask user for missing info
    - fetch_data → enriched query ready for MCP
    """
    query = state["query"]
    messages_history = state.get("messages", [])
    # Keep the tenant context (roster + capability map) fresh — cheap no-op within the TTL,
    # so a client added on the platform shows up here without a redeploy.
    try:
        from ...context import cortex_context
        await cortex_context.ensure_fresh()
    except Exception:
        pass  # never let a platform hiccup break the turn — the cache/seed still serves
    # Quick Facts is a standalone-lookup mode: drop in-session conversation history for this
    # turn so no pronoun/follow-up/"answer from session" carries across turns. Every consumer
    # below reads this local, so blanking it makes the node behave as if there were no prior
    # turns — Client Insights / Deep Insight are untouched and keep full history.
    from ...modes import normalize_mode, QUICK_FACTS, CLIENT_INSIGHTS
    _norm_mode = normalize_mode(state.get("mode"))
    is_quick_facts = _norm_mode == QUICK_FACTS
    if config.quickfacts_standalone_only and is_quick_facts:
        messages_history = []
    memory_context = state.get("memory_context", "")
    # Request-level client fallback. In Client Insights the pinned client rides in `selected_client`
    # (not `investor_name`), so a pronoun-only query ("her total AUM") had no name to resolve to and
    # spuriously clarified even though a client was selected. Use `selected_client` as a fallback,
    # but ONLY in Client Insights and NOT for a family selection (M7 handles families). Deep Insight
    # and Quick Facts don't carry a selected_client, and the mode gate keeps them untouched anyway.
    fallback_investor = state.get("investor_name")
    # The frontend passes the logged-in user_id (e.g. "user1") as investor_name — a login ID, NOT a
    # real client. Treat that as "no client" so the Client Insights selected_client can take over
    # (otherwise "user1" is truthy and silently blocks the pinned-client fallback -> spurious "whose?").
    if (not fallback_investor or _is_login_id(fallback_investor)) and _norm_mode == CLIENT_INSIGHTS:
        _sel_client = (state.get("selected_client") or "").strip()
        if _sel_client and not re.search(r"\bfamil(y|ies)\b", _sel_client, re.IGNORECASE):
            fallback_investor = _sel_client
    start = time.perf_counter()

    # NOTE: preference-statement and personal-recall routing are now decided by the
    # LLM via the structured `route` field (see below) instead of brittle keyword
    # lists — this generalizes to any phrasing without extra latency.

    # --- Fast path: general / conversational query — skip LLM entirely ---
    if _is_general_query(query):
        elapsed_ms = round((time.perf_counter() - start) * 1000)
        return {
            "intent": "general",
            "enriched_query": "",
            "enrichment_entities": {},
            "latency_breakdown": {
                **state.get("latency_breakdown", {}),
                "enrichment_ms": elapsed_ms,
            },
        }

    llm = make_llm("fast", temperature=0.0, max_tokens=300)

    # --- Step 1: Extract entities ---
    # Include recent conversation context for resolving pronouns/references
    # In-session memory = ONLY the immediately preceding exchange (1 question + 1 answer),
    # passed in FULL (no truncation), PLUS that turn's resolved (MCP-bound) query so the
    # follow-up inherits a self-contained antecedent. Bounding to one exchange keeps the now-
    # untruncated answer from bloating the prompt. Empty for Quick Facts (history blanked above).
    conversation_context = _format_exchange(messages_history, state.get("last_enriched_query") or "", query)

    # Enrichment runs CONCURRENTLY with the memory-retrieval track (see parallel.py),
    # so on a session's first turn `memory_context` is still empty here. Pull the user's
    # preference SUMMARY directly — a deterministic point fetch (no vector search), cheap
    # enough inline — so standing preferences can shape the query we send to MCP (e.g.
    # which holding fields to fetch), not just how the answer is formatted later.
    if not memory_context:
        try:
            from .memory import memory_manager
            memory_context = await memory_manager.get_summary(state["user_id"]) or ""
        except Exception:
            memory_context = ""

    extraction_input = f"User query: {query}"
    if conversation_context:
        extraction_input += f"\n\nRecent conversation context:\n{conversation_context}"
    if memory_context:
        extraction_input += f"\n\nMemory context:\n{memory_context}"
    # NOTE: the platform field menu (schema_digest) is intentionally NOT injected here.
    # It was being read as a checklist — the extractor pulled columns off it (e.g.
    # "Weight in Portfolio (%)") into standalone_query even when nothing asked for them.
    # standalone_query should only resolve references and carry fields a preference
    # explicitly names; the answer presents whatever MCP returns.

    structured_llm = llm.with_structured_output(WealthEntities)
    entities = await structured_llm.ainvoke([
        SystemMessage(content=EXTRACTION_SYSTEM_PROMPT),
        HumanMessage(content=extraction_input),
    ])

    entities_dict = entities.model_dump()
    # The LLM's OWN data_type, captured before the keyword fallback below. The keyword
    # inference can false-positive (e.g. "risk" inside "risk-averse"), so acknowledge-route
    # gating uses this clean value rather than the inferred one.
    llm_data_type = entities_dict.get("data_type")
    # --- Metric-bleed correction (own_data_type authority) ------------------------------------
    # The message's OWN metric (own_data_type, classified context-free by the extractor) is
    # authoritative over any metric that bled in from the prior turn. When they disagree, the
    # extractor inherited the wrong metric (e.g. "provide her investment detail" -> data_type
    # "aum" copied from a prior AUM turn, but own_data_type "holdings"). Override the entity
    # here (Edit 2) and, below, discard the bled standalone rewrite (Edit 3). Robust to any
    # phrasing because it's a semantic enum, not a substring match. own_data_type == null (a
    # genuinely elliptical message like "and last quarter?") -> _metric_bled False -> no-op, so
    # legitimate inheritance is preserved.
    _own_dt = (entities_dict.get("own_data_type") or "").strip().lower()
    _metric_bled = bool(_own_dt and (entities_dict.get("data_type") or "").strip().lower() != _own_dt)
    if _metric_bled:
        entities_dict["data_type"] = _own_dt
    # A login / system identifier (e.g. "user2", "admin") is NOT a client name — the
    # frontend passes the logged-in user_id as investor_name, and it must never be
    # injected into the query (that produces "portfolio of user2" and breaks resolution).
    if entities_dict.get("investor_name") and _is_login_id(entities_dict["investor_name"]):
        entities_dict["investor_name"] = None
    # Request-level fallback (the client selected/defaulted in the UI). Only apply it to a
    # question that is actually ABOUT a client — never to a product/market/aggregate or a
    # multi-client list query, otherwise an unrelated lookup ("bonds with AA+ rating") gets
    # silently scoped to that client and answered "... for Ram Krishnan" (issue #4).
    _q_low = query.lower()
    if (not entities_dict.get("investor_name") and fallback_investor
            and not _is_login_id(fallback_investor)
            and not _query_needs_no_client(_q_low)
            and not _query_is_multi_investor(query)):
        entities_dict["investor_name"] = fallback_investor

    # Resolve the investor/client name to its canonical full form (e.g. "Ram" →
    # "Ram Krishnan", "Sita Devi" → "Sita Krishnan", "Rama Krishana" → "Ram Krishnan").
    # Sending the full canonical name makes the platform resolve far more reliably and
    # avoids its poisoned short-name cache keys. If the name is ambiguous (e.g. a bare
    # surname/family shared by multiple clients) we ask the user to clarify — unless the
    # question is explicitly about a whole family/group.
    # Skip resolution for investor IDs (INV-001 etc.) — they're valid as-is.
    name_clarify_candidates: list[str] | None = None
    raw_name = entities_dict.get("investor_name")
    if raw_name and not _INVESTOR_ID_PATTERN.match(raw_name):
        canonical, candidates = _resolve_client_name(raw_name)
        if canonical:
            entities_dict["investor_name"] = canonical
        elif candidates:
            q_low = query.lower()
            is_family_or_group = (
                "family" in q_low or "families" in q_low or _query_is_multi_investor(query)
            )
            if not is_family_or_group:
                name_clarify_candidates = candidates

    # Issue #4 (context bleed): a product/market/aggregate question must NEVER be silently scoped
    # to a client the CURRENT query doesn't reference. The extractor sometimes pulls the most
    # salient client out of the conversation/memory context (e.g. a prior turn about "Ram
    # Krishnan") onto a question that names no one ("details of bonds with AA+ rating"). When the
    # query needs no client (product/market) or is a multi-client question, drop an extracted name
    # UNLESS the query itself names a client or refers to one by pronoun (a real follow-up).
    _inv = (entities_dict.get("investor_name") or "").strip()
    if _inv and not _INVESTOR_ID_PATTERN.match(_inv):
        _ql = query.lower()
        _named_here = (
            _inv.lower() in _ql
            or (_inv.split()[0].lower() in _ql.split())
            or _has_name_reference(_ql)
        )
        _pron_here = bool(_PRONOUN_RE.search(_ql))
        if ((_query_needs_no_client(_ql) or _query_is_multi_investor(query))
                and not (_named_here or _pron_here)):
            entities_dict["investor_name"] = None
            raw_name = None

    # If entity extraction didn't set data_type, try to infer from query keywords
    # so that the cache check below is type-specific (not overly broad).
    if not entities_dict.get("data_type"):
        _query_data_keywords = {
            "portfolio": "portfolio", "holdings": "holdings", "profile": "profile",
            "transactions": "transactions", "performance": "performance",
            "returns": "performance", "aum": "aum", "nav": "nav",
            "risk": "risk", "allocation": "portfolio",
        }
        query_lower = query.lower()
        for kw, dtype in _query_data_keywords.items():
            if kw in query_lower:
                entities_dict["data_type"] = dtype
                break

    # --- Routing: derive from the speech act (intent_kind) + subject, fall back to LLM route ---
    # intent_kind is the primary signal — it classifies WHAT the user is doing (independent of
    # finance keywords), and _derive_route combines it with WHO (investor_name). This avoids the
    # old topic-word bias that misrouted directives ("always include the expense ratio") and
    # concept questions ("what is an index fund") into fetch_data.
    _derived = _derive_route(entities_dict.get("intent_kind"), entities_dict.get("investor_name"))
    route = _derived or (entities_dict.get("route") or "fetch_data").strip()
    has_data_signal = bool(entities_dict.get("investor_name")) and bool(llm_data_type)

    # Speech-act short-circuits (ungated by has_data_signal — the speech act is authoritative):
    # concept/smalltalk -> general (never MCP); a standing directive / self preference -> acknowledge.
    if route == "general_knowledge":
        elapsed_ms = round((time.perf_counter() - start) * 1000)
        return {
            "intent": "general",
            "enriched_query": "",
            "enrichment_entities": entities_dict,
            "latency_breakdown": {**state.get("latency_breakdown", {}), "enrichment_ms": elapsed_ms},
        }
    # Bug-1 backstop: "Store his preference: debt only" reads as an instruction (instruct ->
    # preference_statement -> advisor), but it's actually a fact ABOUT a resolved client. When a
    # client is resolved AND the text carries client-fact language, treat it as a client note so
    # EVERY extracted fact attaches to the client — deterministic, no LLM coin-flip.
    if (route == "preference_statement"
            and entities_dict.get("investor_name")
            and _is_client_fact_statement(query)):
        entities_dict["_note_scope"] = "investor"
        elapsed_ms = round((time.perf_counter() - start) * 1000)
        return {
            "intent": "acknowledge",
            "enriched_query": "",
            "enrichment_entities": entities_dict,
            "latency_breakdown": {**state.get("latency_breakdown", {}), "enrichment_ms": elapsed_ms},
        }

    if route == "preference_statement":
        # Advisor's OWN preference/directive — store advisor-scoped (no client attribution).
        entities_dict["_note_scope"] = "advisor"
        elapsed_ms = round((time.perf_counter() - start) * 1000)
        return {
            "intent": "acknowledge",
            "enriched_query": "",
            "enrichment_entities": entities_dict,
            "latency_breakdown": {**state.get("latency_breakdown", {}), "enrichment_ms": elapsed_ms},
        }

    # recall_client: recall a NOTE the user recorded about a named client — ungated from
    # has_data_signal so attribute words like "risk"/"profile" don't divert to MCP.
    if route == "recall_client":
        recall_investor = entities_dict.get("investor_name")
        fresh_ctx = await _fetch_personal_context(query, state, investor_name=recall_investor, client_recall_only=True)
        if fresh_ctx and fresh_ctx.strip():
            entities_dict["_context_source"] = "recall_ltm"
            elapsed_ms = round((time.perf_counter() - start) * 1000)
            return {
                "intent": "from_memory",
                "memory_context": fresh_ctx,
                "enriched_query": "",
                "enrichment_entities": entities_dict,
                "latency_breakdown": {**state.get("latency_breakdown", {}), "enrichment_ms": elapsed_ms},
            }

    # client_note: RM stating a durable fact ABOUT a named client — ungated from
    # has_data_signal because declarative sentences contain financial words that set data_type.
    if route == "client_note":
        conf = entities_dict.get("confidence") or 0.0
        # NOTE: `_looks_like_data_request(query)` backstop temporarily disabled — it
        # false-positives on declarative notes containing bare data nouns (e.g.
        # "prefers quarterly portfolio reviews" tripped on "portfolio"). Re-enable
        # with a smarter (verb/?-only) check later.
        if conf >= 0.7:  # and not _looks_like_data_request(query):
            # Fact ABOUT a named client — store client-scoped (attribute to the resolved
            # investor, even for preference/persona facts). Carries the resolved name so the
            # store step can attach it even when the user referred to the client by pronoun.
            entities_dict["_note_scope"] = "investor"
            elapsed_ms = round((time.perf_counter() - start) * 1000)
            return {
                "intent": "acknowledge",
                "enriched_query": "",
                "enrichment_entities": entities_dict,
                "latency_breakdown": {**state.get("latency_breakdown", {}), "enrichment_ms": elapsed_ms},
            }

    # --- Context continuity: smalltalk, follow-ups & clarification answers ----------
    msg_kind = entities_dict.get("message_kind") or "data"
    q_low = query.lower()
    prev_clarify = _prev_turn_was_clarification(messages_history)

    if msg_kind == "smalltalk" and not prev_clarify:
        has_data_signal = (
            _has_data_vocabulary(q_low)
            or _INVESTOR_ID_PATTERN.search(query)
            or _has_name_reference(q_low)
        )
        if not has_data_signal:
            elapsed_ms = round((time.perf_counter() - start) * 1000)
            return {
                "intent": "general",
                "enriched_query": "",
                "enrichment_entities": entities_dict,
                "latency_breakdown": {**state.get("latency_breakdown", {}), "enrichment_ms": elapsed_ms},
            }

    # --- Always-on context resolution -----------------------------------------------------
    # The data platform is STATELESS, so any data turn that leans on the conversation must be
    # rewritten into a self-contained query BEFORE it is sent. The extractor now produces that
    # rewrite (standalone_query) for EVERY data request — resolving pronouns, referential phrases
    # ("the same fund", "that family", "those"), and implied/omitted subjects — not just the
    # regex-detectable pronoun cases the old gate covered. So we treat the turn as referential
    # when EITHER an explicit cue fires (clarification answer, followup label, strong/pronoun
    # reference) OR the extractor itself produced a rewrite that differs from the raw text (the
    # general case the regex gate missed). _is_safe_followup_rewrite still rejects any rewrite
    # that invents a client never seen in the session, so a bad rewrite degrades to a
    # clarification (explicit cue) or to the raw self-contained wording (extractor-only) — never
    # a wrong-client fetch.
    # Select-from-prior-list follow-up ("which of those are large cap mutual funds", "rank those
    # by current value") — a FILTER/RANK over the rows just shown, NOT a fresh data request.
    # The stateless platform can't honour "those" (it returns an unrelated semantic match), so
    # answer from the prior turn's data: route to from_memory, where generate filters/ranks the
    # shown rows (and asks before pulling any attribute they don't carry). Gated on a real prior
    # answer existing — so it can't fire in Quick Facts (history blanked) or on a first turn.
    if _is_select_from_prior_list(query) and _has_prior_ai_answer(messages_history):
        entities_dict["_context_source"] = "recall_session"
        return _from_memory_result(state, entities_dict, start)

    _resolved = (entities_dict.get("standalone_query") or "").strip()
    _extractor_rewrote = (
        bool(_resolved)
        and _resolved.lower() != q_low
        and _is_safe_followup_rewrite(_resolved, query, messages_history)
        and not _rewrite_changes_metric(query, _resolved)   # 2nd guard: reject prior-turn metric bleed
    )
    _explicit_referential = (
        prev_clarify
        or (msg_kind == "followup" and not _has_group_antecedent(query))
        or (_has_strong_reference(query) and _has_prior_data_question(messages_history)
            and not _has_group_antecedent(query))
        or _has_unresolved_reference(query, messages_history)
    )
    # Issue #4 (context bleed in the rewrite): a product/market/aggregate query that names no
    # person and uses no pronoun is self-contained — it must NOT be run through follow-up
    # resolution, which would otherwise splice the most salient client from the conversation
    # into the rewritten query (e.g. "bonds with AA+ rating" -> "Ram Krishnan's bonds ..."). The
    # pronoun check keeps genuine clientless follow-ups ("what about their coupon rates?") working.
    _clientless = (
        _query_needs_no_client(q_low)
        and not _has_name_reference(q_low)
        and not _PRONOUN_RE.search(q_low)
    )
    # Edit 3 (own_data_type authority): when the message names its OWN metric that disagrees with
    # the inherited one (_metric_bled), skip the follow-up resolver entirely — both the extractor's
    # standalone AND the LLM re-resolver read the same bleeding context and would re-copy the wrong
    # metric. Falling through hands the turn to the deterministic pass-through + pronoun-substitution
    # path below, which resolves the SUBJECT (her -> the client) while keeping the message's own
    # metric ("her investment detail" -> "<client>'s investment detail", holdings not aum).
    if (_explicit_referential or _extractor_rewrote) and not _clientless and not _metric_bled:
        standalone = _resolved
        if not (standalone and _is_safe_followup_rewrite(standalone, query, messages_history)
                and not _rewrite_changes_metric(query, standalone)):
            standalone = (await _resolve_followup_query(
                llm, messages_history, query, state.get("last_enriched_query") or "")).strip()

        if (standalone and _has_prior_data_question(messages_history)
                and _is_safe_followup_rewrite(standalone, query, messages_history)
                and not _rewrite_changes_metric(query, standalone)   # 2nd guard on the fallback too
                and _is_actionable_query(standalone)):
            entities_dict["_context_source"] = "followup"
            elapsed_ms = round((time.perf_counter() - start) * 1000)
            return {
                "intent": "fetch_data",
                "enriched_query": standalone,
                "enrichment_entities": entities_dict,
                "latency_breakdown": {**state.get("latency_breakdown", {}), "enrichment_ms": elapsed_ms},
            }
        # Clarify ONLY when the message genuinely cannot stand alone — i.e. a bare unresolved
        # pronoun ("what about them?"), or the user is answering a clarifying question we just
        # asked. A turn that is merely flagged referential because history exists (the LLM
        # labelling it 'followup', or a strong-reference cue) but is otherwise SELF-CONTAINED must
        # NOT be forced into a clarification — that is the "works on refresh, clarifies mid-session"
        # bug. Those fall through to normal handling on their own wording (and reach the platform).
        # An unvalidatable extractor-only rewrite likewise falls through (already self-contained).
        _genuinely_anaphoric = prev_clarify or _has_unresolved_reference(query, messages_history)
        if _explicit_referential and _genuinely_anaphoric:
            clarification = (
                "Could you clarify what you'd like me to look up — for example, which client, "
                "metric, or list you're referring to?"
            )
            elapsed_ms = round((time.perf_counter() - start) * 1000)
            return {
                "intent": "clarification",
                "enriched_query": "",
                "enrichment_entities": entities_dict,
                "messages": [AIMessage(content=clarification)],
                "latency_breakdown": {**state.get("latency_breakdown", {}), "enrichment_ms": elapsed_ms},
            }

    if not has_data_signal:
        if route == "recall_personal":
            fresh_ctx = memory_context
            try:
                from .memory import memory_manager
                fetched = await memory_manager.retrieve(query, state["user_id"])
                if fetched and fetched.strip():
                    fresh_ctx = fetched
            except Exception:
                pass
            if fresh_ctx and fresh_ctx.strip():
                entities_dict["_context_source"] = "recall_ltm"
                elapsed_ms = round((time.perf_counter() - start) * 1000)
                return {
                    "intent": "from_memory",
                    "memory_context": fresh_ctx,
                    "enriched_query": "",
                    "enrichment_entities": entities_dict,
                    "latency_breakdown": {**state.get("latency_breakdown", {}), "enrichment_ms": elapsed_ms},
                }
        if route == "chitchat":
            elapsed_ms = round((time.perf_counter() - start) * 1000)
            return {
                "intent": "general",
                "enriched_query": "",
                "enrichment_entities": entities_dict,
                "latency_breakdown": {**state.get("latency_breakdown", {}), "enrichment_ms": elapsed_ms},
            }

    # --- Ambiguous / unresolved client name: ask the user which person they mean ---
    if name_clarify_candidates:
        listed = ", ".join(name_clarify_candidates[:-1]) + " or " + name_clarify_candidates[-1] \
            if len(name_clarify_candidates) > 1 else name_clarify_candidates[0]
        if len(name_clarify_candidates) > 1:
            # Several similar clients — genuine ambiguity.
            clarification = (
                f"There are multiple clients matching that name. Which one did you mean — {listed}?"
            )
        else:
            # A single close-but-different name (e.g. typed "Kavya Reddy", closest is "Kavya Patel").
            # Confirm rather than silently acting on a different client.
            clarification = (
                f"I couldn't find a client named “{raw_name}”. Did you mean {listed}? "
                f"If you meant someone else, please share the full name as it appears in agent."
            )
        elapsed_ms = round((time.perf_counter() - start) * 1000)
        return {
            "intent": "clarification",
            "enriched_query": "",
            "enrichment_entities": entities_dict,
            "messages": [AIMessage(content=clarification)],
            "latency_breakdown": {
                **state.get("latency_breakdown", {}),
                "enrichment_ms": elapsed_ms,
            },
        }

    # --- Data-aware metric check: clarify unknown metrics that are a near-miss of a real
    # one (Plan 1). e.g. "AEM across all clients" -> there is no AEM metric, but AUM is a
    # confident match, so ask "did you mean AUM?" instead of silently answering with the
    # wrong figure. Deterministic + LLM-free. Skipped for genuine recall (handled below).
    if not _is_recall_intent(query):
        metric_issue = find_metric_issue(query, entities_dict)
        if metric_issue:
            elapsed_ms = round((time.perf_counter() - start) * 1000)
            return {
                "intent": "clarification",
                "enriched_query": "",
                "enrichment_entities": entities_dict,
                "messages": [AIMessage(content=metric_issue.question)],
                "latency_breakdown": {
                    **state.get("latency_breakdown", {}),
                    "enrichment_ms": elapsed_ms,
                },
            }

    # --- Step 2: from_memory ONLY for genuine recall; otherwise ALWAYS fetch live ---
    # Default = live platform fetch. We answer from memory ONLY when the user is explicitly
    # asking about the prior conversation, the data already shown, or their own profile
    # (see _is_recall_intent). Any other question — even a keyword-less follow-up — is
    # re-fetched, so the agent can never fabricate a financial answer from a half-remembered
    # earlier turn, and mcp_fetch's retries get a fresh chance on non-deterministic misses.
    if _is_recall_intent(query):
        if memory_context and memory_context.strip():
            entities_dict["_context_source"] = "recall_ltm"
            return _from_memory_result(state, entities_dict, start)
        if _conversation_has_data(entities_dict, messages_history):
            entities_dict["_context_source"] = "recall_session"
            return _from_memory_result(state, entities_dict, start)
        # Generic recall fallback: the question explicitly asks about the prior turn
        # ("what did you just tell me", "summarize that", "repeat the list"), but the
        # answer is NOT investor-scoped — e.g. an aggregate/list reply like "10 clients
        # invested in AIF". _conversation_has_data requires an investor_name and so
        # bails for these. The stateless MCP can NEVER answer a recall question, so
        # routing it to fetch_data just errors; instead answer from the session
        # transcript (the from_memory path replays prior turns to the LLM) whenever the
        # session holds at least one real prior assistant answer.
        if _has_prior_ai_answer(messages_history):
            entities_dict["_context_source"] = "recall_session"
            return _from_memory_result(state, entities_dict, start)

    # --- Step 3: Check if critical entities are present ---
    missing = _identify_missing(entities_dict, query, messages_history)

    if missing:
        # Generate a natural clarification question
        clarification = await _build_clarification(llm, query, missing)
        elapsed_ms = round((time.perf_counter() - start) * 1000)

        return {
            "intent": "clarification",
            "enriched_query": "",
            "enrichment_entities": entities_dict,
            "messages": [AIMessage(content=clarification)],
            "latency_breakdown": {
                **state.get("latency_breakdown", {}),
                "enrichment_ms": elapsed_ms,
            },
        }

    # --- Step 3: Build enriched query ---
    # personal-context memory, fetched ONLY for suitability / preference_match so it
    # can be applied at generation. Direct data queries never fetch or use it.
    personal_ctx = ""

    # preference_match: a NAMED client's data judged against the user's preferences.
    # Build a clean client-scoped data query from the extracted entities (NOT the raw
    # "match what I prefer" sentence MCP can't resolve); compare in generate.
    if route == "preference_match" and entities_dict.get("investor_name"):
        dtype = entities_dict.get("data_type") or "holdings"
        # Avoid "portfolio portfolio" when the extracted data_type is already
        # "portfolio" — use the dtype alone in that case.
        scope = dtype if dtype == "portfolio" else f"portfolio {dtype}"
        enriched = f"{entities_dict['investor_name']}'s {scope}"
        entities_dict["_preference_filter"] = True
        personal_ctx = await _fetch_personal_context(query, state)
    # suitability: "is X a good fit for me" with no client. Rewrite to a clean factual
    # data query about the asset; judge fit from the user's profile in generate.
    elif route == "suitability":
        # Suitability rewrites intentionally drop the personal framing, so the
        # entity-preservation _is_safe_rewrite guard does not apply here. Use the
        # rewrite when produced, else fall back to the raw query.
        candidate = await _build_data_query_from_suitability(llm, query)
        enriched = candidate or query
        entities_dict["_suitability"] = True
        personal_ctx = await _fetch_personal_context(query, state)
    elif _is_macro_impact_query(query) and entities_dict.get("investor_name"):
        # Macro / current-events impact question ("Because of the ongoing war, how is Ram
        # Krishnan's portfolio affected?"). The platform resolver only understands data
        # lookups — sending the causal/macro framing makes it return an internal error
        # (0/1 platform call succeeded). So send it a CLEAN client-scoped portfolio query;
        # the real-world backdrop + impact analysis is layered on in generate (Deep Insight
        # web search + the LLM). Because the macro words ("war") are stripped from the platform
        # query, stash the ORIGINAL macro question on the entities so generate's web-search
        # trigger (which reads enriched_query) still sees the macro signal and fires.
        dtype = entities_dict.get("data_type") or "portfolio"
        scope = dtype if dtype in ("portfolio", "holdings") else f"portfolio {dtype}"
        enriched = (
            f"{entities_dict['investor_name']}'s {scope} with current value, "
            f"sector and asset class for each holding"
        )
        entities_dict["_macro_query"] = query
    elif _is_investment_type_split(query):
        # Aggregate breakdown — pin the clean phrasing so the LLM can't append a
        # spurious "that include <product>" filter from earlier conversation context.
        enriched = _CANONICAL_INVESTMENT_SPLIT_QUERY
    elif _should_substitute_name(query, entities_dict, raw_name):
        # Issue #1A: the user typed a partial/misspelled client name (e.g. "Rama Krishna")
        # that we canonicalised to a real client ("Ram Krishnan"), but the canonical name is
        # NOT in the query — so a plain pass-through would send the unresolvable name to the
        # platform and get back NOTHING (the "no holdings" false-negative). Substitute the
        # exact platform name in place, deterministically (no LLM rewrite, no scope drift).
        enriched = re.sub(re.escape(raw_name), entities_dict["investor_name"], query, flags=re.IGNORECASE)
    elif _needs_name_resolution(query, entities_dict):
        # The query refers to a client by pronoun or short/misspelled name and we need to
        # inject the canonical full name (already resolved into investor_name).
        if _PRONOUN_RE.search(query.lower()):
            # Pronoun with a KNOWN antecedent — substitute deterministically. We already
            # have the name, so don't gamble on an LLM rewrite (which intermittently failed
            # to inject it, sending raw "her"/"his" to the stateless platform).
            enriched = _substitute_pronoun(query, entities_dict["investor_name"])
        else:
            # Short/misspelled name — let the LLM rewrite, but GUARD it: if it changed scope,
            # added a filter, or dropped/added an entity, discard it and send original wording.
            candidate = await _build_enriched_query(llm, query, entities_dict)
            enriched = candidate if _is_safe_rewrite(query, candidate, entities_dict) else query
    else:
        # PASS-THROUGH by default. Sending the user's exact wording to the platform is more
        # accurate than rephrasing every query — free-form rewrites silently change scope,
        # inject filters from earlier turns, or shorten names. Aggregation/“total” needs are
        # handled deterministically downstream (mcp_fetch/generate), not by rewriting here.
        enriched = query
    elapsed_ms = round((time.perf_counter() - start) * 1000)

    result = {
        "intent": "fetch_data",
        "enriched_query": enriched,
        "enrichment_entities": entities_dict,
        "latency_breakdown": {
            **state.get("latency_breakdown", {}),
            "enrichment_ms": elapsed_ms,
        },
    }
    # Only carry memory into generation for suitability / preference_match; direct
    # data queries leave memory_context untouched.
    if personal_ctx:
        result["memory_context"] = personal_ctx
    return result


# Patterns that indicate the query is about ALL investors or a GROUP (no specific name needed)
_MULTI_INVESTOR_PATTERNS = [
    "all investors", "each investor", "every investor", "per investor",
    "how many investors", "count of investors", "list all investors",
    "which investors", "list investors", "show all investors",
    "top 5 investors", "top 10 investors", "top investors",
    "across all investors", "across all", "for all",
    "all clients", "each client", "every client",
    "how many", "average", "total across",
    "for investors whose", "for investors with",
    "show all", "list all", "all scenario",
    "all investment goals", "all holdings",
    "all transactions", "all rebalancing",
    "triggered by", "behind target",
    "who prefer", "who have",
    "for each investor", "per investor",
    "investors with", "investors whose",
    "grouped by", "per scenario type",
    "all goals", "all large transactions",
    "count of investors", "risk tolerance group",
    "loss-making", "sector overweight",
    "highest total investment",
    "average goal match", "average annual return",
    "show investor name,", "show investor name ",
    "investor name, their",
    "for all high risk", "for all moderate",
    "for all low risk",
    # RM / advisor persona — questions about "my clients" across the book
    "my clients", "my client's", "my clients'", "all my", "all of my",
    "each client", "each rm", "each relationship manager", "each banker",
    "relationship manager", "servicing banker", "each family", "all families",
    "across all clients", "across my", "by amc", "by sector", "by sbu",
    "by lob", "each lob", "highest aum", "top client", "top 5", "top 10",
    "which client", "which amc", "which banker", "which fund", "which sector",
    "how many clients", "how many families", "how many bonds",
]

# Regex for investor IDs (e.g. INV-001, INV001, INV-123)
_INVESTOR_ID_PATTERN = re.compile(r'\bINV[-]?\d{1,4}\b', re.IGNORECASE)


def _query_is_multi_investor(query: str) -> bool:
    """Return True if the query asks about multiple/all investors (no specific name needed)."""
    q = query.lower()
    if any(p in q for p in _MULTI_INVESTOR_PATTERNS):
        return True
    # Generic list/identify-clients phrasing the static phrase list misses — e.g. "show clients
    # and their aum", "show investors with their returns". _CLIENT_LISTING_RE matches a request
    # verb (list/show/which/who/name/identify/display/give me/...) followed by client(s)/investor(s),
    # which is unambiguously a multi-client question when no single name was extracted.
    if _CLIENT_LISTING_RE.search(query or ""):
        return True
    return False


# A query that asks to LIST / SHOW / IDENTIFY clients. Used only to recognise a multi-client
# question (see _query_is_multi_investor) when no single name was extracted.
_CLIENT_LISTING_RE = re.compile(
    r"\b(list|show|name|names|which|who|identify|display|give me|tell me|enumerate)\b"
    r"[^?.]*\b(client|clients|investor|investors)\b",
    re.IGNORECASE,
)


# The "split of investment types across all client portfolios" question is another
# aggregate where the LLM rewrite corrupts the query — when the session has a prior
# turn about one product type (e.g. AIF), enrichment "makes context explicit" and
# appends "that include AIF", which makes the platform return ONLY that one type. The
# clean phrasing reliably returns the full per-product breakdown, so we pin it.
_CANONICAL_INVESTMENT_SPLIT_QUERY = "What is the split of investment types across all client portfolios?"


def _is_investment_type_split(query: str) -> bool:
    """True for the aggregate 'split / breakdown of investment (product) types across
    all client portfolios' question — judged from the query text only."""
    q = query.lower()
    has_type = any(t in q for t in ("investment type", "product type", "asset class", "asset type"))
    has_breakdown = any(b in q for b in ("split", "breakdown", "break down", "distribution", "composition", "mix"))
    aggregate_scope = any(s in q for s in ("across", "all", "client", "portfolio", "everyone"))
    return has_type and has_breakdown and aggregate_scope


# A CAUSAL / "impact" framing — the phrasings the platform resolver cannot answer (it only
# does data lookups, not "because of <event> …" reasoning). Combined with a macro/event keyword
# (web_search.needs_web_search) this marks a question whose data part is just the client's
# portfolio, and whose real-world cause is handled by Deep Insight web search + the LLM.
_MACRO_CAUSAL_RE = re.compile(
    r"\b(because of|due to|owing to|on account of|as a result of|in light of|amid|amidst|"
    r"thanks to|impact of|affected by|exposed to|"
    r"how (is|are|will|would|might|has|have|does|do|did|could|can)\b.{0,80}?"
    r"\b(affect|affected|affecting|impact|impacted|impacting|hit|hurt|expos|fare|hold up))\b",
    re.IGNORECASE,
)


def _is_macro_impact_query(query: str) -> bool:
    """True for a 'how does <real-world event> affect this portfolio?' question — causal/impact
    framing AND a macro/current-events keyword. Such a query must NOT be sent verbatim to the
    platform (it returns an internal error); we send a clean portfolio lookup instead."""
    from ...web_search import needs_web_search
    q = query or ""
    return bool(_MACRO_CAUSAL_RE.search(q)) and needs_web_search(q)


# Product / asset keywords used by the divergence guard to detect an injected filter.
_GUARD_PRODUCTS = (
    "aif", "pms", "etf", "reit", "invit", "bond", "mutual fund", "direct equity",
    "equity", "debt", "fixed income", "hybrid", "sip", "nps", "ppf", "epf", "gold",
    "sgb", "fixed deposit", "fd",
)
# Scope words whose appearance/disappearance signals a single<->aggregate flip.
_GUARD_SCOPE = ("all ", "each ", "every ", "across all", "per client", "per investor")
_PRONOUN_RE = re.compile(r"\b(his|her|hers|their|theirs|he|she|him|them)\b", re.I)
# Possessive vs subject/object pronouns — drives how we splice in the resolved name.
_POSSESSIVE_PRONOUN_RE = re.compile(r"\b(his|her|hers|their|theirs|its)\b", re.I)
_SUBJOBJ_PRONOUN_RE = re.compile(r"\b(he|she|him|them|they)\b", re.I)


def _substitute_pronoun(query: str, name: str) -> str:
    """Deterministically replace the FIRST personal pronoun with the already-resolved client
    name (possessive -> "<name>'s", subject/object -> "<name>"). Used when the antecedent is
    known, so we never gamble on an LLM rewrite to inject a name we already have."""
    if _POSSESSIVE_PRONOUN_RE.search(query):
        return _POSSESSIVE_PRONOUN_RE.sub(f"{name}'s", query, count=1)
    if _SUBJOBJ_PRONOUN_RE.search(query):
        return _SUBJOBJ_PRONOUN_RE.sub(name, query, count=1)
    return query


def _should_substitute_name(query: str, entities: dict, raw_name: str | None) -> bool:
    """True when we can deterministically swap a misspelled/partial client name in the query
    for its resolved canonical name (issue #1A). Requires: a resolved canonical name that is
    NOT already in the query, and the exact original text (raw_name) IS in the query, and the
    query isn't an all/aggregate question. No LLM, no scope change — just a safe string swap."""
    canonical = (entities.get("investor_name") or "").strip()
    if not canonical or _INVESTOR_ID_PATTERN.match(canonical):
        return False
    if _query_is_multi_investor(query):
        return False
    q = query.lower()
    if canonical.lower() in q:
        return False  # canonical already present -> nothing to fix
    if not raw_name:
        return False
    rn = raw_name.strip().lower()
    return bool(rn) and rn != canonical.lower() and rn in q


def _needs_name_resolution(query: str, entities: dict) -> bool:
    """True only when the query references a client by pronoun or short/partial name and
    the canonical full name is NOT already present — the one case worth an LLM rewrite."""
    name = (entities.get("investor_name") or "").strip()
    if not name:
        return False
    q = query.lower()
    if name.lower() in q:               # full canonical name already there -> no rewrite
        return False
    if _PRONOUN_RE.search(q):            # "his portfolio" with a known antecedent
        return True
    first = name.split()[0].lower()      # "Ram" present, canonical is "Ram Krishnan"
    return bool(re.search(rf"\b{re.escape(first)}\b", q))


def _is_safe_rewrite(original: str, candidate: str, entities: dict) -> bool:
    """Divergence guard: accept a rewrite ONLY if it preserves the original meaning.

    Rejects the failure modes we actually saw: scope flips (single<->all), product/asset
    filters injected from earlier turns ("...that include AIF"), and runaway expansions.
    A rejected rewrite means we fall back to the user's original wording.
    """
    if not candidate or not candidate.strip():
        return False
    o, c = original.lower(), candidate.lower()
    name = (entities.get("investor_name") or "").lower()
    # The only thing the rewrite is allowed to ADD is the canonical name.
    if name and name not in c:
        return False
    # No new aggregate scope.
    if any(w in c for w in _GUARD_SCOPE) and not any(w in o for w in _GUARD_SCOPE):
        return False
    # No product/asset filter that wasn't in the original (context bleed).
    if any(p in c and p not in o for p in _GUARD_PRODUCTS):
        return False
    # No runaway expansion (injected constraints).
    if len(candidate) > len(original) * 2.2 + 40:
        return False
    return True


# Product / market / instrument / advisor / catalog subjects that do NOT need a
# specific client to be named (resolve_context answers them directly).
_NO_CLIENT_KEYWORDS = [
    # products & funds
    "fund", "scheme", "expense ratio", "nav", "amc", "isin", "benchmark",
    "fund manager", "mutual fund", "etf", "product catalog", "active schemes",
    "catalog",
    # market / instruments
    "price", "sector", "credit rating", "rating", "index", "nifty", "sensex",
    "bond", "stock", "share", "company", "reit", "invit", "ltd", "1-year return",
    "3-year return", "closing value",
    # advisor / org / aggregate
    "lob", "sbu", "banker", "relationship manager", " rm ", "revenue",
    "ultra hni", "hni", "nri", "sub-lob", "service office", "client-to-banker",
    "highest", "lowest", "top ", "most ", "across all", "by amc", "by sector",
    "by lob", "by sbu", "each rm", "each banker", "each relationship",
    "all clients", "all families", "which client", "which amc", "which banker",
    "which fund", "which sector", "which lob", "how many clients",
    "how many families", "how many bonds", "total number of",
]


def _query_needs_no_client(q: str) -> bool:
    """Return True for product/market/advisor/aggregate questions that don't
    reference a specific client (q is already lowercased)."""
    return any(kw in q for kw in _NO_CLIENT_KEYWORDS)


def _query_has_investor_id(query: str) -> str | None:
    """Extract investor ID from query if present (e.g. INV-001, INV002)."""
    match = _INVESTOR_ID_PATTERN.search(query)
    if match:
        # Normalize: ensure hyphen format (INV-001)
        raw = match.group(0).upper()
        if '-' not in raw:
            raw = raw[:3] + '-' + raw[3:]
        return raw
    return None


def _identify_missing(entities: dict, query: str, messages: list | None = None) -> list[str]:
    """Determine which critical entities are missing for a wealth query.

    Required: investor_name AND data_type (at minimum).
    If the query itself contains enough implicit info, we may not need both.
    If data_type is missing but the preceding clarification exchange had one, inherit it.

    Exceptions where investor_name is NOT required:
    - Multi-investor queries ("all investors with X", "which investors...", "how many...")
    - Queries with an investor ID (INV-001) — treated as the investor identifier
    - General knowledge queries ("what is ELSS", "lock-in period")
    """
    # resolve_context is a smart natural-language resolver — it can answer product,
    # market, advisor/RM, family and aggregate questions that do NOT name a client,
    # and it resolves names itself. So we attempt the fetch in almost every case and
    # only ask for clarification when the question truly hinges on an unnamed person
    # (a bare third-person pronoun with no antecedent and no other concrete subject).

    # Multi-client / aggregate questions never need a specific name
    if _query_is_multi_investor(query):
        return []

    # A specific client was identified — good to go
    if entities.get("investor_name"):
        return []

    # Investor ID (INV-001 etc.) acts as the identifier
    inv_id = _query_has_investor_id(query)
    if inv_id:
        entities["investor_name"] = inv_id
        return []

    q = query.lower()

    # Product / market / advisor / catalog lookups don't reference a client at all
    if _query_needs_no_client(q):
        return []

    # "my / I / our" persona questions (client or RM viewpoint) — attempt the fetch
    # rather than blocking; the resolver decides what it can return.
    if re.search(r"\b(my|i|me|we|our|us)\b", q):
        return []

    # Genuinely ambiguous follow-up: a pronoun referring to someone never named.
    # EXCEPTION: a possessive pronoun bound to a plural person-group named in the SAME query
    # ("show clients and THEIR aum", "list investors and THEIR returns") is not anaphoric — the
    # antecedent is right here, so it's a multi-client question, not a missing-name one.
    if re.search(r"\b(his|her|hers|their|theirs|he|she|him|them)\b", q):
        if _has_group_antecedent(q):
            return []
        inferred = _infer_data_type_from_history(entities, messages or [])
        if inferred:
            entities["data_type"] = inferred
            return []
        return ["investor_name"]

    # Default: attempt the fetch — better to try resolve_context than to block.
    return []


def _infer_data_type_from_history(entities: dict, messages: list) -> str | None:
    """Look at recent conversation to inherit data_type from a preceding clarification.

    When the bot asked "which Sharma?" in response to "what is Mr. Sharma's risk profile",
    and the user now replies with just "Neha Sharma's", we should carry forward 'risk'
    as the data_type from the original question.
    """
    if not messages:
        return None

    implicit_data_keywords = [
        "portfolio", "holdings", "profile", "transactions",
        "performance", "returns", "aum", "nav", "risk",
        "investment", "allocation", "funds", "schemes",
    ]

    # Walk backwards through recent human messages (skip the current one, which is last)
    human_messages = [
        m for m in messages
        if hasattr(m, "type") and m.type == "human"
        and hasattr(m, "content") and m.content
    ]

    # Check the previous human messages (up to last 3, excluding current)
    for msg in reversed(human_messages[:-1] if len(human_messages) > 1 else []):
        msg_lower = msg.content.lower()
        for kw in implicit_data_keywords:
            if kw in msg_lower:
                return kw
    return None


async def _build_clarification(
    llm: ClaudeChat, query: str, missing: list[str]
) -> str:
    """Generate a natural clarification question for missing entities."""
    missing_desc = []
    if "investor_name" in missing:
        missing_desc.append("which investor/client they're asking about")
    if "data_type" in missing:
        missing_desc.append("what data they want (portfolio, profile, performance, etc.)")

    response = await llm.ainvoke([
        SystemMessage(content=_clarification_system_prompt()),
        HumanMessage(content=(
            f"User asked: \"{query}\"\n"
            f"Missing: {', '.join(missing_desc)}\n"
            "Write one short clarification question."
        )),
    ])
    return response.content.strip()


async def _build_enriched_query(
    llm: ClaudeChat, query: str, entities: dict
) -> str:
    """Produce a refined query incorporating all extracted entities."""
    response = await llm.ainvoke([
        SystemMessage(content=ENRICHMENT_SYSTEM_PROMPT),
        HumanMessage(content=(
            f"Original query: {query}\n\n"
            f"Extracted entities:\n{json.dumps(entities, indent=2)}\n\n"
            "Rewrite as one precise question."
        )),
    ])
    return response.content.strip()


def _conversation_has_data(entities: dict, messages: list) -> bool:
    """Return True if a recent AI message already contains data about the queried investor.

    Detects follow-up questions whose answers are already in the session history,
    so we skip re-fetching from MCP and answer from conversation context instead.

    Important: if the user is asking for a SPECIFIC data_type (e.g. "portfolio"),
    the cached message must actually contain keywords relevant to that data type.
    A cached risk profile should NOT satisfy a portfolio request.
    """
    investor_name = (entities.get("investor_name") or "").strip()
    if not investor_name:
        return False

    # Collect recent AI messages (last 5 AI turns)
    ai_messages = [
        m for m in messages
        if hasattr(m, "type") and m.type == "ai"
        and hasattr(m, "content") and m.content
    ]
    if not ai_messages:
        return False

    # Map data_type to keywords that MUST be present in the cached message
    _data_type_keywords = {
        "portfolio": ["portfolio", "holdings", "aum", "allocation", "fund", "equity", "debt", "current_value", "total_investment"],
        "holdings": ["holdings", "fund", "scheme", "equity", "debt", "etf", "mutual fund", "sgb", "ppf", "epf"],
        "profile": ["profile", "pan", "email", "phone", "kyc", "investment goal", "sector focus", "segment", "category"],
        "risk": ["risk"],
        "risk_assessment": ["risk"],
        "transactions": ["transaction", "purchase", "sell", "buy", "redemption"],
        "performance": ["performance", "returns", "xirr", "cagr", "gain", "loss"],
        "aum": ["aum", "total_investment", "current_value", "net worth"],
        "nav": ["nav"],
    }

    # General data indicators (used when no specific data_type is requested)
    _data_indicators = [
        "portfolio", "holdings", "investment", "returns", "allocation",
        "aum", "nav", "xirr", "cagr", "current_value", "total_investment",
        "risk", "transactions", "funds", "schemes", "equity", "debt",
        "profile", "pan", "email", "phone", "kyc",
    ]

    requested_data_type = (entities.get("data_type") or "").lower().strip()
    # Get the specific keywords needed for the requested data type
    required_keywords = _data_type_keywords.get(requested_data_type)

    name_lower = investor_name.lower()
    for msg in ai_messages[-5:]:
        content = msg.content
        content_lower = content.lower()

        # A failed / empty / refusal turn is NOT data — never reuse it as the answer.
        if _looks_like_nonanswer(content_lower):
            continue

        # Must mention this investor
        if name_lower not in content_lower:
            continue

        # The investor must be the PRIMARY SUBJECT of the message.
        first_line = content_lower.split("\n", 1)[0]
        if name_lower not in first_line:
            continue

        # If user asked for a specific data_type, check that the cached message
        # actually contains keywords relevant to THAT type
        if required_keywords:
            if any(kw in content_lower for kw in required_keywords):
                return True
        else:
            # No specific data_type requested — any data keyword matches
            if any(kw in content_lower for kw in _data_indicators):
                return True

    return False


# --- Instrument at the bottom of the file ---
from ...observe_utils import observed_node as _observed_node


def _intent_detail(state, result):
    r = result if isinstance(result, dict) else {}
    return {
        "input": state.get("query", ""),
        "output": {
            "intent": r.get("intent"),
            "enriched_query": r.get("enriched_query"),
            "entities": r.get("enrichment_entities"),
        },
        "ns_intent": str(r.get("intent", "")),
        "ns_enriched_query": str(r.get("enriched_query", "") or state.get("query", "")),
    }


intent_enrichment = _observed_node(intent_enrichment, name="Intent Enrichment", detail=_intent_detail)