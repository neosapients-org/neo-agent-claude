"""Data dictionary — the platform semantic layer, for data-aware clarifying questions.

Compiled from ``app/data/agent_metrics.xlsx`` into ``app/data/data_dictionary.json``
(see ``app/data/build_dictionary.py``). Loaded with the stdlib only — no openpyxl at runtime.

Purpose (Plan 1): when a user references a metric that does NOT exist in the platform
(e.g. "AEM"), the agent should notice and ask "did you mean AUM?" instead of confidently
answering with the wrong metric. This module supplies that check. It is deterministic and
LLM-free, so it adds no latency.

Public API:
    find_metric_issue(query, entities) -> MetricIssue | None
        The single entry point intent_enrichment calls. Returns an issue (with a ready-made
        clarification ``question``) only when a referenced term is a confident near-miss of a
        real metric. Returns None when everything checks out — biased towards NOT interrupting.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher, get_close_matches
from pathlib import Path

_DATA_PATH = Path(__file__).parent / "data" / "data_dictionary.json"


# --- load the compiled dictionary ------------------------------------------------

def _load_entries() -> list[dict]:
    try:
        return json.loads(_DATA_PATH.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        # Defensive: a missing/corrupt artefact must never crash the agent — it just
        # means no metric validation happens (the prior behaviour).
        return []


_ENTRIES: list[dict] = _load_entries()


# --- acronym vocabulary ----------------------------------------------------------
# Curated finance/platform acronyms with a human expansion for the clarification text.
# These are the abbreviations a user is likely to (mis)type. "AEM" deliberately ISN'T
# here — it should fuzzy-match to "AUM" and trigger a clarification.
_SEED_ACRONYMS: dict[str, str] = {
    "AUM": "AUM (assets under management)",
    "NAV": "NAV (net asset value)",
    "NNM": "NNM (net new money)",
    "XIRR": "XIRR (annualised return)",
    "CAGR": "CAGR (compound annual growth rate)",
    "STCG": "STCG (short-term capital gain)",
    "LTCG": "LTCG (long-term capital gain)",
    "AMC": "AMC (asset management company)",
    "LOB": "LOB (line of business)",
    "SBU": "SBU (strategic business unit)",
    "ETF": "ETF", "AIF": "AIF", "PMS": "PMS", "REIT": "REIT", "INVIT": "InvIT",
    "PAN": "PAN", "KYC": "KYC status", "SIP": "SIP", "DP": "DP (demat) account",
    "HNI": "HNI", "NRI": "NRI", "FD": "FD (fixed deposit)",
    "EPF": "EPF", "PPF": "PPF", "NPS": "NPS", "SGB": "SGB",
}

# Acronyms that also appear inside Display Names (e.g. "Total AUM") — auto-harvested so the
# known set stays in sync with the sheet. Tokens 2-6 uppercase letters.
_ACRONYM_IN_NAME = re.compile(r"\b[A-Z]{2,6}\b")


def _harvest_acronyms() -> dict[str, str]:
    found = dict(_SEED_ACRONYMS)
    for e in _ENTRIES:
        for tok in _ACRONYM_IN_NAME.findall(e.get("display_name", "")):
            found.setdefault(tok, tok)
    return found


# canonical (upper) acronym -> human expansion
_ACRONYMS: dict[str, str] = _harvest_acronyms()
_ACRONYM_KEYS_LOWER: list[str] = [a.lower() for a in _ACRONYMS]

# Tokens that look like acronyms but are NOT metric references — never clarify these.
_ACRONYM_STOPWORDS = {
    "INV",            # client id prefix (INV-001)
    "AND", "OR", "THE", "FOR", "ALL", "PER", "YTD", "USD", "INR", "API", "URL",
    "PDF", "CSV", "ID", "OK", "VS", "TV", "IT", "AI",
}

# Regex to pull acronym-shaped tokens out of a raw query.
_ACRONYM_TOKEN = re.compile(r"\b[A-Za-z]{2,6}\b")


@dataclass
class MetricIssue:
    """A metric term the agent should clarify before answering."""
    term: str                       # what the user wrote, e.g. "AEM"
    outcome: str                    # "fuzzy_one" | "fuzzy_many"
    candidates: list[str] = field(default_factory=list)  # human expansions

    @property
    def question(self) -> str:
        if self.outcome == "fuzzy_one" and self.candidates:
            return (
                f"I couldn't find “{self.term}” in the platform data. "
                f"Did you mean {self.candidates[0]}?"
            )
        if self.candidates:
            listed = ", ".join(self.candidates[:-1]) + f", or {self.candidates[-1]}"
            return (
                f"I couldn't find “{self.term}” in the platform data. "
                f"Did you mean {listed}?"
            )
        return f"I couldn't find “{self.term}” in the platform data — could you rephrase?"


def _is_known_acronym(tok_upper: str) -> bool:
    return tok_upper in _ACRONYMS


def _fuzzy_acronym(tok_upper: str) -> MetricIssue | None:
    """Match an unknown acronym-shaped token against the known metric acronyms.

    Only returns an issue for a CONFIDENT near-miss (e.g. AEM->AUM). A token that is far
    from every known acronym returns None — we don't interrupt on terms that might be a
    fund/ticker/company name the platform can still resolve.
    """
    low = tok_upper.lower()
    matches = get_close_matches(low, _ACRONYM_KEYS_LOWER, n=3, cutoff=0.6)
    if not matches:
        return None
    # Map back to canonical expansions, de-duplicated, preserving order.
    expansions: list[str] = []
    for m in matches:
        canon = next(a for a in _ACRONYMS if a.lower() == m)
        exp = _ACRONYMS[canon]
        if exp not in expansions:
            expansions.append(exp)
    # One clearly-best candidate -> a confirm question; otherwise offer the top few.
    if len(expansions) == 1:
        return MetricIssue(term=tok_upper, outcome="fuzzy_one", candidates=expansions)
    # If the best is clearly closer than the runner-up, treat as a single confirm.
    s0 = SequenceMatcher(None, low, matches[0]).ratio()
    s1 = SequenceMatcher(None, low, matches[1]).ratio()
    if s0 - s1 >= 0.12:
        return MetricIssue(term=tok_upper, outcome="fuzzy_one", candidates=expansions[:1])
    return MetricIssue(term=tok_upper, outcome="fuzzy_many", candidates=expansions[:3])


def find_metric_issue(query: str, entities: dict | None = None) -> MetricIssue | None:
    """Return a clarification issue if the query references an unknown metric that is a
    confident near-miss of a real one. Returns None otherwise (the common case).

    Strategy (deliberately conservative to avoid over-clarifying):
      1. Look at the extracted ``specific_metric`` first, then acronym-shaped tokens in the
         raw query.
      2. For each, if it's a KNOWN acronym/term -> fine. If it's unknown but a confident
         fuzzy match to a known metric acronym -> raise an issue. If it's far from
         everything -> let it pass (might be a fund/company the platform can resolve).
    """
    if not _ENTRIES:
        return None

    entities = entities or {}
    seen: set[str] = set()
    candidates: list[str] = []

    sm = (entities.get("specific_metric") or "").strip()
    if sm:
        candidates.append(sm)
    # Raw-query scan: only UPPERCASE tokens. Genuine acronyms (AEM, AUM, NAV) are typed in
    # caps; this single rule eliminates false positives from ordinary words ("is", "all",
    # "the") without needing an exhaustive stopword list.
    candidates.extend(t for t in _ACRONYM_TOKEN.findall(query or "") if t.isupper())

    for raw in candidates:
        tok = raw.strip()
        if not tok:
            continue
        up = tok.upper()
        if up in seen:
            continue
        seen.add(up)
        # Single-token, acronym-shaped only. Multi-word phrases are left to the platform
        # for now (a documented v1 boundary — see the improvement plan, Plan 1 §1.5).
        if not re.fullmatch(r"[A-Za-z]{2,6}", tok):
            continue
        if up in _ACRONYM_STOPWORDS:
            continue
        if _is_known_acronym(up):
            continue
        issue = _fuzzy_acronym(up)
        if issue:
            return issue
    return None
