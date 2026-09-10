"""Fund / scheme / AMC name resolution for the distinct-on-empty recovery (Phase 1).

The platform's ``resolve_context`` maps a natural-language question to SQL with
EXACT matching, so a fund name spelled slightly differently from the stored canonical value
(e.g. "HDFC Midcap opportunities" vs the stored "HDFC Mid-Cap Opportunities") returns no
rows — even though the data is there. Unlike client names, there is no fixed roster to match
against: the fund universe lives only in the platform. So the recovery flow (in mcp_fetch)
discovers the real names at runtime via a broad DISTINCT lookup, and this module does the
local normalization + similarity matching.

Deterministic and dependency-free (stdlib ``difflib``), mirroring the investor-name resolver
in ``intent_enrichment.py``.
"""

import re
from difflib import SequenceMatcher

# Common Indian AMCs — used to pick a broad, reliable anchor for the DISTINCT query. Listing
# the multi-word variants too lets ``anchor_token`` prefer the most specific match.
_KNOWN_AMCS = (
    "aditya birla sun life", "aditya birla", "icici prudential", "nippon india",
    "mirae asset", "franklin templeton", "motilal oswal", "kotak mahindra",
    "parag parikh", "canara robeco", "bank of india", "360 one",
    "hdfc", "icici", "sbi", "axis", "kotak", "nippon", "uti", "dsp", "mirae",
    "franklin", "tata", "quant", "ppfas", "motilal", "edelweiss", "invesco",
    "bandhan", "hsbc", "lic", "sundaram", "navi", "mahindra", "whiteoak", "groww",
)

# Words that don't help tell one fund from another — dropped before scoring.
_FILLER = (
    "fund", "scheme", "plan", "direct", "regular", "growth", "idcw", "dividend",
    "option", "the",
)

# Short forms users type → expanded so they line up with the stored names.
_ABBREV = (
    (r"\bopps?\b", "opportunities"),
    (r"\bopp\b", "opportunities"),
    (r"\bmid\s*[- ]?\s*cap\b", "midcap"),
    (r"\bsmall\s*[- ]?\s*cap\b", "smallcap"),
    (r"\blarge\s*[- ]?\s*cap\b", "largecap"),
    (r"\bflexi\s*[- ]?\s*cap\b", "flexicap"),
    (r"\bmulti\s*[- ]?\s*cap\b", "multicap"),
    (r"\bblue\s*[- ]?\s*chip\b", "bluechip"),
)


# Static fallback catalog — the platform's exact mutual-fund scheme names (the full universe is
# small and stable). Used to canonicalize a user's fund name when the LIVE catalog fetch fails,
# which it does often because the platform is intermittently degraded. These strings are the
# exact stored values, so substituting them into a query resolves reliably when the platform is
# up. Refresh if the catalog changes.
_STATIC_FUND_CATALOG = (
    "Aditya Birla SL Frontline Equity Fund - Direct Growth",
    "Axis Bluechip Fund - Direct Growth",
    "Axis Focused 25 Fund - Direct Growth",
    "Axis Healthcare Fund - Direct Growth",
    "Axis Long Term Equity Fund - Direct Growth",
    "Bandhan Balanced Advantage Fund - Direct Growth",
    "Canara Robeco Equity Hybrid Fund - Direct Growth",
    "DSP Tax Saver Fund - Direct Growth",
    "HDFC Corporate Bond Fund - Direct Growth",
    "HDFC Liquid Fund - Direct Growth",
    "HDFC Mid-Cap Opportunities Fund - Direct Growth",
    "HSBC Value Fund - Direct Growth",
    "ICICI Pru Dividend Yield Equity Fund - Direct Growth",
    "ICICI Prudential All Seasons Bond Fund - Direct Growth",
    "ICICI Prudential Bluechip Fund - Direct Growth",
    "ICICI Prudential Technology Fund - Direct Growth",
    "Kotak Emerging Equity Fund - Direct Growth",
    "Kotak Flexi Cap Fund - Direct Growth",
    "Kotak International REIT FoF - Direct Growth",
    "Kotak Small Cap Fund - Direct Growth",
    "Mirae Asset Large Cap Fund - Direct Growth",
    "Motilal Oswal Nasdaq 100 FoF - Direct Growth",
    "Nippon India Growth Fund - Direct Growth",
    "Parag Parikh Flexi Cap Fund - Direct Growth",
    "Quant Active Fund - Direct Growth",
    "SBI Contra Fund - Direct Growth",
    "SBI Magnum Gilt Fund - Direct Growth",
    "SBI Small Cap Fund - Direct Growth",
    "Tata Digital India Fund - Direct Growth",
    "UTI Nifty 50 Index Fund - Direct Growth",
)


def _normalize(s: str) -> str:
    """Collapse a fund/scheme name to a comparison key: lowercase, expand common short
    forms, drop filler words, then strip every non-alphanumeric character. So
    "HDFC Mid-Cap Opportunities Fund" and "hdfc midcap opp" both become
    "hdfcmidcapopportunities"."""
    s = (s or "").lower()
    for pat, repl in _ABBREV:
        s = re.sub(pat, repl, s)
    s = re.sub(r"\b(" + "|".join(_FILLER) + r")\b", " ", s)
    return re.sub(r"[^a-z0-9]", "", s)


def anchor_token(fund: str) -> str:
    """A broad search term the platform CAN match even when the full name can't — prefer a
    recognised AMC name (most specific wins), else the first word of the fund name."""
    low = (fund or "").lower()
    for amc in _KNOWN_AMCS:  # ordered most-specific first
        # Whole-word/phrase match so "quant" (the AMC) doesn't fire on "Quantum".
        if re.search(r"\b" + re.escape(amc) + r"\b", low):
            return amc
    toks = (fund or "").strip().split()
    return toks[0].lower() if toks else (fund or "").strip().lower()


def best_match(user_fund: str, candidates: list[str], cutoff: float = 0.84,
               tie_gap: float = 0.06):
    """Similarity-match the user's fund name against the discovered distinct list.

    Returns one of:
        (canonical, None)    -> a single confident match
        (None, [a, b, ...])  -> ambiguous near-tie; caller should ask "did you mean ...?"
        (None, None)         -> nothing close enough; a genuine no-data result
    """
    target = _normalize(user_fund)
    if not target or not candidates:
        return (None, None)

    scored: list[tuple[float, str]] = []
    seen: set[str] = set()
    for c in candidates:
        c = (c or "").strip()
        if not c or c in seen:
            continue
        seen.add(c)
        scored.append((SequenceMatcher(None, target, _normalize(c)).ratio(), c))
    if not scored:
        return (None, None)

    scored.sort(key=lambda x: x[0], reverse=True)
    best_ratio = scored[0][0]
    if best_ratio < cutoff:
        return (None, None)

    # Everything within tie_gap of the best is "too close to call" — offer the top few.
    close = [c for r, c in scored if best_ratio - r < tie_gap]
    if len(close) > 1:
        return (None, close[:3])
    return (scored[0][1], None)
