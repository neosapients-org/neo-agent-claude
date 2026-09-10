"""CortexContextRegistry — front-loads the tenant's live context map from the Cortex platform.

Three artifacts, each fetched via the platform MCP server and cached in RAM with a TTL
(mirroring the SkillRegistry lifecycle in app/skills/registry.py):

  * ``capabilities`` — the tenant context map from ``get_cortex_capabilities``: the query
    families (primitives) Cortex can answer, verified example questions, and clarification
    guidance. This is the source of truth for "what is answerable", replacing hardcoded
    keyword lists and static prompt scope.
  * ``roster``        — the client book from ``resolve_context`` ("list all clients …"),
    enriched with family / AUM / city / banker so the SAME fetch feeds both the agent's
    name resolution and the frontend's Client Insights picker.
  * ``dimensions``    — reference vocabularies (sectors, asset classes, LOBs) used to offer
    concrete options in clarification questions.

Every fetch is best-effort: a failed/empty refresh keeps the last-good value, and the
registry falls back to the bundled seeds so the agent behaves exactly as before when the
platform is unreachable. Nothing here raises.
"""

import asyncio
import json
import logging
import re
import time

from ..config import config

logger = logging.getLogger(__name__)

# --- seeds (fallback only — the live platform is the source of truth) --------------------
# The pre-registry hardcoded roster. Kept ONLY so a cold start with the MCP server down
# behaves exactly like the old build. Never extended by hand — fix the platform instead.
SEED_ROSTER: list[dict] = [
    {"name": n, "family": f}
    for f, members in {
        "Krishnan Family": ["Ram Krishnan", "Sita Krishnan", "Karthik Krishnan",
                            "Lakshmi Krishnan", "Venkat Krishnan", "Priya Krishnan"],
        "Mehra Family": ["Arjun Mehra", "Nisha Mehra", "Rohan Mehra", "Ananya Mehra",
                         "Suresh Mehra"],
        "Patel Family": ["Nikhil Patel", "Meera Patel", "Aditya Patel", "Kavya Patel",
                         "Dinesh Patel", "Rashi Patel", "Jayesh Patel"],
        "Choudhary Family": ["Vivek Choudhary", "Sunita Choudhary", "Rahul Choudhary",
                             "Neha Choudhary", "Mohan Choudhary", "Kamala Choudhary"],
        "Fernandes Family": ["Roshni Fernandes", "Martin Fernandes", "Sarah Fernandes",
                             "Daniel Fernandes", "Grace Fernandes", "Joseph Fernandes"],
    }.items()
    for n in members
]

# The roster query sent to resolve_context. Asks for the enriched columns the frontend
# needs; the parser is header-driven so a platform that returns fewer columns still works.
# NOTE: asking for the RM name IN THIS query makes the platform fail with EXECUTION_ERROR
# (the client->banker join isn't resolvable alongside the AUM aggregation). The RM name is
# fetched by RM_QUERY below and merged into the roster by client name.
ROSTER_QUERY = ("List all clients with client id, full name, family name, city and total AUM")

# The client -> relationship-manager mapping, fetched separately (the join succeeds on its
# own, just not combined with the AUM roster above) and merged into the roster by full name.
RM_QUERY = "List each client full name and their relationship manager name"

# Reference vocabularies for clarification options. Each resolves through the platform;
# any that fail simply stay absent (clarification then omits that option list).
DIMENSION_QUERIES: dict[str, str] = {
    "sectors": "List all distinct market sectors in the product sector mapping",
    "asset_classes": "List all distinct asset classes in the product catalogue",
    "lobs": "List all LOB names with their sub-LOB names",
}

# Flexible header -> canonical roster field. resolve_context column names vary with
# phrasing, so map by substring (checked in order; first hit wins).
_ROSTER_FIELD_ALIASES: list[tuple[str, str]] = [
    ("client_id", "id"), ("client id", "id"),
    ("full_name", "name"), ("full name", "name"), ("client_name", "name"),
    ("family", "family"),
    ("city", "location"), ("location", "location"),
    ("aum", "aum"), ("portfolio_value", "aum"), ("total_value", "aum"),
    ("banker", "rm"), ("relationship", "rm"), ("rm_name", "rm"), ("advisor", "rm"),
    ("id", "id"), ("name", "name"),  # bare fallbacks LAST so specific ones win
]


def _canon_field(header: str) -> str | None:
    h = (header or "").strip().strip('"').lower()
    for alias, canon in _ROSTER_FIELD_ALIASES:
        if alias in h:
            return canon
    return None


def _extract_result_text(tool_result: dict) -> str:
    """Data text out of an MCP tool result (same shape mcp_fetch consumes)."""
    resp = tool_result.get("response")
    if not (tool_result.get("success") and isinstance(resp, dict)):
        return ""
    blocks = [item.get("text", "") for item in resp.get("content", []) or []
              if isinstance(item, dict) and item.get("type") == "text" and item.get("text")]
    return max(blocks, key=len) if blocks else ""


def _rows_from_payload(text: str) -> list[dict]:
    """Parse a resolve_context payload into rows via the shared mcp_fetch parser."""
    if not (text or "").strip():
        return []
    from ..graph.nodes.mcp_fetch import _parse_mcp_text
    try:
        parsed = json.loads(_parse_mcp_text(text))
    except (json.JSONDecodeError, TypeError):
        return []
    if isinstance(parsed, dict):
        parsed = [parsed]
    return [r for r in parsed if isinstance(r, dict)]


def parse_roster_rows(rows: list[dict]) -> list[dict]:
    """Map platform rows (arbitrary headers) to canonical roster records.

    A record is kept only if it has a plausible client name; extra columns (id, aum,
    location, rm) are carried when present so the frontend can consume them.
    """
    roster = []
    for row in rows:
        rec: dict = {}
        for header, value in row.items():
            canon = _canon_field(header)
            if canon and canon not in rec and str(value).strip():
                rec[canon] = str(value).strip()
        name = rec.get("name", "")
        # A real client name: at least two words, alphabetic, no digits.
        if not re.match(r"^[A-Za-z][A-Za-z.'-]*(\s+[A-Za-z][A-Za-z.'-]*)+$", name):
            continue
        if "aum" in rec:
            try:
                rec["aum"] = float(rec["aum"].replace(",", ""))
            except ValueError:
                rec.pop("aum")
        roster.append(rec)
    return roster


def _rm_map_from_rows(rows: list[dict]) -> dict[str, str]:
    """Build {client_full_name_lower -> rm_name} from a client->RM payload. The name column
    is whichever header canonicalises to 'name'; the RM column is any remaining header that
    looks like a banker/relationship-manager field (or, failing that, the other column)."""
    mapping: dict[str, str] = {}
    for row in rows:
        name = rm = None
        leftover = []
        for header, value in row.items():
            v = str(value).strip()
            if not v:
                continue
            canon = _canon_field(header)
            if canon == "name" and name is None:
                name = v
            elif canon == "rm" and rm is None:
                rm = v
            else:
                leftover.append(v)
        # Two-column payload where the RM header didn't alias cleanly: take the leftover.
        if name and rm is None and leftover:
            rm = leftover[0]
        if name and rm:
            mapping[name.lower()] = rm
    return mapping


def parse_capabilities(result: dict) -> dict:
    """Distill a ``get_cortex_capabilities`` tool result into the compact map we keep:
    ``{"primitives": [descriptions], "example_questions": [...], "guidance": [...]}``."""
    text = _extract_result_text(result)
    if not text:
        return {}
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return {}
    primitives, examples = [], []
    for domain in data.get("domains", []) or []:
        for fam in domain.get("queryFamilies", []) or []:
            desc = (fam.get("description") or "").strip()
            if desc:
                primitives.append(desc)
        examples.extend(q for q in domain.get("exampleQuestions", []) or [] if q)
    if not primitives and not examples:
        return {}
    return {
        "primitives": primitives,
        "example_questions": examples,
        "guidance": [g for g in data.get("guidance", []) or [] if g],
    }


# Tokens too generic to signal "this is a data question" even though they appear all over
# the capability map (stopwords + question scaffolding).
_VOCAB_STOP = frozenset(
    "the a an of in on for with and or to is are was were what which who how many much show "
    "me my all list give per by from as at their his her its this that these those last next "
    "each any than more most does do did has have had can could i we you it one two three "
    "between across under above below into over out not no yes new used using use".split()
)


def build_capability_vocabulary(capabilities: dict) -> frozenset:
    """Domain vocabulary derived from the capability map — every meaningful word that appears
    in a primitive description or a verified example question. Used to recognise that a query
    is about platform data WITHOUT a hand-maintained keyword list."""
    words: set[str] = set()
    for text in (capabilities.get("primitives") or []) + (capabilities.get("example_questions") or []):
        for w in re.findall(r"[a-z][a-z&'-]{2,}", text.lower()):
            if w not in _VOCAB_STOP:
                words.add(w)
    return frozenset(words)


class CortexContextRegistry:
    """RAM cache of the tenant context map + roster + dimensions, TTL-refreshed from MCP."""

    def __init__(self, seed_roster: list[dict] | None = None):
        self._roster: list[dict] = list(seed_roster or [])
        self._capabilities: dict = {}
        self._dimensions: dict[str, list[str]] = {}
        self._vocab: frozenset = frozenset()
        self._source = "seed"
        self._loaded_at = 0.0
        self._version = 0          # bumped on every adopted refresh — memo key for callers
        self._lock = asyncio.Lock()

    # --- lifecycle (mirrors SkillRegistry) -------------------------------------------
    async def ensure_fresh(self) -> None:
        """Refresh from the platform if stale. Cheap no-op within the TTL."""
        if not config.cortex_context_sync:
            return
        if (time.time() - self._loaded_at) < config.skills_refresh_ttl:
            return
        async with self._lock:
            if (time.time() - self._loaded_at) < config.skills_refresh_ttl:
                return
            await self._refresh()

    async def force_refresh(self) -> dict:
        async with self._lock:
            await self._refresh()
        return {"source": self._source, "clients": len(self._roster),
                "primitives": len(self._capabilities.get("primitives", [])),
                "example_questions": len(self._capabilities.get("example_questions", [])),
                "dimensions": {k: len(v) for k, v in self._dimensions.items()}}

    async def _refresh(self) -> None:
        """Suppress per-request platform-call logging around the refresh: the roster / RM /
        dimension queries are internal bootstrap calls and must not appear in the UI's
        "queries sent to the platform" panel (which shows what answered the user's question)."""
        from ..graph.nodes.stream_channel import suppress_platform_log
        _tok = suppress_platform_log.set(True)
        try:
            await self._refresh_impl()
        finally:
            suppress_platform_log.reset(_tok)

    async def _refresh_impl(self) -> None:
        """Best-effort refresh of all three artifacts. Each keeps its last-good value on
        failure; the TTL clock advances regardless so a down platform isn't hammered."""
        from ..mcp.client import mcp_client
        adopted = []

        # 1) capabilities map
        try:
            result = await mcp_client.call_tool("get_cortex_capabilities", {"role": "advisor"})
            caps = parse_capabilities(result)
            if caps:
                self._capabilities = caps
                self._vocab = build_capability_vocabulary(caps)
                adopted.append(f"capabilities({len(caps.get('primitives', []))} primitives)")
        except Exception as e:
            logger.warning("[context] capabilities refresh failed: %s", e)

        # 2) client roster (SEQUENTIAL calls — parallel MCP requests overload the gateway)
        try:
            result = await mcp_client.call_tool("resolve_context", {"query": ROSTER_QUERY})
            roster = parse_roster_rows(_rows_from_payload(_extract_result_text(result)))
            if roster:
                self._roster = roster
                self._source = "mcp"
                adopted.append(f"roster({len(roster)} clients)")
        except Exception as e:
            logger.warning("[context] roster refresh failed: %s", e)

        # 2b) client -> RM mapping (separate query; the join fails when combined with AUM).
        # Merge onto the current roster by full name so the RM column front-loads like the
        # rest. Best-effort: a failure just leaves rm absent (frontend falls back).
        try:
            result = await mcp_client.call_tool("resolve_context", {"query": RM_QUERY})
            rm_by_name = _rm_map_from_rows(_rows_from_payload(_extract_result_text(result)))
            if rm_by_name:
                merged = 0
                for rec in self._roster:
                    rm = rm_by_name.get((rec.get("name") or "").strip().lower())
                    if rm:
                        rec["rm"] = rm
                        merged += 1
                adopted.append(f"rm({merged} mapped)")
        except Exception as e:
            logger.warning("[context] RM mapping refresh failed: %s", e)

        # 3) dimension vocabularies
        for dim, query in DIMENSION_QUERIES.items():
            try:
                result = await mcp_client.call_tool("resolve_context", {"query": query})
                rows = _rows_from_payload(_extract_result_text(result))
                values = sorted({str(v).strip() for r in rows for v in r.values()
                                 if str(v).strip() and not str(v).strip().isdigit()})
                if values:
                    self._dimensions[dim] = values
            except Exception as e:
                logger.warning("[context] dimension %s refresh failed: %s", dim, e)

        self._loaded_at = time.time()
        if adopted:
            self._version += 1
            logger.info("[context] refreshed from platform: %s", ", ".join(adopted))
        else:
            logger.warning("[context] refresh adopted nothing — keeping %s data", self._source)

    # --- read API (sync — always serves the current cache) ---------------------------
    @property
    def version(self) -> int:
        return self._version

    @property
    def source(self) -> str:
        return self._source

    @property
    def loaded_at(self) -> float:
        return self._loaded_at

    def roster(self) -> list[dict]:
        return list(self._roster)

    def roster_names(self) -> list[str]:
        return [r["name"] for r in self._roster if r.get("name")]

    def families(self) -> dict[str, list[str]]:
        """family display name -> member full names (insertion order preserved)."""
        fams: dict[str, list[str]] = {}
        for r in self._roster:
            fam = r.get("family") or "Other"
            fams.setdefault(fam, []).append(r["name"])
        return fams

    def capabilities(self) -> dict:
        return self._capabilities

    def capability_vocabulary(self) -> frozenset:
        return self._vocab

    def dimensions(self) -> dict[str, list[str]]:
        return dict(self._dimensions)

    def scope_block(self, max_examples: int = 40) -> str:
        """Compact prompt block describing what the platform can answer — primitives,
        clarification guidance, and dimension option lists. Empty string when the map
        hasn't loaded (callers then keep their static prompt text)."""
        caps = self._capabilities
        if not caps:
            return ""
        lines = ["The data platform can answer questions about:"]
        for desc in caps.get("primitives", []):
            # First sentence of each primitive description keeps the block small.
            lines.append(f"- {desc.split('. ')[0].strip().rstrip('.')}")
        for dim, values in self._dimensions.items():
            if values and len(values) <= 40:
                lines.append(f"Known {dim.replace('_', ' ')}: {', '.join(values)}")
        examples = caps.get("example_questions", [])[:max_examples]
        if examples:
            lines.append("Examples of questions it answers well: " + " | ".join(examples))
        return "\n".join(lines)


# Singleton — front-loaded in main.lifespan, TTL-refreshed via ensure_fresh() per turn.
cortex_context = CortexContextRegistry(seed_roster=SEED_ROSTER)
