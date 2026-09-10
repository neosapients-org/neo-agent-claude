"""Tests for the Cortex context registry (app/context/registry.py) and its consumers.

Covers:
  * parsing a get_cortex_capabilities tool result into the compact capability map
  * parsing an enriched resolve_context roster payload (header-alias mapping, AUM coercion,
    junk-row rejection)
  * seed fallback — a failed/empty refresh keeps the last-good roster, so name resolution
    behaves exactly like the pre-registry build when the platform is down
  * refresh adoption — a new roster from the platform flows into intent_enrichment's
    name-resolution indexes (memo invalidation by version)
  * the dynamic clarification prompt and the capability-grounded data-vocabulary check

All platform I/O is mocked — no MCP server needed.
"""

import json
import os
import sys

import pytest
from unittest.mock import AsyncMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.context.registry import (
    CortexContextRegistry,
    SEED_ROSTER,
    build_capability_vocabulary,
    parse_capabilities,
    parse_roster_rows,
    _rows_from_payload,
    _rm_map_from_rows,
)


# --- fixtures ---------------------------------------------------------------------

CAPABILITIES_TOOL_RESULT = {
    "success": True,
    "response": {"content": [{"type": "text", "text": json.dumps({
        "domains": [{
            "queryFamilies": [
                {"id": "atom:1", "description": "Current portfolio positions for all clients. Primary atom for AUM."},
                {"id": "atom:2", "description": "Monthly rolling return snapshots for mutual fund schemes."},
            ],
            "exampleQuestions": [
                "What is the total AUM across all clients?",
                "Which large cap fund has the best 3-year return?",
            ],
        }],
        "guidance": ["Ask one clarifying question if the user request lacks a primitive."],
    })}]},
}

ROSTER_PAYLOAD = (
    '[3]{client_id,full_name,family_name,city,total_aum,relationship_manager_name}:\n'
    '  7001,Ram Krishnan,Krishnan Family,Bengaluru,"952875170.42",Rajesh Kumar\n'
    '  7031,Aarav Nair,Nair Family,Kochi,1234567.89,Priya Sharma\n'
    '  ,,,,,\n'
)


def _tool_result(text: str) -> dict:
    return {"success": True, "response": {"content": [{"type": "text", "text": text}]}}


# --- parsers ----------------------------------------------------------------------

def test_parse_capabilities_extracts_primitives_examples_guidance():
    caps = parse_capabilities(CAPABILITIES_TOOL_RESULT)
    assert len(caps["primitives"]) == 2
    assert "Current portfolio positions" in caps["primitives"][0]
    assert len(caps["example_questions"]) == 2
    assert caps["guidance"]


def test_parse_capabilities_empty_on_garbage():
    assert parse_capabilities({"success": False, "response": {}}) == {}
    assert parse_capabilities(_tool_result("not json")) == {}


def test_parse_roster_maps_aliases_and_rejects_junk():
    rows = _rows_from_payload(ROSTER_PAYLOAD)
    roster = parse_roster_rows(rows)
    assert len(roster) == 2  # the blank row is rejected
    ram = roster[0]
    assert ram["id"] == "7001"
    assert ram["name"] == "Ram Krishnan"
    assert ram["family"] == "Krishnan Family"
    assert ram["location"] == "Bengaluru"
    assert ram["aum"] == pytest.approx(952875170.42)
    assert ram["rm"] == "Rajesh Kumar"


def test_parse_roster_rejects_non_name_rows():
    rows = [{"full_name": "12345", "family_name": "X"},
            {"full_name": "OnlyOneWord"},
            {"full_name": "Valid Person", "family_name": "Valid Family"}]
    roster = parse_roster_rows(rows)
    assert [r["name"] for r in roster] == ["Valid Person"]


RM_PAYLOAD = (
    '[3]{client_full_name,relationship_manager_name}:\n'
    '  Ram Krishnan,Rajesh Kumar\n'
    '  Sita Krishnan,Rajesh Kumar\n'
    '  Karthik Krishnan,Priya Sharma\n'
)


def test_rm_map_from_rows_keys_by_lowercased_name():
    m = _rm_map_from_rows(_rows_from_payload(RM_PAYLOAD))
    assert m["ram krishnan"] == "Rajesh Kumar"
    assert m["karthik krishnan"] == "Priya Sharma"


def test_rm_map_handles_alternate_headers():
    rows = [{"client_name": "Aditya Patel", "servicing_banker_name": "Vikram Singh"}]
    assert _rm_map_from_rows(rows) == {"aditya patel": "Vikram Singh"}


def test_capability_vocabulary_keeps_domain_words_drops_stopwords():
    vocab = build_capability_vocabulary(parse_capabilities(CAPABILITIES_TOOL_RESULT))
    assert "aum" in vocab and "portfolio" in vocab and "fund" in vocab
    assert "what" not in vocab and "the" not in vocab


# --- registry lifecycle -----------------------------------------------------------

def _registry_with(monkeypatch_target, tool_results: dict):
    """Registry whose mcp_client.call_tool returns canned results per tool/query."""
    reg = CortexContextRegistry(seed_roster=SEED_ROSTER)

    async def fake_call_tool(tool, args):
        key = tool if tool != "resolve_context" else args["query"]
        return tool_results.get(key, {"success": False, "response": {}})

    return reg, fake_call_tool


@pytest.mark.asyncio
async def test_refresh_adopts_platform_roster_and_capabilities():
    from app.context import registry as regmod
    reg, fake = _registry_with(regmod, {
        "get_cortex_capabilities": CAPABILITIES_TOOL_RESULT,
        regmod.ROSTER_QUERY: _tool_result(ROSTER_PAYLOAD),
    })
    with patch("app.mcp.client.mcp_client") as mc:
        mc.call_tool = AsyncMock(side_effect=fake)
        info = await reg.force_refresh()
    assert reg.source == "mcp"
    assert info["clients"] == 2
    assert "Aarav Nair" in reg.roster_names()
    assert reg.capability_vocabulary()  # vocab built
    assert reg.version == 1


@pytest.mark.asyncio
async def test_failed_refresh_keeps_seed_roster():
    from app.context import registry as regmod
    reg, fake = _registry_with(regmod, {})  # every call fails
    with patch("app.mcp.client.mcp_client") as mc:
        mc.call_tool = AsyncMock(side_effect=fake)
        await reg.force_refresh()
    assert reg.source == "seed"
    assert reg.roster_names() == [r["name"] for r in SEED_ROSTER]
    assert reg.version == 0  # nothing adopted


@pytest.mark.asyncio
async def test_empty_payload_keeps_last_good_roster():
    from app.context import registry as regmod
    reg, fake = _registry_with(regmod, {
        regmod.ROSTER_QUERY: _tool_result(ROSTER_PAYLOAD),
    })
    with patch("app.mcp.client.mcp_client") as mc:
        mc.call_tool = AsyncMock(side_effect=fake)
        await reg.force_refresh()
    assert "Aarav Nair" in reg.roster_names()
    # now the platform starts returning nothing — last-good roster must survive
    with patch("app.mcp.client.mcp_client") as mc:
        mc.call_tool = AsyncMock(return_value={"success": False, "response": {}})
        await reg.force_refresh()
    assert "Aarav Nair" in reg.roster_names()


def test_families_grouping_and_scope_block():
    reg = CortexContextRegistry(seed_roster=SEED_ROSTER)
    fams = reg.families()
    assert set(fams) == {"Krishnan Family", "Mehra Family", "Patel Family",
                         "Choudhary Family", "Fernandes Family"}
    assert "Ram Krishnan" in fams["Krishnan Family"]
    assert reg.scope_block() == ""  # no capabilities loaded -> empty block


# --- intent_enrichment consumers ---------------------------------------------------

def _swap_registry(new_reg):
    """Point the app.context singleton at a test registry (and reset the memo)."""
    import app.context as ctx
    import app.context.registry as regmod
    from app.graph.nodes import intent_enrichment as ie
    ctx.cortex_context = new_reg
    regmod.cortex_context = new_reg
    ie._roster_memo["version"] = None  # force index rebuild
    return ie


def test_name_resolution_from_seed_matches_old_behaviour():
    ie = _swap_registry(CortexContextRegistry(seed_roster=SEED_ROSTER))
    # exact, fuzzy-typo, first-name, ambiguous-family — the pre-registry contract
    assert ie._resolve_client_name("Ram Krishnan") == ("Ram Krishnan", None)
    assert ie._resolve_client_name("Rama Krishana")[0] == "Ram Krishnan"
    assert ie._resolve_client_name("Arjun") == ("Arjun Mehra", None)
    canonical, candidates = ie._resolve_client_name("Patel")
    assert canonical is None and len(candidates) == 7


def test_new_platform_client_resolves_after_refresh():
    reg = CortexContextRegistry(seed_roster=SEED_ROSTER)
    reg._roster = [{"name": "Aarav Nair", "family": "Nair Family"}] + SEED_ROSTER
    reg._version = 99  # simulate an adopted refresh
    ie = _swap_registry(reg)
    assert ie._resolve_client_name("Aarav Nair") == ("Aarav Nair", None)
    assert ie._resolve_client_name("Aarav") == ("Aarav Nair", None)
    assert "nair" in ie._known_surnames()


def test_clarification_prompt_lists_live_roster():
    reg = CortexContextRegistry(seed_roster=SEED_ROSTER)
    reg._roster = [{"name": "Aarav Nair", "family": "Nair Family"}]
    reg._capabilities = parse_capabilities(CAPABILITIES_TOOL_RESULT)
    reg._version = 100
    ie = _swap_registry(reg)
    prompt = ie._clarification_system_prompt()
    assert "Nair Family: Aarav Nair" in prompt
    assert "Current portfolio positions" in prompt      # platform scope included
    assert "Krishnan" not in prompt                     # no hardcoded roster left


def test_data_vocabulary_check_uses_capability_map():
    reg = CortexContextRegistry(seed_roster=SEED_ROSTER)
    reg._capabilities = parse_capabilities(CAPABILITIES_TOOL_RESULT)
    reg._vocab = build_capability_vocabulary(reg._capabilities)
    ie = _swap_registry(reg)
    assert ie._has_data_vocabulary("what is the total aum across clients")
    assert not ie._has_data_vocabulary("tell me a joke about penguins")
    # fallback: no capability map -> the seed keyword list still fires
    reg2 = CortexContextRegistry(seed_roster=SEED_ROSTER)
    ie = _swap_registry(reg2)
    assert ie._has_data_vocabulary("show me the portfolio")
