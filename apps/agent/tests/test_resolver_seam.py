"""The seam: `resolve_context` answered locally instead of by the platform.

This is the single point at which the two variants differ. Everything above it — the graph
node, the retry loop, the fund-name recovery, the decomposition plan — is byte-identical
between the two builds, which is what makes the comparison a measurement of two data paths
rather than of two different agents.

The shape is the contract. `_extract_mcp_text` reads `response.content[].text` and
`_parse_mcp_text` returns already-valid JSON unchanged, so emitting the resolver's JSON array
there means every downstream node behaves exactly as it does on the platform path.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.graph.nodes.mcp_fetch import _extract_mcp_text, _parse_mcp_text  # noqa: E402
from app.mcp import client as client_mod  # noqa: E402

ROWS = [{"name": "Ram Krishnan", "aum": 120}]


@pytest.fixture
def direct(monkeypatch):
    monkeypatch.setenv("DATA_RESOLVER", "direct")

    async def fake_resolve(question, **kw):
        return {"_parsed_text": json.dumps(ROWS), "success": True, "error": None,
                "latency_ms": 12, "queries": ["SELECT name, aum FROM clients LIMIT 200"]}

    import app.resolver.direct as d
    monkeypatch.setattr(d, "resolve_direct", fake_resolve)
    return fake_resolve


class TestTheSeamSwitchesOnlyTheDataPath:
    async def test_resolve_context_is_answered_locally(self, direct):
        result = await client_mod.MCPClient().call_tool(
            "resolve_context", {"query": "who has the most AUM?"}
        )

        assert result["success"] is True
        assert _parse_mcp_text(_extract_mcp_text(result)) == json.dumps(ROWS), (
            "the payload must arrive downstream as the same JSON array the platform "
            "path produces, or every node below this behaves differently"
        )

    async def test_the_generated_sql_is_surfaced(self, direct):
        """The UI shows what was asked of the data platform; this variant shows the SQL."""
        recorded = []
        import app.graph.nodes.stream_channel as sc
        original = sc.record_platform_call
        sc.record_platform_call = lambda c: recorded.append(c)
        try:
            await client_mod.MCPClient().call_tool("resolve_context", {"query": "q"})
        finally:
            sc.record_platform_call = original

        assert recorded and recorded[0]["tool"] == "direct_sql"
        assert any("SELECT" in s.upper() for s in recorded[0]["sql"])

    async def test_other_tools_are_untouched(self, direct, monkeypatch):
        """Only the data tool is intercepted. Skills and capabilities still go out."""
        seen = {}

        async def fake_platform(self, tool_name, arguments):
            seen["tool"] = tool_name
            return {"tool": tool_name, "success": True, "response": {}}

        monkeypatch.setattr(client_mod.MCPClient, "_call_tool_platform", fake_platform)
        await client_mod.MCPClient().call_tool("list_skills", {})

        assert seen["tool"] == "list_skills"


class TestPlatformModeIsTheDefault:
    async def test_without_the_switch_the_platform_is_used(self, monkeypatch):
        """A build that has not opted in must never quietly resolve locally."""
        monkeypatch.delenv("DATA_RESOLVER", raising=False)
        seen = {}

        async def fake_platform(self, tool_name, arguments):
            seen["tool"] = tool_name
            return {"tool": tool_name, "success": True, "response": {}}

        monkeypatch.setattr(client_mod.MCPClient, "_call_tool_platform", fake_platform)
        await client_mod.MCPClient().call_tool("resolve_context", {"query": "q"})

        assert seen["tool"] == "resolve_context", "silently fell back to local resolution"
