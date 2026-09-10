"""Tests for Issue #9: Premature Cache Fallback fixes.

Validates three independent fixes:
1. generate.py — only routes to from_memory when intent is explicitly "from_memory"
2. mcp_fetch.py — no-tool-match returns a non-empty error entry in mcp_tool_calls
3. intent_enrichment.py — _conversation_has_data requires data-keyword proximity
"""

import sys
import os
import pytest
from unittest.mock import AsyncMock, patch, MagicMock

# Ensure agent package is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# --- Test 1: generate.py fallback condition ---

class FakeMessage:
    """Minimal message mock for testing."""
    def __init__(self, content: str, msg_type: str = "ai"):
        self.content = content
        self.type = msg_type


class TestGenerateResponseRouting:
    """Verify generate_response no longer falls into from_memory for fetch_data intent with empty MCP results."""

    @pytest.mark.asyncio
    async def test_fetch_data_with_empty_mcp_goes_to_data_path(self):
        """Root cause #1: fetch_data intent + empty mcp_results should NOT use from_memory path."""
        from app.graph.nodes.generate import generate_response

        state = {
            "query": "Show me Deepak Sharma's portfolio",
            "mcp_tool_calls": [],  # empty — previously triggered from_memory fallback
            "memory_context": "",
            "intent": "fetch_data",  # explicitly fetch_data, NOT from_memory
            "messages": [
                FakeMessage("Show me Rajesh Iyer's profile", "human"),
                FakeMessage("Here is Rajesh Iyer's profile: PAN: ABCDE1234F, Email: rajesh@example.com, Phone: 9876543210, KYC: Verified", "ai"),
                FakeMessage("Show me Deepak Sharma's portfolio", "human"),
            ],
            "latency_breakdown": {},
        }

        # Patch target is make_llm, not the provider class: every call site now goes
        # through one factory, so which provider it returns is a runtime setting. A
        # test pinned to a provider class would break on every provider change.
        with patch("app.graph.nodes.generate.make_llm") as MockLLM:
            mock_instance = AsyncMock()
            mock_response = MagicMock()
            mock_response.content = "I couldn't find relevant data for Deepak Sharma. Please try rephrasing your query."
            mock_instance.ainvoke = AsyncMock(return_value=mock_response)
            MockLLM.return_value = mock_instance

            result = await generate_response(state)

            # Verify LLM was called with SYSTEM_PROMPT (data path), not SYSTEM_PROMPT_FROM_MEMORY
            call_args = mock_instance.ainvoke.call_args[0][0]
            system_msg = call_args[0].content
            # The data-path system prompt contains "Format the data below"
            assert "Format the data below" in system_msg or "tool call failed" in system_msg.lower() or "read-only assistant" in system_msg.lower(), \
                f"Expected data-path system prompt, got from_memory path. System prompt starts with: {system_msg[:100]}"

    @pytest.mark.asyncio
    async def test_from_memory_intent_still_works(self):
        """Genuine from_memory routing should still work."""
        from app.graph.nodes.generate import generate_response

        state = {
            "query": "What is his portfolio allocation?",
            "mcp_tool_calls": [],
            "memory_context": "",
            "intent": "from_memory",  # explicitly from_memory
            "messages": [
                FakeMessage("Show me Rajesh Iyer's portfolio", "human"),
                FakeMessage("Here is Rajesh Iyer's portfolio: Equity 60%, Debt 30%, Gold 10%. Total AUM: ₹1.5Cr", "ai"),
                FakeMessage("What is his portfolio allocation?", "human"),
            ],
            "latency_breakdown": {},
        }

        with patch("app.graph.nodes.generate.make_llm") as MockLLM:
            mock_instance = AsyncMock()
            mock_response = MagicMock()
            mock_response.content = "Rajesh Iyer's allocation: Equity 60%, Debt 30%, Gold 10%"
            mock_instance.ainvoke = AsyncMock(return_value=mock_response)
            MockLLM.return_value = mock_instance

            result = await generate_response(state)

            # Verify from_memory system prompt was used
            call_args = mock_instance.ainvoke.call_args[0][0]
            system_msg = call_args[0].content
            assert "recalling saved notes" in system_msg.lower() or "saved notes" in system_msg.lower(), \
                f"Expected from_memory system prompt, got: {system_msg[:100]}"


# --- Test 2: mcp_fetch.py no-tool-match ---

class TestMCPFetchNoToolMatch:
    """Verify mcp_fetch returns a non-empty error entry when no tool matches."""

    @pytest.mark.asyncio
    async def test_no_tool_match_returns_error_entry(self):
        """Root cause #2: a no-tool-match must return a list with an error dict, not empty [].

        Tool selection is now deterministic (_looks_like_data_query) — the old LLM
        tool-selection hop was removed. The no-match path is reached for a non-data
        query (chitchat), and returns BEFORE any LLM or MCP-client call, so no mocks
        are needed."""
        from app.graph.nodes.mcp_fetch import mcp_fetch

        state = {
            "query": "hello",
            "enriched_query": "hello",
            "latency_breakdown": {},
            "messages": [],
        }

        result = await mcp_fetch(state)

        # mcp_tool_calls must NOT be empty
        assert result["mcp_tool_calls"], "mcp_tool_calls should not be empty on no-tool-match"
        assert len(result["mcp_tool_calls"]) == 1
        entry = result["mcp_tool_calls"][0]
        assert entry["success"] is False
        assert entry["tool"] == "no_match"


# --- Test 3: _conversation_has_data strictness ---

class TestConversationHasData:
    """Verify _conversation_has_data requires financial keywords near investor name."""

    def test_incidental_mention_does_not_trigger(self):
        """Root cause #3: investor mentioned in passing should NOT be treated as session hit."""
        from app.graph.nodes.intent_enrichment import _conversation_has_data

        # Rajesh Iyer's data response mentions Deepak Sharma as a co-investor
        messages = [
            FakeMessage(
                "Here is Rajesh Iyer's portfolio summary:\n"
                "- Total AUM: ₹1.5 Crore\n"
                "- Equity: 60% (includes co-investment with Deepak Sharma in ABC Fund)\n"
                "- Debt: 30%\n"
                "- Gold: 10%\n"
                "- Returns: 14.2% CAGR\n"
                "- Risk Profile: Moderate\n"
                + "x" * 200,  # pad to > 300 chars
                "ai"
            ),
        ]

        entities_deepak = {"investor_name": "Deepak Sharma", "data_type": "portfolio"}

        # Deepak Sharma is mentioned but NOT the subject — should return False
        assert _conversation_has_data(entities_deepak, messages) is False

    def test_primary_subject_triggers(self):
        """Investor that IS the subject of the data response should trigger session hit."""
        from app.graph.nodes.intent_enrichment import _conversation_has_data

        messages = [
            FakeMessage(
                "Here is Rajesh Iyer's portfolio summary:\n"
                "- Total AUM: ₹1.5 Crore\n"
                "- Equity: 60%\n"
                "- Debt: 30%\n"
                "- Gold: 10%\n"
                "- Returns: 14.2% CAGR\n"
                "- Risk Profile: Moderate\n"
                + "x" * 200,  # pad to > 300 chars
                "ai"
            ),
        ]

        entities_rajesh = {"investor_name": "Rajesh Iyer", "data_type": "portfolio"}

        # Rajesh Iyer IS the subject with financial keywords nearby — should return True
        assert _conversation_has_data(entities_rajesh, messages) is True

    def test_no_investor_name_returns_false(self):
        """Empty investor name should always return False."""
        from app.graph.nodes.intent_enrichment import _conversation_has_data

        messages = [FakeMessage("Some long response " * 50, "ai")]
        assert _conversation_has_data({"investor_name": ""}, messages) is False
        assert _conversation_has_data({"investor_name": None}, messages) is False

    def test_short_message_without_data_keywords_not_treated_as_data(self):
        """Short messages without data keywords should not be treated as data responses."""
        from app.graph.nodes.intent_enrichment import _conversation_has_data

        messages = [FakeMessage("Rajesh Iyer's looks good, let me know if you need more.", "ai")]
        entities = {"investor_name": "Rajesh Iyer"}
        assert _conversation_has_data(entities, messages) is False

    def test_short_message_with_matching_data_type_is_cache_hit(self):
        """Short messages with investor as subject + matching data_type ARE valid cache hits."""
        from app.graph.nodes.intent_enrichment import _conversation_has_data

        messages = [FakeMessage("Neha Sharma's risk profile: High", "ai")]
        entities = {"investor_name": "Neha Sharma", "data_type": "risk"}
        assert _conversation_has_data(entities, messages) is True

    def test_cached_risk_does_not_match_portfolio_request(self):
        """A cached risk profile should NOT satisfy a portfolio request."""
        from app.graph.nodes.intent_enrichment import _conversation_has_data

        messages = [FakeMessage("Neha Sharma's risk profile: High", "ai")]
        entities = {"investor_name": "Neha Sharma", "data_type": "portfolio"}
        assert _conversation_has_data(entities, messages) is False

    def test_cached_risk_does_not_match_profile_request(self):
        """A cached risk-only response should NOT satisfy a full profile request."""
        from app.graph.nodes.intent_enrichment import _conversation_has_data

        messages = [FakeMessage("Deepak Sharma - Investor ID: INV-001, Risk tolerance: Moderate", "ai")]
        entities = {"investor_name": "Deepak Sharma", "data_type": "profile"}
        # "profile" requires keywords like pan, email, phone, kyc, investment goal, etc.
        # This message only has risk info — should NOT match
        assert _conversation_has_data(entities, messages) is False

    def test_new_investor_after_session_has_others(self):
        """Test case #5: brand new investor in session with others should not be a cache hit."""
        from app.graph.nodes.intent_enrichment import _conversation_has_data

        messages = [
            FakeMessage(
                "Here is Rajesh Iyer's profile:\n- PAN: ABCDE1234F\n- Email: rajesh@example.com\n- Phone: 9876543210\n- KYC: Verified\n" + "x" * 200,
                "ai"
            ),
            FakeMessage(
                "Here is Nisha Singh's portfolio:\n- AUM: ₹2Cr\n- Equity: 70%\n- Debt: 20%\n- Gold: 10%\n" + "x" * 200,
                "ai"
            ),
        ]

        # Deepak Sharma was never discussed — must return False
        entities = {"investor_name": "Deepak Sharma", "data_type": "portfolio"}
        assert _conversation_has_data(entities, messages) is False


# --- Test 4: data_type inheritance from clarification context ---

class TestDataTypeInheritance:
    """Entity gating in _identify_missing.

    Contract (post-refactor): resolve_context is a smart NL resolver and the
    follow-up carry-forward of WHAT data is wanted is handled by the transcript-
    driven LLM rewrite (FOLLOWUP_RESOLVE_PROMPT), NOT by stuffing a keyword into
    entities["data_type"]. So when a client name is known, _identify_missing
    attempts the fetch (returns []) instead of blocking on a missing data_type.
    The keyword inheritance (_infer_data_type_from_history) survives ONLY as a
    name-rescue fallback in the pronoun branch (no name present)."""

    def test_named_followup_is_not_blocked(self):
        """After 'what is Mr. Sharma's risk profile' → clarification → 'Neha Sharma's':
        the name is known, so attempt the fetch (the LLM rewrite carries 'risk
        profile' forward from the transcript). Never block on data_type."""
        from app.graph.nodes.intent_enrichment import _identify_missing

        messages = [
            FakeMessage("what is Mr. Sharma's risk profile", "human"),
            FakeMessage("Which Sharma do you mean—Deepak Sharma, Neha Sharma, Aditya Sharma?", "ai"),
            FakeMessage("can you give me Neha Sharma's", "human"),
        ]
        entities = {"investor_name": "Neha Sharma", "data_type": None}
        missing = _identify_missing(entities, "can you give me Neha Sharma's", messages)

        assert missing == []

    def test_named_reply_after_clarification_is_not_blocked(self):
        """After 'show me Sharma's portfolio' → clarification → 'Deepak Sharma':
        name known → attempt the fetch, no clarification."""
        from app.graph.nodes.intent_enrichment import _identify_missing

        messages = [
            FakeMessage("show me Sharma's portfolio", "human"),
            FakeMessage("Which Sharma do you mean?", "ai"),
            FakeMessage("Deepak Sharma", "human"),
        ]
        entities = {"investor_name": "Deepak Sharma", "data_type": None}
        missing = _identify_missing(entities, "Deepak Sharma", messages)

        assert missing == []

    def test_bare_name_attempts_fetch_instead_of_blocking(self):
        """A known name with no prior data context is enough to attempt the fetch —
        resolve_context can return a sensible client summary — so do NOT block."""
        from app.graph.nodes.intent_enrichment import _identify_missing

        messages = [FakeMessage("Neha Sharma", "human")]
        entities = {"investor_name": "Neha Sharma", "data_type": None}
        missing = _identify_missing(entities, "Neha Sharma", messages)

        assert missing == []

    def test_pronoun_followup_inherits_data_type(self):
        """The surviving inheritance path: NO name, just a pronoun → inherit data_type
        from the prior question so we can fetch instead of asking 'who?'."""
        from app.graph.nodes.intent_enrichment import _identify_missing

        messages = [
            FakeMessage("what is Mr. Sharma's risk profile", "human"),
            FakeMessage("Which Sharma do you mean?", "ai"),
            FakeMessage("what about her", "human"),
        ]
        entities = {"investor_name": None, "data_type": None}
        missing = _identify_missing(entities, "what about her", messages)

        # Pronoun with no name → inherit data_type from history ("risk profile"
        # contains both "risk" and "profile"; either is valid), so it doesn't block.
        assert entities["data_type"] in ("risk", "profile")
        assert missing == []


# --- Test 5: Integration scenario (full routing) ---

class TestIntegrationRouting:
    """End-to-end routing tests — verify the full pipeline doesn't short-circuit."""

    @pytest.mark.asyncio
    async def test_different_investor_routes_to_mcp(self):
        """After fetching Rajesh, asking about Deepak must route to fetch_data (not from_memory)."""
        from app.graph.nodes.intent_enrichment import intent_enrichment

        state = {
            "query": "Show me Deepak Sharma's portfolio",
            "messages": [
                FakeMessage("Show me Rajesh Iyer's profile", "human"),
                FakeMessage(
                    "Here is Rajesh Iyer's profile:\n- PAN: ABCDE1234F\n- Email: rajesh@example.com\n- Phone: 9876543210\n- KYC: Verified\n" + "x" * 200,
                    "ai"
                ),
                FakeMessage("Show me Deepak Sharma's portfolio", "human"),
            ],
            "memory_context": "",
            "investor_name": None,
            "latency_breakdown": {},
        }

        with patch("app.graph.nodes.intent_enrichment.make_llm") as MockLLM:
            # Mock entity extraction
            mock_structured = AsyncMock()
            mock_entities = MagicMock()
            mock_entities.model_dump.return_value = {
                "investor_name": "Deepak Sharma",
                "data_type": "portfolio",
                "specific_metric": None,
                "timeframe": None,
                "filters": [],
                "comparison": None,
                "confidence": 0.95,
            }
            mock_structured.ainvoke = AsyncMock(return_value=mock_entities)

            # Mock enrichment query
            mock_enrich_response = MagicMock()
            mock_enrich_response.content = "What is Deepak Sharma's portfolio?"

            mock_instance = MagicMock()
            mock_instance.with_structured_output.return_value = mock_structured
            mock_instance.ainvoke = AsyncMock(return_value=mock_enrich_response)
            MockLLM.return_value = mock_instance

            result = await intent_enrichment(state)

            # Must route to fetch_data, NOT from_memory
            assert result["intent"] == "fetch_data", \
                f"Expected fetch_data intent but got: {result['intent']}"
            assert result["enriched_query"], "Enriched query should not be empty for fetch_data"


# --- Test 6: Always-on context resolution (extractor-driven) ---

class TestAlwaysOnContextResolution:
    """The stateless MCP must receive a self-contained query. The extractor now resolves
    EVERY data turn's back-references (not just regex-detectable pronouns), and the enrichment
    node uses that rewrite — guarded so an invented client name can never reach the platform."""

    def test_rewrite_with_seen_client_is_accepted(self):
        """A rewrite that uses a client actually named in the conversation passes the guard."""
        from app.graph.nodes.intent_enrichment import _is_safe_followup_rewrite

        messages = [
            FakeMessage("Show me Ram Krishnan's portfolio", "human"),
            FakeMessage("Ram Krishnan's portfolio: Equity 60%, Debt 40%." + "x" * 60, "ai"),
            FakeMessage("what about him", "human"),
        ]
        assert _is_safe_followup_rewrite(
            "What is Ram Krishnan's portfolio value?", "what about him", messages
        ) is True

    def test_rewrite_inventing_unseen_client_is_rejected(self):
        """A rewrite that injects a client never seen in the session is rejected — so a
        hallucinated subject can never be sent to the stateless platform."""
        from app.graph.nodes.intent_enrichment import _is_safe_followup_rewrite

        messages = [
            FakeMessage("Show me Ram Krishnan's portfolio", "human"),
            FakeMessage("Ram Krishnan's portfolio: Equity 60%, Debt 40%." + "x" * 60, "ai"),
            FakeMessage("what about him", "human"),
        ]
        # "Arjun Mehra" is a known roster name but never appeared in THIS conversation.
        assert _is_safe_followup_rewrite(
            "What is Arjun Mehra's portfolio value?", "what about him", messages
        ) is False

    @pytest.mark.asyncio
    async def test_implied_subject_followup_is_resolved_for_mcp(self):
        """Regression for the reported bug: an implied-subject follow-up with NO pronoun and
        NO 'those'/'the same' cue ('what about last quarter') used to be sent to MCP verbatim
        because the old regex gate didn't fire. The extractor now rewrites it, and the node
        must hand the RESOLVED query (carrying the prior client) to MCP."""
        from app.graph.nodes.intent_enrichment import intent_enrichment

        resolved = "What was Ram Krishnan's portfolio performance last quarter?"
        state = {
            "query": "what about last quarter",
            "messages": [
                FakeMessage("Show me Ram Krishnan's portfolio performance this year", "human"),
                FakeMessage(
                    "Ram Krishnan's portfolio returned 14% this year across equity and debt."
                    + "x" * 60,
                    "ai",
                ),
                FakeMessage("what about last quarter", "human"),
            ],
            "memory_context": "",
            "investor_name": None,
            "latency_breakdown": {},
        }

        with patch("app.graph.nodes.intent_enrichment.make_llm") as MockLLM:
            mock_structured = AsyncMock()
            mock_entities = MagicMock()
            mock_entities.model_dump.return_value = {
                "investor_name": "Ram Krishnan",
                "data_type": "performance",
                "specific_metric": None,
                "timeframe": "last quarter",
                "filters": [],
                "fund_name": None,
                "comparison": None,
                "confidence": 0.9,
                "route": "fetch_data",
                "intent_kind": "request_data",
                # The LLM under-labels the speech act as plain 'data' (no pronoun, no cue) —
                # exactly the case the old regex gate missed — but still emits the rewrite.
                "message_kind": "data",
                "standalone_query": resolved,
            }
            mock_structured.ainvoke = AsyncMock(return_value=mock_entities)

            mock_instance = MagicMock()
            mock_instance.with_structured_output.return_value = mock_structured
            # If this were hit, it'd mean the deterministic rewrite path was skipped.
            mock_instance.ainvoke = AsyncMock(
                return_value=MagicMock(content="SHOULD-NOT-BE-USED")
            )
            MockLLM.return_value = mock_instance

            result = await intent_enrichment(state)

            assert result["intent"] == "fetch_data", f"got {result['intent']}"
            # MCP must receive the resolved, self-contained query — not the bare follow-up.
            assert result["enriched_query"] == resolved
            assert "Ram Krishnan" in result["enriched_query"]
            # The deterministic extractor rewrite was used, not the fallback LLM rewrite call.
            mock_instance.ainvoke.assert_not_called()


class TestQuickFactsStandaloneOnly:
    """Quick Facts is a standalone-lookup mode: in-session conversation history is dropped for
    the turn, so no prior-turn context reaches the extractor and no follow-up carries over.
    Client Insights / Deep Insight are untouched and still pass conversation context."""

    def _state_with_history(self, mode: str) -> dict:
        return {
            "query": "what about last quarter",
            "mode": mode,
            "messages": [
                FakeMessage("Show me Ram Krishnan's portfolio performance this year", "human"),
                FakeMessage("Ram Krishnan's portfolio returned 14% this year." + "x" * 60, "ai"),
                FakeMessage("what about last quarter", "human"),
            ],
            # Non-empty so enrichment skips the inline get_summary() preference fetch.
            "memory_context": "(no standing preferences)",
            "user_id": "rm1",
            "investor_name": None,
            "latency_breakdown": {},
        }

    async def _captured_extraction_input(self, mode: str) -> str:
        """Run intent_enrichment with a mocked extractor and return the HumanMessage text the
        extractor received (which contains 'Recent conversation context' only if history was used)."""
        from app.graph.nodes.intent_enrichment import intent_enrichment

        with patch("app.graph.nodes.intent_enrichment.make_llm") as MockLLM:
            mock_structured = AsyncMock()
            mock_entities = MagicMock()
            mock_entities.model_dump.return_value = {
                "investor_name": None, "data_type": "performance", "specific_metric": None,
                "timeframe": "last quarter", "filters": [], "fund_name": None, "comparison": None,
                "confidence": 0.9, "route": "fetch_data", "intent_kind": "request_data",
                "message_kind": "data", "standalone_query": None,
            }
            mock_structured.ainvoke = AsyncMock(return_value=mock_entities)
            mock_instance = MagicMock()
            mock_instance.with_structured_output.return_value = mock_structured
            mock_instance.ainvoke = AsyncMock(return_value=MagicMock(content=""))
            MockLLM.return_value = mock_instance

            await intent_enrichment(self._state_with_history(mode))

            # The extractor is invoked with [SystemMessage, HumanMessage]; grab the HumanMessage.
            call_args = mock_structured.ainvoke.call_args[0][0]
            return call_args[1].content

    @pytest.mark.asyncio
    async def test_quick_facts_drops_conversation_context(self):
        """Quick Facts: prior turns are NOT fed to the extractor (standalone query)."""
        human = await self._captured_extraction_input("quick")
        assert "Recent conversation context" not in human
        assert "Ram Krishnan" not in human

    @pytest.mark.asyncio
    async def test_client_insights_keeps_conversation_context(self):
        """Client Insights is untouched: prior turns ARE fed to the extractor."""
        human = await self._captured_extraction_input("client")
        assert "Recent conversation context" in human
        assert "Ram Krishnan" in human


class TestPriorEnrichedInContext:
    """The prior turn's resolved (MCP-bound) query is fed to the resolver as a
    'User (resolved):' line, so a follow-up inherits a self-contained antecedent.
    It is bounded to ONE prior exchange and stays off when there's no history."""

    def test_resolved_line_injected(self):
        from app.graph.nodes.intent_enrichment import _format_exchange
        prior = [
            FakeMessage("show his holdings", "human"),
            FakeMessage("Here are the holdings: A, B, C", "ai"),
            FakeMessage("only equity ones", "human"),  # current
        ]
        out = _format_exchange(prior, "Venkat Krishnan's holdings")
        assert "User: show his holdings" in out
        assert "User (resolved): Venkat Krishnan's holdings" in out
        assert out.index("User (resolved):") < out.index("Assistant:")  # order: raw, resolved, answer

    def test_no_duplicate_when_enriched_equals_raw(self):
        from app.graph.nodes.intent_enrichment import _format_exchange
        prior = [FakeMessage("list funds", "human"), FakeMessage("x", "ai"), FakeMessage("cur", "human")]
        out = _format_exchange(prior, "list funds")
        assert "User (resolved):" not in out

    def test_empty_when_no_prior_exchange(self):
        """No history (e.g. Quick Facts blanks it) -> empty, so nothing leaks in."""
        from app.graph.nodes.intent_enrichment import _format_exchange
        assert _format_exchange([FakeMessage("only current", "human")], "something") == ""
        assert _format_exchange([], "something") == ""


class TestSelectFromPriorList:
    """A follow-up that selects/filters/ranks over the rows just shown is answered in-session
    (from_memory) — never sent to the stateless platform, which can't honour 'those'."""

    def test_detector_matches_selections(self):
        from app.graph.nodes.intent_enrichment import _is_select_from_prior_list as f
        for q in ["which of those are large cap mutual funds", "rank those by current value",
                  "of those, which have the highest value", "which of the above are equity",
                  "which ones are large cap", "filter those by sector"]:
            assert f(q), q

    def test_detector_ignores_fresh_queries(self):
        from app.graph.nodes.intent_enrichment import _is_select_from_prior_list as f
        for q in ["show Venkat Krishnan holdings", "what about last quarter",
                  "list all large cap funds in his portfolio", "his equity holdings"]:
            assert not f(q), q

    @pytest.mark.asyncio
    async def test_select_from_prior_list_routes_to_from_memory(self):
        """'which of those are X' with a prior data answer -> from_memory (no MCP)."""
        from app.graph.nodes.intent_enrichment import intent_enrichment
        state = {
            "query": "which of those are large cap mutual funds",
            "messages": [
                FakeMessage("Show Venkat Krishnan's holdings", "human"),
                FakeMessage("Venkat Krishnan's holdings: Maruti Suzuki, Dixon Technologies, "
                            "Trent Ltd, with market values." + "x" * 60, "ai"),
                FakeMessage("which of those are large cap mutual funds", "human"),
            ],
            "memory_context": "", "investor_name": None, "latency_breakdown": {},
        }
        with patch("app.graph.nodes.intent_enrichment.make_llm") as MockLLM:
            mock_structured = AsyncMock()
            mock_entities = MagicMock()
            mock_entities.model_dump.return_value = {
                "investor_name": "Venkat Krishnan", "data_type": "holdings", "route": "fetch_data",
                "intent_kind": "request_data", "message_kind": "followup", "standalone_query": None,
                "confidence": 0.9,
            }
            mock_structured.ainvoke = AsyncMock(return_value=mock_entities)
            mi = MagicMock()
            mi.with_structured_output.return_value = mock_structured
            mi.ainvoke = AsyncMock(return_value=MagicMock(content="SHOULD-NOT-BE-USED"))
            MockLLM.return_value = mi

            result = await intent_enrichment(state)
            assert result["intent"] == "from_memory", f"got {result['intent']}"
            assert not result.get("enriched_query")  # nothing sent to the platform


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
