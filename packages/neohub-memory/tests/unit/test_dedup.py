"""Tests for MemoryDeduplicator with mocked search_fn and OpenAI."""
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from neo_memory_hub.dedup.deduplicator import DedupDecision, MemoryDeduplicator


class TestDedupDecision:
    """Test the DedupDecision dataclass."""

    def test_default_values(self):
        d = DedupDecision(action="NONE")
        assert d.action == "NONE"
        assert d.updated_memory is None
        assert d.reason == ""
        assert d.existing_memory_id is None

    def test_full_construction(self):
        d = DedupDecision(
            action="REPLACE",
            updated_memory="new text",
            reason="supersedes old",
            existing_memory_id="mem_123",
        )
        assert d.action == "REPLACE"
        assert d.updated_memory == "new text"


class TestMemoryDeduplicatorInit:
    """Test MemoryDeduplicator initialization."""

    def test_defaults(self):
        dd = MemoryDeduplicator()
        assert dd._similarity_threshold == 0.55
        assert dd._max_candidates == 5
        assert dd._llm_model == "gpt-4o-mini"

    def test_custom(self):
        dd = MemoryDeduplicator(
            similarity_threshold=0.7,
            max_candidates=10,
            llm_model="gpt-4o",
            dedup_prompt="Custom prompt",
        )
        assert dd._similarity_threshold == 0.7
        assert dd._dedup_prompt == "Custom prompt"


class TestFindAndResolve:
    """Test the full dedup flow."""

    @pytest.fixture
    def dedup(self):
        return MemoryDeduplicator(similarity_threshold=0.5, max_candidates=3)

    @pytest.mark.asyncio
    async def test_no_candidates_returns_none(self, dedup):
        search_fn = AsyncMock(return_value={"results": []})

        decision = await dedup.find_and_resolve(
            new_fact="User prefers dark mode",
            category="preference",
            investor_name="John",
            user_id="u1",
            agent_id="a1",
            search_fn=search_fn,
        )

        assert decision.action == "NONE"
        assert "no similar" in decision.reason
        search_fn.assert_called_once()

    @pytest.mark.asyncio
    async def test_search_fn_called_with_correct_params(self, dedup):
        search_fn = AsyncMock(return_value={"results": []})

        await dedup.find_and_resolve(
            new_fact="Prefers equity",
            category="preference",
            investor_name="Senthil Kumar",
            user_id="rm_001",
            agent_id="vic_l3",
            search_fn=search_fn,
            metadata_filters={"tenant_id": "t1"},
        )

        search_fn.assert_called_once_with(
            query="Prefers equity",
            user_id="rm_001",
            agent_id="vic_l3",
            categories=["preference"],
            limit=3,
            threshold=0.5,
            metadata_filters={"tenant_id": "t1", "investor_name": "Senthil Kumar"},
        )

    @pytest.mark.asyncio
    async def test_search_fn_without_investor_name(self, dedup):
        search_fn = AsyncMock(return_value={"results": []})

        await dedup.find_and_resolve(
            new_fact="Team guideline",
            category="procedural",
            investor_name=None,
            user_id="__team_default__",
            agent_id="a1",
            search_fn=search_fn,
        )

        search_fn.assert_called_once_with(
            query="Team guideline",
            user_id="__team_default__",
            agent_id="a1",
            categories=["procedural"],
            limit=3,
            threshold=0.5,
            metadata_filters=None,
        )

    @pytest.mark.asyncio
    @patch("openai.AsyncOpenAI")
    async def test_replace_decision(self, mock_openai_cls, dedup):
        search_fn = AsyncMock(return_value={
            "results": [
                {
                    "id": "mem_old",
                    "memory": "User likes light mode",
                    "metadata": {"categories": ["preference"], "investor_name": "John"},
                }
            ]
        })

        mock_client = AsyncMock()
        mock_openai_cls.return_value = mock_client
        llm_resp = MagicMock()
        llm_resp.choices = [MagicMock()]
        llm_resp.choices[0].message.content = json.dumps({
            "action": "REPLACE",
            "target_memory_id": "mem_old",
            "updated_memory": "User prefers dark mode",
            "reason": "Updated preference",
        })
        mock_client.chat.completions.create = AsyncMock(return_value=llm_resp)

        decision = await dedup.find_and_resolve(
            new_fact="User prefers dark mode",
            category="preference",
            investor_name="John",
            user_id="u1",
            agent_id="a1",
            search_fn=search_fn,
        )

        assert decision.action == "REPLACE"
        assert decision.existing_memory_id == "mem_old"
        assert decision.updated_memory == "User prefers dark mode"

    @pytest.mark.asyncio
    @patch("openai.AsyncOpenAI")
    async def test_keep_existing_decision(self, mock_openai_cls, dedup):
        search_fn = AsyncMock(return_value={
            "results": [
                {"id": "mem_1", "memory": "User likes dark mode", "metadata": {}},
            ]
        })

        mock_client = AsyncMock()
        mock_openai_cls.return_value = mock_client
        llm_resp = MagicMock()
        llm_resp.choices = [MagicMock()]
        llm_resp.choices[0].message.content = json.dumps({
            "action": "KEEP_EXISTING",
            "target_memory_id": "mem_1",
            "reason": "Same information",
        })
        mock_client.chat.completions.create = AsyncMock(return_value=llm_resp)

        decision = await dedup.find_and_resolve(
            new_fact="Dark mode preference",
            category="preference",
            investor_name=None,
            user_id="u1",
            agent_id="a1",
            search_fn=search_fn,
        )

        assert decision.action == "KEEP_EXISTING"
        assert decision.existing_memory_id == "mem_1"

    @pytest.mark.asyncio
    @patch("openai.AsyncOpenAI")
    async def test_merge_decision(self, mock_openai_cls, dedup):
        search_fn = AsyncMock(return_value={
            "results": [
                {"id": "mem_2", "memory": "User is conservative", "metadata": {}},
            ]
        })

        mock_client = AsyncMock()
        mock_openai_cls.return_value = mock_client
        llm_resp = MagicMock()
        llm_resp.choices = [MagicMock()]
        llm_resp.choices[0].message.content = json.dumps({
            "action": "MERGE",
            "target_memory_id": "mem_2",
            "updated_memory": "User is conservative and prefers fixed income",
            "reason": "Added detail",
        })
        mock_client.chat.completions.create = AsyncMock(return_value=llm_resp)

        decision = await dedup.find_and_resolve(
            new_fact="User prefers fixed income",
            category="persona",
            investor_name=None,
            user_id="u1",
            agent_id="a1",
            search_fn=search_fn,
        )

        assert decision.action == "MERGE"
        assert "fixed income" in decision.updated_memory

    @pytest.mark.asyncio
    @patch("openai.AsyncOpenAI")
    async def test_invalid_action_defaults_to_none(self, mock_openai_cls, dedup):
        search_fn = AsyncMock(return_value={
            "results": [{"id": "mem_3", "memory": "test", "metadata": {}}]
        })

        mock_client = AsyncMock()
        mock_openai_cls.return_value = mock_client
        llm_resp = MagicMock()
        llm_resp.choices = [MagicMock()]
        llm_resp.choices[0].message.content = json.dumps({
            "action": "INVALID_ACTION",
        })
        mock_client.chat.completions.create = AsyncMock(return_value=llm_resp)

        decision = await dedup.find_and_resolve(
            new_fact="test",
            category="persona",
            investor_name=None,
            user_id="u1",
            agent_id="a1",
            search_fn=search_fn,
        )

        assert decision.action == "NONE"

    @pytest.mark.asyncio
    @patch("openai.AsyncOpenAI")
    async def test_llm_failure_defaults_to_none(self, mock_openai_cls, dedup):
        search_fn = AsyncMock(return_value={
            "results": [{"id": "mem_4", "memory": "test", "metadata": {}}]
        })

        mock_client = AsyncMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create = AsyncMock(
            side_effect=Exception("API error")
        )

        decision = await dedup.find_and_resolve(
            new_fact="test",
            category="persona",
            investor_name=None,
            user_id="u1",
            agent_id="a1",
            search_fn=search_fn,
        )

        assert decision.action == "NONE"
        assert "LLM error" in decision.reason

    @pytest.mark.asyncio
    @patch("openai.AsyncOpenAI")
    async def test_missing_target_id_uses_first_candidate(self, mock_openai_cls, dedup):
        search_fn = AsyncMock(return_value={
            "results": [
                {"id": "first_mem", "memory": "test", "metadata": {}},
                {"id": "second_mem", "memory": "test2", "metadata": {}},
            ]
        })

        mock_client = AsyncMock()
        mock_openai_cls.return_value = mock_client
        llm_resp = MagicMock()
        llm_resp.choices = [MagicMock()]
        llm_resp.choices[0].message.content = json.dumps({
            "action": "REPLACE",
            "updated_memory": "Updated text",
            "reason": "test",
            # Note: no target_memory_id
        })
        mock_client.chat.completions.create = AsyncMock(return_value=llm_resp)

        decision = await dedup.find_and_resolve(
            new_fact="test",
            category="persona",
            investor_name=None,
            user_id="u1",
            agent_id="a1",
            search_fn=search_fn,
        )

        assert decision.action == "REPLACE"
        assert decision.existing_memory_id == "first_mem"
