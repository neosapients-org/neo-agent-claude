"""Tests for HighLevelMemoryConnector with mocked Mem0 backend."""
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from neo_memory_hub.config.hub_config import MemoryHubConfig, SalienceConfig, ExtractionConfig, DedupConfig, BufferingConfig, IsolationConfig, FeatureConfig
from neo_memory_hub.connector import HighLevelMemoryConnector


def _make_extraction_response(facts):
    """Create a mock LLM extraction response."""
    msg = MagicMock()
    msg.content = json.dumps({"memories": facts})
    choice = MagicMock()
    choice.message = msg
    resp = MagicMock()
    resp.choices = [choice]
    return resp


def _make_dedup_response(action, target_id=None, updated_memory=None, reason=""):
    """Create a mock LLM dedup response."""
    msg = MagicMock()
    msg.content = json.dumps({
        "action": action,
        "target_memory_id": target_id,
        "updated_memory": updated_memory,
        "reason": reason,
    })
    choice = MagicMock()
    choice.message = msg
    resp = MagicMock()
    resp.choices = [choice]
    return resp


@pytest.fixture
def config():
    """Config with salience gating disabled for simpler testing."""
    return MemoryHubConfig(
        salience=SalienceConfig(enabled=False),
        extraction=ExtractionConfig(enabled=True),
        dedup=DedupConfig(enabled=False),
        buffering=BufferingConfig(enabled=False),
        isolation=IsolationConfig(require_user_id=False),
    )


@pytest.fixture
def config_with_salience():
    """Config with salience gating enabled."""
    return MemoryHubConfig(
        salience=SalienceConfig(enabled=True, min_threshold=0.3),
        extraction=ExtractionConfig(enabled=True),
        dedup=DedupConfig(enabled=False),
        buffering=BufferingConfig(enabled=False),
        isolation=IsolationConfig(require_user_id=False),
    )


@pytest.fixture
def config_with_dedup():
    """Config with dedup enabled."""
    return MemoryHubConfig(
        salience=SalienceConfig(enabled=False),
        extraction=ExtractionConfig(enabled=True),
        dedup=DedupConfig(enabled=True, similarity_threshold=0.5),
        buffering=BufferingConfig(enabled=False),
        isolation=IsolationConfig(require_user_id=False),
    )


class TestHighLevelConnectorInit:
    """Test connector initialization."""

    def test_creates_with_default_config(self):
        hlc = HighLevelMemoryConnector()
        assert hlc._config is not None
        assert hlc.is_ready is False

    def test_creates_with_custom_config(self, config):
        hlc = HighLevelMemoryConnector(config=config)
        assert hlc._config.salience.enabled is False

    def test_config_property(self, config):
        hlc = HighLevelMemoryConnector(config=config)
        assert hlc.config is config


class TestStoreExchange:
    """Test store_exchange pipeline with mocked OpenAI and connector."""

    @pytest.mark.asyncio
    @patch("openai.AsyncOpenAI")
    @patch("neo_memory_hub.connector.NeoMemoryConnector")
    async def test_basic_store_exchange(self, mock_connector_cls, mock_openai_cls, config):
        # Setup mock connector
        mock_connector = AsyncMock()
        mock_connector.initialize = AsyncMock()
        mock_connector.close = AsyncMock()
        mock_connector.store_fact = AsyncMock(return_value={"results": [{"id": "new_1"}]})
        mock_connector.search = AsyncMock(return_value={"results": []})
        mock_connector._memory = MagicMock()
        mock_connector._memory._mem0 = None
        mock_connector_cls.return_value = mock_connector

        # Setup mock OpenAI
        mock_client = AsyncMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create = AsyncMock(
            return_value=_make_extraction_response([
                {
                    "category": "persona",
                    "investor_name": "John",
                    "key": "risk",
                    "value": "Conservative",
                    "salience": 0.8,
                    "reasoning": "Core identity",
                },
            ])
        )

        hlc = HighLevelMemoryConnector(config=config)
        hlc._connector = mock_connector
        hlc._initialized = True
        # Manually initialize extractor
        from neo_memory_hub.extraction import FactExtractor
        hlc._extractor = FactExtractor()
        from neo_memory_hub.retrieval.strategy import MultiCategoryRetriever
        from neo_memory_hub.retrieval.formatter import ProfileAssembler, ContextFormatter
        hlc._retriever = MultiCategoryRetriever()
        hlc._assembler = ProfileAssembler()
        hlc._formatter = ContextFormatter()

        result = await hlc.store_exchange(
            query="Tell me about John",
            response="John is a conservative investor",
            user_id="u1",
        )

        assert result["facts_extracted"] == 1
        assert result["facts_stored"] == 1
        assert result["investor_name"] == "John"
        mock_connector.store_fact.assert_called_once()

    @pytest.mark.asyncio
    async def test_store_exchange_not_ready(self, config):
        hlc = HighLevelMemoryConnector(config=config)
        result = await hlc.store_exchange(query="q", response="r", user_id="u1")
        assert result["facts_stored"] == 0

    @pytest.mark.asyncio
    async def test_store_exchange_storage_disabled(self):
        config = MemoryHubConfig(
            features=FeatureConfig(storage_enabled=False),
            isolation=IsolationConfig(require_user_id=False),
        )
        hlc = HighLevelMemoryConnector(config=config)
        hlc._initialized = True
        hlc._connector = MagicMock()
        result = await hlc.store_exchange(query="q", response="r", user_id="u1")
        assert result["facts_stored"] == 0

    @pytest.mark.asyncio
    async def test_store_exchange_requires_user_id(self):
        config = MemoryHubConfig(
            isolation=IsolationConfig(require_user_id=True),
        )
        hlc = HighLevelMemoryConnector(config=config)
        hlc._initialized = True
        hlc._connector = MagicMock()
        result = await hlc.store_exchange(query="q", response="r", user_id="")
        assert result["facts_stored"] == 0

    @pytest.mark.asyncio
    async def test_store_exchange_requires_tenant_id_in_strict_mode(self):
        config = MemoryHubConfig(
            isolation=IsolationConfig(require_user_id=False, strict_tenant_isolation=True),
        )
        hlc = HighLevelMemoryConnector(config=config)
        hlc._initialized = True
        hlc._connector = MagicMock()
        result = await hlc.store_exchange(query="q", response="r", user_id="u1")
        assert result["facts_stored"] == 0


class TestStoreExchangeWithSalience:
    """Test salience gating in store_exchange."""

    @pytest.mark.asyncio
    @patch("openai.AsyncOpenAI")
    async def test_low_salience_fact_skipped(self, mock_openai_cls, config_with_salience):
        mock_client = AsyncMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create = AsyncMock(
            return_value=_make_extraction_response([
                {"category": "persona", "value": "Likes coffee", "salience": 0.1},
            ])
        )

        mock_connector = AsyncMock()
        mock_connector.store_fact = AsyncMock(return_value={"results": [{"id": "new_1"}]})

        hlc = HighLevelMemoryConnector(config=config_with_salience)
        hlc._connector = mock_connector
        hlc._initialized = True

        from neo_memory_hub.extraction import FactExtractor
        from neo_memory_hub.core.storage_gateway import StorageGateway, StorageGatewayConfig
        hlc._extractor = FactExtractor()
        hlc._gateway = StorageGateway(StorageGatewayConfig(min_salience_threshold=0.3))

        result = await hlc.store_exchange(
            query="q", response="r", user_id="u1"
        )

        assert result["facts_stored"] == 0
        assert result["facts_skipped"] == 1
        mock_connector.store_fact.assert_not_called()


class TestRetrieveContext:
    """Test retrieve_context with mocked search."""

    @pytest.mark.asyncio
    async def test_basic_retrieval(self, config):
        mock_connector = AsyncMock()
        mock_connector.search = AsyncMock(return_value={
            "results": [
                {
                    "id": "m1",
                    "memory": "Conservative investor",
                    "metadata": {"profile_key": "risk", "salience_score": 0.9},
                },
            ]
        })

        hlc = HighLevelMemoryConnector(config=config)
        hlc._connector = mock_connector
        hlc._initialized = True

        from neo_memory_hub.retrieval.strategy import MultiCategoryRetriever
        from neo_memory_hub.retrieval.formatter import ProfileAssembler, ContextFormatter
        hlc._retriever = MultiCategoryRetriever()
        hlc._assembler = ProfileAssembler()
        hlc._formatter = ContextFormatter()

        result = await hlc.retrieve_context(
            query="Tell me about the client",
            user_id="u1",
        )

        assert result["count"] > 0
        assert result["context"] != ""
        assert len(result["memories"]) > 0

    @pytest.mark.asyncio
    async def test_retrieval_disabled(self):
        config = MemoryHubConfig(
            features=FeatureConfig(retrieval_enabled=False),
            isolation=IsolationConfig(require_user_id=False),
        )
        hlc = HighLevelMemoryConnector(config=config)
        hlc._connector = MagicMock()
        hlc._initialized = True

        result = await hlc.retrieve_context(query="q", user_id="u1")
        assert result["count"] == 0
        assert result["context"] == ""

    @pytest.mark.asyncio
    async def test_retrieval_not_ready(self, config):
        hlc = HighLevelMemoryConnector(config=config)
        result = await hlc.retrieve_context(query="q", user_id="u1")
        assert result["count"] == 0

    @pytest.mark.asyncio
    async def test_retrieval_empty_results(self, config):
        mock_connector = AsyncMock()
        mock_connector.search = AsyncMock(return_value={"results": []})

        hlc = HighLevelMemoryConnector(config=config)
        hlc._connector = mock_connector
        hlc._initialized = True

        from neo_memory_hub.retrieval.strategy import MultiCategoryRetriever
        from neo_memory_hub.retrieval.formatter import ProfileAssembler, ContextFormatter
        hlc._retriever = MultiCategoryRetriever()
        hlc._assembler = ProfileAssembler()
        hlc._formatter = ContextFormatter()

        result = await hlc.retrieve_context(query="q", user_id="u1")
        assert result["count"] == 0
        assert result["context"] == ""


class TestBuffering:
    """Test exchange buffering."""

    @pytest.mark.asyncio
    async def test_buffer_exchange_without_buffer_falls_through(self, config):
        """When buffering is disabled, buffer_exchange should call store_exchange directly."""
        hlc = HighLevelMemoryConnector(config=config)
        hlc._initialized = True
        hlc._connector = MagicMock()
        hlc._buffer = None
        hlc._extractor = None

        result = await hlc.buffer_exchange(
            query="q", response="r", user_id="u1"
        )
        # Without extractor, will return 0 stored
        assert result["facts_stored"] == 0

    @pytest.mark.asyncio
    async def test_buffer_exchange_adds_to_buffer(self):
        config = MemoryHubConfig(
            buffering=BufferingConfig(enabled=True, flush_threshold=5),
            isolation=IsolationConfig(require_user_id=False),
        )
        hlc = HighLevelMemoryConnector(config=config)
        hlc._initialized = True
        hlc._connector = MagicMock()

        from neo_memory_hub.buffering import ExchangeBuffer
        hlc._buffer = ExchangeBuffer(flush_threshold=5)

        result = await hlc.buffer_exchange(
            query="q", response="r", user_id="u1"
        )
        assert result["buffered"] is True
        assert result["buffer_size"] == 1


class TestFlushSessionBuffer:
    """Test flush_session_buffer."""

    @pytest.mark.asyncio
    async def test_flush_empty(self, config):
        hlc = HighLevelMemoryConnector(config=config)
        hlc._initialized = True
        hlc._buffer = None

        result = await hlc.flush_session_buffer(user_id="u1")
        assert result["flushed"] == 0


class TestLowLevelCRUD:
    """Test low-level CRUD delegation."""

    @pytest.mark.asyncio
    async def test_search_delegates(self, config):
        hlc = HighLevelMemoryConnector(config=config)
        hlc._connector = AsyncMock()
        hlc._connector.search = AsyncMock(return_value={"results": []})
        hlc._initialized = True

        await hlc.search("query", user_id="u1")
        hlc._connector.search.assert_called_once_with(
            "query", user_id="u1", agent_id="default_agent"
        )

    @pytest.mark.asyncio
    async def test_get_delegates(self, config):
        hlc = HighLevelMemoryConnector(config=config)
        hlc._connector = AsyncMock()
        hlc._connector.get = AsyncMock(return_value={"id": "m1"})
        hlc._initialized = True

        result = await hlc.get("m1")
        hlc._connector.get.assert_called_once_with("m1")

    @pytest.mark.asyncio
    async def test_get_all_delegates(self, config):
        hlc = HighLevelMemoryConnector(config=config)
        hlc._connector = AsyncMock()
        hlc._connector.get_all = AsyncMock(return_value={"results": []})
        hlc._initialized = True

        await hlc.get_all(user_id="u1")
        hlc._connector.get_all.assert_called_once_with(
            user_id="u1", agent_id="default_agent"
        )

    @pytest.mark.asyncio
    async def test_delete_delegates(self, config):
        hlc = HighLevelMemoryConnector(config=config)
        hlc._connector = AsyncMock()
        hlc._connector.delete = AsyncMock(return_value={"deleted": True})
        hlc._initialized = True

        await hlc.delete("m1")
        hlc._connector.delete.assert_called_once_with("m1")

    @pytest.mark.asyncio
    async def test_update_delegates(self, config):
        hlc = HighLevelMemoryConnector(config=config)
        hlc._connector = AsyncMock()
        hlc._connector.update = AsyncMock(return_value={"updated": True})
        hlc._initialized = True

        await hlc.update("m1", "new data")
        hlc._connector.update.assert_called_once_with("m1", data="new data")

    @pytest.mark.asyncio
    async def test_store_fact_delegates(self, config):
        hlc = HighLevelMemoryConnector(config=config)
        hlc._connector = AsyncMock()
        hlc._connector.store_fact = AsyncMock(return_value={"results": []})
        hlc._initialized = True

        await hlc.store_fact("fact text", user_id="u1")
        hlc._connector.store_fact.assert_called_once_with(
            "fact text", user_id="u1", agent_id="default_agent"
        )

    @pytest.mark.asyncio
    async def test_search_with_tenant_isolation(self):
        cfg = MemoryHubConfig(
            isolation=IsolationConfig(
                require_user_id=True, strict_tenant_isolation=True
            ),
            buffering=BufferingConfig(enabled=False),
        )
        hlc = HighLevelMemoryConnector(config=cfg)
        hlc._connector = AsyncMock()
        hlc._connector.search = AsyncMock(return_value={"results": []})
        hlc._initialized = True

        # Missing tenant_id → error
        with pytest.raises(ValueError, match="tenant_id is required"):
            await hlc.search("q", user_id="u1")

        # With tenant_id → passes and adds metadata filter
        await hlc.search("q", user_id="u1", tenant_id="t1")
        hlc._connector.search.assert_called_once()
        call_kwargs = hlc._connector.search.call_args
        assert call_kwargs.kwargs.get("metadata_filters", {}).get("tenant_id") == "t1"

    @pytest.mark.asyncio
    async def test_store_fact_requires_user_id(self):
        cfg = MemoryHubConfig(
            isolation=IsolationConfig(require_user_id=True),
            buffering=BufferingConfig(enabled=False),
        )
        hlc = HighLevelMemoryConnector(config=cfg)
        hlc._connector = AsyncMock()
        hlc._initialized = True

        with pytest.raises(ValueError, match="user_id is required"):
            await hlc.store_fact("fact", user_id="")

    @pytest.mark.asyncio
    async def test_get_all_requires_user_id(self):
        cfg = MemoryHubConfig(
            isolation=IsolationConfig(require_user_id=True),
            buffering=BufferingConfig(enabled=False),
        )
        hlc = HighLevelMemoryConnector(config=cfg)
        hlc._connector = AsyncMock()
        hlc._initialized = True

        with pytest.raises(ValueError, match="user_id is required"):
            await hlc.get_all(user_id="")


class TestHistory:
    """Test history API delegation."""

    @pytest.mark.asyncio
    async def test_history_without_manager(self, config):
        hlc = HighLevelMemoryConnector(config=config)
        hlc._history = None
        result = await hlc.history("m1")
        assert result == []

    @pytest.mark.asyncio
    async def test_history_with_manager(self, config):
        hlc = HighLevelMemoryConnector(config=config)
        hlc._history = AsyncMock()
        hlc._history.get_memory_history = AsyncMock(return_value=[{"event": "ADD"}])
        result = await hlc.history("m1")
        assert len(result) == 1

    def test_get_history_stats_without_manager(self, config):
        hlc = HighLevelMemoryConnector(config=config)
        hlc._history = None
        stats = hlc.get_history_stats()
        assert stats["total"] == 0
        assert stats["history_enabled"] is False


class TestLifecycle:
    """Test lifecycle operations."""

    @pytest.mark.asyncio
    async def test_cleanup_expired_delegates(self, config):
        hlc = HighLevelMemoryConnector(config=config)
        hlc._connector = AsyncMock()
        hlc._connector.cleanup_expired = AsyncMock(return_value={"deleted": 5})

        result = await hlc.cleanup_expired(user_id="u1")
        assert result["deleted"] == 5

    @pytest.mark.asyncio
    async def test_count_memories_delegates(self, config):
        hlc = HighLevelMemoryConnector(config=config)
        hlc._connector = AsyncMock()
        hlc._connector.count_memories = AsyncMock(return_value={"total": 10})

        result = await hlc.count_memories(user_id="u1")
        assert result["total"] == 10

    @pytest.mark.asyncio
    async def test_cleanup_without_connector(self, config):
        hlc = HighLevelMemoryConnector(config=config)
        result = await hlc.cleanup_expired()
        assert result == {}

    @pytest.mark.asyncio
    async def test_count_without_connector(self, config):
        hlc = HighLevelMemoryConnector(config=config)
        result = await hlc.count_memories()
        assert result["total"] == 0


class TestTelemetryHook:
    """Test telemetry hook integration."""

    @pytest.mark.asyncio
    @patch("openai.AsyncOpenAI")
    async def test_telemetry_on_storage(self, mock_openai_cls, config):
        from neo_memory_hub.hooks.callbacks import TelemetryHook

        class TestTelemetry(TelemetryHook):
            def __init__(self):
                self.storage_calls = []

            def on_storage(self, facts_stored, facts_skipped, facts_extracted, investor_name=None):
                self.storage_calls.append({
                    "stored": facts_stored,
                    "skipped": facts_skipped,
                    "extracted": facts_extracted,
                })

        telemetry = TestTelemetry()
        mock_client = AsyncMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create = AsyncMock(
            return_value=_make_extraction_response([
                {"category": "persona", "value": "Test fact", "salience": 0.8},
            ])
        )

        mock_connector = AsyncMock()
        mock_connector.store_fact = AsyncMock(return_value={"results": [{"id": "new_1"}]})

        hlc = HighLevelMemoryConnector(config=config, telemetry=telemetry)
        hlc._connector = mock_connector
        hlc._initialized = True

        from neo_memory_hub.extraction import FactExtractor
        hlc._extractor = FactExtractor()

        await hlc.store_exchange(query="q", response="r", user_id="u1")

        assert len(telemetry.storage_calls) == 1
        assert telemetry.storage_calls[0]["extracted"] == 1
        assert telemetry.storage_calls[0]["stored"] == 1


class TestRetrieveContextWithInvestor:
    """Test retrieve_context with investor_name parameter."""

    @pytest.mark.asyncio
    async def test_retrieval_with_investor(self, config):
        mock_connector = AsyncMock()
        mock_connector.search = AsyncMock(return_value={
            "results": [
                {
                    "id": "m1",
                    "memory": "Conservative investor",
                    "metadata": {"profile_key": "risk", "salience_score": 0.9, "investor_name": "John"},
                },
            ]
        })

        hlc = HighLevelMemoryConnector(config=config)
        hlc._connector = mock_connector
        hlc._initialized = True

        from neo_memory_hub.retrieval.strategy import MultiCategoryRetriever
        from neo_memory_hub.retrieval.formatter import ProfileAssembler, ContextFormatter
        hlc._retriever = MultiCategoryRetriever()
        hlc._assembler = ProfileAssembler()
        hlc._formatter = ContextFormatter()

        result = await hlc.retrieve_context(
            query="Tell me about John",
            user_id="u1",
            investor_name="John Smith",
        )

        assert result["investor_name"] == "John Smith"
        assert result["count"] > 0

    @pytest.mark.asyncio
    async def test_retrieval_requires_user_id_strict(self):
        config = MemoryHubConfig(
            isolation=IsolationConfig(require_user_id=True),
        )
        hlc = HighLevelMemoryConnector(config=config)
        hlc._initialized = True
        hlc._connector = MagicMock()
        result = await hlc.retrieve_context(query="q", user_id="")
        assert result["count"] == 0

    @pytest.mark.asyncio
    async def test_retrieval_requires_tenant_strict(self):
        config = MemoryHubConfig(
            isolation=IsolationConfig(require_user_id=False, strict_tenant_isolation=True),
        )
        hlc = HighLevelMemoryConnector(config=config)
        hlc._initialized = True
        hlc._connector = MagicMock()
        result = await hlc.retrieve_context(query="q", user_id="u1")
        assert result["count"] == 0


class TestStoreExchangeMetadata:
    """Test metadata building in store_exchange (investor, tenant, session, TTL, pool)."""

    @pytest.mark.asyncio
    @patch("openai.AsyncOpenAI")
    async def test_store_with_full_metadata(self, mock_openai_cls, config):
        mock_client = AsyncMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create = AsyncMock(
            return_value=_make_extraction_response([
                {
                    "category": "procedural",
                    "investor_name": "John",
                    "key": "reminder",
                    "value": "Monthly review due",
                    "detail": "Review portfolio quarterly",
                    "behavioral_note": "Prefers email",
                    "trigger": "End of quarter",
                    "salience": 0.9,
                    "reasoning": "Important procedure",
                },
            ])
        )

        mock_connector = AsyncMock()
        mock_connector.store_fact = AsyncMock(return_value={"results": [{"id": "new_1"}]})
        mock_connector.search = AsyncMock(return_value={"results": []})

        hlc = HighLevelMemoryConnector(config=config)
        hlc._connector = mock_connector
        hlc._initialized = True

        from neo_memory_hub.extraction import FactExtractor
        hlc._extractor = FactExtractor()

        result = await hlc.store_exchange(
            query="Review reminder",
            response="Monthly review due",
            user_id="u1",
            tenant_id="acme",
            session_id="s1",
            investor_name="John",
        )

        assert result["facts_stored"] == 1
        call_kwargs = mock_connector.store_fact.call_args
        metadata = call_kwargs.kwargs.get("metadata") or call_kwargs[1].get("metadata")
        assert metadata["tenant_id"] == "acme"
        assert metadata["session_id"] == "s1"
        assert metadata["investor_name"] == "John"
        assert "detail" in metadata
        assert "behavioral_note" in metadata
        assert "trigger" in metadata

    @pytest.mark.asyncio
    async def test_store_without_extractor(self, config):
        hlc = HighLevelMemoryConnector(config=config)
        hlc._connector = MagicMock()
        hlc._initialized = True
        hlc._extractor = None

        result = await hlc.store_exchange(query="q", response="r", user_id="u1")
        assert result["facts_stored"] == 0


class TestHistoryWithManager:
    """Test history API with actual manager mock."""

    def test_get_all_history_without_manager(self, config):
        hlc = HighLevelMemoryConnector(config=config)
        hlc._history = None
        result = hlc.get_all_history()
        assert result == []

    def test_get_all_history_with_manager(self, config):
        hlc = HighLevelMemoryConnector(config=config)
        hlc._history = MagicMock()
        hlc._history.get_all_history = MagicMock(return_value=[{"event": "ADD"}, {"event": "UPDATE"}])
        result = hlc.get_all_history(limit=50)
        assert len(result) == 2

    def test_get_history_stats_with_manager(self, config):
        hlc = HighLevelMemoryConnector(config=config)
        hlc._history = MagicMock()
        hlc._history.get_stats = MagicMock(return_value={"total": 42, "events": {"ADD": 30, "UPDATE": 12}})
        stats = hlc.get_history_stats()
        assert stats["total"] == 42


class TestContextManager:
    """Test async context manager."""

    @pytest.mark.asyncio
    @patch("neo_memory_hub.connector.NeoMemoryConnector")
    async def test_async_context_manager(self, mock_connector_cls, config):
        mock_connector = AsyncMock()
        mock_connector.initialize = AsyncMock()
        mock_connector.close = AsyncMock()
        mock_connector._memory = MagicMock()
        mock_connector._memory._mem0 = None
        mock_connector_cls.return_value = mock_connector

        async with HighLevelMemoryConnector(config=config) as hlc:
            assert hlc.is_ready is True

        mock_connector.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_close_without_connector(self, config):
        hlc = HighLevelMemoryConnector(config=config)
        await hlc.close()
        assert hlc.is_ready is False


class TestDeleteAll:
    """Test delete_all() delegation and isolation — G-02."""

    @pytest.mark.asyncio
    async def test_delete_all_delegates(self, config):
        hlc = HighLevelMemoryConnector(config=config)
        hlc._connector = AsyncMock()
        hlc._connector.delete_all = AsyncMock(return_value={"deleted": 5})
        hlc._initialized = True

        result = await hlc.delete_all(user_id="u1")
        hlc._connector.delete_all.assert_called_once_with(
            user_id="u1", agent_id="default_agent"
        )
        assert result == {"deleted": 5}

    @pytest.mark.asyncio
    async def test_delete_all_requires_user_id(self):
        cfg = MemoryHubConfig(
            isolation=IsolationConfig(require_user_id=True),
            buffering=BufferingConfig(enabled=False),
        )
        hlc = HighLevelMemoryConnector(config=cfg)
        hlc._connector = AsyncMock()
        hlc._initialized = True

        with pytest.raises(ValueError, match="user_id is required"):
            await hlc.delete_all(user_id="")

    @pytest.mark.asyncio
    async def test_delete_all_tenant_isolation(self):
        cfg = MemoryHubConfig(
            isolation=IsolationConfig(
                require_user_id=True, strict_tenant_isolation=True
            ),
            buffering=BufferingConfig(enabled=False),
        )
        hlc = HighLevelMemoryConnector(config=cfg)
        hlc._connector = AsyncMock()
        hlc._initialized = True

        with pytest.raises(ValueError, match="tenant_id is required"):
            await hlc.delete_all(user_id="u1")


class TestRetrievalConfigIntegration:
    """G-07/G-08: Verify retrieval config wires into connector correctly."""

    def test_default_query_hints(self):
        from neo_memory_hub.config.hub_config import RetrievalConfig

        rc = RetrievalConfig()
        assert "persona" in rc.query_hints
        assert "preference" in rc.query_hints
        assert "episodic" in rc.query_hints
        assert "procedural" in rc.query_hints

    def test_default_investor_aware_thresholds(self):
        from neo_memory_hub.config.hub_config import RetrievalConfig

        rc = RetrievalConfig()
        assert rc.investor_aware_thresholds["persona"] == 0.03
        assert rc.investor_aware_thresholds["preference"] == 0.03
        assert rc.investor_aware_thresholds["episodic"] == 0.03
        assert rc.investor_aware_thresholds["procedural"] == 0.15

    def test_storage_routing_has_docstring_hint(self):
        from neo_memory_hub.config.hub_config import StorageRoutingConfig

        doc = StorageRoutingConfig.__doc__
        assert "application" in doc.lower()


class TestEnvVarResolution:
    """G-11: Validate both env-var syntaxes work."""

    def test_colon_hyphen_syntax(self):
        from neo_memory_hub.config.hub_config import _resolve_env_vars
        import os
        os.environ.pop("__TEST_ENVAR__", None)
        result = _resolve_env_vars("${__TEST_ENVAR__:-fallback_val}")
        assert result == "fallback_val"

    def test_env_var_present(self):
        from neo_memory_hub.config.hub_config import _resolve_env_vars
        import os
        os.environ["__TEST_ENVAR__"] = "from_env"
        try:
            result = _resolve_env_vars("${__TEST_ENVAR__:-fallback_val}")
            assert result == "from_env"
        finally:
            del os.environ["__TEST_ENVAR__"]

    def test_null_default(self):
        from neo_memory_hub.config.hub_config import _resolve_env_vars
        import os
        os.environ.pop("__TEST_ENVAR__", None)
        result = _resolve_env_vars("${__TEST_ENVAR__:-null}")
        assert result is None


# =============================================================================
# GAP-1: EntityResolver wired in pipeline
# =============================================================================

class TestEntityResolverWiring:
    """GAP-1: EntityResolver.resolve() called when investor_name is None."""

    @pytest.mark.asyncio
    @patch("openai.AsyncOpenAI")
    async def test_entity_resolver_called_on_store_when_investor_none(
        self, mock_openai_cls, config
    ):
        """EntityResolver should be called in store_exchange when investor_name is not provided."""
        from neo_memory_hub.hooks.callbacks import EntityResolver

        class CapturingResolver(EntityResolver):
            def __init__(self):
                self.calls = []

            async def resolve(self, query, session_context=None):
                self.calls.append(query)
                return "resolved_investor"

        resolver = CapturingResolver()

        mock_client = AsyncMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create = AsyncMock(
            return_value=_make_extraction_response([
                {"category": "persona", "value": "Test fact", "salience": 0.8},
            ])
        )

        mock_connector = AsyncMock()
        mock_connector.store_fact = AsyncMock(return_value={"results": [{"id": "r1"}]})
        mock_connector.search = AsyncMock(return_value={"results": []})

        hlc = HighLevelMemoryConnector(config=config, entity_resolver=resolver)
        hlc._connector = mock_connector
        hlc._initialized = True

        from neo_memory_hub.extraction import FactExtractor
        hlc._extractor = FactExtractor()

        result = await hlc.store_exchange(
            query="Tell me about the client",
            response="The client is risk-averse",
            user_id="u1",
            # investor_name NOT provided
        )

        # Resolver should have been called once
        assert len(resolver.calls) == 1
        assert resolver.calls[0] == "Tell me about the client"
        # Resolved name should appear in result (normalize_investor_name applies .title())
        assert result["investor_name"] == "Resolved_Investor"

    @pytest.mark.asyncio
    @patch("openai.AsyncOpenAI")
    async def test_entity_resolver_not_called_when_investor_provided(
        self, mock_openai_cls, config
    ):
        """EntityResolver should NOT be called when investor_name is explicitly provided."""
        from neo_memory_hub.hooks.callbacks import EntityResolver

        class CapturingResolver(EntityResolver):
            def __init__(self):
                self.calls = []

            async def resolve(self, query, session_context=None):
                self.calls.append(query)
                return "resolver_name"

        resolver = CapturingResolver()

        mock_client = AsyncMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create = AsyncMock(
            return_value=_make_extraction_response([
                {"category": "persona", "value": "Test fact", "salience": 0.8},
            ])
        )

        mock_connector = AsyncMock()
        mock_connector.store_fact = AsyncMock(return_value={"results": [{"id": "r1"}]})
        mock_connector.search = AsyncMock(return_value={"results": []})

        hlc = HighLevelMemoryConnector(config=config, entity_resolver=resolver)
        hlc._connector = mock_connector
        hlc._initialized = True

        from neo_memory_hub.extraction import FactExtractor
        hlc._extractor = FactExtractor()

        result = await hlc.store_exchange(
            query="Tell me about Alice",
            response="Alice prefers email",
            user_id="u1",
            investor_name="alice_jones",  # explicitly provided
        )

        # Resolver should NOT have been called
        assert len(resolver.calls) == 0
        # investor_name is title-cased by normalize_investor_name
        assert result["investor_name"] == "Alice_Jones"

    @pytest.mark.asyncio
    async def test_entity_resolver_called_on_retrieve_when_investor_none(self, config):
        """EntityResolver should be called in retrieve_context when investor_name is None."""
        from neo_memory_hub.hooks.callbacks import EntityResolver

        class CapturingResolver(EntityResolver):
            def __init__(self):
                self.calls = []

            async def resolve(self, query, session_context=None):
                self.calls.append(query)
                return "resolved_from_retrieval"

        resolver = CapturingResolver()
        mock_connector = AsyncMock()
        mock_connector.search = AsyncMock(return_value={"results": []})

        hlc = HighLevelMemoryConnector(config=config, entity_resolver=resolver)
        hlc._connector = mock_connector
        hlc._initialized = True

        from neo_memory_hub.retrieval.strategy import MultiCategoryRetriever
        from neo_memory_hub.retrieval.formatter import ProfileAssembler, ContextFormatter
        hlc._retriever = MultiCategoryRetriever()
        hlc._assembler = ProfileAssembler()
        hlc._formatter = ContextFormatter()

        result = await hlc.retrieve_context(
            query="What do we know about the investor?",
            user_id="u1",
            # investor_name NOT provided
        )

        assert len(resolver.calls) == 1
        # investor_name in result reflects resolved + .title()-normalized name
        assert result["investor_name"] == "Resolved_From_Retrieval"

    @pytest.mark.asyncio
    async def test_entity_resolver_not_called_on_retrieve_when_investor_provided(self, config):
        """EntityResolver should NOT be called in retrieve_context when investor_name is given."""
        from neo_memory_hub.hooks.callbacks import EntityResolver

        class CapturingResolver(EntityResolver):
            def __init__(self):
                self.calls = []

            async def resolve(self, query, session_context=None):
                self.calls.append(query)
                return "resolver_name"

        resolver = CapturingResolver()
        mock_connector = AsyncMock()
        mock_connector.search = AsyncMock(return_value={"results": []})

        hlc = HighLevelMemoryConnector(config=config, entity_resolver=resolver)
        hlc._connector = mock_connector
        hlc._initialized = True

        from neo_memory_hub.retrieval.strategy import MultiCategoryRetriever
        from neo_memory_hub.retrieval.formatter import ProfileAssembler, ContextFormatter
        hlc._retriever = MultiCategoryRetriever()
        hlc._assembler = ProfileAssembler()
        hlc._formatter = ContextFormatter()

        await hlc.retrieve_context(
            query="q",
            user_id="u1",
            investor_name="explicit_investor",
        )

        assert len(resolver.calls) == 0

    @pytest.mark.asyncio
    @patch("openai.AsyncOpenAI")
    async def test_entity_resolver_exception_is_swallowed(self, mock_openai_cls, config):
        """An EntityResolver that raises should not break the pipeline."""
        from neo_memory_hub.hooks.callbacks import EntityResolver

        class FailingResolver(EntityResolver):
            async def resolve(self, query, session_context=None):
                raise RuntimeError("resolution service down")

        mock_client = AsyncMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create = AsyncMock(
            return_value=_make_extraction_response([
                {"category": "persona", "value": "Test fact", "salience": 0.8},
            ])
        )

        mock_connector = AsyncMock()
        mock_connector.store_fact = AsyncMock(return_value={"results": [{"id": "r1"}]})
        mock_connector.search = AsyncMock(return_value={"results": []})

        hlc = HighLevelMemoryConnector(
            config=config, entity_resolver=FailingResolver()
        )
        hlc._connector = mock_connector
        hlc._initialized = True

        from neo_memory_hub.extraction import FactExtractor
        hlc._extractor = FactExtractor()

        # Should not raise
        result = await hlc.store_exchange(
            query="q", response="r", user_id="u1"
        )
        # Pipeline continues even when resolver fails
        assert "facts_stored" in result


# =============================================================================
# GAP-2/3/4: dept_id per-call override
# =============================================================================

class TestDeptIdPerCall:
    """GAP-2/3/4: dept_id passed at call time overrides config default."""

    @pytest.mark.asyncio
    @patch("openai.AsyncOpenAI")
    async def test_store_exchange_dept_id_per_call_stored_in_metadata(
        self, mock_openai_cls, config
    ):
        """dept_id passed to store_exchange should appear in stored metadata."""
        mock_client = AsyncMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create = AsyncMock(
            return_value=_make_extraction_response([
                {
                    "category": "persona",
                    "investor_name": "Alice",
                    "value": "Risk-averse",
                    "salience": 0.9,
                    "pool": "team",
                },
            ])
        )

        mock_connector = AsyncMock()
        mock_connector.store_fact = AsyncMock(return_value={"results": [{"id": "r1"}]})
        mock_connector.search = AsyncMock(return_value={"results": []})

        hlc = HighLevelMemoryConnector(config=config)
        hlc._connector = mock_connector
        hlc._initialized = True

        from neo_memory_hub.extraction import FactExtractor
        hlc._extractor = FactExtractor()

        await hlc.store_exchange(
            query="Tell me about Alice",
            response="Alice is conservative",
            user_id="u1",
            investor_name="alice",
            dept_id="wealth_dept",  # per-call override
        )

        call_kwargs = mock_connector.store_fact.call_args
        metadata = call_kwargs.kwargs.get("metadata") or call_kwargs[1].get("metadata")
        assert metadata.get("dept_id") == "wealth_dept"

    @pytest.mark.asyncio
    @patch("openai.AsyncOpenAI")
    async def test_store_exchange_dept_id_config_default_used_when_no_override(
        self, mock_openai_cls
    ):
        """When dept_id is not passed, isolation.default_dept_id from config is used."""
        from neo_memory_hub.config.hub_config import IsolationConfig

        config_with_dept = MemoryHubConfig(
            salience=SalienceConfig(enabled=False),
            extraction=ExtractionConfig(enabled=True),
            dedup=DedupConfig(enabled=False),
            buffering=BufferingConfig(enabled=False),
            isolation=IsolationConfig(
                require_user_id=False,
                default_dept_id="config_dept",
            ),
        )

        mock_client = AsyncMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create = AsyncMock(
            return_value=_make_extraction_response([
                {"category": "persona", "value": "Fact", "salience": 0.9, "pool": "team"},
            ])
        )

        mock_connector = AsyncMock()
        mock_connector.store_fact = AsyncMock(return_value={"results": [{"id": "r1"}]})
        mock_connector.search = AsyncMock(return_value={"results": []})

        hlc = HighLevelMemoryConnector(config=config_with_dept)
        hlc._connector = mock_connector
        hlc._initialized = True

        from neo_memory_hub.extraction import FactExtractor
        hlc._extractor = FactExtractor()

        # No dept_id passed at call time
        await hlc.store_exchange(query="q", response="r", user_id="u1")

        call_kwargs = mock_connector.store_fact.call_args
        metadata = call_kwargs.kwargs.get("metadata") or call_kwargs[1].get("metadata")
        assert metadata.get("dept_id") == "config_dept"

    @pytest.mark.asyncio
    @patch("openai.AsyncOpenAI")
    async def test_store_exchange_per_call_dept_id_overrides_config(
        self, mock_openai_cls
    ):
        """Per-call dept_id should override isolation.default_dept_id from config."""
        from neo_memory_hub.config.hub_config import IsolationConfig

        config_with_dept = MemoryHubConfig(
            salience=SalienceConfig(enabled=False),
            extraction=ExtractionConfig(enabled=True),
            dedup=DedupConfig(enabled=False),
            buffering=BufferingConfig(enabled=False),
            isolation=IsolationConfig(
                require_user_id=False,
                default_dept_id="config_dept",
            ),
        )

        mock_client = AsyncMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create = AsyncMock(
            return_value=_make_extraction_response([
                {"category": "persona", "value": "Fact", "salience": 0.9},
            ])
        )

        mock_connector = AsyncMock()
        mock_connector.store_fact = AsyncMock(return_value={"results": [{"id": "r1"}]})
        mock_connector.search = AsyncMock(return_value={"results": []})

        hlc = HighLevelMemoryConnector(config=config_with_dept)
        hlc._connector = mock_connector
        hlc._initialized = True

        from neo_memory_hub.extraction import FactExtractor
        hlc._extractor = FactExtractor()

        await hlc.store_exchange(
            query="q", response="r", user_id="u1", dept_id="override_dept"
        )

        call_kwargs = mock_connector.store_fact.call_args
        metadata = call_kwargs.kwargs.get("metadata") or call_kwargs[1].get("metadata")
        # Per-call override wins over config default
        assert metadata.get("dept_id") == "override_dept"

    @pytest.mark.asyncio
    async def test_retrieve_context_dept_id_passed_to_retriever(self, config):
        """dept_id passed to retrieve_context should flow through to the retriever."""
        mock_connector = AsyncMock()
        # Capture the search calls to verify dept_id intent
        mock_connector.search = AsyncMock(return_value={"results": []})

        hlc = HighLevelMemoryConnector(config=config)
        hlc._connector = mock_connector
        hlc._initialized = True

        from neo_memory_hub.retrieval.strategy import MultiCategoryRetriever
        from neo_memory_hub.retrieval.formatter import ProfileAssembler, ContextFormatter
        hlc._retriever = MultiCategoryRetriever()
        hlc._assembler = ProfileAssembler()
        hlc._formatter = ContextFormatter()

        # Should not raise and should complete
        result = await hlc.retrieve_context(
            query="q",
            user_id="u1",
            dept_id="retrieval_dept",
        )
        # Empty results but call succeeded
        assert result["count"] == 0

    @pytest.mark.asyncio
    async def test_buffer_exchange_dept_id_stored_in_buffered_exchange(self, config):
        """dept_id passed to buffer_exchange should be stored in BufferedExchange."""
        cfg = MemoryHubConfig(
            buffering=BufferingConfig(enabled=True, flush_threshold=5),
            isolation=IsolationConfig(require_user_id=False),
            salience=SalienceConfig(enabled=False),
            extraction=ExtractionConfig(enabled=True),
            dedup=DedupConfig(enabled=False),
        )
        hlc = HighLevelMemoryConnector(config=cfg)
        hlc._connector = MagicMock()
        hlc._initialized = True

        from neo_memory_hub.buffering import ExchangeBuffer
        hlc._buffer = ExchangeBuffer(flush_threshold=5)

        await hlc.buffer_exchange(
            query="q",
            response="r",
            user_id="u1",
            dept_id="buffered_dept",
        )

        # Verify the buffered exchange has dept_id set
        key = "u1:default"
        buffered = hlc._buffer._buffers.get(key, [])
        assert len(buffered) == 1
        assert buffered[0].dept_id == "buffered_dept"
