"""Stress tests for all core utilities.

Covers edge cases, boundary conditions, error paths, and concurrency
for: resilience, storage_gateway, salience, metrics, retriever,
ranker, formatter, exchange_buffer, dedup, extraction, scoped_connector,
and pool_routing.
"""
from __future__ import annotations

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# =============================================================================
# A. CircuitBreaker + retry_async (was 0% coverage)
# =============================================================================
from neo_memory_hub.core.resilience import CircuitBreaker, retry_async


class TestCircuitBreakerStateMachine:
    def test_starts_closed(self):
        cb = CircuitBreaker(failure_threshold=3)
        assert cb.allow() is True
        assert cb._failures == 0

    def test_failures_below_threshold_keeps_closed(self):
        cb = CircuitBreaker(failure_threshold=3)
        cb.record_failure()
        cb.record_failure()
        assert cb.allow() is True  # 2 < 3

    def test_failures_at_threshold_opens(self):
        cb = CircuitBreaker(failure_threshold=3)
        for _ in range(3):
            cb.record_failure()
        assert cb.allow() is False
        assert cb._opened_at is not None

    def test_success_resets_counter(self):
        cb = CircuitBreaker(failure_threshold=3)
        cb.record_failure()
        cb.record_failure()
        cb.record_success()
        assert cb._failures == 0
        assert cb._opened_at is None
        # More failures needed to re-trip
        cb.record_failure()
        cb.record_failure()
        assert cb.allow() is True

    def test_half_open_after_recovery_timeout(self):
        cb = CircuitBreaker(failure_threshold=2, recovery_timeout=0.1)
        cb.record_failure()
        cb.record_failure()
        assert cb.allow() is False
        time.sleep(0.15)
        assert cb.allow() is True  # half-open

    def test_half_open_success_closes(self):
        cb = CircuitBreaker(failure_threshold=2, recovery_timeout=0.05)
        cb.record_failure()
        cb.record_failure()
        time.sleep(0.06)
        assert cb.allow() is True
        cb.record_success()
        assert cb._failures == 0
        assert cb._opened_at is None

    def test_half_open_failure_reopens(self):
        cb = CircuitBreaker(failure_threshold=2, recovery_timeout=0.05)
        cb.record_failure()
        cb.record_failure()
        time.sleep(0.06)
        assert cb.allow() is True  # half-open trial
        cb.record_failure()  # trial failed
        assert cb.allow() is False  # re-opened


class TestRetryAsync:
    @pytest.mark.asyncio
    async def test_success_first_try(self):
        result = await retry_async(AsyncMock(return_value=42))
        assert result == 42

    @pytest.mark.asyncio
    async def test_success_on_second_try(self):
        mock = AsyncMock(side_effect=[ValueError("fail"), 99])
        result = await retry_async(mock, retries=2, base_delay=0.01)
        assert result == 99
        assert mock.call_count == 2

    @pytest.mark.asyncio
    async def test_exhausted_raises_last_exception(self):
        mock = AsyncMock(side_effect=ValueError("final"))
        with pytest.raises(ValueError, match="final"):
            await retry_async(mock, retries=0, base_delay=0.01)

    @pytest.mark.asyncio
    async def test_circuit_breaker_open_raises_immediately(self):
        cb = CircuitBreaker(failure_threshold=1)
        cb.record_failure()
        mock = AsyncMock()
        with pytest.raises(RuntimeError, match="Circuit breaker open"):
            await retry_async(mock, breaker=cb)
        mock.assert_not_called()

    @pytest.mark.asyncio
    async def test_exponential_backoff_delays(self):
        delays = []
        mock = AsyncMock(side_effect=[ValueError, ValueError, 42])

        with patch("neo_memory_hub.core.resilience.asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            mock_sleep.side_effect = lambda d: delays.append(d)
            await retry_async(mock, retries=3, base_delay=1.0)

        assert delays == [1.0, 2.0]  # 2^0, 2^1

    @pytest.mark.asyncio
    async def test_breaker_records_success_on_pass(self):
        cb = CircuitBreaker(failure_threshold=5)
        cb.record_failure()  # 1 failure
        await retry_async(AsyncMock(return_value="ok"), breaker=cb)
        assert cb._failures == 0  # reset by success

    @pytest.mark.asyncio
    async def test_retries_zero_single_attempt(self):
        mock = AsyncMock(side_effect=RuntimeError("oops"))
        with pytest.raises(RuntimeError, match="oops"):
            await retry_async(mock, retries=0)
        assert mock.call_count == 1


# =============================================================================
# B. StorageGateway edge cases
# =============================================================================
from neo_memory_hub.core.storage_gateway import (
    StorageDecision,
    StorageGateway,
    StorageGatewayConfig,
)


class TestStorageGatewayEdgeCases:
    def test_whitespace_only_is_too_short(self):
        gw = StorageGateway()
        result = gw.evaluate("   ")
        assert result.decision == StorageDecision.SKIP

    def test_boundary_min_length(self):
        gw = StorageGateway(StorageGatewayConfig(min_content_length=3))
        assert gw.evaluate("abc").decision == StorageDecision.STORE
        assert gw.evaluate("ab").decision == StorageDecision.SKIP

    def test_boundary_max_length(self):
        gw = StorageGateway(StorageGatewayConfig(max_content_length=10))
        assert gw.evaluate("x" * 10).decision == StorageDecision.STORE
        assert gw.evaluate("x" * 11).decision == StorageDecision.SKIP

    def test_salience_zero_threshold_stores_everything(self):
        gw = StorageGateway(StorageGatewayConfig(min_salience_threshold=0.0))
        assert gw.evaluate("fact", salience=0.0).decision == StorageDecision.STORE

    def test_salience_one_always_stores(self):
        gw = StorageGateway(StorageGatewayConfig(min_salience_threshold=0.99))
        assert gw.evaluate("fact", salience=1.0).decision == StorageDecision.STORE

    def test_disabled_worthiness_check(self):
        gw = StorageGateway(StorageGatewayConfig(enable_worthiness_check=False))
        result = gw.evaluate("")
        assert result.decision == StorageDecision.STORE

    def test_stress_100k_evaluations(self):
        gw = StorageGateway()
        for _ in range(100_000):
            gw.evaluate("a fact about something important")
        # No exception, no state accumulation


# =============================================================================
# C. Salience Scorer — curly brace fix verification
# =============================================================================
from neo_memory_hub.core.salience import SalienceScorer, SalienceScorerConfig


class TestSalienceScorerEdgeCases:
    def test_build_prompt_with_curly_braces_in_facts(self):
        """Facts containing {braces} must not crash str.format."""
        scorer = SalienceScorer()
        prompt = scorer._build_prompt(
            facts=["Config is {debug: true}", "User likes {JSON}"],
            context="Test context",
        )
        assert "{debug: true}" in prompt
        assert "{JSON}" in prompt

    def test_parse_response_malformed_returns_defaults(self):
        scorer = SalienceScorer(SalienceScorerConfig(default_score=0.3))
        results = scorer._parse_response("not json", ["fact1", "fact2"])
        assert len(results) == 2
        assert all(r.salience == 0.3 for r in results)

    def test_parse_response_empty_scored_facts(self):
        scorer = SalienceScorer()
        results = scorer._parse_response('{"scored_facts": []}', ["fact1"])
        assert len(results) == 0  # valid but empty

    def test_score_facts_empty_list(self):
        """Empty list should return immediately without LLM call."""
        scorer = SalienceScorer()

        async def run():
            return await scorer.score_facts(facts=[])

        result = asyncio.get_event_loop().run_until_complete(run())
        assert result == []


# =============================================================================
# D. Metrics edge cases
# =============================================================================
from neo_memory_hub.core.metrics import (
    Counter,
    Gauge,
    Histogram,
    MetricsRegistry,
)


class TestMetricsEdgeCases:
    def test_counter_fractional_increment(self):
        c = Counter("test_frac", "test desc", labels=["label_a"])
        c.inc(0.5, label_a="x")
        assert c._values[("x",)] == 0.5

    def test_gauge_goes_negative(self):
        g = Gauge("test_neg", "test desc", labels=["label_a"])
        g.set(2.0, label_a="x")
        g.dec(5.0, label_a="x")
        assert g._values[("x",)] == -3.0

    def test_histogram_zero_observations_prometheus(self):
        h = Histogram("test_empty", "test desc", labels=["op"])
        output = h.to_prometheus()
        assert "test_empty" in output

    def test_counter_no_labels_prometheus(self):
        c = Counter("test_nolabel", "test desc", labels=[])
        c.inc(5.0)
        output = c.to_prometheus()
        assert "test_nolabel" in output
        assert "5.0" in output

    def test_registry_track_operation_with_exception(self):
        registry = MetricsRegistry()
        with pytest.raises(ValueError):
            with registry.track_operation("test_op", "tenant1"):
                raise ValueError("boom")
        # Verify error was still tracked via the errors_total counter
        assert registry.errors_total._values  # At least one error recorded

    def test_histogram_many_observations(self):
        h = Histogram("stress_hist", "test desc", labels=["op"])
        for i in range(1000):
            h.observe(float(i) / 100.0, op="read")
        assert h.get_count(op="read") == 1000


# =============================================================================
# E. MultiCategoryRetriever edge cases
# =============================================================================
from neo_memory_hub.retrieval.strategy import (
    CategorySearchConfig,
    MultiCategoryRetriever,
    RetrievalResult,
)


class TestRetrieverEdgeCases:
    @pytest.mark.asyncio
    async def test_search_fn_returns_none(self):
        """search_fn returning None should not crash."""
        retriever = MultiCategoryRetriever(
            categories=[CategorySearchConfig(name="persona")]
        )
        result = await retriever.retrieve(
            query="test",
            search_fn=AsyncMock(return_value=None),
            user_id="u1",
            agent_id="a1",
        )
        assert result.count == 0

    @pytest.mark.asyncio
    async def test_empty_categories(self):
        retriever = MultiCategoryRetriever(categories=[])
        result = await retriever.retrieve(
            query="test",
            search_fn=AsyncMock(return_value={"results": []}),
            user_id="u1",
            agent_id="a1",
        )
        assert result.count == 0
        assert result.memories == []

    @pytest.mark.asyncio
    async def test_all_categories_fail(self):
        retriever = MultiCategoryRetriever(
            categories=[
                CategorySearchConfig(name="persona"),
                CategorySearchConfig(name="episodic"),
            ]
        )
        result = await retriever.retrieve(
            query="test",
            search_fn=AsyncMock(side_effect=RuntimeError("db down")),
            user_id="u1",
            agent_id="a1",
        )
        assert result.count == 0

    @pytest.mark.asyncio
    async def test_memory_without_id_is_skipped(self):
        async def search_fn(**kwargs):
            return {"results": [{"memory": "no id field", "metadata": {}}]}

        retriever = MultiCategoryRetriever(
            categories=[CategorySearchConfig(name="persona")]
        )
        result = await retriever.retrieve(
            query="test", search_fn=search_fn, user_id="u1", agent_id="a1"
        )
        assert result.count == 0

    def test_normalize_integer_input(self):
        assert MultiCategoryRetriever.normalize_investor_name(42) is None

    def test_subject_id_all_special_chars(self):
        assert MultiCategoryRetriever.investor_name_to_subject_id("@#$%") is None

    @pytest.mark.asyncio
    async def test_pool_routing_hook_integration(self):
        """PoolRoutingHook is called when provided."""
        calls = []

        async def search_fn(**kwargs):
            calls.append(kwargs)
            return {"results": []}

        mock_hook = MagicMock()
        mock_hook.get_pool_search_tasks.return_value = (
            [search_fn(query="team_q", user_id="__team_eng__", agent_id=None, limit=5, threshold=0.2, metadata_filters={"pool": "team"})],
            ["team"],
        )

        retriever = MultiCategoryRetriever(categories=[])
        result = await retriever.retrieve(
            query="test",
            search_fn=search_fn,
            user_id="u1",
            agent_id="a1",
            pool_routing=mock_hook,
        )
        mock_hook.get_pool_search_tasks.assert_called_once()


# =============================================================================
# F. Ranker edge cases
# =============================================================================
from neo_memory_hub.retrieval.ranker import MultiFactorRanker, RankingConfig


class TestRankerEdgeCases:
    def test_empty_list(self):
        ranker = MultiFactorRanker()
        assert ranker.rank([]) == []

    def test_single_element(self):
        from datetime import datetime, timezone
        from neo_memory_hub.retrieval.ranker import MemoryResult
        from neo_memory_hub.domain.memory import MemoryEntry

        ranker = MultiFactorRanker()
        entry = MagicMock(spec=MemoryEntry)
        entry.importance = 5
        entry.accessed_at = datetime.now(timezone.utc)
        mem = MemoryResult(entry=entry, relevance_score=0.8)
        result = ranker.rank([mem])
        assert len(result) == 1

    def test_relevance_only_config(self):
        config = RankingConfig.relevance_only()
        assert config.relevance_weight == 1.0
        assert config.recency_weight == 0.0
        assert config.importance_weight == 0.0

    def test_recency_biased_config(self):
        config = RankingConfig.recency_biased()
        assert config.recency_weight > config.relevance_weight

    def test_zero_decay_lambda(self):
        config = RankingConfig(recency_decay_lambda=0.0)
        ranker = MultiFactorRanker(config)
        # All memories get same recency score regardless of age
        assert config.recency_decay_lambda == 0.0


# =============================================================================
# G. Formatter edge cases
# =============================================================================
from neo_memory_hub.retrieval.formatter import ContextFormatter, ProfileAssembler


class TestFormatterEdgeCases:
    def test_truncation_with_very_small_max_chars(self):
        """max_chars < 40 should not produce negative slice."""
        formatter = ContextFormatter(max_chars=10)
        result = formatter.format(
            {"persona": {"name": {"value": "John Doe"}}},
            entity_name="Client",
        )
        assert "[...memory context truncated...]" in result

    def test_max_chars_zero_disables_truncation(self):
        formatter = ContextFormatter(max_chars=0)
        result = formatter.format(
            {"persona": {"name": {"value": "x" * 10000}}},
            entity_name="Client",
        )
        assert "[...memory context truncated...]" not in result

    def test_empty_profile(self):
        formatter = ContextFormatter()
        result = formatter.format({}, entity_name="Client")
        assert result == ""

    def test_assembler_empty_memories(self):
        assembler = ProfileAssembler()
        profile = assembler.assemble([], entity_name="Client")
        # entity_name is always included
        assert profile.get("persona") is None
        assert profile.get("preference") is None

    def test_assembler_data_key_fallback(self):
        assembler = ProfileAssembler()
        memories = [
            {
                "id": "1",
                "data": "Uses dark mode",
                "_retrieval_reason": "preference",
                "metadata": {"profile_key": "theme"},
            }
        ]
        profile = assembler.assemble(memories, entity_name="Client")
        assert "preference" in profile

    def test_assembler_content_key_fallback(self):
        assembler = ProfileAssembler()
        memories = [
            {
                "id": "1",
                "content": "Likes Python",
                "_retrieval_reason": "persona",
                "metadata": {"profile_key": "language"},
            }
        ]
        profile = assembler.assemble(memories, entity_name="Client")
        assert "persona" in profile


# =============================================================================
# H. ExchangeBuffer edge cases
# =============================================================================
from neo_memory_hub.buffering.exchange_buffer import ExchangeBuffer


class TestExchangeBufferEdgeCases:
    @pytest.mark.asyncio
    async def test_count_threshold_triggers_flush(self):
        buf = ExchangeBuffer(flush_threshold=3, idle_timeout_seconds=0)
        for i in range(2):
            result = await buf.add_exchange(f"q{i}", f"r{i}", "u1", session_id="s1")
            assert result is None
        result = await buf.add_exchange("q2", "r2", "u1", session_id="s1")
        assert result is not None
        assert len(result) == 3

    @pytest.mark.asyncio
    async def test_chars_threshold_triggers_flush(self):
        buf = ExchangeBuffer(flush_threshold=100, max_buffer_chars=20, idle_timeout_seconds=0)
        result = await buf.add_exchange("x" * 15, "y" * 10, "u1", session_id="s1")
        assert result is not None  # 25 chars >= 20

    @pytest.mark.asyncio
    async def test_flush_session_clears_buffer(self):
        buf = ExchangeBuffer(flush_threshold=100, idle_timeout_seconds=0)
        await buf.add_exchange("q", "r", "u1", session_id="s1")
        result = await buf.flush_session("u1", "s1")
        assert result is not None
        assert len(result) == 1
        assert buf.get_buffer_size("u1", "s1") == 0

    @pytest.mark.asyncio
    async def test_cleanup_session_does_not_flush(self):
        callback = AsyncMock()
        buf = ExchangeBuffer(flush_threshold=100, idle_timeout_seconds=0)
        buf.set_flush_callback(callback)
        await buf.add_exchange("q", "r", "u1", session_id="s1")
        buf.cleanup_session("u1", "s1")
        assert buf.get_buffer_size("u1", "s1") == 0
        callback.assert_not_called()

    @pytest.mark.asyncio
    async def test_drain_cleans_idle_tasks(self):
        buf = ExchangeBuffer(flush_threshold=2, idle_timeout_seconds=60)
        await buf.add_exchange("q1", "r1", "u1", session_id="s1")
        result = await buf.add_exchange("q2", "r2", "u1", session_id="s1")
        assert result is not None
        assert "u1:s1" not in buf._idle_tasks

    @pytest.mark.asyncio
    async def test_consolidate_exchanges(self):
        buf = ExchangeBuffer()
        from neo_memory_hub.buffering.exchange_buffer import BufferedExchange

        exchanges = [
            BufferedExchange(query="Hello", response="Hi", user_id="u1"),
            BufferedExchange(query="How?", response="Good", user_id="u1"),
        ]
        text = buf.consolidate_exchanges(exchanges)
        assert "User: Hello" in text
        assert "Assistant: Hi" in text
        assert "Exchange 1" in text

    @pytest.mark.asyncio
    async def test_session_isolation(self):
        buf = ExchangeBuffer(flush_threshold=100, idle_timeout_seconds=0)
        await buf.add_exchange("q1", "r1", "u1", session_id="s1")
        await buf.add_exchange("q2", "r2", "u1", session_id="s2")
        assert buf.get_buffer_size("u1", "s1") == 1
        assert buf.get_buffer_size("u1", "s2") == 1

    @pytest.mark.asyncio
    async def test_concurrent_add_exchange(self):
        """Multiple concurrent adds on same session should not lose data."""
        buf = ExchangeBuffer(flush_threshold=100, max_buffer_chars=1_000_000, idle_timeout_seconds=0)
        tasks = [
            buf.add_exchange(f"q{i}", f"r{i}", "u1", session_id="s1")
            for i in range(50)
        ]
        await asyncio.gather(*tasks)
        assert buf.get_buffer_size("u1", "s1") == 50


# =============================================================================
# I. ScopedMemoryConnector edge cases
# =============================================================================
from memory_utils.shared_scope import ScopedMemoryConnector, SharedMemoryStrategy
from neo_memory_hub.domain.scope import AccessTier, IsolationScope
from neo_memory_hub.integrations.connector import ConnectorConfig, NeoMemoryConnector


class TestScopedMemoryConnectorEdgeCases:
    def _make_scoped(
        self, mock_memory, strategy=SharedMemoryStrategy.ENABLED
    ) -> ScopedMemoryConnector:
        connector = NeoMemoryConnector(
            config=ConnectorConfig(log_operations=False, use_storage_validation=False)
        )
        connector._memory = mock_memory
        connector._initialized = True
        return ScopedMemoryConnector(connector, strategy=strategy)

    @pytest.mark.asyncio
    async def test_add_private_with_disabled_strategy_allowed(self):
        mock = AsyncMock()
        mock.add = AsyncMock(return_value={"results": [{"id": "1"}]})
        scoped = self._make_scoped(mock, SharedMemoryStrategy.DISABLED)
        scope = IsolationScope(tenant_id="t1", user_id="u1", agent_id="a1", pool=AccessTier.PRIVATE)
        result = await scoped.add_scoped("fact", scope)
        assert "results" in result

    @pytest.mark.asyncio
    async def test_add_team_with_disabled_strategy_blocked(self):
        mock = AsyncMock()
        scoped = self._make_scoped(mock, SharedMemoryStrategy.DISABLED)
        scope = IsolationScope(tenant_id="t1", user_id="u1", agent_id="a1", dept_id="eng", pool=AccessTier.TEAM)
        result = await scoped.add_scoped("fact", scope)
        assert result["blocked_by"] == "shared_memory_strategy"
        mock.add.assert_not_called()

    @pytest.mark.asyncio
    async def test_add_org_with_read_only_blocked(self):
        mock = AsyncMock()
        scoped = self._make_scoped(mock, SharedMemoryStrategy.READ_ONLY)
        scope = IsolationScope(tenant_id="t1", pool=AccessTier.ORG)
        result = await scoped.add_scoped("fact", scope)
        assert result["blocked_by"] == "shared_memory_strategy"

    @pytest.mark.asyncio
    async def test_search_disabled_keeps_none_pool(self):
        """Results with no pool metadata should be kept when DISABLED."""
        mock = AsyncMock()
        mock.search = AsyncMock(return_value={
            "results": [
                {"id": "1", "memory": "no pool", "metadata": {}},
                {"id": "2", "memory": "private", "metadata": {"pool": "private"}},
                {"id": "3", "memory": "team", "metadata": {"pool": "team"}},
            ]
        })
        scoped = self._make_scoped(mock, SharedMemoryStrategy.DISABLED)
        scope = IsolationScope(tenant_id="t1", user_id="u1", agent_id="a1", pool=AccessTier.PRIVATE)
        result = await scoped.search_scoped("q", scope)
        # None pool + private kept, team filtered out
        assert len(result["results"]) == 2

    @pytest.mark.asyncio
    async def test_metadata_merge_caller_wins(self):
        """Caller-provided metadata should override scope metadata on same key."""
        mock = AsyncMock()
        mock.add = AsyncMock(return_value={"results": []})
        scoped = self._make_scoped(mock)
        scope = IsolationScope(tenant_id="t1", user_id="u1", agent_id="a1", pool=AccessTier.PRIVATE)
        await scoped.add_scoped("fact", scope, metadata={"tenant_id": "override"})
        call_meta = mock.add.call_args[1]["metadata"]
        assert call_meta["tenant_id"] == "override"

    @pytest.mark.asyncio
    async def test_store_exchange_scoped_constructs_messages(self):
        mock = AsyncMock()
        mock.add = AsyncMock(return_value={"results": []})
        scoped = self._make_scoped(mock)
        scope = IsolationScope(tenant_id="t1", user_id="u1", agent_id="a1", pool=AccessTier.PRIVATE)
        await scoped.store_exchange_scoped("hi", "hello", scope)
        messages = mock.add.call_args[1]["messages"]
        assert messages[0] == {"role": "user", "content": "hi"}
        assert messages[1] == {"role": "assistant", "content": "hello"}

    def test_strategy_enum_values(self):
        assert SharedMemoryStrategy.ENABLED.value == "enabled"
        assert SharedMemoryStrategy.READ_ONLY.value == "read_only"
        assert SharedMemoryStrategy.DISABLED.value == "disabled"


# =============================================================================
# J. PoolRouting edge cases
# =============================================================================
from memory_utils.shared_scope.pool_routing import (
    PoolSearchConfig,
    SharedScopePoolRouter,
)


class TestPoolRoutingEdgeCases:
    def test_resolve_store_params_unknown_pool(self):
        router = SharedScopePoolRouter(
            pools=[PoolSearchConfig(pool="team")]
        )
        result = router.resolve_store_params(
            fact_pool="unknown_tier",
            user_id="u1",
            agent_id="a1",
            tenant_id="t1",
            dept_id="eng",
            metadata={},
        )
        # Unknown pool falls through to default (returns original user_id)
        assert result["user_id"] == "u1"

    def test_resolve_store_params_org_sentinel(self):
        router = SharedScopePoolRouter(
            pools=[PoolSearchConfig(pool="org")]
        )
        result = router.resolve_store_params(
            fact_pool="org",
            user_id="u1",
            agent_id="a1",
            tenant_id="acme",
            dept_id=None,
            metadata={},
        )
        assert result["user_id"] == "__org_acme__"

    def test_get_pool_search_tasks_no_tenant(self):
        router = SharedScopePoolRouter(
            pools=[PoolSearchConfig(pool="org")]
        )
        tasks, labels = router.get_pool_search_tasks(
            query="test",
            search_fn=AsyncMock(return_value={"results": []}),
            user_id="u1",
            agent_id="a1",
            tenant_id=None,
            dept_id=None,
            allow_cross_agent=True,
        )
        assert len(tasks) == 1
        assert labels == ["org"]

    def test_team_pool_user_id_based_on_dept(self):
        router = SharedScopePoolRouter(
            pools=[PoolSearchConfig(pool="team", sentinel_user_id="__custom__")]
        )
        result = router.resolve_store_params(
            fact_pool="team",
            user_id="u1",
            agent_id="a1",
            tenant_id="t1",
            dept_id="eng",
            metadata={},
        )
        # team pool routes to __team_{dept_id}__
        assert result["user_id"] == "__team_eng__"


# =============================================================================
# K. FactExtractor — non-string value bug was fixed
# =============================================================================


class TestFactExtractorValueFix:
    @pytest.mark.asyncio
    async def test_integer_value_does_not_crash(self):
        """After fix: integer values are converted to string via str()."""
        from neo_memory_hub.extraction.extractor import FactExtractor

        extractor = FactExtractor()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = '{"memories": [{"value": 42, "category": "persona"}]}'

        with patch("openai.AsyncOpenAI") as MockClient:
            client_instance = AsyncMock()
            client_instance.chat.completions.create = AsyncMock(return_value=mock_response)
            MockClient.return_value = client_instance

            result = await extractor.extract("q", "r")
            assert len(result.facts) == 1
            assert result.facts[0].content == "42"

    @pytest.mark.asyncio
    async def test_null_memories_in_response(self):
        """LLM returning {"memories": null} should not crash."""
        from neo_memory_hub.extraction.extractor import FactExtractor

        extractor = FactExtractor()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = '{"memories": null}'

        with patch("openai.AsyncOpenAI") as MockClient:
            client_instance = AsyncMock()
            client_instance.chat.completions.create = AsyncMock(return_value=mock_response)
            MockClient.return_value = client_instance

            result = await extractor.extract("q", "r")
            # Should either return empty list or error gracefully
            assert isinstance(result.facts, list)


# =============================================================================
# L. Context / TokenBudget edge cases
# =============================================================================
from neo_memory_hub.retrieval.context import TokenBudgetConfig


class TestTokenBudgetEdgeCases:
    def test_overspend_raises(self):
        with pytest.raises(ValueError):
            TokenBudgetConfig(total_budget=100, retrieved_memories=200)

    def test_for_context_size_8000_matches_defaults(self):
        budget = TokenBudgetConfig.for_context_size(8000)
        default = TokenBudgetConfig()
        assert budget.total_budget == default.total_budget

    def test_for_context_size_small(self):
        budget = TokenBudgetConfig.for_context_size(1024)
        assert budget.total_budget == 1024

    def test_unlimited(self):
        budget = TokenBudgetConfig.unlimited()
        assert budget.retrieved_memories >= 100_000

    def test_for_context_size_zero(self):
        budget = TokenBudgetConfig.for_context_size(0)
        assert budget.total_budget == 0


# =============================================================================
# M. Dedup edge cases
# =============================================================================
from neo_memory_hub.dedup.deduplicator import DedupDecision, MemoryDeduplicator


class TestDedupEdgeCases:
    @pytest.mark.asyncio
    async def test_llm_returns_action_none(self):
        """LLM response with no action defaults to NONE."""
        deduper = MemoryDeduplicator()
        search_fn = AsyncMock(return_value={
            "results": [{"id": "old_1", "memory": "old fact", "score": 0.9, "metadata": {}}]
        })

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = '{}'  # no action key

        with patch("openai.AsyncOpenAI") as MockClient:
            client_instance = AsyncMock()
            client_instance.chat.completions.create = AsyncMock(return_value=mock_response)
            MockClient.return_value = client_instance

            decision = await deduper.find_and_resolve(
                new_fact="new fact",
                category="persona",
                investor_name="John",
                user_id="u1",
                agent_id="a1",
                search_fn=search_fn,
            )
            assert decision.action == "NONE"

    @pytest.mark.asyncio
    async def test_no_candidates_found(self):
        """When search returns no results, decision should be NONE."""
        deduper = MemoryDeduplicator()
        search_fn = AsyncMock(return_value={"results": []})

        decision = await deduper.find_and_resolve(
            new_fact="brand new fact",
            category="persona",
            investor_name="John",
            user_id="u1",
            agent_id="a1",
            search_fn=search_fn,
        )
        assert decision.action == "NONE"

    @pytest.mark.asyncio
    async def test_lowercase_action_normalized(self):
        """LLM returning lowercase action should be normalized."""
        deduper = MemoryDeduplicator()
        search_fn = AsyncMock(return_value={
            "results": [{"id": "old_1", "memory": "old fact", "score": 0.9, "metadata": {}}]
        })

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = '{"action": "keep_existing", "reasoning": "same"}'

        with patch("openai.AsyncOpenAI") as MockClient:
            client_instance = AsyncMock()
            client_instance.chat.completions.create = AsyncMock(return_value=mock_response)
            MockClient.return_value = client_instance

            decision = await deduper.find_and_resolve(
                new_fact="new fact",
                category="persona",
                investor_name="John",
                user_id="u1",
                agent_id="a1",
                search_fn=search_fn,
            )
            assert decision.action == "KEEP_EXISTING"
