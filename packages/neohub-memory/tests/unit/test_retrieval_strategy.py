"""Tests for MultiCategoryRetriever."""
from unittest.mock import AsyncMock

import pytest

from neo_memory_hub.retrieval.strategy import (
    CategorySearchConfig,
    MultiCategoryRetriever,
    RetrievalResult,
)


class TestNormalizeInvestorName:
    """Test static investor name normalization."""

    def test_none(self):
        assert MultiCategoryRetriever.normalize_investor_name(None) is None

    def test_empty(self):
        assert MultiCategoryRetriever.normalize_investor_name("") is None

    def test_null_like(self):
        for name in ["unknown", "Unknown Client", "None", "null"]:
            assert MultiCategoryRetriever.normalize_investor_name(name) is None

    def test_valid_names(self):
        assert MultiCategoryRetriever.normalize_investor_name("john doe") == "John Doe"
        assert MultiCategoryRetriever.normalize_investor_name("SENTHIL KUMAR") == "Senthil Kumar"

    def test_strips_whitespace(self):
        assert MultiCategoryRetriever.normalize_investor_name("  alice  ") == "Alice"


class TestInvestorNameToSubjectId:
    """Test investor name to subject_id conversion."""

    def test_none(self):
        assert MultiCategoryRetriever.investor_name_to_subject_id(None) is None

    def test_empty(self):
        assert MultiCategoryRetriever.investor_name_to_subject_id("") is None

    def test_conversion(self):
        assert MultiCategoryRetriever.investor_name_to_subject_id("John Doe") == "John_Doe"

    def test_special_chars_replaced(self):
        result = MultiCategoryRetriever.investor_name_to_subject_id("O'Brien Jr.")
        assert "'" not in result
        assert "." not in result


class TestMultiCategoryRetrieverInit:
    """Test MultiCategoryRetriever initialization."""

    def test_default_categories(self):
        r = MultiCategoryRetriever()
        assert len(r._categories) == 4
        names = {c.name for c in r._categories}
        assert names == {"persona", "preference", "episodic", "procedural"}

    def test_custom_categories(self):
        r = MultiCategoryRetriever(
            categories=[CategorySearchConfig(name="custom", threshold=0.1)]
        )
        assert len(r._categories) == 1
        assert r._categories[0].name == "custom"


class TestRetrieve:
    """Test the retrieve method with mocked search_fn."""

    @pytest.fixture
    def search_fn(self):
        """Create a mock search function."""
        fn = AsyncMock()
        fn.return_value = {"results": []}
        return fn

    @pytest.mark.asyncio
    async def test_basic_retrieval_fires_category_searches(self, search_fn):
        retriever = MultiCategoryRetriever(
            categories=[
                CategorySearchConfig(name="persona", threshold=0.2, limit=5),
                CategorySearchConfig(name="episodic", threshold=0.3, limit=3),
            ]
        )

        result = await retriever.retrieve(
            query="Tell me about client",
            search_fn=search_fn,
            user_id="u1",
            agent_id="a1",
        )

        assert isinstance(result, RetrievalResult)
        assert search_fn.call_count == 2

    @pytest.mark.asyncio
    async def test_parallel_execution(self, search_fn):
        retriever = MultiCategoryRetriever(parallel=True)
        await retriever.retrieve(
            query="test", search_fn=search_fn, user_id="u1", agent_id="a1"
        )
        assert search_fn.call_count == 4  # 4 default categories

    @pytest.mark.asyncio
    async def test_sequential_execution(self, search_fn):
        retriever = MultiCategoryRetriever(parallel=False)
        await retriever.retrieve(
            query="test", search_fn=search_fn, user_id="u1", agent_id="a1"
        )
        assert search_fn.call_count == 4

    @pytest.mark.asyncio
    async def test_deduplication_across_categories(self):
        """Same memory ID appearing in multiple categories should be deduplicated."""
        call_count = 0

        async def search_fn(**kwargs):
            nonlocal call_count
            call_count += 1
            return {
                "results": [
                    {"id": "shared_mem", "memory": "test", "metadata": {}},
                    {"id": f"unique_{call_count}", "memory": f"unique {call_count}", "metadata": {}},
                ]
            }

        retriever = MultiCategoryRetriever(
            categories=[
                CategorySearchConfig(name="persona"),
                CategorySearchConfig(name="preference"),
            ]
        )
        result = await retriever.retrieve(
            query="test", search_fn=search_fn, user_id="u1", agent_id="a1"
        )

        # shared_mem should appear once only
        ids = [m["id"] for m in result.memories]
        assert ids.count("shared_mem") == 1
        assert result.count == 3  # 1 shared + 2 unique

    @pytest.mark.asyncio
    async def test_investor_aware_threshold(self):
        """When investor_name is set, procedural should use lower threshold."""
        calls = []

        async def search_fn(**kwargs):
            calls.append(kwargs)
            return {"results": []}

        retriever = MultiCategoryRetriever(
            categories=[
                CategorySearchConfig(
                    name="procedural",
                    threshold=0.30,
                    investor_aware_threshold=0.15,
                ),
            ]
        )
        await retriever.retrieve(
            query="test",
            search_fn=search_fn,
            user_id="u1",
            agent_id="a1",
            investor_name="Senthil Kumar",
        )

        assert calls[0]["threshold"] == 0.15

    @pytest.mark.asyncio
    async def test_procedural_query_broadened_with_investor(self):
        calls = []

        async def search_fn(**kwargs):
            calls.append(kwargs)
            return {"results": []}

        retriever = MultiCategoryRetriever(
            categories=[
                CategorySearchConfig(
                    name="procedural",
                    threshold=0.3,
                    query_hint="reminders alerts procedures tasks notifications recurring",
                ),
            ]
        )
        await retriever.retrieve(
            query="portfolio review",
            search_fn=search_fn,
            user_id="u1",
            agent_id="a1",
            investor_name="Senthil Kumar",
        )

        assert "Senthil Kumar" in calls[0]["query"]
        assert "reminders" in calls[0]["query"]

    @pytest.mark.asyncio
    async def test_investor_filter_in_metadata(self):
        calls = []

        async def search_fn(**kwargs):
            calls.append(kwargs)
            return {"results": []}

        retriever = MultiCategoryRetriever(
            categories=[CategorySearchConfig(name="persona")]
        )
        await retriever.retrieve(
            query="test",
            search_fn=search_fn,
            user_id="u1",
            agent_id="a1",
            investor_name="John Doe",
        )

        assert calls[0]["metadata_filters"]["investor_name"] == "John Doe"

    @pytest.mark.asyncio
    async def test_retrieval_result_structure(self):
        async def search_fn(**kwargs):
            cat = kwargs.get("categories", ["unknown"])[0]
            return {
                "results": [
                    {"id": f"{cat}_1", "memory": f"Memory from {cat}", "metadata": {}},
                ]
            }

        retriever = MultiCategoryRetriever(
            categories=[
                CategorySearchConfig(name="persona"),
                CategorySearchConfig(name="episodic"),
            ]
        )
        result = await retriever.retrieve(
            query="test", search_fn=search_fn, user_id="u1", agent_id="a1"
        )

        assert result.count == 2
        assert result.user_id == "u1"
        assert result.agent_id == "a1"
        assert len(result.by_category["persona"]) == 1
        assert len(result.by_category["episodic"]) == 1
        # Check _retrieval_reason tag
        assert result.memories[0]["_retrieval_reason"] == "persona"

    @pytest.mark.asyncio
    async def test_search_failure_handled_gracefully(self):
        """Failed searches should not crash the whole retrieval."""
        call_count = 0

        async def search_fn(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise ValueError("Search failed")
            return {"results": [{"id": "ok_1", "memory": "test", "metadata": {}}]}

        retriever = MultiCategoryRetriever(
            categories=[
                CategorySearchConfig(name="persona"),
                CategorySearchConfig(name="episodic"),
            ],
            parallel=False,  # sequential to control failure order
        )
        result = await retriever.retrieve(
            query="test", search_fn=search_fn, user_id="u1", agent_id="a1"
        )

        # Should still return results from the successful search
        assert result.count == 1

    @pytest.mark.asyncio
    async def test_tenant_filter_propagated(self):
        calls = []

        async def search_fn(**kwargs):
            calls.append(kwargs)
            return {"results": []}

        retriever = MultiCategoryRetriever(
            categories=[CategorySearchConfig(name="persona")]
        )
        await retriever.retrieve(
            query="test",
            search_fn=search_fn,
            user_id="u1",
            agent_id="a1",
            tenant_id="acme",
        )

        assert calls[0]["metadata_filters"]["tenant_id"] == "acme"


class TestQueryHints:
    """Test query_hint field on CategorySearchConfig — G-07."""

    @pytest.mark.asyncio
    async def test_persona_query_enhanced_with_investor(self):
        calls = []

        async def search_fn(**kwargs):
            calls.append(kwargs)
            return {"results": []}

        retriever = MultiCategoryRetriever(
            categories=[
                CategorySearchConfig(
                    name="persona",
                    threshold=0.2,
                    query_hint="personality behavior risk profile",
                ),
            ]
        )
        await retriever.retrieve(
            query="what is his risk profile",
            search_fn=search_fn,
            user_id="u1",
            agent_id="a1",
            investor_name="Senthil Kumar",
        )

        q = calls[0]["query"]
        assert "Senthil Kumar" in q
        assert "what is his risk profile" in q
        assert "personality" in q

    @pytest.mark.asyncio
    async def test_no_hint_when_no_investor(self):
        calls = []

        async def search_fn(**kwargs):
            calls.append(kwargs)
            return {"results": []}

        retriever = MultiCategoryRetriever(
            categories=[
                CategorySearchConfig(
                    name="persona",
                    threshold=0.2,
                    query_hint="personality behavior risk profile",
                ),
            ]
        )
        await retriever.retrieve(
            query="what is his risk profile",
            search_fn=search_fn,
            user_id="u1",
            agent_id="a1",
        )

        # Without investor, raw query used
        assert calls[0]["query"] == "what is his risk profile"

    @pytest.mark.asyncio
    async def test_no_hint_when_query_override_set(self):
        calls = []

        async def search_fn(**kwargs):
            calls.append(kwargs)
            return {"results": []}

        retriever = MultiCategoryRetriever(
            categories=[
                CategorySearchConfig(
                    name="persona",
                    query_override="custom query",
                    query_hint="personality behavior",
                ),
            ]
        )
        await retriever.retrieve(
            query="test",
            search_fn=search_fn,
            user_id="u1",
            agent_id="a1",
            investor_name="John Doe",
        )

        # query_override takes precedence — no hints appended
        assert calls[0]["query"] == "custom query"

    @pytest.mark.asyncio
    async def test_all_categories_enhanced_with_investor(self):
        """All 4 categories get query hints when investor is known."""
        calls = []

        async def search_fn(**kwargs):
            calls.append(kwargs)
            return {"results": []}

        retriever = MultiCategoryRetriever(
            categories=[
                CategorySearchConfig(name="persona", query_hint="personality"),
                CategorySearchConfig(name="preference", query_hint="communication"),
                CategorySearchConfig(name="episodic", query_hint="events"),
                CategorySearchConfig(name="procedural", query_hint="reminders"),
            ]
        )
        await retriever.retrieve(
            query="review",
            search_fn=search_fn,
            user_id="u1",
            agent_id="a1",
            investor_name="Kumar",
        )

        for call in calls:
            assert "Kumar" in call["query"]

    @pytest.mark.asyncio
    async def test_investor_aware_threshold_all_categories(self):
        """G-08: All categories should use investor-aware thresholds."""
        calls = []

        async def search_fn(**kwargs):
            calls.append(kwargs)
            return {"results": []}

        retriever = MultiCategoryRetriever(
            categories=[
                CategorySearchConfig(name="persona", threshold=0.20, investor_aware_threshold=0.03),
                CategorySearchConfig(name="preference", threshold=0.15, investor_aware_threshold=0.03),
                CategorySearchConfig(name="episodic", threshold=0.20, investor_aware_threshold=0.03),
                CategorySearchConfig(name="procedural", threshold=0.30, investor_aware_threshold=0.15),
            ]
        )
        await retriever.retrieve(
            query="test",
            search_fn=search_fn,
            user_id="u1",
            agent_id="a1",
            investor_name="Senthil",
        )

        thresholds = {c["categories"][0]: c["threshold"] for c in calls}
        assert thresholds["persona"] == 0.03
        assert thresholds["preference"] == 0.03
        assert thresholds["episodic"] == 0.03
        assert thresholds["procedural"] == 0.15
