"""Tests for MemoryEngine ABC."""
import pytest

from neo_memory_hub.integrations.base import MemoryEngine


class TestMemoryEngineABC:
    """Verify MemoryEngine ABC contract."""

    def test_cannot_instantiate_directly(self):
        """MemoryEngine is abstract and cannot be instantiated."""
        with pytest.raises(TypeError, match="abstract"):
            MemoryEngine()

    def test_concrete_implementation_must_implement_all_methods(self):
        """A subclass missing any method raises TypeError."""

        class PartialEngine(MemoryEngine):
            async def initialize(self):
                pass

            async def add(self, messages, **kwargs):
                return {}

            # Missing: search, get, get_all, update, delete, close

        with pytest.raises(TypeError, match="abstract"):
            PartialEngine()

    def test_concrete_implementation_works(self):
        """A complete subclass can be instantiated."""

        class CompleteEngine(MemoryEngine):
            async def initialize(self):
                pass

            async def add(self, messages, **kwargs):
                return {"results": []}

            async def search(self, query, **kwargs):
                return {"results": []}

            async def get(self, memory_id):
                return None

            async def get_all(self, **kwargs):
                return {"results": []}

            async def update(self, memory_id, data):
                return {"message": "updated"}

            async def delete(self, memory_id):
                return {"message": "deleted"}

            async def close(self):
                pass

        engine = CompleteEngine()
        assert engine is not None

    @pytest.mark.asyncio
    async def test_complete_engine_methods_callable(self):
        """All methods on a complete implementation are callable."""

        class MockEngine(MemoryEngine):
            async def initialize(self):
                self.initialized = True

            async def add(self, messages, **kwargs):
                return {"results": [{"id": "m1", "memory": str(messages)}]}

            async def search(self, query, **kwargs):
                return {"results": [{"id": "m1", "memory": "found", "score": 0.9}]}

            async def get(self, memory_id):
                return {"id": memory_id, "memory": "test"}

            async def get_all(self, **kwargs):
                return {"results": [{"id": "m1"}]}

            async def update(self, memory_id, data):
                return {"message": "updated"}

            async def delete(self, memory_id):
                return {"message": "deleted"}

            async def close(self):
                self.initialized = False

        engine = MockEngine()
        await engine.initialize()
        assert engine.initialized is True

        add_result = await engine.add("test fact", user_id="u1")
        assert add_result["results"][0]["id"] == "m1"

        search_result = await engine.search("query", user_id="u1", limit=5)
        assert search_result["results"][0]["score"] == 0.9

        get_result = await engine.get("m1")
        assert get_result["id"] == "m1"

        get_all_result = await engine.get_all(user_id="u1")
        assert len(get_all_result["results"]) == 1

        update_result = await engine.update("m1", "updated content")
        assert update_result["message"] == "updated"

        delete_result = await engine.delete("m1")
        assert delete_result["message"] == "deleted"

        await engine.close()
        assert engine.initialized is False
