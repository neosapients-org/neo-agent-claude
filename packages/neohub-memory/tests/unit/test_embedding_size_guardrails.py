import pytest

from neo_memory_hub.integrations.memory import AsyncMemory, MemoryConfig


@pytest.mark.unit
@pytest.mark.asyncio
async def test_memory_add_rejects_oversized_embedding_input_by_tokens(monkeypatch) -> None:
    # Set a tiny token limit so we can trigger deterministically.
    cfg = MemoryConfig(max_embedding_input_tokens=1, max_embedding_input_chars=1_000_000)
    m = AsyncMemory(config=cfg)

    # Avoid importing/initializing real mem0 in this unit test.
    m._initialized = True

    class DummyMem0:
        async def add(self, *args, **kwargs):  # noqa: ANN001
            raise AssertionError("should not call mem0")

    m._mem0 = DummyMem0()

    with pytest.raises(ValueError, match="Input too large for embedding token limits"):
        await m.add("this will exceed", user_id="u")


@pytest.mark.unit
@pytest.mark.asyncio
async def test_memory_add_rejects_oversized_embedding_input_by_chars(monkeypatch) -> None:
    cfg = MemoryConfig(max_embedding_input_tokens=1_000_000, max_embedding_input_chars=3)
    m = AsyncMemory(config=cfg)

    m._initialized = True

    class DummyMem0:
        async def add(self, *args, **kwargs):  # noqa: ANN001
            raise AssertionError("should not call mem0")

    m._mem0 = DummyMem0()

    with pytest.raises(ValueError, match="Input too large for embedding"):
        await m.add("abcd", user_id="u")


@pytest.mark.unit
@pytest.mark.asyncio
async def test_memory_update_rejects_oversized_embedding_input_by_chars() -> None:
    cfg = MemoryConfig(max_embedding_input_tokens=1_000_000, max_embedding_input_chars=3)
    m = AsyncMemory(config=cfg)

    m._initialized = True

    class DummyMem0:
        async def update(self, *args, **kwargs):  # noqa: ANN001
            raise AssertionError("should not call mem0")

    m._mem0 = DummyMem0()

    with pytest.raises(ValueError, match="Input too large for embedding"):
        await m.update("mem_1", "abcd")


@pytest.mark.unit
@pytest.mark.asyncio
async def test_memory_search_rejects_oversized_query_by_chars() -> None:
    cfg = MemoryConfig(max_embedding_input_tokens=1_000_000, max_embedding_input_chars=3)
    m = AsyncMemory(config=cfg)

    m._initialized = True

    class DummyMem0:
        async def search(self, *args, **kwargs):  # noqa: ANN001
            raise AssertionError("should not call mem0")

    m._mem0 = DummyMem0()

    with pytest.raises(ValueError, match="Input too large for embedding"):
        await m.search("abcd", user_id="u")


@pytest.mark.unit
@pytest.mark.asyncio
async def test_async_memory_update_rejects_oversized_embedding_input_by_chars() -> None:
    cfg = MemoryConfig(max_embedding_input_tokens=1_000_000, max_embedding_input_chars=3)
    m = AsyncMemory(config=cfg)

    m._initialized = True

    class DummyMem0:
        async def update(self, *args, **kwargs):  # noqa: ANN001
            raise AssertionError("should not call mem0")

    m._mem0 = DummyMem0()

    with pytest.raises(ValueError, match="Input too large for embedding"):
        await m.update("mem_1", "abcd")
