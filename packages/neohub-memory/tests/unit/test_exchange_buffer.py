"""Tests for ExchangeBuffer."""
import asyncio
import time
from unittest.mock import AsyncMock

import pytest

from neo_memory_hub.buffering.exchange_buffer import BufferedExchange, ExchangeBuffer


class TestBufferedExchange:
    """Test the BufferedExchange dataclass."""

    def test_construction(self):
        ex = BufferedExchange(query="q", response="r", user_id="u1")
        assert ex.query == "q"
        assert ex.response == "r"
        assert ex.user_id == "u1"
        assert ex.tenant_id is None
        assert ex.session_id is None
        assert ex.investor_name is None
        assert ex.timestamp > 0

    def test_full_construction(self):
        ex = BufferedExchange(
            query="q",
            response="r",
            user_id="u1",
            tenant_id="t1",
            session_id="s1",
            investor_name="John",
        )
        assert ex.tenant_id == "t1"
        assert ex.session_id == "s1"
        assert ex.investor_name == "John"


class TestExchangeBufferInit:
    """Test ExchangeBuffer initialization."""

    def test_defaults(self):
        buf = ExchangeBuffer()
        assert buf._flush_threshold == 10
        assert buf._max_buffer_chars == 50000
        assert buf._flush_on_session_end is True

    def test_custom(self):
        buf = ExchangeBuffer(
            flush_threshold=5, max_buffer_chars=1000, flush_on_session_end=False
        )
        assert buf._flush_threshold == 5
        assert buf._max_buffer_chars == 1000


class TestAddExchange:
    """Test add_exchange with threshold triggering."""

    @pytest.mark.asyncio
    async def test_add_below_threshold(self):
        buf = ExchangeBuffer(flush_threshold=3)
        result = await buf.add_exchange(query="q1", response="r1", user_id="u1")
        assert result is None
        assert buf.get_buffer_size("u1") == 1

    @pytest.mark.asyncio
    async def test_flush_at_count_threshold(self):
        buf = ExchangeBuffer(flush_threshold=3)
        await buf.add_exchange(query="q1", response="r1", user_id="u1")
        await buf.add_exchange(query="q2", response="r2", user_id="u1")
        result = await buf.add_exchange(query="q3", response="r3", user_id="u1")

        assert result is not None
        assert len(result) == 3
        assert result[0].query == "q1"
        assert result[2].query == "q3"
        # Buffer should be drained
        assert buf.get_buffer_size("u1") == 0

    @pytest.mark.asyncio
    async def test_flush_at_chars_threshold(self):
        buf = ExchangeBuffer(flush_threshold=100, max_buffer_chars=20)
        result = await buf.add_exchange(
            query="x" * 15, response="y" * 15, user_id="u1"
        )

        assert result is not None
        assert len(result) == 1

    @pytest.mark.asyncio
    async def test_session_isolation(self):
        buf = ExchangeBuffer(flush_threshold=2)
        await buf.add_exchange(
            query="q1", response="r1", user_id="u1", session_id="s1"
        )
        await buf.add_exchange(
            query="q2", response="r2", user_id="u1", session_id="s2"
        )

        assert buf.get_buffer_size("u1", "s1") == 1
        assert buf.get_buffer_size("u1", "s2") == 1
        assert buf.active_sessions == 2

    @pytest.mark.asyncio
    async def test_user_isolation(self):
        buf = ExchangeBuffer(flush_threshold=5)
        await buf.add_exchange(query="q1", response="r1", user_id="u1")
        await buf.add_exchange(query="q2", response="r2", user_id="u2")

        assert buf.get_buffer_size("u1") == 1
        assert buf.get_buffer_size("u2") == 1


class TestFlushSession:
    """Test flush_session."""

    @pytest.mark.asyncio
    async def test_flush_returns_buffered_exchanges(self):
        buf = ExchangeBuffer(flush_threshold=100)
        await buf.add_exchange(query="q1", response="r1", user_id="u1")
        await buf.add_exchange(query="q2", response="r2", user_id="u1")

        result = await buf.flush_session("u1")
        assert result is not None
        assert len(result) == 2
        assert buf.get_buffer_size("u1") == 0

    @pytest.mark.asyncio
    async def test_flush_empty_session(self):
        buf = ExchangeBuffer()
        result = await buf.flush_session("nonexistent")
        assert result is None

    @pytest.mark.asyncio
    async def test_flush_disabled(self):
        buf = ExchangeBuffer(flush_on_session_end=False)
        await buf.add_exchange(query="q1", response="r1", user_id="u1")
        result = await buf.flush_session("u1")
        assert result is None
        assert buf.get_buffer_size("u1") == 1


class TestConsolidateExchanges:
    """Test consolidate_exchanges."""

    def test_consolidation_format(self):
        buf = ExchangeBuffer()
        exchanges = [
            BufferedExchange(query="Hello", response="Hi there", user_id="u1"),
            BufferedExchange(query="How are you?", response="Fine thanks", user_id="u1"),
        ]

        text = buf.consolidate_exchanges(exchanges)
        assert "--- Exchange 1 ---" in text
        assert "--- Exchange 2 ---" in text
        assert "User: Hello" in text
        assert "Assistant: Hi there" in text
        assert "User: How are you?" in text

    def test_empty_consolidation(self):
        buf = ExchangeBuffer()
        text = buf.consolidate_exchanges([])
        assert text == ""


class TestCleanupSession:
    """Test cleanup_session."""

    @pytest.mark.asyncio
    async def test_cleanup_removes_buffer(self):
        buf = ExchangeBuffer()
        await buf.add_exchange(query="q1", response="r1", user_id="u1", session_id="s1")
        assert buf.get_buffer_size("u1", "s1") == 1

        buf.cleanup_session("u1", "s1")
        assert buf.get_buffer_size("u1", "s1") == 0

    @pytest.mark.asyncio
    async def test_cleanup_nonexistent_session(self):
        buf = ExchangeBuffer()
        # Should not raise
        buf.cleanup_session("u1", "nonexistent")


class TestActiveSessionsProperty:
    """Test active_sessions property."""

    @pytest.mark.asyncio
    async def test_active_sessions_count(self):
        buf = ExchangeBuffer()
        assert buf.active_sessions == 0

        await buf.add_exchange(query="q1", response="r1", user_id="u1")
        assert buf.active_sessions == 1

        await buf.add_exchange(query="q2", response="r2", user_id="u2")
        assert buf.active_sessions == 2

        buf.cleanup_session("u1")
        assert buf.active_sessions == 1


class TestBufferedExchangeAgentId:
    """Test agent_id field on BufferedExchange."""

    def test_default_agent_id_is_none(self):
        ex = BufferedExchange(query="q", response="r", user_id="u1")
        assert ex.agent_id is None

    def test_explicit_agent_id(self):
        ex = BufferedExchange(
            query="q", response="r", user_id="u1", agent_id="agent_x"
        )
        assert ex.agent_id == "agent_x"


class TestAddExchangeAgentId:
    """Test agent_id plumbing through add_exchange."""

    @pytest.mark.asyncio
    async def test_agent_id_stored_in_exchange(self):
        buf = ExchangeBuffer(flush_threshold=2)
        await buf.add_exchange(
            query="q1", response="r1", user_id="u1", agent_id="a1"
        )
        result = await buf.add_exchange(
            query="q2", response="r2", user_id="u1", agent_id="a1"
        )
        assert result is not None
        assert result[0].agent_id == "a1"
        assert result[1].agent_id == "a1"


class TestIdleTimeoutConfig:
    """Test idle_timeout_seconds configuration."""

    def test_default_idle_timeout(self):
        buf = ExchangeBuffer()
        assert buf.idle_timeout_seconds == 60.0

    def test_custom_idle_timeout(self):
        buf = ExchangeBuffer(idle_timeout_seconds=120.0)
        assert buf.idle_timeout_seconds == 120.0

    def test_zero_disables_idle_timer(self):
        buf = ExchangeBuffer(idle_timeout_seconds=0)
        assert buf.idle_timeout_seconds == 0


class TestLastActivity:
    """Test activity tracking."""

    @pytest.mark.asyncio
    async def test_last_activity_updated_on_add(self):
        buf = ExchangeBuffer(idle_timeout_seconds=0)  # disable timer tasks
        before = time.time()
        await buf.add_exchange(query="q1", response="r1", user_id="u1")
        after = time.time()

        last = buf.get_last_activity("u1")
        assert last is not None
        assert before <= last <= after

    @pytest.mark.asyncio
    async def test_idle_seconds(self):
        buf = ExchangeBuffer(idle_timeout_seconds=0)
        await buf.add_exchange(query="q1", response="r1", user_id="u1")
        idle = buf.get_idle_seconds("u1")
        assert idle < 1.0  # just added

    def test_idle_seconds_no_activity(self):
        buf = ExchangeBuffer()
        idle = buf.get_idle_seconds("u1")
        assert idle == 0.0  # no buffer = 0.0 per docstring


class TestFlushCallback:
    """Test set_flush_callback and idle flush."""

    def test_set_callback(self):
        buf = ExchangeBuffer()
        cb = AsyncMock()
        buf.set_flush_callback(cb)
        assert buf._flush_callback is cb

    @pytest.mark.asyncio
    async def test_idle_flush_fires_callback(self):
        """Idle flush task should invoke callback after timeout."""
        callback = AsyncMock()
        buf = ExchangeBuffer(
            flush_threshold=100,  # won't trigger count flush
            idle_timeout_seconds=0.1,  # 100ms for fast test
        )
        buf.set_flush_callback(callback)

        await buf.add_exchange(query="q1", response="r1", user_id="u1")
        # Wait for idle timer to fire
        await asyncio.sleep(0.3)

        callback.assert_called_once()
        args = callback.call_args[0]
        session_key = args[0]
        exchanges = args[1]
        assert session_key == "u1:default"
        assert len(exchanges) == 1
        assert exchanges[0].query == "q1"
        # Buffer should be drained
        assert buf.get_buffer_size("u1") == 0

    @pytest.mark.asyncio
    async def test_idle_timer_cancelled_on_explicit_flush(self):
        """Explicit flush should cancel idle timer so callback doesn't fire."""
        callback = AsyncMock()
        buf = ExchangeBuffer(
            flush_threshold=100,
            idle_timeout_seconds=0.2,
        )
        buf.set_flush_callback(callback)

        await buf.add_exchange(query="q1", response="r1", user_id="u1")
        # Flush immediately before idle fires
        result = await buf.flush_session("u1")
        assert result is not None

        # Wait past the idle timeout
        await asyncio.sleep(0.4)
        # Callback should NOT have been called (timer was cancelled)
        callback.assert_not_called()

    @pytest.mark.asyncio
    async def test_idle_timer_restarted_on_new_exchange(self):
        """Adding another exchange should restart the idle timer."""
        callback = AsyncMock()
        buf = ExchangeBuffer(
            flush_threshold=100,
            idle_timeout_seconds=0.2,
        )
        buf.set_flush_callback(callback)

        await buf.add_exchange(query="q1", response="r1", user_id="u1")
        await asyncio.sleep(0.1)  # Half the timeout
        # Add another exchange - should restart timer
        await buf.add_exchange(query="q2", response="r2", user_id="u1")
        await asyncio.sleep(0.15)  # Past original timeout but not new one
        callback.assert_not_called()

        # Now wait for the full new timeout
        await asyncio.sleep(0.15)
        callback.assert_called_once()
        exchanges = callback.call_args[0][1]
        assert len(exchanges) == 2

    @pytest.mark.asyncio
    async def test_count_threshold_cancels_idle_timer(self):
        """Count threshold flush should cancel idle timer."""
        callback = AsyncMock()
        buf = ExchangeBuffer(
            flush_threshold=2,
            idle_timeout_seconds=0.2,
        )
        buf.set_flush_callback(callback)

        await buf.add_exchange(query="q1", response="r1", user_id="u1")
        result = await buf.add_exchange(query="q2", response="r2", user_id="u1")
        assert result is not None  # count threshold triggered

        # Wait past idle timeout
        await asyncio.sleep(0.4)
        # Callback should NOT fire (buffer was already drained by count flush)
        callback.assert_not_called()


class TestCleanupSessionIdleTimer:
    """Test cleanup_session also cleans idle timer."""

    @pytest.mark.asyncio
    async def test_cleanup_cancels_idle_timer(self):
        callback = AsyncMock()
        buf = ExchangeBuffer(
            flush_threshold=100,
            idle_timeout_seconds=0.2,
        )
        buf.set_flush_callback(callback)

        await buf.add_exchange(query="q1", response="r1", user_id="u1")
        buf.cleanup_session("u1")

        await asyncio.sleep(0.4)
        callback.assert_not_called()
        assert buf.active_sessions == 0
