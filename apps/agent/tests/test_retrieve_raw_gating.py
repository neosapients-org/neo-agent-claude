"""Tests for the debug-panel retrieve_raw gating in main.py.

The per-request debug-panel memory fetch (memory_manager.retrieve_raw) used to
run on EVERY request, duplicating the graph's own once-per-session memory fetch.
It is now gated to the FIRST turn of a session (empty chat_history). These tests
drive run_agent_stream and assert the call count for both cases.
"""

import sys
import os
import pytest
from unittest.mock import AsyncMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import app.main as main


def _empty_graph_events():
    """Patch replacement for agent_graph.astream_events: yields no events so the
    stream runs to completion quickly without invoking the real graph."""
    async def _gen(*a, **k):
        return
        yield  # noqa: unreachable — makes this an async generator
    return _gen


async def _drain(agen):
    async for _ in agen:
        pass


@pytest.mark.asyncio
async def test_first_turn_fetches_debug_memories():
    """Empty chat_history (turn 1) → retrieve_raw is called once."""
    raw = AsyncMock(return_value=[])
    with patch.object(main, "NS_PROBE_ENABLED", False), \
         patch.object(main.agent_graph, "astream_events", _empty_graph_events()), \
         patch.object(main.memory_manager, "retrieve_raw", raw):
        await _drain(main.run_agent_stream("hello", "user1", None, chat_history=[]))
    # Invoked once (it's fired as a background task; we assert it was kicked off,
    # not necessarily awaited-to-completion within the stream's lifetime).
    raw.assert_called_once()


@pytest.mark.asyncio
async def test_followup_turn_skips_debug_memories():
    """Non-empty chat_history (turn 2+) → retrieve_raw is NOT called."""
    raw = AsyncMock(return_value=[])
    history = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]
    with patch.object(main, "NS_PROBE_ENABLED", False), \
         patch.object(main.agent_graph, "astream_events", _empty_graph_events()), \
         patch.object(main.memory_manager, "retrieve_raw", raw):
        await _drain(main.run_agent_stream("and again", "user1", None, chat_history=history))
    raw.assert_not_called()
