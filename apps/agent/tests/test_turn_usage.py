"""Per-turn token accounting.

The comparison UI puts cost side by side with the answer, so it needs a turn's token
totals the moment the turn ends — the observability pipeline lags 20-36s, far too late.

These numbers come from the SDK's own usage fields, not an estimate. `_astream` is the
case that matters: it previously read only `content_block_delta` and dropped usage
entirely, so every streamed turn contributed zero.
"""
import asyncio
import types

import pytest
from langchain_core.messages import HumanMessage

from app import llm as llm_mod


def _usage(inp, out):
    return types.SimpleNamespace(input_tokens=inp, output_tokens=out)


class _FakeMessages:
    """Stands in for client.messages, returning a canned non-streaming response."""

    def __init__(self, usage):
        self._usage = usage

    async def create(self, **payload):
        return types.SimpleNamespace(
            content=[types.SimpleNamespace(type="text", text="hi")],
            usage=self._usage,
            model=payload.get("model", "claude-sonnet-5"),
            stop_reason="end_turn",
        )


class _FakeStreamMessages:
    """Stands in for client.messages when stream=True.

    Mirrors the real event order: message_start carries input tokens, message_delta
    carries the running output count, content_block_delta carries text.
    """

    def __init__(self, inp, out):
        self._inp, self._out = inp, out

    async def create(self, **payload):
        inp, out = self._inp, self._out

        async def gen():
            yield types.SimpleNamespace(
                type="message_start",
                message=types.SimpleNamespace(usage=_usage(inp, 0)),
            )
            yield types.SimpleNamespace(
                type="content_block_delta",
                delta=types.SimpleNamespace(text="hi"),
            )
            yield types.SimpleNamespace(
                type="message_delta", usage=_usage(0, out)
            )

        return gen()


def _chat(monkeypatch, fake):
    chat = llm_mod.ClaudeChat(model="claude-sonnet-5", provider="anthropic")
    monkeypatch.setattr(
        llm_mod, "_client", lambda *a, **k: types.SimpleNamespace(messages=fake)
    )
    return chat


def test_reset_clears_previous_turn():
    llm_mod.reset_turn_usage()
    assert llm_mod.get_turn_usage() == {
        "input_tokens": 0, "output_tokens": 0, "calls": 0
    }


def test_non_streaming_call_accumulates(monkeypatch):
    chat = _chat(monkeypatch, _FakeMessages(_usage(100, 20)))
    llm_mod.reset_turn_usage()
    asyncio.run(chat._agenerate([HumanMessage(content="q")]))
    assert llm_mod.get_turn_usage() == {
        "input_tokens": 100, "output_tokens": 20, "calls": 1
    }


def test_streaming_call_accumulates(monkeypatch):
    """The regression that matters: streamed turns used to record nothing."""
    chat = _chat(monkeypatch, _FakeStreamMessages(300, 45))
    llm_mod.reset_turn_usage()

    async def drain():
        async for _ in chat._astream([HumanMessage(content="q")]):
            pass

    asyncio.run(drain())
    assert llm_mod.get_turn_usage() == {
        "input_tokens": 300, "output_tokens": 45, "calls": 1
    }


def test_multiple_calls_sum(monkeypatch):
    """A turn makes several model calls; the UI shows the turn total, not the last call."""
    chat = _chat(monkeypatch, _FakeMessages(_usage(10, 5)))
    llm_mod.reset_turn_usage()
    asyncio.run(chat._agenerate([HumanMessage(content="q")]))
    asyncio.run(chat._agenerate([HumanMessage(content="q")]))
    assert llm_mod.get_turn_usage() == {
        "input_tokens": 20, "output_tokens": 10, "calls": 2
    }
