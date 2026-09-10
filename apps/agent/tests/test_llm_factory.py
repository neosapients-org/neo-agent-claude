"""The Bedrock adapter, and why it exists at all.

The agent builds a chat model in 14 places and passes LangChain message objects around in
roughly 96 more. Moving it to Claude on Bedrock had three candidate routes and two of them
are dead ends:

  - `langchain_aws.ChatBedrockConverse` drops straight in, but reaches AWS through boto3,
    which ns_probe does not instrument. Every model call would be invisible and the cost
    experiment would produce no numbers at all.
  - `langchain_anthropic.ChatAnthropic` cannot be pointed at Bedrock: it builds its own
    client from `anthropic_api_key` / `anthropic_api_url` and exposes no way to inject one,
    so SigV4 auth is unreachable.

So this adapter wraps the Anthropic SDK's own async client — whose `.messages` resource IS
`anthropic.resources.AsyncMessages`, the class ns_probe patches — in the LangChain
interface the existing call sites already speak. That identity is the whole point, and
`test_it_calls_through_the_instrumented_client` is what stops someone swapping in a boto3
client and silently zeroing the experiment.

The other trap these cover: Anthropic takes `system` as a TOP-LEVEL request parameter, not
a message role. A conversion that leaves SystemMessage in the messages list either errors
or silently demotes the system prompt to a user turn.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import llm as llm_mod  # noqa: E402


# --- stub Anthropic client -----------------------------------------------------------

class _StubMessages:
    """Records what the adapter actually sent, which is what the tests assert on."""

    def __init__(self, sink):
        self._sink = sink

    async def create(self, **kwargs):
        self._sink.append(kwargs)
        if kwargs.get("stream"):
            async def _agen():
                for piece in ("Hel", "lo"):
                    yield types.SimpleNamespace(
                        type="content_block_delta",
                        delta=types.SimpleNamespace(text=piece),
                    )
            return _agen()
        return types.SimpleNamespace(
            content=[types.SimpleNamespace(type="text", text="Hello")],
            model=kwargs["model"],
            stop_reason="end_turn",
            usage=types.SimpleNamespace(input_tokens=11, output_tokens=7),
        )


class _StubClient:
    def __init__(self, sink, **kwargs):
        self.init_kwargs = kwargs
        self.messages = _StubMessages(sink)


class _Sink(list):
    """The payloads sent, plus the clients constructed. A bare list cannot carry both."""

    made: list


@pytest.fixture
def sent(monkeypatch):
    """The request payloads the adapter sent to the SDK, newest last."""
    sink = _Sink()
    sink.made = []

    def _factory(**kwargs):
        client = _StubClient(sink, **kwargs)
        sink.made.append(client)
        return client

    monkeypatch.setattr(llm_mod, "AsyncAnthropic", _factory)
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    # The client is cached per region, so a stub installed after a real one was built
    # would never be used — and the test would pass while exercising nothing.
    llm_mod._client.cache_clear()
    monkeypatch.setenv("AWS_REGION", "us-east-1")
    monkeypatch.setenv("LLM_MODEL", "claude-sonnet-4-6")
    yield sink
    llm_mod._client.cache_clear()


# --- the LangChain surface the 14 call sites use ------------------------------------

class TestItSpeaksTheInterfaceTheAgentAlreadyUses:
    async def test_ainvoke_returns_the_assistant_text(self, sent):
        model = llm_mod.make_llm()

        result = await model.ainvoke([HumanMessage(content="hi")])

        assert isinstance(result, AIMessage)
        assert result.content == "Hello"

    async def test_astream_yields_the_pieces_in_order(self, sent):
        model = llm_mod.make_llm()

        pieces = [chunk.content async for chunk in model.astream([HumanMessage(content="hi")])]

        assert "".join(pieces) == "Hello"

    async def test_usage_is_reported_back_on_the_message(self, sent):
        """LangChain callers read token counts off usage_metadata; ns_probe reads them off
        the SDK call. Both must see them, or one of the two views reads zero."""
        model = llm_mod.make_llm()

        result = await model.ainvoke([HumanMessage(content="hi")])

        assert result.usage_metadata["input_tokens"] == 11
        assert result.usage_metadata["output_tokens"] == 7


# --- the system-prompt trap --------------------------------------------------------

class TestSystemPromptsAreLiftedOutOfTheMessageList:
    async def test_a_system_message_becomes_the_top_level_system_param(self, sent):
        model = llm_mod.make_llm()

        await model.ainvoke([
            SystemMessage(content="you are terse"),
            HumanMessage(content="hi"),
        ])

        payload = sent[0]
        assert payload["system"] == "you are terse"
        assert [m["role"] for m in payload["messages"]] == ["user"], (
            "SystemMessage left in the messages list — Anthropic takes system as a "
            "top-level parameter, and a stray system role is either an error or a "
            "silently demoted system prompt"
        )

    async def test_several_system_messages_are_joined_not_dropped(self, sent):
        model = llm_mod.make_llm()

        await model.ainvoke([
            SystemMessage(content="rule one"),
            SystemMessage(content="rule two"),
            HumanMessage(content="hi"),
        ])

        assert "rule one" in sent[0]["system"] and "rule two" in sent[0]["system"]

    async def test_no_system_param_is_sent_when_there_is_no_system_message(self, sent):
        model = llm_mod.make_llm()

        await model.ainvoke([HumanMessage(content="hi")])

        assert "system" not in sent[0], "an empty system prompt must be absent, not ''"


# --- provider wiring -------------------------------------------------------------

class TestProviderWiring:
    async def test_the_model_id_is_sent_unprefixed(self, sent):
        """First-party ids are bare. The `anthropic.` prefix is Bedrock-only and would be
        an unknown model here."""
        model = llm_mod.make_llm()

        await model.ainvoke([HumanMessage(content="hi")])

        assert sent[0]["model"] == "claude-sonnet-4-6"

    async def test_it_calls_through_the_instrumented_client(self, sent):
        """The adapter must reach Claude via the Anthropic SDK.

        `AsyncAnthropic().messages` is `anthropic.resources.AsyncMessages`, the class
        ns_probe patches (verified). Any other transport — boto3, raw httpx — bypasses
        instrumentation and every Claude call records zero tokens, with no error to notice.
        """
        model = llm_mod.make_llm()
        await model.ainvoke([HumanMessage(content="hi")])

        assert sent.made, "no Anthropic SDK client was constructed"


class TestSamplingParamsAreModelAware:
    """`temperature` is not universally accepted, and the failure is a hard 400.

    All ten call sites pass `temperature=0.0`. On Sonnet 4.6 that is allowed. On Sonnet 5
    and the Opus 5 family, sampling parameters were REMOVED and are rejected outright — so
    flipping `LLM_MODEL` to a newer (and cheaper) model would 400 every single call. Since
    this is a cost experiment, someone will try exactly that.
    """

    async def test_temperature_is_sent_to_a_model_that_accepts_it(self, sent):
        model = llm_mod.make_llm(temperature=0.0)

        await model.ainvoke([HumanMessage(content="hi")])

        assert sent[0]["temperature"] == 0.0

    async def test_temperature_is_dropped_for_a_model_that_rejects_it(self, sent, monkeypatch):
        monkeypatch.setenv("LLM_MODEL", "claude-sonnet-5")
        model = llm_mod.make_llm(temperature=0.0)

        await model.ainvoke([HumanMessage(content="hi")])

        assert "temperature" not in sent[0], (
            "temperature sent to a model that removed sampling params — this is a 400, "
            "not a warning"
        )

    async def test_budget_tokens_is_stripped_from_thinking(self, sent):
        """`budget_tokens` is deprecated on 4.6 and a 400 on 5. Adaptive thinking replaced it."""
        model = llm_mod.make_llm(thinking={"type": "adaptive", "budget_tokens": 2048})

        await model.ainvoke([HumanMessage(content="hi")])

        assert "budget_tokens" not in sent[0].get("thinking", {})

    async def test_budget_tokens_is_never_sent(self, sent):
        """`budget_tokens` returns a 400 on Sonnet 5. Adaptive thinking replaced it."""
        model = llm_mod.make_llm()

        await model.ainvoke([HumanMessage(content="hi")])

        thinking = sent[0].get("thinking") or {}
        assert "budget_tokens" not in thinking

    async def test_no_assistant_prefill_is_sent(self, sent):
        """A trailing assistant turn is a prefill, which is a 400 on Sonnet 5."""
        model = llm_mod.make_llm()

        await model.ainvoke([
            HumanMessage(content="hi"),
            AIMessage(content="partial"),
        ])

        assert sent[0]["messages"][-1]["role"] != "assistant", (
            "a trailing assistant message is a prefill and Sonnet 5 rejects it with a 400"
        )


# --- structured output ------------------------------------------------------------

class _Plan(BaseModel):
    """Stand-in for QueryPlan / WealthEntities, the two real structured-output schemas."""

    steps: list[str] = Field(description="ordered steps")
    reason: str = Field(default="", description="why")


class _StubToolMessages(_StubMessages):
    """Answers with a tool_use block, which is how Anthropic returns structured output."""

    async def create(self, **kwargs):
        self._sink.append(kwargs)
        return types.SimpleNamespace(
            content=[types.SimpleNamespace(
                type="tool_use",
                id="toolu_01",
                name=kwargs["tools"][0]["name"],
                input={"steps": ["a", "b"], "reason": "because"},
            )],
            model=kwargs["model"],
            stop_reason="tool_use",
            usage=types.SimpleNamespace(input_tokens=11, output_tokens=7),
        )


@pytest.fixture
def tool_sent(monkeypatch):
    sink = _Sink()
    sink.made = []

    def _factory(**kwargs):
        client = _StubClient(sink, **kwargs)
        client.messages = _StubToolMessages(sink)
        sink.made.append(client)
        return client

    monkeypatch.setattr(llm_mod, "AsyncAnthropic", _factory)
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    llm_mod._client.cache_clear()
    monkeypatch.setenv("AWS_REGION", "us-east-1")
    monkeypatch.setenv("LLM_MODEL", "claude-sonnet-4-6")
    yield sink
    llm_mod._client.cache_clear()


class TestStructuredOutput:
    """Two real call sites depend on this: planner.py:336 and intent_enrichment.py:1326.

    Both call `with_structured_output(SomePydanticModel)`, whose default implementation
    routes through `bind_tools`. Without bind_tools the adapter raises NotImplementedError
    and those two nodes fail outright.
    """

    async def test_with_structured_output_returns_a_populated_model(self, tool_sent):
        model = llm_mod.make_llm()

        result = await model.with_structured_output(_Plan).ainvoke(
            [HumanMessage(content="plan it")]
        )

        assert isinstance(result, _Plan)
        assert result.steps == ["a", "b"]
        assert result.reason == "because"

    async def test_tools_are_sent_in_anthropic_shape_not_openai_shape(self, tool_sent):
        model = llm_mod.make_llm()

        await model.with_structured_output(_Plan).ainvoke([HumanMessage(content="plan it")])

        tool = tool_sent[0]["tools"][0]
        assert set(tool) >= {"name", "input_schema"}, (
            "Anthropic takes name/description/input_schema; the OpenAI "
            "function/parameters shape is rejected"
        )
        assert "function" not in tool and "parameters" not in tool
        assert tool["input_schema"]["properties"]["steps"]

    async def test_a_tool_use_reply_becomes_langchain_tool_calls(self, tool_sent):
        """The base class's parser reads .tool_calls; a tool_use block that is not
        translated leaves it empty and structured output silently returns None."""
        model = llm_mod.make_llm().bind_tools([_Plan])

        result = await model.ainvoke([HumanMessage(content="plan it")])

        assert result.tool_calls, "tool_use block was not translated to tool_calls"
        assert result.tool_calls[0]["name"] == "_Plan"
        assert result.tool_calls[0]["args"]["steps"] == ["a", "b"]
        assert result.tool_calls[0]["id"] == "toolu_01"
