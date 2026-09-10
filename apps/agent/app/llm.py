"""The one place the agent builds a chat model.

Before this module the agent constructed `ChatOpenAI(...)` in 14 separate places, each with
its own model name and settings. Moving providers meant editing 14 files and hoping none was
missed — and a missed one keeps working, just against a different provider, which is the
worst kind of bug in a cost comparison.

WHY THIS ADAPTER EXISTS RATHER THAN AN OFF-THE-SHELF CLASS
----------------------------------------------------------
Two ready-made options exist for Claude on Bedrock and both are dead ends:

  * ``langchain_aws.ChatBedrockConverse`` drops straight in, but reaches AWS through boto3.
    ns_probe does not instrument boto3, so every model call would be invisible: no tokens,
    no cost, no error to tell you. The cost experiment would output nothing.
  * ``langchain_anthropic.ChatAnthropic`` cannot be pointed at Bedrock at all. It builds its
    own client from ``anthropic_api_key`` / ``anthropic_api_url`` and exposes no injection
    point, so SigV4 auth is unreachable.

So this wraps ``AsyncAnthropicBedrockMantle``, whose ``.messages`` resource *is*
``anthropic.resources.AsyncMessages`` — the exact class ns_probe patches (verified on
anthropic 0.125 and 1.4). One instrumented seam, every platform.

The wrapper is deliberately thin. The agent's LangChain usage is narrow — measured: 21
``ainvoke``, 2 ``astream``, 2 ``with_structured_output``, no ``bind_tools``, no LCEL chains
— so implementing ``_agenerate`` and ``_astream`` is enough for the base class to supply the
rest, and the change stays in one file instead of spreading across ~96 message-object
references.
"""

from __future__ import annotations

import logging
import os
from functools import lru_cache
from typing import Any, AsyncIterator, Optional, Sequence

from anthropic import AsyncAnthropic, AsyncAnthropicBedrockMantle
from langchain_core.callbacks import AsyncCallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    SystemMessage,
)
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.utils.function_calling import convert_to_openai_tool

from contextvars import ContextVar

# Per-turn token totals, read by the side-by-side comparison UI the moment a turn ends.
#
# The observability pipeline stays the system of record for cost, but it lags 20-36s
# between a span being emitted and the row being queryable — far too late to show numbers
# next to the answer. What is accumulated here is the SDK's own usage field, the same
# figure the pipeline will later price, so this is a faster view of one number rather than
# a second, disagreeing estimate.
_turn_usage: ContextVar[dict | None] = ContextVar("_turn_usage", default=None)

_ZERO = {"input_tokens": 0, "output_tokens": 0, "calls": 0}


def reset_turn_usage() -> None:
    """Begin a new turn. Call once per user message, before the graph runs."""
    _turn_usage.set(dict(_ZERO))


def get_turn_usage() -> dict:
    """Totals for the current turn; zeros when no turn was started."""
    return dict(_turn_usage.get() or _ZERO)


def _record_usage(inp: int, out: int) -> None:
    """Add one model call. A no-op outside a turn, so library use never fails here."""
    acc = _turn_usage.get()
    if acc is None:
        return
    acc["input_tokens"] += inp or 0
    acc["output_tokens"] += out or 0
    acc["calls"] += 1

logger = logging.getLogger(__name__)

# Bedrock model ids are namespaced; the bare first-party id does not resolve there.
BEDROCK_PREFIX = "anthropic."

DEFAULT_MODEL = "claude-sonnet-4-6"
DEFAULT_MAX_TOKENS = 4096


def _bedrock_model_id(model: str) -> str:
    return model if model.startswith(BEDROCK_PREFIX) else BEDROCK_PREFIX + model


def _text_of(message: BaseMessage) -> str:
    """Flatten LangChain content, which may be a string or a list of content blocks."""
    content = message.content
    if isinstance(content, str):
        return content
    parts = []
    for block in content or []:
        if isinstance(block, str):
            parts.append(block)
        elif isinstance(block, dict) and block.get("type") == "text":
            parts.append(block.get("text", ""))
    return "".join(parts)


def to_anthropic_messages(messages: Sequence[BaseMessage]) -> tuple[str, list[dict]]:
    """Convert LangChain messages into Anthropic's (system, messages) shape.

    Two conversions here are not cosmetic:

    1. **`system` is a top-level request parameter, not a message role.** Leaving a
       SystemMessage in the list either errors or silently demotes the system prompt to a
       user turn — the prompt appears to be ignored, with nothing in the logs.
    2. **A trailing assistant turn is a "prefill", which Sonnet 5 rejects with a 400.**
       Dropping it is the only way to send the request at all; it is logged rather than
       discarded quietly, because a prefill in the history usually means a caller is
       trying to steer output shape and should use a system instruction instead.
    """
    system_parts: list[str] = []
    converted: list[dict] = []

    for message in messages:
        if isinstance(message, SystemMessage):
            text = _text_of(message)
            if text:
                system_parts.append(text)
            continue
        role = "assistant" if message.type == "ai" else "user"
        converted.append({"role": role, "content": _text_of(message)})

    while converted and converted[-1]["role"] == "assistant":
        dropped = converted.pop()
        logger.warning(
            "dropped a trailing assistant message (%d chars): Sonnet 5 rejects assistant "
            "prefills with a 400. Use a system instruction to shape output instead.",
            len(dropped["content"] or ""),
        )

    return "\n\n".join(system_parts), converted


def _to_anthropic_tool(tool: Any) -> dict:
    """Normalise any LangChain-accepted tool form into Anthropic's tool shape."""
    if isinstance(tool, dict) and "input_schema" in tool:
        return tool  # already Anthropic-shaped
    fn = convert_to_openai_tool(tool)["function"]
    return {
        "name": fn["name"],
        "description": fn.get("description", "") or "",
        "input_schema": fn.get("parameters") or {"type": "object", "properties": {}},
    }


def _to_anthropic_tool_choice(choice: Any) -> dict:
    """Map LangChain's tool_choice spellings onto Anthropic's."""
    if isinstance(choice, dict):
        return choice
    if choice in ("any", "required"):
        return {"type": "any"}
    if choice == "auto":
        return {"type": "auto"}
    return {"type": "tool", "name": str(choice)}


# Models that REMOVED sampling parameters. Sending `temperature` to one of these is a
# hard 400, not a warning — and every call site in this agent passes a temperature, so
# flipping LLM_MODEL to one of them without stripping would break every request.
_NO_SAMPLING_PREFIXES = (
    "claude-sonnet-5",
    "claude-opus-5",
    "claude-opus-4-8",
    "claude-opus-4-7",
    "claude-fable-5",
    "claude-mythos-5",
)


def _accepts_sampling(model: str) -> bool:
    bare = model[len(BEDROCK_PREFIX):] if model.startswith(BEDROCK_PREFIX) else model
    return not bare.startswith(_NO_SAMPLING_PREFIXES)


@lru_cache(maxsize=8)
def _client(provider: str, aws_region: str) -> Any:
    """One client per (provider, region). Cached because construction resolves credentials.

    Both branches return an SDK whose `.messages` resource IS
    `anthropic.resources.AsyncMessages` — the class ns_probe patches. Verified for the
    first-party and Bedrock clients on anthropic 0.125 and 1.4.
    """
    if provider == "bedrock":
        return AsyncAnthropicBedrockMantle(aws_region=aws_region)
    return AsyncAnthropic()  # reads ANTHROPIC_API_KEY


class ClaudeChat(BaseChatModel):
    """Claude on Bedrock, wearing the LangChain chat-model interface.

    Only the async paths are implemented, because the agent only uses async paths. A sync
    call raises rather than silently blocking an event loop.
    """

    model: str = DEFAULT_MODEL
    max_tokens: int = DEFAULT_MAX_TOKENS
    temperature: Optional[float] = None
    # Left unset by default and only sent when a caller asks for it. On Sonnet 5 `thinking`
    # has no `budget_tokens` (it is a 400); depth is controlled with `effort` instead.
    thinking: Optional[dict] = None
    effort: Optional[str] = None
    provider: str = "anthropic"
    aws_region: str = ""

    @property
    def _llm_type(self) -> str:
        return "claude-bedrock"

    def _request(self, messages: Sequence[BaseMessage], **kwargs: Any) -> dict:
        system, converted = to_anthropic_messages(messages)
        payload: dict[str, Any] = {
            "model": _bedrock_model_id(self.model) if self.provider == "bedrock" else self.model,
            "max_tokens": self.max_tokens,
            "messages": converted,
        }
        # Absent beats empty: `system=""` is a real (blank) system prompt.
        if system:
            payload["system"] = system
        if self.temperature is not None:
            if _accepts_sampling(payload["model"]):
                payload["temperature"] = self.temperature
            else:
                # Dropped rather than passed through: this is a 400 on these models, and
                # every call site here sets a temperature.
                logger.debug(
                    "temperature dropped: %s removed sampling parameters", payload["model"]
                )
        if self.thinking:
            # budget_tokens is deprecated on 4.6 and rejected on 5; adaptive replaced it.
            payload["thinking"] = {
                k: v for k, v in self.thinking.items() if k != "budget_tokens"
            }
        if self.effort:
            payload["output_config"] = {"effort": self.effort}
        payload.update(kwargs)
        return payload

    def _region(self) -> str:
        return self.aws_region or os.getenv("AWS_REGION", "us-east-1")

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: Optional[list[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        payload = self._request(messages, **kwargs)
        if stop:
            payload["stop_sequences"] = stop

        response = await _client(self.provider, self._region()).messages.create(**payload)

        blocks = getattr(response, "content", []) or []
        text = "".join(
            block.text for block in blocks if getattr(block, "type", None) == "text"
        )
        # tool_use blocks must be translated, not ignored. The base class's structured-output
        # parser reads `.tool_calls`; leaving it empty makes with_structured_output return
        # None with no error to explain why.
        tool_calls = [
            {
                "name": block.name,
                "args": block.input or {},
                "id": getattr(block, "id", None),
                "type": "tool_call",
            }
            for block in blocks
            if getattr(block, "type", None) == "tool_use"
        ]
        usage = getattr(response, "usage", None)
        usage_metadata = None
        if usage is not None:
            inp = getattr(usage, "input_tokens", 0) or 0
            out = getattr(usage, "output_tokens", 0) or 0
            usage_metadata = {
                "input_tokens": inp,
                "output_tokens": out,
                "total_tokens": inp + out,
            }
            _record_usage(inp, out)

        message = AIMessage(
            content=text,
            tool_calls=tool_calls,
            usage_metadata=usage_metadata,
            response_metadata={
                "model": getattr(response, "model", payload["model"]),
                "stop_reason": getattr(response, "stop_reason", None),
            },
        )
        return ChatResult(generations=[ChatGeneration(message=message)])

    async def _astream(
        self,
        messages: list[BaseMessage],
        stop: Optional[list[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        payload = self._request(messages, **kwargs)
        payload["stream"] = True
        if stop:
            payload["stop_sequences"] = stop

        stream = await _client(self.provider, self._region()).messages.create(**payload)
        # Usage does not ride the text deltas: input tokens arrive on `message_start` and
        # the running output total on `message_delta`. Reading only `content_block_delta`
        # — as this loop used to — dropped every streamed call from the turn totals, which
        # in a cost comparison is the one number that must never be quietly wrong.
        stream_in = stream_out = 0
        try:
            async for event in stream:
                etype = getattr(event, "type", None)
                if etype == "message_start":
                    u = getattr(getattr(event, "message", None), "usage", None)
                    if u is not None:
                        stream_in = getattr(u, "input_tokens", 0) or 0
                        stream_out = max(stream_out, getattr(u, "output_tokens", 0) or 0)
                    continue
                if etype == "message_delta":
                    u = getattr(event, "usage", None)
                    if u is not None:
                        # message_delta carries a cumulative count, not an increment.
                        stream_out = max(stream_out, getattr(u, "output_tokens", 0) or 0)
                    continue
                if etype != "content_block_delta":
                    continue
                piece = getattr(getattr(event, "delta", None), "text", None)
                if not piece:
                    continue
                chunk = ChatGenerationChunk(message=AIMessageChunk(content=piece))
                if run_manager:
                    await run_manager.on_llm_new_token(piece, chunk=chunk)
                yield chunk
        finally:
            # `finally` because a consumer that stops early still spent these tokens.
            if stream_in or stream_out:
                _record_usage(stream_in, stream_out)

    def bind_tools(
        self,
        tools: Sequence[Any],
        *,
        tool_choice: Optional[Any] = None,
        **kwargs: Any,
    ) -> Any:
        """Attach tools, converting them to Anthropic's shape.

        Needed for more than tool use: LangChain's default `with_structured_output`
        implementation routes through `bind_tools`, so without this the two call sites
        that ask for a structured answer (planner, entity extraction) fail outright with
        NotImplementedError.

        Anthropic wants ``{name, description, input_schema}``. The OpenAI
        ``{function: {name, parameters}}`` envelope is rejected, so the OpenAI conversion
        is used only to normalise the many accepted input forms (Pydantic model, dict,
        callable) and then unwrapped.
        """
        formatted = [_to_anthropic_tool(tool) for tool in tools]
        if tool_choice is not None:
            kwargs["tool_choice"] = _to_anthropic_tool_choice(tool_choice)
        return super().bind(tools=formatted, **kwargs)

    def _generate(self, *args: Any, **kwargs: Any) -> ChatResult:
        raise NotImplementedError(
            "ClaudeChat is async-only. The agent uses ainvoke/astream everywhere "
            "(measured: 21 ainvoke, 2 astream, 0 sync invoke); a sync path would block the "
            "event loop rather than help."
        )


def make_llm(
    role: str = "answer",
    *,
    temperature: Optional[float] = None,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    thinking: Optional[dict] = None,
    effort: Optional[str] = None,
):
    """Build the chat model for a given role.

    PROVIDER IS AN ENV FLIP, NOT A CODE CHANGE. `LLM_PROVIDER` selects between:

      openai    (default for now) — the existing, working path. Kept while the Anthropic
                account has no credits, so the agent can be built and run end to end.
      anthropic — first-party Claude API, reads ANTHROPIC_API_KEY.
      bedrock   — Claude via AWS Bedrock, reads the usual AWS credential chain.

    All three are instrumented by ns_probe: the OpenAI path through its openai patch, and
    both Claude paths because their `.messages` resource is the same
    `anthropic.resources.AsyncMessages` class it patches. Whichever is selected, tokens and
    cost still reach the dashboard — which is the whole reason this indirection exists.

    `role` selects a model per call site so a cheap step (intent classification) and an
    expensive one (writing the answer) need not share a model.
    """
    provider = (os.getenv("LLM_PROVIDER") or "openai").strip().lower()

    if provider == "openai":
        # Imported here, not at module scope, so an Anthropic-only deployment does not
        # need langchain-openai installed.
        from langchain_openai import ChatOpenAI

        from app.config import config

        model = {
            "fast": config.openai_model_fast,
            "answer": config.openai_model_answer,
        }.get(role, config.openai_model)
        return ChatOpenAI(
            api_key=config.openai_api_key,
            base_url=config.openai_base_url,
            model=model,
            temperature=temperature if temperature is not None else 0.0,
            max_tokens=max_tokens,
        )

    env_key = f"LLM_MODEL_{role.upper()}"
    model = os.getenv(env_key) or os.getenv("LLM_MODEL") or DEFAULT_MODEL
    return ClaudeChat(
        model=model,
        provider=provider,
        max_tokens=max_tokens,
        temperature=temperature,
        thinking=thinking,
        effort=effort,
    )
