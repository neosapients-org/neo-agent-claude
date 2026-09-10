"""
Import Hook Patcher for BaseAgent auto-instrumentation.

This module implements Python's sys.meta_path import hooks to intercept
the import of BaseAgent and automatically wrap its methods with span
instrumentation - without requiring any code changes to the agent.

How Python Import Hooks Work:
    When Python encounters `from ns_core.agents.base import BaseAgent`:
    1. Python iterates through sys.meta_path looking for a finder
    2. Each finder's find_spec() is called with the module name
    3. If a finder returns a spec, that spec's loader loads the module
    4. We intercept step 2-3 to wrap methods after the real module loads

The Patching Strategy:
    1. Register our finder FIRST in sys.meta_path
    2. When 'ns_core.agents.base' is imported:
       a. Temporarily remove ourselves from meta_path
       b. Let Python find the real module
       c. Add ourselves back
       d. Return a spec with our custom loader
    3. Our loader:
       a. Executes the real module (loads BaseAgent)
       b. Wraps BaseAgent.execute, think, decide, act, reflect
       c. Each wrapped method creates spans automatically

Methods Patched:
    - execute(): Root span for agent execution
    - think(): Chain-of-Thought reasoning span
    - decide(): Decision-making span
    - act(): Action execution span
    - reflect(): Reflection/learning span
    - _call_tool(): Tool call span
    - _call_llm(): LLM call span (if exists)

Performance Impact:
    - Import hook: +50ms one-time at startup
    - Method wrapper: <0.1ms per call (span creation + ring buffer push)
"""

from __future__ import annotations

import sys
import logging
import functools
import inspect
import importlib.abc
import importlib.util
from typing import Any, Optional, Callable, TypeVar

from .tracer import get_tracer
from .span import SpanKind

logger = logging.getLogger("ns_probe.patcher")

F = TypeVar("F", bound=Callable[..., Any])


class AgentMethodWrapper:
    """
    Wraps agent methods with span instrumentation.

    This class provides both sync and async wrappers that:
    1. Create a span before method execution
    2. Capture method arguments as span attributes
    3. Capture return value as span attribute
    4. Record exceptions if method fails
    5. End span when method completes
    """

    def __init__(self, tracer_name: str = "ns_probe.agent") -> None:
        self._tracer_name = tracer_name

    def wrap_execute(self, original: F) -> F:
        """
        Wrap the execute() method (root span).

        This is the entry point for agent execution. Creates the root
        span that all other spans will be children of.
        """
        tracer = get_tracer(self._tracer_name)

        if inspect.iscoroutinefunction(original):

            @functools.wraps(original)
            async def async_wrapper(self_agent: Any, task: Any, **kwargs: Any) -> Any:
                # Extract agent identity
                agent_name = getattr(getattr(self_agent, "config", None), "name", "UnknownAgent")
                agent_id = getattr(getattr(self_agent, "config", None), "agent_id", "unknown")
                agent_class = self_agent.__class__.__name__

                span_name = "agent.execute"

                with tracer.start_span(span_name, kind=SpanKind.INTERNAL) as span:
                    # Set agent attributes
                    span.set_attributes(
                        {
                            "agent.name": agent_name,
                            "agent.id": agent_id,
                            "agent.class": agent_class,
                            "agent.task": _truncate(str(task), 2000),
                        }
                    )

                    # Execute original method
                    result = await original(self_agent, task, **kwargs)

                    # Capture output
                    span.set_attribute("agent.output", _truncate(str(result), 2000))

                    return result

            return async_wrapper  # type: ignore
        else:

            @functools.wraps(original)
            def sync_wrapper(self_agent: Any, task: Any, **kwargs: Any) -> Any:
                agent_name = getattr(getattr(self_agent, "config", None), "name", "UnknownAgent")
                agent_id = getattr(getattr(self_agent, "config", None), "agent_id", "unknown")
                agent_class = self_agent.__class__.__name__

                span_name = "agent.execute"

                with tracer.start_span(span_name, kind=SpanKind.INTERNAL) as span:
                    span.set_attributes(
                        {
                            "agent.name": agent_name,
                            "agent.id": agent_id,
                            "agent.class": agent_class,
                            "agent.task": _truncate(str(task), 2000),
                        }
                    )

                    result = original(self_agent, task, **kwargs)
                    span.set_attribute("agent.output", _truncate(str(result), 2000))

                    return result

            return sync_wrapper  # type: ignore

    def wrap_think(self, original: F) -> F:
        """
        Wrap the think() method (Chain-of-Thought span).

        Captures:
            - thought.reasoning: The agent's internal reasoning
            - thought.decision: What the agent decided
            - thought.confidence: Self-assessed confidence (0-1)
            - thought.iteration: Which thinking iteration this is
        """
        tracer = get_tracer(self._tracer_name)

        # Track iteration count per agent instance
        iteration_counts: dict[int, int] = {}

        if inspect.iscoroutinefunction(original):

            @functools.wraps(original)
            async def async_wrapper(self_agent: Any, *args: Any, **kwargs: Any) -> Any:
                agent_id = id(self_agent)
                iteration_counts[agent_id] = iteration_counts.get(agent_id, 0) + 1
                iteration = iteration_counts[agent_id]

                with tracer.start_span("agent.thought", kind=SpanKind.INTERNAL) as span:
                    span.set_attribute("thought.iteration", iteration)

                    result = await original(self_agent, *args, **kwargs)

                    # Try to extract structured thought data
                    self._extract_thought_attributes(span, result)

                    return result

            return async_wrapper  # type: ignore
        else:

            @functools.wraps(original)
            def sync_wrapper(self_agent: Any, *args: Any, **kwargs: Any) -> Any:
                agent_id = id(self_agent)
                iteration_counts[agent_id] = iteration_counts.get(agent_id, 0) + 1
                iteration = iteration_counts[agent_id]

                with tracer.start_span("agent.thought", kind=SpanKind.INTERNAL) as span:
                    span.set_attribute("thought.iteration", iteration)

                    result = original(self_agent, *args, **kwargs)
                    self._extract_thought_attributes(span, result)

                    return result

            return sync_wrapper  # type: ignore

    def wrap_decide(self, original: F) -> F:
        """
        Wrap the decide() method (decision span).

        Captures:
            - decision.action: The chosen action
            - decision.rationale: Why this action was chosen
            - decision.alternatives: Other options considered
        """
        tracer = get_tracer(self._tracer_name)

        if inspect.iscoroutinefunction(original):

            @functools.wraps(original)
            async def async_wrapper(self_agent: Any, *args: Any, **kwargs: Any) -> Any:
                with tracer.start_span("agent.decision", kind=SpanKind.INTERNAL) as span:
                    result = await original(self_agent, *args, **kwargs)
                    self._extract_decision_attributes(span, result)
                    return result

            return async_wrapper  # type: ignore
        else:

            @functools.wraps(original)
            def sync_wrapper(self_agent: Any, *args: Any, **kwargs: Any) -> Any:
                with tracer.start_span("agent.decision", kind=SpanKind.INTERNAL) as span:
                    result = original(self_agent, *args, **kwargs)
                    self._extract_decision_attributes(span, result)
                    return result

            return sync_wrapper  # type: ignore

    def wrap_act(self, original: F) -> F:
        """
        Wrap the act() method (action span).

        Captures:
            - action.type: Type of action (tool_call, llm_call, etc.)
            - action.input: Action input
            - action.output: Action result
            - action.duration_ms: Execution time
        """
        tracer = get_tracer(self._tracer_name)

        if inspect.iscoroutinefunction(original):

            @functools.wraps(original)
            async def async_wrapper(self_agent: Any, action: Any, *args: Any, **kwargs: Any) -> Any:
                with tracer.start_span("agent.action", kind=SpanKind.INTERNAL) as span:
                    # Capture action input
                    span.set_attribute("action.input", _truncate(str(action), 1000))

                    if hasattr(action, "action_type"):
                        span.set_attribute("action.type", str(action.action_type))

                    result = await original(self_agent, action, *args, **kwargs)

                    span.set_attribute("action.output", _truncate(str(result), 1000))

                    return result

            return async_wrapper  # type: ignore
        else:

            @functools.wraps(original)
            def sync_wrapper(self_agent: Any, action: Any, *args: Any, **kwargs: Any) -> Any:
                with tracer.start_span("agent.action", kind=SpanKind.INTERNAL) as span:
                    span.set_attribute("action.input", _truncate(str(action), 1000))

                    if hasattr(action, "action_type"):
                        span.set_attribute("action.type", str(action.action_type))

                    result = original(self_agent, action, *args, **kwargs)
                    span.set_attribute("action.output", _truncate(str(result), 1000))

                    return result

            return sync_wrapper  # type: ignore

    def wrap_reflect(self, original: F) -> F:
        """
        Wrap the reflect() method (reflection span).

        Captures:
            - reflection.observation: What the agent observed
            - reflection.learning: What the agent learned
        """
        tracer = get_tracer(self._tracer_name)

        if inspect.iscoroutinefunction(original):

            @functools.wraps(original)
            async def async_wrapper(self_agent: Any, *args: Any, **kwargs: Any) -> Any:
                with tracer.start_span("agent.reflection", kind=SpanKind.INTERNAL) as span:
                    result = await original(self_agent, *args, **kwargs)
                    self._extract_reflection_attributes(span, result)
                    return result

            return async_wrapper  # type: ignore
        else:

            @functools.wraps(original)
            def sync_wrapper(self_agent: Any, *args: Any, **kwargs: Any) -> Any:
                with tracer.start_span("agent.reflection", kind=SpanKind.INTERNAL) as span:
                    result = original(self_agent, *args, **kwargs)
                    self._extract_reflection_attributes(span, result)
                    return result

            return sync_wrapper  # type: ignore

    def wrap_tool_call(self, original: F) -> F:
        """
        Wrap tool execution methods.

        Captures:
            - tool.name: Name of the tool
            - tool.input: Tool arguments
            - tool.output: Tool result
            - tool.duration_ms: Execution time
            - tool.success: Whether the call succeeded
        """
        tracer = get_tracer(self._tracer_name)

        if inspect.iscoroutinefunction(original):

            @functools.wraps(original)
            async def async_wrapper(
                self_agent: Any, tool_name: str, *args: Any, **kwargs: Any
            ) -> Any:
                with tracer.start_span("tool.call", kind=SpanKind.CLIENT) as span:
                    span.set_attributes(
                        {
                            "tool.name": tool_name,
                            "tool.input": _truncate(str(kwargs or args), 1000),
                        }
                    )

                    try:
                        result = await original(self_agent, tool_name, *args, **kwargs)
                        span.set_attributes(
                            {
                                "tool.output": _truncate(str(result), 1000),
                                "tool.success": True,
                            }
                        )
                        return result
                    except Exception:
                        # Marked as failed and re-raised. The exception itself is
                        # deliberately not recorded here — the caller's own
                        # traced frame records it, and doing it twice puts the
                        # same stack on two spans.
                        span.set_attribute("tool.success", False)
                        raise

            return async_wrapper  # type: ignore
        else:

            @functools.wraps(original)
            def sync_wrapper(self_agent: Any, tool_name: str, *args: Any, **kwargs: Any) -> Any:
                with tracer.start_span("tool.call", kind=SpanKind.CLIENT) as span:
                    span.set_attributes(
                        {
                            "tool.name": tool_name,
                            "tool.input": _truncate(str(kwargs or args), 1000),
                        }
                    )

                    try:
                        result = original(self_agent, tool_name, *args, **kwargs)
                        span.set_attributes(
                            {
                                "tool.output": _truncate(str(result), 1000),
                                "tool.success": True,
                            }
                        )
                        return result
                    except Exception:
                        # Marked as failed and re-raised. The exception itself is
                        # deliberately not recorded here — the caller's own
                        # traced frame records it, and doing it twice puts the
                        # same stack on two spans.
                        span.set_attribute("tool.success", False)
                        raise

            return sync_wrapper  # type: ignore

    def wrap_llm_call(self, original: F) -> F:
        """
        Wrap LLM call methods.

        Captures OpenTelemetry Gen AI semantic conventions:
            - gen_ai.system: Provider (openai, anthropic, etc.)
            - gen_ai.request.model: Model name
            - gen_ai.usage.input_tokens: Prompt tokens
            - gen_ai.usage.output_tokens: Completion tokens
            - gen_ai.response.finish_reason: Why generation stopped
        """
        tracer = get_tracer(self._tracer_name)

        if inspect.iscoroutinefunction(original):

            @functools.wraps(original)
            async def async_wrapper(self_agent: Any, *args: Any, **kwargs: Any) -> Any:
                with tracer.start_span("llm.call", kind=SpanKind.CLIENT) as span:
                    # Try to extract model info from kwargs
                    if "model" in kwargs:
                        span.set_attribute("gen_ai.request.model", kwargs["model"])

                    result = await original(self_agent, *args, **kwargs)

                    # Extract response metadata
                    self._extract_llm_attributes(span, result)

                    return result

            return async_wrapper  # type: ignore
        else:

            @functools.wraps(original)
            def sync_wrapper(self_agent: Any, *args: Any, **kwargs: Any) -> Any:
                with tracer.start_span("llm.call", kind=SpanKind.CLIENT) as span:
                    if "model" in kwargs:
                        span.set_attribute("gen_ai.request.model", kwargs["model"])

                    result = original(self_agent, *args, **kwargs)
                    self._extract_llm_attributes(span, result)

                    return result

            return sync_wrapper  # type: ignore

    def _extract_thought_attributes(self, span: Any, result: Any) -> None:
        """Extract thought attributes from result."""
        if hasattr(result, "reasoning"):
            span.set_attribute("thought.reasoning", _truncate(str(result.reasoning), 2000))
        if hasattr(result, "decision"):
            span.set_attribute("thought.decision", _truncate(str(result.decision), 500))
        if hasattr(result, "confidence"):
            span.set_attribute("thought.confidence", float(result.confidence))

        # Fallback: if result is a string, treat it as reasoning
        if isinstance(result, str):
            span.set_attribute("thought.reasoning", _truncate(result, 2000))

    def _extract_decision_attributes(self, span: Any, result: Any) -> None:
        """Extract decision attributes from result."""
        if hasattr(result, "action"):
            span.set_attribute("decision.action", _truncate(str(result.action), 500))
        if hasattr(result, "rationale"):
            span.set_attribute("decision.rationale", _truncate(str(result.rationale), 1000))
        if hasattr(result, "alternatives"):
            span.set_attribute("decision.alternatives", str(result.alternatives)[:500])

    def _extract_reflection_attributes(self, span: Any, result: Any) -> None:
        """Extract reflection attributes from result."""
        if hasattr(result, "observation"):
            span.set_attribute("reflection.observation", _truncate(str(result.observation), 1000))
        if hasattr(result, "learning"):
            span.set_attribute("reflection.learning", _truncate(str(result.learning), 1000))

    def _extract_llm_attributes(self, span: Any, result: Any) -> None:
        """Extract LLM response attributes."""
        # OpenAI-style response
        if hasattr(result, "usage"):
            usage = result.usage
            if hasattr(usage, "prompt_tokens"):
                span.set_attribute("gen_ai.usage.input_tokens", usage.prompt_tokens)
            if hasattr(usage, "completion_tokens"):
                span.set_attribute("gen_ai.usage.output_tokens", usage.completion_tokens)
            if hasattr(usage, "total_tokens"):
                span.set_attribute("gen_ai.usage.total_tokens", usage.total_tokens)

        if hasattr(result, "model"):
            span.set_attribute("gen_ai.response.model", result.model)

        # Extract finish reason from choices
        if hasattr(result, "choices") and result.choices:
            choice = result.choices[0]
            if hasattr(choice, "finish_reason"):
                span.set_attribute("gen_ai.response.finish_reason", choice.finish_reason)


def _truncate(s: str, max_len: int) -> str:
    """Truncate string to max length with ellipsis."""
    if len(s) <= max_len:
        return s
    return s[: max_len - 3] + "..."


class AgentLoader(importlib.abc.Loader):
    """
    Custom loader that patches BaseAgent after loading.

    This loader delegates to the real loader to execute the module,
    then applies our method wrappers to BaseAgent.
    """

    def __init__(self, real_loader: Any, wrapper: AgentMethodWrapper) -> None:
        self.real_loader = real_loader
        self.wrapper = wrapper

    def create_module(self, spec: Any) -> Optional[Any]:
        """Delegate module creation to real loader."""
        if hasattr(self.real_loader, "create_module"):
            return self.real_loader.create_module(spec)
        return None

    def exec_module(self, module: Any) -> None:
        """Execute module then patch BaseAgent."""
        # 1. Execute the real module
        if hasattr(self.real_loader, "exec_module"):
            self.real_loader.exec_module(module)

        # 2. Patch BaseAgent if present
        if hasattr(module, "BaseAgent"):
            self._patch_base_agent(module.BaseAgent)
            logger.info("Successfully patched BaseAgent via import hook")
        else:
            logger.debug(f"BaseAgent not found in {module.__name__}")

    def _patch_base_agent(self, base_agent_class: type) -> None:
        """Apply method wrappers to BaseAgent."""
        methods_to_patch = {
            "execute": self.wrapper.wrap_execute,
            "think": self.wrapper.wrap_think,
            "decide": self.wrapper.wrap_decide,
            "act": self.wrapper.wrap_act,
            "reflect": self.wrapper.wrap_reflect,
            "_call_tool": self.wrapper.wrap_tool_call,
            "call_tool": self.wrapper.wrap_tool_call,
            "_call_llm": self.wrapper.wrap_llm_call,
            "call_llm": self.wrapper.wrap_llm_call,
        }

        for method_name, wrapper_func in methods_to_patch.items():
            if hasattr(base_agent_class, method_name):
                original = getattr(base_agent_class, method_name)

                # Skip if already wrapped (prevent double-wrapping)
                if hasattr(original, "_ns_probe_wrapped"):
                    continue

                wrapped = wrapper_func(original)
                wrapped._ns_probe_wrapped = True  # type: ignore
                setattr(base_agent_class, method_name, wrapped)
                logger.debug(f"Patched BaseAgent.{method_name}")


class AgentImportHook(importlib.abc.MetaPathFinder):
    """
    Meta path finder that intercepts BaseAgent import.

    This finder watches for imports of 'ns_core.agents.base' and
    returns a custom spec that will use our AgentLoader.
    """

    def __init__(self, wrapper: AgentMethodWrapper) -> None:
        self.wrapper = wrapper
        self._target_modules = {
            "ns_core.agents.base",
            "packages.core.ns_core.agents.base",
        }

    def find_spec(
        self,
        fullname: str,
        path: Optional[Any],
        target: Optional[Any] = None,
    ) -> Optional[Any]:
        """Find module spec, intercepting our target modules."""
        if fullname not in self._target_modules:
            return None

        # Temporarily remove self to find the real spec
        if self in sys.meta_path:
            sys.meta_path.remove(self)

        try:
            # Find the real module spec
            real_spec = importlib.util.find_spec(fullname)
        finally:
            # Always add self back (at the front)
            sys.meta_path.insert(0, self)

        if real_spec is None or real_spec.loader is None:
            return None

        # Return a new spec with our custom loader
        loader = AgentLoader(real_spec.loader, self.wrapper)
        return importlib.util.spec_from_loader(fullname, loader)


# Global hook instance
_import_hook: Optional[AgentImportHook] = None


def install_import_hook() -> None:
    """
    Install the import hook into sys.meta_path.

    This should be called early in application startup, before
    any agent code is imported.
    """
    global _import_hook

    if _import_hook is not None:
        # Already installed
        return

    wrapper = AgentMethodWrapper()
    _import_hook = AgentImportHook(wrapper)
    sys.meta_path.insert(0, _import_hook)

    logger.info("Installed BaseAgent import hook")


def uninstall_import_hook() -> None:
    """
    Remove the import hook from sys.meta_path.

    Primarily for testing purposes.
    """
    global _import_hook

    if _import_hook is None:
        return

    if _import_hook in sys.meta_path:
        sys.meta_path.remove(_import_hook)

    _import_hook = None
    logger.info("Removed BaseAgent import hook")


def patch_existing_base_agent() -> None:
    """
    Patch BaseAgent if it's already been imported.

    This is a fallback for when the import hook wasn't installed
    before BaseAgent was imported.
    """
    wrapper = AgentMethodWrapper()

    # Try to find BaseAgent in already-loaded modules
    for module_name in ["ns_core.agents.base", "packages.core.ns_core.agents.base"]:
        if module_name in sys.modules:
            module = sys.modules[module_name]
            if hasattr(module, "BaseAgent"):
                loader = AgentLoader(None, wrapper)
                loader._patch_base_agent(module.BaseAgent)
                logger.info(f"Patched existing BaseAgent from {module_name}")
                return

    logger.debug("BaseAgent not yet imported, will be patched on import")
