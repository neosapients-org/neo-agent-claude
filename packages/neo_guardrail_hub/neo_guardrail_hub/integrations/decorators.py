"""Decorator for adding guardrails to functions.

This module provides the @guarded decorator for easily
adding guardrail protection to any async function.
"""

import functools
from pathlib import Path
from typing import Any, Callable, Optional, TypeVar, Union

from ..core.exceptions import GuardrailError
from ..core.models import ActionOnFail, AggregatedResult
from ..core.orchestrator import NeoGuardrailOrchestrator

F = TypeVar("F", bound=Callable[..., Any])


def guarded(
    agent_id: str = "default",
    config_path: Union[str, Path] = "./configs",
    guard_input: bool = True,
    guard_output: bool = True,
    guard_context: bool = False,
    on_fail: str = "block",
    timeout_ms: Optional[int] = None,
    input_arg: str = "user_input",
    raise_on_fail: bool = True,
    models_dir: Optional[str] = None,
) -> Callable[[F], F]:
    """Decorator to add guardrails to an async function.

    Automatically validates input and/or output using the
    configured guardrails.

    Args:
        agent_id: Agent identifier for config lookup
        config_path: Path to configuration directory
        guard_input: Whether to validate input
        guard_output: Whether to validate output
        guard_context: Whether to validate context
        on_fail: Action on failure: "block", "warn", "sanitize"
        timeout_ms: Optional timeout override
        input_arg: Name of the input argument to validate
        raise_on_fail: Whether to raise exception on failure
        models_dir: Optional path to directory containing local models for LLM Guard guardrails

    Returns:
        Decorated function

    Example:
        @guarded(agent_id="my_agent", guard_input=True, guard_output=True, models_dir="./models/llm_guard")
        async def my_function(user_input: str) -> str:
            return await llm.generate(user_input)
    """
    _orchestrator: Optional[NeoGuardrailOrchestrator] = None
    _initialized: bool = False

    async def get_orchestrator() -> NeoGuardrailOrchestrator:
        nonlocal _orchestrator, _initialized
        if _orchestrator is None:
            _orchestrator = NeoGuardrailOrchestrator(
                config_path=config_path,
                models_dir=models_dir
            )
        if not _initialized:
            await _orchestrator.initialize(agent_id=agent_id)
            _initialized = True
        return _orchestrator

    def decorator(func: F) -> F:
        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            orchestrator = await get_orchestrator()

            # Get input text from arguments
            input_text = None
            if guard_input:
                if input_arg in kwargs:
                    input_text = kwargs[input_arg]
                elif args:
                    input_text = args[0]

                if input_text is not None:
                    input_result = await orchestrator.guard_input(
                        text=str(input_text),
                        agent_id=agent_id,
                    )

                    if not input_result.passed:
                        if raise_on_fail:
                            raise GuardrailError(
                                message=_get_failure_message(input_result),
                                guardrail_name=_get_failed_guardrail(input_result),
                            )
                        return _create_blocked_response(input_result)

            # Call the original function
            result = await func(*args, **kwargs)

            # Validate output if enabled
            if guard_output and result is not None:
                output_text = str(result)
                output_result = await orchestrator.guard_output(
                    text=output_text,
                    agent_id=agent_id,
                )

                if not output_result.passed:
                    if raise_on_fail:
                        raise GuardrailError(
                            message=_get_failure_message(output_result),
                            guardrail_name=_get_failed_guardrail(output_result),
                        )
                    return _create_blocked_response(output_result)

                # Return sanitized text if available
                if output_result.final_text and output_result.final_text != output_text:
                    return output_result.final_text

            return result

        return wrapper  # type: ignore

    return decorator


def _get_failure_message(result: AggregatedResult) -> str:
    """Extract failure message from aggregated result."""
    for r in result.results:
        if not r.passed and r.message:
            return r.message
    return "Guardrail check failed"


def _get_failed_guardrail(result: AggregatedResult) -> str:
    """Get the name of the first failed guardrail."""
    for r in result.results:
        if not r.passed:
            return r.guardrail_name
    return "unknown"


def _create_blocked_response(result: AggregatedResult) -> dict:
    """Create a blocked response dictionary."""
    return {
        "error": "blocked",
        "message": _get_failure_message(result),
        "guardrail": _get_failed_guardrail(result),
        "risk_score": result.max_risk_score,
    }


# Sync version of the decorator
def guarded_sync(
    agent_id: str = "default",
    config_path: Union[str, Path] = "./configs",
    guard_input: bool = True,
    guard_output: bool = True,
    on_fail: str = "block",
    input_arg: str = "user_input",
    raise_on_fail: bool = True,
) -> Callable[[F], F]:
    """Synchronous version of the @guarded decorator.

    For use with non-async functions.

    Example:
        @guarded_sync(agent_id="my_agent")
        def my_function(user_input: str) -> str:
            return llm.generate_sync(user_input)
    """
    _orchestrator: Optional[NeoGuardrailOrchestrator] = None

    def get_orchestrator() -> NeoGuardrailOrchestrator:
        nonlocal _orchestrator
        if _orchestrator is None:
            _orchestrator = NeoGuardrailOrchestrator(config_path=config_path)
        return _orchestrator

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            orchestrator = get_orchestrator()

            # Get input text
            input_text = None
            if guard_input:
                if input_arg in kwargs:
                    input_text = kwargs[input_arg]
                elif args:
                    input_text = args[0]

                if input_text is not None:
                    input_result = orchestrator.guard_input_sync(
                        text=str(input_text),
                        agent_id=agent_id,
                    )

                    if not input_result.passed:
                        if raise_on_fail:
                            raise GuardrailError(
                                message=_get_failure_message(input_result),
                                guardrail_name=_get_failed_guardrail(input_result),
                            )
                        return _create_blocked_response(input_result)

            # Call the original function
            result = func(*args, **kwargs)

            # Validate output
            if guard_output and result is not None:
                output_result = orchestrator.guard_output_sync(
                    text=str(result),
                    agent_id=agent_id,
                )

                if not output_result.passed:
                    if raise_on_fail:
                        raise GuardrailError(
                            message=_get_failure_message(output_result),
                            guardrail_name=_get_failed_guardrail(output_result),
                        )
                    return _create_blocked_response(output_result)

                if output_result.final_text and output_result.final_text != str(result):
                    return output_result.final_text

            return result

        return wrapper  # type: ignore

    return decorator
