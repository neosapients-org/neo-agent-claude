"""Parallel and sequential executor for guardrails.

This module handles the execution of multiple guardrails,
supporting both parallel (concurrent) and sequential execution modes.
"""

import asyncio
import time
from typing import List, Optional

from .exceptions import ExecutionTimeoutError
from .interfaces import BaseGuardrail
from .models import (
    ActionOnFail,
    AggregatedResult,
    GuardrailContext,
    GuardrailLayer,
    GuardrailResult,
)


class ParallelExecutor:
    """Execute guardrails in parallel or sequential mode.

    Provides methods for running multiple guardrails concurrently
    with timeout support, or sequentially with early stopping.

    Example:
        executor = ParallelExecutor()
        results = await executor.execute_parallel(
            guardrails=[guard1, guard2],
            text="user input",
            timeout_ms=5000
        )
    """

    def __init__(self, max_concurrent: int = 10) -> None:
        """Initialize the executor.

        Args:
            max_concurrent: Maximum number of concurrent guardrail executions
        """
        self.max_concurrent = max_concurrent
        self._semaphore: Optional[asyncio.Semaphore] = None

    async def execute_parallel(
        self,
        guardrails: List[BaseGuardrail],
        text: str,
        context: Optional[GuardrailContext] = None,
        timeout_ms: int = 5000,
    ) -> List[GuardrailResult]:
        """Run guardrails in parallel and return when all complete.

        Args:
            guardrails: List of guardrails to execute
            text: Text to check
            context: Optional context for stateful checks
            timeout_ms: Timeout in milliseconds for all checks

        Returns:
            List of GuardrailResult objects in the same order as input

        Raises:
            ExecutionTimeoutError: If execution exceeds timeout
        """
        if not guardrails:
            return []

        # Create semaphore if not exists
        if self._semaphore is None:
            self._semaphore = asyncio.Semaphore(self.max_concurrent)

        async def run_with_semaphore(guardrail: BaseGuardrail) -> GuardrailResult:
            async with self._semaphore:  # type: ignore
                return await self._execute_single(guardrail, text, context)

        try:
            timeout_seconds = timeout_ms / 1000.0
            tasks = [run_with_semaphore(g) for g in guardrails]

            results = await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=True),
                timeout=timeout_seconds,
            )

            # Process results, converting exceptions to failed results
            processed_results: List[GuardrailResult] = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    processed_results.append(
                        self._create_error_result(guardrails[i], result)
                    )
                else:
                    processed_results.append(result)

            return processed_results

        except asyncio.TimeoutError:
            guardrail_names = [g.name for g in guardrails]
            raise ExecutionTimeoutError(
                message="Parallel execution timed out",
                timeout_ms=timeout_ms,
                guardrail_names=guardrail_names,
            )

    async def execute_sequential(
        self,
        guardrails: List[BaseGuardrail],
        text: str,
        context: Optional[GuardrailContext] = None,
        stop_on_fail: bool = True,
        timeout_ms: Optional[int] = None,
    ) -> List[GuardrailResult]:
        """Run guardrails sequentially, optionally stopping on first failure.

        Args:
            guardrails: List of guardrails to execute in order
            text: Text to check
            context: Optional context for stateful checks
            stop_on_fail: Whether to stop on first failed check
            timeout_ms: Optional timeout for all checks combined

        Returns:
            List of GuardrailResult objects for executed guardrails

        Raises:
            ExecutionTimeoutError: If execution exceeds timeout
        """
        if not guardrails:
            return []

        results: List[GuardrailResult] = []
        start_time = time.monotonic()

        for guardrail in guardrails:
            # Check timeout if specified
            if timeout_ms is not None:
                elapsed_ms = (time.monotonic() - start_time) * 1000
                if elapsed_ms >= timeout_ms:
                    raise ExecutionTimeoutError(
                        message="Sequential execution timed out",
                        timeout_ms=timeout_ms,
                        guardrail_names=[g.name for g in guardrails],
                    )

            try:
                result = await self._execute_single(guardrail, text, context)
                results.append(result)

                # Stop if check failed and stop_on_fail is True
                if stop_on_fail and not result.passed:
                    break

            except Exception as e:
                results.append(self._create_error_result(guardrail, e))
                if stop_on_fail:
                    break

        return results

    async def _execute_single(
        self,
        guardrail: BaseGuardrail,
        text: str,
        context: Optional[GuardrailContext] = None,
    ) -> GuardrailResult:
        """Execute a single guardrail with timing.

        Args:
            guardrail: The guardrail to execute
            text: Text to check
            context: Optional context

        Returns:
            GuardrailResult with latency populated
        """
        start_time = time.monotonic()

        try:
            result = await guardrail.check(text, context)
            latency_ms = (time.monotonic() - start_time) * 1000
            result.latency_ms = latency_ms
            return result

        except Exception as e:
            latency_ms = (time.monotonic() - start_time) * 1000
            return GuardrailResult(
                passed=False,
                guardrail_name=guardrail.name,
                layer=guardrail.layer,
                risk_score=1.0,
                message=f"Guardrail error: {str(e)}",
                latency_ms=latency_ms,
                metadata={"error": str(e), "error_type": type(e).__name__},
            )

    def _create_error_result(
        self, guardrail: BaseGuardrail, error: Exception
    ) -> GuardrailResult:
        """Create a GuardrailResult for an error case.

        Args:
            guardrail: The guardrail that failed
            error: The exception that was raised

        Returns:
            GuardrailResult indicating failure
        """
        return GuardrailResult(
            passed=False,
            guardrail_name=guardrail.name,
            layer=guardrail.layer,
            risk_score=1.0,
            message=f"Execution error: {str(error)}",
            metadata={"error": str(error), "error_type": type(error).__name__},
        )


class ResultAggregator:
    """Aggregate multiple guardrail results into a single result.

    Combines results from parallel or sequential execution
    into an AggregatedResult with overall pass/fail status.
    """

    def aggregate(
        self,
        results: List[GuardrailResult],
        layer: GuardrailLayer,
        original_text: str,
        default_action: ActionOnFail = ActionOnFail.BLOCK,
    ) -> AggregatedResult:
        """Aggregate multiple results into a single AggregatedResult.

        Args:
            results: List of individual guardrail results
            layer: The layer these results are from
            original_text: The original text that was checked
            default_action: Default action when a check fails

        Returns:
            AggregatedResult combining all individual results
        """
        if not results:
            return AggregatedResult(
                passed=True,
                layer=layer,
                results=[],
                action_taken=ActionOnFail.BLOCK,
                final_text=original_text,
                total_latency_ms=0.0,
            )

        # Calculate overall pass/fail
        # Failed if any guardrail with BLOCK action failed
        passed = True
        action_taken = ActionOnFail.WARN  # Default if all pass
        final_text = original_text

        for result in results:
            if not result.passed:
                # Determine action from metadata or use default
                action = result.metadata.get("on_fail", default_action)
                if isinstance(action, str):
                    action = ActionOnFail(action)

                if action == ActionOnFail.BLOCK:
                    passed = False
                    action_taken = ActionOnFail.BLOCK
                    break
                elif action == ActionOnFail.SANITIZE:
                    if result.sanitized_text:
                        final_text = result.sanitized_text
                    action_taken = ActionOnFail.SANITIZE
            else:
                # Even for passing guardrails, check if they provide sanitized text
                # (e.g., PII redaction that passes but sanitizes output)
                if result.sanitized_text and result.sanitized_text != original_text:
                    final_text = result.sanitized_text
                    action_taken = ActionOnFail.SANITIZE

        # If all checks passed, status is passed (but final_text may be sanitized)
        if all(r.passed for r in results):
            passed = True

        # Calculate total latency
        total_latency = sum(r.latency_ms for r in results)

        return AggregatedResult(
            passed=passed,
            layer=layer,
            results=results,
            action_taken=action_taken,
            final_text=final_text,
            total_latency_ms=total_latency,
        )

    def merge_sanitized_text(
        self, results: List[GuardrailResult], original_text: str
    ) -> str:
        """Merge sanitized text from multiple guardrails.

        When multiple guardrails sanitize text, this applies
        all sanitizations in order.

        Args:
            results: List of guardrail results
            original_text: The original text

        Returns:
            Text with all sanitizations applied
        """
        current_text = original_text

        for result in results:
            if result.sanitized_text and result.sanitized_text != current_text:
                # Apply this sanitization
                # Note: This is a simple replacement, more sophisticated
                # merging might be needed for overlapping sanitizations
                current_text = result.sanitized_text

        return current_text
