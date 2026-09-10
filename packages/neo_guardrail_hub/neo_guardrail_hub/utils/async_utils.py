"""Async utility functions for Neo Guardrail Hub.

This module provides utilities for running async code from
sync contexts and vice versa.
"""

import asyncio
import functools
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Coroutine, TypeVar

T = TypeVar("T")


def run_sync(coro: Coroutine[Any, Any, T]) -> T:
    """Run an async coroutine synchronously.

    Handles the case where there's already an event loop running
    by using a thread pool executor.

    Args:
        coro: Coroutine to run

    Returns:
        Result from the coroutine
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        # No running loop, safe to use asyncio.run
        return asyncio.run(coro)

    # There's a running loop, run in a thread
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(asyncio.run, coro)
        return future.result()


def ensure_async(func: Callable[..., T]) -> Callable[..., Coroutine[Any, Any, T]]:
    """Ensure a function returns a coroutine.

    Wraps sync functions to be async-compatible.

    Args:
        func: Function to wrap

    Returns:
        Async-compatible function
    """
    if asyncio.iscoroutinefunction(func):
        return func

    @functools.wraps(func)
    async def wrapper(*args: Any, **kwargs: Any) -> T:
        return func(*args, **kwargs)

    return wrapper


async def run_in_thread(func: Callable[..., T], *args: Any, **kwargs: Any) -> T:
    """Run a blocking function in a thread pool.

    Useful for running CPU-bound or blocking I/O operations
    without blocking the event loop.

    Args:
        func: Function to run
        *args: Positional arguments
        **kwargs: Keyword arguments

    Returns:
        Result from the function
    """
    loop = asyncio.get_running_loop()
    partial_func = functools.partial(func, *args, **kwargs)
    return await loop.run_in_executor(None, partial_func)


async def gather_with_concurrency(
    limit: int,
    *coros: Coroutine[Any, Any, T],
) -> list[T]:
    """Run coroutines with a concurrency limit.

    Args:
        limit: Maximum number of concurrent tasks
        *coros: Coroutines to run

    Returns:
        List of results
    """
    semaphore = asyncio.Semaphore(limit)

    async def limited_coro(coro: Coroutine[Any, Any, T]) -> T:
        async with semaphore:
            return await coro

    return await asyncio.gather(*[limited_coro(c) for c in coros])


class AsyncContextManager:
    """Base class for async context managers."""

    async def __aenter__(self) -> "AsyncContextManager":
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        pass
