"""Tests for the concurrent guardrail layers in parallel.py:_do_guardrail.

The two layers (guard_input + guard_context) now run via asyncio.gather instead
of sequentially. These tests verify both the latency win (they overlap) and that
the original failure policy is preserved: input fails CLOSED, context fails OPEN.
"""

import sys
import os
import asyncio
import pytest
from unittest.mock import AsyncMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.graph.nodes import parallel


class FakeScan:
    """A single guardrail scanner result."""
    def __init__(self, passed=True, name="scanner", message="msg"):
        self.passed = passed
        self.guardrail_name = name
        self.message = message


class FakeResult:
    """A GuardSession.guard_* result."""
    def __init__(self, passed=True, scans=None, max_risk=0.0):
        self.passed = passed
        self.results = scans if scans is not None else []
        self.max_risk_score = max_risk


def _session(input_result=None, context_result=None, delay=0.0):
    """Fake guardrail session whose guard_input/guard_context return the given
    results (or raise, if an Exception instance is passed). Uses async side-effect
    functions so AsyncMock actually awaits them and returns the real value."""
    async def _guard_input(*a, **k):
        if delay:
            await asyncio.sleep(delay)
        if isinstance(input_result, BaseException):
            raise input_result
        return input_result

    async def _guard_context(*a, **k):
        if delay:
            await asyncio.sleep(delay)
        if isinstance(context_result, BaseException):
            raise context_result
        return context_result

    sess = AsyncMock()
    sess.guard_input = AsyncMock(side_effect=_guard_input)
    sess.guard_context = AsyncMock(side_effect=_guard_context)
    return sess


def _patch_session(sess):
    return patch.object(parallel, "init_guardrail_session", AsyncMock(return_value=sess))


class TestGuardrailConcurrency:
    @pytest.mark.asyncio
    async def test_both_pass(self):
        sess = _session(FakeResult(passed=True), FakeResult(passed=True))
        with _patch_session(sess):
            out = await parallel._do_guardrail({"query": "what is my portfolio value"})
        assert out["guardrail_passed"] is True
        sess.guard_input.assert_awaited_once()
        sess.guard_context.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_input_failure_fails_closed(self):
        scans = [FakeScan(passed=False, name="pii_detection", message="SSN found")]
        sess = _session(FakeResult(passed=False, scans=scans, max_risk=0.9),
                        FakeResult(passed=True))
        with _patch_session(sess):
            out = await parallel._do_guardrail({"query": "my ssn is 123-45-6789"})
        assert out["guardrail_passed"] is False
        assert out["guardrail_result"]["blocked_by"] == "pii_detection"

    @pytest.mark.asyncio
    async def test_context_failure_blocks_when_input_passes(self):
        scans = [FakeScan(passed=False, name="topical_rail", message="off topic")]
        sess = _session(FakeResult(passed=True),
                        FakeResult(passed=False, scans=scans, max_risk=0.7))
        with _patch_session(sess):
            out = await parallel._do_guardrail({"query": "who will win the election"})
        assert out["guardrail_passed"] is False
        assert out["guardrail_result"]["blocked_by"] == "topical_rail"

    @pytest.mark.asyncio
    async def test_context_exception_fails_open(self):
        """A provider/LLM error in the topical rail must NOT block a valid query."""
        sess = _session(FakeResult(passed=True), RuntimeError("nemo provider down"))
        with _patch_session(sess):
            out = await parallel._do_guardrail({"query": "show my aum"})
        assert out["guardrail_passed"] is True

    @pytest.mark.asyncio
    async def test_input_exception_fails_open(self):
        """An input-scanner crash is caught by the outer handler (session-level
        fail-open) so infra errors never hard-block."""
        sess = _session(RuntimeError("scanner crashed"), FakeResult(passed=True))
        with _patch_session(sess):
            out = await parallel._do_guardrail({"query": "show my aum"})
        assert out["guardrail_passed"] is True

    @pytest.mark.asyncio
    async def test_layers_run_concurrently(self):
        """Each layer sleeps 50ms; concurrently total wall-time must be well under
        the 100ms they'd take sequentially."""
        sess = _session(FakeResult(passed=True), FakeResult(passed=True), delay=0.05)
        with _patch_session(sess):
            start = asyncio.get_event_loop().time()
            out = await parallel._do_guardrail({"query": "valid query"})
            elapsed = asyncio.get_event_loop().time() - start
        assert out["guardrail_passed"] is True
        assert elapsed < 0.09, f"layers did not overlap (took {elapsed:.3f}s)"
