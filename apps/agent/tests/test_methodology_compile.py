"""Approach A — deriving Cortex questions from a skill's markdown methodology.

The LLM is mocked so these are deterministic: they verify the wiring (methodology → derived
plan_steps → decomposition plan) and the graceful fallbacks, not the model's extraction quality.
"""

import asyncio
from unittest.mock import patch

import pytest

from app.skills.models import Skill
from app.skills import methodology_compiler as mc
from app.graph.nodes.skill_compiler import compile_skill_plan


def _methodology_skill():
    return Skill(
        skill_id="pre_meeting_client_brief",
        name="Pre-Meeting Client Brief",
        domain="wealth_management",
        skill_type="reasoning_methodology",
        description="Assemble a pre-meeting brief.",
        content="## Data to fetch\n1. What is the total AUM of {client}?\n2. What is the XIRR of {client}?",
        plan_steps=[],            # platform ships no structured steps — Approach A must derive them
        requires_client=False,
    )


@pytest.fixture(autouse=True)
def _clear_cache():
    mc._CACHE.clear()
    yield
    mc._CACHE.clear()


def _fake_derive(steps, requires_client=True):
    async def _inner(skill):
        return {"steps": [{"question": q} for q in steps], "requires_client": requires_client}
    return _inner


def test_compile_derives_steps_from_methodology():
    """A methodology skill with content but no plan_steps compiles into a decomposition plan
    whose steps come from the derived questions, with {client} filled in."""
    skill = _methodology_skill()
    with patch("app.graph.nodes.skill_compiler.derive_plan_steps",
               _fake_derive(["What is the total AUM of {client}?", "What is the XIRR of {client}?"])):
        plan = asyncio.run(compile_skill_plan([skill], {"selected_client": "Ram Krishnan"}))
    assert plan is not None
    assert plan["strategy"] == "decompose"
    assert plan["skill_id"] == "pre_meeting_client_brief"
    qs = [s["question"] for s in plan["steps"]]
    assert qs == ["What is the total AUM of Ram Krishnan?", "What is the XIRR of Ram Krishnan?"]


def test_derived_requires_client_gate():
    """If the methodology says a client is required, no plan runs without a resolved client."""
    skill = _methodology_skill()
    with patch("app.graph.nodes.skill_compiler.derive_plan_steps",
               _fake_derive(["What is the total AUM of {client}?", "What is the XIRR of {client}?"], requires_client=True)):
        plan = asyncio.run(compile_skill_plan([skill], {}))   # no client resolved
    assert plan is None


def test_compile_falls_back_when_no_steps_derived():
    """If derivation yields nothing (e.g. LLM failure), the skill is content-only → no plan."""
    skill = _methodology_skill()
    with patch("app.graph.nodes.skill_compiler.derive_plan_steps", _fake_derive([])):
        plan = asyncio.run(compile_skill_plan([skill], {"selected_client": "Ram Krishnan"}))
    assert plan is None


def test_structured_plan_steps_skip_derivation():
    """A skill that already has plan_steps must NOT hit the methodology compiler at all."""
    skill = _methodology_skill()
    skill.plan_steps = [{"question": "What is the total AUM of {client}?"},
                        {"question": "What is the XIRR of {client}?"}]

    async def _boom(_skill):
        raise AssertionError("derive_plan_steps should not be called when plan_steps exist")

    with patch("app.graph.nodes.skill_compiler.derive_plan_steps", _boom):
        plan = asyncio.run(compile_skill_plan([skill], {"selected_client": "Priya"}))
    assert plan is not None
    assert [s["question"] for s in plan["steps"]] == [
        "What is the total AUM of Priya?", "What is the XIRR of Priya?"]


def test_derive_caches_per_content_hash():
    """derive_plan_steps caches by (skill_id, content-hash): a second call for the same content
    does not re-invoke the LLM."""
    skill = _methodology_skill()
    calls = {"n": 0}

    class _Resp:
        content = '{"requires_client": true, "steps": ["What is the total AUM of {client}?"]}'

    class _LLM:
        async def ainvoke(self, _msgs):
            calls["n"] += 1
            return _Resp()

    with patch.object(mc, "_llm", lambda: _LLM()):
        r1 = asyncio.run(mc.derive_plan_steps(skill))
        r2 = asyncio.run(mc.derive_plan_steps(skill))
    assert r1 == r2
    assert calls["n"] == 1            # second call served from cache
