"""Tests for auto-attaching the Nifty 50 1Y/3Y benchmark to client-XIRR questions.

Whenever an RM asks for a client's XIRR, the plan should also fetch the benchmark as a
client-independent parallel sub-step, and the synthesis should say to show it only WHEN
AVAILABLE (so it's safe before the MCP returns it). List/aggregate XIRR questions and plans
that already cover the benchmark are left untouched. The deterministic simple path is also
exercised through ``plan_query`` (which makes no LLM call for an obviously-simple question).
"""

import sys
import os
import asyncio

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.graph.nodes.planner import (
    _attach_benchmark,
    _direct_plan,
    _BENCHMARK_STEP_Q,
    plan_query,
)


def _has_benchmark_step(plan):
    return any(s["question"] == _BENCHMARK_STEP_Q for s in plan.get("steps", []))


# --- _attach_benchmark ----------------------------------------------------------

def test_direct_xirr_question_gains_benchmark_step():
    plan = _direct_plan("What is the XIRR of Ram Krishnan?")
    out = _attach_benchmark(plan, "What is the XIRR of Ram Krishnan?")
    assert out["strategy"] == "decompose"
    assert _has_benchmark_step(out)
    # original client step is preserved, benchmark appended (no dependency -> parallel)
    assert out["steps"][0]["question"] == "What is the XIRR of Ram Krishnan?"
    assert out["steps"][-1]["depends_on"] == []
    assert "when available" in out["synthesis"].lower()


def test_decompose_plan_with_xirr_step_gets_benchmark_appended():
    plan = {
        "strategy": "decompose", "reason": "multi", "synthesis": "Combine AUM and XIRR.",
        "steps": [
            {"question": "What is the total AUM of Ram?", "depends_on": [], "fills": None, "for_each": False},
            {"question": "What is the XIRR of Ram?", "depends_on": [], "fills": None, "for_each": False},
        ],
    }
    out = _attach_benchmark(plan, "Show Ram's AUM and XIRR.")
    assert _has_benchmark_step(out)
    assert len(out["steps"]) == 3
    assert out["synthesis"].startswith("Combine AUM and XIRR.")


def test_non_xirr_question_unchanged():
    plan = _direct_plan("What is the total AUM of Ram?")
    out = _attach_benchmark(plan, "What is the total AUM of Ram?")
    assert out["strategy"] == "direct"
    assert not _has_benchmark_step(out)


def test_list_xirr_question_left_alone():
    # A cross-client / list XIRR ask would make a single benchmark line noise.
    for q in ("List the XIRR of all clients", "Top 5 clients by XIRR", "Compare XIRR across clients"):
        plan = _direct_plan(q)
        out = _attach_benchmark(plan, q)
        assert not _has_benchmark_step(out), q


def test_existing_benchmark_not_duplicated():
    plan = {
        "strategy": "decompose", "reason": "", "synthesis": "",
        "steps": [
            {"question": "What is the XIRR of Ram?", "depends_on": [], "fills": None, "for_each": False},
            {"question": "What is the Nifty 50 1-year return?", "depends_on": [], "fills": None, "for_each": False},
        ],
    }
    out = _attach_benchmark(plan, "What is the XIRR of Ram vs the Nifty 50?")
    assert sum(1 for s in out["steps"] if "nifty" in s["question"].lower()) == 1
    assert not _has_benchmark_step(out)  # the literal injected step was NOT added


# --- plan_query (deterministic simple path, no LLM) -----------------------------

def test_plan_query_simple_xirr_attaches_benchmark():
    state = {"enriched_query": "What is the XIRR of Ram Krishnan?",
             "mode_config": {"planner_enabled": True, "decomposition_enabled": True}}
    plan = asyncio.run(plan_query(state))
    assert plan["strategy"] == "decompose"
    assert _has_benchmark_step(plan)


def test_plan_query_quickfacts_mode_stays_single_shot():
    # Decomposition disabled (Quick Facts) -> no benchmark fan-out, answer stays direct.
    state = {"enriched_query": "What is the XIRR of Ram Krishnan?",
             "mode_config": {"planner_enabled": True, "decomposition_enabled": False}}
    plan = asyncio.run(plan_query(state))
    assert plan["strategy"] == "direct"
    assert not _has_benchmark_step(plan)


# --- end-to-end through the real mode config + skill matcher ---------------------
# A plain client-XIRR question must NOT match a methodology skill (which would bypass the
# benchmark injection), and must get the benchmark in Client Insights / Deep Insight but not in
# Quick Facts. This guards the actual routing parallel.py uses.

import pytest
from app.modes import mode_config
from app.skills.registry import registry


def _route_plan(query, mode):
    """Mirror parallel.py: real skill selection, then plan_query for an XIRR question."""
    mc = mode_config(mode)
    state = {"enriched_query": query, "query": query, "mode_config": mc,
             "investor_name": "Ram Krishnan", "enrichment_entities": {"investor_name": "Ram Krishnan"}}
    skill = None
    if mc.get("skills_enabled"):
        skill = registry.match(query, domain="wealth_management", client_resolved=True, intent="lookup")
    assert skill is None, f"plain XIRR question unexpectedly matched skill {getattr(skill,'skill_id',None)}"
    return asyncio.run(plan_query(state))


@pytest.mark.parametrize("mode", ["client", "deep"])
def test_insight_modes_attach_benchmark(mode):
    plan = _route_plan("What is the XIRR of Ram Krishnan?", mode)
    assert plan["strategy"] == "decompose"
    assert _has_benchmark_step(plan)


def test_quickfacts_mode_no_benchmark_end_to_end():
    plan = _route_plan("What is the XIRR of Ram Krishnan?", "quick")
    assert plan["strategy"] == "direct"
    assert not _has_benchmark_step(plan)
