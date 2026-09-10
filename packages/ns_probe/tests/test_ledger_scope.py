"""Per-turn scope on a ledger row.

A worker can serve several workspaces — it configures one MCP endpoint per workspace — so the
workspace a turn served is a property of the TURN, not of the deployment. `agent.turn()` has always
accepted it; what these cover is the half that did not: a row written when the turn's ambient
context is no longer in scope, which is the case `record(parent=...)` already exists for.

The rule running through all of it: an unscoped row is not an error anywhere. It is written,
ingested and stored like any other, and the READER scopes strictly — so the only symptom is an empty
Outcome Ledger page for an agent that is plainly serving traffic. Measured in dev before this
existed: 30 of 78 rows unreadable, reported twice as "nothing shows up".
"""

import logging

import pytest

from ns_probe import outcome_ledger as ol
from ns_probe.tracer import context_attributes


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    """No ambient deployment scope unless a case asks for it, and a fresh warning latch."""
    monkeypatch.delenv(ol.TENANT_ID_ENV, raising=False)
    monkeypatch.delenv(ol.WORKSPACE_ID_ENV, raising=False)
    monkeypatch.setattr(ol, "_warned_unscoped", False)


def scope(attrs):
    """Just the two scope attributes a row carries, if any."""
    return {k: v for k, v in attrs.items() if k in ("tenant.id", "workspace.id")}


class TestResolutionOrder:
    def test_an_explicit_workspace_is_written_onto_the_row(self):
        # The only input that can differ per turn. A worker serving three workspaces has one
        # deployment and three answers, and nothing but the turn itself knows which applies.
        attrs = ol.build_attributes(tenant_id="t-1", workspace_id="ws-1")

        assert scope(attrs) == {"tenant.id": "t-1", "workspace.id": "ws-1"}

    def test_the_environment_fills_in_outside_a_turn(self, monkeypatch):
        monkeypatch.setenv(ol.TENANT_ID_ENV, "env-t")
        monkeypatch.setenv(ol.WORKSPACE_ID_ENV, "env-ws")

        assert scope(ol.build_attributes()) == {"tenant.id": "env-t", "workspace.id": "env-ws"}

    def test_an_explicit_value_beats_the_environment(self, monkeypatch):
        monkeypatch.setenv(ol.WORKSPACE_ID_ENV, "env-ws")

        assert ol.build_attributes(workspace_id="ws-turn")["workspace.id"] == "ws-turn"

    def test_an_ambient_turn_identity_is_left_ALONE(self, monkeypatch):
        # ⚠️ The subtle one. When a turn published scope, the tracer already folds it under the
        # span's own attributes — so writing the ENV value here would override a correct per-turn
        # answer with a deployment-wide constant, turning a working agent into a mis-scoped one.
        # Writing nothing is what lets the fold win.
        monkeypatch.setenv(ol.WORKSPACE_ID_ENV, "env-ws")

        with context_attributes(**{"tenant.id": "turn-t", "workspace.id": "turn-ws"}):
            assert scope(ol.build_attributes()) == {}

    def test_an_explicit_value_still_beats_the_ambient_turn(self):
        # A turn that fans out across workspaces can still scope one row differently.
        with context_attributes(**{"workspace.id": "turn-ws"}):
            assert ol.build_attributes(workspace_id="row-ws")["workspace.id"] == "row-ws"

    def test_nothing_is_written_when_nothing_has_a_value(self):
        # Absent, never ''. An empty string lands as a real column value that no scoped read can
        # ever match — indistinguishable from a wrong workspace rather than from a missing one.
        assert scope(ol.build_attributes()) == {}

    def test_whitespace_is_not_a_workspace(self):
        assert scope(ol.build_attributes(workspace_id="   ", tenant_id="  ")) == {}


class TestTheCaseParentAlreadyExistsFor:
    def test_a_row_recorded_outside_its_turn_can_still_be_scoped(self):
        # `record(parent=...)` exists because ContextVar parenting does not survive an async
        # generator's `yield`. `parent` restores the TRACE link; attributes are not inherited from a
        # parent SpanContext, so scope was silently lost in exactly that case. Now it can be passed.
        outside_the_turn = ol.build_attributes(tenant_id="t-1", workspace_id="ws-1")

        assert scope(outside_the_turn) == {"tenant.id": "t-1", "workspace.id": "ws-1"}

    def test_without_it_the_row_is_unscoped_and_therefore_unreadable(self):
        # Pinned as the documented shape of the failure, not as acceptable behaviour.
        assert scope(ol.build_attributes()) == {}


class TestTheWarning:
    def test_it_names_the_problem_and_the_fix(self, caplog):
        with caplog.at_level(logging.WARNING, logger="ns_probe.outcome_ledger"):
            ol.build_attributes()

        message = caplog.text
        assert "NOT be readable" in message
        assert ol.WORKSPACE_ID_ENV in message
        assert "turn()" in message

    def test_it_fires_once_per_process_not_once_per_turn(self, caplog):
        # An agent that is misconfigured is misconfigured for every turn it will serve. Thousands of
        # copies would bury the one line that matters.
        with caplog.at_level(logging.WARNING, logger="ns_probe.outcome_ledger"):
            for _ in range(50):
                ol.build_attributes()

        assert len(caplog.records) == 1

    def test_it_stays_quiet_when_the_turn_published_scope(self, caplog):
        # The ambient case writes nothing onto the row, so a naive check of the row's attributes
        # would warn on every correctly-scoped turn — the exact noise that gets a warning muted.
        with caplog.at_level(logging.WARNING, logger="ns_probe.outcome_ledger"):
            with context_attributes(**{"tenant.id": "turn-t", "workspace.id": "turn-ws"}):
                ol.build_attributes()

        assert caplog.records == []

    def test_it_stays_quiet_when_the_environment_supplies_scope(self, caplog, monkeypatch):
        monkeypatch.setenv(ol.TENANT_ID_ENV, "env-t")
        monkeypatch.setenv(ol.WORKSPACE_ID_ENV, "env-ws")

        with caplog.at_level(logging.WARNING, logger="ns_probe.outcome_ledger"):
            ol.build_attributes()

        assert caplog.records == []

    def test_it_warns_about_a_partial_scope_too(self, caplog, monkeypatch):
        # Tenant without workspace is the state the concierge shipped in: readable by the tenant
        # predicate, invisible to the page, which requires a workspace.
        monkeypatch.setenv(ol.TENANT_ID_ENV, "env-t")

        with caplog.at_level(logging.WARNING, logger="ns_probe.outcome_ledger"):
            ol.build_attributes()

        assert "workspace.id" in caplog.text
        assert "tenant.id" not in caplog.text.split("Pass")[0]


class TestPropagation:
    def test_workspace_travels_beside_tenant_in_tracestate(self):
        # Scope is a TRIPLE, and workspace is the enforced boundary — a downstream service handed
        # only the tenant cannot scope to it. It was reachable through `additional_state`, which
        # left every caller to spell the key differently.
        from ns_probe.propagation import inject_context
        from ns_probe.span import SpanContext

        headers: dict = {}
        inject_context(
            headers,
            SpanContext(trace_id="4bf92f3577b34da6a3ce929d0e0e4736", span_id="00f067aa0ba902b7"),
            tenant_id="t-1",
            workspace_id="ws-1",
        )

        assert "tenant:t-1" in headers["tracestate"]
        assert "workspace:ws-1" in headers["tracestate"]

    def test_an_absent_workspace_adds_nothing(self):
        from ns_probe.propagation import inject_context
        from ns_probe.span import SpanContext

        headers: dict = {}
        inject_context(
            headers,
            SpanContext(trace_id="4bf92f3577b34da6a3ce929d0e0e4736", span_id="00f067aa0ba902b7"),
            tenant_id="t-1",
        )

        assert "workspace" not in headers.get("tracestate", "")
