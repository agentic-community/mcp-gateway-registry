"""Endpoint tests for POST /api/agents/{path}/pull-card.

These exercise the route logic (permissions, federated/protocol/URL guards,
diff, apply, single-write side effect) with agent_service and the remote fetch
mocked via unittest.mock, mirroring the existing test_agent_routes.py style.
A respx + live-MongoDB integration variant (testing.md section 6 "integration")
is a follow-up that would require adding respx as a dev dependency.

Covers testing.md sections 1.1.x (functional/permission) and 6.6.x (apply path).
"""

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI, HTTPException, status
from fastapi.testclient import TestClient

from registry.api.agent_routes import router
from tests.fixtures.factories import AgentCardFactory

AGENT_PATH = "/test-a2a-agent"
CARD_URL = "https://agent.example.com/.well-known/agent-card.json"


@pytest.fixture
def mock_user_context() -> dict[str, Any]:
    return {
        "username": "testuser",
        "groups": ["dev-group"],
        "scopes": ["write:agents"],
        "auth_method": "session",
        "provider": "local",
        "accessible_agents": ["all"],
        "ui_permissions": {"modify_service": ["all"]},
        "is_admin": False,
    }


@pytest.fixture
def test_client(mock_user_context):
    app = FastAPI()
    app.include_router(router)
    from registry.api.agent_routes import nginx_proxied_auth

    app.dependency_overrides[nginx_proxied_auth] = lambda: mock_user_context
    client = TestClient(app)
    yield client
    app.dependency_overrides.clear()


def _a2a_agent(**overrides):
    defaults = {
        "name": "agent",
        "description": "d",
        "version": "1.0.0",
        "url": "http://localhost:9000/a",
        "skills": [],
        "path": AGENT_PATH,
        "registered_by": "testuser",
        "supported_protocol": "a2a",
        "sync_metadata": None,
    }
    defaults.update(overrides)
    return AgentCardFactory(**defaults)


# Patch the permission helper to a no-op so these tests isolate the route's own
# owner/federated/protocol logic (the permission layer is covered elsewhere).
def _run(client, agent, remote_card, *, query="", update_return=None, enabled=True):
    """Helper: wire up all mocks and POST to pull-card. Returns (response, mocks)."""
    svc = MagicMock()
    svc.get_agent_info = AsyncMock(return_value=agent)
    svc.update_agent = AsyncMock(return_value=update_return or agent)
    svc.is_agent_enabled = AsyncMock(return_value=enabled)

    with (
        patch("registry.api.agent_routes._check_agent_permission", MagicMock()),
        patch("registry.api.agent_routes.agent_service", svc),
        patch(
            "registry.api.agent_routes._fetch_remote_agent_card",
            AsyncMock(return_value=(remote_card, CARD_URL)),
        ),
    ):
        resp = client.post(f"/agents{AGENT_PATH}/pull-card{query}")
    return resp, svc


class TestPullCardDryRun:
    def test_111_dry_run_returns_diff(self, test_client):
        agent = _a2a_agent(version="1.0.0", name="agent")
        resp, svc = _run(
            test_client,
            agent,
            {"version": "2.0.0", "name": "renamed"},
            query="?dry_run=true",
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["dry_run"] is True and body["applied"] is False
        fields = {c["field"] for c in body["changes"]}
        assert fields == {"version", "name"}
        assert body["has_changes"] is True
        assert body["health_status"] == "healthy"
        # Health side-effect still writes (single update_agent), but no A2A fields.
        svc.update_agent.assert_awaited_once()
        written = svc.update_agent.await_args.args[1]
        assert "version" not in written and "name" not in written
        assert written["health_status"] == "healthy"

    def test_112_default_dry_run_is_true(self, test_client):
        agent = _a2a_agent()
        resp, _ = _run(test_client, agent, {"version": "2.0.0"})  # no query param
        assert resp.status_code == 200
        assert resp.json()["dry_run"] is True

    def test_114_no_changes_is_noop(self, test_client):
        agent = _a2a_agent(version="1.0.0", name="agent")
        resp, svc = _run(
            test_client,
            agent,
            {"version": "1.0.0", "name": "agent"},
            query="?dry_run=true",
        )
        body = resp.json()
        assert body["has_changes"] is False and body["applied"] is False
        # Health refresh still runs.
        svc.update_agent.assert_awaited_once()


class TestPullCardApply:
    def test_113_apply_updates_a2a_and_preserves_registry(self, test_client):
        agent = _a2a_agent(version="1.0.0")
        resp, svc = _run(
            test_client,
            agent,
            {"version": "2.0.0", "skills": [{"id": "s2", "name": "New"}]},
            query="?dry_run=false",
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["applied"] is True and body["has_changes"] is True
        # P1: exactly one update_agent write merging health + A2A fields.
        svc.update_agent.assert_awaited_once()
        written = svc.update_agent.await_args.args[1]
        assert written["version"] == "2.0.0"
        assert written["health_status"] == "healthy"
        # S5: no registry-managed field is written.
        for forbidden in ("num_stars", "registered_by", "tags", "visibility", "trust_level"):
            assert forbidden not in written

    def test_661_registrant_only_fields_never_written(self, test_client):
        # Remote maliciously echoes registry-managed fields; the diff ignores them.
        agent = _a2a_agent(version="1.0.0")
        resp, svc = _run(
            test_client,
            agent,
            {
                "version": "2.0.0",
                "num_stars": 999,
                "registered_by": "attacker",
                "sync_metadata": {"is_federated": True},
            },
            query="?dry_run=false",
        )
        assert resp.status_code == 200
        written = svc.update_agent.await_args.args[1]
        assert written["version"] == "2.0.0"
        assert "num_stars" not in written
        assert "registered_by" not in written
        assert "sync_metadata" not in written

    def test_662_single_update_agent_call_on_apply(self, test_client):
        agent = _a2a_agent(version="1.0.0")
        _, svc = _run(
            test_client,
            agent,
            {"version": "2.0.0"},
            query="?dry_run=false",
        )
        assert svc.update_agent.await_count == 1


class TestPullCardGuards:
    def test_115_non_a2a_rejected(self, test_client):
        agent = _a2a_agent(supported_protocol="mcp")
        resp, _ = _run(test_client, agent, {"version": "2.0.0"})
        assert resp.status_code == 400
        assert "A2A protocol" in resp.json()["detail"]

    def test_116_no_url_rejected(self, test_client):
        # Blank the URL via model_copy (factory requires a valid URL at construction).
        agent = _a2a_agent().model_copy(update={"url": ""})
        resp, _ = _run(test_client, agent, {"version": "2.0.0"})
        assert resp.status_code == 400
        assert "no registered URL" in resp.json()["detail"]

    def test_117_federated_rejected(self, test_client):
        agent = _a2a_agent(sync_metadata={"is_federated": True, "source_peer_id": "peer-1"})
        resp, _ = _run(test_client, agent, {"version": "2.0.0"})
        assert resp.status_code == 403
        assert "cannot be updated locally" in resp.json()["detail"]

    def test_118_non_owner_rejected(self, test_client):
        agent = _a2a_agent(registered_by="someone-else")
        resp, _ = _run(test_client, agent, {"version": "2.0.0"})
        assert resp.status_code == 403
        assert "agents you registered" in resp.json()["detail"]

    def test_119_unknown_path_404(self, test_client):
        resp, _ = _run(test_client, None, {"version": "2.0.0"})
        assert resp.status_code == 404
        assert "not found" in resp.json()["detail"].lower()

    def test_1110_remote_fetch_failure_502(self, test_client):
        agent = _a2a_agent()
        svc = MagicMock()
        svc.get_agent_info = AsyncMock(return_value=agent)
        with (
            patch("registry.api.agent_routes._check_agent_permission", MagicMock()),
            patch("registry.api.agent_routes.agent_service", svc),
            patch(
                "registry.api.agent_routes._fetch_remote_agent_card",
                AsyncMock(
                    side_effect=HTTPException(
                        status_code=status.HTTP_502_BAD_GATEWAY,
                        detail="Timeout fetching agent card from ...",
                    )
                ),
            ),
        ):
            resp = test_client.post(f"/agents{AGENT_PATH}/pull-card?dry_run=true")
        assert resp.status_code == 502
