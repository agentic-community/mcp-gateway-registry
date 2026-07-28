"""Tests for GET /api/egress-auth/available-servers connect_url field.

The endpoint lists per-user-egress servers the caller can reach. This suite
covers the new ``connect_url`` field (issue #1495 / egress-connect affordance):
- oauth_user rows carry a registry_url-based connect_url.
- pat rows carry connect_url = None (pat cannot use the OAuth front door).
- the URL base tracks settings.registry_url.
- unchanged gating: [] for non-per-user callers, 404 when disabled.
"""

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import registry.api.egress_auth_routes as routes


def _servers():
    """Two egress servers (oauth_user + pat) plus a non-egress one."""
    return {
        "/slack": {
            "server_name": "Slack MCP",
            "egress_auth_mode": "oauth_user",
            "egress_oauth": {"provider": "slack"},
        },
        "/acme-pat": {
            "server_name": "Acme PAT",
            "egress_auth_mode": "pat",
            "egress_oauth": {"provider": "custom"},
        },
        "/plain": {
            "server_name": "Plain",
            "egress_auth_mode": "none",
            "egress_oauth": None,
        },
    }


class _StubServerService:
    def __init__(self, servers):
        self._servers = servers

    async def get_all_servers(self):
        return self._servers


@pytest.fixture
def make_client(monkeypatch):
    """Factory: TestClient with controllable user context, servers, settings."""

    def _build(
        user_context,
        servers,
        enabled=True,
        registry_url="https://gw.example.com",
    ):
        monkeypatch.setattr(routes.settings, "egress_auth_enabled", enabled)
        monkeypatch.setattr(routes.settings, "registry_url", registry_url)
        monkeypatch.setattr(routes, "server_service", _StubServerService(servers))

        app = FastAPI()
        app.include_router(routes.router)
        app.dependency_overrides[routes.nginx_proxied_auth] = lambda: user_context
        return TestClient(app)

    return _build


def _ctx(**over):
    base = {"auth_method": "oauth2", "accessible_servers": ["*"]}
    base.update(over)
    return base


@pytest.mark.unit
class TestAvailableServersConnectUrl:
    def test_oauth_user_row_has_connect_url(self, make_client):
        client = make_client(_ctx(), _servers())
        rows = client.get("/egress-auth/available-servers").json()
        slack = next(r for r in rows if r["server_path"] == "/slack")
        assert (
            slack["connect_url"] == "https://gw.example.com/oauth2/egress/connect?server=%2Fslack"
        )

    def test_pat_row_connect_url_is_none(self, make_client):
        client = make_client(_ctx(), _servers())
        rows = client.get("/egress-auth/available-servers").json()
        pat = next(r for r in rows if r["server_path"] == "/acme-pat")
        assert pat["egress_auth_mode"] == "pat"
        assert pat["connect_url"] is None

    def test_connect_url_uses_configured_registry_url(self, make_client):
        client = make_client(_ctx(), _servers(), registry_url="http://localhost:9999")
        rows = client.get("/egress-auth/available-servers").json()
        slack = next(r for r in rows if r["server_path"] == "/slack")
        assert slack["connect_url"].startswith("http://localhost:9999/oauth2/egress/connect")

    def test_non_egress_server_absent(self, make_client):
        client = make_client(_ctx(), _servers())
        rows = client.get("/egress-auth/available-servers").json()
        assert all(r["server_path"] != "/plain" for r in rows)

    def test_non_per_user_caller_gets_empty(self, make_client):
        client = make_client(_ctx(auth_method="network-trusted"), _servers())
        assert client.get("/egress-auth/available-servers").json() == []

    def test_feature_disabled_404(self, make_client):
        client = make_client(_ctx(), _servers(), enabled=False)
        assert client.get("/egress-auth/available-servers").status_code == 404
