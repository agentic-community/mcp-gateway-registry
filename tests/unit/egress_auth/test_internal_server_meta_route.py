"""Tests for POST /internal/server-meta.

The endpoint returns a registered server's per-server metadata
(egress_auth_mode + expires_in_hours) for the ingress /validate hot path.
Security shape mirrors vend_egress_token: gated by validate_internal_auth.
"""

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import registry.api.egress_auth_routes as routes


class _StubRepo:
    def __init__(self, server):
        self._server = server
        self.queried_paths: list[str] = []

    async def get(self, path):
        self.queried_paths.append(path)
        return self._server


@pytest.fixture
def make_client(monkeypatch):
    def _build(server):
        repo = _StubRepo(server)
        monkeypatch.setattr(routes, "get_server_repository", lambda: repo)
        app = FastAPI()
        app.include_router(routes.router)
        app.dependency_overrides[routes.validate_internal_auth] = lambda: "auth-server"
        client = TestClient(app)
        client._repo = repo
        return client

    return _build


def _post(client, server_path="/github-mcp"):
    return client.post("/internal/server-meta", json={"server_path": server_path})


@pytest.mark.unit
class TestInternalServerMetaRoute:
    def test_returns_both_fields(self, make_client):
        client = make_client({"egress_auth_mode": "obo_exchange", "expires_in_hours": 4})
        r = _post(client)
        assert r.status_code == 200
        assert r.json() == {"egress_auth_mode": "obo_exchange", "expires_in_hours": 4}

    def test_plain_server_maps_none_and_null_ttl(self, make_client):
        # A plain server (no egress mode, no per-server ceiling) reports "none"
        # and a null expires_in_hours -> auth_server falls back to the default.
        client = make_client({})
        r = _post(client)
        assert r.status_code == 200
        assert r.json() == {"egress_auth_mode": "none", "expires_in_hours": None}

    def test_unknown_server_404(self, make_client):
        client = make_client(None)
        r = _post(client)
        assert r.status_code == 404

    def test_leading_slash_normalized(self, make_client):
        # Accept the bare first-segment form and normalize before lookup.
        client = make_client({"egress_auth_mode": "oauth_user"})
        r = _post(client, server_path="github-mcp")
        assert r.status_code == 200
        assert client._repo.queried_paths == ["/github-mcp"]
