"""Authorization tests for federation_routes management endpoints.

The /api/federation/* mutating routes are JWT mirrors of admin-gated
operations that previously enforced no authorization. These tests verify the
_check_federation_management_scope gate added in fix/servers-api-authz:
admin OR federation/peers scope may manage federation config; everyone else
gets 403. Mirrors the sibling gate in peer_management_routes.py.
"""

from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient

from registry.api.federation_routes import _get_federation_repo
from registry.auth.dependencies import nginx_proxied_auth
from registry.main import app


def _non_admin_ctx():
    return {
        "username": "ctuser",
        "groups": ["currenttime-users"],
        "scopes": ["currenttime-users"],
        "is_admin": False,
    }


def _admin_ctx():
    return {
        "username": "admin",
        "groups": ["mcp-registry-admin"],
        "scopes": ["mcp-registry-admin"],
        "is_admin": True,
    }


def _federation_scope_ctx():
    return {
        "username": "peer-1",
        "groups": [],
        "scopes": ["federation/peers"],
        "is_admin": False,
    }


@pytest.fixture
def _mock_repo():
    """A repo whose methods are AsyncMocks so we can assert no-write on 403."""
    repo = AsyncMock()
    repo.delete_config = AsyncMock(return_value=True)
    repo.get_config = AsyncMock(return_value={"id": "default"})
    return repo


def _override(auth_ctx, repo):
    app.dependency_overrides[nginx_proxied_auth] = lambda: auth_ctx
    app.dependency_overrides[_get_federation_repo] = lambda: repo


def _clear():
    app.dependency_overrides.clear()


class TestFederationManagementAuthz:
    """All federation management routes require admin or federation/peers."""

    def test_delete_config_rejects_non_admin(self, _mock_repo):
        _override(_non_admin_ctx(), _mock_repo)
        try:
            client = TestClient(app)
            resp = client.delete("/api/federation/config/default")
        finally:
            _clear()

        assert resp.status_code == 403
        _mock_repo.delete_config.assert_not_called()

    def test_sync_rejects_non_admin(self, _mock_repo):
        _override(_non_admin_ctx(), _mock_repo)
        try:
            client = TestClient(app)
            resp = client.post("/api/federation/sync")
        finally:
            _clear()

        assert resp.status_code == 403

    def test_delete_config_allows_admin(self, _mock_repo):
        _override(_admin_ctx(), _mock_repo)
        try:
            client = TestClient(app)
            resp = client.delete("/api/federation/config/default")
        finally:
            _clear()

        # Admin passes the gate; the route proceeds (not 403).
        assert resp.status_code != 403

    def test_delete_config_allows_federation_scope(self, _mock_repo):
        _override(_federation_scope_ctx(), _mock_repo)
        try:
            client = TestClient(app)
            resp = client.delete("/api/federation/config/default")
        finally:
            _clear()

        # federation/peers scope passes the gate (matches peer management).
        assert resp.status_code != 403
