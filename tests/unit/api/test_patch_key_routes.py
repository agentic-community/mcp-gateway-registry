"""Unit tests for registry/api/patch_key_routes.py (wire-platform-v1 2.1).

Covers the console API surface:
- POST   /api/patch-keys        (mint; plaintext returned once)
- GET    /api/patch-keys        (list own keys; metadata only)
- DELETE /api/patch-keys/{id}   (revoke own key; 404 for other users' keys)

The service layer is mocked (its logic has its own suite); auth is faked via
the nginx_proxied_auth dependency override, mirroring the m2m routes tests.
"""

import logging
from datetime import datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from registry.schemas.patch_key import (
    PATCH_KEY_STATUS_ACTIVE,
    PATCH_KEY_STATUS_REVOKED,
    PatchKeyCreated,
    PatchKeyInfo,
)
from registry.services.patch_key_service import (
    PatchKeyAlreadyRevoked,
    PatchKeyNotFound,
    PatchKeyQuotaExceeded,
)

logger = logging.getLogger(__name__)


@pytest.fixture
def console_user_context() -> dict[str, Any]:
    return {
        "username": "alice",
        "email": "alice@example.com",
        "groups": ["developers"],
        "scopes": ["read:servers"],
        "auth_method": "oauth2",
        "is_admin": False,
    }


@pytest.fixture
def mock_service() -> MagicMock:
    service = MagicMock()
    service.mint_key = AsyncMock()
    service.list_keys_for_user = AsyncMock(return_value=[])
    service.revoke_key = AsyncMock()
    return service


def _override_auth(user_context: dict | None) -> None:
    from registry.auth.dependencies import nginx_proxied_auth
    from registry.main import app

    app.dependency_overrides[nginx_proxied_auth] = lambda: user_context


@pytest.fixture
def client(console_user_context, mock_settings, mock_service):
    from registry.main import app

    _override_auth(console_user_context)
    with patch(
        "registry.api.patch_key_routes._get_service",
        new=AsyncMock(return_value=mock_service),
    ):
        client = TestClient(app, cookies={"mcp_gateway_session": "test-session"})
        yield client, mock_service
    app.dependency_overrides.clear()


@pytest.fixture
def anon_client(mock_settings, mock_service):
    from registry.main import app

    _override_auth(None)
    with patch(
        "registry.api.patch_key_routes._get_service",
        new=AsyncMock(return_value=mock_service),
    ):
        client = TestClient(app, cookies={"mcp_gateway_session": "test-session"})
        yield client, mock_service
    app.dependency_overrides.clear()


def _info(status: str = PATCH_KEY_STATUS_ACTIVE) -> PatchKeyInfo:
    return PatchKeyInfo(
        key_id="k1",
        name="ci-runner",
        key_prefix="wgk-ab12cd34",
        username="alice",
        email="alice@example.com",
        groups=["developers"],
        status=status,
        created_at=datetime.utcnow(),
        last_used_at=None,
        revoked_at=datetime.utcnow() if status == PATCH_KEY_STATUS_REVOKED else None,
    )


@pytest.mark.unit
@pytest.mark.api
class TestMintPatchKey:
    def test_unauthenticated_401(self, anon_client):
        client, service = anon_client
        response = client.post("/api/patch-keys", json={"name": "ci"})
        assert response.status_code == 401
        service.mint_key.assert_not_awaited()

    def test_mint_returns_plaintext_once(self, client):
        client, service = client
        created = PatchKeyCreated(key="wgk-" + "x" * 43, info=_info())
        service.mint_key.return_value = created

        response = client.post("/api/patch-keys", json={"name": "ci-runner"})

        assert response.status_code == 201
        body = response.json()
        assert body["key"] == created.key
        assert body["info"]["key_id"] == "k1"
        assert body["info"]["status"] == PATCH_KEY_STATUS_ACTIVE
        # the stored hash never appears in the response
        assert "key_hash" not in body["info"]
        # mint captured the console user's identity/groups snapshot
        service.mint_key.assert_awaited_once()
        kwargs = service.mint_key.call_args.kwargs
        assert kwargs["username"] == "alice"
        assert kwargs["groups"] == ["developers"]
        assert kwargs["name"] == "ci-runner"

    def test_mint_quota_429(self, client):
        client, service = client
        service.mint_key.side_effect = PatchKeyQuotaExceeded("alice", 20)
        response = client.post("/api/patch-keys", json={"name": "ci"})
        assert response.status_code == 429

    def test_blank_name_rejected(self, client):
        client, _ = client
        response = client.post("/api/patch-keys", json={"name": "   "})
        assert response.status_code == 422


@pytest.mark.unit
@pytest.mark.api
class TestListPatchKeys:
    def test_list_returns_own_keys_metadata_only(self, client):
        client, service = client
        service.list_keys_for_user.return_value = [_info(), _info(PATCH_KEY_STATUS_REVOKED)]
        response = client.get("/api/patch-keys")
        assert response.status_code == 200
        body = response.json()
        assert body["total"] == 2
        assert all("key" not in item and "key_hash" not in item for item in body["items"])
        service.list_keys_for_user.assert_awaited_once_with("alice")

    def test_unauthenticated_401(self, anon_client):
        client, service = anon_client
        response = client.get("/api/patch-keys")
        assert response.status_code == 401


@pytest.mark.unit
@pytest.mark.api
class TestRevokePatchKey:
    def test_revoke_own_key(self, client):
        client, service = client
        service.revoke_key.return_value = _info(PATCH_KEY_STATUS_REVOKED)
        response = client.delete("/api/patch-keys/k1")
        assert response.status_code == 200
        assert response.json()["status"] == PATCH_KEY_STATUS_REVOKED
        service.revoke_key.assert_awaited_once_with(key_id="k1", username="alice")

    def test_revoke_unknown_or_foreign_key_404(self, client):
        client, service = client
        service.revoke_key.side_effect = PatchKeyNotFound("k9")
        response = client.delete("/api/patch-keys/k9")
        assert response.status_code == 404

    def test_revoke_twice_409(self, client):
        client, service = client
        service.revoke_key.side_effect = PatchKeyAlreadyRevoked("k1")
        response = client.delete("/api/patch-keys/k1")
        assert response.status_code == 409

    def test_unauthenticated_401(self, anon_client):
        client, _ = anon_client
        response = client.delete("/api/patch-keys/k1")
        assert response.status_code == 401


@pytest.mark.unit
@pytest.mark.api
class TestPlaintextLogSafety:
    """The mint plaintext must not reach the application log."""

    def test_mint_does_not_log_plaintext(self, client, caplog):
        client, service = client
        plaintext = "wgk-SUPERSECRETplaintextvalue-do-not-log-123"
        service.mint_key.return_value = PatchKeyCreated(key=plaintext, info=_info())

        with caplog.at_level(logging.DEBUG):
            response = client.post("/api/patch-keys", json={"name": "ci"})
        assert response.status_code == 201
        assert plaintext not in caplog.text
        assert plaintext not in response.headers.get("X-Audit-Action", "")
