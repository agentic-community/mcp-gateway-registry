"""Unit tests for registry/api/m2m_management_routes.py (issue #851).

Tests the direct M2M client registration endpoints:
- POST   /api/iam/m2m-clients
- GET    /api/iam/m2m-clients
- GET    /api/iam/m2m-clients/{client_id}
- PATCH  /api/iam/m2m-clients/{client_id}
- DELETE /api/iam/m2m-clients/{client_id}
"""

import logging
from datetime import datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from registry.schemas.idp_m2m_client import MANUAL_PROVIDER, IdPM2MClient
from registry.services.m2m_management_service import (
    M2MClientConflict,
    M2MClientImmutable,
    M2MClientNotFound,
)

logger = logging.getLogger(__name__)


@pytest.fixture
def admin_user_context() -> dict[str, Any]:
    return {
        "username": "admin",
        "is_admin": True,
        "groups": ["mcp-registry-admin"],
        "scopes": ["mcp-servers-unrestricted/read"],
        "accessible_servers": ["all"],
        "accessible_services": ["all"],
        "accessible_agents": ["all"],
        "ui_permissions": {"register_service": ["all"]},
        "auth_method": "session",
    }


@pytest.fixture
def regular_user_context() -> dict[str, Any]:
    return {
        "username": "user",
        "is_admin": False,
        "groups": ["test-group"],
        "scopes": [],
        "accessible_servers": [],
        "accessible_services": [],
        "accessible_agents": [],
        "ui_permissions": {},
        "auth_method": "session",
    }


@pytest.fixture
def sample_client() -> IdPM2MClient:
    now = datetime.utcnow()
    return IdPM2MClient(
        client_id="test-client-id",
        name="Test Client",
        description="desc",
        groups=["group-a"],
        enabled=True,
        provider=MANUAL_PROVIDER,
        idp_app_id=None,
        created_by="admin",
        created_at=now,
        updated_at=now,
    )


def _override_auth(user_context: dict | None) -> None:
    """Override the nginx_proxied_auth FastAPI dependency."""
    from registry.auth.dependencies import nginx_proxied_auth
    from registry.main import app

    def _override() -> dict | None:
        return user_context

    app.dependency_overrides[nginx_proxied_auth] = _override


@pytest.fixture
def mock_service() -> MagicMock:
    service = MagicMock()
    service.create = AsyncMock()
    service.list_paged = AsyncMock()
    service.get = AsyncMock()
    service.patch = AsyncMock()
    service.delete = AsyncMock()
    return service


@pytest.fixture
def client_admin(mock_settings, admin_user_context, mock_service):
    from registry.main import app

    _override_auth(admin_user_context)
    with patch(
        "registry.api.m2m_management_routes._get_service",
        new=AsyncMock(return_value=mock_service),
    ):
        client = TestClient(app, cookies={"mcp_gateway_session": "test-session"})
        yield client, mock_service
    app.dependency_overrides.clear()


@pytest.fixture
def client_regular(mock_settings, regular_user_context, mock_service):
    from registry.main import app

    _override_auth(regular_user_context)
    with patch(
        "registry.api.m2m_management_routes._get_service",
        new=AsyncMock(return_value=mock_service),
    ):
        client = TestClient(app, cookies={"mcp_gateway_session": "test-session"})
        yield client, mock_service
    app.dependency_overrides.clear()


@pytest.fixture
def client_anon(mock_settings, mock_service):
    from registry.main import app

    _override_auth(None)
    with patch(
        "registry.api.m2m_management_routes._get_service",
        new=AsyncMock(return_value=mock_service),
    ):
        client = TestClient(app, cookies={"mcp_gateway_session": "test-session"})
        yield client, mock_service
    app.dependency_overrides.clear()


@pytest.mark.unit
@pytest.mark.api
class TestCreateM2MClient:
    """Tests for POST /api/iam/m2m-clients."""

    def test_unauthenticated_returns_401(self, client_anon):
        client, _ = client_anon

        response = client.post(
            "/api/iam/m2m-clients",
            json={"client_id": "abc", "client_name": "x"},
        )

        assert response.status_code == 401

    def test_non_admin_returns_403(self, client_regular):
        client, _ = client_regular

        response = client.post(
            "/api/iam/m2m-clients",
            json={"client_id": "abc", "client_name": "x"},
        )

        assert response.status_code == 403

    def test_happy_path_returns_201(
        self,
        client_admin,
        sample_client,
    ):
        client, service = client_admin
        service.create.return_value = sample_client

        response = client.post(
            "/api/iam/m2m-clients",
            json={
                "client_id": "test-client-id",
                "client_name": "Test Client",
                "groups": ["group-a"],
                "description": "desc",
            },
        )

        assert response.status_code == 201
        body = response.json()
        assert body["client_id"] == "test-client-id"
        assert body["provider"] == MANUAL_PROVIDER
        service.create.assert_awaited_once()
        create_kwargs = service.create.await_args.kwargs
        assert create_kwargs["created_by"] == "admin"

    def test_conflict_returns_409(self, client_admin):
        client, service = client_admin
        service.create.side_effect = M2MClientConflict("dup")

        response = client.post(
            "/api/iam/m2m-clients",
            json={"client_id": "dup", "client_name": "x"},
        )

        assert response.status_code == 409

    def test_invalid_client_id_returns_422(self, client_admin):
        client, _ = client_admin

        response = client.post(
            "/api/iam/m2m-clients",
            json={"client_id": "bad id with space", "client_name": "x"},
        )

        assert response.status_code == 422


@pytest.mark.unit
@pytest.mark.api
class TestListM2MClients:
    """Tests for GET /api/iam/m2m-clients."""

    def test_unauthenticated_returns_401(self, client_anon):
        client, _ = client_anon

        response = client.get("/api/iam/m2m-clients")

        assert response.status_code == 401

    def test_returns_paginated_envelope(
        self,
        client_admin,
        sample_client,
    ):
        client, service = client_admin
        service.list_paged.return_value = ([sample_client], 1)

        response = client.get("/api/iam/m2m-clients")

        assert response.status_code == 200
        body = response.json()
        assert body["total"] == 1
        assert body["limit"] == 500
        assert body["skip"] == 0
        assert len(body["items"]) == 1
        assert body["items"][0]["client_id"] == "test-client-id"

    def test_passes_provider_filter(self, client_admin):
        client, service = client_admin
        service.list_paged.return_value = ([], 0)

        client.get("/api/iam/m2m-clients?provider=manual")

        kwargs = service.list_paged.await_args.kwargs
        assert kwargs["provider"] == "manual"

    def test_enforces_limit_cap(self, client_admin):
        client, service = client_admin
        service.list_paged.return_value = ([], 0)

        response = client.get("/api/iam/m2m-clients?limit=5000")

        assert response.status_code == 422  # exceeds le=1000

    def test_passes_skip_and_limit(self, client_admin):
        client, service = client_admin
        service.list_paged.return_value = ([], 0)

        client.get("/api/iam/m2m-clients?limit=25&skip=10")

        kwargs = service.list_paged.await_args.kwargs
        assert kwargs["limit"] == 25
        assert kwargs["skip"] == 10


@pytest.mark.unit
@pytest.mark.api
class TestGetM2MClient:
    """Tests for GET /api/iam/m2m-clients/{client_id}."""

    def test_unauthenticated_returns_401(self, client_anon):
        client, _ = client_anon

        response = client.get("/api/iam/m2m-clients/x")

        assert response.status_code == 401

    def test_returns_200_on_found(
        self,
        client_admin,
        sample_client,
    ):
        client, service = client_admin
        service.get.return_value = sample_client

        response = client.get("/api/iam/m2m-clients/test-client-id")

        assert response.status_code == 200
        assert response.json()["client_id"] == "test-client-id"

    def test_returns_404_on_missing(self, client_admin):
        client, service = client_admin
        service.get.side_effect = M2MClientNotFound("missing")

        response = client.get("/api/iam/m2m-clients/missing")

        assert response.status_code == 404


@pytest.mark.unit
@pytest.mark.api
class TestPatchM2MClient:
    """Tests for PATCH /api/iam/m2m-clients/{client_id}."""

    def test_non_admin_returns_403(self, client_regular):
        client, _ = client_regular

        response = client.patch(
            "/api/iam/m2m-clients/x",
            json={"groups": ["g1"]},
        )

        assert response.status_code == 403

    def test_happy_path_returns_200(
        self,
        client_admin,
        sample_client,
    ):
        client, service = client_admin
        service.patch.return_value = sample_client

        response = client.patch(
            "/api/iam/m2m-clients/test-client-id",
            json={"groups": ["new-group"]},
        )

        assert response.status_code == 200

    def test_not_found_returns_404(self, client_admin):
        client, service = client_admin
        service.patch.side_effect = M2MClientNotFound("missing")

        response = client.patch(
            "/api/iam/m2m-clients/missing",
            json={"groups": ["g1"]},
        )

        assert response.status_code == 404

    def test_immutable_returns_403(self, client_admin):
        client, service = client_admin
        service.patch.side_effect = M2MClientImmutable("sync-id")

        response = client.patch(
            "/api/iam/m2m-clients/sync-id",
            json={"groups": ["g1"]},
        )

        assert response.status_code == 403


@pytest.mark.unit
@pytest.mark.api
class TestDeleteM2MClient:
    """Tests for DELETE /api/iam/m2m-clients/{client_id}."""

    def test_non_admin_returns_403(self, client_regular):
        client, _ = client_regular

        response = client.delete("/api/iam/m2m-clients/x")

        assert response.status_code == 403

    def test_happy_path_returns_204(self, client_admin):
        client, service = client_admin
        service.delete.return_value = None

        response = client.delete("/api/iam/m2m-clients/test-client-id")

        assert response.status_code == 204

    def test_not_found_returns_404(self, client_admin):
        client, service = client_admin
        service.delete.side_effect = M2MClientNotFound("missing")

        response = client.delete("/api/iam/m2m-clients/missing")

        assert response.status_code == 404

    def test_immutable_returns_403(self, client_admin):
        client, service = client_admin
        service.delete.side_effect = M2MClientImmutable("sync-id")

        response = client.delete("/api/iam/m2m-clients/sync-id")

        assert response.status_code == 403
