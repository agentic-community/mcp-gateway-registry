"""
Unit tests for GET /api/servers/{path}/versions endpoint (SA-19).

The versions endpoint previously ran no authorization and no backend-URL
redaction, so any authenticated user could read every version's internal
proxy_pass_url for any server. These tests lock in the fix, mirroring the
guards on GET /api/servers/{path}:
- 404 when the server does not exist
- 403 when a non-admin lacks access to the server
- admin bypasses the access check
- proxy_pass_url stripped per version for non-admins in with-gateway mode
- proxy_pass_url kept in registry-only mode and always for admins
"""

import logging
from typing import Any
from unittest.mock import (
    AsyncMock,
    MagicMock,
    patch,
)

import pytest
from fastapi.testclient import TestClient

logger = logging.getLogger(__name__)


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def admin_user_context() -> dict[str, Any]:
    """Create admin user context."""
    return {
        "username": "admin",
        "is_admin": True,
        "groups": ["mcp-registry-admin"],
        "accessible_servers": ["all"],
        "auth_method": "session",
    }


@pytest.fixture
def regular_user_context() -> dict[str, Any]:
    """Create regular (non-admin) user context."""
    return {
        "username": "testuser",
        "is_admin": False,
        "groups": ["test-group"],
        "accessible_servers": ["test-server"],
        "auth_method": "session",
    }


@pytest.fixture
def sample_versions() -> dict[str, Any]:
    """Sample return value of server_service.get_server_versions()."""
    return {
        "path": "/test-server",
        "default_version": "v1.1.0",
        "versions": [
            {
                "version": "v1.1.0",
                "proxy_pass_url": "http://internal-backend:8080",
                "status": "active",
                "is_default": True,
            },
            {
                "version": "v1.0.0",
                "proxy_pass_url": "http://internal-backend-old:8080",
                "status": "inactive",
                "is_default": False,
            },
        ],
    }


@pytest.fixture
def mock_server_service():
    """Mock server_service dependency."""
    mock_service = MagicMock()
    # Default: server exists (versions guard resolves it via get_server_info)
    mock_service.get_server_info = AsyncMock(return_value={"server_name": "Test Server"})
    mock_service.user_can_access_server_path = AsyncMock(return_value=True)
    mock_service.get_server_versions = AsyncMock(return_value={"path": "/test-server", "versions": []})
    return mock_service


def _create_test_client(user_context: dict[str, Any]):
    """Yield a TestClient with server_service patched and auth overridden."""
    from registry.auth.dependencies import nginx_proxied_auth
    from registry.main import app

    app.dependency_overrides[nginx_proxied_auth] = lambda: user_context
    client = TestClient(app, cookies={"mcp_gateway_session": "test-session"})
    yield client
    app.dependency_overrides.pop(nginx_proxied_auth, None)


@pytest.fixture
def test_client_admin(mock_settings, mock_server_service, admin_user_context):
    """Test client authenticated as admin, with server_service patched."""
    with patch("registry.api.server_routes.server_service", mock_server_service):
        yield from _create_test_client(admin_user_context)


@pytest.fixture
def test_client_regular(mock_settings, mock_server_service, regular_user_context):
    """Test client authenticated as a non-admin, with server_service patched."""
    with patch("registry.api.server_routes.server_service", mock_server_service):
        yield from _create_test_client(regular_user_context)


# =============================================================================
# TESTS
# =============================================================================


class TestGetServerVersionsAuthz:
    """Authorization + redaction tests for GET /api/servers/{path}/versions."""

    def test_not_found_returns_404(self, test_client_admin, mock_server_service):
        """404 when the server path does not exist."""
        mock_server_service.get_server_info.return_value = None

        response = test_client_admin.get("/api/servers/nope/versions")

        assert response.status_code == 404

    def test_non_admin_without_access_forbidden(
        self, test_client_regular, mock_server_service, sample_versions
    ):
        """403 when a non-admin lacks access to the server."""
        mock_server_service.user_can_access_server_path.return_value = False
        mock_server_service.get_server_versions.return_value = dict(sample_versions)

        response = test_client_regular.get("/api/servers/test-server/versions")

        assert response.status_code == 403
        assert "access" in response.json()["detail"].lower()

    def test_admin_bypasses_access_check(
        self, test_client_admin, mock_server_service, sample_versions
    ):
        """Admin does not run user_can_access_server_path."""
        mock_server_service.get_server_versions.return_value = dict(sample_versions)

        response = test_client_admin.get("/api/servers/test-server/versions")

        assert response.status_code == 200
        mock_server_service.user_can_access_server_path.assert_not_called()

    def test_proxy_pass_url_stripped_for_non_admin_with_gateway(
        self, test_client_regular, mock_server_service, mock_settings, sample_versions
    ):
        """proxy_pass_url stripped from every version for a non-admin in with-gateway mode."""
        from registry.core.config import DeploymentMode

        mock_settings.deployment_mode = DeploymentMode.WITH_GATEWAY
        mock_server_service.user_can_access_server_path.return_value = True
        # Deep copy so the endpoint's dict.pop does not mutate the fixture
        mock_server_service.get_server_versions.return_value = {
            "path": sample_versions["path"],
            "versions": [dict(v) for v in sample_versions["versions"]],
        }

        response = test_client_regular.get("/api/servers/test-server/versions")

        assert response.status_code == 200
        data = response.json()
        assert data["versions"], "expected versions in response"
        for version in data["versions"]:
            assert "proxy_pass_url" not in version

    def test_proxy_pass_url_kept_for_non_admin_registry_only(
        self, test_client_regular, mock_server_service, sample_versions
    ):
        """proxy_pass_url retained for a non-admin in registry-only mode."""
        from registry.core.config import DeploymentMode

        mock_server_service.user_can_access_server_path.return_value = True
        mock_server_service.get_server_versions.return_value = {
            "path": sample_versions["path"],
            "versions": [dict(v) for v in sample_versions["versions"]],
        }

        with patch(
            "registry.api.server_routes.settings.deployment_mode",
            DeploymentMode.REGISTRY_ONLY,
        ):
            response = test_client_regular.get("/api/servers/test-server/versions")

        assert response.status_code == 200
        data = response.json()
        assert all("proxy_pass_url" in v for v in data["versions"])

    def test_proxy_pass_url_kept_for_admin(
        self, test_client_admin, mock_server_service, mock_settings, sample_versions
    ):
        """Admin always sees proxy_pass_url, even in with-gateway mode."""
        from registry.core.config import DeploymentMode

        mock_settings.deployment_mode = DeploymentMode.WITH_GATEWAY
        mock_server_service.get_server_versions.return_value = {
            "path": sample_versions["path"],
            "versions": [dict(v) for v in sample_versions["versions"]],
        }

        response = test_client_admin.get("/api/servers/test-server/versions")

        assert response.status_code == 200
        data = response.json()
        assert all("proxy_pass_url" in v for v in data["versions"])
