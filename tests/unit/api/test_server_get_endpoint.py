"""
Unit tests for GET /api/servers/{path} endpoint.

Tests the single server retrieval endpoint including:
- Successful retrieval for admin and regular users
- 404 when server not found
- 403 when user lacks access
- Path normalization (with/without leading slash)
- Credentials are never in the response
- proxy_pass_url stripping behavior based on deployment mode
- Audit logging
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
        "scopes": ["mcp-servers-unrestricted/read"],
        "accessible_servers": ["all"],
        "accessible_services": ["all"],
        "accessible_agents": ["all"],
        "ui_permissions": {
            "list_service": ["all"],
            "toggle_service": ["all"],
            "register_service": ["all"],
            "view_tools": ["all"],
            "refresh_service": ["all"],
            "modify_service": ["all"],
        },
        "auth_method": "session",
    }


@pytest.fixture
def regular_user_context() -> dict[str, Any]:
    """Create regular (non-admin) user context."""
    return {
        "username": "testuser",
        "is_admin": False,
        "groups": ["test-group"],
        "scopes": ["test-server/read"],
        "accessible_servers": ["test-server"],
        "accessible_services": ["test-server"],
        "accessible_agents": ["test-agent"],
        "ui_permissions": {
            "list_service": ["test-server"],
            "view_tools": ["test-server"],
        },
        "auth_method": "session",
    }


@pytest.fixture
def sample_server_info() -> dict[str, Any]:
    """Create sample server info dict as returned by server_service.get_server_info()."""
    return {
        "server_name": "Test Server",
        "description": "A test MCP server",
        "path": "/test-server",
        "proxy_pass_url": "http://internal-backend:8080",
        "tags": ["test", "demo"],
        "num_tools": 2,
        "tool_list": [
            {
                "name": "get_weather",
                "description": "Get weather data",
                "inputSchema": {"type": "object"},
            },
            {
                "name": "search_docs",
                "description": "Search documents",
                "inputSchema": {"type": "object"},
            },
        ],
        "is_enabled": True,
        "health_status": "healthy",
        "transport": "sse",
        "supported_transports": ["sse", "streamable-http"],
        "version": "v1.0.0",
        "versions": [{"version": "v1.0.0", "status": "active", "is_default": True}],
        "license": "Apache-2.0",
        "registered_at": "2026-04-01T00:00:00Z",
        "registered_by": "admin",
    }


@pytest.fixture
def mock_server_service():
    """Mock server_service dependency."""
    mock_service = MagicMock()
    mock_service.get_server_info = AsyncMock(return_value=None)
    mock_service.get_all_servers = AsyncMock(return_value={})
    mock_service.get_all_servers_with_permissions = AsyncMock(return_value={})
    mock_service.is_service_enabled = AsyncMock(return_value=True)
    mock_service.toggle_service = AsyncMock(return_value=True)
    mock_service.register_server = AsyncMock(
        return_value={
            "success": True,
            "message": "Server registered successfully",
            "is_new_version": False,
        }
    )
    mock_service.update_server = AsyncMock(return_value=True)
    mock_service.remove_server = AsyncMock(return_value=True)
    mock_service.get_enabled_services = AsyncMock(return_value=[])
    mock_service.user_can_access_server_path = AsyncMock(return_value=True)
    return mock_service


@pytest.fixture
def _mock_auth_admin(admin_user_context, mock_settings):
    """Mock authentication dependencies with admin user."""
    from registry.auth.dependencies import (
        enhanced_auth,
        nginx_proxied_auth,
    )
    from registry.main import app

    def mock_enhanced_auth_override():
        return admin_user_context

    def mock_nginx_proxied_auth_override():
        return admin_user_context

    app.dependency_overrides[enhanced_auth] = mock_enhanced_auth_override
    app.dependency_overrides[nginx_proxied_auth] = mock_nginx_proxied_auth_override

    yield admin_user_context

    app.dependency_overrides.clear()


@pytest.fixture
def _mock_auth_regular(regular_user_context, mock_settings):
    """Mock authentication dependencies with regular user."""
    from registry.auth.dependencies import (
        enhanced_auth,
        nginx_proxied_auth,
    )
    from registry.main import app

    def mock_enhanced_auth_override():
        return regular_user_context

    def mock_nginx_proxied_auth_override():
        return regular_user_context

    app.dependency_overrides[enhanced_auth] = mock_enhanced_auth_override
    app.dependency_overrides[nginx_proxied_auth] = mock_nginx_proxied_auth_override

    yield regular_user_context

    app.dependency_overrides.clear()


def _create_test_client(
    mock_server_service: MagicMock,
    user_context: dict[str, Any],
) -> TestClient:
    """Create a FastAPI test client with mocked services.

    Args:
        mock_server_service: Mocked server service
        user_context: User context for auth

    Returns:
        TestClient instance
    """

    def mock_enhanced_auth_func(session=None):
        return user_context

    with (
        patch("registry.api.server_routes.server_service", mock_server_service),
        patch("registry.search.service.faiss_service", MagicMock()),
        patch("registry.health.service.health_service", MagicMock()),
        patch("registry.core.nginx_service.nginx_service", MagicMock()),
        patch("registry.api.server_routes.security_scanner_service", MagicMock()),
        patch("registry.utils.scopes_manager.update_server_scopes", new_callable=AsyncMock),
        patch("registry.api.server_routes.enhanced_auth", mock_enhanced_auth_func),
    ):
        from registry.auth.csrf import verify_csrf_token_flexible
        from registry.main import app

        app.dependency_overrides[verify_csrf_token_flexible] = lambda: None

        client = TestClient(app, cookies={"mcp_gateway_session": "test-session"})
        yield client

        app.dependency_overrides.pop(verify_csrf_token_flexible, None)


@pytest.fixture
def test_client_admin(
    mock_settings,
    mock_server_service,
    _mock_auth_admin,
    admin_user_context,
):
    """Create test client with admin auth."""
    yield from _create_test_client(mock_server_service, admin_user_context)


@pytest.fixture
def test_client_regular(
    mock_settings,
    mock_server_service,
    _mock_auth_regular,
    regular_user_context,
):
    """Create test client with regular user auth."""
    yield from _create_test_client(mock_server_service, regular_user_context)


# =============================================================================
# TESTS: GET /api/servers/{path}
# =============================================================================


class TestGetServer:
    """Tests for GET /api/servers/{path} endpoint."""

    def test_get_server_success_admin(
        self,
        test_client_admin,
        mock_server_service,
        sample_server_info,
    ):
        """Test successful server retrieval as admin."""
        mock_server_service.get_server_info.return_value = sample_server_info

        response = test_client_admin.get("/api/servers/test-server")

        assert response.status_code == 200
        data = response.json()
        assert data["server_name"] == "Test Server"
        assert data["path"] == "/test-server"
        assert data["description"] == "A test MCP server"
        assert data["num_tools"] == 2
        assert len(data["tool_list"]) == 2
        assert data["is_enabled"] is True

    def test_get_server_success_regular_user(
        self,
        test_client_regular,
        mock_server_service,
        sample_server_info,
    ):
        """Test successful server retrieval as regular user."""
        mock_server_service.get_server_info.return_value = sample_server_info
        mock_server_service.user_can_access_server_path.return_value = True

        response = test_client_regular.get("/api/servers/test-server")

        assert response.status_code == 200
        data = response.json()
        assert data["server_name"] == "Test Server"

    def test_get_server_not_found(
        self,
        test_client_admin,
        mock_server_service,
    ):
        """Test 404 when server does not exist."""
        mock_server_service.get_server_info.return_value = None

        response = test_client_admin.get("/api/servers/nonexistent-server")

        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()

    def test_get_server_forbidden(
        self,
        test_client_regular,
        mock_server_service,
        sample_server_info,
    ):
        """Test 403 when user lacks access to the server."""
        mock_server_service.get_server_info.return_value = sample_server_info
        mock_server_service.user_can_access_server_path.return_value = False

        response = test_client_regular.get("/api/servers/test-server")

        assert response.status_code == 403
        assert "access" in response.json()["detail"].lower()

    def test_get_server_admin_bypasses_access_check(
        self,
        test_client_admin,
        mock_server_service,
        sample_server_info,
    ):
        """Test that admin users bypass the access control check."""
        mock_server_service.get_server_info.return_value = sample_server_info

        response = test_client_admin.get("/api/servers/test-server")

        assert response.status_code == 200
        # user_can_access_server_path should NOT be called for admin
        mock_server_service.user_can_access_server_path.assert_not_called()

    def test_get_server_path_normalization_no_slash(
        self,
        test_client_admin,
        mock_server_service,
        sample_server_info,
    ):
        """Test that paths without leading slash are normalized."""
        mock_server_service.get_server_info.return_value = sample_server_info

        response = test_client_admin.get("/api/servers/my-server")

        assert response.status_code == 200
        # Verify get_server_info was called with normalized path (leading slash)
        mock_server_service.get_server_info.assert_called_once_with("/my-server")

    def test_get_server_credentials_stripped(
        self,
        test_client_admin,
        mock_server_service,
    ):
        """Test that credentials are never included in the response."""
        server_info = {
            "server_name": "Test Server",
            "path": "/test-server",
            "description": "Test",
            "is_enabled": True,
        }
        mock_server_service.get_server_info.return_value = server_info

        response = test_client_admin.get("/api/servers/test-server")

        assert response.status_code == 200
        data = response.json()
        assert "auth_credential_encrypted" not in data
        assert "auth_credential" not in data

    def test_get_server_includes_tools(
        self,
        test_client_admin,
        mock_server_service,
        sample_server_info,
    ):
        """Test that the response includes tool_list."""
        mock_server_service.get_server_info.return_value = sample_server_info

        response = test_client_admin.get("/api/servers/test-server")

        assert response.status_code == 200
        data = response.json()
        assert "tool_list" in data
        assert len(data["tool_list"]) == 2
        assert data["tool_list"][0]["name"] == "get_weather"

    def test_get_server_includes_versions(
        self,
        test_client_admin,
        mock_server_service,
        sample_server_info,
    ):
        """Test that the response includes versions for multi-version servers."""
        mock_server_service.get_server_info.return_value = sample_server_info

        response = test_client_admin.get("/api/servers/test-server")

        assert response.status_code == 200
        data = response.json()
        assert "versions" in data
        assert len(data["versions"]) == 1
        assert data["versions"][0]["version"] == "v1.0.0"

    def test_get_server_proxy_pass_url_stripped_for_non_admin_with_gateway(
        self,
        test_client_regular,
        mock_server_service,
        mock_settings,
        sample_server_info,
    ):
        """Test proxy_pass_url is stripped for non-admin users in with-gateway mode."""
        from registry.core.config import DeploymentMode

        mock_settings.deployment_mode = DeploymentMode.WITH_GATEWAY
        # Use a copy so dict.pop in the endpoint doesn't affect other tests
        mock_server_service.get_server_info.return_value = dict(sample_server_info)
        mock_server_service.user_can_access_server_path.return_value = True

        response = test_client_regular.get("/api/servers/test-server")

        assert response.status_code == 200
        data = response.json()
        assert "proxy_pass_url" not in data

    def test_get_server_proxy_pass_url_kept_for_non_admin_registry_only(
        self,
        test_client_regular,
        mock_server_service,
        sample_server_info,
    ):
        """Test proxy_pass_url is kept for non-admin users in registry-only mode."""
        from registry.core.config import DeploymentMode

        # Use a copy so dict.pop in the endpoint doesn't affect other tests
        mock_server_service.get_server_info.return_value = dict(sample_server_info)
        mock_server_service.user_can_access_server_path.return_value = True

        # Patch deployment_mode at the module level where the endpoint reads it
        with patch(
            "registry.api.server_routes.settings.deployment_mode",
            DeploymentMode.REGISTRY_ONLY,
        ):
            response = test_client_regular.get("/api/servers/test-server")

        assert response.status_code == 200
        data = response.json()
        assert "proxy_pass_url" in data
        assert data["proxy_pass_url"] == "http://internal-backend:8080"

    def test_get_server_proxy_pass_url_kept_for_admin(
        self,
        test_client_admin,
        mock_server_service,
        sample_server_info,
    ):
        """Test proxy_pass_url is always kept for admin users."""
        mock_server_service.get_server_info.return_value = sample_server_info

        response = test_client_admin.get("/api/servers/test-server")

        assert response.status_code == 200
        data = response.json()
        assert "proxy_pass_url" in data
        assert data["proxy_pass_url"] == "http://internal-backend:8080"

    def test_get_server_audit_logged(
        self,
        test_client_admin,
        mock_server_service,
        sample_server_info,
    ):
        """Test that the read action is audit logged."""
        mock_server_service.get_server_info.return_value = sample_server_info

        with patch("registry.api.server_routes.set_audit_action") as mock_audit:
            response = test_client_admin.get("/api/servers/test-server")

            assert response.status_code == 200
            mock_audit.assert_called_once()
            call_args = mock_audit.call_args
            assert call_args[0][1] == "read"
            assert call_args[0][2] == "server"
