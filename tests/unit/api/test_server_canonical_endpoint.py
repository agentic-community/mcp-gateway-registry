"""Unit tests for GET /api/servers/{path}/server.json (canonical export endpoint).

Exercises the HTTP route added in #1187:
- returns the canonical server.json shape (and routes correctly past the
  catch-all /servers/{path} endpoint)
- 404 when the server is not registered
- 403 when a non-admin lacks access; admin bypasses the access check
- path normalization and greedy nested-path matching
- backend-URL redaction for non-admin callers in with-gateway mode, plus the
  cases where the URL is kept (registry-only mode, and admin callers)
- the X-Description-Truncated response header
- audit logging
"""

import logging
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

logger = logging.getLogger(__name__)

# Backend URL that must be redacted from non-admin, with-gateway responses.
BACKEND_URL = "http://internal-backend:8080/mcp"

URL = "/api/servers/calculator/server.json"


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def admin_user_context() -> dict[str, Any]:
    return {
        "username": "admin",
        "is_admin": True,
        "accessible_servers": ["all"],
        "auth_method": "session",
    }


@pytest.fixture
def regular_user_context() -> dict[str, Any]:
    return {
        "username": "testuser",
        "is_admin": False,
        "accessible_servers": ["calculator"],
        "auth_method": "session",
    }


@pytest.fixture
def canonical_server_info() -> dict[str, Any]:
    """Remote server, no preserved spec. to_canonical synthesizes remotes from
    proxy_pass_url, so both remotes[].url and the internal _meta carry the
    backend URL that redaction must strip."""
    return {
        "id": "srv-1",
        "server_name": "Calculator",
        "path": "/calculator",
        "description": "A calculator MCP server",
        "version": "1.2.0",
        "proxy_pass_url": BACKEND_URL,
        "deployment": "remote",
        "supported_transports": ["streamable-http"],
        "tags": ["math"],
        "num_tools": 1,
        "tool_list": [{"name": "add"}],
        "auth_scheme": "none",
        "metadata": {},
    }


@pytest.fixture
def local_server_info() -> dict[str, Any]:
    """Local stdio server, no spec. to_canonical synthesizes a packages[] entry."""
    return {
        "id": "srv-2",
        "server_name": "Local Calc",
        "path": "/local-calc",
        "description": "Local stdio MCP server",
        "version": "1.0.0",
        "deployment": "local",
        "supported_transports": ["stdio"],
        "local_runtime": {
            "type": "npx",
            "package": "@example/calc-mcp",
            "version": "1.0.0",
        },
        "metadata": {},
    }


@pytest.fixture
def mock_server_service():
    mock_service = MagicMock()
    mock_service.get_server_info = AsyncMock(return_value=None)
    mock_service.user_can_access_server_path = AsyncMock(return_value=True)
    return mock_service


@pytest.fixture
def _mock_auth_admin(admin_user_context, mock_settings):
    from registry.auth.dependencies import enhanced_auth, nginx_proxied_auth
    from registry.main import app

    app.dependency_overrides[enhanced_auth] = lambda: admin_user_context
    app.dependency_overrides[nginx_proxied_auth] = lambda: admin_user_context
    yield admin_user_context
    app.dependency_overrides.clear()


@pytest.fixture
def _mock_auth_regular(regular_user_context, mock_settings):
    from registry.auth.dependencies import enhanced_auth, nginx_proxied_auth
    from registry.main import app

    app.dependency_overrides[enhanced_auth] = lambda: regular_user_context
    app.dependency_overrides[nginx_proxied_auth] = lambda: regular_user_context
    yield regular_user_context
    app.dependency_overrides.clear()


def _create_test_client(mock_server_service, user_context):
    def mock_enhanced_auth_func(session=None):
        return user_context

    with (
        patch("registry.api.server_routes.server_service", mock_server_service),
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
def test_client_admin(mock_settings, mock_server_service, _mock_auth_admin, admin_user_context):
    yield from _create_test_client(mock_server_service, admin_user_context)


@pytest.fixture
def test_client_regular(
    mock_settings, mock_server_service, _mock_auth_regular, regular_user_context
):
    yield from _create_test_client(mock_server_service, regular_user_context)


# =============================================================================
# TESTS
# =============================================================================


class TestCanonicalEndpoint:
    def test_returns_canonical_shape(
        self, test_client_admin, mock_server_service, canonical_server_info
    ):
        """200 + canonical shape. The $schema field also confirms the request
        routed here and not to the catch-all /servers/{path}."""
        mock_server_service.get_server_info.return_value = canonical_server_info

        response = test_client_admin.get(URL)

        assert response.status_code == 200
        data = response.json()
        assert "$schema" in data
        assert {"name", "description", "version"}.issubset(data.keys())
        assert "_meta" in data

    def test_not_found(self, test_client_admin, mock_server_service):
        mock_server_service.get_server_info.return_value = None
        response = test_client_admin.get("/api/servers/missing/server.json")
        assert response.status_code == 404

    def test_forbidden_for_non_admin_without_access(
        self, test_client_regular, mock_server_service, canonical_server_info
    ):
        mock_server_service.get_server_info.return_value = canonical_server_info
        mock_server_service.user_can_access_server_path.return_value = False
        response = test_client_regular.get(URL)
        assert response.status_code == 403

    def test_admin_bypasses_access_check(
        self, test_client_admin, mock_server_service, canonical_server_info
    ):
        mock_server_service.get_server_info.return_value = canonical_server_info
        response = test_client_admin.get(URL)
        assert response.status_code == 200
        mock_server_service.user_can_access_server_path.assert_not_called()

    def test_path_normalized_with_leading_slash(
        self, test_client_admin, mock_server_service, canonical_server_info
    ):
        mock_server_service.get_server_info.return_value = canonical_server_info
        test_client_admin.get(URL)
        mock_server_service.get_server_info.assert_called_once_with("/calculator")

    def test_nested_path_matches(
        self, test_client_admin, mock_server_service, canonical_server_info
    ):
        """Greedy {path:path} must capture a nested path before the .json suffix."""
        mock_server_service.get_server_info.return_value = canonical_server_info
        response = test_client_admin.get("/api/servers/team/calculator/server.json")
        assert response.status_code == 200
        mock_server_service.get_server_info.assert_called_once_with("/team/calculator")

    def test_local_server_returns_packages(
        self, test_client_admin, mock_server_service, local_server_info
    ):
        mock_server_service.get_server_info.return_value = local_server_info
        response = test_client_admin.get("/api/servers/local-calc/server.json")
        assert response.status_code == 200
        data = response.json()
        assert "packages" in data
        assert "remotes" not in data

    def test_backend_url_redacted_for_non_admin_with_gateway(
        self, test_client_regular, mock_server_service, canonical_server_info
    ):
        from registry.core.config import DeploymentMode

        mock_server_service.get_server_info.return_value = dict(canonical_server_info)
        mock_server_service.user_can_access_server_path.return_value = True

        with patch(
            "registry.api.server_routes.settings.deployment_mode", DeploymentMode.WITH_GATEWAY
        ):
            response = test_client_regular.get(URL)

        assert response.status_code == 200
        assert BACKEND_URL not in response.text
        assert "url" not in response.json()["remotes"][0]

    def test_backend_url_kept_for_non_admin_registry_only(
        self, test_client_regular, mock_server_service, canonical_server_info
    ):
        from registry.core.config import DeploymentMode

        mock_server_service.get_server_info.return_value = dict(canonical_server_info)
        mock_server_service.user_can_access_server_path.return_value = True

        with patch(
            "registry.api.server_routes.settings.deployment_mode", DeploymentMode.REGISTRY_ONLY
        ):
            response = test_client_regular.get(URL)

        assert response.status_code == 200
        assert BACKEND_URL in response.text

    def test_backend_url_kept_for_admin_with_gateway(
        self, test_client_admin, mock_server_service, canonical_server_info
    ):
        from registry.core.config import DeploymentMode

        mock_server_service.get_server_info.return_value = dict(canonical_server_info)

        with patch(
            "registry.api.server_routes.settings.deployment_mode", DeploymentMode.WITH_GATEWAY
        ):
            response = test_client_admin.get(URL)

        assert response.status_code == 200
        assert BACKEND_URL in response.text

    def test_truncation_header_present_when_truncated(
        self, test_client_admin, mock_server_service, canonical_server_info
    ):
        mock_server_service.get_server_info.return_value = dict(
            canonical_server_info, description="x" * 150
        )
        response = test_client_admin.get(URL)
        assert response.status_code == 200
        assert response.headers.get("X-Description-Truncated") == "true"

    def test_truncation_header_absent_when_not_truncated(
        self, test_client_admin, mock_server_service, canonical_server_info
    ):
        mock_server_service.get_server_info.return_value = canonical_server_info
        response = test_client_admin.get(URL)
        assert response.status_code == 200
        assert "x-description-truncated" not in response.headers

    def test_audit_logged(self, test_client_admin, mock_server_service, canonical_server_info):
        mock_server_service.get_server_info.return_value = canonical_server_info
        with patch("registry.api.server_routes.set_audit_action") as mock_audit:
            response = test_client_admin.get(URL)
        assert response.status_code == 200
        mock_audit.assert_called_once()
        args = mock_audit.call_args[0]
        assert args[1] == "read"
        assert args[2] == "server"
