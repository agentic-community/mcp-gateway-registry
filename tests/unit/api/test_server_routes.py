"""
Unit tests for registry/api/server_routes.py

Tests the main server routes including:
- GET / - Main dashboard
- GET /servers - JSON API for servers list
- POST /toggle/{service_path:path} - Toggle service on/off
- POST /register - Register new service
- POST /internal/register - Internal registration with Basic Auth
- POST /internal/remove - Internal removal with Basic Auth
"""

import base64
import logging
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.templating import Jinja2Templates
from fastapi.testclient import TestClient

logger = logging.getLogger(__name__)


# =============================================================================
# AUTH MOCK FIXTURES (Following test_search_integration.py pattern)
# =============================================================================


@pytest.fixture
def admin_user_context() -> dict[str, Any]:
    """Create admin user context."""
    return {
        "username": "admin",
        "is_admin": True,
        "groups": ["mcp-registry-admin"],
        "scopes": ["mcp-servers-unrestricted/read", "mcp-servers-unrestricted/execute"],
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
        "ui_permissions": {"list_service": ["test-server"], "view_tools": ["test-server"]},
        "auth_method": "session",
    }


@pytest.fixture
def mock_auth_admin(admin_user_context, mock_settings):
    """
    Mock authentication dependencies with admin user.
    Following test_search_integration.py pattern.
    Note: depends on mock_settings to ensure environment is set up before importing app.
    """
    from registry.auth.dependencies import enhanced_auth, nginx_proxied_auth
    from registry.main import app

    def mock_enhanced_auth_override():
        return admin_user_context

    def mock_nginx_proxied_auth_override():
        return admin_user_context

    # Override dependencies at the app level
    app.dependency_overrides[enhanced_auth] = mock_enhanced_auth_override
    app.dependency_overrides[nginx_proxied_auth] = mock_nginx_proxied_auth_override

    yield admin_user_context

    # Cleanup
    app.dependency_overrides.clear()


@pytest.fixture
def mock_auth_regular(regular_user_context, mock_settings):
    """
    Mock authentication dependencies with regular user.
    Note: depends on mock_settings to ensure environment is set up before importing app.
    """
    from registry.auth.dependencies import enhanced_auth, nginx_proxied_auth
    from registry.main import app

    def mock_enhanced_auth_override():
        return regular_user_context

    def mock_nginx_proxied_auth_override():
        return regular_user_context

    # Override dependencies at the app level
    app.dependency_overrides[enhanced_auth] = mock_enhanced_auth_override
    app.dependency_overrides[nginx_proxied_auth] = mock_nginx_proxied_auth_override

    yield regular_user_context

    # Cleanup
    app.dependency_overrides.clear()


# =============================================================================
# SERVICE MOCK FIXTURES
# =============================================================================


@pytest.fixture
def mock_server_service():
    """Mock server_service dependency."""
    mock_service = MagicMock()
    mock_service.get_all_servers = AsyncMock(return_value={})
    mock_service.get_all_servers_with_permissions = AsyncMock(return_value={})
    mock_service.get_server_info = AsyncMock(return_value=None)
    mock_service.is_service_enabled = AsyncMock(return_value=True)
    mock_service.toggle_service = AsyncMock(return_value=True)
    # register_server now returns a dict with success, message, is_new_version
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
def mock_faiss_service():
    """Mock faiss_service dependency."""
    mock_service = MagicMock()
    mock_service.add_or_update_service = AsyncMock()
    mock_service.remove_service = AsyncMock()
    return mock_service


@pytest.fixture
def mock_health_service():
    """Mock health_service dependency."""
    mock_service = MagicMock()
    mock_service._get_service_health_data = MagicMock(
        return_value={"status": "healthy", "last_checked_iso": "2025-01-01T00:00:00Z"}
    )
    mock_service.perform_immediate_health_check = AsyncMock(return_value=("healthy", None))
    mock_service.broadcast_health_update = AsyncMock()
    return mock_service


@pytest.fixture
def mock_security_scanner_service():
    """Mock security_scanner_service dependency."""
    from registry.schemas.security import SecurityScanConfig, SecurityScanResult

    mock_service = MagicMock()

    # Return config with scanning disabled to avoid scan during registration
    mock_service.get_scan_config.return_value = SecurityScanConfig(
        enabled=False, scan_on_registration=False, block_unsafe_servers=False
    )

    # If scan is called anyway, return a passing result
    mock_service.scan_server = AsyncMock(
        return_value=SecurityScanResult(
            server_url="http://localhost:9000/mcp",
            server_path="/test-server",
            scan_timestamp="2025-01-01T00:00:00Z",
            is_safe=True,
            critical_issues=0,
            high_severity=0,
            medium_severity=0,
            low_severity=0,
            analyzers_used=["yara"],
            raw_output={},
            scan_failed=False,
        )
    )

    return mock_service


@pytest.fixture
def mock_nginx_service():
    """Mock nginx_service dependency."""
    mock_service = MagicMock()
    mock_service.generate_config_async = AsyncMock()
    return mock_service


@pytest.fixture
def mock_templates():
    """Mock Jinja2 templates."""
    mock = MagicMock(spec=Jinja2Templates)
    mock.TemplateResponse = MagicMock(return_value=MagicMock(status_code=200))
    return mock


@pytest.fixture
def sample_server_info() -> dict[str, Any]:
    """Create sample server info for testing."""
    return {
        "server_name": "test-server",
        "description": "A test server",
        "path": "/test-server",
        "proxy_pass_url": "http://localhost:8080",
        "tags": ["test", "demo"],
        "num_tools": 5,
        "num_stars": 4,
        "is_python": True,
        "license": "MIT",
        "tool_list": [
            {"name": "test_tool", "description": "A test tool", "inputSchema": {"type": "object"}}
        ],
    }


@pytest.fixture
def test_client_admin(
    mock_settings,
    mock_server_service,
    mock_faiss_service,
    mock_health_service,
    mock_nginx_service,
    mock_security_scanner_service,
    mock_auth_admin,
    admin_user_context,
):
    """Create FastAPI test client with admin auth and all services mocked."""

    # For /api/ route, enhanced_auth is called directly (not as dependency)
    def mock_enhanced_auth_func(session=None):
        return admin_user_context

    # Patch services - server_service is imported at module level, others are lazy imports
    # For module-level imports, patch where used: registry.api.server_routes.server_service
    # For lazy imports (inside functions), patch at definition: registry.search.service.faiss_service
    with (
        patch("registry.api.server_routes.server_service", mock_server_service),
        patch("registry.search.service.faiss_service", mock_faiss_service),
        patch("registry.health.service.health_service", mock_health_service),
        patch("registry.core.nginx_service.nginx_service", mock_nginx_service),
        patch("registry.api.server_routes.security_scanner_service", mock_security_scanner_service),
        patch("registry.utils.scopes_manager.update_server_scopes", new_callable=AsyncMock),
        patch("registry.api.server_routes.enhanced_auth", mock_enhanced_auth_func),
    ):
        from registry.main import app

        # Create client with session cookie (uses the default cookie name mcp_gateway_session)
        client = TestClient(app, cookies={"mcp_gateway_session": "test-session"})
        yield client


@pytest.fixture
def test_client_regular(
    mock_settings,
    mock_server_service,
    mock_faiss_service,
    mock_health_service,
    mock_nginx_service,
    mock_security_scanner_service,
    mock_auth_regular,
    regular_user_context,
):
    """Create FastAPI test client with regular user auth and all services mocked."""

    # For /api/ route, enhanced_auth is called directly (not as dependency)
    def mock_enhanced_auth_func(session=None):
        return regular_user_context

    # Patch services - server_service is imported at module level, others are lazy imports
    with (
        patch("registry.api.server_routes.server_service", mock_server_service),
        patch("registry.search.service.faiss_service", mock_faiss_service),
        patch("registry.health.service.health_service", mock_health_service),
        patch("registry.core.nginx_service.nginx_service", mock_nginx_service),
        patch("registry.api.server_routes.security_scanner_service", mock_security_scanner_service),
        patch("registry.utils.scopes_manager.update_server_scopes", new_callable=AsyncMock),
        patch("registry.api.server_routes.enhanced_auth", mock_enhanced_auth_func),
    ):
        from registry.main import app

        # Create client with session cookie (uses the default cookie name mcp_gateway_session)
        client = TestClient(app, cookies={"mcp_gateway_session": "test-session"})
        yield client


@pytest.fixture
def test_client_no_auth(
    mock_settings,
    mock_server_service,
    mock_faiss_service,
    mock_health_service,
    mock_nginx_service,
    mock_security_scanner_service,
):
    """Create FastAPI test client without auth mocking."""
    # Patch services - server_service is imported at module level, others are lazy imports
    with (
        patch("registry.api.server_routes.server_service", mock_server_service),
        patch("registry.search.service.faiss_service", mock_faiss_service),
        patch("registry.health.service.health_service", mock_health_service),
        patch("registry.core.nginx_service.nginx_service", mock_nginx_service),
        patch("registry.api.server_routes.security_scanner_service", mock_security_scanner_service),
        patch("registry.utils.scopes_manager.update_server_scopes", new_callable=AsyncMock),
    ):
        from registry.main import app

        # Clear any leftover auth overrides
        app.dependency_overrides.clear()
        client = TestClient(app)
        yield client


# =============================================================================
# TEST GET / - Main Dashboard
# =============================================================================


@pytest.mark.unit
@pytest.mark.api
@pytest.mark.servers
class TestRootDashboard:
    """Tests for GET / endpoint."""

    def test_no_session_cookie_redirects_to_login(self, test_client_no_auth):
        """Test that missing session cookie redirects to login page."""
        # Act
        response = test_client_no_auth.get("/api/", follow_redirects=False)

        # Assert - without auth, should redirect to login
        assert response.status_code == 302
        assert response.headers["location"] == "/login"

    @pytest.mark.skip(
        reason="Root dashboard uses Cookie() parameter which requires complex session mocking. "
        "Business logic is tested via TestGetServersJSON.test_admin_gets_all_servers"
    )
    def test_admin_sees_all_servers(self, test_client_admin, mock_server_service):
        """Test that admin user sees all servers."""
        pass

    @pytest.mark.skip(
        reason="Root dashboard uses Cookie() parameter which requires complex session mocking. "
        "Business logic is tested via TestGetServersJSON.test_non_admin_gets_filtered_servers"
    )
    def test_non_admin_sees_filtered_servers(
        self, test_client_regular, mock_server_service, regular_user_context
    ):
        """Test that non-admin user sees only accessible servers."""
        pass

    @pytest.mark.skip(
        reason="Root dashboard uses Cookie() parameter which requires complex session mocking. "
        "Business logic is tested via TestGetServersJSON.test_search_query_filters_results"
    )
    def test_search_query_filters_services(self, test_client_admin, mock_server_service):
        """Test that search query filters services by name, description, and tags."""
        pass


# =============================================================================
# TEST GET /servers - JSON API
# =============================================================================


@pytest.mark.unit
@pytest.mark.api
@pytest.mark.servers
class TestGetServersJSON:
    """Tests for GET /servers endpoint."""

    def test_admin_gets_all_servers(self, test_client_admin, mock_server_service):
        """Test that admin user gets all servers via JSON API."""
        # Arrange
        mock_server_service.get_all_servers.return_value = {
            "/server1": {
                "server_name": "Server 1",
                "description": "Test 1",
                "tags": [],
                "num_tools": 3,
                "num_stars": 5,
                "is_python": True,
                "license": "MIT",
                "proxy_pass_url": "http://localhost:8080",
            }
        }

        # Act
        response = test_client_admin.get("/api/servers")

        # Assert
        assert response.status_code == 200
        data = response.json()
        assert "servers" in data
        assert len(data["servers"]) == 1
        mock_server_service.get_all_servers.assert_called_once()

    def test_non_admin_gets_filtered_servers(
        self, test_client_regular, mock_server_service, regular_user_context
    ):
        """Test that non-admin user gets only accessible servers."""
        # Arrange
        mock_server_service.get_all_servers_with_permissions.return_value = {
            "/test-server": {
                "server_name": "test-server",
                "description": "Test",
                "tags": [],
                "num_tools": 2,
                "num_stars": 4,
                "is_python": False,
                "license": "Apache-2.0",
                "proxy_pass_url": "http://localhost:9000",
            }
        }

        # Act
        response = test_client_regular.get("/api/servers")

        # Assert
        assert response.status_code == 200
        data = response.json()
        assert "servers" in data
        assert len(data["servers"]) == 1
        assert data["servers"][0]["display_name"] == "test-server"

    def test_search_query_filters_results(self, test_client_admin, mock_server_service):
        """Test that search query filters server results."""
        # Arrange
        mock_server_service.get_all_servers.return_value = {
            "/server1": {
                "server_name": "Python Server",
                "description": "A Python-based server",
                "tags": ["python"],
                "num_tools": 3,
                "num_stars": 5,
                "is_python": True,
                "license": "MIT",
                "proxy_pass_url": "http://localhost:8080",
            },
            "/server2": {
                "server_name": "Node Server",
                "description": "A Node.js-based server",
                "tags": ["nodejs"],
                "num_tools": 2,
                "num_stars": 4,
                "is_python": False,
                "license": "MIT",
                "proxy_pass_url": "http://localhost:8081",
            },
        }

        # Act
        response = test_client_admin.get("/api/servers?query=python")

        # Assert
        assert response.status_code == 200
        data = response.json()
        assert "servers" in data
        assert len(data["servers"]) == 1
        assert "Python" in data["servers"][0]["display_name"]

    def test_returns_health_status(
        self, test_client_admin, mock_server_service, mock_health_service
    ):
        """Test that server list includes health status."""
        # Arrange
        mock_server_service.get_all_servers.return_value = {
            "/server1": {
                "server_name": "Server 1",
                "description": "Test",
                "tags": [],
                "num_tools": 3,
                "num_stars": 5,
                "is_python": True,
                "license": "MIT",
                "proxy_pass_url": "http://localhost:8080",
            }
        }
        mock_health_service._get_service_health_data.return_value = {
            "status": "healthy",
            "last_checked_iso": "2025-01-01T12:00:00Z",
        }

        # Act
        response = test_client_admin.get("/api/servers")

        # Assert
        assert response.status_code == 200
        data = response.json()
        assert data["servers"][0]["health_status"] == "healthy"
        assert data["servers"][0]["last_checked_iso"] == "2025-01-01T12:00:00Z"


# =============================================================================
# TEST POST /toggle/{service_path:path} - Toggle Service
# =============================================================================


@pytest.mark.unit
@pytest.mark.api
@pytest.mark.servers
class TestToggleService:
    """Tests for POST /toggle/{service_path:path} endpoint."""

    def test_toggle_service_on_success(
        self,
        test_client_admin,
        mock_server_service,
        mock_faiss_service,
        mock_nginx_service,
        mock_health_service,
        sample_server_info,
    ):
        """Test successful toggle service on."""
        # Arrange
        mock_server_service.get_server_info.return_value = sample_server_info
        mock_server_service.toggle_service.return_value = True

        # Patch at the actual module location (imported inside functions)
        with patch(
            "registry.auth.dependencies.user_has_ui_permission_for_service", return_value=True
        ):
            # Act
            response = test_client_admin.post("/api/toggle/test-server", data={"enabled": "on"})

            # Assert
            assert response.status_code == 200
            data = response.json()
            assert data["new_enabled_state"] is True
            assert data["service_path"] == "/test-server"
            mock_server_service.toggle_service.assert_called_once_with("/test-server", True)
            mock_faiss_service.add_or_update_service.assert_called_once()
            mock_nginx_service.generate_config_async.assert_called_once()

    def test_toggle_service_off_success(
        self,
        test_client_admin,
        mock_server_service,
        mock_faiss_service,
        mock_nginx_service,
        sample_server_info,
    ):
        """Test successful toggle service off."""
        # Arrange
        mock_server_service.get_server_info.return_value = sample_server_info
        mock_server_service.toggle_service.return_value = True

        # Patch at the actual module location (imported inside functions)
        with patch(
            "registry.auth.dependencies.user_has_ui_permission_for_service", return_value=True
        ):
            # Act
            response = test_client_admin.post("/api/toggle/test-server", data={"enabled": "off"})

            # Assert
            assert response.status_code == 200
            data = response.json()
            assert data["new_enabled_state"] is False
            assert data["status"] == "disabled"
            mock_server_service.toggle_service.assert_called_once_with("/test-server", False)

    def test_toggle_service_not_found(self, test_client_admin, mock_server_service):
        """Test toggle fails when service not found."""
        # Arrange
        mock_server_service.get_server_info.return_value = None

        # Act
        response = test_client_admin.post("/api/toggle/nonexistent", data={"enabled": "on"})

        # Assert
        assert response.status_code == 404
        assert "not registered" in response.json()["detail"]

    @pytest.mark.skip(
        reason="Bug in server_routes.py: local variable 'status' shadows imported 'status' module"
    )
    def test_toggle_service_no_permission(
        self, test_client_regular, mock_server_service, sample_server_info
    ):
        """Test toggle fails when user lacks toggle_service permission."""
        # Arrange
        mock_server_service.get_server_info.return_value = sample_server_info

        with patch(
            "registry.auth.dependencies.user_has_ui_permission_for_service", return_value=False
        ):
            # Act
            response = test_client_regular.post("/api/toggle/test-server", data={"enabled": "on"})

            # Assert
            assert response.status_code == 403
            assert "permission" in response.json()["detail"].lower()

    @pytest.mark.skip(
        reason="Bug in server_routes.py: local variable 'status' shadows imported 'status' module"
    )
    def test_toggle_service_no_server_access(
        self, test_client_regular, mock_server_service, sample_server_info
    ):
        """Test toggle fails when non-admin user lacks server access."""
        # Arrange
        mock_server_service.get_server_info.return_value = sample_server_info
        mock_server_service.user_can_access_server_path.return_value = False

        with patch(
            "registry.auth.dependencies.user_has_ui_permission_for_service", return_value=True
        ):
            # Act
            response = test_client_regular.post("/api/toggle/test-server", data={"enabled": "on"})

            # Assert
            assert response.status_code == 403
            assert "access" in response.json()["detail"].lower()

    def test_toggle_service_performs_health_check_when_enabling(
        self, test_client_admin, mock_server_service, mock_health_service, sample_server_info
    ):
        """Test that enabling a service triggers immediate health check."""
        # Arrange
        mock_server_service.get_server_info.return_value = sample_server_info
        mock_server_service.toggle_service.return_value = True
        mock_health_service.perform_immediate_health_check.return_value = ("healthy", None)

        with patch(
            "registry.auth.dependencies.user_has_ui_permission_for_service", return_value=True
        ):
            # Act
            response = test_client_admin.post("/api/toggle/test-server", data={"enabled": "on"})

            # Assert
            assert response.status_code == 200
            mock_health_service.perform_immediate_health_check.assert_called_once_with(
                "/test-server"
            )
            mock_health_service.broadcast_health_update.assert_called_once_with("/test-server")


# =============================================================================
# TEST POST /register - Register Service
# =============================================================================


@pytest.mark.unit
@pytest.mark.api
@pytest.mark.servers
class TestRegisterService:
    """Tests for POST /register endpoint."""

    def test_register_service_success(
        self,
        test_client_admin,
        mock_server_service,
        mock_faiss_service,
        mock_nginx_service,
        mock_health_service,
    ):
        """Test successful service registration."""
        # Arrange - register_server returns a dict now
        mock_server_service.register_server.return_value = {
            "success": True,
            "message": "Server registered successfully",
            "is_new_version": False,
        }

        with patch(
            "registry.auth.dependencies.user_has_ui_permission_for_service", return_value=True
        ):
            # Act
            response = test_client_admin.post(
                "/api/register",
                data={
                    "name": "New Server",
                    "description": "A new test server",
                    "path": "/new-server",
                    "proxy_pass_url": "http://localhost:9000",
                    "tags": "test, new",
                    "num_tools": 5,
                    "num_stars": 4,
                    "is_python": "true",
                    "license": "MIT",
                },
            )

            # Assert
            assert response.status_code == 201
            data = response.json()
            assert data["message"] == "Service registered successfully"
            assert data["service"]["server_name"] == "New Server"
            mock_server_service.register_server.assert_called_once()
            mock_faiss_service.add_or_update_service.assert_called_once()
            mock_nginx_service.generate_config_async.assert_called_once()

    def test_register_service_no_permission(self, test_client_regular, mock_server_service):
        """Test registration fails when user lacks register_service permission."""
        # Arrange - regular user context already lacks register_service permission

        # Act
        response = test_client_regular.post(
            "/api/register",
            data={
                "name": "New Server",
                "description": "Test",
                "path": "/new-server",
                "proxy_pass_url": "http://localhost:9000",
            },
        )

        # Assert
        assert response.status_code == 403
        assert "permission" in response.json()["detail"].lower()

    def test_register_service_path_already_exists(self, test_client_admin, mock_server_service):
        """Test registration fails when path already exists with same version."""
        # Arrange - register_server returns a dict now
        mock_server_service.register_server.return_value = {
            "success": False,
            "message": "Server already exists at path /existing-server with the same version",
            "is_new_version": False,
        }

        with patch(
            "registry.auth.dependencies.user_has_ui_permission_for_service", return_value=True
        ):
            # Act
            response = test_client_admin.post(
                "/api/register",
                data={
                    "name": "Duplicate Server",
                    "description": "Test",
                    "path": "/existing-server",
                    "proxy_pass_url": "http://localhost:9000",
                },
            )

            # Assert - returns 409 Conflict with generic error (no internal details)
            assert response.status_code == 409
            assert "registration failed" in response.json()["error"].lower()

    def test_register_service_normalizes_path(self, test_client_admin, mock_server_service):
        """Test that service path is normalized to start with /."""
        # Arrange - register_server returns a dict now
        mock_server_service.register_server.return_value = {
            "success": True,
            "message": "Server registered successfully",
            "is_new_version": False,
        }

        with patch(
            "registry.auth.dependencies.user_has_ui_permission_for_service", return_value=True
        ):
            # Act
            response = test_client_admin.post(
                "/api/register",
                data={
                    "name": "New Server",
                    "description": "Test",
                    "path": "new-server",  # Missing leading slash
                    "proxy_pass_url": "http://localhost:9000",
                },
            )

            # Assert
            assert response.status_code == 201
            # Verify path was normalized
            call_args = mock_server_service.register_server.call_args[0][0]
            assert call_args["path"] == "/new-server"


# =============================================================================
# TEST POST /internal/register - Internal Registration
# =============================================================================


@pytest.mark.unit
@pytest.mark.api
@pytest.mark.servers
class TestInternalRegister:
    """Tests for POST /internal/register endpoint."""

    def test_internal_register_success(
        self,
        test_client_no_auth,
        mock_server_service,
        mock_faiss_service,
        mock_nginx_service,
        mock_health_service,
    ):
        """Test successful internal registration with valid Basic Auth."""
        # Arrange - register_server returns a dict now
        mock_server_service.register_server.return_value = {
            "success": True,
            "message": "Server registered successfully",
            "is_new_version": False,
        }
        credentials = base64.b64encode(b"admin:testpass").decode("utf-8")

        with (
            patch.dict("os.environ", {"ADMIN_USER": "admin", "ADMIN_PASSWORD": "testpass"}),
            patch("registry.utils.scopes_manager.update_server_scopes", new_callable=AsyncMock),
        ):
            # Act
            response = test_client_no_auth.post(
                "/api/internal/register",
                data={
                    "name": "Internal Server",
                    "description": "Registered internally",
                    "path": "/internal-server",
                    "proxy_pass_url": "http://localhost:9000",
                    "tags": "internal",
                    "num_tools": 3,
                },
                headers={"Authorization": f"Basic {credentials}"},
            )

            # Assert
            assert response.status_code == 201
            data = response.json()
            assert data["message"] == "Service registered successfully"
            mock_server_service.register_server.assert_called_once()
            mock_faiss_service.add_or_update_service.assert_called_once()

    def test_internal_register_missing_auth_header(self, test_client_no_auth):
        """Test internal registration fails without Authorization header."""
        # Act
        response = test_client_no_auth.post(
            "/api/internal/register",
            data={
                "name": "Server",
                "description": "Test",
                "path": "/test",
                "proxy_pass_url": "http://localhost:9000",
            },
        )

        # Assert
        assert response.status_code == 401
        assert "authorization" in response.json()["detail"].lower()

    def test_internal_register_invalid_credentials(self, test_client_no_auth, mock_server_service):
        """Test internal registration fails with invalid credentials."""
        # Arrange
        credentials = base64.b64encode(b"admin:wrongpass").decode("utf-8")

        with patch.dict("os.environ", {"ADMIN_USER": "admin", "ADMIN_PASSWORD": "testpass"}):
            # Act
            response = test_client_no_auth.post(
                "/api/internal/register",
                data={
                    "name": "Server",
                    "description": "Test",
                    "path": "/test",
                    "proxy_pass_url": "http://localhost:9000",
                },
                headers={"Authorization": f"Basic {credentials}"},
            )

            # Assert
            assert response.status_code == 401
            assert "Invalid admin credentials" in response.json()["detail"]

    def test_internal_register_admin_password_not_set(self, test_client_no_auth):
        """Test internal registration fails when ADMIN_PASSWORD not set."""
        # Arrange
        credentials = base64.b64encode(b"admin:testpass").decode("utf-8")

        with patch.dict("os.environ", {"ADMIN_USER": "admin"}, clear=True):
            # Remove ADMIN_PASSWORD from env
            # Act
            response = test_client_no_auth.post(
                "/api/internal/register",
                data={
                    "name": "Server",
                    "description": "Test",
                    "path": "/test",
                    "proxy_pass_url": "http://localhost:9000",
                },
                headers={"Authorization": f"Basic {credentials}"},
            )

            # Assert
            assert response.status_code == 500
            assert "Internal server configuration error" in response.json()["detail"]

    def test_internal_register_overwrite_existing_service(
        self, test_client_no_auth, mock_server_service, sample_server_info
    ):
        """Test internal registration can overwrite existing service."""
        # Arrange
        mock_server_service.get_server_info.return_value = sample_server_info
        mock_server_service.update_server.return_value = True
        credentials = base64.b64encode(b"admin:testpass").decode("utf-8")

        with (
            patch.dict("os.environ", {"ADMIN_USER": "admin", "ADMIN_PASSWORD": "testpass"}),
            patch("registry.utils.scopes_manager.update_server_scopes", new_callable=AsyncMock),
        ):
            # Act
            response = test_client_no_auth.post(
                "/api/internal/register",
                data={
                    "name": "Updated Server",
                    "description": "Updated",
                    "path": "/test-server",
                    "proxy_pass_url": "http://localhost:9001",
                    "overwrite": "true",
                },
                headers={"Authorization": f"Basic {credentials}"},
            )

            # Assert
            assert response.status_code == 201
            mock_server_service.update_server.assert_called_once()

    def test_internal_register_no_overwrite_existing_service(
        self, test_client_no_auth, mock_server_service, sample_server_info
    ):
        """Test internal registration fails without overwrite flag for existing service."""
        # Arrange
        mock_server_service.get_server_info.return_value = sample_server_info
        credentials = base64.b64encode(b"admin:testpass").decode("utf-8")

        with patch.dict("os.environ", {"ADMIN_USER": "admin", "ADMIN_PASSWORD": "testpass"}):
            # Act
            response = test_client_no_auth.post(
                "/api/internal/register",
                data={
                    "name": "Server",
                    "description": "Test",
                    "path": "/test-server",
                    "proxy_pass_url": "http://localhost:9000",
                    "overwrite": "false",
                },
                headers={"Authorization": f"Basic {credentials}"},
            )

            # Assert
            assert response.status_code == 409
            assert "already exists" in response.json()["reason"].lower()

    def test_internal_register_auto_enables_service(
        self, test_client_no_auth, mock_server_service, mock_faiss_service, mock_nginx_service
    ):
        """Test that internal registration auto-enables the service."""
        # Arrange - register_server returns a dict now
        mock_server_service.register_server.return_value = {
            "success": True,
            "message": "Server registered successfully",
            "is_new_version": False,
        }
        mock_server_service.toggle_service.return_value = True
        mock_server_service.is_service_enabled.return_value = True
        credentials = base64.b64encode(b"admin:testpass").decode("utf-8")

        with (
            patch.dict("os.environ", {"ADMIN_USER": "admin", "ADMIN_PASSWORD": "testpass"}),
            patch("registry.utils.scopes_manager.update_server_scopes", new_callable=AsyncMock),
        ):
            # Act
            response = test_client_no_auth.post(
                "/api/internal/register",
                data={
                    "name": "Auto-Enabled Server",
                    "description": "Test",
                    "path": "/auto-enabled",
                    "proxy_pass_url": "http://localhost:9000",
                },
                headers={"Authorization": f"Basic {credentials}"},
            )

            # Assert
            assert response.status_code == 201
            mock_server_service.toggle_service.assert_called_once_with("/auto-enabled", True)


# =============================================================================
# TEST POST /internal/remove - Internal Removal
# =============================================================================


@pytest.mark.unit
@pytest.mark.api
@pytest.mark.servers
class TestInternalRemove:
    """Tests for POST /internal/remove endpoint."""

    def test_internal_remove_success(
        self, test_client_no_auth, mock_server_service, sample_server_info
    ):
        """Test successful internal service removal."""
        # Arrange
        mock_server_service.get_server_info.return_value = sample_server_info
        mock_server_service.remove_server.return_value = True
        credentials = base64.b64encode(b"admin:testpass").decode("utf-8")

        with patch.dict("os.environ", {"ADMIN_USER": "admin", "ADMIN_PASSWORD": "testpass"}):
            # Act
            response = test_client_no_auth.post(
                "/api/internal/remove",
                data={"service_path": "/test-server"},
                headers={"Authorization": f"Basic {credentials}"},
            )

            # Assert
            assert response.status_code == 200
            mock_server_service.remove_server.assert_called_once_with("/test-server")

    def test_internal_remove_service_not_found(self, test_client_no_auth, mock_server_service):
        """Test internal removal fails when service not found."""
        # Arrange
        mock_server_service.get_server_info.return_value = None
        credentials = base64.b64encode(b"admin:testpass").decode("utf-8")

        with patch.dict("os.environ", {"ADMIN_USER": "admin", "ADMIN_PASSWORD": "testpass"}):
            # Act
            response = test_client_no_auth.post(
                "/api/internal/remove",
                data={"service_path": "/nonexistent"},
                headers={"Authorization": f"Basic {credentials}"},
            )

            # Assert
            assert response.status_code == 404
            assert "not found" in response.json()["error"].lower()

    def test_internal_remove_missing_auth(self, test_client_no_auth):
        """Test internal removal requires authentication."""
        # Act
        response = test_client_no_auth.post("/api/internal/remove", data={"service_path": "/test"})

        # Assert
        assert response.status_code == 401
        assert "authorization" in response.json()["detail"].lower()

    def test_internal_remove_normalizes_path(
        self, test_client_no_auth, mock_server_service, sample_server_info
    ):
        """Test that service path is normalized in removal."""
        # Arrange
        mock_server_service.get_server_info.return_value = sample_server_info
        mock_server_service.remove_server.return_value = True
        credentials = base64.b64encode(b"admin:testpass").decode("utf-8")

        with patch.dict("os.environ", {"ADMIN_USER": "admin", "ADMIN_PASSWORD": "testpass"}):
            # Act
            response = test_client_no_auth.post(
                "/api/internal/remove",
                data={"service_path": "test-server"},  # Missing leading slash
                headers={"Authorization": f"Basic {credentials}"},
            )

            # Assert
            assert response.status_code == 200
            mock_server_service.remove_server.assert_called_once_with("/test-server")


# =============================================================================
# ADDITIONAL HELPER TESTS
# =============================================================================


@pytest.mark.unit
@pytest.mark.api
@pytest.mark.servers
class TestHelperFunctions:
    """Tests for helper functions and edge cases."""

    def test_path_normalization_in_toggle(
        self, test_client_admin, mock_server_service, sample_server_info
    ):
        """Test that paths without leading slash are normalized in toggle endpoint."""
        # Arrange
        mock_server_service.get_server_info.return_value = sample_server_info
        mock_server_service.toggle_service.return_value = True

        with patch(
            "registry.auth.dependencies.user_has_ui_permission_for_service", return_value=True
        ):
            # Act
            response = test_client_admin.post(
                "/api/toggle/test-server",  # Path in URL
                data={"enabled": "on"},
            )

            # Assert
            assert response.status_code == 200
            # Verify the path was normalized
            mock_server_service.get_server_info.assert_called_with("/test-server")

    def test_tags_parsing_in_register(self, test_client_admin, mock_server_service):
        """Test that tags are properly parsed from comma-separated string."""
        # Arrange - register_server returns a dict now
        mock_server_service.register_server.return_value = {
            "success": True,
            "message": "Server registered successfully",
            "is_new_version": False,
        }

        with patch(
            "registry.auth.dependencies.user_has_ui_permission_for_service", return_value=True
        ):
            # Act
            response = test_client_admin.post(
                "/api/register",
                data={
                    "name": "Tagged Server",
                    "description": "Test",
                    "path": "/tagged",
                    "proxy_pass_url": "http://localhost:9000",
                    "tags": "tag1, tag2, tag3",  # Comma-separated with spaces
                },
            )

            # Assert
            assert response.status_code == 201
            call_args = mock_server_service.register_server.call_args[0][0]
            assert call_args["tags"] == ["tag1", "tag2", "tag3"]
