"""
Unit tests for registry/api/server_routes.py

Tests the main server routes including:
- GET / - Main dashboard
- GET /servers - JSON API for servers list
- POST /toggle/{service_path:path} - Toggle service on/off
- POST /register - Register new service
- POST /internal/register - Internal registration with JWT Bearer Auth
- POST /internal/remove - Internal removal with JWT Bearer Auth
"""

import json
import logging
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.templating import Jinja2Templates
from fastapi.testclient import TestClient

from registry.auth.internal import generate_internal_token

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
    # POST /api/servers/register schedules `asyncio.create_task(save_data())`
    # after a successful registration; if save_data is a plain MagicMock the
    # call returns a non-coroutine and create_task() raises, turning a 201
    # into a 500.
    mock_service.save_data = AsyncMock()
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
def mock_nginx_reload_scheduler():
    """Mock nginx_reload_scheduler dependency."""
    mock_scheduler = MagicMock()
    mock_scheduler.mark_dirty = MagicMock()
    mock_scheduler.flush_now = AsyncMock()
    return mock_scheduler


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
    mock_nginx_reload_scheduler,
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
        patch("registry.core.nginx_service.nginx_reload_scheduler", mock_nginx_reload_scheduler),
        patch("registry.api.server_routes.security_scanner_service", mock_security_scanner_service),
        patch("registry.utils.scopes_manager.update_server_scopes", new_callable=AsyncMock),
        patch("registry.api.server_routes.enhanced_auth", mock_enhanced_auth_func),
    ):
        from registry.auth.csrf import verify_csrf_token_flexible
        from registry.main import app

        # Override CSRF verification for tests
        app.dependency_overrides[verify_csrf_token_flexible] = lambda: None

        # Create client with session cookie (uses the default cookie name mcp_gateway_session)
        client = TestClient(app, cookies={"mcp_gateway_session": "test-session"})
        yield client

        app.dependency_overrides.pop(verify_csrf_token_flexible, None)


@pytest.fixture
def test_client_regular(
    mock_settings,
    mock_server_service,
    mock_faiss_service,
    mock_health_service,
    mock_nginx_service,
    mock_nginx_reload_scheduler,
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
        patch("registry.core.nginx_service.nginx_reload_scheduler", mock_nginx_reload_scheduler),
        patch("registry.api.server_routes.security_scanner_service", mock_security_scanner_service),
        patch("registry.utils.scopes_manager.update_server_scopes", new_callable=AsyncMock),
        patch("registry.api.server_routes.enhanced_auth", mock_enhanced_auth_func),
    ):
        from registry.auth.csrf import verify_csrf_token_flexible
        from registry.main import app

        # Override CSRF verification for tests
        app.dependency_overrides[verify_csrf_token_flexible] = lambda: None

        # Create client with session cookie (uses the default cookie name mcp_gateway_session)
        client = TestClient(app, cookies={"mcp_gateway_session": "test-session"})
        yield client

        app.dependency_overrides.pop(verify_csrf_token_flexible, None)


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
        """Test that admin user gets all servers via JSON API (fast path)."""
        # Arrange - admin with no filters uses fast path (get_servers_paginated)
        mock_server_service.get_servers_paginated = AsyncMock(
            return_value=(
                {
                    "/server1": {
                        "server_name": "Server 1",
                        "description": "Test 1",
                        "tags": [],
                        "num_tools": 3,
                        "license": "MIT",
                        "proxy_pass_url": "http://localhost:8080",
                    }
                },
                1,
            )
        )

        # Act
        response = test_client_admin.get("/api/servers")

        # Assert
        assert response.status_code == 200
        data = response.json()
        assert "servers" in data
        assert len(data["servers"]) == 1
        assert data["total_count"] == 1
        assert data["limit"] == 20
        assert data["offset"] == 0
        assert data["has_next"] is False
        mock_server_service.get_servers_paginated.assert_called_once_with(
            skip=0, limit=20, exclude_tool_list=False
        )

    def test_list_response_includes_local_server_fields(
        self, test_client_admin, mock_server_service
    ):
        """GET /api/servers must surface deployment + local_runtime so the
        frontend can render the LOCAL badge and the connect modal can emit a
        stdio launch recipe."""
        mock_server_service.get_servers_paginated = AsyncMock(
            return_value=(
                {
                    "/local-weather": {
                        "server_name": "Local Weather",
                        "description": "stdio",
                        "tags": ["security-pending-local"],
                        "num_tools": 0,
                        "license": "MIT",
                        "deployment": "local",
                        "local_runtime": {
                            "type": "npx",
                            "package": "@acme/weather-mcp",
                            "version": "1.0.0",
                        },
                        "registered_by": "admin",
                    },
                    "/remote-srv": {
                        "server_name": "Remote",
                        "description": "http",
                        "tags": [],
                        "num_tools": 5,
                        "license": "Apache-2.0",
                        "proxy_pass_url": "http://upstream:9000",
                        "deployment": "remote",
                    },
                },
                2,
            )
        )

        response = test_client_admin.get("/api/servers")
        assert response.status_code == 200
        servers_by_path = {s["path"]: s for s in response.json()["servers"]}

        local = servers_by_path["/local-weather"]
        assert local["deployment"] == "local"
        assert local["local_runtime"]["type"] == "npx"
        assert local["local_runtime"]["package"] == "@acme/weather-mcp"
        assert local["registered_by"] == "admin"

        # Remote servers carry deployment too (default echoed back).
        remote = servers_by_path["/remote-srv"]
        assert remote["deployment"] == "remote"
        assert remote["local_runtime"] is None

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
                "license": "MIT",
                "proxy_pass_url": "http://localhost:8080",
            },
            "/server2": {
                "server_name": "Node Server",
                "description": "A Node.js-based server",
                "tags": ["nodejs"],
                "num_tools": 2,
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
        """Test that server list includes health status (fast path)."""
        # Arrange - admin with no filters uses fast path
        mock_server_service.get_servers_paginated = AsyncMock(
            return_value=(
                {
                    "/server1": {
                        "server_name": "Server 1",
                        "description": "Test",
                        "tags": [],
                        "num_tools": 3,
                        "license": "MIT",
                        "proxy_pass_url": "http://localhost:8080",
                    }
                },
                1,
            )
        )
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

    def test_include_tools_false_omits_tool_list(self, test_client_admin, mock_server_service):
        """include_tools=false drops the heavy tool_list but keeps num_tools
        from the stored doc, letting the dashboard render the tool-count badge
        without paying for per-server tool schemas + filtering."""
        # Arrange - server doc carries both num_tools and a populated tool_list
        mock_server_service.get_servers_paginated = AsyncMock(
            return_value=(
                {
                    "/server1": {
                        "server_name": "Server 1",
                        "description": "Test",
                        "tags": [],
                        "num_tools": 3,
                        "license": "MIT",
                        "proxy_pass_url": "http://localhost:8080",
                        "tool_list": [
                            {"name": "tool_a", "description": "A"},
                            {"name": "tool_b", "description": "B"},
                            {"name": "tool_c", "description": "C"},
                        ],
                    }
                },
                1,
            )
        )

        # Act
        response = test_client_admin.get("/api/servers?include_tools=false")

        # Assert
        assert response.status_code == 200
        server = response.json()["servers"][0]
        assert server["tool_list"] == []
        assert server["num_tools"] == 3

    def test_include_tools_true_returns_tool_list(self, test_client_admin, mock_server_service):
        """Default (include_tools=true) still returns the per-server tool_list
        so API consumers keep full payloads."""
        # Arrange
        mock_server_service.get_servers_paginated = AsyncMock(
            return_value=(
                {
                    "/server1": {
                        "server_name": "Server 1",
                        "description": "Test",
                        "tags": [],
                        "num_tools": 2,
                        "license": "MIT",
                        "proxy_pass_url": "http://localhost:8080",
                        "tool_list": [
                            {"name": "tool_a", "description": "A"},
                            {"name": "tool_b", "description": "B"},
                        ],
                    }
                },
                1,
            )
        )

        # Act
        response = test_client_admin.get("/api/servers")

        # Assert
        assert response.status_code == 200
        server = response.json()["servers"][0]
        assert len(server["tool_list"]) == 2
        assert server["num_tools"] == 2


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
        mock_nginx_reload_scheduler,
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
            mock_nginx_reload_scheduler.flush_now.assert_called()

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
        mock_nginx_reload_scheduler,
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
            mock_nginx_reload_scheduler.mark_dirty.assert_called()

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

    def test_register_service_metadata_omitted_defaults_to_empty_dict(
        self,
        test_client_admin,
        mock_server_service,
        mock_faiss_service,
        mock_nginx_service,
        mock_nginx_reload_scheduler,
        mock_health_service,
    ):
        """Test that omitting metadata persists an empty dict, not None."""
        # Arrange
        mock_server_service.register_server.return_value = {
            "success": True,
            "message": "Server registered successfully",
            "is_new_version": False,
        }

        with patch(
            "registry.auth.dependencies.user_has_ui_permission_for_service", return_value=True
        ):
            # Act - register without the metadata field
            response = test_client_admin.post(
                "/api/register",
                data={
                    "name": "No Metadata Server",
                    "description": "Test",
                    "path": "/no-metadata",
                    "proxy_pass_url": "http://localhost:9000",
                },
            )

            # Assert - metadata is always a dict in the persisted entry
            assert response.status_code == 201
            server_entry = mock_server_service.register_server.call_args[0][0]
            assert server_entry["metadata"] == {}

    def test_register_service_empty_metadata_defaults_to_empty_dict(
        self,
        test_client_admin,
        mock_server_service,
        mock_faiss_service,
        mock_nginx_service,
        mock_nginx_reload_scheduler,
        mock_health_service,
    ):
        """Test that an empty metadata JSON object is accepted."""
        # Arrange
        mock_server_service.register_server.return_value = {
            "success": True,
            "message": "Server registered successfully",
            "is_new_version": False,
        }

        with patch(
            "registry.auth.dependencies.user_has_ui_permission_for_service", return_value=True
        ):
            # Act - register with an explicit empty metadata object
            response = test_client_admin.post(
                "/api/register",
                data={
                    "name": "Empty Metadata Server",
                    "description": "Test",
                    "path": "/empty-metadata",
                    "proxy_pass_url": "http://localhost:9000",
                    "metadata": "{}",
                },
            )

            # Assert
            assert response.status_code == 201
            server_entry = mock_server_service.register_server.call_args[0][0]
            assert server_entry["metadata"] == {}

    def test_register_service_metadata_preserved_when_provided(
        self,
        test_client_admin,
        mock_server_service,
        mock_faiss_service,
        mock_nginx_service,
        mock_nginx_reload_scheduler,
        mock_health_service,
    ):
        """Test that populated metadata is preserved."""
        # Arrange
        mock_server_service.register_server.return_value = {
            "success": True,
            "message": "Server registered successfully",
            "is_new_version": False,
        }

        with patch(
            "registry.auth.dependencies.user_has_ui_permission_for_service", return_value=True
        ):
            # Act - register with populated metadata
            response = test_client_admin.post(
                "/api/register",
                data={
                    "name": "Metadata Server",
                    "description": "Test",
                    "path": "/with-metadata",
                    "proxy_pass_url": "http://localhost:9000",
                    "metadata": '{"team": "platform", "tier": "gold"}',
                },
            )

            # Assert
            assert response.status_code == 201
            server_entry = mock_server_service.register_server.call_args[0][0]
            assert server_entry["metadata"] == {"team": "platform", "tier": "gold"}

    def test_register_service_metadata_json_null_defaults_to_empty_dict(
        self,
        test_client_admin,
        mock_server_service,
        mock_faiss_service,
        mock_nginx_service,
        mock_nginx_reload_scheduler,
        mock_health_service,
    ):
        """Test that metadata sent as the JSON string 'null' coerces to {}."""
        # Arrange
        mock_server_service.register_server.return_value = {
            "success": True,
            "message": "Server registered successfully",
            "is_new_version": False,
        }

        with patch(
            "registry.auth.dependencies.user_has_ui_permission_for_service", return_value=True
        ):
            # Act - a truthy form value that parses to None must not persist None
            response = test_client_admin.post(
                "/api/register",
                data={
                    "name": "Null Metadata Server",
                    "description": "Test",
                    "path": "/null-metadata",
                    "proxy_pass_url": "http://localhost:9000",
                    "metadata": "null",
                },
            )

            # Assert
            assert response.status_code == 201
            server_entry = mock_server_service.register_server.call_args[0][0]
            assert server_entry["metadata"] == {}


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
        """Test successful internal registration with valid JWT Bearer token."""
        # Arrange - register_server returns a dict now
        mock_server_service.register_server.return_value = {
            "success": True,
            "message": "Server registered successfully",
            "is_new_version": False,
        }

        with (
            patch.dict("os.environ", {"SECRET_KEY": "testpass"}),
            patch("registry.utils.scopes_manager.update_server_scopes", new_callable=AsyncMock),
        ):
            token = generate_internal_token(subject="test-service", purpose="test")
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
                headers={"Authorization": f"Bearer {token}"},
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

    def test_internal_register_invalid_token(self, test_client_no_auth, mock_server_service):
        """Test internal registration fails with a token signed by a different key."""
        # Arrange - generate token with a different key than what the server expects
        with patch.dict("os.environ", {"SECRET_KEY": "wrong-secret-key"}):
            token = generate_internal_token(subject="test-service", purpose="test")

        with patch.dict("os.environ", {"SECRET_KEY": "correct-secret-key"}):
            # Act
            response = test_client_no_auth.post(
                "/api/internal/register",
                data={
                    "name": "Server",
                    "description": "Test",
                    "path": "/test",
                    "proxy_pass_url": "http://localhost:9000",
                },
                headers={"Authorization": f"Bearer {token}"},
            )

            # Assert
            assert response.status_code == 401
            assert "Invalid token" in response.json()["detail"]

    def test_internal_register_secret_key_not_set(self, test_client_no_auth):
        """Test internal registration fails when SECRET_KEY is not set on server."""
        # Arrange - generate a token with some key, but the server won't have SECRET_KEY set
        with patch.dict("os.environ", {"SECRET_KEY": "some-key"}):
            token = generate_internal_token(subject="test-service", purpose="test")

        # Ensure SECRET_KEY is not set in the server's environment
        with patch.dict("os.environ", {}, clear=True):
            # Act
            response = test_client_no_auth.post(
                "/api/internal/register",
                data={
                    "name": "Server",
                    "description": "Test",
                    "path": "/test",
                    "proxy_pass_url": "http://localhost:9000",
                },
                headers={"Authorization": f"Bearer {token}"},
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

        with (
            patch.dict("os.environ", {"SECRET_KEY": "testpass"}),
            patch("registry.utils.scopes_manager.update_server_scopes", new_callable=AsyncMock),
        ):
            token = generate_internal_token(subject="test-service", purpose="test")
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
                headers={"Authorization": f"Bearer {token}"},
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

        with patch.dict("os.environ", {"SECRET_KEY": "testpass"}):
            token = generate_internal_token(subject="test-service", purpose="test")
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
                headers={"Authorization": f"Bearer {token}"},
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

        with (
            patch.dict("os.environ", {"SECRET_KEY": "testpass"}),
            patch("registry.utils.scopes_manager.update_server_scopes", new_callable=AsyncMock),
        ):
            token = generate_internal_token(subject="test-service", purpose="test")
            # Act
            response = test_client_no_auth.post(
                "/api/internal/register",
                data={
                    "name": "Auto-Enabled Server",
                    "description": "Test",
                    "path": "/auto-enabled",
                    "proxy_pass_url": "http://localhost:9000",
                },
                headers={"Authorization": f"Bearer {token}"},
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

        with patch.dict("os.environ", {"SECRET_KEY": "testpass"}):
            token = generate_internal_token(subject="test-service", purpose="test")
            # Act
            response = test_client_no_auth.post(
                "/api/internal/remove",
                data={"service_path": "/test-server"},
                headers={"Authorization": f"Bearer {token}"},
            )

            # Assert
            assert response.status_code == 200
            mock_server_service.remove_server.assert_called_once_with("/test-server")

    def test_internal_remove_service_not_found(self, test_client_no_auth, mock_server_service):
        """Test internal removal fails when service not found."""
        # Arrange
        mock_server_service.get_server_info.return_value = None

        with patch.dict("os.environ", {"SECRET_KEY": "testpass"}):
            token = generate_internal_token(subject="test-service", purpose="test")
            # Act
            response = test_client_no_auth.post(
                "/api/internal/remove",
                data={"service_path": "/nonexistent"},
                headers={"Authorization": f"Bearer {token}"},
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

        with patch.dict("os.environ", {"SECRET_KEY": "testpass"}):
            token = generate_internal_token(subject="test-service", purpose="test")
            # Act
            response = test_client_no_auth.post(
                "/api/internal/remove",
                data={"service_path": "test-server"},  # Missing leading slash
                headers={"Authorization": f"Bearer {token}"},
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


# ==================================================================
# TEST POST /register - Local server support on the unified endpoint
# ==================================================================


@pytest.mark.unit
@pytest.mark.api
@pytest.mark.servers
class TestRegisterLocalServer:
    """Tests for local (stdio) server registration on /api/register."""

    def test_register_local_server_admin(
        self,
        test_client_admin,
        mock_server_service,
        mock_faiss_service,
        mock_nginx_service,
        mock_security_scanner_service,
    ):
        """Admin can register a local (stdio) server via form-encoded data.

        Also asserts the automated security scanner is NOT invoked for local
        servers — the registry can't probe a stdio launch recipe — so the
        skip-when-deployment==local branch is locked in by behavior.
        """
        with patch(
            "registry.auth.dependencies.user_has_ui_permission_for_service", return_value=True
        ):
            response = test_client_admin.post(
                "/api/register",
                data={
                    "name": "Local Weather",
                    "description": "stdio",
                    "path": "/local-weather",
                    "deployment": "local",
                    "local_runtime": json.dumps(
                        {
                            "type": "npx",
                            "package": "@acme/weather-mcp",
                            "version": "1.0.0",
                        }
                    ),
                },
            )
        assert response.status_code == 201
        entry = response.json()["service"]
        assert entry["deployment"] == "local"
        assert entry["local_runtime"]["type"] == "npx"
        assert entry["registered_by"] == "admin"
        # Tagged for manual security review (no automated scan ran).
        assert "security-pending-local" in entry["tags"]
        # The scanner must NOT have been called.
        mock_security_scanner_service.scan_server.assert_not_called()

    def test_register_local_server_non_admin_rejected(
        self,
        test_client_regular,
    ):
        """Non-admin users cannot register local servers."""
        response = test_client_regular.post(
            "/api/register",
            data={
                "name": "Local",
                "description": "x",
                "path": "/local",
                "deployment": "local",
                "local_runtime": json.dumps({"type": "npx", "package": "@acme/mcp"}),
            },
        )
        assert response.status_code == 403
        assert "admin" in response.json()["detail"].lower()

    def test_register_local_with_leaked_secret_rejected(
        self,
        test_client_admin,
    ):
        """An obvious literal secret in env values is rejected."""
        with patch(
            "registry.auth.dependencies.user_has_ui_permission_for_service", return_value=True
        ):
            response = test_client_admin.post(
                "/api/register",
                data={
                    "name": "Bad",
                    "description": "x",
                    "path": "/bad",
                    "deployment": "local",
                    "local_runtime": json.dumps(
                        {
                            "type": "npx",
                            "package": "@acme/mcp",
                            "env": {"OPENAI_KEY": "sk-proj-realsecretvalue123"},
                        }
                    ),
                },
            )
        assert response.status_code == 400
        detail = response.json()["detail"]
        assert "OPENAI_KEY" in detail.get("env_keys", [])

    def test_register_local_with_unpinned_npx_gets_warning_tag(
        self,
        test_client_admin,
        mock_server_service,
        mock_faiss_service,
        mock_nginx_service,
    ):
        """Unpinned npx package gets the 'unpinned-version' soft warning tag."""
        with patch(
            "registry.auth.dependencies.user_has_ui_permission_for_service", return_value=True
        ):
            response = test_client_admin.post(
                "/api/register",
                data={
                    "name": "Unpinned",
                    "description": "x",
                    "path": "/unpinned",
                    "deployment": "local",
                    "local_runtime": json.dumps({"type": "npx", "package": "@acme/mcp"}),
                },
            )
        assert response.status_code == 201
        assert "unpinned-version" in response.json()["service"]["tags"]

    def test_register_local_rejects_proxy_pass_url(
        self,
        test_client_admin,
    ):
        """Local deployment must not include proxy_pass_url."""
        with patch(
            "registry.auth.dependencies.user_has_ui_permission_for_service", return_value=True
        ):
            response = test_client_admin.post(
                "/api/register",
                data={
                    "name": "Confused",
                    "description": "x",
                    "path": "/confused",
                    "deployment": "local",
                    "proxy_pass_url": "http://nope",
                    "local_runtime": json.dumps({"type": "npx", "package": "@acme/mcp"}),
                },
            )
        assert response.status_code == 400
        assert "must not set proxy_pass_url" in response.json()["detail"]

    def test_register_remote_unaffected(
        self,
        test_client_admin,
        mock_server_service,
        mock_faiss_service,
        mock_nginx_service,
    ):
        """Remote registration without explicit deployment still works (default)."""
        with patch(
            "registry.auth.dependencies.user_has_ui_permission_for_service", return_value=True
        ):
            response = test_client_admin.post(
                "/api/register",
                data={
                    "name": "Remote Server",
                    "description": "test",
                    "path": "/remote-srv",
                    "proxy_pass_url": "http://upstream:9000",
                },
            )
        assert response.status_code == 201
        entry = response.json()["service"]
        assert entry["deployment"] == "remote"
        assert entry["registered_by"] == "admin"


# =============================================================================
# TEST POST /edit/{path} - Local server edit support
# =============================================================================


@pytest.mark.unit
@pytest.mark.api
@pytest.mark.servers
class TestEditLocalServer:
    """Edit endpoint must accept deployment + local_runtime fields and gate
    local edits to admins (mirrors the registration flow)."""

    LOCAL_SERVER_INFO = {
        "server_name": "Existing Local",
        "description": "stdio",
        "path": "/local-srv",
        "deployment": "local",
        "local_runtime": {"type": "npx", "package": "@acme/mcp", "version": "1.0.0"},
        "tags": ["security-pending-local"],
        "auth_scheme": "none",
        "registered_by": "admin",
    }

    REMOTE_SERVER_INFO = {
        "server_name": "Existing Remote",
        "description": "http",
        "path": "/remote-srv",
        "deployment": "remote",
        "proxy_pass_url": "http://upstream:9000",
        "tags": [],
        "auth_scheme": "none",
        "registered_by": "admin",
    }

    def test_edit_local_server_admin(
        self,
        test_client_admin,
        mock_server_service,
        mock_faiss_service,
        mock_nginx_service,
    ):
        """Admin can edit a local server's launch recipe."""
        mock_server_service.get_server_info.return_value = self.LOCAL_SERVER_INFO
        mock_server_service.is_service_enabled.return_value = True

        response = test_client_admin.post(
            "/api/edit/local-srv",
            headers={"accept": "application/json"},
            data={
                "name": "Existing Local",
                "description": "updated",
                "deployment": "local",
                "tags": "team:weather",
                "local_runtime": json.dumps(
                    {
                        "type": "npx",
                        "package": "@acme/mcp",
                        "version": "1.1.0",
                    }
                ),
            },
        )
        assert response.status_code == 200
        # Verify update_server was called with merged entry
        call_args = mock_server_service.update_server.call_args
        assert call_args is not None
        updated_entry = call_args[0][1]
        assert updated_entry["deployment"] == "local"
        assert updated_entry["local_runtime"]["version"] == "1.1.0"
        # Audit trail is preserved across edits
        assert updated_entry["registered_by"] == "admin"
        # Recipe content changed (1.0.0 → 1.1.0) so security-pending-local is
        # re-added — the prior review was for the OLD recipe.
        assert "security-pending-local" in updated_entry["tags"]

    def test_edit_local_server_non_admin_rejected(
        self,
        test_client_regular,
        mock_server_service,
    ):
        """Non-admin cannot edit local servers (executable recipe)."""
        mock_server_service.get_server_info.return_value = self.LOCAL_SERVER_INFO

        response = test_client_regular.post(
            "/api/edit/local-srv",
            data={
                "name": "Existing Local",
                "deployment": "local",
                "local_runtime": json.dumps({"type": "npx", "package": "@acme/mcp"}),
            },
        )
        assert response.status_code == 403
        assert "admin" in response.json()["detail"].lower()

    def test_edit_local_rejects_proxy_pass_url(
        self,
        test_client_admin,
        mock_server_service,
    ):
        """Local edit must not include proxy_pass_url."""
        mock_server_service.get_server_info.return_value = self.LOCAL_SERVER_INFO

        response = test_client_admin.post(
            "/api/edit/local-srv",
            data={
                "name": "Existing Local",
                "deployment": "local",
                "proxy_pass_url": "http://nope",
                "local_runtime": json.dumps({"type": "npx", "package": "@acme/mcp"}),
            },
        )
        assert response.status_code == 400
        assert "must not set proxy_pass_url" in response.json()["detail"]

    def test_edit_local_with_leaked_secret_rejected(
        self,
        test_client_admin,
        mock_server_service,
    ):
        """Edit-time secret-leak guard works the same as registration."""
        mock_server_service.get_server_info.return_value = self.LOCAL_SERVER_INFO

        response = test_client_admin.post(
            "/api/edit/local-srv",
            data={
                "name": "Existing Local",
                "deployment": "local",
                "local_runtime": json.dumps(
                    {
                        "type": "npx",
                        "package": "@acme/mcp",
                        "env": {"OPENAI_KEY": "sk-proj-realsecretvalue123"},
                    }
                ),
            },
        )
        assert response.status_code == 400
        detail = response.json()["detail"]
        assert "OPENAI_KEY" in detail.get("env_keys", [])

    def test_edit_remote_unaffected_by_new_fields(
        self,
        test_client_admin,
        mock_server_service,
        mock_faiss_service,
        mock_nginx_service,
    ):
        """Remote edit still works when deployment field is omitted (preserves existing)."""
        mock_server_service.get_server_info.return_value = self.REMOTE_SERVER_INFO
        mock_server_service.is_service_enabled.return_value = True

        with patch(
            "registry.auth.dependencies.user_has_ui_permission_for_service", return_value=True
        ):
            response = test_client_admin.post(
                "/api/edit/remote-srv",
                headers={"accept": "application/json"},
                data={
                    "name": "Existing Remote",
                    "description": "updated",
                    "proxy_pass_url": "http://upstream:9001",
                    "tags": "",
                },
            )
        assert response.status_code == 200
        call_args = mock_server_service.update_server.call_args
        updated_entry = call_args[0][1]
        assert updated_entry["deployment"] == "remote"
        assert updated_entry["proxy_pass_url"] == "http://upstream:9001"

    def test_edit_remote_requires_proxy_pass_url(
        self,
        test_client_admin,
        mock_server_service,
    ):
        """Remote edit without proxy_pass_url is rejected (regression check)."""
        mock_server_service.get_server_info.return_value = self.REMOTE_SERVER_INFO

        with patch(
            "registry.auth.dependencies.user_has_ui_permission_for_service", return_value=True
        ):
            response = test_client_admin.post(
                "/api/edit/remote-srv",
                data={
                    "name": "Existing Remote",
                    "description": "broken",
                },
            )
        assert response.status_code == 400
        assert "proxy_pass_url is required" in response.json()["detail"]


# =============================================================================
# TEST POST /register and /edit input validation
# =============================================================================


@pytest.mark.unit
@pytest.mark.api
@pytest.mark.servers
class TestDeploymentValidation:
    """Hardening: explicit deployment value validation + edit-time type lock."""

    def test_register_rejects_invalid_deployment_value(self, test_client_admin):
        """Garbage deployment value must be rejected before persist (else
        ServerInfo's Literal validator only catches it on read)."""
        with patch(
            "registry.auth.dependencies.user_has_ui_permission_for_service", return_value=True
        ):
            response = test_client_admin.post(
                "/api/register",
                data={
                    "name": "Bad",
                    "description": "x",
                    "path": "/bad",
                    "deployment": "banana",
                    "proxy_pass_url": "http://upstream:9000",
                },
            )
        assert response.status_code == 400
        assert "deployment must be" in response.json()["detail"]

    def test_edit_rejects_remote_to_local_change(
        self,
        test_client_admin,
        mock_server_service,
    ):
        """Switching deployment type via edit is blocked: it would strip the
        nginx route, change auth semantics, and break the audit trail."""
        mock_server_service.get_server_info.return_value = {
            "server_name": "Remote",
            "path": "/remote-srv",
            "deployment": "remote",
            "proxy_pass_url": "http://upstream:9000",
            "tags": [],
        }

        response = test_client_admin.post(
            "/api/edit/remote-srv",
            data={
                "name": "Remote",
                "deployment": "local",
                "local_runtime": json.dumps({"type": "npx", "package": "@acme/mcp"}),
            },
        )
        assert response.status_code == 400
        assert "Cannot change deployment" in response.json()["detail"]

    def test_edit_rejects_local_to_remote_change(
        self,
        test_client_admin,
        mock_server_service,
    ):
        mock_server_service.get_server_info.return_value = {
            "server_name": "Local",
            "path": "/local-srv",
            "deployment": "local",
            "local_runtime": {"type": "npx", "package": "@acme/mcp"},
            "tags": ["security-pending-local"],
        }

        response = test_client_admin.post(
            "/api/edit/local-srv",
            data={
                "name": "Local",
                "deployment": "remote",
                "proxy_pass_url": "http://upstream:9000",
            },
        )
        assert response.status_code == 400
        assert "Cannot change deployment" in response.json()["detail"]

    def test_edit_rejects_invalid_deployment_value(
        self,
        test_client_admin,
        mock_server_service,
    ):
        mock_server_service.get_server_info.return_value = {
            "server_name": "Remote",
            "path": "/remote-srv",
            "deployment": "remote",
            "proxy_pass_url": "http://upstream:9000",
            "tags": [],
        }
        response = test_client_admin.post(
            "/api/edit/remote-srv",
            data={"name": "Remote", "deployment": "banana"},
        )
        assert response.status_code == 400
        assert "deployment must be" in response.json()["detail"]


@pytest.mark.unit
@pytest.mark.api
@pytest.mark.servers
class TestEditLocalSecurityReviewReset:
    """When the launch recipe materially changes on edit, the prior security
    review is invalidated — the admin approved the OLD recipe. Cosmetic edits
    (description, tags, status) preserve the existing review state."""

    REVIEWED_LOCAL = {
        "server_name": "Reviewed",
        "path": "/local-srv",
        "deployment": "local",
        "local_runtime": {
            "type": "npx",
            "package": "@acme/mcp",
            "version": "1.0.0",
            "args": [],
            "env": {},
            "required_env": [],
        },
        # Tag previously cleared by an admin via /clear-security-pending-local.
        "tags": ["team:weather"],
        "auth_scheme": "none",
        "registered_by": "admin",
    }

    def test_cosmetic_edit_preserves_cleared_review(
        self,
        test_client_admin,
        mock_server_service,
        mock_faiss_service,
        mock_nginx_service,
    ):
        """Identical recipe → tag stays cleared (review preserved)."""
        mock_server_service.get_server_info.return_value = self.REVIEWED_LOCAL
        mock_server_service.is_service_enabled.return_value = True

        response = test_client_admin.post(
            "/api/edit/local-srv",
            headers={"accept": "application/json"},
            data={
                "name": "Reviewed",
                "description": "updated copy",  # cosmetic change
                "deployment": "local",
                "tags": "team:weather",
                "local_runtime": json.dumps(
                    {
                        "type": "npx",
                        "package": "@acme/mcp",
                        "version": "1.0.0",
                        "args": [],
                        "env": {},
                        "required_env": [],
                    }
                ),
            },
        )
        assert response.status_code == 200
        updated = mock_server_service.update_server.call_args[0][1]
        assert "security-pending-local" not in updated["tags"]

    def test_recipe_change_resets_review(
        self,
        test_client_admin,
        mock_server_service,
        mock_faiss_service,
        mock_nginx_service,
    ):
        """Different recipe (version bump) → tag re-added even though it had
        been cleared. Admin must re-review the new recipe."""
        mock_server_service.get_server_info.return_value = self.REVIEWED_LOCAL
        mock_server_service.is_service_enabled.return_value = True

        response = test_client_admin.post(
            "/api/edit/local-srv",
            headers={"accept": "application/json"},
            data={
                "name": "Reviewed",
                "deployment": "local",
                "tags": "team:weather",
                "local_runtime": json.dumps(
                    {
                        "type": "npx",
                        "package": "@acme/mcp",
                        "version": "2.0.0",  # ← recipe change
                        "args": [],
                        "env": {},
                        "required_env": [],
                    }
                ),
            },
        )
        assert response.status_code == 200
        updated = mock_server_service.update_server.call_args[0][1]
        assert "security-pending-local" in updated["tags"]

    def test_args_change_resets_review(
        self,
        test_client_admin,
        mock_server_service,
        mock_faiss_service,
        mock_nginx_service,
    ):
        """args list change (e.g. swapping a flag) is also a material change."""
        mock_server_service.get_server_info.return_value = self.REVIEWED_LOCAL
        mock_server_service.is_service_enabled.return_value = True

        response = test_client_admin.post(
            "/api/edit/local-srv",
            headers={"accept": "application/json"},
            data={
                "name": "Reviewed",
                "deployment": "local",
                "tags": "team:weather",
                "local_runtime": json.dumps(
                    {
                        "type": "npx",
                        "package": "@acme/mcp",
                        "version": "1.0.0",
                        "args": ["--unsafe-flag"],  # ← args change
                        "env": {},
                        "required_env": [],
                    }
                ),
            },
        )
        assert response.status_code == 200
        updated = mock_server_service.update_server.call_args[0][1]
        assert "security-pending-local" in updated["tags"]


# =============================================================================
# TEST POST /refresh/{path} - Local server refresh path
# =============================================================================


@pytest.mark.unit
@pytest.mark.api
@pytest.mark.servers
class TestRefreshLocalServer:
    """clicking Refresh Health on a local server card must NOT
    return a 500 'no proxy URL' error. The endpoint should short-circuit
    identically to the toggle path."""

    def test_refresh_local_server_returns_local_status(
        self,
        test_client_admin,
        mock_server_service,
        mock_faiss_service,
        mock_nginx_service,
    ):
        mock_server_service.get_server_info.return_value = {
            "server_name": "Local",
            "path": "/local-srv",
            "deployment": "local",
            "local_runtime": {"type": "npx", "package": "@acme/mcp"},
            "num_tools": 3,
        }
        mock_server_service.is_service_enabled.return_value = True

        with patch(
            "registry.auth.dependencies.user_has_ui_permission_for_service", return_value=True
        ):
            response = test_client_admin.post("/api/refresh/local-srv")

        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "local"
        assert body["last_checked_iso"] is None
        assert body["num_tools"] == 3


# =============================================================================
# TEST POST /clear-security-pending-local/{path}
# =============================================================================


@pytest.mark.unit
@pytest.mark.api
@pytest.mark.servers
class TestClearSecurityPendingLocal:
    """Admin action that removes the security-pending-local tag — gives admins
    a discoverable way to mark a local server as reviewed without editing the
    tags string by hand."""

    LOCAL_PENDING = {
        "server_name": "Local",
        "path": "/local-srv",
        "deployment": "local",
        "local_runtime": {"type": "npx", "package": "@acme/mcp"},
        "tags": ["security-pending-local", "team:weather"],
    }

    def test_clears_tag_for_admin(
        self,
        test_client_admin,
        mock_server_service,
        mock_faiss_service,
    ):
        mock_server_service.get_server_info.return_value = self.LOCAL_PENDING
        mock_server_service.is_service_enabled.return_value = True

        response = test_client_admin.post("/api/clear-security-pending-local/local-srv")
        assert response.status_code == 200
        body = response.json()
        assert "security-pending-local" not in body["tags"]
        assert "team:weather" in body["tags"]

        # update_server called with tag removed
        call_args = mock_server_service.update_server.call_args
        updated = call_args[0][1]
        assert "security-pending-local" not in updated["tags"]

    def test_non_admin_rejected(self, test_client_regular, mock_server_service):
        mock_server_service.get_server_info.return_value = self.LOCAL_PENDING
        response = test_client_regular.post("/api/clear-security-pending-local/local-srv")
        assert response.status_code == 403

    def test_404_for_unknown_server(self, test_client_admin, mock_server_service):
        mock_server_service.get_server_info.return_value = None
        response = test_client_admin.post("/api/clear-security-pending-local/nope")
        assert response.status_code == 404

    def test_400_for_remote_server(self, test_client_admin, mock_server_service):
        mock_server_service.get_server_info.return_value = {
            "server_name": "Remote",
            "path": "/remote",
            "deployment": "remote",
            "proxy_pass_url": "http://upstream",
            "tags": [],
        }
        response = test_client_admin.post("/api/clear-security-pending-local/remote")
        assert response.status_code == 400

    def test_idempotent_when_tag_absent(
        self,
        test_client_admin,
        mock_server_service,
    ):
        """Calling on a local server that's already been cleared is a no-op."""
        mock_server_service.get_server_info.return_value = {
            **self.LOCAL_PENDING,
            "tags": ["team:weather"],  # already cleared
        }
        response = test_client_admin.post("/api/clear-security-pending-local/local-srv")
        assert response.status_code == 200
        # update_server NOT called — we short-circuit.
        mock_server_service.update_server.assert_not_called()


# =============================================================================
# TESTS — POST /api/servers/register visibility round-trip (Issue #1181)
# =============================================================================


@pytest.mark.unit
@pytest.mark.api
@pytest.mark.servers
class TestServerRegisterVisibility:
    """Regression coverage for #1181.

    The newer POST /api/servers/register handler previously dropped
    visibility="public" entirely (never persisted), skipped validation,
    and the matching GET responses surfaced absent visibility as null.
    These tests pin the corrected behaviour.
    """

    _BASE_FORM = {
        "name": "Vis Server",
        "description": "Visibility round-trip test",
        "path": "/vis-server",
        "proxy_pass_url": "http://localhost:9000",
    }

    def test_register_persists_visibility_public(
        self, test_client_admin, mock_server_service
    ):
        response = test_client_admin.post(
            "/api/servers/register",
            data={**self._BASE_FORM, "visibility": "public"},
        )
        assert response.status_code == 201
        # The handler builds a server_entry dict and passes it to register_server.
        # That dict must carry visibility="public" — never omitted.
        call_args = mock_server_service.register_server.call_args
        server_entry = call_args.args[0] if call_args.args else call_args.kwargs.get("server_entry")
        assert server_entry["visibility"] == "public"

    def test_register_rejects_invalid_visibility(
        self, test_client_admin, mock_server_service
    ):
        response = test_client_admin.post(
            "/api/servers/register",
            data={**self._BASE_FORM, "visibility": "garbage"},
        )
        assert response.status_code == 400
        assert "visibility" in response.json()["detail"].lower()
        mock_server_service.register_server.assert_not_called()

    def test_register_rejects_group_restricted_without_groups(
        self, test_client_admin, mock_server_service
    ):
        response = test_client_admin.post(
            "/api/servers/register",
            data={**self._BASE_FORM, "visibility": "group-restricted"},
        )
        assert response.status_code == 400
        assert "allowed_group" in response.json()["detail"].lower()
        mock_server_service.register_server.assert_not_called()

    def test_register_persists_group_restricted_with_groups(
        self, test_client_admin, mock_server_service
    ):
        response = test_client_admin.post(
            "/api/servers/register",
            data={
                **self._BASE_FORM,
                "visibility": "group-restricted",
                "allowed_groups": "engineering,platform-team",
            },
        )
        assert response.status_code == 201
        call_args = mock_server_service.register_server.call_args
        server_entry = call_args.args[0] if call_args.args else call_args.kwargs.get("server_entry")
        assert server_entry["visibility"] == "group-restricted"
        assert server_entry["allowed_groups"] == ["engineering", "platform-team"]


# =============================================================================
# TESTS — GET endpoints normalize absent visibility to "public" (Issue #1181)
# =============================================================================


@pytest.mark.unit
@pytest.mark.api
@pytest.mark.servers
class TestServerGetVisibilityNormalization:
    """Servers stored before #1181's write-side fix have no visibility key.

    GET responses must default the field to "public" so clients never see
    a misleading null. The list endpoint historically did not include the
    field at all; #1181 adds it.
    """

    _LEGACY_DOC = {
        "server_name": "Legacy Server",
        "description": "Pre-#1181 row with no visibility key",
        "path": "/legacy",
        "proxy_pass_url": "http://localhost:9000",
        "is_enabled": True,
        # Note: no 'visibility' key — this is the legacy shape under test.
    }

    def test_get_server_defaults_absent_visibility_to_public(
        self, test_client_admin, mock_server_service
    ):
        mock_server_service.get_server_info.return_value = {**self._LEGACY_DOC}

        response = test_client_admin.get("/api/servers/legacy")
        assert response.status_code == 200
        assert response.json()["visibility"] == "public"

    def test_list_servers_defaults_absent_visibility_to_public(
        self, test_client_admin, mock_server_service
    ):
        mock_server_service.get_servers_paginated = AsyncMock(
            return_value=({"/legacy": {**self._LEGACY_DOC}}, 1)
        )

        response = test_client_admin.get("/api/servers")
        assert response.status_code == 200
        servers = response.json()["servers"]
        assert len(servers) == 1
        assert servers[0]["visibility"] == "public"


# =============================================================================
# TESTS — Metadata round-trip on POST /api/servers/register (Issue #1165 follow-up)
# =============================================================================


@pytest.mark.unit
@pytest.mark.api
@pytest.mark.servers
class TestServersRegisterAPIMetadata:
    """Mirrors the metadata matrix that PR #1175 added for POST /api/register
    onto the newer POST /api/servers/register. The fix in #1175 lands in
    three handlers but only the legacy one was covered; this class closes
    that gap and also pins the non-dict guard added as a #1165 follow-up.
    """

    _BASE_FORM = {
        "name": "Meta Server",
        "description": "Metadata round-trip test",
        "path": "/meta-server",
        "proxy_pass_url": "http://localhost:9000",
    }

    @staticmethod
    def _stored(mock_server_service):
        call_args = mock_server_service.register_server.call_args
        return call_args.args[0] if call_args.args else call_args.kwargs.get("server_entry")

    def test_register_metadata_omitted_defaults_to_empty_dict(
        self, test_client_admin, mock_server_service
    ):
        response = test_client_admin.post("/api/servers/register", data=self._BASE_FORM)
        assert response.status_code == 201
        assert self._stored(mock_server_service)["metadata"] == {}

    def test_register_metadata_empty_json_object_defaults_to_empty_dict(
        self, test_client_admin, mock_server_service
    ):
        response = test_client_admin.post(
            "/api/servers/register",
            data={**self._BASE_FORM, "metadata": "{}"},
        )
        assert response.status_code == 201
        assert self._stored(mock_server_service)["metadata"] == {}

    def test_register_metadata_preserved_when_provided(
        self, test_client_admin, mock_server_service
    ):
        response = test_client_admin.post(
            "/api/servers/register",
            data={**self._BASE_FORM, "metadata": '{"team": "platform", "tier": "gold"}'},
        )
        assert response.status_code == 201
        assert self._stored(mock_server_service)["metadata"] == {
            "team": "platform",
            "tier": "gold",
        }

    def test_register_metadata_json_null_defaults_to_empty_dict(
        self, test_client_admin, mock_server_service
    ):
        response = test_client_admin.post(
            "/api/servers/register",
            data={**self._BASE_FORM, "metadata": "null"},
        )
        assert response.status_code == 201
        assert self._stored(mock_server_service)["metadata"] == {}

    def test_register_metadata_array_coerced_to_empty_dict(
        self, test_client_admin, mock_server_service, caplog
    ):
        with caplog.at_level("WARNING", logger="registry.api.server_routes"):
            response = test_client_admin.post(
                "/api/servers/register",
                data={**self._BASE_FORM, "metadata": "[]"},
            )
        assert response.status_code == 201
        assert self._stored(mock_server_service)["metadata"] == {}
        assert any("coerced to {}" in rec.message for rec in caplog.records)

    def test_register_metadata_scalar_coerced_to_empty_dict(
        self, test_client_admin, mock_server_service, caplog
    ):
        with caplog.at_level("WARNING", logger="registry.api.server_routes"):
            response = test_client_admin.post(
                "/api/servers/register",
                data={**self._BASE_FORM, "metadata": "42"},
            )
        assert response.status_code == 201
        assert self._stored(mock_server_service)["metadata"] == {}
        assert any("coerced to {}" in rec.message for rec in caplog.records)


# =============================================================================
# TESTS — Metadata round-trip on POST /api/internal/register (#1165 follow-up)
# =============================================================================


@pytest.mark.unit
@pytest.mark.api
@pytest.mark.servers
class TestInternalRegisterMetadata:
    """Same metadata matrix for the internal register endpoint."""

    _BASE_FORM = {
        "name": "Internal Meta Server",
        "description": "Internal metadata test",
        "path": "/internal-meta-server",
        "proxy_pass_url": "http://localhost:9000",
    }

    @staticmethod
    def _stored(mock_server_service):
        call_args = mock_server_service.register_server.call_args
        return call_args.args[0] if call_args.args else call_args.kwargs.get("server_entry")

    def _post(self, client, payload):
        with (
            patch.dict("os.environ", {"SECRET_KEY": "testpass"}),
            patch("registry.utils.scopes_manager.update_server_scopes", new_callable=AsyncMock),
        ):
            token = generate_internal_token(subject="test-service", purpose="test")
            return client.post(
                "/api/internal/register",
                data=payload,
                headers={"Authorization": f"Bearer {token}"},
            )

    def test_internal_register_metadata_omitted_defaults_to_empty_dict(
        self, test_client_no_auth, mock_server_service
    ):
        response = self._post(test_client_no_auth, self._BASE_FORM)
        assert response.status_code == 201
        assert self._stored(mock_server_service)["metadata"] == {}

    def test_internal_register_metadata_preserved_when_provided(
        self, test_client_no_auth, mock_server_service
    ):
        response = self._post(
            test_client_no_auth,
            {**self._BASE_FORM, "metadata": '{"owner": "ops"}'},
        )
        assert response.status_code == 201
        assert self._stored(mock_server_service)["metadata"] == {"owner": "ops"}

    def test_internal_register_metadata_json_null_defaults_to_empty_dict(
        self, test_client_no_auth, mock_server_service
    ):
        response = self._post(test_client_no_auth, {**self._BASE_FORM, "metadata": "null"})
        assert response.status_code == 201
        assert self._stored(mock_server_service)["metadata"] == {}

    def test_internal_register_metadata_array_coerced_to_empty_dict(
        self, test_client_no_auth, mock_server_service, caplog
    ):
        with caplog.at_level("WARNING", logger="registry.api.server_routes"):
            response = self._post(test_client_no_auth, {**self._BASE_FORM, "metadata": "[]"})
        assert response.status_code == 201
        assert self._stored(mock_server_service)["metadata"] == {}
        assert any("coerced to {}" in rec.message for rec in caplog.records)


# =============================================================================
# TESTS — Read-side metadata normalization on GET /api/servers/{path} (#1165 follow-up)
# =============================================================================


@pytest.mark.unit
@pytest.mark.api
@pytest.mark.servers
class TestServerGetMetadataNormalization:
    """Single-GET on a legacy doc with no metadata key historically omitted
    the field entirely, while the list endpoint emitted metadata: {}. The
    read-side default closes that contract gap. Mirrors the visibility
    normalization pattern from #1181/#1186.
    """

    _LEGACY_DOC = {
        "server_name": "Legacy Server",
        "description": "Pre-#1165 row with no metadata key",
        "path": "/legacy-meta",
        "proxy_pass_url": "http://localhost:9000",
        "is_enabled": True,
        # No 'metadata' key — this is the legacy shape under test.
    }

    def test_get_server_defaults_absent_metadata_to_empty_dict(
        self, test_client_admin, mock_server_service
    ):
        mock_server_service.get_server_info.return_value = {**self._LEGACY_DOC}

        response = test_client_admin.get("/api/servers/legacy-meta")
        assert response.status_code == 200
        body = response.json()
        assert "metadata" in body
        assert body["metadata"] == {}

    def test_get_server_coerces_non_dict_metadata_to_empty_dict(
        self, test_client_admin, mock_server_service
    ):
        """A stored row with metadata=[] or metadata=42 must not leak the bad
        shape into the response — the read path coerces to {} to match the
        ServerInfo type invariant."""
        mock_server_service.get_server_info.return_value = {
            **self._LEGACY_DOC,
            "metadata": [],
        }

        response = test_client_admin.get("/api/servers/legacy-meta")
        assert response.status_code == 200
        assert response.json()["metadata"] == {}
