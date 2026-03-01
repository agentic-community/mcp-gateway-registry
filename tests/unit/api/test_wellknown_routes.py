"""
Unit tests for registry/api/wellknown_routes.py

Tests the well-known URL discovery endpoint including:
- GET /.well-known/mcp-servers - MCP server discovery
- Health status retrieval from health service
- Status normalization for client consumption
"""

import logging
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

logger = logging.getLogger(__name__)


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def mock_server_service():
    """Mock server_service dependency."""
    mock_service = MagicMock()
    mock_service.get_all_servers = AsyncMock(return_value={})
    mock_service.is_service_enabled = AsyncMock(return_value=True)
    return mock_service


@pytest.fixture
def mock_health_service():
    """Mock health_service dependency with server_health_status dict."""
    mock_service = MagicMock()
    mock_service.server_health_status = {}
    return mock_service


@pytest.fixture
def sample_server_info() -> dict[str, Any]:
    """Create sample server information for testing."""
    return {
        "path": "test-server",
        "server_name": "Test Server",
        "description": "A test MCP server",
        "transport": "streamable-http",
        "auth_type": "oauth",
        "auth_provider": "keycloak",
        "tool_list": [
            {"name": "get_data", "description": "Get data from source"},
            {"name": "process_data", "description": "Process data"},
        ],
        "proxy_pass_url": "http://localhost:8000",
        "is_enabled": True,
    }


# =============================================================================
# UNIT TESTS FOR _get_normalized_health_status
# =============================================================================


class TestGetNormalizedHealthStatus:
    """Tests for the _get_normalized_health_status helper function."""

    def test_healthy_status_normalized(self, mock_health_service, mock_settings):
        """Test that 'healthy' status is returned as 'healthy'."""
        mock_health_service.server_health_status = {"test-server": "healthy"}

        with patch("registry.api.wellknown_routes.health_service", mock_health_service):
            from registry.api.wellknown_routes import _get_normalized_health_status

            result = _get_normalized_health_status("test-server")
            assert result == "healthy"

    def test_healthy_auth_expired_normalized_to_healthy(self, mock_health_service, mock_settings):
        """Test that 'healthy-auth-expired' is normalized to 'healthy'."""
        mock_health_service.server_health_status = {"test-server": "healthy-auth-expired"}

        with patch("registry.api.wellknown_routes.health_service", mock_health_service):
            from registry.api.wellknown_routes import _get_normalized_health_status

            result = _get_normalized_health_status("test-server")
            assert result == "healthy"

    def test_unhealthy_timeout_normalized(self, mock_health_service, mock_settings):
        """Test that 'unhealthy: timeout' is normalized to 'unhealthy'."""
        mock_health_service.server_health_status = {"test-server": "unhealthy: timeout"}

        with patch("registry.api.wellknown_routes.health_service", mock_health_service):
            from registry.api.wellknown_routes import _get_normalized_health_status

            result = _get_normalized_health_status("test-server")
            assert result == "unhealthy"

    def test_unhealthy_connection_error_normalized(self, mock_health_service, mock_settings):
        """Test that 'unhealthy: connection error' is normalized to 'unhealthy'."""
        mock_health_service.server_health_status = {"test-server": "unhealthy: connection error"}

        with patch("registry.api.wellknown_routes.health_service", mock_health_service):
            from registry.api.wellknown_routes import _get_normalized_health_status

            result = _get_normalized_health_status("test-server")
            assert result == "unhealthy"

    def test_error_status_normalized_to_unhealthy(self, mock_health_service, mock_settings):
        """Test that error statuses are normalized to 'unhealthy'."""
        mock_health_service.server_health_status = {"test-server": "error: ConnectionError"}

        with patch("registry.api.wellknown_routes.health_service", mock_health_service):
            from registry.api.wellknown_routes import _get_normalized_health_status

            result = _get_normalized_health_status("test-server")
            assert result == "unhealthy"

    def test_disabled_status_normalized(self, mock_health_service, mock_settings):
        """Test that 'disabled' status is returned as 'disabled'."""
        mock_health_service.server_health_status = {"test-server": "disabled"}

        with patch("registry.api.wellknown_routes.health_service", mock_health_service):
            from registry.api.wellknown_routes import _get_normalized_health_status

            result = _get_normalized_health_status("test-server")
            assert result == "disabled"

    def test_checking_status_normalized_to_unknown(self, mock_health_service, mock_settings):
        """Test that 'checking' status is normalized to 'unknown'."""
        mock_health_service.server_health_status = {"test-server": "checking"}

        with patch("registry.api.wellknown_routes.health_service", mock_health_service):
            from registry.api.wellknown_routes import _get_normalized_health_status

            result = _get_normalized_health_status("test-server")
            assert result == "unknown"

    def test_unknown_server_returns_unknown(self, mock_health_service, mock_settings):
        """Test that unknown servers return 'unknown' status."""
        mock_health_service.server_health_status = {}

        with patch("registry.api.wellknown_routes.health_service", mock_health_service):
            from registry.api.wellknown_routes import _get_normalized_health_status

            result = _get_normalized_health_status("nonexistent-server")
            assert result == "unknown"


# =============================================================================
# UNIT TESTS FOR _format_server_discovery
# =============================================================================


class TestFormatServerDiscovery:
    """Tests for the _format_server_discovery function."""

    def test_format_includes_health_status(
        self, mock_health_service, mock_settings, sample_server_info
    ):
        """Test that formatted server includes actual health status."""
        mock_health_service.server_health_status = {"test-server": "healthy"}

        with patch("registry.api.wellknown_routes.health_service", mock_health_service):
            from registry.api.wellknown_routes import _format_server_discovery

            # Create a mock request
            mock_request = MagicMock()
            mock_request.headers = {"host": "localhost:7860"}
            mock_request.url.scheme = "http"

            result = _format_server_discovery(sample_server_info, mock_request)

            assert result["health_status"] == "healthy"
            assert result["name"] == "Test Server"
            assert result["description"] == "A test MCP server"

    def test_format_uses_unhealthy_status_from_health_service(
        self, mock_health_service, mock_settings, sample_server_info
    ):
        """Test that formatted server uses unhealthy status from health service."""
        mock_health_service.server_health_status = {"test-server": "unhealthy: timeout"}

        with patch("registry.api.wellknown_routes.health_service", mock_health_service):
            from registry.api.wellknown_routes import _format_server_discovery

            mock_request = MagicMock()
            mock_request.headers = {"host": "localhost:7860"}
            mock_request.url.scheme = "http"

            result = _format_server_discovery(sample_server_info, mock_request)

            # Should be normalized to 'unhealthy'
            assert result["health_status"] == "unhealthy"

    def test_format_unknown_server_has_unknown_status(self, mock_health_service, mock_settings):
        """Test that servers not in health service have 'unknown' status."""
        mock_health_service.server_health_status = {}

        server_info = {
            "path": "new-server",
            "server_name": "New Server",
            "description": "A new server",
        }

        with patch("registry.api.wellknown_routes.health_service", mock_health_service):
            from registry.api.wellknown_routes import _format_server_discovery

            mock_request = MagicMock()
            mock_request.headers = {"host": "localhost:7860"}
            mock_request.url.scheme = "http"

            result = _format_server_discovery(server_info, mock_request)

            assert result["health_status"] == "unknown"


# =============================================================================
# INTEGRATION TESTS FOR GET /.well-known/mcp-servers
# =============================================================================


class TestWellKnownMcpServersEndpoint:
    """Integration tests for the well-known MCP servers endpoint."""

    def test_endpoint_returns_actual_health_status(
        self,
        mock_server_service,
        mock_health_service,
        mock_settings,
        sample_server_info,
    ):
        """Test that the endpoint returns actual health status, not hardcoded."""
        # Set up mock data
        mock_server_service.get_all_servers = AsyncMock(
            return_value={"test-server": sample_server_info}
        )
        mock_server_service.is_service_enabled = AsyncMock(return_value=True)
        mock_health_service.server_health_status = {"test-server": "unhealthy: connection error"}

        # Patch settings to enable discovery
        mock_settings.enable_wellknown_discovery = True
        mock_settings.wellknown_cache_ttl = 300

        with (
            patch("registry.api.wellknown_routes.server_service", mock_server_service),
            patch("registry.api.wellknown_routes.health_service", mock_health_service),
            patch("registry.api.wellknown_routes.settings", mock_settings),
        ):
            from registry.api.wellknown_routes import router
            from fastapi import FastAPI

            app = FastAPI()
            app.include_router(router, prefix="/.well-known")

            client = TestClient(app)
            response = client.get("/.well-known/mcp-servers")

            assert response.status_code == 200
            data = response.json()
            assert len(data["servers"]) == 1
            # Verify health_status is normalized from "unhealthy: connection error" to "unhealthy"
            assert data["servers"][0]["health_status"] == "unhealthy"

    def test_endpoint_returns_healthy_status(
        self,
        mock_server_service,
        mock_health_service,
        mock_settings,
        sample_server_info,
    ):
        """Test that healthy servers show as healthy."""
        mock_server_service.get_all_servers = AsyncMock(
            return_value={"test-server": sample_server_info}
        )
        mock_server_service.is_service_enabled = AsyncMock(return_value=True)
        mock_health_service.server_health_status = {"test-server": "healthy"}

        mock_settings.enable_wellknown_discovery = True
        mock_settings.wellknown_cache_ttl = 300

        with (
            patch("registry.api.wellknown_routes.server_service", mock_server_service),
            patch("registry.api.wellknown_routes.health_service", mock_health_service),
            patch("registry.api.wellknown_routes.settings", mock_settings),
        ):
            from registry.api.wellknown_routes import router
            from fastapi import FastAPI

            app = FastAPI()
            app.include_router(router, prefix="/.well-known")

            client = TestClient(app)
            response = client.get("/.well-known/mcp-servers")

            assert response.status_code == 200
            data = response.json()
            assert data["servers"][0]["health_status"] == "healthy"

    def test_endpoint_returns_unknown_for_unchecked_servers(
        self,
        mock_server_service,
        mock_health_service,
        mock_settings,
        sample_server_info,
    ):
        """Test that servers not yet health-checked show as unknown."""
        mock_server_service.get_all_servers = AsyncMock(
            return_value={"test-server": sample_server_info}
        )
        mock_server_service.is_service_enabled = AsyncMock(return_value=True)
        # Empty health status dict means no health checks have run yet
        mock_health_service.server_health_status = {}

        mock_settings.enable_wellknown_discovery = True
        mock_settings.wellknown_cache_ttl = 300

        with (
            patch("registry.api.wellknown_routes.server_service", mock_server_service),
            patch("registry.api.wellknown_routes.health_service", mock_health_service),
            patch("registry.api.wellknown_routes.settings", mock_settings),
        ):
            from registry.api.wellknown_routes import router
            from fastapi import FastAPI

            app = FastAPI()
            app.include_router(router, prefix="/.well-known")

            client = TestClient(app)
            response = client.get("/.well-known/mcp-servers")

            assert response.status_code == 200
            data = response.json()
            assert data["servers"][0]["health_status"] == "unknown"

    def test_multiple_servers_with_different_health_statuses(
        self,
        mock_server_service,
        mock_health_service,
        mock_settings,
    ):
        """Test that multiple servers show their individual health statuses."""
        servers = {
            "healthy-server": {
                "path": "healthy-server",
                "server_name": "Healthy Server",
                "description": "A healthy server",
            },
            "unhealthy-server": {
                "path": "unhealthy-server",
                "server_name": "Unhealthy Server",
                "description": "An unhealthy server",
            },
            "unknown-server": {
                "path": "unknown-server",
                "server_name": "Unknown Server",
                "description": "A server with unknown status",
            },
        }

        mock_server_service.get_all_servers = AsyncMock(return_value=servers)
        mock_server_service.is_service_enabled = AsyncMock(return_value=True)
        mock_health_service.server_health_status = {
            "healthy-server": "healthy",
            "unhealthy-server": "unhealthy: timeout",
            # unknown-server not in dict, should return "unknown"
        }

        mock_settings.enable_wellknown_discovery = True
        mock_settings.wellknown_cache_ttl = 300

        with (
            patch("registry.api.wellknown_routes.server_service", mock_server_service),
            patch("registry.api.wellknown_routes.health_service", mock_health_service),
            patch("registry.api.wellknown_routes.settings", mock_settings),
        ):
            from registry.api.wellknown_routes import router
            from fastapi import FastAPI

            app = FastAPI()
            app.include_router(router, prefix="/.well-known")

            client = TestClient(app)
            response = client.get("/.well-known/mcp-servers")

            assert response.status_code == 200
            data = response.json()
            assert len(data["servers"]) == 3

            # Create a dict for easier verification
            server_statuses = {s["name"]: s["health_status"] for s in data["servers"]}

            assert server_statuses["Healthy Server"] == "healthy"
            assert server_statuses["Unhealthy Server"] == "unhealthy"
            assert server_statuses["Unknown Server"] == "unknown"
