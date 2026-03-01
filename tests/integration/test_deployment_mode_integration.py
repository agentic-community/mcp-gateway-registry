"""Integration tests for deployment mode configuration endpoints.

These tests require a running MongoDB instance. They are skipped in CI
where MongoDB is not available.
"""

import pytest
from unittest.mock import AsyncMock, patch

from registry.core.config import DeploymentMode, RegistryMode


# Skip all tests in this module - requires MongoDB running
pytestmark = pytest.mark.skip(reason="Requires MongoDB running - not available in CI environment")


@pytest.fixture
def mock_peer_federation():
    """Mock peer federation service to avoid MongoDB event loop issues."""
    mock_service = AsyncMock()
    mock_service.registered_peers = []
    mock_service.load_peers_and_state = AsyncMock()

    mock_scheduler = AsyncMock()
    mock_scheduler.start = AsyncMock()
    mock_scheduler.stop = AsyncMock()

    with (
        patch(
            "registry.main.get_peer_federation_service",
            return_value=mock_service,
        ),
        patch(
            "registry.main.get_peer_sync_scheduler",
            return_value=mock_scheduler,
        ),
    ):
        yield mock_service


@pytest.fixture
def mock_auth_admin():
    """Mock authentication returning admin user context."""
    admin_context = {
        "username": "admin",
        "groups": ["mcp-registry-admin"],
        "scopes": [
            "mcp-servers-unrestricted/read",
            "mcp-servers-unrestricted/execute",
        ],
        "is_admin": True,
        "can_modify_servers": True,
    }
    with patch(
        "registry.api.server_routes.nginx_proxied_auth",
        return_value=admin_context,
    ):
        yield admin_context


@pytest.fixture
def integration_client(mock_settings, mock_peer_federation):
    """Test client with peer federation mocked to avoid event loop issues."""
    from fastapi.testclient import TestClient
    from registry.main import app

    with TestClient(app) as client:
        yield client


@pytest.mark.integration
class TestDeploymentModeIntegration:
    """Integration tests for deployment mode endpoints."""

    def test_config_endpoint_returns_mode(self, integration_client):
        """Config endpoint should return deployment mode fields."""
        response = integration_client.get("/api/config")
        assert response.status_code == 200
        data = response.json()
        assert "deployment_mode" in data
        assert "registry_mode" in data
        assert "nginx_updates_enabled" in data
        assert "features" in data
        assert "gateway_proxy" in data["features"]
        assert "mcp_servers" in data["features"]
        assert "agents" in data["features"]
        assert "skills" in data["features"]
        assert "federation" in data["features"]

    def test_health_includes_deployment_mode(self, integration_client):
        """Health endpoint should include deployment mode info."""
        response = integration_client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "deployment_mode" in data
        assert "registry_mode" in data
        assert "nginx_updates_enabled" in data

    def test_server_registration_works_in_registry_only(self, integration_client, mock_auth_admin):
        """Server registration should not 500 in registry-only mode."""
        response = integration_client.post(
            "/api/servers/register",
            json={
                "server_name": "test-server",
                "path": "/test-server",
                "transport": "sse",
                "proxy_pass_url": "http://localhost:8080/mcp",
            },
        )
        assert response.status_code != 500

    def test_server_toggle_works_in_registry_only(self, integration_client, mock_auth_admin):
        """Server toggle should not fail due to nginx in registry-only mode."""
        response = integration_client.post(
            "/api/servers/test-server/toggle",
            json={"enabled": True},
        )
        assert response.status_code != 500
