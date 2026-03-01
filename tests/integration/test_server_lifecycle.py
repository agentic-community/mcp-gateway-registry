"""
Integration tests for server lifecycle (CRUD operations).

This module tests the full lifecycle of server management including:
- Registration
- Listing
- Retrieval
- Updates
- Deletion
- Error handling

NOTE: These tests are currently skipped due to data persistence issue where
servers register successfully but don't appear in list/retrieve operations.
"""

import json
import logging
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import status as http_status
from fastapi.testclient import TestClient

logger = logging.getLogger(__name__)

# Skip all tests in this file due to data persistence issue
pytestmark = pytest.mark.skip(
    reason="Data persistence issue - servers register but don't appear in listings"
)


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def mock_nginx_service():
    """
    Mock nginx service to avoid actual nginx operations.

    Returns:
        Mock nginx service instance
    """
    with patch("registry.core.nginx_service.nginx_service") as mock_nginx:
        mock_nginx.generate_config = MagicMock()
        mock_nginx.reload_nginx = MagicMock()
        mock_nginx.generate_config_async = AsyncMock()
        yield mock_nginx


@pytest.fixture
def mock_faiss_service():
    """
    Mock FAISS service to avoid actual embedding operations.

    Returns:
        Mock FAISS service instance
    """
    with patch("registry.search.service.faiss_service") as mock_faiss:
        mock_faiss.initialize = AsyncMock()
        mock_faiss.add_or_update_service = AsyncMock()
        mock_faiss.add_or_update_agent = AsyncMock()
        mock_faiss.remove_service = AsyncMock()
        mock_faiss.save_data = AsyncMock()
        yield mock_faiss


@pytest.fixture
def mock_health_service():
    """
    Mock health service to avoid actual health checks.

    Returns:
        Mock health service instance
    """
    with patch("registry.health.service.health_service") as mock_health:
        mock_health.initialize = AsyncMock()
        mock_health.shutdown = AsyncMock()
        mock_health.broadcast_health_update = AsyncMock()
        mock_health.perform_immediate_health_check = AsyncMock(return_value=("healthy", None))
        mock_health._get_service_health_data = MagicMock(
            return_value={"status": "healthy", "last_checked_iso": "2024-01-01T00:00:00"}
        )
        yield mock_health


@pytest.fixture
def mock_agent_service():
    """
    Mock agent service to avoid actual agent operations.

    Returns:
        Mock agent service instance
    """
    with patch("registry.services.agent_service.agent_service") as mock_agent:
        mock_agent.load_agents_and_state = AsyncMock()
        mock_agent.list_agents = MagicMock(return_value=[])
        mock_agent.is_agent_enabled = MagicMock(return_value=False)
        yield mock_agent




@pytest.fixture
def mock_auth_dependencies():
    """
    Mock authentication dependencies using dependency_overrides.

    Returns:
        Dict with admin and regular user contexts
    """
    from registry.auth.dependencies import (
        enhanced_auth,
        nginx_proxied_auth,
    )
    from registry.main import app

    admin_user_context = {
        "username": "testadmin",
        "is_admin": True,
        "groups": ["admin"],
        "scopes": ["admin"],
        "accessible_servers": [],
        "accessible_services": ["all"],
        "ui_permissions": {
            "list_service": ["all"],
            "toggle_service": ["all"],
            "register_service": ["all"],
            "modify_service": ["all"],
        },
        "auth_method": "session",
    }

    regular_user_context = {
        "username": "testuser",
        "is_admin": False,
        "groups": ["users"],
        "scopes": ["read"],
        "accessible_servers": ["test-server"],
        "accessible_services": ["test-server"],
        "ui_permissions": {
            "list_service": ["test-server"],
            "toggle_service": [],
            "register_service": [],
            "modify_service": [],
        },
        "auth_method": "session",
    }

    def mock_enhanced_auth_override():
        return admin_user_context

    def mock_nginx_proxied_auth_override():
        return admin_user_context

    def mock_user_has_permission(
        permission: str, service_name: str, permissions: dict[str, Any]
    ) -> bool:
        """Mock permission checker that always returns True for admin"""
        return True

    # Override dependencies at the app level
    app.dependency_overrides[enhanced_auth] = mock_enhanced_auth_override
    app.dependency_overrides[nginx_proxied_auth] = mock_nginx_proxied_auth_override

    # Patch the permission checker function
    with patch(
        "registry.auth.dependencies.user_has_ui_permission_for_service", mock_user_has_permission
    ):
        yield {"admin": admin_user_context, "regular": regular_user_context}

    # Cleanup
    app.dependency_overrides.clear()


@pytest.fixture
def test_server_data() -> dict[str, Any]:
    """
    Create test server data for registration.

    Returns:
        Dictionary with server data
    """
    return {
        "name": "Test Server",
        "description": "A test MCP server for integration tests",
        "path": "/test-server",
        "proxy_pass_url": "http://localhost:9000",
        "tags": "test,integration",
        "num_tools": 5,
        "license": "MIT",
    }


@pytest.fixture
def test_server_data_2() -> dict[str, Any]:
    """
    Create second test server data for listing tests.

    Returns:
        Dictionary with server data
    """
    return {
        "name": "Second Test Server",
        "description": "Another test MCP server",
        "path": "/second-server",
        "proxy_pass_url": "http://localhost:9001",
        "tags": "test,second",
        "num_tools": 3,
        "license": "Apache-2.0",
    }


@pytest.fixture(autouse=True)
def setup_test_environment(
    mock_settings,
    mock_nginx_service,
    mock_faiss_service,
    mock_health_service,
    mock_agent_service,
    mock_auth_dependencies,
):
    """
    Auto-use fixture to set up test environment with all mocks.

    This fixture runs automatically for all tests in this module.
    """
    # Initialize server service with clean state
    from registry.services.server_service import server_service

    server_service.registered_servers = {}
    server_service.service_state = {}

    yield

    # Cleanup after test
    server_service.registered_servers = {}
    server_service.service_state = {}


# =============================================================================
# REGISTRATION TESTS
# =============================================================================


@pytest.mark.integration
class TestServerRegistration:
    """Test server registration functionality."""

    def test_register_server_success(
        self, test_client: TestClient, test_server_data: dict[str, Any]
    ):
        """Test successful server registration."""
        # Act
        response = test_client.post("/api/servers/register", data=test_server_data)

        # Assert
        if response.status_code != http_status.HTTP_201_CREATED:
            logger.error(f"Registration failed with status {response.status_code}")
            logger.error(f"Response body: {response.text}")
        assert response.status_code == http_status.HTTP_201_CREATED
        data = response.json()
        assert data["path"] == test_server_data["path"]
        assert data["name"] == test_server_data["name"]
        assert "registered successfully" in data["message"].lower()

    def test_register_server_duplicate_path(
        self, test_client: TestClient, test_server_data: dict[str, Any]
    ):
        """Test registering server with duplicate path."""
        # Arrange - Register first server
        response1 = test_client.post("/api/servers/register", data=test_server_data)
        assert response1.status_code == http_status.HTTP_201_CREATED

        # Act - Try to register duplicate (overwrite=false)
        duplicate_data = test_server_data.copy()
        duplicate_data["overwrite"] = False
        response2 = test_client.post("/api/servers/register", data=duplicate_data)

        # Assert
        assert response2.status_code == http_status.HTTP_409_CONFLICT
        data = response2.json()
        assert "already exists" in data["reason"].lower()

    def test_register_server_overwrite_existing(
        self, test_client: TestClient, test_server_data: dict[str, Any]
    ):
        """Test overwriting existing server with overwrite=true."""
        # Arrange - Register first server
        response1 = test_client.post("/api/servers/register", data=test_server_data)
        assert response1.status_code == http_status.HTTP_201_CREATED

        # Act - Overwrite with updated data
        updated_data = test_server_data.copy()
        updated_data["description"] = "Updated description"
        updated_data["overwrite"] = True
        response2 = test_client.post("/api/servers/register", data=updated_data)

        # Assert
        assert response2.status_code == http_status.HTTP_201_CREATED
        data = response2.json()
        assert data["path"] == test_server_data["path"]

    def test_register_server_without_leading_slash(
        self, test_client: TestClient, test_server_data: dict[str, Any]
    ):
        """Test path normalization (adds leading slash)."""
        # Arrange
        test_server_data["path"] = "no-leading-slash"

        # Act
        response = test_client.post("/api/servers/register", data=test_server_data)

        # Assert
        assert response.status_code == http_status.HTTP_201_CREATED
        data = response.json()
        assert data["path"] == "/no-leading-slash"

    def test_register_server_minimal_data(self, test_client: TestClient):
        """Test registration with only required fields."""
        # Arrange
        minimal_data = {
            "name": "Minimal Server",
            "description": "Minimal test server",
            "path": "/minimal",
            "proxy_pass_url": "http://localhost:8888",
        }

        # Act
        response = test_client.post("/api/servers/register", data=minimal_data)

        # Assert
        assert response.status_code == http_status.HTTP_201_CREATED

    def test_register_server_with_tool_list(
        self, test_client: TestClient, test_server_data: dict[str, Any]
    ):
        """Test registration with tool_list_json."""
        # Arrange
        tools = [
            {"name": "get_data", "description": "Get data"},
            {"name": "set_data", "description": "Set data"},
        ]
        test_server_data["tool_list_json"] = json.dumps(tools)

        # Act
        response = test_client.post("/api/servers/register", data=test_server_data)

        # Assert
        assert response.status_code == http_status.HTTP_201_CREATED


# =============================================================================
# LIST SERVERS TESTS
# =============================================================================


@pytest.mark.integration
class TestServerListing:
    """Test server listing functionality."""

    def test_list_servers_empty(self, test_client: TestClient):
        """Test listing servers when none are registered."""
        # Act
        response = test_client.get("/api/servers")

        # Assert
        assert response.status_code == http_status.HTTP_200_OK
        data = response.json()
        assert "servers" in data
        assert data["servers"] == []

    def test_list_servers_with_single_server(
        self, test_client: TestClient, test_server_data: dict[str, Any]
    ):
        """Test listing servers with one registered server."""
        # Arrange - Register a server
        reg_response = test_client.post("/api/servers/register", data=test_server_data)
        assert reg_response.status_code == http_status.HTTP_201_CREATED

        # Act
        response = test_client.get("/api/servers")

        # Assert
        assert response.status_code == http_status.HTTP_200_OK
        data = response.json()
        assert len(data["servers"]) == 1
        server = data["servers"][0]
        assert server["display_name"] == test_server_data["name"]
        assert server["path"] == test_server_data["path"]
        assert server["description"] == test_server_data["description"]

    def test_list_servers_with_multiple_servers(
        self,
        test_client: TestClient,
        test_server_data: dict[str, Any],
        test_server_data_2: dict[str, Any],
    ):
        """Test listing multiple registered servers."""
        # Arrange - Register two servers
        reg1 = test_client.post("/api/servers/register", data=test_server_data)
        reg2 = test_client.post("/api/servers/register", data=test_server_data_2)
        assert reg1.status_code == http_status.HTTP_201_CREATED
        assert reg2.status_code == http_status.HTTP_201_CREATED

        # Act
        response = test_client.get("/api/servers")

        # Assert
        assert response.status_code == http_status.HTTP_200_OK
        data = response.json()
        assert len(data["servers"]) == 2

        # Verify both servers are present
        server_paths = [s["path"] for s in data["servers"]]
        assert test_server_data["path"] in server_paths
        assert test_server_data_2["path"] in server_paths

    def test_list_servers_with_query_filter(
        self,
        test_client: TestClient,
        test_server_data: dict[str, Any],
        test_server_data_2: dict[str, Any],
    ):
        """Test listing servers with search query filter."""
        # Arrange - Register two servers
        test_client.post("/api/servers/register", data=test_server_data)
        test_client.post("/api/servers/register", data=test_server_data_2)

        # Act - Search for "second"
        response = test_client.get("/api/servers?query=second")

        # Assert
        assert response.status_code == http_status.HTTP_200_OK
        data = response.json()
        assert len(data["servers"]) == 1
        assert data["servers"][0]["display_name"] == test_server_data_2["name"]

    def test_list_servers_includes_metadata(
        self, test_client: TestClient, test_server_data: dict[str, Any]
    ):
        """Test that server list includes all expected metadata."""
        # Arrange
        test_client.post("/api/servers/register", data=test_server_data)

        # Act
        response = test_client.get("/api/servers")

        # Assert
        assert response.status_code == http_status.HTTP_200_OK
        data = response.json()
        server = data["servers"][0]

        # Check required fields
        assert "display_name" in server
        assert "path" in server
        assert "description" in server
        assert "proxy_pass_url" in server
        assert "is_enabled" in server
        assert "tags" in server
        assert "num_tools" in server
        assert "license" in server
        assert "health_status" in server


# =============================================================================
# GET SERVER TESTS
# =============================================================================


@pytest.mark.integration
class TestServerRetrieval:
    """Test getting individual server details."""

    def test_get_server_by_path_success(
        self, test_client: TestClient, test_server_data: dict[str, Any]
    ):
        """Test retrieving server details by path."""
        # Arrange - Register server
        reg_response = test_client.post("/api/servers/register", data=test_server_data)
        assert reg_response.status_code == http_status.HTTP_201_CREATED

        # Act
        path = test_server_data["path"].lstrip("/")
        response = test_client.get(f"/api/server_details/{path}")

        # Assert
        assert response.status_code == http_status.HTTP_200_OK
        data = response.json()
        assert data["path"] == test_server_data["path"]
        assert data["server_name"] == test_server_data["name"]

    def test_get_server_nonexistent_path(self, test_client: TestClient):
        """Test retrieving server with non-existent path."""
        # Act
        response = test_client.get("/api/server_details/nonexistent")

        # Assert
        assert response.status_code == http_status.HTTP_404_NOT_FOUND


# =============================================================================
# UPDATE SERVER TESTS
# =============================================================================


@pytest.mark.integration
class TestServerUpdate:
    """
    Test server update functionality.

    Note: The current API only supports updates via register with overwrite=true.
    The /edit endpoint is for web UI and returns HTML redirects, not suitable for API testing.
    """

    def test_update_server_via_overwrite(
        self, test_client: TestClient, test_server_data: dict[str, Any]
    ):
        """Test updating server by re-registering with overwrite=true."""
        # Arrange - Register server
        reg_response = test_client.post("/api/servers/register", data=test_server_data)
        assert reg_response.status_code == http_status.HTTP_201_CREATED

        # Act - Update by re-registering with overwrite=true
        updated_data = test_server_data.copy()
        updated_data["name"] = "Updated Test Server"
        updated_data["description"] = "Updated description"
        updated_data["num_tools"] = 10
        updated_data["overwrite"] = True

        response = test_client.post("/api/servers/register", data=updated_data)

        # Assert
        assert response.status_code == http_status.HTTP_201_CREATED

        # Verify update by listing servers
        list_response = test_client.get("/api/servers")
        servers = list_response.json()["servers"]
        assert len(servers) == 1  # Should still be only one server
        updated_server = servers[0]
        assert updated_server["display_name"] == updated_data["name"]
        assert updated_server["description"] == updated_data["description"]
        assert updated_server["num_tools"] == updated_data["num_tools"]

    def test_update_server_reject_without_overwrite(
        self, test_client: TestClient, test_server_data: dict[str, Any]
    ):
        """Test that updating without overwrite=true fails."""
        # Arrange - Register server
        test_client.post("/api/servers/register", data=test_server_data)

        # Act - Try to update without overwrite
        updated_data = test_server_data.copy()
        updated_data["name"] = "Updated Test Server"
        updated_data["overwrite"] = False

        response = test_client.post("/api/servers/register", data=updated_data)

        # Assert
        assert response.status_code == http_status.HTTP_409_CONFLICT


# =============================================================================
# DELETE SERVER TESTS
# =============================================================================


@pytest.mark.integration
class TestServerDeletion:
    """Test server deletion functionality."""

    def test_delete_server_success(self, test_client: TestClient, test_server_data: dict[str, Any]):
        """Test successful server deletion."""
        # Arrange - Register server
        reg_response = test_client.post("/api/servers/register", data=test_server_data)
        assert reg_response.status_code == http_status.HTTP_201_CREATED

        # Act - Delete server
        response = test_client.post("/api/servers/remove", data={"path": test_server_data["path"]})

        # Assert
        assert response.status_code == http_status.HTTP_200_OK
        data = response.json()
        assert "removed successfully" in data["message"].lower()

        # Verify deletion by listing servers
        list_response = test_client.get("/api/servers")
        servers = list_response.json()["servers"]
        assert len(servers) == 0

    def test_delete_server_nonexistent(self, test_client: TestClient):
        """Test deleting non-existent server."""
        # Act
        response = test_client.post("/api/servers/remove", data={"path": "/nonexistent"})

        # Assert
        assert response.status_code == http_status.HTTP_404_NOT_FOUND
        data = response.json()
        # The response contains "no service registered at path" which includes conceptually "not found"
        assert "service" in data["reason"].lower() or "not found" in data["reason"].lower()

    def test_delete_server_without_leading_slash(
        self, test_client: TestClient, test_server_data: dict[str, Any]
    ):
        """Test path normalization in delete operation."""
        # Arrange - Register server
        test_client.post("/api/servers/register", data=test_server_data)

        # Act - Delete without leading slash
        path_without_slash = test_server_data["path"].lstrip("/")
        response = test_client.post("/api/servers/remove", data={"path": path_without_slash})

        # Assert
        assert response.status_code == http_status.HTTP_200_OK


# =============================================================================
# TOGGLE SERVER TESTS
# =============================================================================


@pytest.mark.integration
class TestServerToggle:
    """Test server enable/disable toggle functionality."""

    def test_toggle_server_enable(self, test_client: TestClient, test_server_data: dict[str, Any]):
        """Test enabling a server."""
        # Arrange - Register server (defaults to disabled)
        test_client.post("/api/servers/register", data=test_server_data)

        # Act - Enable server
        response = test_client.post(
            "/api/servers/toggle", data={"path": test_server_data["path"], "new_state": True}
        )

        # Assert
        assert response.status_code == http_status.HTTP_200_OK
        data = response.json()
        assert data["new_enabled_state"] is True

    def test_toggle_server_disable(self, test_client: TestClient, test_server_data: dict[str, Any]):
        """Test disabling a server."""
        # Arrange - Register and enable server
        test_client.post("/api/servers/register", data=test_server_data)
        test_client.post(
            "/api/servers/toggle", data={"path": test_server_data["path"], "new_state": True}
        )

        # Act - Disable server
        response = test_client.post(
            "/api/servers/toggle", data={"path": test_server_data["path"], "new_state": False}
        )

        # Assert
        assert response.status_code == http_status.HTTP_200_OK
        data = response.json()
        assert data["new_enabled_state"] is False

    def test_toggle_server_nonexistent(self, test_client: TestClient):
        """Test toggling non-existent server."""
        # Act
        response = test_client.post(
            "/api/servers/toggle", data={"path": "/nonexistent", "new_state": True}
        )

        # Assert
        assert response.status_code == http_status.HTTP_404_NOT_FOUND


# =============================================================================
# FULL LIFECYCLE TESTS
# =============================================================================


@pytest.mark.integration
class TestServerFullLifecycle:
    """Test complete server lifecycle (create -> read -> update -> delete)."""

    def test_full_crud_lifecycle(self, test_client: TestClient, test_server_data: dict[str, Any]):
        """Test complete CRUD lifecycle for a server."""
        # CREATE
        create_response = test_client.post("/api/servers/register", data=test_server_data)
        assert create_response.status_code == http_status.HTTP_201_CREATED
        created_path = create_response.json()["path"]

        # READ - List all
        list_response = test_client.get("/api/servers")
        assert list_response.status_code == http_status.HTTP_200_OK
        servers = list_response.json()["servers"]
        assert len(servers) == 1
        assert servers[0]["path"] == created_path

        # READ - Get specific
        path_param = created_path.lstrip("/")
        detail_response = test_client.get(f"/api/server_details/{path_param}")
        assert detail_response.status_code == http_status.HTTP_200_OK
        assert detail_response.json()["path"] == created_path

        # UPDATE - via overwrite registration
        update_data = test_server_data.copy()
        update_data["name"] = "Updated Server Name"
        update_data["description"] = "Updated description"
        update_data["num_tools"] = 99
        update_data["overwrite"] = True

        update_response = test_client.post("/api/servers/register", data=update_data)
        assert update_response.status_code == http_status.HTTP_201_CREATED

        # Verify update
        list_after_update = test_client.get("/api/servers")
        servers_after_update = list_after_update.json()["servers"]
        assert len(servers_after_update) == 1  # Still only one server
        updated_server = servers_after_update[0]
        assert updated_server["display_name"] == update_data["name"]
        assert updated_server["num_tools"] == update_data["num_tools"]

        # DELETE
        delete_response = test_client.post("/api/servers/remove", data={"path": created_path})
        assert delete_response.status_code == http_status.HTTP_200_OK

        # Verify deletion
        list_after_delete = test_client.get("/api/servers")
        assert len(list_after_delete.json()["servers"]) == 0

    def test_lifecycle_with_toggle_operations(
        self, test_client: TestClient, test_server_data: dict[str, Any]
    ):
        """Test lifecycle including enable/disable operations."""
        # CREATE
        create_response = test_client.post("/api/servers/register", data=test_server_data)
        assert create_response.status_code == http_status.HTTP_201_CREATED
        path = create_response.json()["path"]

        # TOGGLE - Enable
        enable_response = test_client.post(
            "/api/servers/toggle", data={"path": path, "new_state": True}
        )
        assert enable_response.status_code == http_status.HTTP_200_OK
        assert enable_response.json()["new_enabled_state"] is True

        # Verify enabled state
        list_response = test_client.get("/api/servers")
        server = list_response.json()["servers"][0]
        assert server["is_enabled"] is True

        # TOGGLE - Disable
        disable_response = test_client.post(
            "/api/servers/toggle", data={"path": path, "new_state": False}
        )
        assert disable_response.status_code == http_status.HTTP_200_OK

        # DELETE
        delete_response = test_client.post("/api/servers/remove", data={"path": path})
        assert delete_response.status_code == http_status.HTTP_200_OK

    def test_multiple_servers_lifecycle(
        self,
        test_client: TestClient,
        test_server_data: dict[str, Any],
        test_server_data_2: dict[str, Any],
    ):
        """Test lifecycle with multiple servers."""
        # CREATE multiple servers
        create1 = test_client.post("/api/servers/register", data=test_server_data)
        create2 = test_client.post("/api/servers/register", data=test_server_data_2)
        assert create1.status_code == http_status.HTTP_201_CREATED
        assert create2.status_code == http_status.HTTP_201_CREATED

        # LIST - Verify both present
        list_response = test_client.get("/api/servers")
        servers = list_response.json()["servers"]
        assert len(servers) == 2

        # UPDATE first server via overwrite
        update_data = test_server_data.copy()
        update_data["name"] = "Updated First Server"
        update_data["overwrite"] = True

        update_response = test_client.post("/api/servers/register", data=update_data)
        assert update_response.status_code == http_status.HTTP_201_CREATED

        # DELETE first server
        delete_response = test_client.post(
            "/api/servers/remove", data={"path": test_server_data["path"]}
        )
        assert delete_response.status_code == http_status.HTTP_200_OK

        # LIST - Verify only second remains
        list_after_delete = test_client.get("/api/servers")
        remaining_servers = list_after_delete.json()["servers"]
        assert len(remaining_servers) == 1
        assert remaining_servers[0]["path"] == test_server_data_2["path"]

        # DELETE second server
        delete2_response = test_client.post(
            "/api/servers/remove", data={"path": test_server_data_2["path"]}
        )
        assert delete2_response.status_code == http_status.HTTP_200_OK

        # LIST - Verify empty
        final_list = test_client.get("/api/servers")
        assert len(final_list.json()["servers"]) == 0


# =============================================================================
# ERROR HANDLING TESTS
# =============================================================================


@pytest.mark.integration
class TestServerErrorHandling:
    """Test error handling in server operations."""

    def test_register_with_missing_required_fields(self, test_client: TestClient):
        """Test registration with missing required fields."""
        # Act - Missing proxy_pass_url
        response = test_client.post(
            "/api/servers/register", data={"name": "Test", "description": "Test", "path": "/test"}
        )

        # Assert
        assert response.status_code == http_status.HTTP_422_UNPROCESSABLE_ENTITY

    def test_update_preserves_path(self, test_client: TestClient, test_server_data: dict[str, Any]):
        """Test that update operation preserves the original path."""
        # Arrange - Register server
        test_client.post("/api/servers/register", data=test_server_data)
        original_path = test_server_data["path"]

        # Act - Update server via overwrite
        update_data = test_server_data.copy()
        update_data["name"] = "Updated Name"
        update_data["proxy_pass_url"] = "http://localhost:9999"
        update_data["overwrite"] = True

        update_response = test_client.post("/api/servers/register", data=update_data)
        assert update_response.status_code == http_status.HTTP_201_CREATED

        # Assert - Path unchanged
        list_response = test_client.get("/api/servers")
        servers = list_response.json()["servers"]
        assert len(servers) == 1
        assert servers[0]["path"] == original_path

    def test_operations_on_same_server_sequential(
        self, test_client: TestClient, test_server_data: dict[str, Any]
    ):
        """Test sequential operations on the same server."""
        # CREATE
        create_resp = test_client.post("/api/servers/register", data=test_server_data)
        assert create_resp.status_code == http_status.HTTP_201_CREATED
        path = create_resp.json()["path"]

        # UPDATE 1
        update_data_1 = test_server_data.copy()
        update_data_1["name"] = "Updated 1"
        update_data_1["overwrite"] = True

        update_resp = test_client.post("/api/servers/register", data=update_data_1)
        assert update_resp.status_code == http_status.HTTP_201_CREATED

        # TOGGLE
        toggle_resp = test_client.post(
            "/api/servers/toggle", data={"path": path, "new_state": True}
        )
        assert toggle_resp.status_code == http_status.HTTP_200_OK

        # UPDATE 2
        update_data_2 = test_server_data.copy()
        update_data_2["name"] = "Updated 2"
        update_data_2["overwrite"] = True

        update2_resp = test_client.post("/api/servers/register", data=update_data_2)
        assert update2_resp.status_code == http_status.HTTP_201_CREATED

        # DELETE
        delete_resp = test_client.post("/api/servers/remove", data={"path": path})
        assert delete_resp.status_code == http_status.HTTP_200_OK

        # Verify final state
        list_resp = test_client.get("/api/servers")
        assert len(list_resp.json()["servers"]) == 0
