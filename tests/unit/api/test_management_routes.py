"""
Unit tests for registry/api/management_routes.py

Tests the IAM-related management endpoints including:
- GET /management/iam/users - List users from identity provider
- POST /management/iam/groups - Create group in IdP and MongoDB
- DELETE /management/iam/groups/{group_name} - Delete group from IdP and MongoDB
- GET /management/iam/groups - List groups from identity provider
"""

import logging
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient


logger = logging.getLogger(__name__)


# =============================================================================
# AUTH MOCK FIXTURES
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
        "ui_permissions": {
            "list_service": ["test-server"],
            "view_tools": ["test-server"],
        },
        "auth_method": "session",
    }


@pytest.fixture
def mock_auth_admin(admin_user_context, mock_settings):
    """
    Mock authentication dependencies with admin user.

    Note: depends on mock_settings to ensure environment is set up before importing app.
    """
    from registry.auth.dependencies import nginx_proxied_auth
    from registry.main import app

    def mock_nginx_proxied_auth_override():
        return admin_user_context

    app.dependency_overrides[nginx_proxied_auth] = mock_nginx_proxied_auth_override

    yield admin_user_context

    app.dependency_overrides.clear()


@pytest.fixture
def mock_auth_regular(regular_user_context, mock_settings):
    """
    Mock authentication dependencies with regular user.

    Note: depends on mock_settings to ensure environment is set up before importing app.
    """
    from registry.auth.dependencies import nginx_proxied_auth
    from registry.main import app

    def mock_nginx_proxied_auth_override():
        return regular_user_context

    app.dependency_overrides[nginx_proxied_auth] = mock_nginx_proxied_auth_override

    yield regular_user_context

    app.dependency_overrides.clear()


# =============================================================================
# IAM MANAGER MOCK FIXTURES
# =============================================================================


@pytest.fixture
def mock_iam_manager():
    """Create a mock IAM manager for testing."""
    mock = MagicMock()
    mock.list_users = AsyncMock(return_value=[])
    mock.list_groups = AsyncMock(return_value=[])
    mock.create_group = AsyncMock(
        return_value={
            "id": "test-group-id",
            "name": "test-group",
            "path": "/test-group",
            "attributes": None,
        }
    )
    mock.delete_group = AsyncMock(return_value=True)
    mock.create_human_user = AsyncMock(
        return_value={
            "id": "test-user-id",
            "username": "testuser",
            "email": "test@example.com",
            "firstName": "Test",
            "lastName": "User",
            "enabled": True,
            "groups": ["test-group"],
        }
    )
    mock.delete_user = AsyncMock(return_value=True)
    mock.create_service_account = AsyncMock(
        return_value={
            "client_id": "test-client",
            "client_secret": "test-secret",
            "groups": ["test-group"],
        }
    )
    return mock


# =============================================================================
# TEST CLIENT FIXTURES
# =============================================================================


@pytest.fixture
def test_client_admin(mock_settings, mock_auth_admin, mock_iam_manager):
    """Create FastAPI test client with admin auth and IAM manager mocked."""
    with patch(
        "registry.api.management_routes.get_iam_manager",
        return_value=mock_iam_manager,
    ):
        from registry.main import app

        client = TestClient(app, cookies={"mcp_gateway_session": "test-session"})
        yield client, mock_iam_manager


@pytest.fixture
def test_client_regular(mock_settings, mock_auth_regular, mock_iam_manager):
    """Create FastAPI test client with regular user auth and IAM manager mocked."""
    with patch(
        "registry.api.management_routes.get_iam_manager",
        return_value=mock_iam_manager,
    ):
        from registry.main import app

        client = TestClient(app, cookies={"mcp_gateway_session": "test-session"})
        yield client, mock_iam_manager


# =============================================================================
# TEST GET /management/iam/users - List Users
# =============================================================================


@pytest.mark.unit
@pytest.mark.api
class TestManagementListUsers:
    """Tests for GET /management/iam/users endpoint."""

    def test_list_users_success(self, test_client_admin):
        """Test successful listing of users."""
        # Arrange
        client, mock_iam = test_client_admin
        mock_iam.list_users.return_value = [
            {
                "id": "user-1",
                "username": "user1",
                "email": "user1@example.com",
                "firstName": "User",
                "lastName": "One",
                "enabled": True,
                "groups": ["group-a"],
            },
            {
                "id": "user-2",
                "username": "user2",
                "email": "user2@example.com",
                "firstName": "User",
                "lastName": "Two",
                "enabled": False,
                "groups": [],
            },
        ]

        # Act
        response = client.get("/api/management/iam/users")

        # Assert
        assert response.status_code == 200
        data = response.json()
        assert "users" in data
        assert "total" in data
        assert data["total"] == 2
        assert len(data["users"]) == 2
        assert data["users"][0]["username"] == "user1"
        assert data["users"][1]["username"] == "user2"
        mock_iam.list_users.assert_called_once_with(search=None, max_results=500)

    def test_list_users_with_search(self, test_client_admin):
        """Test listing users with search parameter."""
        # Arrange
        client, mock_iam = test_client_admin
        mock_iam.list_users.return_value = [
            {
                "id": "user-1",
                "username": "john",
                "email": "john@example.com",
                "firstName": "John",
                "lastName": "Doe",
                "enabled": True,
                "groups": [],
            },
        ]

        # Act
        response = client.get("/api/management/iam/users?search=john&limit=100")

        # Assert
        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 1
        mock_iam.list_users.assert_called_once_with(search="john", max_results=100)

    def test_list_users_requires_admin(self, test_client_regular):
        """Test that listing users requires admin permissions."""
        # Arrange
        client, _ = test_client_regular

        # Act
        response = client.get("/api/management/iam/users")

        # Assert
        assert response.status_code == 403
        assert "Administrator permissions" in response.json()["detail"]

    def test_list_users_iam_error(self, test_client_admin):
        """Test error handling when IAM manager fails."""
        # Arrange
        client, mock_iam = test_client_admin
        mock_iam.list_users.side_effect = Exception("Connection refused")

        # Act
        response = client.get("/api/management/iam/users")

        # Assert
        assert response.status_code == 502
        assert "Connection refused" in response.json()["detail"]


# =============================================================================
# TEST GET /management/iam/groups - List Groups
# =============================================================================


@pytest.mark.unit
@pytest.mark.api
class TestManagementListGroups:
    """Tests for GET /management/iam/groups endpoint."""

    def test_list_groups_success(self, test_client_admin):
        """Test successful listing of groups."""
        # Arrange
        client, mock_iam = test_client_admin
        mock_iam.list_groups.return_value = [
            {
                "id": "group-1",
                "name": "developers",
                "path": "/developers",
                "attributes": {"department": ["engineering"]},
            },
            {
                "id": "group-2",
                "name": "admins",
                "path": "/admins",
                "attributes": None,
            },
        ]

        # Act
        response = client.get("/api/management/iam/groups")

        # Assert
        assert response.status_code == 200
        data = response.json()
        assert "groups" in data
        assert "total" in data
        assert data["total"] == 2
        assert len(data["groups"]) == 2
        assert data["groups"][0]["name"] == "developers"
        assert data["groups"][1]["name"] == "admins"
        mock_iam.list_groups.assert_called_once()

    def test_list_groups_returns_group_summary(self, test_client_admin):
        """Test that groups are returned as GroupSummary objects."""
        # Arrange
        client, mock_iam = test_client_admin
        mock_iam.list_groups.return_value = [
            {
                "id": "test-id",
                "name": "test-group",
                "path": "/test-group",
                "attributes": {"key": ["value"]},
            },
        ]

        # Act
        response = client.get("/api/management/iam/groups")

        # Assert
        assert response.status_code == 200
        data = response.json()
        group = data["groups"][0]
        assert "id" in group
        assert "name" in group
        assert "path" in group
        assert "attributes" in group
        assert group["id"] == "test-id"
        assert group["name"] == "test-group"
        assert group["path"] == "/test-group"

    def test_list_groups_requires_admin(self, test_client_regular):
        """Test that listing groups requires admin permissions."""
        # Arrange
        client, _ = test_client_regular

        # Act
        response = client.get("/api/management/iam/groups")

        # Assert
        assert response.status_code == 403
        assert "Administrator permissions" in response.json()["detail"]

    def test_list_groups_iam_error(self, test_client_admin):
        """Test error handling when IAM manager fails."""
        # Arrange
        client, mock_iam = test_client_admin
        mock_iam.list_groups.side_effect = Exception("Keycloak unavailable")

        # Act
        response = client.get("/api/management/iam/groups")

        # Assert
        assert response.status_code == 502
        assert "Unable to list IAM groups" in response.json()["detail"]


# =============================================================================
# TEST POST /management/iam/groups - Create Group
# =============================================================================


@pytest.mark.unit
@pytest.mark.api
class TestManagementCreateGroup:
    """Tests for POST /management/iam/groups endpoint."""

    def test_create_group_success_keycloak(self, test_client_admin):
        """Test successful group creation with Keycloak provider."""
        # Arrange
        client, mock_iam = test_client_admin
        mock_iam.create_group.return_value = {
            "id": "new-group-id",
            "name": "new-group",
            "path": "/new-group",
            "attributes": None,
        }

        with (
            patch(
                "registry.api.management_routes.scope_service.import_group",
                new_callable=AsyncMock,
                return_value=True,
            ) as mock_import_group,
            patch("registry.api.management_routes.AUTH_PROVIDER", "keycloak"),
        ):
            # Act
            response = client.post(
                "/api/management/iam/groups",
                json={"name": "new-group", "description": "A new test group"},
            )

            # Assert
            assert response.status_code == 200
            data = response.json()
            assert data["id"] == "new-group-id"
            assert data["name"] == "new-group"

            mock_iam.create_group.assert_called_once_with(
                group_name="new-group", description="A new test group"
            )

            # Keycloak uses group name in group_mappings
            mock_import_group.assert_called_once_with(
                scope_name="new-group",
                description="A new test group",
                group_mappings=["new-group"],
                server_access=[],
                ui_permissions={},
                agent_access=[],
            )

    def test_create_group_success_entra(self, test_client_admin):
        """Test successful group creation with Entra ID provider."""
        # Arrange
        client, mock_iam = test_client_admin
        entra_group_id = "12345678-1234-1234-1234-123456789abc"
        mock_iam.create_group.return_value = {
            "id": entra_group_id,
            "name": "new-group",
            "path": "/new-group",
            "attributes": None,
        }

        with (
            patch(
                "registry.api.management_routes.scope_service.import_group",
                new_callable=AsyncMock,
                return_value=True,
            ) as mock_import_group,
            patch("registry.api.management_routes.AUTH_PROVIDER", "entra"),
        ):
            # Act
            response = client.post(
                "/api/management/iam/groups",
                json={"name": "new-group", "description": "Entra test group"},
            )

            # Assert
            assert response.status_code == 200
            data = response.json()
            assert data["id"] == entra_group_id
            assert data["name"] == "new-group"

            mock_iam.create_group.assert_called_once_with(
                group_name="new-group", description="Entra test group"
            )

            # Entra ID uses Object ID (GUID) in group_mappings
            mock_import_group.assert_called_once_with(
                scope_name="new-group",
                description="Entra test group",
                group_mappings=[entra_group_id],
                server_access=[],
                ui_permissions={},
                agent_access=[],
            )

    def test_create_group_requires_admin(self, test_client_regular):
        """Test that creating groups requires admin permissions."""
        # Arrange
        client, _ = test_client_regular

        # Act
        response = client.post(
            "/api/management/iam/groups",
            json={"name": "new-group"},
        )

        # Assert
        assert response.status_code == 403
        assert "Administrator permissions" in response.json()["detail"]

    def test_create_group_already_exists(self, test_client_admin):
        """Test error handling when group already exists."""
        # Arrange
        client, mock_iam = test_client_admin
        mock_iam.create_group.side_effect = Exception("Group 'existing-group' already exists")

        # Act
        response = client.post(
            "/api/management/iam/groups",
            json={"name": "existing-group"},
        )

        # Assert
        assert response.status_code == 400
        assert "already exists" in response.json()["detail"]

    def test_create_group_iam_error(self, test_client_admin):
        """Test error handling when IAM manager fails."""
        # Arrange
        client, mock_iam = test_client_admin
        mock_iam.create_group.side_effect = Exception("IAM service unavailable")

        # Act
        response = client.post(
            "/api/management/iam/groups",
            json={"name": "new-group"},
        )

        # Assert
        assert response.status_code == 502
        assert "IAM service unavailable" in response.json()["detail"]

    def test_create_group_scope_import_failure_logs_warning(self, test_client_admin):
        """Test that scope import failure is logged but doesn't fail the request."""
        # Arrange
        client, mock_iam = test_client_admin
        mock_iam.create_group.return_value = {
            "id": "group-id",
            "name": "partial-group",
            "path": "/partial-group",
            "attributes": None,
        }

        with (
            patch(
                "registry.api.management_routes.scope_service.import_group",
                new_callable=AsyncMock,
                return_value=False,
            ),
            patch("registry.api.management_routes.AUTH_PROVIDER", "keycloak"),
        ):
            # Act
            response = client.post(
                "/api/management/iam/groups",
                json={"name": "partial-group"},
            )

            # Assert - should still succeed (IdP creation succeeded)
            assert response.status_code == 200
            data = response.json()
            assert data["name"] == "partial-group"

    def test_create_group_without_description(self, test_client_admin):
        """Test group creation without description uses empty string."""
        # Arrange
        client, mock_iam = test_client_admin
        mock_iam.create_group.return_value = {
            "id": "group-id",
            "name": "minimal-group",
            "path": "/minimal-group",
            "attributes": None,
        }

        with (
            patch(
                "registry.api.management_routes.scope_service.import_group",
                new_callable=AsyncMock,
                return_value=True,
            ) as mock_import_group,
            patch("registry.api.management_routes.AUTH_PROVIDER", "keycloak"),
        ):
            # Act
            response = client.post(
                "/api/management/iam/groups",
                json={"name": "minimal-group"},
            )

            # Assert
            assert response.status_code == 200
            mock_iam.create_group.assert_called_once_with(
                group_name="minimal-group", description=""
            )
            mock_import_group.assert_called_once_with(
                scope_name="minimal-group",
                description="",
                group_mappings=["minimal-group"],
                server_access=[],
                ui_permissions={},
                agent_access=[],
            )


# =============================================================================
# TEST DELETE /management/iam/groups/{group_name} - Delete Group
# =============================================================================


@pytest.mark.unit
@pytest.mark.api
class TestManagementDeleteGroup:
    """Tests for DELETE /management/iam/groups/{group_name} endpoint."""

    def test_delete_group_success(self, test_client_admin):
        """Test successful group deletion."""
        # Arrange
        client, mock_iam = test_client_admin
        mock_iam.delete_group.return_value = True

        with patch(
            "registry.api.management_routes.scope_service.delete_group",
            new_callable=AsyncMock,
            return_value=True,
        ) as mock_delete_scope:
            # Act
            response = client.delete("/api/management/iam/groups/test-group")

            # Assert
            assert response.status_code == 200
            data = response.json()
            assert data["name"] == "test-group"
            assert data["deleted"] is True

            mock_iam.delete_group.assert_called_once_with(group_name="test-group")
            mock_delete_scope.assert_called_once_with(
                group_name="test-group", remove_from_mappings=True
            )

    def test_delete_group_requires_admin(self, test_client_regular):
        """Test that deleting groups requires admin permissions."""
        # Arrange
        client, _ = test_client_regular

        # Act
        response = client.delete("/api/management/iam/groups/test-group")

        # Assert
        assert response.status_code == 403
        assert "Administrator permissions" in response.json()["detail"]

    def test_delete_group_not_found(self, test_client_admin):
        """Test error handling when group is not found."""
        # Arrange
        client, mock_iam = test_client_admin
        mock_iam.delete_group.side_effect = Exception("Group 'nonexistent' not found")

        # Act
        response = client.delete("/api/management/iam/groups/nonexistent")

        # Assert
        assert response.status_code == 404
        assert "not found" in response.json()["detail"]

    def test_delete_group_iam_error(self, test_client_admin):
        """Test error handling when IAM manager fails."""
        # Arrange
        client, mock_iam = test_client_admin
        mock_iam.delete_group.side_effect = Exception("IAM service error")

        # Act
        response = client.delete("/api/management/iam/groups/test-group")

        # Assert
        assert response.status_code == 502
        assert "IAM service error" in response.json()["detail"]

    def test_delete_group_scope_deletion_failure_logs_warning(self, test_client_admin):
        """Test that scope deletion failure is logged but doesn't fail the request."""
        # Arrange
        client, mock_iam = test_client_admin
        mock_iam.delete_group.return_value = True

        with patch(
            "registry.api.management_routes.scope_service.delete_group",
            new_callable=AsyncMock,
            return_value=False,
        ):
            # Act
            response = client.delete("/api/management/iam/groups/partial-delete")

            # Assert - should still succeed (IdP deletion succeeded)
            assert response.status_code == 200
            data = response.json()
            assert data["name"] == "partial-delete"
            assert data["deleted"] is True


# =============================================================================
# TEST HELPER FUNCTIONS
# =============================================================================


@pytest.mark.unit
@pytest.mark.api
class TestManagementHelpers:
    """Tests for management routes helper functions."""

    def test_translate_iam_error_already_exists(self):
        """Test error translation for 'already exists' errors."""
        from registry.api.management_routes import _translate_iam_error

        exc = Exception("Group 'test' already exists in Keycloak")
        http_exc = _translate_iam_error(exc)
        assert http_exc.status_code == 400

    def test_translate_iam_error_not_found(self):
        """Test error translation for 'not found' errors."""
        from registry.api.management_routes import _translate_iam_error

        exc = Exception("User not found in identity provider")
        http_exc = _translate_iam_error(exc)
        assert http_exc.status_code == 400

    def test_translate_iam_error_generic(self):
        """Test error translation for generic errors."""
        from registry.api.management_routes import _translate_iam_error

        exc = Exception("Connection timeout to Keycloak")
        http_exc = _translate_iam_error(exc)
        assert http_exc.status_code == 502

    def test_require_admin_passes_for_admin(self, admin_user_context):
        """Test _require_admin passes for admin users."""
        from registry.api.management_routes import _require_admin

        # Should not raise
        _require_admin(admin_user_context)

    def test_require_admin_raises_for_non_admin(self, regular_user_context):
        """Test _require_admin raises HTTPException for non-admin users."""
        from fastapi import HTTPException

        from registry.api.management_routes import _require_admin

        with pytest.raises(HTTPException) as exc_info:
            _require_admin(regular_user_context)

        assert exc_info.value.status_code == 403
        assert "Administrator permissions" in exc_info.value.detail
