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

from registry.utils.iam_errors import IdPNotFoundError

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

        # Assert - generic message returned, raw IdP error NOT reflected (stack-trace exposure)
        assert response.status_code == 502
        detail = response.json()["detail"]
        assert detail == "IAM provider error"
        assert "Connection refused" not in detail

    def test_dedup_skips_mongo_entries_matching_idp(self, test_client_admin):
        """MongoDB entries whose client_id already appears in IdP are skipped.

        Covers the dedup logic introduced in PR #942.
        """
        client, mock_iam = test_client_admin
        mock_iam.list_users.return_value = [
            {
                "id": "svc-1",
                "username": "service-1",
                "email": "service-1@example.com",
                "firstName": None,
                "lastName": None,
                "enabled": True,
                "groups": [],
            },
        ]

        mock_cursor = MagicMock()
        mock_cursor.to_list = AsyncMock(
            return_value=[{"client_id": "service-1", "name": "service-1"}]
        )
        mock_collection = MagicMock()
        mock_collection.find = MagicMock(return_value=mock_cursor)
        mock_db = MagicMock()
        mock_db.__getitem__.return_value = mock_collection

        with patch(
            "registry.api.management_routes.get_documentdb_client",
            new=AsyncMock(return_value=mock_db),
        ):
            response = client.get("/api/management/iam/users")

        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 1
        assert data["users"][0]["username"] == "service-1"

    def test_dedup_case_insensitive(self, test_client_admin):
        """Dedup matches usernames case-insensitively."""
        client, mock_iam = test_client_admin
        mock_iam.list_users.return_value = [
            {
                "id": "svc-1",
                "username": "Service-1",
                "email": "svc@example.com",
                "firstName": None,
                "lastName": None,
                "enabled": True,
                "groups": [],
            },
        ]

        mock_cursor = MagicMock()
        mock_cursor.to_list = AsyncMock(
            return_value=[{"client_id": "service-1", "name": "service-1"}]
        )
        mock_collection = MagicMock()
        mock_collection.find = MagicMock(return_value=mock_cursor)
        mock_db = MagicMock()
        mock_db.__getitem__.return_value = mock_collection

        with patch(
            "registry.api.management_routes.get_documentdb_client",
            new=AsyncMock(return_value=mock_db),
        ):
            response = client.get("/api/management/iam/users")

        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 1


# =============================================================================
# TEST DELETE /management/iam/users/{username} - Delete User
# =============================================================================


@pytest.mark.unit
@pytest.mark.api
class TestManagementDeleteUser:
    """Tests for DELETE /management/iam/users/{username} endpoint.

    Covers the orphan-delete behavior introduced in PR #942. The intra-admin
    last-admin guard is neutralized here (a no-op) so these tests exercise the
    orphan-delete path; the guard itself is covered by
    TestManagementDeleteUserSafetyGuards.
    """

    @pytest.fixture(autouse=True)
    def _neutralize_last_admin_guard(self):
        with patch(
            "registry.api.management_routes.admin_safety.assert_not_last_admin",
            new=AsyncMock(return_value=None),
        ):
            yield

    @staticmethod
    def _make_mock_db(deleted_count: int = 0):
        """Build a mock MongoDB database whose delete_one returns deleted_count."""
        delete_result = MagicMock()
        delete_result.deleted_count = deleted_count
        mock_collection = MagicMock()
        mock_collection.delete_one = AsyncMock(return_value=delete_result)
        mock_db = MagicMock()
        mock_db.__getitem__.return_value = mock_collection
        return mock_db, mock_collection

    def test_deletes_idp_only_user(self, test_client_admin):
        """IdP delete succeeds, MongoDB has no matching record - returns 200."""
        client, mock_iam = test_client_admin
        mock_iam.delete_user.return_value = True
        mock_db, mock_collection = self._make_mock_db(deleted_count=0)

        with patch(
            "registry.api.management_routes.get_documentdb_client",
            new=AsyncMock(return_value=mock_db),
        ):
            response = client.delete("/api/management/iam/users/human-user")

        assert response.status_code == 200
        mock_iam.delete_user.assert_called_once_with(username="human-user")
        mock_collection.delete_one.assert_called_once()

    def test_deletes_mongo_only_orphan(self, test_client_admin):
        """IdP raises 'not found', MongoDB delete succeeds - returns 200."""
        client, mock_iam = test_client_admin
        mock_iam.delete_user.side_effect = Exception("User not found in IdP")
        mock_db, mock_collection = self._make_mock_db(deleted_count=1)

        with patch(
            "registry.api.management_routes.get_documentdb_client",
            new=AsyncMock(return_value=mock_db),
        ):
            response = client.delete("/api/management/iam/users/orphan-client")

        assert response.status_code == 200
        mock_collection.delete_one.assert_called_once()

    def test_rejects_truly_missing_user(self, test_client_admin):
        """IdP raises 'not found', MongoDB has no record - returns 400."""
        client, mock_iam = test_client_admin
        mock_iam.delete_user.side_effect = Exception("User not found")
        mock_db, _ = self._make_mock_db(deleted_count=0)

        with patch(
            "registry.api.management_routes.get_documentdb_client",
            new=AsyncMock(return_value=mock_db),
        ):
            response = client.delete("/api/management/iam/users/ghost-user")

        assert response.status_code == 400
        assert "not found" in response.json()["detail"].lower()

    def test_propagates_non_not_found_idp_errors(self, test_client_admin):
        """IdP raises '502 Bad Gateway' - error is propagated, MongoDB not touched."""
        client, mock_iam = test_client_admin
        mock_iam.delete_user.side_effect = Exception("502 Bad Gateway")
        mock_db, mock_collection = self._make_mock_db(deleted_count=0)

        with patch(
            "registry.api.management_routes.get_documentdb_client",
            new=AsyncMock(return_value=mock_db),
        ):
            response = client.delete("/api/management/iam/users/some-user")

        assert response.status_code == 502
        detail = response.json()["detail"]
        assert detail == "IAM provider error"
        assert "Bad Gateway" not in detail
        mock_collection.delete_one.assert_not_called()

    def test_mongo_delete_filter_is_case_insensitive(self, test_client_admin):
        """The MongoDB delete filter uses a case-insensitive regex match.

        Covers the case-insensitive filter added as PR #942 follow-up (P1.3).
        """
        import re as _re

        client, mock_iam = test_client_admin
        mock_iam.delete_user.return_value = True
        mock_db, mock_collection = self._make_mock_db(deleted_count=1)

        with patch(
            "registry.api.management_routes.get_documentdb_client",
            new=AsyncMock(return_value=mock_db),
        ):
            response = client.delete("/api/management/iam/users/Mixed-Case")

        assert response.status_code == 200
        args, _kwargs = mock_collection.delete_one.call_args
        filter_doc = args[0]
        or_clauses = filter_doc["$or"]
        patterns = [clause.get("client_id") or clause.get("name") for clause in or_clauses]
        # Ensure each side of the $or is a compiled case-insensitive regex
        assert all(isinstance(p, _re.Pattern) for p in patterns)
        assert all(p.flags & _re.IGNORECASE for p in patterns)


# =============================================================================
# TEST DELETE /management/iam/users/{username} - Intra-admin safety guards
# =============================================================================


@pytest.mark.unit
@pytest.mark.api
class TestManagementDeleteUserSafetyGuards:
    """Self-deletion and last-admin lockout guards on user delete."""

    @staticmethod
    def _make_mock_db(deleted_count: int = 0):
        delete_result = MagicMock()
        delete_result.deleted_count = deleted_count
        mock_collection = MagicMock()
        mock_collection.delete_one = AsyncMock(return_value=delete_result)
        mock_db = MagicMock()
        mock_db.__getitem__.return_value = mock_collection
        return mock_db, mock_collection

    def test_admin_cannot_delete_own_account(self, test_client_admin):
        """An admin deleting their OWN account is rejected with 403.

        The admin fixture's username is 'admin'; deleting 'admin' must fail
        before any IdP/DB call.
        """
        client, mock_iam = test_client_admin

        response = client.delete("/api/management/iam/users/admin")

        assert response.status_code == 403
        assert "your own account" in response.json()["detail"].lower()
        mock_iam.delete_user.assert_not_called()

    def test_self_delete_check_is_case_insensitive(self, test_client_admin):
        client, mock_iam = test_client_admin

        response = client.delete("/api/management/iam/users/ADMIN")

        assert response.status_code == 403
        mock_iam.delete_user.assert_not_called()

    def test_deleting_last_admin_is_rejected(self, test_client_admin):
        """Deleting the final administrator is refused with 409."""
        from registry.services.admin_safety import AdminSafetyError

        client, mock_iam = test_client_admin

        with patch(
            "registry.api.management_routes.admin_safety.assert_not_last_admin",
            new=AsyncMock(
                side_effect=AdminSafetyError(409, "Refusing to remove the last administrator.")
            ),
        ):
            response = client.delete("/api/management/iam/users/other-admin")

        assert response.status_code == 409
        assert "last administrator" in response.json()["detail"].lower()
        mock_iam.delete_user.assert_not_called()

    def test_delete_fails_closed_when_admin_count_unknown(self, test_client_admin):
        """If the admin population can't be verified, the delete is refused."""
        from registry.services.admin_safety import AdminSafetyError

        client, mock_iam = test_client_admin

        with patch(
            "registry.api.management_routes.admin_safety.assert_not_last_admin",
            new=AsyncMock(
                side_effect=AdminSafetyError(
                    503, "Unable to verify administrator population; operation refused"
                )
            ),
        ):
            response = client.delete("/api/management/iam/users/other-user")

        assert response.status_code == 503
        mock_iam.delete_user.assert_not_called()

    def test_delete_non_admin_user_succeeds(self, test_client_admin):
        """A normal (non-self, non-last-admin) delete still works."""
        client, mock_iam = test_client_admin
        mock_iam.delete_user.return_value = True
        mock_db, _ = self._make_mock_db(deleted_count=0)

        with (
            patch(
                "registry.api.management_routes.admin_safety.assert_not_last_admin",
                new=AsyncMock(return_value=None),
            ),
            patch(
                "registry.api.management_routes.get_documentdb_client",
                new=AsyncMock(return_value=mock_db),
            ),
        ):
            response = client.delete("/api/management/iam/users/regular-user")

        assert response.status_code == 200
        mock_iam.delete_user.assert_called_once_with(username="regular-user")


# =============================================================================
# TEST PATCH /management/iam/users/{username}/groups - Safety + admin-grant audit
# =============================================================================


@pytest.mark.unit
@pytest.mark.api
class TestManagementUpdateUserGroupsSafety:
    """Last-admin demotion guard and admin-grant audit on group update."""

    def test_demoting_last_admin_is_rejected(self, test_client_admin):
        from registry.services.admin_safety import AdminSafetyError

        client, _ = test_client_admin

        with patch(
            "registry.api.management_routes.admin_safety.would_remove_last_admin_via_groups",
            new=AsyncMock(
                side_effect=AdminSafetyError(409, "Refusing to demote the last administrator.")
            ),
        ):
            response = client.patch(
                "/api/management/iam/users/only-admin/groups",
                json={"groups": ["readers"]},
            )

        assert response.status_code == 409
        assert "last administrator" in response.json()["detail"].lower()

    def test_group_update_emits_admin_grant_audit(self, test_client_admin):
        """Granting an admin group triggers the dedicated admin-grant audit."""
        client, mock_iam = test_client_admin
        mock_iam.update_user_groups = AsyncMock(
            return_value={
                "username": "promoted-user",
                "groups": ["mcp-registry-admin"],
                "added": ["mcp-registry-admin"],
                "removed": [],
            }
        )

        # M2M lookup returns nothing => IdP path.
        mock_collection = MagicMock()
        mock_collection.find_one = AsyncMock(return_value=None)
        mock_db = MagicMock()
        mock_db.__getitem__.return_value = mock_collection

        with (
            patch(
                "registry.api.management_routes.admin_safety.would_remove_last_admin_via_groups",
                new=AsyncMock(return_value=None),
            ),
            patch(
                "registry.api.management_routes.get_documentdb_client",
                new=AsyncMock(return_value=mock_db),
            ),
            patch(
                "registry.api.management_routes.admin_safety.desired_groups_grant_admin",
                new=AsyncMock(return_value=True),
            ) as mock_grant,
            patch(
                "registry.api.management_routes.set_audit_action",
            ) as mock_audit,
        ):
            response = client.patch(
                "/api/management/iam/users/promoted-user/groups",
                json={"groups": ["mcp-registry-admin"]},
            )

        assert response.status_code == 200
        mock_grant.assert_awaited()
        # A distinct admin-grant audit event was recorded.
        assert any(
            call.kwargs.get("metadata", {}).get("admin_grant") is True
            for call in mock_audit.call_args_list
        )

    def test_group_update_no_audit_for_non_admin_grant(self, test_client_admin):
        """A non-admin group change does NOT emit the admin-grant event."""
        client, mock_iam = test_client_admin
        mock_iam.update_user_groups = AsyncMock(
            return_value={
                "username": "user",
                "groups": ["readers"],
                "added": ["readers"],
                "removed": [],
            }
        )

        mock_collection = MagicMock()
        mock_collection.find_one = AsyncMock(return_value=None)
        mock_db = MagicMock()
        mock_db.__getitem__.return_value = mock_collection

        with (
            patch(
                "registry.api.management_routes.admin_safety.would_remove_last_admin_via_groups",
                new=AsyncMock(return_value=None),
            ),
            patch(
                "registry.api.management_routes.get_documentdb_client",
                new=AsyncMock(return_value=mock_db),
            ),
            patch(
                "registry.api.management_routes.admin_safety.desired_groups_grant_admin",
                new=AsyncMock(return_value=False),
            ),
            patch(
                "registry.api.management_routes.set_audit_action",
            ) as mock_audit,
        ):
            response = client.patch(
                "/api/management/iam/users/user/groups",
                json={"groups": ["readers"]},
            )

        assert response.status_code == 200
        assert not any(
            call.kwargs.get("metadata", {}).get("admin_grant") is True
            for call in mock_audit.call_args_list
        )


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
                json={
                    "name": "new-group",
                    "description": "A new test group",
                    "scope_config": {"create_in_idp": True},
                },
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
                is_idp_managed=True,
                allow_privileged=True,
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
                json={
                    "name": "new-group",
                    "description": "Entra test group",
                    "scope_config": {"create_in_idp": True},
                },
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
                is_idp_managed=True,
                allow_privileged=True,
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
            json={
                "name": "existing-group",
                "scope_config": {"create_in_idp": True},
            },
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
            json={
                "name": "new-group",
                "scope_config": {"create_in_idp": True},
            },
        )

        # Assert - generic message returned, raw IdP error NOT reflected (stack-trace exposure)
        assert response.status_code == 502
        detail = response.json()["detail"]
        assert detail == "IAM provider error"
        assert "IAM service unavailable" not in detail

    def test_create_group_scope_import_refusal_returns_400(self, test_client_admin):
        """A refused scope write (guard rejection) surfaces as an actionable 400.

        Previously this returned 200 with only a server-side warning, so the
        UI showed a success toast while nothing was persisted (issue #1494).
        """
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
                json={
                    "name": "partial-group",
                    "scope_config": {"create_in_idp": True},
                },
            )

            # Assert - refusal is surfaced, not swallowed into a success toast
            assert response.status_code == 400
            assert "was not saved" in response.json()["detail"]

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
                json={
                    "name": "minimal-group",
                    "scope_config": {"create_in_idp": True},
                },
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
                is_idp_managed=True,
                allow_privileged=True,
            )


# =============================================================================
# TEST POST /management/iam/groups - Create Group with create_in_idp flag
# =============================================================================


@pytest.mark.unit
@pytest.mark.api
class TestManagementCreateGroupCreateInIdp:
    """Tests for create_in_idp flag handling in group creation."""

    def test_create_group_with_create_in_idp_false(self, test_client_admin):
        """When create_in_idp is False, group should only be created in MongoDB."""
        # Arrange
        client, mock_iam = test_client_admin

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
                json={
                    "name": "local-only-group",
                    "description": "Local only group",
                    "scope_config": {"create_in_idp": False},
                },
            )

            # Assert
            assert response.status_code == 200
            data = response.json()
            assert data["name"] == "local-only-group"

            # IdP create_group should NOT have been called
            mock_iam.create_group.assert_not_called()

            # MongoDB scope should still be created with group name as mapping.
            # is_idp_managed=False is persisted so later PATCH/DELETE won't
            # call the IdP (see issue #946).
            mock_import_group.assert_called_once_with(
                scope_name="local-only-group",
                description="Local only group",
                group_mappings=["local-only-group"],
                server_access=[],
                ui_permissions={},
                agent_access=[],
                is_idp_managed=False,
                allow_privileged=True,
            )

    def test_create_group_with_create_in_idp_true(self, test_client_admin):
        """When create_in_idp is True, group should be created in both IdP and MongoDB."""
        # Arrange
        client, mock_iam = test_client_admin
        entra_group_id = "12345678-1234-1234-1234-123456789abc"
        mock_iam.create_group.return_value = {
            "id": entra_group_id,
            "name": "idp-group",
            "path": "/idp-group",
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
                json={
                    "name": "idp-group",
                    "description": "IdP group",
                    "scope_config": {"create_in_idp": True},
                },
            )

            # Assert
            assert response.status_code == 200
            data = response.json()
            assert data["id"] == entra_group_id

            # IdP create_group SHOULD have been called
            mock_iam.create_group.assert_called_once_with(
                group_name="idp-group",
                description="IdP group",
            )

            # MongoDB scope created with Entra Object ID as mapping
            mock_import_group.assert_called_once_with(
                scope_name="idp-group",
                description="IdP group",
                group_mappings=[entra_group_id],
                server_access=[],
                ui_permissions={},
                agent_access=[],
                is_idp_managed=True,
                allow_privileged=True,
            )

    def test_create_group_default_does_not_create_in_idp(self, test_client_admin):
        """When create_in_idp not in scope_config, default to NOT creating in IdP."""
        # Arrange
        client, mock_iam = test_client_admin

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
                json={"name": "default-group"},
            )

            # Assert
            assert response.status_code == 200
            data = response.json()
            assert data["name"] == "default-group"

            # IdP create_group should NOT be called (default is False)
            mock_iam.create_group.assert_not_called()

            # MongoDB scope should still be created with group name as mapping.
            # Default create_in_idp=False => is_idp_managed=False persisted.
            mock_import_group.assert_called_once_with(
                scope_name="default-group",
                description="",
                group_mappings=["default-group"],
                server_access=[],
                ui_permissions={},
                agent_access=[],
                is_idp_managed=False,
                allow_privileged=True,
            )


# =============================================================================
# TEST DELETE /management/iam/groups/{group_name} - Delete Group (with local-only)
# =============================================================================


@pytest.mark.unit
@pytest.mark.api
class TestManagementDeleteGroupLocalOnly:
    """Tests for deleting groups that only exist in MongoDB (local-only)."""

    def test_delete_local_only_group_succeeds(self, test_client_admin):
        """Delete succeeds when group only exists in MongoDB (IdP returns not found).

        Two paths both lead to success:
        - is_idp_managed=False persisted (issue #946): IdP call is skipped.
        - is_idp_managed=True but IdP raises not-found: typed exception
          fall-through in the route handler still lets MongoDB delete proceed.

        This test covers the second path (legacy record, IdP raises not found).
        """
        # Arrange
        client, mock_iam = test_client_admin
        # Provider managers translate a 404 error to IdPNotFoundError; the
        # route catches that typed exception and falls through.
        mock_iam.delete_group.side_effect = IdPNotFoundError("Group 'local-group' not found")

        with (
            patch(
                "registry.api.management_routes.scope_service.get_group",
                new_callable=AsyncMock,
                return_value={
                    "scope_name": "local-group",
                    "is_idp_managed": True,
                },
            ),
            patch(
                "registry.api.management_routes.scope_service.delete_group",
                new_callable=AsyncMock,
                return_value=True,
            ) as mock_delete_scope,
        ):
            # Act
            response = client.delete("/api/management/iam/groups/local-group")

            # Assert - should succeed because IdP "not found" is handled gracefully
            assert response.status_code == 200
            data = response.json()
            assert data["name"] == "local-group"

            # MongoDB deletion should still proceed
            mock_delete_scope.assert_called_once_with(
                group_name="local-group", remove_from_mappings=True
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

    def test_delete_group_not_found_in_idp_still_deletes_from_mongodb(self, test_client_admin):
        """Test that IdP 'not found' is handled gracefully (legacy record delete)."""
        # Arrange
        client, mock_iam = test_client_admin
        mock_iam.delete_group.side_effect = IdPNotFoundError("Group 'nonexistent' not found")

        with (
            patch(
                "registry.api.management_routes.scope_service.get_group",
                new_callable=AsyncMock,
                return_value={
                    "scope_name": "nonexistent",
                    "is_idp_managed": True,
                },
            ),
            patch(
                "registry.api.management_routes.scope_service.delete_group",
                new_callable=AsyncMock,
                return_value=True,
            ) as mock_delete_scope,
        ):
            # Act
            response = client.delete("/api/management/iam/groups/nonexistent")

            # Assert - should succeed because IdP "not found" is handled gracefully
            assert response.status_code == 200
            data = response.json()
            assert data["name"] == "nonexistent"

            # MongoDB deletion should still proceed
            mock_delete_scope.assert_called_once_with(
                group_name="nonexistent", remove_from_mappings=True
            )

    def test_delete_group_iam_error(self, test_client_admin):
        """Test error handling when IAM manager fails."""
        # Arrange
        client, mock_iam = test_client_admin
        mock_iam.delete_group.side_effect = Exception("IAM service error")

        # Act
        response = client.delete("/api/management/iam/groups/test-group")

        # Assert - generic message returned, raw IdP error NOT reflected (stack-trace exposure)
        assert response.status_code == 502
        detail = response.json()["detail"]
        assert detail == "IAM provider error"
        assert "IAM service error" not in detail

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


# =============================================================================
# TESTS - agent_access / ui_permissions wildcard normalization (#1189 follow-up)
# =============================================================================


@pytest.mark.unit
@pytest.mark.api
class TestAgentWildcardNormalization:
    """The agent-access wildcard convention is the literal string "all".
    Clients may submit "*" (from the PR #1191 dropdown's prior shape),
    "all" (admin scope convention), or "/*" (pre-fix UI artifact). All
    three must collapse to "all" before storage so the downstream filter
    at agent_routes.py and visibility.py (both of which only match
    "all") grant access correctly.

    Pre-fix, "_normalize_agent_path" prepended a slash to "*" -> "/*",
    which the resolver treated as a literal path. The group then
    silently filtered every agent.
    """

    def test_coerce_wildcard_recognizes_all(self):
        from registry.api.management_routes import _coerce_agent_wildcard

        assert _coerce_agent_wildcard("all") == "all"

    def test_coerce_wildcard_recognizes_star(self):
        from registry.api.management_routes import _coerce_agent_wildcard

        assert _coerce_agent_wildcard("*") == "all"

    def test_coerce_wildcard_recognizes_slash_star(self):
        from registry.api.management_routes import _coerce_agent_wildcard

        assert _coerce_agent_wildcard("/*") == "all"

    def test_coerce_wildcard_strips_whitespace(self):
        from registry.api.management_routes import _coerce_agent_wildcard

        assert _coerce_agent_wildcard("  *  ") == "all"

    def test_coerce_wildcard_rejects_literal_paths(self):
        from registry.api.management_routes import _coerce_agent_wildcard

        assert _coerce_agent_wildcard("/flight-booking") is None
        assert _coerce_agent_wildcard("agents/foo") is None

    def test_coerce_wildcard_rejects_non_strings(self):
        from registry.api.management_routes import _coerce_agent_wildcard

        assert _coerce_agent_wildcard(None) is None
        assert _coerce_agent_wildcard(42) is None
        assert _coerce_agent_wildcard([]) is None

    def test_normalize_agent_access_star_collapses_to_all(self):
        from registry.api.management_routes import _normalize_agent_paths_in_scope_config

        agent_access, _ = _normalize_agent_paths_in_scope_config(["*"], None)
        assert agent_access == ["all"]

    def test_normalize_agent_access_slash_star_collapses_to_all(self):
        """Existing broken scopes have '/*' stored. Saving the group
        again must auto-repair to 'all'."""
        from registry.api.management_routes import _normalize_agent_paths_in_scope_config

        agent_access, _ = _normalize_agent_paths_in_scope_config(["/*"], None)
        assert agent_access == ["all"]

    def test_normalize_agent_access_preserves_literal_paths(self):
        from registry.api.management_routes import _normalize_agent_paths_in_scope_config

        agent_access, _ = _normalize_agent_paths_in_scope_config(
            ["flight-booking", "/jewel-homes-support"], None
        )
        assert agent_access == ["/flight-booking", "/jewel-homes-support"]

    def test_normalize_agent_access_dedupes_wildcards(self):
        """A caller passing both 'all' and '*' must not produce ['all', 'all']."""
        from registry.api.management_routes import _normalize_agent_paths_in_scope_config

        agent_access, _ = _normalize_agent_paths_in_scope_config(["*", "all"], None)
        assert agent_access == ["all"]

    def test_normalize_agent_access_mixed_wildcard_and_paths(self):
        from registry.api.management_routes import _normalize_agent_paths_in_scope_config

        agent_access, _ = _normalize_agent_paths_in_scope_config(["*", "flight-booking"], None)
        assert agent_access == ["all", "/flight-booking"]

    def test_normalize_ui_permissions_collapses_wildcards(self):
        """ui_permissions previously only special-cased 'all'. After the
        fix, '*' and '/*' there must collapse the same way."""
        from registry.api.management_routes import _normalize_agent_paths_in_scope_config

        _, ui_permissions = _normalize_agent_paths_in_scope_config(
            None,
            {
                "list_agents": ["*"],
                "get_agent": ["/*"],
                "modify_agent": ["all", "/foo"],
            },
        )
        assert ui_permissions["list_agents"] == ["all"]
        assert ui_permissions["get_agent"] == ["all"]
        assert ui_permissions["modify_agent"] == ["all", "/foo"]

    def test_normalize_handles_none_and_empty(self):
        from registry.api.management_routes import _normalize_agent_paths_in_scope_config

        agent_access, ui_permissions = _normalize_agent_paths_in_scope_config(None, None)
        assert agent_access is None
        assert ui_permissions is None

        agent_access, _ = _normalize_agent_paths_in_scope_config([], None)
        assert agent_access == []


# =============================================================================
# TEST issue #1494 - Scope creation from the UI (reload + validated config)
# =============================================================================


@pytest.mark.unit
@pytest.mark.api
class TestScopeMutationsTriggerAuthReload:
    """Create/update/delete via /iam/groups reload the auth server (#1494)."""

    def test_create_group_triggers_reload(self, test_client_admin):
        """A successful create triggers exactly one auth-server reload."""
        client, mock_iam = test_client_admin
        mock_iam.create_group.return_value = {
            "id": "gid",
            "name": "reload-group",
            "path": "/reload-group",
            "attributes": None,
        }

        with (
            patch(
                "registry.api.management_routes.scope_service.import_group",
                new_callable=AsyncMock,
                return_value=True,
            ),
            patch(
                "registry.api.management_routes.scope_service.trigger_auth_server_reload",
                new_callable=AsyncMock,
                return_value=True,
            ) as mock_reload,
            patch("registry.api.management_routes.AUTH_PROVIDER", "keycloak"),
        ):
            response = client.post(
                "/api/management/iam/groups",
                json={
                    "name": "reload-group",
                    "description": "d",
                    "scope_config": {"create_in_idp": True},
                },
            )

        assert response.status_code == 200
        mock_reload.assert_awaited_once()

    def test_create_reload_failure_is_non_fatal(self, test_client_admin):
        """Reload returning False still yields 200 (persisted change wins)."""
        client, mock_iam = test_client_admin
        mock_iam.create_group.return_value = {
            "id": "gid",
            "name": "reload-fail-group",
            "path": "/reload-fail-group",
            "attributes": None,
        }

        with (
            patch(
                "registry.api.management_routes.scope_service.import_group",
                new_callable=AsyncMock,
                return_value=True,
            ),
            patch(
                "registry.api.management_routes.scope_service.trigger_auth_server_reload",
                new_callable=AsyncMock,
                return_value=False,
            ) as mock_reload,
            patch("registry.api.management_routes.AUTH_PROVIDER", "keycloak"),
        ):
            response = client.post(
                "/api/management/iam/groups",
                json={
                    "name": "reload-fail-group",
                    "scope_config": {"create_in_idp": True},
                },
            )

        assert response.status_code == 200
        mock_reload.assert_awaited_once()

    def test_create_refusal_returns_400_and_skips_reload(self, test_client_admin):
        """A refused import (guard) surfaces a 400 and does not reload."""
        client, mock_iam = test_client_admin
        mock_iam.create_group.return_value = {
            "id": "gid",
            "name": "wild-group",
            "path": "/wild-group",
            "attributes": None,
        }

        with (
            patch(
                "registry.api.management_routes.scope_service.import_group",
                new_callable=AsyncMock,
                return_value=False,
            ),
            patch(
                "registry.api.management_routes.scope_service.trigger_auth_server_reload",
                new_callable=AsyncMock,
                return_value=True,
            ) as mock_reload,
            patch("registry.api.management_routes.AUTH_PROVIDER", "keycloak"),
        ):
            response = client.post(
                "/api/management/iam/groups",
                json={
                    "name": "wild-group",
                    "scope_config": {
                        "server_access": [{"server": "*", "methods": [], "tools": "*"}]
                    },
                },
            )

        assert response.status_code == 400
        assert "was not saved" in response.json()["detail"]
        mock_reload.assert_not_awaited()

    def test_update_group_triggers_reload(self, test_client_admin):
        """A successful PATCH triggers exactly one auth-server reload."""
        client, _ = test_client_admin
        existing = {
            "id": "gid",
            "name": "upd-group",
            "scope_name": "upd-group",
            "description": "old",
            "is_idp_managed": False,
            "group_mappings": ["upd-group"],
            "agent_access": [],
        }

        with (
            patch(
                "registry.api.management_routes.scope_service.get_group",
                new_callable=AsyncMock,
                return_value=existing,
            ),
            patch(
                "registry.api.management_routes.scope_service.import_group",
                new_callable=AsyncMock,
                return_value=True,
            ),
            patch(
                "registry.api.management_routes.scope_service.trigger_auth_server_reload",
                new_callable=AsyncMock,
                return_value=True,
            ) as mock_reload,
        ):
            response = client.patch(
                "/api/management/iam/groups/upd-group",
                json={
                    "scope_config": {
                        "server_access": [
                            {"server": "api", "methods": ["tools/call"], "tools": ["t1"]}
                        ]
                    }
                },
            )

        assert response.status_code == 200
        mock_reload.assert_awaited_once()

    def test_update_refusal_returns_400_and_skips_reload(self, test_client_admin):
        """A refused import on PATCH surfaces a 400 and does not reload."""
        client, _ = test_client_admin
        existing = {
            "id": "gid",
            "name": "upd-wild",
            "scope_name": "upd-wild",
            "is_idp_managed": False,
            "group_mappings": ["upd-wild"],
            "agent_access": [],
        }

        with (
            patch(
                "registry.api.management_routes.scope_service.get_group",
                new_callable=AsyncMock,
                return_value=existing,
            ),
            patch(
                "registry.api.management_routes.scope_service.import_group",
                new_callable=AsyncMock,
                return_value=False,
            ),
            patch(
                "registry.api.management_routes.scope_service.trigger_auth_server_reload",
                new_callable=AsyncMock,
                return_value=True,
            ) as mock_reload,
        ):
            response = client.patch(
                "/api/management/iam/groups/upd-wild",
                json={
                    "scope_config": {
                        "server_access": [{"server": "*", "methods": [], "tools": "*"}]
                    }
                },
            )

        assert response.status_code == 400
        assert "was not updated" in response.json()["detail"]
        mock_reload.assert_not_awaited()

    def test_delete_group_triggers_reload(self, test_client_admin):
        """A successful delete triggers exactly one auth-server reload."""
        client, mock_iam = test_client_admin
        mock_iam.delete_group.return_value = True

        with (
            patch(
                "registry.api.management_routes.scope_service.get_group",
                new_callable=AsyncMock,
                return_value={"scope_name": "del-group", "is_idp_managed": True},
            ),
            patch(
                "registry.api.management_routes.scope_service.delete_group",
                new_callable=AsyncMock,
                return_value=True,
            ),
            patch(
                "registry.api.management_routes.scope_service.trigger_auth_server_reload",
                new_callable=AsyncMock,
                return_value=True,
            ) as mock_reload,
        ):
            response = client.delete("/api/management/iam/groups/del-group")

        assert response.status_code == 200
        mock_reload.assert_awaited_once()


@pytest.mark.unit
@pytest.mark.api
class TestScopeConfigValidation:
    """Typed ScopeConfig on the IAM group endpoints (#1494)."""

    def test_invalid_scope_config_returns_422(self, test_client_admin):
        """Unknown scope_config keys are rejected with a 422, not silently dropped."""
        client, _ = test_client_admin

        response = client.post(
            "/api/management/iam/groups",
            json={
                "name": "bad-group",
                "scope_config": {"sever_access": []},  # typo'd key
            },
        )

        assert response.status_code == 422
        assert "sever_access" in str(response.json()["detail"])

    def test_malformed_server_access_rule_returns_422(self, test_client_admin):
        """A server_access entry missing both server and agent is a 422."""
        client, _ = test_client_admin

        response = client.post(
            "/api/management/iam/groups",
            json={
                "name": "bad-rule-group",
                "scope_config": {"server_access": [{"methods": ["tools/call"]}]},
            },
        )

        assert response.status_code == 422

    def test_scope_config_applied(self, test_client_admin):
        """server_access / ui_permissions / agent_access reach import_group."""
        client, mock_iam = test_client_admin
        mock_iam.create_group.return_value = {
            "id": "gid",
            "name": "applied-group",
            "path": "/applied-group",
            "attributes": None,
        }

        with (
            patch(
                "registry.api.management_routes.scope_service.import_group",
                new_callable=AsyncMock,
                return_value=True,
            ) as mock_import,
            patch(
                "registry.api.management_routes.scope_service.trigger_auth_server_reload",
                new_callable=AsyncMock,
                return_value=True,
            ),
            patch("registry.api.management_routes.AUTH_PROVIDER", "keycloak"),
        ):
            response = client.post(
                "/api/management/iam/groups",
                json={
                    "name": "applied-group",
                    "description": "d",
                    "scope_config": {
                        "create_in_idp": True,
                        "server_access": [
                            {
                                "server": "currenttime",
                                "methods": ["initialize", "tools/call"],
                                "tools": ["current_time_by_timezone"],
                            },
                            {"agent": "/my-planner", "actions": ["invoke_agent"]},
                        ],
                        "ui_permissions": {"list_service": ["currenttime"]},
                        "agent_access": ["/my-planner"],
                    },
                },
            )

        assert response.status_code == 200
        kwargs = mock_import.await_args.kwargs
        assert kwargs["server_access"] == [
            {
                "server": "currenttime",
                "methods": ["initialize", "tools/call"],
                "tools": ["current_time_by_timezone"],
            },
            {"agent": "/my-planner", "actions": ["invoke_agent"]},
        ]
        assert kwargs["ui_permissions"] == {"list_service": ["currenttime"]}
        assert kwargs["agent_access"] == ["/my-planner"]

    def test_update_omitted_fields_preserve_existing(self, test_client_admin):
        """PATCH with a partial scope_config preserves omitted fields.

        Critical for group_mappings, which holds Entra Object IDs.
        """
        client, _ = test_client_admin
        existing = {
            "id": "gid",
            "name": "partial-upd",
            "scope_name": "partial-upd",
            "description": "old",
            "is_idp_managed": False,
            "group_mappings": ["12345678-entra-object-id"],
            "agent_access": ["/kept-agent"],
        }

        with (
            patch(
                "registry.api.management_routes.scope_service.get_group",
                new_callable=AsyncMock,
                return_value=existing,
            ),
            patch(
                "registry.api.management_routes.scope_service.import_group",
                new_callable=AsyncMock,
                return_value=True,
            ) as mock_import,
            patch(
                "registry.api.management_routes.scope_service.trigger_auth_server_reload",
                new_callable=AsyncMock,
                return_value=True,
            ),
        ):
            response = client.patch(
                "/api/management/iam/groups/partial-upd",
                json={"scope_config": {"ui_permissions": {"list_service": ["api"]}}},
            )

        assert response.status_code == 200
        kwargs = mock_import.await_args.kwargs
        assert kwargs["group_mappings"] == ["12345678-entra-object-id"]
        assert kwargs["agent_access"] == ["/kept-agent"]
        assert kwargs["server_access"] is None  # omitted -> preserved downstream
        assert kwargs["ui_permissions"] == {"list_service": ["api"]}
