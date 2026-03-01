"""
Unit tests for registry.utils.iam_manager module.

This module tests the IAM manager factory and provider-specific implementations
(KeycloakIAMManager, EntraIAMManager) with mocked underlying provider functions.
"""

import logging
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

logger = logging.getLogger(__name__)


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def sample_user_list() -> list[dict[str, Any]]:
    """
    Create sample user list for testing.

    Returns:
        List of user dictionaries
    """
    return [
        {
            "id": "user-1",
            "username": "testuser1",
            "email": "user1@example.com",
            "firstName": "Test",
            "lastName": "User1",
            "groups": ["admin", "developers"],
        },
        {
            "id": "user-2",
            "username": "testuser2",
            "email": "user2@example.com",
            "firstName": "Test",
            "lastName": "User2",
            "groups": ["developers"],
        },
    ]


@pytest.fixture
def sample_group_list() -> list[dict[str, Any]]:
    """
    Create sample group list for testing.

    Returns:
        List of group dictionaries
    """
    return [
        {
            "id": "group-1",
            "name": "admin",
            "description": "Admin group",
        },
        {
            "id": "group-2",
            "name": "developers",
            "description": "Developers group",
        },
    ]


@pytest.fixture
def sample_created_group() -> dict[str, Any]:
    """
    Create sample created group response for testing.

    Returns:
        Dictionary with created group details
    """
    return {
        "id": "new-group-id",
        "name": "new-group",
        "description": "A new group",
    }


@pytest.fixture
def sample_created_user() -> dict[str, Any]:
    """
    Create sample created user response for testing.

    Returns:
        Dictionary with created user details
    """
    return {
        "id": "new-user-id",
        "username": "newuser",
        "email": "newuser@example.com",
        "firstName": "New",
        "lastName": "User",
        "groups": ["developers"],
    }


@pytest.fixture
def sample_service_account() -> dict[str, Any]:
    """
    Create sample service account response for testing.

    Returns:
        Dictionary with service account details
    """
    return {
        "client_id": "test-service-account",
        "client_secret": "generated-secret-123",
        "groups": ["api-access"],
    }


# =============================================================================
# TEST: get_iam_manager() Factory Function
# =============================================================================


@pytest.mark.unit
class TestGetIAMManagerFactory:
    """Test the get_iam_manager factory function."""

    def test_returns_keycloak_manager_when_auth_provider_is_keycloak(
        self,
        monkeypatch,
    ):
        """Test returns KeycloakIAMManager when AUTH_PROVIDER is 'keycloak'."""
        # Arrange
        monkeypatch.setenv("AUTH_PROVIDER", "keycloak")

        # Need to reimport to pick up the new env var
        import importlib
        import registry.utils.iam_manager as iam_module

        importlib.reload(iam_module)

        # Act
        manager = iam_module.get_iam_manager()

        # Assert
        assert isinstance(manager, iam_module.KeycloakIAMManager)

    def test_returns_entra_manager_when_auth_provider_is_entra(
        self,
        monkeypatch,
    ):
        """Test returns EntraIAMManager when AUTH_PROVIDER is 'entra'."""
        # Arrange
        monkeypatch.setenv("AUTH_PROVIDER", "entra")

        # Need to reimport to pick up the new env var
        import importlib
        import registry.utils.iam_manager as iam_module

        importlib.reload(iam_module)

        # Act
        manager = iam_module.get_iam_manager()

        # Assert
        assert isinstance(manager, iam_module.EntraIAMManager)

    def test_defaults_to_keycloak_when_auth_provider_not_set(
        self,
        monkeypatch,
    ):
        """Test defaults to KeycloakIAMManager when AUTH_PROVIDER is not set."""
        # Arrange
        monkeypatch.delenv("AUTH_PROVIDER", raising=False)

        # Need to reimport to pick up the new env var
        import importlib
        import registry.utils.iam_manager as iam_module

        importlib.reload(iam_module)

        # Act
        manager = iam_module.get_iam_manager()

        # Assert
        assert isinstance(manager, iam_module.KeycloakIAMManager)

    def test_defaults_to_keycloak_for_unknown_provider(
        self,
        monkeypatch,
    ):
        """Test defaults to KeycloakIAMManager for unknown provider value."""
        # Arrange
        monkeypatch.setenv("AUTH_PROVIDER", "unknown-provider")

        # Need to reimport to pick up the new env var
        import importlib
        import registry.utils.iam_manager as iam_module

        importlib.reload(iam_module)

        # Act
        manager = iam_module.get_iam_manager()

        # Assert
        assert isinstance(manager, iam_module.KeycloakIAMManager)

    def test_auth_provider_case_insensitive(
        self,
        monkeypatch,
    ):
        """Test AUTH_PROVIDER comparison is case-insensitive."""
        # Arrange - test with uppercase
        monkeypatch.setenv("AUTH_PROVIDER", "KEYCLOAK")

        # Need to reimport to pick up the new env var
        import importlib
        import registry.utils.iam_manager as iam_module

        importlib.reload(iam_module)

        # Act
        manager = iam_module.get_iam_manager()

        # Assert
        assert isinstance(manager, iam_module.KeycloakIAMManager)

    def test_entra_auth_provider_case_insensitive(
        self,
        monkeypatch,
    ):
        """Test AUTH_PROVIDER='ENTRA' (uppercase) returns EntraIAMManager."""
        # Arrange
        monkeypatch.setenv("AUTH_PROVIDER", "ENTRA")

        # Need to reimport to pick up the new env var
        import importlib
        import registry.utils.iam_manager as iam_module

        importlib.reload(iam_module)

        # Act
        manager = iam_module.get_iam_manager()

        # Assert
        assert isinstance(manager, iam_module.EntraIAMManager)


# =============================================================================
# TEST: KeycloakIAMManager Methods
# =============================================================================


@pytest.mark.unit
class TestKeycloakIAMManager:
    """Test KeycloakIAMManager implementation."""

    @pytest.mark.asyncio
    async def test_list_users_delegates_to_keycloak_manager(
        self,
        sample_user_list: list[dict[str, Any]],
    ):
        """Test list_users() delegates to list_keycloak_users()."""
        # Arrange
        from registry.utils.iam_manager import KeycloakIAMManager

        manager = KeycloakIAMManager()
        mock_list_users = AsyncMock(return_value=sample_user_list)

        # Act - patch at the keycloak_manager module where function is defined
        with patch(
            "registry.utils.keycloak_manager.list_keycloak_users",
            mock_list_users,
        ):
            result = await manager.list_users(
                search="test",
                max_results=100,
                include_groups=True,
            )

        # Assert
        mock_list_users.assert_called_once_with(
            search="test",
            max_results=100,
            include_groups=True,
        )
        assert result == sample_user_list

    @pytest.mark.asyncio
    async def test_create_group_delegates_to_keycloak_manager(
        self,
        sample_created_group: dict[str, Any],
    ):
        """Test create_group() delegates to create_keycloak_group()."""
        # Arrange
        from registry.utils.iam_manager import KeycloakIAMManager

        manager = KeycloakIAMManager()
        mock_create_group = AsyncMock(return_value=sample_created_group)

        # Act
        with patch(
            "registry.utils.keycloak_manager.create_keycloak_group",
            mock_create_group,
        ):
            result = await manager.create_group(
                group_name="new-group",
                description="A new group",
            )

        # Assert
        mock_create_group.assert_called_once_with(
            group_name="new-group",
            description="A new group",
        )
        assert result == sample_created_group

    @pytest.mark.asyncio
    async def test_delete_group_delegates_to_keycloak_manager(self):
        """Test delete_group() delegates to delete_keycloak_group()."""
        # Arrange
        from registry.utils.iam_manager import KeycloakIAMManager

        manager = KeycloakIAMManager()
        mock_delete_group = AsyncMock(return_value=True)

        # Act
        with patch(
            "registry.utils.keycloak_manager.delete_keycloak_group",
            mock_delete_group,
        ):
            result = await manager.delete_group(group_name="test-group")

        # Assert
        mock_delete_group.assert_called_once_with(group_name="test-group")
        assert result is True

    @pytest.mark.asyncio
    async def test_list_groups_delegates_to_keycloak_manager(
        self,
        sample_group_list: list[dict[str, Any]],
    ):
        """Test list_groups() delegates to list_keycloak_groups()."""
        # Arrange
        from registry.utils.iam_manager import KeycloakIAMManager

        manager = KeycloakIAMManager()
        mock_list_groups = AsyncMock(return_value=sample_group_list)

        # Act
        with patch(
            "registry.utils.keycloak_manager.list_keycloak_groups",
            mock_list_groups,
        ):
            result = await manager.list_groups()

        # Assert
        mock_list_groups.assert_called_once()
        assert result == sample_group_list

    @pytest.mark.asyncio
    async def test_create_human_user_delegates_to_keycloak_manager(
        self,
        sample_created_user: dict[str, Any],
    ):
        """Test create_human_user() delegates to create_human_user_account()."""
        # Arrange
        from registry.utils.iam_manager import KeycloakIAMManager

        manager = KeycloakIAMManager()
        mock_create_user = AsyncMock(return_value=sample_created_user)

        # Act
        with patch(
            "registry.utils.keycloak_manager.create_human_user_account",
            mock_create_user,
        ):
            result = await manager.create_human_user(
                username="newuser",
                email="newuser@example.com",
                first_name="New",
                last_name="User",
                groups=["developers"],
                password="temppass123",
            )

        # Assert
        mock_create_user.assert_called_once_with(
            username="newuser",
            email="newuser@example.com",
            first_name="New",
            last_name="User",
            groups=["developers"],
            password="temppass123",
        )
        assert result == sample_created_user

    @pytest.mark.asyncio
    async def test_delete_user_delegates_to_keycloak_manager(self):
        """Test delete_user() delegates to delete_keycloak_user()."""
        # Arrange
        from registry.utils.iam_manager import KeycloakIAMManager

        manager = KeycloakIAMManager()
        mock_delete_user = AsyncMock(return_value=True)

        # Act
        with patch(
            "registry.utils.keycloak_manager.delete_keycloak_user",
            mock_delete_user,
        ):
            result = await manager.delete_user(username="testuser")

        # Assert
        mock_delete_user.assert_called_once_with(username="testuser")
        assert result is True

    @pytest.mark.asyncio
    async def test_create_service_account_delegates_to_keycloak_manager(
        self,
        sample_service_account: dict[str, Any],
    ):
        """Test create_service_account() delegates to create_service_account_client()."""
        # Arrange
        from registry.utils.iam_manager import KeycloakIAMManager

        manager = KeycloakIAMManager()
        mock_create_sa = AsyncMock(return_value=sample_service_account)

        # Act
        with patch(
            "registry.utils.keycloak_manager.create_service_account_client",
            mock_create_sa,
        ):
            result = await manager.create_service_account(
                client_id="test-service-account",
                groups=["api-access"],
                description="Test service account",
            )

        # Assert
        mock_create_sa.assert_called_once_with(
            client_id="test-service-account",
            group_names=["api-access"],
            description="Test service account",
        )
        assert result == sample_service_account


# =============================================================================
# TEST: EntraIAMManager Methods
# =============================================================================


@pytest.mark.unit
class TestEntraIAMManager:
    """Test EntraIAMManager implementation."""

    @pytest.mark.asyncio
    async def test_list_users_delegates_to_entra_manager(
        self,
        sample_user_list: list[dict[str, Any]],
    ):
        """Test list_users() delegates to list_entra_users()."""
        # Arrange
        from registry.utils.iam_manager import EntraIAMManager

        manager = EntraIAMManager()
        mock_list_users = AsyncMock(return_value=sample_user_list)

        # Act
        with patch(
            "registry.utils.entra_manager.list_entra_users",
            mock_list_users,
        ):
            result = await manager.list_users(
                search="test",
                max_results=100,
                include_groups=True,
            )

        # Assert
        mock_list_users.assert_called_once_with(
            search="test",
            max_results=100,
            include_groups=True,
        )
        assert result == sample_user_list

    @pytest.mark.asyncio
    async def test_create_group_delegates_to_entra_manager(
        self,
        sample_created_group: dict[str, Any],
    ):
        """Test create_group() delegates to create_entra_group()."""
        # Arrange
        from registry.utils.iam_manager import EntraIAMManager

        manager = EntraIAMManager()
        mock_create_group = AsyncMock(return_value=sample_created_group)

        # Act
        with patch(
            "registry.utils.entra_manager.create_entra_group",
            mock_create_group,
        ):
            result = await manager.create_group(
                group_name="new-group",
                description="A new group",
            )

        # Assert
        mock_create_group.assert_called_once_with(
            group_name="new-group",
            description="A new group",
        )
        assert result == sample_created_group

    @pytest.mark.asyncio
    async def test_delete_group_delegates_to_entra_manager(self):
        """Test delete_group() delegates to delete_entra_group()."""
        # Arrange
        from registry.utils.iam_manager import EntraIAMManager

        manager = EntraIAMManager()
        mock_delete_group = AsyncMock(return_value=True)

        # Act
        with patch(
            "registry.utils.entra_manager.delete_entra_group",
            mock_delete_group,
        ):
            result = await manager.delete_group(group_name="test-group")

        # Assert
        mock_delete_group.assert_called_once_with(group_name_or_id="test-group")
        assert result is True

    @pytest.mark.asyncio
    async def test_list_groups_delegates_to_entra_manager(
        self,
        sample_group_list: list[dict[str, Any]],
    ):
        """Test list_groups() delegates to list_entra_groups()."""
        # Arrange
        from registry.utils.iam_manager import EntraIAMManager

        manager = EntraIAMManager()
        mock_list_groups = AsyncMock(return_value=sample_group_list)

        # Act
        with patch(
            "registry.utils.entra_manager.list_entra_groups",
            mock_list_groups,
        ):
            result = await manager.list_groups()

        # Assert
        mock_list_groups.assert_called_once()
        assert result == sample_group_list

    @pytest.mark.asyncio
    async def test_create_human_user_delegates_to_entra_manager(
        self,
        sample_created_user: dict[str, Any],
    ):
        """Test create_human_user() delegates to create_entra_human_user()."""
        # Arrange
        from registry.utils.iam_manager import EntraIAMManager

        manager = EntraIAMManager()
        mock_create_user = AsyncMock(return_value=sample_created_user)

        # Act
        with patch(
            "registry.utils.entra_manager.create_entra_human_user",
            mock_create_user,
        ):
            result = await manager.create_human_user(
                username="newuser",
                email="newuser@example.com",
                first_name="New",
                last_name="User",
                groups=["developers"],
                password="temppass123",
            )

        # Assert
        mock_create_user.assert_called_once_with(
            username="newuser",
            email="newuser@example.com",
            first_name="New",
            last_name="User",
            groups=["developers"],
            password="temppass123",
        )
        assert result == sample_created_user

    @pytest.mark.asyncio
    async def test_delete_user_delegates_to_entra_manager(self):
        """Test delete_user() delegates to delete_entra_user()."""
        # Arrange
        from registry.utils.iam_manager import EntraIAMManager

        manager = EntraIAMManager()
        mock_delete_user = AsyncMock(return_value=True)

        # Act
        with patch(
            "registry.utils.entra_manager.delete_entra_user",
            mock_delete_user,
        ):
            result = await manager.delete_user(username="testuser")

        # Assert
        mock_delete_user.assert_called_once_with(username_or_id="testuser")
        assert result is True

    @pytest.mark.asyncio
    async def test_create_service_account_delegates_to_entra_manager(
        self,
        sample_service_account: dict[str, Any],
    ):
        """Test create_service_account() delegates to create_service_principal_client()."""
        # Arrange
        from registry.utils.iam_manager import EntraIAMManager

        manager = EntraIAMManager()
        mock_create_sa = AsyncMock(return_value=sample_service_account)

        # Act
        with patch(
            "registry.utils.entra_manager.create_service_principal_client",
            mock_create_sa,
        ):
            result = await manager.create_service_account(
                client_id="test-service-account",
                groups=["api-access"],
                description="Test service account",
            )

        # Assert
        mock_create_sa.assert_called_once_with(
            client_id_name="test-service-account",
            group_names=["api-access"],
            description="Test service account",
        )
        assert result == sample_service_account


# =============================================================================
# TEST: IAMManager Protocol Compliance
# =============================================================================


@pytest.mark.unit
class TestIAMManagerProtocol:
    """Test that IAM managers implement the IAMManager protocol."""

    def test_keycloak_manager_is_runtime_checkable(self):
        """Test KeycloakIAMManager satisfies IAMManager protocol."""
        from registry.utils.iam_manager import (
            IAMManager,
            KeycloakIAMManager,
        )

        manager = KeycloakIAMManager()

        # Check that manager is instance of protocol (runtime_checkable)
        assert isinstance(manager, IAMManager)

    def test_entra_manager_is_runtime_checkable(self):
        """Test EntraIAMManager satisfies IAMManager protocol."""
        from registry.utils.iam_manager import (
            EntraIAMManager,
            IAMManager,
        )

        manager = EntraIAMManager()

        # Check that manager is instance of protocol (runtime_checkable)
        assert isinstance(manager, IAMManager)

    def test_keycloak_manager_has_all_protocol_methods(self):
        """Test KeycloakIAMManager has all required protocol methods."""
        from registry.utils.iam_manager import KeycloakIAMManager

        manager = KeycloakIAMManager()

        # Verify all protocol methods exist
        assert hasattr(manager, "list_users")
        assert hasattr(manager, "create_human_user")
        assert hasattr(manager, "delete_user")
        assert hasattr(manager, "list_groups")
        assert hasattr(manager, "create_group")
        assert hasattr(manager, "delete_group")
        assert hasattr(manager, "create_service_account")

        # Verify methods are callable
        assert callable(manager.list_users)
        assert callable(manager.create_human_user)
        assert callable(manager.delete_user)
        assert callable(manager.list_groups)
        assert callable(manager.create_group)
        assert callable(manager.delete_group)
        assert callable(manager.create_service_account)

    def test_entra_manager_has_all_protocol_methods(self):
        """Test EntraIAMManager has all required protocol methods."""
        from registry.utils.iam_manager import EntraIAMManager

        manager = EntraIAMManager()

        # Verify all protocol methods exist
        assert hasattr(manager, "list_users")
        assert hasattr(manager, "create_human_user")
        assert hasattr(manager, "delete_user")
        assert hasattr(manager, "list_groups")
        assert hasattr(manager, "create_group")
        assert hasattr(manager, "delete_group")
        assert hasattr(manager, "create_service_account")

        # Verify methods are callable
        assert callable(manager.list_users)
        assert callable(manager.create_human_user)
        assert callable(manager.delete_user)
        assert callable(manager.list_groups)
        assert callable(manager.create_group)
        assert callable(manager.delete_group)
        assert callable(manager.create_service_account)
