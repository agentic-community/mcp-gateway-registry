"""
Unit tests for registry/utils/entra_manager.py

Tests for Microsoft Entra ID group and user management utilities.
Includes tests for:
- GUID validation helper
- Temporary password generation
- Graph API token acquisition
- User listing operations
- Group CRUD operations
"""

import logging
import string
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from registry.utils.entra_manager import (
    EntraAdminError,
    _generate_temp_password,
    _is_guid,
    create_entra_group,
    delete_entra_group,
    list_entra_groups,
    list_entra_users,
)

logger = logging.getLogger(__name__)


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def mock_admin_token() -> str:
    """Provide a mock admin token for testing."""
    return "mock-access-token-12345"


@pytest.fixture
def mock_token_response(mock_admin_token: str) -> dict[str, Any]:
    """Provide a mock token response from Entra ID."""
    return {
        "access_token": mock_admin_token,
        "token_type": "Bearer",
        "expires_in": 3600,
    }


@pytest.fixture
def mock_users_response() -> dict[str, Any]:
    """Provide a mock users response from Graph API."""
    return {
        "value": [
            {
                "id": "user-id-123",
                "displayName": "John Doe",
                "userPrincipalName": "john.doe@example.com",
                "mail": "john.doe@example.com",
                "givenName": "John",
                "surname": "Doe",
                "accountEnabled": True,
            },
            {
                "id": "user-id-456",
                "displayName": "Jane Smith",
                "userPrincipalName": "jane.smith@example.com",
                "mail": "jane.smith@example.com",
                "givenName": "Jane",
                "surname": "Smith",
                "accountEnabled": False,
            },
        ]
    }


@pytest.fixture
def mock_groups_response() -> dict[str, Any]:
    """Provide a mock groups response from Graph API."""
    return {
        "value": [
            {
                "id": "group-id-123",
                "displayName": "Registry Admins",
                "description": "Admin group for registry",
                "securityEnabled": True,
            },
            {
                "id": "group-id-456",
                "displayName": "Registry Users",
                "description": "User group for registry",
                "securityEnabled": True,
            },
        ]
    }


@pytest.fixture
def mock_create_group_response() -> dict[str, Any]:
    """Provide a mock create group response from Graph API."""
    return {
        "id": "new-group-id-789",
        "displayName": "New Test Group",
        "description": "A new test group",
        "securityEnabled": True,
    }


@pytest.fixture
def entra_env_vars(monkeypatch):
    """Set up environment variables for Entra ID authentication."""
    monkeypatch.setenv("ENTRA_TENANT_ID", "test-tenant-id")
    monkeypatch.setenv("ENTRA_CLIENT_ID", "test-client-id")
    monkeypatch.setenv("ENTRA_CLIENT_SECRET", "test-client-secret")

    # Also patch the module-level constants
    monkeypatch.setattr("registry.utils.entra_manager.ENTRA_TENANT_ID", "test-tenant-id")
    monkeypatch.setattr("registry.utils.entra_manager.ENTRA_CLIENT_ID", "test-client-id")
    monkeypatch.setattr("registry.utils.entra_manager.ENTRA_CLIENT_SECRET", "test-client-secret")


# =============================================================================
# TEST: _is_guid() helper
# =============================================================================


@pytest.mark.unit
class TestIsGuid:
    """Tests for _is_guid helper function."""

    def test_valid_guid_lowercase(self):
        """Test valid lowercase GUID returns True."""
        # Arrange
        valid_guid = "12345678-1234-1234-1234-123456789abc"

        # Act
        result = _is_guid(valid_guid)

        # Assert
        assert result is True

    def test_valid_guid_uppercase(self):
        """Test valid uppercase GUID returns True."""
        # Arrange
        valid_guid = "12345678-1234-1234-1234-123456789ABC"

        # Act
        result = _is_guid(valid_guid)

        # Assert
        assert result is True

    def test_valid_guid_mixed_case(self):
        """Test valid mixed case GUID returns True."""
        # Arrange
        valid_guid = "12345678-1234-1234-1234-123456789AbC"

        # Act
        result = _is_guid(valid_guid)

        # Assert
        assert result is True

    def test_invalid_guid_wrong_format(self):
        """Test invalid GUID format returns False."""
        # Arrange
        invalid_guid = "12345678123412341234123456789abc"  # No dashes

        # Act
        result = _is_guid(invalid_guid)

        # Assert
        assert result is False

    def test_invalid_guid_short_string(self):
        """Test short string returns False."""
        # Arrange
        invalid_guid = "12345678"

        # Act
        result = _is_guid(invalid_guid)

        # Assert
        assert result is False

    def test_invalid_guid_display_name(self):
        """Test display name string returns False."""
        # Arrange
        display_name = "Registry Admins"

        # Act
        result = _is_guid(display_name)

        # Assert
        assert result is False

    def test_invalid_guid_empty_string(self):
        """Test empty string returns False."""
        # Arrange
        empty_string = ""

        # Act
        result = _is_guid(empty_string)

        # Assert
        assert result is False

    def test_invalid_guid_contains_invalid_chars(self):
        """Test GUID with invalid characters returns False."""
        # Arrange
        invalid_guid = "12345678-1234-1234-1234-123456789xyz"  # xyz not valid hex

        # Act
        result = _is_guid(invalid_guid)

        # Assert
        assert result is False

    def test_invalid_guid_wrong_segment_lengths(self):
        """Test GUID with wrong segment lengths returns False."""
        # Arrange
        invalid_guid = "1234-12345678-1234-1234-123456789abc"  # Wrong segment order

        # Act
        result = _is_guid(invalid_guid)

        # Assert
        assert result is False


# =============================================================================
# TEST: _generate_temp_password()
# =============================================================================


@pytest.mark.unit
class TestGenerateTempPassword:
    """Tests for _generate_temp_password helper function."""

    def test_password_length(self):
        """Test password meets length requirements (16 chars)."""
        # Act
        password = _generate_temp_password()

        # Assert
        assert len(password) == 16

    def test_password_contains_allowed_characters(self):
        """Test password contains only allowed character types."""
        # Arrange
        allowed_chars = string.ascii_letters + string.digits + "!@#$%^&*()"

        # Act
        password = _generate_temp_password()

        # Assert
        for char in password:
            assert char in allowed_chars, f"Character '{char}' not in allowed set"

    def test_password_randomness(self):
        """Test that generated passwords are different each time."""
        # Act
        passwords = [_generate_temp_password() for _ in range(10)]

        # Assert - all passwords should be unique
        assert len(set(passwords)) == 10, "Passwords should be randomly generated"

    def test_password_contains_letters(self):
        """Test password typically contains letters."""
        # Act - generate multiple to increase probability of coverage
        passwords = [_generate_temp_password() for _ in range(20)]

        # Assert - at least some passwords should contain letters
        has_letters = False
        for password in passwords:
            if any(c in string.ascii_letters for c in password):
                has_letters = True
                break
        assert has_letters, "At least some passwords should contain letters"

    def test_password_is_string(self):
        """Test password is returned as string."""
        # Act
        password = _generate_temp_password()

        # Assert
        assert isinstance(password, str)


# =============================================================================
# TEST: _get_entra_admin_token() error handling
# =============================================================================


@pytest.mark.unit
class TestGetEntraAdminToken:
    """Tests for _get_entra_admin_token error handling."""

    @pytest.mark.asyncio
    async def test_raises_error_when_client_secret_missing(self, monkeypatch):
        """Test raises EntraAdminError when ENTRA_CLIENT_SECRET not set."""
        # Arrange
        monkeypatch.setattr("registry.utils.entra_manager.ENTRA_CLIENT_SECRET", "")
        monkeypatch.setattr("registry.utils.entra_manager.ENTRA_TENANT_ID", "test-tenant")
        monkeypatch.setattr("registry.utils.entra_manager.ENTRA_CLIENT_ID", "test-client")

        # Import after patching
        from registry.utils.entra_manager import _get_entra_admin_token

        # Act & Assert
        with pytest.raises(EntraAdminError) as exc_info:
            await _get_entra_admin_token()

        assert "ENTRA_CLIENT_SECRET" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_raises_error_when_tenant_id_missing(self, monkeypatch):
        """Test raises EntraAdminError when ENTRA_TENANT_ID not set."""
        # Arrange
        monkeypatch.setattr("registry.utils.entra_manager.ENTRA_CLIENT_SECRET", "test-secret")
        monkeypatch.setattr("registry.utils.entra_manager.ENTRA_TENANT_ID", "")
        monkeypatch.setattr("registry.utils.entra_manager.ENTRA_CLIENT_ID", "test-client")

        # Import after patching
        from registry.utils.entra_manager import _get_entra_admin_token

        # Act & Assert
        with pytest.raises(EntraAdminError) as exc_info:
            await _get_entra_admin_token()

        assert "ENTRA_TENANT_ID" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_raises_error_when_client_id_missing(self, monkeypatch):
        """Test raises EntraAdminError when ENTRA_CLIENT_ID not set."""
        # Arrange
        monkeypatch.setattr("registry.utils.entra_manager.ENTRA_CLIENT_SECRET", "test-secret")
        monkeypatch.setattr("registry.utils.entra_manager.ENTRA_TENANT_ID", "test-tenant")
        monkeypatch.setattr("registry.utils.entra_manager.ENTRA_CLIENT_ID", "")

        # Import after patching
        from registry.utils.entra_manager import _get_entra_admin_token

        # Act & Assert
        with pytest.raises(EntraAdminError) as exc_info:
            await _get_entra_admin_token()

        assert "ENTRA_CLIENT_ID" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_raises_error_on_http_error(self, entra_env_vars):
        """Test raises EntraAdminError on HTTP error response."""
        # Arrange
        from registry.utils.entra_manager import _get_entra_admin_token

        # Create a mock response that raises HTTPStatusError
        mock_response = MagicMock()
        mock_response.status_code = 401
        mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "Unauthorized", request=MagicMock(), response=mock_response
        )

        mock_client = AsyncMock()
        mock_client.post.return_value = mock_response
        mock_client.__aenter__.return_value = mock_client
        mock_client.__aexit__.return_value = None

        # Act & Assert
        with patch("registry.utils.entra_manager.httpx.AsyncClient", return_value=mock_client):
            with pytest.raises(EntraAdminError) as exc_info:
                await _get_entra_admin_token()

        assert "authentication failed" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_raises_error_when_no_access_token_in_response(self, entra_env_vars):
        """Test raises EntraAdminError when response has no access_token."""
        # Arrange
        from registry.utils.entra_manager import _get_entra_admin_token

        # Create a mock response with no access_token
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {"error": "something went wrong"}

        mock_client = AsyncMock()
        mock_client.post.return_value = mock_response
        mock_client.__aenter__.return_value = mock_client
        mock_client.__aexit__.return_value = None

        # Act & Assert
        with patch("registry.utils.entra_manager.httpx.AsyncClient", return_value=mock_client):
            with pytest.raises(EntraAdminError) as exc_info:
                await _get_entra_admin_token()

        assert "No access token" in str(exc_info.value)


# =============================================================================
# TEST: list_entra_users()
# =============================================================================


@pytest.mark.unit
class TestListEntraUsers:
    """Tests for list_entra_users function."""

    @pytest.mark.asyncio
    async def test_list_users_success(
        self,
        entra_env_vars,
        mock_token_response: dict[str, Any],
        mock_users_response: dict[str, Any],
    ):
        """Test listing users successfully."""
        # Arrange
        mock_token_resp = MagicMock()
        mock_token_resp.status_code = 200
        mock_token_resp.raise_for_status.return_value = None
        mock_token_resp.json.return_value = mock_token_response

        mock_users_resp = MagicMock()
        mock_users_resp.status_code = 200
        mock_users_resp.raise_for_status.return_value = None
        mock_users_resp.json.return_value = mock_users_response

        mock_groups_resp = MagicMock()
        mock_groups_resp.status_code = 200
        mock_groups_resp.raise_for_status.return_value = None
        mock_groups_resp.json.return_value = {"value": []}

        mock_client = AsyncMock()
        mock_client.post.return_value = mock_token_resp
        mock_client.get.side_effect = [mock_users_resp, mock_groups_resp, mock_groups_resp]
        mock_client.__aenter__.return_value = mock_client
        mock_client.__aexit__.return_value = None

        # Act
        with patch("registry.utils.entra_manager.httpx.AsyncClient", return_value=mock_client):
            result = await list_entra_users(include_groups=True)

        # Assert
        assert len(result) == 2
        assert result[0]["id"] == "user-id-123"
        assert result[0]["username"] == "john.doe@example.com"
        assert result[0]["email"] == "john.doe@example.com"
        assert result[0]["firstName"] == "John"
        assert result[0]["lastName"] == "Doe"
        assert result[0]["enabled"] is True

        assert result[1]["id"] == "user-id-456"
        assert result[1]["username"] == "jane.smith@example.com"
        assert result[1]["enabled"] is False

    @pytest.mark.asyncio
    async def test_list_users_without_groups(
        self,
        entra_env_vars,
        mock_token_response: dict[str, Any],
        mock_users_response: dict[str, Any],
    ):
        """Test listing users without group memberships."""
        # Arrange
        mock_token_resp = MagicMock()
        mock_token_resp.status_code = 200
        mock_token_resp.raise_for_status.return_value = None
        mock_token_resp.json.return_value = mock_token_response

        mock_users_resp = MagicMock()
        mock_users_resp.status_code = 200
        mock_users_resp.raise_for_status.return_value = None
        mock_users_resp.json.return_value = mock_users_response

        mock_client = AsyncMock()
        mock_client.post.return_value = mock_token_resp
        mock_client.get.return_value = mock_users_resp
        mock_client.__aenter__.return_value = mock_client
        mock_client.__aexit__.return_value = None

        # Act
        with patch("registry.utils.entra_manager.httpx.AsyncClient", return_value=mock_client):
            result = await list_entra_users(include_groups=False)

        # Assert
        assert len(result) == 2
        # Groups should be empty lists since we didn't fetch them
        assert result[0]["groups"] == []
        assert result[1]["groups"] == []

    @pytest.mark.asyncio
    async def test_list_users_transforms_data_correctly(
        self,
        entra_env_vars,
        mock_token_response: dict[str, Any],
    ):
        """Test that user data is transformed correctly."""
        # Arrange
        users_response = {
            "value": [
                {
                    "id": "user-123",
                    "displayName": "Test User",
                    "userPrincipalName": "test@example.com",
                    "mail": "test.mail@example.com",
                    "givenName": "Test",
                    "surname": "User",
                    "accountEnabled": True,
                },
            ]
        }

        mock_token_resp = MagicMock()
        mock_token_resp.status_code = 200
        mock_token_resp.raise_for_status.return_value = None
        mock_token_resp.json.return_value = mock_token_response

        mock_users_resp = MagicMock()
        mock_users_resp.status_code = 200
        mock_users_resp.raise_for_status.return_value = None
        mock_users_resp.json.return_value = users_response

        mock_client = AsyncMock()
        mock_client.post.return_value = mock_token_resp
        mock_client.get.return_value = mock_users_resp
        mock_client.__aenter__.return_value = mock_client
        mock_client.__aexit__.return_value = None

        # Act
        with patch("registry.utils.entra_manager.httpx.AsyncClient", return_value=mock_client):
            result = await list_entra_users(include_groups=False)

        # Assert
        assert len(result) == 1
        user = result[0]
        assert user["id"] == "user-123"
        assert user["username"] == "test@example.com"
        assert user["email"] == "test.mail@example.com"
        assert user["firstName"] == "Test"
        assert user["lastName"] == "User"
        assert user["enabled"] is True
        assert "groups" in user


# =============================================================================
# TEST: create_entra_group()
# =============================================================================


@pytest.mark.unit
class TestCreateEntraGroup:
    """Tests for create_entra_group function."""

    @pytest.mark.asyncio
    async def test_create_group_success(
        self,
        entra_env_vars,
        mock_token_response: dict[str, Any],
        mock_create_group_response: dict[str, Any],
    ):
        """Test creating a group successfully."""
        # Arrange
        mock_token_resp = MagicMock()
        mock_token_resp.status_code = 200
        mock_token_resp.raise_for_status.return_value = None
        mock_token_resp.json.return_value = mock_token_response

        mock_create_resp = MagicMock()
        mock_create_resp.status_code = 201
        mock_create_resp.raise_for_status.return_value = None
        mock_create_resp.json.return_value = mock_create_group_response

        mock_client = AsyncMock()
        mock_client.post.side_effect = [mock_token_resp, mock_create_resp]
        mock_client.__aenter__.return_value = mock_client
        mock_client.__aexit__.return_value = None

        # Act
        with patch("registry.utils.entra_manager.httpx.AsyncClient", return_value=mock_client):
            result = await create_entra_group("New Test Group", "A new test group")

        # Assert
        assert result["id"] == "new-group-id-789"
        assert result["name"] == "New Test Group"
        assert result["path"] == "/New Test Group"
        assert "attributes" in result
        assert result["attributes"]["description"] == ["A new test group"]

    @pytest.mark.asyncio
    async def test_create_group_returns_correct_document(
        self,
        entra_env_vars,
        mock_token_response: dict[str, Any],
    ):
        """Test that create group returns correctly formatted document."""
        # Arrange
        create_response = {
            "id": "group-abc-123",
            "displayName": "My Custom Group",
            "description": "Custom description",
            "securityEnabled": True,
        }

        mock_token_resp = MagicMock()
        mock_token_resp.status_code = 200
        mock_token_resp.raise_for_status.return_value = None
        mock_token_resp.json.return_value = mock_token_response

        mock_create_resp = MagicMock()
        mock_create_resp.status_code = 201
        mock_create_resp.raise_for_status.return_value = None
        mock_create_resp.json.return_value = create_response

        mock_client = AsyncMock()
        mock_client.post.side_effect = [mock_token_resp, mock_create_resp]
        mock_client.__aenter__.return_value = mock_client
        mock_client.__aexit__.return_value = None

        # Act
        with patch("registry.utils.entra_manager.httpx.AsyncClient", return_value=mock_client):
            result = await create_entra_group("My Custom Group", "Custom description")

        # Assert
        assert result["id"] == "group-abc-123"
        assert result["name"] == "My Custom Group"
        assert result["path"] == "/My Custom Group"
        assert result["attributes"]["description"] == ["Custom description"]

    @pytest.mark.asyncio
    async def test_create_group_already_exists_raises_error(
        self,
        entra_env_vars,
        mock_token_response: dict[str, Any],
    ):
        """Test that creating an existing group raises EntraAdminError."""
        # Arrange
        mock_token_resp = MagicMock()
        mock_token_resp.status_code = 200
        mock_token_resp.raise_for_status.return_value = None
        mock_token_resp.json.return_value = mock_token_response

        mock_error_resp = MagicMock()
        mock_error_resp.status_code = 400
        mock_error_resp.json.return_value = {
            "error": {"message": "A group with this name already exists."}
        }

        mock_client = AsyncMock()
        mock_client.post.side_effect = [mock_token_resp, mock_error_resp]
        mock_client.__aenter__.return_value = mock_client
        mock_client.__aexit__.return_value = None

        # Act & Assert
        with patch("registry.utils.entra_manager.httpx.AsyncClient", return_value=mock_client):
            with pytest.raises(EntraAdminError) as exc_info:
                await create_entra_group("Existing Group")

        assert "already exists" in str(exc_info.value)


# =============================================================================
# TEST: delete_entra_group()
# =============================================================================


@pytest.mark.unit
class TestDeleteEntraGroup:
    """Tests for delete_entra_group function."""

    @pytest.mark.asyncio
    async def test_delete_group_by_id_success(
        self,
        entra_env_vars,
        mock_token_response: dict[str, Any],
    ):
        """Test deleting a group by ID successfully."""
        # Arrange
        group_id = "12345678-1234-1234-1234-123456789abc"

        mock_token_resp = MagicMock()
        mock_token_resp.status_code = 200
        mock_token_resp.raise_for_status.return_value = None
        mock_token_resp.json.return_value = mock_token_response

        mock_delete_resp = MagicMock()
        mock_delete_resp.status_code = 204

        mock_client = AsyncMock()
        mock_client.post.return_value = mock_token_resp
        mock_client.delete.return_value = mock_delete_resp
        mock_client.__aenter__.return_value = mock_client
        mock_client.__aexit__.return_value = None

        # Act
        with patch("registry.utils.entra_manager.httpx.AsyncClient", return_value=mock_client):
            result = await delete_entra_group(group_id)

        # Assert
        assert result is True

    @pytest.mark.asyncio
    async def test_delete_group_by_name_success(
        self,
        entra_env_vars,
        mock_token_response: dict[str, Any],
    ):
        """Test deleting a group by name successfully."""
        # Arrange
        group_name = "Test Group"
        group_id = "12345678-1234-1234-1234-123456789abc"

        mock_token_resp = MagicMock()
        mock_token_resp.status_code = 200
        mock_token_resp.raise_for_status.return_value = None
        mock_token_resp.json.return_value = mock_token_response

        # Mock finding the group by name
        mock_find_resp = MagicMock()
        mock_find_resp.status_code = 200
        mock_find_resp.raise_for_status.return_value = None
        mock_find_resp.json.return_value = {"value": [{"id": group_id}]}

        mock_delete_resp = MagicMock()
        mock_delete_resp.status_code = 204

        mock_client = AsyncMock()
        mock_client.post.return_value = mock_token_resp
        mock_client.get.return_value = mock_find_resp
        mock_client.delete.return_value = mock_delete_resp
        mock_client.__aenter__.return_value = mock_client
        mock_client.__aexit__.return_value = None

        # Act
        with patch("registry.utils.entra_manager.httpx.AsyncClient", return_value=mock_client):
            result = await delete_entra_group(group_name)

        # Assert
        assert result is True

    @pytest.mark.asyncio
    async def test_delete_group_not_found_by_name_raises_error(
        self,
        entra_env_vars,
        mock_token_response: dict[str, Any],
    ):
        """Test that deleting a non-existent group by name raises error."""
        # Arrange
        group_name = "Non Existent Group"

        mock_token_resp = MagicMock()
        mock_token_resp.status_code = 200
        mock_token_resp.raise_for_status.return_value = None
        mock_token_resp.json.return_value = mock_token_response

        # Mock finding the group - returns empty
        mock_find_resp = MagicMock()
        mock_find_resp.status_code = 200
        mock_find_resp.raise_for_status.return_value = None
        mock_find_resp.json.return_value = {"value": []}

        mock_client = AsyncMock()
        mock_client.post.return_value = mock_token_resp
        mock_client.get.return_value = mock_find_resp
        mock_client.__aenter__.return_value = mock_client
        mock_client.__aexit__.return_value = None

        # Act & Assert
        with patch("registry.utils.entra_manager.httpx.AsyncClient", return_value=mock_client):
            with pytest.raises(EntraAdminError) as exc_info:
                await delete_entra_group(group_name)

        assert "not found" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_delete_group_404_raises_error(
        self,
        entra_env_vars,
        mock_token_response: dict[str, Any],
    ):
        """Test that 404 response raises EntraAdminError."""
        # Arrange
        group_id = "12345678-1234-1234-1234-123456789abc"

        mock_token_resp = MagicMock()
        mock_token_resp.status_code = 200
        mock_token_resp.raise_for_status.return_value = None
        mock_token_resp.json.return_value = mock_token_response

        mock_delete_resp = MagicMock()
        mock_delete_resp.status_code = 404

        mock_client = AsyncMock()
        mock_client.post.return_value = mock_token_resp
        mock_client.delete.return_value = mock_delete_resp
        mock_client.__aenter__.return_value = mock_client
        mock_client.__aexit__.return_value = None

        # Act & Assert
        with patch("registry.utils.entra_manager.httpx.AsyncClient", return_value=mock_client):
            with pytest.raises(EntraAdminError) as exc_info:
                await delete_entra_group(group_id)

        assert "not found" in str(exc_info.value).lower()


# =============================================================================
# TEST: list_entra_groups()
# =============================================================================


@pytest.mark.unit
class TestListEntraGroups:
    """Tests for list_entra_groups function."""

    @pytest.mark.asyncio
    async def test_list_groups_success(
        self,
        entra_env_vars,
        mock_token_response: dict[str, Any],
        mock_groups_response: dict[str, Any],
    ):
        """Test listing groups successfully."""
        # Arrange
        mock_token_resp = MagicMock()
        mock_token_resp.status_code = 200
        mock_token_resp.raise_for_status.return_value = None
        mock_token_resp.json.return_value = mock_token_response

        mock_groups_resp = MagicMock()
        mock_groups_resp.status_code = 200
        mock_groups_resp.raise_for_status.return_value = None
        mock_groups_resp.json.return_value = mock_groups_response

        mock_client = AsyncMock()
        mock_client.post.return_value = mock_token_resp
        mock_client.get.return_value = mock_groups_resp
        mock_client.__aenter__.return_value = mock_client
        mock_client.__aexit__.return_value = None

        # Act
        with patch("registry.utils.entra_manager.httpx.AsyncClient", return_value=mock_client):
            result = await list_entra_groups()

        # Assert
        assert len(result) == 2
        assert result[0]["id"] == "group-id-123"
        assert result[0]["name"] == "Registry Admins"
        assert result[0]["path"] == "/Registry Admins"

        assert result[1]["id"] == "group-id-456"
        assert result[1]["name"] == "Registry Users"
        assert result[1]["path"] == "/Registry Users"

    @pytest.mark.asyncio
    async def test_list_groups_transforms_correctly(
        self,
        entra_env_vars,
        mock_token_response: dict[str, Any],
    ):
        """Test that groups are transformed correctly to match Keycloak format."""
        # Arrange
        groups_response = {
            "value": [
                {
                    "id": "group-abc",
                    "displayName": "Test Group",
                    "description": "Test description",
                    "securityEnabled": True,
                },
            ]
        }

        mock_token_resp = MagicMock()
        mock_token_resp.status_code = 200
        mock_token_resp.raise_for_status.return_value = None
        mock_token_resp.json.return_value = mock_token_response

        mock_groups_resp = MagicMock()
        mock_groups_resp.status_code = 200
        mock_groups_resp.raise_for_status.return_value = None
        mock_groups_resp.json.return_value = groups_response

        mock_client = AsyncMock()
        mock_client.post.return_value = mock_token_resp
        mock_client.get.return_value = mock_groups_resp
        mock_client.__aenter__.return_value = mock_client
        mock_client.__aexit__.return_value = None

        # Act
        with patch("registry.utils.entra_manager.httpx.AsyncClient", return_value=mock_client):
            result = await list_entra_groups()

        # Assert
        assert len(result) == 1
        group = result[0]
        assert group["id"] == "group-abc"
        assert group["name"] == "Test Group"
        assert group["path"] == "/Test Group"
        assert "attributes" in group
        assert group["attributes"]["description"] == ["Test description"]
        assert group["attributes"]["securityEnabled"] is True

    @pytest.mark.asyncio
    async def test_list_groups_empty_response(
        self,
        entra_env_vars,
        mock_token_response: dict[str, Any],
    ):
        """Test listing groups with empty response."""
        # Arrange
        mock_token_resp = MagicMock()
        mock_token_resp.status_code = 200
        mock_token_resp.raise_for_status.return_value = None
        mock_token_resp.json.return_value = mock_token_response

        mock_groups_resp = MagicMock()
        mock_groups_resp.status_code = 200
        mock_groups_resp.raise_for_status.return_value = None
        mock_groups_resp.json.return_value = {"value": []}

        mock_client = AsyncMock()
        mock_client.post.return_value = mock_token_resp
        mock_client.get.return_value = mock_groups_resp
        mock_client.__aenter__.return_value = mock_client
        mock_client.__aexit__.return_value = None

        # Act
        with patch("registry.utils.entra_manager.httpx.AsyncClient", return_value=mock_client):
            result = await list_entra_groups()

        # Assert
        assert result == []
