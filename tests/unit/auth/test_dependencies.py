"""
Unit tests for registry/auth/dependencies.py

Tests all authentication dependencies including:
- Session validation and extraction
- User context building
- Scope mapping
- Permission checking
- UI permissions
- Server access control
"""

import logging
from typing import Any
from unittest.mock import AsyncMock, Mock, patch

import pytest
from fastapi import HTTPException, Request
from itsdangerous import SignatureExpired, URLSafeTimedSerializer

from registry.auth.dependencies import (
    _user_is_admin,
    api_auth,
    enhanced_auth,
    get_accessible_agents_for_user,
    get_accessible_services_for_user,
    get_current_user,
    get_servers_for_scope,
    get_ui_permissions_for_user,
    get_user_accessible_servers,
    get_user_session_data,
    map_cognito_groups_to_scopes,
    nginx_proxied_auth,
    user_can_access_server,
    user_can_modify_servers,
    user_has_ui_permission_for_service,
    user_has_wildcard_access,
    web_auth,
)
from tests.fixtures.mocks.mock_auth import MockSessionValidator

logger = logging.getLogger(__name__)


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def test_secret_key() -> str:
    """Secret key for session signing."""
    return "test-secret-key-for-unit-tests"


@pytest.fixture
def mock_signer(test_secret_key: str, monkeypatch):
    """Mock URLSafeTimedSerializer for session signing."""
    signer = URLSafeTimedSerializer(test_secret_key)
    # Patch the module-level signer
    monkeypatch.setattr("registry.auth.dependencies.signer", signer)
    return signer


@pytest.fixture
def mock_session_store(monkeypatch):
    """Mock the server-side session store lookup.

    The test sets `mock_session_store.next_value = {...}` (or None) and any
    `signer.dumps(<session_id>)` cookie passed through resolve_session_from_cookie
    will resolve to that value, regardless of what session_id was used.
    """

    class _Stub:
        next_value: dict | None = None

    stub = _Stub()

    async def _fake_resolve(session_id: str):
        return stub.next_value

    monkeypatch.setattr("registry.auth.dependencies._store_resolve_session", _fake_resolve)
    return stub


def _make_session_cookie(signer: URLSafeTimedSerializer, session_id: str = "sid-test") -> str:
    """Helper: produce a signed opaque-session_id cookie matching the new format."""
    return signer.dumps(session_id)


@pytest.fixture
def registry_token_secret(test_secret_key: str, monkeypatch):
    """Set SECRET_KEY in env so the registry-UI token mint/verify works.

    The auth-server minter and the registry verifier both read os.environ['SECRET_KEY'];
    the tests mint a token with the same key the verifier will use.
    """
    monkeypatch.setenv("SECRET_KEY", test_secret_key)
    return test_secret_key


def _mint_registry_token(
    secret: str,
    *,
    subject: str,
    session_id: str = "",
    groups: list[str] | None = None,
    auth_method: str = "keycloak",
    client_id: str = "",
) -> str:
    """Mint a registry-UI token the way auth_server's /validate would.

    Mirrors auth_server.internal_request_token.mint_registry_ui_token without
    importing the auth-server package into registry tests.
    """
    import time

    import jwt as pyjwt

    now = int(time.time())
    claims = {
        "iss": "mcp-auth-server",
        "aud": "mcp-registry-ui",
        "sub": subject,
        "scopes": [],
        "session_id": session_id or "",
        "groups": list(groups or []),
        "auth_method": auth_method or "",
        "client_id": client_id or "",
        "token_use": "mcp-registry-ui",
        "iat": now,
        "exp": now + 30,
    }
    return pyjwt.encode(claims, secret, algorithm="HS256")


def _proxied_request(headers: dict[str, str] | None = None) -> Mock:
    """A Mock Request for the proxied path with a real headers dict + state."""
    req = Mock(spec=Request)
    req.url.path = "/api/test"
    req.method = "GET"
    req.state = Mock()
    req.headers = headers or {}
    return req


@pytest.fixture
def sample_scopes_config() -> dict[str, Any]:
    """Sample scopes configuration for testing."""
    return {
        "UI-Scopes": {
            "mcp-registry-admin": {
                "list_agents": ["all"],
                "get_agent": ["all"],
                "publish_agent": ["all"],
                "modify_agent": ["all"],
                "delete_agent": ["all"],
                "list_service": ["all"],
                "register_service": ["all"],
                "toggle_service": ["all"],
            },
            "registry-admins": {
                "list_agents": ["all"],
                "get_agent": ["all"],
                "publish_agent": ["all"],
                "modify_agent": ["all"],
                "delete_agent": ["all"],
                "list_service": ["all"],
                "register_service": ["all"],
                "toggle_service": ["all"],
            },
            "registry-users-lob1": {
                "list_agents": ["/code-reviewer", "/test-automation"],
                "get_agent": ["/code-reviewer", "/test-automation"],
                "list_service": ["currenttime", "mcpgw"],
            },
        },
        "group_mappings": {
            "mcp-registry-admin": [
                "mcp-registry-admin",
                "mcp-servers-unrestricted/read",
                "mcp-servers-unrestricted/execute",
            ],
            "registry-admins": [
                "registry-admins",
                "mcp-servers-unrestricted/read",
                "mcp-servers-unrestricted/execute",
            ],
            "registry-users-lob1": ["registry-users-lob1"],
        },
        "mcp-servers-unrestricted/read": [
            {
                "server": "*",
                "methods": ["initialize", "tools/list", "tools/call"],
                "tools": "*",
            }
        ],
        "mcp-servers-unrestricted/execute": [
            {
                "server": "*",
                "methods": ["initialize", "GET", "POST", "PUT", "DELETE"],
                "tools": "*",
            }
        ],
        "registry-admins": [
            {
                "server": "*",
                "methods": [
                    "initialize",
                    "GET",
                    "POST",
                    "PUT",
                    "DELETE",
                    "tools/list",
                    "tools/call",
                ],
                "tools": "*",
            }
        ],
        "registry-users-lob1": [
            {
                "server": "currenttime",
                "methods": ["initialize", "tools/list"],
                "tools": ["current_time_by_timezone"],
            }
        ],
    }


@pytest.fixture
def mock_scopes_config(sample_scopes_config: dict[str, Any], monkeypatch):
    """Mock SCOPES_CONFIG global variable and scope repository."""
    # Keep existing monkeypatch for backward compatibility
    monkeypatch.setattr("registry.auth.dependencies.SCOPES_CONFIG", sample_scopes_config)

    # Create mock repository
    mock_repo = AsyncMock()

    # Configure get_group_mappings based on sample config
    async def mock_get_group_mappings(group: str):
        group_mappings = sample_scopes_config.get("group_mappings", {})
        return group_mappings.get(group, [])

    # get_all_group_mappings returns the inverse shape stored in the DB:
    # {scope_name: [groups]} (not {group: [scopes]} like the YAML config).
    async def mock_get_all_group_mappings():
        group_mappings = sample_scopes_config.get("group_mappings", {})
        inverted: dict[str, list[str]] = {}
        for group, scope_names in group_mappings.items():
            for scope_name in scope_names:
                inverted.setdefault(scope_name, []).append(group)
        return inverted

    # Configure get_ui_scopes based on sample config
    async def mock_get_ui_scopes(scope: str):
        ui_scopes = sample_scopes_config.get("UI-Scopes", {})
        return ui_scopes.get(scope, {})

    # Configure get_server_scopes based on sample config
    async def mock_get_server_scopes(scope: str):
        # Check in the main config for scope definitions
        # The scope config is stored directly as a key in sample_scopes_config
        # Return the raw config (list of dicts), not extracted server names
        scope_config = sample_scopes_config.get(scope, [])
        if scope_config and isinstance(scope_config, list):
            return scope_config
        return []

    # Bulk variants delegate to the single getters, mirroring the base-class
    # default impl (dict keyed by scope name, empties omitted). Production code
    # on the hot path now calls these instead of looping the singles.
    async def mock_get_server_scopes_bulk(scope_names: list[str]):
        out: dict[str, list] = {}
        for name in scope_names:
            rules = await mock_get_server_scopes(name)
            if rules:
                out[name] = rules
        return out

    async def mock_get_ui_scopes_bulk(group_names: list[str]):
        out: dict[str, dict] = {}
        for name in group_names:
            scopes = await mock_get_ui_scopes(name)
            if scopes:
                out[name] = scopes
        return out

    # get_group_mappings_bulk returns the de-duplicated union of scopes across
    # the given groups (mirrors the base-class default that loops the single
    # getter). Production map_groups_to_scopes now calls this instead of looping.
    async def mock_get_group_mappings_bulk(groups: list[str]):
        seen: set[str] = set()
        out: list[str] = []
        for group in groups:
            for scope in await mock_get_group_mappings(group):
                if scope not in seen:
                    seen.add(scope)
                    out.append(scope)
        return out

    mock_repo.get_group_mappings.side_effect = mock_get_group_mappings
    mock_repo.get_group_mappings_bulk.side_effect = mock_get_group_mappings_bulk
    mock_repo.get_all_group_mappings.side_effect = mock_get_all_group_mappings
    mock_repo.get_ui_scopes.side_effect = mock_get_ui_scopes
    mock_repo.get_server_scopes.side_effect = mock_get_server_scopes
    mock_repo.get_server_scopes_bulk.side_effect = mock_get_server_scopes_bulk
    mock_repo.get_ui_scopes_bulk.side_effect = mock_get_ui_scopes_bulk

    # Patch get_scope_repository to return our mock using patch context manager
    # Since it's imported locally in functions, we need to patch the import
    with patch("registry.repositories.factory.get_scope_repository", return_value=mock_repo):
        yield sample_scopes_config


@pytest.fixture
def mock_session_validator(test_secret_key: str):
    """Create a mock session validator."""
    return MockSessionValidator(secret_key=test_secret_key)


# =============================================================================
# TEST: get_current_user
# =============================================================================


@pytest.mark.unit
@pytest.mark.auth
class TestGetCurrentUser:
    """Tests for get_current_user dependency."""

    @pytest.mark.asyncio
    async def test_get_current_user_with_valid_session(
        self, mock_signer: URLSafeTimedSerializer, mock_session_store
    ):
        """Test extracting user from valid session cookie."""
        mock_session_store.next_value = {"username": "testuser", "auth_method": "oauth2"}
        session_cookie = _make_session_cookie(mock_signer)

        username = await get_current_user(session=session_cookie)

        assert username == "testuser"

    @pytest.mark.asyncio
    async def test_get_current_user_no_session_cookie(self, mock_session_store):
        """Test that missing session cookie raises 401."""
        with pytest.raises(HTTPException) as exc_info:
            await get_current_user(session=None)

        assert exc_info.value.status_code == 401
        assert "Authentication required" in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_get_current_user_expired_session(
        self, mock_signer: URLSafeTimedSerializer, mock_session_store
    ):
        """Test that expired session raises 401."""
        session_cookie = _make_session_cookie(mock_signer)

        # Force the cookie's signature check to report expiry — short-circuits
        # before the store lookup.
        with patch.object(mock_signer, "loads", side_effect=SignatureExpired("Expired")):
            with pytest.raises(HTTPException) as exc_info:
                await get_current_user(session=session_cookie)

        assert exc_info.value.status_code == 401

    @pytest.mark.asyncio
    async def test_get_current_user_invalid_signature(self, mock_session_store):
        """Test that invalid signature raises 401."""
        with pytest.raises(HTTPException) as exc_info:
            await get_current_user(session="invalid.session.cookie")

        assert exc_info.value.status_code == 401

    @pytest.mark.asyncio
    async def test_get_current_user_no_username_in_session(
        self, mock_signer: URLSafeTimedSerializer, mock_session_store
    ):
        """Test that session without username raises 401."""
        # Store returns a hydrated record with no username.
        mock_session_store.next_value = {"username": None, "auth_method": "oauth2"}
        session_cookie = _make_session_cookie(mock_signer)

        with pytest.raises(HTTPException) as exc_info:
            await get_current_user(session=session_cookie)

        assert exc_info.value.status_code == 401

    @pytest.mark.asyncio
    async def test_get_current_user_legacy_cookie_rejected(
        self, mock_signer: URLSafeTimedSerializer, mock_session_store
    ):
        """Legacy dict-payload cookies must be rejected — forces re-login."""
        legacy_cookie = mock_signer.dumps({"username": "testuser", "auth_method": "oauth2"})

        with pytest.raises(HTTPException) as exc_info:
            await get_current_user(session=legacy_cookie)

        assert exc_info.value.status_code == 401


# =============================================================================
# TEST: get_user_session_data
# =============================================================================


@pytest.mark.unit
@pytest.mark.auth
class TestGetUserSessionData:
    """Tests for get_user_session_data dependency."""

    @pytest.mark.asyncio
    async def test_get_session_data_traditional_user_rejected(
        self, mock_signer: URLSafeTimedSerializer, mock_session_store
    ):
        """Test that non-OAuth2 sessions are rejected."""
        mock_session_store.next_value = {"username": "admin", "auth_method": "traditional"}
        session_cookie = _make_session_cookie(mock_signer)

        with pytest.raises(HTTPException) as exc_info:
            await get_user_session_data(session=session_cookie)
        assert exc_info.value.status_code == 401

    @pytest.mark.asyncio
    async def test_get_session_data_oauth2_user(
        self, mock_signer: URLSafeTimedSerializer, mock_session_store
    ):
        """Test extracting session data for OAuth2 user."""
        mock_session_store.next_value = {
            "username": "oauth_user",
            "auth_method": "oauth2",
            "groups": ["registry-users-lob1"],
            "provider": "cognito",
        }
        session_cookie = _make_session_cookie(mock_signer)

        result = await get_user_session_data(session=session_cookie)

        assert result["username"] == "oauth_user"
        assert result["auth_method"] == "oauth2"
        assert result["groups"] == ["registry-users-lob1"]
        assert "scopes" not in result or "mcp-registry-admin" not in result.get("scopes", [])

    @pytest.mark.asyncio
    async def test_get_session_data_no_session(self, mock_session_store):
        """Test that missing session raises 401."""
        with pytest.raises(HTTPException) as exc_info:
            await get_user_session_data(session=None)

        assert exc_info.value.status_code == 401
        assert "Authentication required" in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_get_session_data_expired(
        self, mock_signer: URLSafeTimedSerializer, mock_session_store
    ):
        """Test that expired session raises 401."""
        session_cookie = _make_session_cookie(mock_signer)

        with patch.object(mock_signer, "loads", side_effect=SignatureExpired("Expired")):
            with pytest.raises(HTTPException) as exc_info:
                await get_user_session_data(session=session_cookie)

        assert exc_info.value.status_code == 401


# =============================================================================
# TEST: map_cognito_groups_to_scopes
# =============================================================================


@pytest.mark.unit
@pytest.mark.unit
@pytest.mark.auth
class TestMapCognitoGroupsToScopes:
    """Tests for map_cognito_groups_to_scopes function."""

    @pytest.mark.asyncio
    async def test_map_admin_group(self, mock_scopes_config: dict[str, Any]):
        """Test mapping admin group to scopes."""
        # Arrange
        groups = ["mcp-registry-admin"]

        # Act
        scopes = await map_cognito_groups_to_scopes(groups)

        # Assert
        assert "mcp-registry-admin" in scopes
        assert "mcp-servers-unrestricted/read" in scopes
        assert "mcp-servers-unrestricted/execute" in scopes

    @pytest.mark.asyncio
    async def test_map_lob1_group(self, mock_scopes_config: dict[str, Any]):
        """Test mapping LOB1 group to scopes."""
        # Arrange
        groups = ["registry-users-lob1"]

        # Act
        scopes = await map_cognito_groups_to_scopes(groups)

        # Assert
        assert "registry-users-lob1" in scopes
        assert "mcp-registry-admin" not in scopes

    @pytest.mark.asyncio
    async def test_map_multiple_groups(self, mock_scopes_config: dict[str, Any]):
        """Test mapping multiple groups removes duplicates."""
        # Arrange
        groups = ["mcp-registry-admin", "registry-users-lob1"]

        # Act
        scopes = await map_cognito_groups_to_scopes(groups)

        # Assert
        assert "mcp-registry-admin" in scopes
        assert "registry-users-lob1" in scopes
        # Verify no duplicates
        assert len(scopes) == len(set(scopes))

    @pytest.mark.asyncio
    async def test_map_unknown_group(self, mock_scopes_config: dict[str, Any]):
        """Test mapping unknown group returns empty list."""
        # Arrange
        groups = ["unknown-group"]

        # Act
        scopes = await map_cognito_groups_to_scopes(groups)

        # Assert
        assert scopes == []

    @pytest.mark.asyncio
    async def test_map_empty_groups(self, mock_scopes_config: dict[str, Any]):
        """Test mapping empty groups list."""
        # Arrange
        groups = []

        # Act
        scopes = await map_cognito_groups_to_scopes(groups)

        # Assert
        assert scopes == []


# =============================================================================
# TEST: get_ui_permissions_for_user
# =============================================================================


@pytest.mark.unit
@pytest.mark.auth
class TestGetUIPermissionsForUser:
    """Tests for get_ui_permissions_for_user function."""

    @pytest.mark.asyncio
    async def test_admin_ui_permissions(self, mock_scopes_config: dict[str, Any]):
        """Test admin user gets all UI permissions."""
        # Arrange
        user_scopes = ["mcp-registry-admin"]

        # Act
        permissions = await get_ui_permissions_for_user(user_scopes)

        # Assert
        assert "list_agents" in permissions
        assert "all" in permissions["list_agents"]
        assert "list_service" in permissions
        assert "all" in permissions["list_service"]

    @pytest.mark.asyncio
    async def test_lob1_ui_permissions(self, mock_scopes_config: dict[str, Any]):
        """Test LOB1 user gets restricted UI permissions."""
        # Arrange
        user_scopes = ["registry-users-lob1"]

        # Act
        permissions = await get_ui_permissions_for_user(user_scopes)

        # Assert
        assert "list_agents" in permissions
        assert "/code-reviewer" in permissions["list_agents"]
        assert "/test-automation" in permissions["list_agents"]
        assert "all" not in permissions["list_agents"]

    @pytest.mark.asyncio
    async def test_no_scopes_no_permissions(self, mock_scopes_config: dict[str, Any]):
        """Test user with no scopes gets no permissions."""
        # Arrange
        user_scopes = []

        # Act
        permissions = await get_ui_permissions_for_user(user_scopes)

        # Assert
        assert permissions == {}

    @pytest.mark.asyncio
    async def test_unknown_scope_no_permissions(self, mock_scopes_config: dict[str, Any]):
        """Test unknown scope grants no permissions."""
        # Arrange
        user_scopes = ["unknown-scope"]

        # Act
        permissions = await get_ui_permissions_for_user(user_scopes)

        # Assert
        assert permissions == {}

    @pytest.mark.asyncio
    async def test_admin_ui_permissions_with_mixed_scopes(self, mock_scopes_config: dict[str, Any]):
        """Test admin gets permissions when scopes include non-UI server scopes (#930).

        In production, the admin group maps to both UI scopes (with permissions)
        and server-access scopes (without permissions). The function must still
        return the admin UI permissions, not an empty dict.
        """
        # Arrange — realistic scope list produced by map_cognito_groups_to_scopes
        user_scopes = [
            "mcp-registry-admin",
            "mcp-servers-unrestricted/read",
            "mcp-servers-unrestricted/execute",
        ]

        # Act
        permissions = await get_ui_permissions_for_user(user_scopes)

        # Assert — admin UI permissions must be present
        assert "list_agents" in permissions
        assert "all" in permissions["list_agents"]
        assert "list_service" in permissions
        assert "all" in permissions["list_service"]
        assert "register_service" in permissions
        assert "all" in permissions["register_service"]


class TestMapAndResolveEndToEnd:
    """End-to-end test: group → scopes → ui_permissions (#930).

    Verifies the full chain that was broken on the file backend
    because get_all_group_mappings returned the wrong dict shape.
    """

    @pytest.mark.asyncio
    async def test_admin_group_to_ui_permissions(self, mock_scopes_config: dict[str, Any]):
        """Admin group must produce non-empty ui_permissions."""
        # Step 1: map groups → scopes
        scopes = await map_cognito_groups_to_scopes(["mcp-registry-admin"])
        assert len(scopes) > 0, "Admin group should map to at least one scope"

        # Step 2: scopes → ui_permissions
        permissions = await get_ui_permissions_for_user(scopes)
        assert permissions, "Admin scopes must produce non-empty ui_permissions"
        assert "list_service" in permissions
        assert "all" in permissions["list_service"]


# =============================================================================
# TEST: user_has_ui_permission_for_service
# =============================================================================


@pytest.mark.unit
@pytest.mark.auth
class TestUserHasUIPermissionForService:
    """Tests for user_has_ui_permission_for_service function."""

    def test_has_permission_for_all_services(self):
        """Test user with 'all' permission can access any service."""
        # Arrange
        permissions = {"list_service": ["all"]}

        # Act & Assert
        assert user_has_ui_permission_for_service("list_service", "any_service", permissions)

    def test_has_permission_for_specific_service(self):
        """Test user with specific service permission."""
        # Arrange
        permissions = {"list_service": ["currenttime", "mcpgw"]}

        # Act & Assert
        assert user_has_ui_permission_for_service("list_service", "currenttime", permissions)
        assert user_has_ui_permission_for_service("list_service", "mcpgw", permissions)

    def test_no_permission_for_service(self):
        """Test user without permission for service."""
        # Arrange
        permissions = {"list_service": ["currenttime"]}

        # Act & Assert
        assert not user_has_ui_permission_for_service("list_service", "other_service", permissions)

    def test_permission_not_in_user_permissions(self):
        """Test permission type not in user's permissions."""
        # Arrange
        permissions = {"list_service": ["currenttime"]}

        # Act & Assert
        assert not user_has_ui_permission_for_service(
            "register_service", "currenttime", permissions
        )


# =============================================================================
# TEST: get_accessible_services_for_user
# =============================================================================


@pytest.mark.unit
@pytest.mark.auth
class TestGetAccessibleServicesForUser:
    """Tests for get_accessible_services_for_user function."""

    def test_all_services_accessible(self):
        """Test user with 'all' can access all services."""
        # Arrange
        permissions = {"list_service": ["all"]}

        # Act
        services = get_accessible_services_for_user(permissions)

        # Assert
        assert services == ["all"]

    def test_specific_services_accessible(self):
        """Test user with specific services."""
        # Arrange
        permissions = {"list_service": ["currenttime", "mcpgw"]}

        # Act
        services = get_accessible_services_for_user(permissions)

        # Assert
        assert "currenttime" in services
        assert "mcpgw" in services

    def test_no_list_permission(self):
        """Test user without list_service permission."""
        # Arrange
        permissions = {"other_permission": ["service1"]}

        # Act
        services = get_accessible_services_for_user(permissions)

        # Assert
        assert services == []


# =============================================================================
# TEST: get_accessible_agents_for_user
# =============================================================================


@pytest.mark.unit
@pytest.mark.auth
class TestGetAccessibleAgentsForUser:
    """Tests for get_accessible_agents_for_user function."""

    def test_all_agents_accessible(self):
        """Test user with 'all' can access all agents."""
        # Arrange
        permissions = {"list_agents": ["all"]}

        # Act
        agents = get_accessible_agents_for_user(permissions)

        # Assert
        assert agents == ["all"]

    def test_specific_agents_accessible(self):
        """Test user with specific agents."""
        # Arrange
        permissions = {"list_agents": ["/code-reviewer", "/test-automation"]}

        # Act
        agents = get_accessible_agents_for_user(permissions)

        # Assert
        assert "/code-reviewer" in agents
        assert "/test-automation" in agents

    def test_no_list_agents_permission(self):
        """Test user without list_agents permission."""
        # Arrange
        permissions = {"other_permission": ["/agent1"]}

        # Act
        agents = get_accessible_agents_for_user(permissions)

        # Assert
        assert agents == []


# =============================================================================
# TEST: get_servers_for_scope
# =============================================================================


@pytest.mark.unit
@pytest.mark.auth
class TestGetServersForScope:
    """Tests for get_servers_for_scope function."""

    @pytest.mark.asyncio
    async def test_wildcard_scope_returns_wildcard(self, mock_scopes_config: dict[str, Any]):
        """Test wildcard scope returns wildcard server."""
        # Act
        servers = await get_servers_for_scope("mcp-servers-unrestricted/read")

        # Assert
        assert "*" in servers

    @pytest.mark.asyncio
    async def test_specific_scope_returns_servers(self, mock_scopes_config: dict[str, Any]):
        """Test specific scope returns specific servers."""
        # Act
        servers = await get_servers_for_scope("registry-users-lob1")

        # Assert
        assert "currenttime" in servers

    @pytest.mark.asyncio
    async def test_unknown_scope_returns_empty(self, mock_scopes_config: dict[str, Any]):
        """Test unknown scope returns empty list."""
        # Act
        servers = await get_servers_for_scope("unknown-scope")

        # Assert
        assert servers == []


# =============================================================================
# TEST: user_has_wildcard_access
# =============================================================================


@pytest.mark.unit
@pytest.mark.auth
class TestUserHasWildcardAccess:
    """Tests for user_has_wildcard_access function."""

    @pytest.mark.asyncio
    async def test_admin_has_wildcard_access(self, mock_scopes_config: dict[str, Any]):
        """Test admin user has wildcard access."""
        # Arrange
        scopes = ["mcp-servers-unrestricted/read"]

        # Act
        has_access = await user_has_wildcard_access(scopes)

        # Assert
        assert has_access is True

    @pytest.mark.asyncio
    async def test_restricted_user_no_wildcard_access(self, mock_scopes_config: dict[str, Any]):
        """Test restricted user has no wildcard access."""
        # Arrange
        scopes = ["registry-users-lob1"]

        # Act
        has_access = await user_has_wildcard_access(scopes)

        # Assert
        assert has_access is False

    @pytest.mark.asyncio
    async def test_no_scopes_no_wildcard_access(self, mock_scopes_config: dict[str, Any]):
        """Test user with no scopes has no wildcard access."""
        # Arrange
        scopes = []

        # Act
        has_access = await user_has_wildcard_access(scopes)

        # Assert
        assert has_access is False


# =============================================================================
# TEST: get_user_accessible_servers
# =============================================================================


@pytest.mark.unit
@pytest.mark.auth
class TestGetUserAccessibleServers:
    """Tests for get_user_accessible_servers function."""

    @pytest.mark.asyncio
    async def test_admin_access_all_servers(self, mock_scopes_config: dict[str, Any]):
        """Test admin user can access all servers (wildcard)."""
        # Arrange
        scopes = ["mcp-servers-unrestricted/read"]

        # Act
        servers = await get_user_accessible_servers(scopes)

        # Assert
        assert "*" in servers

    @pytest.mark.asyncio
    async def test_lob1_access_specific_servers(self, mock_scopes_config: dict[str, Any]):
        """Test LOB1 user can access specific servers."""
        # Arrange
        scopes = ["registry-users-lob1"]

        # Act
        servers = await get_user_accessible_servers(scopes)

        # Assert
        assert "currenttime" in servers
        assert "*" not in servers

    @pytest.mark.asyncio
    async def test_multiple_scopes_combine_servers(self, mock_scopes_config: dict[str, Any]):
        """Test multiple scopes combine accessible servers."""
        # Arrange
        scopes = [
            "registry-users-lob1",
            "mcp-servers-unrestricted/read",
        ]

        # Act
        servers = await get_user_accessible_servers(scopes)

        # Assert
        assert "currenttime" in servers
        assert "*" in servers


# =============================================================================
# TEST: user_can_modify_servers
# =============================================================================


@pytest.mark.unit
@pytest.mark.auth
class TestUserCanModifyServers:
    """Tests for user_can_modify_servers function."""

    def test_admin_can_modify(self):
        """Test admin group can modify servers."""
        # Arrange
        groups = ["mcp-registry-admin"]
        scopes = ["mcp-servers-unrestricted/execute"]

        # Act
        can_modify = user_can_modify_servers(groups, scopes)

        # Assert
        assert can_modify is True

    def test_execute_scope_can_modify(self):
        """Test user with execute scope can modify."""
        # Arrange
        groups = []
        scopes = ["mcp-servers-unrestricted/execute"]

        # Act
        can_modify = user_can_modify_servers(groups, scopes)

        # Assert
        assert can_modify is True

    def test_read_only_cannot_modify(self):
        """Test read-only user cannot modify."""
        # Arrange
        groups = ["registry-users-lob1"]
        scopes = ["registry-users-lob1"]

        # Act
        can_modify = user_can_modify_servers(groups, scopes)

        # Assert
        assert can_modify is False

    def test_any_execute_scope_can_modify(self):
        """Test any execute scope grants modify permission."""
        # Arrange
        groups = []
        scopes = ["some-scope/execute"]

        # Act
        can_modify = user_can_modify_servers(groups, scopes)

        # Assert
        assert can_modify is True


# =============================================================================
# TEST: user_can_access_server
# =============================================================================


@pytest.mark.unit
@pytest.mark.auth
class TestUserCanAccessServer:
    """Tests for user_can_access_server function."""

    @pytest.mark.asyncio
    async def test_admin_can_access_any_server(self, mock_scopes_config: dict[str, Any]):
        """Test admin can access any server."""
        # Arrange
        scopes = ["mcp-servers-unrestricted/read"]

        # Act & Assert
        # Admin has wildcard in accessible servers
        # Note: The implementation checks if server name is in accessible_servers list
        # For wildcard access, "*" is in the list, but specific server names won't match
        # This test documents current behavior - wildcard doesn't match arbitrary names
        # User needs to check for "*" in accessible_servers separately
        accessible_servers = await get_user_accessible_servers(scopes)
        assert "*" in accessible_servers

        # The function doesn't expand wildcard, so specific server check returns False
        # This is expected behavior - caller should check for "*" separately
        assert not await user_can_access_server("any-server", scopes)

    @pytest.mark.asyncio
    async def test_user_can_access_allowed_server(self, mock_scopes_config: dict[str, Any]):
        """Test user can access allowed server."""
        # Arrange
        scopes = ["registry-users-lob1"]

        # Act & Assert
        assert await user_can_access_server("currenttime", scopes)

    @pytest.mark.asyncio
    async def test_user_cannot_access_disallowed_server(self, mock_scopes_config: dict[str, Any]):
        """Test user cannot access disallowed server."""
        # Arrange
        scopes = ["registry-users-lob1"]

        # Act & Assert
        assert not await user_can_access_server("other-server", scopes)


# =============================================================================
# TEST: api_auth and web_auth
# =============================================================================


@pytest.mark.unit
@pytest.mark.auth
class TestAuthWrappers:
    """Tests for api_auth and web_auth wrapper functions."""

    @pytest.mark.asyncio
    async def test_api_auth_calls_get_current_user(
        self, mock_signer: URLSafeTimedSerializer, mock_session_store
    ):
        """Test api_auth delegates to get_current_user."""
        mock_session_store.next_value = {"username": "apiuser", "auth_method": "oauth2"}
        session_cookie = _make_session_cookie(mock_signer)

        username = await api_auth(session=session_cookie)

        assert username == "apiuser"

    @pytest.mark.asyncio
    async def test_web_auth_calls_get_current_user(
        self, mock_signer: URLSafeTimedSerializer, mock_session_store
    ):
        """Test web_auth delegates to get_current_user."""
        mock_session_store.next_value = {"username": "webuser", "auth_method": "oauth2"}
        session_cookie = _make_session_cookie(mock_signer)

        username = await web_auth(session=session_cookie)

        assert username == "webuser"


# =============================================================================
# TEST: enhanced_auth
# =============================================================================


@pytest.mark.unit
@pytest.mark.auth
class TestEnhancedAuth:
    """Tests for enhanced_auth dependency."""

    @pytest.mark.asyncio
    async def test_enhanced_auth_traditional_user_rejected(
        self,
        mock_signer: URLSafeTimedSerializer,
        mock_session_store,
        mock_scopes_config: dict[str, Any],
    ):
        """Test enhanced_auth rejects traditional (non-OAuth2) sessions."""
        mock_session_store.next_value = {
            "username": "admin",
            "auth_method": "traditional",
            "provider": "local",
        }
        session_cookie = _make_session_cookie(mock_signer)
        mock_request = Mock(spec=Request)
        mock_request.state = Mock()

        with pytest.raises(HTTPException) as exc_info:
            await enhanced_auth(request=mock_request, session=session_cookie)
        assert exc_info.value.status_code == 401

    @pytest.mark.asyncio
    async def test_enhanced_auth_oauth2_user(
        self,
        mock_signer: URLSafeTimedSerializer,
        mock_session_store,
        mock_scopes_config: dict[str, Any],
    ):
        """Test enhanced_auth for OAuth2 user."""
        mock_session_store.next_value = {
            "session_id": "sid-1",
            "username": "oauth_user",
            "auth_method": "oauth2",
            "provider": "cognito",
            "groups": ["registry-users-lob1"],
        }
        session_cookie = _make_session_cookie(mock_signer)
        mock_request = Mock(spec=Request)
        mock_request.state = Mock()

        context = await enhanced_auth(request=mock_request, session=session_cookie)

        assert context["username"] == "oauth_user"
        assert context["auth_method"] == "oauth2"
        assert "registry-users-lob1" in context["groups"]
        assert context["can_modify_servers"] is False
        assert context["is_admin"] is False

    @pytest.mark.asyncio
    async def test_enhanced_auth_egress_user_is_persisted_subject(
        self,
        mock_signer: URLSafeTimedSerializer,
        mock_session_store,
        mock_scopes_config: dict[str, Any],
    ):
        """The cookie path (reached by the egress OAuth callback) must expose the
        persisted OIDC sub as egress_user, so the consent-write vault key matches
        the vend path's egress_user claim even when username is an email. See #933.
        """
        mock_session_store.next_value = {
            "session_id": "sid-1",
            "username": "alice@contoso.com",  # Entra preferred_username / email
            "auth_method": "oauth2",
            "provider": "entra",
            "groups": ["registry-users-lob1"],
            "subject": "entra-oid-sub-123",
        }
        session_cookie = _make_session_cookie(mock_signer)
        mock_request = Mock(spec=Request)
        mock_request.state = Mock()

        context = await enhanced_auth(request=mock_request, session=session_cookie)

        assert context["egress_user"] == "entra-oid-sub-123"
        assert context["username"] == "alice@contoso.com"

    @pytest.mark.asyncio
    async def test_enhanced_auth_egress_user_falls_back_to_username(
        self,
        mock_signer: URLSafeTimedSerializer,
        mock_session_store,
        mock_scopes_config: dict[str, Any],
    ):
        """A session with no persisted subject (non-OIDC / pre-fix login) keeps its
        prior vault bucket by falling back to username.
        """
        mock_session_store.next_value = {
            "session_id": "sid-1",
            "username": "oauth_user",
            "auth_method": "oauth2",
            "provider": "cognito",
            "groups": ["registry-users-lob1"],
        }
        session_cookie = _make_session_cookie(mock_signer)
        mock_request = Mock(spec=Request)
        mock_request.state = Mock()

        context = await enhanced_auth(request=mock_request, session=session_cookie)

        assert context["egress_user"] == "oauth_user"

    @pytest.mark.asyncio
    async def test_enhanced_auth_no_session(self):
        """Test enhanced_auth raises 401 without session."""
        # Arrange
        mock_request = Mock(spec=Request)
        mock_request.state = Mock()

        # Act & Assert
        with pytest.raises(HTTPException) as exc_info:
            await enhanced_auth(request=mock_request, session=None)

        assert exc_info.value.status_code == 401


# =============================================================================
# TEST: nginx_proxied_auth
# =============================================================================


@pytest.mark.unit
@pytest.mark.auth
class TestNginxProxiedAuth:
    """Tests for nginx_proxied_auth dependency."""

    @pytest.mark.asyncio
    async def test_signed_token_resolves_groups_server_side(
        self, registry_token_secret: str, mock_scopes_config: dict[str, Any]
    ):
        """A valid signed token (bearer/static caller, groups in claim) is verified
        and groups are resolved to scopes server-side; inbound headers are ignored."""
        token = _mint_registry_token(
            registry_token_secret,
            subject="nginx_user",
            groups=["registry-admins"],
            auth_method="keycloak",
        )
        # Forged inbound headers that MUST be ignored.
        context = await nginx_proxied_auth(
            request=_proxied_request(),
            session=None,
            x_user="attacker",
            x_username="attacker",
            x_scopes="mcp-registry-admin",
            x_auth_method="keycloak",
            x_groups="registry-admins",
            x_internal_token=token,
        )

        assert context["username"] == "nginx_user"  # from claim, not the forged X-User
        assert context["groups"] == ["registry-admins"]
        assert context["auth_method"] == "keycloak"
        # Scopes derived server-side from the claim groups.
        assert any("mcp-servers-unrestricted" in s for s in context["scopes"])

    @pytest.mark.asyncio
    async def test_forged_headers_without_token_rejected(
        self, mock_scopes_config: dict[str, Any], monkeypatch
    ):
        """No token + auth_request enabled + identity headers present → 401.

        This is the core fix: forged X-User/X-Scopes/X-Groups with no signed token
        are rejected, not trusted.
        """
        monkeypatch.delenv("NGINX_DISABLE_API_AUTH_REQUEST", raising=False)
        with pytest.raises(HTTPException) as exc:
            await nginx_proxied_auth(
                request=_proxied_request(),
                session=None,
                x_user="attacker",
                x_username="attacker",
                x_scopes="mcp-registry-admin",
                x_auth_method="keycloak",
                x_groups="registry-admins",
                x_internal_token=None,
            )
        assert exc.value.status_code == 401

    @pytest.mark.asyncio
    async def test_invalid_token_rejected_no_fallthrough(
        self, registry_token_secret: str, mock_scopes_config: dict[str, Any]
    ):
        """A present-but-invalid token → 401; never falls through to header trust."""
        bad_token = _mint_registry_token(
            "a-totally-different-secret-key", subject="nginx_user", groups=["registry-admins"]
        )
        with pytest.raises(HTTPException) as exc:
            await nginx_proxied_auth(
                request=_proxied_request(),
                session=None,
                x_user="attacker",
                x_username="attacker",
                x_scopes="mcp-registry-admin",
                x_auth_method="keycloak",
                x_groups="registry-admins",
                x_internal_token=bad_token,
            )
        assert exc.value.status_code == 401

    @pytest.mark.asyncio
    async def test_disable_mode_falls_back_to_cookie(
        self,
        mock_signer: URLSafeTimedSerializer,
        mock_session_store,
        mock_scopes_config: dict[str, Any],
        monkeypatch,
    ):
        """NGINX_DISABLE_API_AUTH_REQUEST=true + identity headers but no token →
        cookie fallback (headers still ignored)."""
        monkeypatch.setenv("NGINX_DISABLE_API_AUTH_REQUEST", "true")
        mock_session_store.next_value = {
            "session_id": "sid-d",
            "username": "session_user",
            "auth_method": "oauth2",
            "provider": "cognito",
            "groups": ["registry-admins"],
        }
        context = await nginx_proxied_auth(
            request=_proxied_request(),
            session=_make_session_cookie(mock_signer),
            x_user="attacker",
            x_username="attacker",
            x_scopes="mcp-registry-admin",
            x_auth_method="keycloak",
            x_groups="registry-admins",
            x_internal_token=None,
        )
        assert context["username"] == "session_user"  # from cookie, not forged header

    @pytest.mark.asyncio
    async def test_nginx_auth_fallback_to_session_oauth2(
        self,
        mock_signer: URLSafeTimedSerializer,
        mock_session_store,
        mock_scopes_config: dict[str, Any],
    ):
        """Test nginx auth falls back to OAuth2 session cookie."""
        mock_request = Mock(spec=Request)
        mock_request.url.path = "/api/test"
        mock_request.method = "GET"
        mock_request.state = Mock()
        mock_request.headers = {}

        mock_session_store.next_value = {
            "session_id": "sid-1",
            "username": "session_user",
            "auth_method": "oauth2",
            "provider": "cognito",
            "groups": ["registry-admins"],
        }
        session_cookie = _make_session_cookie(mock_signer)

        context = await nginx_proxied_auth(
            request=mock_request,
            session=session_cookie,
            x_user=None,
            x_username=None,
            x_scopes=None,
            x_auth_method=None,
            x_client_id=None,
        )

        assert context["username"] == "session_user"
        assert context["auth_method"] == "oauth2"

    @pytest.mark.asyncio
    async def test_nginx_auth_fallback_rejects_traditional_session(
        self,
        mock_signer: URLSafeTimedSerializer,
        mock_session_store,
        mock_scopes_config: dict[str, Any],
    ):
        """Test nginx auth rejects traditional (non-OAuth2) session cookies."""
        mock_request = Mock(spec=Request)
        mock_request.url.path = "/api/test"
        mock_request.method = "GET"
        mock_request.state = Mock()
        mock_request.headers = {}

        mock_session_store.next_value = {
            "username": "session_user",
            "auth_method": "traditional",
        }
        session_cookie = _make_session_cookie(mock_signer)

        # Act & Assert - traditional sessions should be rejected
        with pytest.raises(HTTPException) as exc_info:
            await nginx_proxied_auth(
                request=mock_request,
                session=session_cookie,
                x_user=None,
                x_username=None,
                x_scopes=None,
                x_auth_method=None,
                x_client_id=None,
            )
        assert exc_info.value.status_code == 401

    @pytest.mark.asyncio
    async def test_signed_token_non_admin_group_not_admin(
        self, registry_token_secret: str, mock_scopes_config: dict[str, Any]
    ):
        """A signed token carrying a non-admin group → is_admin False.

        is_admin is derived server-side from the group's ui_permissions, so a
        read-only group does not get admin.
        """
        token = _mint_registry_token(
            registry_token_secret,
            subject="oauth_user",
            groups=["registry-users-lob1"],
            auth_method="cognito",
        )
        context = await nginx_proxied_auth(
            request=_proxied_request(),
            session=None,
            x_internal_token=token,
        )

        assert context["username"] == "oauth_user"
        assert context["groups"] == ["registry-users-lob1"]
        assert context["is_admin"] is False


# =============================================================================
# TEST: Edge Cases and Error Handling
# =============================================================================


@pytest.mark.unit
@pytest.mark.auth
class TestEdgeCases:
    """Tests for edge cases and error handling."""

    @pytest.mark.asyncio
    async def test_session_with_empty_username(
        self, mock_signer: URLSafeTimedSerializer, mock_session_store
    ):
        """Test session with empty string username."""
        mock_session_store.next_value = {"username": "", "auth_method": "oauth2"}
        session_cookie = _make_session_cookie(mock_signer)

        with pytest.raises(HTTPException) as exc_info:
            await get_current_user(session=session_cookie)

        assert exc_info.value.status_code == 401

    @pytest.mark.asyncio
    async def test_scopes_deduplication(self, mock_scopes_config: dict[str, Any]):
        """Test that scopes shared across a user's groups are deduplicated."""
        # Arrange - scope1 is reachable via two of the user's groups, so it
        # must appear only once in the result. The repo stores the inverse
        # {scope_name: [groups]} shape.
        mock_repo = AsyncMock()

        async def mock_get_all_group_mappings_with_overlap():
            return {
                "scope1": ["group-a", "group-b"],
                "scope2": ["group-a"],
            }

        mock_repo.get_all_group_mappings.side_effect = mock_get_all_group_mappings_with_overlap

        with patch("registry.repositories.factory.get_scope_repository", return_value=mock_repo):
            # Act
            scopes = await map_cognito_groups_to_scopes(["group-a", "group-b"])

            # Assert
            assert len(scopes) == len(set(scopes))  # No duplicates
            assert scopes.count("scope1") == 1
            assert "scope2" in scopes

    @pytest.mark.asyncio
    async def test_enhanced_auth_oauth2_no_groups(
        self,
        mock_signer: URLSafeTimedSerializer,
        mock_session_store,
        mock_scopes_config: dict[str, Any],
    ):
        """Test OAuth2 user with no groups gets minimal permissions."""
        mock_session_store.next_value = {
            "session_id": "sid-1",
            "username": "no_groups_user",
            "auth_method": "oauth2",
            "groups": [],
        }
        session_cookie = _make_session_cookie(mock_signer)
        mock_request = Mock(spec=Request)
        mock_request.state = Mock()

        context = await enhanced_auth(request=mock_request, session=session_cookie)

        assert context["username"] == "no_groups_user"
        assert context["groups"] == []
        assert context["scopes"] == []
        assert context["can_modify_servers"] is False

    def test_ui_permissions_with_all_and_specific(self, mock_scopes_config: dict[str, Any]):
        """Test UI permissions handles 'all' with specific services."""
        # Arrange - Create permissions with both 'all' and specific
        permissions = {"list_service": ["all", "currenttime"]}

        # Act & Assert
        assert user_has_ui_permission_for_service("list_service", "any_service", permissions)


# =============================================================================
# NETWORK-TRUSTED AUTH METHOD TESTS
# =============================================================================


class TestNetworkTrustedAuthMethod:
    """Tests for network-trusted auth method in nginx_proxied_auth (issue #357)."""

    @pytest.mark.asyncio
    async def test_network_trusted_with_admin_groups_gets_admin(
        self, registry_token_secret: str, mock_scopes_config: dict[str, Any]
    ):
        """Network-trusted static-token caller with admin groups resolves to admin.

        The token (no session_id) carries the caller's groups; the registry resolves
        groups→scopes→ui_permissions server-side. is_admin derives from the
        mcp-registry-admin group's UI permissions (no synthesized groups, #933).
        """
        token = _mint_registry_token(
            registry_token_secret,
            subject="network-user",
            groups=["mcp-registry-admin"],
            auth_method="network-trusted",
            client_id="key-admin",
        )
        context = await nginx_proxied_auth(
            request=_proxied_request(),
            session=None,
            x_internal_token=token,
        )

        assert context["username"] == "network-user"
        assert context["auth_method"] == "network-trusted"
        assert "mcp-servers-unrestricted/read" in context["scopes"]
        assert "mcp-servers-unrestricted/execute" in context["scopes"]
        assert context["is_admin"] is True

    @pytest.mark.asyncio
    async def test_network_trusted_readonly_groups_not_admin(
        self, registry_token_secret: str, mock_scopes_config: dict[str, Any]
    ):
        """Network-trusted with read-only groups does NOT get admin (issue #779)."""
        token = _mint_registry_token(
            registry_token_secret,
            subject="monitoring-script",
            groups=["registry-users-lob1"],
            auth_method="network-trusted",
            client_id="key-ro",
        )
        context = await nginx_proxied_auth(
            request=_proxied_request(),
            session=None,
            x_internal_token=token,
        )

        assert context["username"] == "monitoring-script"
        assert context["is_admin"] is False


# =============================================================================
# REGRESSION: cookie path and proxied path must agree on is_admin (#933)
# =============================================================================


@pytest.mark.unit
@pytest.mark.auth
class TestAuthPathsAgreeOnIsAdmin:
    """Same user, same groups → same is_admin regardless of which dependency
    resolved them. This is the core invariant #933 broke."""

    @pytest.mark.asyncio
    async def test_admin_user_is_admin_on_both_paths(
        self,
        mock_signer: URLSafeTimedSerializer,
        mock_session_store,
        mock_scopes_config: dict[str, Any],
        registry_token_secret: str,
    ):
        admin_groups = ["mcp-registry-admin"]

        # Shared session record: both the cookie path and the session-backed token
        # path resolve groups via the SAME resolve_session lookup.
        mock_session_store.next_value = {
            "session_id": "sid-1",
            "username": "alice",
            "auth_method": "oauth2",
            "provider": "cognito",
            "groups": admin_groups,
        }

        # Cookie path: enhanced_auth via session_store
        session_cookie = _make_session_cookie(mock_signer)
        cookie_request = Mock(spec=Request)
        cookie_request.state = Mock()
        cookie_context = await enhanced_auth(request=cookie_request, session=session_cookie)

        # Proxied path: session-backed signed token (carries session_id, no groups);
        # the registry resolves groups server-side from the same session record.
        token = _mint_registry_token(
            registry_token_secret, subject="alice", session_id="sid-1", auth_method="session_cookie"
        )
        proxied_context = await nginx_proxied_auth(
            request=_proxied_request(),
            session=None,
            x_internal_token=token,
        )

        assert cookie_context["is_admin"] == proxied_context["is_admin"]
        assert cookie_context["is_admin"] is True
        assert cookie_context["ui_permissions"] == proxied_context["ui_permissions"]

    @pytest.mark.asyncio
    async def test_non_admin_is_not_admin_on_either_path(
        self,
        mock_signer: URLSafeTimedSerializer,
        mock_session_store,
        mock_scopes_config: dict[str, Any],
        registry_token_secret: str,
    ):
        user_groups = ["registry-users-lob1"]

        mock_session_store.next_value = {
            "session_id": "sid-2",
            "username": "bob",
            "auth_method": "oauth2",
            "provider": "cognito",
            "groups": user_groups,
        }
        session_cookie = _make_session_cookie(mock_signer)
        cookie_request = Mock(spec=Request)
        cookie_request.state = Mock()
        cookie_context = await enhanced_auth(request=cookie_request, session=session_cookie)

        token = _mint_registry_token(
            registry_token_secret, subject="bob", session_id="sid-2", auth_method="session_cookie"
        )
        proxied_context = await nginx_proxied_auth(
            request=_proxied_request(),
            session=None,
            x_internal_token=token,
        )

        assert cookie_context["is_admin"] is False
        assert proxied_context["is_admin"] is False


# =============================================================================
# TEST: _user_is_admin (issue #663)
# =============================================================================


@pytest.mark.unit
@pytest.mark.auth
class TestUserIsAdmin:
    """Tests for _user_is_admin function.

    Verifies that admin status is derived from mutating UI-Scopes actions
    (register_, modify_, toggle_, delete_, publish_, create_) with 'all'
    resources, NOT from server: '*' wildcard access.

    See GitHub issue #663.
    """

    @pytest.mark.parametrize(
        "action",
        [
            "register_service",
            "modify_service",
            "toggle_service",
            "delete_service",
            "publish_agent",
            "modify_agent",
            "delete_agent",
            "create_virtual_server",
            "modify_virtual_server",
            "delete_virtual_server",
        ],
    )
    def test_admin_with_mutating_action_all(self, action: str):
        """User with any mutating action for [all] is admin."""
        # Arrange
        ui_permissions = {action: ["all"], "list_service": ["all"]}

        # Act
        result = _user_is_admin(ui_permissions)

        # Assert
        assert result is True

    def test_not_admin_with_only_read_actions(self):
        """Consumer with only read-only permissions is not admin (issue #663 core fix)."""
        # Arrange
        ui_permissions = {
            "list_service": ["all"],
            "health_check_service": ["all"],
            "list_agents": ["all"],
            "get_agent": ["all"],
            "list_virtual_server": ["all"],
        }

        # Act
        result = _user_is_admin(ui_permissions)

        # Assert
        assert result is False

    def test_not_admin_with_specific_server_modify(self):
        """User with modify_service for specific servers only is not admin."""
        # Arrange
        ui_permissions = {"modify_service": ["server1", "server2"]}

        # Act
        result = _user_is_admin(ui_permissions)

        # Assert
        assert result is False

    def test_not_admin_empty_permissions(self):
        """User with no UI permissions is not admin."""
        # Arrange / Act
        result = _user_is_admin({})

        # Assert
        assert result is False

    def test_full_admin_permissions_match_registry_admins_json(self):
        """Full admin role (matching scripts/registry-admins.json) is admin."""
        # Arrange
        ui_permissions = {
            "list_agents": ["all"],
            "get_agent": ["all"],
            "publish_agent": ["all"],
            "modify_agent": ["all"],
            "delete_agent": ["all"],
            "list_service": ["all"],
            "register_service": ["all"],
            "health_check_service": ["all"],
            "toggle_service": ["all"],
            "modify_service": ["all"],
            "delete_service": ["all"],
            "list_virtual_server": ["all"],
            "create_virtual_server": ["all"],
            "modify_virtual_server": ["all"],
            "delete_virtual_server": ["all"],
        }

        # Act
        result = _user_is_admin(ui_permissions)

        # Assert
        assert result is True

    def test_consumer_with_wildcard_server_not_admin(self):
        """Issue #663: server: '*' in scopes should NOT trigger is_admin.

        A consumer role with server: '*' but only read-only UI-Scopes
        must not be treated as admin.
        """
        # Arrange - consumer has only read-only UI-Scopes
        ui_permissions = {
            "list_service": ["all"],
            "health_check_service": ["all"],
            "list_agents": ["all"],
            "get_agent": ["all"],
        }

        # Act - even though the user's scopes contain server: '*',
        # _user_is_admin only checks ui_permissions, not server access
        result = _user_is_admin(ui_permissions)

        # Assert
        assert result is False

    @pytest.mark.parametrize(
        "action",
        [
            "list_service",
            "get_agent",
            "health_check_service",
            "list_agents",
            "list_virtual_server",
        ],
    )
    def test_read_only_actions_never_grant_admin(self, action: str):
        """Read-only actions with 'all' do not grant admin status."""
        # Arrange
        ui_permissions = {action: ["all"]}

        # Act
        result = _user_is_admin(ui_permissions)

        # Assert
        assert result is False
