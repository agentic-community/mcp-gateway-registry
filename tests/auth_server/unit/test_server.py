"""
Unit tests for auth_server/server.py

Tests cover token validation, session management, scope validation,
rate limiting, and helper functions.
"""

import logging
import time
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import jwt
import pytest
from fastapi.testclient import TestClient

logger = logging.getLogger(__name__)


# Mark all tests in this file
pytestmark = [pytest.mark.unit, pytest.mark.auth]


# =============================================================================
# HELPER FUNCTION TESTS
# =============================================================================


class TestMaskingFunctions:
    """Tests for sensitive data masking functions."""

    def test_mask_sensitive_id_short(self):
        """Test masking short IDs."""
        from auth_server.server import mask_sensitive_id

        # Arrange
        short_id = "abc"

        # Act
        result = mask_sensitive_id(short_id)

        # Assert
        assert result == "***MASKED***"

    def test_mask_sensitive_id_normal(self):
        """Test masking normal length IDs."""
        from auth_server.server import mask_sensitive_id

        # Arrange
        normal_id = "us-east-1_ABCD12345"

        # Act
        result = mask_sensitive_id(normal_id)

        # Assert
        assert result.startswith("us-e")
        assert result.endswith("2345")
        assert "..." in result

    def test_mask_token(self):
        """Test masking JWT tokens showing first 4 characters."""
        from auth_server.server import mask_token

        # Arrange
        token = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.test"

        # Act
        result = mask_token(token)

        # Assert
        assert result.startswith("eyJh")
        assert result.endswith("...")
        assert len(result) < len(token)

    def test_anonymize_ip_ipv4(self):
        """Test IPv4 anonymization."""
        from auth_server.server import anonymize_ip

        # Arrange
        ipv4 = "192.168.1.100"

        # Act
        result = anonymize_ip(ipv4)

        # Assert
        assert result == "192.168.1.xxx"

    def test_anonymize_ip_ipv6(self):
        """Test IPv6 anonymization."""
        from auth_server.server import anonymize_ip

        # Arrange
        ipv6 = "2001:0db8:85a3:0000:0000:8a2e:0370:7334"

        # Act
        result = anonymize_ip(ipv6)

        # Assert
        assert result.endswith(":xxxx")
        assert "2001" in result

    def test_hash_username(self):
        """Test username hashing for privacy."""
        from auth_server.server import hash_username

        # Arrange
        username = "testuser"

        # Act
        result = hash_username(username)

        # Assert
        assert result.startswith("user_")
        assert len(result) > len(username)
        # Same input produces same hash
        assert hash_username(username) == result


class TestServerNameNormalization:
    """Tests for server name normalization and matching."""

    def test_normalize_server_name_with_trailing_slash(self):
        """Test removing trailing slash."""
        from auth_server.server import _normalize_server_name

        # Arrange
        name_with_slash = "test-server/"

        # Act
        result = _normalize_server_name(name_with_slash)

        # Assert
        assert result == "test-server"

    def test_normalize_server_name_without_trailing_slash(self):
        """Test name without trailing slash."""
        from auth_server.server import _normalize_server_name

        # Arrange
        name = "test-server"

        # Act
        result = _normalize_server_name(name)

        # Assert
        assert result == "test-server"

    def test_server_names_match_exact(self):
        """Test exact server name matching."""
        from auth_server.server import _server_names_match

        # Act & Assert
        assert _server_names_match("test-server", "test-server")

    def test_server_names_match_with_trailing_slash(self):
        """Test server name matching with trailing slash."""
        from auth_server.server import _server_names_match

        # Act & Assert
        assert _server_names_match("test-server/", "test-server")
        assert _server_names_match("test-server", "test-server/")

    def test_server_names_match_wildcard(self):
        """Test wildcard matching."""
        from auth_server.server import _server_names_match

        # Act & Assert
        assert _server_names_match("*", "any-server")
        assert _server_names_match("*", "another-server")


class TestGroupToScopeMapping:
    """Tests for mapping IdP groups to MCP scopes."""

    @pytest.mark.asyncio
    async def test_map_groups_to_scopes_basic(self, mock_scopes_config):
        """Test basic group to scope mapping."""
        from auth_server.server import map_groups_to_scopes

        # Arrange - Mock the repository to return scopes for groups
        mock_repo = AsyncMock()
        mock_repo.get_group_mappings.side_effect = lambda group: {
            "users": ["read:servers", "read:tools"],
            "developers": ["write:servers"],
        }.get(group, [])

        with patch("auth_server.server.get_scope_repository", return_value=mock_repo):
            groups = ["users", "developers"]

            # Act
            scopes = await map_groups_to_scopes(groups)

            # Assert
            assert "read:servers" in scopes
            assert "write:servers" in scopes
            assert "read:tools" in scopes

    @pytest.mark.asyncio
    async def test_map_groups_to_scopes_no_duplicates(self, mock_scopes_config):
        """Test that duplicate scopes are removed."""
        from auth_server.server import map_groups_to_scopes

        # Arrange - Mock the repository to return scopes for groups
        mock_repo = AsyncMock()
        # Both groups return "read:servers" to test deduplication
        mock_repo.get_group_mappings.side_effect = lambda group: {
            "users": ["read:servers", "read:tools"],
            "developers": ["read:servers", "write:servers"],
        }.get(group, [])

        with patch("auth_server.server.get_scope_repository", return_value=mock_repo):
            # Both groups have "read:servers"
            groups = ["users", "developers"]

            # Act
            scopes = await map_groups_to_scopes(groups)

            # Assert
            # Should only appear once (duplicates removed)
            assert scopes.count("read:servers") == 1
            assert "write:servers" in scopes
            assert "read:tools" in scopes

    @pytest.mark.asyncio
    async def test_map_groups_to_scopes_unknown_group(self, mock_scopes_config):
        """Test mapping with unknown group."""
        from auth_server.server import map_groups_to_scopes

        # Arrange - Mock repository to return empty list for unknown groups
        mock_repo = AsyncMock()
        mock_repo.get_group_mappings.return_value = []

        with patch("auth_server.server.get_scope_repository", return_value=mock_repo):
            groups = ["unknown-group"]

            # Act
            scopes = await map_groups_to_scopes(groups)

            # Assert
            assert len(scopes) == 0


class TestScopeValidation:
    """Tests for scope-based access validation."""

    @pytest.mark.asyncio
    async def test_validate_server_tool_access_allowed(self, mock_scope_repository_with_data):
        """Test access validation when allowed."""
        from auth_server.server import validate_server_tool_access

        # Arrange
        with patch(
            "auth_server.server.get_scope_repository", return_value=mock_scope_repository_with_data
        ):
            server_name = "test-server"
            method = "initialize"
            tool_name = None
            user_scopes = ["read:servers"]

            # Act
            result = await validate_server_tool_access(server_name, method, tool_name, user_scopes)

            # Assert
            assert result is True

    @pytest.mark.asyncio
    async def test_validate_server_tool_access_denied(self, mock_scope_repository_with_data):
        """Test access validation when denied."""
        from auth_server.server import validate_server_tool_access

        # Arrange
        with patch(
            "auth_server.server.get_scope_repository", return_value=mock_scope_repository_with_data
        ):
            server_name = "other-server"
            method = "initialize"
            tool_name = None
            user_scopes = ["read:servers"]  # Only for test-server

            # Act
            result = await validate_server_tool_access(server_name, method, tool_name, user_scopes)

            # Assert
            assert result is False

    @pytest.mark.asyncio
    async def test_validate_server_tool_access_wildcard_server(
        self, mock_scope_repository_with_data
    ):
        """Test wildcard server access."""
        from auth_server.server import validate_server_tool_access

        # Arrange
        with patch(
            "auth_server.server.get_scope_repository", return_value=mock_scope_repository_with_data
        ):
            server_name = "any-server"
            method = "initialize"
            tool_name = None
            user_scopes = ["admin:all"]

            # Act
            result = await validate_server_tool_access(server_name, method, tool_name, user_scopes)

            # Assert
            assert result is True

    @pytest.mark.asyncio
    async def test_validate_server_tool_access_tools_call(self, mock_scope_repository_with_data):
        """Test access validation for tools/call method."""
        from auth_server.server import validate_server_tool_access

        # Arrange
        with patch(
            "auth_server.server.get_scope_repository", return_value=mock_scope_repository_with_data
        ):
            server_name = "test-server"
            method = "tools/call"
            tool_name = "test-tool"
            user_scopes = ["write:servers"]  # Has wildcard tools

            # Act
            result = await validate_server_tool_access(server_name, method, tool_name, user_scopes)

            # Assert
            assert result is True

    def test_validate_scope_subset_valid(self):
        """Test that requested scopes are subset of user scopes."""
        from auth_server.server import validate_scope_subset

        # Arrange
        user_scopes = ["read:servers", "write:servers", "admin:all"]
        requested_scopes = ["read:servers", "write:servers"]

        # Act
        result = validate_scope_subset(user_scopes, requested_scopes)

        # Assert
        assert result is True

    def test_validate_scope_subset_invalid(self):
        """Test that requested scopes exceed user scopes."""
        from auth_server.server import validate_scope_subset

        # Arrange
        user_scopes = ["read:servers"]
        requested_scopes = ["read:servers", "write:servers"]

        # Act
        result = validate_scope_subset(user_scopes, requested_scopes)

        # Assert
        assert result is False


class TestRateLimiting:
    """Tests for token generation rate limiting."""

    def test_check_rate_limit_under_limit(self):
        """Test rate limiting when under limit."""
        from auth_server.server import check_rate_limit, user_token_generation_counts

        # Arrange
        user_token_generation_counts.clear()
        username = "testuser"

        # Act
        result = check_rate_limit(username)

        # Assert
        assert result is True

    def test_check_rate_limit_exceeded(self, monkeypatch):
        """Test rate limiting when limit exceeded."""
        from auth_server.server import check_rate_limit, user_token_generation_counts

        # Arrange
        monkeypatch.setenv("MAX_TOKENS_PER_USER_PER_HOUR", "3")
        from auth_server import server

        server.MAX_TOKENS_PER_USER_PER_HOUR = 3

        user_token_generation_counts.clear()
        username = "testuser"

        # Generate tokens up to limit
        for _ in range(3):
            check_rate_limit(username)

        # Act - try one more
        result = check_rate_limit(username)

        # Assert
        assert result is False

    def test_check_rate_limit_cleanup_old_entries(self):
        """Test that old rate limit entries are cleaned up."""
        from auth_server.server import check_rate_limit, user_token_generation_counts

        # Arrange
        user_token_generation_counts.clear()
        username = "testuser"
        current_time = int(time.time())
        old_hour = (current_time // 3600) - 2  # 2 hours ago

        # Add old entry
        user_token_generation_counts[f"{username}:{old_hour}"] = 5

        # Act
        check_rate_limit(username)

        # Assert - old entry should be removed
        assert f"{username}:{old_hour}" not in user_token_generation_counts


# =============================================================================
# SESSION COOKIE VALIDATION TESTS
# =============================================================================


class TestSessionCookieValidation:
    """Tests for session cookie validation."""

    @pytest.mark.asyncio
    async def test_validate_session_cookie_valid(self, auth_env_vars, valid_session_cookie):
        """Test validating a valid session cookie.

        With server-side sessions, validation does signer.loads(cookie) →
        session_id (string), then store.resolve_session(session_id) →
        hydrated record. We mock the store lookup.
        """
        from itsdangerous import URLSafeTimedSerializer

        from auth_server.server import validate_session_cookie

        test_signer = URLSafeTimedSerializer(auth_env_vars["SECRET_KEY"])

        async def _fake_resolve(_session_id):
            return {
                "session_id": _session_id,
                "username": "testuser",
                "email": "testuser@example.com",
                "groups": ["users", "developers"],
                "provider": "cognito",
                "auth_method": "oauth2",
            }

        with (
            patch("auth_server.server.signer", test_signer),
            patch("session_store.resolve_session", _fake_resolve),
        ):
            result = await validate_session_cookie(valid_session_cookie)

            assert result["valid"] is True
            assert result["username"] == "testuser"
            assert result["method"] == "session_cookie"
            assert "users" in result["groups"]

    @pytest.mark.asyncio
    async def test_validate_session_cookie_expired(self, auth_env_vars):
        """Test validating an expired session cookie."""
        from itsdangerous import URLSafeTimedSerializer

        from auth_server.server import validate_session_cookie

        # Create signer with test key
        test_signer = URLSafeTimedSerializer(auth_env_vars["SECRET_KEY"])

        # Create cookie with far past timestamp
        old_data = {"username": "testuser", "groups": []}
        import time

        old_time = time.time() - 30000  # Way past max_age
        with patch("time.time", return_value=old_time):
            old_cookie = test_signer.dumps(old_data)

        # Patch the module's signer to use test key
        with patch("auth_server.server.signer", test_signer):
            # Act & Assert
            with pytest.raises(ValueError, match="expired"):
                await validate_session_cookie(old_cookie)

    @pytest.mark.asyncio
    async def test_validate_session_cookie_invalid_signature(self, auth_env_vars):
        """Test validating cookie with invalid signature."""
        from auth_server.server import validate_session_cookie

        # Arrange
        invalid_cookie = "invalid.signature.data"

        # Act & Assert
        with pytest.raises(ValueError, match="Invalid session cookie"):
            await validate_session_cookie(invalid_cookie)


# =============================================================================
# SIMPLIFIED COGNITO VALIDATOR TESTS
# =============================================================================


class TestSimplifiedCognitoValidator:
    """Tests for SimplifiedCognitoValidator class."""

    def test_validator_initialization(self):
        """Test validator initialization."""
        from auth_server.server import SimplifiedCognitoValidator

        # Act
        validator = SimplifiedCognitoValidator(region="us-west-2")

        # Assert
        assert validator.default_region == "us-west-2"
        assert validator._jwks_cache == {}

    @patch("auth_server.server.requests.get")
    def test_get_jwks_success(self, mock_get, mock_jwks_response):
        """Test successful JWKS retrieval."""
        from auth_server.server import SimplifiedCognitoValidator

        # Arrange
        mock_response = MagicMock()
        mock_response.json.return_value = mock_jwks_response
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        validator = SimplifiedCognitoValidator()
        user_pool_id = "us-east-1_TEST"
        region = "us-east-1"

        # Act
        jwks = validator._get_jwks(user_pool_id, region)

        # Assert
        assert "keys" in jwks
        assert len(jwks["keys"]) == 2
        mock_get.assert_called_once()

    @patch("auth_server.server.requests.get")
    def test_get_jwks_cached(self, mock_get, mock_jwks_response):
        """Test JWKS caching."""
        from auth_server.server import SimplifiedCognitoValidator

        # Arrange
        mock_response = MagicMock()
        mock_response.json.return_value = mock_jwks_response
        mock_get.return_value = mock_response

        validator = SimplifiedCognitoValidator()
        user_pool_id = "us-east-1_TEST"
        region = "us-east-1"

        # Act - call twice
        jwks1 = validator._get_jwks(user_pool_id, region)
        jwks2 = validator._get_jwks(user_pool_id, region)

        # Assert - should only call once due to caching
        assert mock_get.call_count == 1
        assert jwks1 == jwks2

    def test_validate_self_signed_token_valid(self, auth_env_vars, self_signed_token):
        """Test validating a valid self-signed token."""
        from auth_server.server import SimplifiedCognitoValidator

        # Arrange
        validator = SimplifiedCognitoValidator()

        # Patch SECRET_KEY at module level (loaded at import time before fixture sets env)
        with patch("auth_server.server.SECRET_KEY", auth_env_vars["SECRET_KEY"]):
            # Act
            result = validator.validate_self_signed_token(self_signed_token)

            # Assert
            assert result["valid"] is True
            assert result["method"] == "self_signed"
            assert result["username"] == "testuser"
            assert "read:servers" in result["scopes"]

    def test_validate_self_signed_token_expired(self, auth_env_vars):
        """Test validating an expired self-signed token."""
        from auth_server.server import SimplifiedCognitoValidator

        # Arrange
        validator = SimplifiedCognitoValidator()
        secret_key = auth_env_vars["SECRET_KEY"]
        now = int(time.time())

        # Create expired token
        payload = {
            "iss": "mcp-auth-server",
            "aud": "mcp-registry",
            "sub": "testuser",
            "exp": now - 3600,  # Expired 1 hour ago
            "iat": now - 7200,
            "token_use": "access",
        }
        expired_token = jwt.encode(payload, secret_key, algorithm="HS256")

        # Patch SECRET_KEY at module level (loaded at import time before fixture sets env)
        with patch("auth_server.server.SECRET_KEY", secret_key):
            # Act & Assert
            with pytest.raises(ValueError, match="expired"):
                validator.validate_self_signed_token(expired_token)


# =============================================================================
# FASTAPI ENDPOINT TESTS
# =============================================================================


class TestHealthEndpoint:
    """Tests for /health endpoint."""

    @patch("auth_server.server.get_auth_provider")
    def test_health_check(self, mock_get_provider):
        """Test health check endpoint."""
        # Arrange - import after mocking
        import auth_server.server as server_module

        client = TestClient(server_module.app)

        # Act
        response = client.get("/health")

        # Assert
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["service"] == "simplified-auth-server"


class TestValidateEndpoint:
    """Tests for /validate endpoint."""

    @patch("auth_server.server.get_auth_provider")
    def test_validate_with_valid_token(
        self,
        mock_get_provider,
        mock_cognito_provider,
        auth_env_vars,
        mock_scope_repository_with_data,
    ):
        """Test validation with valid JWT token."""
        # Arrange
        mock_get_provider.return_value = mock_cognito_provider

        import auth_server.server as server_module

        # Patch scope repository to return test data
        with patch(
            "auth_server.server.get_scope_repository", return_value=mock_scope_repository_with_data
        ):
            client = TestClient(server_module.app)

            # Act
            # URL format: /server-name/mcp-endpoint where endpoint is mcp, sse, or messages
            response = client.get(
                "/validate",
                headers={
                    "Authorization": "Bearer test-token",
                    "X-Original-URL": "https://example.com/test-server/mcp",
                },
            )

            # Assert
            assert response.status_code == 200
            data = response.json()
            assert data["valid"] is True
            assert data["username"] == "testuser"

    @patch("auth_server.server.get_auth_provider")
    def test_validate_missing_auth_header(self, mock_get_provider, auth_env_vars):
        """Test validation without Authorization header returns 401."""
        # Arrange
        import auth_server.server as server_module

        client = TestClient(server_module.app)

        # Act
        response = client.get("/validate")

        # Assert
        assert response.status_code == 401
        assert "Missing or invalid Authorization header" in response.json()["detail"]

    @patch("auth_server.server.get_auth_provider")
    def test_validate_with_session_cookie(
        self,
        mock_get_provider,
        auth_env_vars,
        valid_session_cookie,
        mock_scope_repository_with_data,
    ):
        """Test validation with valid session cookie."""
        from itsdangerous import URLSafeTimedSerializer

        import auth_server.server as server_module

        test_signer = URLSafeTimedSerializer(auth_env_vars["SECRET_KEY"])

        async def _fake_resolve(_session_id):
            return {
                "session_id": _session_id,
                "username": "testuser",
                "groups": ["users", "developers"],
                "provider": "cognito",
                "auth_method": "oauth2",
            }

        with (
            patch(
                "auth_server.server.get_scope_repository",
                return_value=mock_scope_repository_with_data,
            ),
            patch("auth_server.server.signer", test_signer),
            patch("session_store.resolve_session", _fake_resolve),
        ):
            client = TestClient(server_module.app)

            response = client.get(
                "/validate",
                headers={
                    "Cookie": f"mcp_gateway_session={valid_session_cookie}",
                    "X-Original-URL": "https://example.com/test-server/mcp",
                },
            )

            assert response.status_code == 200
            data = response.json()
            assert data["valid"] is True

    @patch("auth_server.server.get_auth_provider")
    def test_validate_token_failure_returns_401_not_500(
        self,
        mock_get_provider,
        auth_env_vars,
    ):
        """A ValueError from the provider's validate_token() must surface as
        401 with WWW-Authenticate, not 500. Pre-#989 this was a 500 which
        prevented MCP discovery clients from re-triggering OAuth on a stale
        token.

        Regression test for the auth_server fix shipped alongside #989.
        """
        from unittest.mock import MagicMock

        provider = MagicMock()
        provider.validate_token.side_effect = ValueError(
            "Token missing 'kid' in header"
        )
        mock_get_provider.return_value = provider

        import auth_server.server as server_module

        client = TestClient(server_module.app)

        response = client.get(
            "/validate",
            headers={
                "Authorization": "Bearer junk-token",
                "X-Original-URL": "https://example.com/test-server/mcp",
            },
        )

        assert response.status_code == 401
        assert "www-authenticate" in {k.lower() for k in response.headers.keys()}
        assert response.headers["www-authenticate"].startswith("Bearer")


class TestConfigEndpoint:
    """Tests for /config endpoint."""

    @patch("auth_server.server.get_auth_provider")
    def test_config_keycloak(self, mock_get_provider, mock_keycloak_provider):
        """Test config endpoint with Keycloak provider."""
        # Arrange
        mock_get_provider.return_value = mock_keycloak_provider

        import auth_server.server as server_module

        client = TestClient(server_module.app)

        # Act
        response = client.get("/config")

        # Assert
        assert response.status_code == 200
        data = response.json()
        assert data["auth_type"] == "keycloak"


def _internal_auth_headers(auth_env_vars: dict) -> dict:
    """Build an Authorization header carrying a valid internal JWT.

    ``/internal/tokens`` and ``/internal/reload-scopes`` both require a
    Bearer JWT signed with the shared SECRET_KEY (see
    ``registry.auth.internal.generate_internal_token``). Tests that POST
    to either endpoint must attach this header.
    """
    from registry.auth.internal import generate_internal_token

    token = generate_internal_token(
        subject="test-suite",
        purpose="unit-test",
    )
    return {"Authorization": f"Bearer {token}"}


class TestGenerateTokenEndpoint:
    """Tests for /internal/tokens endpoint."""

    @patch("auth_server.server.get_auth_provider")
    def test_generate_token_success(self, mock_get_provider, auth_env_vars):
        """Test successful token generation using Keycloak M2M."""
        # Arrange
        import auth_server.server as server_module

        # Mock Keycloak provider
        mock_provider = Mock()
        mock_provider.get_provider_info.return_value = {"provider_type": "keycloak"}
        # M2M token uses fixed scopes for IdP compatibility, not user-requested scopes
        mock_provider.get_m2m_token.return_value = {
            "access_token": "mock_keycloak_m2m_token",
            "refresh_token": None,
            "expires_in": 28800,
            "refresh_expires_in": 0,
            "scope": "openid email profile",
        }
        mock_get_provider.return_value = mock_provider

        client = TestClient(server_module.app)

        request_data = {
            "user_context": {"username": "testuser", "scopes": ["read:servers", "write:servers"]},
            "requested_scopes": ["read:servers"],
            "expires_in_hours": 8,
            "description": "Test token",
        }

        # Act
        response = client.post(
            "/internal/tokens",
            json=request_data,
            headers=_internal_auth_headers(auth_env_vars),
        )

        # Assert
        assert response.status_code == 200
        data = response.json()
        assert "access_token" in data
        assert data["access_token"] == "mock_keycloak_m2m_token"
        assert data["token_type"] == "Bearer"
        # Scope in response comes from Keycloak M2M client configuration
        assert data["scope"] == "openid email profile"
        # Verify Keycloak M2M was called with IdP-compatible scopes
        mock_provider.get_m2m_token.assert_called_once_with(scope="openid email profile")

    @patch("auth_server.server.get_auth_provider")
    def test_generate_token_missing_username(self, mock_get_provider, auth_env_vars):
        """Test token generation without username."""
        # Arrange
        import auth_server.server as server_module

        client = TestClient(server_module.app)

        request_data = {
            "user_context": {"scopes": ["read:servers"]},
            "requested_scopes": ["read:servers"],
            "expires_in_hours": 8,
        }

        # Act
        response = client.post(
            "/internal/tokens",
            json=request_data,
            headers=_internal_auth_headers(auth_env_vars),
        )

        # Assert
        assert response.status_code == 400
        assert "Username is required" in response.json()["detail"]

    @patch("auth_server.server.get_auth_provider")
    def test_generate_token_invalid_scopes(self, mock_get_provider, auth_env_vars):
        """Test token generation with invalid scopes."""
        # Arrange
        import auth_server.server as server_module

        client = TestClient(server_module.app)

        request_data = {
            "user_context": {"username": "testuser", "scopes": ["read:servers"]},
            "requested_scopes": ["admin:all"],  # User doesn't have this
            "expires_in_hours": 8,
        }

        # Act
        response = client.post(
            "/internal/tokens",
            json=request_data,
            headers=_internal_auth_headers(auth_env_vars),
        )

        # Assert
        assert response.status_code == 403
        assert "exceed user permissions" in response.json()["detail"]

    @patch("auth_server.server.get_auth_provider")
    def test_generate_token_rate_limit(self, mock_get_provider, auth_env_vars, monkeypatch):
        """Test token generation rate limiting."""
        # Arrange
        monkeypatch.setenv("MAX_TOKENS_PER_USER_PER_HOUR", "2")

        import auth_server.server as server_module

        server_module.MAX_TOKENS_PER_USER_PER_HOUR = 2
        server_module.user_token_generation_counts.clear()

        # Mock Keycloak provider for successful token generation
        mock_provider = Mock()
        mock_provider.get_provider_info.return_value = {"provider_type": "keycloak"}
        mock_provider.get_m2m_token.return_value = {
            "access_token": "mock_keycloak_m2m_token",
            "refresh_token": None,
            "expires_in": 28800,
            "refresh_expires_in": 0,
            "scope": "read:servers",
        }
        mock_get_provider.return_value = mock_provider

        client = TestClient(server_module.app)

        request_data = {
            "user_context": {"username": "testuser", "scopes": ["read:servers"]},
            "requested_scopes": ["read:servers"],
            "expires_in_hours": 8,
        }

        # Act - generate tokens up to limit
        for _ in range(2):
            response = client.post(
                "/internal/tokens",
                json=request_data,
                headers=_internal_auth_headers(auth_env_vars),
            )
            assert response.status_code == 200

        # Try one more - should fail
        response = client.post(
            "/internal/tokens",
            json=request_data,
            headers=_internal_auth_headers(auth_env_vars),
        )

        # Assert
        assert response.status_code == 429
        assert "Rate limit exceeded" in response.json()["detail"]


class TestInternalRouterGate:
    """Meta-test: every route under the ``/internal/`` prefix on the
    auth-server must require the signed-Bearer internal-JWT gate.

    The router-level dependency in ``auth_server.server.internal_router``
    is the mechanism that provides this guarantee. This test enumerates
    the routes at runtime and asserts each one returns 401 when called
    without an ``Authorization`` header — so a future developer who
    adds a new ``/internal/*`` handler by accident on ``@app.post`` or
    without the router dependency will get a failing build instead of
    an unauthenticated privileged endpoint.
    """

    def _internal_routes(self, server_module) -> list:
        """Return every (path, method) on app.routes whose path starts
        with ``/internal/``. Filters out non-HTTP things like
        ``Mount``/``WebSocketRoute`` which don't have a ``methods``
        attribute.
        """
        collected: list[tuple[str, str]] = []
        for route in server_module.app.routes:
            path = getattr(route, "path", None)
            methods = getattr(route, "methods", None)
            if not path or not methods:
                continue
            if not path.startswith("/internal/"):
                continue
            for method in methods:
                # HEAD/OPTIONS are auto-added and not interesting.
                if method in ("HEAD", "OPTIONS"):
                    continue
                collected.append((path, method))
        return collected

    def test_at_least_the_known_endpoints_are_present(self, auth_env_vars):
        """Guard against the meta-test trivially passing when the
        router is empty. If someone deletes both endpoints this catches it."""
        import auth_server.server as server_module

        paths = {path for path, _ in self._internal_routes(server_module)}
        assert "/internal/tokens" in paths
        assert "/internal/reload-scopes" in paths

    def test_every_internal_route_rejects_unauthenticated_request(self, auth_env_vars):
        """For every /internal/* route, a request without Authorization
        must return 401. A future /internal/foo endpoint registered on
        ``@app.post`` (bypassing the router) will fail here because the
        handler runs without the gate and returns something other than
        401.
        """
        import auth_server.server as server_module

        client = TestClient(server_module.app)
        routes = self._internal_routes(server_module)
        assert routes, "expected at least one /internal/* route"

        failures: list[str] = []
        for path, method in routes:
            response = client.request(method, path, json={})
            if response.status_code != 401:
                failures.append(
                    f"  {method} {path} returned {response.status_code} "
                    f"(expected 401); body={response.text[:200]}"
                )
        if failures:
            raise AssertionError(
                "One or more /internal/* routes accept requests without the "
                "internal-JWT gate. This is almost always because a handler "
                "was registered directly on ``app`` (e.g. "
                "``@app.post('/internal/foo')``) instead of on the "
                "``internal_router`` defined in auth_server/server.py.\n"
                "\n"
                "Fix: decorate the handler with ``@internal_router.post(...)`` "
                "(and drop the ``/internal`` prefix from the path, since the "
                "router already provides it). This inherits the router-level "
                "``Depends(validate_internal_auth)`` so the handler cannot "
                "ship without the signed-Bearer check.\n"
                "\n"
                "Offending routes:\n" + "\n".join(failures)
            )


class TestGenerateTokenEndpointInternalAuth:
    """Regression coverage for the internal-JWT gate on /internal/tokens.

    The endpoint mints JWTs — any caller that can reach it can issue a
    token for any user. We require the caller to prove knowledge of the
    shared ``SECRET_KEY`` via a short-lived internal JWT signed the same
    way the existing ``/internal/reload-scopes`` handler does it.
    """

    def test_rejects_missing_authorization(self, auth_env_vars):
        import auth_server.server as server_module

        client = TestClient(server_module.app)
        response = client.post(
            "/internal/tokens",
            json={
                "user_context": {"username": "alice", "scopes": []},
                "requested_scopes": [],
                "expires_in_hours": 8,
            },
        )
        assert response.status_code == 401
        assert "Missing authorization header" in response.json()["detail"]

    def test_rejects_non_bearer_scheme(self, auth_env_vars):
        import auth_server.server as server_module

        client = TestClient(server_module.app)
        response = client.post(
            "/internal/tokens",
            json={
                "user_context": {"username": "alice", "scopes": []},
                "requested_scopes": [],
                "expires_in_hours": 8,
            },
            headers={"Authorization": "Basic YWxpY2U6cGFzcw=="},
        )
        assert response.status_code == 401

    def test_rejects_bearer_signed_with_wrong_key(self, auth_env_vars):
        # Sign a JWT with a DIFFERENT secret — identical shape, wrong key.
        # Models the realistic threat: an attacker on the internal network
        # who does not possess SECRET_KEY.
        import time as _time

        import auth_server.server as server_module

        wrong_key_token = jwt.encode(
            {
                "iss": "mcp-auth-server",
                "aud": "mcp-registry",
                "sub": "attacker",
                "purpose": "forged",
                "token_use": "access",
                "iat": int(_time.time()),
                "exp": int(_time.time()) + 60,
            },
            "not-the-real-secret",
            algorithm="HS256",
        )
        client = TestClient(server_module.app)
        response = client.post(
            "/internal/tokens",
            json={
                "user_context": {"username": "alice", "scopes": []},
                "requested_scopes": [],
                "expires_in_hours": 8,
            },
            headers={"Authorization": f"Bearer {wrong_key_token}"},
        )
        assert response.status_code == 401

    def test_rejects_expired_bearer(self, auth_env_vars):
        # A token correctly signed with SECRET_KEY but whose ``exp`` is in
        # the past must be rejected. Models the realistic threat of an
        # attacker who captured a valid internal JWT from an earlier
        # request and tries to replay it after the short TTL.
        import time as _time

        import auth_server.server as server_module

        secret = auth_env_vars["SECRET_KEY"]
        now = int(_time.time())
        expired_token = jwt.encode(
            {
                "iss": "mcp-auth-server",
                "aud": "mcp-registry",
                "sub": "registry-service",
                "purpose": "generate-token",
                "token_use": "access",
                # ``leeway=30`` on validation, so push exp well past that.
                "iat": now - 600,
                "exp": now - 120,
            },
            secret,
            algorithm="HS256",
        )
        client = TestClient(server_module.app)
        response = client.post(
            "/internal/tokens",
            json={
                "user_context": {"username": "alice", "scopes": []},
                "requested_scopes": [],
                "expires_in_hours": 8,
            },
            headers={"Authorization": f"Bearer {expired_token}"},
        )
        assert response.status_code == 401


class TestReloadScopesEndpoint:
    """Tests for /internal/reload-scopes endpoint."""

    @patch("registry.common.scopes_loader.reload_scopes_config")
    @patch("auth_server.server.get_auth_provider")
    def test_reload_scopes_success_with_jwt(
        self, mock_get_provider, mock_reload_scopes, auth_env_vars
    ):
        """Test successful scopes reload using self-signed JWT."""
        # Arrange
        mock_reload_scopes.return_value = {"group_mappings": {}}

        import jwt

        import auth_server.server as server_module

        # Patch module-level SECRET_KEY to match the test env var
        # (it may already be set to a different value from earlier test imports)
        secret_key = auth_env_vars["SECRET_KEY"]
        original_secret_key = server_module.SECRET_KEY
        server_module.SECRET_KEY = secret_key

        try:
            client = TestClient(server_module.app)

            now = int(time.time())
            token = jwt.encode(
                {
                    "iss": "mcp-auth-server",
                    "aud": "mcp-registry",
                    "sub": "registry-service",
                    "purpose": "reload-scopes",
                    "token_use": "access",
                    "iat": now,
                    "exp": now + 30,
                },
                secret_key,
                algorithm="HS256",
            )

            # Act
            response = client.post(
                "/internal/reload-scopes", headers={"Authorization": f"Bearer {token}"}
            )

            # Assert
            assert response.status_code == 200
            data = response.json()
            assert "successfully" in data["message"]
        finally:
            server_module.SECRET_KEY = original_secret_key

    @patch("auth_server.server.get_auth_provider")
    def test_reload_scopes_no_auth(self, mock_get_provider):
        """Test scopes reload without authentication."""
        # Arrange
        import auth_server.server as server_module

        client = TestClient(server_module.app)

        # Act
        response = client.post("/internal/reload-scopes")

        # Assert
        assert response.status_code == 401

    @patch("auth_server.server.get_auth_provider")
    def test_reload_scopes_invalid_jwt(self, mock_get_provider, auth_env_vars):
        """Test scopes reload with an invalid JWT token."""
        # Arrange
        import auth_server.server as server_module

        client = TestClient(server_module.app)

        # Act
        response = client.post(
            "/internal/reload-scopes", headers={"Authorization": "Bearer invalid-token"}
        )

        # Assert
        assert response.status_code == 401

    @patch("registry.common.scopes_loader.reload_scopes_config")
    @patch("auth_server.server.get_auth_provider")
    def test_reload_scopes_basic_auth_rejected(self, mock_get_provider, auth_env_vars):
        """Test that Basic Auth is rejected (no longer supported)."""
        # Arrange
        import base64

        import auth_server.server as server_module

        client = TestClient(server_module.app)

        credentials = base64.b64encode(b"testadmin:testadminpass").decode()

        # Act
        response = client.post(
            "/internal/reload-scopes", headers={"Authorization": f"Basic {credentials}"}
        )

        # Assert - Basic Auth is no longer supported
        assert response.status_code == 401
        assert "Unsupported authentication scheme" in response.json()["detail"]


# =============================================================================
# NETWORK-TRUSTED MODE TESTS
# =============================================================================


class TestNetworkTrustedMode:
    """Tests for network-trusted auth bypass mode (issue #357)."""

    def test_network_trusted_bypasses_registry_api(self):
        """When enabled, registry API requests bypass JWT validation."""
        # Arrange
        import auth_server.server as server_module

        token_map = _make_legacy_token_map("test-api-key")
        with (
            patch.object(server_module, "REGISTRY_STATIC_TOKEN_AUTH_ENABLED", True),
            patch.object(server_module, "REGISTRY_API_TOKEN", "test-api-key"),
            patch.object(server_module, "_STATIC_TOKEN_MAP", token_map),
        ):
            client = TestClient(server_module.app)

            # Act
            response = client.get(
                "/validate",
                headers={
                    "Authorization": "Bearer test-api-key",
                    "X-Original-URL": "https://example.com/api/servers",
                },
            )

            # Assert
            assert response.status_code == 200
            data = response.json()
            assert data["valid"] is True
            assert data["username"] == "network-user"
            assert data["client_id"] == "network-trusted"
            assert data["method"] == "network-trusted"
            assert "mcp-servers-unrestricted/read" in data["scopes"]
            assert "mcp-servers-unrestricted/execute" in data["scopes"]
            assert response.headers["X-Auth-Method"] == "network-trusted"
            assert response.headers["X-Username"] == "network-user"

    def test_network_trusted_missing_auth_falls_through_to_jwt(self):
        """Missing Authorization header falls through to JWT/session validation.

        Before issue #871 the static-token block terminated with a 401. After
        the fix the block falls through so Okta JWT / self-signed JWT callers
        still work. An absent Authorization header ultimately reaches the JWT
        block which returns 401 with a different detail message.
        """
        # Arrange
        import auth_server.server as server_module

        token_map = _make_legacy_token_map("test-api-key")
        with (
            patch.object(server_module, "REGISTRY_STATIC_TOKEN_AUTH_ENABLED", True),
            patch.object(server_module, "REGISTRY_API_TOKEN", "test-api-key"),
            patch.object(server_module, "_STATIC_TOKEN_MAP", token_map),
        ):
            client = TestClient(server_module.app)

            # Act
            response = client.get(
                "/validate",
                headers={
                    "X-Original-URL": "https://example.com/api/servers",
                },
            )

            # Assert: 401 comes from the downstream JWT block, not the static
            # token block. The detail text changed to the JWT-block message.
            assert response.status_code == 401
            assert "Missing or invalid Authorization header" in response.json()["detail"]

    @patch("auth_server.server.get_auth_provider")
    def test_network_trusted_does_not_bypass_mcp_gateway(
        self,
        mock_get_provider,
        auth_env_vars,
    ):
        """MCP server access still requires full validation even when bypass is enabled."""
        # Arrange
        import auth_server.server as server_module

        mock_provider = MagicMock()
        mock_provider.validate_token = AsyncMock(side_effect=ValueError("Invalid token"))
        mock_get_provider.return_value = mock_provider

        token_map = _make_legacy_token_map("test-api-key")
        with (
            patch.object(server_module, "REGISTRY_STATIC_TOKEN_AUTH_ENABLED", True),
            patch.object(server_module, "REGISTRY_API_TOKEN", "test-api-key"),
            patch.object(server_module, "_STATIC_TOKEN_MAP", token_map),
        ):
            client = TestClient(server_module.app)

            # Act - request to an MCP server path, not /api/ or /v0.1/
            response = client.get(
                "/validate",
                headers={
                    "Authorization": "Bearer test-api-key",
                    "X-Original-URL": "https://example.com/mcpserver/messages",
                },
            )

            # Assert - should NOT be bypassed, falls through to normal validation
            assert response.status_code != 200 or response.json().get("method") != "network-trusted"

    def test_network_trusted_disabled_by_default(self, auth_env_vars):
        """Default behavior requires full authentication, no bypass."""
        # Arrange
        import auth_server.server as server_module

        with patch.object(server_module, "REGISTRY_STATIC_TOKEN_AUTH_ENABLED", False):
            client = TestClient(server_module.app)

            # Act
            response = client.get(
                "/validate",
                headers={
                    "Authorization": "Bearer network-trusted",
                    "X-Original-URL": "https://example.com/api/servers",
                },
            )

            # Assert - should NOT return network-trusted response
            if response.status_code == 200:
                assert response.json().get("method") != "network-trusted"

    def test_network_trusted_bypasses_v01_api(self):
        """When enabled, /v0.1/* requests also bypass JWT validation."""
        # Arrange
        import auth_server.server as server_module

        token_map = _make_legacy_token_map("test-api-key")
        with (
            patch.object(server_module, "REGISTRY_STATIC_TOKEN_AUTH_ENABLED", True),
            patch.object(server_module, "REGISTRY_API_TOKEN", "test-api-key"),
            patch.object(server_module, "_STATIC_TOKEN_MAP", token_map),
        ):
            client = TestClient(server_module.app)

            # Act
            response = client.get(
                "/validate",
                headers={
                    "Authorization": "Bearer test-api-key",
                    "X-Original-URL": "https://example.com/v0.1/servers",
                },
            )

            # Assert
            assert response.status_code == 200
            data = response.json()
            assert data["valid"] is True
            assert data["username"] == "network-user"
            assert data["method"] == "network-trusted"

    def test_network_trusted_valid_api_token(self):
        """When REGISTRY_API_TOKEN is set, matching Bearer token is accepted."""
        # Arrange
        import auth_server.server as server_module

        token_map = _make_legacy_token_map("my-secret-key")
        with (
            patch.object(server_module, "REGISTRY_STATIC_TOKEN_AUTH_ENABLED", True),
            patch.object(server_module, "REGISTRY_API_TOKEN", "my-secret-key"),
            patch.object(server_module, "_STATIC_TOKEN_MAP", token_map),
        ):
            client = TestClient(server_module.app)

            # Act
            response = client.get(
                "/validate",
                headers={
                    "Authorization": "Bearer my-secret-key",
                    "X-Original-URL": "https://example.com/api/servers",
                },
            )

            # Assert
            assert response.status_code == 200
            data = response.json()
            assert data["valid"] is True
            assert data["method"] == "network-trusted"

    @patch("auth_server.server.get_auth_provider")
    def test_network_trusted_invalid_api_token_falls_through_to_jwt(
        self,
        mock_get_provider,
        auth_env_vars,
    ):
        """A mismatched Bearer now falls through to JWT validation (issue #871).

        Pre-#871 the static-token block returned 403 "Invalid API token". After
        #871 a mismatched bearer is handed to the JWT block. When the JWT
        provider rejects it, the final response does NOT contain the old
        static-token-block detail text.
        """
        # Arrange - provider returns an invalid-token result
        mock_provider = MagicMock()
        mock_provider.validate_token = MagicMock(side_effect=ValueError("Invalid token"))
        mock_get_provider.return_value = mock_provider

        import auth_server.server as server_module

        token_map = _make_legacy_token_map("my-secret-key")
        with (
            patch.object(server_module, "REGISTRY_STATIC_TOKEN_AUTH_ENABLED", True),
            patch.object(server_module, "REGISTRY_API_TOKEN", "my-secret-key"),
            patch.object(server_module, "_STATIC_TOKEN_MAP", token_map),
        ):
            client = TestClient(server_module.app)

            # Act
            response = client.get(
                "/validate",
                headers={
                    "Authorization": "Bearer wrong-key",
                    "X-Original-URL": "https://example.com/api/servers",
                },
            )

            # Assert: response is no longer the static-token block's 403 with
            # "Invalid API token". The terminal status depends on the JWT
            # provider's failure handling (pre-existing 500 path wraps
            # ValueError), but either way it must NOT be the old 403 body.
            assert response.status_code != 403
            assert "Invalid API token" not in response.json().get("detail", "")

    def test_network_trusted_disabled_when_no_token_configured(self):
        """When REGISTRY_API_TOKEN is empty, static token auth is disabled (falls back to JWT)."""
        # Arrange
        import auth_server.server as server_module

        # Simulate: enabled flag was set to False at startup because token was empty
        with (
            patch.object(server_module, "REGISTRY_STATIC_TOKEN_AUTH_ENABLED", False),
            patch.object(server_module, "REGISTRY_API_TOKEN", ""),
        ):
            client = TestClient(server_module.app)

            # Act
            response = client.get(
                "/validate",
                headers={
                    "Authorization": "Bearer anything-goes",
                    "X-Original-URL": "https://example.com/api/servers",
                },
            )

            # Assert - should NOT return network-trusted (falls through to JWT validation)
            if response.status_code == 200:
                assert response.json().get("method") != "network-trusted"

    def test_network_trusted_skips_bypass_when_session_cookie_present(self):
        """When session cookie is present, bypass is skipped for normal cookie auth flow."""
        # Arrange
        import auth_server.server as server_module

        token_map = _make_legacy_token_map("test-api-key")
        with (
            patch.object(server_module, "REGISTRY_STATIC_TOKEN_AUTH_ENABLED", True),
            patch.object(server_module, "REGISTRY_API_TOKEN", "test-api-key"),
            patch.object(server_module, "_STATIC_TOKEN_MAP", token_map),
        ):
            client = TestClient(server_module.app)

            # Act - send with session cookie but no Bearer token
            response = client.get(
                "/validate",
                headers={
                    "X-Original-URL": "https://example.com/api/servers",
                    "Cookie": "mcp_gateway_session=some-session-value",
                },
            )

            # Assert - should NOT get 401 from bypass (bypass was skipped)
            # It will fail session validation, but not with the bypass 401 message
            if response.status_code == 401:
                assert "Authorization header required" not in response.json().get("detail", "")

    def test_network_trusted_non_bearer_scheme_falls_through_to_jwt(self):
        """Non-Bearer scheme now falls through to JWT validation (issue #871).

        Before #871 the static-token block returned 401 with detail mentioning
        "Bearer scheme". After #871 the block falls through; the JWT block
        returns 401 with its own detail message.
        """
        # Arrange
        import auth_server.server as server_module

        token_map = _make_legacy_token_map("test-api-key")
        with (
            patch.object(server_module, "REGISTRY_STATIC_TOKEN_AUTH_ENABLED", True),
            patch.object(server_module, "REGISTRY_API_TOKEN", "test-api-key"),
            patch.object(server_module, "_STATIC_TOKEN_MAP", token_map),
        ):
            client = TestClient(server_module.app)

            # Act - send Basic auth instead of Bearer
            response = client.get(
                "/validate",
                headers={
                    "Authorization": "Basic dXNlcjpwYXNz",
                    "X-Original-URL": "https://example.com/api/servers",
                },
            )

            # Assert: 401 from JWT block, not the old "Bearer scheme" detail
            assert response.status_code == 401
            assert "Bearer scheme" not in response.json()["detail"]

    @patch("auth_server.server.get_auth_provider")
    def test_network_trusted_empty_bearer_falls_through_to_jwt(
        self,
        mock_get_provider,
        auth_env_vars,
    ):
        """Empty Bearer token now falls through to JWT validation (issue #871)."""
        # Arrange - provider rejects empty token
        mock_provider = MagicMock()
        mock_provider.validate_token = MagicMock(side_effect=ValueError("Empty token"))
        mock_get_provider.return_value = mock_provider

        import auth_server.server as server_module

        token_map = _make_legacy_token_map("test-api-key")
        with (
            patch.object(server_module, "REGISTRY_STATIC_TOKEN_AUTH_ENABLED", True),
            patch.object(server_module, "REGISTRY_API_TOKEN", "test-api-key"),
            patch.object(server_module, "_STATIC_TOKEN_MAP", token_map),
        ):
            client = TestClient(server_module.app)

            # Act - send Bearer with empty token
            response = client.get(
                "/validate",
                headers={
                    "Authorization": "Bearer ",
                    "X-Original-URL": "https://example.com/api/servers",
                },
            )

            # Assert: fall-through → JWT block rejects → no longer the old 403
            # "Invalid API token" detail.
            assert response.status_code != 403
            assert "Invalid API token" not in response.json().get("detail", "")


# =============================================================================
# HELPER UNIT TESTS (issue #871)
# =============================================================================


def _make_legacy_token_map(token: str) -> dict[str, dict]:
    """Build a _STATIC_TOKEN_MAP with just the legacy entry for test helpers."""
    return {
        "legacy": {
            "key_bytes": token.encode("utf-8"),
            "groups": ["mcp-registry-admin"],
            "scopes": [
                "mcp-registry-admin",
                "mcp-servers-unrestricted/read",
                "mcp-servers-unrestricted/execute",
            ],
            "username_override": "network-user",
            "client_id_override": "network-trusted",
        },
    }


class TestCheckRegistryStaticToken:
    """Unit tests for the _check_registry_static_token helper.

    Updated for issue #779 (multi-key map iteration).
    """

    def test_legacy_match_returns_network_trusted_identity(self):
        """Matching bearer for legacy key returns the back-compat identity dict."""
        import auth_server.server as server_module

        token_map = _make_legacy_token_map("expected-token")
        with patch.object(server_module, "_STATIC_TOKEN_MAP", token_map):
            identity = server_module._check_registry_static_token("expected-token")

        assert identity is not None
        assert identity["username"] == "network-user"
        assert identity["client_id"] == "network-trusted"
        assert identity["groups"] == ["mcp-registry-admin"]
        assert "mcp-servers-unrestricted/read" in identity["scopes"]
        assert "mcp-servers-unrestricted/execute" in identity["scopes"]

    def test_mismatch_returns_none(self):
        """Non-matching bearer returns None (not an exception, not a falsy dict)."""
        import auth_server.server as server_module

        token_map = _make_legacy_token_map("expected-token")
        with patch.object(server_module, "_STATIC_TOKEN_MAP", token_map):
            assert server_module._check_registry_static_token("something-else") is None

    def test_empty_bearer_returns_none(self):
        """Empty-string bearer must not match any configured token."""
        import auth_server.server as server_module

        token_map = _make_legacy_token_map("expected-token")
        with patch.object(server_module, "_STATIC_TOKEN_MAP", token_map):
            assert server_module._check_registry_static_token("") is None

    def test_empty_map_returns_none(self):
        """When no keys are configured, any bearer returns None."""
        import auth_server.server as server_module

        with patch.object(server_module, "_STATIC_TOKEN_MAP", {}):
            assert server_module._check_registry_static_token("any-token") is None

    def test_uses_timing_safe_comparison(self):
        """Guard against regression: must use hmac.compare_digest, not ==."""
        import inspect

        import auth_server.server as server_module

        source = inspect.getsource(server_module._check_registry_static_token)
        assert "hmac.compare_digest" in source

    def test_multi_key_match_returns_correct_identity(self):
        """With multiple keys, the matched entry's identity is returned."""
        import auth_server.server as server_module

        token_map = {
            "monitoring": {
                "key_bytes": b"aaaa" * 8,
                "groups": ["mcp-readonly"],
                "scopes": ["mcp-readonly/read"],
            },
            "deploy": {
                "key_bytes": b"bbbb" * 8,
                "groups": ["mcp-registry-admin"],
                "scopes": ["mcp-servers-unrestricted/read"],
            },
        }
        with patch.object(server_module, "_STATIC_TOKEN_MAP", token_map):
            identity = server_module._check_registry_static_token("bbbb" * 8)

        assert identity is not None
        assert identity["username"] == "deploy"
        assert identity["client_id"] == "deploy"
        assert identity["groups"] == ["mcp-registry-admin"]

    def test_multi_key_no_match_returns_none(self):
        """With multiple keys, a non-matching bearer returns None."""
        import auth_server.server as server_module

        token_map = {
            "monitoring": {
                "key_bytes": b"aaaa" * 8,
                "groups": ["mcp-readonly"],
                "scopes": ["mcp-readonly/read"],
            },
        }
        with patch.object(server_module, "_STATIC_TOKEN_MAP", token_map):
            assert server_module._check_registry_static_token("wrong-token") is None

    def test_legacy_username_override_preserved(self):
        """Legacy entry uses username_override / client_id_override for back-compat."""
        import auth_server.server as server_module

        token_map = _make_legacy_token_map("legacy-token")
        with patch.object(server_module, "_STATIC_TOKEN_MAP", token_map):
            identity = server_module._check_registry_static_token("legacy-token")

        assert identity["username"] == "network-user"
        assert identity["client_id"] == "network-trusted"

    def test_non_legacy_key_uses_name_as_username(self):
        """Non-legacy entries use the key name as username and client_id."""
        import auth_server.server as server_module

        token_map = {
            "ci-pipeline": {
                "key_bytes": b"x" * 32,
                "groups": ["mcp-registry-admin"],
                "scopes": ["admin/all"],
            },
        }
        with patch.object(server_module, "_STATIC_TOKEN_MAP", token_map):
            identity = server_module._check_registry_static_token("x" * 32)

        assert identity["username"] == "ci-pipeline"
        assert identity["client_id"] == "ci-pipeline"


# =============================================================================
# JWT / STATIC TOKEN COEXISTENCE TESTS (issue #871)
# =============================================================================


class TestStaticTokenFallthrough:
    """Tests verifying that static-token mode accepts Okta/self-signed JWTs
    as ADDITIONAL credentials, not as replacements. See issue #871.
    """

    @patch("auth_server.server.get_auth_provider")
    def test_valid_jwt_accepted_when_static_token_enabled(
        self,
        mock_get_provider,
        mock_cognito_provider,
        auth_env_vars,
        mock_scope_repository_with_data,
    ):
        """A valid IdP JWT must be accepted on /api/* even when static-token
        mode is on. Pre-#871 the static-token block returned 403 here.
        """
        # Arrange
        mock_get_provider.return_value = mock_cognito_provider

        import auth_server.server as server_module

        token_map = _make_legacy_token_map("static-key")
        with (
            patch.object(server_module, "REGISTRY_STATIC_TOKEN_AUTH_ENABLED", True),
            patch.object(server_module, "REGISTRY_API_TOKEN", "static-key"),
            patch.object(server_module, "_STATIC_TOKEN_MAP", token_map),
            patch(
                "auth_server.server.get_scope_repository",
                return_value=mock_scope_repository_with_data,
            ),
        ):
            client = TestClient(server_module.app)

            # Act: send a non-matching Bearer that the JWT provider accepts
            response = client.get(
                "/validate",
                headers={
                    "Authorization": "Bearer some-valid-idp-jwt",
                    "X-Original-URL": "https://example.com/api/servers",
                },
            )

            # Assert: JWT path wins; response is 200 but NOT network-trusted.
            assert response.status_code == 200
            data = response.json()
            assert data["valid"] is True
            assert data["method"] != "network-trusted"
            # The cognito mock returns method="cognito".
            assert data["username"] == "testuser"

    def test_static_token_match_still_returns_network_trusted(self):
        """The happy path for the static token is unchanged by #871."""
        import auth_server.server as server_module

        token_map = _make_legacy_token_map("static-key")
        with (
            patch.object(server_module, "REGISTRY_STATIC_TOKEN_AUTH_ENABLED", True),
            patch.object(server_module, "REGISTRY_API_TOKEN", "static-key"),
            patch.object(server_module, "_STATIC_TOKEN_MAP", token_map),
        ):
            client = TestClient(server_module.app)

            response = client.get(
                "/validate",
                headers={
                    "Authorization": "Bearer static-key",
                    "X-Original-URL": "https://example.com/api/servers",
                },
            )

            assert response.status_code == 200
            data = response.json()
            assert data["method"] == "network-trusted"
            assert data["client_id"] == "network-trusted"
            assert response.headers["X-Auth-Method"] == "network-trusted"

    @patch("auth_server.server.get_auth_provider")
    def test_mismatched_bearer_and_invalid_jwt_returns_401(
        self,
        mock_get_provider,
        auth_env_vars,
    ):
        """Bearer that matches neither static token nor any valid JWT returns
        401 from the JWT block (previously 403 from static-token block).
        """
        # Arrange - provider rejects the token
        mock_provider = MagicMock()
        mock_provider.validate_token = MagicMock(side_effect=ValueError("Invalid token"))
        mock_get_provider.return_value = mock_provider

        import auth_server.server as server_module

        token_map = _make_legacy_token_map("static-key")
        with (
            patch.object(server_module, "REGISTRY_STATIC_TOKEN_AUTH_ENABLED", True),
            patch.object(server_module, "REGISTRY_API_TOKEN", "static-key"),
            patch.object(server_module, "_STATIC_TOKEN_MAP", token_map),
        ):
            client = TestClient(server_module.app)

            response = client.get(
                "/validate",
                headers={
                    "Authorization": "Bearer neither-static-nor-jwt",
                    "X-Original-URL": "https://example.com/api/servers",
                },
            )

            # Assert: the terminal rejection is no longer the static-token
            # block's 403 "Invalid API token". Downstream JWT failure
            # semantics (401 on empty / 500 on provider ValueError etc.) are
            # out of scope for #871; we only assert the removal of the old
            # static-token rejection.
            assert "Invalid API token" not in response.json().get("detail", "")


# =============================================================================
# OAUTH2 CALLBACK SESSION STORAGE INTEGRATION TESTS
# =============================================================================


class TestOAuth2CallbackTokenStorage:
    """The browser cookie always carries only an opaque session_id — never
    user data, groups, or tokens. id_token is always persisted server-side
    (encrypted) so SSO logout via id_token_hint keeps working.
    """

    # Cookie ceiling: we sign an opaque 32-byte session_id; the resulting
    # value comfortably fits well under 512 bytes. Lock that in — the whole
    # point of the server-side store is to keep the cookie small.
    COOKIE_SIZE_CEILING_BYTES = 512

    def _call_oauth2_callback(self) -> tuple[dict, str]:
        """Drive the real oauth2_callback endpoint with create_session mocked.

        Returns (kwargs, session_cookie_value) so tests can assert what would
        have landed in the persistent record AND can verify the cookie value
        bounds.
        """
        from auth_server.server import app, signer

        mock_token_data = {
            "access_token": "mock-access-token-value",
            "refresh_token": "mock-refresh-token-value",
            "expires_in": 3600,
            "id_token": "mock-id-token",
        }
        mock_user_info = {
            "sub": "testuser",
            "email": "test@example.com",
            "name": "Test User",
        }
        temp_session_data = {
            "state": "test-state",
            "provider": "github",
            "callback_uri": "http://localhost:8888/oauth2/callback/github",
        }
        temp_cookie = signer.dumps(temp_session_data)

        captured: dict = {}

        async def _fake_create_session(**kwargs):
            captured.update(kwargs)
            return "fake-session-id"

        client = TestClient(app, raise_server_exceptions=False)

        with (
            patch(
                "auth_server.server.exchange_code_for_token",
                new_callable=AsyncMock,
                return_value=mock_token_data,
            ),
            patch(
                "auth_server.server.get_user_info",
                new_callable=AsyncMock,
                return_value=mock_user_info,
            ),
            patch(
                "auth_server.server.map_user_info",
                return_value={
                    "username": "testuser",
                    "email": "test@example.com",
                    "name": "Test User",
                    "groups": [],
                },
            ),
            patch("session_store.create_session", _fake_create_session),
        ):
            response = client.get(
                "/oauth2/callback/github",
                params={"code": "test-code", "state": "test-state"},
                cookies={"oauth2_temp_session": temp_cookie},
                follow_redirects=False,
            )

        assert response.status_code == 302
        # Cookie payload is the signed opaque session_id, never user data.
        session_cookie = response.cookies.get("mcp_gateway_session")
        assert session_cookie is not None, "Session cookie not set in response"
        assert signer.loads(session_cookie) == "fake-session-id"

        return captured, session_cookie

    def test_id_token_persisted_in_session_store(self):
        """id_token is always passed to the session store (regression guard
        for SSO logout via id_token_hint). access_token / refresh_token are
        never stored.
        """
        kwargs, _cookie = self._call_oauth2_callback()
        assert kwargs["username"] == "testuser"
        assert kwargs["auth_method"] == "oauth2"
        assert kwargs["id_token"] == "mock-id-token"
        assert "access_token" not in kwargs
        assert "refresh_token" not in kwargs

    def test_session_cookie_stays_well_under_size_ceiling(self):
        """The whole point of the server-side store is a small cookie.

        Lock in the win: the cookie should be a bounded-size signed opaque
        session_id, not a serialized payload of user/groups/id_token. If a
        future change reintroduces inline user data, this test catches it.
        """
        _kwargs, cookie = self._call_oauth2_callback()
        assert len(cookie) < self.COOKIE_SIZE_CEILING_BYTES, (
            f"Session cookie is {len(cookie)} bytes; expected < "
            f"{self.COOKIE_SIZE_CEILING_BYTES}. The server-side session store "
            "should keep this small."
        )


# =============================================================================
# MULTI-KEY STATIC TOKEN PARSER TESTS (issue #779)
# =============================================================================


class TestParseRegistryApiKeys:
    """Unit tests for _parse_registry_api_keys config parser."""

    def test_empty_string_returns_empty_list(self):
        """Empty raw string produces no entries."""
        import auth_server.server as server_module

        result = server_module._parse_registry_api_keys("")
        assert result == []

    def test_valid_single_entry(self):
        """A single valid entry parses correctly."""
        import json

        import auth_server.server as server_module

        raw = json.dumps(
            {
                "deploy-pipeline": {
                    "key": "a" * 32,
                    "groups": ["mcp-registry-admin"],
                }
            }
        )
        result = server_module._parse_registry_api_keys(raw)
        assert len(result) == 1
        assert result[0].name == "deploy-pipeline"
        assert result[0].key == "a" * 32
        assert result[0].groups == ["mcp-registry-admin"]

    def test_valid_multiple_entries(self):
        """Multiple valid entries parse correctly."""
        import json

        import auth_server.server as server_module

        raw = json.dumps(
            {
                "monitoring": {"key": "m" * 32, "groups": ["mcp-readonly"]},
                "deploy": {"key": "d" * 32, "groups": ["mcp-registry-admin"]},
            }
        )
        result = server_module._parse_registry_api_keys(raw)
        assert len(result) == 2
        names = {e.name for e in result}
        assert names == {"monitoring", "deploy"}

    def test_malformed_json_raises(self):
        """Non-JSON input raises ValueError."""
        import auth_server.server as server_module

        with pytest.raises(ValueError, match="not valid JSON"):
            server_module._parse_registry_api_keys("{bad json")

    def test_non_object_json_raises(self):
        """A JSON array (not object) raises ValueError."""
        import auth_server.server as server_module

        with pytest.raises(ValueError, match="must be a JSON object"):
            server_module._parse_registry_api_keys('[{"key":"abc"}]')

    def test_reserved_name_legacy_raises(self):
        """The name 'legacy' is reserved and must be rejected."""
        import json

        import auth_server.server as server_module

        raw = json.dumps(
            {
                "legacy": {"key": "x" * 32, "groups": ["admin"]},
            }
        )
        with pytest.raises(ValueError, match="reserved"):
            server_module._parse_registry_api_keys(raw)

    def test_reserved_name_network_user_raises(self):
        """The name 'network-user' is reserved."""
        import json

        import auth_server.server as server_module

        raw = json.dumps(
            {
                "network-user": {"key": "x" * 32, "groups": ["admin"]},
            }
        )
        with pytest.raises(ValueError, match="reserved"):
            server_module._parse_registry_api_keys(raw)

    def test_reserved_name_network_trusted_raises(self):
        """The name 'network-trusted' is reserved."""
        import json

        import auth_server.server as server_module

        raw = json.dumps(
            {
                "network-trusted": {"key": "x" * 32, "groups": ["admin"]},
            }
        )
        with pytest.raises(ValueError, match="reserved"):
            server_module._parse_registry_api_keys(raw)

    def test_key_too_short_raises(self):
        """Key shorter than 32 chars raises."""
        import json

        import auth_server.server as server_module

        raw = json.dumps(
            {
                "short-key": {"key": "abc", "groups": ["admin"]},
            }
        )
        with pytest.raises(ValueError, match="Invalid entry"):
            server_module._parse_registry_api_keys(raw)

    def test_empty_groups_raises(self):
        """Empty groups list raises."""
        import json

        import auth_server.server as server_module

        raw = json.dumps(
            {
                "no-groups": {"key": "x" * 32, "groups": []},
            }
        )
        with pytest.raises(ValueError, match="Invalid entry"):
            server_module._parse_registry_api_keys(raw)

    def test_duplicate_key_value_raises(self):
        """Two entries with the same key value raises."""
        import json

        import auth_server.server as server_module

        same_key = "k" * 32
        raw = json.dumps(
            {
                "entry-a": {"key": same_key, "groups": ["g1"]},
                "entry-b": {"key": same_key, "groups": ["g2"]},
            }
        )
        with pytest.raises(ValueError, match="Duplicate key value"):
            server_module._parse_registry_api_keys(raw)

    def test_invalid_name_format_raises(self):
        """Name with uppercase or special chars raises."""
        import json

        import auth_server.server as server_module

        raw = json.dumps(
            {
                "Invalid-Name!": {"key": "x" * 32, "groups": ["admin"]},
            }
        )
        with pytest.raises(ValueError, match="Invalid"):
            server_module._parse_registry_api_keys(raw)

    def test_entry_not_object_raises(self):
        """Entry value that is not a dict raises."""
        import json

        import auth_server.server as server_module

        raw = json.dumps(
            {
                "bad-entry": "just-a-string",
            }
        )
        with pytest.raises(ValueError, match="must be an object"):
            server_module._parse_registry_api_keys(raw)

    def test_empty_object_returns_empty_list(self):
        """An empty JSON object '{}' returns an empty list."""
        import auth_server.server as server_module

        result = server_module._parse_registry_api_keys("{}")
        assert result == []


# =============================================================================
# MULTI-KEY BUILD TOKEN MAP TESTS (issue #779)
# =============================================================================


class TestBuildStaticTokenMap:
    """Unit tests for _build_static_token_map startup builder."""

    @pytest.mark.asyncio
    async def test_disabled_flag_does_nothing(self):
        """When REGISTRY_STATIC_TOKEN_AUTH_ENABLED is False, map stays empty."""
        import auth_server.server as server_module

        with (
            patch.object(server_module, "REGISTRY_STATIC_TOKEN_AUTH_ENABLED", False),
            patch.object(server_module, "_STATIC_TOKEN_MAP", {}),
        ):
            await server_module._build_static_token_map()
            assert server_module._STATIC_TOKEN_MAP == {}

    @pytest.mark.asyncio
    async def test_legacy_only_builds_single_entry(self):
        """With only REGISTRY_API_TOKEN set (no REGISTRY_API_KEYS), map has one legacy entry."""
        import auth_server.server as server_module

        with (
            patch.object(server_module, "REGISTRY_STATIC_TOKEN_AUTH_ENABLED", True),
            patch.object(server_module, "REGISTRY_API_TOKEN", "t" * 32),
            patch.object(server_module, "_REGISTRY_API_KEYS_RAW", ""),
            patch.object(server_module, "_STATIC_TOKEN_MAP", {}),
        ):
            await server_module._build_static_token_map()
            assert "legacy" in server_module._STATIC_TOKEN_MAP
            assert len(server_module._STATIC_TOKEN_MAP) == 1
            legacy = server_module._STATIC_TOKEN_MAP["legacy"]
            assert legacy["username_override"] == "network-user"
            assert legacy["client_id_override"] == "network-trusted"

    @pytest.mark.asyncio
    async def test_bad_json_disables_feature(self):
        """Malformed REGISTRY_API_KEYS disables static-token auth (fail-closed)."""
        import auth_server.server as server_module

        with (
            patch.object(server_module, "REGISTRY_STATIC_TOKEN_AUTH_ENABLED", True),
            patch.object(server_module, "REGISTRY_API_TOKEN", ""),
            patch.object(server_module, "_REGISTRY_API_KEYS_RAW", "{bad json"),
            patch.object(server_module, "_STATIC_TOKEN_MAP", {}),
        ):
            await server_module._build_static_token_map()
            assert server_module.REGISTRY_STATIC_TOKEN_AUTH_ENABLED is False

    @pytest.mark.asyncio
    async def test_valid_keys_plus_legacy_merged(self):
        """Both REGISTRY_API_KEYS and REGISTRY_API_TOKEN produce merged map."""
        import json

        import auth_server.server as server_module

        raw = json.dumps(
            {
                "monitoring": {"key": "m" * 32, "groups": ["mcp-readonly"]},
            }
        )

        mock_repo = AsyncMock()
        mock_repo.get_group_mappings.return_value = ["mcp-readonly/read"]

        with (
            patch.object(server_module, "REGISTRY_STATIC_TOKEN_AUTH_ENABLED", True),
            patch.object(server_module, "REGISTRY_API_TOKEN", "t" * 32),
            patch.object(server_module, "_REGISTRY_API_KEYS_RAW", raw),
            patch.object(server_module, "_STATIC_TOKEN_MAP", {}),
            patch(
                "auth_server.server.get_scope_repository",
                return_value=mock_repo,
            ),
        ):
            await server_module._build_static_token_map()
            assert "monitoring" in server_module._STATIC_TOKEN_MAP
            assert "legacy" in server_module._STATIC_TOKEN_MAP
            assert len(server_module._STATIC_TOKEN_MAP) == 2

    @pytest.mark.asyncio
    async def test_zero_keys_warns_but_stays_enabled(self):
        """Empty REGISTRY_API_KEYS and empty REGISTRY_API_TOKEN logs warning."""
        import auth_server.server as server_module

        with (
            patch.object(server_module, "REGISTRY_STATIC_TOKEN_AUTH_ENABLED", True),
            patch.object(server_module, "REGISTRY_API_TOKEN", ""),
            patch.object(server_module, "_REGISTRY_API_KEYS_RAW", ""),
            patch.object(server_module, "_STATIC_TOKEN_MAP", {}),
        ):
            await server_module._build_static_token_map()
            assert server_module._STATIC_TOKEN_MAP == {}
            # Feature stays enabled (callers just fall through to JWT)
            assert server_module.REGISTRY_STATIC_TOKEN_AUTH_ENABLED is True


# =============================================================================
# MULTI-KEY VALIDATE INTEGRATION TESTS (issue #779)
# =============================================================================


class TestMultiKeyStaticTokenValidate:
    """Integration tests for multi-key static token through /validate."""

    def test_named_key_returns_key_name_as_username(self):
        """A named key match returns the key name as X-Username."""
        import auth_server.server as server_module

        token_map = {
            "ci-runner": {
                "key_bytes": ("c" * 32).encode("utf-8"),
                "groups": ["mcp-registry-admin"],
                "scopes": ["mcp-servers-unrestricted/read"],
            },
        }
        with (
            patch.object(server_module, "REGISTRY_STATIC_TOKEN_AUTH_ENABLED", True),
            patch.object(server_module, "_STATIC_TOKEN_MAP", token_map),
        ):
            client = TestClient(server_module.app)
            response = client.get(
                "/validate",
                headers={
                    "Authorization": f"Bearer {'c' * 32}",
                    "X-Original-URL": "https://example.com/api/servers",
                },
            )

            assert response.status_code == 200
            data = response.json()
            assert data["username"] == "ci-runner"
            assert data["client_id"] == "ci-runner"
            assert data["method"] == "network-trusted"
            assert response.headers["X-Username"] == "ci-runner"

    def test_readonly_key_gets_limited_scopes(self):
        """A read-only key gets only the scopes configured for its groups."""
        import auth_server.server as server_module

        token_map = {
            "readonly-monitor": {
                "key_bytes": ("r" * 32).encode("utf-8"),
                "groups": ["mcp-readonly"],
                "scopes": ["mcp-readonly/read"],
            },
        }
        with (
            patch.object(server_module, "REGISTRY_STATIC_TOKEN_AUTH_ENABLED", True),
            patch.object(server_module, "_STATIC_TOKEN_MAP", token_map),
        ):
            client = TestClient(server_module.app)
            response = client.get(
                "/validate",
                headers={
                    "Authorization": f"Bearer {'r' * 32}",
                    "X-Original-URL": "https://example.com/api/servers",
                },
            )

            assert response.status_code == 200
            data = response.json()
            assert data["scopes"] == ["mcp-readonly/read"]
            assert data["groups"] == ["mcp-readonly"]

    def test_key_with_empty_scopes_still_matches(self):
        """A key whose groups map to no scopes still matches (but will 403 at registry)."""
        import auth_server.server as server_module

        token_map = {
            "empty-scope-key": {
                "key_bytes": ("e" * 32).encode("utf-8"),
                "groups": ["ghost-group"],
                "scopes": [],
            },
        }
        with (
            patch.object(server_module, "REGISTRY_STATIC_TOKEN_AUTH_ENABLED", True),
            patch.object(server_module, "_STATIC_TOKEN_MAP", token_map),
        ):
            client = TestClient(server_module.app)
            response = client.get(
                "/validate",
                headers={
                    "Authorization": f"Bearer {'e' * 32}",
                    "X-Original-URL": "https://example.com/api/servers",
                },
            )

            assert response.status_code == 200
            data = response.json()
            assert data["scopes"] == []
            assert data["username"] == "empty-scope-key"


# =============================================================================
# MCP PROXY RESPONSE-HEADER ALLOWLIST TESTS
# =============================================================================
#
# The /mcp-proxy/{server_name} hop introduced in Issue #1026 buffers the
# upstream MCP response and re-emits it via Starlette. Two regressions live
# on the same code path:
#
#   1. Earlier revisions silently dropped every upstream response header,
#      including Mcp-Session-Id which streamable-http MCP servers emit
#      during initialize and which the client requires on follow-up
#      requests. These tests pin the three return-site branches so that
#      cannot recur.
#
#   2. The auth-server sits on a trust boundary in front of arbitrary
#      upstream MCP servers. Forwarding every header (denylist) lets an
#      upstream dictate Set-Cookie, Location, Strict-Transport-Security,
#      Content-Security-Policy, Access-Control-Allow-*, Server, etc. The
#      handler must instead use an allowlist (_FORWARDED_RESPONSE_HEADERS)
#      that opts in the small set of MCP-essential headers only. These
#      tests pin the allowlist and assert dangerous headers are dropped.


def _build_mock_upstream_response(
    status_code: int,
    headers: dict[str, str],
    body: bytes,
) -> MagicMock:
    """Build a MagicMock that mimics the surface of httpx.Response used by
    auth_server.server.mcp_proxy: aiter_bytes(chunk_size=...), status_code,
    and headers.
    """
    mock_resp = MagicMock()
    mock_resp.status_code = status_code
    mock_resp.headers = headers

    async def _aiter(chunk_size: int = 64 * 1024):
        yield body

    mock_resp.aiter_bytes = _aiter
    return mock_resp


def _patch_httpx_async_client(mock_upstream_response: MagicMock):
    """Return a patch context manager that replaces
    ``auth_server.server.httpx.AsyncClient`` so the ``async with
    httpx.AsyncClient(...) as client: async with client.stream(...) as
    response:`` chain yields ``mock_upstream_response``.
    """
    mock_stream_cm = AsyncMock()
    mock_stream_cm.__aenter__ = AsyncMock(return_value=mock_upstream_response)
    mock_stream_cm.__aexit__ = AsyncMock(return_value=False)

    mock_client = AsyncMock()
    mock_client.stream = MagicMock(return_value=mock_stream_cm)
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)

    return patch("auth_server.server.httpx.AsyncClient", return_value=mock_client)


class TestForwardedResponseHeadersAllowlist:
    """Tests for ``_select_forwarded_response_headers`` -- the function that
    enforces the upstream-response-header allowlist.

    Two responsibilities under test:
      1. Allowlisted headers (Mcp-Session-Id, X-Mcp-Session-Id, WWW-
         Authenticate, Retry-After) survive, case-insensitively.
      2. Everything else, including framing headers and headers in the
         review specifically called out as security-sensitive
         (Set-Cookie, Location, HSTS, CSP, ACAO, Server), is dropped.
    """

    def test_allowlist_membership_matches_constant(self):
        """Pin the allowlist contents. Adding to the set is a
        security-relevant change; this test forces the PR author to
        update it intentionally and the reviewer to re-evaluate.
        """
        from auth_server.server import _FORWARDED_RESPONSE_HEADERS

        assert _FORWARDED_RESPONSE_HEADERS == frozenset(
            {
                "mcp-session-id",
                "x-mcp-session-id",
                "www-authenticate",
                "retry-after",
            }
        )

    def test_keeps_mcp_session_id(self):
        """Mcp-Session-Id is the original symptom of issue #1096 -- it
        must survive the allowlist or streamable-http MCP clients lose
        their session.
        """
        from auth_server.server import _select_forwarded_response_headers

        upstream = {
            "content-type": "application/json",
            "mcp-session-id": "sess-abc-123",
        }

        forwarded = _select_forwarded_response_headers(upstream)

        assert forwarded == {"mcp-session-id": "sess-abc-123"}

    def test_keeps_x_mcp_session_id_legacy_alias(self):
        """Some MCP servers emit the legacy X-Mcp-Session-Id alias; the
        allowlist must include it for the same session-management reason.
        """
        from auth_server.server import _select_forwarded_response_headers

        upstream = {"x-mcp-session-id": "sess-legacy-007"}

        assert _select_forwarded_response_headers(upstream) == {
            "x-mcp-session-id": "sess-legacy-007",
        }

    def test_keeps_www_authenticate_for_prm_flow(self):
        """PR #1115 introduced the OAuth PRM / resource-metadata 401
        flow that depends on WWW-Authenticate reaching the MCP client.
        """
        from auth_server.server import _select_forwarded_response_headers

        upstream = {
            "www-authenticate": 'Bearer resource_metadata="https://idp.example/.well-known/oauth-protected-resource"'
        }

        assert _select_forwarded_response_headers(upstream) == upstream

    def test_keeps_retry_after_for_backoff(self):
        """Retry-After must be forwarded so MCP clients honor upstream
        429/503 backoff instead of retry-storming the upstream.
        """
        from auth_server.server import _select_forwarded_response_headers

        upstream = {"retry-after": "30"}

        assert _select_forwarded_response_headers(upstream) == {"retry-after": "30"}

    def test_match_is_case_insensitive_for_allowlist(self):
        """HTTP header names are case-insensitive (RFC 9110 §5.1). The
        upstream may emit any casing; the allowlist match must work
        regardless, and the helper must preserve the wire casing on
        return so the client sees what the MCP server actually sent.
        """
        from auth_server.server import _select_forwarded_response_headers

        upstream = {
            "Mcp-Session-Id": "sess-xyz",
            "WWW-Authenticate": "Bearer realm=mcp",
            "Retry-After": "5",
        }

        forwarded = _select_forwarded_response_headers(upstream)

        assert forwarded["Mcp-Session-Id"] == "sess-xyz"
        assert forwarded["WWW-Authenticate"] == "Bearer realm=mcp"
        assert forwarded["Retry-After"] == "5"

    def test_drops_set_cookie_and_location_regression(self):
        """Regression for review feedback on PR #1097: an upstream MCP
        server must not be able to set cookies or redirect targets on
        the gateway response. These are the headers the reviewer
        specifically called out -- they are the canonical "trust
        boundary leak" pair and the most likely abuse vector.
        """
        from auth_server.server import _select_forwarded_response_headers

        upstream = {
            "Set-Cookie": "session=evil; Path=/; HttpOnly",
            "Location": "https://attacker.example/oauth/authorize",
            "Mcp-Session-Id": "sess-keep-me",
        }

        forwarded = _select_forwarded_response_headers(upstream)

        assert "Set-Cookie" not in forwarded
        assert "Location" not in forwarded
        # And the allowlisted header is still there so the test is not
        # vacuously asserting an empty dict.
        assert forwarded["Mcp-Session-Id"] == "sess-keep-me"

    def test_drops_security_policy_headers(self):
        """Other reviewer-cited headers -- HSTS, CSP, ACAO, Server -- let
        an upstream override the gateway's security posture. Drop them.
        """
        from auth_server.server import _select_forwarded_response_headers

        upstream = {
            "Strict-Transport-Security": "max-age=0",
            "Content-Security-Policy": "default-src *",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Credentials": "true",
            "Server": "evil-upstream/1.0",
        }

        assert _select_forwarded_response_headers(upstream) == {}

    def test_drops_framing_headers(self):
        """Framing / encoding headers describe the upstream wire
        message; Starlette must recompute them for the body it actually
        serializes (which may be httpx-decoded when the upstream used
        Content-Encoding: gzip). They are not on the allowlist, so the
        allowlist drops them for free.
        """
        from auth_server.server import _select_forwarded_response_headers

        upstream = {
            "content-length": "1024",
            "content-encoding": "gzip",
            "transfer-encoding": "chunked",
            "connection": "keep-alive",
        }

        assert _select_forwarded_response_headers(upstream) == {}

    def test_drops_arbitrary_x_headers(self):
        """The reviewer wants opt-in, not opt-out. Custom X-* headers
        from the upstream must not survive even if they look innocuous.
        """
        from auth_server.server import _select_forwarded_response_headers

        upstream = {
            "X-Frame-Options": "DENY",
            "X-Custom-Tracking": "abc123",
            "X-Powered-By": "MCP/0.1",
        }

        assert _select_forwarded_response_headers(upstream) == {}

    def test_empty_input_returns_empty(self):
        """Defensive: an empty upstream-header dict must produce an
        empty forwarded dict, never a None or an error.
        """
        from auth_server.server import _select_forwarded_response_headers

        assert _select_forwarded_response_headers({}) == {}


class TestMcpProxyEndpointHeaderPassthrough:
    """End-to-end-style tests that drive the FastAPI /mcp-proxy/... route
    with a mocked httpx upstream and assert the client-visible response
    headers include the upstream Mcp-Session-Id.
    """

    def test_session_id_forwarded_on_passthrough_branch(self):
        """SSE / non-tools-list response: the early-return branch at
        ``if not should_filter`` must include upstream Mcp-Session-Id on
        the Starlette Response sent to the client.
        """
        import auth_server.server as server_module

        upstream_resp = _build_mock_upstream_response(
            status_code=200,
            headers={
                "content-type": "text/event-stream",
                "mcp-session-id": "sess-passthrough-001",
                "content-length": "12",
            },
            body=b"data: hello\n\n",
        )

        with (
            _patch_httpx_async_client(upstream_resp),
            patch.object(server_module, "_read_mcp_filter_enabled", return_value=False),
        ):
            client = TestClient(server_module.app)
            response = client.post(
                "/mcp-proxy/office-docs",
                json={"jsonrpc": "2.0", "id": 1, "method": "initialize"},
                headers={"X-Upstream-Url": "https://upstream.example/mcp"},
            )

        assert response.status_code == 200
        assert response.headers.get("mcp-session-id") == "sess-passthrough-001"
        # Framing header must not survive: Starlette sets its own.
        assert response.headers.get("content-length") != "12"

    def test_session_id_forwarded_on_filtered_tools_list_branch(self):
        """tools/list with JSON body and filter enabled: the filtered
        JSONResponse branch must also include upstream Mcp-Session-Id.
        """
        import auth_server.server as server_module

        upstream_body = b'{"jsonrpc":"2.0","id":1,"result":{"tools":[{"name":"t1"},{"name":"t2"}]}}'
        upstream_resp = _build_mock_upstream_response(
            status_code=200,
            headers={
                "content-type": "application/json",
                "mcp-session-id": "sess-filtered-002",
            },
            body=upstream_body,
        )

        async def _no_filter(server_name, user_scopes, tools):
            return tools

        with (
            _patch_httpx_async_client(upstream_resp),
            patch.object(server_module, "_read_mcp_filter_enabled", return_value=True),
            patch.object(
                server_module,
                "filter_tools_list_response",
                side_effect=_no_filter,
            ),
        ):
            client = TestClient(server_module.app)
            response = client.post(
                "/mcp-proxy/office-docs",
                json={"jsonrpc": "2.0", "id": 1, "method": "tools/list"},
                headers={"X-Upstream-Url": "https://upstream.example/mcp"},
            )

        assert response.status_code == 200
        assert response.headers.get("mcp-session-id") == "sess-filtered-002"

    def test_session_id_forwarded_on_tools_list_non_json_fallback(self):
        """tools/list whose upstream body fails JSON parsing falls into
        the ``_safe_parse_body`` JSONResponse branch; that branch must
        still forward upstream Mcp-Session-Id.
        """
        import auth_server.server as server_module

        upstream_resp = _build_mock_upstream_response(
            status_code=200,
            headers={
                "content-type": "application/json",
                "mcp-session-id": "sess-malformed-003",
            },
            body=b"not-json-at-all",
        )

        with (
            _patch_httpx_async_client(upstream_resp),
            patch.object(server_module, "_read_mcp_filter_enabled", return_value=True),
        ):
            client = TestClient(server_module.app)
            response = client.post(
                "/mcp-proxy/office-docs",
                json={"jsonrpc": "2.0", "id": 1, "method": "tools/list"},
                headers={"X-Upstream-Url": "https://upstream.example/mcp"},
            )

        assert response.status_code == 200
        assert response.headers.get("mcp-session-id") == "sess-malformed-003"

    def test_session_id_preserved_when_filter_flag_is_false(self):
        """Regression guard for the workaround question: setting
        MCP_TOOLS_LIST_FILTER_ENABLED=false alone is NOT enough --
        nginx still routes through this endpoint, and the response
        builder must forward Mcp-Session-Id regardless of the flag.
        """
        import auth_server.server as server_module

        upstream_resp = _build_mock_upstream_response(
            status_code=200,
            headers={
                "content-type": "application/json",
                "mcp-session-id": "sess-flag-off-004",
            },
            body=b'{"jsonrpc":"2.0","id":1,"result":{"tools":[]}}',
        )

        with (
            _patch_httpx_async_client(upstream_resp),
            patch.object(server_module, "_read_mcp_filter_enabled", return_value=False),
        ):
            client = TestClient(server_module.app)
            response = client.post(
                "/mcp-proxy/office-docs",
                json={"jsonrpc": "2.0", "id": 1, "method": "tools/list"},
                headers={"X-Upstream-Url": "https://upstream.example/mcp"},
            )

        assert response.status_code == 200
        assert response.headers.get("mcp-session-id") == "sess-flag-off-004"

    def test_missing_upstream_url_header_rejected(self):
        """Sanity: the existing 400 guard for missing X-Upstream-Url is
        unaffected by the patch. Prevents accidental loosening.
        """
        import auth_server.server as server_module

        client = TestClient(server_module.app)
        response = client.post(
            "/mcp-proxy/office-docs",
            json={"jsonrpc": "2.0", "id": 1, "method": "initialize"},
        )

        assert response.status_code == 400
        assert "X-Upstream-Url" in response.json()["detail"]

    def test_set_cookie_and_location_dropped_end_to_end(self):
        """End-to-end regression: even if an upstream MCP server emits
        Set-Cookie and Location, the Starlette response the gateway
        sends back to the client must not carry them. This is the
        regression test the reviewer explicitly asked for on the
        allowlist conversion.
        """
        import auth_server.server as server_module

        upstream_resp = _build_mock_upstream_response(
            status_code=200,
            headers={
                "content-type": "application/json",
                "mcp-session-id": "sess-allowlist-100",
                "set-cookie": "session=evil; Path=/; HttpOnly",
                "location": "https://attacker.example/oauth/authorize",
                "strict-transport-security": "max-age=0",
                "access-control-allow-origin": "*",
            },
            body=b'{"jsonrpc":"2.0","id":1,"result":{}}',
        )

        with (
            _patch_httpx_async_client(upstream_resp),
            patch.object(server_module, "_read_mcp_filter_enabled", return_value=False),
        ):
            client = TestClient(server_module.app)
            response = client.post(
                "/mcp-proxy/office-docs",
                json={"jsonrpc": "2.0", "id": 1, "method": "initialize"},
                headers={"X-Upstream-Url": "https://upstream.example/mcp"},
            )

        assert response.status_code == 200
        # Allowlisted header survives.
        assert response.headers.get("mcp-session-id") == "sess-allowlist-100"
        # The reviewer-cited headers must not reach the client.
        assert "set-cookie" not in {k.lower() for k in response.headers.keys()}
        assert response.headers.get("location") is None
        assert response.headers.get("strict-transport-security") is None
        assert response.headers.get("access-control-allow-origin") is None

    def test_www_authenticate_and_retry_after_forwarded(self):
        """The PRM/OAuth flow (PR #1115) needs WWW-Authenticate on 401s,
        and rate-limit-aware MCP clients honor Retry-After. Both must
        round-trip through the gateway.
        """
        import auth_server.server as server_module

        upstream_resp = _build_mock_upstream_response(
            status_code=401,
            headers={
                "content-type": "application/json",
                "www-authenticate": 'Bearer resource_metadata="https://idp.example/.well-known/oauth-protected-resource"',
                "retry-after": "15",
            },
            body=b'{"error":"unauthorized"}',
        )

        with (
            _patch_httpx_async_client(upstream_resp),
            patch.object(server_module, "_read_mcp_filter_enabled", return_value=False),
        ):
            client = TestClient(server_module.app)
            response = client.post(
                "/mcp-proxy/office-docs",
                json={"jsonrpc": "2.0", "id": 1, "method": "initialize"},
                headers={"X-Upstream-Url": "https://upstream.example/mcp"},
            )

        assert response.status_code == 401
        assert response.headers.get("www-authenticate", "").startswith("Bearer")
        assert response.headers.get("retry-after") == "15"
