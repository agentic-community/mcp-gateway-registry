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
        """Test validating a valid session cookie."""
        from itsdangerous import URLSafeTimedSerializer

        from auth_server.server import validate_session_cookie

        # Create a signer with the test SECRET_KEY
        test_signer = URLSafeTimedSerializer(auth_env_vars["SECRET_KEY"])

        # Patch the module's signer to use test key (loaded at import time)
        with patch("auth_server.server.signer", test_signer):
            # Act
            result = await validate_session_cookie(valid_session_cookie)

            # Assert
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
        # Arrange
        from itsdangerous import URLSafeTimedSerializer

        import auth_server.server as server_module

        # Create signer with test SECRET_KEY (module's signer uses different key loaded at import)
        test_signer = URLSafeTimedSerializer(auth_env_vars["SECRET_KEY"])

        with patch(
            "auth_server.server.get_scope_repository", return_value=mock_scope_repository_with_data
        ):
            with patch("auth_server.server.signer", test_signer):
                client = TestClient(server_module.app)

                # Act
                # URL format: /server-name/mcp-endpoint where endpoint is mcp, sse, or messages
                response = client.get(
                    "/validate",
                    headers={
                        "Cookie": f"mcp_gateway_session={valid_session_cookie}",
                        "X-Original-URL": "https://example.com/test-server/mcp",
                    },
                )

                # Assert
                assert response.status_code == 200
                data = response.json()
                assert data["valid"] is True


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
        response = client.post("/internal/tokens", json=request_data)

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
        response = client.post("/internal/tokens", json=request_data)

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
        response = client.post("/internal/tokens", json=request_data)

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
            response = client.post("/internal/tokens", json=request_data)
            assert response.status_code == 200

        # Try one more - should fail
        response = client.post("/internal/tokens", json=request_data)

        # Assert
        assert response.status_code == 429
        assert "Rate limit exceeded" in response.json()["detail"]


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
    def test_reload_scopes_basic_auth_backward_compat(
        self, mock_get_provider, mock_reload_scopes, auth_env_vars
    ):
        """Test that deprecated Basic Auth still works for backward compatibility."""
        # Arrange
        mock_reload_scopes.return_value = {"group_mappings": {}}

        import base64

        import auth_server.server as server_module

        client = TestClient(server_module.app)

        credentials = base64.b64encode(b"testadmin:testadminpass").decode()

        # Act
        response = client.post(
            "/internal/reload-scopes", headers={"Authorization": f"Basic {credentials}"}
        )

        # Assert
        assert response.status_code == 200
        data = response.json()
        assert "successfully" in data["message"]

    @patch("auth_server.server.get_auth_provider")
    def test_reload_scopes_invalid_basic_auth(self, mock_get_provider, auth_env_vars):
        """Test scopes reload with invalid Basic Auth credentials."""
        # Arrange
        import base64

        import auth_server.server as server_module

        client = TestClient(server_module.app)

        credentials = base64.b64encode(b"wrong:password").decode()

        # Act
        response = client.post(
            "/internal/reload-scopes", headers={"Authorization": f"Basic {credentials}"}
        )

        # Assert
        assert response.status_code == 401


# =============================================================================
# NETWORK-TRUSTED MODE TESTS
# =============================================================================


class TestNetworkTrustedMode:
    """Tests for network-trusted auth bypass mode (issue #357)."""

    def test_network_trusted_bypasses_registry_api(self):
        """When enabled, registry API requests bypass JWT validation."""
        # Arrange
        import auth_server.server as server_module

        with (
            patch.object(server_module, "REGISTRY_STATIC_TOKEN_AUTH_ENABLED", True),
            patch.object(server_module, "REGISTRY_API_TOKEN", "test-api-key"),
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

    def test_network_trusted_requires_auth_header(self):
        """Auth header must be present even in network-trusted mode."""
        # Arrange
        import auth_server.server as server_module

        with (
            patch.object(server_module, "REGISTRY_STATIC_TOKEN_AUTH_ENABLED", True),
            patch.object(server_module, "REGISTRY_API_TOKEN", "test-api-key"),
        ):
            client = TestClient(server_module.app)

            # Act
            response = client.get(
                "/validate",
                headers={
                    "X-Original-URL": "https://example.com/api/servers",
                },
            )

            # Assert
            assert response.status_code == 401
            assert "Authorization header required" in response.json()["detail"]

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

        with (
            patch.object(server_module, "REGISTRY_STATIC_TOKEN_AUTH_ENABLED", True),
            patch.object(server_module, "REGISTRY_API_TOKEN", "test-api-key"),
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

        with (
            patch.object(server_module, "REGISTRY_STATIC_TOKEN_AUTH_ENABLED", True),
            patch.object(server_module, "REGISTRY_API_TOKEN", "test-api-key"),
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

        with (
            patch.object(server_module, "REGISTRY_STATIC_TOKEN_AUTH_ENABLED", True),
            patch.object(server_module, "REGISTRY_API_TOKEN", "my-secret-key"),
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

    def test_network_trusted_invalid_api_token(self):
        """When REGISTRY_API_TOKEN is set, mismatched Bearer token is rejected."""
        # Arrange
        import auth_server.server as server_module

        with (
            patch.object(server_module, "REGISTRY_STATIC_TOKEN_AUTH_ENABLED", True),
            patch.object(server_module, "REGISTRY_API_TOKEN", "my-secret-key"),
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

            # Assert
            assert response.status_code == 403
            assert "Invalid API token" in response.json()["detail"]

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

        with (
            patch.object(server_module, "REGISTRY_STATIC_TOKEN_AUTH_ENABLED", True),
            patch.object(server_module, "REGISTRY_API_TOKEN", "test-api-key"),
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

    def test_network_trusted_rejects_non_bearer_scheme(self):
        """Authorization header with non-Bearer scheme is rejected."""
        # Arrange
        import auth_server.server as server_module

        with (
            patch.object(server_module, "REGISTRY_STATIC_TOKEN_AUTH_ENABLED", True),
            patch.object(server_module, "REGISTRY_API_TOKEN", "test-api-key"),
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

            # Assert
            assert response.status_code == 401
            assert "Bearer scheme" in response.json()["detail"]

    def test_network_trusted_rejects_empty_bearer_token(self):
        """Bearer token with empty value is rejected."""
        # Arrange
        import auth_server.server as server_module

        with (
            patch.object(server_module, "REGISTRY_STATIC_TOKEN_AUTH_ENABLED", True),
            patch.object(server_module, "REGISTRY_API_TOKEN", "test-api-key"),
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

            # Assert
            assert response.status_code == 403
            assert "Invalid API token" in response.json()["detail"]


# =============================================================================
# OAUTH TOKEN STORAGE CONFIGURATION TESTS
# =============================================================================


class TestOAuthTokenStorageConfiguration:
    """Tests for OAUTH_STORE_TOKENS_IN_SESSION configuration."""

    def test_oauth_store_tokens_default_true(self, monkeypatch):
        """Test that OAUTH_STORE_TOKENS_IN_SESSION defaults to True."""
        # Arrange - ensure env var is not set
        monkeypatch.delenv("OAUTH_STORE_TOKENS_IN_SESSION", raising=False)

        # Act - test the parsing logic (module is already imported at test collection)
        import os

        result = os.environ.get("OAUTH_STORE_TOKENS_IN_SESSION", "true").lower() == "true"

        # Assert
        assert result is True

    def test_oauth_store_tokens_env_true(self, monkeypatch):
        """Test OAUTH_STORE_TOKENS_IN_SESSION=true is parsed correctly."""
        # Arrange
        import os

        monkeypatch.setenv("OAUTH_STORE_TOKENS_IN_SESSION", "true")

        # Act
        result = os.environ.get("OAUTH_STORE_TOKENS_IN_SESSION", "true").lower() == "true"

        # Assert
        assert result is True

    def test_oauth_store_tokens_env_false(self, monkeypatch):
        """Test OAUTH_STORE_TOKENS_IN_SESSION=false is parsed correctly."""
        # Arrange
        import os

        monkeypatch.setenv("OAUTH_STORE_TOKENS_IN_SESSION", "false")

        # Act
        result = os.environ.get("OAUTH_STORE_TOKENS_IN_SESSION", "true").lower() == "true"

        # Assert
        assert result is False

    def test_oauth_store_tokens_env_false_uppercase(self, monkeypatch):
        """Test OAUTH_STORE_TOKENS_IN_SESSION=FALSE (case insensitive)."""
        # Arrange
        import os

        monkeypatch.setenv("OAUTH_STORE_TOKENS_IN_SESSION", "FALSE")

        # Act
        result = os.environ.get("OAUTH_STORE_TOKENS_IN_SESSION", "true").lower() == "true"

        # Assert
        assert result is False

    def test_session_data_includes_tokens_when_enabled(self):
        """Test session data includes OAuth tokens when OAUTH_STORE_TOKENS_IN_SESSION=true."""
        # Arrange
        mapped_user = {
            "username": "testuser",
            "email": "test@example.com",
            "name": "Test User",
            "groups": ["users"],
        }
        provider = "entra"
        token_data = {
            "access_token": "eyJ0eXAiOiJKV1QiLCJhbGciOiJSUzI1NiIsIng1dCI6InRlc3QifQ...",
            "refresh_token": "refresh_token_value",
            "expires_in": 3600,
        }

        # Act - simulate the session data creation logic
        session_data = {
            "username": mapped_user["username"],
            "email": mapped_user.get("email"),
            "name": mapped_user.get("name"),
            "groups": mapped_user.get("groups", []),
            "provider": provider,
            "auth_method": "oauth2",
        }

        # Simulate OAUTH_STORE_TOKENS_IN_SESSION=true
        oauth_store_tokens = True
        if oauth_store_tokens:
            session_data.update(
                {
                    "access_token": token_data.get("access_token"),
                    "refresh_token": token_data.get("refresh_token"),
                    "token_expires_in": token_data.get("expires_in"),
                    "token_obtained_at": 1234567890,
                }
            )

        # Assert
        assert "access_token" in session_data
        assert "refresh_token" in session_data
        assert "token_expires_in" in session_data
        assert "token_obtained_at" in session_data
        assert session_data["access_token"] == token_data["access_token"]

    def test_session_data_excludes_tokens_when_disabled(self):
        """Test session data excludes OAuth tokens when OAUTH_STORE_TOKENS_IN_SESSION=false."""
        # Arrange
        mapped_user = {
            "username": "testuser",
            "email": "test@example.com",
            "name": "Test User",
            "groups": ["users"],
        }
        provider = "entra"
        token_data = {
            "access_token": "eyJ0eXAiOiJKV1QiLCJhbGciOiJSUzI1NiIsIng1dCI6InRlc3QifQ...",
            "refresh_token": "refresh_token_value",
            "expires_in": 3600,
        }

        # Act - simulate the session data creation logic
        session_data = {
            "username": mapped_user["username"],
            "email": mapped_user.get("email"),
            "name": mapped_user.get("name"),
            "groups": mapped_user.get("groups", []),
            "provider": provider,
            "auth_method": "oauth2",
        }

        # Simulate OAUTH_STORE_TOKENS_IN_SESSION=false
        oauth_store_tokens = False
        if oauth_store_tokens:
            session_data.update(
                {
                    "access_token": token_data.get("access_token"),
                    "refresh_token": token_data.get("refresh_token"),
                    "token_expires_in": token_data.get("expires_in"),
                    "token_obtained_at": 1234567890,
                }
            )

        # Assert - tokens should NOT be in session_data
        assert "access_token" not in session_data
        assert "refresh_token" not in session_data
        assert "token_expires_in" not in session_data
        assert "token_obtained_at" not in session_data
        # But user info should still be present
        assert session_data["username"] == "testuser"
        assert session_data["email"] == "test@example.com"
        assert session_data["provider"] == "entra"

    def test_session_data_size_reduction_when_disabled(self):
        """Test that disabling token storage significantly reduces session data size."""
        # Arrange - simulate a large Entra ID token (typical size ~2000+ chars)
        large_access_token = "eyJ0eXAiOiJKV1QiLCJhbGciOiJSUzI1NiIsIng1dCI6InRlc3QifQ." + "a" * 2000
        large_refresh_token = "refresh_" + "b" * 500

        mapped_user = {
            "username": "testuser@example.com",
            "email": "testuser@example.com",
            "name": "Test User",
            "groups": ["group1", "group2"],
        }

        token_data = {
            "access_token": large_access_token,
            "refresh_token": large_refresh_token,
            "expires_in": 3600,
        }

        # Act - create session with tokens enabled
        session_with_tokens = {
            "username": mapped_user["username"],
            "email": mapped_user.get("email"),
            "name": mapped_user.get("name"),
            "groups": mapped_user.get("groups", []),
            "provider": "entra",
            "auth_method": "oauth2",
            "access_token": token_data.get("access_token"),
            "refresh_token": token_data.get("refresh_token"),
            "token_expires_in": token_data.get("expires_in"),
            "token_obtained_at": 1234567890,
        }

        # Act - create session without tokens
        session_without_tokens = {
            "username": mapped_user["username"],
            "email": mapped_user.get("email"),
            "name": mapped_user.get("name"),
            "groups": mapped_user.get("groups", []),
            "provider": "entra",
            "auth_method": "oauth2",
        }

        # Assert - session without tokens should be much smaller
        import json

        size_with_tokens = len(json.dumps(session_with_tokens))
        size_without_tokens = len(json.dumps(session_without_tokens))

        # Session without tokens should be significantly smaller
        assert size_without_tokens < size_with_tokens
        # With large tokens, the difference should be substantial (>2000 bytes)
        assert size_with_tokens - size_without_tokens > 2000
        # Session without tokens should be under cookie limit (4096 bytes)
        assert size_without_tokens < 4096


# =============================================================================
# OAUTH2 CALLBACK TOKEN STORAGE INTEGRATION TESTS
# =============================================================================


class TestOAuth2CallbackTokenStorage:
    """Test that OAUTH_STORE_TOKENS_IN_SESSION controls actual session cookie content."""

    def _call_oauth2_callback(
        self,
        store_tokens: bool,
    ) -> dict:
        """Call the real oauth2_callback endpoint and return decoded session data.

        Args:
            store_tokens: Value for OAUTH_STORE_TOKENS_IN_SESSION flag

        Returns:
            Decoded session cookie data dict
        """
        from itsdangerous import URLSafeTimedSerializer

        from auth_server.server import (
            SECRET_KEY,
            app,
            signer,
        )

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

        client = TestClient(app, raise_server_exceptions=False)

        with (
            patch("auth_server.server.OAUTH_STORE_TOKENS_IN_SESSION", store_tokens),
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
        ):
            response = client.get(
                "/oauth2/callback/github",
                params={"code": "test-code", "state": "test-state"},
                cookies={"oauth2_temp_session": temp_cookie},
                follow_redirects=False,
            )

        # Extract session cookie from redirect response
        assert response.status_code == 302
        session_cookie = response.cookies.get("mcp_gateway_session")
        assert session_cookie is not None, "Session cookie not set in response"

        # Decode session cookie
        decoder = URLSafeTimedSerializer(SECRET_KEY)
        return decoder.loads(session_cookie)

    def test_tokens_excluded_when_disabled(self):
        """oauth2_callback omits tokens from session when flag is False."""
        session_data = self._call_oauth2_callback(store_tokens=False)

        assert session_data["username"] == "testuser"
        assert session_data["auth_method"] == "oauth2"
        assert "access_token" not in session_data
        assert "refresh_token" not in session_data
        assert "token_expires_in" not in session_data
        assert "token_obtained_at" not in session_data

    def test_tokens_included_when_enabled(self):
        """oauth2_callback includes tokens in session when flag is True."""
        session_data = self._call_oauth2_callback(store_tokens=True)

        assert session_data["username"] == "testuser"
        assert session_data["auth_method"] == "oauth2"
        assert session_data["access_token"] == "mock-access-token-value"
        assert session_data["refresh_token"] == "mock-refresh-token-value"
        assert session_data["token_expires_in"] == 3600
        assert "token_obtained_at" in session_data
