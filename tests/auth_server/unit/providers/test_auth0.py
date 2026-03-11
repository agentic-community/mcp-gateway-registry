"""
Unit tests for auth_server/providers/auth0.py

Tests the Auth0 authentication provider implementation including
token validation, JWKS handling, OAuth2 flows, and M2M authentication.
"""

import logging
import time
from unittest.mock import MagicMock, patch

import jwt
import pytest
import requests

logger = logging.getLogger(__name__)


# Mark all tests in this file
pytestmark = [pytest.mark.unit, pytest.mark.auth]


# =============================================================================
# AUTH0 PROVIDER INITIALIZATION TESTS
# =============================================================================


class TestAuth0ProviderInit:
    """Tests for Auth0Provider initialization."""

    def test_provider_initialization_basic(self):
        """Test basic provider initialization."""
        from auth_server.providers.auth0 import Auth0Provider

        # Act
        provider = Auth0Provider(
            domain="test-tenant.auth0.com",
            client_id="test-client",
            client_secret="test-secret",
        )

        # Assert
        assert provider.domain == "test-tenant.auth0.com"
        assert provider.client_id == "test-client"
        assert provider.client_secret == "test-secret"
        assert provider.audience is None

    def test_provider_initialization_with_audience(self):
        """Test initialization with API audience."""
        from auth_server.providers.auth0 import Auth0Provider

        # Act
        provider = Auth0Provider(
            domain="test-tenant.auth0.com",
            client_id="test-client",
            client_secret="test-secret",
            audience="https://api.example.com",
        )

        # Assert
        assert provider.audience == "https://api.example.com"

    def test_provider_initialization_removes_trailing_slashes(self):
        """Test that trailing slashes are removed from domain."""
        from auth_server.providers.auth0 import Auth0Provider

        # Act
        provider = Auth0Provider(
            domain="test-tenant.auth0.com/",
            client_id="test-client",
            client_secret="test-secret",
        )

        # Assert
        assert not provider.domain.endswith("/")

    def test_provider_initialization_m2m_defaults(self):
        """Test M2M client defaults to main client."""
        from auth_server.providers.auth0 import Auth0Provider

        # Act
        provider = Auth0Provider(
            domain="test-tenant.auth0.com",
            client_id="test-client",
            client_secret="test-secret",
        )

        # Assert
        assert provider.m2m_client_id == "test-client"
        assert provider.m2m_client_secret == "test-secret"

    def test_provider_initialization_separate_m2m_client(self):
        """Test initialization with separate M2M client."""
        from auth_server.providers.auth0 import Auth0Provider

        # Act
        provider = Auth0Provider(
            domain="test-tenant.auth0.com",
            client_id="web-client",
            client_secret="web-secret",
            m2m_client_id="m2m-client",
            m2m_client_secret="m2m-secret",
        )

        # Assert
        assert provider.client_id == "web-client"
        assert provider.m2m_client_id == "m2m-client"
        assert provider.m2m_client_secret == "m2m-secret"

    def test_provider_initialization_custom_groups_claim(self):
        """Test initialization with custom groups claim."""
        from auth_server.providers.auth0 import Auth0Provider

        # Act
        provider = Auth0Provider(
            domain="test-tenant.auth0.com",
            client_id="test-client",
            client_secret="test-secret",
            groups_claim="https://custom-ns/roles",
        )

        # Assert
        assert provider.groups_claim == "https://custom-ns/roles"

    def test_provider_endpoints(self):
        """Test that Auth0 endpoints are correctly constructed."""
        from auth_server.providers.auth0 import Auth0Provider

        # Act
        provider = Auth0Provider(
            domain="test-tenant.auth0.com",
            client_id="test-client",
            client_secret="test-secret",
        )

        # Assert
        assert provider.auth_url == "https://test-tenant.auth0.com/authorize"
        assert provider.token_url == "https://test-tenant.auth0.com/oauth/token"
        assert provider.userinfo_url == "https://test-tenant.auth0.com/userinfo"
        assert provider.jwks_url == "https://test-tenant.auth0.com/.well-known/jwks.json"
        assert provider.logout_url == "https://test-tenant.auth0.com/v2/logout"
        assert provider.issuer == "https://test-tenant.auth0.com/"


# =============================================================================
# JWKS RETRIEVAL TESTS
# =============================================================================


class TestAuth0JWKS:
    """Tests for JWKS retrieval and caching."""

    @patch("auth_server.providers.auth0.requests.get")
    def test_get_jwks_success(self, mock_get, mock_jwks_response):
        """Test successful JWKS retrieval."""
        from auth_server.providers.auth0 import Auth0Provider

        # Arrange
        mock_response = MagicMock()
        mock_response.json.return_value = mock_jwks_response
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        provider = Auth0Provider(
            domain="test-tenant.auth0.com",
            client_id="test-client",
            client_secret="test-secret",
        )

        # Act
        jwks = provider.get_jwks()

        # Assert
        assert "keys" in jwks
        assert len(jwks["keys"]) == 2
        mock_get.assert_called_once()
        assert "/.well-known/jwks.json" in mock_get.call_args[0][0]

    @patch("auth_server.providers.auth0.requests.get")
    def test_get_jwks_caching(self, mock_get, mock_jwks_response):
        """Test that JWKS is cached and not fetched repeatedly."""
        from auth_server.providers.auth0 import Auth0Provider

        # Arrange
        mock_response = MagicMock()
        mock_response.json.return_value = mock_jwks_response
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        provider = Auth0Provider(
            domain="test-tenant.auth0.com",
            client_id="test-client",
            client_secret="test-secret",
        )

        # Act - call multiple times
        jwks1 = provider.get_jwks()
        jwks2 = provider.get_jwks()
        jwks3 = provider.get_jwks()

        # Assert - should only call once due to caching
        assert mock_get.call_count == 1
        assert jwks1 == jwks2 == jwks3

    @patch("auth_server.providers.auth0.requests.get")
    @patch("auth_server.providers.auth0.time.time")
    def test_get_jwks_cache_expiration(self, mock_time, mock_get, mock_jwks_response):
        """Test that JWKS cache expires after TTL."""
        from auth_server.providers.auth0 import Auth0Provider

        # Arrange
        mock_response = MagicMock()
        mock_response.json.return_value = mock_jwks_response
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        provider = Auth0Provider(
            domain="test-tenant.auth0.com",
            client_id="test-client",
            client_secret="test-secret",
        )

        # First call
        mock_time.return_value = 1000
        provider.get_jwks()

        # Second call - cache should still be valid
        mock_time.return_value = 1100
        provider.get_jwks()

        # Third call - cache should be expired (TTL is 3600 seconds)
        mock_time.return_value = 5000
        provider.get_jwks()

        # Assert
        assert mock_get.call_count == 2  # First call + after expiration

    @patch("auth_server.providers.auth0.requests.get")
    def test_get_jwks_network_error(self, mock_get):
        """Test JWKS retrieval with network error."""
        from auth_server.providers.auth0 import Auth0Provider

        # Arrange
        mock_get.side_effect = requests.RequestException("Network error")

        provider = Auth0Provider(
            domain="test-tenant.auth0.com",
            client_id="test-client",
            client_secret="test-secret",
        )

        # Act & Assert
        with pytest.raises(ValueError, match="Cannot retrieve JWKS"):
            provider.get_jwks()


# =============================================================================
# TOKEN VALIDATION TESTS
# =============================================================================


class TestAuth0TokenValidation:
    """Tests for JWT token validation."""

    @patch("auth_server.providers.auth0.requests.get")
    def test_validate_token_success(self, mock_get, mock_jwks_response):
        """Test successful token validation."""
        from auth_server.providers.auth0 import Auth0Provider

        # Arrange
        mock_response = MagicMock()
        mock_response.json.return_value = mock_jwks_response
        mock_get.return_value = mock_response

        provider = Auth0Provider(
            domain="test-tenant.auth0.com",
            client_id="test-client",
            client_secret="test-secret",
        )

        now = int(time.time())
        payload = {
            "iss": "https://test-tenant.auth0.com/",
            "aud": "test-client",
            "sub": "auth0|user-123",
            "nickname": "testuser",
            "email": "testuser@example.com",
            "https://mcp-gateway/groups": ["registry-admins", "developers"],
            "scope": "openid profile email",
            "azp": "test-client",
            "exp": now + 3600,
            "iat": now,
        }

        with patch("auth_server.providers.auth0.jwt.get_unverified_header") as mock_header:
            with patch("auth_server.providers.auth0.jwt.decode") as mock_decode:
                mock_header.return_value = {"kid": "test-key-id-1"}
                mock_decode.return_value = payload

                with patch("jwt.PyJWK") as mock_pyjwk:
                    mock_key = MagicMock()
                    mock_pyjwk.return_value.key = mock_key

                    # Act
                    result = provider.validate_token("test-token")

                    # Assert
                    assert result["valid"] is True
                    assert result["username"] == "testuser"
                    assert result["email"] == "testuser@example.com"
                    assert "registry-admins" in result["groups"]
                    assert "developers" in result["groups"]
                    assert result["method"] == "auth0"

    @patch("auth_server.providers.auth0.requests.get")
    def test_validate_token_with_permissions_fallback(self, mock_get, mock_jwks_response):
        """Test token validation falls back to permissions claim for groups."""
        from auth_server.providers.auth0 import Auth0Provider

        # Arrange
        mock_response = MagicMock()
        mock_response.json.return_value = mock_jwks_response
        mock_get.return_value = mock_response

        provider = Auth0Provider(
            domain="test-tenant.auth0.com",
            client_id="test-client",
            client_secret="test-secret",
        )

        now = int(time.time())
        payload = {
            "iss": "https://test-tenant.auth0.com/",
            "aud": "test-client",
            "sub": "auth0|user-123",
            "nickname": "testuser",
            "permissions": ["read:servers", "write:servers"],
            "exp": now + 3600,
            "iat": now,
        }

        with patch("auth_server.providers.auth0.jwt.get_unverified_header") as mock_header:
            with patch("auth_server.providers.auth0.jwt.decode") as mock_decode:
                mock_header.return_value = {"kid": "test-key-id-1"}
                mock_decode.return_value = payload

                with patch("jwt.PyJWK") as mock_pyjwk:
                    mock_pyjwk.return_value.key = MagicMock()

                    # Act
                    result = provider.validate_token("test-token")

                    # Assert
                    assert result["valid"] is True
                    assert result["groups"] == ["read:servers", "write:servers"]

    @patch("auth_server.providers.auth0.requests.get")
    def test_validate_token_expired(self, mock_get, mock_jwks_response):
        """Test validation of expired token."""
        from auth_server.providers.auth0 import Auth0Provider

        # Arrange
        mock_response = MagicMock()
        mock_response.json.return_value = mock_jwks_response
        mock_get.return_value = mock_response

        provider = Auth0Provider(
            domain="test-tenant.auth0.com",
            client_id="test-client",
            client_secret="test-secret",
        )

        with patch("auth_server.providers.auth0.jwt.get_unverified_header") as mock_header:
            with patch("auth_server.providers.auth0.jwt.decode") as mock_decode:
                mock_header.return_value = {"kid": "test-key-id-1"}
                mock_decode.side_effect = jwt.ExpiredSignatureError("Token expired")

                # Act & Assert
                with pytest.raises(ValueError, match="expired"):
                    provider.validate_token("expired-token")

    @patch("auth_server.providers.auth0.requests.get")
    def test_validate_token_no_kid(self, mock_get, mock_jwks_response):
        """Test validation of token without kid header."""
        from auth_server.providers.auth0 import Auth0Provider

        # Arrange
        mock_response = MagicMock()
        mock_response.json.return_value = mock_jwks_response
        mock_get.return_value = mock_response

        provider = Auth0Provider(
            domain="test-tenant.auth0.com",
            client_id="test-client",
            client_secret="test-secret",
        )

        with patch("auth_server.providers.auth0.jwt.get_unverified_header") as mock_header:
            mock_header.return_value = {}  # No kid

            # Act & Assert
            with pytest.raises(ValueError, match="missing 'kid'"):
                provider.validate_token("token-without-kid")

    @patch("auth_server.providers.auth0.requests.get")
    def test_validate_token_key_not_found(self, mock_get, mock_jwks_response):
        """Test validation when signing key is not found."""
        from auth_server.providers.auth0 import Auth0Provider

        # Arrange
        mock_response = MagicMock()
        mock_response.json.return_value = mock_jwks_response
        mock_get.return_value = mock_response

        provider = Auth0Provider(
            domain="test-tenant.auth0.com",
            client_id="test-client",
            client_secret="test-secret",
        )

        with patch("auth_server.providers.auth0.jwt.get_unverified_header") as mock_header:
            mock_header.return_value = {"kid": "unknown-key-id"}

            # Act & Assert
            with pytest.raises(ValueError, match="No matching key found"):
                provider.validate_token("token-with-unknown-kid")

    @patch("auth_server.providers.auth0.requests.get")
    def test_validate_token_with_audience(self, mock_get, mock_jwks_response):
        """Test validation includes audience in valid audiences."""
        from auth_server.providers.auth0 import Auth0Provider

        # Arrange
        mock_response = MagicMock()
        mock_response.json.return_value = mock_jwks_response
        mock_get.return_value = mock_response

        provider = Auth0Provider(
            domain="test-tenant.auth0.com",
            client_id="test-client",
            client_secret="test-secret",
            audience="https://api.example.com",
        )

        now = int(time.time())
        payload = {
            "iss": "https://test-tenant.auth0.com/",
            "aud": "https://api.example.com",
            "sub": "auth0|user-123",
            "nickname": "testuser",
            "exp": now + 3600,
            "iat": now,
        }

        with patch("auth_server.providers.auth0.jwt.get_unverified_header") as mock_header:
            with patch("auth_server.providers.auth0.jwt.decode") as mock_decode:
                mock_header.return_value = {"kid": "test-key-id-1"}
                mock_decode.return_value = payload

                with patch("jwt.PyJWK") as mock_pyjwk:
                    mock_pyjwk.return_value.key = MagicMock()

                    # Act
                    result = provider.validate_token("test-token")

                    # Assert
                    assert result["valid"] is True
                    # Verify audience list includes both client_id and API audience
                    decode_call = mock_decode.call_args
                    assert "https://api.example.com" in decode_call[1]["audience"]
                    assert "test-client" in decode_call[1]["audience"]


# =============================================================================
# OAUTH2 FLOW TESTS
# =============================================================================


class TestAuth0OAuth2:
    """Tests for OAuth2 authorization code flow."""

    @patch("auth_server.providers.auth0.requests.post")
    def test_exchange_code_for_token_success(self, mock_post):
        """Test successful code exchange."""
        from auth_server.providers.auth0 import Auth0Provider

        # Arrange
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "access_token": "access-token-value",
            "id_token": "id-token-value",
            "refresh_token": "refresh-token-value",
            "token_type": "Bearer",
            "expires_in": 3600,
        }
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        provider = Auth0Provider(
            domain="test-tenant.auth0.com",
            client_id="test-client",
            client_secret="test-secret",
        )

        # Act
        result = provider.exchange_code_for_token(
            code="auth-code", redirect_uri="https://app.example.com/callback"
        )

        # Assert
        assert result["access_token"] == "access-token-value"
        assert result["token_type"] == "Bearer"
        assert result["expires_in"] == 3600
        mock_post.assert_called_once()

    @patch("auth_server.providers.auth0.requests.post")
    def test_exchange_code_for_token_error(self, mock_post):
        """Test code exchange with error."""
        from auth_server.providers.auth0 import Auth0Provider

        # Arrange
        mock_post.side_effect = requests.RequestException("Token endpoint error")

        provider = Auth0Provider(
            domain="test-tenant.auth0.com",
            client_id="test-client",
            client_secret="test-secret",
        )

        # Act & Assert
        with pytest.raises(ValueError, match="Token exchange failed"):
            provider.exchange_code_for_token(
                code="invalid-code", redirect_uri="https://app.example.com/callback"
            )

    def test_get_auth_url(self):
        """Test authorization URL generation."""
        from auth_server.providers.auth0 import Auth0Provider

        # Arrange
        provider = Auth0Provider(
            domain="test-tenant.auth0.com",
            client_id="test-client",
            client_secret="test-secret",
        )

        # Act
        auth_url = provider.get_auth_url(
            redirect_uri="https://app.example.com/callback",
            state="random-state",
            scope="openid email profile",
        )

        # Assert
        assert "test-tenant.auth0.com/authorize" in auth_url
        assert "client_id=test-client" in auth_url
        assert "redirect_uri=https" in auth_url
        assert "state=random-state" in auth_url
        assert "scope=openid" in auth_url

    def test_get_auth_url_includes_audience(self):
        """Test authorization URL includes audience when configured."""
        from auth_server.providers.auth0 import Auth0Provider

        # Arrange
        provider = Auth0Provider(
            domain="test-tenant.auth0.com",
            client_id="test-client",
            client_secret="test-secret",
            audience="https://api.example.com",
        )

        # Act
        auth_url = provider.get_auth_url(
            redirect_uri="https://app.example.com/callback",
            state="random-state",
        )

        # Assert
        assert "audience=https" in auth_url

    def test_get_auth_url_no_audience(self):
        """Test authorization URL without audience parameter."""
        from auth_server.providers.auth0 import Auth0Provider

        # Arrange
        provider = Auth0Provider(
            domain="test-tenant.auth0.com",
            client_id="test-client",
            client_secret="test-secret",
        )

        # Act
        auth_url = provider.get_auth_url(
            redirect_uri="https://app.example.com/callback",
            state="random-state",
        )

        # Assert
        assert "audience" not in auth_url

    def test_get_logout_url(self):
        """Test logout URL generation uses Auth0's returnTo parameter."""
        from auth_server.providers.auth0 import Auth0Provider

        # Arrange
        provider = Auth0Provider(
            domain="test-tenant.auth0.com",
            client_id="test-client",
            client_secret="test-secret",
        )

        # Act
        logout_url = provider.get_logout_url(redirect_uri="https://app.example.com/logout")

        # Assert
        assert "test-tenant.auth0.com/v2/logout" in logout_url
        assert "client_id=test-client" in logout_url
        assert "returnTo=https" in logout_url


# =============================================================================
# USER INFO TESTS
# =============================================================================


class TestAuth0UserInfo:
    """Tests for user information retrieval."""

    @patch("auth_server.providers.auth0.requests.get")
    def test_get_user_info_success(self, mock_get):
        """Test successful user info retrieval."""
        from auth_server.providers.auth0 import Auth0Provider

        # Arrange
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "sub": "auth0|user-123",
            "nickname": "testuser",
            "email": "testuser@example.com",
            "email_verified": True,
            "name": "Test User",
        }
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        provider = Auth0Provider(
            domain="test-tenant.auth0.com",
            client_id="test-client",
            client_secret="test-secret",
        )

        # Act
        user_info = provider.get_user_info("access-token")

        # Assert
        assert user_info["nickname"] == "testuser"
        assert user_info["email"] == "testuser@example.com"

    @patch("auth_server.providers.auth0.requests.get")
    def test_get_user_info_error(self, mock_get):
        """Test user info retrieval with error."""
        from auth_server.providers.auth0 import Auth0Provider

        # Arrange
        mock_get.side_effect = requests.RequestException("UserInfo error")

        provider = Auth0Provider(
            domain="test-tenant.auth0.com",
            client_id="test-client",
            client_secret="test-secret",
        )

        # Act & Assert
        with pytest.raises(ValueError, match="User info retrieval failed"):
            provider.get_user_info("invalid-token")


# =============================================================================
# TOKEN REFRESH TESTS
# =============================================================================


class TestAuth0TokenRefresh:
    """Tests for token refresh functionality."""

    @patch("auth_server.providers.auth0.requests.post")
    def test_refresh_token_success(self, mock_post):
        """Test successful token refresh."""
        from auth_server.providers.auth0 import Auth0Provider

        # Arrange
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "access_token": "new-access-token",
            "refresh_token": "new-refresh-token",
            "token_type": "Bearer",
            "expires_in": 3600,
        }
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        provider = Auth0Provider(
            domain="test-tenant.auth0.com",
            client_id="test-client",
            client_secret="test-secret",
        )

        # Act
        result = provider.refresh_token("old-refresh-token")

        # Assert
        assert result["access_token"] == "new-access-token"
        assert result["token_type"] == "Bearer"

    @patch("auth_server.providers.auth0.requests.post")
    def test_refresh_token_error(self, mock_post):
        """Test token refresh with error."""
        from auth_server.providers.auth0 import Auth0Provider

        # Arrange
        mock_post.side_effect = requests.RequestException("Refresh failed")

        provider = Auth0Provider(
            domain="test-tenant.auth0.com",
            client_id="test-client",
            client_secret="test-secret",
        )

        # Act & Assert
        with pytest.raises(ValueError, match="Token refresh failed"):
            provider.refresh_token("invalid-refresh-token")


# =============================================================================
# M2M AUTHENTICATION TESTS
# =============================================================================


class TestAuth0M2M:
    """Tests for machine-to-machine authentication."""

    @patch("auth_server.providers.auth0.requests.post")
    def test_get_m2m_token_success(self, mock_post):
        """Test successful M2M token generation."""
        from auth_server.providers.auth0 import Auth0Provider

        # Arrange
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "access_token": "m2m-access-token",
            "token_type": "Bearer",
            "expires_in": 3600,
        }
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        provider = Auth0Provider(
            domain="test-tenant.auth0.com",
            client_id="web-client",
            client_secret="web-secret",
            audience="https://api.example.com",
            m2m_client_id="m2m-client",
            m2m_client_secret="m2m-secret",
        )

        # Act
        result = provider.get_m2m_token()

        # Assert
        assert result["access_token"] == "m2m-access-token"
        assert result["token_type"] == "Bearer"
        # Should use M2M credentials
        call_data = mock_post.call_args[1]["data"]
        assert call_data["client_id"] == "m2m-client"
        assert call_data["client_secret"] == "m2m-secret"
        assert call_data["grant_type"] == "client_credentials"
        assert call_data["audience"] == "https://api.example.com"

    @patch("auth_server.providers.auth0.requests.post")
    def test_get_m2m_token_custom_credentials(self, mock_post):
        """Test M2M token generation with custom credentials."""
        from auth_server.providers.auth0 import Auth0Provider

        # Arrange
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "access_token": "custom-m2m-token",
            "token_type": "Bearer",
            "expires_in": 3600,
        }
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        provider = Auth0Provider(
            domain="test-tenant.auth0.com",
            client_id="default-client",
            client_secret="default-secret",
            audience="https://api.example.com",
        )

        # Act
        result = provider.get_m2m_token(
            client_id="custom-client", client_secret="custom-secret", scope="custom-scope"
        )

        # Assert
        assert result["access_token"] == "custom-m2m-token"
        call_data = mock_post.call_args[1]["data"]
        assert call_data["client_id"] == "custom-client"
        assert call_data["client_secret"] == "custom-secret"
        assert call_data["scope"] == "custom-scope"

    @patch("auth_server.providers.auth0.requests.post")
    def test_get_m2m_token_no_audience(self, mock_post):
        """Test M2M token without audience configured."""
        from auth_server.providers.auth0 import Auth0Provider

        # Arrange
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "access_token": "m2m-token",
            "token_type": "Bearer",
        }
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        provider = Auth0Provider(
            domain="test-tenant.auth0.com",
            client_id="test-client",
            client_secret="test-secret",
        )

        # Act
        provider.get_m2m_token()

        # Assert - audience should not be in request data
        call_data = mock_post.call_args[1]["data"]
        assert "audience" not in call_data

    def test_validate_m2m_token(self):
        """Test that M2M token validation uses same method as regular tokens."""
        from auth_server.providers.auth0 import Auth0Provider

        # Arrange
        provider = Auth0Provider(
            domain="test-tenant.auth0.com",
            client_id="test-client",
            client_secret="test-secret",
        )

        # Mock validate_token
        with patch.object(provider, "validate_token") as mock_validate:
            mock_validate.return_value = {"valid": True}

            # Act
            result = provider.validate_m2m_token("m2m-token")

            # Assert
            assert result["valid"] is True
            mock_validate.assert_called_once_with("m2m-token")


# =============================================================================
# PROVIDER INFO TESTS
# =============================================================================


class TestAuth0ProviderInfo:
    """Tests for provider information."""

    def test_get_provider_info(self):
        """Test getting provider information."""
        from auth_server.providers.auth0 import Auth0Provider

        # Arrange
        provider = Auth0Provider(
            domain="test-tenant.auth0.com",
            client_id="test-client",
            client_secret="test-secret",
            audience="https://api.example.com",
        )

        # Act
        info = provider.get_provider_info()

        # Assert
        assert info["provider_type"] == "auth0"
        assert info["domain"] == "test-tenant.auth0.com"
        assert info["client_id"] == "test-client"
        assert info["audience"] == "https://api.example.com"
        assert "endpoints" in info
        assert "auth" in info["endpoints"]
        assert "token" in info["endpoints"]
        assert "userinfo" in info["endpoints"]
        assert "jwks" in info["endpoints"]
        assert "logout" in info["endpoints"]
        assert info["issuer"] == "https://test-tenant.auth0.com/"


# =============================================================================
# FACTORY TESTS
# =============================================================================


class TestAuth0Factory:
    """Tests for Auth0 provider factory creation."""

    def test_factory_creates_auth0_provider(self, auth0_env_vars):
        """Test that factory creates Auth0 provider correctly."""
        from auth_server.providers.factory import get_auth_provider

        # Act
        provider = get_auth_provider("auth0")

        # Assert
        from auth_server.providers.auth0 import Auth0Provider

        assert isinstance(provider, Auth0Provider)
        assert provider.domain == "test-tenant.auth0.com"
        assert provider.client_id == "test-client-id"
        assert provider.audience == "https://api.example.com"

    def test_factory_missing_domain(self, monkeypatch):
        """Test factory raises error when domain is missing."""
        from auth_server.providers.factory import get_auth_provider

        # Arrange - set client_id and secret but not domain
        monkeypatch.setenv("AUTH0_CLIENT_ID", "test-client")
        monkeypatch.setenv("AUTH0_CLIENT_SECRET", "test-secret")
        monkeypatch.delenv("AUTH0_DOMAIN", raising=False)

        # Act & Assert
        with pytest.raises(ValueError, match="AUTH0_DOMAIN"):
            get_auth_provider("auth0")

    def test_factory_missing_client_id(self, monkeypatch):
        """Test factory raises error when client_id is missing."""
        from auth_server.providers.factory import get_auth_provider

        # Arrange
        monkeypatch.setenv("AUTH0_DOMAIN", "test.auth0.com")
        monkeypatch.setenv("AUTH0_CLIENT_SECRET", "test-secret")
        monkeypatch.delenv("AUTH0_CLIENT_ID", raising=False)

        # Act & Assert
        with pytest.raises(ValueError, match="AUTH0_CLIENT_ID"):
            get_auth_provider("auth0")

    def test_factory_missing_client_secret(self, monkeypatch):
        """Test factory raises error when client_secret is missing."""
        from auth_server.providers.factory import get_auth_provider

        # Arrange
        monkeypatch.setenv("AUTH0_DOMAIN", "test.auth0.com")
        monkeypatch.setenv("AUTH0_CLIENT_ID", "test-client")
        monkeypatch.delenv("AUTH0_CLIENT_SECRET", raising=False)

        # Act & Assert
        with pytest.raises(ValueError, match="AUTH0_CLIENT_SECRET"):
            get_auth_provider("auth0")
