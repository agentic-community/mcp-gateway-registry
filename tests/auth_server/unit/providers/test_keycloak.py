"""
Unit tests for auth_server/providers/keycloak.py

Tests the Keycloak authentication provider implementation including
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
# KEYCLOAK PROVIDER INITIALIZATION TESTS
# =============================================================================


class TestKeycloakProviderInit:
    """Tests for KeycloakProvider initialization."""

    def test_provider_initialization_basic(self):
        """Test basic provider initialization."""
        from auth_server.providers.keycloak import KeycloakProvider

        # Act
        provider = KeycloakProvider(
            keycloak_url="http://localhost:8080",
            realm="test-realm",
            client_id="test-client",
            client_secret="test-secret",
        )

        # Assert
        assert provider.keycloak_url == "http://localhost:8080"
        assert provider.realm == "test-realm"
        assert provider.client_id == "test-client"
        assert provider.client_secret == "test-secret"

    def test_provider_initialization_with_external_url(self):
        """Test initialization with separate external URL."""
        from auth_server.providers.keycloak import KeycloakProvider

        # Act
        provider = KeycloakProvider(
            keycloak_url="http://keycloak:8080",
            realm="test-realm",
            client_id="test-client",
            client_secret="test-secret",
            keycloak_external_url="https://keycloak.example.com",
        )

        # Assert
        assert provider.keycloak_url == "http://keycloak:8080"
        assert provider.keycloak_external_url == "https://keycloak.example.com"
        # Auth URL should use external URL
        assert "keycloak.example.com" in provider.auth_url
        # Token URL should use internal URL
        assert "keycloak:8080" in provider.token_url

    def test_provider_initialization_removes_trailing_slashes(self):
        """Test that trailing slashes are removed from URLs."""
        from auth_server.providers.keycloak import KeycloakProvider

        # Act
        provider = KeycloakProvider(
            keycloak_url="http://localhost:8080/",
            realm="test-realm",
            client_id="test-client",
            client_secret="test-secret",
        )

        # Assert
        assert not provider.keycloak_url.endswith("/")
        assert not provider.keycloak_external_url.endswith("/")

    def test_provider_initialization_m2m_defaults(self):
        """Test M2M client defaults to main client."""
        from auth_server.providers.keycloak import KeycloakProvider

        # Act
        provider = KeycloakProvider(
            keycloak_url="http://localhost:8080",
            realm="test-realm",
            client_id="test-client",
            client_secret="test-secret",
        )

        # Assert
        assert provider.m2m_client_id == "test-client"
        assert provider.m2m_client_secret == "test-secret"

    def test_provider_initialization_separate_m2m_client(self):
        """Test initialization with separate M2M client."""
        from auth_server.providers.keycloak import KeycloakProvider

        # Act
        provider = KeycloakProvider(
            keycloak_url="http://localhost:8080",
            realm="test-realm",
            client_id="web-client",
            client_secret="web-secret",
            m2m_client_id="m2m-client",
            m2m_client_secret="m2m-secret",
        )

        # Assert
        assert provider.client_id == "web-client"
        assert provider.m2m_client_id == "m2m-client"
        assert provider.m2m_client_secret == "m2m-secret"


# =============================================================================
# JWKS RETRIEVAL TESTS
# =============================================================================


class TestKeycloakJWKS:
    """Tests for JWKS retrieval and caching."""

    @patch("auth_server.providers.keycloak.requests.get")
    def test_get_jwks_success(self, mock_get, mock_jwks_response):
        """Test successful JWKS retrieval."""
        from auth_server.providers.keycloak import KeycloakProvider

        # Arrange
        mock_response = MagicMock()
        mock_response.json.return_value = mock_jwks_response
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        provider = KeycloakProvider(
            keycloak_url="http://localhost:8080",
            realm="test-realm",
            client_id="test-client",
            client_secret="test-secret",
        )

        # Act
        jwks = provider.get_jwks()

        # Assert
        assert "keys" in jwks
        assert len(jwks["keys"]) == 2
        mock_get.assert_called_once()
        assert "/protocol/openid-connect/certs" in mock_get.call_args[0][0]

    @patch("auth_server.providers.keycloak.requests.get")
    def test_get_jwks_caching(self, mock_get, mock_jwks_response):
        """Test that JWKS is cached and not fetched repeatedly."""
        from auth_server.providers.keycloak import KeycloakProvider

        # Arrange
        mock_response = MagicMock()
        mock_response.json.return_value = mock_jwks_response
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        provider = KeycloakProvider(
            keycloak_url="http://localhost:8080",
            realm="test-realm",
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

    @patch("auth_server.providers.keycloak.requests.get")
    @patch("auth_server.providers.keycloak.time.time")
    def test_get_jwks_cache_expiration(self, mock_time, mock_get, mock_jwks_response):
        """Test that JWKS cache expires after TTL."""
        from auth_server.providers.keycloak import KeycloakProvider

        # Arrange
        mock_response = MagicMock()
        mock_response.json.return_value = mock_jwks_response
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        provider = KeycloakProvider(
            keycloak_url="http://localhost:8080",
            realm="test-realm",
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

    @patch("auth_server.providers.keycloak.requests.get")
    def test_get_jwks_network_error(self, mock_get):
        """Test JWKS retrieval with network error."""
        from auth_server.providers.keycloak import KeycloakProvider

        # Arrange
        mock_get.side_effect = requests.RequestException("Network error")

        provider = KeycloakProvider(
            keycloak_url="http://localhost:8080",
            realm="test-realm",
            client_id="test-client",
            client_secret="test-secret",
        )

        # Act & Assert
        with pytest.raises(ValueError, match="Cannot retrieve JWKS"):
            provider.get_jwks()


# =============================================================================
# TOKEN VALIDATION TESTS
# =============================================================================


class TestKeycloakTokenValidation:
    """Tests for JWT token validation."""

    @patch("auth_server.providers.keycloak.requests.get")
    def test_validate_token_success(self, mock_get, mock_jwks_response):
        """Test successful token validation."""
        from auth_server.providers.keycloak import KeycloakProvider

        # Arrange
        mock_response = MagicMock()
        mock_response.json.return_value = mock_jwks_response
        mock_get.return_value = mock_response

        provider = KeycloakProvider(
            keycloak_url="http://localhost:8080",
            realm="test-realm",
            client_id="test-client",
            client_secret="test-secret",
        )

        # Create a mock token that will pass basic structure checks
        now = int(time.time())
        payload = {
            "iss": "http://localhost:8080/realms/test-realm",
            "aud": "account",
            "sub": "user-123",
            "preferred_username": "testuser",
            "email": "testuser@example.com",
            "groups": ["users", "admins"],
            "scope": "openid profile email",
            "azp": "test-client",
            "exp": now + 3600,
            "iat": now,
        }

        # Mock JWT validation
        with patch("auth_server.providers.keycloak.jwt.get_unverified_header") as mock_header:
            with patch("auth_server.providers.keycloak.jwt.decode") as mock_decode:
                mock_header.return_value = {"kid": "test-key-id-1"}
                mock_decode.return_value = payload

                # Mock PyJWK - imported dynamically inside function so patch at source
                with patch("jwt.PyJWK") as mock_pyjwk:
                    mock_key = MagicMock()
                    mock_pyjwk.return_value.key = mock_key

                    # Act
                    result = provider.validate_token("test-token")

                    # Assert
                    assert result["valid"] is True
                    assert result["username"] == "testuser"
                    assert result["email"] == "testuser@example.com"
                    assert "users" in result["groups"]
                    assert "admins" in result["groups"]
                    assert result["method"] == "keycloak"

    @patch("auth_server.providers.keycloak.requests.get")
    def test_validate_token_expired(self, mock_get, mock_jwks_response):
        """Test validation of expired token."""
        from auth_server.providers.keycloak import KeycloakProvider

        # Arrange
        mock_response = MagicMock()
        mock_response.json.return_value = mock_jwks_response
        mock_get.return_value = mock_response

        provider = KeycloakProvider(
            keycloak_url="http://localhost:8080",
            realm="test-realm",
            client_id="test-client",
            client_secret="test-secret",
        )

        with patch("auth_server.providers.keycloak.jwt.get_unverified_header") as mock_header:
            with patch("auth_server.providers.keycloak.jwt.decode") as mock_decode:
                mock_header.return_value = {"kid": "test-key-id-1"}
                mock_decode.side_effect = jwt.ExpiredSignatureError("Token expired")

                # Act & Assert
                with pytest.raises(ValueError, match="expired"):
                    provider.validate_token("expired-token")

    @patch("auth_server.providers.keycloak.requests.get")
    def test_validate_token_no_kid(self, mock_get, mock_jwks_response):
        """Test validation of token without kid header."""
        from auth_server.providers.keycloak import KeycloakProvider

        # Arrange
        mock_response = MagicMock()
        mock_response.json.return_value = mock_jwks_response
        mock_get.return_value = mock_response

        provider = KeycloakProvider(
            keycloak_url="http://localhost:8080",
            realm="test-realm",
            client_id="test-client",
            client_secret="test-secret",
        )

        with patch("auth_server.providers.keycloak.jwt.get_unverified_header") as mock_header:
            mock_header.return_value = {}  # No kid

            # Act & Assert
            with pytest.raises(ValueError, match="missing 'kid'"):
                provider.validate_token("token-without-kid")

    @patch("auth_server.providers.keycloak.requests.get")
    def test_validate_token_key_not_found(self, mock_get, mock_jwks_response):
        """Test validation when signing key is not found."""
        from auth_server.providers.keycloak import KeycloakProvider

        # Arrange
        mock_response = MagicMock()
        mock_response.json.return_value = mock_jwks_response
        mock_get.return_value = mock_response

        provider = KeycloakProvider(
            keycloak_url="http://localhost:8080",
            realm="test-realm",
            client_id="test-client",
            client_secret="test-secret",
        )

        with patch("auth_server.providers.keycloak.jwt.get_unverified_header") as mock_header:
            mock_header.return_value = {"kid": "unknown-key-id"}

            # Act & Assert
            with pytest.raises(ValueError, match="No matching key found"):
                provider.validate_token("token-with-unknown-kid")

    @patch("auth_server.providers.keycloak.requests.get")
    def test_validate_token_multiple_issuers(self, mock_get, mock_jwks_response):
        """Test validation with multiple valid issuers."""
        from auth_server.providers.keycloak import KeycloakProvider

        # Arrange
        mock_response = MagicMock()
        mock_response.json.return_value = mock_jwks_response
        mock_get.return_value = mock_response

        provider = KeycloakProvider(
            keycloak_url="http://keycloak:8080",
            realm="test-realm",
            client_id="test-client",
            client_secret="test-secret",
            keycloak_external_url="https://keycloak.example.com",
        )

        # Create payload with external issuer
        now = int(time.time())
        payload = {
            "iss": "https://keycloak.example.com/realms/test-realm",
            "aud": "account",
            "sub": "user-123",
            "preferred_username": "testuser",
            "exp": now + 3600,
            "iat": now,
        }

        with patch("auth_server.providers.keycloak.jwt.get_unverified_header") as mock_header:
            with patch("auth_server.providers.keycloak.jwt.decode") as mock_decode:
                mock_header.return_value = {"kid": "test-key-id-1"}
                mock_decode.return_value = payload

                # Mock PyJWK - imported dynamically inside function so patch at source
                with patch("jwt.PyJWK") as mock_pyjwk:
                    mock_key = MagicMock()
                    mock_pyjwk.return_value.key = mock_key

                    # Act
                    result = provider.validate_token("test-token")

                    # Assert
                    assert result["valid"] is True


# =============================================================================
# OAUTH2 FLOW TESTS
# =============================================================================


class TestKeycloakOAuth2:
    """Tests for OAuth2 authorization code flow."""

    @patch("auth_server.providers.keycloak.requests.post")
    def test_exchange_code_for_token_success(self, mock_post):
        """Test successful code exchange."""
        from auth_server.providers.keycloak import KeycloakProvider

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

        provider = KeycloakProvider(
            keycloak_url="http://localhost:8080",
            realm="test-realm",
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

    @patch("auth_server.providers.keycloak.requests.post")
    def test_exchange_code_for_token_error(self, mock_post):
        """Test code exchange with error."""
        from auth_server.providers.keycloak import KeycloakProvider

        # Arrange
        mock_post.side_effect = requests.RequestException("Token endpoint error")

        provider = KeycloakProvider(
            keycloak_url="http://localhost:8080",
            realm="test-realm",
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
        from auth_server.providers.keycloak import KeycloakProvider

        # Arrange
        provider = KeycloakProvider(
            keycloak_url="http://localhost:8080",
            realm="test-realm",
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
        assert "protocol/openid-connect/auth" in auth_url
        assert "client_id=test-client" in auth_url
        assert "redirect_uri=https" in auth_url
        assert "state=random-state" in auth_url
        assert "scope=openid" in auth_url

    def test_get_logout_url(self):
        """Test logout URL generation."""
        from auth_server.providers.keycloak import KeycloakProvider

        # Arrange
        provider = KeycloakProvider(
            keycloak_url="http://localhost:8080",
            realm="test-realm",
            client_id="test-client",
            client_secret="test-secret",
        )

        # Act
        logout_url = provider.get_logout_url(redirect_uri="https://app.example.com/logout")

        # Assert
        assert "protocol/openid-connect/logout" in logout_url
        assert "client_id=test-client" in logout_url
        assert "post_logout_redirect_uri=https" in logout_url


# =============================================================================
# USER INFO TESTS
# =============================================================================


class TestKeycloakUserInfo:
    """Tests for user information retrieval."""

    @patch("auth_server.providers.keycloak.requests.get")
    def test_get_user_info_success(self, mock_get):
        """Test successful user info retrieval."""
        from auth_server.providers.keycloak import KeycloakProvider

        # Arrange
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "sub": "user-123",
            "preferred_username": "testuser",
            "email": "testuser@example.com",
            "email_verified": True,
            "groups": ["users", "developers"],
        }
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        provider = KeycloakProvider(
            keycloak_url="http://localhost:8080",
            realm="test-realm",
            client_id="test-client",
            client_secret="test-secret",
        )

        # Act
        user_info = provider.get_user_info("access-token")

        # Assert
        assert user_info["preferred_username"] == "testuser"
        assert user_info["email"] == "testuser@example.com"
        assert "users" in user_info["groups"]

    @patch("auth_server.providers.keycloak.requests.get")
    def test_get_user_info_error(self, mock_get):
        """Test user info retrieval with error."""
        from auth_server.providers.keycloak import KeycloakProvider

        # Arrange
        mock_get.side_effect = requests.RequestException("UserInfo error")

        provider = KeycloakProvider(
            keycloak_url="http://localhost:8080",
            realm="test-realm",
            client_id="test-client",
            client_secret="test-secret",
        )

        # Act & Assert
        with pytest.raises(ValueError, match="User info retrieval failed"):
            provider.get_user_info("invalid-token")


# =============================================================================
# TOKEN REFRESH TESTS
# =============================================================================


class TestKeycloakTokenRefresh:
    """Tests for token refresh functionality."""

    @patch("auth_server.providers.keycloak.requests.post")
    def test_refresh_token_success(self, mock_post):
        """Test successful token refresh."""
        from auth_server.providers.keycloak import KeycloakProvider

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

        provider = KeycloakProvider(
            keycloak_url="http://localhost:8080",
            realm="test-realm",
            client_id="test-client",
            client_secret="test-secret",
        )

        # Act
        result = provider.refresh_token("old-refresh-token")

        # Assert
        assert result["access_token"] == "new-access-token"
        assert result["token_type"] == "Bearer"

    @patch("auth_server.providers.keycloak.requests.post")
    def test_refresh_token_error(self, mock_post):
        """Test token refresh with error."""
        from auth_server.providers.keycloak import KeycloakProvider

        # Arrange
        mock_post.side_effect = requests.RequestException("Refresh failed")

        provider = KeycloakProvider(
            keycloak_url="http://localhost:8080",
            realm="test-realm",
            client_id="test-client",
            client_secret="test-secret",
        )

        # Act & Assert
        with pytest.raises(ValueError, match="Token refresh failed"):
            provider.refresh_token("invalid-refresh-token")


# =============================================================================
# M2M AUTHENTICATION TESTS
# =============================================================================


class TestKeycloakM2M:
    """Tests for machine-to-machine authentication."""

    @patch("auth_server.providers.keycloak.requests.post")
    def test_get_m2m_token_success(self, mock_post):
        """Test successful M2M token generation."""
        from auth_server.providers.keycloak import KeycloakProvider

        # Arrange
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "access_token": "m2m-access-token",
            "token_type": "Bearer",
            "expires_in": 3600,
        }
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        provider = KeycloakProvider(
            keycloak_url="http://localhost:8080",
            realm="test-realm",
            client_id="web-client",
            client_secret="web-secret",
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

    @patch("auth_server.providers.keycloak.requests.post")
    def test_get_m2m_token_custom_credentials(self, mock_post):
        """Test M2M token generation with custom credentials."""
        from auth_server.providers.keycloak import KeycloakProvider

        # Arrange
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "access_token": "custom-m2m-token",
            "token_type": "Bearer",
            "expires_in": 3600,
        }
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        provider = KeycloakProvider(
            keycloak_url="http://localhost:8080",
            realm="test-realm",
            client_id="default-client",
            client_secret="default-secret",
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

    def test_validate_m2m_token(self):
        """Test that M2M token validation uses same method as regular tokens."""
        from auth_server.providers.keycloak import KeycloakProvider

        # Arrange
        provider = KeycloakProvider(
            keycloak_url="http://localhost:8080",
            realm="test-realm",
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


class TestKeycloakProviderInfo:
    """Tests for provider information."""

    def test_get_provider_info(self):
        """Test getting provider information."""
        from auth_server.providers.keycloak import KeycloakProvider

        # Arrange
        provider = KeycloakProvider(
            keycloak_url="http://localhost:8080",
            realm="test-realm",
            client_id="test-client",
            client_secret="test-secret",
        )

        # Act
        info = provider.get_provider_info()

        # Assert
        assert info["provider_type"] == "keycloak"
        assert info["realm"] == "test-realm"
        assert info["client_id"] == "test-client"
        assert "endpoints" in info
        assert "auth" in info["endpoints"]
        assert "token" in info["endpoints"]
        assert "userinfo" in info["endpoints"]

    @patch("auth_server.providers.keycloak.requests.get")
    def test_check_keycloak_health(self, mock_get):
        """Test Keycloak health check."""
        from auth_server.providers.keycloak import KeycloakProvider

        # Arrange
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_get.return_value = mock_response

        provider = KeycloakProvider(
            keycloak_url="http://localhost:8080",
            realm="test-realm",
            client_id="test-client",
            client_secret="test-secret",
        )

        # Act
        is_healthy = provider._check_keycloak_health()

        # Assert
        assert is_healthy is True
        mock_get.assert_called_once()
        assert "/health/ready" in mock_get.call_args[0][0]
