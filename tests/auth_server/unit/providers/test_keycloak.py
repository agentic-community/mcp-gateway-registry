"""
Unit tests for auth_server/providers/keycloak.py

Tests the Keycloak authentication provider implementation including
token validation, JWKS handling, OAuth2 flows, and M2M authentication.
"""

import logging
import time
from unittest.mock import MagicMock, patch
from urllib.parse import urlparse

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
        assert urlparse(provider.auth_url).hostname == "keycloak.example.com"
        # Token URL should use internal URL
        assert urlparse(provider.token_url).hostname == "keycloak"

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


class TestKeycloakAuthorizationServerMetadata:
    """Tests for RFC 8414 metadata exposure via authorization_server_metadata()."""

    def _openid_config(self, base_url: str) -> dict:
        return {
            "issuer": f"{base_url}/realms/test-realm",
            "authorization_endpoint": f"{base_url}/realms/test-realm/protocol/openid-connect/auth",
            "token_endpoint": f"{base_url}/realms/test-realm/protocol/openid-connect/token",
            "userinfo_endpoint": f"{base_url}/realms/test-realm/protocol/openid-connect/userinfo",
            "jwks_uri": f"{base_url}/realms/test-realm/protocol/openid-connect/certs",
            "end_session_endpoint": f"{base_url}/realms/test-realm/protocol/openid-connect/logout",
            "registration_endpoint": f"{base_url}/realms/test-realm/clients-registrations/openid-connect",
            "response_types_supported": ["code"],
        }

    @patch("auth_server.providers.keycloak.requests.get")
    def test_metadata_passthrough_when_no_external_url(self, mock_get):
        """Internal == external URL: doc is returned unchanged."""
        from auth_server.providers.keycloak import KeycloakProvider

        config = self._openid_config("http://localhost:8080")
        mock_response = MagicMock()
        mock_response.json.return_value = config
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        provider = KeycloakProvider(
            keycloak_url="http://localhost:8080",
            realm="test-realm",
            client_id="c",
            client_secret="s",
        )

        metadata = provider.authorization_server_metadata()

        assert metadata == config

    @patch("auth_server.providers.keycloak.requests.get")
    def test_metadata_rewrites_internal_to_external(self, mock_get):
        """When external URL differs, browser-facing endpoints are rewritten."""
        from auth_server.providers.keycloak import KeycloakProvider

        config = self._openid_config("http://keycloak:8080")
        mock_response = MagicMock()
        mock_response.json.return_value = config
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        provider = KeycloakProvider(
            keycloak_url="http://keycloak:8080",
            keycloak_external_url="https://auth.example.com",
            realm="test-realm",
            client_id="c",
            client_secret="s",
        )

        metadata = provider.authorization_server_metadata()

        assert metadata["issuer"] == "https://auth.example.com/realms/test-realm"
        assert metadata["authorization_endpoint"].startswith("https://auth.example.com/")
        assert metadata["token_endpoint"].startswith("https://auth.example.com/")
        assert metadata["jwks_uri"].startswith("https://auth.example.com/")
        assert metadata["end_session_endpoint"].startswith("https://auth.example.com/")
        assert metadata["registration_endpoint"].startswith("https://auth.example.com/")
        assert metadata["response_types_supported"] == ["code"]

    @patch("auth_server.providers.keycloak.requests.get")
    def test_authorization_server_issuer_uses_external_url(self, mock_get):
        """The PRM `authorization_servers` entry must be the external issuer."""
        from auth_server.providers.keycloak import KeycloakProvider

        config = self._openid_config("http://keycloak:8080")
        mock_response = MagicMock()
        mock_response.json.return_value = config
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        provider = KeycloakProvider(
            keycloak_url="http://keycloak:8080",
            keycloak_external_url="https://auth.example.com",
            realm="test-realm",
            client_id="c",
            client_secret="s",
        )

        assert (
            provider.authorization_server_issuer() == "https://auth.example.com/realms/test-realm"
        )


class TestKeycloakMetadataRewritesEveryUrl:
    """No internal URL may survive anywhere in the discovery document.

    The rewrite used to enumerate ten field names, so everything it did not list
    kept the internal Docker URL: ``check_session_iframe``, ``pushed_authorization_
    request_endpoint``, ``backchannel_authentication_endpoint`` and the whole
    nested ``mtls_endpoint_aliases`` object. The primary endpoints looked
    correct, so a client using PAR, CIBA, mTLS client auth or OIDC session
    management failed in a way that is hard to attribute (issue #1649).
    """

    INTERNAL = "http://keycloak:8080"
    EXTERNAL = "https://gateway.example.com"

    def _full_openid_config(self, base_url: str) -> dict:
        """A discovery document covering the fields the allowlist missed."""
        realm = f"{base_url}/realms/test-realm"
        oidc = f"{realm}/protocol/openid-connect"
        return {
            "issuer": realm,
            "authorization_endpoint": f"{oidc}/auth",
            "token_endpoint": f"{oidc}/token",
            "userinfo_endpoint": f"{oidc}/userinfo",
            "jwks_uri": f"{oidc}/certs",
            "end_session_endpoint": f"{oidc}/logout",
            "introspection_endpoint": f"{oidc}/token/introspect",
            "revocation_endpoint": f"{oidc}/revoke",
            "device_authorization_endpoint": f"{oidc}/auth/device",
            "registration_endpoint": f"{base_url}/realms/test-realm/clients-registrations/openid-connect",
            # Fields the allowlist never covered:
            "check_session_iframe": f"{oidc}/login-status-iframe.html",
            "backchannel_authentication_endpoint": f"{oidc}/ext/ciba/auth",
            "pushed_authorization_request_endpoint": f"{oidc}/ext/par/request",
            "mtls_endpoint_aliases": {
                "token_endpoint": f"{oidc}/token",
                "revocation_endpoint": f"{oidc}/revoke",
                "introspection_endpoint": f"{oidc}/token/introspect",
                "device_authorization_endpoint": f"{oidc}/auth/device",
                "registration_endpoint": f"{base_url}/realms/test-realm/clients-registrations/openid-connect",
                "userinfo_endpoint": f"{oidc}/userinfo",
                "pushed_authorization_request_endpoint": f"{oidc}/ext/par/request",
                "backchannel_authentication_endpoint": f"{oidc}/ext/ciba/auth",
            },
            # Non-URL values that must survive untouched.
            "response_types_supported": ["code", "id_token"],
            "request_parameter_supported": True,
            "require_request_uri_registration": False,
        }

    def _provider(self, mock_get, config: dict, external: str | None = None):
        from auth_server.providers.keycloak import KeycloakProvider

        mock_response = MagicMock()
        mock_response.json.return_value = config
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response
        return KeycloakProvider(
            keycloak_url=self.INTERNAL,
            keycloak_external_url=self.EXTERNAL if external is None else external,
            realm="test-realm",
            client_id="c",
            client_secret="s",
        )

    @staticmethod
    def _all_strings(node):
        """Yield every string anywhere in a nested JSON structure."""
        if isinstance(node, str):
            yield node
        elif isinstance(node, dict):
            for value in node.values():
                yield from TestKeycloakMetadataRewritesEveryUrl._all_strings(value)
        elif isinstance(node, list):
            for value in node:
                yield from TestKeycloakMetadataRewritesEveryUrl._all_strings(value)

    @patch("auth_server.providers.keycloak.requests.get")
    def test_no_internal_url_survives_anywhere(self, mock_get):
        """The whole-document invariant, stated once.

        Asserting on the document as a whole rather than on named fields is the
        point of the fix: a field Keycloak adds in a future release is covered
        without anyone updating a list.
        """
        provider = self._provider(mock_get, self._full_openid_config(self.INTERNAL))

        metadata = provider.authorization_server_metadata()

        leaked = [s for s in self._all_strings(metadata) if self.INTERNAL in s]
        assert not leaked, f"internal URL survived in {len(leaked)} value(s): {leaked}"

    @pytest.mark.parametrize(
        "field",
        [
            "check_session_iframe",
            "backchannel_authentication_endpoint",
            "pushed_authorization_request_endpoint",
        ],
    )
    @patch("auth_server.providers.keycloak.requests.get")
    def test_previously_missed_top_level_fields(self, mock_get, field):
        """Each field the reporter found still internal on 1.29.0."""
        provider = self._provider(mock_get, self._full_openid_config(self.INTERNAL))

        metadata = provider.authorization_server_metadata()

        assert metadata[field].startswith(self.EXTERNAL), metadata[field]

    @patch("auth_server.providers.keycloak.requests.get")
    def test_nested_mtls_endpoint_aliases_rewritten(self, mock_get):
        """The nested object was not walked at all, so every key inside leaked."""
        provider = self._provider(mock_get, self._full_openid_config(self.INTERNAL))

        aliases = provider.authorization_server_metadata()["mtls_endpoint_aliases"]

        assert aliases, "mtls_endpoint_aliases missing from the rewritten document"
        for key, value in aliases.items():
            assert value.startswith(self.EXTERNAL), f"{key} -> {value}"

    @patch("auth_server.providers.keycloak.requests.get")
    def test_non_url_values_are_untouched(self, mock_get):
        """Lists, booleans and unrelated strings must pass through unchanged."""
        config = self._full_openid_config(self.INTERNAL)
        provider = self._provider(mock_get, config)

        metadata = provider.authorization_server_metadata()

        assert metadata["response_types_supported"] == ["code", "id_token"]
        assert metadata["request_parameter_supported"] is True
        assert metadata["require_request_uri_registration"] is False

    @patch("auth_server.providers.keycloak.requests.get")
    def test_rewrite_anchors_on_a_url_boundary(self, mock_get):
        """A different port on the same host must not be rewritten.

        "http://keycloak:8080" is a string prefix of "http://keycloak:80801",
        which is a different service. A bare startswith would rewrite it.
        """
        config = self._full_openid_config(self.INTERNAL)
        config["token_endpoint"] = "http://keycloak:80801/realms/test-realm/token"
        provider = self._provider(mock_get, config)

        metadata = provider.authorization_server_metadata()

        assert metadata["token_endpoint"] == "http://keycloak:80801/realms/test-realm/token"

    @patch("auth_server.providers.keycloak.requests.get")
    def test_bare_internal_origin_is_rewritten(self, mock_get):
        """An exact match with no path is a boundary too, not a near-miss."""
        config = self._full_openid_config(self.INTERNAL)
        config["issuer"] = self.INTERNAL
        provider = self._provider(mock_get, config)

        assert provider.authorization_server_metadata()["issuer"] == self.EXTERNAL

    @patch("auth_server.providers.keycloak.requests.get")
    def test_cached_configuration_is_not_mutated(self, mock_get):
        """The rewrite must not poison the lru_cache it reads from.

        _get_openid_configuration is lru_cached, so rewriting nested objects in
        place would corrupt the document for every later caller -- including the
        internal-URL consumers that must keep pointing at the cluster host.
        """
        provider = self._provider(mock_get, self._full_openid_config(self.INTERNAL))

        first = provider.authorization_server_metadata()
        cached = provider._get_openid_configuration()
        second = provider.authorization_server_metadata()

        assert cached["token_endpoint"].startswith(self.INTERNAL)
        assert cached["mtls_endpoint_aliases"]["token_endpoint"].startswith(self.INTERNAL)
        assert first == second

    @patch("auth_server.providers.keycloak.requests.get")
    def test_passthrough_when_internal_equals_external(self, mock_get):
        """No external URL configured: the document is returned as-is."""
        config = self._full_openid_config(self.INTERNAL)
        provider = self._provider(mock_get, config, external=self.INTERNAL)

        assert provider.authorization_server_metadata() == config


# =============================================================================
# KEYCLOAK_EXTERNAL_URL STARTUP DIAGNOSTIC
# =============================================================================


class TestKeycloakExternalUrlDiagnostic:
    """The factory must say so at startup when the advertised URL cannot work.

    The registry and the auth server publish KEYCLOAK_EXTERNAL_URL in their OAuth
    discovery documents. A value that resolves only inside the container network
    yields a well-formed metadata document that no external MCP client can act
    on: nothing fails server-side and the client reports a generic OAuth error,
    so without a startup line the misconfiguration is effectively invisible.
    """

    @pytest.fixture(autouse=True)
    def _reset_diagnostic_dedup(self):
        """The diagnostic logs each (url, configured) state once per process; clear
        that de-dup state before every test so cases stay independent."""
        from auth_server.providers import factory

        factory._reset_external_url_diagnostic()
        yield
        factory._reset_external_url_diagnostic()

    def _create(self, monkeypatch, external_url: str | None):
        """Build a Keycloak provider through the factory with a patched env."""
        from auth_server.providers import factory

        monkeypatch.setenv("KEYCLOAK_URL", "http://keycloak:8080")
        monkeypatch.setenv("KEYCLOAK_CLIENT_ID", "test-client")
        monkeypatch.setenv("KEYCLOAK_CLIENT_SECRET", "test-secret")
        if external_url is None:
            monkeypatch.delenv("KEYCLOAK_EXTERNAL_URL", raising=False)
        else:
            monkeypatch.setenv("KEYCLOAK_EXTERNAL_URL", external_url)
        return factory._create_keycloak_provider()

    def test_unset_external_url_warns(self, monkeypatch, caplog):
        """An unset variable silently advertises the internal URL, so warn."""
        with caplog.at_level(logging.WARNING, logger="auth_server.providers.factory"):
            provider = self._create(monkeypatch, None)

        assert provider.keycloak_external_url == "http://keycloak:8080"
        warnings = [r.message for r in caplog.records if r.levelno == logging.WARNING]
        assert any("KEYCLOAK_EXTERNAL_URL is not set" in m for m in warnings), warnings

    @pytest.mark.parametrize(
        "external_url",
        [
            "http://keycloak:8080",  # container DNS name, the reported case
            "http://10.0.1.5:8080",  # RFC 1918
            "http://192.168.1.10:8080",  # RFC 1918
            "http://169.254.10.10:8080",  # link-local
            "http://203.0.113.10:8080",  # RFC 5737 documentation range
        ],
    )
    def test_unroutable_external_url_warns(self, monkeypatch, caplog, external_url):
        """A set-but-unroutable value is the same outage with a different cause."""
        with caplog.at_level(logging.WARNING, logger="auth_server.providers.factory"):
            self._create(monkeypatch, external_url)

        warnings = [r.message for r in caplog.records if r.levelno == logging.WARNING]
        assert any("fail discovery" in m for m in warnings), warnings

    @pytest.mark.parametrize(
        "external_url",
        [
            "https://gateway.example.com",
            "https://gateway.example.com:8443",
            "http://93.184.216.34:8080",  # globally routable literal
        ],
    )
    def test_routable_external_url_is_silent(self, monkeypatch, caplog, external_url):
        """A correctly configured deployment must not emit a warning every boot."""
        with caplog.at_level(logging.WARNING, logger="auth_server.providers.factory"):
            self._create(monkeypatch, external_url)

        warnings = [r.message for r in caplog.records if r.levelno == logging.WARNING]
        assert not warnings, warnings

    @pytest.mark.parametrize(
        "external_url",
        ["http://localhost:8080", "http://127.0.0.1:8080"],
    )
    def test_loopback_external_url_is_not_a_warning(self, monkeypatch, caplog, external_url):
        """Loopback is the shipped default for local dev, so it must not warn.

        It is still worth an INFO line, because it does exclude remote clients.
        """
        with caplog.at_level(logging.INFO, logger="auth_server.providers.factory"):
            self._create(monkeypatch, external_url)

        assert not [r for r in caplog.records if r.levelno >= logging.WARNING]
        infos = [r.message for r in caplog.records if r.levelno == logging.INFO]
        assert any("only works for clients on this host" in m for m in infos), infos

    def test_diagnostic_logs_once_per_config(self, monkeypatch, caplog):
        """The factory runs per OAuth-discovery request; the warning must not repeat.

        `.well-known` endpoints are public and crawled anonymously, so a warning
        on every hit is noise. The same misconfiguration should log exactly once.
        """
        with caplog.at_level(logging.WARNING, logger="auth_server.providers.factory"):
            for _ in range(5):
                self._create(monkeypatch, "http://keycloak:8080")

        warnings = [
            r.message
            for r in caplog.records
            if r.levelno == logging.WARNING and "fail discovery" in r.message
        ]
        assert len(warnings) == 1, warnings

    def test_distinct_configs_each_log(self, monkeypatch, caplog):
        """A genuinely new (url, configured) state still logs, even after another."""
        with caplog.at_level(logging.WARNING, logger="auth_server.providers.factory"):
            self._create(monkeypatch, "http://keycloak:8080")  # single-label WARNING
            self._create(monkeypatch, "http://10.0.1.5:8080")  # private-address WARNING

        warnings = [
            r.message
            for r in caplog.records
            if r.levelno == logging.WARNING and "fail discovery" in r.message
        ]
        assert len(warnings) == 2, warnings
