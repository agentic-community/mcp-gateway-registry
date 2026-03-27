"""Unit tests for OktaProvider."""

import time
from unittest.mock import MagicMock, patch

import jwt as pyjwt
import pytest

from auth_server.providers.okta import OktaProvider

# =============================================================================
# INITIALIZATION TESTS
# =============================================================================


class TestOktaProviderInit:
    """Tests for OktaProvider initialization."""

    def test_provider_initialization(self):
        """Test provider initializes with valid config."""
        provider = OktaProvider(
            okta_domain="dev-123456.okta.com",
            client_id="test-client-id",
            client_secret="test-client-secret",
        )
        assert provider.okta_domain == "dev-123456.okta.com"
        assert provider.client_id == "test-client-id"
        assert provider.issuer == "https://dev-123456.okta.com"
        assert provider.token_url == "https://dev-123456.okta.com/oauth2/v1/token"

    def test_provider_initialization_removes_https(self):
        """Test domain normalization strips https:// prefix."""
        provider = OktaProvider(
            okta_domain="https://dev-123456.okta.com/",
            client_id="cid",
            client_secret="csecret",
        )
        assert provider.okta_domain == "dev-123456.okta.com"

    def test_provider_initialization_m2m_defaults(self):
        """Test M2M credentials default to primary credentials."""
        provider = OktaProvider(
            okta_domain="dev-123456.okta.com",
            client_id="web-client",
            client_secret="web-secret",
        )
        assert provider.m2m_client_id == "web-client"
        assert provider.m2m_client_secret == "web-secret"


# =============================================================================
# JWKS TESTS
# =============================================================================


class TestOktaJWKS:
    """Tests for JWKS retrieval and caching."""

    @patch("auth_server.providers.okta.requests.get")
    def test_get_jwks_success(self, mock_get):
        """Test successful JWKS retrieval."""
        mock_jwks = {"keys": [{"kid": "key1", "kty": "RSA"}]}
        mock_response = MagicMock()
        mock_response.json.return_value = mock_jwks
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        provider = OktaProvider("dev-123.okta.com", "cid", "cs")
        result = provider.get_jwks()

        assert result == mock_jwks
        mock_get.assert_called_once()

    @patch("auth_server.providers.okta.requests.get")
    def test_get_jwks_caching(self, mock_get):
        """Test JWKS cache returns cached data within TTL."""
        mock_jwks = {"keys": [{"kid": "key1", "kty": "RSA"}]}
        mock_response = MagicMock()
        mock_response.json.return_value = mock_jwks
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        provider = OktaProvider("dev-123.okta.com", "cid", "cs")
        provider.get_jwks()
        provider.get_jwks()

        # Should only fetch once due to caching
        assert mock_get.call_count == 1

    @patch("auth_server.providers.okta.requests.get")
    def test_get_jwks_cache_expiration(self, mock_get):
        """Test JWKS cache expires after TTL."""
        mock_jwks = {"keys": [{"kid": "key1", "kty": "RSA"}]}
        mock_response = MagicMock()
        mock_response.json.return_value = mock_jwks
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        provider = OktaProvider("dev-123.okta.com", "cid", "cs")

        # First call — populates cache
        provider.get_jwks()

        # Simulate TTL expiration by backdating the cache time
        provider._jwks_cache_time = provider._jwks_cache_time - 3601

        # Second call should re-fetch
        provider.get_jwks()

        assert mock_get.call_count == 2


# =============================================================================
# TOKEN VALIDATION TESTS
# =============================================================================


class TestOktaTokenValidation:
    """Tests for token validation."""

    @patch("auth_server.providers.okta.requests.get")
    def test_validate_token_success(self, mock_get):
        """Test successful token validation with correct claim extraction."""
        mock_jwks = {"keys": [{"kid": "test-key-id-1", "kty": "RSA"}]}
        mock_response = MagicMock()
        mock_response.json.return_value = mock_jwks
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        provider = OktaProvider("dev-123.okta.com", "test-client", "cs")

        now = int(time.time())
        payload = {
            "iss": "https://dev-123.okta.com",
            "aud": "test-client",
            "sub": "user-123",
            "preferred_username": "testuser",
            "email": "testuser@example.com",
            "groups": ["users", "admins"],
            "scp": ["openid", "profile"],
            "cid": "test-client",
            "exp": now + 3600,
            "iat": now,
        }

        with patch("auth_server.providers.okta.jwt.get_unverified_header") as mock_header:
            with patch("auth_server.providers.okta.jwt.decode") as mock_decode:
                mock_header.return_value = {"kid": "test-key-id-1"}
                mock_decode.return_value = payload

                with patch("jwt.PyJWK") as mock_pyjwk:
                    mock_pyjwk.return_value.key = MagicMock()

                    result = provider.validate_token("test-token")

                    assert result["valid"] is True
                    assert result["username"] == "user-123"
                    assert result["email"] == "testuser@example.com"
                    assert "users" in result["groups"]
                    assert "admins" in result["groups"]
                    assert result["scopes"] == ["openid", "profile"]
                    assert result["client_id"] == "test-client"
                    assert result["method"] == "okta"

    def test_validate_token_expired(self):
        """Test expired token raises ValueError."""
        provider = OktaProvider("dev-123.okta.com", "cid", "cs")

        with patch.object(provider, "get_jwks", return_value={"keys": [{"kid": "k1"}]}):
            with patch("auth_server.providers.okta.jwt.get_unverified_header") as mock_header:
                mock_header.return_value = {"kid": "k1"}
                with patch("jwt.PyJWK") as mock_pyjwk:
                    mock_pyjwk.return_value.key = MagicMock()
                    with patch("auth_server.providers.okta.jwt.decode") as mock_decode:
                        from jwt.exceptions import ExpiredSignatureError

                        mock_decode.side_effect = ExpiredSignatureError("Token has expired")

                        with pytest.raises(ValueError, match="Token has expired"):
                            provider.validate_token("expired-token")

    def test_validate_token_no_kid(self):
        """Test missing kid header raises ValueError."""
        provider = OktaProvider("dev-123.okta.com", "cid", "cs")

        with patch.object(provider, "get_jwks", return_value={"keys": []}):
            with patch("auth_server.providers.okta.jwt.get_unverified_header") as mock_header:
                mock_header.return_value = {}  # No kid

                with pytest.raises(ValueError, match="kid"):
                    provider.validate_token("no-kid-token")

    def test_validate_token_self_signed(self):
        """Test self-signed token path delegates correctly."""
        provider = OktaProvider("dev-123.okta.com", "cid", "cs")

        now = int(time.time())
        token = pyjwt.encode(
            {
                "iss": "mcp-auth-server",
                "aud": "mcp-registry",
                "sub": "testuser",
                "email": "test@example.com",
                "groups": ["admin"],
                "scope": "read write",
                "token_use": "access",
                "exp": now + 3600,
                "iat": now,
            },
            "development-secret-key",
            algorithm="HS256",
        )

        result = provider.validate_token(token)
        assert result["method"] == "self_signed"
        assert result["username"] == "testuser"
        assert result["groups"] == ["admin"]
        assert result["scopes"] == ["read", "write"]


# =============================================================================
# OAUTH2 FLOW TESTS
# =============================================================================


class TestOktaOAuth2:
    """Tests for OAuth2 flows."""

    @patch("auth_server.providers.okta.requests.post")
    def test_exchange_code_for_token(self, mock_post):
        """Test OAuth2 code exchange sends correct parameters."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"access_token": "at", "id_token": "it"}
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        provider = OktaProvider("dev-123.okta.com", "cid", "cs")
        result = provider.exchange_code_for_token("auth-code", "http://localhost/callback")

        assert result["access_token"] == "at"
        call_data = mock_post.call_args[1]["data"]
        assert call_data["grant_type"] == "authorization_code"
        assert call_data["code"] == "auth-code"
        assert call_data["client_id"] == "cid"

    def test_get_auth_url(self):
        """Test auth URL generation with correct parameters and default scope."""
        provider = OktaProvider("dev-123.okta.com", "cid", "cs")
        url = provider.get_auth_url("http://localhost/callback", "state123")

        assert "https://dev-123.okta.com/oauth2/v1/authorize" in url
        assert "client_id=cid" in url
        assert "response_type=code" in url
        assert "state=state123" in url
        assert "openid" in url
        assert "email" in url
        assert "profile" in url
        assert "groups" in url

    def test_get_logout_url(self):
        """Test logout URL generation with correct parameters."""
        provider = OktaProvider("dev-123.okta.com", "cid", "cs")
        url = provider.get_logout_url("http://localhost")

        assert "https://dev-123.okta.com/oauth2/v1/logout" in url
        assert "client_id=cid" in url
        assert "post_logout_redirect_uri" in url

    @patch("auth_server.providers.okta.requests.post")
    def test_refresh_token(self, mock_post):
        """Test token refresh sends correct parameters."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"access_token": "new-at"}
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        provider = OktaProvider("dev-123.okta.com", "cid", "cs")
        result = provider.refresh_token("refresh-tok")

        assert result["access_token"] == "new-at"
        call_data = mock_post.call_args[1]["data"]
        assert call_data["grant_type"] == "refresh_token"
        assert call_data["refresh_token"] == "refresh-tok"
        assert call_data["client_id"] == "cid"


# =============================================================================
# M2M TESTS
# =============================================================================


class TestOktaM2M:
    """Tests for M2M client credentials flow."""

    @patch("auth_server.providers.okta.requests.post")
    def test_get_m2m_token(self, mock_post):
        """Test client credentials flow with M2M credentials."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"access_token": "m2m-token"}
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        provider = OktaProvider(
            "dev-123.okta.com",
            "cid",
            "cs",
            m2m_client_id="m2m-cid",
            m2m_client_secret="m2m-cs",
        )
        result = provider.get_m2m_token()

        assert result["access_token"] == "m2m-token"
        call_data = mock_post.call_args[1]["data"]
        assert call_data["grant_type"] == "client_credentials"
        assert call_data["client_id"] == "m2m-cid"
        assert call_data["client_secret"] == "m2m-cs"


# =============================================================================
# PROVIDER INFO TESTS
# =============================================================================


class TestOktaProviderInfo:
    """Tests for provider info."""

    def test_get_provider_info(self):
        """Test provider info returns correct structure."""
        provider = OktaProvider("dev-123.okta.com", "cid", "cs")
        info = provider.get_provider_info()

        assert info["provider_type"] == "okta"
        assert info["okta_domain"] == "dev-123.okta.com"
        assert info["client_id"] == "cid"
        assert info["issuer"] == "https://dev-123.okta.com"
        assert "endpoints" in info
        assert "auth" in info["endpoints"]
        assert "token" in info["endpoints"]
        assert "jwks" in info["endpoints"]


# =============================================================================
# FACTORY INTEGRATION TESTS
# =============================================================================


class TestOktaFactoryIntegration:
    """Tests for factory integration."""

    def test_factory_creates_okta_provider(self, monkeypatch):
        """Factory returns OktaProvider when AUTH_PROVIDER=okta."""
        monkeypatch.setenv("OKTA_DOMAIN", "dev-123.okta.com")
        monkeypatch.setenv("OKTA_CLIENT_ID", "test-cid")
        monkeypatch.setenv("OKTA_CLIENT_SECRET", "test-cs")

        import importlib

        import auth_server.providers.factory as factory_module

        importlib.reload(factory_module)

        provider = factory_module.get_auth_provider("okta")
        assert isinstance(provider, OktaProvider)
        assert provider.okta_domain == "dev-123.okta.com"
