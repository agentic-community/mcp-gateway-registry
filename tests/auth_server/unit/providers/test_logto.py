"""Unit tests for LogtoProvider."""

import time
from unittest.mock import MagicMock, patch

import jwt as pyjwt
import pytest

from auth_server.providers.logto import LogtoProvider

INTERNAL_URL = "http://idp-logto:3001"
EXTERNAL_URL = "https://auth.example.com"


def make_provider(**overrides) -> LogtoProvider:
    """Provider with the internal/external URL split a real deployment uses."""
    kwargs = {
        "logto_url": INTERNAL_URL,
        "client_id": "web-client",
        "client_secret": "web-secret",
        "m2m_client_id": "m2m-client",
        "m2m_client_secret": "m2m-secret",
        "m2m_resource": "https://api.example.com",
        "logto_external_url": EXTERNAL_URL,
    }
    kwargs.update(overrides)
    return LogtoProvider(**kwargs)


def mock_json_response(payload, status_code=200):
    response = MagicMock()
    response.json.return_value = payload
    response.status_code = status_code
    response.raise_for_status.return_value = None
    return response


# =============================================================================
# INITIALIZATION TESTS
# =============================================================================


class TestLogtoProviderInit:
    """Tests for LogtoProvider initialization."""

    def test_provider_initialization(self):
        """Endpoints derive from the configured URLs."""
        provider = make_provider()
        assert provider.logto_url == INTERNAL_URL
        assert provider.logto_external_url == EXTERNAL_URL
        assert provider.issuer == f"{EXTERNAL_URL}/oidc"
        # Server-to-server endpoints use the internal URL
        assert provider.token_url == f"{INTERNAL_URL}/oidc/token"
        assert provider.userinfo_url == f"{INTERNAL_URL}/oidc/me"
        assert provider.jwks_url == f"{INTERNAL_URL}/oidc/jwks"
        # Browser-facing endpoints use the external URL
        assert provider.auth_url == f"{EXTERNAL_URL}/oidc/auth"
        assert provider.logout_url == f"{EXTERNAL_URL}/oidc/session/end"

    def test_provider_initialization_strips_trailing_slash(self):
        """URLs are normalized without a trailing slash."""
        provider = LogtoProvider(
            logto_url="http://idp-logto:3001/",
            client_id="cid",
            client_secret="cs",
            m2m_client_id="m2m-cid",
            m2m_client_secret="m2m-cs",
            m2m_resource="https://api.example.com",
        )
        assert provider.logto_url == "http://idp-logto:3001"

    def test_provider_initialization_external_defaults_to_internal(self):
        """Without an external URL, browser endpoints use the internal one."""
        provider = LogtoProvider(
            logto_url=INTERNAL_URL,
            client_id="cid",
            client_secret="cs",
            m2m_client_id="m2m-cid",
            m2m_client_secret="m2m-cs",
            m2m_resource="https://api.example.com",
        )
        assert provider.logto_external_url == INTERNAL_URL
        assert provider.auth_url == f"{INTERNAL_URL}/oidc/auth"

    def test_provider_initialization_keeps_dedicated_m2m_credentials(self):
        """M2M credentials are the ones passed in — no web-secret substitution."""
        provider = make_provider()
        assert provider.m2m_client_id == "m2m-client"
        assert provider.m2m_client_secret == "m2m-secret"
        assert provider.m2m_resource == "https://api.example.com"
        assert provider.m2m_client_id != provider.client_id


# =============================================================================
# JWKS TESTS
# =============================================================================


class TestLogtoJWKS:
    """Tests for JWKS retrieval and caching."""

    @patch("auth_server.providers.logto.requests.get")
    def test_get_jwks_success(self, mock_get):
        """Test successful JWKS retrieval."""
        mock_get.return_value = mock_json_response({"keys": [{"kid": "key1", "kty": "RSA"}]})

        provider = make_provider()
        result = provider.get_jwks()

        assert result == {"keys": [{"kid": "key1", "kty": "RSA"}]}
        mock_get.assert_called_once_with(provider.jwks_url, timeout=10)

    @patch("auth_server.providers.logto.requests.get")
    def test_get_jwks_caching(self, mock_get):
        """Test JWKS cache returns cached data within TTL."""
        mock_get.return_value = mock_json_response({"keys": [{"kid": "key1", "kty": "RSA"}]})

        provider = make_provider()
        provider.get_jwks()
        provider.get_jwks()

        assert mock_get.call_count == 1

    @patch("auth_server.providers.logto.requests.get")
    def test_get_jwks_cache_expiration(self, mock_get):
        """Test JWKS cache expires after TTL."""
        mock_get.return_value = mock_json_response({"keys": [{"kid": "key1", "kty": "RSA"}]})

        provider = make_provider()
        provider.get_jwks()

        # Simulate TTL expiration by backdating the cache time
        provider._jwks_cache_time = provider._jwks_cache_time - 3601

        provider.get_jwks()

        assert mock_get.call_count == 2

    @patch("auth_server.providers.logto.requests.get")
    def test_get_jwks_failure(self, mock_get):
        """A transport failure raises ValueError."""
        mock_get.side_effect = Exception("connection refused")

        provider = make_provider()
        with pytest.raises(ValueError, match="Cannot retrieve JWKS"):
            provider.get_jwks()


# =============================================================================
# TOKEN VALIDATION TESTS
# =============================================================================


class TestLogtoTokenValidation:
    """Tests for token validation."""

    @patch("auth_server.providers.logto.jwt.decode")
    @patch("auth_server.providers.logto.jwt.get_unverified_header")
    def test_validate_token_success(self, mock_header, mock_decode):
        """A valid Logto JWT maps claims onto the provider result shape."""
        provider = make_provider(m2m_resource="https://api.example.com")

        now = int(time.time())
        payload = {
            "iss": f"{EXTERNAL_URL}/oidc",
            "aud": "web-client",
            "sub": "user-123",
            "username": "testuser",
            "email": "testuser@example.com",
            "roles": ["users", "admins"],
            "scope": "openid profile",
            "exp": now + 3600,
            "iat": now,
        }

        with patch.object(provider, "get_jwks", return_value={"keys": [{"kid": "k1"}]}):
            with patch("jwt.PyJWK") as mock_pyjwk:
                mock_pyjwk.return_value.key = MagicMock()
                mock_header.return_value = {"kid": "k1"}
                mock_decode.return_value = payload

                result = provider.validate_token("test-token")

        assert result["valid"] is True
        assert result["username"] == "testuser"
        assert result["email"] == "testuser@example.com"
        assert result["groups"] == ["users", "admins"]
        assert result["scopes"] == ["openid", "profile"]
        assert result["client_id"] == "web-client"
        assert result["method"] == "logto"

        # Signature algorithms restricted to what Logto signs with
        assert mock_decode.call_args.kwargs["algorithms"] == ["ES384", "RS256"]
        assert mock_decode.call_args.kwargs["issuer"] == f"{EXTERNAL_URL}/oidc"
        # Audiences include the M2M resource indicator
        assert "https://api.example.com" in mock_decode.call_args.kwargs["audience"]

    @patch("auth_server.providers.logto.jwt.get_unverified_header")
    def test_validate_token_expired(self, mock_header):
        """An expired token raises ValueError."""
        provider = make_provider()

        with patch.object(provider, "get_jwks", return_value={"keys": [{"kid": "k1"}]}):
            mock_header.return_value = {"kid": "k1"}
            with patch("jwt.PyJWK") as mock_pyjwk:
                mock_pyjwk.return_value.key = MagicMock()
                with patch(
                    "auth_server.providers.logto.jwt.decode",
                    side_effect=pyjwt.ExpiredSignatureError("Token has expired"),
                ):
                    with pytest.raises(ValueError, match="Token has expired"):
                        provider.validate_token("expired-token")

    @patch("auth_server.providers.logto.jwt.get_unverified_header")
    def test_validate_token_invalid(self, mock_header):
        """A structurally invalid token raises ValueError."""
        provider = make_provider()

        with patch.object(provider, "get_jwks", return_value={"keys": [{"kid": "k1"}]}):
            mock_header.return_value = {"kid": "k1"}
            with patch("jwt.PyJWK") as mock_pyjwk:
                mock_pyjwk.return_value.key = MagicMock()
                with patch(
                    "auth_server.providers.logto.jwt.decode",
                    side_effect=pyjwt.InvalidTokenError("bad signature"),
                ):
                    with pytest.raises(ValueError, match="Invalid token"):
                        provider.validate_token("bad-token")

    @patch("auth_server.providers.logto.jwt.get_unverified_header")
    def test_validate_token_no_kid(self, mock_header):
        """A token header without kid raises ValueError."""
        provider = make_provider()

        with patch.object(provider, "get_jwks", return_value={"keys": [{"kid": "k1"}]}):
            mock_header.return_value = {}
            with pytest.raises(ValueError, match="kid"):
                provider.validate_token("no-kid-token")

    @patch("auth_server.providers.logto.jwt.get_unverified_header")
    def test_validate_token_unknown_kid(self, mock_header):
        """A kid absent from the JWKS raises ValueError."""
        provider = make_provider()

        with patch.object(provider, "get_jwks", return_value={"keys": [{"kid": "other"}]}):
            mock_header.return_value = {"kid": "k1"}
            with pytest.raises(ValueError, match="No matching key found"):
                provider.validate_token("unknown-kid-token")

    def test_validate_token_roles_as_string(self):
        """A scalar roles claim is normalized to a list."""
        provider = make_provider()

        now = int(time.time())
        payload = {
            "iss": f"{EXTERNAL_URL}/oidc",
            "aud": "web-client",
            "sub": "user-123",
            "roles": "admins",
            "exp": now + 3600,
            "iat": now,
        }

        with patch.object(provider, "get_jwks", return_value={"keys": [{"kid": "k1"}]}):
            with patch("auth_server.providers.logto.jwt.get_unverified_header") as mock_header:
                with patch("auth_server.providers.logto.jwt.decode") as mock_decode:
                    with patch("jwt.PyJWK") as mock_pyjwk:
                        mock_pyjwk.return_value.key = MagicMock()
                        mock_header.return_value = {"kid": "k1"}
                        mock_decode.return_value = payload

                        result = provider.validate_token("test-token")

        assert result["groups"] == ["admins"]
        # username falls back to sub when username is absent
        assert result["username"] == "user-123"

    def test_validate_token_self_signed(self):
        """A self-signed auth-server token validates via the HS256 path."""
        provider = make_provider()
        secret = "test-secret-key-for-logto-provider-tests"

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
            secret,
            algorithm="HS256",
        )

        with patch("auth_server.providers.logto.SECRET_KEY", secret):
            result = provider.validate_token(token)

        assert result["method"] == "self_signed"
        assert result["username"] == "testuser"
        assert result["groups"] == ["admin"]
        assert result["scopes"] == ["read", "write"]

    def test_validate_token_self_signed_expired(self):
        """An expired self-signed token reports its expiry, not a JWKS error."""
        provider = make_provider()
        secret = "test-secret-key-for-logto-provider-tests"

        now = int(time.time())
        token = pyjwt.encode(
            {
                "iss": "mcp-auth-server",
                "aud": "mcp-registry",
                "sub": "testuser",
                "token_use": "access",
                "exp": now - 3600,
                "iat": now - 7200,
            },
            secret,
            algorithm="HS256",
        )

        # Through the public path: the self-signed pre-check must let the
        # expiry error propagate instead of falling through to the JWKS path
        # (which would surface a misleading "Cannot retrieve JWKS").
        with patch("auth_server.providers.logto.SECRET_KEY", secret):
            with pytest.raises(ValueError, match="^Token has expired$"):
                provider.validate_token(token)

    def test_validate_self_signed_missing_secret_key(self):
        """Without SECRET_KEY the self-signed path fails closed."""
        provider = make_provider()

        now = int(time.time())
        token = pyjwt.encode(
            {
                "iss": "mcp-auth-server",
                "aud": "mcp-registry",
                "sub": "testuser",
                "token_use": "access",
                "exp": now + 3600,
                "iat": now,
            },
            "some-secret",
            algorithm="HS256",
        )

        with patch("auth_server.providers.logto.SECRET_KEY", None):
            with pytest.raises(ValueError, match="SECRET_KEY is required"):
                provider._validate_self_signed_token(token)

    def test_validate_self_signed_wrong_token_use(self):
        """A self-signed token with the wrong token_use is rejected."""
        provider = make_provider()
        secret = "test-secret-key-for-logto-provider-tests"

        now = int(time.time())
        token = pyjwt.encode(
            {
                "iss": "mcp-auth-server",
                "aud": "mcp-registry",
                "sub": "testuser",
                "token_use": "id",
                "exp": now + 3600,
                "iat": now,
            },
            secret,
            algorithm="HS256",
        )

        with patch("auth_server.providers.logto.SECRET_KEY", secret):
            with pytest.raises(ValueError, match="Invalid token_use"):
                provider._validate_self_signed_token(token)


# =============================================================================
# ID TOKEN TESTS
# =============================================================================


class TestLogtoIdToken:
    """Tests for OIDC id_token verification wiring."""

    def test_validate_id_token_delegates_with_logto_parameters(self):
        """Verification runs against the Logto issuer/audience allowlists."""
        provider = make_provider(m2m_client_id="m2m-cid")

        with patch.object(provider, "_verify_id_token_with_jwks") as mock_verify:
            mock_verify.return_value = {"sub": "user-123"}
            result = provider.validate_id_token("id-token", expected_nonce="nonce-1")

        assert result == {"sub": "user-123"}
        args, kwargs = mock_verify.call_args
        assert args[0] == "id-token"
        assert args[1] == [f"{EXTERNAL_URL}/oidc"]
        assert args[2] == ["web-client", "m2m-cid"]
        assert kwargs["expected_nonce"] == "nonce-1"
        assert kwargs["algorithms"] == ["ES384", "RS256"]


# =============================================================================
# OAUTH2 FLOW TESTS
# =============================================================================


class TestLogtoOAuth2:
    """Tests for OAuth2 flows."""

    @patch("auth_server.providers.logto.requests.post")
    def test_exchange_code_for_token(self, mock_post):
        """Code exchange sends the correct grant parameters."""
        mock_post.return_value = mock_json_response({"access_token": "at"})

        provider = make_provider()
        result = provider.exchange_code_for_token("auth-code", "http://localhost/callback")

        assert result["access_token"] == "at"
        call_data = mock_post.call_args[1]["data"]
        assert call_data["grant_type"] == "authorization_code"
        assert call_data["code"] == "auth-code"
        assert call_data["client_id"] == "web-client"
        assert call_data["client_secret"] == "web-secret"
        assert call_data["redirect_uri"] == "http://localhost/callback"

    @patch("auth_server.providers.logto.requests.post")
    def test_refresh_token(self, mock_post):
        """Token refresh sends the correct grant parameters."""
        mock_post.return_value = mock_json_response({"access_token": "new-at"})

        provider = make_provider()
        result = provider.refresh_token("refresh-tok")

        assert result["access_token"] == "new-at"
        call_data = mock_post.call_args[1]["data"]
        assert call_data["grant_type"] == "refresh_token"
        assert call_data["refresh_token"] == "refresh-tok"

    @patch("auth_server.providers.logto.requests.get")
    def test_get_user_info(self, mock_get):
        """Userinfo is fetched with the bearer token."""
        mock_get.return_value = mock_json_response({"sub": "user-123", "username": "testuser"})

        provider = make_provider()
        result = provider.get_user_info("access-tok")

        assert result["sub"] == "user-123"
        assert mock_get.call_args[1]["headers"]["Authorization"] == "Bearer access-tok"

    def test_get_auth_url_default_scope(self):
        """Auth URL carries the first-party scope set and OAuth2 parameters."""
        provider = make_provider()
        url = provider.get_auth_url("http://localhost/callback", "state123")

        assert url.startswith(f"{EXTERNAL_URL}/oidc/auth?")
        assert "client_id=web-client" in url
        assert "response_type=code" in url
        assert "state=state123" in url
        assert "openid" in url
        assert "offline_access" in url
        assert "profile" in url
        assert "email" in url
        assert "roles" in url

    def test_get_auth_url_custom_scope(self):
        """An explicit scope overrides the default."""
        provider = make_provider()
        url = provider.get_auth_url("http://localhost/callback", "state123", scope="openid")

        assert "scope=openid" in url
        assert "profile" not in url

    def test_get_logout_url(self):
        """Logout URL targets the external end-session endpoint."""
        provider = make_provider()
        url = provider.get_logout_url("http://localhost")

        assert url.startswith(f"{EXTERNAL_URL}/oidc/session/end?")
        assert "client_id=web-client" in url
        assert "post_logout_redirect_uri" in url


# =============================================================================
# M2M TESTS
# =============================================================================


class TestLogtoM2M:
    """Tests for M2M client credentials flow."""

    @patch("auth_server.providers.logto.requests.post")
    def test_get_m2m_token(self, mock_post):
        """Client credentials request carries the Logto resource indicator."""
        mock_post.return_value = mock_json_response({"access_token": "m2m-token"})

        provider = make_provider(
            m2m_client_id="m2m-cid",
            m2m_client_secret="m2m-cs",
            m2m_resource="https://api.example.com",
        )
        result = provider.get_m2m_token()

        assert result["access_token"] == "m2m-token"
        call_data = mock_post.call_args[1]["data"]
        assert call_data["grant_type"] == "client_credentials"
        assert call_data["client_id"] == "m2m-cid"
        assert call_data["client_secret"] == "m2m-cs"
        assert call_data["resource"] == "https://api.example.com"

    @patch("auth_server.providers.logto.requests.post")
    def test_get_m2m_token_explicit_credentials(self, mock_post):
        """Explicit credentials and scope override the configured ones."""
        mock_post.return_value = mock_json_response({"access_token": "at"})

        provider = make_provider()
        provider.get_m2m_token(
            client_id="other-cid",
            client_secret="other-cs",
            scope="read",
        )

        call_data = mock_post.call_args[1]["data"]
        assert call_data["client_id"] == "other-cid"
        assert call_data["client_secret"] == "other-cs"
        assert call_data["scope"] == "read"

    def test_validate_m2m_token_delegates(self):
        """M2M tokens validate through the same path as user tokens."""
        provider = make_provider()

        with patch.object(
            provider, "validate_token", return_value={"valid": True}
        ) as mock_validate:
            result = provider.validate_m2m_token("m2m-token")

        assert result == {"valid": True}
        mock_validate.assert_called_once_with("m2m-token")


# =============================================================================
# DISCOVERY METADATA TESTS
# =============================================================================


class TestLogtoDiscoveryMetadata:
    """Tests for the OIDC discovery document exposure."""

    @patch("auth_server.providers.logto.requests.get")
    def test_metadata_same_url_returned_unchanged(self, mock_get):
        """Without an internal/external split, discovery is passed through."""
        config = {"issuer": f"{INTERNAL_URL}/oidc"}
        mock_get.return_value = mock_json_response(config)

        provider = make_provider(logto_external_url=INTERNAL_URL)
        assert provider.authorization_server_metadata() == config

    @patch("auth_server.providers.logto.requests.get")
    def test_metadata_rewrites_internal_urls(self, mock_get):
        """Browser-facing endpoints are rewritten onto the external URL."""
        config = {
            "issuer": f"{INTERNAL_URL}/oidc",
            "authorization_endpoint": f"{INTERNAL_URL}/oidc/auth",
            "token_endpoint": f"{INTERNAL_URL}/oidc/token",
            "jwks_uri": f"{INTERNAL_URL}/oidc/jwks",
            "end_session_endpoint": f"{INTERNAL_URL}/oidc/session/end",
            "response_modes_supported": ["query", "fragment"],
        }
        mock_get.return_value = mock_json_response(config)

        provider = make_provider()
        metadata = provider.authorization_server_metadata()

        assert metadata["issuer"] == f"{EXTERNAL_URL}/oidc"
        assert metadata["authorization_endpoint"] == f"{EXTERNAL_URL}/oidc/auth"
        assert metadata["token_endpoint"] == f"{EXTERNAL_URL}/oidc/token"
        assert metadata["jwks_uri"] == f"{EXTERNAL_URL}/oidc/jwks"
        assert metadata["end_session_endpoint"] == f"{EXTERNAL_URL}/oidc/session/end"
        # Non-URL fields pass through untouched
        assert metadata["response_modes_supported"] == ["query", "fragment"]

    @patch("auth_server.providers.logto.requests.get")
    def test_openid_configuration_failure(self, mock_get):
        """A failed discovery fetch raises ValueError."""
        import requests

        mock_get.side_effect = requests.RequestException("connection refused")

        provider = make_provider()
        with pytest.raises(ValueError, match="OpenID configuration retrieval failed"):
            provider._get_openid_configuration()


# =============================================================================
# PROVIDER INFO TESTS
# =============================================================================


class TestLogtoProviderInfo:
    """Tests for provider info."""

    @patch("auth_server.providers.logto.requests.get")
    def test_get_provider_info(self, mock_get):
        """Provider info reports the provider shape and health."""
        mock_get.return_value = mock_json_response({})

        provider = make_provider()
        info = provider.get_provider_info()

        assert info["provider_type"] == "logto"
        assert info["logto_url"] == INTERNAL_URL
        assert info["logto_external_url"] == EXTERNAL_URL
        assert info["client_id"] == "web-client"
        assert info["healthy"] is True
        assert info["endpoints"]["auth"] == provider.auth_url
        assert info["endpoints"]["token"] == provider.token_url

    @patch("auth_server.providers.logto.requests.get")
    def test_health_check_failure_is_false(self, mock_get):
        """An unreachable Logto reports unhealthy, not an exception."""
        mock_get.side_effect = Exception("connection refused")

        provider = make_provider()
        assert provider._check_logto_health() is False


# =============================================================================
# FACTORY INTEGRATION TESTS
# =============================================================================


class TestLogtoFactoryIntegration:
    """Tests for factory integration."""

    def test_factory_creates_logto_provider(self, monkeypatch):
        """Factory returns LogtoProvider when AUTH_PROVIDER=logto."""
        monkeypatch.setenv("LOGTO_URL", "http://idp-logto:3001")
        monkeypatch.setenv("LOGTO_CLIENT_ID", "test-cid")
        monkeypatch.setenv("LOGTO_CLIENT_SECRET", "test-cs")
        monkeypatch.setenv("LOGTO_M2M_CLIENT_ID", "test-m2m-cid")
        monkeypatch.setenv("LOGTO_M2M_CLIENT_SECRET", "test-m2m-cs")
        monkeypatch.setenv("LOGTO_M2M_RESOURCE", "https://api.example.com")

        import importlib

        import auth_server.providers.factory as factory_module

        importlib.reload(factory_module)

        provider = factory_module.get_auth_provider("logto")
        assert isinstance(provider, LogtoProvider)
        assert provider.logto_url == "http://idp-logto:3001"
        assert provider.m2m_client_id == "test-m2m-cid"

    def test_factory_requires_explicit_m2m_credentials(self, monkeypatch):
        """A deployment without M2M config refuses to start — no web-secret fallback."""
        monkeypatch.setenv("LOGTO_URL", "http://idp-logto:3001")
        monkeypatch.setenv("LOGTO_CLIENT_ID", "test-cid")
        monkeypatch.setenv("LOGTO_CLIENT_SECRET", "test-cs")
        monkeypatch.delenv("LOGTO_M2M_CLIENT_ID", raising=False)
        monkeypatch.delenv("LOGTO_M2M_CLIENT_SECRET", raising=False)
        monkeypatch.delenv("LOGTO_M2M_RESOURCE", raising=False)

        import importlib

        import auth_server.providers.factory as factory_module

        importlib.reload(factory_module)

        with pytest.raises(ValueError, match="LOGTO_M2M_CLIENT_ID.*LOGTO_M2M_CLIENT_SECRET"):
            factory_module.get_auth_provider("logto")
