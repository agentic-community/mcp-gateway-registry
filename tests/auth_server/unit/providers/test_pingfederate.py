"""Unit tests for PingFederateProvider."""

import time
from unittest.mock import MagicMock, patch

import jwt as pyjwt
import pytest
import requests

from auth_server.providers.pingfederate import PingFederateProvider

MOCK_DISCOVERY_DOC = {
    "issuer": "https://pf.example.com",
    "authorization_endpoint": "https://pf.example.com/as/authorization.oauth2",
    "token_endpoint": "https://pf.example.com/as/token.oauth2",
    "userinfo_endpoint": "https://pf.example.com/idp/userinfo.openid",
    "jwks_uri": "https://pf.example.com/pf/JWKS",
    "end_session_endpoint": "https://pf.example.com/idp/startSLO.ping",
    "response_types_supported": ["code"],
    "grant_types_supported": ["authorization_code", "client_credentials", "refresh_token"],
    "subject_types_supported": ["public"],
    "id_token_signing_alg_values_supported": ["RS256"],
    "scopes_supported": ["openid", "email", "profile"],
}


def _create_provider(
    base_url: str = "https://pf.example.com:9031",
    client_id: str = "mcp-gateway",
    client_secret: str = "test-secret",
    m2m_client_id: str | None = None,
    m2m_client_secret: str | None = None,
    application_id_uri: str | None = None,
    groups_claim: str = "groups",
) -> PingFederateProvider:
    """Helper to create a provider instance for testing."""
    return PingFederateProvider(
        base_url=base_url,
        client_id=client_id,
        client_secret=client_secret,
        m2m_client_id=m2m_client_id,
        m2m_client_secret=m2m_client_secret,
        application_id_uri=application_id_uri,
        groups_claim=groups_claim,
    )


# =============================================================================
# INITIALIZATION TESTS
# =============================================================================


class TestPingFederateProviderInit:
    """Tests for PingFederateProvider initialization."""

    def test_init_with_valid_config(self):
        """Test provider initializes with valid configuration."""
        provider = _create_provider()
        assert provider.base_url == "https://pf.example.com:9031"
        assert provider.client_id == "mcp-gateway"
        assert provider.client_secret == "test-secret"
        assert provider.config_url == "https://pf.example.com:9031/.well-known/openid-configuration"

    def test_init_strips_trailing_slash(self):
        """Test base_url normalization strips trailing slash."""
        provider = _create_provider(base_url="https://pf.example.com:9031/")
        assert provider.base_url == "https://pf.example.com:9031"

    def test_init_m2m_defaults_to_web_client(self):
        """Test M2M credentials default to primary credentials."""
        provider = _create_provider()
        assert provider.m2m_client_id == "mcp-gateway"
        assert provider.m2m_client_secret == "test-secret"

    def test_init_m2m_separate_credentials(self):
        """Test M2M can use separate credentials."""
        provider = _create_provider(
            m2m_client_id="m2m-client",
            m2m_client_secret="m2m-secret",
        )
        assert provider.m2m_client_id == "m2m-client"
        assert provider.m2m_client_secret == "m2m-secret"

    def test_init_application_id_uri(self):
        """Test application_id_uri is stored correctly."""
        provider = _create_provider(application_id_uri="api://mcp-gateway")
        assert provider.application_id_uri == "api://mcp-gateway"

    def test_init_custom_groups_claim(self):
        """Test custom groups_claim is stored correctly."""
        provider = _create_provider(groups_claim="memberOf")
        assert provider.groups_claim == "memberOf"

    def test_init_default_groups_claim(self):
        """Test default groups_claim is 'groups'."""
        provider = _create_provider()
        assert provider.groups_claim == "groups"


# =============================================================================
# FACTORY TESTS
# =============================================================================


class TestPingFederateFactory:
    """Tests for the factory function."""

    @patch.dict(
        "os.environ",
        {
            "AUTH_PROVIDER": "pingfederate",
            "PINGFEDERATE_BASE_URL": "https://pf.example.com:9031",
            "PINGFEDERATE_CLIENT_ID": "mcp-gateway",
            "PINGFEDERATE_CLIENT_SECRET": "secret",
        },
    )
    def test_factory_creates_provider(self):
        """Test factory creates PingFederateProvider with valid env vars."""
        from auth_server.providers.factory import _create_pingfederate_provider

        provider = _create_pingfederate_provider()
        assert isinstance(provider, PingFederateProvider)
        assert provider.base_url == "https://pf.example.com:9031"

    @patch.dict(
        "os.environ",
        {"AUTH_PROVIDER": "pingfederate"},
        clear=True,
    )
    def test_init_missing_base_url_raises(self):
        """Test factory raises ValueError when PINGFEDERATE_BASE_URL is missing."""
        from auth_server.providers.factory import _create_pingfederate_provider

        with pytest.raises(ValueError, match="PINGFEDERATE_BASE_URL"):
            _create_pingfederate_provider()

    @patch.dict(
        "os.environ",
        {
            "AUTH_PROVIDER": "pingfederate",
            "PINGFEDERATE_BASE_URL": "https://pf.example.com:9031",
        },
        clear=True,
    )
    def test_init_missing_client_id_raises(self):
        """Test factory raises ValueError when PINGFEDERATE_CLIENT_ID is missing."""
        from auth_server.providers.factory import _create_pingfederate_provider

        with pytest.raises(ValueError, match="PINGFEDERATE_CLIENT_ID"):
            _create_pingfederate_provider()

    @patch.dict(
        "os.environ",
        {
            "AUTH_PROVIDER": "pingfederate",
            "PINGFEDERATE_BASE_URL": "https://pf.example.com:9031",
            "PINGFEDERATE_CLIENT_ID": "mcp-gateway",
            "PINGFEDERATE_CLIENT_SECRET": "secret",
            "PINGFEDERATE_GROUPS_CLAIM": "memberOf",
        },
    )
    def test_factory_passes_groups_claim(self):
        """Test factory passes PINGFEDERATE_GROUPS_CLAIM to provider."""
        from auth_server.providers.factory import _create_pingfederate_provider

        provider = _create_pingfederate_provider()
        assert provider.groups_claim == "memberOf"


# =============================================================================
# DISCOVERY TESTS
# =============================================================================


class TestPingFederateDiscovery:
    """Tests for OpenID Connect discovery."""

    @patch("auth_server.providers.pingfederate.requests.get")
    def test_discovery_fetches_and_caches(self, mock_get):
        """Test discovery document is fetched and cached."""
        mock_response = MagicMock()
        mock_response.json.return_value = MOCK_DISCOVERY_DOC
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        provider = _create_provider()
        config = provider._get_openid_configuration()

        assert config["issuer"] == "https://pf.example.com"
        assert config["token_endpoint"] == "https://pf.example.com/as/token.oauth2"
        mock_get.assert_called_once_with(provider.config_url, timeout=10)

    @patch("auth_server.providers.pingfederate.requests.get")
    def test_discovery_uses_lru_cache(self, mock_get):
        """Test subsequent calls use cached discovery doc."""
        mock_response = MagicMock()
        mock_response.json.return_value = MOCK_DISCOVERY_DOC
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        provider = _create_provider()
        provider._get_openid_configuration()
        provider._get_openid_configuration()

        # Should only fetch once due to @lru_cache
        assert mock_get.call_count == 1

    @patch("auth_server.providers.pingfederate.requests.get")
    def test_discovery_failure_raises(self, mock_get):
        """Test discovery fetch failure raises ValueError."""
        mock_get.side_effect = requests.ConnectionError("Connection refused")

        provider = _create_provider()
        with pytest.raises(ValueError, match="OpenID configuration retrieval failed"):
            provider._get_openid_configuration()

    @patch("auth_server.providers.pingfederate.requests.get")
    def test_endpoint_properties_resolve_from_discovery(self, mock_get):
        """Test endpoint properties use discovery document."""
        mock_response = MagicMock()
        mock_response.json.return_value = MOCK_DISCOVERY_DOC
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        provider = _create_provider()
        assert provider.token_url == "https://pf.example.com/as/token.oauth2"
        assert provider.jwks_url == "https://pf.example.com/pf/JWKS"
        assert provider.auth_url == "https://pf.example.com/as/authorization.oauth2"
        assert provider.issuer == "https://pf.example.com"

    @patch("auth_server.providers.pingfederate.requests.get")
    def test_logout_url_raises_when_endpoint_missing(self, mock_get):
        """Test logout_url raises when end_session_endpoint not in discovery."""
        discovery_no_logout = {**MOCK_DISCOVERY_DOC}
        del discovery_no_logout["end_session_endpoint"]

        mock_response = MagicMock()
        mock_response.json.return_value = discovery_no_logout
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        provider = _create_provider()
        with pytest.raises(ValueError, match="end_session_endpoint"):
            _ = provider.logout_url

    @patch("auth_server.providers.pingfederate.requests.get")
    def test_userinfo_url_raises_when_endpoint_missing(self, mock_get):
        """Test userinfo_url raises when userinfo_endpoint not in discovery."""
        discovery_no_userinfo = {**MOCK_DISCOVERY_DOC}
        del discovery_no_userinfo["userinfo_endpoint"]

        mock_response = MagicMock()
        mock_response.json.return_value = discovery_no_userinfo
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        provider = _create_provider()
        with pytest.raises(ValueError, match="userinfo_endpoint"):
            _ = provider.userinfo_url


# =============================================================================
# JWKS TESTS
# =============================================================================


class TestPingFederateJWKS:
    """Tests for JWKS retrieval and caching."""

    @patch("auth_server.providers.pingfederate.requests.get")
    def test_get_jwks_success(self, mock_get):
        """Test successful JWKS retrieval."""
        mock_jwks = {"keys": [{"kid": "key1", "kty": "RSA"}]}

        # First call returns discovery, second returns JWKS
        mock_discovery_response = MagicMock()
        mock_discovery_response.json.return_value = MOCK_DISCOVERY_DOC
        mock_discovery_response.raise_for_status.return_value = None

        mock_jwks_response = MagicMock()
        mock_jwks_response.json.return_value = mock_jwks
        mock_jwks_response.raise_for_status.return_value = None

        mock_get.side_effect = [mock_discovery_response, mock_jwks_response]

        provider = _create_provider()
        result = provider.get_jwks()

        assert result == mock_jwks

    @patch("auth_server.providers.pingfederate.requests.get")
    def test_get_jwks_caches_for_ttl(self, mock_get):
        """Test JWKS cache returns cached data within TTL."""
        mock_jwks = {"keys": [{"kid": "key1", "kty": "RSA"}]}

        mock_discovery_response = MagicMock()
        mock_discovery_response.json.return_value = MOCK_DISCOVERY_DOC
        mock_discovery_response.raise_for_status.return_value = None

        mock_jwks_response = MagicMock()
        mock_jwks_response.json.return_value = mock_jwks
        mock_jwks_response.raise_for_status.return_value = None

        mock_get.side_effect = [mock_discovery_response, mock_jwks_response]

        provider = _create_provider()
        provider.get_jwks()
        provider.get_jwks()

        # Discovery + JWKS = 2 calls total (JWKS cached on second call)
        assert mock_get.call_count == 2

    @patch("auth_server.providers.pingfederate.requests.get")
    def test_get_jwks_cache_expiration(self, mock_get):
        """Test JWKS cache expires after TTL."""
        mock_jwks = {"keys": [{"kid": "key1", "kty": "RSA"}]}

        mock_discovery_response = MagicMock()
        mock_discovery_response.json.return_value = MOCK_DISCOVERY_DOC
        mock_discovery_response.raise_for_status.return_value = None

        mock_jwks_response = MagicMock()
        mock_jwks_response.json.return_value = mock_jwks
        mock_jwks_response.raise_for_status.return_value = None

        mock_get.return_value = mock_jwks_response

        provider = _create_provider()
        # Pre-cache discovery to avoid side_effect complexity
        provider._get_openid_configuration.cache_clear()
        with patch.object(provider, "_get_openid_configuration", return_value=MOCK_DISCOVERY_DOC):
            provider.get_jwks()

            # Expire the cache
            provider._jwks_cache_time = time.time() - 3601
            provider.get_jwks()

        # Should fetch JWKS twice (cache expired)
        assert mock_get.call_count >= 2

    @patch("auth_server.providers.pingfederate.requests.get")
    def test_get_jwks_falls_back_to_stale_cache_on_error(self, mock_get):
        """Test JWKS falls back to stale cache when fetch fails."""
        mock_jwks = {"keys": [{"kid": "key1", "kty": "RSA"}]}

        provider = _create_provider()
        # Pre-populate stale cache
        provider._jwks_cache = mock_jwks
        provider._jwks_cache_time = time.time() - 7200  # 2 hours old (expired)

        # Mock discovery
        with patch.object(provider, "_get_openid_configuration", return_value=MOCK_DISCOVERY_DOC):
            # All JWKS fetches fail
            mock_get.side_effect = Exception("Network error")

            result = provider.get_jwks()
            assert result == mock_jwks  # Falls back to stale cache

    @patch("auth_server.providers.pingfederate.requests.get")
    def test_get_jwks_no_cache_raises(self, mock_get):
        """Test JWKS raises when no cache available and fetch fails."""
        provider = _create_provider()

        with patch.object(provider, "_get_openid_configuration", return_value=MOCK_DISCOVERY_DOC):
            mock_get.side_effect = Exception("Network error")

            with pytest.raises(ValueError, match="Cannot retrieve JWKS"):
                provider.get_jwks()


# =============================================================================
# TOKEN VALIDATION TESTS
# =============================================================================


class TestPingFederateTokenValidation:
    """Tests for token validation."""

    @patch("auth_server.providers.pingfederate.requests.get")
    def test_validate_token_happy_path(self, mock_get):
        """Test successful token validation."""
        from cryptography.hazmat.primitives.asymmetric import rsa

        # Generate a test RSA key pair
        private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        public_key = private_key.public_key()

        # Create a test JWT
        claims = {
            "sub": "testuser",
            "email": "test@example.com",
            "groups": ["admin", "users"],
            "scope": "openid email profile",
            "iss": "https://pf.example.com",
            "aud": "mcp-gateway",
            "exp": int(time.time()) + 3600,
            "iat": int(time.time()),
        }

        # Encode with PyJWT

        # Get public key numbers for JWKS
        pub_numbers = public_key.public_numbers()
        import base64

        def _int_to_base64url(n, length=None):
            b = n.to_bytes((n.bit_length() + 7) // 8, byteorder="big")
            return base64.urlsafe_b64encode(b).rstrip(b"=").decode("ascii")

        jwks = {
            "keys": [
                {
                    "kid": "test-key-1",
                    "kty": "RSA",
                    "use": "sig",
                    "alg": "RS256",
                    "n": _int_to_base64url(pub_numbers.n),
                    "e": _int_to_base64url(pub_numbers.e),
                }
            ]
        }

        token = pyjwt.encode(
            claims,
            private_key,
            algorithm="RS256",
            headers={"kid": "test-key-1"},
        )

        # Mock discovery
        mock_discovery_response = MagicMock()
        mock_discovery_response.json.return_value = MOCK_DISCOVERY_DOC
        mock_discovery_response.raise_for_status.return_value = None

        # Mock JWKS
        mock_jwks_response = MagicMock()
        mock_jwks_response.json.return_value = jwks
        mock_jwks_response.raise_for_status.return_value = None

        mock_get.side_effect = [mock_discovery_response, mock_jwks_response]

        provider = _create_provider()
        result = provider.validate_token(token)

        assert result["valid"] is True
        assert result["username"] == "testuser"
        assert result["email"] == "test@example.com"
        assert result["groups"] == ["admin", "users"]
        assert result["method"] == "pingfederate"
        assert "openid" in result["scopes"]

    @patch("auth_server.providers.pingfederate.SECRET_KEY", "test-secret-key")
    def test_validate_token_self_signed_short_circuit(self):
        """Test self-signed gateway tokens bypass PingFederate validation."""
        claims = {
            "sub": "testuser",
            "email": "test@example.com",
            "groups": ["admin"],
            "scope": "openid mcp-servers/all/read",
            "iss": "mcp-auth-server",
            "aud": "mcp-registry",
            "exp": int(time.time()) + 3600,
            "iat": int(time.time()),
            "token_use": "access",
            "client_id": "user-generated",
        }

        token = pyjwt.encode(claims, "test-secret-key", algorithm="HS256")

        provider = _create_provider()
        result = provider.validate_token(token)

        assert result["valid"] is True
        assert result["method"] == "self_signed"
        assert result["username"] == "testuser"
        assert result["groups"] == ["admin"]

    def test_validate_token_expired_raises(self):
        """Test expired token raises ValueError."""
        from cryptography.hazmat.primitives.asymmetric import rsa

        private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)

        claims = {
            "sub": "testuser",
            "iss": "https://pf.example.com",
            "aud": "mcp-gateway",
            "exp": int(time.time()) - 3600,  # Expired 1 hour ago
            "iat": int(time.time()) - 7200,
        }

        token = pyjwt.encode(claims, private_key, algorithm="RS256", headers={"kid": "test-key-1"})

        provider = _create_provider()

        with patch.object(provider, "get_jwks") as mock_jwks:
            pub_key = private_key.public_key()
            pub_numbers = pub_key.public_numbers()
            import base64

            def _int_to_base64url(n):
                b = n.to_bytes((n.bit_length() + 7) // 8, byteorder="big")
                return base64.urlsafe_b64encode(b).rstrip(b"=").decode("ascii")

            mock_jwks.return_value = {
                "keys": [
                    {
                        "kid": "test-key-1",
                        "kty": "RSA",
                        "use": "sig",
                        "alg": "RS256",
                        "n": _int_to_base64url(pub_numbers.n),
                        "e": _int_to_base64url(pub_numbers.e),
                    }
                ]
            }

            with patch.object(
                provider, "_get_openid_configuration", return_value=MOCK_DISCOVERY_DOC
            ):
                with pytest.raises(ValueError, match="Token has expired"):
                    provider.validate_token(token)

    @patch("auth_server.providers.pingfederate.requests.get")
    def test_validate_token_accepts_client_id_as_aud(self, mock_get):
        """Test token validation accepts client_id as valid audience."""
        import base64

        from cryptography.hazmat.primitives.asymmetric import rsa

        private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        public_key = private_key.public_key()
        pub_numbers = public_key.public_numbers()

        def _int_to_base64url(n):
            b = n.to_bytes((n.bit_length() + 7) // 8, byteorder="big")
            return base64.urlsafe_b64encode(b).rstrip(b"=").decode("ascii")

        jwks = {
            "keys": [
                {
                    "kid": "test-key-1",
                    "kty": "RSA",
                    "use": "sig",
                    "alg": "RS256",
                    "n": _int_to_base64url(pub_numbers.n),
                    "e": _int_to_base64url(pub_numbers.e),
                }
            ]
        }

        # Token with client_id as audience (PingFederate default behavior)
        claims = {
            "sub": "testuser",
            "iss": "https://pf.example.com",
            "aud": "mcp-gateway",  # client_id as audience
            "exp": int(time.time()) + 3600,
            "iat": int(time.time()),
            "scope": "openid",
        }

        token = pyjwt.encode(claims, private_key, algorithm="RS256", headers={"kid": "test-key-1"})

        mock_discovery_response = MagicMock()
        mock_discovery_response.json.return_value = MOCK_DISCOVERY_DOC
        mock_discovery_response.raise_for_status.return_value = None

        mock_jwks_response = MagicMock()
        mock_jwks_response.json.return_value = jwks
        mock_jwks_response.raise_for_status.return_value = None

        mock_get.side_effect = [mock_discovery_response, mock_jwks_response]

        provider = _create_provider(client_id="mcp-gateway")
        result = provider.validate_token(token)

        assert result["valid"] is True

    @patch("auth_server.providers.pingfederate.requests.get")
    def test_validate_token_accepts_application_id_uri_as_aud(self, mock_get):
        """Test token validation accepts application_id_uri as valid audience."""
        import base64

        from cryptography.hazmat.primitives.asymmetric import rsa

        private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        public_key = private_key.public_key()
        pub_numbers = public_key.public_numbers()

        def _int_to_base64url(n):
            b = n.to_bytes((n.bit_length() + 7) // 8, byteorder="big")
            return base64.urlsafe_b64encode(b).rstrip(b"=").decode("ascii")

        jwks = {
            "keys": [
                {
                    "kid": "test-key-1",
                    "kty": "RSA",
                    "use": "sig",
                    "alg": "RS256",
                    "n": _int_to_base64url(pub_numbers.n),
                    "e": _int_to_base64url(pub_numbers.e),
                }
            ]
        }

        # Token with application_id_uri as audience
        claims = {
            "sub": "testuser",
            "iss": "https://pf.example.com",
            "aud": "api://mcp-gateway",  # application_id_uri
            "exp": int(time.time()) + 3600,
            "iat": int(time.time()),
            "scope": "openid",
        }

        token = pyjwt.encode(claims, private_key, algorithm="RS256", headers={"kid": "test-key-1"})

        mock_discovery_response = MagicMock()
        mock_discovery_response.json.return_value = MOCK_DISCOVERY_DOC
        mock_discovery_response.raise_for_status.return_value = None

        mock_jwks_response = MagicMock()
        mock_jwks_response.json.return_value = jwks
        mock_jwks_response.raise_for_status.return_value = None

        mock_get.side_effect = [mock_discovery_response, mock_jwks_response]

        provider = _create_provider(application_id_uri="api://mcp-gateway")
        result = provider.validate_token(token)

        assert result["valid"] is True

    @patch("auth_server.providers.pingfederate.requests.get")
    def test_validate_token_empty_groups_returns_empty_list(self, mock_get):
        """Test token without groups claim returns empty list."""
        import base64

        from cryptography.hazmat.primitives.asymmetric import rsa

        private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        public_key = private_key.public_key()
        pub_numbers = public_key.public_numbers()

        def _int_to_base64url(n):
            b = n.to_bytes((n.bit_length() + 7) // 8, byteorder="big")
            return base64.urlsafe_b64encode(b).rstrip(b"=").decode("ascii")

        jwks = {
            "keys": [
                {
                    "kid": "test-key-1",
                    "kty": "RSA",
                    "use": "sig",
                    "alg": "RS256",
                    "n": _int_to_base64url(pub_numbers.n),
                    "e": _int_to_base64url(pub_numbers.e),
                }
            ]
        }

        claims = {
            "sub": "testuser",
            "iss": "https://pf.example.com",
            "aud": "mcp-gateway",
            "exp": int(time.time()) + 3600,
            "iat": int(time.time()),
            "scope": "openid",
            # No groups claim — simulates operator not configuring ATM
        }

        token = pyjwt.encode(claims, private_key, algorithm="RS256", headers={"kid": "test-key-1"})

        mock_discovery_response = MagicMock()
        mock_discovery_response.json.return_value = MOCK_DISCOVERY_DOC
        mock_discovery_response.raise_for_status.return_value = None

        mock_jwks_response = MagicMock()
        mock_jwks_response.json.return_value = jwks
        mock_jwks_response.raise_for_status.return_value = None

        mock_get.side_effect = [mock_discovery_response, mock_jwks_response]

        provider = _create_provider()
        result = provider.validate_token(token)

        assert result["valid"] is True
        assert result["groups"] == []

    @patch("auth_server.providers.pingfederate.requests.get")
    def test_validate_token_custom_groups_claim_name(self, mock_get):
        """Test token validation uses configured groups_claim name."""
        import base64

        from cryptography.hazmat.primitives.asymmetric import rsa

        private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        public_key = private_key.public_key()
        pub_numbers = public_key.public_numbers()

        def _int_to_base64url(n):
            b = n.to_bytes((n.bit_length() + 7) // 8, byteorder="big")
            return base64.urlsafe_b64encode(b).rstrip(b"=").decode("ascii")

        jwks = {
            "keys": [
                {
                    "kid": "test-key-1",
                    "kty": "RSA",
                    "use": "sig",
                    "alg": "RS256",
                    "n": _int_to_base64url(pub_numbers.n),
                    "e": _int_to_base64url(pub_numbers.e),
                }
            ]
        }

        claims = {
            "sub": "testuser",
            "iss": "https://pf.example.com",
            "aud": "mcp-gateway",
            "exp": int(time.time()) + 3600,
            "iat": int(time.time()),
            "scope": "openid",
            "memberOf": ["engineering", "platform"],  # Custom claim name
        }

        token = pyjwt.encode(claims, private_key, algorithm="RS256", headers={"kid": "test-key-1"})

        mock_discovery_response = MagicMock()
        mock_discovery_response.json.return_value = MOCK_DISCOVERY_DOC
        mock_discovery_response.raise_for_status.return_value = None

        mock_jwks_response = MagicMock()
        mock_jwks_response.json.return_value = jwks
        mock_jwks_response.raise_for_status.return_value = None

        mock_get.side_effect = [mock_discovery_response, mock_jwks_response]

        provider = _create_provider(groups_claim="memberOf")
        result = provider.validate_token(token)

        assert result["valid"] is True
        assert result["groups"] == ["engineering", "platform"]


# =============================================================================
# CODE EXCHANGE TESTS
# =============================================================================


class TestPingFederateCodeExchange:
    """Tests for authorization code exchange."""

    @patch("auth_server.providers.pingfederate.requests.post")
    @patch("auth_server.providers.pingfederate.requests.get")
    def test_exchange_code_for_token_happy(self, mock_get, mock_post):
        """Test successful code exchange."""
        mock_discovery_response = MagicMock()
        mock_discovery_response.json.return_value = MOCK_DISCOVERY_DOC
        mock_discovery_response.raise_for_status.return_value = None
        mock_get.return_value = mock_discovery_response

        token_response = {
            "access_token": "eyJ...",
            "id_token": "eyJ...",
            "token_type": "Bearer",
            "expires_in": 3600,
        }
        mock_post_response = MagicMock()
        mock_post_response.json.return_value = token_response
        mock_post_response.raise_for_status.return_value = None
        mock_post.return_value = mock_post_response

        provider = _create_provider()
        result = provider.exchange_code_for_token("auth-code-123", "http://localhost/callback")

        assert result["access_token"] == "eyJ..."
        mock_post.assert_called_once()
        call_kwargs = mock_post.call_args
        assert call_kwargs[1]["data"]["grant_type"] == "authorization_code"
        assert call_kwargs[1]["data"]["code"] == "auth-code-123"

    @patch("auth_server.providers.pingfederate.requests.post")
    @patch("auth_server.providers.pingfederate.requests.get")
    def test_exchange_code_failure_raises(self, mock_get, mock_post):
        """Test code exchange failure raises ValueError."""
        mock_discovery_response = MagicMock()
        mock_discovery_response.json.return_value = MOCK_DISCOVERY_DOC
        mock_discovery_response.raise_for_status.return_value = None
        mock_get.return_value = mock_discovery_response

        mock_post.side_effect = requests.ConnectionError("Connection refused")

        provider = _create_provider()
        with pytest.raises(ValueError, match="Token exchange failed"):
            provider.exchange_code_for_token("bad-code", "http://localhost/callback")


# =============================================================================
# M2M TOKEN TESTS
# =============================================================================


class TestPingFederateM2M:
    """Tests for machine-to-machine token operations."""

    @patch("auth_server.providers.pingfederate.requests.post")
    @patch("auth_server.providers.pingfederate.requests.get")
    def test_get_m2m_token_happy(self, mock_get, mock_post):
        """Test successful M2M token generation."""
        mock_discovery_response = MagicMock()
        mock_discovery_response.json.return_value = MOCK_DISCOVERY_DOC
        mock_discovery_response.raise_for_status.return_value = None
        mock_get.return_value = mock_discovery_response

        token_response = {
            "access_token": "m2m-token-xyz",
            "token_type": "Bearer",
            "expires_in": 7200,
        }
        mock_post_response = MagicMock()
        mock_post_response.json.return_value = token_response
        mock_post_response.raise_for_status.return_value = None
        mock_post.return_value = mock_post_response

        provider = _create_provider(m2m_client_id="m2m-app", m2m_client_secret="m2m-secret")
        result = provider.get_m2m_token()

        assert result["access_token"] == "m2m-token-xyz"
        call_kwargs = mock_post.call_args
        assert call_kwargs[1]["data"]["grant_type"] == "client_credentials"
        assert call_kwargs[1]["data"]["client_id"] == "m2m-app"
        assert call_kwargs[1]["data"]["client_secret"] == "m2m-secret"

    @patch("auth_server.providers.pingfederate.requests.post")
    @patch("auth_server.providers.pingfederate.requests.get")
    def test_get_m2m_token_uses_override_credentials(self, mock_get, mock_post):
        """Test M2M token generation with override credentials."""
        mock_discovery_response = MagicMock()
        mock_discovery_response.json.return_value = MOCK_DISCOVERY_DOC
        mock_discovery_response.raise_for_status.return_value = None
        mock_get.return_value = mock_discovery_response

        mock_post_response = MagicMock()
        mock_post_response.json.return_value = {"access_token": "t"}
        mock_post_response.raise_for_status.return_value = None
        mock_post.return_value = mock_post_response

        provider = _create_provider()
        provider.get_m2m_token(client_id="override-id", client_secret="override-secret")

        call_kwargs = mock_post.call_args
        assert call_kwargs[1]["data"]["client_id"] == "override-id"
        assert call_kwargs[1]["data"]["client_secret"] == "override-secret"

    def test_validate_m2m_token_delegates_to_validate_token(self):
        """Test validate_m2m_token delegates to validate_token."""
        provider = _create_provider()
        with patch.object(provider, "validate_token", return_value={"valid": True}) as mock:
            result = provider.validate_m2m_token("test-token")
            mock.assert_called_once_with("test-token")
            assert result["valid"] is True


# =============================================================================
# AUTH URL AND LOGOUT TESTS
# =============================================================================


class TestPingFederateAuthUrl:
    """Tests for authorization URL generation."""

    @patch("auth_server.providers.pingfederate.requests.get")
    def test_get_auth_url_default_scope(self, mock_get):
        """Test auth URL generation with default scope."""
        mock_response = MagicMock()
        mock_response.json.return_value = MOCK_DISCOVERY_DOC
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        provider = _create_provider()
        url = provider.get_auth_url("http://localhost/callback", "state123")

        assert "authorization.oauth2" in url
        assert "client_id=mcp-gateway" in url
        assert "response_type=code" in url
        assert "state=state123" in url
        assert "scope=openid+email+profile+groups" in url

    @patch("auth_server.providers.pingfederate.requests.get")
    def test_get_auth_url_custom_scope(self, mock_get):
        """Test auth URL generation with custom scope."""
        mock_response = MagicMock()
        mock_response.json.return_value = MOCK_DISCOVERY_DOC
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        provider = _create_provider()
        url = provider.get_auth_url("http://localhost/callback", "state", scope="openid custom")

        assert "scope=openid+custom" in url


class TestPingFederateLogout:
    """Tests for logout URL generation."""

    @patch("auth_server.providers.pingfederate.requests.get")
    def test_get_logout_url(self, mock_get):
        """Test logout URL generation."""
        mock_response = MagicMock()
        mock_response.json.return_value = MOCK_DISCOVERY_DOC
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        provider = _create_provider()
        url = provider.get_logout_url("http://localhost/")

        assert "startSLO.ping" in url
        assert "client_id=mcp-gateway" in url
        assert "post_logout_redirect_uri" in url


# =============================================================================
# METADATA TESTS
# =============================================================================


class TestPingFederateMetadata:
    """Tests for authorization server metadata."""

    @patch("auth_server.providers.pingfederate.requests.get")
    def test_authorization_server_metadata_returns_discovery_doc(self, mock_get):
        """Test metadata returns the cached discovery document."""
        mock_response = MagicMock()
        mock_response.json.return_value = MOCK_DISCOVERY_DOC
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        provider = _create_provider()
        metadata = provider.authorization_server_metadata()

        assert metadata == MOCK_DISCOVERY_DOC
        assert metadata["issuer"] == "https://pf.example.com"
        assert metadata["token_endpoint"] == "https://pf.example.com/as/token.oauth2"

    @patch("auth_server.providers.pingfederate.requests.get")
    def test_get_provider_info(self, mock_get):
        """Test get_provider_info returns expected structure."""
        mock_response = MagicMock()
        mock_response.json.return_value = MOCK_DISCOVERY_DOC
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        provider = _create_provider()
        info = provider.get_provider_info()

        assert info["provider_type"] == "pingfederate"
        assert info["base_url"] == "https://pf.example.com:9031"
        assert info["client_id"] == "mcp-gateway"
        assert "endpoints" in info
        assert info["issuer"] == "https://pf.example.com"
