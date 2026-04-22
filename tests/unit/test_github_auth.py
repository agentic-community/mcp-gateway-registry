"""Unit tests for GitHubAuthProvider."""

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import jwt
import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa


@pytest.fixture()
def rsa_private_key_pem() -> str:
    """Generate a fresh RSA private key in PEM format for testing."""
    private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    return private_key.private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.PKCS8,
        serialization.NoEncryption(),
    ).decode()


def _mock_app_settings(mock_settings, pem: str, pat: str = "") -> None:
    """Configure mock_settings for GitHub App auth tests."""
    mock_settings.github_pat = pat
    mock_settings.github_app_id = "12345"
    mock_settings.github_app_installation_id = "67890"
    mock_settings.github_app_private_key = pem
    mock_settings.github_extra_hosts = ""
    mock_settings.github_api_base_url = "https://api.github.com"


class TestDomainMatching:
    """Tests for _is_allowed_host and host allowlist logic."""

    def test_github_com_is_allowed(self):
        """Public github.com is allowed by default."""
        from registry.services.github_auth import GitHubAuthProvider

        provider = GitHubAuthProvider()
        assert provider._is_allowed_host("https://github.com/owner/repo") is True

    def test_raw_githubusercontent_is_allowed(self):
        """raw.githubusercontent.com is allowed by default."""
        from registry.services.github_auth import GitHubAuthProvider

        provider = GitHubAuthProvider()
        assert (
            provider._is_allowed_host("https://raw.githubusercontent.com/owner/repo/main/SKILL.md")
            is True
        )

    def test_non_github_host_is_not_allowed(self):
        """Non-GitHub hosts are rejected."""
        from registry.services.github_auth import GitHubAuthProvider

        provider = GitHubAuthProvider()
        assert provider._is_allowed_host("https://gitlab.com/owner/repo") is False

    def test_case_insensitive_matching(self):
        """Host matching is case-insensitive."""
        from registry.services.github_auth import GitHubAuthProvider

        provider = GitHubAuthProvider()
        assert provider._is_allowed_host("https://GitHub.COM/owner/repo") is True

    @patch("registry.services.github_auth.settings")
    def test_extra_hosts_from_config(self, mock_settings):
        """Extra hosts from config are included in allowlist."""
        mock_settings.github_pat = ""
        mock_settings.github_app_id = ""
        mock_settings.github_app_installation_id = ""
        mock_settings.github_app_private_key = ""
        mock_settings.github_extra_hosts = "github.mycompany.com,raw.github.mycompany.com"

        from registry.services.github_auth import GitHubAuthProvider

        provider = GitHubAuthProvider()
        assert provider._is_allowed_host("https://github.mycompany.com/org/repo") is True
        assert provider._is_allowed_host("https://raw.github.mycompany.com/org/repo/main/f") is True

    @patch("registry.services.github_auth.settings")
    def test_empty_extra_hosts(self, mock_settings):
        """Empty extra hosts config doesn't break anything."""
        mock_settings.github_pat = ""
        mock_settings.github_app_id = ""
        mock_settings.github_app_installation_id = ""
        mock_settings.github_app_private_key = ""
        mock_settings.github_extra_hosts = ""

        from registry.services.github_auth import GitHubAuthProvider

        provider = GitHubAuthProvider()
        assert provider._is_allowed_host("https://github.com/owner/repo") is True
        assert provider._is_allowed_host("https://example.com/foo") is False


class TestPATAuth:
    """Tests for Personal Access Token authentication."""

    @patch("registry.services.github_auth.settings")
    async def test_pat_returns_bearer_header(self, mock_settings):
        """PAT produces Authorization: Bearer header."""
        mock_settings.github_pat = "ghp_test_token_123"
        mock_settings.github_app_id = ""
        mock_settings.github_app_installation_id = ""
        mock_settings.github_app_private_key = ""
        mock_settings.github_extra_hosts = ""

        from registry.services.github_auth import GitHubAuthProvider

        provider = GitHubAuthProvider()
        headers = await provider.get_auth_headers("https://github.com/owner/repo")
        assert headers == {"Authorization": "Bearer ghp_test_token_123"}

    @patch("registry.services.github_auth.settings")
    async def test_no_credentials_returns_empty(self, mock_settings):
        """No credentials configured returns empty headers."""
        mock_settings.github_pat = ""
        mock_settings.github_app_id = ""
        mock_settings.github_app_installation_id = ""
        mock_settings.github_app_private_key = ""
        mock_settings.github_extra_hosts = ""

        from registry.services.github_auth import GitHubAuthProvider

        provider = GitHubAuthProvider()
        headers = await provider.get_auth_headers("https://github.com/owner/repo")
        assert headers == {}

    @patch("registry.services.github_auth.settings")
    async def test_non_github_host_returns_empty_even_with_pat(self, mock_settings):
        """PAT is not sent to non-GitHub hosts."""
        mock_settings.github_pat = "ghp_test_token_123"
        mock_settings.github_app_id = ""
        mock_settings.github_app_installation_id = ""
        mock_settings.github_app_private_key = ""
        mock_settings.github_extra_hosts = ""

        from registry.services.github_auth import GitHubAuthProvider

        provider = GitHubAuthProvider()
        headers = await provider.get_auth_headers("https://gitlab.com/owner/repo")
        assert headers == {}

    @patch("registry.services.github_auth.settings")
    async def test_pat_works_with_raw_githubusercontent(self, mock_settings):
        """PAT is sent to raw.githubusercontent.com."""
        mock_settings.github_pat = "ghp_test_token_123"
        mock_settings.github_app_id = ""
        mock_settings.github_app_installation_id = ""
        mock_settings.github_app_private_key = ""
        mock_settings.github_extra_hosts = ""

        from registry.services.github_auth import GitHubAuthProvider

        provider = GitHubAuthProvider()
        headers = await provider.get_auth_headers(
            "https://raw.githubusercontent.com/owner/repo/main/SKILL.md"
        )
        assert headers == {"Authorization": "Bearer ghp_test_token_123"}


class TestJWTCreation:
    """Tests for GitHub App JWT creation."""

    @patch("registry.services.github_auth.settings")
    def test_jwt_has_correct_claims(self, mock_settings, rsa_private_key_pem):
        """JWT contains iat, exp, iss claims."""
        _mock_app_settings(mock_settings, rsa_private_key_pem)

        from registry.services.github_auth import GitHubAuthProvider

        provider = GitHubAuthProvider()
        token = provider._create_jwt()

        claims = jwt.decode(token, options={"verify_signature": False})
        assert claims["iss"] == "12345"
        assert "iat" in claims
        assert "exp" in claims
        assert claims["exp"] - claims["iat"] <= 660

    @patch("registry.services.github_auth.settings")
    def test_jwt_uses_rs256(self, mock_settings, rsa_private_key_pem):
        """JWT is signed with RS256 algorithm."""
        _mock_app_settings(mock_settings, rsa_private_key_pem)

        from registry.services.github_auth import GitHubAuthProvider

        provider = GitHubAuthProvider()
        token = provider._create_jwt()

        header = jwt.get_unverified_header(token)
        assert header["alg"] == "RS256"


class TestTokenExchange:
    """Tests for GitHub App token exchange and caching."""

    @patch("registry.services.github_auth.settings")
    async def test_successful_token_exchange(self, mock_settings, rsa_private_key_pem):
        """Successful token exchange returns bearer header."""
        _mock_app_settings(mock_settings, rsa_private_key_pem, pat="ghp_fallback")

        from registry.services.github_auth import GitHubAuthProvider

        provider = GitHubAuthProvider()

        mock_response = MagicMock()
        mock_response.status_code = 201
        mock_response.json.return_value = {"token": "ghs_installation_token_abc"}

        with patch("registry.services.github_auth.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post.return_value = mock_response
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            headers = await provider.get_auth_headers("https://github.com/owner/repo")
            assert headers == {"Authorization": "Bearer ghs_installation_token_abc"}

    @patch("registry.services.github_auth.settings")
    async def test_cached_token_reused(self, mock_settings, rsa_private_key_pem):
        """Second call within TTL reuses cached token."""
        _mock_app_settings(mock_settings, rsa_private_key_pem)

        from registry.services.github_auth import GitHubAuthProvider

        provider = GitHubAuthProvider()

        mock_response = MagicMock()
        mock_response.status_code = 201
        mock_response.json.return_value = {"token": "ghs_cached_token"}

        with patch("registry.services.github_auth.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post.return_value = mock_response
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            headers1 = await provider.get_auth_headers("https://github.com/owner/repo")
            headers2 = await provider.get_auth_headers("https://github.com/owner/repo")

            assert headers1 == {"Authorization": "Bearer ghs_cached_token"}
            assert headers2 == {"Authorization": "Bearer ghs_cached_token"}
            assert mock_client.post.call_count == 1

    @patch("registry.services.github_auth.settings")
    async def test_exchange_failure_falls_back_to_pat(self, mock_settings, rsa_private_key_pem):
        """Failed token exchange falls back to PAT."""
        _mock_app_settings(mock_settings, rsa_private_key_pem, pat="ghp_fallback_token")

        from registry.services.github_auth import GitHubAuthProvider

        provider = GitHubAuthProvider()

        mock_response = MagicMock()
        mock_response.status_code = 401
        mock_response.text = "Bad credentials"

        with patch("registry.services.github_auth.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post.return_value = mock_response
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            headers = await provider.get_auth_headers("https://github.com/owner/repo")
            assert headers == {"Authorization": "Bearer ghp_fallback_token"}

    @patch("registry.services.github_auth.settings")
    async def test_exchange_failure_no_pat_returns_empty(self, mock_settings, rsa_private_key_pem):
        """Failed token exchange with no PAT returns empty headers."""
        _mock_app_settings(mock_settings, rsa_private_key_pem)

        from registry.services.github_auth import GitHubAuthProvider

        provider = GitHubAuthProvider()

        with patch("registry.services.github_auth.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post.side_effect = httpx.ConnectError("Connection refused")
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            headers = await provider.get_auth_headers("https://github.com/owner/repo")
            assert headers == {}

    @patch("registry.services.github_auth.settings")
    async def test_custom_api_base_url_used_in_exchange(self, mock_settings, rsa_private_key_pem):
        """Custom github_api_base_url is used for token exchange requests."""
        _mock_app_settings(mock_settings, rsa_private_key_pem)
        mock_settings.github_api_base_url = "https://github.mycompany.com/api/v3"

        from registry.services.github_auth import GitHubAuthProvider

        provider = GitHubAuthProvider()

        mock_response = MagicMock()
        mock_response.status_code = 201
        mock_response.json.return_value = {"token": "ghs_enterprise_token"}

        with patch("registry.services.github_auth.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post.return_value = mock_response
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            # Need to allow the enterprise host for auth headers
            mock_settings.github_extra_hosts = "github.mycompany.com"
            provider._allowed_hosts = provider._build_allowed_hosts()

            headers = await provider.get_auth_headers("https://github.mycompany.com/org/repo")
            assert headers == {"Authorization": "Bearer ghs_enterprise_token"}

            # Verify the POST was made to the custom API URL
            post_call = mock_client.post.call_args
            assert post_call.args[0] == (
                "https://github.mycompany.com/api/v3/app/installations/67890/access_tokens"
            )
