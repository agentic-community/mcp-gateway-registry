"""
Unit tests for auth_server/providers/cognito.py

Currently scoped to the RFC 8414 metadata exposure added in issue #989. The
broader Cognito provider has historically been exercised via integration
tests; we add unit coverage here as new surface lands.

Note on URL assertions: we parse URLs with `urllib.parse.urlsplit` and compare
the hostname exactly rather than using substring `in` or `startswith` checks,
because CodeQL flags substring URL checks under py/incomplete-url-substring-sanitization.
The parsed-host comparison is also a stricter test.
"""

from unittest.mock import MagicMock, patch
from urllib.parse import urlsplit

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.auth]


def _hostname_of(url: str) -> str:
    """Return the host part of a URL, validating scheme and host explicitly."""
    parsed = urlsplit(url)
    assert parsed.scheme == "https", f"Expected https URL, got: {url}"
    assert parsed.hostname, f"URL has no hostname: {url}"
    return parsed.hostname


class TestCognitoAuthorizationServerMetadata:
    """Tests for RFC 8414 metadata exposure via authorization_server_metadata()."""

    def test_endpoints_rehomed_onto_cognito_domain(self):
        """authorization/token/userinfo/logout live on the cognito-domain host;
        only jwks_uri and issuer stay on cognito-idp.{region}.amazonaws.com."""
        from auth_server.providers.cognito import CognitoProvider

        provider = CognitoProvider(
            user_pool_id="us-east-1_abc123",
            client_id="c",
            client_secret="s",
            region="us-east-1",
            domain="my-app",
        )

        metadata = provider.authorization_server_metadata()

        assert metadata["issuer"] == "https://cognito-idp.us-east-1.amazonaws.com/us-east-1_abc123"
        cognito_domain_host = "my-app.auth.us-east-1.amazoncognito.com"
        assert _hostname_of(metadata["authorization_endpoint"]) == cognito_domain_host
        assert _hostname_of(metadata["token_endpoint"]) == cognito_domain_host
        assert _hostname_of(metadata["userinfo_endpoint"]) == cognito_domain_host
        assert _hostname_of(metadata["end_session_endpoint"]) == cognito_domain_host
        # JWKS stays on the cognito-idp host
        assert (
            metadata["jwks_uri"]
            == "https://cognito-idp.us-east-1.amazonaws.com/us-east-1_abc123/.well-known/jwks.json"
        )

    def test_default_domain_when_not_provided(self):
        """When no `domain` is configured, the cognito-domain host is derived
        from the user pool ID (Cognito's auto-generated domain)."""
        from auth_server.providers.cognito import CognitoProvider

        provider = CognitoProvider(
            user_pool_id="us-west-2_xyz789",
            client_id="c",
            client_secret="s",
            region="us-west-2",
        )

        metadata = provider.authorization_server_metadata()

        # Auto-derived domain strips the underscore from the user pool id.
        # Parse the URL and compare hostname exactly, not as a substring,
        # so the test catches host-spoofing variants and so CodeQL's
        # py/incomplete-url-substring-sanitization rule is satisfied.
        assert (
            _hostname_of(metadata["token_endpoint"])
            == "us-west-2xyz789.auth.us-west-2.amazoncognito.com"
        )
        assert metadata["issuer"] == "https://cognito-idp.us-west-2.amazonaws.com/us-west-2_xyz789"

    def test_authorization_server_issuer_returns_cognito_idp_url(self):
        from auth_server.providers.cognito import CognitoProvider

        provider = CognitoProvider(
            user_pool_id="us-east-1_abc123",
            client_id="c",
            client_secret="s",
            region="us-east-1",
        )

        assert (
            provider.authorization_server_issuer()
            == "https://cognito-idp.us-east-1.amazonaws.com/us-east-1_abc123"
        )

    def test_includes_pkce_support(self):
        """PKCE is mandatory in OAuth 2.1; metadata must advertise S256."""
        from auth_server.providers.cognito import CognitoProvider

        provider = CognitoProvider(
            user_pool_id="us-east-1_abc123",
            client_id="c",
            client_secret="s",
            region="us-east-1",
        )

        metadata = provider.authorization_server_metadata()

        assert "S256" in metadata["code_challenge_methods_supported"]


class TestCognitoValidateToken:
    """Token validation: id tokens are audience-bound; access tokens are not
    (no `aud`) and must be checked against the accepted client_id allowlist.

    Regression guard for the bug where access tokens were rejected with
    'Token is missing the "aud" claim' because verify_aud was always True.
    """

    @staticmethod
    def _provider(ide_client_id=None):
        from auth_server.providers.cognito import CognitoProvider

        return CognitoProvider(
            user_pool_id="us-east-1_abc123",
            client_id="web-client",
            client_secret="s",
            region="us-east-1",
            ide_oauth_client_id=ide_client_id,
        )

    def _run_validate(self, provider, decoded_claims):
        """Run validate_token offline.

        Bypasses JWKS/signature handling by mocking get_jwks, the unverified
        header, PyJWK, and jwt.decode. jwt.decode is called twice (unverified to
        read token_use, then verified); both return the same claims here, which
        is fine for exercising the token_use / client_id allowlist branch.
        """
        with (
            patch.object(provider, "get_jwks", return_value={"keys": [{"kid": "k1"}]}),
            patch(
                "auth_server.providers.cognito.jwt.get_unverified_header",
                return_value={"kid": "k1"},
            ),
            patch("jwt.PyJWK", return_value=MagicMock(key="key")),
            patch(
                "auth_server.providers.cognito.jwt.decode",
                return_value=decoded_claims,
            ),
        ):
            return provider.validate_token("fake.jwt.token")

    def test_access_token_accepted_for_ide_client(self):
        """An access token (no aud) from the IDE public client is accepted when
        that client id is in the allowlist."""
        provider = self._provider(ide_client_id="ide-public-client")
        result = self._run_validate(
            provider,
            {
                "token_use": "access",
                "client_id": "ide-public-client",
                "sub": "user-1",
                "cognito:groups": ["registry-admins"],
                "scope": "openid email profile",
            },
        )

        assert result["valid"] is True
        assert result["groups"] == ["registry-admins"]
        assert result["client_id"] == "ide-public-client"

    def test_access_token_rejected_for_unknown_client(self):
        """An access token from a client id NOT in the allowlist is rejected."""
        provider = self._provider(ide_client_id="ide-public-client")
        with pytest.raises(ValueError, match="not in the accepted client"):
            self._run_validate(
                provider,
                {"token_use": "access", "client_id": "some-other-client", "sub": "user-1"},
            )

    def test_access_token_accepted_for_web_client(self):
        """The configured web client_id is always accepted (no IDE client set)."""
        provider = self._provider()
        result = self._run_validate(
            provider,
            {
                "token_use": "access",
                "client_id": "web-client",
                "sub": "user-1",
                "cognito:groups": [],
            },
        )

        assert result["valid"] is True
