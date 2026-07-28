"""Unit tests for Keycloak access-token audience validation.

These tests exercise the real ``KeycloakProvider.validate_token`` decode path
with a genuine RSA keypair and PyJWT's ``verify_aud=True``, so the audience
allowlist is enforced by the cryptography/decoder rather than by a mock.

Regression coverage for the same-realm cross-client confused-deputy finding:
Keycloak's default ``account`` audience is present on EVERY realm user token
regardless of which client requested it. A token whose only audience is
``account`` (i.e. minted for some other client in the realm) must be REJECTED,
while tokens that carry an audience naming THIS gateway (``client_id``,
``m2m_client_id``, or ``mcp-gateway``) must still validate.
"""

import json
import logging
import time
from unittest.mock import patch

import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric import rsa
from jwt.algorithms import RSAAlgorithm

logger = logging.getLogger(__name__)

pytestmark = [pytest.mark.unit, pytest.mark.auth]


# =============================================================================
# Test helpers: real RSA keypair + JWKS + signed access token
# =============================================================================


def _build_keypair(kid: str = "test-kid") -> tuple[rsa.RSAPrivateKey, dict]:
    """Generate an RSA keypair and the matching single-key JWKS document."""
    private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    public_jwk = json.loads(RSAAlgorithm.to_jwk(private_key.public_key()))
    public_jwk["kid"] = kid
    public_jwk["alg"] = "RS256"
    public_jwk["use"] = "sig"
    return private_key, {"keys": [public_jwk]}


def _sign_token(
    private_key: rsa.RSAPrivateKey,
    claims: dict,
    kid: str = "test-kid",
) -> str:
    """Sign an access token with the given RSA private key (RS256)."""
    return jwt.encode(claims, private_key, algorithm="RS256", headers={"kid": kid})


def _base_claims(issuer: str, audience: str | list[str]) -> dict:
    now = int(time.time())
    return {
        "iss": issuer,
        "aud": audience,
        "sub": "user-123",
        "preferred_username": "alice",
        "email": "alice@example.com",
        "groups": ["mcp-registry-admin"],
        "scope": "openid profile email",
        "azp": "some-other-client",
        "iat": now,
        "exp": now + 3600,
    }


def _make_provider():
    from providers.keycloak import KeycloakProvider

    return KeycloakProvider(
        keycloak_url="http://keycloak:8080",
        realm="test-realm",
        client_id="gateway-web",
        client_secret="secret",  # noqa: S106 - test fixture, not a real secret
        m2m_client_id="gateway-m2m",
        m2m_client_secret="m2m-secret",  # noqa: S106 - test fixture, not a real secret
        keycloak_external_url="https://keycloak.example.com",
    )


# =============================================================================
# Rejection: a token audienced ONLY to "account" must be rejected
# =============================================================================


class TestAccountAudienceRejected:
    """The default ``account`` audience alone is not a gateway audience."""

    def test_account_only_audience_rejected(self):
        """A user token minted for another client (aud=account) is rejected."""
        provider = _make_provider()
        private_key, jwks = _build_keypair()
        token = _sign_token(private_key, _base_claims(provider.realm_url, "account"))

        with patch.object(provider, "get_jwks", return_value=jwks):
            with pytest.raises(ValueError, match="[Ii]nvalid|[Aa]udience"):
                provider.validate_token(token)

    def test_unrelated_client_audience_rejected(self):
        """An audience naming a different client entirely is rejected."""
        provider = _make_provider()
        private_key, jwks = _build_keypair()
        token = _sign_token(private_key, _base_claims(provider.realm_url, "some-unrelated-client"))

        with patch.object(provider, "get_jwks", return_value=jwks):
            with pytest.raises(ValueError):
                provider.validate_token(token)

    def test_account_in_list_without_gateway_audience_rejected(self):
        """aud as a list containing only account (+ noise) is still rejected."""
        provider = _make_provider()
        private_key, jwks = _build_keypair()
        token = _sign_token(
            private_key,
            _base_claims(provider.realm_url, ["account", "realm-management"]),
        )

        with patch.object(provider, "get_jwks", return_value=jwks):
            with pytest.raises(ValueError):
                provider.validate_token(token)


# =============================================================================
# Acceptance: tokens carrying a legitimate gateway audience still validate
# =============================================================================


class TestGatewayAudienceAccepted:
    """Audiences that name this gateway are accepted."""

    def test_mcp_gateway_audience_accepted(self):
        """The realm audience-mapper attaches aud=mcp-gateway to every token."""
        provider = _make_provider()
        private_key, jwks = _build_keypair()
        token = _sign_token(private_key, _base_claims(provider.realm_url, "mcp-gateway"))

        with patch.object(provider, "get_jwks", return_value=jwks):
            result = provider.validate_token(token)

        assert result["valid"] is True
        assert result["username"] == "alice"

    def test_client_id_audience_accepted(self):
        """A token audienced to the gateway's own web client is accepted."""
        provider = _make_provider()
        private_key, jwks = _build_keypair()
        token = _sign_token(private_key, _base_claims(provider.realm_url, provider.client_id))

        with patch.object(provider, "get_jwks", return_value=jwks):
            result = provider.validate_token(token)

        assert result["valid"] is True

    def test_m2m_client_id_audience_accepted(self):
        """A token audienced to the gateway's M2M client is accepted."""
        provider = _make_provider()
        private_key, jwks = _build_keypair()
        token = _sign_token(private_key, _base_claims(provider.realm_url, provider.m2m_client_id))

        with patch.object(provider, "get_jwks", return_value=jwks):
            result = provider.validate_token(token)

        assert result["valid"] is True

    def test_account_plus_gateway_audience_accepted(self):
        """A real DCR token carries both account and mcp-gateway; accepted."""
        provider = _make_provider()
        private_key, jwks = _build_keypair()
        token = _sign_token(
            private_key,
            _base_claims(provider.realm_url, ["account", "mcp-gateway"]),
        )

        with patch.object(provider, "get_jwks", return_value=jwks):
            result = provider.validate_token(token)

        assert result["valid"] is True

    def test_external_issuer_with_gateway_audience_accepted(self):
        """The external realm issuer plus a gateway audience validates."""
        provider = _make_provider()
        private_key, jwks = _build_keypair()
        token = _sign_token(private_key, _base_claims(provider.external_realm_url, "mcp-gateway"))

        with patch.object(provider, "get_jwks", return_value=jwks):
            result = provider.validate_token(token)

        assert result["valid"] is True
