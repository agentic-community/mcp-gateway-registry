"""Unit tests for KeycloakProvider.exchanged_token_gateway_audiences.

The OBO engine forwards the token minted by a Keycloak token exchange to the
upstream MCP server, so that token must be verified and must not also be a
credential for this gateway. Keycloak's legacy token exchange mints it with
the audience client's default client scopes, and the realm's ``mcp-gateway``
audience mapper on the ``basic`` scope then lands in ``aud``.

These tests use a genuine RSA keypair and the real verification path; only the
JWKS fetch is patched.
"""

import json
import time
from unittest.mock import patch

import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric import rsa
from jwt.algorithms import RSAAlgorithm

pytestmark = [pytest.mark.unit, pytest.mark.auth]

TARGET_AUDIENCE: str = "finance-mcp-server"
REALM_URL: str = "http://keycloak:8080/realms/test-realm"


def _build_keypair(kid: str = "test-kid") -> tuple[rsa.RSAPrivateKey, dict]:
    """Generate an RSA keypair and the matching single-key JWKS document."""
    private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    public_jwk = json.loads(RSAAlgorithm.to_jwk(private_key.public_key()))
    public_jwk["kid"] = kid
    public_jwk["alg"] = "RS256"
    public_jwk["use"] = "sig"
    return private_key, {"keys": [public_jwk]}


def _sign(
    private_key: rsa.RSAPrivateKey,
    audience: str | list[str],
    issuer: str = REALM_URL,
    expires_in: int = 300,
    iat_offset: int = 0,
    sub: str | None = "user-123",
) -> str:
    """Sign an exchanged access token (RS256) with the given audience.

    ``iat_offset`` shifts the issue time into the future to simulate an IdP
    clock running ahead; ``sub=None`` omits the subject claim.
    """
    now = int(time.time())
    claims = {
        "iss": issuer,
        "aud": audience,
        "azp": "gateway-web",
        "iat": now + iat_offset,
        "exp": now + expires_in,
    }
    if sub is not None:
        claims["sub"] = sub
    return jwt.encode(claims, private_key, algorithm="RS256", headers={"kid": "test-kid"})


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


class TestSafeTokens:
    """Tokens audienced only to the target are safe to forward."""

    def test_target_only_audience_returns_empty(self):
        provider = _make_provider()
        private_key, jwks = _build_keypair()
        token = _sign(private_key, [TARGET_AUDIENCE])
        with patch.object(provider, "get_jwks", return_value=jwks):
            assert provider.exchanged_token_gateway_audiences(token, TARGET_AUDIENCE) == []

    def test_string_audience_returns_empty(self):
        provider = _make_provider()
        private_key, jwks = _build_keypair()
        token = _sign(private_key, TARGET_AUDIENCE)
        with patch.object(provider, "get_jwks", return_value=jwks):
            assert provider.exchanged_token_gateway_audiences(token, TARGET_AUDIENCE) == []

    def test_account_audience_is_not_a_gateway_audience(self):
        """Keycloak's default ``account`` audience does not make a token valid here."""
        provider = _make_provider()
        private_key, jwks = _build_keypair()
        token = _sign(private_key, [TARGET_AUDIENCE, "account"])
        with patch.object(provider, "get_jwks", return_value=jwks):
            assert provider.exchanged_token_gateway_audiences(token, TARGET_AUDIENCE) == []


class TestGatewayAudiencesReported:
    """Any audience the gateway accepts is reported, so the engine can refuse it."""

    def test_mcp_gateway_audience_is_reported(self):
        provider = _make_provider()
        private_key, jwks = _build_keypair()
        token = _sign(private_key, [TARGET_AUDIENCE, "mcp-gateway", "account"])
        with patch.object(provider, "get_jwks", return_value=jwks):
            assert provider.exchanged_token_gateway_audiences(token, TARGET_AUDIENCE) == [
                "mcp-gateway"
            ]

    def test_gateway_client_ids_are_reported_sorted(self):
        provider = _make_provider()
        private_key, jwks = _build_keypair()
        token = _sign(private_key, [TARGET_AUDIENCE, "gateway-web", "gateway-m2m"])
        with patch.object(provider, "get_jwks", return_value=jwks):
            assert provider.exchanged_token_gateway_audiences(token, TARGET_AUDIENCE) == [
                "gateway-m2m",
                "gateway-web",
            ]

    @pytest.mark.parametrize("gateway_audience", ["gateway-web", "gateway-m2m", "mcp-gateway"])
    def test_every_audience_validate_token_accepts_is_reported(self, gateway_audience):
        """The check uses the same audience list as the gateway's own validation."""
        provider = _make_provider()
        private_key, jwks = _build_keypair()
        with patch.object(provider, "get_jwks", return_value=jwks):
            assert provider.validate_token(_sign(private_key, [gateway_audience]))["valid"] is True
            token = _sign(private_key, [TARGET_AUDIENCE, gateway_audience])
            assert provider.exchanged_token_gateway_audiences(token, TARGET_AUDIENCE) == [
                gateway_audience
            ]

    def test_a_token_reported_clean_is_rejected_by_validate_token(self):
        provider = _make_provider()
        private_key, jwks = _build_keypair()
        token = _sign(private_key, [TARGET_AUDIENCE, "account"])
        with patch.object(provider, "get_jwks", return_value=jwks):
            assert provider.exchanged_token_gateway_audiences(token, TARGET_AUDIENCE) == []
            with pytest.raises(ValueError):
                provider.validate_token(token)

    def test_target_that_is_itself_a_gateway_audience_is_reported(self):
        provider = _make_provider()
        private_key, jwks = _build_keypair()
        token = _sign(private_key, ["mcp-gateway"])
        with patch.object(provider, "get_jwks", return_value=jwks):
            assert provider.exchanged_token_gateway_audiences(token, "mcp-gateway") == [
                "mcp-gateway"
            ]


class TestClockSkew:
    """The token is checked moments after Keycloak minted it; a small skew between
    the IdP's clock and this process must not fail it."""

    def test_small_clock_skew_is_tolerated(self):
        provider = _make_provider()
        private_key, jwks = _build_keypair()
        token = _sign(private_key, [TARGET_AUDIENCE], iat_offset=2)
        with patch.object(provider, "get_jwks", return_value=jwks):
            assert provider.exchanged_token_gateway_audiences(token, TARGET_AUDIENCE) == []

    def test_large_clock_skew_is_rejected(self):
        from providers.base import IdTokenVerificationError

        provider = _make_provider()
        private_key, jwks = _build_keypair()
        token = _sign(private_key, [TARGET_AUDIENCE], iat_offset=60)
        with patch.object(provider, "get_jwks", return_value=jwks):
            with pytest.raises(IdTokenVerificationError):
                provider.exchanged_token_gateway_audiences(token, TARGET_AUDIENCE)


class TestVerificationFailsClosed:
    """An unverifiable token raises instead of being inspected."""

    def test_missing_sub_raises(self):
        """A token without the user's subject is not a delegated token."""
        from providers.base import IdTokenVerificationError

        provider = _make_provider()
        private_key, jwks = _build_keypair()
        token = _sign(private_key, [TARGET_AUDIENCE], sub=None)
        with patch.object(provider, "get_jwks", return_value=jwks):
            with pytest.raises(IdTokenVerificationError, match="sub"):
                provider.exchanged_token_gateway_audiences(token, TARGET_AUDIENCE)

    def test_missing_target_audience_raises(self):
        from providers.base import IdTokenVerificationError

        provider = _make_provider()
        private_key, jwks = _build_keypair()
        token = _sign(private_key, ["mcp-gateway"])
        with patch.object(provider, "get_jwks", return_value=jwks):
            with pytest.raises(IdTokenVerificationError):
                provider.exchanged_token_gateway_audiences(token, TARGET_AUDIENCE)

    def test_expired_token_raises(self):
        from providers.base import IdTokenVerificationError

        provider = _make_provider()
        private_key, jwks = _build_keypair()
        token = _sign(private_key, [TARGET_AUDIENCE], expires_in=-60)
        with patch.object(provider, "get_jwks", return_value=jwks):
            with pytest.raises(IdTokenVerificationError):
                provider.exchanged_token_gateway_audiences(token, TARGET_AUDIENCE)

    def test_unknown_issuer_raises(self):
        from providers.base import IdTokenVerificationError

        provider = _make_provider()
        private_key, jwks = _build_keypair()
        token = _sign(private_key, [TARGET_AUDIENCE], issuer="https://evil.example/realms/x")
        with patch.object(provider, "get_jwks", return_value=jwks):
            with pytest.raises(IdTokenVerificationError):
                provider.exchanged_token_gateway_audiences(token, TARGET_AUDIENCE)

    def test_token_signed_by_another_key_raises(self):
        from providers.base import IdTokenVerificationError

        provider = _make_provider()
        _, jwks = _build_keypair()
        other_key, _ = _build_keypair()
        token = _sign(other_key, [TARGET_AUDIENCE])
        with patch.object(provider, "get_jwks", return_value=jwks):
            with pytest.raises(IdTokenVerificationError):
                provider.exchanged_token_gateway_audiences(token, TARGET_AUDIENCE)
