"""Unit tests for KeycloakProvider.exchanged_token_gateway_audiences.

The OBO engine forwards the token minted by a Keycloak token exchange to the
upstream MCP server, so that token must be verified, must belong to the user
whose token was exchanged, and must not also be a credential for this gateway.
Keycloak's legacy token exchange mints it with the audience client's default
client scopes, and the realm's ``mcp-gateway`` audience mapper on the ``basic``
scope then lands in ``aud``.

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
USER_SUB: str = "user-123"
# The subject token is read without signature verification, so any key works.
SUBJECT_TOKEN_KEY: str = "subject-token-test-key-0123456789abcdef"  # noqa: S105 - test fixture


def _build_keypair(kid: str = "test-kid") -> tuple[rsa.RSAPrivateKey, dict]:
    """Generate an RSA keypair and the matching single-key JWKS document."""
    private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    public_jwk = json.loads(RSAAlgorithm.to_jwk(private_key.public_key()))
    public_jwk["kid"] = kid
    public_jwk["alg"] = "RS256"
    public_jwk["use"] = "sig"
    return private_key, {"keys": [public_jwk]}


@pytest.fixture(scope="module")
def realm_keys() -> tuple[rsa.RSAPrivateKey, dict]:
    """The realm's signing key and JWKS, generated once for the module."""
    return _build_keypair()


@pytest.fixture(scope="module")
def foreign_key() -> rsa.RSAPrivateKey:
    """A signing key the realm JWKS does not contain."""
    private_key, _ = _build_keypair()
    return private_key


def _sign(
    private_key: rsa.RSAPrivateKey,
    audience: str | list[str],
    issuer: str = REALM_URL,
    expires_in: int = 300,
    iat_offset: int = 0,
    sub: str | None = USER_SUB,
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


def _subject_token(sub: str | None = USER_SUB) -> str:
    """Build the ingress token that was exchanged; ``sub=None`` omits the subject."""
    claims = {"iss": REALM_URL, "aud": "mcp-gateway"}
    if sub is not None:
        claims["sub"] = sub
    return jwt.encode(claims, SUBJECT_TOKEN_KEY, algorithm="HS256")


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


def _audiences(provider, token: str, target_audience: str = TARGET_AUDIENCE) -> list[str]:
    """Run the check for a token exchanged from the default user's subject token."""
    return provider.exchanged_token_gateway_audiences(token, target_audience, _subject_token())


class TestSafeTokens:
    """Tokens audienced only to the target are safe to forward."""

    def test_target_only_audience_returns_empty(self, realm_keys):
        provider = _make_provider()
        private_key, jwks = realm_keys
        token = _sign(private_key, [TARGET_AUDIENCE])
        with patch.object(provider, "get_jwks", return_value=jwks):
            assert _audiences(provider, token) == []

    def test_string_audience_returns_empty(self, realm_keys):
        provider = _make_provider()
        private_key, jwks = realm_keys
        token = _sign(private_key, TARGET_AUDIENCE)
        with patch.object(provider, "get_jwks", return_value=jwks):
            assert _audiences(provider, token) == []

    def test_account_audience_is_not_a_gateway_audience(self, realm_keys):
        """Keycloak's default ``account`` audience does not make a token valid here."""
        provider = _make_provider()
        private_key, jwks = realm_keys
        token = _sign(private_key, [TARGET_AUDIENCE, "account"])
        with patch.object(provider, "get_jwks", return_value=jwks):
            assert _audiences(provider, token) == []


class TestGatewayAudiencesReported:
    """Any audience the gateway accepts is reported, so the engine can refuse it."""

    def test_mcp_gateway_audience_is_reported(self, realm_keys):
        provider = _make_provider()
        private_key, jwks = realm_keys
        token = _sign(private_key, [TARGET_AUDIENCE, "mcp-gateway", "account"])
        with patch.object(provider, "get_jwks", return_value=jwks):
            assert _audiences(provider, token) == ["mcp-gateway"]

    def test_gateway_client_ids_are_reported_sorted(self, realm_keys):
        provider = _make_provider()
        private_key, jwks = realm_keys
        token = _sign(private_key, [TARGET_AUDIENCE, "gateway-web", "gateway-m2m"])
        with patch.object(provider, "get_jwks", return_value=jwks):
            assert _audiences(provider, token) == ["gateway-m2m", "gateway-web"]

    @pytest.mark.parametrize("gateway_audience", ["gateway-web", "gateway-m2m", "mcp-gateway"])
    def test_every_audience_validate_token_accepts_is_reported(
        self,
        realm_keys,
        gateway_audience,
    ):
        """The check uses the same audience list as the gateway's own validation."""
        provider = _make_provider()
        private_key, jwks = realm_keys
        with patch.object(provider, "get_jwks", return_value=jwks):
            assert provider.validate_token(_sign(private_key, [gateway_audience]))["valid"] is True
            token = _sign(private_key, [TARGET_AUDIENCE, gateway_audience])
            assert _audiences(provider, token) == [gateway_audience]

    def test_a_token_reported_clean_is_rejected_by_validate_token(self, realm_keys):
        provider = _make_provider()
        private_key, jwks = realm_keys
        token = _sign(private_key, [TARGET_AUDIENCE, "account"])
        with patch.object(provider, "get_jwks", return_value=jwks):
            assert _audiences(provider, token) == []
            with pytest.raises(ValueError):
                provider.validate_token(token)

    def test_validate_token_ignores_per_server_extra_audiences(self, realm_keys):
        """/validate passes per-server ``extra_audiences``, and Entra accepts them.

        The leak check reads only ``accepted_audiences()``, so it is complete
        only while Keycloak's ``validate_token`` ignores those extras. If this
        test fails, the leak check must learn the same extras before the
        change ships, or an exchanged token carrying one becomes replayable.
        """
        provider = _make_provider()
        private_key, jwks = realm_keys
        per_server_audience = "https://gateway.example.com/finance/mcp"
        token = _sign(private_key, [per_server_audience])
        with patch.object(provider, "get_jwks", return_value=jwks):
            with pytest.raises(ValueError):
                provider.validate_token(token, extra_audiences=[per_server_audience])

    def test_target_that_is_itself_a_gateway_audience_is_reported(self, realm_keys):
        provider = _make_provider()
        private_key, jwks = realm_keys
        token = _sign(private_key, ["mcp-gateway"])
        with patch.object(provider, "get_jwks", return_value=jwks):
            assert _audiences(provider, token, "mcp-gateway") == ["mcp-gateway"]


class TestSubjectBinding:
    """The exchanged token must belong to the user whose token was exchanged."""

    def test_different_subject_raises(self, realm_keys):
        from providers.base import IdTokenVerificationError

        provider = _make_provider()
        private_key, jwks = realm_keys
        token = _sign(private_key, [TARGET_AUDIENCE], sub="someone-else")
        with patch.object(provider, "get_jwks", return_value=jwks):
            with pytest.raises(IdTokenVerificationError, match="does not match"):
                _audiences(provider, token)

    def test_subject_token_without_sub_raises(self, realm_keys):
        from providers.base import IdTokenVerificationError

        provider = _make_provider()
        private_key, jwks = realm_keys
        token = _sign(private_key, [TARGET_AUDIENCE])
        with patch.object(provider, "get_jwks", return_value=jwks):
            with pytest.raises(IdTokenVerificationError, match="subject token"):
                provider.exchanged_token_gateway_audiences(
                    token, TARGET_AUDIENCE, _subject_token(sub=None)
                )

    def test_undecodable_subject_token_raises(self, realm_keys):
        from providers.base import IdTokenVerificationError

        provider = _make_provider()
        private_key, jwks = realm_keys
        token = _sign(private_key, [TARGET_AUDIENCE])
        with patch.object(provider, "get_jwks", return_value=jwks):
            with pytest.raises(IdTokenVerificationError, match="subject token"):
                provider.exchanged_token_gateway_audiences(token, TARGET_AUDIENCE, "not-a-jwt")


class TestClockSkew:
    """The token is checked moments after Keycloak minted it; a small skew between
    the IdP's clock and this process must not fail it."""

    def test_small_clock_skew_is_tolerated(self, realm_keys):
        provider = _make_provider()
        private_key, jwks = realm_keys
        token = _sign(private_key, [TARGET_AUDIENCE], iat_offset=2)
        with patch.object(provider, "get_jwks", return_value=jwks):
            assert _audiences(provider, token) == []

    def test_large_clock_skew_is_rejected(self, realm_keys):
        from providers.base import IdTokenVerificationError

        provider = _make_provider()
        private_key, jwks = realm_keys
        token = _sign(private_key, [TARGET_AUDIENCE], iat_offset=60)
        with patch.object(provider, "get_jwks", return_value=jwks):
            with pytest.raises(IdTokenVerificationError):
                _audiences(provider, token)


class TestVerificationFailsClosed:
    """An unverifiable token raises instead of being inspected."""

    def test_missing_sub_raises(self, realm_keys):
        """A token without the user's subject is not a delegated token."""
        from providers.base import IdTokenVerificationError

        provider = _make_provider()
        private_key, jwks = realm_keys
        token = _sign(private_key, [TARGET_AUDIENCE], sub=None)
        with patch.object(provider, "get_jwks", return_value=jwks):
            with pytest.raises(IdTokenVerificationError, match="sub"):
                _audiences(provider, token)

    def test_missing_target_audience_raises(self, realm_keys):
        from providers.base import IdTokenVerificationError

        provider = _make_provider()
        private_key, jwks = realm_keys
        token = _sign(private_key, ["mcp-gateway"])
        with patch.object(provider, "get_jwks", return_value=jwks):
            with pytest.raises(IdTokenVerificationError):
                _audiences(provider, token)

    def test_expired_token_raises(self, realm_keys):
        from providers.base import IdTokenVerificationError

        provider = _make_provider()
        private_key, jwks = realm_keys
        token = _sign(private_key, [TARGET_AUDIENCE], expires_in=-60)
        with patch.object(provider, "get_jwks", return_value=jwks):
            with pytest.raises(IdTokenVerificationError):
                _audiences(provider, token)

    def test_unknown_issuer_raises(self, realm_keys):
        from providers.base import IdTokenVerificationError

        provider = _make_provider()
        private_key, jwks = realm_keys
        token = _sign(private_key, [TARGET_AUDIENCE], issuer="https://evil.example/realms/x")
        with patch.object(provider, "get_jwks", return_value=jwks):
            with pytest.raises(IdTokenVerificationError):
                _audiences(provider, token)

    def test_token_signed_by_another_key_raises(self, realm_keys, foreign_key):
        from providers.base import IdTokenVerificationError

        provider = _make_provider()
        _, jwks = realm_keys
        token = _sign(foreign_key, [TARGET_AUDIENCE])
        with patch.object(provider, "get_jwks", return_value=jwks):
            with pytest.raises(IdTokenVerificationError):
                _audiences(provider, token)
