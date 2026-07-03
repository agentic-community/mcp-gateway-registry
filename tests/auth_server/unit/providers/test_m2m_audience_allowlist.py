"""Unit tests for M2M audience-allowlist enforcement in the Okta and
PingFederate providers.

These tests exercise the access-token validation path with a real RSA keypair so
that audience enforcement is genuinely performed by the JWT library, not merely
by a mock. The security property under test:

  A machine-to-machine token that is genuinely signed by the configured IdP and
  carries the correct issuer, but whose ``aud`` was minted for a DIFFERENT
  resource/API in the same tenant, MUST be rejected. The gateway only accepts an
  M2M ``aud`` that appears on an explicit, config-driven allowlist (plus the
  configured client ids). Audience verification is never derived from unverified
  token claims, and an unconfigured allowlist fails closed.
"""

import json
import time
from unittest.mock import patch

import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric import rsa
from jwt.algorithms import RSAAlgorithm

from auth_server.providers.okta import OktaProvider
from auth_server.providers.pingfederate import PingFederateProvider

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


def _sign(
    private_key: rsa.RSAPrivateKey,
    claims: dict,
    kid: str = "test-kid",
) -> str:
    """Sign a token with the given RSA private key (RS256)."""
    return jwt.encode(claims, private_key, algorithm="RS256", headers={"kid": kid})


def _m2m_claims(issuer: str, audience: str) -> dict:
    """Build a genuine M2M (client-credentials) access-token claim set."""
    now = int(time.time())
    return {
        "iss": issuer,
        "aud": audience,
        "sub": "svc-account",
        "cid": "0oaSERVICEclient",
        "client_id": "0oaSERVICEclient",
        "scope": "mcp:read",
        "groups": ["registry-admins"],
        "iat": now,
        "exp": now + 3600,
    }


# =============================================================================
# Okta provider
# =============================================================================


class TestOktaM2MAudienceAllowlist:
    """Okta M2M tokens are only accepted for allowlisted audiences."""

    def test_m2m_token_with_allowlisted_audience_accepted(self):
        """A token whose aud is in the configured M2M allowlist is accepted."""
        provider = OktaProvider(
            "dev-123.okta.com",
            "gateway-client",
            "secret",
            m2m_allowed_audiences=["api://ai-registry"],
        )
        private_key, jwks = _build_keypair()
        token = _sign(private_key, _m2m_claims(provider.issuer, "api://ai-registry"))

        with patch.object(provider, "get_jwks", return_value=jwks):
            result = provider.validate_token(token)

        assert result["valid"] is True
        assert result["client_id"] == "0oaSERVICEclient"

    def test_m2m_token_for_other_resource_rejected(self):
        """A genuinely-signed token minted for a DIFFERENT resource in the same
        tenant (aud not on the allowlist) is rejected — this is the fix."""
        provider = OktaProvider(
            "dev-123.okta.com",
            "gateway-client",
            "secret",
            m2m_allowed_audiences=["api://ai-registry"],
        )
        private_key, jwks = _build_keypair()
        # aud is a legitimate but DIFFERENT API in the same Okta tenant.
        token = _sign(private_key, _m2m_claims(provider.issuer, "api://other-service"))

        with patch.object(provider, "get_jwks", return_value=jwks):
            with pytest.raises(ValueError):
                provider.validate_token(token)

    def test_m2m_token_rejected_when_allowlist_unconfigured(self):
        """With no M2M audience allowlist configured, a custom-audience M2M token
        fails closed (only the configured client ids are accepted)."""
        provider = OktaProvider("dev-123.okta.com", "gateway-client", "secret")
        private_key, jwks = _build_keypair()
        token = _sign(private_key, _m2m_claims(provider.issuer, "api://ai-registry"))

        with patch.object(provider, "get_jwks", return_value=jwks):
            with pytest.raises(ValueError):
                provider.validate_token(token)

    def test_token_with_client_id_audience_still_accepted(self):
        """A token whose aud equals the gateway client_id is accepted without any
        allowlist entry (normal audience-bound token)."""
        provider = OktaProvider("dev-123.okta.com", "gateway-client", "secret")
        private_key, jwks = _build_keypair()
        token = _sign(private_key, _m2m_claims(provider.issuer, "gateway-client"))

        with patch.object(provider, "get_jwks", return_value=jwks):
            result = provider.validate_token(token)

        assert result["valid"] is True

    def test_verify_aud_never_disabled(self):
        """Regression guard: validate_token must always call jwt.decode with
        verify_aud=True — verify_aud must never be derived from token claims."""
        provider = OktaProvider(
            "dev-123.okta.com",
            "gateway-client",
            "secret",
            m2m_allowed_audiences=["api://ai-registry"],
        )
        private_key, jwks = _build_keypair()
        token = _sign(private_key, _m2m_claims(provider.issuer, "api://ai-registry"))

        real_decode = jwt.decode
        seen_options: list[dict] = []

        def _spy_decode(*args, **kwargs):
            options = kwargs.get("options")
            if options is not None:
                # Snapshot: PyJWT mutates the passed options dict in place,
                # backfilling default verify_* keys. We record only what the
                # caller explicitly set.
                seen_options.append(dict(options))
            return real_decode(*args, **kwargs)

        with patch.object(provider, "get_jwks", return_value=jwks):
            with patch("auth_server.providers.okta.jwt.decode", side_effect=_spy_decode):
                provider.validate_token(token)

        # The signature-verifying decode call (options carrying verify_aud) must
        # always have verify_aud True. The only verify_signature=False decode
        # is the self-signed sniff, which does not carry verify_aud.
        aud_options = [o for o in seen_options if "verify_aud" in o]
        assert aud_options, "expected a decode call that enforces audience"
        assert all(o["verify_aud"] is True for o in aud_options)


# =============================================================================
# Factory: env-var parsing of the allowlist
# =============================================================================


class TestAllowedAudienceParsing:
    """The factory parses the allowlist env var and wires it into the provider."""

    def test_parse_comma_and_space_separated(self):
        from auth_server.providers.factory import _parse_allowed_audiences

        with patch.dict(
            "os.environ",
            {"OKTA_M2M_ALLOWED_AUDIENCES": "api://a, api://b api://a"},
            clear=False,
        ):
            assert _parse_allowed_audiences("OKTA_M2M_ALLOWED_AUDIENCES") == [
                "api://a",
                "api://b",
            ]

    def test_unset_env_yields_empty_list(self):
        from auth_server.providers.factory import _parse_allowed_audiences

        with patch.dict("os.environ", {}, clear=True):
            assert _parse_allowed_audiences("OKTA_M2M_ALLOWED_AUDIENCES") == []

    def test_okta_factory_wires_allowlist(self, monkeypatch):
        monkeypatch.setenv("OKTA_DOMAIN", "dev-123.okta.com")
        monkeypatch.setenv("OKTA_CLIENT_ID", "test-cid")
        monkeypatch.setenv("OKTA_CLIENT_SECRET", "test-cs")
        monkeypatch.setenv("OKTA_M2M_ALLOWED_AUDIENCES", "api://ai-registry")

        from auth_server.providers.factory import get_auth_provider

        provider = get_auth_provider("okta")
        assert isinstance(provider, OktaProvider)
        assert provider.m2m_allowed_audiences == ["api://ai-registry"]

    def test_pingfederate_factory_wires_allowlist(self, monkeypatch):
        monkeypatch.setenv("PINGFEDERATE_BASE_URL", "https://pf.example.com:9031")
        monkeypatch.setenv("PINGFEDERATE_CLIENT_ID", "test-cid")
        monkeypatch.setenv("PINGFEDERATE_CLIENT_SECRET", "test-cs")
        monkeypatch.setenv("PINGFEDERATE_M2M_ALLOWED_AUDIENCES", "api://a,api://b")

        from auth_server.providers.factory import get_auth_provider

        provider = get_auth_provider("pingfederate")
        assert isinstance(provider, PingFederateProvider)
        assert provider.m2m_allowed_audiences == ["api://a", "api://b"]


# =============================================================================
# PingFederate provider
# =============================================================================


class TestPingFederateM2MAudienceAllowlist:
    """PingFederate M2M tokens are only accepted for allowlisted audiences."""

    def _provider(self, **kwargs) -> PingFederateProvider:
        return PingFederateProvider(
            base_url="https://pf.example.com:9031",
            client_id="gateway-client",
            client_secret="secret",
            **kwargs,
        )

    def test_m2m_token_with_allowlisted_audience_accepted(self):
        provider = self._provider(m2m_allowed_audiences=["api://ai-registry"])
        private_key, jwks = _build_keypair()
        token = _sign(private_key, _m2m_claims("https://pf.example.com", "api://ai-registry"))

        with patch.object(
            provider, "_get_openid_configuration", return_value={"issuer": "https://pf.example.com"}
        ):
            with patch.object(provider, "get_jwks", return_value=jwks):
                result = provider.validate_token(token)

        assert result["valid"] is True

    def test_m2m_token_for_other_resource_rejected(self):
        provider = self._provider(m2m_allowed_audiences=["api://ai-registry"])
        private_key, jwks = _build_keypair()
        token = _sign(private_key, _m2m_claims("https://pf.example.com", "api://other-service"))

        with patch.object(
            provider, "_get_openid_configuration", return_value={"issuer": "https://pf.example.com"}
        ):
            with patch.object(provider, "get_jwks", return_value=jwks):
                with pytest.raises(ValueError):
                    provider.validate_token(token)

    def test_m2m_token_rejected_when_allowlist_unconfigured(self):
        provider = self._provider()
        private_key, jwks = _build_keypair()
        token = _sign(private_key, _m2m_claims("https://pf.example.com", "api://ai-registry"))

        with patch.object(
            provider, "_get_openid_configuration", return_value={"issuer": "https://pf.example.com"}
        ):
            with patch.object(provider, "get_jwks", return_value=jwks):
                with pytest.raises(ValueError):
                    provider.validate_token(token)

    def test_application_id_uri_still_accepted(self):
        """application_id_uri remains a valid audience without an allowlist entry."""
        provider = self._provider(application_id_uri="api://mcp-gateway")
        private_key, jwks = _build_keypair()
        token = _sign(private_key, _m2m_claims("https://pf.example.com", "api://mcp-gateway"))

        with patch.object(
            provider, "_get_openid_configuration", return_value={"issuer": "https://pf.example.com"}
        ):
            with patch.object(provider, "get_jwks", return_value=jwks):
                result = provider.validate_token(token)

        assert result["valid"] is True

    def test_verify_aud_never_disabled(self):
        provider = self._provider(m2m_allowed_audiences=["api://ai-registry"])
        private_key, jwks = _build_keypair()
        token = _sign(private_key, _m2m_claims("https://pf.example.com", "api://ai-registry"))

        real_decode = jwt.decode
        seen_options: list[dict] = []

        def _spy_decode(*args, **kwargs):
            options = kwargs.get("options")
            if options is not None:
                # Snapshot: PyJWT mutates the passed options dict in place,
                # backfilling default verify_* keys. We record only what the
                # caller explicitly set.
                seen_options.append(dict(options))
            return real_decode(*args, **kwargs)

        with patch.object(
            provider, "_get_openid_configuration", return_value={"issuer": "https://pf.example.com"}
        ):
            with patch.object(provider, "get_jwks", return_value=jwks):
                with patch(
                    "auth_server.providers.pingfederate.jwt.decode", side_effect=_spy_decode
                ):
                    provider.validate_token(token)

        aud_options = [o for o in seen_options if "verify_aud" in o]
        assert aud_options, "expected a decode call that enforces audience"
        assert all(o["verify_aud"] is True for o in aud_options)
