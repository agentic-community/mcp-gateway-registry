import json
import time
from unittest.mock import patch

import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric import rsa
from jwt.algorithms import RSAAlgorithm

pytestmark = [pytest.mark.unit, pytest.mark.auth]


class TestNormalizeGroupsClaim:
    def test_array_of_strings_passes_through(self):
        from providers.base import normalize_groups_claim

        value = ["mcp-registry-admin", "mcp-registry-user"]
        assert normalize_groups_claim(value, source="test") == value

    def test_empty_array_is_allowed(self):
        from providers.base import normalize_groups_claim

        assert normalize_groups_claim([], source="test") == []

    def test_absent_claim_maps_to_empty_list(self):
        from providers.base import normalize_groups_claim

        assert normalize_groups_claim(None, source="test") == []

    def test_object_claim_is_rejected(self):
        from providers.base import normalize_groups_claim

        with pytest.raises(ValueError, match="array of strings"):
            normalize_groups_claim({"mcp-registry-admin": True}, source="test")

    def test_string_claim_is_rejected(self):
        from providers.base import normalize_groups_claim

        with pytest.raises(ValueError, match="array of strings"):
            normalize_groups_claim("mcp-registry-admin", source="test")

    def test_list_with_non_string_element_is_rejected(self):
        from providers.base import normalize_groups_claim

        with pytest.raises(ValueError, match="array of strings"):
            normalize_groups_claim(["ok", {"nested": True}], source="test")

    def test_list_of_booleans_is_rejected(self):
        from providers.base import normalize_groups_claim

        with pytest.raises(ValueError, match="array of strings"):
            normalize_groups_claim([True, False], source="test")


def _build_keypair(kid: str = "test-kid") -> tuple[rsa.RSAPrivateKey, dict]:
    private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    public_jwk = json.loads(RSAAlgorithm.to_jwk(private_key.public_key()))
    public_jwk["kid"] = kid
    public_jwk["alg"] = "RS256"
    public_jwk["use"] = "sig"
    return private_key, {"keys": [public_jwk]}


def _sign_token(private_key: rsa.RSAPrivateKey, claims: dict, kid: str = "test-kid") -> str:
    return jwt.encode(claims, private_key, algorithm="RS256", headers={"kid": kid})


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


def _claims(provider, groups) -> dict:
    now = int(time.time())
    claims = {
        "iss": provider.realm_url,
        "aud": "mcp-gateway",
        "sub": "service-account-vrfy-esc-9a",
        "preferred_username": "service-account-vrfy-esc-9a",
        "azp": "vrfy-esc-9a",
        "scope": "openid",
        "iat": now,
        "exp": now + 3600,
    }
    if groups is not _ABSENT:
        claims["groups"] = groups
    return claims


_ABSENT = object()


class TestKeycloakGroupsType:
    def test_object_valued_groups_claim_is_rejected(self):
        provider = _make_provider()
        private_key, jwks = _build_keypair()
        token = _sign_token(private_key, _claims(provider, {"mcp-registry-admin": True}))

        with patch.object(provider, "get_jwks", return_value=jwks):
            with pytest.raises(ValueError):
                provider.validate_token(token)

    def test_string_valued_groups_claim_is_rejected(self):
        provider = _make_provider()
        private_key, jwks = _build_keypair()
        token = _sign_token(private_key, _claims(provider, "mcp-registry-admin"))

        with patch.object(provider, "get_jwks", return_value=jwks):
            with pytest.raises(ValueError):
                provider.validate_token(token)

    def test_absent_groups_claim_yields_empty_list(self):
        provider = _make_provider()
        private_key, jwks = _build_keypair()
        token = _sign_token(private_key, _claims(provider, _ABSENT))

        with patch.object(provider, "get_jwks", return_value=jwks):
            result = provider.validate_token(token)

        assert result["valid"] is True
        assert result["groups"] == []

    def test_array_valued_groups_claim_still_validates(self):
        provider = _make_provider()
        private_key, jwks = _build_keypair()
        token = _sign_token(private_key, _claims(provider, ["mcp-registry-user"]))

        with patch.object(provider, "get_jwks", return_value=jwks):
            result = provider.validate_token(token)

        assert result["valid"] is True
        assert result["groups"] == ["mcp-registry-user"]


class TestFilterSessionGroupsFailClosed:
    @pytest.mark.asyncio
    async def test_object_valued_groups_rejected(self):
        from group_filter import filter_session_groups

        result = await filter_session_groups({"mcp-registry-admin": True}, username_hash="h")
        assert result == []

    @pytest.mark.asyncio
    async def test_string_valued_groups_rejected(self):
        from group_filter import filter_session_groups

        result = await filter_session_groups("mcp-registry-admin", username_hash="h")
        assert result == []
