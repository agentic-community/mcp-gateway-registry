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


KEYCLOAK_GROUP_PATH_CASES = [
    pytest.param("/registry-admins", "registry-admins", True, id="top-level-admin"),
    pytest.param("/mcp-registry-user", "mcp-registry-user", False, id="top-level-other"),
    pytest.param(
        "/team/registry-admins", "team/registry-admins", False, id="subgroup-named-like-admin"
    ),
    pytest.param(
        "/a/b/registry-admins", "a/b/registry-admins", False, id="deeply-nested-admin-name"
    ),
    pytest.param(
        "/mcp-servers-unrestricted/read",
        "mcp-servers-unrestricted/read",
        False,
        id="top-level-name-containing-slash",
    ),
    pytest.param("registry-admins", "registry-admins", True, id="legacy-flat-admin"),
    pytest.param(
        "team/registry-admins", "team/registry-admins", False, id="legacy-flat-with-slash"
    ),
    pytest.param("//registry-admins", "/registry-admins", False, id="double-leading-slash"),
    pytest.param("/", None, False, id="root-path"),
    pytest.param("", None, False, id="empty"),
]


class TestKeycloakBearerGroupPaths:
    """Only the dedicated signed claim is interpreted as Keycloak paths."""

    def _validate(self, groups) -> dict:
        provider = _make_provider()
        private_key, jwks = _build_keypair()
        claims = _claims(provider, _ABSENT)
        claims["mcpgw_group_paths"] = groups
        token = _sign_token(private_key, claims)
        with patch.object(provider, "get_jwks", return_value=jwks):
            return provider.validate_token(token)

    @pytest.mark.parametrize(("claim", "expected", "is_admin"), KEYCLOAK_GROUP_PATH_CASES)
    def test_group_claim_maps_to_gateway_name(self, claim, expected, is_admin):
        from registry.auth.privileged_constants import ADMIN_GROUP_MARKERS

        result = self._validate([claim, "/mcp-registry-user"])

        expected_groups = ([expected] if expected is not None else []) + ["mcp-registry-user"]
        assert result["groups"] == expected_groups
        assert ("registry-admins" in ADMIN_GROUP_MARKERS.intersection(result["groups"])) is is_admin

    @pytest.mark.parametrize(("claim", "expected", "is_admin"), KEYCLOAK_GROUP_PATH_CASES)
    def test_registry_names_admin_api_groups_like_the_token(self, claim, expected, is_admin):
        """Group updates must use the same names as the token's authorization groups."""
        from providers.keycloak import _normalize_keycloak_groups

        from registry.utils.keycloak_manager import _membership_group_name

        token_side = _normalize_keycloak_groups([claim], source="test")
        registry_side = _membership_group_name({"id": "g1", "path": claim})

        assert token_side == ([expected] if expected is not None else [])
        assert registry_side == (expected or "")

    @pytest.mark.asyncio
    async def test_subgroup_does_not_inherit_admin_group_scopes(self):
        """Scope mapping is by exact name, so only the top-level group maps."""
        from auth_server import server as srv

        mappings = {"registry-admins": ["mcp-registry-admin"]}

        class _FakeScopeRepo:
            async def get_group_mappings_bulk(self, groups):
                return [scope for group in groups for scope in mappings.get(group, [])]

        admin = self._validate(["/registry-admins"])
        subgroup = self._validate(["/x/registry-admins"])
        with patch.object(srv, "get_scope_repository", return_value=_FakeScopeRepo()):
            admin_scopes = await srv.map_groups_to_scopes(admin["groups"])
            subgroup_scopes = await srv.map_groups_to_scopes(subgroup["groups"])

        assert admin_scopes == ["mcp-registry-admin"]
        assert subgroup_scopes == []

    @pytest.mark.parametrize(
        ("group_claims", "expected"),
        [
            ({"groups": ["/registry-admins"]}, ["/registry-admins"]),
            ({"groups": ["registry-admins"]}, ["registry-admins"]),
            ({"groups": ["registry-admins"], "mcpgw_group_paths": []}, []),
            (
                {"groups": ["registry-admins"], "mcpgw_group_paths": ["/team/registry-admins"]},
                ["team/registry-admins"],
            ),
            (
                {"groups": ["/registry-admins"], "mcpgw_group_paths": ["/registry-admins"]},
                ["registry-admins"],
            ),
            (
                {"groups": {"registry-admins": True}, "mcpgw_group_paths": ["/mcp-registry-user"]},
                ["mcp-registry-user"],
            ),
        ],
    )
    def test_signed_claim_cutover_never_reinterprets_or_merges_legacy_groups(
        self, group_claims, expected
    ):
        provider = _make_provider()
        private_key, jwks = _build_keypair()
        claims = {**_claims(provider, _ABSENT), **group_claims}
        with patch.object(provider, "get_jwks", return_value=jwks):
            result = provider.validate_token(_sign_token(private_key, claims))
        assert result["groups"] == expected

    @pytest.mark.parametrize("malformed", [None, "/registry-admins", {}, [True], ["/ok", 1]])
    def test_malformed_path_claim_denies_despite_legacy_admin(self, malformed):
        provider = _make_provider()
        private_key, jwks = _build_keypair()
        claims = _claims(provider, ["registry-admins"])
        claims["mcpgw_group_paths"] = malformed
        with patch.object(provider, "get_jwks", return_value=jwks), pytest.raises(ValueError):
            provider.validate_token(_sign_token(private_key, claims))


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
