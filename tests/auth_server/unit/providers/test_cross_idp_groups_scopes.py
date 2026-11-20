"""Signed bearer-token contracts across the six supported identity providers.

Only JWKS retrieval is mocked. PingFederate's discovery document is preloaded
in the real discovery cache. JWT decoding, signature checks, and claim extraction
run through each provider's validate_token implementation.

Keycloak path normalization and Entra's scp/legacy-scope precedence already have
signed coverage in test_groups_claim_type.py and test_entra.py.
"""

import json
import time
from unittest.mock import patch

import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric import rsa
from jwt.algorithms import RSAAlgorithm

from auth_server.providers.auth0 import Auth0Provider
from auth_server.providers.base import AuthProvider
from auth_server.providers.cognito import CognitoProvider
from auth_server.providers.entra import EntraIdProvider
from auth_server.providers.keycloak import KeycloakProvider
from auth_server.providers.okta import OktaProvider
from auth_server.providers.pingfederate import PingFederateProvider

pytestmark = [pytest.mark.unit, pytest.mark.auth]

PROVIDERS = ("keycloak", "cognito", "entra", "okta", "auth0", "pingfederate")
OTHER_PROVIDERS = PROVIDERS[1:]
SCOPES = ["mcp-servers/read", "api://registry/tools.invoke"]
SCOPE_STRING = "  mcp-servers/read\tapi://registry/tools.invoke\n"
ABSENT = object()
MALFORMED_GROUPS = [
    pytest.param({}, id="empty-object"),
    pytest.param("", id="empty-string"),
    pytest.param(False, id="false"),
    pytest.param(0, id="zero"),
    pytest.param(["readers", 1], id="mixed-array"),
]


@pytest.fixture(scope="module")
def signing_key():
    private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    jwk = json.loads(RSAAlgorithm.to_jwk(private_key.public_key()))
    jwk.update(kid="cross-idp-key", alg="RS256", use="sig")
    return private_key, {"keys": [jwk]}


@pytest.fixture(params=PROVIDERS)
def provider(request, monkeypatch, signing_key):
    name = request.param
    if name == "keycloak":
        instance = KeycloakProvider(
            keycloak_url="https://keycloak.example.test",
            realm="cross-idp",
            client_id="gateway-web",
            client_secret="test-secret",
        )
    elif name == "cognito":
        instance = CognitoProvider(
            user_pool_id="us-east-1_crossidp",
            region="us-east-1",
            client_id="gateway-web",
            client_secret="test-secret",
        )
    elif name == "entra":
        for variable in (
            "ENTRA_LOGIN_BASE_URL",
            "ENTRA_GRAPH_BASE_URL",
            "ENTRA_GRAPH_RESOURCE_URL",
        ):
            monkeypatch.delenv(variable, raising=False)
        instance = EntraIdProvider(
            tenant_id="cross-idp-tenant",
            client_id="gateway-web",
            client_secret="test-secret",
            scope_format="v2",
            application_id_uri="api://gateway-web",
        )
    elif name == "okta":
        monkeypatch.delenv("OKTA_AUTH_SERVER_ID", raising=False)
        instance = OktaProvider(
            okta_domain="cross-idp.okta.com",
            client_id="gateway-web",
            client_secret="test-secret",
        )
    elif name == "auth0":
        instance = Auth0Provider(
            domain="cross-idp.auth0.com",
            client_id="gateway-web",
            client_secret="test-secret",
            audience="https://gateway.example.test",
        )
    elif name == "pingfederate":
        instance = PingFederateProvider(
            base_url="https://ping.example.test",
            client_id="gateway-web",
            client_secret="test-secret",
        )
        monkeypatch.setitem(
            AuthProvider._shared_discovery,
            instance.config_url,
            {
                "issuer": "https://ping.example.test",
                "jwks_uri": "https://ping.example.test/pf/JWKS",
            },
        )
    else:
        raise AssertionError(f"Unknown test provider: {name}")

    with patch.object(instance, "get_jwks", return_value=signing_key[1]):
        yield instance


@pytest.fixture
def claims(provider):
    if isinstance(provider, KeycloakProvider):
        issuer = provider.external_realm_url
    elif isinstance(provider, EntraIdProvider):
        issuer = provider.issuer_v2
    else:
        issuer = provider.issuer
    now = int(time.time())
    result = {
        "iss": issuer,
        "aud": provider.client_id,
        "sub": "alice",
        "iat": now,
        "exp": now + 3600,
    }
    if isinstance(provider, CognitoProvider):
        # Cognito access tokens bind to client_id instead of aud.
        result.pop("aud")
        result.update(token_use="access", client_id=provider.client_id, username="alice")
    return result


@pytest.fixture
def groups_claim(provider):
    if isinstance(provider, CognitoProvider):
        return "cognito:groups"
    if isinstance(provider, Auth0Provider):
        return "https://mcp-gateway/groups"
    return "groups"


def _sign(signing_key, claims):
    return jwt.encode(claims, signing_key[0], algorithm="RS256", headers={"kid": "cross-idp-key"})


@pytest.mark.parametrize(
    ("provider", "scope_claims"),
    [
        pytest.param("keycloak", {"scope": SCOPE_STRING}, id="keycloak-scope-string"),
        pytest.param("cognito", {"scope": SCOPE_STRING}, id="cognito-scope-string"),
        pytest.param("entra", {"scp": SCOPE_STRING}, id="entra-scp-string"),
        pytest.param("okta", {"scp": SCOPES}, id="okta-scp-array"),
        pytest.param("okta", {"scp": SCOPE_STRING}, id="okta-scp-string"),
        pytest.param("okta", {"scope": SCOPE_STRING}, id="okta-legacy-scope"),
        pytest.param("auth0", {"scope": SCOPE_STRING}, id="auth0-scope-string"),
        pytest.param("pingfederate", {"scope": SCOPE_STRING}, id="ping-scope-string"),
        pytest.param("pingfederate", {"scope": SCOPES}, id="ping-scope-array"),
    ],
    indirect=["provider"],
)
def test_signed_canonical_groups_and_scopes(
    provider, claims, groups_claim, scope_claims, signing_key
):
    claims.update(scope_claims)
    claims[groups_claim] = ["registry-readers", "team/tools"]

    result = provider.validate_token(_sign(signing_key, claims))

    assert result["valid"] is True
    assert result["groups"] == ["registry-readers", "team/tools"]
    assert result["scopes"] == SCOPES


@pytest.mark.parametrize("provider", OTHER_PROVIDERS, indirect=True)
@pytest.mark.parametrize(
    "keycloak_paths",
    [pytest.param(["/registry-admins"], id="valid-paths"), pytest.param({}, id="malformed-paths")],
)
def test_other_idps_preserve_literal_slashes_and_ignore_keycloak_claim(
    provider, claims, groups_claim, signing_key, keycloak_paths
):
    names = ["/registry-admins", "//literal", "/team/registry-admins", "team/tools"]
    claims[groups_claim] = names
    claims["mcpgw_group_paths"] = keycloak_paths

    result = provider.validate_token(_sign(signing_key, claims))

    assert result["groups"] == names
    assert "registry-admins" not in result["groups"]


@pytest.mark.parametrize("provider", OTHER_PROVIDERS, indirect=True)
@pytest.mark.parametrize(
    "empty_groups",
    [
        pytest.param(ABSENT, id="absent"),
        pytest.param(None, id="null"),
        pytest.param([], id="empty"),
    ],
)
def test_other_idps_do_not_invent_groups_from_keycloak_claim(
    provider, claims, groups_claim, signing_key, empty_groups
):
    if empty_groups is not ABSENT:
        claims[groups_claim] = empty_groups
    claims["mcpgw_group_paths"] = ["/registry-admins"]
    if groups_claim != "groups":
        claims["groups"] = ["registry-admins"]

    result = provider.validate_token(_sign(signing_key, claims))

    assert result["groups"] == []
    assert result["scopes"] == []


@pytest.mark.parametrize("malformed", MALFORMED_GROUPS)
def test_signed_malformed_canonical_groups_rejected(
    provider, claims, groups_claim, signing_key, malformed
):
    claims[groups_claim] = malformed

    with pytest.raises(ValueError, match="array of strings"):
        provider.validate_token(_sign(signing_key, claims))


@pytest.mark.parametrize(
    ("provider", "fallback_claim"),
    [("auth0", "permissions"), ("entra", "roles")],
    indirect=["provider"],
)
@pytest.mark.parametrize("malformed", MALFORMED_GROUPS)
def test_signed_malformed_groups_cannot_fall_back_to_permissions_or_roles(
    provider, claims, groups_claim, signing_key, fallback_claim, malformed
):
    claims[groups_claim] = malformed
    claims[fallback_claim] = ["registry-admins"]

    with pytest.raises(ValueError, match="array of strings"):
        provider.validate_token(_sign(signing_key, claims))


@pytest.mark.parametrize(
    ("provider", "fallback_claim"),
    [("auth0", "permissions"), ("entra", "roles")],
    indirect=["provider"],
)
@pytest.mark.parametrize(
    "empty_groups",
    [
        pytest.param(ABSENT, id="absent"),
        pytest.param(None, id="null"),
        pytest.param([], id="empty"),
    ],
)
def test_signed_permissions_and_roles_keep_existing_empty_groups_fallback(
    provider, claims, groups_claim, signing_key, fallback_claim, empty_groups
):
    if empty_groups is not ABSENT:
        claims[groups_claim] = empty_groups
    claims[fallback_claim] = ["/registry-admins", "tools-reader"]

    result = provider.validate_token(_sign(signing_key, claims))

    assert result["groups"] == ["/registry-admins", "tools-reader"]
    assert result["scopes"] == []


@pytest.mark.parametrize(
    ("provider", "fallback_claim"),
    [("auth0", "permissions"), ("entra", "roles")],
    indirect=["provider"],
)
@pytest.mark.parametrize("malformed", MALFORMED_GROUPS)
def test_signed_malformed_selected_permissions_or_roles_rejected(
    provider, claims, signing_key, fallback_claim, malformed
):
    claims[fallback_claim] = malformed

    with pytest.raises(ValueError, match="array of strings"):
        provider.validate_token(_sign(signing_key, claims))


@pytest.mark.parametrize(
    ("provider", "fallback_claim"),
    [("auth0", "permissions"), ("entra", "roles")],
    indirect=["provider"],
)
def test_signed_nonempty_groups_take_precedence_without_merging_fallback(
    provider, claims, groups_claim, signing_key, fallback_claim
):
    claims[groups_claim] = ["registry-readers"]
    claims[fallback_claim] = ["registry-admins"]

    result = provider.validate_token(_sign(signing_key, claims))

    assert result["groups"] == ["registry-readers"]
    assert result["scopes"] == []


@pytest.mark.parametrize("provider", ["entra"], indirect=True)
def test_signed_empty_entra_scp_does_not_fall_back_or_convert_roles_to_scopes(
    provider, claims, signing_key
):
    claims.update(scp="", scope="mcp-admin", roles=["/registry-admins"])

    result = provider.validate_token(_sign(signing_key, claims))

    assert result["groups"] == ["/registry-admins"]
    assert result["scopes"] == []


def test_wrong_signed_issuer_rejected_before_returning_groups(
    provider, claims, groups_claim, signing_key
):
    claims.update(iss="https://untrusted.example.test", **{groups_claim: ["registry-admins"]})

    with pytest.raises(ValueError, match="[Ii]ssuer"):
        provider.validate_token(_sign(signing_key, claims))


# Keycloak, Okta, and PingFederate already have signed audience-rejection suites.
@pytest.mark.parametrize("provider", ["auth0", "entra"], indirect=True)
def test_wrong_signed_audience_rejected_before_returning_groups(
    provider, claims, groups_claim, signing_key
):
    claims.update(aud="another-application", **{groups_claim: ["registry-admins"]})

    with pytest.raises(ValueError, match="[Aa]udience"):
        provider.validate_token(_sign(signing_key, claims))


@pytest.mark.parametrize("provider", ["cognito"], indirect=True)
def test_cognito_signed_access_token_rejects_wrong_client_id(provider, claims, signing_key):
    claims.update(client_id="another-application", **{"cognito:groups": ["registry-admins"]})

    with pytest.raises(ValueError, match="client_id.*accepted client list"):
        provider.validate_token(_sign(signing_key, claims))


@pytest.mark.parametrize("provider", ["cognito"], indirect=True)
def test_cognito_signed_id_token_rejects_wrong_audience(provider, claims, signing_key):
    claims.update(token_use="id", aud="another-application")

    with pytest.raises(ValueError, match="[Aa]udience"):
        provider.validate_token(_sign(signing_key, claims))


def test_groups_tampered_after_signing_rejected(provider, claims, groups_claim, signing_key):
    claims[groups_claim] = ["registry-readers"]
    header, _, signature = _sign(signing_key, claims).split(".")
    claims[groups_claim] = ["registry-admins"]
    forged_payload = jwt.utils.base64url_encode(json.dumps(claims).encode()).decode()
    forged_token = f"{header}.{forged_payload}.{signature}"

    with pytest.raises(ValueError, match="[Ss]ignature"):
        provider.validate_token(forged_token)


@pytest.mark.parametrize("provider", ["auth0"], indirect=True)
@pytest.mark.parametrize("malformed", MALFORMED_GROUPS)
def test_auth0_signed_id_token_malformed_groups_cannot_fall_back_to_permissions(
    provider, claims, groups_claim, signing_key, malformed
):
    claims.update(nonce="login-bound-nonce", permissions=["registry-admins"])
    claims[groups_claim] = malformed
    token_data = {"id_token": _sign(signing_key, claims)}

    with pytest.raises(ValueError, match="array of strings"):
        provider.extract_user_from_tokens(token_data, expected_nonce="login-bound-nonce")


@pytest.mark.parametrize("provider", ["auth0"], indirect=True)
@pytest.mark.parametrize(
    "empty_groups",
    [
        pytest.param(ABSENT, id="absent"),
        pytest.param(None, id="null"),
        pytest.param([], id="empty"),
    ],
)
def test_auth0_signed_id_token_empty_groups_keep_permissions_fallback(
    provider, claims, groups_claim, signing_key, empty_groups
):
    claims.update(nonce="login-bound-nonce", permissions=["/registry-admins", "tools-reader"])
    if empty_groups is not ABSENT:
        claims[groups_claim] = empty_groups
    token_data = {"id_token": _sign(signing_key, claims)}

    result = provider.extract_user_from_tokens(token_data, expected_nonce="login-bound-nonce")

    assert result["subject"] == "alice"
    assert result["groups"] == ["/registry-admins", "tools-reader"]
