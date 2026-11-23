"""Property tests: the egress vault key is the IdP's immutable account id.

The per-user egress credential vault stores each user's upstream provider
tokens under one key, computed by ``_canonical_egress_user``. A user reaches
the vault on two paths. The browser consent path presents a session cookie
whose record the OAuth2 login callback wrote, along with the session's vault
key (``egress_user``); the vend path presents a bearer token, either the IdP's
own access token or a token this gateway minted from the session. The
properties pinned here:

* every path derives the same key for one account (agreement);
* every reader of a session (the auth server's cookie path, the registry's raw
  cookie path and the token mint) gives the key stored at login, even after the
  stored ``provider`` or ``username`` changes;
* two accounts with distinct immutable ids get distinct keys even when every
  mutable or reusable claim is identical (injectivity), which covers a Cognito
  username reused after its account is deleted and an Okta login moved to a
  new user;
* an account keeps its key when only those claims change (stability);
* an OIDC login that yields no subject, or a subject that is not a string,
  creates no session;
* a provider's access token without ``sub`` gets no key;
* a minted token carries the session's key whatever the request body says,
  and never a username fallback key;
* an Okta client-credentials token keys on its client id, and ``uid`` counts
  only for Okta;
* the registry's vault principal equals the vend's for one caller, including
  the callers the auth server gives no key (both refuse them) and a session
  with no subject (every path, the registry's raw cookie path included, keys it
  on the one namespaced username-fallback key stored at login);
* a session record without ``egress_user`` owns no vault key, whatever its
  subject.

The immutable id is the token ``sub`` everywhere except Okta access tokens,
whose ``sub`` is the user's editable login and whose ``uid`` is the user id the
Okta id_token puts in ``sub``. Bearer tokens are signed with a local RSA key
and verified for real; on the login path the provider's id_token verification
returns the claims as verified, and the session store runs against an
in-memory collection.
"""

from __future__ import annotations

import asyncio
import functools
import json
import os
import string
import time
from contextlib import (
    ExitStack,
    contextmanager,
)
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import (
    AsyncMock,
    MagicMock,
    patch,
)

import jwt
import pytest
import yaml
from cryptography.hazmat.primitives.asymmetric import rsa
from hypothesis import (
    assume,
    example,
    given,
    settings,
)
from hypothesis import strategies as st
from jwt.algorithms import RSAAlgorithm

os.environ.setdefault("SECRET_KEY", "Zq8v3Lp0Wm7Xr2Tn5Yb9Kc4Hd6Jf1Ga8Qe3Us0Vt")

import session_store  # noqa: E402
from providers.auth0 import Auth0Provider  # noqa: E402
from providers.cognito import CognitoProvider  # noqa: E402
from providers.entra import EntraIdProvider  # noqa: E402
from providers.keycloak import KeycloakProvider  # noqa: E402
from providers.okta import OktaProvider  # noqa: E402
from providers.pingfederate import PingFederateProvider  # noqa: E402

import auth_server.server as server_module  # noqa: E402
import registry.auth.dependencies as registry_dependencies  # noqa: E402
from registry.auth.dependencies import _context_from_internal_token  # noqa: E402
from registry.auth.proxied_token import verify_mcp_proxy_token  # noqa: E402
from registry.egress_auth.fallback_key import username_fallback_key  # noqa: E402
from registry.egress_auth.service import (  # noqa: E402
    canonical_auth_method,
    is_per_user_auth_method,
    vault_principal,
)

pytestmark = [pytest.mark.unit, pytest.mark.auth]


KID: str = "egress-vault-identity-kid"
WEB_CLIENT_ID: str = "gateway-web"
COGNITO_POOL_ID: str = "us-east-1_TESTPOOL"
COGNITO_REGION: str = "us-east-1"
PINGFEDERATE_ISSUER: str = "https://pf.example.com"
OAUTH2_PROVIDERS_YML: Path = (
    Path(__file__).resolve().parents[3] / "auth_server" / "oauth2_providers.yml"
)

# Claims an account holder or IdP administrator can change, or that a later
# account can hold. ``cognito:username`` is the Cognito username, which Cognito
# frees for reuse when its account is deleted; ``login`` doubles as the Okta
# login that an Okta user access token carries in ``sub``.
MUTABLE_CLAIMS: tuple[str, ...] = (
    "preferred_username",
    "email",
    "nickname",
    "login",
    "upn",
    "cognito:username",
)

# Bearer paths: path id -> provider. ``cognito`` is ``CognitoProvider`` (what
# ``/validate`` uses when AUTH_PROVIDER=cognito); the ``cognito-validator-*``
# paths are the two branches of ``SimplifiedCognitoValidator.validate_token``.
BEARER_PATHS: dict[str, str] = {
    "keycloak": "keycloak",
    "entra": "entra",
    "okta": "okta",
    "auth0": "auth0",
    "pingfederate": "pingfederate",
    "cognito": "cognito",
    "cognito-validator-jwt": "cognito",
    "cognito-validator-boto3": "cognito",
}

# Login paths through ``/oauth2/callback/{provider}``: path id -> provider.
# Bare names use the verified id_token, or for Cognito the verified access
# token (``cognito-boto3`` is the GetUser fallback). ``-userinfo`` paths are the
# userInfo fallback taken when there is no id_token, or no Cognito pool is
# configured. GitHub and Google only have the userInfo path. ``corp-oidc`` is
# a provider an operator added to ``oauth2_providers.yml``: the callback has no
# branch for it, so both of its paths read userInfo, and the bare path's token
# response also carries an id_token.
LOGIN_PATHS: dict[str, str] = {
    "keycloak": "keycloak",
    "entra": "entra",
    "okta": "okta",
    "auth0": "auth0",
    "pingfederate": "pingfederate",
    "cognito": "cognito",
    "cognito-boto3": "cognito",
    "keycloak-userinfo": "keycloak",
    "okta-userinfo": "okta",
    "auth0-userinfo": "auth0",
    "pingfederate-userinfo": "pingfederate",
    "cognito-userinfo": "cognito",
    "github": "github",
    "google": "google",
    "corp-oidc": "corp-oidc",
    "corp-oidc-userinfo": "corp-oidc",
}

# Login paths whose token response carries no id_token, from a provider the
# callback does not know speaks OIDC: a missing subject only warns.
NON_OIDC_LOGIN_PATHS: frozenset[str] = frozenset({"github", "google", "corp-oidc-userinfo"})

_OPERATOR_PROVIDER: str = "corp-oidc"
_OPERATOR_PROVIDER_CONFIG: dict[str, Any] = {
    "display_name": "Corp OIDC",
    "client_id": WEB_CLIENT_ID,
    "client_secret": "secret",
    "auth_url": "https://idp.example.com/authorize",
    "token_url": "https://idp.example.com/token",
    "user_info_url": "https://idp.example.com/userinfo",
    "scopes": ["openid", "email", "profile"],
    "response_type": "code",
    "grant_type": "authorization_code",
    "username_claim": "preferred_username",
    "subject_claim": "sub",
    "groups_claim": "groups",
    "email_claim": "email",
    "name_claim": "name",
    "enabled": True,
}

PROVIDERS_WITH_BOTH_PATHS: tuple[str, ...] = (
    "keycloak",
    "entra",
    "okta",
    "auth0",
    "pingfederate",
    "cognito",
)

_ENV_VARS_CLEARED: tuple[str, ...] = (
    "OKTA_AUTH_SERVER_ID",
    "ENTRA_LOGIN_BASE_URL",
    "ENTRA_GRAPH_BASE_URL",
    "ENTRA_SCOPE_FORMAT",
    "ENTRA_APPLICATION_ID_URI",
    "ENTRA_GRAPH_RESOURCE_URL",
    "ENTRA_GROUP_RECOVERY_ENABLED",
)

HYPOTHESIS_SETTINGS = settings(
    max_examples=200,
    derandomize=True,
    database=None,
    report_multiple_bugs=False,
)
# For tests that send a /validate request (and often an /internal/tokens mint)
# per example: 50 examples keep the file fast under -n 8, and the drawn values
# are opaque ids, so more examples reach no new branch. No deadline: the first
# request pays FastAPI's route compilation.
HOP_HYPOTHESIS_SETTINGS = settings(
    max_examples=50,
    derandomize=True,
    database=None,
    report_multiple_bugs=False,
    deadline=None,
)

# The MCP request /validate authorizes in the hop-token tests, as nginx forwards it.
_HOP_ORIGINAL_URL: str = "https://gw.example.com/github/mcp"
# A browser sends Origin on an MCP POST, and /validate refuses a cookie request
# whose Origin is not the gateway's own, so the hop tests send the gateway's.
_HOP_ORIGIN: str = "https://gw.example.com"
_HOP_UPSTREAM: str = "https://api.githubcopilot.com/mcp"
_HOP_BODY: str = '{"jsonrpc":"2.0","id":1,"method":"tools/list"}'
# /validate refuses an MCP request from a caller with no scopes, so every group
# maps to this one; tool authorization itself is allowed in the hop tests.
_HOP_SCOPE: str = "hop-test-scope"

_COLLIDING_PROFILE: dict[str, str] = {
    "preferred_username": "alice",
    "email": "alice@example.com",
    "nickname": "alice",
    "login": "alice@example.com",
    "upn": "alice@example.com",
    "cognito:username": "alice@example.com",
}

# The same account after a rename: every mutable claim has a new value.
_RENAMED_PROFILE: dict[str, str] = {
    "preferred_username": "alice.smith",
    "email": "alice.smith@example.com",
    "nickname": "alice.smith",
    "login": "alice.smith@example.com",
    "upn": "alice.smith@example.com",
    "cognito:username": "alice.smith@example.com",
}

# Cognito username ``alice@example.com`` deleted and recreated: a new ``sub``.
_COGNITO_USERNAME_REUSE = (
    "0f6c-uuid-old-account",
    "0f6c-uuid-new-account",
    _COLLIDING_PROFILE,
    _COLLIDING_PROFILE,
)
# Okta login ``alice@example.com`` moved from user id A to user id B.
_OKTA_LOGIN_REASSIGNED = (
    "00u-alice-a",
    "00u-alice-b",
    _COLLIDING_PROFILE,
    _COLLIDING_PROFILE,
)

_PRIVATE_KEY = rsa.generate_private_key(public_exponent=65537, key_size=2048)
_JWKS: dict[str, Any] = {
    "keys": [
        {
            **json.loads(RSAAlgorithm.to_jwk(_PRIVATE_KEY.public_key())),
            "kid": KID,
            "alg": "RS256",
        }
    ]
}

_ACCOUNT_IDS = st.text(
    alphabet=string.ascii_lowercase + string.digits + "-", min_size=1, max_size=16
)
_CLAIM_VALUES = st.text(
    alphabet=string.ascii_lowercase + string.digits + "._@-", min_size=1, max_size=16
)
_PROFILES = st.fixed_dictionaries(dict.fromkeys(MUTABLE_CLAIMS, _CLAIM_VALUES))


class _FakeSessionCollection:
    """In-memory stand-in for the ``oauth_sessions`` collection."""

    def __init__(self) -> None:
        self.documents: list[dict[str, Any]] = []

    async def insert_one(
        self,
        document: dict[str, Any],
    ) -> None:
        """Store a copy of ``document``."""
        self.documents.append(dict(document))

    async def find_one(
        self,
        query: dict[str, Any],
    ) -> dict[str, Any] | None:
        """Return the document whose ``session_id`` matches, or None."""
        for document in self.documents:
            if document["session_id"] == query["session_id"]:
                return document
        return None


@dataclass
class _Login:
    """What one ``/oauth2/callback`` request produced."""

    status_code: int
    cookie: str | None
    store: _FakeSessionCollection
    set_cookie_headers: list[str]
    location: str | None


def _clean_env() -> dict[str, str]:
    """Return the process environment without provider-shaping variables.

    Returns:
        A copy of ``os.environ`` with ``_ENV_VARS_CLEARED`` removed.
    """
    return {k: v for k, v in os.environ.items() if k not in _ENV_VARS_CLEARED}


@functools.cache
def _provider(
    name: str,
):
    """Build the named provider once, with a clean provider environment.

    Args:
        name: Provider name, one of ``PROVIDERS_WITH_BOTH_PATHS``.

    Returns:
        A provider configured with ``WEB_CLIENT_ID`` as its client.
    """
    with patch.dict(os.environ, _clean_env(), clear=True):
        if name == "keycloak":
            return KeycloakProvider(
                keycloak_url="http://keycloak:8080",
                realm="mcp-gateway",
                client_id=WEB_CLIENT_ID,
                client_secret="secret",
            )
        if name == "entra":
            return EntraIdProvider(
                tenant_id="tenant-abc", client_id=WEB_CLIENT_ID, client_secret="secret"
            )
        if name == "okta":
            return OktaProvider("dev-123456.okta.com", WEB_CLIENT_ID, "secret")
        if name == "auth0":
            return Auth0Provider(
                domain="tenant.auth0.com", client_id=WEB_CLIENT_ID, client_secret="secret"
            )
        if name == "cognito":
            return CognitoProvider(
                user_pool_id=COGNITO_POOL_ID,
                client_id=WEB_CLIENT_ID,
                client_secret="secret",
                region=COGNITO_REGION,
            )
        if name == "pingfederate":
            pingfederate = PingFederateProvider(
                base_url=PINGFEDERATE_ISSUER, client_id=WEB_CLIENT_ID, client_secret="secret"
            )
            # Stands in for the discovery document fetch.
            pingfederate._get_openid_configuration = lambda: {"issuer": PINGFEDERATE_ISSUER}
            return pingfederate
    raise ValueError(f"unknown provider {name}")


@functools.cache
def _yml_provider_config(
    name: str,
) -> dict[str, Any]:
    """Return the provider config the callback reads for ``name``.

    For a shipped provider this is its ``oauth2_providers.yml`` entry: the claim
    mapping (``username_claim``, ``subject_claim`` and so on) is the production
    configuration; only the endpoint URL is replaced. The operator-added
    provider gets ``_OPERATOR_PROVIDER_CONFIG``.

    Args:
        name: Provider key in the YAML file, or ``_OPERATOR_PROVIDER``.

    Returns:
        The provider config dict the callback reads.
    """
    if name == _OPERATOR_PROVIDER:
        return dict(_OPERATOR_PROVIDER_CONFIG)
    shipped = yaml.safe_load(OAUTH2_PROVIDERS_YML.read_text())["providers"][name]
    return {
        **shipped,
        "client_id": WEB_CLIENT_ID,
        "user_info_url": "https://idp.example.com/userinfo",
    }


@functools.cache
def _client():
    """Return a TestClient bound to the auth server app.

    Returns:
        The shared ``TestClient``.
    """
    from fastapi.testclient import TestClient

    return TestClient(server_module.app, raise_server_exceptions=False)


def _sign(
    claims: dict[str, Any],
) -> str:
    """Sign ``claims`` with the test RSA key, valid for ten minutes.

    Args:
        claims: Token claims without ``iat``/``exp``.

    Returns:
        The encoded RS256 JWT.
    """
    now = int(time.time())
    return jwt.encode(
        {**claims, "iat": now, "exp": now + 600},
        _PRIVATE_KEY,
        algorithm="RS256",
        headers={"kid": KID},
    )


def _drop_absent(
    claims: dict[str, Any],
) -> dict[str, Any]:
    """Remove claims whose value is None (an account with no subject).

    Args:
        claims: Claims that may hold None values.

    Returns:
        The claims without the None-valued entries.
    """
    return {k: v for k, v in claims.items() if v is not None}


def _id_claims(
    account_id: str | None,
    profile: dict[str, str],
) -> dict[str, Any]:
    """The claims an id_token or OIDC userInfo response carries for an account.

    Args:
        account_id: The account's immutable id, or None for no subject.
        profile: The account's mutable claims.

    Returns:
        The identity claims (no protocol claims such as ``iss``).
    """
    return _drop_absent({**profile, "sub": account_id})


def _access_claims(
    provider: str,
    account_id: str | None,
    profile: dict[str, str],
) -> dict[str, Any]:
    """The identity claims a user access token from ``provider`` carries.

    An Okta user access token puts the user's login in ``sub`` and the user id
    in ``uid``. A Cognito access token carries the Cognito username in
    ``username``. Every other provider puts the account id in ``sub``.

    Args:
        provider: Provider name.
        account_id: The account's immutable id, or None for no subject.
        profile: The account's mutable claims.

    Returns:
        The identity claims (no protocol claims such as ``iss``).
    """
    if provider == "okta":
        return _drop_absent({**profile, "sub": profile["login"], "uid": account_id})
    if provider == "cognito":
        return _drop_absent({**profile, "sub": account_id, "username": profile["cognito:username"]})
    return _id_claims(account_id, profile)


def _cognito_access_token(
    account_id: str | None,
    profile: dict[str, str],
) -> str:
    """Sign a Cognito user access token for the account.

    Args:
        account_id: The account's ``sub``, or None for no subject.
        profile: The account's mutable claims.

    Returns:
        The encoded access token.
    """
    return _sign(
        {
            **_access_claims("cognito", account_id, profile),
            "iss": _provider("cognito").issuer,
            "token_use": "access",
            "client_id": WEB_CLIENT_ID,
        }
    )


def _cognito_get_user_client(
    account_id: str | None,
    profile: dict[str, str],
) -> MagicMock:
    """A boto3 ``cognito-idp`` client whose ``get_user`` returns the account.

    Args:
        account_id: The account's ``sub`` attribute, or None for no subject.
        profile: The account's mutable claims.

    Returns:
        The mock client.
    """
    attributes = [{"Name": "email", "Value": profile["email"]}]
    if account_id is not None:
        attributes.append({"Name": "sub", "Value": account_id})
    client = MagicMock()
    client.get_user.return_value = {
        "Username": profile["cognito:username"],
        "UserAttributes": attributes,
        "UserStatus": "CONFIRMED",
    }
    return client


def _signed_bearer(
    provider: str,
    account_id: str | None,
    profile: dict[str, str],
    extra_claims: dict[str, Any] | None = None,
) -> str:
    """Sign the user access token ``provider`` issues for the account.

    Args:
        provider: A provider in ``PROVIDERS_WITH_BOTH_PATHS``.
        account_id: The account's immutable id.
        profile: The account's mutable claims.
        extra_claims: Claims added to the token, overriding the account's.

    Returns:
        The encoded RS256 token, with the issuer and audience the provider checks.
    """
    instance = _provider(provider)
    claims: dict[str, Any] = _access_claims(provider, account_id, profile)
    if provider == "keycloak":
        claims.update(iss=instance.realm_url, aud=WEB_CLIENT_ID, azp=WEB_CLIENT_ID)
    elif provider == "entra":
        claims.update(iss=instance.issuer_v2, aud=WEB_CLIENT_ID, azp=WEB_CLIENT_ID)
    elif provider == "cognito":
        claims.update(iss=instance.issuer, token_use="access", client_id=WEB_CLIENT_ID)
    elif provider == "pingfederate":
        claims.update(iss=PINGFEDERATE_ISSUER, aud=WEB_CLIENT_ID)
    else:
        claims.update(iss=instance.issuer, aud=WEB_CLIENT_ID)
    claims.update(extra_claims or {})
    return _sign(claims)


def _bearer_result(
    bearer_path: str,
    account_id: str | None,
    profile: dict[str, str],
    extra_claims: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return the validation result a bearer path builds for the account.

    Args:
        bearer_path: Key of ``BEARER_PATHS``.
        account_id: The account's immutable id.
        profile: The account's mutable claims.
        extra_claims: Claims added to the token, overriding the account's.

    Returns:
        The validation result.
    """
    if bearer_path.startswith("cognito-validator-"):
        return _cognito_validator_result(bearer_path, account_id, profile, extra_claims)
    instance = _provider(BEARER_PATHS[bearer_path])
    token = _signed_bearer(BEARER_PATHS[bearer_path], account_id, profile, extra_claims)
    with patch.object(instance, "get_jwks", return_value=_JWKS):
        result = instance.validate_token(token)
    assert result["valid"] is True, result
    return result


def _cognito_validator_result(
    bearer_path: str,
    account_id: str | None,
    profile: dict[str, str],
    extra_claims: dict[str, Any] | None,
) -> dict[str, Any]:
    """Run ``SimplifiedCognitoValidator.validate_token`` on the account's token.

    ``cognito-validator-jwt`` verifies a signed access token against the test
    JWKS. ``cognito-validator-boto3`` makes JWT verification fail so the
    validator falls back to boto3 ``GetUser``.

    Args:
        bearer_path: ``cognito-validator-jwt`` or ``cognito-validator-boto3``.
        account_id: The account's ``sub``.
        profile: The account's mutable claims.
        extra_claims: Claims added to the JWT (ignored on the boto3 branch).

    Returns:
        The validation result.
    """
    validator = server_module.SimplifiedCognitoValidator(COGNITO_REGION)
    claims = {
        **_access_claims("cognito", account_id, profile),
        "iss": _provider("cognito").issuer,
        "token_use": "access",
        "client_id": WEB_CLIENT_ID,
        **(extra_claims or {}),
    }
    token = _sign(claims)
    with ExitStack() as stack:
        stack.enter_context(patch.object(validator, "_get_jwks", return_value=_JWKS))
        if bearer_path == "cognito-validator-boto3":
            stack.enter_context(
                patch.object(
                    validator,
                    "validate_jwt_token",
                    side_effect=ValueError("JWKS unreachable"),
                )
            )
            stack.enter_context(
                patch.object(
                    validator,
                    "_get_cognito_client",
                    return_value=_cognito_get_user_client(account_id, profile),
                )
            )
        result = validator.validate_token(token, COGNITO_POOL_ID, WEB_CLIENT_ID, COGNITO_REGION)
    assert result["valid"] is True, result
    expected_method = "boto3" if bearer_path == "cognito-validator-boto3" else "jwt"
    assert result["method"] == expected_method, result
    return result


def _bearer_key(
    bearer_path: str,
    account_id: str | None,
    profile: dict[str, str],
    extra_claims: dict[str, Any] | None = None,
) -> str:
    """Return the vault key a bearer path derives for the account.

    Args:
        bearer_path: Key of ``BEARER_PATHS``.
        account_id: The account's immutable id.
        profile: The account's mutable claims.
        extra_claims: Claims added to the token, overriding the account's.

    Returns:
        ``_canonical_egress_user`` of the validation result.
    """
    result = _bearer_result(bearer_path, account_id, profile, extra_claims)
    return server_module._canonical_egress_user(result)


@contextmanager
def _session_store(
    store: _FakeSessionCollection,
):
    """Back the real ``session_store`` functions with ``store``.

    Args:
        store: The in-memory session collection.

    Yields:
        None.
    """
    with patch.object(session_store, "_get_collection", AsyncMock(return_value=store)):
        yield


def _login_patches(
    login_path: str,
    account_id: str | None,
    profile: dict[str, str],
    getuser_has_sub: bool = True,
) -> list[Any]:
    """Build the patches that make the IdP return the account at ``login_path``.

    The code exchange, the userInfo fetch, id_token verification and (on
    ``cognito-boto3``) the JWKS check are replaced so the claims arrive as
    verified. The Cognito access token is signed and verified for real. The
    callback's own mapping from claims to session fields runs unchanged.

    Args:
        login_path: Key of ``LOGIN_PATHS``.
        account_id: The account's immutable id, or None for no subject.
        profile: The account's mutable claims.
        getuser_has_sub: Whether the boto3 ``GetUser`` response on
            ``cognito-boto3`` carries the ``sub`` attribute.

    Returns:
        The patches to enter.
    """
    provider = LOGIN_PATHS[login_path]
    provider_config = _yml_provider_config(provider)
    userinfo_path = login_path.endswith("-userinfo") or login_path in NON_OIDC_LOGIN_PATHS

    userinfo = _id_claims(account_id, profile)
    if provider == "cognito":
        userinfo["username"] = profile["cognito:username"]
    subject_claim = provider_config.get("subject_claim")
    if subject_claim and "sub" in userinfo:
        userinfo[subject_claim] = userinfo.pop("sub")

    token_data: dict[str, Any] = {"access_token": "opaque-access-token"}
    if provider == "cognito" and not userinfo_path:
        token_data["access_token"] = _cognito_access_token(account_id, profile)
    elif not userinfo_path:
        token_data["id_token"] = "opaque-id-token"

    providers_by_name = {name: _provider(name) for name in PROVIDERS_WITH_BOTH_PATHS}
    config = dict(server_module.OAUTH2_CONFIG)
    config["providers"] = {**config.get("providers", {}), provider: provider_config}

    env = _clean_env()
    if provider == "cognito" and not userinfo_path:
        env.update(COGNITO_USER_POOL_ID=COGNITO_POOL_ID, AWS_REGION=COGNITO_REGION)
    else:
        env.pop("COGNITO_USER_POOL_ID", None)

    validator_cls = server_module.SimplifiedCognitoValidator
    patches: list[Any] = [
        patch.dict(os.environ, env, clear=True),
        patch.object(server_module, "OAUTH2_CONFIG", config),
        patch.object(server_module, "IDP_USER_GROUP_FALLBACK_ENABLED_PROVIDERS", []),
        patch.object(server_module, "exchange_code_for_token", AsyncMock(return_value=token_data)),
        patch.object(server_module, "get_user_info", AsyncMock(return_value=userinfo)),
        patch.object(
            server_module,
            "get_auth_provider",
            side_effect=lambda name=None: providers_by_name[name],
        ),
        patch.object(validator_cls, "_get_jwks", return_value=_JWKS),
        patch("group_filter.filter_session_groups", AsyncMock(side_effect=lambda g, **_: g)),
    ]
    if login_path == "cognito-boto3":
        patches.append(
            patch.object(
                validator_cls,
                "validate_jwt_token",
                side_effect=ValueError("JWKS unreachable"),
            )
        )
        patches.append(
            patch.object(
                validator_cls,
                "_get_cognito_client",
                return_value=_cognito_get_user_client(
                    account_id if getuser_has_sub else None, profile
                ),
            )
        )
    verified = _id_claims(account_id, profile)
    patches.extend(
        patch.object(instance, "validate_id_token", return_value=verified)
        for instance in providers_by_name.values()
    )
    return patches


def _login(
    login_path: str,
    account_id: str | None,
    profile: dict[str, str],
    getuser_has_sub: bool = True,
) -> _Login:
    """Drive ``/oauth2/callback/{provider}`` for the account.

    Args:
        login_path: Key of ``LOGIN_PATHS``.
        account_id: The account's immutable id, or None for no subject.
        profile: The account's mutable claims.
        getuser_has_sub: Whether the boto3 ``GetUser`` response on
            ``cognito-boto3`` carries the ``sub`` attribute.

    Returns:
        The response status, the session cookie it set (None when it set
        none) and the session store it wrote to.
    """
    provider = LOGIN_PATHS[login_path]
    temp_cookie = server_module.signer.dumps(
        {
            "state": "state-1",
            "provider": provider,
            "callback_uri": f"http://localhost:8888/oauth2/callback/{provider}",
            "nonce": "nonce-1",
            "code_verifier": "verifier-1",
        }
    )
    store = _FakeSessionCollection()
    client = _client()
    with ExitStack() as stack:
        for active in _login_patches(login_path, account_id, profile, getuser_has_sub):
            stack.enter_context(active)
        stack.enter_context(_session_store(store))
        response = client.get(
            f"/oauth2/callback/{provider}",
            params={"code": "code-1", "state": "state-1"},
            headers={"Cookie": f"oauth2_temp_session={temp_cookie}"},
            follow_redirects=False,
        )
    client.cookies.clear()
    return _Login(
        status_code=response.status_code,
        cookie=response.cookies.get(server_module.settings.session_cookie_effective_name),
        store=store,
        set_cookie_headers=response.headers.get_list("set-cookie"),
        location=response.headers.get("location"),
    )


def _session_cookie_header(
    cookie: str | None,
) -> dict[str, str]:
    """Return the ``Cookie`` header a browser sends with a login's session cookie.

    Args:
        cookie: The session cookie value the login set.

    Returns:
        The header, under the effective session cookie name.
    """
    return {"Cookie": f"{server_module.settings.session_cookie_effective_name}={cookie}"}


def _cookie_result(
    login: _Login,
) -> dict[str, Any]:
    """Return the validation result the consent path builds from a login's cookie.

    Runs the auth server's own ``validate_session_cookie`` (what ``/validate``
    uses for a cookie) against the session the login wrote.

    Args:
        login: A login that created a session.

    Returns:
        The session-cookie validation result.
    """
    assert login.cookie, (login.status_code, login.store.documents)
    with (
        _session_store(login.store),
        patch.object(server_module, "map_groups_to_scopes", AsyncMock(return_value=[])),
    ):
        result = asyncio.run(server_module.validate_session_cookie(login.cookie))
    assert result["method"] == "session_cookie", result
    return result


def _cookie_key(
    login: _Login,
) -> str:
    """Return the vault key the consent path derives from a login's cookie.

    Args:
        login: A login that created a session.

    Returns:
        ``_canonical_egress_user`` of the session-cookie validation result.
    """
    return server_module._canonical_egress_user(_cookie_result(login))


def _login_key(
    login_path: str,
    account_id: str,
    profile: dict[str, str],
) -> str:
    """Log the account in at ``login_path`` and return its cookie-path key.

    Args:
        login_path: Key of ``LOGIN_PATHS``.
        account_id: The account's immutable id.
        profile: The account's mutable claims.

    Returns:
        The vault key of the resulting session.
    """
    login = _login(login_path, account_id, profile)
    assert login.status_code == 302, (login_path, login.status_code)
    return _cookie_key(login)


def _internal_auth_headers() -> dict[str, str]:
    """Return an Authorization header carrying a valid internal JWT.

    Returns:
        The header dict for ``/internal/tokens``.
    """
    from registry.auth.internal import generate_internal_token

    token = generate_internal_token(subject="test-suite", purpose="unit-test")
    return {"Authorization": f"Bearer {token}"}


def _mint_token(
    user_context: dict[str, Any],
    store: _FakeSessionCollection,
) -> str:
    """Mint a self-signed user token through the real ``/internal/tokens``.

    Args:
        user_context: The ``user_context`` body field, as the registry sends it.
        store: Session collection a ``session_id`` in the context resolves in.

    Returns:
        The minted access token.
    """
    body = {
        "user_context": {"scopes": [_HOP_SCOPE], "groups": [], **user_context},
        "requested_scopes": [],
        "expires_in_hours": 1,
        "description": "vault identity property test",
    }
    server_module.user_token_generation_counts.clear()
    with (
        _session_store(store),
        patch.object(server_module, "map_groups_to_scopes", AsyncMock(return_value=[_HOP_SCOPE])),
        # The mint's audit record would otherwise go to the durable audit store.
        patch.object(server_module, "_emit_token_mint_audit", AsyncMock(return_value=None)),
    ):
        response = _client().post("/internal/tokens", json=body, headers=_internal_auth_headers())
    assert response.status_code == 200, response.text
    return response.json()["access_token"]


def _session_mint_context(
    login: _Login,
) -> dict[str, Any]:
    """Return the ``user_context`` the registry sends to mint from a login's session.

    Args:
        login: A login that created a session.

    Returns:
        The session-backed ``user_context``.
    """
    document = login.store.documents[0]
    return {
        "username": document["username"],
        "auth_method": "oauth2",
        "provider": document["provider"],
        "session_id": server_module.signer.loads(login.cookie),
    }


def _minted_token_key(
    login: _Login,
) -> str:
    """Mint a user token from the login's session and return its vault key.

    Calls ``/internal/tokens`` the way the registry does for a session-backed
    user, then validates the token as the vend path would.

    Args:
        login: A login that created a session.

    Returns:
        ``_canonical_egress_user`` of the minted token's validation result.
    """
    token = _mint_token(_session_mint_context(login), login.store)
    result = server_module.validator.validate_self_signed_token(token)
    return server_module._canonical_egress_user(result)


@dataclass
class _HopTokens:
    """The two internal tokens one ``/validate`` response carried, as verified downstream."""

    registry_context: dict[str, Any]
    mcp_proxy_claims: dict[str, Any]


def _registry_context(
    registry_token: str,
    store: _FakeSessionCollection,
) -> dict[str, Any]:
    """Build the registry's user context from an ``X-Internal-Token-Registry``.

    Runs the registry's own ``_context_from_internal_token``, which verifies the
    token and, for a session-backed token, resolves the session through the
    registry's session store. Only the stores are replaced: sessions read from
    ``store`` and the scope repository maps every group to nothing.

    Args:
        registry_token: The token ``/validate`` minted for the registry hop.
        store: The session collection the login wrote.

    Returns:
        The registry user context.
    """
    with (
        patch("registry.auth.session_store._get_collection", AsyncMock(return_value=store)),
        patch(
            "registry.auth.dependencies.map_cognito_groups_to_scopes",
            AsyncMock(return_value=[]),
        ),
        patch(
            "registry.auth.dependencies.resolve_scope_access",
            AsyncMock(return_value=SimpleNamespace(servers=[], tools={})),
        ),
        patch(
            "registry.auth.dependencies.get_ui_permissions_for_user",
            AsyncMock(return_value={}),
        ),
    ):
        request = SimpleNamespace(state=SimpleNamespace())
        return asyncio.run(_context_from_internal_token(request, registry_token))


def _validate_hop_tokens(
    credential_headers: dict[str, str],
    store: _FakeSessionCollection,
    bearer_provider: str | None = None,
) -> _HopTokens:
    """Send one MCP request's ``/validate`` and verify both internal tokens it mints.

    The request carries what nginx forwards for a browser's MCP POST on a
    registry location: the method, the gateway's own Origin, the original URL
    and body, the resolved upstream (which mints the mcp-proxy token the vend
    reads) and the registry-API marker (which mints the registry token). Tool
    authorization is allowed outright, since the test is about identity, not
    access.

    Args:
        credential_headers: The caller's ``Authorization`` or ``Cookie`` header.
        store: The session collection a cookie resolves in.
        bearer_provider: Provider that verifies a raw IdP bearer, or None.

    Returns:
        The registry user context and the vend's verified mcp-proxy claims.
    """
    marker = server_module.settings.auth_server_nginx_marker_secret
    headers = {
        **credential_headers,
        "X-Original-URL": _HOP_ORIGINAL_URL,
        "X-Original-Method": "POST",
        "Origin": _HOP_ORIGIN,
        "X-Body": _HOP_BODY,
        "X-Resolved-Upstream": _HOP_UPSTREAM,
        "X-Registry-Api-Auth": "1",
        "X-Validate-Source-Secret": marker,
    }
    client = _client()
    with ExitStack() as stack:
        stack.enter_context(_session_store(store))
        stack.enter_context(
            patch.object(
                server_module,
                "_gateway_trusted_origins",
                return_value=frozenset({_HOP_ORIGIN}),
            )
        )
        stack.enter_context(
            patch.object(
                server_module, "map_groups_to_scopes", AsyncMock(return_value=[_HOP_SCOPE])
            )
        )
        stack.enter_context(
            patch.object(server_module, "validate_server_tool_access", AsyncMock(return_value=True))
        )
        # The MCP access record would otherwise go to the durable audit store.
        stack.enter_context(patch.object(server_module, "get_mcp_logger", return_value=None))
        if bearer_provider is not None:
            instance = _provider(bearer_provider)
            stack.enter_context(patch.object(instance, "get_jwks", return_value=_JWKS))
            stack.enter_context(
                patch.object(server_module, "get_auth_provider", return_value=instance)
            )
        response = client.get("/validate", headers=headers)
    client.cookies.clear()
    assert response.status_code == 200, response.text
    return _HopTokens(
        registry_context=_registry_context(response.headers["X-Internal-Token-Registry"], store),
        mcp_proxy_claims=verify_mcp_proxy_token(response.headers["X-Internal-Token"]),
    )


@st.composite
def _distinct_accounts(
    draw: st.DrawFn,
) -> tuple[str, str, dict[str, str], dict[str, str]]:
    """Draw two distinct account ids and their profiles; often the profiles match.

    Args:
        draw: Hypothesis draw function.

    Returns:
        ``(id_a, id_b, profile_a, profile_b)`` with ``id_a != id_b``.
    """
    id_a = draw(_ACCOUNT_IDS)
    id_b = draw(_ACCOUNT_IDS.filter(lambda s: s != id_a))
    profile_a = draw(_PROFILES)
    profile_b = draw(st.one_of(st.just(profile_a), _PROFILES))
    return id_a, id_b, profile_a, profile_b


def _paths_for(
    provider: str,
) -> tuple[list[str], list[str]]:
    """Return the login paths and bearer paths of ``provider``.

    Args:
        provider: Provider name.

    Returns:
        ``(login_paths, bearer_paths)``.
    """
    logins = [path for path, name in LOGIN_PATHS.items() if name == provider]
    bearers = [path for path, name in BEARER_PATHS.items() if name == provider]
    return logins, bearers


@pytest.mark.parametrize("provider", PROVIDERS_WITH_BOTH_PATHS)
@HYPOTHESIS_SETTINGS
@given(account_id=_ACCOUNT_IDS, profile=_PROFILES)
@example(account_id="0f6c-uuid-new-account", profile=_COLLIDING_PROFILE)
@example(account_id="00u-alice-b", profile=_COLLIDING_PROFILE)
def test_every_path_derives_the_same_vault_key_for_one_account(provider, account_id, profile):
    """Cookie and bearer paths derive one vault key, the immutable account id.

    For every provider with both paths and every account, each login path's
    session-cookie key and each bearer path's key equal the account id.
    """
    logins, bearers = _paths_for(provider)
    keys = {f"login:{path}": _login_key(path, account_id, profile) for path in logins}
    keys.update({f"bearer:{path}": _bearer_key(path, account_id, profile) for path in bearers})
    assert set(keys.values()) == {account_id}, (provider, account_id, keys)


@pytest.mark.parametrize("bearer_path", list(BEARER_PATHS))
@HYPOTHESIS_SETTINGS
@given(accounts=_distinct_accounts())
@example(accounts=_COGNITO_USERNAME_REUSE)
@example(accounts=_OKTA_LOGIN_REASSIGNED)
def test_bearer_vault_key_injective_over_account_id(bearer_path, accounts):
    """Distinct account ids get distinct bearer-path vault keys.

    For every bearer path and every two accounts with distinct immutable ids,
    whatever their mutable claims (identical in the pinned reuse cases), the
    derived keys differ.
    """
    id_a, id_b, profile_a, profile_b = accounts
    key_a = _bearer_key(bearer_path, id_a, profile_a)
    key_b = _bearer_key(bearer_path, id_b, profile_b)
    assert key_a != key_b, (bearer_path, key_a, accounts)


@pytest.mark.parametrize("login_path", list(LOGIN_PATHS))
@HYPOTHESIS_SETTINGS
@given(accounts=_distinct_accounts())
@example(accounts=_COGNITO_USERNAME_REUSE)
@example(accounts=_OKTA_LOGIN_REASSIGNED)
def test_session_vault_key_injective_over_account_id(login_path, accounts):
    """Distinct account ids get distinct session-cookie vault keys.

    For every login path and every two accounts with distinct immutable ids,
    whatever their mutable claims, the sessions the logins create key the
    vault differently.
    """
    id_a, id_b, profile_a, profile_b = accounts
    key_a = _login_key(login_path, id_a, profile_a)
    key_b = _login_key(login_path, id_b, profile_b)
    assert key_a != key_b, (login_path, key_a, accounts)


@pytest.mark.parametrize("bearer_path", list(BEARER_PATHS))
@HYPOTHESIS_SETTINGS
@given(account_id=_ACCOUNT_IDS, before=_PROFILES, after=_PROFILES)
@example(account_id="00u-alice-a", before=_COLLIDING_PROFILE, after=_RENAMED_PROFILE)
def test_bearer_vault_key_stable_under_mutable_claim_edits(bearer_path, account_id, before, after):
    """Changing only mutable claims keeps the bearer-path vault key.

    For every bearer path, account id and two mutable-claim sets, the key is
    the same before and after the edit.
    """
    key_before = _bearer_key(bearer_path, account_id, before)
    key_after = _bearer_key(bearer_path, account_id, after)
    assert key_before == key_after, (bearer_path, key_before, key_after)


@pytest.mark.parametrize("login_path", list(LOGIN_PATHS))
@HYPOTHESIS_SETTINGS
@given(account_id=_ACCOUNT_IDS, before=_PROFILES, after=_PROFILES)
@example(account_id="00u-alice-a", before=_COLLIDING_PROFILE, after=_RENAMED_PROFILE)
def test_session_vault_key_stable_under_mutable_claim_edits(login_path, account_id, before, after):
    """Changing only mutable claims keeps the session-cookie vault key.

    For every login path, account id and two mutable-claim sets, the session
    created before the edit and the one created after key the vault the same.
    """
    key_before = _login_key(login_path, account_id, before)
    key_after = _login_key(login_path, account_id, after)
    assert key_before == key_after, (login_path, key_before, key_after)


@pytest.mark.parametrize(
    "login_path",
    [path for path in LOGIN_PATHS if path not in NON_OIDC_LOGIN_PATHS],
)
@HYPOTHESIS_SETTINGS
@given(profile=_PROFILES)
@example(profile=_COLLIDING_PROFILE)
def test_oidc_login_without_subject_is_refused(login_path, profile):
    """An OIDC login whose IdP response has no subject creates no session.

    For every OIDC login path and every set of mutable claims, a login whose
    token or userInfo response omits the subject returns 401, sets no session
    cookie and writes no session record. That includes the operator-added
    provider, which the callback recognizes as OIDC only by the id_token in its
    token response.
    """
    login = _login(login_path, None, profile)
    assert login.status_code == 401, (login_path, login.status_code, login.store.documents)
    assert login.cookie is None
    assert login.store.documents == []


# Account ids a provider can return when its subject_claim names a value users
# choose: each is the fallback key of some subject-less login.
_FALLBACK_SHAPED_IDS = ("name:github:alice", "name:")


@pytest.mark.parametrize("login_path", list(LOGIN_PATHS))
@pytest.mark.parametrize("account_id", _FALLBACK_SHAPED_IDS)
def test_login_with_a_subject_shaped_like_a_fallback_key_is_refused(login_path, account_id):
    """A login whose subject starts with the fallback-key prefix creates no session.

    A provider whose ``subject_claim`` names a user-chosen value could return
    ``name:github:alice`` as an account id, which is the vault key of a GitHub
    login with no subject. For every login path, such a login returns 401, sets
    no session cookie and writes no session record.
    """
    login = _login(login_path, account_id, _COLLIDING_PROFILE)
    assert login.status_code == 401, (login_path, login.status_code, login.store.documents)
    assert login.cookie is None
    assert login.store.documents == []


# Login paths whose subject comes from a verified id_token's ``sub`` as is.
# ``map_user_info`` turns a userInfo subject into a string, and a Cognito access
# token with a non-string ``sub`` fails JWT verification, so neither is here.
_ID_TOKEN_LOGIN_PATHS: tuple[str, ...] = ("keycloak", "entra", "okta", "auth0", "pingfederate")
_NON_STRING_SUBJECTS: tuple[Any, ...] = (12345, 0, True, False, 1.5, ["sub-1"], {"id": "sub-1"})


@pytest.mark.parametrize("login_path", _ID_TOKEN_LOGIN_PATHS)
@pytest.mark.parametrize("subject", _NON_STRING_SUBJECTS, ids=repr)
def test_login_with_a_non_string_subject_is_refused(login_path, subject, caplog):
    """A login whose id_token ``sub`` is not a string creates no session.

    The callback stores the subject as the session's vault key, and a key that
    is not a string keys nothing, so the user would pass login and then be
    refused by every vault route with nothing logged. For every id_token login
    path and every non-string ``sub``, the login returns 401, sets no cookie,
    writes no session record and logs why.
    """
    with caplog.at_level("WARNING"):
        login = _login(login_path, subject, _COLLIDING_PROFILE)

    assert login.status_code == 401, (login_path, login.status_code, login.store.documents)
    assert login.cookie is None
    assert login.store.documents == []
    assert any("subject that is not a string" in r.getMessage() for r in caplog.records)


_FALLBACK_LOGIN_PROVIDERS = st.sampled_from(["github", "google", "corp-oidc"])
_REALISTIC_ACCOUNT_IDS = st.one_of(
    st.uuids().map(str),
    st.integers(min_value=0, max_value=10**12).map(str),
    st.emails(),
    st.text(alphabet="0123456789abcdef", min_size=8, max_size=24).map(lambda s: f"auth0|{s}"),
    st.text(alphabet=string.ascii_letters + string.digits, min_size=17, max_size=17).map(
        lambda s: f"00u{s}"
    ),
    st.text(alphabet=string.ascii_letters + string.digits + "-_", min_size=43, max_size=43),
    st.sampled_from(_FALLBACK_SHAPED_IDS),
)


def _fallback_result(
    provider: str,
    username: str,
) -> dict[str, Any]:
    """The session-cookie validation result of a login that mapped no subject.

    The callback stored the username fallback key as the session's
    ``egress_user``.
    """
    return {
        "method": "session_cookie",
        "username": username,
        "data": {
            "provider": provider,
            "username": username,
            "egress_user": username_fallback_key(provider, username),
        },
    }


def _account_id_results(
    account_id: str,
) -> dict[str, dict[str, Any]]:
    """Every validation-result shape that carries ``account_id`` as an IdP account id.

    A gateway-minted token's ``egress_user`` is not one of them: only the mint
    stamps it, from the session's stored key or a non-fallback body value, so a
    fallback-shaped value there is the session's own fallback key (see
    ``test_session_without_subject_keys_every_path_on_one_fallback_key``).
    """
    return {
        "bearer-sub": {"method": "keycloak", "username": "u", "data": {"sub": account_id}},
        "okta-uid": {"method": "okta", "username": "u", "data": {"sub": "u", "uid": account_id}},
    }


@HYPOTHESIS_SETTINGS
@given(
    provider=_FALLBACK_LOGIN_PROVIDERS, username=_CLAIM_VALUES, account_id=_REALISTIC_ACCOUNT_IDS
)
@example(provider="github", username="alice", account_id="name:github:alice")
@example(provider="corp-oidc", username="bob", account_id="name:corp-oidc:bob")
@example(provider="github", username="12345", account_id="12345")
def test_fallback_key_never_equals_an_account_ids_key(provider, username, account_id):
    """Property: no account id keys the vault where a subject-less login's fallback key does.

    For every fallback login provider, username and account id, including ids a
    user-chosen ``subject_claim`` could shape like a fallback key, the key
    ``_canonical_egress_user`` derives for that account id on every path differs
    from the fallback key of the subject-less login.
    """
    fallback = server_module._canonical_egress_user(_fallback_result(provider, username))
    assert fallback == username_fallback_key(provider, username)
    for name, result in _account_id_results(account_id).items():
        key = server_module._canonical_egress_user(result)
        assert key != fallback, (name, account_id, key)


@pytest.mark.parametrize("login_path", sorted(NON_OIDC_LOGIN_PATHS))
@HYPOTHESIS_SETTINGS
@given(profile=_PROFILES)
def test_non_oidc_login_without_subject_still_creates_session(login_path, profile):
    """A non-OIDC login without a subject still logs in, with no subject stored.

    For GitHub, Google and an operator-added provider whose token response has
    no id_token, a userInfo response missing the configured ``subject_claim``
    still creates a session, and the session record carries no ``subject``.
    """
    login = _login(login_path, None, profile)
    assert login.status_code == 302, (login_path, login.status_code)
    assert login.cookie
    assert len(login.store.documents) == 1
    assert "subject" not in login.store.documents[0]


@HYPOTHESIS_SETTINGS
@given(client_id=_ACCOUNT_IDS)
@example(client_id="0oa-machine-client")
def test_okta_client_credentials_token_keys_on_its_sub(client_id):
    """An Okta client-credentials token, which has no ``uid``, keys on ``sub``.

    For every client id, a machine token whose ``sub`` and ``cid`` are the
    client id derives the client id as its vault key and its audit subject.
    """
    instance = _provider("okta")
    token = _sign(
        {
            "iss": instance.issuer,
            "aud": WEB_CLIENT_ID,
            "sub": client_id,
            "cid": client_id,
            "scp": ["mcp-servers/read"],
        }
    )
    with patch.object(instance, "get_jwks", return_value=_JWKS):
        result = instance.validate_token(token)
    assert server_module._canonical_egress_user(result) == client_id
    assert server_module._audit_identity_claims(result)["subject"] == client_id


# ``uid`` values an Okta user access token can carry that name no user id;
# ``_ABSENT_CLAIM`` leaves the claim out of the token.
_ABSENT_CLAIM = object()
_UNUSABLE_OKTA_UIDS = st.one_of(
    st.just(_ABSENT_CLAIM),
    st.none(),
    st.just(""),
    st.integers(),
    st.booleans(),
    st.lists(_ACCOUNT_IDS, max_size=2),
)


@HYPOTHESIS_SETTINGS
@given(profile=_PROFILES, uid=_UNUSABLE_OKTA_UIDS)
@example(profile=_COLLIDING_PROFILE, uid="")
@example(profile=_COLLIDING_PROFILE, uid=123)
@example(profile=_COLLIDING_PROFILE, uid=["00u-alice"])
def test_okta_user_token_without_a_usable_uid_has_no_vault_key(profile, uid):
    """An Okta user token with no usable ``uid`` never keys on its ``sub``.

    A user access token's ``sub`` is the editable login and its ``cid`` is the
    client it was issued to, so ``sub != cid``. For every profile and every
    ``uid`` that is absent, empty or not a string, the vault key is ``""`` (the
    vend refuses it) and the audit subject is None, rather than the login.
    """
    assume(profile["login"] != WEB_CLIENT_ID)
    claims = {"cid": WEB_CLIENT_ID}
    if uid is not _ABSENT_CLAIM:
        claims["uid"] = uid
    result = _bearer_result("okta", None, profile, extra_claims=claims)
    assert server_module._canonical_egress_user(result) == ""
    assert server_module._audit_identity_claims(result)["subject"] is None


@pytest.mark.parametrize(
    "bearer_path",
    [path for path in BEARER_PATHS if path not in ("okta", "cognito-validator-boto3")],
)
@HYPOTHESIS_SETTINGS
@given(account_id=_ACCOUNT_IDS, uid=_ACCOUNT_IDS)
@example(account_id="user-a", uid="user-b")
def test_uid_claim_ignored_outside_okta(bearer_path, account_id, uid):
    """A ``uid`` claim never changes the vault key of a non-Okta token.

    For every non-Okta bearer path, account id and ``uid`` value, the key is
    the account id.
    """
    key = _bearer_key(bearer_path, account_id, _COLLIDING_PROFILE, extra_claims={"uid": uid})
    assert key == account_id


@pytest.mark.parametrize(
    "bearer_path",
    [path for path in BEARER_PATHS if path not in ("okta", "cognito-validator-boto3")],
)
@HYPOTHESIS_SETTINGS
@given(profile=_PROFILES)
@example(profile=_COLLIDING_PROFILE)
def test_bearer_without_sub_has_no_vault_key(bearer_path, profile):
    """A provider's access token without ``sub`` never keys on its username.

    The validators accept a token without ``sub`` (a Keycloak client without
    the ``basic`` scope issues one), and the only name left on it is the login,
    which an admin can rename or a later account can take. For every validator
    path and profile, the key is empty, so ``/validate`` stamps no
    ``egress_user`` on either internal token. Okta keys on ``uid``, and a
    ``GetUser`` response without ``sub`` fails validation, so neither applies.
    """
    assert _bearer_key(bearer_path, None, profile) == ""


@HYPOTHESIS_SETTINGS
@given(
    login_path=st.sampled_from(list(LOGIN_PATHS)),
    account_id=_ACCOUNT_IDS,
    profile=_PROFILES,
)
@example(login_path="cognito", account_id="0f6c-uuid-new-account", profile=_COLLIDING_PROFILE)
@example(login_path="okta", account_id="00u-alice-b", profile=_COLLIDING_PROFILE)
@example(login_path="github", account_id="12345", profile=_COLLIDING_PROFILE)
def test_every_reader_of_a_session_gives_the_key_stored_at_login(login_path, account_id, profile):
    """Every reader of a session's vault key returns the key the login stored.

    For every login path and account, the callback stores the account id as the
    session's ``egress_user``. The auth server's cookie path, the registry's raw
    cookie path (``enhanced_auth``, which the consent callback uses) and the
    self-signed token ``/internal/tokens`` mints from the session all key the
    vault on that stored value.
    """
    login = _login(login_path, account_id, profile)
    assert login.status_code == 302, (login_path, login.status_code)

    keys = {
        "stored": login.store.documents[0].get("egress_user"),
        "auth-server-cookie": _cookie_key(login),
        "registry-cookie": _registry_cookie_key(login),
        "minted": _minted_token_key(login),
    }

    assert set(keys.values()) == {account_id}, (login_path, keys)


# Values a session's ``provider`` takes after the stored document drifts from
# what the login wrote. ``_ABSENT_CLAIM`` removes the field.
_DRIFTED_PROVIDERS = st.one_of(
    st.just(_ABSENT_CLAIM),
    st.none(),
    st.just(""),
    st.sampled_from(["github", "google", "keycloak", "corp-oidc"]),
)


@pytest.mark.parametrize(
    ("login_path", "has_subject"),
    [(path, False) for path in sorted(NON_OIDC_LOGIN_PATHS)]
    + [("github", True), ("keycloak", True)],
)
@HYPOTHESIS_SETTINGS
@given(profile=_PROFILES, provider=_DRIFTED_PROVIDERS, username=_CLAIM_VALUES)
@example(profile=_COLLIDING_PROFILE, provider="google", username="mallory")
def test_session_vault_key_ignores_drift_in_provider_and_username(
    login_path, has_subject, profile, provider, username
):
    """Changing a stored session's ``provider`` or ``username`` keeps its vault key.

    Both services read the shared ``oauth_sessions`` document through their own
    projection. For every login path, with or without a subject, and every new
    ``provider`` (removed, empty or another name) and ``username``, the auth
    server's cookie path and the registry's raw cookie path still give the key
    the session had at login, which is its stored ``egress_user``.
    """
    account_id = "account-1" if has_subject else None
    login = _login(login_path, account_id, profile)
    assert login.status_code == 302, (login_path, login.status_code)
    key_at_login = _cookie_key(login)
    document = login.store.documents[0]

    document["username"] = username
    if provider is _ABSENT_CLAIM:
        document.pop("provider", None)
    else:
        document["provider"] = provider
    keys = {
        "auth-server-cookie": _cookie_key(login),
        "registry-cookie": _registry_cookie_key(login),
    }

    assert set(keys.values()) == {key_at_login}, (login_path, key_at_login, keys)
    assert key_at_login and key_at_login == document.get("egress_user"), (login_path, document)


_AUDIT_PATHS: list[str] = [f"bearer:{path}" for path in BEARER_PATHS] + [
    f"login:{path}" for path in LOGIN_PATHS
]


@pytest.mark.parametrize("path", _AUDIT_PATHS)
@HYPOTHESIS_SETTINGS
@given(account_id=_ACCOUNT_IDS, profile=_PROFILES)
@example(account_id="00u-alice-b", profile=_COLLIDING_PROFILE)
@example(account_id="0f6c-uuid-new-account", profile=_COLLIDING_PROFILE)
def test_audit_subject_names_the_vault_key(path, account_id, profile):
    """The audit record's ``subject`` is the vault key of the same caller.

    For every bearer path and every login path's session-cookie result, and
    every account, ``_audit_identity_claims(result)["subject"]`` equals
    ``_canonical_egress_user(result)`` whenever that key is non-empty, so an
    audit record joins to the egress connection of the same human.
    """
    kind, name = path.split(":", 1)
    if kind == "bearer":
        result = _bearer_result(name, account_id, profile)
    else:
        login = _login(name, account_id, profile)
        assert login.status_code == 302, (name, login.status_code)
        result = _cookie_result(login)
    vault_key = server_module._canonical_egress_user(result)
    audit_subject = server_module._audit_identity_claims(result)["subject"]
    assert vault_key
    assert audit_subject == vault_key, (path, audit_subject, vault_key)


@HYPOTHESIS_SETTINGS
@given(profile=_PROFILES)
@example(profile=_COLLIDING_PROFILE)
def test_cognito_getuser_without_sub_fails_bearer_validation(profile):
    """A boto3 ``GetUser`` response without ``sub`` is not a valid bearer.

    For every Cognito username and attribute set, when JWT verification is
    unavailable and ``GetUser`` returns no ``sub`` attribute,
    ``SimplifiedCognitoValidator.validate_token`` raises a ``ValueError``
    naming ``sub`` instead of returning a result keyed on the username.
    """
    validator = server_module.SimplifiedCognitoValidator(COGNITO_REGION)
    with (
        patch.object(validator, "validate_jwt_token", side_effect=ValueError("JWKS unreachable")),
        patch.object(
            validator,
            "_get_cognito_client",
            return_value=_cognito_get_user_client(None, profile),
        ),
        pytest.raises(ValueError, match="sub"),
    ):
        validator.validate_token(
            "opaque-access-token", COGNITO_POOL_ID, WEB_CLIENT_ID, COGNITO_REGION
        )


@HYPOTHESIS_SETTINGS
@given(account_id=_ACCOUNT_IDS, profile=_PROFILES)
@example(account_id="0f6c-uuid-new-account", profile=_COLLIDING_PROFILE)
def test_cognito_login_without_getuser_sub_uses_userinfo_sub(account_id, profile):
    """A Cognito login whose ``GetUser`` lacks ``sub`` takes its subject from userInfo.

    For every account, when JWT verification is unavailable and ``GetUser``
    returns no ``sub``, the login falls back to the userInfo endpoint, whose
    ``sub`` becomes the session subject and the vault key.
    """
    login = _login("cognito-boto3", account_id, profile, getuser_has_sub=False)
    assert login.status_code == 302, login.status_code
    assert login.store.documents[0]["subject"] == account_id
    assert _cookie_key(login) == account_id


@HOP_HYPOTHESIS_SETTINGS
@given(key_name=_ACCOUNT_IDS)
@example(key_name="alice")
def test_token_minted_for_a_static_key_owns_no_vault_bucket(key_name):
    """A token minted for a network-trusted caller never addresses a per-user bucket.

    ``/internal/tokens`` mints a self-signed token for a ``network-trusted``
    caller (a REGISTRY_API_TOKEN or REGISTRY_API_KEYS key) and stamps that
    method into the token's ``auth_method`` claim. For every key name, the
    token's validation result canonicalizes to a non-per-user method, and the
    registry context ``/validate`` builds from it owns no vault principal, so a
    static key named like a user cannot reach that user's connections.
    """
    store = _FakeSessionCollection()
    token = _mint_token(
        {"username": key_name, "auth_method": "network-trusted", "provider": "network-trusted"},
        store,
    )
    result = server_module.validator.validate_self_signed_token(token)

    hop = _validate_hop_tokens({"Authorization": f"Bearer {token}"}, store)

    canonical = canonical_auth_method(result)
    assert not is_per_user_auth_method(canonical), canonical
    assert vault_principal(hop.registry_context) is None, hop.registry_context
    assert not is_per_user_auth_method(hop.mcp_proxy_claims["auth_method"])


@st.composite
def _session_logins(
    draw: st.DrawFn,
) -> tuple[str, str | None]:
    """Draw a login path and an account id, or None on a path that allows no subject.

    Args:
        draw: Hypothesis draw function.

    Returns:
        ``(login_path, account_id)``.
    """
    login_path = draw(st.sampled_from(list(LOGIN_PATHS)))
    if login_path in NON_OIDC_LOGIN_PATHS:
        return login_path, draw(st.one_of(st.none(), _ACCOUNT_IDS))
    return login_path, draw(_ACCOUNT_IDS)


@HOP_HYPOTHESIS_SETTINGS
@given(session=_session_logins(), profile=_PROFILES)
@example(session=("keycloak", "oidc-sub-1"), profile=_COLLIDING_PROFILE)
@example(session=("github", None), profile=_COLLIDING_PROFILE)
def test_session_without_a_stored_key_owns_no_vault_key_whatever_its_subject(session, profile):
    """A session record without ``egress_user`` keys nothing, with or without a subject.

    No reader falls back to the session's ``subject`` or rebuilds a key from
    its ``provider`` and ``username``, so a session created before the login
    stored ``egress_user`` has no vault key until the user signs in again. For
    every login and profile, removing ``egress_user`` from the stored document
    leaves the auth server's cookie path, both internal tokens ``/validate``
    mints, the registry's raw cookie path and a token minted from the session
    without a key.
    """
    login_path, account_id = session
    login = _login(login_path, account_id, profile)
    assert login.status_code == 302, (login_path, login.status_code)
    document = login.store.documents[0]
    document.pop("egress_user", None)
    assert document.get("subject") == account_id, (login_path, document)

    hop = _validate_hop_tokens(_session_cookie_header(login.cookie), login.store, None)
    raw_cookie = _registry_cookie_context(login.cookie, login.store)

    keys = {
        "auth-server-cookie": _cookie_key(login),
        "registry-cookie": raw_cookie["egress_user"],
        "minted": _minted_token_key(login),
    }
    principals = {
        "registry-token": vault_principal(hop.registry_context),
        "vend": _vend_principal(hop.mcp_proxy_claims),
        "raw-cookie": vault_principal(raw_cookie),
    }
    assert set(keys.values()) == {""}, (login_path, keys)
    assert set(principals.values()) == {None}, (login_path, principals)


@HYPOTHESIS_SETTINGS
@given(
    session=_session_logins(),
    other=st.one_of(_ACCOUNT_IDS, st.sampled_from(_FALLBACK_SHAPED_IDS)),
    profile=_PROFILES,
)
@example(session=("keycloak", "sub-alice"), other="sub-bob", profile=_COLLIDING_PROFILE)
@example(session=("github", None), other="sub-bob", profile=_COLLIDING_PROFILE)
def test_session_backed_mint_ignores_the_body_egress_user_and_subject(session, other, profile):
    """A mint with a ``session_id`` keys the token on the session, never on the body.

    For every login and every other user's id in the request body's
    ``egress_user`` and ``subject``, the minted token carries the session's
    stored key (the username fallback key for a subject-less session), so a
    caller holding the internal signing key cannot point a session's token at
    someone else.
    """
    login_path, account_id = session
    assume(other != account_id)
    login = _login(login_path, account_id, profile)
    assert login.status_code == 302, (login_path, login.status_code)
    stored_key = login.store.documents[0]["egress_user"]
    assume(other != stored_key)

    token = _mint_token(
        {**_session_mint_context(login), "egress_user": other, "subject": other}, login.store
    )
    claims = server_module.validator.validate_self_signed_token(token)["data"]

    assert claims.get("egress_user") == stored_key, (login_path, claims)


@HYPOTHESIS_SETTINGS
@given(
    field=st.sampled_from(["egress_user", "subject"]),
    suffix=st.text(max_size=12),
    username=_CLAIM_VALUES,
)
@example(field="egress_user", suffix="github:alice", username="alice")
@example(field="subject", suffix="", username="alice")
def test_mint_without_a_session_never_stamps_a_fallback_key(field, suffix, username):
    """A mint with no ``session_id`` leaves a fallback-shaped body key off the token.

    A caller without a session may assert its vault key in the body. For every
    fallback-shaped value in ``egress_user`` or ``subject``, the minted token
    carries no ``egress_user``, as a token minted from a session never does.
    """
    token = _mint_token(
        {
            "username": username,
            "auth_method": "oauth2",
            "provider": "keycloak",
            field: f"name:{suffix}",
        },
        _FakeSessionCollection(),
    )
    claims = server_module.validator.validate_self_signed_token(token)["data"]

    assert "egress_user" not in claims, claims


@pytest.mark.parametrize("value", [12345, True, ["sub-1"], {"id": "sub-1"}], ids=repr)
def test_mint_without_a_session_never_stamps_a_non_string_key(value):
    """A mint with no ``session_id`` drops a non-string body ``egress_user``.

    The body value is a vault key the token would carry, so a wrong type is
    refused rather than stamped or coerced: the minted token carries no
    ``egress_user``, and the vend refuses it.
    """
    token = _mint_token(
        {
            "username": "alice",
            "auth_method": "oauth2",
            "provider": "keycloak",
            "egress_user": value,
        },
        _FakeSessionCollection(),
    )
    claims = server_module.validator.validate_self_signed_token(token)["data"]

    assert "egress_user" not in claims, claims


def _vend_principal(
    mcp_proxy_claims: dict[str, Any],
) -> tuple[str, str] | None:
    """The vault principal the egress vend reads from verified mcp-proxy claims.

    ``vend_egress_token`` derives it with ``vault_principal`` from the verified
    claims, so this is that call on the mcp-proxy token rather than on the
    registry token.

    Args:
        mcp_proxy_claims: Claims ``verify_mcp_proxy_token`` returned.

    Returns:
        ``(auth_method, egress_user)``, or None when the vend refuses the caller.
    """
    return vault_principal(mcp_proxy_claims)


@pytest.mark.parametrize("provider", ["keycloak", "okta"])
@HOP_HYPOTHESIS_SETTINGS
@given(account_id=_ACCOUNT_IDS, profile=_PROFILES)
@example(account_id="00u-alice-b", profile=_COLLIDING_PROFILE)
def test_registry_and_vend_key_one_caller_on_one_principal(provider, account_id, profile):
    """The registry's vault principal equals the vend's for the same caller.

    One account logs in, then makes the same MCP request three ways: with the
    session cookie, with a token ``/internal/tokens`` minted from that session,
    and with the IdP's own access token. For each, ``/validate`` mints the
    registry token and the mcp-proxy token from one validation result. The
    registry builds its user context from the first through its real
    verification path, and the vend reads the second. For every account and
    profile, both name ``("oauth2", account_id)``, so a connection the browser
    saves is the one each of those callers vends.
    """
    login = _login(provider, account_id, profile)
    assert login.status_code == 302, (provider, login.status_code)
    minted = _mint_token(_session_mint_context(login), login.store)
    idp_bearer = _signed_bearer(provider, account_id, profile, {"groups": ["hop-test-group"]})
    callers = {
        "cookie": (_session_cookie_header(login.cookie), None),
        "minted": ({"Authorization": f"Bearer {minted}"}, None),
        "idp-bearer": ({"Authorization": f"Bearer {idp_bearer}"}, provider),
    }

    principals = {
        name: _validate_hop_tokens(headers, login.store, bearer_provider)
        for name, (headers, bearer_provider) in callers.items()
    }

    expected = ("oauth2", account_id)
    for name, hop in principals.items():
        registry = vault_principal(hop.registry_context)
        vend = _vend_principal(hop.mcp_proxy_claims)
        assert registry == vend == expected, (provider, name, registry, vend)
    raw_cookie = vault_principal(_registry_cookie_context(login.cookie, login.store))
    assert raw_cookie == expected, (provider, "raw-cookie", raw_cookie)


def _registry_cookie_context(
    cookie: str,
    store: _FakeSessionCollection,
) -> dict[str, Any]:
    """Build the registry's user context from a raw session cookie.

    This is the path the consent callback and ``/oauth2/egress/connect`` take:
    no internal token, so the registry's ``enhanced_auth`` resolves the cookie
    against the session store itself. The registry verifies the cookie with
    the auth server's signer, as a deployment that shares ``SECRET_KEY`` does.

    Args:
        cookie: The session cookie the login set.
        store: The session collection the login wrote.

    Returns:
        The registry user context.
    """
    with (
        patch.object(registry_dependencies, "signer", server_module.signer),
        patch("registry.auth.session_store._get_collection", AsyncMock(return_value=store)),
        patch(
            "registry.auth.dependencies.map_cognito_groups_to_scopes",
            AsyncMock(return_value=[]),
        ),
        patch(
            "registry.auth.dependencies.resolve_scope_access",
            AsyncMock(return_value=SimpleNamespace(servers=[], tools={})),
        ),
        patch(
            "registry.auth.dependencies.get_ui_permissions_for_user",
            AsyncMock(return_value={}),
        ),
    ):
        request = SimpleNamespace(state=SimpleNamespace())
        return asyncio.run(registry_dependencies.enhanced_auth(request, session=cookie))


def _registry_cookie_key(
    login: _Login,
) -> str:
    """Return the vault key the registry's raw cookie path gives a login's session.

    Args:
        login: A login that created a session.

    Returns:
        The ``egress_user`` of the registry's ``enhanced_auth`` user context.
    """
    return _registry_cookie_context(login.cookie, login.store)["egress_user"]


def _okta_user_bearer_without_uid(
    profile: dict[str, str],
) -> tuple[dict[str, str], str | None]:
    """An Okta user access token with no ``uid``; the auth server keys it on nothing."""
    token = _signed_bearer(
        "okta", None, profile, {"cid": WEB_CLIENT_ID, "groups": ["hop-test-group"]}
    )
    return {"Authorization": f"Bearer {token}"}, "okta"


def _minted_bearer_without_egress_user(
    profile: dict[str, str],
) -> tuple[dict[str, str], str | None]:
    """A gateway-minted user token with no ``egress_user``; the auth server keys it on nothing."""
    token = _mint_token(
        {"username": profile["login"], "auth_method": "oauth2", "provider": "keycloak"},
        _FakeSessionCollection(),
    )
    return {"Authorization": f"Bearer {token}"}, None


def _provider_bearer_without_sub(
    provider: str,
    profile: dict[str, str],
) -> tuple[dict[str, str], str | None]:
    """A provider's user access token with no ``sub``; the auth server keys it on nothing."""
    token = _signed_bearer(provider, None, profile, {"groups": ["hop-test-group"]})
    return {"Authorization": f"Bearer {token}"}, provider


@pytest.mark.parametrize(
    "caller",
    [
        pytest.param(_okta_user_bearer_without_uid, id="okta-user-token-without-uid"),
        pytest.param(_minted_bearer_without_egress_user, id="minted-token-without-egress-user"),
        *(
            pytest.param(
                functools.partial(_provider_bearer_without_sub, provider),
                id=f"{provider}-bearer-without-sub",
            )
            # The hop harness grants scopes from a ``groups`` claim, which these
            # two providers read; test_bearer_without_sub_has_no_vault_key
            # covers every validator path at the key /validate stamps.
            for provider in ("keycloak", "entra")
        ),
    ],
)
@HOP_HYPOTHESIS_SETTINGS
@given(profile=_PROFILES)
@example(profile=_COLLIDING_PROFILE)
def test_registry_and_vend_both_refuse_a_caller_the_auth_server_does_not_key(caller, profile):
    """A caller the auth server gives no vault key owns no principal in the registry either.

    The auth server returns an empty key for an Okta user token without a usable
    ``uid``, for a gateway-minted token without ``egress_user`` and for a
    provider's access token without ``sub``, because the only other names those
    tokens carry are the login (a Keycloak ``preferred_username``, for example,
    which an admin can rename or a later user can take). For every profile, both
    the registry's principal (built from the registry token) and the vend's
    (from the mcp-proxy token) are None, so PAT submit, initiate, list and
    disconnect cannot address a bucket keyed on that login.
    """
    assume(profile["login"] != WEB_CLIENT_ID)
    headers, bearer_provider = caller(profile)

    hop = _validate_hop_tokens(headers, _FakeSessionCollection(), bearer_provider)

    registry = vault_principal(hop.registry_context)
    vend = _vend_principal(hop.mcp_proxy_claims)
    assert registry is None and vend is None, (registry, vend, hop.registry_context)


@pytest.mark.parametrize("login_path", sorted(NON_OIDC_LOGIN_PATHS))
@HOP_HYPOTHESIS_SETTINGS
@given(profile=_PROFILES)
@example(profile=_COLLIDING_PROFILE)
def test_session_without_subject_keys_every_path_on_one_fallback_key(login_path, profile):
    """A session with no subject keys the browser consent and every token on one key.

    A GitHub, Google or operator-added login without an id_token and without
    the provider's ``subject_claim`` creates a session with no subject, and the
    callback stores ``username_fallback_key(provider, username)`` as its
    ``egress_user``. For every profile, the registry's raw cookie path
    (``enhanced_auth``, which the consent callback and ``/oauth2/egress/connect``
    use, so the bucket a browser consent writes), the auth server's cookie path
    (through ``/validate``, for both internal tokens), and a token
    ``/internal/tokens`` minted from that session (its registry principal and
    its vend principal) all give that stored key. So an MCP client holding the
    minted token vends what the browser consented, and the key never equals the
    bare username another provider might issue as an account id.
    """
    login = _login(login_path, None, profile)
    assert login.status_code == 302, (login_path, login.status_code)
    session = login.store.documents[0]
    expected = ("oauth2", username_fallback_key(session["provider"], session["username"]))

    cookie_hop = _validate_hop_tokens(_session_cookie_header(login.cookie), login.store, None)
    minted = _mint_token(_session_mint_context(login), login.store)
    minted_hop = _validate_hop_tokens({"Authorization": f"Bearer {minted}"}, login.store, None)
    raw_cookie = _registry_cookie_context(login.cookie, login.store)

    principals = {
        "stored": ("oauth2", session.get("egress_user")),
        "auth-server-cookie": ("oauth2", _cookie_key(login)),
        "registry-token": vault_principal(cookie_hop.registry_context),
        "vend": _vend_principal(cookie_hop.mcp_proxy_claims),
        "raw-cookie": vault_principal(raw_cookie),
        "minted-registry-token": vault_principal(minted_hop.registry_context),
        "minted-vend": _vend_principal(minted_hop.mcp_proxy_claims),
    }
    assert set(principals.values()) == {expected}, (login_path, principals)
    assert expected[1] != session["username"]


def _provider_bearer_with_fallback_shaped_sub(
    provider: str,
    profile: dict[str, str],
) -> tuple[dict[str, str], str | None]:
    """A provider's access token whose ``sub`` is shaped like a fallback key."""
    token = _signed_bearer(provider, "name:github:alice", profile, {"groups": ["hop-test-group"]})
    return {"Authorization": f"Bearer {token}"}, provider


@pytest.mark.parametrize("provider", ["keycloak", "entra"])
@HOP_HYPOTHESIS_SETTINGS
@given(profile=_PROFILES)
@example(profile=_COLLIDING_PROFILE)
def test_idp_bearer_with_a_fallback_shaped_sub_owns_no_principal(provider, profile):
    """A raw IdP bearer whose ``sub`` starts with ``name:`` keys nothing.

    Only a gateway-minted token may carry a fallback key, because only the
    session-backed mint stamps one. A bearer the IdP issued with
    ``sub = "name:github:alice"`` would otherwise read that subject-less GitHub
    user's connections, so for every profile both the registry's principal and
    the vend's are None.
    """
    assume(profile["login"] != WEB_CLIENT_ID)
    headers, bearer_provider = _provider_bearer_with_fallback_shaped_sub(provider, profile)

    hop = _validate_hop_tokens(headers, _FakeSessionCollection(), bearer_provider)

    registry = vault_principal(hop.registry_context)
    vend = _vend_principal(hop.mcp_proxy_claims)
    assert registry is None and vend is None, (registry, vend, hop.registry_context)
