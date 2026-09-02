"""Generic OAuth 2.0 authorization-code engine.

Three pure-ish functions cover ~90% of every provider; the ~3 providers that
bend the spec are handled by a small ``token_response_parser`` hook table, not
bespoke code (the AgentCore one-engine model).

- ``build_authorize_url``  -- construct the consent URL (PKCE S256, scopes, state).
- ``exchange_code``        -- code -> tokens at the token endpoint.
- ``refresh_token``        -- refresh_token grant -> new tokens.

PKCE helpers live here too. The engine never touches the SecretStore or the
provider config table directly -- callers pass the resolved ``OAuthProviderConfig``
plus the operator ``client_id``/``client_secret``. Token material is returned as
``StoredToken``; the caller persists it.
"""

import base64
import hashlib
import json
import logging
import secrets
from datetime import UTC, datetime, timedelta
from urllib.parse import urlencode

import httpx

from registry.egress_auth.schemas import (
    OAuthProviderConfig,
    StoredToken,
    TokenEndpointAuthStyle,
)
from registry.exceptions import UrlValidationError
from registry.utils.url_guard import CREDENTIALED_OAUTH_PROFILE, guarded_async_client

logger = logging.getLogger(__name__)

_HTTP_TIMEOUT = 30.0


class OAuthEngineError(Exception):
    """OAuth token-endpoint failure (network, non-2xx, unparseable response)."""


class DeadRefreshTokenError(OAuthEngineError):
    """The refresh token was rejected by the provider (invalid_grant) -> re-consent."""


# --------------------------------------------------------------------------- #
# PKCE
# --------------------------------------------------------------------------- #


def generate_pkce_verifier() -> str:
    """RFC 7636 code_verifier: 43-128 chars of unreserved URL-safe base64."""
    return base64.urlsafe_b64encode(secrets.token_bytes(32)).rstrip(b"=").decode("ascii")


def pkce_challenge_s256(verifier: str) -> str:
    """S256 code_challenge = base64url(sha256(verifier)), no padding."""
    digest = hashlib.sha256(verifier.encode("ascii")).digest()
    return base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")


# --------------------------------------------------------------------------- #
# Authorize URL
# --------------------------------------------------------------------------- #


def build_authorize_url(
    cfg: OAuthProviderConfig,
    client_id: str,
    redirect_uri: str,
    scopes: list[str],
    state: str,
    pkce_challenge: str | None = None,
) -> str:
    """Build the provider authorization-code consent URL."""
    params = {
        "response_type": "code",
        "client_id": client_id,
        "redirect_uri": redirect_uri,
        "state": state,
    }
    if scopes:
        params["scope"] = cfg.scope_separator.join(scopes)
    if cfg.use_pkce and pkce_challenge:
        params["code_challenge"] = pkce_challenge
        params["code_challenge_method"] = "S256"
    # RFC 8707 resource indicator. Sent here on the authorize leg AND on the
    # token/refresh legs (see exchange_code/refresh_token) -- providers like
    # Atlassian's Rovo MCP require it on BOTH or they reject the flow. Set before
    # extra_authorize_params so a built-in's static params still win a collision.
    if cfg.resource:
        params["resource"] = cfg.resource
    params.update(cfg.extra_authorize_params)
    return f"{cfg.authorize_url}?{urlencode(params)}"


# --------------------------------------------------------------------------- #
# Token-endpoint quirk hooks
# --------------------------------------------------------------------------- #


def _parse_standard(payload: dict) -> dict:
    return payload


def _parse_github_form(payload: dict) -> dict:
    """GitHub already arrives as a dict here (we force Accept: json); identity.

    Kept as an explicit hook so a future form-encoded edge has one place to live.
    """
    return payload


def _parse_slack_nested(payload: dict) -> dict:
    """Slack nests the user token under ``authed_user``.

    Slack v2 returns ``{ok, authed_user: {access_token, token_type, scope, ...}}``
    for user tokens. Lift the user token to the top level the engine expects.
    """
    if not payload.get("ok", True):
        raise OAuthEngineError(f"Slack token error: {payload.get('error')}")
    authed = payload.get("authed_user")
    if isinstance(authed, dict) and authed.get("access_token"):
        merged = dict(payload)
        merged["access_token"] = authed.get("access_token")
        merged["token_type"] = authed.get("token_type", "Bearer")
        if authed.get("scope"):
            merged["scope"] = authed["scope"]
        if authed.get("refresh_token"):
            merged["refresh_token"] = authed["refresh_token"]
        if authed.get("expires_in"):
            merged["expires_in"] = authed["expires_in"]
        return merged
    return payload


_TOKEN_RESPONSE_PARSERS = {
    "github_form": _parse_github_form,
    "slack_nested": _parse_slack_nested,
}


def _parse_token_response(cfg: OAuthProviderConfig, payload: dict) -> dict:
    parser = _TOKEN_RESPONSE_PARSERS.get(cfg.token_response_parser or "", _parse_standard)
    return parser(payload)


# --------------------------------------------------------------------------- #
# Token endpoint calls
# --------------------------------------------------------------------------- #


def _jwt_exp(access_token: str | None) -> int | None:
    """Best-effort ``exp`` (epoch seconds) from a JWT access token, else None.

    Opaque (non-JWT) tokens and any decode/parse failure return None; the payload
    is base64url-decoded WITHOUT signature verification purely to read the
    provider-asserted lifetime, never to trust the token.
    """
    if not access_token or access_token.count(".") != 2:
        return None
    payload_b64 = access_token.split(".")[1]
    payload_b64 += "=" * (-len(payload_b64) % 4)
    try:
        claims = json.loads(base64.urlsafe_b64decode(payload_b64.encode("ascii")))
    except (ValueError, TypeError):
        return None
    exp = claims.get("exp")
    return int(exp) if isinstance(exp, int | float) and not isinstance(exp, bool) else None


def _expires_at(expires_in: int | None, access_token: str | None = None) -> str | None:
    """ISO expiry from ``expires_in`` when present, else the access token's JWT ``exp``.

    Some providers (e.g. Salesforce) omit ``expires_in`` from the token response
    but issue a JWT access token bounded by an ``exp`` claim. Without this
    fallback ``expires_at`` stays None, the vend path treats the token as
    long-lived and never fires the single-flight refresh, so the gateway keeps
    injecting a token the upstream has already expired ("Invalid token").
    """
    if expires_in:
        return (datetime.now(UTC) + timedelta(seconds=int(expires_in))).isoformat()
    exp = _jwt_exp(access_token)
    if exp is not None:
        return datetime.fromtimestamp(exp, tz=UTC).isoformat()
    return None


def _build_token_request(
    cfg: OAuthProviderConfig,
    client_id: str,
    client_secret: str | None,
    form: dict,
) -> tuple[dict, dict]:
    """Return (form_data, headers), placing the client secret per the provider's style.

    ``NONE`` (RFC 7591 ``token_endpoint_auth_method=none``) is a public client:
    only ``client_id`` goes in the body and no secret exists anywhere in the
    request. The confidential styles fail closed on a missing secret rather
    than sending an empty one (which providers accept-then-misattribute).
    """
    headers = {"Accept": "application/json"}
    data = dict(form)
    if cfg.token_endpoint_auth_style == TokenEndpointAuthStyle.NONE:
        data["client_id"] = client_id
        return data, headers
    if not client_secret:
        raise OAuthEngineError(
            f"provider {cfg.name!r} uses token_endpoint_auth_style="
            f"{cfg.token_endpoint_auth_style.value!r} but no client_secret is configured"
        )
    if cfg.token_endpoint_auth_style == TokenEndpointAuthStyle.BASIC_HEADER:
        basic = base64.b64encode(f"{client_id}:{client_secret}".encode()).decode()
        headers["Authorization"] = f"Basic {basic}"
        data["client_id"] = client_id
    else:  # POST_BODY (default)
        data["client_id"] = client_id
        data["client_secret"] = client_secret
    return data, headers


async def _post_token(cfg: OAuthProviderConfig, data: dict, headers: dict) -> dict:
    # The token endpoint receives the operator client_secret (and, on refresh, the
    # user's refresh_token). For a 'custom' provider cfg.token_url is registrant-
    # supplied, so the request MUST go through the SSRF/rebinding-safe client: it
    # pins the connection to a validated public IP at connect time (blocking a
    # post-registration DNS rebind to a private/metadata address) and rejects a
    # non-http(s) scheme, so the credential can never be exfiltrated to an
    # internal target. The dedicated profile has an empty allowlist and
    # requires HTTPS, so proxy allowlist entries cannot weaken this path.
    # Built-in providers resolve to public HTTPS hosts and pass unchanged.
    try:
        async with guarded_async_client(
            profile=CREDENTIALED_OAUTH_PROFILE,
            timeout=_HTTP_TIMEOUT,
        ) as client:
            resp = await client.post(cfg.token_url, data=data, headers=headers)
    except UrlValidationError as exc:
        # The pinned guard rejected the target before sending any credential.
        # Keep the wrapped detail out of higher-level logs and browser responses.
        raise OAuthEngineError("token endpoint blocked by security policy") from exc
    except httpx.HTTPError as exc:
        raise OAuthEngineError("token endpoint unreachable") from exc

    try:
        payload = resp.json()
    except ValueError as exc:
        raise OAuthEngineError(
            f"token endpoint returned non-JSON (status {resp.status_code})"
        ) from exc

    if resp.status_code >= 400 or payload.get("error"):
        err = payload.get("error", f"http {resp.status_code}")
        if err in ("invalid_grant", "bad_refresh_token"):
            raise DeadRefreshTokenError(f"refresh rejected by provider: {err}")
        raise OAuthEngineError(f"token endpoint error: {err}")
    return payload


def _to_stored_token(
    cfg: OAuthProviderConfig,
    payload: dict,
    client_id: str,
    fallback_refresh: str | None = None,
) -> StoredToken:
    parsed = _parse_token_response(cfg, payload)
    access = parsed.get("access_token")
    if not access:
        raise OAuthEngineError("token response missing access_token")
    scope_raw = parsed.get("scope", "")
    scopes = scope_raw.split(cfg.scope_separator) if scope_raw else []
    now = datetime.now(UTC).isoformat()
    return StoredToken(
        access_token=access,
        # Providers that rotate refresh tokens return a new one; otherwise keep
        # the prior one (some don't re-send it on refresh).
        refresh_token=parsed.get("refresh_token") or fallback_refresh,
        token_type=parsed.get("token_type", "Bearer"),
        expires_at=_expires_at(parsed.get("expires_in"), access),
        scopes=[s for s in scopes if s],
        status="active",
        client_id=client_id,
        created_at=now,
        last_refreshed_at=now,
    )


async def exchange_code(
    cfg: OAuthProviderConfig,
    client_id: str,
    client_secret: str | None,
    code: str,
    redirect_uri: str,
    pkce_verifier: str | None = None,
) -> StoredToken:
    """Exchange an authorization code for tokens."""
    form = {
        "grant_type": "authorization_code",
        "code": code,
        "redirect_uri": redirect_uri,
    }
    if cfg.use_pkce and pkce_verifier:
        form["code_verifier"] = pkce_verifier
    # RFC 8707: the resource indicator MUST match the one sent on authorize, or a
    # resource server like Atlassian's Rovo MCP rejects the exchange.
    if cfg.resource:
        form["resource"] = cfg.resource
    data, headers = _build_token_request(cfg, client_id, client_secret, form)
    payload = await _post_token(cfg, data, headers)
    return _to_stored_token(cfg, payload, client_id)


async def refresh_token(
    cfg: OAuthProviderConfig,
    client_id: str,
    client_secret: str | None,
    refresh_token_value: str,
) -> StoredToken:
    """Exchange a refresh token for a new access token (and possibly new refresh token).

    Raises ``DeadRefreshTokenError`` when the provider rejects the refresh token
    (invalid_grant) so the caller marks the entry ``refresh_failed`` -> re-consent.
    """
    form = {
        "grant_type": "refresh_token",
        "refresh_token": refresh_token_value,
    }
    # RFC 8707: carry the resource indicator on refresh too so the rotated access
    # token stays bound to the same protected resource.
    if cfg.resource:
        form["resource"] = cfg.resource
    data, headers = _build_token_request(cfg, client_id, client_secret, form)
    payload = await _post_token(cfg, data, headers)
    return _to_stored_token(cfg, payload, client_id, fallback_refresh=refresh_token_value)


# --------------------------------------------------------------------------- #
# Dynamic Client Registration (RFC 7591)
# --------------------------------------------------------------------------- #


async def _get_json(url: str) -> dict:
    """GET a discovery document through the SSRF/rebinding-safe guarded client.

    Uses CREDENTIALED_OAUTH_PROFILE (same as the token endpoint) because the
    discovery chain for a ``requires_dcr`` provider can be derived from a
    registrant-influenced protected-resource metadata document; pinning to a
    validated public IP blocks a DNS rebind to a private/metadata address.
    """
    try:
        async with guarded_async_client(
            profile=CREDENTIALED_OAUTH_PROFILE, timeout=_HTTP_TIMEOUT
        ) as client:
            resp = await client.get(url, headers={"Accept": "application/json"})
    except UrlValidationError as exc:
        raise OAuthEngineError(f"discovery blocked by SSRF guard: {exc}") from exc
    except httpx.HTTPError as exc:
        raise OAuthEngineError(f"discovery endpoint unreachable: {exc}") from exc
    try:
        return resp.json()
    except ValueError as exc:
        raise OAuthEngineError(
            f"discovery endpoint returned non-JSON (status {resp.status_code})"
        ) from exc


async def fetch_protected_resource_metadata(cfg: OAuthProviderConfig) -> dict:
    """Fetch the RFC 9728 protected-resource metadata document for a ``requires_dcr`` provider.

    Returns the full PRM dict so callers can inspect both ``authorization_servers``
    (needed for the discovery walk in ``_discover_registration_url``) and
    ``scopes_supported`` (needed for config-time scope validation) from a
    **single** network round-trip.  Uses the same SSRF-safe guarded client as all
    other discovery fetches.
    """
    if not cfg.protected_resource_metadata_url:
        raise OAuthEngineError("protected_resource_metadata_url is not configured")
    return await _get_json(cfg.protected_resource_metadata_url)


def validate_scopes_against_prm(scopes: list[str], prm: dict) -> list[str]:
    """Return the subset of ``scopes`` not listed in ``prm['scopes_supported']``.

    An empty return value means all requested scopes are valid (or the PRM does
    not advertise ``scopes_supported``, in which case validation is skipped and
    the empty list is returned regardless).  The caller decides whether to treat
    unsupported scopes as an error.
    """
    supported = prm.get("scopes_supported") or []
    if not supported:
        return []
    supported_set = set(supported)
    return [s for s in scopes if s not in supported_set]


async def _discover_registration_url(
    cfg: OAuthProviderConfig,
    prm: dict | None = None,
) -> str:
    """Resolve the RFC 7591 registration endpoint for a ``requires_dcr`` provider.

    Pinned ``registration_url`` wins; otherwise walk RFC 9728 (protected-resource
    metadata) -> RFC 8414 (authorization-server metadata) -> ``registration_endpoint``.
    The AS metadata is read at the append form (``{as}/.well-known/oauth-authorization-server``)
    that Atlassian's authv2 AS serves.

    ``prm`` may be a pre-fetched PRM dict (from ``fetch_protected_resource_metadata``).
    When provided, the PRM fetch is skipped so the caller can reuse a document
    already retrieved for another purpose (e.g. scope validation) without a
    second network round-trip.
    """
    if cfg.registration_url:
        return cfg.registration_url
    if not cfg.protected_resource_metadata_url:
        raise OAuthEngineError(
            "DCR required but neither registration_url nor "
            "protected_resource_metadata_url is configured"
        )
    if prm is None:
        prm = await _get_json(cfg.protected_resource_metadata_url)
    servers = prm.get("authorization_servers") or []
    if not servers:
        raise OAuthEngineError("protected-resource metadata lists no authorization_servers")
    as_meta_url = str(servers[0]).rstrip("/") + "/.well-known/oauth-authorization-server"
    as_meta = await _get_json(as_meta_url)
    reg = as_meta.get("registration_endpoint")
    if not reg:
        raise OAuthEngineError("authorization-server metadata has no registration_endpoint")
    return reg


async def _post_dcr(reg_url: str, body: dict) -> dict:
    """POST an RFC 7591 registration request through the guarded client."""
    try:
        async with guarded_async_client(
            profile=CREDENTIALED_OAUTH_PROFILE, timeout=_HTTP_TIMEOUT
        ) as client:
            resp = await client.post(reg_url, json=body, headers={"Accept": "application/json"})
    except UrlValidationError as exc:
        raise OAuthEngineError(f"DCR endpoint blocked by SSRF guard: {exc}") from exc
    except httpx.HTTPError as exc:
        raise OAuthEngineError(f"DCR endpoint unreachable: {exc}") from exc
    try:
        payload = resp.json()
    except ValueError as exc:
        raise OAuthEngineError(
            f"DCR endpoint returned non-JSON (status {resp.status_code})"
        ) from exc
    if resp.status_code >= 400 or payload.get("error"):
        raise OAuthEngineError(f"DCR failed: {payload.get('error', f'http {resp.status_code}')}")
    return payload


async def register_dcr_client(
    cfg: OAuthProviderConfig,
    redirect_uri: str,
    scopes: list[str],
    prm: dict | None = None,
) -> tuple[str, str | None]:
    """RFC 7591 Dynamic Client Registration. Returns ``(client_id, client_secret|None)``.

    Registers the gateway as an OAuth client at the provider's Authorization
    Server so a static operator app is not required. When the provider's
    ``token_endpoint_auth_style`` is ``NONE`` the registration requests
    ``token_endpoint_auth_method=none`` (public PKCE client); otherwise
    ``client_secret_post`` (confidential client). ``redirect_uri`` MUST be the
    gateway callback the consent/exchange legs use, or the AS rejects the later
    authorize request.

    ``prm`` may be a pre-fetched PRM dict; when provided it is forwarded to
    ``_discover_registration_url`` so the PRM is not fetched a second time.
    """
    is_none_style = cfg.token_endpoint_auth_style == TokenEndpointAuthStyle.NONE
    reg_url = await _discover_registration_url(cfg, prm=prm)
    body: dict = {
        "client_name": cfg.dcr_client_name,
        "redirect_uris": [redirect_uri],
        "grant_types": ["authorization_code", "refresh_token"],
        "response_types": ["code"],
        "token_endpoint_auth_method": "none" if is_none_style else "client_secret_post",
    }
    if scopes:
        body["scope"] = cfg.scope_separator.join(scopes)
    payload = await _post_dcr(reg_url, body)
    client_id = payload.get("client_id")
    if not client_id:
        raise OAuthEngineError("DCR response missing client_id")
    return client_id, (payload.get("client_secret") or None)
