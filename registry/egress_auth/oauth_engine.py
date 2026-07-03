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


def _expires_at(expires_in: int | None) -> str | None:
    if not expires_in:
        return None
    return (datetime.now(UTC) + timedelta(seconds=int(expires_in))).isoformat()


def _build_token_request(
    cfg: OAuthProviderConfig,
    client_id: str,
    client_secret: str,
    form: dict,
) -> tuple[dict, dict]:
    """Return (form_data, headers), placing the client secret per the provider's style."""
    headers = {"Accept": "application/json"}
    data = dict(form)
    if cfg.token_endpoint_auth_style == TokenEndpointAuthStyle.BASIC_HEADER:
        basic = base64.b64encode(f"{client_id}:{client_secret}".encode()).decode()
        headers["Authorization"] = f"Basic {basic}"
        data["client_id"] = client_id
    else:  # POST_BODY (default)
        data["client_id"] = client_id
        data["client_secret"] = client_secret
    return data, headers


async def _post_token(cfg: OAuthProviderConfig, data: dict, headers: dict) -> dict:
    try:
        async with httpx.AsyncClient(timeout=_HTTP_TIMEOUT) as client:
            resp = await client.post(cfg.token_url, data=data, headers=headers)
    except httpx.HTTPError as exc:
        raise OAuthEngineError(f"token endpoint unreachable: {exc}") from exc

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
        expires_at=_expires_at(parsed.get("expires_in")),
        scopes=[s for s in scopes if s],
        status="active",
        client_id=client_id,
        created_at=now,
        last_refreshed_at=now,
    )


async def exchange_code(
    cfg: OAuthProviderConfig,
    client_id: str,
    client_secret: str,
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
    data, headers = _build_token_request(cfg, client_id, client_secret, form)
    payload = await _post_token(cfg, data, headers)
    return _to_stored_token(cfg, payload, client_id)


async def refresh_token(
    cfg: OAuthProviderConfig,
    client_id: str,
    client_secret: str,
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
    data, headers = _build_token_request(cfg, client_id, client_secret, form)
    payload = await _post_token(cfg, data, headers)
    return _to_stored_token(cfg, payload, client_id, fallback_refresh=refresh_token_value)
