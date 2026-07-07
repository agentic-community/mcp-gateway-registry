"""Egress credential vault API routes.

- POST /internal/egress-token: internal vend endpoint for auth_server's
  mcp_proxy hop.
- POST/GET /servers/{path}/egress-auth: operator config (admin-only).
- POST /egress-auth/initiate, GET /oauth2/egress/callback,
  GET/DELETE /egress-auth/connections/...: end-user consent + management.

Security model for POST /internal/egress-token:
- validate_internal_auth gates the caller (auth_server presents a fresh
  mcp-registry-audience service token) -- bound to the internal network.
- The forwarded X-Internal-Token (the mcp-proxy token /validate minted) is
  RE-VERIFIED here; sub + auth_method are taken from the verified claims,
  never from the request body.
- Non-per-user auth_method is rejected so a static-key/federation caller
  can never address a per-user vault bucket.
- claims["upstream_url"] is cross-checked against the server's registered
  proxy_pass_url union so a forged X-Resolved-Upstream (minted via a
  direct /validate call) cannot vend a token to an attacker-controlled host.
"""

import logging
import secrets
from html import escape
from typing import Annotated
from urllib.parse import urlencode, urlparse

from fastapi import APIRouter, Depends, Header, HTTPException, Request, status
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from registry.auth.csrf import verify_csrf_token_flexible
from registry.auth.dependencies import nginx_proxied_auth
from registry.auth.internal import validate_internal_auth
from registry.auth.proxied_token import verify_mcp_proxy_token
from registry.core.config import settings
from registry.core.schemas import _is_gateway_own_audience
from registry.egress_auth.factory import get_egress_auth_service
from registry.egress_auth.providers import list_provider_names, resolve_provider
from registry.egress_auth.service import EgressAuthError, is_per_user_auth_method
from registry.exceptions import UrlValidationError
from registry.repositories.factory import get_server_repository
from registry.services.server_service import server_service
from registry.utils.credential_encryption import encrypt_credential
from registry.utils.url_guard import PROXY_PROFILE, validate_url

logger = logging.getLogger(__name__)

router = APIRouter()


def _feature_enabled_or_404() -> None:
    if not settings.egress_auth_enabled:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="egress auth disabled")


def _callback_url() -> str:
    return settings.egress_oauth_callback_base.rstrip("/") + "/oauth2/egress/callback"


def _build_connect_url(server_path: str) -> str:
    """Build the session-verified elicitation front-door URL for a server.

    This is the ``url`` the MCP client opens for URL-mode elicitation. It points
    at the gateway's own ``/oauth2/egress/connect`` (mounted at the registry
    root, with ROOT_PATH), which verifies the opener's session before starting
    the provider consent -- so the client itself performs NO OAuth/DCR.
    """
    base = settings.registry_url.rstrip("/")
    path = server_path if server_path.startswith("/") else "/" + server_path
    return f"{base}/oauth2/egress/connect?{urlencode({'server': path})}"


def _build_request_state(
    user_id: str,
    auth_method: str,
    provider: str,
    server_path: str,
    client_id_audit: str,
) -> str | None:
    """Build the MRTR ``requestState`` blob the client echoes on retry.

    Reuses the egress AEAD ``OAuthState`` codec: the blob is integrity-protected
    and carries the principal + server + issue time, satisfying the MRTR
    requirement to reject tampered, replayed, or cross-user retries. It carries
    no ``pkce_verifier`` (this is the client<->gateway retry binding, not a
    provider-leg state). Returns None if state cannot be built (non-fatal: the
    elicitation still works via the unchanged bearer + vault re-check on retry).
    """
    from datetime import UTC, datetime

    from registry.egress_auth.schemas import OAuthState
    from registry.egress_auth.state_codec import encode_state

    try:
        state = OAuthState(
            user_id=user_id,
            auth_method=auth_method,
            client_id=client_id_audit,
            provider=provider,
            server_path=server_path,
            session_id="",
            pkce_verifier=None,
            nonce=secrets.token_urlsafe(16),
            issued_at=datetime.now(UTC).isoformat(),
        )
        return encode_state(state)
    except Exception as exc:
        logger.warning("egress vend: could not build request_state: %s", exc)
        return None


class EgressTokenRequest(BaseModel):
    """Body for POST /internal/egress-token.

    server_path identifies the registered server whose egress config + upstream
    allowlist the vend is checked against. Identity (sub/auth_method) is NOT in
    the body -- it is re-derived from the forwarded mcp-proxy token.
    """

    server_path: str


class EgressTokenResponse(BaseModel):
    """Vend result. ``access_token`` is None on a clean miss (consent required).

    On a consent-required miss for a properly egress-configured server, the
    registry builds the provider ``authorize_url`` so mcp_proxy can return it to
    the caller (the user clicks it to connect). ``authorize_url`` is None when
    the server isn't egress-configured / the caller isn't a per-user principal
    (nothing to connect)."""

    access_token: str | None = None
    consent_required: bool = False
    authorize_url: str | None = None
    connect_url: str | None = Field(
        default=None,
        description="Session-verified gateway front door for MCP URL-mode "
        "elicitation (``GET /oauth2/egress/connect?server=<path>``). The mcp_proxy "
        "hop puts this in the ``elicitation/create`` ``url`` field. Unlike "
        "``authorize_url`` (the provider-direct URL), this route re-verifies the "
        "opener's gateway session against the elicited principal (anti-phishing) "
        "and needs NO client-side OAuth/DCR -- so it works with providers like "
        "Entra that do not support RFC 7591 DCR.",
    )
    request_state: str | None = Field(
        default=None,
        description="Opaque AEAD blob the MCP client echoes back on retry "
        "(MRTR ``requestState``). Binds the principal + server + issue time so a "
        "tampered/replayed/cross-user retry is rejected. Built with the egress "
        "OAuth state codec.",
    )
    provider: str | None = Field(
        default=None,
        description="Provider key (github/google/entra/...) for the human-readable "
        "elicitation message.",
    )
    # obo_exchange directive (returned instead of a token; the exchange runs in
    # auth_server, which holds the gateway's IdP creds and the raw ingress JWT).
    mode: str | None = Field(
        default=None,
        description="Egress mode for this server: 'obo_exchange' when the caller "
        "should perform a same-IdP OBO token exchange instead of a vault vend.",
    )
    obo_target_audience: str | None = Field(
        default=None,
        description="obo_exchange: the 'aud' the auth_server requests in the OBO exchange.",
    )
    obo_scopes: list[str] | None = Field(
        default=None,
        description="obo_exchange: audience-scoped scopes for the exchange request.",
    )


def _base_url(url: str) -> str:
    """scheme://host[:port] of a URL, lowercased -- the comparison surface for the upstream cross-check.

    The mcp_proxy sub-path append is confined to the bound host, so the cross-check
    compares the BASE (scheme+host+port), not the full post-append path.
    """
    p = urlparse(url)
    return f"{(p.scheme or '').lower()}://{(p.netloc or '').lower()}"


def _registered_upstreams(server: dict) -> set[str]:
    """The legal upstream base-URL set for a server: proxy_pass_url ∪ versions[*]."""
    bases: set[str] = set()
    if server.get("proxy_pass_url"):
        bases.add(_base_url(server["proxy_pass_url"]))
    for ver in server.get("versions") or []:
        ppu = (
            ver.get("proxy_pass_url")
            if isinstance(ver, dict)
            else getattr(ver, "proxy_pass_url", None)
        )
        if ppu:
            bases.add(_base_url(ppu))
    return bases


@router.post("/internal/egress-token", response_model=EgressTokenResponse)
async def vend_egress_token(
    body: EgressTokenRequest,
    _caller: Annotated[str, Depends(validate_internal_auth)],
    x_internal_token: Annotated[str | None, Header(alias="X-Internal-Token")] = None,
) -> EgressTokenResponse:
    """Vend a per-user egress access token for the mcp_proxy hop.

    Returns 401 if the feature is off or the forwarded mcp-proxy token is
    missing/invalid; a clean miss (no connection, non-per-user caller, upstream
    mismatch, etc.) returns ``consent_required=True`` with no token.
    """
    if not settings.egress_auth_enabled:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="egress auth disabled")

    if not x_internal_token:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, detail="missing X-Internal-Token")

    # Independently re-verify the mcp-proxy token; identity is the verified
    # claim, never an asserted body field.
    claims = verify_mcp_proxy_token(x_internal_token)
    # Egress vault user id: the canonical OIDC-sub-based ``egress_user`` claim,
    # which /validate stamps identically into the consent-write and vend tokens
    # so one human maps to one bucket across providers (see _canonical_egress_user
    # in auth_server; the browser-consent and bearer-vend paths otherwise diverged
    # when an IdP omits preferred_username from access tokens, e.g. Entra). Fall
    # back to ``sub`` for tokens minted before this claim existed.
    sub = claims.get("egress_user") or claims.get("sub") or ""
    auth_method = claims.get("auth_method") or ""
    token_upstream = claims.get("upstream_url") or ""

    # Only real per-user principals may vend.
    if not is_per_user_auth_method(auth_method):
        logger.info("egress vend: non-per-user auth_method %r -> consent", auth_method)
        return EgressTokenResponse(consent_required=True)

    # Normalize the server path: mcp_proxy passes the first path segment without a
    # leading slash ("github"), but server entries, the vault key, and the consent
    # state all use the slash-prefixed path ("/github"). Without this, the lookup
    # misses and consent loops forever. Use the canonical form everywhere below.
    server_path = body.server_path if body.server_path.startswith("/") else "/" + body.server_path

    server = await get_server_repository().get(server_path)
    if server is None:
        return EgressTokenResponse(consent_required=True)

    # Per-server enablement: a misconfigured/half-deleted server never vends.
    egress_mode = server.get("egress_auth_mode")
    if egress_mode not in ("oauth_user", "obo_exchange") or not server.get("egress_oauth"):
        return EgressTokenResponse(consent_required=True)

    # The bound upstream MUST match a registered upstream for this server. This
    # cross-check applies to BOTH egress modes: an OBO directive must only be
    # handed out for a legitimately-bound upstream, same as a vault vend.
    legal = _registered_upstreams(server)
    if _base_url(token_upstream) not in legal:
        logger.warning(
            "egress vend REFUSED: upstream %r not in registered set %r for %s",
            _base_url(token_upstream),
            legal,
            server_path,
        )
        raise HTTPException(
            status.HTTP_403_FORBIDDEN, detail="upstream not registered for this server"
        )

    egress_oauth = server["egress_oauth"]

    # obo_exchange: return the exchange DIRECTIVE, not a token. The actual IdP
    # token exchange runs in auth_server (which holds the gateway's own IdP
    # credentials and the raw ingress JWT); the registry never sees the JWT and
    # holds no per-user token for this mode. Stateless -- no vault lookup.
    if egress_mode == "obo_exchange":
        return EgressTokenResponse(
            mode="obo_exchange",
            obo_target_audience=egress_oauth.get("target_audience"),
            obo_scopes=egress_oauth.get("scopes") or [],
        )

    svc = get_egress_auth_service()
    access_token = await svc.get_valid_token(
        auth_method=auth_method,
        user_id=sub,
        server_path=server_path,
        egress_oauth=egress_oauth,
    )
    if access_token is not None:
        return EgressTokenResponse(access_token=access_token)

    # Miss: no usable token (never connected, or refresh dead). Build the consent
    # URL so mcp_proxy can hand it back to the user to self-serve (the gateway
    # triggers consent automatically rather than forwarding unauthenticated).
    try:
        authorize_url = svc.build_consent_url(
            auth_method=auth_method,
            user_id=sub,
            client_id_audit=claims.get("client_id") or "",
            session_id="",
            server_path=server_path,
            egress_oauth=egress_oauth,
        )
    except Exception as exc:  # bad provider config etc. -- still a clean miss
        logger.warning("egress vend: could not build consent URL: %s", exc)
        authorize_url = None

    # MCP URL-mode elicitation: a session-verified gateway front door the client
    # opens verbatim (no client-side OAuth/DCR -- works with Entra). The matching
    # ``request_state`` is an AEAD blob the client echoes back on retry; it binds
    # the principal + server + issue time so a tampered/replayed/cross-user retry
    # is rejected (MRTR requestState integrity requirement).
    provider = egress_oauth.get("provider")
    connect_url = _build_connect_url(server_path)
    request_state = _build_request_state(
        user_id=sub,
        auth_method=auth_method,
        provider=provider or "",
        server_path=server_path,
        client_id_audit=claims.get("client_id") or "",
    )

    return EgressTokenResponse(
        consent_required=True,
        authorize_url=authorize_url,
        connect_url=connect_url,
        request_state=request_state,
        provider=provider,
    )


# ---------------------------------------------------------------------------- #
# Public endpoints. Operator config + end-user consent/connections.
# ---------------------------------------------------------------------------- #


class EgressConfigRequest(BaseModel):
    """Configure egress auth on a server (admin/registrant)."""

    egress_auth_mode: str = "oauth_user"  # "none" | "oauth_user" | "obo_exchange"
    egress_provider: str = ""
    client_id: str = ""
    client_secret: str | None = None  # write-only; encrypted, never echoed
    scopes: list[str] = []
    custom_authorize_url: str | None = None
    custom_token_url: str | None = None
    custom_scope_separator: str | None = None
    custom_token_auth_style: str | None = None
    # obo_exchange only: the internal MCP server's audience (IdP-shaped).
    target_audience: str | None = None


def _egress_config_view(server: dict) -> dict:
    """Non-secret view of a server's egress config + the callback URL to register."""
    eo = server.get("egress_oauth") or {}
    return {
        "path": server.get("path"),
        "egress_auth_mode": server.get("egress_auth_mode", "none"),
        "egress_provider": eo.get("provider"),
        "scopes": eo.get("scopes", []),
        "target_audience": eo.get("target_audience"),
        "callback_url": _callback_url(),
        "custom_authorize_url": eo.get("custom_authorize_url"),
        "custom_token_url": eo.get("custom_token_url"),
    }


def _require_admin(user_context: dict) -> None:
    if not user_context.get("is_admin"):
        raise HTTPException(status.HTTP_403_FORBIDDEN, detail="admin required")


@router.post("/servers/{server_path:path}/egress-auth")
async def configure_egress_auth(
    request: Request,
    server_path: str,
    body: EgressConfigRequest,
    user_context: Annotated[dict, Depends(nginx_proxied_auth)],
    _csrf: Annotated[None, Depends(verify_csrf_token_flexible)],
):
    """Configure (or disable) per-user egress OAuth on a server. Admin only.

    The client_secret is Fernet-encrypted and never returned. Returns the
    callback URL the operator must register in the provider's OAuth app.
    """
    _feature_enabled_or_404()
    _require_admin(user_context)

    if not server_path.startswith("/"):
        server_path = "/" + server_path

    server = await server_service.get_server_info(server_path, include_credentials=True)
    if not server:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="server not found")

    if body.egress_auth_mode == "none":
        server["egress_auth_mode"] = "none"
        server["egress_oauth"] = None
    elif body.egress_auth_mode == "oauth_user":
        if body.egress_provider not in list_provider_names():
            raise HTTPException(
                status.HTTP_400_BAD_REQUEST,
                detail=f"unknown provider; valid: {list_provider_names()}",
            )
        eo: dict = {
            "provider": body.egress_provider,
            "client_id": body.client_id,
            "scopes": body.scopes,
            "custom_authorize_url": body.custom_authorize_url,
            "custom_token_url": body.custom_token_url,
            "custom_scope_separator": body.custom_scope_separator,
            "custom_token_auth_style": body.custom_token_auth_style,
        }
        # For a 'custom' provider the authorize/token URLs are registrant-supplied
        # and become an outbound token POST (carrying the client_secret) and a
        # browser 302. Fail closed at registration: require https and reject any
        # literal private/metadata IP or bad scheme via the shared SSRF guard, so
        # a config that would exfiltrate the secret to an internal target (e.g.
        # 169.254.169.254) can never be persisted. resolve=False keeps this a
        # structural check; the rebinding-safe block for hostname targets is the
        # pinned guarded client at token-exchange time.
        if body.egress_provider == "custom":
            for field, url in (
                ("custom_authorize_url", body.custom_authorize_url),
                ("custom_token_url", body.custom_token_url),
            ):
                try:
                    validate_url(
                        url or "",
                        profile=PROXY_PROFILE,
                        require_https=True,
                        resolve=False,
                    )
                except UrlValidationError as exc:
                    raise HTTPException(
                        status.HTTP_400_BAD_REQUEST,
                        detail=f"{field} rejected: {exc}",
                    ) from exc
        # Validate provider resolution (custom requires URLs) before persisting.
        try:
            resolve_provider(eo)
        except ValueError as exc:
            raise HTTPException(status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
        # Encrypt the secret; keep the prior one if the field is omitted on edit.
        if body.client_secret:
            eo["client_secret_encrypted"] = encrypt_credential(body.client_secret)
        else:
            prior = (server.get("egress_oauth") or {}).get("client_secret_encrypted")
            eo["client_secret_encrypted"] = prior
        if not eo["client_secret_encrypted"]:
            raise HTTPException(status.HTTP_400_BAD_REQUEST, detail="client_secret required")
        server["egress_auth_mode"] = "oauth_user"
        server["egress_oauth"] = eo
    elif body.egress_auth_mode == "obo_exchange":
        target = (body.target_audience or "").strip()
        if not target:
            raise HTTPException(
                status.HTTP_400_BAD_REQUEST,
                detail="obo_exchange requires target_audience",
            )
        if _is_gateway_own_audience(target):
            raise HTTPException(
                status.HTTP_400_BAD_REQUEST,
                detail="target_audience must differ from the gateway's own IdP client id",
            )
        # Same-IdP exchange: no per-server provider/client_id/secret. Only the
        # target audience and (optional) audience-scoped scopes are stored.
        server["egress_auth_mode"] = "obo_exchange"
        server["egress_oauth"] = {
            "target_audience": target,
            "scopes": body.scopes,
        }
    else:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, detail="invalid egress_auth_mode")

    if not await server_service.update_server(server_path, server):
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, detail="update failed")
    return _egress_config_view(server)


@router.get("/servers/{server_path:path}/egress-auth")
async def get_egress_auth_config(
    server_path: str,
    user_context: Annotated[dict, Depends(nginx_proxied_auth)],
):
    """Read a server's egress config (secret stripped). Admin only."""
    _feature_enabled_or_404()
    _require_admin(user_context)
    if not server_path.startswith("/"):
        server_path = "/" + server_path
    server = await server_service.get_server_info(server_path, include_credentials=False)
    if not server:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="server not found")
    return _egress_config_view(server)


@router.get("/egress-auth/available-servers")
async def list_available_egress_servers(
    user_context: Annotated[dict, Depends(nginx_proxied_auth)],
):
    """List egress-enabled servers the current user can access (for the
    Connected Accounts dropdown).

    Returns only servers with ``egress_auth_mode == 'oauth_user'`` AND a valid
    ``egress_oauth`` config, intersected with the user's accessible servers, so
    a user is never offered a server they cannot reach. Tokens/secrets are never
    included -- only path, display name, and provider.
    """
    _feature_enabled_or_404()

    # Only per-user principals can connect an account; non-per-user callers
    # (e.g. federation/network-trusted) get an empty list rather than an error.
    auth_method = user_context.get("auth_method") or ""
    if not is_per_user_auth_method(auth_method):
        return []

    # "*" in accessible_servers = unrestricted (admin); otherwise the user only
    # sees servers whose path is explicitly granted. accessible_servers stores
    # names WITHOUT a leading slash (e.g. "slack") while server paths carry one
    # (e.g. "/slack"), so compare on the slash-stripped form.
    accessible = user_context.get("accessible_servers") or []
    unrestricted = "*" in accessible
    accessible_norm = {s.lstrip("/") for s in accessible}
    all_servers = await server_service.get_all_servers()
    results: list[dict] = []
    for path, server in all_servers.items():
        if server.get("egress_auth_mode") != "oauth_user" or not server.get("egress_oauth"):
            continue
        if not unrestricted and str(path).lstrip("/") not in accessible_norm:
            continue
        eo = server.get("egress_oauth") or {}
        results.append(
            {
                "server_path": path,
                "server_name": server.get("server_name") or path,
                "provider": eo.get("provider") or "custom",
            }
        )
    results.sort(key=lambda r: r["server_name"].lower())
    return results


@router.get("/egress/obo-identifier-uris")
async def get_obo_identifier_uris(
    user_context: Annotated[dict, Depends(nginx_proxied_auth)],
):
    """List the Entra Application ID URIs the operator must register.

    Each server that logs the client in at the gateway via a per-server PRM
    (``obo_exchange`` and the 3LO vault's ``oauth_user`` ingress leg) has a
    per-server resource URL -- the value the gateway advertises in its PRM and
    validates as the ingress ``aud``. On Entra, every one of those URLs must be
    present in the gateway app's ``identifierUris`` list. This endpoint returns
    the exact set so the operator can keep Entra in sync as such servers are
    added/removed -- the registry side is automatic; only this list is manual.

    Admin only. Returns ``{"identifier_uris": [...], "count": N}``.
    """
    _feature_enabled_or_404()
    _require_admin(user_context)

    from registry.api.wellknown_routes import server_needs_per_server_prm
    from registry.auth.oauth_metadata import build_per_server_resource_url
    from registry.core.config import settings

    servers = await server_service.get_all_servers(include_inactive=True)
    uris: list[str] = []
    for path, info in (servers or {}).items():
        if not server_needs_per_server_prm((info or {}).get("egress_auth_mode")):
            continue
        append_mcp = info.get("append_mcp_path") is not False
        uris.append(
            build_per_server_resource_url(settings.registry_url, path, append_mcp=append_mcp)
        )
    uris = sorted(set(uris))
    return {"identifier_uris": uris, "count": len(uris)}


class InitiateRequest(BaseModel):
    server_path: str


@router.post("/egress-auth/initiate")
async def initiate_consent(
    request: Request,
    body: InitiateRequest,
    user_context: Annotated[dict, Depends(nginx_proxied_auth)],
    _csrf: Annotated[None, Depends(verify_csrf_token_flexible)],
):
    """Begin the OAuth consent for the current user; returns the authorize URL."""
    _feature_enabled_or_404()
    server_path = body.server_path
    if not server_path.startswith("/"):
        server_path = "/" + server_path
    server = await server_service.get_server_info(server_path, include_credentials=True)
    if (
        not server
        or server.get("egress_auth_mode") != "oauth_user"
        or not server.get("egress_oauth")
    ):
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST, detail="server has no per-user egress auth configured"
        )

    auth_method = user_context.get("auth_method") or ""
    if not is_per_user_auth_method(auth_method):
        raise HTTPException(
            status.HTTP_403_FORBIDDEN, detail="this caller cannot connect a per-user account"
        )

    url = get_egress_auth_service().build_consent_url(
        auth_method=auth_method,
        # Canonical egress user (OIDC sub, else username): must match the id the
        # vend path derives from the mcp-proxy token so one human maps to one
        # vault bucket regardless of token type / provider. See #933.
        user_id=user_context.get("egress_user") or user_context.get("username") or "",
        client_id_audit=user_context.get("client_id") or "",
        session_id=user_context.get("session_id") or "",
        server_path=server_path,
        egress_oauth=server["egress_oauth"],
    )
    return {"authorize_url": url}


@router.get("/oauth2/egress/callback")
async def egress_callback(
    request: Request,
    code: str = "",
    state: str = "",
):
    """Provider redirect target. No ingress auth -- the signed+encrypted state is
    the authority. Verifies state (TTL + single-use + account-swap), exchanges the
    code, and stores the token. Reached via nginx -> registry (no /validate)."""
    _feature_enabled_or_404()
    if not code or not state:
        return HTMLResponse("<h3>Connection failed: missing code/state.</h3>", status_code=400)

    # The provider+server are bound in the signed state; we resolve the server's
    # egress config to get client_id/secret for the code exchange. We decode the
    # state-bound server_path indirectly via handle_callback, so fetch by the
    # state after a light pre-decode is avoided -- instead the service needs the
    # egress_oauth; resolve it from the state's server_path.
    from registry.egress_auth.state_codec import InvalidState, decode_state

    try:
        st = decode_state(state)
    except InvalidState:
        return HTMLResponse("<h3>Connection failed: invalid state.</h3>", status_code=400)

    server = await server_service.get_server_info(st.server_path, include_credentials=True)
    if not server or not server.get("egress_oauth"):
        return HTMLResponse("<h3>Connection failed: server not configured.</h3>", status_code=400)

    # Account-swap guard: cross-check the live session principal when present.
    # The provider redirect often lands in a fresh tab with a valid session
    # cookie (same browser), in which case we enforce it; if there is no live
    # session, the signed+single-use state remains the authority (handle_callback
    # still enforces TTL + replay + the state-bound (user, auth_method)).
    current_user = None
    current_method = None
    session_cookie = request.cookies.get(settings.session_cookie_name)
    if session_cookie:
        try:
            # Pass the cookie explicitly: nginx_proxied_auth's `session` is a
            # FastAPI Cookie(...) param only populated by dependency injection, so
            # a direct call without it always sees session=None (the account-swap
            # guard would silently never engage).
            ctx = await nginx_proxied_auth(request, session=session_cookie)
            # Compare on the canonical egress user (OIDC sub, else username) so the
            # account-swap guard matches the id the consent state was built with.
            current_user = ctx.get("egress_user") or ctx.get("username")
            current_method = ctx.get("auth_method")
        except Exception:
            pass

    try:
        conn = await get_egress_auth_service().handle_callback(
            code=code,
            state_blob=state,
            egress_oauth=server["egress_oauth"],
            current_user_id=current_user,
            current_auth_method=current_method,
        )
    except EgressAuthError as exc:
        # Detail to server logs only. Do NOT reflect the exception text into the
        # browser response: EgressAuthError messages embed internal state (e.g.
        # decryption / SECRET_KEY hints, wrapped upstream errors) — a
        # stack-trace/internal-detail exposure. Show a generic message.
        logger.warning("egress callback failed: %s", exc)
        return HTMLResponse(
            "<h3>Connection failed. Please close this tab and try connecting again.</h3>",
            status_code=400,
        )

    # The egress consent is the web Connected-Accounts / MCP URL-mode elicitation
    # flow: the token is now vaulted, so show the close-tab page and let the user
    # retry their original request. HTML-escape the interpolated values: the
    # server_path is admin-registrant-supplied and validate_server_path blocks
    # nginx metacharacters but not '<'/'>' (a different sink), so escape here to
    # keep a crafted path (e.g. /<svg onload=...>) from executing in the browser.
    return HTMLResponse(
        f"<h3>Connected {escape(conn.provider)} for {escape(conn.server_path)}.</h3>"
        "<p>You can close this tab and retry your request.</p>"
    )


@router.get("/egress-auth/connections")
async def list_connections(
    user_context: Annotated[dict, Depends(nginx_proxied_auth)],
):
    """List the current user's egress connections (tokens stripped)."""
    _feature_enabled_or_404()
    conns = await get_egress_auth_service().list_connections(
        auth_method=user_context.get("auth_method") or "",
        user_id=user_context.get("egress_user") or user_context.get("username") or "",
    )
    return [c.model_dump() for c in conns]


@router.delete("/egress-auth/connections/{provider}/{server_path:path}")
async def disconnect(
    request: Request,
    provider: str,
    server_path: str,
    user_context: Annotated[dict, Depends(nginx_proxied_auth)],
    _csrf: Annotated[None, Depends(verify_csrf_token_flexible)],
):
    """Delete the current user's vault entry for (provider, server_path)."""
    _feature_enabled_or_404()
    if not server_path.startswith("/"):
        server_path = "/" + server_path
    await get_egress_auth_service().disconnect(
        auth_method=user_context.get("auth_method") or "",
        user_id=user_context.get("egress_user") or user_context.get("username") or "",
        provider=provider,
        server_path=server_path,
    )
    return {"status": "revoked"}
