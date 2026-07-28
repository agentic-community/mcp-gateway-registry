"""Egress consent route: the front door for MCP URL-mode elicitation.

When an MCP tool call needs a per-user third-party token (e.g. GitHub) that the
vault does not yet hold, the gateway returns an ``elicitation/create``
``mode: "url"`` pointing the user's client at ``GET /oauth2/egress/connect``.
This module serves that route.

The client performs NO OAuth itself: it just opens the connect URL (with user
consent) and retries the original tool call. The connect route session-verifies
the opener, then starts the real provider (GitHub/Google/Entra/...) OAuth leg.
The provider leg reuses the existing ``/oauth2/egress/callback`` in
``egress_auth_routes.py``; the third-party token stays in the vault and never
transits the MCP client.

SECURITY MODEL
--------------
- ``/connect`` REQUIRES a live gateway session; the principal that starts the
  consent is the SESSION principal, never anything the client asserts. This is
  the elicitation anti-phishing requirement: a stolen connect URL opened by a
  different user cannot bind a token to the elicited user (the provider-leg AEAD
  state binds the token write to the authenticated principal). When no session
  exists (first IDE use), it redirects the browser through the gateway's
  Keycloak login and returns to ``/connect``.
- ``/connect`` takes NO client-side OAuth parameters (no redirect_uri / PKCE /
  Dynamic Client Registration), so it works with providers (e.g. Entra) that do
  not support RFC 7591 DCR.
"""

import logging
from urllib.parse import urlencode

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, RedirectResponse

from registry.auth.dependencies import nginx_proxied_auth
from registry.core.config import settings
from registry.egress_auth import as_facade
from registry.egress_auth.factory import get_egress_auth_service
from registry.egress_auth.service import is_per_user_auth_method
from registry.services.server_service import server_service

logger = logging.getLogger(__name__)

router = APIRouter()


def _gateway_origin(request: Request) -> str:
    """Return the public gateway ORIGIN (scheme://host, no ROOT_PATH path).

    The browser-facing auth routes (``/oauth2/login/*``, ``/oauth2/callback/*``)
    are served by nginx at the ORIGIN ROOT in BOTH routing modes -- they are
    NOT mounted under the registry's ROOT_PATH. So a login bounce must target
    ``{origin}/oauth2/login/keycloak``, NOT ``{registry_url}/oauth2/login/...``
    (in path mode registry_url carries ``/registry`` and the prefixed login path
    falls through nginx to the SPA -> HTML -> the login never happens, which is
    the path-mode-only egress consent failure; subdomain mode worked because
    registry_url has no path so the two coincide).

    Derive the origin from settings.registry_url (authoritative public URL,
    HTTPS-correct behind the ALB) rather than the request Host, stripping any
    path component.
    """
    from urllib.parse import urlsplit

    parts = urlsplit(settings.registry_url)
    return f"{parts.scheme}://{parts.netloc}"


def _feature_on() -> bool:
    return bool(settings.egress_auth_enabled)


async def _optional_session(request: Request) -> dict | None:
    """Read the gateway session, returning None instead of raising when there is
    no valid session (so /connect can bounce to login).

    ``nginx_proxied_auth`` declares ``session`` as a FastAPI ``Cookie(...)``
    parameter that is only populated by dependency injection. Since we call it
    directly (not as a route dependency), we must extract the session cookie off
    the request and pass it explicitly -- otherwise it always sees ``session=None``
    and reports "no session" even when the cookie is present (the login-loop bug)."""
    session = request.cookies.get(settings.session_cookie_name)
    try:
        return await nginx_proxied_auth(request, session=session)
    except Exception:
        return None


def _login_bootstrap_redirect(request: Request) -> RedirectResponse:
    """302 the browser through the gateway's Keycloak login, returning to this
    exact /connect URL (with its query) once a session is established.

    Uses the auth-server's validated same-origin redirect_uri round-trip. The
    return URL is the request's own path+query on the gateway origin."""
    return_to = request.url.path
    if request.url.query:
        return_to = f"{return_to}?{request.url.query}"
    # /oauth2/login/* is served at the ORIGIN root in both routing modes, so the
    # login URL must NOT carry the registry ROOT_PATH (see _gateway_origin).
    login = f"{_gateway_origin(request)}/oauth2/login/keycloak?" + urlencode(
        {"redirect_uri": return_to}
    )
    logger.info("egress connect: no session -> bounce to gateway login")
    return RedirectResponse(login, status_code=302)


@router.get("/oauth2/egress/connect")
async def egress_connect(
    request: Request,
    server: str = "",
) -> object:
    """Session-verified front door for MCP URL-mode elicitation.

    This is the ``url`` the MCP client opens (verbatim, with user consent) when
    the gateway returns an ``elicitation/create`` ``mode: "url"`` for an egress
    server the user has not connected yet. It takes NO client-side OAuth
    parameters (no ``redirect_uri``/PKCE/``response_type`` and no Dynamic Client
    Registration) -- the CLIENT performs no OAuth at all, so this works with
    providers like Entra that do not support RFC 7591 DCR.

    Flow:
      1. If the browser has no gateway session, bounce through Keycloak login and
         return here (so the opener is always an authenticated principal).
      2. Verify the session principal is a per-user identity. This is the
         elicitation anti-phishing requirement: the user who OPENS the URL must
         be a real authenticated user, and the provider-leg AEAD state binds the
         token write to THIS principal -- a stolen connect URL opened by a
         different user cannot bind a token to the elicited user.
      3. 302 the browser to the provider consent (``build_consent_url``). The
         existing ``/oauth2/egress/callback`` stores the token and shows the
         close-tab page; the client then retries the original tool call (same
         bearer) and the vend now HITs.
    """
    if not _feature_on():
        return JSONResponse({"error": "not_found"}, status_code=404)

    server_path = server or request.query_params.get("resource", "")
    server_path = _server_path_from_resource(server_path) or server_path
    if not server_path:
        return JSONResponse(
            {"error": "invalid_request", "error_description": "missing server"},
            status_code=400,
        )
    server_path = as_facade._normalize_server_path(server_path)

    server_info = await server_service.get_server_info(server_path, include_credentials=True)
    if not as_facade.is_server_egress_configured(server_info):
        return JSONResponse(
            {"error": "invalid_request", "error_description": "server has no per-user egress auth"},
            status_code=400,
        )

    # Session bootstrap: no gateway session -> log in via Keycloak, return here.
    user_context = await _optional_session(request)
    if user_context is None:
        return _login_bootstrap_redirect(request)

    auth_method = user_context.get("auth_method") or ""
    if not is_per_user_auth_method(auth_method):
        return JSONResponse(
            {
                "error": "access_denied",
                "error_description": "this caller cannot connect a per-user account",
            },
            status_code=403,
        )

    # Bind the consent state to the canonical egress principal (OIDC-sub-based
    # ``egress_user``, else ``username``) -- the SAME id the vend, the vault, and
    # the callback account-swap guard use. Using bare ``username`` here would bind
    # the state to a different id than the callback resolves, so the guard would
    # reject every callback with "state user mismatch" whenever egress_user is set.
    egress_user_id = user_context.get("egress_user") or user_context.get("username") or ""

    # Provider consent leg, via the existing web Connected-Accounts path: the
    # callback stores the token + shows the close-tab page. No client-side code
    # exchange -- the client just retries the original tool call.
    provider_authorize_url = get_egress_auth_service().build_consent_url(
        auth_method=auth_method,
        user_id=egress_user_id,
        client_id_audit=user_context.get("client_id") or "",
        session_id=user_context.get("session_id") or "",
        server_path=server_path,
        egress_oauth=server_info["egress_oauth"],
    )
    logger.info(
        "egress connect: user=%s server=%s -> provider consent",
        egress_user_id,
        server_path,
    )
    return RedirectResponse(provider_authorize_url, status_code=302)


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #


def _server_path_from_resource(resource: str) -> str:
    """Recover the server path from a client-supplied ``resource`` value.

    A client may pass the canonical MCP server identifier ``<registry>/<server>``
    (e.g. ``https://gw.example.com/github`` -> ``/github``). Defensively also
    accepts the well-known PRM document URL form
    (``<registry>/.well-known/oauth-protected-resource/<server>``) -- strip the
    well-known prefix first, then fall back to stripping the registry base.
    Returns "" when neither matches.
    """
    if not resource:
        return ""
    # Form 1: PRM document URL -- strip the well-known prefix.
    marker = as_facade.PRM_WELLKNOWN_PREFIX
    idx = resource.find(marker)
    if idx != -1:
        return resource[idx + len(marker) :] or ""
    # Form 2: canonical resource identifier -- strip the registry base origin.
    base = settings.registry_url.rstrip("/")
    if resource.startswith(base):
        return resource[len(base) :] or ""
    # Last resort: if it looks like a bare path, accept it.
    return resource if resource.startswith("/") else ""
