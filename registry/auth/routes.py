import logging
import os
import re
import urllib.parse
from typing import Annotated

import httpx
from fastapi import APIRouter, Cookie, Depends, Request, status
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates

from registry.observability.meters import (
    logout_id_token_hint_missing_total as logout_id_token_hint_missing,
)
from registry.observability.meters import (
    logout_id_token_hint_present_total as logout_id_token_hint_present,
)
from registry.observability.meters import (
    logout_jwt_validation_failed_total as logout_jwt_validation_failed,
)
from registry.observability.meters import (
    logout_url_length_warning_total as logout_url_length_warning,
)

from ..audit.context import set_audit_action
from ..core.config import settings
from .csrf import generate_csrf_token, verify_csrf_token_flexible

logger = logging.getLogger(__name__)

_ROOT_PATH: str = os.environ.get("ROOT_PATH", "").rstrip("/")

# Server-to-server logout call to auth-server. The timeout is short because the
# hop is cluster-internal (loopback/service-discovery); if it can't answer in a
# few seconds it is down, and the user should not stare at a blank page — the
# local session is already invalidated by that point, so we fall back to /login.
_S2S_LOGOUT_TIMEOUT_SECONDS: float = 5.0


def _fallback_external_host() -> str:
    """Return the host (host[:port]) to use when the inbound Host is untrusted.

    Derived from the configured ``registry_url`` so the redirect target is a
    deployment-controlled hostname rather than an attacker-supplied one. Fails
    closed to ``localhost:7860`` if ``registry_url`` cannot be parsed.
    """
    try:
        parsed = urllib.parse.urlparse(settings.registry_url)
        if parsed.hostname:
            if parsed.port:
                return f"{parsed.hostname}:{parsed.port}"
            return parsed.netloc or parsed.hostname
    except ValueError:
        pass
    return "localhost:7860"


def _resolve_trusted_host(request: Request) -> str:
    """Resolve a trusted host for external-URL construction.

    The inbound ``Host`` header is attacker-controlled and feeds the OAuth
    ``redirect_uri``; a spoofed Host would redirect the login flow to an
    attacker origin. Validate it against the configured allowlist
    (:attr:`settings.trusted_external_hosts_set`) and, on any mismatch or
    missing header, fall back to the deployment's own ``registry_url`` host
    (fail closed).

    Args:
        request: The FastAPI request object.

    Returns:
        A trusted host string (host or host:port).
    """
    host = (request.headers.get("host") or "").strip()
    allowlist = settings.trusted_external_hosts_set

    if host and host.lower() in allowlist:
        return host

    if host:
        logger.warning(
            "Rejected untrusted Host header %r for external URL; "
            "falling back to configured registry host",
            host,
        )
    return _fallback_external_host()


def _build_external_url(
    request: Request,
    path: str = "",
) -> str:
    """Build an external URL with proper scheme, host, and ROOT_PATH.

    The host is validated against a configured allowlist rather than trusting
    the inbound Host header (which feeds the OAuth redirect_uri). An unexpected
    Host falls back to the deployment's own configured host (fail closed).

    Args:
        request: The FastAPI request object
        path: The path to append (e.g., "/logout", "/")

    Returns:
        Full external URL (e.g., "https://host/registry/logout")
    """
    host = _resolve_trusted_host(request)

    cloudfront_proto = request.headers.get("x-cloudfront-forwarded-proto", "")
    x_forwarded_proto = request.headers.get("x-forwarded-proto", "")

    if (
        cloudfront_proto.lower() == "https"
        or x_forwarded_proto.lower() == "https"
        or request.url.scheme == "https"
    ):
        scheme = "https"
    else:
        scheme = "http"

    if "localhost" in host and ":" not in host:
        host = "localhost:7860"

    if path and not path.startswith("/"):
        path = f"/{path}"

    return f"{scheme}://{host}{_ROOT_PATH}{path}"


# Logout observability counters are imported above from
# registry.observability.meters as part of the OTel migration (issue #1122).
# Call sites continue to use ``.inc()`` via the _CounterAdapter shim.

router = APIRouter()

# Templates (will be injected via dependency later, but for now keep it simple)
templates = Jinja2Templates(directory=settings.templates_dir)


def _validate_jwt_format(token: str) -> bool:
    """Validate that a token matches JWT format (header.payload.signature).

    Args:
        token: The token string to validate

    Returns:
        True if token matches JWT format, False otherwise
    """
    jwt_pattern = r"^[A-Za-z0-9-_]+\.[A-Za-z0-9-_]+\.[A-Za-z0-9-_]+$"
    return bool(re.match(jwt_pattern, token))


def _is_safe_logout_redirect(url: str) -> bool:
    """Validate the IdP logout Location before redirecting the browser to it.

    The Location comes from the trusted internal auth-server, but this is a
    cheap defense-in-depth check so a malformed or unexpected value can never
    turn into a ``javascript:``/``data:`` redirect. Absolute URLs must use
    http/https; relative paths are allowed (auth-server falls back to a local
    path when the provider has no logout URL). Fails closed on anything else.

    Args:
        url: The Location header value returned by the auth-server.

    Returns:
        True if the URL is safe to redirect the browser to, False otherwise.
    """
    if not url:
        return False
    parsed = urllib.parse.urlparse(url)
    if not parsed.scheme and not parsed.netloc:
        return True
    return parsed.scheme in ("http", "https")


async def get_oauth2_providers():
    """Fetch available OAuth2 providers from auth server"""
    try:
        async with httpx.AsyncClient() as client:
            logger.info(
                f"Fetching OAuth2 providers from {settings.auth_server_url}/oauth2/providers"
            )
            response = await client.get(f"{settings.auth_server_url}/oauth2/providers", timeout=5.0)
            logger.info(f"OAuth2 providers response: status={response.status_code}")
            if response.status_code == 200:
                data = response.json()
                providers = data.get("providers", [])
                logger.info(f"Successfully fetched {len(providers)} OAuth2 providers: {providers}")
                return providers
            else:
                logger.warning(
                    f"Auth server returned non-200 status: {response.status_code}, body: {response.text}"
                )
    except Exception as e:
        logger.warning(f"Failed to fetch OAuth2 providers from auth server: {e}", exc_info=True)
    return []


@router.get("/login", response_class=HTMLResponse)
async def login_form(request: Request, error: str | None = None):
    """Show login form with OAuth2 providers"""
    oauth_providers = await get_oauth2_providers()
    return templates.TemplateResponse(
        request, "login.html", {"error": error, "oauth_providers": oauth_providers}
    )


@router.get("/auth/{provider}")
async def oauth2_login_redirect(provider: str, request: Request):
    """Redirect to auth server for OAuth2 login"""
    try:
        registry_url = _build_external_url(request, "/")
        auth_external_url = settings.auth_server_external_url
        auth_url = f"{auth_external_url}/oauth2/login/{provider}?redirect_uri={registry_url}"
        logger.info(f"Redirecting to OAuth2 login for provider {provider}: {auth_url}")
        return RedirectResponse(url=auth_url, status_code=302)

    except Exception as e:
        logger.error(f"Error redirecting to OAuth2 login for {provider}: {e}")
        return RedirectResponse(url="/login?error=oauth2_redirect_failed", status_code=302)


@router.get("/auth/callback")
async def oauth2_callback(request: Request, error: str = None, details: str = None):
    """Handle OAuth2 callback from auth server"""
    try:
        if error:
            logger.warning(f"OAuth2 callback received error: {error}, details: {details}")
            error_message = "Authentication failed"
            if error == "oauth2_error":
                # Sanitize user-supplied details to prevent injection
                safe_details = re.sub(r"[^\w\s.:-]", "", str(details or ""))[:200]
                error_message = f"OAuth2 provider error: {safe_details}"
            elif error == "oauth2_init_failed":
                error_message = "Failed to initiate OAuth2 login"
            elif error == "oauth2_callback_failed":
                error_message = "OAuth2 authentication failed"

            # Redirect to /login with URL-encoded error message (safe relative URL)
            safe_redirect = f"/login?error={urllib.parse.quote(error_message)}"
            return RedirectResponse(url=safe_redirect, status_code=302)

        # If we reach here, the auth server should have set the session cookie
        # Verify the session is valid by resolving it against the server-side store.
        session_cookie = request.cookies.get(settings.session_cookie_name)
        if session_cookie:
            from .dependencies import resolve_session_from_cookie

            session_data = await resolve_session_from_cookie(session_cookie)
            if session_data and session_data.get("username"):
                logger.info(
                    f"OAuth2 callback successful for user {session_data['username']} "
                    f"via {session_data.get('auth_method', 'unknown')}"
                )
                return RedirectResponse(url="/", status_code=302)
            logger.warning("Invalid session cookie in OAuth2 callback")

        # If no valid session, redirect to login with error
        logger.warning("OAuth2 callback completed but no valid session found")
        return RedirectResponse(url="/login?error=oauth2_session_invalid", status_code=302)

    except Exception as e:
        logger.error(f"Error in OAuth2 callback: {e}")
        return RedirectResponse(url="/login?error=oauth2_callback_error", status_code=302)


async def logout_handler(
    request: Request,
    session: Annotated[str | None, Cookie(alias=settings.session_cookie_name)] = None,
):
    """Shared logout logic for both GET and POST requests"""
    # Set audit action for logout
    set_audit_action(request, "logout", "auth", description="User logged out")

    try:
        # Resolve the server-side session record up front. We need the
        # provider for the IdP redirect, the id_token for id_token_hint, and
        # the session_id so we can delete the server record before clearing
        # the cookie (so a stolen cookie cannot be replayed).
        provider = None
        id_token = None
        session_id = None
        session_data: dict | None = None
        if session:
            from .dependencies import resolve_session_from_cookie

            session_data = await resolve_session_from_cookie(session)
            if session_data:
                if session_data.get("auth_method") == "oauth2":
                    provider = session_data.get("provider")
                    logger.info(f"User was authenticated via OAuth2 provider: {provider}")
                id_token = session_data.get("id_token")
                session_id = session_data.get("session_id")

        # Invalidate the server-side session before issuing the cookie clear.
        # If the resolve above failed (legacy cookie, expired, store outage),
        # there is no record to delete; the TTL cleans up either way.
        if session_id:
            from .session_store import delete_session

            try:
                await delete_session(session_id)
            except Exception as e:
                logger.warning(f"Best-effort session_store delete failed during logout: {e}")

        # Clear local session cookie. Must match (name, domain, path) of the
        # Set-Cookie used by auth_server to create the cookie, or the browser
        # will ignore the deletion. secure=False is intentional: an expired
        # empty cookie has no secrets, and secure=True on an HTTP request
        # would cause the browser to reject the Set-Cookie header entirely.
        response = RedirectResponse(url="/login", status_code=status.HTTP_303_SEE_OTHER)
        response.delete_cookie(
            settings.session_cookie_name,
            path="/",
            domain=settings.session_cookie_domain,
        )

        # If user was logged in via OAuth2, terminate the IdP session too.
        #
        # The id_token_hint is a full OIDC JWT (typically 1-3 KB). It must
        # never travel through a browser-facing URL (issue #1503): WAFs and
        # reverse proxies routinely block long or JWT-looking query strings
        # (returning 403 before the auth-server is reached, so logout fails
        # silently after the local session has already been cleared), and the
        # token would otherwise leak into browser history and access/proxy
        # logs while it is being actively revoked.
        #
        # Instead the registry calls the auth-server directly over the
        # internal network (settings.auth_server_url) and redirects the
        # browser straight to the IdP logout URL returned in the Location
        # header. The JWT only ever appears on the direct browser->IdP hop,
        # bypassing the deployment's public infrastructure entirely. This is
        # the same internal S2S URL already used by get_oauth2_providers(),
        # so it is configured and reachable in every deployment mode (EKS,
        # Docker/Compose, ECS, EC2, local).
        if provider:
            # Post-logout landing page: "/logout" renders the "Successfully Logged
            # Out" confirmation screen (frontend/src/pages/Logout.tsx), then
            # auto-redirects to /login after 5s. The IdP returns the browser here
            # after terminating the session, so this URL MUST be registered as a
            # valid post-logout / sign-out redirect in the IdP:
            #   - Cognito: add "<host>/logout" to the app client's Sign-out URLs
            #     (an unregistered logout_uri is rejected with "Required parameters
            #     missing" -- issue #1503).
            #   - Keycloak/Entra/Okta: add "<host>/logout" to the client's valid
            #     post-logout redirect URIs.
            # See docs/idp/*.md. If a deployment cannot register "/logout", point
            # this at a registered page (e.g. "/login") at the cost of the screen.
            redirect_uri = _build_external_url(request, "/logout")
            logout_params: dict[str, str] = {"redirect_uri": redirect_uri}

            # Include id_token_hint for proper SSO session termination if we
            # have a well-formed id_token from the server-side session record.
            #
            # We deliberately do NOT length-guard the id_token here. The logout
            # URL-length concern (Entra's AADSTS90015 "QueryStringTooLong") is
            # owned entirely by auth_server's oauth2_logout, which drops the hint
            # per-provider when the composed IdP URL exceeds MAX_LOGOUT_URL_LENGTH
            # (issue #1502 / PR #1508). That guard still runs on this S2S hop, so
            # adding a second registry-side size check would be a redundant,
            # divergent mechanism. The id_token also comes from our own AES-
            # encrypted session store, not user input, so _validate_jwt_format is
            # the only sanity check needed before forwarding.
            if id_token:
                if not _validate_jwt_format(id_token):
                    logger.debug("id_token failed JWT format validation, not forwarding")
                    logout_jwt_validation_failed.inc()
                else:
                    logout_params["id_token_hint"] = id_token
                    logger.debug("id_token extracted and forwarded, has_id_token=True")
                    logout_id_token_hint_present.inc()
            else:
                logger.debug("id_token not present in session, has_id_token=False")
                logout_id_token_hint_missing.inc()

            # Forward X-Forwarded-Host/Proto so auth-server's redirect_uri
            # same-origin validation (_is_redirect_within_cookie_domain) sees
            # the real public hostname and approves the post-logout URI. The
            # host is the same trusted, allowlist-validated value baked into
            # redirect_uri above, and the proto is derived from it so the two
            # can never disagree.
            s2s_url = f"{settings.auth_server_url}/oauth2/logout/{provider}"
            trusted_host = _resolve_trusted_host(request)
            x_forwarded_proto = "https" if redirect_uri.startswith("https") else "http"

            idp_redirect_url: str | None = None
            try:
                # Bare client (not the SSRF-guarded client) is correct here:
                # auth_server_url is a deployment-controlled internal service
                # (loopback/RFC-1918/cluster-internal) that the SSRF guard is
                # designed to block; it is not a user/registry-supplied URL.
                async with httpx.AsyncClient(follow_redirects=False) as client:
                    s2s_resp = await client.get(
                        s2s_url,
                        params=logout_params,
                        headers={
                            "X-Forwarded-Host": trusted_host,
                            "X-Forwarded-Proto": x_forwarded_proto,
                        },
                        timeout=_S2S_LOGOUT_TIMEOUT_SECONDS,
                    )
                if s2s_resp.status_code in (301, 302, 303, 307, 308):
                    idp_redirect_url = s2s_resp.headers.get("location")
                    # Observability only — this does NOT drop or truncate the
                    # URL (auth_server already applies the functional length
                    # guard, PR #1508). It just surfaces when the final IdP URL
                    # is unusually long so operators can correlate logout issues.
                    if idp_redirect_url and len(idp_redirect_url) > 2000:
                        logger.debug(
                            f"IdP logout URL length ({len(idp_redirect_url)}) "
                            "exceeds recommended limit (2000)"
                        )
                        logout_url_length_warning.inc()
                else:
                    logger.warning(
                        f"auth-server logout returned status {s2s_resp.status_code}; "
                        "falling back to local clear"
                    )
            except Exception as exc:
                logger.warning(f"S2S auth-server logout failed: {exc}; falling back to local clear")

            # Redirect the browser to the IdP logout URL when we have a safe
            # one, else fall back to /login so logout always completes locally
            # (the server-side session and cookie were already invalidated
            # above). Only http(s) and relative targets are followed to guard
            # against a javascript:/data: Location, even though the source is
            # the trusted internal auth-server.
            if idp_redirect_url and _is_safe_logout_redirect(idp_redirect_url):
                redirect_target = idp_redirect_url
            else:
                redirect_target = "/login"

            logger.debug(f"Redirecting browser after {provider} logout")
            response = RedirectResponse(url=redirect_target, status_code=status.HTTP_303_SEE_OTHER)
            response.delete_cookie(
                settings.session_cookie_name,
                path="/",
                domain=settings.session_cookie_domain,
            )

        logger.info("User logged out.")
        return response

    except Exception as e:
        logger.error(f"Error during logout: {e}")
        # Fallback to simple logout
        response = RedirectResponse(url="/login", status_code=status.HTTP_303_SEE_OTHER)
        response.delete_cookie(
            settings.session_cookie_name,
            path="/",
            domain=settings.session_cookie_domain,
        )
        return response


@router.get("/logout")
async def logout_get(
    request: Request,
    session: Annotated[str | None, Cookie(alias=settings.session_cookie_name)] = None,
):
    """Handle logout via GET request (for URL navigation)"""
    return await logout_handler(request, session)


@router.post("/logout")
async def logout_post(
    request: Request,
    session: Annotated[str | None, Cookie(alias=settings.session_cookie_name)] = None,
    _csrf: Annotated[None, Depends(verify_csrf_token_flexible)] = None,
):
    """Handle logout via POST request (for forms with CSRF validation)"""
    return await logout_handler(request, session)


@router.get("/providers")
async def get_providers_api():
    """API endpoint to get available OAuth2 providers for React frontend"""
    providers = await get_oauth2_providers()
    return {"providers": providers}


@router.get("/config")
async def get_auth_config():
    """API endpoint to get auth configuration for React frontend"""
    return {"auth_server_url": settings.auth_server_external_url}


@router.get("/csrf-token")
async def get_csrf_token(
    request: Request,
    session: Annotated[str | None, Cookie(alias=settings.session_cookie_name)] = None,
):
    """API endpoint to get a CSRF token for React/SPA applications.

    Returns a CSRF token bound to the current session that can be used
    in X-CSRF-Token headers for API requests.
    """
    from fastapi.responses import JSONResponse

    from .dependencies import resolve_session_from_cookie

    session_data = await resolve_session_from_cookie(session)
    if not session_data or not session_data.get("session_id"):
        return JSONResponse(
            status_code=status.HTTP_401_UNAUTHORIZED, content={"error": "No session found"}
        )

    csrf_token = generate_csrf_token(session_data["session_id"])
    return {"csrf_token": csrf_token}
