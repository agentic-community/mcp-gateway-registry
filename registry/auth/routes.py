import logging
import re
import urllib.parse
from typing import Annotated

import httpx
from fastapi import APIRouter, Cookie, Depends, Request, status
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from prometheus_client import Counter

from ..audit.context import set_audit_action
from ..core.config import settings
from .csrf import generate_csrf_token, verify_csrf_token_flexible

logger = logging.getLogger(__name__)


# Prometheus metrics for logout observability
logout_id_token_hint_present = Counter(
    "registry_logout_id_token_hint_present_total",
    "Number of Registry logout requests where id_token was successfully extracted and forwarded",
)

logout_id_token_hint_missing = Counter(
    "registry_logout_id_token_hint_missing_total",
    "Number of Registry logout requests where id_token was missing from session",
)

logout_jwt_validation_failed = Counter(
    "registry_logout_jwt_validation_failed_total",
    "Number of Registry logout requests where id_token failed JWT format validation",
)

logout_url_length_warning = Counter(
    "registry_logout_url_length_warning_total",
    "Number of Registry logout requests where the logout URL exceeded recommended length",
)

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
        "login.html", {"request": request, "error": error, "oauth_providers": oauth_providers}
    )


@router.get("/auth/{provider}")
async def oauth2_login_redirect(provider: str, request: Request):
    """Redirect to auth server for OAuth2 login"""
    try:
        # Build redirect URL to auth server - use external URL for browser redirects
        # When behind CloudFront, request.base_url may have wrong scheme/host
        # Check CloudFront and X-Forwarded headers to build correct URL
        host = request.headers.get("host", "")
        cloudfront_proto = request.headers.get("x-cloudfront-forwarded-proto", "")
        x_forwarded_proto = request.headers.get("x-forwarded-proto", "")

        # Determine scheme - prefer CloudFront header, then X-Forwarded-Proto
        if cloudfront_proto.lower() == "https" or x_forwarded_proto.lower() == "https":
            scheme = "https"
        else:
            scheme = request.url.scheme

        # Build registry URL from headers (more reliable behind proxies)
        if host:
            registry_url = f"{scheme}://{host}"
        else:
            registry_url = str(request.base_url).rstrip("/")

        auth_external_url = settings.auth_server_external_url
        auth_url = f"{auth_external_url}/oauth2/login/{provider}?redirect_uri={registry_url}/"
        logger.info(
            f"request.base_url: {request.base_url}, registry_url: {registry_url}, auth_external_url: {auth_external_url}, auth_url: {auth_url}"
        )
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
                error_message = f"OAuth2 provider error: {details}"
            elif error == "oauth2_init_failed":
                error_message = "Failed to initiate OAuth2 login"
            elif error == "oauth2_callback_failed":
                error_message = "OAuth2 authentication failed"

            return RedirectResponse(
                url=f"/login?error={urllib.parse.quote(error_message)}", status_code=302
            )

        # If we reach here, the auth server should have set the session cookie
        # Verify the session is valid by checking the cookie
        session_cookie = request.cookies.get(settings.session_cookie_name)
        if session_cookie:
            try:
                from .dependencies import signer

                # Validate session cookie
                session_data = signer.loads(
                    session_cookie, max_age=settings.session_max_age_seconds
                )
                username = session_data.get("username")
                auth_method = session_data.get("auth_method", "unknown")

                logger.info(f"OAuth2 callback successful for user {username} via {auth_method}")
                return RedirectResponse(url="/", status_code=302)

            except Exception as e:
                logger.warning(f"Invalid session cookie in OAuth2 callback: {e}")

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
        # Check if user was logged in via OAuth2
        provider = None
        if session:
            try:
                from itsdangerous import BadSignature, SignatureExpired, URLSafeTimedSerializer

                serializer = URLSafeTimedSerializer(settings.secret_key)
                session_data = serializer.loads(session, max_age=settings.session_max_age_seconds)

                if session_data.get("auth_method") == "oauth2":
                    provider = session_data.get("provider")
                    logger.info(f"User was authenticated via OAuth2 provider: {provider}")

            except (SignatureExpired, BadSignature, Exception) as e:
                logger.debug(f"Could not decode session for logout: {e}")

        # Clear local session cookie
        response = RedirectResponse(url="/login", status_code=status.HTTP_303_SEE_OTHER)
        response.delete_cookie(settings.session_cookie_name)

        # If user was logged in via OAuth2, redirect to provider logout
        if provider:
            auth_external_url = settings.auth_server_external_url

            # Build redirect URI based on current host
            # Check CloudFront header first, then x-forwarded-proto, then request scheme
            host = request.headers.get("host", "localhost:7860")
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

            # Handle localhost specially to ensure correct port
            if "localhost" in host and ":" not in host:
                redirect_uri = f"{scheme}://localhost:7860/logout"
            else:
                redirect_uri = f"{scheme}://{host}/logout"

            logout_url = f"{auth_external_url}/oauth2/logout/{provider}?redirect_uri={redirect_uri}"

            # Extract id_token from session and append as id_token_hint if present
            # This enables proper SSO session termination for Keycloak and Entra ID
            if session:
                try:
                    from itsdangerous import URLSafeTimedSerializer

                    serializer = URLSafeTimedSerializer(settings.secret_key)
                    session_data = serializer.loads(
                        session, max_age=settings.session_max_age_seconds
                    )
                    id_token = session_data.get("id_token")

                    if id_token:
                        # Validate JWT format before forwarding
                        if not _validate_jwt_format(id_token):
                            logger.debug("id_token failed JWT format validation, not forwarding")
                            logout_jwt_validation_failed.inc()
                        else:
                            # URL encode the token
                            encoded_token = urllib.parse.quote(id_token, safe="")

                            # Append id_token_hint to logout URL
                            logout_url = f"{logout_url}&id_token_hint={encoded_token}"

                            # Validate URL length (warn if > 2000 chars)
                            if len(logout_url) > 2000:
                                logger.debug(
                                    f"Logout URL length ({len(logout_url)}) exceeds recommended limit (2000)"
                                )
                                logout_url_length_warning.inc()

                            logger.debug("id_token extracted and forwarded, has_id_token=True")
                            logout_id_token_hint_present.inc()
                    else:
                        logger.debug("id_token not found in session, has_id_token=False")
                        logout_id_token_hint_missing.inc()

                except Exception as e:
                    logger.debug(f"Could not extract id_token from session: {e}")
                    logout_id_token_hint_missing.inc()

            logger.debug(f"Redirecting to {provider} logout")
            response = RedirectResponse(url=logout_url, status_code=status.HTTP_303_SEE_OTHER)
            response.delete_cookie(settings.session_cookie_name)

        logger.info("User logged out.")
        return response

    except Exception as e:
        logger.error(f"Error during logout: {e}")
        # Fallback to simple logout
        response = RedirectResponse(url="/login", status_code=status.HTTP_303_SEE_OTHER)
        response.delete_cookie(settings.session_cookie_name)
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
    if not session:
        from fastapi.responses import JSONResponse

        return JSONResponse(
            status_code=status.HTTP_401_UNAUTHORIZED, content={"error": "No session found"}
        )

    csrf_token = generate_csrf_token(session)
    return {"csrf_token": csrf_token}
