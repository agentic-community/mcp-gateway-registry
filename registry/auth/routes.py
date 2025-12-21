import urllib.parse
import logging
from typing import Annotated

from fastapi import APIRouter, Request, Form, HTTPException, status, Cookie
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse
from fastapi.templating import Jinja2Templates
import httpx

from ..core.config import settings
from .dependencies import (
    create_session_cookie,
    get_user_session_data,
    validate_login_credentials,
)
from gateway_csrf import (
    mint_csrf_token,
)
from gateway_session import (
    normalize_session_data,
)
from auth_server.enforceai.stores.sqlite.session_store import (
    SqliteSessionStore,
)
from auth_server.enforceai.db.data_layer import (
    EnforceAIDataLayer,
)
from auth_server.enforceai.tokens.ui_session import (
    mint_enforceai_ui_session_token,
)

logger = logging.getLogger(__name__)

router = APIRouter()

# Templates (will be injected via dependency later, but for now keep it simple)
templates = Jinja2Templates(directory=settings.templates_dir)


def _normalize_redirect_uri(
    request: Request,
    redirect_uri: str | None,
) -> str:
    if redirect_uri is None:
        return str(request.base_url).rstrip("/") + "/"

    candidate = redirect_uri.strip()
    if not candidate:
        return str(request.base_url).rstrip("/") + "/"

    if candidate.startswith("/"):
        base = str(request.base_url).rstrip("/")
        return base + candidate

    parsed = urllib.parse.urlparse(candidate)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid redirect_uri",
        )

    request_host = request.url.hostname
    if request_host and parsed.hostname != request_host:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="redirect_uri host mismatch",
        )

    return candidate


async def get_oauth2_providers():
    """Fetch available OAuth2 providers from auth server"""
    try:
        async with httpx.AsyncClient() as client:
            logger.info(f"Fetching OAuth2 providers from {settings.auth_server_url}/oauth2/providers")
            response = await client.get(f"{settings.auth_server_url}/oauth2/providers", timeout=5.0)
            logger.info(f"OAuth2 providers response: status={response.status_code}")
            if response.status_code == 200:
                data = response.json()
                providers = data.get("providers", [])
                logger.info(f"Successfully fetched {len(providers)} OAuth2 providers: {providers}")
                return providers
            else:
                logger.warning(f"Auth server returned non-200 status: {response.status_code}, body: {response.text}")
    except Exception as e:
        logger.warning(f"Failed to fetch OAuth2 providers from auth server: {e}", exc_info=True)
    return []


@router.get("/login", response_class=HTMLResponse)
async def login_form(request: Request, error: str | None = None):
    """Show login form with OAuth2 providers"""
    oauth_providers = await get_oauth2_providers()
    return templates.TemplateResponse(
        "login.html", 
        {
            "request": request, 
            "error": error,
            "oauth_providers": oauth_providers
        }
    )


@router.get("/auth/{provider}")
async def oauth2_login_redirect(
    provider: str,
    request: Request,
    redirect_uri: str | None = None,
):
    """Redirect to auth server for OAuth2 login"""
    try:
        registry_url = _normalize_redirect_uri(
            request,
            redirect_uri,
        )

        auth_external_url = settings.auth_server_external_url
        if not auth_external_url or "your-domain.com" in auth_external_url:
            auth_external_url = str(request.base_url).rstrip("/")
        auth_url = (
            f"{auth_external_url}/oauth2/login/{provider}"
            f"?redirect_uri={urllib.parse.quote(registry_url)}"
        )
        logger.info(f"request.base_url: {request.base_url}, registry_url: {registry_url}, auth_external_url: {auth_external_url}, auth_url: {auth_url}")
        logger.info(f"Redirecting to OAuth2 login for provider {provider}: {auth_url}")
        return RedirectResponse(url=auth_url, status_code=302)

    except Exception as e:
        logger.error(f"Error redirecting to OAuth2 login for {provider}: {e}")
        return RedirectResponse(url="/login?error=oauth2_redirect_failed", status_code=302)


@router.get("/login/{provider}")
async def oauth2_login_redirect_compat(
    provider: str,
    request: Request,
    redirect_uri: str | None = None,
):
    """Compatibility route for SPAs: /api/auth/login/{provider}."""
    return await oauth2_login_redirect(
        provider=provider,
        request=request,
        redirect_uri=redirect_uri,
    )


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
                url=f"/login?error={urllib.parse.quote(error_message)}", 
                status_code=302
            )
        
        # If we reach here, the auth server should have set the session cookie
        # Verify the session is valid by checking the cookie
        session_cookie = request.cookies.get(settings.session_cookie_name)
        if session_cookie:
            try:
                from .dependencies import signer
                # Validate session cookie
                session_data = signer.loads(session_cookie, max_age=settings.session_max_age_seconds)
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


@router.post("/login")
async def login_submit(
    request: Request,
):
    """Handle login form submission - supports both traditional and API calls"""
    content_type = (request.headers.get("content-type") or "").lower()

    username: str | None = None
    password: str | None = None

    if "application/json" in content_type:
        try:
            payload = await request.json()
        except Exception:
            payload = None

        if isinstance(payload, dict):
            username = payload.get("username")
            password = payload.get("password")
    else:
        form = await request.form()
        username = form.get("username")
        password = form.get("password")

    if not isinstance(username, str) or not username.strip():
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="username is required",
        )

    if not isinstance(password, str) or not password.strip():
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="password is required",
        )

    username = username.strip()

    logger.info(f"Login attempt for username: {username}")
    
    # Check if this is an API call (React) or traditional form submission
    accept_header = (request.headers.get("accept") or "").lower()
    is_api_call = "application/json" in accept_header
    
    if validate_login_credentials(username, password):
        session_data = create_session_cookie(username)
        
        if is_api_call:
            # API response for React
            response = JSONResponse(content={"success": True, "message": "Login successful"})
        else:
            # Traditional redirect response
            response = RedirectResponse(url="/", status_code=status.HTTP_303_SEE_OTHER)
        
        # Security Note: This implementation uses domain cookies for single-tenant deployments
        # where cross-subdomain authentication is required (e.g., auth.domain.com and registry.domain.com).
        # For multi-tenant SaaS deployments with tenant-based subdomains, do NOT use domain cookies
        # as they would allow cross-tenant session sharing. Consider alternative authentication methods
        # such as token-based auth or separate auth domains per tenant.
        cookie_params = {
            "key": settings.session_cookie_name,
            "value": session_data,
            "max_age": settings.session_max_age_seconds,
            "httponly": True,  # Prevents JavaScript access (XSS protection)
            "samesite": "lax",  # CSRF protection
            "secure": settings.session_cookie_secure,  # Only transmit over HTTPS when True
            "path": "/",  # Explicit path for clarity
        }

        # Add domain attribute if configured for cross-subdomain cookie sharing
        if settings.session_cookie_domain:
            cookie_params["domain"] = settings.session_cookie_domain

        response.set_cookie(**cookie_params)
        logger.info(f"User '{username}' logged in successfully.")
        return response
    else:
        logger.info(f"Login failed for user '{username}'.")
        
        if is_api_call:
            # API error response for React
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid username or password"
            )
        else:
            # Traditional redirect with error
            return RedirectResponse(
                url="/login?error=Invalid+username+or+password",
                status_code=status.HTTP_303_SEE_OTHER,
            )





async def logout_handler(
    request: Request,
    session: Annotated[str | None, Cookie(alias=settings.session_cookie_name)] = None
):
    """Shared logout logic for both GET and POST requests"""
    try:
        # Check if user was logged in via OAuth2
        provider = None
        if session:
            try:
                from itsdangerous import URLSafeTimedSerializer, SignatureExpired, BadSignature
                serializer = URLSafeTimedSerializer(settings.secret_key)
                session_data = serializer.loads(session, max_age=settings.session_max_age_seconds)
                
                legacy_auth_method = session_data.get("legacy_auth_method") or session_data.get(
                    "auth_method"
                )
                if legacy_auth_method == "oauth2":
                    provider = session_data.get('provider')
                    logger.info(f"User was authenticated via OAuth2 provider: {provider}")
                    
            except (SignatureExpired, BadSignature, Exception) as e:
                logger.debug(f"Could not decode session for logout: {e}")
        
        session_id = None
        if session:
            try:
                from .dependencies import signer
                session_payload = signer.loads(session, max_age=settings.session_max_age_seconds)
                normalized = normalize_session_data(
                    session_payload,
                    default_provider="local",
                    max_age_seconds=settings.session_max_age_seconds,
                )
                session_id = normalized.session_id
            except Exception:
                session_id = None

        enforceai_db_path = getattr(settings, "enforceai_db_path", None)
        if session_id and enforceai_db_path is not None:
            try:
                EnforceAIDataLayer(db_path=enforceai_db_path).initialize()
                store = SqliteSessionStore(db_path=enforceai_db_path)
                store.revoke_session(session_id=session_id, revoked_reason="logout")
            except Exception:
                logger.exception("Failed to revoke server-side session on logout")

        # Clear local session cookie
        response = RedirectResponse(url="/login", status_code=status.HTTP_303_SEE_OTHER)
        response.delete_cookie(settings.session_cookie_name)
        
        # If user was logged in via OAuth2, redirect to provider logout
        if provider:
            auth_external_url = settings.auth_server_external_url
            
            # Build redirect URI based on current host
            host = request.headers.get("host", "localhost:7860")
            scheme = "https" if request.headers.get("x-forwarded-proto") == "https" or request.url.scheme == "https" else "http"
            
            # Handle localhost specially to ensure correct port
            if "localhost" in host and ":" not in host:
                redirect_uri = f"{scheme}://localhost:7860/logout"
            else:
                redirect_uri = f"{scheme}://{host}/logout"
            
            logout_url = f"{auth_external_url}/oauth2/logout/{provider}?redirect_uri={redirect_uri}"
            logger.info(f"Redirecting to {provider} logout: {logout_url}")
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
    session: Annotated[str | None, Cookie(alias=settings.session_cookie_name)] = None
):
    """Handle logout via GET request (for URL navigation)"""
    return await logout_handler(request, session)


@router.post("/logout")
async def logout_post(
    request: Request,
    session: Annotated[str | None, Cookie(alias=settings.session_cookie_name)] = None
):
    """Handle logout via POST request (for forms)"""
    return await logout_handler(request, session)


# Backwards-compatible alias used by unit tests.
logout = logout_get


@router.get("/providers")
async def get_providers_api():
    """API endpoint to get available OAuth2 providers for React frontend"""
    providers = await get_oauth2_providers()
    return {"providers": providers}

@router.get("/config")
async def get_auth_config():
    """API endpoint to get auth configuration for React frontend"""
    return {"auth_server_url": settings.auth_server_external_url}


@router.get("/csrf")
async def get_csrf_token(
    session: Annotated[str | None, Cookie(alias=settings.session_cookie_name)] = None,
):
    """Return a CSRF token bound to the current session.

    This endpoint requires an authenticated session cookie.
    """
    data = get_user_session_data(session)
    session_id = data.get("session_id")
    if not isinstance(session_id, str) or not session_id.strip():
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
        )

    return {
        "csrf_token": mint_csrf_token(
            secret_key=settings.secret_key,
            session_id=session_id,
        ),
        "expires_in_seconds": settings.csrf_token_max_age_seconds,
    }


@router.post("/enforceai/token")
async def mint_enforceai_management_token(
    session: Annotated[str | None, Cookie(alias=settings.session_cookie_name)] = None,
):
    """Vend a short-lived EnforceAI management token for the current session.

    This endpoint is intended for the browser UI. It requires a valid session
    cookie and is CSRF-protected by the registry middleware.
    """
    data = get_user_session_data(session)

    session_id = data.get("session_id")
    user_id = data.get("user_id")
    groups = data.get("groups", [])

    if not isinstance(session_id, str) or not session_id.strip():
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
        )

    if not isinstance(user_id, str) or not user_id.strip():
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
        )

    token, expires_at = mint_enforceai_ui_session_token(
        secret_key=settings.secret_key,
        user_id=user_id,
        session_id=session_id,
        groups=groups if isinstance(groups, list) else [],
    )

    return {
        "access_token": token,
        "expires_at": expires_at.isoformat(),
    }
