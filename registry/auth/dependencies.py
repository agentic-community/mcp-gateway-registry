from __future__ import annotations

from typing import Annotated, Dict, Any
import logging

from fastapi import Depends, HTTPException, status, Cookie, Header, Request
from itsdangerous import URLSafeTimedSerializer

from ..core.config import settings
from .session import (
    create_session_cookie_value,
    get_current_user_from_cookie,
    get_user_session_data_from_cookie,
)
from .scopes import (
    SCOPES_CONFIG,
    get_accessible_agents_for_user,
    get_accessible_services_for_user,
    get_ui_permissions_for_user,
    get_user_accessible_servers,
    map_cognito_groups_to_scopes,
    user_can_access_server,
    user_can_modify_servers,
    user_has_ui_permission_for_service,
    user_has_wildcard_access,
)
from .user_context import (
    build_registry_user_context,
)

logger = logging.getLogger(__name__)

# Initialize session signer
signer = URLSafeTimedSerializer(settings.secret_key)

_SENSITIVE_HEADER_KEYS: set[str] = {
    "authorization",
    "proxy-authorization",
    "cookie",
    "set-cookie",
    "x-api-key",
    "api-key",
    "x-auth-token",
    "x-amz-security-token",
}


def _redact_headers_for_logging(
    headers: Dict[str, str],
) -> Dict[str, str]:
    redacted: Dict[str, str] = {}
    for key, value in headers.items():
        if key.lower() in _SENSITIVE_HEADER_KEYS:
            redacted[key] = "***REDACTED***"
        else:
            redacted[key] = value
    return redacted


def get_current_user(
    session: Annotated[str | None, Cookie(alias=settings.session_cookie_name)] = None,
) -> str:
    """
    Get the current authenticated user from session cookie.
    
    Returns:
        str: Username of the authenticated user
        
    Raises:
        HTTPException: If user is not authenticated
    """
    return get_current_user_from_cookie(
        signer=signer,
        session_cookie=session,
        max_age_seconds=settings.session_max_age_seconds,
    )


def get_user_session_data(
    session: Annotated[str | None, Cookie(alias=settings.session_cookie_name)] = None,
) -> Dict[str, Any]:
    """
    Get the full session data for the authenticated user.
    
    Returns:
        Dict containing username, groups, auth_method, provider, etc.
        
    Raises:
        HTTPException: If user is not authenticated
    """
    return get_user_session_data_from_cookie(
        signer=signer,
        session_cookie=session,
        max_age_seconds=settings.session_max_age_seconds,
        enforceai_db_path=getattr(settings, "enforceai_db_path", None),
        default_provider="local",
    )


def api_auth(
    session: Annotated[str | None, Cookie(alias=settings.session_cookie_name)] = None,
) -> str:
    """
    API authentication dependency that returns the username.
    Used for API endpoints that need authentication.
    """
    return get_current_user(session)


def web_auth(
    session: Annotated[str | None, Cookie(alias=settings.session_cookie_name)] = None,
) -> str:
    """
    Web authentication dependency that returns the username.
    Used for web pages that need authentication.
    """
    return get_current_user(session)


def enhanced_auth(
    session: Annotated[str | None, Cookie(alias=settings.session_cookie_name)] = None,
) -> Dict[str, Any]:
    """
    Enhanced authentication dependency that returns full user context.
    Returns username, groups, scopes, and permission flags.
    """
    session_data = get_user_session_data(session)
    
    username = session_data["username"]
    groups = session_data.get("groups", [])
    auth_method = session_data.get("auth_method", "password")
    legacy_auth_method = session_data.get("legacy_auth_method") or ""
    
    logger.info(
        f"Enhanced auth debug for {username}: groups={groups}, auth_method={auth_method}, legacy_auth_method={legacy_auth_method}"
    )
    
    is_oauth2_user = legacy_auth_method == "oauth2" or auth_method == "oidc"

    # Map groups to scopes for OAuth2 users
    if is_oauth2_user:
        scopes = map_cognito_groups_to_scopes(groups)
        logger.info(f"OAuth2 user {username} with groups {groups} mapped to scopes: {scopes}")
        # If OAuth2 user has no groups, they should get minimal permissions, not admin
        if not groups:
            logger.warning(f"OAuth2 user {username} has no groups! This user may not have proper group assignments in Cognito.")
    else:
        # Traditional users dynamically map to admin
        if not groups:
            groups = ['mcp-registry-admin']
        # Map traditional admin groups to scopes dynamically
        scopes = map_cognito_groups_to_scopes(groups)
        if not scopes:
            # Fallback for traditional users if no mapping exists
            scopes = ['mcp-registry-admin', 'mcp-servers-unrestricted/read', 'mcp-servers-unrestricted/execute']
        logger.info(f"Traditional user {username} with groups {groups} mapped to scopes: {scopes}")
    
    user_context = build_registry_user_context(
        username=username,
        groups=groups,
        scopes=scopes,
        auth_method=auth_method,
        provider=session_data.get("provider", "local"),
        extra={
            "user_id": session_data.get("user_id", f"local|{username}"),
            "session_id": session_data.get("session_id"),
            "email": session_data.get("email"),
            "legacy_auth_method": legacy_auth_method,
        },
    )

    logger.debug(f"Enhanced auth context for {username}: {user_context}")
    return user_context


def nginx_proxied_auth(
    request: Request,
    session: Annotated[str | None, Cookie(alias=settings.session_cookie_name)] = None,
    x_user: Annotated[str | None, Header(alias="X-User")] = None,
    x_username: Annotated[str | None, Header(alias="X-Username")] = None,
    x_scopes: Annotated[str | None, Header(alias="X-Scopes")] = None,
    x_auth_method: Annotated[str | None, Header(alias="X-Auth-Method")] = None,
) -> Dict[str, Any]:
    """
    Authentication dependency that works with both nginx-proxied requests and direct requests.

    For nginx-proxied requests: Reads user context from headers set by nginx after auth validation
    For direct requests: Falls back to session cookie authentication

    This allows Anthropic Registry API endpoints to work both when accessed through nginx (with JWT tokens)
    and when accessed directly (with session cookies).

    Returns:
        Dict containing username, groups, scopes, and permission flags
    """
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug("[NGINX_AUTH_DEBUG] Request path: %s", request.url.path)
        logger.debug("[NGINX_AUTH_DEBUG] Request method: %s", request.method)
        logger.debug("[NGINX_AUTH_DEBUG] X-User header present: %s", bool(x_user))
        logger.debug("[NGINX_AUTH_DEBUG] X-Username header present: %s", bool(x_username))
        logger.debug("[NGINX_AUTH_DEBUG] X-Scopes header present: %s", bool(x_scopes))
        logger.debug("[NGINX_AUTH_DEBUG] X-Auth-Method header: %s", x_auth_method)
        logger.debug("[NGINX_AUTH_DEBUG] Session cookie present: %s", session is not None)
        logger.debug(
            "[NGINX_AUTH_DEBUG] Request headers (redacted): %s",
            _redact_headers_for_logging(dict(request.headers)),
        )

    # First, try to get user context from nginx headers (JWT Bearer token flow)
    if x_user or x_username:
        username = x_username or x_user

        # Parse scopes from space-separated header
        scopes = x_scopes.split() if x_scopes else []

        # Map scopes to get groups based on auth method
        groups = []
        if x_auth_method in ['keycloak', 'entra', 'cognito']:
            # User authenticated via OAuth2 JWT (Keycloak, Entra ID, or Cognito)
            # Scopes already contain mapped permissions
            # Check if user has admin scopes
            if 'mcp-servers-unrestricted/read' in scopes and 'mcp-servers-unrestricted/execute' in scopes:
                groups = ['mcp-registry-admin']
            else:
                groups = ['mcp-registry-user']

        logger.info(f"nginx-proxied auth for user: {username}, method: {x_auth_method}, scopes: {scopes}")

        user_context = build_registry_user_context(
            username=username,
            groups=groups,
            scopes=scopes,
            auth_method=x_auth_method or "keycloak",
            provider=x_auth_method or "keycloak",
        )

        logger.debug(f"nginx-proxied auth context for {username}: {user_context}")
        return user_context

    # Fallback to session cookie authentication
    logger.debug("No nginx auth headers found, falling back to session cookie auth")
    return enhanced_auth(session)


def create_session_cookie(
    username: str,
    auth_method: str = "traditional",
    provider: str = "local",
) -> str:
    """Create a session cookie for a user."""
    return create_session_cookie_value(
        signer=signer,
        username=username,
        auth_method=auth_method,
        provider=provider,
        session_max_age_seconds=settings.session_max_age_seconds,
        enforceai_db_path=getattr(settings, "enforceai_db_path", None),
    )


def validate_login_credentials(username: str, password: str) -> bool:
    """Validate traditional login credentials."""
    return username == settings.admin_user and password == settings.admin_password


def ui_permission_required(permission: str, service_name: str = None):
    """
    Decorator to require a specific UI permission for a route.
    
    Args:
        permission: The UI permission required (e.g., 'register_service')
        service_name: Optional service name to check permission for. If None, checks if user has permission for any service.
    
    Returns:
        Dependency function that checks the permission
    """
    def check_permission(user_context: Dict[str, Any] = Depends(enhanced_auth)) -> Dict[str, Any]:
        ui_permissions = user_context.get('ui_permissions', {})
        
        if service_name:
            # Check permission for specific service
            if not user_has_ui_permission_for_service(permission, service_name, ui_permissions):
                logger.warning(f"User {user_context.get('username')} lacks UI permission '{permission}' for service '{service_name}'")
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=f"Insufficient permissions. Required: {permission} for {service_name}"
                )
        else:
            # Check if user has permission for any service
            if permission not in ui_permissions or not ui_permissions[permission]:
                logger.warning(f"User {user_context.get('username')} lacks UI permission: {permission}")
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=f"Insufficient permissions. Required: {permission}"
                )
        
        return user_context
    
    return check_permission 
