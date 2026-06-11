import logging
from typing import Annotated, Any

from fastapi import Cookie, Depends, Header, HTTPException, Request, status
from itsdangerous import BadSignature, SignatureExpired, URLSafeTimedSerializer

from ..core.config import settings
from .access_resolver import (
    get_user_accessible_tools,  # noqa: F401 - re-exported for external callers
    resolve_scope_access,
)
from .session_store import resolve_session as _store_resolve_session

logger = logging.getLogger(__name__)

# Initialize session signer
signer = URLSafeTimedSerializer(settings.secret_key)


async def resolve_session_from_cookie(
    cookie_value: str | None,
) -> dict[str, Any] | None:
    """Resolve a session cookie to its server-side record.

    Returns None for any non-recoverable failure: missing cookie, bad
    signature, expired signature, legacy dict-payload (pre-server-side
    rollout — forces re-login), or store unavailable. Callers must translate
    None into 401, never into 500.
    """
    if not cookie_value:
        return None

    try:
        payload = signer.loads(cookie_value, max_age=settings.session_max_age_seconds)
    except SignatureExpired:
        logger.debug("Session cookie has expired")
        return None
    except BadSignature:
        logger.debug("Session cookie has invalid signature")
        return None
    except Exception as e:
        logger.warning(f"Unexpected error decoding session cookie: {e}")
        return None

    # Legacy cookies signed before this rollout carried a dict payload. With
    # the server-side store, the cookie holds only an opaque session_id
    # string. Anything else is a stale cookie and the user must re-login.
    if not isinstance(payload, str):
        logger.debug("Legacy session cookie format detected; forcing re-login")
        return None

    return await _store_resolve_session(payload)


async def get_current_user(
    session: Annotated[str | None, Cookie(alias=settings.session_cookie_name)] = None,
) -> str:
    """Get the username for the authenticated user.

    Raises 401 for missing, tampered, expired, or legacy-format cookies.
    """
    data = await resolve_session_from_cookie(session)
    if data is None or not data.get("username"):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Authentication required"
        )
    return data["username"]


async def get_user_session_data(
    session: Annotated[str | None, Cookie(alias=settings.session_cookie_name)] = None,
) -> dict[str, Any]:
    """Get the full session record for the authenticated user.

    The record contains username, email, name, groups, provider, auth_method,
    and id_token (always, when the IdP returned one). Loaded from the
    server-side session store via the opaque session_id in the signed cookie.
    """
    data = await resolve_session_from_cookie(session)
    if data is None or not data.get("username"):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Authentication required"
        )

    if data.get("auth_method") != "oauth2":
        logger.warning(
            f"Rejecting non-OAuth2 session for user {data.get('username')} "
            f"(auth_method={data.get('auth_method')}). Please re-login via OAuth2."
        )
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Session expired. Please login again via OAuth2.",
        )

    return data


# Global scopes configuration - will be loaded during app startup
SCOPES_CONFIG = {}


async def reload_scopes_from_repository():
    """
    Async function to reload scopes from repository during app startup.
    Uses shared scopes loader from common module.
    """
    global SCOPES_CONFIG

    try:
        from ..common.scopes_loader import reload_scopes_config

        config = await reload_scopes_config()

        SCOPES_CONFIG.clear()
        SCOPES_CONFIG.update(config)

        group_mappings = config.get("group_mappings", {})
        ui_scopes = config.get("UI-Scopes", {})
        scope_defs = len([k for k in config.keys() if k not in ["group_mappings", "UI-Scopes"]])

        logger.info(
            f"Loaded scopes configuration: {len(group_mappings)} group mappings, "
            f"{scope_defs} scope definitions, {len(ui_scopes)} UI scopes"
        )

    except Exception as e:
        logger.error(f"Failed to reload scopes from repository: {e}", exc_info=True)


async def map_cognito_groups_to_scopes(groups: list[str]) -> list[str]:
    """
    Map Cognito groups to MCP scopes - queries repository directly.

    Args:
        groups: List of Cognito group names

    Returns:
        List of MCP scopes
    """
    from ..repositories.factory import get_scope_repository

    scope_repo = get_scope_repository()

    # Single read of all scope->groups mappings, then invert in memory.
    # Avoids one find() per group on this hot auth path.
    all_mappings = await scope_repo.get_all_group_mappings()
    group_to_scopes: dict[str, list[str]] = {}
    for scope_name, mapped_groups in all_mappings.items():
        for mapped_group in mapped_groups:
            group_to_scopes.setdefault(mapped_group, []).append(scope_name)

    # Collect scopes in group order, deduping while preserving order.
    seen: set[str] = set()
    unique_scopes: list[str] = []
    for group in groups:
        group_scopes = group_to_scopes.get(group, [])
        if group_scopes:
            logger.debug(f"Mapped group '{group}' to scopes: {group_scopes}")
        else:
            logger.debug(f"No scope mapping found for group: {group}")
        for scope in group_scopes:
            if scope not in seen:
                seen.add(scope)
                unique_scopes.append(scope)

    logger.info(f"Final mapped scopes: {unique_scopes}")
    return unique_scopes


async def get_ui_permissions_for_user(user_scopes: list[str]) -> dict[str, list[str]]:
    """
    Get UI permissions for a user based on their scopes - queries repository directly.

    Args:
        user_scopes: List of user's scopes (includes UI scope names like 'mcp-registry-admin')

    Returns:
        Dict mapping UI actions to lists of services they can perform the action on
        Example: {'list_service': ['mcpgw', 'auth_server'], 'toggle_service': ['mcpgw']}
    """
    from ..repositories.factory import get_scope_repository

    ui_permissions = {}
    scope_repo = get_scope_repository()

    for scope in user_scopes:
        scope_config = await scope_repo.get_ui_scopes(scope)
        if scope_config:
            logger.debug(f"Processing UI scope '{scope}' with config: {scope_config}")

            # Process each permission in the scope
            for permission, services in scope_config.items():
                if permission not in ui_permissions:
                    ui_permissions[permission] = set()

                # Handle "all" case
                if services == ["all"] or (isinstance(services, list) and "all" in services):
                    ui_permissions[permission].add("all")
                    logger.debug(f"UI permission '{permission}' granted for all services")
                else:
                    # Add specific services
                    if isinstance(services, list):
                        ui_permissions[permission].update(services)
                        logger.debug(
                            f"UI permission '{permission}' granted for services: {services}"
                        )

    # Convert sets back to lists
    result = {k: list(v) for k, v in ui_permissions.items()}
    logger.info(f"Final UI permissions for user: {result}")
    return result


def user_has_ui_permission_for_service(
    permission: str, service_name: str, user_ui_permissions: dict[str, list[str]]
) -> bool:
    """
    Check if user has a specific UI permission for a specific service.

    Args:
        permission: The UI permission to check (e.g., 'list_service', 'toggle_service')
        service_name: The service name to check permission for
        user_ui_permissions: User's UI permissions dict from get_ui_permissions_for_user()

    Returns:
        True if user has the permission for the service, False otherwise
    """
    if permission not in user_ui_permissions:
        return False

    allowed_services = user_ui_permissions[permission]

    # Check if user has permission for all services or the specific service
    has_permission = "all" in allowed_services or service_name in allowed_services

    logger.debug(
        f"Permission check: {permission} for {service_name} = {has_permission} (allowed: {allowed_services})"
    )
    return has_permission


def get_accessible_services_for_user(user_ui_permissions: dict[str, list[str]]) -> list[str]:
    """
    Get list of services the user can see based on their list_service permission.

    Args:
        user_ui_permissions: User's UI permissions dict from get_ui_permissions_for_user()

    Returns:
        List of service names the user can see, or ['all'] if they can see all services
    """
    list_permissions = user_ui_permissions.get("list_service", [])

    if "all" in list_permissions:
        return ["all"]

    return list_permissions


def get_accessible_agents_for_user(user_ui_permissions: dict[str, list[str]]) -> list[str]:
    """
    Get list of agents the user can see based on their list_agents permission.

    Args:
        user_ui_permissions: User's UI permissions dict from get_ui_permissions_for_user()

    Returns:
        List of agent paths the user can see, or ['all'] if they can see all agents
    """
    list_permissions = user_ui_permissions.get("list_agents", [])

    if "all" in list_permissions:
        return ["all"]

    return list_permissions


async def get_servers_for_scope(scope: str) -> list[str]:
    """
    Get list of server names that a scope provides access to - queries repository directly.

    Args:
        scope: The scope to check (e.g., 'mcp-servers-restricted/read')

    Returns:
        List of server names the scope grants access to
    """
    from ..repositories.factory import get_scope_repository

    scope_repo = get_scope_repository()
    scope_config = await scope_repo.get_server_scopes(scope)
    server_names = []

    for server_config in scope_config:
        if isinstance(server_config, dict) and "server" in server_config:
            server_names.append(server_config["server"])

    return list(set(server_names))  # Remove duplicates


async def user_has_wildcard_access(user_scopes: list[str]) -> bool:
    """
    Check if user has wildcard access to all servers via their scopes - queries repository directly.

    A user has wildcard access if any of their scopes includes server: '*'.
    This is determined dynamically from the scopes configuration, not hardcoded group names.

    Note: This function checks server access only. It is NOT used to determine
    admin status. See _user_is_admin() for admin determination.

    Args:
        user_scopes: List of user's scopes

    Returns:
        True if user has wildcard access to all servers, False otherwise
    """
    for scope in user_scopes:
        servers = await get_servers_for_scope(scope)
        if "*" in servers:
            logger.debug(f"User scope '{scope}' grants wildcard access to all servers")
            return True

    return False


# Prefixes for mutating (management) UI-Scopes actions.
# Any action starting with these prefixes is a management action.
# A user with any such action for "all" resources is considered an admin.
# Read-only prefixes (list_, get_, health_check_) are NOT included.
#
# SECURITY BOUNDARY: Changes to this tuple affect who is considered an admin.
# Reference: scripts/registry-admins.json for the complete admin permissions set.
_ADMIN_ACTION_PREFIXES: tuple[str, ...] = (
    "register_",
    "modify_",
    "toggle_",
    "delete_",
    "publish_",
    "create_",
)


def _user_is_admin(
    ui_permissions: dict[str, list[str]],
) -> bool:
    """Check if user has admin privileges based on UI-Scopes management actions.

    Admin status is determined by whether the user has any mutating
    (write/delete) UI-Scopes action with wildcard ("all") access.
    This decouples admin status from server: '*' wildcard access, allowing
    consumer roles to access all servers without gaining admin privileges.

    Mutating actions are identified by prefix: register_, modify_, toggle_,
    delete_, publish_, create_. Read-only actions (list_, get_, health_check_)
    do not grant admin status.

    See GitHub issue #663 for the motivation behind this design.

    Args:
        ui_permissions: Dict mapping UI actions to lists of allowed resources.
            Example: {'list_service': ['all'], 'register_service': ['all']}

    Returns:
        True if user has admin-level management permissions, False otherwise.
    """
    for action, resources in ui_permissions.items():
        if action.startswith(_ADMIN_ACTION_PREFIXES) and "all" in resources:
            logger.debug(f"Admin check: action '{action}' with 'all' grants admin status")
            return True

    return False


async def get_user_accessible_servers(user_scopes: list[str]) -> list[str]:
    """
    Get list of all servers the user has access to based on their scopes.

    Thin wrapper around `resolve_scope_access` to preserve the historical
    signature used by `user_can_access_server`, tests, and CLI tooling.
    The heavy lifting (scope repo walk, wildcard handling, fail-closed
    semantics) lives in `registry.auth.access_resolver`.

    Args:
        user_scopes: List of user's scopes

    Returns:
        List of server names the user can access
    """
    access = await resolve_scope_access(user_scopes)
    logger.debug(
        "User with scopes %s has access to servers: %s",
        user_scopes,
        access.servers,
    )
    return access.servers


def user_can_modify_servers(user_groups: list[str], user_scopes: list[str]) -> bool:
    """
    Check if user can modify servers (toggle, edit).

    Args:
        user_groups: List of user's groups
        user_scopes: List of user's scopes

    Returns:
        True if user can modify servers, False otherwise
    """
    # Admin users can always modify (check both groups and scopes)
    if "mcp-registry-admin" in user_groups or "mcp-registry-admin" in user_scopes:
        return True
    if "registry-admins" in user_groups or "registry-admins" in user_scopes:
        return True

    # Users with unrestricted execute access can modify
    if "mcp-servers-unrestricted/execute" in user_scopes:
        return True

    # mcp-registry-user group cannot modify servers (unless they're also admin)
    is_admin = "mcp-registry-admin" in user_groups or "mcp-registry-admin" in user_scopes
    if "mcp-registry-user" in user_groups and not is_admin:
        return False

    # For other cases, check if they have any execute permissions
    execute_scopes = [scope for scope in user_scopes if "/execute" in scope]
    return len(execute_scopes) > 0


async def user_can_access_server(server_name: str, user_scopes: list[str]) -> bool:
    """
    Check if user can access a specific server - queries repository directly.

    Args:
        server_name: Name of the server to check
        user_scopes: List of user's scopes

    Returns:
        True if user can access the server, False otherwise
    """
    accessible_servers = await get_user_accessible_servers(user_scopes)
    return server_name in accessible_servers


async def api_auth(
    session: Annotated[str | None, Cookie(alias=settings.session_cookie_name)] = None,
) -> str:
    """API authentication dependency that returns the username."""
    return await get_current_user(session)


async def web_auth(
    session: Annotated[str | None, Cookie(alias=settings.session_cookie_name)] = None,
) -> str:
    """Web authentication dependency that returns the username."""
    return await get_current_user(session)


async def _derive_user_context(
    *,
    username: str,
    groups: list[str],
    scopes: list[str],
    auth_method: str,
    provider: str,
    session_id: str | None = None,
    client_id: str = "",
) -> dict[str, Any]:
    """Build the canonical user_context dict from authenticated identity inputs.

    Single source of truth for scope→ui_permissions→is_admin derivation. Both
    `enhanced_auth` (cookie path) and `nginx_proxied_auth` (header path) call
    this so a given (groups, scopes, auth_method) tuple produces the same
    authorization result regardless of how the user was authenticated.

    The two callers differ only in how they obtain `scopes`: the cookie path
    derives scopes from `groups` via `map_cognito_groups_to_scopes`; the
    header path uses pre-computed scopes forwarded from the auth_server in
    `X-Scopes`. Both should equal each other for the same user — see #933.

    `auth_method == "federation-static"` short-circuits to a no-access
    context (federation static tokens get scoped access via separate routing,
    not registry-side derivation).
    """
    if auth_method == "federation-static":
        return {
            "username": username,
            "client_id": client_id,
            "groups": groups,
            "scopes": scopes,
            "auth_method": auth_method,
            "provider": provider,
            "session_id": session_id,
            "accessible_servers": [],
            "accessible_tools": {},
            "accessible_services": [],
            "accessible_agents": [],
            "ui_permissions": {},
            "can_modify_servers": False,
            "is_admin": False,
        }

    # Resolve servers + tools in a single scope-repo pass (Issue #1026).
    access = await resolve_scope_access(scopes)
    ui_permissions = await get_ui_permissions_for_user(scopes)

    return {
        "username": username,
        "client_id": client_id,
        "groups": groups,
        "scopes": scopes,
        "auth_method": auth_method,
        "provider": provider,
        # session_id is the opaque server-side identifier — used by template
        # callers to mint CSRF tokens via generate_csrf_token(session_id).
        "session_id": session_id,
        "accessible_servers": access.servers,
        "accessible_tools": access.tools,
        "accessible_services": get_accessible_services_for_user(ui_permissions),
        "accessible_agents": get_accessible_agents_for_user(ui_permissions),
        "ui_permissions": ui_permissions,
        "can_modify_servers": user_can_modify_servers(groups, scopes),
        "is_admin": _user_is_admin(ui_permissions),
    }


async def enhanced_auth(
    request: Request,
    session: Annotated[str | None, Cookie(alias=settings.session_cookie_name)] = None,
) -> dict[str, Any]:
    """
    Enhanced authentication dependency that returns full user context.
    Returns username, groups, scopes, and permission flags.
    Also sets request.state.user_context for audit logging middleware.
    """
    session_data = await get_user_session_data(session)

    username = session_data["username"]
    groups = session_data.get("groups", [])
    auth_method = session_data.get("auth_method", "oauth2")

    logger.info(f"Enhanced auth debug for {username}: groups={groups}, auth_method={auth_method}")

    # Map groups to scopes via OAuth2 group-to-scope mapping
    scopes = await map_cognito_groups_to_scopes(groups)
    logger.info(f"OAuth2 user {username} with groups {groups} mapped to scopes: {scopes}")
    if not groups:
        logger.warning(
            f"OAuth2 user {username} has no groups! This user may not have proper group assignments."
        )

    user_context = await _derive_user_context(
        username=username,
        groups=groups,
        scopes=scopes,
        auth_method=auth_method,
        provider=session_data.get("provider", "local"),
        session_id=session_data.get("session_id"),
    )

    # Set user context on request state for audit logging middleware
    request.state.user_context = user_context

    logger.debug(f"Enhanced auth context for {username}: {user_context}")
    return user_context


async def nginx_proxied_auth(
    request: Request,
    session: Annotated[
        str | None, Cookie(alias=settings.session_cookie_name, include_in_schema=False)
    ] = None,
    x_user: Annotated[str | None, Header(alias="X-User", include_in_schema=False)] = None,
    x_username: Annotated[str | None, Header(alias="X-Username", include_in_schema=False)] = None,
    x_scopes: Annotated[str | None, Header(alias="X-Scopes", include_in_schema=False)] = None,
    x_auth_method: Annotated[
        str | None, Header(alias="X-Auth-Method", include_in_schema=False)
    ] = None,
    x_client_id: Annotated[str | None, Header(alias="X-Client-Id", include_in_schema=False)] = None,
    x_groups: Annotated[str | None, Header(alias="X-Groups", include_in_schema=False)] = None,
) -> dict[str, Any]:
    """
    Authentication dependency that works with both nginx-proxied requests and direct requests.

    For nginx-proxied requests: Reads user context from headers set by nginx after auth validation
    For direct requests: Falls back to session cookie authentication

    This allows Anthropic Registry API endpoints to work both when accessed through nginx (with JWT tokens)
    and when accessed directly (with session cookies).

    Returns:
        Dict containing username, groups, scopes, and permission flags
    """
    # CRITICAL DIAGNOSTIC: Log ALL incoming headers and auth parameters
    logger.debug(f"[NGINX_AUTH_DEBUG] Request path: {request.url.path}")
    logger.debug(f"[NGINX_AUTH_DEBUG] Request method: {request.method}")
    logger.debug(f"[NGINX_AUTH_DEBUG] X-User header: '{x_user}' (type: {type(x_user).__name__})")
    logger.debug(
        f"[NGINX_AUTH_DEBUG] X-Username header: '{x_username}' (type: {type(x_username).__name__})"
    )
    logger.debug(
        f"[NGINX_AUTH_DEBUG] X-Scopes header: '{x_scopes}' (type: {type(x_scopes).__name__})"
    )
    logger.debug(
        f"[NGINX_AUTH_DEBUG] X-Auth-Method header: '{x_auth_method}' (type: {type(x_auth_method).__name__})"
    )
    logger.debug(f"[NGINX_AUTH_DEBUG] Session cookie present: {session is not None}")
    logger.debug(
        f"[NGINX_AUTH_DEBUG] Authorization header: {request.headers.get('authorization', 'NOT PRESENT')[:50] if request.headers.get('authorization') else 'NOT PRESENT'}"
    )

    # Log ALL headers for complete diagnostic, with sensitive values redacted.
    # cookie/authorization carry the session and bearer token; even at DEBUG we
    # don't want them in logs.
    _redacted_header_names = {"cookie", "authorization"}
    all_headers = {
        name: ("[REDACTED]" if name.lower() in _redacted_header_names else value)
        for name, value in request.headers.items()
    }
    logger.debug(f"[NGINX_AUTH_DEBUG] ALL REQUEST HEADERS: {all_headers}")

    # First, try to get user context from nginx headers (JWT Bearer token flow)
    if x_user or x_username:
        username = x_username or x_user

        # Parse scopes from space-separated header
        scopes = x_scopes.split() if x_scopes else []

        # Parse groups from X-Groups header (set by auth server from JWT claims).
        # If X-Groups is missing, groups stay empty: admin status is then
        # derived purely from scope-based ui_permissions (which the auth_server
        # already computed when it set X-Scopes). Synthesizing groups from a
        # scope heuristic — the previous behavior — granted admin to non-admins
        # via the proxied path while the cookie path correctly said non-admin
        # for the same user (#933).
        groups = x_groups.split() if x_groups else []

        logger.info(
            f"nginx-proxied auth for user: {username}, method: {x_auth_method}, "
            f"groups: {groups}, scopes: {scopes}"
        )

        # Defaulting auth_method/provider when X-Auth-Method is absent is a
        # legacy fallback for misconfigured nginx (the auth_server should
        # always set this header). Warn so operators can fix the upstream
        # config rather than seeing every header-auth request silently
        # labeled "keycloak" in audit logs.
        if not x_auth_method:
            logger.warning(
                f"nginx-proxied auth for {username}: X-Auth-Method header missing; "
                "defaulting to 'keycloak' (audit logs may be inaccurate). "
                "Verify auth_server is setting X-Auth-Method on proxied requests."
            )

        user_context = await _derive_user_context(
            username=username,
            groups=groups,
            scopes=scopes,
            auth_method=x_auth_method or "keycloak",
            provider=x_auth_method or "keycloak",
            client_id=x_client_id or "",
        )

        # Set user context on request state for audit logging middleware
        request.state.user_context = user_context

        logger.debug(
            f"nginx-proxied auth context for {username} "
            f"(is_admin={user_context['is_admin']}): {user_context}"
        )
        return user_context

    # Fallback to session cookie authentication
    logger.info(
        "[NGINX_AUTH_FALLBACK] No nginx auth headers found, falling back to session cookie auth"
    )
    logger.info(
        f"[NGINX_AUTH_FALLBACK] Session cookie value: {session[:20] if session else 'None'}..."
    )
    logger.info(f"[NGINX_AUTH_FALLBACK] Request path: {request.url.path}")
    try:
        return await enhanced_auth(request, session)
    except HTTPException as e:
        logger.error(
            f"[NGINX_AUTH_FALLBACK] enhanced_auth raised HTTPException: status={e.status_code}, detail={e.detail}"
        )
        raise
    except Exception as e:
        logger.error(
            f"[NGINX_AUTH_FALLBACK] enhanced_auth raised unexpected exception: {type(e).__name__}: {str(e)}"
        )
        raise


def ui_permission_required(permission: str, service_name: str = None):
    """
    Decorator to require a specific UI permission for a route.

    Args:
        permission: The UI permission required (e.g., 'register_service')
        service_name: Optional service name to check permission for. If None, checks if user has permission for any service.

    Returns:
        Dependency function that checks the permission
    """

    def check_permission(user_context: dict[str, Any] = Depends(enhanced_auth)) -> dict[str, Any]:
        ui_permissions = user_context.get("ui_permissions", {})

        if service_name:
            # Check permission for specific service
            if not user_has_ui_permission_for_service(permission, service_name, ui_permissions):
                logger.warning(
                    f"User {user_context.get('username')} lacks UI permission '{permission}' for service '{service_name}'"
                )
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=f"Insufficient permissions. Required: {permission} for {service_name}",
                )
        else:
            # Check if user has permission for any service
            if permission not in ui_permissions or not ui_permissions[permission]:
                logger.warning(
                    f"User {user_context.get('username')} lacks UI permission: {permission}"
                )
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=f"Insufficient permissions. Required: {permission}",
                )

        return user_context

    return check_permission
