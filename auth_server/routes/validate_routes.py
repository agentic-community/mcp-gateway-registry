from __future__ import annotations

import json
import logging
import os
from urllib.parse import (
    urlparse,
)

from fastapi import (
    APIRouter,
    HTTPException,
    Request,
)
from fastapi.responses import (
    JSONResponse,
)

try:
    from ..enforceai_runtime import (
        _load_enforceai_runtime,
        evaluate_tool_call,
        get_enforceai_settings,
        get_enforceai_stores,
        get_identity_resolver,
        get_upstream_oauth_token_client,
        load_scope_catalog,
        resolve_callable_tools_for_server,
    )
    from ..enforceai_support import (
        emit_enforceai_audit_event as _emit_enforceai_audit_event,
        resolve_enforceai_scopes_catalog_path as _resolve_enforceai_scopes_catalog_path,
    )
    from ..providers.factory import (
        get_auth_provider,
    )
    from ..session_config import (
        SESSION_COOKIE_NAME,
    )
    from ..validation_utils import (
        anonymize_ip,
        hash_username,
        map_groups_to_scopes,
        mask_headers,
        mask_sensitive_id,
        parse_server_and_tool_from_url,
        validate_server_tool_access,
        validate_session_cookie,
    )
except ImportError:  # pragma: no cover
    from enforceai_runtime import (  # type: ignore[no-redef]
        _load_enforceai_runtime,
        evaluate_tool_call,
        get_enforceai_settings,
        get_enforceai_stores,
        get_identity_resolver,
        get_upstream_oauth_token_client,
        load_scope_catalog,
        resolve_callable_tools_for_server,
    )
    from enforceai_support import (  # type: ignore[no-redef]
        emit_enforceai_audit_event as _emit_enforceai_audit_event,
        resolve_enforceai_scopes_catalog_path as _resolve_enforceai_scopes_catalog_path,
    )
    from providers.factory import (  # type: ignore[no-redef]
        get_auth_provider,
    )
    from session_config import (  # type: ignore[no-redef]
        SESSION_COOKIE_NAME,
    )
    from validation_utils import (  # type: ignore[no-redef]
        anonymize_ip,
        hash_username,
        map_groups_to_scopes,
        mask_headers,
        mask_sensitive_id,
        parse_server_and_tool_from_url,
        validate_server_tool_access,
        validate_session_cookie,
    )

logger = logging.getLogger(__name__)

router = APIRouter()

_SESSION_COOKIE_PREFIX: str = f"{SESSION_COOKIE_NAME}="


def _extract_cookie_value(
    request: Request,
    cookie_header: str,
    cookie_name: str,
) -> str | None:
    value = request.cookies.get(cookie_name)
    if value:
        return value

    for cookie in cookie_header.split(";"):
        stripped = cookie.strip()
        if stripped.startswith(f"{cookie_name}="):
            return stripped.split("=", 1)[1]
    return None


async def _validate_request_with_enforceai(
    request: Request,
    original_path: str,
    server_name_from_url: str | None,
    server_name: str | None,
    tool_name: str | None,
    request_payload: object,
) -> JSONResponse:
    runtime = _load_enforceai_runtime()
    dependency_unavailable_error = runtime.DependencyUnavailableError
    enforceai_error = runtime.EnforceAIError

    has_non_cookie_credentials = any(
        value and value.strip()
        for value in (
            request.headers.get("authorization"),
            request.headers.get("x-authorization"),
            request.headers.get("x-gateway-token"),
            request.headers.get("x-api-key"),
        )
    )

    allow_cookie_auth = original_path.startswith("/api/")
    cookie_header = request.headers.get("Cookie", "")

    if allow_cookie_auth and not has_non_cookie_credentials:
        if _SESSION_COOKIE_PREFIX not in cookie_header and SESSION_COOKIE_NAME not in request.cookies:
            raise HTTPException(
                status_code=401,
                detail="Authentication required",
                headers={"Connection": "close"},
            )

        cookie_value = _extract_cookie_value(
            request=request,
            cookie_header=cookie_header,
            cookie_name=SESSION_COOKIE_NAME,
        )

        if not cookie_value:
            raise HTTPException(
                status_code=401,
                detail="Authentication required",
                headers={"Connection": "close"},
            )

        try:
            validation_result = validate_session_cookie(cookie_value)
        except ValueError as exc:
            raise HTTPException(
                status_code=401,
                detail=str(exc),
                headers={"Connection": "close"},
            ) from exc

        response_data = {
            "valid": True,
            "username": validation_result.get("username") or "",
            "client_id": validation_result.get("client_id") or "",
            "scopes": validation_result.get("scopes", []),
            "method": validation_result.get("method") or "",
            "groups": validation_result.get("groups", []),
            "server_name": server_name_from_url,
            "tool_name": tool_name,
        }
        response = JSONResponse(
            content=response_data,
            status_code=200,
        )
        response.headers["X-User"] = validation_result.get("username") or ""
        response.headers["X-Username"] = validation_result.get("username") or ""
        response.headers["X-Client-Id"] = validation_result.get("client_id") or ""
        response.headers["X-Scopes"] = " ".join(validation_result.get("scopes", []))
        response.headers["X-Auth-Method"] = validation_result.get("method") or ""
        response.headers["X-Server-Name"] = server_name_from_url or ""
        response.headers["X-Tool-Name"] = tool_name or ""
        return response

    try:
        resolver = get_identity_resolver()
        catalog_path = _resolve_enforceai_scopes_catalog_path()
        if catalog_path is None:
            catalog = load_scope_catalog()
        else:
            catalog = load_scope_catalog(path=catalog_path)
        identity = await resolver.resolve_identity(headers=dict(request.headers))
    except dependency_unavailable_error as exc:
        raise HTTPException(
            status_code=503,
            detail=exc.public_message,
            headers={"Connection": "close"},
        ) from exc
    except enforceai_error as exc:
        raise HTTPException(
            status_code=exc.status_code,
            detail=exc.public_message,
            headers={"Connection": "close"},
        ) from exc
    except Exception as exc:  # noqa: BLE001
        logger.exception("Unexpected EnforceAI failure during identity resolution")
        raise HTTPException(
            status_code=503,
            detail="Enforcement dependency unavailable",
            headers={"Connection": "close"},
        ) from exc

    method = tool_name or "initialize"
    actual_tool_name = None
    if method == "tools/call" and isinstance(request_payload, dict):
        params = request_payload.get("params")
        if isinstance(params, dict):
            actual_tool_name = params.get("name")

    request_id = request.headers.get("X-Request-Id")
    if not request_id and isinstance(request_payload, dict) and "id" in request_payload:
        request_id_value = request_payload.get("id")
        if request_id_value is not None:
            request_id = str(request_id_value)

    allowed_tools = None
    if isinstance(identity.metadata, dict):
        allowed_tools = identity.metadata.get("agent_allowed_tools")

    allowed_tools_header_value = ""
    if server_name and method == "tools/list":
        tool_policy = resolve_callable_tools_for_server(
            identity=identity,
            catalog=catalog,
            server=server_name,
            allowed_tools=allowed_tools,
        )
        if tool_policy.all_tools:
            allowed_tools_header_value = "*"
        else:
            allowed_tools_header_value = json.dumps(sorted(tool_policy.tools))

    if server_name and method in {"tools/list", "tools/call"}:
        if method == "tools/list":
            _emit_enforceai_audit_event(
                action="tools/list",
                outcome="allow",
                user_id=identity.user_id,
                agent_id=identity.agent_id,
                request_id=request_id,
                details={
                    "provider": identity.provider,
                    "server": server_name,
                    "allowed_tools": allowed_tools_header_value,
                },
                get_stores=get_enforceai_stores,
            )

        if method == "tools/call":
            if not actual_tool_name:
                _emit_enforceai_audit_event(
                    action="tools/call",
                    outcome="deny",
                    user_id=identity.user_id,
                    agent_id=identity.agent_id,
                    request_id=request_id,
                    details={
                        "provider": identity.provider,
                        "server": server_name,
                        "reason": "missing_tool_name",
                    },
                    get_stores=get_enforceai_stores,
                )
                raise HTTPException(
                    status_code=403,
                    detail="Forbidden",
                    headers={"Connection": "close"},
                )

            decision = evaluate_tool_call(
                identity=identity,
                catalog=catalog,
                server=server_name,
                tool=actual_tool_name,
                allowed_tools=allowed_tools,
            )
            _emit_enforceai_audit_event(
                action="tools/call",
                outcome="allow" if decision.allowed else "deny",
                user_id=identity.user_id,
                agent_id=identity.agent_id,
                request_id=request_id,
                details={
                    "provider": identity.provider,
                    "server": server_name,
                    "tool": actual_tool_name,
                    "reason": decision.reason,
                    "matched_scope": decision.matched_scope,
                },
                get_stores=get_enforceai_stores,
            )
            if not decision.allowed:
                raise HTTPException(
                    status_code=403,
                    detail="Forbidden",
                    headers={"Connection": "close"},
                )

    from auth_server.enforceai.models.upstream_auth import (
        UpstreamAuthConfig,
        UpstreamAuthInjection,
    )
    from auth_server.enforceai.upstream.headers import (
        ENFORCEAI_ERROR_CODE_HEADER,
        ENFORCEAI_UPSTREAM_API_KEY_HEADER,
        ENFORCEAI_UPSTREAM_API_KEY_HEADER_NAME_HEADER,
        ENFORCEAI_UPSTREAM_AUTHORIZATION_HEADER,
        ENFORCEAI_UPSTREAM_MODE_HEADER,
        MCP_AUTH_TYPE_HEADER,
        MCP_CLAIMS_HEADER,
        MCP_PRINCIPAL_HEADER,
        MCP_PROVIDER_HEADER,
        MCP_SCOPES_HEADER,
    )
    from auth_server.enforceai.upstream.resolver import (
        UpstreamInjectionError,
        resolve_upstream_injection,
    )

    server_path = request.headers.get("X-EnforceAI-Server-Path")
    if not server_path and server_name:
        server_path = f"/{server_name}"

    upstream_type = (request.headers.get("X-EnforceAI-Upstream-Auth-Type") or "none").strip()
    upstream_binding = (
        request.headers.get("X-EnforceAI-Upstream-Credential-Binding") or "service"
    ).strip()
    upstream_provider = request.headers.get("X-EnforceAI-Upstream-Provider")
    default_upstream_mode = "none" if upstream_type == "none" else "gateway-managed"
    upstream_mode = (request.headers.get("X-EnforceAI-Upstream-Mode") or default_upstream_mode).strip()
    upstream_mode = upstream_mode or default_upstream_mode
    upstream_header_name = request.headers.get("X-EnforceAI-Upstream-Header-Name")
    upstream_scheme = request.headers.get("X-EnforceAI-Upstream-Scheme")

    injection = None
    if upstream_type in {"api-key", "jwt", "oauth2", "oidc", "provider-oauth"}:
        header_name = upstream_header_name
        scheme = upstream_scheme
        if not header_name:
            if upstream_type == "api-key":
                header_name = "X-API-Key"
                scheme = None
            else:
                header_name = "Authorization"
                scheme = scheme or "Bearer"
        injection = UpstreamAuthInjection(
            header_name=header_name,
            scheme=scheme,
        )

    upstream_auth = UpstreamAuthConfig(
        mode=upstream_mode,
        type=upstream_type,
        provider=upstream_provider,
        credential_binding=upstream_binding,
        injection=injection,
    )

    try:
        oauth_providers = None
        oauth_token_client = None
        oauth_refresh_skew_seconds = 0
        if upstream_auth.type in {"oauth2", "oidc", "provider-oauth"}:
            settings = get_enforceai_settings()
            oauth_providers = settings.upstream_oauth_providers
            oauth_token_client = get_upstream_oauth_token_client()
            oauth_refresh_skew_seconds = settings.upstream_oauth_refresh_skew_seconds

        allow_missing_upstream = False
        if tool_name == "tools/list" and upstream_auth.type in {
            "oauth2",
            "oidc",
            "provider-oauth",
        }:
            allow_missing_upstream = True

        injection_result = await resolve_upstream_injection(
            server_path=server_path,
            upstream_auth=upstream_auth,
            identity=identity,
            stores=get_enforceai_stores(),
            oauth_providers=oauth_providers,
            oauth_token_client=oauth_token_client,
            oauth_refresh_skew_seconds=oauth_refresh_skew_seconds,
            allow_missing_credential=allow_missing_upstream,
        )
    except UpstreamInjectionError as exc:
        raise HTTPException(
            status_code=exc.status_code,
            detail=exc.public_message,
            headers={
                "Connection": "close",
                ENFORCEAI_ERROR_CODE_HEADER: exc.error_code,
            },
        ) from exc

    response_data = {
        "valid": True,
        "username": identity.user_id,
        "client_id": "",
        "scopes": identity.scopes,
        "method": identity.provider,
        "groups": [],
        "server_name": server_name,
        "tool_name": tool_name,
    }

    response = JSONResponse(
        content=response_data,
        status_code=200,
    )
    response.headers["X-User"] = identity.user_id
    response.headers["X-Username"] = identity.user_id
    response.headers["X-Client-Id"] = ""
    response.headers["X-Scopes"] = " ".join(identity.scopes)
    response.headers["X-Auth-Method"] = identity.provider
    response.headers["X-Server-Name"] = server_name or ""
    response.headers["X-Tool-Name"] = tool_name or ""
    response.headers["X-Agent-Id"] = identity.agent_id
    response.headers["X-Allowed-Tools"] = allowed_tools_header_value
    response.headers[MCP_PRINCIPAL_HEADER] = injection_result.mcp_principal
    response.headers[MCP_AUTH_TYPE_HEADER] = injection_result.mcp_auth_type
    response.headers[MCP_SCOPES_HEADER] = injection_result.mcp_scopes
    response.headers[MCP_PROVIDER_HEADER] = injection_result.mcp_provider
    response.headers[MCP_CLAIMS_HEADER] = injection_result.mcp_claims
    response.headers[ENFORCEAI_UPSTREAM_MODE_HEADER] = injection_result.mode
    response.headers[ENFORCEAI_UPSTREAM_AUTHORIZATION_HEADER] = injection_result.upstream_authorization
    response.headers[ENFORCEAI_UPSTREAM_API_KEY_HEADER] = injection_result.upstream_api_key
    response.headers[ENFORCEAI_UPSTREAM_API_KEY_HEADER_NAME_HEADER] = injection_result.upstream_api_key_header
    return response


def _safe_validation_result_for_logging(
    validation_result: dict,
) -> dict:
    safe_result = {key: value for key, value in validation_result.items() if key != "username"}
    safe_result["username"] = hash_username(validation_result.get("username", ""))
    return safe_result


async def _validate_request_legacy(
    request: Request,
    authorization: str | None,
    cookie_header: str,
    user_pool_id: str | None,
    client_id: str | None,
    region: str,
    original_url: str | None,
    server_name_from_url: str | None,
    request_payload: object,
) -> JSONResponse:
    # Log request for debugging with anonymized IP
    client_ip = request.client.host if request.client else "unknown"
    logger.info("Validation request from %s", anonymize_ip(client_ip))
    logger.info("Request Method: %s", request.method)

    # Log masked HTTP headers for GDPR/SOX compliance
    all_headers = dict(request.headers)
    masked_headers = mask_headers(all_headers)
    logger.debug("HTTP Headers (masked): %s", json.dumps(masked_headers, indent=2))

    # Log specific headers for debugging with masked sensitive data
    logger.info(
        "Key Headers: Authorization=%s, Cookie=%s, User-Pool-Id=%s, Client-Id=%s, Region=%s, Original-URL=%s",
        bool(authorization),
        bool(cookie_header),
        mask_sensitive_id(user_pool_id) if user_pool_id else "None",
        mask_sensitive_id(client_id) if client_id else "None",
        region,
        original_url,
    )
    logger.info("Server Name from URL: %s", server_name_from_url)

    # Initialize validation result
    validation_result = None

    # FIRST: Check for session cookie if present
    if _SESSION_COOKIE_PREFIX in cookie_header or SESSION_COOKIE_NAME in request.cookies:
        logger.info("Session cookie detected, attempting session validation")
        cookie_value = _extract_cookie_value(
            request=request,
            cookie_header=cookie_header,
            cookie_name=SESSION_COOKIE_NAME,
        )

        if cookie_value:
            try:
                validation_result = validate_session_cookie(cookie_value)
                logger.info(
                    "Session cookie validation result: %s",
                    _safe_validation_result_for_logging(validation_result),
                )
                logger.info(
                    "Session cookie validation successful for user: %s",
                    hash_username(validation_result["username"]),
                )
            except ValueError as exc:
                logger.warning("Session cookie validation failed: %s", exc)
                # Fall through to JWT validation

    # SECOND: If no valid session cookie, check for JWT token
    if not validation_result:
        # Validate required headers for JWT
        if not authorization or not authorization.startswith("Bearer "):
            logger.warning("Missing or invalid Authorization header and no valid session cookie")
            raise HTTPException(
                status_code=401,
                detail=(
                    "Missing or invalid Authorization header. Expected: Bearer <token> "
                    "or valid session cookie"
                ),
                headers={"WWW-Authenticate": "Bearer", "Connection": "close"},
            )

        # Extract token
        access_token = authorization.split(" ")[1]

        # Get authentication provider based on AUTH_PROVIDER environment variable
        try:
            auth_provider = get_auth_provider()
            logger.info("Using authentication provider: %s", auth_provider.__class__.__name__)

            # Provider-specific validation
            if hasattr(auth_provider, "validate_token"):
                # For Keycloak, no additional headers needed
                validation_result = auth_provider.validate_token(access_token)
                logger.info(
                    "Token validation successful using %s",
                    auth_provider.__class__.__name__,
                )
            else:
                # Fallback to old validation for compatibility
                if not user_pool_id:
                    logger.warning("Missing X-User-Pool-Id header for Cognito validation")
                    raise HTTPException(
                        status_code=400,
                        detail="Missing X-User-Pool-Id header",
                        headers={"Connection": "close"},
                    )

                if not client_id:
                    logger.warning("Missing X-Client-Id header for Cognito validation")
                    raise HTTPException(
                        status_code=400,
                        detail="Missing X-Client-Id header",
                        headers={"Connection": "close"},
                    )

                # Use old validator for backward compatibility
                validator = getattr(request.app.state, "validator", None)
                if validator is None:
                    raise HTTPException(
                        status_code=500,
                        detail="Token validator not configured",
                        headers={"Connection": "close"},
                    )
                validation_result = validator.validate_token(
                    access_token=access_token,
                    user_pool_id=user_pool_id,
                    client_id=client_id,
                    region=region,
                )

        except Exception as exc:
            logger.error("Authentication provider error: %s", exc)
            raise HTTPException(
                status_code=500,
                detail=f"Authentication provider configuration error: {str(exc)}",
                headers={"Connection": "close"},
            ) from exc

    logger.info("Token validation successful using method: %s", validation_result["method"])

    # Parse server and tool information from original URL if available
    server_name = server_name_from_url  # Use the server_name we extracted earlier
    tool_name = None

    if original_url and request_payload:
        # We already extracted server_name above, now just get tool_name from URL parsing
        _, tool_name = parse_server_and_tool_from_url(original_url)
        logger.debug("Parsed from original URL: server='%s', tool='%s'", server_name, tool_name)

        # Try to extract tool name from request payload if not found in URL
        if server_name and not tool_name and request_payload:
            try:
                # Look for tool name in JSON-RPC 2.0 format and other MCP patterns
                if isinstance(request_payload, dict):
                    # JSON-RPC 2.0 format: method field contains the tool name
                    tool_name = request_payload.get("method")

                    # If not found in method, check other common patterns
                    if not tool_name:
                        tool_name = request_payload.get("tool") or request_payload.get("name")

                    # Check for nested tool reference in params
                    if not tool_name and "params" in request_payload:
                        params = request_payload["params"]
                        if isinstance(params, dict):
                            tool_name = (
                                params.get("name")
                                or params.get("tool")
                                or params.get("method")
                            )

                    logger.debug("Extracted tool name from JSON-RPC payload: '%s'", tool_name)
                else:
                    logger.warning("Payload is not a dictionary: %s", type(request_payload))
            except Exception as exc:
                logger.error("Error processing request payload for tool extraction: %s", exc)

    # Validate scope-based access if we have server/tool information
    # For providers that use groups (Keycloak, Entra ID, Cognito), map groups to scopes
    user_groups = validation_result.get("groups", [])
    auth_method = validation_result.get("method", "")
    if user_groups and auth_method in ["keycloak", "entra", "cognito"]:
        # Map IdP groups to scopes using the group mappings
        user_scopes = map_groups_to_scopes(user_groups)
        logger.info("Mapped %s groups %s to scopes: %s", auth_method, user_groups, user_scopes)
    else:
        user_scopes = validation_result.get("scopes", [])
    if server_name:
        # For ANY server access, enforce scope validation (fail closed principle)
        # This includes MCP initialization methods that may not have a specific tool

        method = tool_name if tool_name else "initialize"  # Default to initialize if no tool specified
        actual_tool_name = None

        # For tools/call, extract the actual tool name from params
        if method == "tools/call" and isinstance(request_payload, dict):
            params = request_payload.get("params", {})
            if isinstance(params, dict):
                actual_tool_name = params.get("name")
                logger.info("Extracted actual tool name for tools/call: '%s'", actual_tool_name)

        # Check if user has any scopes - if not, deny access (fail closed)
        if not user_scopes:
            logger.warning(
                "Access denied for user %s to %s.%s (tool: %s) - no scopes configured",
                hash_username(validation_result.get("username", "")),
                server_name,
                method,
                actual_tool_name,
            )
            raise HTTPException(
                status_code=403,
                detail=(
                    f"Access denied to {server_name}.{method} - user has no scopes configured"
                ),
                headers={"Connection": "close"},
            )

        if not validate_server_tool_access(server_name, method, actual_tool_name, user_scopes):
            logger.warning(
                "Access denied for user %s to %s.%s (tool: %s)",
                hash_username(validation_result.get("username", "")),
                server_name,
                method,
                actual_tool_name,
            )
            raise HTTPException(
                status_code=403,
                detail=f"Access denied to {server_name}.{method}",
                headers={"Connection": "close"},
            )
        logger.info(
            "Scope validation passed for %s.%s (tool: %s)",
            server_name,
            method,
            actual_tool_name,
        )
    else:
        logger.debug("No server information available, skipping scope validation")

    # Prepare JSON response data
    response_data = {
        "valid": True,
        "username": validation_result.get("username") or "",
        "client_id": validation_result.get("client_id") or "",
        "scopes": user_scopes,
        "method": validation_result.get("method") or "",
        "groups": validation_result.get("groups", []),
        "server_name": server_name,
        "tool_name": tool_name,
    }
    logger.debug(
        "Full validation result: %s",
        json.dumps(_safe_validation_result_for_logging(validation_result), indent=2),
    )
    logger.debug("Response data being sent: %s", json.dumps(response_data, indent=2))
    # Create JSON response with headers that nginx can use
    response = JSONResponse(content=response_data, status_code=200)

    # Set headers for nginx auth_request_set directives
    response.headers["X-User"] = validation_result.get("username") or ""
    response.headers["X-Username"] = validation_result.get("username") or ""
    response.headers["X-Client-Id"] = validation_result.get("client_id") or ""
    response.headers["X-Scopes"] = " ".join(user_scopes)
    response.headers["X-Auth-Method"] = validation_result.get("method") or ""
    response.headers["X-Server-Name"] = server_name or ""
    response.headers["X-Tool-Name"] = tool_name or ""

    return response


@router.get("/validate")
async def validate_request(request: Request):
    """
    Validate a request by extracting configuration from headers and validating the bearer token.

    Expected headers:
    - Authorization: Bearer <token>
    - X-User-Pool-Id: <user_pool_id>
    - X-Client-Id: <client_id>
    - X-Region: <region> (optional, defaults to us-east-1)
    - X-Original-URL: <original_url> (optional, for scope validation)

    Returns:
        HTTP 200 with user info headers if valid, HTTP 401/403 if invalid

    Raises:
        HTTPException: If the token is missing, invalid, or configuration is incomplete
    """

    try:
        enforceai_enabled = bool(os.environ.get("ENFORCEAI_DB_PATH"))
        # Extract headers
        # Check for X-Authorization first (custom header used by this gateway)
        # Only if X-Authorization is not present, check standard Authorization header
        authorization = request.headers.get("X-Authorization")
        if not authorization:
            authorization = request.headers.get("Authorization")
        cookie_header = request.headers.get("Cookie", "")
        user_pool_id = request.headers.get("X-User-Pool-Id")
        client_id = request.headers.get("X-Client-Id")
        region = request.headers.get("X-Region", "us-east-1")
        original_url = request.headers.get("X-Original-URL")
        original_path = ""
        if original_url:
            try:
                original_path = urlparse(original_url).path or ""
            except Exception:
                original_path = ""
        body = request.headers.get("X-Body")

        is_registry_api_request = original_path.startswith("/api/")

        # Extract server_name from original_url early for logging
        server_name_from_url = None
        if original_url:
            try:
                parsed_url = urlparse(original_url)
                path = parsed_url.path.strip("/")
                path_parts = path.split("/") if path else []
                server_name_from_url = path_parts[0] if path_parts else None
                logger.info(
                    "Extracted server_name '%s' from original_url: %s",
                    server_name_from_url,
                    original_url,
                )
            except Exception as exc:
                logger.warning(
                    "Failed to extract server_name from original_url %s: %s",
                    original_url,
                    exc,
                )

        if is_registry_api_request:
            server_name_from_url = None

        # Read request body
        request_payload = None
        try:
            if body:
                payload_text = body  # .decode('utf-8')
                logger.debug(
                    "Raw Request Payload (%s chars): %s...",
                    len(payload_text),
                    payload_text[:1000],
                )
                request_payload = json.loads(payload_text)
                logger.debug("JSON RPC Request Payload: %s", json.dumps(request_payload, indent=2))
            else:
                logger.debug("No request body provided, skipping payload parsing")
        except UnicodeDecodeError as exc:
            logger.warning("Could not decode body as UTF-8: %s", exc)
        except json.JSONDecodeError as exc:
            logger.warning("Could not parse JSON RPC payload: %s", exc)
        except Exception as exc:
            logger.error("Error reading request payload: %s: %s", type(exc).__name__, exc)

        server_name = server_name_from_url
        tool_name = None
        if request_payload and isinstance(request_payload, dict):
            tool_name = request_payload.get("method") or request_payload.get("tool") or request_payload.get("name")
            if not tool_name and "params" in request_payload and isinstance(
                request_payload.get("params"),
                dict,
            ):
                tool_name = (
                    request_payload["params"].get("method")
                    or request_payload["params"].get("tool")
                    or request_payload["params"].get("name")
                )

        if enforceai_enabled:
            return await _validate_request_with_enforceai(
                request=request,
                original_path=original_path,
                server_name_from_url=server_name_from_url,
                server_name=server_name,
                tool_name=tool_name,
                request_payload=request_payload,
            )
        return await _validate_request_legacy(
            request=request,
            authorization=authorization,
            cookie_header=cookie_header,
            user_pool_id=user_pool_id,
            client_id=client_id,
            region=region,
            original_url=original_url,
            server_name_from_url=server_name_from_url,
            request_payload=request_payload,
        )

    except ValueError as e:
        logger.warning(f"Token validation failed: {e}")
        raise HTTPException(
            status_code=401,
            detail=str(e),
            headers={"WWW-Authenticate": "Bearer", "Connection": "close"},
        )
    except HTTPException as e:
        # Preserve explicit auth/enforcement HTTP status codes
        if e.status_code in {401, 403, 409, 424, 503}:
            raise
        # For other HTTPExceptions, let them fall through to general handler
        logger.error(f"HTTP error during validation: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal validation error: {str(e)}",
            headers={"Connection": "close"},
        )
    except Exception as e:
        logger.error(f"Unexpected error during validation: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal validation error: {str(e)}",
            headers={"Connection": "close"}
        )
    finally:
        pass
