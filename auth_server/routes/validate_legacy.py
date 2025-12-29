from __future__ import annotations

import json
import logging

from fastapi import (
    HTTPException,
    Request,
)
from fastapi.responses import (
    JSONResponse,
)

try:
    from .validate_common import (
        SESSION_COOKIE_NAME,
        _SESSION_COOKIE_PREFIX,
        _extract_cookie_value,
    )
except ImportError:  # pragma: no cover
    from validate_common import (  # type: ignore[no-redef]
        SESSION_COOKIE_NAME,
        _SESSION_COOKIE_PREFIX,
        _extract_cookie_value,
    )

try:
    from ..providers.factory import (
        get_auth_provider,
    )
    from ..validation_utils import (
        hash_username,
        map_groups_to_scopes,
        mask_headers,
        mask_sensitive_id,
        parse_server_and_tool_from_url,
        validate_server_tool_access,
        validate_session_cookie,
        anonymize_ip,
    )
except ImportError:  # pragma: no cover
    from providers.factory import (  # type: ignore[no-redef]
        get_auth_provider,
    )
    from validation_utils import (  # type: ignore[no-redef]
        hash_username,
        map_groups_to_scopes,
        mask_headers,
        mask_sensitive_id,
        parse_server_and_tool_from_url,
        validate_server_tool_access,
        validate_session_cookie,
        anonymize_ip,
    )

logger = logging.getLogger(__name__)


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
                validation_result = auth_provider.validate_token(access_token)
                logger.info(
                    "Token validation successful using %s",
                    auth_provider.__class__.__name__,
                )
            else:
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

        except HTTPException:
            raise
        except Exception as exc:
            logger.error("Authentication provider error: %s", exc)
            raise HTTPException(
                status_code=500,
                detail=f"Authentication provider configuration error: {str(exc)}",
                headers={"Connection": "close"},
            ) from exc

    logger.info("Token validation successful using method: %s", validation_result["method"])

    server_name = server_name_from_url
    tool_name = None

    if original_url and request_payload:
        _, tool_name = parse_server_and_tool_from_url(original_url)
        logger.debug("Parsed from original URL: server='%s', tool='%s'", server_name, tool_name)

        if server_name and not tool_name and request_payload:
            try:
                if isinstance(request_payload, dict):
                    tool_name = request_payload.get("method")

                    if not tool_name:
                        tool_name = request_payload.get("tool") or request_payload.get("name")

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

    user_groups = validation_result.get("groups", [])
    auth_method = validation_result.get("method", "")
    if user_groups and auth_method in ["keycloak", "entra", "cognito"]:
        user_scopes = map_groups_to_scopes(user_groups)
        logger.info("Mapped %s groups %s to scopes: %s", auth_method, user_groups, user_scopes)
    else:
        user_scopes = validation_result.get("scopes", [])

    if server_name:
        method = tool_name if tool_name else "initialize"
        actual_tool_name = None

        if method == "tools/call" and isinstance(request_payload, dict):
            params = request_payload.get("params", {})
            if isinstance(params, dict):
                actual_tool_name = params.get("name")
                logger.info("Extracted actual tool name for tools/call: '%s'", actual_tool_name)

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
    response = JSONResponse(content=response_data, status_code=200)

    response.headers["X-User"] = validation_result.get("username") or ""
    response.headers["X-Username"] = validation_result.get("username") or ""
    response.headers["X-Client-Id"] = validation_result.get("client_id") or ""
    response.headers["X-Scopes"] = " ".join(user_scopes)
    response.headers["X-Auth-Method"] = validation_result.get("method") or ""
    response.headers["X-Server-Name"] = server_name or ""
    response.headers["X-Tool-Name"] = tool_name or ""

    return response

