import asyncio
import logging
from typing import (
    Annotated,
)

from fastapi import (
    APIRouter,
    Depends,
    Form,
    HTTPException,
    Request,
    status,
)
from fastapi.responses import (
    JSONResponse,
)

from ..auth.dependencies import (
    nginx_proxied_auth,
)
from ..services.server_service import (
    server_service,
)
from .server_internal_routes import (
    internal_healthcheck,
)
from .server_routes_common import (
    _apply_remove_side_effects,
    _apply_toggle_side_effects,
    _build_server_entry_from_form,
    _enforce_proxy_pass_url_allowlist,
    _require_can_modify_servers,
)

logger = logging.getLogger(__name__)

router = APIRouter()


# ============================================================================
# NEW API: /api/servers/* endpoints with JWT Bearer Token Authentication
# ============================================================================
# These are the modern, JWT-authenticated equivalents of the /api/internal/*
# endpoints. They use Depends(nginx_proxied_auth) for authentication and
# support fine-grained permission checks via user context.
#
# Architecture:
# - Both /api/internal/* and /api/servers/* call the same internal functions
# - No code duplication; external API simply wraps existing endpoints
# - User context from JWT is passed through for audit logging
#
# Migration Path:
# Phase 1 (Now): Both endpoints work identically with same business logic
# Phase 2 (Future): Clients migrate to /api/servers/*
# Phase 3 (Future): /api/internal/* deprecated with sunset headers
# Phase 4 (Future): /api/internal/* removed in major version


@router.post("/servers/register")
async def register_service_api(
    request: Request,
    name: Annotated[str, Form()],
    description: Annotated[str, Form()],
    path: Annotated[str, Form()],
    proxy_pass_url: Annotated[str, Form()],
    user_context: Annotated[dict, Depends(nginx_proxied_auth)],
    tags: Annotated[str, Form()] = "",
    num_tools: Annotated[int, Form()] = 0,
    num_stars: Annotated[int, Form()] = 0,
    is_python: Annotated[bool, Form()] = False,
    license_str: Annotated[str, Form(alias="license")] = "N/A",
    overwrite: Annotated[bool, Form()] = True,
    auth_provider: Annotated[str | None, Form()] = None,
    auth_type: Annotated[str | None, Form()] = None,
    upstream_auth: Annotated[str | None, Form()] = None,
    supported_transports: Annotated[str | None, Form()] = None,
    headers: Annotated[str | None, Form()] = None,
    tool_list_json: Annotated[str | None, Form()] = None,
):
    """
    Register a service via JWT Bearer Token authentication (External API).

    This endpoint provides the same functionality as POST /api/internal/register
    but uses modern JWT Bearer token authentication via nginx headers, making it
    suitable for external service-to-service communication.

    **Authentication:** JWT Bearer token (via nginx X-User header)
    **Authorization:** Requires valid JWT token from auth system

    **Request body (form data):**
    - `name` (required): Service name
    - `description` (required): Service description
    - `path` (required): Service path (e.g., /myservice)
    - `proxy_pass_url` (required): Proxy URL (e.g., http://localhost:8000)
    - `tags` (optional): Comma-separated tags
    - `num_tools` (optional): Number of tools
    - `num_stars` (optional): Star rating
    - `is_python` (optional): Is Python server (boolean)
    - `license` (optional): License name
    - `overwrite` (optional): Overwrite if exists (boolean, default true)
    - `auth_provider` (optional): Auth provider name
    - `auth_type` (optional): Auth type (e.g., oauth, basic)
    - `supported_transports` (optional): JSON array of transports
    - `headers` (optional): JSON object of headers
    - `tool_list_json` (optional): JSON array of tool definitions

    **Response:**
    - `201 Created`: Service registered successfully
    - `400 Bad Request`: Invalid input data
    - `401 Unauthorized`: Missing or invalid JWT token
    - `409 Conflict`: Service already exists (unless overwrite=true)
    - `500 Internal Server Error`: Server error

    **Example:**
    ```bash
    curl -X POST https://registry.example.com/api/servers/register \\
      -H "Authorization: Bearer $JWT_TOKEN" \\
      -F "name=My Service" \\
      -F "description=My MCP Service" \\
      -F "path=/myservice" \\
      -F "proxy_pass_url=http://localhost:8000"
    ```
    """
    logger.info(
        "API register service request from user '%s' for service '%s'",
        user_context.get("username"),
        name,
    )
    _require_can_modify_servers(user_context)

    from ..search.service import faiss_service
    from ..health.service import health_service
    from ..core.nginx_service import nginx_service

    if not path.startswith("/"):
        path = "/" + path
    logger.warning("SERVERS REGISTER: Validated path: %s", path)

    _enforce_proxy_pass_url_allowlist(proxy_pass_url=proxy_pass_url)
    path, server_entry = _build_server_entry_from_form(
        name=name,
        description=description,
        path=path,
        proxy_pass_url=proxy_pass_url,
        tags=tags,
        num_tools=num_tools,
        num_stars=num_stars,
        is_python=is_python,
        license_str=license_str,
        auth_provider=auth_provider,
        auth_type=auth_type,
        upstream_auth=upstream_auth,
        supported_transports=supported_transports,
        headers=headers,
        tool_list_json=tool_list_json,
        logger=logger,
    )

    existing_server = server_service.get_server_info(path)
    if existing_server and not overwrite:
        logger.warning(
            "SERVERS REGISTER: Server exists and overwrite=False for path %s",
            path,
        )
        return JSONResponse(
            status_code=409,
            content={
                "error": "Service registration failed",
                "reason": f"A service with path '{path}' already exists",
                "detail": "Use overwrite=true to replace existing service",
            },
        )

    try:
        if existing_server and overwrite:
            logger.info(
                "Overwriting existing service at path %s for user %s",
                path,
                user_context.get("username"),
            )
            success = server_service.update_server(path, server_entry)
        else:
            success = server_service.register_server(server_entry)

        if not success:
            logger.warning("Failed to register service at path %s", path)
            return JSONResponse(
                status_code=400,
                content={
                    "error": "Service registration failed",
                    "reason": f"Failed to register service at path '{path}'",
                    "detail": "Check server logs for more information",
                },
            )

        logger.info(
            "Service registered successfully via API: %s by user %s",
            path,
            user_context.get("username"),
        )

        asyncio.create_task(health_service.perform_immediate_health_check(path))
        asyncio.create_task(faiss_service.save_data())

        return JSONResponse(
            status_code=201,
            content={
                "path": path,
                "name": name,
                "message": f"Service '{name}' registered successfully at path '{path}'",
            },
        )

    except Exception as e:
        logger.error("Service registration failed for %s: %s", path, e, exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Service registration failed: {str(e)}",
        ) from e


@router.post("/servers/toggle")
async def toggle_service_api(
    path: Annotated[str | None, Form()] = None,
    service_path: Annotated[str | None, Form()] = None,
    new_state: Annotated[bool | None, Form()] = None,
    user_context: Annotated[dict, Depends(nginx_proxied_auth)] = None,
):
    """
    Toggle a service's enabled/disabled state via JWT authentication (External API).

    This endpoint provides the same functionality as POST /api/internal/toggle
    but uses modern JWT Bearer token authentication.

    **Authentication:** JWT Bearer token (via nginx X-User header)
    **Authorization:** Requires valid JWT token from auth system

    **Request body (form data):**
    - `path` (preferred): Service path
    - `service_path` (legacy client compatibility): Service path
    - `new_state` (optional): If provided, set to desired state. If omitted, flips current state.

    **Response:**
    Returns the updated service status.

    **Example:**
    ```bash
    curl -X POST https://registry.example.com/api/servers/toggle \\
      -H "Authorization: Bearer $JWT_TOKEN" \\
      -F "path=/myservice" \\
      -F "new_state=true"
    ```
    """
    from ..search.service import faiss_service
    from ..health.service import health_service
    from ..core.nginx_service import nginx_service

    _require_can_modify_servers(user_context)

    raw_path = path or service_path
    if raw_path is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="path is required",
        )

    logger.info(
        "API toggle service request from user '%s' for path '%s' to %s",
        user_context.get("username"),
        raw_path,
        new_state,
    )

    if not raw_path.startswith("/"):
        raw_path = "/" + raw_path
    path = raw_path

    server_info = server_service.get_server_info(path)
    if not server_info:
        raise HTTPException(status_code=404, detail="Service path not registered")

    desired_state = new_state
    if desired_state is None:
        desired_state = not server_service.is_service_enabled(path)

    success = server_service.toggle_service(path, desired_state)
    if not success:
        raise HTTPException(status_code=500, detail="Failed to toggle service")

    logger.info(
        "Toggled '%s' (%s) to %s by user '%s'",
        server_info["server_name"],
        path,
        desired_state,
        user_context.get("username"),
    )

    status_str, last_checked_iso = await _apply_toggle_side_effects(
        service_path=path,
        server_info=server_info,
        new_state=desired_state,
        server_service=server_service,
        faiss_service=faiss_service,
        health_service=health_service,
        nginx_service=nginx_service,
        logger=logger,
    )

    return JSONResponse(
        status_code=200,
        content={
            "message": f"Toggle request for {path} processed.",
            "path": path,
            "is_enabled": desired_state,
            "service_path": path,
            "new_enabled_state": desired_state,
            "status": status_str,
            "last_checked_iso": last_checked_iso,
            "num_tools": server_info.get("num_tools", 0),
        },
    )


@router.post("/servers/remove")
async def remove_service_api(
    path: Annotated[str, Form()],
    user_context: Annotated[dict, Depends(nginx_proxied_auth)] = None,
):
    """
    Remove a service via JWT Bearer Token authentication (External API).

    This endpoint provides the same functionality as POST /api/internal/remove
    but uses modern JWT Bearer token authentication.

    **Authentication:** JWT Bearer token (via nginx X-User header)
    **Authorization:** Requires valid JWT token from auth system

    **Request body (form data):**
    - `path` (required): Service path to remove

    **Response:**
    Returns confirmation of removal.

    **Example:**
    ```bash
    curl -X POST https://registry.example.com/api/servers/remove \\
      -H "Authorization: Bearer $JWT_TOKEN" \\
      -F "path=/myservice"
    ```
    """
    from ..search.service import faiss_service
    from ..health.service import health_service
    from ..core.nginx_service import nginx_service
    from ..utils.scopes_manager import remove_server_scopes

    _require_can_modify_servers(user_context)

    logger.info(
        "API remove service request from user '%s' for path '%s'",
        user_context.get("username"),
        path,
    )

    if not path.startswith("/"):
        path = "/" + path

    server_info = server_service.get_server_info(path)
    if not server_info:
        logger.warning("Service not found at path '%s'", path)
        return JSONResponse(
            status_code=404,
            content={
                "error": "Service not found",
                "reason": f"No service registered at path '{path}'",
                "suggestion": "Check the service path and ensure it is registered",
            },
        )

    success = server_service.remove_server(path)
    if not success:
        logger.warning("Failed to remove service at path '%s'", path)
        return JSONResponse(
            status_code=500,
            content={
                "error": "Service removal failed",
                "reason": f"Failed to remove service at path '{path}'",
                "suggestion": "Check server logs for detailed error information",
            },
        )

    logger.info(
        "Service removed successfully: %s by user %s",
        path,
        user_context.get("username"),
    )

    await _apply_remove_side_effects(
        service_path=path,
        server_service=server_service,
        faiss_service=faiss_service,
        health_service=health_service,
        nginx_service=nginx_service,
        remove_server_scopes=remove_server_scopes,
        scopes_error_log_level="warning",
        logger=logger,
    )

    return JSONResponse(
        status_code=200,
        content={
            "message": "Service removed successfully",
            "path": path,
        },
    )


@router.get("/servers/health")
async def healthcheck_api(
    request: Request,
    user_context: Annotated[dict, Depends(nginx_proxied_auth)] = None,
):
    """
    Get health status for all registered services via JWT authentication (External API).

    This endpoint provides the same functionality as GET /api/internal/healthcheck
    but uses modern JWT Bearer token authentication.

    **Authentication:** JWT Bearer token (via nginx X-User header)
    **Authorization:** Requires valid JWT token from auth system

    **Response:**
    Returns health status for all services.

    **Example:**
    ```bash
    curl -X GET https://registry.example.com/api/servers/health \\
      -H "Authorization: Bearer $JWT_TOKEN"
    ```
    """
    logger.info(
        "API healthcheck request from user '%s'",
        user_context.get("username") if user_context else "unknown",
    )

    return await internal_healthcheck(
        request,
        user_context=user_context,
    )

