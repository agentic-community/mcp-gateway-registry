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
    _require_can_modify_servers,
)
from .server_service_ops import (
    _register_service_external,
    _remove_service_external,
    _toggle_service_external,
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

    try:
        return await _register_service_external(
            name=name,
            description=description,
            path=path,
            proxy_pass_url=proxy_pass_url,
            tags=tags,
            num_tools=num_tools,
            num_stars=num_stars,
            is_python=is_python,
            license_str=license_str,
            overwrite=overwrite,
            auth_provider=auth_provider,
            auth_type=auth_type,
            upstream_auth=upstream_auth,
            supported_transports=supported_transports,
            headers=headers,
            tool_list_json=tool_list_json,
            user_context=user_context,
            require_user_context=_require_can_modify_servers,
            server_service_obj=server_service,
            create_task=asyncio.create_task,
            logger=logger,
        )
    except Exception as e:
        if isinstance(e, HTTPException):
            raise
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
    logger.info(
        "API toggle service request from user '%s' for path '%s' to %s",
        user_context.get("username"),
        path or service_path,
        new_state,
    )
    return await _toggle_service_external(
        path=path,
        service_path=service_path,
        new_state=new_state,
        user_context=user_context,
        require_user_context=_require_can_modify_servers,
        server_service_obj=server_service,
        logger=logger,
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
    logger.info(
        "API remove service request from user '%s' for path '%s'",
        user_context.get("username"),
        path,
    )
    return await _remove_service_external(
        path=path,
        user_context=user_context,
        require_user_context=_require_can_modify_servers,
        server_service_obj=server_service,
        logger=logger,
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
