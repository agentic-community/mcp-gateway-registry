from __future__ import annotations

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
from fastapi.responses import (
    JSONResponse,
)

from ..auth.dependencies import (
    nginx_proxied_auth,
)
from ..services.server_service import (
    server_service,
)
from .server_routes_common import (
    _require_admin_user_context,
)
from .server_service_ops import (
    _register_service_internal,
    _remove_service_internal,
    _toggle_service_internal,
)

logger = logging.getLogger("registry.api.server_routes")

router = APIRouter()


@router.post("/internal/register")
async def internal_register_service(
    request: Request,
    name: Annotated[str, Form()],
    description: Annotated[str, Form()],
    path: Annotated[str, Form()],
    proxy_pass_url: Annotated[str, Form()],
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
    user_context: Annotated[dict, Depends(nginx_proxied_auth)] = None,
):
    """Internal service registration endpoint for mcpgw-server (requires admin auth)."""
    return await _register_service_internal(
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
        require_user_context=_require_admin_user_context,
        server_service_obj=server_service,
        logger=logger,
    )


@router.post("/internal/remove")
async def internal_remove_service(
    request: Request,
    service_path: Annotated[str, Form()],
    user_context: Annotated[dict, Depends(nginx_proxied_auth)] = None,
):
    """Internal service removal endpoint for mcpgw-server (requires admin auth)."""
    return await _remove_service_internal(
        service_path=service_path,
        user_context=user_context,
        require_user_context=_require_admin_user_context,
        server_service_obj=server_service,
        logger=logger,
    )


@router.post("/internal/toggle")
async def internal_toggle_service(
    request: Request,
    service_path: Annotated[str, Form()],
    user_context: Annotated[dict, Depends(nginx_proxied_auth)] = None,
):
    """Internal service toggle endpoint for mcpgw-server (requires admin auth)."""
    return await _toggle_service_internal(
        service_path=service_path,
        user_context=user_context,
        require_user_context=_require_admin_user_context,
        server_service_obj=server_service,
        logger=logger,
    )


@router.post("/internal/healthcheck")
async def internal_healthcheck(
    request: Request,
    user_context: Annotated[dict, Depends(nginx_proxied_auth)] = None,
):
    """Internal health check endpoint for mcpgw-server (requires admin auth)."""
    from ..health.service import health_service

    logger.debug("INTERNAL HEALTHCHECK: Function called - starting execution")

    user_context = _require_admin_user_context(user_context)
    username = user_context.get("username", "unknown")
    logger.debug("INTERNAL HEALTHCHECK: Admin authenticated successfully: %s", username)

    # Get health status for all servers
    try:
        health_data = health_service.get_all_health_status()
        logger.info(f"Retrieved health status for {len(health_data)} servers")

        return JSONResponse(
            status_code=200,
            content=health_data,
        )

    except Exception as e:
        logger.error(f"Failed to retrieve health status: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve health status: {str(e)}",
        )


@router.get("/internal/list")
async def internal_list_services(
    request: Request,
    user_context: Annotated[dict, Depends(nginx_proxied_auth)] = None,
):
    """Internal service listing endpoint for mcpgw-server (requires admin auth)."""
    logger.debug("INTERNAL LIST: Function called - starting execution")

    user_context = _require_admin_user_context(user_context)
    username = user_context.get("username", "unknown")

    logger.info(f"Internal service list request from admin user '{username}'")

    # Get all servers (admin access - no permission filtering)
    all_servers = server_service.get_all_servers()

    logger.debug("INTERNAL LIST: Found %s servers", len(all_servers))

    # Transform the data to include enabled status and health information
    from ..health.service import health_service

    services = []
    for service_path, server_info in all_servers.items():
        # Get real health status from health service
        health_data = health_service._get_service_health_data(service_path)

        service_data = {
            "server_name": server_info.get("server_name", "Unknown"),
            "path": service_path,
            "description": server_info.get("description", ""),
            "proxy_pass_url": server_info.get("proxy_pass_url", ""),
            "is_enabled": server_service.is_service_enabled(service_path),
            "tags": server_info.get("tags", []),
            "num_tools": server_info.get("num_tools", 0),
            "num_stars": server_info.get("num_stars", 0),
            "is_python": server_info.get("is_python", False),
            "license": server_info.get("license", "N/A"),
            "health_status": health_data["status"],
            "last_checked_iso": health_data["last_checked_iso"],
            "tool_list": server_info.get("tool_list", []),
        }
        services.append(service_data)

    logger.debug("INTERNAL LIST: Returning %s services", len(services))
    logger.info(
        f"Internal service list completed for admin user '{username}' - returned {len(services)} services"
    )

    return JSONResponse(
        status_code=200,
        content={
            "services": services,
            "total_count": len(services),
        },
    )
