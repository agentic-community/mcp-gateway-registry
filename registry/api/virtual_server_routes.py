"""
API routes for virtual MCP server management.

Provides CRUD endpoints for virtual servers that aggregate tools
from multiple backend MCP servers, plus a global tool catalog endpoint.
"""

import logging
import re
from typing import Annotated

from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    Path,
    Query,
    Request,
    status,
)
from pydantic import BaseModel

from ..audit.context import set_audit_action
from ..auth.dependencies import nginx_proxied_auth
from ..exceptions import (
    VirtualServerAlreadyExistsError,
    VirtualServerNotFoundError,
    VirtualServerServiceError,
    VirtualServerValidationError,
)
from ..schemas.virtual_server_models import (
    CreateVirtualServerRequest,
    ToggleVirtualServerRequest,
    UpdateVirtualServerRequest,
    VirtualServerConfig,
)
from ..services.tool_catalog_service import get_tool_catalog_service
from ..services.virtual_server_service import get_virtual_server_service

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s,p%(process)s,{%(filename)s:%(lineno)d},%(levelname)s,%(message)s",
)
logger = logging.getLogger(__name__)


router = APIRouter()


def _require_admin(
    user_context: dict,
) -> None:
    """Check that user has admin or server-modify permissions.

    Args:
        user_context: Authenticated user context

    Raises:
        HTTPException: 403 if user lacks permissions
    """
    is_admin = user_context.get("is_admin", False)
    can_modify = user_context.get("can_modify_servers", False)

    # Also check groups and scopes for mcp-registry-admin
    groups = user_context.get("groups", [])
    scopes = user_context.get("scopes", [])
    has_admin_group = "mcp-registry-admin" in groups
    has_admin_scope = "mcp-registry-admin" in scopes

    if not (is_admin or can_modify or has_admin_group or has_admin_scope):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin or server-modify permissions required",
        )


_VALID_VIRTUAL_PATH_RE = re.compile(r"^/virtual/[a-z0-9]+(-[a-z0-9]+)*$")


def _normalize_virtual_path(
    raw_path: str,
) -> str:
    """Normalize and validate a virtual server path from URL path parameter.

    Rejects paths containing '..', special characters, or anything that
    doesn't match the expected /virtual/<slug> format.

    Args:
        raw_path: Raw path from URL (may not have /virtual/ prefix)

    Returns:
        Normalized path with /virtual/ prefix

    Raises:
        HTTPException: 400 if path contains invalid characters or format
    """
    if ".." in raw_path:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid virtual server path: path traversal not allowed",
        )

    if raw_path.startswith("/virtual/"):
        normalized = raw_path
    elif raw_path.startswith("virtual/"):
        normalized = f"/{raw_path}"
    else:
        normalized = f"/virtual/{raw_path}"

    if not _VALID_VIRTUAL_PATH_RE.match(normalized):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                "Invalid virtual server path: must match /virtual/<slug> "
                "where slug is lowercase alphanumeric with hyphens"
            ),
        )

    return normalized


# --- Virtual Server CRUD Endpoints ---
# NOTE: Route order matters! Sub-resource routes (tools, toggle) must be
# declared before the catch-all {vs_path:path} GET route to avoid the
# :path parameter consuming "tools" or "toggle" as part of the path.


@router.get(
    "/virtual-servers",
    response_model=dict,
    summary="List all virtual servers",
)
async def list_virtual_servers(
    user_context: Annotated[dict, Depends(nginx_proxied_auth)],
) -> dict:
    """List all virtual servers with summary information."""
    service = get_virtual_server_service()
    all_servers = await service.list_virtual_servers()

    # Filter based on list_virtual_server permission
    ui_permissions = user_context.get("ui_permissions", {})
    list_virtual_perms = ui_permissions.get("list_virtual_server", [])

    # Admin users or users with "all" permission see everything
    if user_context.get("is_admin") or "all" in list_virtual_perms:
        filtered_servers = all_servers
    else:
        # Filter to only virtual servers the user has explicit permission for
        # Permission values are virtual server paths like "/virtual/my-server"
        normalized_perms = [p.strip("/") for p in list_virtual_perms]
        filtered_servers = [s for s in all_servers if s.path.strip("/") in normalized_perms]

    logger.info(
        f"Returning {len(filtered_servers)} virtual servers for user "
        f"{user_context.get('username', 'unknown')} (filtered from {len(all_servers)} total)"
    )
    return {
        "virtual_servers": [s.model_dump(mode="json") for s in filtered_servers],
        "total_count": len(filtered_servers),
    }


@router.get(
    "/virtual-servers/{vs_path:path}/tools",
    response_model=dict,
    summary="List resolved tools for a virtual server",
)
async def get_virtual_server_tools(
    user_context: Annotated[dict, Depends(nginx_proxied_auth)],
    vs_path: str = Path(..., description="Virtual server path"),
) -> dict:
    """Get the resolved tool list for a virtual server.

    Returns all tools with their final names, sources, and metadata.
    """
    normalized = _normalize_virtual_path(vs_path)
    service = get_virtual_server_service()

    try:
        tools = await service.resolve_tools(normalized)
        return {
            "path": normalized,
            "tools": [t.model_dump(mode="json") for t in tools],
            "total_count": len(tools),
        }
    except VirtualServerNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Virtual server not found: {normalized}",
        )


# --- Rating Endpoints (must be before catch-all GET) ---


class RatingRequest(BaseModel):
    """Request model for rating a virtual server."""

    rating: int


@router.post(
    "/virtual-servers/{vs_path:path}/rate",
    response_model=dict,
    summary="Rate a virtual server",
)
async def rate_virtual_server(
    http_request: Request,
    rating_request: RatingRequest,
    user_context: Annotated[dict, Depends(nginx_proxied_auth)],
    vs_path: str = Path(..., description="Virtual server path"),
) -> dict:
    """Submit or update a rating for a virtual server.

    Requires authentication. Each user can have one rating per server.
    """
    normalized = _normalize_virtual_path(vs_path)
    username = user_context.get("username", "anonymous")

    set_audit_action(
        http_request,
        "rate",
        "virtual_server",
        resource_id=normalized,
        description=f"Rate virtual server with {rating_request.rating} stars",
    )

    service = get_virtual_server_service()

    try:
        result = await service.rate_virtual_server(
            path=normalized,
            username=username,
            rating=rating_request.rating,
        )
        return result
    except VirtualServerNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Virtual server not found: {normalized}",
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )


@router.get(
    "/virtual-servers/{vs_path:path}/rating",
    response_model=dict,
    summary="Get virtual server rating",
)
async def get_virtual_server_rating(
    user_context: Annotated[dict, Depends(nginx_proxied_auth)],
    vs_path: str = Path(..., description="Virtual server path"),
) -> dict:
    """Get rating information for a virtual server."""
    normalized = _normalize_virtual_path(vs_path)
    service = get_virtual_server_service()

    try:
        return await service.get_virtual_server_rating(normalized)
    except VirtualServerNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Virtual server not found: {normalized}",
        )


@router.get(
    "/virtual-servers/{vs_path:path}",
    response_model=VirtualServerConfig,
    summary="Get a virtual server by path",
)
async def get_virtual_server(
    user_context: Annotated[dict, Depends(nginx_proxied_auth)],
    vs_path: str = Path(..., description="Virtual server path"),
) -> VirtualServerConfig:
    """Get detailed configuration for a virtual server."""
    normalized = _normalize_virtual_path(vs_path)
    service = get_virtual_server_service()
    config = await service.get_virtual_server(normalized)

    if not config:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Virtual server not found: {normalized}",
        )

    return config


@router.post(
    "/virtual-servers",
    response_model=VirtualServerConfig,
    status_code=status.HTTP_201_CREATED,
    summary="Create a virtual server",
)
async def create_virtual_server(
    http_request: Request,
    request: CreateVirtualServerRequest,
    user_context: Annotated[dict, Depends(nginx_proxied_auth)],
) -> VirtualServerConfig:
    """Create a new virtual MCP server.

    Requires admin or server-modify permissions.
    """
    _require_admin(user_context)

    set_audit_action(
        http_request,
        "create",
        "virtual_server",
        resource_id=request.server_name,
        description=f"Create virtual server '{request.server_name}'",
    )

    service = get_virtual_server_service()
    created_by = user_context.get("username")

    try:
        config = await service.create_virtual_server(
            request=request,
            created_by=created_by,
        )
        logger.info(
            f"Created virtual server '{config.server_name}' at {config.path} by {created_by}"
        )
        return config

    except VirtualServerAlreadyExistsError as e:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(e),
        )
    except VirtualServerValidationError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except VirtualServerServiceError as e:
        logger.error(f"Failed to create virtual server: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create virtual server",
        )


@router.put(
    "/virtual-servers/{vs_path:path}",
    response_model=VirtualServerConfig,
    summary="Update a virtual server",
)
async def update_virtual_server(
    http_request: Request,
    request: UpdateVirtualServerRequest,
    user_context: Annotated[dict, Depends(nginx_proxied_auth)],
    vs_path: str = Path(..., description="Virtual server path"),
) -> VirtualServerConfig:
    """Update an existing virtual MCP server.

    Requires admin or server-modify permissions.
    """
    _require_admin(user_context)
    normalized = _normalize_virtual_path(vs_path)

    set_audit_action(
        http_request,
        "update",
        "virtual_server",
        resource_id=normalized,
        description=f"Update virtual server at {normalized}",
    )

    service = get_virtual_server_service()

    try:
        config = await service.update_virtual_server(
            path=normalized,
            request=request,
        )

        if not config:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Virtual server not found: {normalized}",
            )

        return config

    except VirtualServerNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Virtual server not found: {normalized}",
        )
    except VirtualServerValidationError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except VirtualServerServiceError as e:
        logger.error(f"Failed to update virtual server: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update virtual server",
        )


@router.delete(
    "/virtual-servers/{vs_path:path}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete a virtual server",
)
async def delete_virtual_server(
    http_request: Request,
    user_context: Annotated[dict, Depends(nginx_proxied_auth)],
    vs_path: str = Path(..., description="Virtual server path"),
) -> None:
    """Delete a virtual MCP server.

    Requires admin or server-modify permissions.
    """
    _require_admin(user_context)
    normalized = _normalize_virtual_path(vs_path)

    set_audit_action(
        http_request,
        "delete",
        "virtual_server",
        resource_id=normalized,
        description=f"Delete virtual server at {normalized}",
    )

    service = get_virtual_server_service()

    try:
        success = await service.delete_virtual_server(normalized)
        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Virtual server not found: {normalized}",
            )
    except VirtualServerNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Virtual server not found: {normalized}",
        )


@router.post(
    "/virtual-servers/{vs_path:path}/toggle",
    response_model=dict,
    summary="Toggle virtual server enabled state",
)
async def toggle_virtual_server(
    http_request: Request,
    request: ToggleVirtualServerRequest,
    user_context: Annotated[dict, Depends(nginx_proxied_auth)],
    vs_path: str = Path(..., description="Virtual server path"),
) -> dict:
    """Enable or disable a virtual MCP server.

    Requires admin or server-modify permissions.
    Enabling triggers nginx configuration regeneration.
    """
    _require_admin(user_context)
    normalized = _normalize_virtual_path(vs_path)

    set_audit_action(
        http_request,
        "toggle",
        "virtual_server",
        resource_id=normalized,
        description=f"Toggle virtual server to {request.enabled}",
    )

    service = get_virtual_server_service()

    try:
        success = await service.toggle_virtual_server(
            path=normalized,
            enabled=request.enabled,
        )

        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Virtual server not found: {normalized}",
            )

        return {"path": normalized, "is_enabled": request.enabled}

    except VirtualServerNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Virtual server not found: {normalized}",
        )
    except VirtualServerValidationError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )


# --- Tool Catalog Endpoint ---


@router.get(
    "/tool-catalog",
    response_model=dict,
    summary="Browse all available tools across servers",
)
async def get_tool_catalog(
    user_context: Annotated[dict, Depends(nginx_proxied_auth)],
    server_path: str | None = Query(
        None,
        description="Filter by server path",
    ),
) -> dict:
    """Get the global tool catalog from all enabled MCP servers.

    Returns tools the authenticated user has access to, filtered by
    their scopes. Includes source server, description, input schema,
    and available versions.
    """
    service = get_tool_catalog_service()
    # Admin users bypass scope filtering (consistent with /api/servers)
    user_scopes = None if user_context.get("is_admin") else user_context.get("scopes", [])
    catalog = await service.get_tool_catalog(
        server_path_filter=server_path,
        user_scopes=user_scopes,
    )

    # Group by server for convenience
    servers: dict[str, list[dict]] = {}
    for entry in catalog:
        server_key = entry.server_path
        if server_key not in servers:
            servers[server_key] = []
        servers[server_key].append(entry.model_dump(mode="json"))

    return {
        "tools": [e.model_dump(mode="json") for e in catalog],
        "total_count": len(catalog),
        "server_count": len(servers),
        "by_server": servers,
    }
