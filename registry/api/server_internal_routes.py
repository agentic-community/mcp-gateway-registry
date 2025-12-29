from __future__ import annotations

import json
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
from .server_routes_common import (
    _apply_remove_side_effects,
    _apply_toggle_side_effects,
    _build_server_entry_from_form,
    _enforce_proxy_pass_url_allowlist,
    _require_admin_user_context,
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
    logger.warning("INTERNAL REGISTER: Function called - starting execution")  # TODO: replace with debug

    from ..search.service import faiss_service
    from ..health.service import health_service
    from ..core.nginx_service import nginx_service

    logger.warning(
        f"INTERNAL REGISTER: Request parameters - name={name}, path={path}, proxy_pass_url={proxy_pass_url}"
    )  # TODO: replace with debug

    user_context = _require_admin_user_context(user_context)
    username = user_context.get("username", "unknown")

    logger.warning(
        f"INTERNAL REGISTER: Authentication successful for user {username}"
    )  # TODO: replace with debug
    logger.info(f"Internal service registration request from admin user '{username}'")

    # Validate path format
    if not path.startswith("/"):
        path = "/" + path
    logger.warning(f"INTERNAL REGISTER: Validated path: {path}")  # TODO: replace with debug

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

    logger.warning(
        f"INTERNAL REGISTER: Created server entry: {server_entry}"
    )  # TODO: replace with debug
    logger.warning(
        f"INTERNAL REGISTER: Overwrite parameter: {overwrite}"
    )  # TODO: replace with debug

    # Check if server exists and handle overwrite logic
    existing_server = server_service.get_server_info(path)
    if existing_server and not overwrite:
        logger.warning(
            f"INTERNAL REGISTER: Server exists and overwrite=False for path {path}"
        )  # TODO: replace with debug
        return JSONResponse(
            status_code=409,  # Conflict status code for existing resource
            content={
                "error": "Service registration failed",
                "reason": f"A service with path '{path}' already exists",
                "suggestion": "Set overwrite=true or use the remove command first",
            },
        )

    # Register the server (this will overwrite if server exists and overwrite=True)
    logger.warning(
        "INTERNAL REGISTER: Calling server_service.register_server"
    )  # TODO: replace with debug
    if existing_server and overwrite:
        logger.warning(
            f"INTERNAL REGISTER: Overwriting existing server at path {path}"
        )  # TODO: replace with debug
        success = server_service.update_server(path, server_entry)
    else:
        success = server_service.register_server(server_entry)

    if not success:
        logger.warning(
            f"INTERNAL REGISTER: Registration failed for path {path}"
        )  # TODO: replace with debug
        return JSONResponse(
            status_code=409,  # Conflict status code for existing resource
            content={
                "error": "Service registration failed",
                "reason": f"Failed to register service at path '{path}'",
                "suggestion": "Check server logs for detailed error information",
            },
        )

    logger.warning(
        "INTERNAL REGISTER: Auto-enabling newly registered server"
    )  # TODO: replace with debug

    # Automatically enable the newly registered server BEFORE FAISS indexing
    try:
        toggle_success = server_service.toggle_service(path, True)
        if toggle_success:
            logger.info(f"Successfully auto-enabled server {path} after registration")
        else:
            logger.warning(f"Failed to auto-enable server {path} after registration")
    except Exception as e:
        logger.error(f"Error auto-enabling server {path}: {e}")
        # Non-fatal error - server is registered but not enabled

    logger.warning(
        "INTERNAL REGISTER: Server registered successfully, adding to FAISS index"
    )  # TODO: replace with debug

    # Add to FAISS index with current enabled state (should be True after auto-enable)
    is_enabled = server_service.is_service_enabled(path)
    await faiss_service.add_or_update_service(path, server_entry, is_enabled)

    logger.warning(
        "INTERNAL REGISTER: Regenerating Nginx configuration"
    )  # TODO: replace with debug

    # Regenerate Nginx configuration
    enabled_servers = {
        server_path: server_service.get_server_info(server_path)
        for server_path in server_service.get_enabled_services()
    }
    await nginx_service.generate_config_async(enabled_servers)

    logger.warning(
        "INTERNAL REGISTER: Broadcasting health status update"
    )  # TODO: replace with debug

    # Broadcast health status update to WebSocket clients
    await health_service.broadcast_health_update(path)

    logger.warning(
        "INTERNAL REGISTER: Updating scopes.yml for new server"
    )  # TODO: replace with debug

    # Update scopes.yml with the new server's tools
    from ..utils.scopes_manager import update_server_scopes

    # Get the tool list from the server entry
    tool_names = []
    if "tool_list" in server_entry and server_entry["tool_list"]:
        tool_names = [
            tool["name"] for tool in server_entry["tool_list"] if "name" in tool
        ]

    # Update scopes and reload auth server
    try:
        await update_server_scopes(path, name, tool_names)
        logger.info(
            f"Successfully updated scopes for server {path} with {len(tool_names)} tools"
        )
    except Exception as e:
        logger.error(f"Failed to update scopes for server {path}: {e}")
        # Non-fatal error - server is registered but scopes not updated

    logger.warning(
        "INTERNAL REGISTER: Registration complete, returning success response"
    )  # TODO: replace with debug
    logger.info(
        f"New service registered via internal endpoint: '{name}' at path '{path}' by admin '{username}'"
    )

    return JSONResponse(
        status_code=201,
        content={
            "message": "Service registered successfully",
            "service": server_entry,
        },
    )


@router.post("/internal/remove")
async def internal_remove_service(
    request: Request,
    service_path: Annotated[str, Form()],
    user_context: Annotated[dict, Depends(nginx_proxied_auth)] = None,
):
    """Internal service removal endpoint for mcpgw-server (requires admin auth)."""
    from ..search.service import faiss_service
    from ..health.service import health_service
    from ..core.nginx_service import nginx_service

    logger.warning("INTERNAL REMOVE: Function called - starting execution")  # TODO: replace with debug

    user_context = _require_admin_user_context(user_context)
    username = user_context.get("username", "unknown")

    logger.info(
        f"Internal service removal request from admin user '{username}' for service '{service_path}'"
    )

    # Validate path format
    if not service_path.startswith("/"):
        service_path = "/" + service_path

    logger.warning(
        f"INTERNAL REMOVE: Normalized service path: {service_path}"
    )  # TODO: replace with debug

    # Check if server exists
    server_info = server_service.get_server_info(service_path)
    if not server_info:
        logger.warning(
            f"INTERNAL REMOVE: Service not found at path '{service_path}'"
        )  # TODO: replace with debug
        return JSONResponse(
            status_code=404,
            content={
                "error": "Service not found",
                "reason": f"No service registered at path '{service_path}'",
                "suggestion": "Check the service path and ensure it is registered",
            },
        )

    logger.warning(
        "INTERNAL REMOVE: Service found, proceeding with removal"
    )  # TODO: replace with debug

    # Remove the server
    success = server_service.remove_server(service_path)

    if not success:
        logger.warning(
            f"INTERNAL REMOVE: Failed to remove service at path '{service_path}'"
        )  # TODO: replace with debug
        return JSONResponse(
            status_code=500,
            content={
                "error": "Service removal failed",
                "reason": f"Failed to remove service at path '{service_path}'",
                "suggestion": "Check server logs for detailed error information",
            },
        )

    from ..utils.scopes_manager import remove_server_scopes

    logger.warning(
        "INTERNAL REMOVE: Service removed successfully, updating FAISS index"
    )  # TODO: replace with debug

    await _apply_remove_side_effects(
        service_path=service_path,
        server_service=server_service,
        faiss_service=faiss_service,
        health_service=health_service,
        nginx_service=nginx_service,
        remove_server_scopes=remove_server_scopes,
        scopes_error_log_level="error",
        logger=logger,
    )

    logger.warning(
        "INTERNAL REMOVE: Removal complete, returning success response"
    )  # TODO: replace with debug
    logger.info(f"Service removed via internal endpoint: '{service_path}' by admin '{username}'")

    return JSONResponse(
        status_code=200,
        content={
            "message": "Service removed successfully",
            "service_path": service_path,
        },
    )


@router.post("/internal/toggle")
async def internal_toggle_service(
    request: Request,
    service_path: Annotated[str, Form()],
    user_context: Annotated[dict, Depends(nginx_proxied_auth)] = None,
):
    """Internal service toggle endpoint for mcpgw-server (requires admin auth)."""
    from ..search.service import faiss_service
    from ..health.service import health_service
    from ..core.nginx_service import nginx_service

    logger.warning("INTERNAL TOGGLE: Function called - starting execution")  # TODO: replace with debug

    user_context = _require_admin_user_context(user_context)
    username = user_context.get("username", "unknown")
    logger.warning(
        f"INTERNAL TOGGLE: Admin authentication successful for user '{username}'"
    )  # TODO: replace with debug

    # Ensure service_path starts with /
    if not service_path.startswith("/"):
        service_path = "/" + service_path

    # Check if server exists
    server_info = server_service.get_server_info(service_path)
    if not server_info:
        logger.warning(
            f"INTERNAL TOGGLE: Service not found at path '{service_path}'"
        )  # TODO: replace with debug
        return JSONResponse(
            status_code=404,
            content={
                "error": "Service not found",
                "reason": f"No service registered at path '{service_path}'",
                "suggestion": "Check the service path and ensure it is registered",
            },
        )

    logger.warning(
        "INTERNAL TOGGLE: Service found, proceeding with toggle"
    )  # TODO: replace with debug

    # Get current state and toggle it
    current_state = server_service.is_service_enabled(service_path)
    new_state = not current_state
    success = server_service.toggle_service(service_path, new_state)

    if not success:
        logger.warning(
            f"INTERNAL TOGGLE: Failed to toggle service at path '{service_path}'"
        )  # TODO: replace with debug
        return JSONResponse(
            status_code=500,
            content={
                "error": "Service toggle failed",
                "reason": f"Failed to toggle service at path '{service_path}'",
                "suggestion": "Check server logs for detailed error information",
            },
        )

    server_name = server_info["server_name"]
    logger.info(f"Toggled '{server_name}' ({service_path}) to {new_state} by admin '{username}'")

    status_result, last_checked_iso = await _apply_toggle_side_effects(
        service_path=service_path,
        server_info=server_info,
        new_state=new_state,
        server_service=server_service,
        faiss_service=faiss_service,
        health_service=health_service,
        nginx_service=nginx_service,
        logger=logger,
    )

    logger.warning("INTERNAL TOGGLE: Toggle complete, returning success response")  # TODO: replace with debug
    return JSONResponse(
        status_code=200,
        content={
            "message": "Service toggled successfully",
            "service_path": service_path,
            "new_enabled_state": new_state,
            "status": status_result,
            "last_checked_iso": last_checked_iso,
            "num_tools": server_info.get("num_tools", 0),
        },
    )


@router.post("/internal/healthcheck")
async def internal_healthcheck(
    request: Request,
    user_context: Annotated[dict, Depends(nginx_proxied_auth)] = None,
):
    """Internal health check endpoint for mcpgw-server (requires admin auth)."""
    from ..health.service import health_service

    logger.warning(
        "INTERNAL HEALTHCHECK: Function called - starting execution"
    )  # TODO: replace with debug

    user_context = _require_admin_user_context(user_context)
    username = user_context.get("username", "unknown")
    logger.warning(
        f"INTERNAL HEALTHCHECK: Admin authenticated successfully: {username}"
    )  # TODO: replace with debug

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


@router.post("/internal/add-to-groups")
async def internal_add_server_to_groups(
    request: Request,
    server_name: Annotated[str, Form()],
    group_names: Annotated[str, Form()],  # Comma-separated list
    user_context: Annotated[dict, Depends(nginx_proxied_auth)] = None,
):
    """Internal endpoint to add a server to specific scopes groups (requires admin auth)."""
    from ..utils.scopes_manager import add_server_to_groups

    user_context = _require_admin_user_context(user_context)
    username = user_context.get("username", "unknown")

    # Parse group names from comma-separated string
    groups = [group.strip() for group in group_names.split(",") if group.strip()]
    if not groups:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No valid group names provided",
        )

    # Convert server name to path format
    server_path = f"/{server_name}" if not server_name.startswith("/") else server_name

    logger.info(f"Adding server {server_path} to groups {groups} via internal endpoint by admin '{username}'")

    try:
        success = await add_server_to_groups(server_path, groups)

        if success:
            return JSONResponse(
                status_code=200,
                content={
                    "message": "Server successfully added to groups",
                    "server_path": server_path,
                    "groups": groups,
                },
            )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Failed to add server to groups",
        )

    except Exception as e:
        logger.error(f"Error adding server {server_path} to groups {groups}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal error: {str(e)}",
        )


@router.post("/internal/remove-from-groups")
async def internal_remove_server_from_groups(
    request: Request,
    server_name: Annotated[str, Form()],
    group_names: Annotated[str, Form()],  # Comma-separated list
    user_context: Annotated[dict, Depends(nginx_proxied_auth)] = None,
):
    """Internal endpoint to remove a server from specific scopes groups (requires admin auth)."""
    from ..utils.scopes_manager import remove_server_from_groups

    user_context = _require_admin_user_context(user_context)
    username = user_context.get("username", "unknown")

    # Parse group names from comma-separated string
    groups = [group.strip() for group in group_names.split(",") if group.strip()]
    if not groups:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No valid group names provided",
        )

    # Convert server name to path format
    server_path = f"/{server_name}" if not server_name.startswith("/") else server_name

    logger.info(
        f"Removing server {server_path} from groups {groups} via internal endpoint by admin '{username}'"
    )

    try:
        success = await remove_server_from_groups(server_path, groups)

        if success:
            return JSONResponse(
                status_code=200,
                content={
                    "message": "Server successfully removed from groups",
                    "server_path": server_path,
                    "groups": groups,
                },
            )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Failed to remove server from groups",
        )

    except Exception as e:
        logger.error(f"Error removing server {server_path} from groups {groups}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal error: {str(e)}",
        )


@router.get("/internal/list")
async def internal_list_services(
    request: Request,
    user_context: Annotated[dict, Depends(nginx_proxied_auth)] = None,
):
    """Internal service listing endpoint for mcpgw-server (requires admin auth)."""
    logger.warning("INTERNAL LIST: Function called - starting execution")  # TODO: replace with debug

    user_context = _require_admin_user_context(user_context)
    username = user_context.get("username", "unknown")

    logger.info(f"Internal service list request from admin user '{username}'")

    # Get all servers (admin access - no permission filtering)
    all_servers = server_service.get_all_servers()

    logger.warning(f"INTERNAL LIST: Found {len(all_servers)} servers")  # TODO: replace with debug

    # Transform the data to include enabled status and health information
    services = []
    for service_path, server_info in all_servers.items():
        from ..health.service import health_service

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

    logger.warning(f"INTERNAL LIST: Returning {len(services)} services")  # TODO: replace with debug
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


@router.post("/internal/create-group")
async def internal_create_group(
    request: Request,
    group_name: Annotated[str, Form()],
    description: Annotated[str, Form()] = "",
    create_in_keycloak: Annotated[bool, Form()] = True,
    user_context: Annotated[dict, Depends(nginx_proxied_auth)] = None,
):
    """Internal endpoint to create a new group in both Keycloak and scopes.yml (requires admin auth)."""
    from ..utils.scopes_manager import create_group_in_scopes
    from ..utils.keycloak_manager import create_keycloak_group, group_exists_in_keycloak

    user_context = _require_admin_user_context(user_context)
    username = user_context.get("username", "unknown")

    # Validate group name
    if not group_name or not group_name.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Group name is required",
        )

    logger.info(f"Creating group '{group_name}' via internal endpoint by admin '{username}'")

    try:
        # Create in Keycloak first if requested
        keycloak_created = False
        if create_in_keycloak:
            try:
                # Check if group already exists in Keycloak
                if await group_exists_in_keycloak(group_name):
                    logger.warning(f"Group '{group_name}' already exists in Keycloak")
                else:
                    await create_keycloak_group(group_name, description)
                    keycloak_created = True
                    logger.info(f"Group '{group_name}' created in Keycloak")
            except Exception as e:
                logger.error(f"Failed to create group in Keycloak: {e}")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Failed to create group in Keycloak: {str(e)}",
                )

        # Create in scopes.yml
        scopes_success = await create_group_in_scopes(group_name, description)

        if scopes_success:
            return JSONResponse(
                status_code=200,
                content={
                    "message": "Group successfully created",
                    "group_name": group_name,
                    "created_in_keycloak": keycloak_created,
                    "created_in_scopes": True,
                },
            )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Failed to create group in scopes.yml (may already exist)",
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating group '{group_name}': {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal error: {str(e)}",
        )


@router.post("/internal/delete-group")
async def internal_delete_group(
    request: Request,
    group_name: Annotated[str, Form()],
    delete_from_keycloak: Annotated[bool, Form()] = True,
    force: Annotated[bool, Form()] = False,
    user_context: Annotated[dict, Depends(nginx_proxied_auth)] = None,
):
    """Internal endpoint to delete a group from both Keycloak and scopes.yml (requires admin auth)."""
    from ..utils.scopes_manager import delete_group_from_scopes
    from ..utils.keycloak_manager import delete_keycloak_group, group_exists_in_keycloak

    user_context = _require_admin_user_context(user_context)
    username = user_context.get("username", "unknown")

    # Validate group name
    if not group_name or not group_name.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Group name is required",
        )

    # Prevent deletion of system groups
    system_groups = [
        "UI-Scopes",
        "group_mappings",
        "mcp-registry-admin",
        "mcp-registry-user",
        "mcp-registry-developer",
        "mcp-registry-operator",
    ]

    if group_name in system_groups:
        logger.warning(f"Attempt to delete system group '{group_name}' by admin '{username}'")
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Cannot delete system group '{group_name}'",
        )

    logger.info(f"Deleting group '{group_name}' via internal endpoint by admin '{username}'")

    try:
        # Delete from scopes.yml first
        scopes_success = await delete_group_from_scopes(group_name, remove_from_mappings=True)

        if not scopes_success:
            logger.warning(f"Group '{group_name}' not found in scopes.yml or deletion failed")

        # Delete from Keycloak if requested
        keycloak_deleted = False
        if delete_from_keycloak:
            try:
                if await group_exists_in_keycloak(group_name):
                    await delete_keycloak_group(group_name)
                    keycloak_deleted = True
                    logger.info(f"Group '{group_name}' deleted from Keycloak")
                else:
                    logger.warning(f"Group '{group_name}' not found in Keycloak")
            except Exception as e:
                logger.error(f"Failed to delete group from Keycloak: {e}")
                # Continue anyway - scopes deletion might have succeeded

        if scopes_success or keycloak_deleted:
            return JSONResponse(
                status_code=200,
                content={
                    "message": "Group deletion completed",
                    "group_name": group_name,
                    "deleted_from_keycloak": keycloak_deleted,
                    "deleted_from_scopes": scopes_success,
                },
            )
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Group '{group_name}' not found in either Keycloak or scopes.yml",
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting group '{group_name}': {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal error: {str(e)}",
        )


@router.get("/internal/list-groups")
async def internal_list_groups(
    request: Request,
    include_keycloak: bool = True,
    include_scopes: bool = True,
    user_context: Annotated[dict, Depends(nginx_proxied_auth)] = None,
):
    """Internal endpoint to list groups from Keycloak and/or scopes.yml (requires admin auth)."""
    from ..utils.scopes_manager import list_groups_from_scopes
    from ..utils.keycloak_manager import list_keycloak_groups

    user_context = _require_admin_user_context(user_context)
    username = user_context.get("username", "unknown")

    logger.info(f"Listing groups via internal endpoint by admin '{username}'")

    try:
        result = {
            "keycloak_groups": [],
            "scopes_groups": {},
            "synchronized": [],
            "keycloak_only": [],
            "scopes_only": [],
        }

        # Get groups from Keycloak
        keycloak_group_names = set()
        if include_keycloak:
            try:
                keycloak_groups = await list_keycloak_groups()
                result["keycloak_groups"] = [
                    {
                        "name": group.get("name"),
                        "id": group.get("id"),
                        "path": group.get("path", ""),
                    }
                    for group in keycloak_groups
                ]
                keycloak_group_names = {group.get("name") for group in keycloak_groups}
                logger.info(f"Found {len(keycloak_groups)} groups in Keycloak")
            except Exception as e:
                logger.error(f"Failed to list Keycloak groups: {e}")
                result["keycloak_error"] = str(e)

        # Get groups from scopes.yml
        scopes_group_names = set()
        if include_scopes:
            try:
                scopes_data = await list_groups_from_scopes()
                result["scopes_groups"] = scopes_data.get("groups", {})
                scopes_group_names = set(scopes_data.get("groups", {}).keys())
                logger.info(f"Found {len(scopes_group_names)} groups in scopes.yml")
            except Exception as e:
                logger.error(f"Failed to list scopes groups: {e}")
                result["scopes_error"] = str(e)

        # Find synchronized and out-of-sync groups
        if include_keycloak and include_scopes:
            result["synchronized"] = list(keycloak_group_names & scopes_group_names)
            result["keycloak_only"] = list(keycloak_group_names - scopes_group_names)
            result["scopes_only"] = list(scopes_group_names - keycloak_group_names)

        return JSONResponse(
            status_code=200,
            content=result,
        )

    except Exception as e:
        logger.error(f"Error listing groups: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal error: {str(e)}",
        )
