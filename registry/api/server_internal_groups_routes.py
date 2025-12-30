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
    status,
)
from fastapi.responses import (
    JSONResponse,
)

from ..auth.dependencies import (
    nginx_proxied_auth,
)
from .server_routes_common import (
    _require_admin_user_context,
)

logger = logging.getLogger("registry.api.server_routes")

router = APIRouter()


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

    logger.info(
        f"Adding server {server_path} to groups {groups} via internal endpoint by admin '{username}'"
    )

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
        scopes_success = await delete_group_from_scopes(
            group_name,
            remove_from_mappings=True,
        )

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

