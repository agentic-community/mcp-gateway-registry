from __future__ import annotations

import logging
from typing import (
    Callable,
)

from fastapi import (
    HTTPException,
    status,
)
from fastapi.responses import (
    JSONResponse,
)

RequireUserContextFn = Callable[[dict | None], dict]

logger = logging.getLogger("registry.api.server_routes")


def _normalize_server_path(
    server_name: str,
) -> str:
    if server_name.startswith("/"):
        return server_name
    return f"/{server_name}"


def _parse_group_names(
    group_names: str,
) -> list[str]:
    groups = [group.strip() for group in group_names.split(",") if group.strip()]
    if not groups:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No valid group names provided",
        )
    return groups


async def _add_server_to_groups(
    *,
    server_name: str,
    group_names: str,
    user_context: dict | None,
    require_user_context: RequireUserContextFn,
) -> JSONResponse:
    from ..utils.scopes_manager import add_server_to_groups

    user_context = require_user_context(user_context)
    username = user_context.get("username", "unknown")

    groups = _parse_group_names(group_names)
    server_path = _normalize_server_path(server_name)

    logger.info(
        "Adding server %s to groups %s via internal endpoint by admin '%s'",
        server_path,
        groups,
        username,
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

    except HTTPException:
        raise
    except Exception as exc:
        logger.error(
            "Error adding server %s to groups %s: %s",
            server_path,
            groups,
            exc,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal error: {str(exc)}",
        ) from exc


async def _remove_server_from_groups(
    *,
    server_name: str,
    group_names: str,
    user_context: dict | None,
    require_user_context: RequireUserContextFn,
) -> JSONResponse:
    from ..utils.scopes_manager import remove_server_from_groups

    user_context = require_user_context(user_context)
    username = user_context.get("username", "unknown")

    groups = _parse_group_names(group_names)
    server_path = _normalize_server_path(server_name)

    logger.info(
        "Removing server %s from groups %s via internal endpoint by admin '%s'",
        server_path,
        groups,
        username,
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

    except HTTPException:
        raise
    except Exception as exc:
        logger.error(
            "Error removing server %s from groups %s: %s",
            server_path,
            groups,
            exc,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal error: {str(exc)}",
        ) from exc


async def _create_group(
    *,
    group_name: str,
    description: str,
    create_in_keycloak: bool,
    user_context: dict | None,
    require_user_context: RequireUserContextFn,
) -> JSONResponse:
    from ..utils.keycloak_manager import (
        create_keycloak_group,
        group_exists_in_keycloak,
    )
    from ..utils.scopes_manager import (
        create_group_in_scopes,
    )

    user_context = require_user_context(user_context)
    username = user_context.get("username", "unknown")

    if not group_name or not group_name.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Group name is required",
        )

    logger.info(
        "Creating group '%s' via internal endpoint by admin '%s'",
        group_name,
        username,
    )

    try:
        keycloak_created = False
        if create_in_keycloak:
            try:
                if await group_exists_in_keycloak(group_name):
                    logger.warning("Group '%s' already exists in Keycloak", group_name)
                else:
                    await create_keycloak_group(group_name, description)
                    keycloak_created = True
                    logger.info("Group '%s' created in Keycloak", group_name)
            except Exception as exc:
                logger.error("Failed to create group in Keycloak: %s", exc)
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Failed to create group in Keycloak: {str(exc)}",
                ) from exc

        scopes_success = await create_group_in_scopes(group_name, description)
        if not scopes_success:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Failed to create group in scopes.yml (may already exist)",
            )

        return JSONResponse(
            status_code=200,
            content={
                "message": "Group successfully created",
                "group_name": group_name,
                "created_in_keycloak": keycloak_created,
                "created_in_scopes": True,
            },
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Error creating group '%s': %s", group_name, exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal error: {str(exc)}",
        ) from exc


async def _delete_group(
    *,
    group_name: str,
    delete_from_keycloak: bool,
    force: bool,
    user_context: dict | None,
    require_user_context: RequireUserContextFn,
) -> JSONResponse:
    from ..utils.keycloak_manager import (
        delete_keycloak_group,
        group_exists_in_keycloak,
    )
    from ..utils.scopes_manager import (
        delete_group_from_scopes,
    )

    user_context = require_user_context(user_context)
    username = user_context.get("username", "unknown")

    if not group_name or not group_name.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Group name is required",
        )

    system_groups = [
        "UI-Scopes",
        "group_mappings",
        "mcp-registry-admin",
        "mcp-registry-user",
        "mcp-registry-developer",
        "mcp-registry-operator",
    ]

    if group_name in system_groups and not force:
        logger.warning(
            "Attempt to delete system group '%s' by admin '%s'",
            group_name,
            username,
        )
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Cannot delete system group '{group_name}'",
        )

    logger.info(
        "Deleting group '%s' via internal endpoint by admin '%s'",
        group_name,
        username,
    )

    try:
        scopes_success = await delete_group_from_scopes(
            group_name,
            remove_from_mappings=True,
        )

        if not scopes_success:
            logger.warning("Group '%s' not found in scopes.yml or deletion failed", group_name)

        keycloak_deleted = False
        if delete_from_keycloak:
            try:
                if await group_exists_in_keycloak(group_name):
                    await delete_keycloak_group(group_name)
                    keycloak_deleted = True
                    logger.info("Group '%s' deleted from Keycloak", group_name)
                else:
                    logger.warning("Group '%s' not found in Keycloak", group_name)
            except Exception as exc:
                logger.error("Failed to delete group from Keycloak: %s", exc)
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
    except Exception as exc:
        logger.error("Error deleting group '%s': %s", group_name, exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal error: {str(exc)}",
        ) from exc


async def _list_groups(
    *,
    include_keycloak: bool,
    include_scopes: bool,
    user_context: dict | None,
    require_user_context: RequireUserContextFn,
) -> JSONResponse:
    from ..utils.keycloak_manager import (
        list_keycloak_groups,
    )
    from ..utils.scopes_manager import (
        list_groups_from_scopes,
    )

    user_context = require_user_context(user_context)
    username = user_context.get("username", "unknown")

    logger.info("Listing groups via internal endpoint by admin '%s'", username)

    try:
        result: dict = {
            "keycloak_groups": [],
            "scopes_groups": {},
            "synchronized": [],
            "keycloak_only": [],
            "scopes_only": [],
        }

        keycloak_group_names: set[str] = set()
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
                logger.info("Found %s groups in Keycloak", len(keycloak_groups))
            except Exception as exc:
                logger.error("Failed to list Keycloak groups: %s", exc)
                result["keycloak_error"] = str(exc)

        scopes_group_names: set[str] = set()
        if include_scopes:
            try:
                scopes_data = await list_groups_from_scopes()
                result["scopes_groups"] = scopes_data.get("groups", {})
                scopes_group_names = set(scopes_data.get("groups", {}).keys())
                logger.info("Found %s groups in scopes.yml", len(scopes_group_names))
            except Exception as exc:
                logger.error("Failed to list scopes groups: %s", exc)
                result["scopes_error"] = str(exc)

        if include_keycloak and include_scopes:
            result["synchronized"] = list(keycloak_group_names & scopes_group_names)
            result["keycloak_only"] = list(keycloak_group_names - scopes_group_names)
            result["scopes_only"] = list(scopes_group_names - keycloak_group_names)

        return JSONResponse(
            status_code=200,
            content=result,
        )

    except Exception as exc:
        logger.error("Error listing groups: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal error: {str(exc)}",
        ) from exc
