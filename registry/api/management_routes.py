from __future__ import annotations

import logging
import os
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status

from ..auth.dependencies import nginx_proxied_auth
from ..schemas.management import (
    GroupCreateRequest,
    GroupDeleteResponse,
    GroupDetailResponse,
    GroupListResponse,
    GroupSummary,
    GroupUpdateRequest,
    HumanUserRequest,
    M2MAccountRequest,
    UpdateUserGroupsRequest,
    UpdateUserGroupsResponse,
    UserDeleteResponse,
    UserListResponse,
    UserSummary,
)
from ..services import scope_service
from ..utils.iam_manager import get_iam_manager

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/management", tags=["Management API"])

AUTH_PROVIDER: str = os.environ.get("AUTH_PROVIDER", "keycloak")


def _translate_iam_error(exc: Exception) -> HTTPException:
    """
    Map IAM admin errors to HTTP responses.

    Works for both Keycloak and Entra ID error messages.

    Args:
        exc: The exception from IAM operations

    Returns:
        HTTPException with appropriate status code
    """
    detail = str(exc)
    lowered = detail.lower()
    status_code = status.HTTP_502_BAD_GATEWAY

    if any(keyword in lowered for keyword in ("already exists", "not found", "provided")):
        status_code = status.HTTP_400_BAD_REQUEST

    return HTTPException(status_code=status_code, detail=detail)


def _normalize_agent_path(path: str) -> str:
    """
    Normalize agent path to ensure it has a leading slash.

    Args:
        path: Agent path to normalize

    Returns:
        Normalized path with leading slash
    """
    if not path:
        return path
    path = path.strip()
    if not path.startswith("/"):
        path = "/" + path
    if path.endswith("/") and len(path) > 1:
        path = path.rstrip("/")
    return path


def _normalize_agent_paths_in_scope_config(
    agent_access: list | None,
    ui_permissions: dict | None,
) -> tuple[list | None, dict | None]:
    """
    Normalize agent paths in agent_access and ui_permissions.

    Ensures all agent paths have leading slashes for consistent matching.

    Args:
        agent_access: List of agent paths
        ui_permissions: Dict of UI permissions

    Returns:
        Tuple of (normalized_agent_access, normalized_ui_permissions)
    """
    # Normalize agent_access
    if agent_access:
        agent_access = [_normalize_agent_path(p) for p in agent_access if p]

    # Normalize agent-related ui_permissions
    if ui_permissions:
        for key in ["list_agents", "get_agent", "publish_agent", "modify_agent", "delete_agent"]:
            if key in ui_permissions and isinstance(ui_permissions[key], list):
                # Don't normalize "all" - it's a special value
                ui_permissions[key] = [
                    p if p == "all" else _normalize_agent_path(p) for p in ui_permissions[key] if p
                ]

    return agent_access, ui_permissions


def _require_admin(user_context: dict) -> None:
    """
    Verify user has admin permissions.

    Args:
        user_context: User context from authentication

    Raises:
        HTTPException: If user is not an admin
    """
    if not user_context.get("is_admin"):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Administrator permissions are required for this operation",
        )


@router.get("/iam/users", response_model=UserListResponse)
async def management_list_users(
    search: str | None = None,
    limit: int = 500,
    user_context: Annotated[dict, Depends(nginx_proxied_auth)] = None,
):
    """List users from the configured identity provider (admin only)."""
    _require_admin(user_context)

    iam = get_iam_manager()

    try:
        raw_users = await iam.list_users(search=search, max_results=limit)
        logger.debug(f"[LIST_USERS] Retrieved {len(raw_users)} users from IAM")
    except Exception as exc:
        logger.error(f"[LIST_USERS] Exception calling list_users: {type(exc).__name__}: {exc}")
        raise _translate_iam_error(exc) from exc

    # Include M2M clients from MongoDB for all providers
    try:
        from registry.repositories.documentdb.client import get_documentdb_client

        db = await get_documentdb_client()
        collection = db["idp_m2m_clients"]

        # Query M2M clients from MongoDB
        cursor = collection.find({})
        m2m_docs = await cursor.to_list(length=None)

        # Add M2M clients as users with special email pattern
        for doc in m2m_docs:
            client_id = doc.get("client_id", "")
            raw_users.append(
                {
                    "id": client_id,
                    "username": doc.get("name", client_id),
                    "email": f"{client_id}@service-account.local",  # Special email pattern for M2M
                    "firstName": None,
                    "lastName": None,
                    "enabled": doc.get("enabled", True),
                    "groups": doc.get("groups", []),
                }
            )

        logger.debug(f"[LIST_USERS] Added {len(m2m_docs)} M2M clients from MongoDB")
    except Exception as e:
        logger.warning(f"Failed to retrieve M2M clients from MongoDB: {e}")
        # Don't fail the entire operation if MongoDB query fails

    summaries = [
        UserSummary(
            id=user.get("id", ""),
            username=user.get("username", ""),
            email=user.get("email"),
            firstName=user.get("firstName"),
            lastName=user.get("lastName"),
            enabled=user.get("enabled", True),
            groups=user.get("groups", []),
        )
        for user in raw_users
    ]
    return UserListResponse(users=summaries, total=len(summaries))


@router.post("/iam/users/m2m")
async def management_create_m2m_user(
    payload: M2MAccountRequest,
    user_context: Annotated[dict, Depends(nginx_proxied_auth)] = None,
):
    """Create a service account client and return its credentials (admin only)."""
    _require_admin(user_context)

    iam = get_iam_manager()

    try:
        result = await iam.create_service_account(
            client_id=payload.name,
            groups=payload.groups,
            description=payload.description,
        )

        # Store M2M client in MongoDB for all providers (authorization database)
        try:
            from datetime import datetime
            from os import environ

            from registry.repositories.documentdb.client import get_documentdb_client

            db = await get_documentdb_client()
            collection = db["idp_m2m_clients"]

            provider = environ.get("AUTH_PROVIDER", "keycloak").lower()

            client_doc = {
                "client_id": result.get("client_id"),
                "name": payload.name,
                "description": payload.description,
                "groups": payload.groups,
                "enabled": True,
                "provider": provider,
                "idp_app_id": result.get("okta_app_id") or result.get("client_id"),
                "created_at": datetime.utcnow(),
                "updated_at": datetime.utcnow(),
            }

            await collection.insert_one(client_doc)
            client_id_val = result.get("client_id", "")
            masked_client_id = f"{client_id_val[:8]}..." if client_id_val else "<none>"
            logger.info(f"Stored M2M client in MongoDB: {masked_client_id} (provider: {provider})")
        except Exception as e:
            logger.warning(f"Failed to store M2M client in MongoDB: {e}")
            # Don't fail the entire operation if MongoDB storage fails

    except Exception as exc:
        raise _translate_iam_error(exc) from exc

    return result


@router.post("/iam/users/human")
async def management_create_human_user(
    payload: HumanUserRequest,
    user_context: Annotated[dict, Depends(nginx_proxied_auth)] = None,
):
    """Create a human user and assign groups (admin only)."""
    _require_admin(user_context)

    iam = get_iam_manager()

    try:
        user_doc = await iam.create_human_user(
            username=payload.username,
            email=payload.email,
            first_name=payload.first_name,
            last_name=payload.last_name,
            groups=payload.groups,
            password=payload.password,
        )
    except Exception as exc:
        raise _translate_iam_error(exc) from exc

    return UserSummary(
        id=user_doc.get("id", ""),
        username=user_doc.get("username", payload.username),
        email=user_doc.get("email"),
        firstName=user_doc.get("firstName"),
        lastName=user_doc.get("lastName"),
        enabled=user_doc.get("enabled", True),
        groups=user_doc.get("groups", payload.groups),
    )


@router.delete("/iam/users/{username}", response_model=UserDeleteResponse)
async def management_delete_user(
    username: str,
    user_context: Annotated[dict, Depends(nginx_proxied_auth)] = None,
):
    """Delete a user by username (admin only)."""
    _require_admin(user_context)

    iam = get_iam_manager()

    try:
        await iam.delete_user(username=username)
    except Exception as exc:
        raise _translate_iam_error(exc) from exc

    return UserDeleteResponse(username=username)


@router.patch("/iam/users/{username}/groups", response_model=UpdateUserGroupsResponse)
async def management_update_user_groups(
    username: str,
    payload: UpdateUserGroupsRequest,
    user_context: Annotated[dict, Depends(nginx_proxied_auth)] = None,
):
    """Update a user's group memberships (admin only).

    This endpoint calculates the diff between current and desired groups,
    then adds or removes group memberships as needed.

    For M2M accounts (service accounts), updates the DocumentDB record directly.
    For human users, delegates to the IdP manager.
    """
    from datetime import datetime

    _require_admin(user_context)

    # Check if this is an M2M account by looking it up in DocumentDB
    try:
        from registry.repositories.documentdb.client import get_documentdb_client

        db = await get_documentdb_client()
        collection = db["idp_m2m_clients"]

        # Try to find M2M client by name (username is the name for M2M accounts in the UI)
        m2m_doc = await collection.find_one({"name": username})

        if m2m_doc:
            # This is an M2M account - update DocumentDB directly
            logger.info(f"Updating groups for M2M account: {username}")

            current_groups = m2m_doc.get("groups", [])
            new_groups = payload.groups

            added = list(set(new_groups) - set(current_groups))
            removed = list(set(current_groups) - set(new_groups))

            # Update the groups in DocumentDB
            await collection.update_one(
                {"name": username},
                {
                    "$set": {
                        "groups": new_groups,
                        "updated_at": datetime.utcnow(),
                    }
                },
            )

            logger.info(f"Updated M2M account {username}: added {added}, removed {removed}")

            return UpdateUserGroupsResponse(
                username=username,
                groups=new_groups,
                added=added,
                removed=removed,
            )
    except Exception as e:
        logger.warning(f"Error checking/updating M2M account in DocumentDB: {e}")
        # Continue to IdP update if DocumentDB check fails

    # If not an M2M account, update through IdP
    iam = get_iam_manager()

    try:
        result = await iam.update_user_groups(
            username=username,
            groups=payload.groups,
        )
    except Exception as exc:
        raise _translate_iam_error(exc) from exc

    return UpdateUserGroupsResponse(
        username=result.get("username", username),
        groups=result.get("groups", []),
        added=result.get("added", []),
        removed=result.get("removed", []),
    )


@router.get("/iam/groups", response_model=GroupListResponse)
async def management_list_groups(
    user_context: Annotated[dict, Depends(nginx_proxied_auth)] = None,
):
    """List IAM groups from the configured identity provider (admin only)."""
    _require_admin(user_context)

    iam = get_iam_manager()

    try:
        raw_groups = await iam.list_groups()
        summaries = [
            GroupSummary(
                id=group.get("id", ""),
                name=group.get("name", ""),
                path=group.get("path", ""),
                attributes=group.get("attributes"),
            )
            for group in raw_groups
        ]
        return GroupListResponse(groups=summaries, total=len(summaries))
    except Exception as exc:
        logger.error("Failed to list IAM groups: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Unable to list IAM groups",
        ) from exc


@router.post("/iam/groups", response_model=GroupSummary)
async def management_create_group(
    payload: GroupCreateRequest,
    user_context: Annotated[dict, Depends(nginx_proxied_auth)] = None,
):
    """
    Create a new group in the identity provider and/or MongoDB (admin only).

    When create_in_idp is True, creates in both the configured
    identity provider and MongoDB scopes collection.
    When create_in_idp is False, creates only in MongoDB scopes collection.
    """
    _require_admin(user_context)

    iam = get_iam_manager()

    # Extract create_in_idp from scope_config (frontend sends it there)
    create_in_idp = False  # default: do not create in IdP
    if payload.scope_config and "create_in_idp" in payload.scope_config:
        create_in_idp = bool(payload.scope_config["create_in_idp"])
    logger.debug(
        "create_in_idp=%s for group '%s' (from scope_config)",
        create_in_idp,
        payload.name,
    )

    try:
        result = {}
        group_mapping_id = payload.name  # default for local-only groups

        # Step 1: Create group in identity provider (only if requested)
        if create_in_idp:
            result = await iam.create_group(
                group_name=payload.name,
                description=payload.description or "",
            )

            # For Entra ID: use Object ID for group mapping
            # For Keycloak/Okta: use group name
            provider = AUTH_PROVIDER.lower()
            if provider == "entra":
                group_mapping_id = result.get("id", payload.name)
        else:
            # Local-only group: build a result dict without calling IdP
            result = {
                "id": payload.name,
                "name": payload.name,
                "path": f"/{payload.name}",
                "attributes": {"description": [payload.description or ""]},
            }
            logger.info(
                "Group '%s' created locally only (create_in_idp=False)",
                payload.name,
            )

        # Step 2: Create in MongoDB scopes collection (always)
        server_access = []
        ui_permissions = {}
        agent_access = []
        if payload.scope_config:
            server_access = payload.scope_config.get("server_access", [])
            ui_permissions = payload.scope_config.get("ui_permissions", {})
            agent_access = payload.scope_config.get("agent_access", [])

        # Normalize agent paths to ensure they have leading slashes
        agent_access, ui_permissions = _normalize_agent_paths_in_scope_config(
            agent_access, ui_permissions
        )

        import_success = await scope_service.import_group(
            scope_name=payload.name,
            description=payload.description or "",
            group_mappings=[group_mapping_id],
            server_access=server_access,
            ui_permissions=ui_permissions,
            agent_access=agent_access,
        )

        if not import_success:
            logger.warning(
                "Group %s in IdP but failed to create in MongoDB: %s",
                "created" if create_in_idp else "skipped",
                payload.name,
            )

        return GroupSummary(
            id=result.get("id", ""),
            name=result.get("name", ""),
            path=result.get("path", ""),
            attributes=result.get("attributes"),
        )

    except Exception as exc:
        logger.error("Failed to create group: %s", exc)
        detail = str(exc).lower()

        if "already exists" in detail:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=str(exc),
            ) from exc

        raise _translate_iam_error(exc) from exc


@router.delete("/iam/groups/{group_name}", response_model=GroupDeleteResponse)
async def management_delete_group(
    group_name: str,
    user_context: Annotated[dict, Depends(nginx_proxied_auth)] = None,
):
    """
    Delete a group from the identity provider and/or MongoDB (admin only).

    Attempts to delete from IdP first. If the group does not exist in the IdP
    (e.g., it was created with create_in_idp=False), the IdP error is logged
    and the MongoDB deletion proceeds.
    """
    _require_admin(user_context)

    iam = get_iam_manager()

    try:
        # Step 1: Attempt to delete from identity provider
        try:
            await iam.delete_group(group_name=group_name)
        except Exception as idp_exc:
            idp_detail = str(idp_exc).lower()
            if "not found" in idp_detail or "404" in idp_detail:
                logger.info(
                    "Group '%s' not found in IdP (may be local-only), "
                    "proceeding with MongoDB deletion",
                    group_name,
                )
            else:
                raise

        # Step 2: Delete from MongoDB scopes collection
        delete_success = await scope_service.delete_group(
            group_name=group_name, remove_from_mappings=True
        )

        if not delete_success:
            logger.warning(
                "Group deleted from IdP but failed to delete from MongoDB: %s",
                group_name,
            )

        return GroupDeleteResponse(name=group_name)

    except Exception as exc:
        logger.error("Failed to delete group: %s", exc)
        detail = str(exc).lower()

        if "not found" in detail:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Group '{group_name}' not found",
            ) from exc

        raise _translate_iam_error(exc) from exc


@router.get("/iam/groups/{group_name}", response_model=GroupDetailResponse)
async def management_get_group(
    group_name: str,
    user_context: Annotated[dict, Depends(nginx_proxied_auth)] = None,
):
    """
    Get detailed information about a group (admin only).

    Returns both identity provider data and MongoDB scope data.
    """
    _require_admin(user_context)

    try:
        # Get group details from MongoDB scopes
        group_data = await scope_service.get_group(group_name)

        if not group_data:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Group '{group_name}' not found",
            )

        return GroupDetailResponse(
            id=group_data.get("id", ""),
            name=group_data.get("name", group_name),
            path=group_data.get("path"),
            description=group_data.get("description"),
            server_access=group_data.get("server_access"),
            group_mappings=group_data.get("group_mappings"),
            ui_permissions=group_data.get("ui_permissions"),
            agent_access=group_data.get("agent_access"),
        )

    except HTTPException:
        raise

    except Exception as exc:
        logger.error("Failed to get group: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Failed to get group details: {exc}",
        ) from exc


@router.patch("/iam/groups/{group_name}", response_model=GroupDetailResponse)
async def management_update_group(
    group_name: str,
    payload: GroupUpdateRequest,
    user_context: Annotated[dict, Depends(nginx_proxied_auth)] = None,
):
    """
    Update a group's properties and scope configuration (admin only).

    This updates the group in both:
    1. The configured identity provider (Keycloak or Entra ID)
    2. MongoDB scopes collection for authorization
    """
    _require_admin(user_context)

    iam = get_iam_manager()

    try:
        # Step 1: Get existing group data to preserve group_mappings if not provided
        existing_group = await scope_service.get_group(group_name)
        if not existing_group:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Group '{group_name}' not found",
            )

        # Step 2: Update group in identity provider (description only)
        if payload.description is not None:
            await iam.update_group(
                group_name=group_name,
                description=payload.description,
            )

        # Step 3: Update in MongoDB scopes collection
        # Extract server_access, ui_permissions, and agent_access from scope_config
        server_access = None
        ui_permissions = None
        group_mappings = None
        agent_access = None

        if payload.scope_config:
            server_access = payload.scope_config.get("server_access")
            ui_permissions = payload.scope_config.get("ui_permissions")
            group_mappings = payload.scope_config.get("group_mappings")
            agent_access = payload.scope_config.get("agent_access")

        # Preserve existing group_mappings if not provided in payload
        # This is critical for Entra ID where group_mappings contains Object IDs
        if group_mappings is None:
            group_mappings = existing_group.get("group_mappings", [group_name])

        # Preserve existing agent_access if not provided in payload
        if agent_access is None:
            agent_access = existing_group.get("agent_access", [])

        # Normalize agent paths to ensure they have leading slashes
        agent_access, ui_permissions = _normalize_agent_paths_in_scope_config(
            agent_access, ui_permissions
        )

        # Use import_group to update the scope data
        import_success = await scope_service.import_group(
            scope_name=group_name,
            description=payload.description or "",
            server_access=server_access,
            group_mappings=group_mappings,
            ui_permissions=ui_permissions,
            agent_access=agent_access,
        )

        if not import_success:
            logger.warning(
                "Group updated in IdP but failed to update in MongoDB: %s",
                group_name,
            )

        # Step 3: Fetch and return updated group details
        group_data = await scope_service.get_group(group_name)

        if not group_data:
            # Fall back to basic response if scope data not available
            return GroupDetailResponse(
                id="",
                name=group_name,
                description=payload.description,
            )

        return GroupDetailResponse(
            id=group_data.get("id", ""),
            name=group_data.get("name", group_name),
            path=group_data.get("path"),
            description=group_data.get("description"),
            server_access=group_data.get("server_access"),
            group_mappings=group_data.get("group_mappings"),
            ui_permissions=group_data.get("ui_permissions"),
            agent_access=group_data.get("agent_access"),
        )

    except Exception as exc:
        logger.error("Failed to update group: %s", exc)
        detail = str(exc).lower()

        if "not found" in detail:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Group '{group_name}' not found",
            ) from exc

        raise _translate_iam_error(exc) from exc
