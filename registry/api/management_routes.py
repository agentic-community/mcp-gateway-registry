from __future__ import annotations

import logging
import os
import re
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request, status

from ..audit.context import set_audit_action
from ..auth.dependencies import nginx_proxied_auth
from ..core.metrics import M2M_ORPHAN_CLEANUPS_TOTAL
from ..repositories.documentdb.client import get_documentdb_client
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
from ..utils.iam_errors import IdPForbiddenError, IdPNotFoundError
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
        db = await get_documentdb_client()
        collection = db["idp_m2m_clients"]

        # Query M2M clients from MongoDB
        cursor = collection.find({})
        m2m_docs = await cursor.to_list(length=None)

        # Deduplicate: skip MongoDB entries whose client_id already appears
        # in the IdP results (e.g. Keycloak already lists M2M service accounts)
        existing_usernames = {u.get("username", "").lower() for u in raw_users}

        # Add only M2M clients that are NOT already in the IdP listing
        added = 0
        for doc in m2m_docs:
            client_id = doc.get("client_id", "")
            name = doc.get("name", client_id)
            if name.lower() in existing_usernames or client_id.lower() in existing_usernames:
                continue
            raw_users.append(
                {
                    "id": client_id,
                    "username": name,
                    "email": f"{client_id}@service-account.local",
                    "firstName": None,
                    "lastName": None,
                    "enabled": doc.get("enabled", True),
                    "groups": doc.get("groups", []),
                }
            )
            added += 1

        logger.debug(f"[LIST_USERS] Added {added} M2M clients from MongoDB (skipped {len(m2m_docs) - added} duplicates)")
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
    request: Request,
    user_context: Annotated[dict, Depends(nginx_proxied_auth)] = None,
):
    """Delete a user by username (admin only)."""
    _require_admin(user_context)

    iam = get_iam_manager()

    idp_deleted = False
    try:
        await iam.delete_user(username=username)
        idp_deleted = True
    except Exception as exc:
        # Check if the error is "not found" — the user might only exist in MongoDB
        if "not found" not in str(exc).lower():
            raise _translate_iam_error(exc) from exc

    # Also remove from MongoDB idp_m2m_clients (handles orphaned records
    # that exist in MongoDB but not in the IdP). Case-insensitive regex match
    # keeps parity with the case-insensitive dedup used by the list handler.
    mongo_deleted = False
    try:
        db = await get_documentdb_client()
        collection = db["idp_m2m_clients"]
        username_ci = re.compile(f"^{re.escape(username)}$", re.IGNORECASE)
        result = await collection.delete_one(
            {"$or": [{"client_id": username_ci}, {"name": username_ci}]}
        )
        if result.deleted_count > 0:
            mongo_deleted = True
            logger.info(f"Removed M2M client '{username}' from MongoDB idp_m2m_clients")
    except Exception as e:
        logger.warning(
            f"Failed to remove M2M client '{username}' from MongoDB "
            f"(idp_deleted={idp_deleted}): {e}"
        )

    if not idp_deleted and not mongo_deleted:
        raise HTTPException(
            status_code=400,
            detail=f"User or M2M account '{username}' not found",
        )

    if mongo_deleted:
        M2M_ORPHAN_CLEANUPS_TOTAL.labels(idp_had_record=str(idp_deleted).lower()).inc()
        if not idp_deleted:
            set_audit_action(
                request,
                operation="delete",
                resource_type="user",
                resource_id=username,
                description=(
                    f"M2M orphan cleanup: removed '{username}' from MongoDB "
                    f"idp_m2m_clients (not present in IdP)"
                ),
                idp_skip_reason="not_found",
            )

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

    # If not an M2M account, update through IdP. The IdP manager computes the
    # add/remove diff against the user's current memberships, so groups that
    # are unchanged (including local-only scopes that happen to be in the
    # payload) never touch the IdP. Only *new* additions hit the IdP, and if
    # one of those doesn't exist there we surface a 400 with a helpful message
    # pointing the operator at the correct "map an existing IdP group via
    # group_mappings" workflow (see issue #946).
    iam = get_iam_manager()

    try:
        result = await iam.update_user_groups(
            username=username,
            groups=payload.groups,
        )
    except Exception as exc:
        detail = str(exc).lower()
        if "group" in detail and "not found" in detail:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=(
                    f"Cannot assign user '{username}' to a group that does "
                    "not exist in the identity provider. If this is a "
                    "local-only group (created with 'Create in identity "
                    "provider' unchecked), add the user to the IdP group "
                    "referenced in the scope's group_mappings instead. "
                    f"Underlying error: {exc}"
                ),
            ) from exc
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
    """List IAM groups from the configured identity provider and local-only
    scope documents (admin only).

    Local-only groups (created with `create_in_idp=False`, see issue #946) live
    only in the MongoDB `scopes` collection and are invisible to the IdP.
    This handler merges the IdP list with any MongoDB scope documents that
    don't appear in the IdP response so every UI picker sees them.
    """
    _require_admin(user_context)

    iam = get_iam_manager()

    try:
        raw_groups = await iam.list_groups()
    except Exception as exc:
        logger.error("Failed to list IAM groups: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Unable to list IAM groups",
        ) from exc

    # Cross-reference with MongoDB scope documents.
    # scope_service.list_groups() also eagerly backfills any legacy docs
    # missing the is_idp_managed flag (see DocumentDBScopeRepository.list_groups).
    scope_groups_map: dict = {}
    try:
        scope_groups = await scope_service.list_groups()
        for name, meta in scope_groups.items():
            if isinstance(meta, dict):
                scope_groups_map[name] = meta
    except Exception as exc:
        logger.warning(
            "Could not cross-reference scope docs for is_idp_managed: %s",
            exc,
        )

    summaries: list[GroupSummary] = []
    idp_names: set[str] = set()
    for group in raw_groups:
        name = group.get("name", "")
        idp_names.add(name)
        scope_meta = scope_groups_map.get(name, {})
        summaries.append(
            GroupSummary(
                id=group.get("id", ""),
                name=name,
                path=group.get("path", ""),
                attributes=group.get("attributes"),
                is_idp_managed=scope_meta.get("is_idp_managed"),
            )
        )

    # Append local-only groups (present in MongoDB scopes, absent from IdP).
    # Synthesize id/path using the scope name, mirroring the create path.
    local_only: list[GroupSummary] = []
    for name, scope_meta in scope_groups_map.items():
        if name in idp_names:
            continue
        description = scope_meta.get("description") if isinstance(scope_meta, dict) else None
        local_only.append(
            GroupSummary(
                id=name,
                name=name,
                path=f"/{name}",
                attributes={"description": [description]} if description else None,
                is_idp_managed=scope_meta.get("is_idp_managed", False),
            )
        )
        logger.debug(
            "iam_groups_list_local_only_included scope=%s is_idp_managed=%s",
            name,
            scope_meta.get("is_idp_managed", False),
        )

    local_only.sort(key=lambda summary: summary.name)
    summaries.extend(local_only)

    return GroupListResponse(groups=summaries, total=len(summaries))


@router.post("/iam/groups", response_model=GroupSummary)
async def management_create_group(
    payload: GroupCreateRequest,
    request: Request,
    user_context: Annotated[dict, Depends(nginx_proxied_auth)] = None,
):
    """
    Create a new group in the identity provider and/or MongoDB (admin only).

    When create_in_idp is True, creates in both the configured
    identity provider and MongoDB scopes collection.
    When create_in_idp is False, creates only in MongoDB scopes collection
    and persists `is_idp_managed=False` so subsequent PATCH/DELETE do not
    call the IdP (see issue #946).
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

    set_audit_action(
        request,
        operation="create",
        resource_type="group",
        resource_id=payload.name,
        description=f"Create group '{payload.name}'",
        idp_skip_reason=None if create_in_idp else "local_only",
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
            is_idp_managed=create_in_idp,
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
            is_idp_managed=create_in_idp,
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
    request: Request,
    user_context: Annotated[dict, Depends(nginx_proxied_auth)] = None,
):
    """
    Delete a group (admin only).

    Behavior (see issue #946):
    - is_idp_managed=False: IdP call is skipped entirely.
    - is_idp_managed=True and IdP 403/404: non-fatal fall-through; MongoDB
      deletion proceeds; audit entry includes idp_skip_reason.
    - Any other IdP error: propagates as before (502).
    """
    _require_admin(user_context)

    set_audit_action(
        request,
        operation="delete",
        resource_type="group",
        resource_id=group_name,
        description=f"Delete group '{group_name}'",
    )

    iam = get_iam_manager()
    skip_reason: str | None = None

    try:
        existing_group = await scope_service.get_group(group_name)
        is_idp_managed = True
        if existing_group is not None:
            is_idp_managed = bool(existing_group.get("is_idp_managed", True))

        if is_idp_managed:
            try:
                await iam.delete_group(group_name=group_name)
            except IdPNotFoundError as idp_exc:
                logger.info(
                    "iam_idp_fallthrough operation=delete resource=%s "
                    "status=404 reason=not_found detail=%s",
                    group_name,
                    idp_exc,
                )
                skip_reason = "not_found"
            except IdPForbiddenError as idp_exc:
                logger.warning(
                    "iam_idp_fallthrough operation=delete resource=%s "
                    "status=403 reason=forbidden detail=%s",
                    group_name,
                    idp_exc,
                )
                skip_reason = "forbidden"
        else:
            logger.info(
                "iam_idp_skipped operation=delete resource=%s reason=local_only",
                group_name,
            )
            skip_reason = "local_only"

        if skip_reason is not None:
            set_audit_action(
                request,
                operation="delete",
                resource_type="group",
                resource_id=group_name,
                description=(
                    f"Delete group '{group_name}' - IdP call skipped "
                    f"({skip_reason})"
                ),
                idp_skip_reason=skip_reason,
            )

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

    except HTTPException:
        raise
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
            is_idp_managed=group_data.get("is_idp_managed", True),
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
    request: Request,
    user_context: Annotated[dict, Depends(nginx_proxied_auth)] = None,
):
    """
    Update a group's properties and scope configuration (admin only).

    Behavior (see issue #946):
    - is_idp_managed=False: IdP call is skipped entirely. MongoDB update
      proceeds. The flag is preserved across edits.
    - is_idp_managed=True and IdP 403/404: non-fatal fall-through; MongoDB
      update proceeds; audit entry includes idp_skip_reason.
    """
    _require_admin(user_context)

    set_audit_action(
        request,
        operation="update",
        resource_type="group",
        resource_id=group_name,
        description=f"Update group '{group_name}'",
    )

    iam = get_iam_manager()
    skip_reason: str | None = None

    try:
        # Step 1: Get existing group data.
        # This also lazily backfills is_idp_managed for legacy documents.
        existing_group = await scope_service.get_group(group_name)
        if not existing_group:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Group '{group_name}' not found",
            )

        is_idp_managed = bool(existing_group.get("is_idp_managed", True))

        # Step 2: Update group in identity provider (description only),
        # gated on is_idp_managed. Catch typed IdP errors as non-fatal.
        if payload.description is not None and is_idp_managed:
            try:
                await iam.update_group(
                    group_name=group_name,
                    description=payload.description,
                )
            except IdPNotFoundError as idp_exc:
                logger.info(
                    "iam_idp_fallthrough operation=update resource=%s "
                    "status=404 reason=not_found detail=%s",
                    group_name,
                    idp_exc,
                )
                skip_reason = "not_found"
            except IdPForbiddenError as idp_exc:
                logger.warning(
                    "iam_idp_fallthrough operation=update resource=%s "
                    "status=403 reason=forbidden detail=%s",
                    group_name,
                    idp_exc,
                )
                skip_reason = "forbidden"
        elif payload.description is not None:
            logger.info(
                "iam_idp_skipped operation=update resource=%s reason=local_only",
                group_name,
            )
            skip_reason = "local_only"

        if skip_reason is not None:
            set_audit_action(
                request,
                operation="update",
                resource_type="group",
                resource_id=group_name,
                description=(
                    f"Update group '{group_name}' - IdP call skipped "
                    f"({skip_reason})"
                ),
                idp_skip_reason=skip_reason,
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

        # Use import_group to update the scope data.
        # Preserve is_idp_managed across edits (import_group replaces the whole doc).
        import_success = await scope_service.import_group(
            scope_name=group_name,
            description=payload.description
            if payload.description is not None
            else existing_group.get("description", ""),
            server_access=server_access,
            group_mappings=group_mappings,
            ui_permissions=ui_permissions,
            agent_access=agent_access,
            is_idp_managed=is_idp_managed,
        )

        if not import_success:
            logger.warning(
                "Group updated in IdP but failed to update in MongoDB: %s",
                group_name,
            )

        # Step 4: Fetch and return updated group details
        group_data = await scope_service.get_group(group_name)

        if not group_data:
            # Fall back to basic response if scope data not available
            return GroupDetailResponse(
                id="",
                name=group_name,
                description=payload.description,
                is_idp_managed=is_idp_managed,
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
            is_idp_managed=group_data.get("is_idp_managed", True),
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Failed to update group: %s", exc)
        detail = str(exc).lower()

        if "not found" in detail:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Group '{group_name}' not found",
            ) from exc

        raise _translate_iam_error(exc) from exc
