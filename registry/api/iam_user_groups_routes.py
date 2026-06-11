"""Direct CRUD endpoints for user-to-group fallback registration.

These endpoints write directly to the ``idp_user_groups`` MongoDB collection
without calling any IdP Admin API. Operators can register usernames and their
group mappings so the auth server can enrich user tokens during authorization
when the JWT's groups claim is empty for a fallback-enabled IdP (e.g.
PingFederate today).

This module is the user-side mirror of
:mod:`registry.api.m2m_management_routes`.

Tracked by issue #1127.
"""

import logging
import os
from typing import Annotated

import httpx
from fastapi import APIRouter, Depends, HTTPException, Query, Request, status

from registry.audit.context import set_audit_action
from registry.auth.dependencies import nginx_proxied_auth
from registry.core.config import settings
from registry.schemas.idp_user_group import (
    IdPUserGroup,
    IdPUserGroupCreate,
    IdPUserGroupPatch,
    PingFederateUserCreateRequest,
    UserGroupListResponse,
)
from registry.services.user_group_management_service import (
    UserGroupConflict,
    UserGroupManagementService,
    UserGroupNotFound,
    get_user_group_management_service,
)

logger = logging.getLogger(__name__)


router = APIRouter(prefix="/api/iam/user-groups", tags=["IAM User Groups"])

_RESOURCE_TYPE: str = "idp_user_group"
_LIST_DEFAULT_LIMIT: int = 500
_LIST_MAX_LIMIT: int = 1000

# PingFederate admin API connection (in-container defaults). Operators can
# override via environment variables when the registry runs outside Docker.
_PF_ADMIN_URL = os.environ.get("PF_ADMIN_URL", "https://pingfederate:9999")
_PF_ADMIN_USER = os.environ.get("PF_ADMIN_USER", "administrator")
_PF_ADMIN_PASS = os.environ.get("PF_ADMIN_PASS", "2FederateM0re")
_PF_USERS_PATH = "/pf-admin-api/v1/passwordCredentialValidators/simple"
_PF_HTTP_TIMEOUT_SECONDS = 10.0
_PF_USERS_TABLE_NAME = "Users"
_PF_USERNAME_FIELD = "Username"
_PF_PASSWORD_FIELD = "Password"
_PF_CONFIRM_PASSWORD_FIELD = "Confirm Password"
_PF_RELAX_PASSWORD_FIELD = "Relax Password Requirements"


def _require_admin(
    user_context: dict | None,
) -> None:
    """Enforce admin permission or raise 401/403."""
    if not user_context:
        raise HTTPException(status_code=401, detail="Not authenticated")
    if not user_context.get("is_admin"):
        raise HTTPException(status_code=403, detail="Administrator permissions are required")


async def _get_service() -> UserGroupManagementService:
    return await get_user_group_management_service()


def _check_pingfederate_enabled() -> None:
    """Ensure the active deployment supports PingFederate user management.

    Reject with 400 if the auth provider is not pingfederate, or if pingfederate
    is not present in the IDP_USER_GROUP_FALLBACK_ENABLED_PROVIDERS list.
    """
    auth_provider = (settings.auth_provider or "").lower()
    if auth_provider != "pingfederate":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="PingFederate user creation is only available when AUTH_PROVIDER=pingfederate",
        )

    fallback_providers = [
        p.lower() for p in settings.idp_user_group_fallback_enabled_providers
    ]
    if "pingfederate" not in fallback_providers:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                "PingFederate user creation requires 'pingfederate' in "
                "IDP_USER_GROUP_FALLBACK_ENABLED_PROVIDERS"
            ),
        )


def _find_users_table(
    pcv_doc: dict,
) -> dict:
    """Locate the Users table within the Simple PCV configuration document.

    Raises:
        HTTPException(502): If the document's shape is unexpected (no
            configuration.tables, or no table named "Users"). We don't surface
            PF's internal structure to the client.
    """
    try:
        tables = pcv_doc["configuration"]["tables"]
    except (KeyError, TypeError):
        logger.error("PingFederate Simple PCV response missing configuration.tables")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="PingFederate admin API error: unexpected response shape",
        )

    for table in tables:
        if table.get("name") == _PF_USERS_TABLE_NAME:
            return table

    logger.error("PingFederate Simple PCV response has no Users table")
    raise HTTPException(
        status_code=status.HTTP_502_BAD_GATEWAY,
        detail="PingFederate admin API error: Users table not found",
    )


def _build_user_row(
    username: str,
    password: str,
) -> dict:
    """Build a Simple PCV row dict for the given username/password."""
    return {
        "fields": [
            {"name": _PF_USERNAME_FIELD, "value": username},
            {"name": _PF_PASSWORD_FIELD, "value": password},
            {"name": _PF_CONFIRM_PASSWORD_FIELD, "value": password},
            {"name": _PF_RELAX_PASSWORD_FIELD, "value": "true"},
        ]
    }


def _upsert_user_row(
    users_table: dict,
    username: str,
    password: str,
) -> str:
    """Insert or replace the user row inside the Users table.

    Returns:
        "created" if the user was appended, "updated" if an existing row's
        password fields were rewritten.
    """
    rows = users_table.setdefault("rows", [])
    for row in rows:
        for field in row.get("fields", []):
            if field.get("name") == _PF_USERNAME_FIELD and field.get("value") == username:
                # Replace password and confirm-password fields in-place so we
                # don't disturb any other fields PF may have added.
                for f in row["fields"]:
                    if f.get("name") in (_PF_PASSWORD_FIELD, _PF_CONFIRM_PASSWORD_FIELD):
                        f["value"] = password
                    elif f.get("name") == _PF_RELAX_PASSWORD_FIELD:
                        f["value"] = "true"
                return "updated"

    rows.append(_build_user_row(username, password))
    return "created"


async def _pingfederate_upsert_user(
    username: str,
    password: str,
) -> str:
    """Create or update a user inside PingFederate's Simple PCV.

    Mirrors Step 5 of pingfederate/setup/init-pingfederate.sh: GET the Simple
    PCV configuration, mutate its Users table, PUT it back.

    Returns:
        "created" or "updated" depending on whether the username pre-existed.

    Raises:
        HTTPException(502): On any failure to talk to PingFederate. Errors are
            logged with type+status server-side; the response body is never
            echoed to the client (it could leak admin credentials).
    """
    headers = {
        "X-XSRF-Header": "PingFederate",
        "Content-Type": "application/json",
    }
    auth = (_PF_ADMIN_USER, _PF_ADMIN_PASS)
    url = f"{_PF_ADMIN_URL}{_PF_USERS_PATH}"

    try:
        async with httpx.AsyncClient(verify=False, timeout=_PF_HTTP_TIMEOUT_SECONDS) as client:  # nosec B501 - PF admin API uses self-signed cert in default deployment
            get_resp = await client.get(url, auth=auth, headers=headers)
            if get_resp.status_code >= 300:
                logger.error(
                    "PingFederate GET %s returned status=%s",
                    _PF_USERS_PATH,
                    get_resp.status_code,
                )
                raise HTTPException(
                    status_code=status.HTTP_502_BAD_GATEWAY,
                    detail="PingFederate admin API error: GET failed",
                )

            pcv_doc = get_resp.json()
            users_table = _find_users_table(pcv_doc)
            outcome = _upsert_user_row(users_table, username, password)

            put_resp = await client.put(
                url,
                auth=auth,
                headers=headers,
                json=pcv_doc,
            )
            if put_resp.status_code >= 300:
                logger.error(
                    "PingFederate PUT %s returned status=%s",
                    _PF_USERS_PATH,
                    put_resp.status_code,
                )
                raise HTTPException(
                    status_code=status.HTTP_502_BAD_GATEWAY,
                    detail="PingFederate admin API error: PUT failed",
                )

            return outcome

    except HTTPException:
        raise
    except httpx.HTTPError as e:
        logger.error(
            "PingFederate admin API connection error type=%s",
            type(e).__name__,
        )
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"PingFederate admin API error: {type(e).__name__}",
        )
    except Exception as e:
        logger.exception("Unexpected error talking to PingFederate admin API")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"PingFederate admin API error: {type(e).__name__}",
        )


@router.post(
    "",
    response_model=IdPUserGroup,
    status_code=status.HTTP_201_CREATED,
)
async def create_user_group(
    payload: IdPUserGroupCreate,
    request: Request,
    user_context: Annotated[dict | None, Depends(nginx_proxied_auth)] = None,
) -> IdPUserGroup:
    """Register a new user-group fallback record (admin only)."""
    _require_admin(user_context)
    service = await _get_service()
    created_by = user_context.get("username") if user_context else None
    try:
        result = await service.register_user_group(payload, created_by=created_by)
    except UserGroupConflict:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"User-group record for {payload.username} already exists",
        )
    set_audit_action(
        request,
        "create",
        _RESOURCE_TYPE,
        resource_id=payload.username,
        description=f"Created user-group record for {payload.username}",
    )
    return result


@router.get("", response_model=UserGroupListResponse)
async def list_user_groups(
    provider: str | None = None,
    q: str | None = Query(default=None, description="Substring filter on username/email"),
    limit: int = Query(default=_LIST_DEFAULT_LIMIT, ge=1, le=_LIST_MAX_LIMIT),
    skip: int = Query(default=0, ge=0),
    user_context: Annotated[dict | None, Depends(nginx_proxied_auth)] = None,
) -> UserGroupListResponse:
    """List user-group fallback records (any authenticated user)."""
    if not user_context:
        raise HTTPException(status_code=401, detail="Not authenticated")
    service = await _get_service()
    items, total = await service.list_user_groups(
        skip=skip,
        limit=limit,
        provider=provider,
        q=q,
    )
    return UserGroupListResponse(total=total, limit=limit, skip=skip, items=items)


@router.get("/{username}", response_model=IdPUserGroup)
async def get_user_group(
    username: str,
    user_context: Annotated[dict | None, Depends(nginx_proxied_auth)] = None,
) -> IdPUserGroup:
    """Get a specific user-group record (any authenticated user)."""
    if not user_context:
        raise HTTPException(status_code=401, detail="Not authenticated")
    service = await _get_service()
    try:
        result = await service.get_user_group(username)
    except UserGroupNotFound:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"User-group record for {username} not found",
        )
    return result


@router.patch("/{username}", response_model=IdPUserGroup)
async def patch_user_group(
    username: str,
    payload: IdPUserGroupPatch,
    request: Request,
    user_context: Annotated[dict | None, Depends(nginx_proxied_auth)] = None,
) -> IdPUserGroup:
    """Update fields of an existing user-group record (admin only)."""
    _require_admin(user_context)
    service = await _get_service()
    try:
        result = await service.update_user_group(username, payload)
    except UserGroupNotFound:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"User-group record for {username} not found",
        )
    set_audit_action(
        request,
        "update",
        _RESOURCE_TYPE,
        resource_id=username,
        description=f"Updated user-group record for {username}",
    )
    return result


@router.delete("/{username}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_user_group(
    username: str,
    request: Request,
    user_context: Annotated[dict | None, Depends(nginx_proxied_auth)] = None,
) -> None:
    """Delete a user-group fallback record (admin only)."""
    _require_admin(user_context)
    service = await _get_service()
    try:
        await service.delete_user_group(username)
    except UserGroupNotFound:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"User-group record for {username} not found",
        )
    set_audit_action(
        request,
        "delete",
        _RESOURCE_TYPE,
        resource_id=username,
        description=f"Deleted user-group record for {username}",
    )


@router.post(
    "/{username}/pingfederate-user",
    status_code=status.HTTP_200_OK,
)
async def create_pingfederate_user(
    username: str,
    payload: PingFederateUserCreateRequest,
    request: Request,
    user_context: Annotated[dict | None, Depends(nginx_proxied_auth)] = None,
) -> dict:
    """Create or update a user inside PingFederate's Simple PCV (admin only).

    The user-group fallback record for this username MUST already exist in the
    registry (the UI flow creates it first via POST /api/iam/user-groups, then
    calls this endpoint when the admin opts in to "Also create in PingFederate").
    The password is sent straight through to PingFederate and is not stored in
    the registry.
    """
    _require_admin(user_context)
    _check_pingfederate_enabled()

    # Confirm the user-group record exists; refuse to create the PF user
    # without one so the auth-server fallback enrichment has groups to attach.
    service = await _get_service()
    try:
        await service.get_user_group(username)
    except UserGroupNotFound:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=(
                f"User-group record for {username} not found. "
                "Create the registry user-group record before creating the PingFederate user."
            ),
        )

    outcome = await _pingfederate_upsert_user(username, payload.password)

    set_audit_action(
        request,
        "create" if outcome == "created" else "update",
        _RESOURCE_TYPE,
        resource_id=username,
        description=f"PingFederate user {outcome} for {username}",
    )

    logger.info(
        "PingFederate Simple PCV user %s for username=%s",
        outcome,
        username,
    )
    return {"username": username, "created_or_updated": outcome}
