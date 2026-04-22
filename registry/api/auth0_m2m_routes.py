"""API routes for Auth0 M2M client management.

This module provides endpoints for syncing Auth0 M2M applications to MongoDB
and managing their group mappings.
"""

import logging
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from registry.auth.dependencies import nginx_proxied_auth
from registry.repositories.documentdb.client import get_documentdb_client
from registry.schemas.idp_m2m_client import (
    IdPM2MClient,
    IdPM2MClientUpdate,
)
from registry.services.auth0_m2m_sync import get_auth0_m2m_sync

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s,p%(process)s,{%(filename)s:%(lineno)d},%(levelname)s,%(message)s",
)

logger = logging.getLogger(__name__)

router = APIRouter()


class Auth0SyncRequest(BaseModel):
    """Request payload for Auth0 M2M sync."""

    force_full_sync: bool = False


class Auth0SyncResponse(BaseModel):
    """Response from Auth0 M2M sync operation."""

    synced_count: int
    added_count: int
    updated_count: int
    removed_count: int
    errors: list[str]


def _require_admin(user_context: dict | None) -> None:
    """Check if user is admin.

    Args:
        user_context: User context from authentication

    Raises:
        HTTPException: If user is not admin
    """
    if not user_context:
        raise HTTPException(status_code=401, detail="Not authenticated")

    groups = user_context.get("groups", [])
    if "registry-admins" not in groups:
        raise HTTPException(
            status_code=403,
            detail="Admin access required",
        )


@router.post("/iam/auth0/m2m/sync", response_model=Auth0SyncResponse)
async def sync_auth0_m2m_clients(
    request: Auth0SyncRequest = Auth0SyncRequest(),
    user_context: Annotated[dict, Depends(nginx_proxied_auth)] = None,
):
    """Sync M2M clients from Auth0 to MongoDB (admin only).

    This endpoint queries the Auth0 Management API to fetch all M2M applications
    and stores/updates their information in MongoDB for authorization decisions.

    Args:
        request: Sync request parameters
        user_context: Authenticated user context


    Returns:
        Sync statistics including number of clients added/updated

    Raises:
        HTTPException: If user is not admin or sync fails
    """
    _require_admin(user_context)

    db = await get_documentdb_client()
    auth0_sync = get_auth0_m2m_sync(db)
    if not auth0_sync:
        raise HTTPException(
            status_code=503,
            detail="Auth0 sync not configured (missing AUTH0_DOMAIN, AUTH0_M2M_CLIENT_ID, or AUTH0_M2M_CLIENT_SECRET)",
        )

    try:
        result = await auth0_sync.sync_from_auth0(force_full_sync=request.force_full_sync)
        return Auth0SyncResponse(**result)

    except Exception as e:
        logger.exception(f"Failed to sync Auth0 M2M clients: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Sync failed: {str(e)}",
        )


@router.get("/iam/auth0/m2m/clients", response_model=list[IdPM2MClient])
async def list_auth0_m2m_clients(
    user_context: Annotated[dict, Depends(nginx_proxied_auth)] = None,
):
    """List all Auth0 M2M clients from MongoDB.

    Returns all M2M service accounts synced from Auth0, including their
    client IDs and group mappings.

    Args:
        user_context: Authenticated user context


    Returns:
        List of Auth0 M2M clients

    Raises:
        HTTPException: If user is not authenticated
    """
    if not user_context:
        raise HTTPException(status_code=401, detail="Not authenticated")

    db = await get_documentdb_client()
    auth0_sync = get_auth0_m2m_sync(db)
    if not auth0_sync:
        # Return empty list if Auth0 not configured
        return []

    try:
        clients = await auth0_sync.get_all_clients()
        return clients

    except Exception as e:
        logger.exception(f"Failed to list Auth0 M2M clients: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve clients: {str(e)}",
        )


@router.get("/iam/auth0/m2m/clients/{client_id}/groups", response_model=list[str])
async def get_client_groups(
    client_id: str,
    user_context: Annotated[dict, Depends(nginx_proxied_auth)] = None,
):
    """Get groups for a specific Auth0 M2M client.

    Args:
        client_id: Auth0 client ID
        user_context: Authenticated user context


    Returns:
        List of group names

    Raises:
        HTTPException: If user is not authenticated or client not found
    """
    if not user_context:
        raise HTTPException(status_code=401, detail="Not authenticated")

    db = await get_documentdb_client()
    auth0_sync = get_auth0_m2m_sync(db)
    if not auth0_sync:
        return []

    try:
        groups = await auth0_sync.get_client_groups(client_id)
        return groups

    except Exception as e:
        logger.exception(f"Failed to get groups for client {client_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve groups: {str(e)}",
        )


@router.patch("/iam/auth0/m2m/clients/{client_id}/groups")
async def update_client_groups(
    client_id: str,
    payload: IdPM2MClientUpdate,
    user_context: Annotated[dict, Depends(nginx_proxied_auth)] = None,
):
    """Update groups for an Auth0 M2M client (admin only).

    This allows administrators to change which groups a service account belongs to.

    Args:
        client_id: Auth0 client ID
        payload: Update payload with new groups
        user_context: Authenticated user context


    Returns:
        Success message

    Raises:
        HTTPException: If user is not admin or update fails
    """
    _require_admin(user_context)

    db = await get_documentdb_client()
    auth0_sync = get_auth0_m2m_sync(db)
    if not auth0_sync:
        raise HTTPException(
            status_code=503,
            detail="Auth0 sync not configured",
        )

    try:
        success = await auth0_sync.update_client_groups(client_id, payload.groups)

        if not success:
            raise HTTPException(
                status_code=404,
                detail=f"Client {client_id} not found",
            )

        return {
            "client_id": client_id,
            "groups": payload.groups,
            "message": "Groups updated successfully",
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Failed to update groups for client {client_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to update groups: {str(e)}",
        )
