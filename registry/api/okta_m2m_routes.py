"""API routes for Okta M2M client management.

This module provides endpoints for syncing Okta M2M applications to MongoDB
and managing their group mappings.
"""

import logging
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException
from registry.core.auth import nginx_proxied_auth
from registry.core.database import get_db
from registry.models.okta_m2m_client import (
    OktaM2MClient,
    OktaM2MClientUpdate,
    OktaSyncRequest,
    OktaSyncResponse,
)

from registry.services.okta_m2m_sync import get_okta_m2m_sync

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s,p%(process)s,{%(filename)s:%(lineno)d},%(levelname)s,%(message)s",
)

logger = logging.getLogger(__name__)

router = APIRouter()


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


@router.post("/iam/okta/m2m/sync", response_model=OktaSyncResponse)
async def sync_okta_m2m_clients(
    request: OktaSyncRequest = OktaSyncRequest(),
    user_context: Annotated[dict, Depends(nginx_proxied_auth)] = None,
    db=Depends(get_db),
):
    """Sync M2M clients from Okta to MongoDB (admin only).

    This endpoint queries the Okta Admin API to fetch all M2M service applications
    and stores/updates their information in MongoDB for authorization decisions.

    Args:
        request: Sync request parameters
        user_context: Authenticated user context
        db: Database connection

    Returns:
        Sync statistics including number of clients added/updated

    Raises:
        HTTPException: If user is not admin or sync fails
    """
    _require_admin(user_context)

    okta_sync = get_okta_m2m_sync(db)
    if not okta_sync:
        raise HTTPException(
            status_code=503,
            detail="Okta sync not configured (missing OKTA_DOMAIN or OKTA_API_TOKEN)",
        )

    try:
        result = await okta_sync.sync_from_okta(force_full_sync=request.force_full_sync)
        return OktaSyncResponse(**result)

    except Exception as e:
        logger.exception(f"Failed to sync Okta M2M clients: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Sync failed: {str(e)}",
        )


@router.get("/iam/okta/m2m/clients", response_model=list[OktaM2MClient])
async def list_okta_m2m_clients(
    user_context: Annotated[dict, Depends(nginx_proxied_auth)] = None,
    db=Depends(get_db),
):
    """List all Okta M2M clients from MongoDB.

    Returns all M2M service accounts synced from Okta, including their
    client IDs and group mappings.

    Args:
        user_context: Authenticated user context
        db: Database connection

    Returns:
        List of Okta M2M clients

    Raises:
        HTTPException: If user is not authenticated
    """
    if not user_context:
        raise HTTPException(status_code=401, detail="Not authenticated")

    okta_sync = get_okta_m2m_sync(db)
    if not okta_sync:
        # Return empty list if Okta not configured
        return []

    try:
        clients = await okta_sync.get_all_clients()
        return clients

    except Exception as e:
        logger.exception(f"Failed to list Okta M2M clients: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve clients: {str(e)}",
        )


@router.get("/iam/okta/m2m/clients/{client_id}/groups", response_model=list[str])
async def get_client_groups(
    client_id: str,
    user_context: Annotated[dict, Depends(nginx_proxied_auth)] = None,
    db=Depends(get_db),
):
    """Get groups for a specific Okta M2M client.

    Args:
        client_id: Okta client ID
        user_context: Authenticated user context
        db: Database connection

    Returns:
        List of group names

    Raises:
        HTTPException: If user is not authenticated or client not found
    """
    if not user_context:
        raise HTTPException(status_code=401, detail="Not authenticated")

    okta_sync = get_okta_m2m_sync(db)
    if not okta_sync:
        return []

    try:
        groups = await okta_sync.get_client_groups(client_id)
        return groups

    except Exception as e:
        logger.exception(f"Failed to get groups for client {client_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve groups: {str(e)}",
        )


@router.patch("/iam/okta/m2m/clients/{client_id}/groups")
async def update_client_groups(
    client_id: str,
    payload: OktaM2MClientUpdate,
    user_context: Annotated[dict, Depends(nginx_proxied_auth)] = None,
    db=Depends(get_db),
):
    """Update groups for an Okta M2M client (admin only).

    This allows administrators to change which groups a service account belongs to
    without modifying the Okta authorization server expression.

    Args:
        client_id: Okta client ID
        payload: Update payload with new groups
        user_context: Authenticated user context
        db: Database connection

    Returns:
        Success message

    Raises:
        HTTPException: If user is not admin or update fails
    """
    _require_admin(user_context)

    okta_sync = get_okta_m2m_sync(db)
    if not okta_sync:
        raise HTTPException(
            status_code=503,
            detail="Okta sync not configured",
        )

    try:
        success = await okta_sync.update_client_groups(client_id, payload.groups)

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
