"""
Peer management API routes.

Provides REST endpoints for managing peer registry configurations
and triggering synchronization operations.

IMPORTANT: Route ordering matters in FastAPI. All specific-path routes
(e.g., /connections/all, /shared-resources, /sync) MUST be defined BEFORE
any parameterized routes (e.g., /{peer_id}) to prevent the catch-all
parameter from shadowing the specific paths.
"""

import logging
from datetime import datetime
from typing import Any

from fastapi import APIRouter, Body, Depends, HTTPException, Query, status

from ..auth.dependencies import nginx_proxied_auth
from ..schemas.peer_federation_schema import (
    PeerRegistryConfig,
    PeerSyncStatus,
    SyncResult,
)
from ..services.federation_audit_service import (
    FederationConnectionLog,
    PeerSyncSummary,
    get_federation_audit_service,
)
from ..services.peer_federation_service import get_peer_federation_service

logger = logging.getLogger(__name__)


def _check_peer_management_scope(
    user_context: dict[str, Any],
) -> None:
    """Check if user has permission to manage peers.

    Allows access for:
    - Admin users (network-trusted or mcp-registry-admin group)
    - Federation-static token users (have federation/peers scope)

    Args:
        user_context: User context from auth dependency

    Raises:
        HTTPException: 403 if user lacks peer management permission
    """
    # Admins always have access
    if user_context.get("is_admin", False):
        return

    # Check for federation/peers scope (federation static token)
    scopes = user_context.get("scopes", [])
    if "federation/peers" in scopes:
        return

    # Check for admin group
    groups = user_context.get("groups", [])
    if "mcp-registry-admin" in groups:
        return

    logger.warning(
        f"User {user_context.get('username')} attempted peer management "
        f"without required scope. Scopes: {scopes}, Groups: {groups}"
    )
    raise HTTPException(
        status_code=status.HTTP_403_FORBIDDEN,
        detail="Peer management requires admin privileges or federation/peers scope",
    )


router = APIRouter(
    prefix="/api/peers",
    tags=["peer-management"],
)


# ============================================================================
# SECTION 1: Routes with FIXED paths (no path parameters)
# These MUST come before any /{peer_id} routes to avoid shadowing.
# ============================================================================


@router.get("", response_model=list[PeerRegistryConfig])
async def list_peers(
    enabled: bool | None = None,
    user_context: dict = Depends(nginx_proxied_auth),
) -> list[PeerRegistryConfig]:
    """
    List all peer registries with optional filtering by enabled status.

    Args:
        enabled: If True, return only enabled peers. If False, return only disabled peers.
                If None, return all peers.
        user_context: Authenticated user context

    Returns:
        List of peer registry configurations

    Example:
        GET /api/peers
        GET /api/peers?enabled=true
    """
    _check_peer_management_scope(user_context)
    logger.info(f"User '{user_context.get('username')}' listing peers (enabled={enabled})")

    service = get_peer_federation_service()
    peers = await service.list_peers(enabled=enabled)

    logger.info(f"Returning {len(peers)} peer configs")
    return peers


@router.post("", response_model=PeerRegistryConfig, status_code=status.HTTP_201_CREATED)
async def create_peer(
    config: PeerRegistryConfig,
    user_context: dict = Depends(nginx_proxied_auth),
) -> PeerRegistryConfig:
    """
    Create a new peer registry configuration.

    Args:
        config: Peer registry configuration to create
        user_context: Authenticated user context

    Returns:
        Created peer registry configuration

    Raises:
        HTTPException: 409 if peer_id already exists
        HTTPException: 400 if validation fails

    Example:
        POST /api/peers
        {
            "peer_id": "central-registry",
            "name": "Central MCP Registry",
            "endpoint": "https://central.registry.company.com",
            "enabled": true,
            "sync_mode": "all",
            "sync_interval_minutes": 30
        }
    """
    _check_peer_management_scope(user_context)
    logger.info(f"User '{user_context.get('username')}' creating peer '{config.peer_id}'")

    service = get_peer_federation_service()

    try:
        created_peer = await service.add_peer(config)
        logger.info(f"Successfully created peer '{config.peer_id}'")
        return created_peer
    except ValueError as e:
        error_msg = str(e)
        if "already exists" in error_msg:
            logger.error(f"Peer ID already exists: {config.peer_id}")
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=error_msg,
            )
        else:
            logger.error(f"Invalid peer config: {error_msg}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=error_msg,
            )


@router.post("/sync", response_model=dict[str, SyncResult])
async def sync_all_peers(
    enabled_only: bool = Query(True, description="If True, only sync enabled peers"),
    user_context: dict = Depends(nginx_proxied_auth),
) -> dict[str, SyncResult]:
    """
    Trigger synchronization for all (or enabled) peers.

    Args:
        enabled_only: If True, only sync enabled peers (default: True)
        user_context: Authenticated user context

    Returns:
        Dictionary mapping peer_id to SyncResult

    Example:
        POST /api/peers/sync
        POST /api/peers/sync?enabled_only=false
    """
    _check_peer_management_scope(user_context)
    logger.info(
        f"User '{user_context.get('username')}' triggering sync for all peers "
        f"(enabled_only={enabled_only})"
    )

    service = get_peer_federation_service()

    results = await service.sync_all_peers(enabled_only=enabled_only)

    # Count successes and failures
    successful = sum(1 for r in results.values() if r.success)
    failed = len(results) - successful
    total_servers = sum(r.servers_synced for r in results.values())
    total_agents = sum(r.agents_synced for r in results.values())

    logger.info(
        f"Sync all completed: {successful} succeeded, {failed} failed. "
        f"Total: {total_servers} servers, {total_agents} agents"
    )

    return results


@router.get("/connections/all", response_model=list[FederationConnectionLog])
async def get_all_connections(
    since: datetime | None = Query(
        None, description="Only return connections after this timestamp"
    ),
    limit: int = Query(100, ge=1, le=1000, description="Maximum entries to return"),
    user_context: dict = Depends(nginx_proxied_auth),
) -> list[FederationConnectionLog]:
    """
    Get all federation connection history.

    Returns a list of all federation connections from all peers, useful for
    monitoring overall federation activity.

    Args:
        since: Only return connections after this timestamp
        limit: Maximum entries to return (default: 100, max: 1000)
        user_context: Authenticated user context

    Returns:
        List of all connection logs

    Example:
        GET /api/peers/connections/all
        GET /api/peers/connections/all?limit=50
    """
    _check_peer_management_scope(user_context)
    logger.info(f"User '{user_context.get('username')}' retrieving all federation connections")

    audit_service = get_federation_audit_service()
    connections = await audit_service.get_all_connections(
        since=since,
        limit=limit,
    )

    logger.info(f"Returning {len(connections)} total connection logs")
    return connections


@router.get("/shared-resources", response_model=dict[str, PeerSyncSummary])
async def get_shared_resources(
    user_context: dict = Depends(nginx_proxied_auth),
) -> dict[str, PeerSyncSummary]:
    """
    Get summary of resources shared with each peer.

    Returns statistics about what has been shared with each peer,
    including connection counts and resource totals.

    Args:
        user_context: Authenticated user context

    Returns:
        Dictionary mapping peer_id to PeerSyncSummary

    Example:
        GET /api/peers/shared-resources
    """
    _check_peer_management_scope(user_context)
    logger.info(f"User '{user_context.get('username')}' retrieving shared resources summary")

    audit_service = get_federation_audit_service()
    summaries = await audit_service.get_shared_resources_summary()

    logger.info(f"Returning shared resources summary for {len(summaries)} peers")
    return summaries


# ============================================================================
# SECTION 2: Routes with /{peer_id} path parameter
# These MUST come after all fixed-path routes above.
# ============================================================================


@router.get("/{peer_id}", response_model=PeerRegistryConfig)
async def get_peer(
    peer_id: str,
    user_context: dict = Depends(nginx_proxied_auth),
) -> PeerRegistryConfig:
    """
    Get a specific peer by ID.

    Args:
        peer_id: Peer identifier
        user_context: Authenticated user context

    Returns:
        Peer registry configuration

    Raises:
        HTTPException: 404 if peer not found

    Example:
        GET /api/peers/central-registry
    """
    _check_peer_management_scope(user_context)
    logger.info(f"User '{user_context.get('username')}' retrieving peer '{peer_id}'")

    service = get_peer_federation_service()

    try:
        peer = await service.get_peer(peer_id)
        return peer
    except ValueError as e:
        logger.error(f"Peer not found: {peer_id}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e),
        )


@router.put("/{peer_id}", response_model=PeerRegistryConfig)
async def update_peer(
    peer_id: str,
    updates: dict[str, Any] = Body(...),
    user_context: dict = Depends(nginx_proxied_auth),
) -> PeerRegistryConfig:
    """
    Update an existing peer configuration.

    Args:
        peer_id: Peer identifier
        updates: Dictionary of fields to update
        user_context: Authenticated user context

    Returns:
        Updated peer registry configuration

    Raises:
        HTTPException: 404 if peer not found
        HTTPException: 400 if validation fails

    Example:
        PUT /api/peers/central-registry
        {
            "enabled": false,
            "sync_interval_minutes": 60
        }
    """
    _check_peer_management_scope(user_context)
    logger.info(
        f"User '{user_context.get('username')}' updating peer '{peer_id}' with updates: {updates}"
    )

    service = get_peer_federation_service()

    try:
        updated_peer = await service.update_peer(peer_id, updates)
        logger.info(f"Successfully updated peer '{peer_id}'")
        return updated_peer
    except ValueError as e:
        error_msg = str(e)
        if "not found" in error_msg:
            logger.error(f"Peer not found: {peer_id}")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=error_msg,
            )
        else:
            logger.error(f"Invalid peer update: {error_msg}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=error_msg,
            )


@router.patch("/{peer_id}/token")
async def update_peer_token(
    peer_id: str,
    federation_token: str = Body(..., embed=True),
    user_context: dict = Depends(nginx_proxied_auth),
) -> dict[str, str]:
    """
    Update federation token for a peer without triggering full update.

    This endpoint specifically updates only the federation token, bypassing
    the regular update_peer() flow. This is useful for recovering from
    issue #561 where tokens were lost during peer updates, or for rotating
    tokens without modifying other peer configuration.

    Args:
        peer_id: Peer identifier
        federation_token: New federation token value
        user_context: Authenticated user context

    Returns:
        Success message with peer ID

    Raises:
        HTTPException: 404 if peer not found
        HTTPException: 400 if token update fails

    Example:
        PATCH /api/peers/central-registry/token
        {
            "federation_token": "new-token-value-here"
        }
    """
    _check_peer_management_scope(user_context)
    username = user_context.get("username", "unknown")
    logger.info(f"User '{username}' updating federation token for peer '{peer_id}'")

    service = get_peer_federation_service()

    try:
        # Get peer info before update for audit context
        peer_name = "unknown"
        try:
            existing_peer = await service.get_peer_by_id(peer_id)
            peer_name = existing_peer.name
        except Exception:
            pass  # Continue even if we can't get peer name

        # Use the standard update_peer with only the token field
        # This goes through the now-fixed update flow
        await service.update_peer(peer_id, {"federation_token": federation_token})

        logger.info(
            f"AUDIT: Federation token update successful via PATCH endpoint. "
            f"Peer: '{peer_id}' (name='{peer_name}'), User: '{username}'"
        )

        return {
            "message": "Federation token updated successfully",
            "peer_id": peer_id,
        }
    except ValueError as e:
        error_msg = str(e)
        if "not found" in error_msg:
            logger.error(f"Peer not found: {peer_id}")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=error_msg,
            )
        else:
            logger.error(f"Failed to update token: {error_msg}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=error_msg,
            )


@router.delete("/{peer_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_peer(
    peer_id: str,
    user_context: dict = Depends(nginx_proxied_auth),
):
    """
    Delete a peer registry configuration.

    Args:
        peer_id: Peer identifier
        user_context: Authenticated user context

    Raises:
        HTTPException: 404 if peer not found

    Example:
        DELETE /api/peers/central-registry
    """
    _check_peer_management_scope(user_context)
    logger.info(f"User '{user_context.get('username')}' deleting peer '{peer_id}'")

    service = get_peer_federation_service()

    try:
        await service.remove_peer(peer_id)
        logger.info(f"Successfully deleted peer '{peer_id}'")
        # Return None for 204 No Content
        return None
    except ValueError as e:
        logger.error(f"Failed to delete peer '{peer_id}': {e}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e),
        )


@router.post("/{peer_id}/sync", response_model=SyncResult)
async def sync_peer(
    peer_id: str,
    user_context: dict = Depends(nginx_proxied_auth),
) -> SyncResult:
    """
    Trigger synchronization for a specific peer.

    Args:
        peer_id: Peer identifier
        user_context: Authenticated user context

    Returns:
        Sync result with statistics

    Raises:
        HTTPException: 404 if peer not found
        HTTPException: 400 if peer is disabled

    Example:
        POST /api/peers/central-registry/sync
    """
    _check_peer_management_scope(user_context)
    logger.info(f"User '{user_context.get('username')}' triggering sync for peer '{peer_id}'")

    service = get_peer_federation_service()

    try:
        result = await service.sync_peer(peer_id)
        logger.info(
            f"Sync completed for peer '{peer_id}': "
            f"success={result.success}, servers={result.servers_synced}, "
            f"agents={result.agents_synced}"
        )
        return result
    except ValueError as e:
        error_msg = str(e)
        if "not found" in error_msg:
            logger.error(f"Peer not found: {peer_id}")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=error_msg,
            )
        else:
            logger.error(f"Sync failed: {error_msg}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=error_msg,
            )


@router.get("/{peer_id}/status", response_model=PeerSyncStatus)
async def get_peer_status(
    peer_id: str,
    user_context: dict = Depends(nginx_proxied_auth),
) -> PeerSyncStatus:
    """
    Get sync status for a specific peer.

    Args:
        peer_id: Peer identifier
        user_context: Authenticated user context

    Returns:
        Peer sync status with health and history

    Raises:
        HTTPException: 404 if peer not found

    Example:
        GET /api/peers/central-registry/status
    """
    _check_peer_management_scope(user_context)
    logger.info(f"User '{user_context.get('username')}' retrieving status for peer '{peer_id}'")

    service = get_peer_federation_service()

    sync_status = await service.get_sync_status(peer_id)

    if not sync_status:
        logger.error(f"Sync status not found for peer: {peer_id}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Sync status not found for peer: {peer_id}",
        )

    return sync_status


@router.post("/{peer_id}/enable", response_model=PeerRegistryConfig)
async def enable_peer(
    peer_id: str,
    user_context: dict = Depends(nginx_proxied_auth),
) -> PeerRegistryConfig:
    """
    Enable a peer registry.

    Args:
        peer_id: Peer identifier
        user_context: Authenticated user context

    Returns:
        Updated peer registry configuration

    Raises:
        HTTPException: 404 if peer not found

    Example:
        POST /api/peers/central-registry/enable
    """
    _check_peer_management_scope(user_context)
    logger.info(f"User '{user_context.get('username')}' enabling peer '{peer_id}'")

    service = get_peer_federation_service()

    try:
        updated_peer = await service.update_peer(peer_id, {"enabled": True})
        logger.info(f"Successfully enabled peer '{peer_id}'")
        return updated_peer
    except ValueError as e:
        logger.error(f"Failed to enable peer '{peer_id}': {e}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e),
        )


@router.post("/{peer_id}/disable", response_model=PeerRegistryConfig)
async def disable_peer(
    peer_id: str,
    user_context: dict = Depends(nginx_proxied_auth),
) -> PeerRegistryConfig:
    """
    Disable a peer registry.

    Args:
        peer_id: Peer identifier
        user_context: Authenticated user context

    Returns:
        Updated peer registry configuration

    Raises:
        HTTPException: 404 if peer not found

    Example:
        POST /api/peers/central-registry/disable
    """
    _check_peer_management_scope(user_context)
    logger.info(f"User '{user_context.get('username')}' disabling peer '{peer_id}'")

    service = get_peer_federation_service()

    try:
        updated_peer = await service.update_peer(peer_id, {"enabled": False})
        logger.info(f"Successfully disabled peer '{peer_id}'")
        return updated_peer
    except ValueError as e:
        logger.error(f"Failed to disable peer '{peer_id}': {e}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e),
        )


@router.get("/{peer_id}/connections", response_model=list[FederationConnectionLog])
async def get_peer_connections(
    peer_id: str,
    since: datetime | None = Query(
        None, description="Only return connections after this timestamp"
    ),
    limit: int = Query(100, ge=1, le=1000, description="Maximum entries to return"),
    user_context: dict = Depends(nginx_proxied_auth),
) -> list[FederationConnectionLog]:
    """
    Get connection history for a specific peer.

    Returns a list of federation connections from this peer, useful for
    debugging and monitoring federation sync operations.

    Args:
        peer_id: Peer identifier
        since: Only return connections after this timestamp
        limit: Maximum entries to return (default: 100, max: 1000)
        user_context: Authenticated user context

    Returns:
        List of connection logs for the peer

    Example:
        GET /api/peers/central-registry/connections
        GET /api/peers/central-registry/connections?since=2024-01-01T00:00:00Z
    """
    _check_peer_management_scope(user_context)
    logger.info(
        f"User '{user_context.get('username')}' retrieving connections for peer '{peer_id}'"
    )

    audit_service = get_federation_audit_service()
    connections = await audit_service.get_peer_connections(
        peer_id=peer_id,
        since=since,
        limit=limit,
    )

    logger.info(f"Returning {len(connections)} connection logs for peer '{peer_id}'")
    return connections


@router.get("/{peer_id}/shared-resources", response_model=PeerSyncSummary)
async def get_peer_shared_resources(
    peer_id: str,
    user_context: dict = Depends(nginx_proxied_auth),
) -> PeerSyncSummary:
    """
    Get summary of resources shared with a specific peer.

    Args:
        peer_id: Peer identifier
        user_context: Authenticated user context

    Returns:
        PeerSyncSummary for the specified peer

    Raises:
        HTTPException: 404 if peer has no connection history

    Example:
        GET /api/peers/central-registry/shared-resources
    """
    _check_peer_management_scope(user_context)
    logger.info(
        f"User '{user_context.get('username')}' retrieving shared resources for peer '{peer_id}'"
    )

    audit_service = get_federation_audit_service()
    summary = await audit_service.get_peer_summary(peer_id)

    if not summary:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No connection history found for peer: {peer_id}",
        )

    return summary
