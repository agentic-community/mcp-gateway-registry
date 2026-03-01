"""
Federation Export API routes for MCP Gateway Registry.

This module provides REST API endpoints for exporting servers and agents to peer
registries in a federated mesh topology. Endpoints enforce visibility-based
access control and support incremental sync via generation numbers.

Based on: docs/federation.md
"""

import logging
import socket
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Query, status

from ..auth.dependencies import nginx_proxied_auth
from ..core.config import settings
from ..schemas.peer_federation_schema import FederationExportResponse
from ..repositories.factory import get_security_scan_repository
from ..services.agent_service import agent_service
from ..services.federation_audit_service import get_federation_audit_service
from ..services.peer_federation_service import get_peer_federation_service
from ..services.server_service import server_service

logger = logging.getLogger(__name__)


router = APIRouter(prefix="/api/federation", tags=["federation"])


# Constants
DEFAULT_PAGE_LIMIT: int = 100
MAX_PAGE_LIMIT: int = 1000


async def _get_current_sync_generation() -> int:
    """
    Compute the current sync generation dynamically.

    Uses the total count of enabled servers and agents as a generation proxy.
    When items are added, removed, or toggled, the count changes, signaling
    peers that a re-sync may be needed. Returns at least 1 so that peers
    with generation 0 always get all items on their first sync.

    Returns:
        Current sync generation number (minimum 1)
    """
    try:
        all_servers = await server_service.get_all_servers()
        enabled_server_count = 0
        for path in all_servers:
            if await server_service.is_service_enabled(path):
                enabled_server_count += 1

        all_agents = await agent_service.get_all_agents()
        enabled_agent_count = len([a for a in all_agents if agent_service.is_agent_enabled(a.path)])

        generation = max(1, enabled_server_count + enabled_agent_count)
        return generation
    except Exception as e:
        logger.warning(f"Failed to compute sync generation, defaulting to 1: {e}")
        return 1


def _get_registry_id() -> str:
    """
    Get the unique identifier for this registry instance.

    Uses REGISTRY_ID from settings if configured, otherwise falls back
    to hostname-based ID.

    Returns:
        Registry identifier string
    """
    # Use configured registry_id if available
    if settings.registry_id:
        return settings.registry_id

    # Fallback to hostname-based ID
    try:
        hostname = socket.gethostname()
        return f"registry-{hostname}"
    except Exception as e:
        logger.warning(f"Failed to get hostname: {e}, using default")
        return "registry-unknown"


def _check_federation_scope(
    user_context: dict[str, Any],
) -> None:
    """Check if user has federation access scope.

    Accepts either:
    - 'federation-service' scope (OAuth2 JWT from Keycloak service account)
    - 'federation/read' scope (federation static token)

    Args:
        user_context: User context from auth dependency

    Raises:
        HTTPException: 403 if no federation scope is present
    """
    scopes = user_context.get("scopes", [])
    has_federation_scope = "federation-service" in scopes or "federation/read" in scopes
    if not has_federation_scope:
        logger.warning(
            f"User {user_context.get('username')} attempted federation access "
            f"without federation scope. Scopes: {scopes}"
        )
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Federation access requires 'federation-service' or 'federation/read' scope",
        )


def _get_item_attr(
    item: Any,
    attr: str,
    default: Any = None,
) -> Any:
    """
    Get attribute from item, supporting both dict and object types.

    Args:
        item: Dict or object to get attribute from
        attr: Attribute name
        default: Default value if not found

    Returns:
        Attribute value or default
    """
    if isinstance(item, dict):
        return item.get(attr, default)
    return getattr(item, attr, default)


def _is_federated_item(
    item: Any,
) -> bool:
    """
    Check if an item was synced from another peer registry.

    This is used for chain prevention: items synced from A to B should not
    be re-exported from B to C. Only locally-created items should be exported.

    Args:
        item: Dict or object to check

    Returns:
        True if the item has sync_metadata.is_federated == True
    """
    sync_metadata = _get_item_attr(item, "sync_metadata", None)
    if not sync_metadata:
        return False

    if isinstance(sync_metadata, dict):
        return sync_metadata.get("is_federated", False) is True

    return getattr(sync_metadata, "is_federated", False) is True


def _filter_by_visibility(
    items: list[Any],
    peer_groups: list[str],
) -> list[Any]:
    """
    Filter items based on visibility and peer's group membership.

    Filtering rules:
    - Federated items (synced from another peer): NEVER included (chain prevention)
    - visibility=public: Always included (default if not specified)
    - visibility=group-restricted: Include only if peer is in allowed_groups
    - visibility=internal: NEVER included

    Args:
        items: List of items (dict or object) to filter
        peer_groups: Groups the peer registry belongs to (from JWT)

    Returns:
        Filtered list of items
    """
    filtered = []
    peer_group_set = set(peer_groups)
    federated_count = 0

    for item in items:
        # Chain prevention: Never re-export items synced from another peer
        # This prevents A->B->C propagation where items from A would be
        # re-exported from B to C
        if _is_federated_item(item):
            federated_count += 1
            continue

        # Default to "public" if visibility not specified (backwards compatibility)
        visibility = _get_item_attr(item, "visibility", "public")

        # Never export internal items
        if visibility == "internal":
            continue

        # Always export public items
        if visibility == "public":
            filtered.append(item)
            continue

        # Export group-restricted only if peer is in allowed_groups
        if visibility == "group-restricted":
            allowed_groups = set(_get_item_attr(item, "allowed_groups", []))
            if allowed_groups & peer_group_set:
                filtered.append(item)
                continue

    logger.debug(
        f"Filtered {len(items)} items to {len(filtered)} based on visibility. "
        f"Excluded {federated_count} federated items (chain prevention). "
        f"Peer groups: {peer_groups}"
    )

    return filtered


def _filter_by_generation(
    items: list[Any],
    since_generation: int | None,
) -> list[Any]:
    """
    Filter items by sync generation for incremental sync.

    Args:
        items: List of items (dict or object) to filter
        since_generation: Minimum generation number (exclusive).
                          If None, returns all items (full sync).
                          If 0, returns only items with generation > 0.

    Returns:
        Filtered list of items with generation > since_generation,
        plus all items without sync_metadata (local items never synced).
    """
    # Full sync: return all items if since_generation is None
    if since_generation is None:
        return items

    filtered = []
    for item in items:
        sync_metadata = _get_item_attr(item, "sync_metadata", None)
        if sync_metadata:
            if isinstance(sync_metadata, dict):
                item_generation = sync_metadata.get("sync_generation", 0)
            else:
                item_generation = getattr(sync_metadata, "sync_generation", 0)

            # Include if item's generation is newer than requested
            if item_generation > since_generation:
                filtered.append(item)
        else:
            # Items without sync_metadata are local items that have never been
            # synced - always include them as they're "new" to the peer
            filtered.append(item)

    logger.debug(
        f"Filtered {len(items)} items to {len(filtered)} with generation > {since_generation} "
        f"(includes items without sync_metadata)"
    )

    return filtered


def _item_to_dict(
    item: Any,
) -> dict[str, Any]:
    """
    Convert item to dictionary, supporting both dict and object types.

    Args:
        item: Dict or Pydantic model

    Returns:
        Dictionary representation
    """
    if isinstance(item, dict):
        return item
    if hasattr(item, "model_dump"):
        return item.model_dump()
    return dict(item)


def _paginate(
    items: list[Any],
    limit: int,
    offset: int,
) -> tuple[list[Any], bool]:
    """
    Paginate items list.

    Args:
        items: List of items to paginate
        limit: Maximum items per page
        offset: Number of items to skip

    Returns:
        Tuple of (paginated items, has_more flag)
    """
    start = offset
    end = offset + limit
    paginated = items[start:end]
    has_more = len(items) > end

    return paginated, has_more


async def federation_auth(
    user_context: Annotated[dict[str, Any], Depends(nginx_proxied_auth)],
) -> dict[str, Any]:
    """
    Authentication dependency for federation endpoints.

    Validates that the requester has federation-service scope in their JWT
    and identifies the peer by matching their client_id (azp claim) to a
    registered peer's expected_client_id.

    Args:
        user_context: User context from nginx_proxied_auth

    Returns:
        User context enriched with peer_id and peer_name if peer is identified

    Raises:
        HTTPException: 403 if federation-service scope not present
    """
    _check_federation_scope(user_context)

    # Extract client_id from token (azp claim)
    client_id = user_context.get("client_id", "")

    # Look up peer by client_id
    if client_id:
        peer_service = get_peer_federation_service()
        peer = await peer_service.get_peer_by_client_id(client_id)
        if peer:
            user_context["peer_id"] = peer.peer_id
            user_context["peer_name"] = peer.name
            logger.info(f"Identified federation peer: {peer.peer_id} (client_id: {client_id})")
        else:
            logger.debug(f"Federation request from unregistered client_id: {client_id}")

    return user_context


@router.get("/health")
async def federation_health():
    """
    Federation health check endpoint.

    This endpoint does NOT require authentication and is used by peer registries
    to check if the federation API is available before attempting sync.

    Returns:
        200 OK with status message
    """
    return {
        "status": "healthy",
        "federation_api_version": "1.0",
        "registry_id": _get_registry_id(),
    }


@router.get(
    "/servers",
    response_model=FederationExportResponse,
)
async def export_servers(
    limit: int = Query(
        DEFAULT_PAGE_LIMIT,
        ge=1,
        le=MAX_PAGE_LIMIT,
        description=f"Maximum items per page (default {DEFAULT_PAGE_LIMIT}, max {MAX_PAGE_LIMIT})",
    ),
    offset: int = Query(
        0,
        ge=0,
        description="Number of items to skip (default 0)",
    ),
    since_generation: int | None = Query(
        None,
        ge=0,
        description="Return only items with generation > this value (for incremental sync)",
    ),
    user_context: Annotated[dict[str, Any], Depends(federation_auth)] = None,
):
    """
    Export servers for federation sync.

    Returns servers with visibility filtering based on the peer's group membership.
    Supports pagination and incremental sync via generation numbers.

    **Authentication:** Requires JWT with 'federation-service' scope

    **Visibility filtering:**
    - public: Returned to all peers
    - group-restricted: Returned only if peer is in allowed_groups
    - internal: NEVER returned

    **Pagination:**
    - Use limit and offset for pagination
    - Check has_more to determine if more pages exist

    **Incremental sync:**
    - Use since_generation to get only changed items
    - Track sync_generation from response for next sync

    Args:
        limit: Maximum items per page
        offset: Number of items to skip
        since_generation: Minimum generation for incremental sync
        user_context: Authenticated peer context

    Returns:
        FederationExportResponse with filtered servers

    Raises:
        HTTPException: 401 if unauthenticated, 403 if missing federation scope
    """
    logger.info(
        f"Federation export request for servers from peer '{user_context['username']}' "
        f"(limit={limit}, offset={offset}, since_generation={since_generation})"
    )

    # Get all servers (enabled and disabled) - returns Dict[str, Dict[str, Any]]
    all_servers_dict = await server_service.get_all_servers()

    # Convert to list and filter out disabled servers - never sync disabled servers
    # Each server is a dict with 'path' key
    enabled_servers = []
    for path, server_data in all_servers_dict.items():
        if await server_service.is_service_enabled(path):
            enabled_servers.append(server_data)

    # Extract peer groups from JWT for visibility filtering
    peer_groups = user_context.get("groups", [])

    # Apply visibility filtering
    visible_servers = _filter_by_visibility(enabled_servers, peer_groups)

    # Apply generation filtering for incremental sync
    if since_generation is not None:
        visible_servers = _filter_by_generation(visible_servers, since_generation)

    # Apply pagination
    total_count = len(visible_servers)
    paginated_servers, has_more = _paginate(visible_servers, limit, offset)

    # Convert to dict format (servers are already dicts from service)
    items = [_item_to_dict(server) for server in paginated_servers]

    logger.info(
        f"Exporting {len(items)} servers to peer '{user_context['username']}' "
        f"(total visible: {total_count}, has_more: {has_more})"
    )

    # Log the connection for audit trail
    audit_service = get_federation_audit_service()
    await audit_service.log_connection(
        peer_id=user_context.get("peer_id", user_context.get("username", "unknown")),
        peer_name=user_context.get("peer_name", ""),
        client_id=user_context.get("client_id", ""),
        endpoint="/api/federation/servers",
        items_requested=len(items),
        success=True,
    )

    return FederationExportResponse(
        items=items,
        sync_generation=await _get_current_sync_generation(),
        total_count=total_count,
        has_more=has_more,
        registry_id=_get_registry_id(),
    )


@router.get(
    "/agents",
    response_model=FederationExportResponse,
)
async def export_agents(
    limit: int = Query(
        DEFAULT_PAGE_LIMIT,
        ge=1,
        le=MAX_PAGE_LIMIT,
        description=f"Maximum items per page (default {DEFAULT_PAGE_LIMIT}, max {MAX_PAGE_LIMIT})",
    ),
    offset: int = Query(
        0,
        ge=0,
        description="Number of items to skip (default 0)",
    ),
    since_generation: int | None = Query(
        None,
        ge=0,
        description="Return only items with generation > this value (for incremental sync)",
    ),
    user_context: Annotated[dict[str, Any], Depends(federation_auth)] = None,
):
    """
    Export agents for federation sync.

    Returns agents with visibility filtering based on the peer's group membership.
    Supports pagination and incremental sync via generation numbers.

    **Authentication:** Requires JWT with 'federation-service' scope

    **Visibility filtering:**
    - public: Returned to all peers
    - group-restricted: Returned only if peer is in allowed_groups
    - internal: NEVER returned

    **Pagination:**
    - Use limit and offset for pagination
    - Check has_more to determine if more pages exist

    **Incremental sync:**
    - Use since_generation to get only changed items
    - Track sync_generation from response for next sync

    Args:
        limit: Maximum items per page
        offset: Number of items to skip
        since_generation: Minimum generation for incremental sync
        user_context: Authenticated peer context

    Returns:
        FederationExportResponse with filtered agents

    Raises:
        HTTPException: 401 if unauthenticated, 403 if missing federation scope
    """
    logger.info(
        f"Federation export request for agents from peer '{user_context['username']}' "
        f"(limit={limit}, offset={offset}, since_generation={since_generation})"
    )

    # Get all agents (enabled and disabled)
    all_agents = await agent_service.get_all_agents()

    # Filter out disabled agents - never sync disabled agents
    enabled_agents = [a for a in all_agents if agent_service.is_agent_enabled(a.path)]

    # Extract peer groups from JWT for visibility filtering
    peer_groups = user_context.get("groups", [])

    # Apply visibility filtering
    visible_agents = _filter_by_visibility(enabled_agents, peer_groups)

    # Apply generation filtering for incremental sync
    if since_generation is not None:
        visible_agents = _filter_by_generation(visible_agents, since_generation)

    # Apply pagination
    total_count = len(visible_agents)
    paginated_agents, has_more = _paginate(visible_agents, limit, offset)

    # Convert to dict format (agents are AgentCard objects)
    items = [_item_to_dict(agent) for agent in paginated_agents]

    logger.info(
        f"Exporting {len(items)} agents to peer '{user_context['username']}' "
        f"(total visible: {total_count}, has_more: {has_more})"
    )

    # Log the connection for audit trail
    audit_service = get_federation_audit_service()
    await audit_service.log_connection(
        peer_id=user_context.get("peer_id", user_context.get("username", "unknown")),
        peer_name=user_context.get("peer_name", ""),
        client_id=user_context.get("client_id", ""),
        endpoint="/api/federation/agents",
        items_requested=len(items),
        success=True,
    )

    return FederationExportResponse(
        items=items,
        sync_generation=await _get_current_sync_generation(),
        total_count=total_count,
        has_more=has_more,
        registry_id=_get_registry_id(),
    )


@router.get(
    "/security-scans",
    response_model=FederationExportResponse,
)
async def export_security_scans(
    limit: int = Query(
        DEFAULT_PAGE_LIMIT,
        ge=1,
        le=MAX_PAGE_LIMIT,
        description=f"Maximum items per page (default {DEFAULT_PAGE_LIMIT}, max {MAX_PAGE_LIMIT})",
    ),
    offset: int = Query(
        0,
        ge=0,
        description="Number of items to skip (default 0)",
    ),
    user_context: Annotated[dict[str, Any], Depends(federation_auth)] = None,
):
    """
    Export security scan results for federation sync.

    Returns security scan results only for servers that are visible to the peer
    based on the server's visibility settings. This ensures peers only receive
    security information for servers they can access.

    **Authentication:** Requires JWT with 'federation-service' or 'federation/read' scope

    **Visibility filtering:**
    - Only returns scans for servers with visibility=public
    - Returns scans for group-restricted servers only if peer is in allowed_groups
    - Never returns scans for internal servers

    **Pagination:**
    - Use limit and offset for pagination
    - Check has_more to determine if more pages exist

    Args:
        limit: Maximum items per page
        offset: Number of items to skip
        user_context: Authenticated peer context

    Returns:
        FederationExportResponse with security scan results

    Raises:
        HTTPException: 401 if unauthenticated, 403 if missing federation scope
    """
    logger.info(
        f"Federation export request for security scans from peer '{user_context['username']}' "
        f"(limit={limit}, offset={offset})"
    )

    # Get all servers to build visibility map
    all_servers_dict = await server_service.get_all_servers()

    # Build a set of visible server paths for this peer
    peer_groups = user_context.get("groups", [])
    peer_group_set = set(peer_groups)
    visible_server_paths: set[str] = set()

    for path, server_data in all_servers_dict.items():
        # Check if server is enabled
        if not await server_service.is_service_enabled(path):
            continue

        # Check visibility
        visibility = server_data.get("visibility", "public")

        if visibility == "internal":
            continue

        if visibility == "public":
            visible_server_paths.add(path)
            continue

        if visibility == "group-restricted":
            allowed_groups = set(server_data.get("allowed_groups", []))
            if allowed_groups & peer_group_set:
                visible_server_paths.add(path)

    logger.debug(f"Visible server paths for peer: {len(visible_server_paths)} servers")

    # Get all security scans from repository
    scan_repo = get_security_scan_repository()
    all_scans = await scan_repo.list_all()

    # Filter scans to only include those for visible servers
    visible_scans = []
    for scan in all_scans:
        server_path = scan.get("server_path", "")
        if server_path in visible_server_paths:
            visible_scans.append(scan)

    logger.debug(f"Filtered {len(all_scans)} scans to {len(visible_scans)} for visible servers")

    # Apply pagination
    total_count = len(visible_scans)
    paginated_scans, has_more = _paginate(visible_scans, limit, offset)

    # Convert to dict format (scans are already dicts)
    items = [_item_to_dict(scan) for scan in paginated_scans]

    logger.info(
        f"Exporting {len(items)} security scans to peer '{user_context['username']}' "
        f"(total visible: {total_count}, has_more: {has_more})"
    )

    # Log the connection for audit trail
    audit_service = get_federation_audit_service()
    await audit_service.log_connection(
        peer_id=user_context.get("peer_id", user_context.get("username", "unknown")),
        peer_name=user_context.get("peer_name", ""),
        client_id=user_context.get("client_id", ""),
        endpoint="/api/federation/security-scans",
        items_requested=len(items),
        success=True,
    )

    return FederationExportResponse(
        items=items,
        sync_generation=await _get_current_sync_generation(),
        total_count=total_count,
        has_more=has_more,
        registry_id=_get_registry_id(),
    )
