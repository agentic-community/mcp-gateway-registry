"""
Pydantic models for peer-to-peer federation in MCP Gateway Registry.

This module defines configuration and metadata models for federated registry
synchronization, enabling mesh topology where any registry can sync from any other.

Based on: docs/federation.md and implementation plan
"""

import logging
from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

logger = logging.getLogger(__name__)


# Constants
MIN_SYNC_INTERVAL_MINUTES: int = 5
MAX_SYNC_INTERVAL_MINUTES: int = 1440  # 24 hours
DEFAULT_SYNC_INTERVAL_MINUTES: int = 60
MAX_SYNC_HISTORY_ENTRIES: int = 100


def _validate_endpoint_url(
    url: str,
) -> str:
    """
    Validate peer registry endpoint URL format.

    Args:
        url: Endpoint URL to validate

    Returns:
        Validated URL string

    Raises:
        ValueError: If URL format is invalid
    """
    if not url:
        raise ValueError("Endpoint URL cannot be empty")

    if not (url.startswith("http://") or url.startswith("https://")):
        raise ValueError("Endpoint URL must use HTTP or HTTPS protocol")

    # Remove trailing slash for consistency
    url = url.rstrip("/")

    try:
        from urllib.parse import urlparse

        parsed = urlparse(url)
        if not parsed.netloc:
            raise ValueError("Endpoint URL must include a valid hostname")
    except Exception as e:
        raise ValueError(f"Invalid endpoint URL format: {e}")

    return url


def _validate_peer_id(
    peer_id: str,
) -> str:
    """
    Validate peer ID format for use as filename.

    Args:
        peer_id: Peer identifier to validate

    Returns:
        Validated peer ID

    Raises:
        ValueError: If peer ID format is invalid
    """
    if not peer_id:
        raise ValueError("Peer ID cannot be empty")

    if not peer_id.strip():
        raise ValueError("Peer ID cannot be whitespace only")

    # Check for invalid filename characters
    invalid_chars = ["/", "\\", ":", "*", "?", '"', "<", ">", "|", "\0"]
    for char in invalid_chars:
        if char in peer_id:
            raise ValueError(f"Peer ID cannot contain '{char}' character")

    # Limit length for filesystem compatibility
    if len(peer_id) > 255:
        raise ValueError("Peer ID cannot exceed 255 characters")

    return peer_id.strip()


class SyncMetadata(BaseModel):
    """
    Metadata for items synced from peer registries.

    Tracks the origin of synced items and local customizations,
    enabling proper merge behavior during subsequent syncs.
    """

    upstream_peer_id: str = Field(
        ...,
        description="ID of the peer registry this item was synced from",
    )
    upstream_path: str = Field(
        ...,
        description="Original path of the item in the upstream registry",
    )
    sync_generation: int = Field(
        default=1,
        ge=1,
        description="Generation number for incremental sync tracking",
    )
    last_synced_at: datetime = Field(
        ...,
        description="Timestamp of the last successful sync",
    )
    is_orphaned: bool = Field(
        default=False,
        description="Whether this item no longer exists in upstream",
    )
    orphaned_at: datetime | None = Field(
        default=None,
        description="Timestamp when item was marked as orphaned",
    )
    local_overrides: dict[str, Any] = Field(
        default_factory=dict,
        description="Fields that have been locally customized",
    )
    is_read_only: bool = Field(
        default=True,
        description="Whether core fields are read-only (synced items are read-only)",
    )

    model_config = ConfigDict(
        populate_by_name=True,
        json_schema_extra={
            "example": {
                "upstream_peer_id": "central-registry",
                "upstream_path": "/finance-tools",
                "sync_generation": 42,
                "last_synced_at": "2024-01-15T10:30:00Z",
                "is_orphaned": False,
                "local_overrides": {"tags": ["local-tag"]},
                "is_read_only": True,
            }
        },
    )

    @model_validator(mode="after")
    def _validate_orphan_timestamp(
        self,
    ) -> "SyncMetadata":
        """Validate orphaned_at is set when is_orphaned is True."""
        if self.is_orphaned and self.orphaned_at is None:
            # Auto-set orphaned_at if not provided
            object.__setattr__(self, "orphaned_at", datetime.now(UTC))
        return self


class PeerRegistryConfig(BaseModel):
    """
    Configuration for a peer registry connection.

    Defines how to connect to and sync from a peer registry,
    including endpoint, sync mode, and filtering options.
    """

    peer_id: str = Field(
        ...,
        min_length=1,
        max_length=255,
        description="Unique identifier for this peer registry",
    )
    name: str = Field(
        ...,
        min_length=1,
        max_length=255,
        description="Human-readable display name for the peer",
    )
    endpoint: str = Field(
        ...,
        description="Base URL of the peer registry API",
    )
    enabled: bool = Field(
        default=True,
        description="Whether sync from this peer is enabled",
    )

    # Sync configuration
    sync_mode: Literal["all", "whitelist", "tag_filter"] = Field(
        default="all",
        description="Sync mode: all items, whitelist only, or tag-based filtering",
    )
    whitelist_servers: list[str] = Field(
        default_factory=list,
        description="Server paths to sync when sync_mode is 'whitelist'",
    )
    whitelist_agents: list[str] = Field(
        default_factory=list,
        description="Agent paths to sync when sync_mode is 'whitelist'",
    )
    tag_filters: list[str] = Field(
        default_factory=list,
        description="Tags to filter by when sync_mode is 'tag_filter'",
    )

    sync_local_servers: bool = Field(
        default=False,
        description=(
            "Whether to accept deployment='local' servers from this peer. "
            "Local servers are executable launch recipes that run on developer "
            "machines — only enable for trusted peers. Default False is the "
            "safe choice; existing peers continue to behave identically."
        ),
    )

    # Scheduling
    sync_interval_minutes: int = Field(
        default=DEFAULT_SYNC_INTERVAL_MINUTES,
        ge=MIN_SYNC_INTERVAL_MINUTES,
        le=MAX_SYNC_INTERVAL_MINUTES,
        description=f"Sync interval in minutes ({MIN_SYNC_INTERVAL_MINUTES}-{MAX_SYNC_INTERVAL_MINUTES})",
    )

    # Federation static token (for peer-to-peer sync without OAuth2)
    # This is the FEDERATION_STATIC_TOKEN value from the remote peer registry.
    # When set, the client uses this directly as Bearer token instead of OAuth2.
    federation_token: str | None = Field(
        default=None,
        description="Federation static token from the remote peer registry. "
        "Used as Bearer token for sync requests when the peer has "
        "FEDERATION_STATIC_TOKEN_AUTH_ENABLED=true.",
    )

    # Identity binding (for peer identification via OAuth2 tokens)
    expected_client_id: str | None = Field(
        default=None,
        description="Azure AD/Keycloak client_id (azp claim) that identifies this peer",
    )
    expected_issuer: str | None = Field(
        default=None,
        description="Expected token issuer URL (for cross-tenant validation)",
    )

    # Metadata (set by service, not user input)
    created_at: datetime | None = Field(
        default=None,
        description="When this peer config was created",
    )
    updated_at: datetime | None = Field(
        default=None,
        description="When this peer config was last updated",
    )

    model_config = ConfigDict(
        populate_by_name=True,
        json_schema_extra={
            "example": {
                "peer_id": "central-registry",
                "name": "Central MCP Registry",
                "endpoint": "https://central.registry.company.com",
                "enabled": True,
                "sync_mode": "all",
                "sync_interval_minutes": 30,
                "federation_token": None,
                "expected_client_id": "uuid-central-1111-2222-3333",
                "expected_issuer": "https://login.microsoftonline.com/tenant-id/v2.0",
            }
        },
    )

    @field_validator("peer_id")
    @classmethod
    def _validate_peer_id_field(
        cls,
        v: str,
    ) -> str:
        """Validate peer ID format."""
        return _validate_peer_id(v)

    @field_validator("endpoint")
    @classmethod
    def _validate_endpoint_field(
        cls,
        v: str,
    ) -> str:
        """Validate endpoint URL format."""
        return _validate_endpoint_url(v)

    @model_validator(mode="after")
    def _validate_sync_mode_config(
        self,
    ) -> "PeerRegistryConfig":
        """Validate sync mode has required configuration."""
        if self.sync_mode == "whitelist":
            if not self.whitelist_servers and not self.whitelist_agents:
                logger.warning(
                    f"Peer '{self.peer_id}' has sync_mode='whitelist' but no "
                    "whitelist_servers or whitelist_agents configured. "
                    "No items will be synced."
                )
        elif self.sync_mode == "tag_filter":
            if not self.tag_filters:
                logger.warning(
                    f"Peer '{self.peer_id}' has sync_mode='tag_filter' but no "
                    "tag_filters configured. No items will be synced."
                )
        return self


class PeerRegistryConfigResponse(PeerRegistryConfig):
    """Response view of a peer registry config.

    ``federation_token`` is the static bearer token this registry uses to
    authenticate against the remote peer. It is write-only: settable via
    create/update and ``PATCH /{peer_id}/token``, but never serialized back out
    (otherwise any principal with the federation scope could harvest peer
    tokens and impersonate this registry). ``exclude`` drops it from every
    response regardless of value.
    """

    federation_token: str | None = Field(default=None, exclude=True)


class SyncHistoryEntry(BaseModel):
    """
    Record of a single sync operation.

    Captures the outcome of a sync attempt including success/failure,
    items synced, and any errors encountered.
    """

    sync_id: str = Field(
        ...,
        description="Unique identifier for this sync operation",
    )
    started_at: datetime = Field(
        ...,
        description="When the sync operation started",
    )
    completed_at: datetime | None = Field(
        default=None,
        description="When the sync operation completed",
    )
    success: bool = Field(
        default=False,
        description="Whether the sync completed successfully",
    )
    servers_synced: int = Field(
        default=0,
        ge=0,
        description="Number of servers synced",
    )
    agents_synced: int = Field(
        default=0,
        ge=0,
        description="Number of agents synced",
    )
    servers_orphaned: int = Field(
        default=0,
        ge=0,
        description="Number of servers marked as orphaned",
    )
    agents_orphaned: int = Field(
        default=0,
        ge=0,
        description="Number of agents marked as orphaned",
    )
    error_message: str | None = Field(
        default=None,
        description="Error message if sync failed",
    )
    sync_generation: int = Field(
        default=0,
        ge=0,
        description="Generation number used for this sync",
    )
    full_sync: bool = Field(
        default=False,
        description="Whether this was a full sync (vs incremental)",
    )

    model_config = ConfigDict(
        populate_by_name=True,
        json_schema_extra={
            "example": {
                "sync_id": "sync-2024-01-15-103000",
                "started_at": "2024-01-15T10:30:00Z",
                "completed_at": "2024-01-15T10:30:15Z",
                "success": True,
                "servers_synced": 42,
                "agents_synced": 15,
                "servers_orphaned": 0,
                "agents_orphaned": 1,
                "sync_generation": 100,
                "full_sync": False,
            }
        },
    )


class PeerSyncStatus(BaseModel):
    """
    Current sync status for a peer registry.

    Tracks the state of synchronization including last sync time,
    health status, and recent sync history.
    """

    peer_id: str = Field(
        ...,
        description="ID of the peer registry",
    )
    is_healthy: bool = Field(
        default=False,
        description="Whether the peer is currently reachable",
    )
    last_health_check: datetime | None = Field(
        default=None,
        description="When health was last checked",
    )
    last_successful_sync: datetime | None = Field(
        default=None,
        description="When last successful sync completed",
    )
    last_sync_attempt: datetime | None = Field(
        default=None,
        description="When last sync was attempted",
    )
    current_generation: int = Field(
        default=0,
        ge=0,
        description="Current sync generation number",
    )
    total_servers_synced: int = Field(
        default=0,
        ge=0,
        description="Total number of servers from this peer",
    )
    total_agents_synced: int = Field(
        default=0,
        ge=0,
        description="Total number of agents from this peer",
    )
    sync_in_progress: bool = Field(
        default=False,
        description="Whether a sync is currently running",
    )
    consecutive_failures: int = Field(
        default=0,
        ge=0,
        description="Number of consecutive sync failures",
    )
    sync_history: list[SyncHistoryEntry] = Field(
        default_factory=list,
        description=f"Recent sync history (max {MAX_SYNC_HISTORY_ENTRIES} entries)",
    )

    model_config = ConfigDict(
        populate_by_name=True,
        json_schema_extra={
            "example": {
                "peer_id": "central-registry",
                "is_healthy": True,
                "last_health_check": "2024-01-15T10:35:00Z",
                "last_successful_sync": "2024-01-15T10:30:15Z",
                "current_generation": 100,
                "total_servers_synced": 42,
                "total_agents_synced": 15,
                "sync_in_progress": False,
                "consecutive_failures": 0,
            }
        },
    )

    def add_history_entry(
        self,
        entry: SyncHistoryEntry,
    ) -> None:
        """
        Add a sync history entry, maintaining max entries limit.

        Args:
            entry: The sync history entry to add
        """
        self.sync_history.insert(0, entry)
        if len(self.sync_history) > MAX_SYNC_HISTORY_ENTRIES:
            self.sync_history = self.sync_history[:MAX_SYNC_HISTORY_ENTRIES]


class SyncResult(BaseModel):
    """
    Result of a sync operation.

    Returned by sync methods to indicate success/failure and
    provide details about what was synced.
    """

    success: bool = Field(
        ...,
        description="Whether the sync completed successfully",
    )
    peer_id: str = Field(
        ...,
        description="ID of the peer that was synced",
    )
    servers_synced: int = Field(
        default=0,
        ge=0,
        description="Number of servers synced",
    )
    agents_synced: int = Field(
        default=0,
        ge=0,
        description="Number of agents synced",
    )
    servers_orphaned: int = Field(
        default=0,
        ge=0,
        description="Number of servers marked as orphaned",
    )
    agents_orphaned: int = Field(
        default=0,
        ge=0,
        description="Number of agents marked as orphaned",
    )
    error_message: str | None = Field(
        default=None,
        description="Error message if sync failed",
    )
    duration_seconds: float = Field(
        default=0.0,
        ge=0.0,
        description="Duration of the sync operation in seconds",
    )
    new_generation: int = Field(
        default=0,
        ge=0,
        description="New generation number after sync",
    )

    model_config = ConfigDict(
        populate_by_name=True,
    )


class FederationExportResponse(BaseModel):
    """
    Response model for federation export API endpoints.

    Contains items to be synced by peer registries along with
    metadata for incremental sync support.
    """

    items: list[dict[str, Any]] = Field(
        default_factory=list,
        description="List of items (servers or agents) to sync",
    )
    sync_generation: int = Field(
        ...,
        description="Current generation number for incremental sync",
    )
    total_count: int = Field(
        ...,
        ge=0,
        description="Total number of items available",
    )
    has_more: bool = Field(
        default=False,
        description="Whether more items are available (pagination)",
    )
    registry_id: str = Field(
        ...,
        description="ID of the source registry",
    )

    model_config = ConfigDict(
        populate_by_name=True,
        json_schema_extra={
            "example": {
                "items": [{"path": "/finance-tools", "name": "Finance Tools"}],
                "sync_generation": 100,
                "total_count": 42,
                "has_more": False,
                "registry_id": "central-registry",
            }
        },
    )
