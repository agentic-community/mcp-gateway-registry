"""
Federation audit service for tracking peer connections.

This module provides services for logging and querying federation
connection history, enabling visibility into peer sync operations.
"""

import logging
from datetime import UTC, datetime
from threading import Lock
from typing import Optional

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# Constants
MAX_CONNECTION_LOG_ENTRIES: int = 1000
DEFAULT_LOG_RETENTION_DAYS: int = 30


class FederationConnectionLog(BaseModel):
    """Record of a federation connection from a peer."""

    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="When the connection occurred",
    )
    peer_id: str = Field(
        ...,
        description="ID of the connecting peer",
    )
    peer_name: str = Field(
        default="",
        description="Display name of the connecting peer",
    )
    client_id: str = Field(
        ...,
        description="OAuth2 client_id from the token",
    )
    endpoint: str = Field(
        ...,
        description="API endpoint accessed",
    )
    items_requested: int = Field(
        default=0,
        ge=0,
        description="Number of items requested/returned",
    )
    success: bool = Field(
        default=True,
        description="Whether the request succeeded",
    )
    error_message: str | None = Field(
        default=None,
        description="Error message if request failed",
    )
    request_id: str | None = Field(
        default=None,
        description="Unique request identifier for correlation",
    )


class PeerSyncSummary(BaseModel):
    """Summary of resources shared with a peer."""

    peer_id: str = Field(
        ...,
        description="ID of the peer",
    )
    peer_name: str = Field(
        default="",
        description="Display name of the peer",
    )
    total_connections: int = Field(
        default=0,
        ge=0,
        description="Total number of connections from this peer",
    )
    last_connection: datetime | None = Field(
        default=None,
        description="Timestamp of last connection",
    )
    servers_shared: int = Field(
        default=0,
        ge=0,
        description="Number of servers shared with this peer",
    )
    agents_shared: int = Field(
        default=0,
        ge=0,
        description="Number of agents shared with this peer",
    )
    successful_requests: int = Field(
        default=0,
        ge=0,
        description="Number of successful requests",
    )
    failed_requests: int = Field(
        default=0,
        ge=0,
        description="Number of failed requests",
    )


class FederationAuditService:
    """Service for tracking peer federation connections.

    Provides in-memory logging of federation connections for visibility
    and debugging purposes. In production, this could be backed by a
    persistent store.
    """

    _instance: Optional["FederationAuditService"] = None
    _lock: Lock = Lock()

    def __new__(cls) -> "FederationAuditService":
        """Singleton pattern with thread-safe double-checked locking."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        """Initialize the audit service."""
        if self._initialized:
            return

        self._connection_logs: list[FederationConnectionLog] = []
        self._peer_summaries: dict[str, PeerSyncSummary] = {}
        self._operation_lock = Lock()

        self._initialized = True
        logger.info("FederationAuditService initialized")

    async def log_connection(
        self,
        peer_id: str,
        client_id: str,
        endpoint: str,
        items_requested: int = 0,
        success: bool = True,
        error_message: str | None = None,
        peer_name: str = "",
        request_id: str | None = None,
    ) -> None:
        """
        Log a federation sync connection.

        Args:
            peer_id: ID of the connecting peer
            client_id: OAuth2 client_id from token
            endpoint: API endpoint accessed
            items_requested: Number of items returned
            success: Whether request succeeded
            error_message: Error message if failed
            peer_name: Display name of peer
            request_id: Unique request ID for correlation
        """
        with self._operation_lock:
            # Create log entry
            log_entry = FederationConnectionLog(
                peer_id=peer_id,
                peer_name=peer_name,
                client_id=client_id,
                endpoint=endpoint,
                items_requested=items_requested,
                success=success,
                error_message=error_message,
                request_id=request_id,
            )

            # Add to logs, maintaining max size
            self._connection_logs.insert(0, log_entry)
            if len(self._connection_logs) > MAX_CONNECTION_LOG_ENTRIES:
                self._connection_logs = self._connection_logs[:MAX_CONNECTION_LOG_ENTRIES]

            # Update peer summary
            if peer_id not in self._peer_summaries:
                self._peer_summaries[peer_id] = PeerSyncSummary(
                    peer_id=peer_id,
                    peer_name=peer_name,
                )

            summary = self._peer_summaries[peer_id]
            summary.total_connections += 1
            summary.last_connection = log_entry.timestamp

            if peer_name and not summary.peer_name:
                summary.peer_name = peer_name

            if success:
                summary.successful_requests += 1
                # Update shared counts based on endpoint
                if "/servers" in endpoint:
                    summary.servers_shared = max(summary.servers_shared, items_requested)
                elif "/agents" in endpoint:
                    summary.agents_shared = max(summary.agents_shared, items_requested)
            else:
                summary.failed_requests += 1

            logger.debug(
                f"Logged federation connection: peer={peer_id}, "
                f"endpoint={endpoint}, items={items_requested}, success={success}"
            )

    async def get_peer_connections(
        self,
        peer_id: str,
        since: datetime | None = None,
        limit: int = 100,
    ) -> list[FederationConnectionLog]:
        """
        Get connection history for a peer.

        Args:
            peer_id: ID of the peer
            since: Only return connections after this timestamp
            limit: Maximum entries to return

        Returns:
            List of connection logs for the peer
        """
        with self._operation_lock:
            # Filter by peer_id
            peer_logs = [log for log in self._connection_logs if log.peer_id == peer_id]

            # Filter by timestamp if specified
            if since:
                peer_logs = [log for log in peer_logs if log.timestamp > since]

            # Apply limit
            return peer_logs[:limit]

    async def get_all_connections(
        self,
        since: datetime | None = None,
        limit: int = 100,
    ) -> list[FederationConnectionLog]:
        """
        Get all connection history.

        Args:
            since: Only return connections after this timestamp
            limit: Maximum entries to return

        Returns:
            List of all connection logs
        """
        with self._operation_lock:
            logs = self._connection_logs.copy()

            # Filter by timestamp if specified
            if since:
                logs = [log for log in logs if log.timestamp > since]

            # Apply limit
            return logs[:limit]

    async def get_shared_resources_summary(self) -> dict[str, PeerSyncSummary]:
        """
        Get summary of what's shared with each peer.

        Returns:
            Dictionary mapping peer_id to PeerSyncSummary
        """
        with self._operation_lock:
            return self._peer_summaries.copy()

    async def get_peer_summary(
        self,
        peer_id: str,
    ) -> PeerSyncSummary | None:
        """
        Get summary for a specific peer.

        Args:
            peer_id: ID of the peer

        Returns:
            PeerSyncSummary if peer has connected, None otherwise
        """
        with self._operation_lock:
            return self._peer_summaries.get(peer_id)

    def clear_logs(self) -> None:
        """Clear all connection logs (for testing)."""
        with self._operation_lock:
            self._connection_logs.clear()
            self._peer_summaries.clear()
            logger.info("Cleared all federation audit logs")


# Global service instance
_federation_audit_service: FederationAuditService | None = None


def get_federation_audit_service() -> FederationAuditService:
    """
    Get the global federation audit service instance.

    Returns:
        Singleton FederationAuditService instance
    """
    global _federation_audit_service
    if _federation_audit_service is None:
        _federation_audit_service = FederationAuditService()
    return _federation_audit_service
