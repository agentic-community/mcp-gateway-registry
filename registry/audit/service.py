"""
AuditLogger service for writing audit events to MongoDB.

This module provides the core audit logging service that writes
audit events to MongoDB for persistent storage and querying.
"""

import asyncio
import logging
from typing import TYPE_CHECKING, Optional, Union

from .models import RegistryApiAccessRecord, TokenMintAuditRecord

if TYPE_CHECKING:
    from ..repositories.audit_repository import AuditRepositoryBase

logger = logging.getLogger(__name__)


class AuditLogger:
    """
    Async audit logger for MongoDB storage.

    Writes audit events to MongoDB for persistent storage. Events can be
    queried through the audit API endpoints.

    Attributes:
        stream_name: Name of the audit stream for categorization
        mongodb_enabled: Whether MongoDB logging is enabled
    """

    def __init__(
        self,
        log_dir: str = "logs/audit",
        rotation_hours: int = 1,
        rotation_max_mb: int = 100,
        local_retention_hours: int = 24,
        stream_name: str = "registry-api-access",
        mongodb_enabled: bool = False,
        audit_repository: Optional["AuditRepositoryBase"] = None,
    ):
        """
        Initialize the AuditLogger.

        Args:
            log_dir: Deprecated - no longer used (kept for backward compatibility)
            rotation_hours: Deprecated - no longer used (kept for backward compatibility)
            rotation_max_mb: Deprecated - no longer used (kept for backward compatibility)
            local_retention_hours: Deprecated - no longer used (kept for backward compatibility)
            stream_name: Name of the audit stream for categorization
            mongodb_enabled: Whether to write audit events to MongoDB
            audit_repository: Repository for MongoDB writes (required if mongodb_enabled)
        """
        self.stream_name = stream_name
        self.mongodb_enabled = mongodb_enabled
        self._audit_repository = audit_repository

        # Lock for thread-safe operations
        self._lock = asyncio.Lock()

        if mongodb_enabled and audit_repository:
            logger.info(f"Audit logging enabled for stream: {stream_name} (MongoDB)")
        elif not mongodb_enabled:
            logger.warning(f"Audit logging disabled for stream: {stream_name}")

    async def log_event(
        self,
        record: Union[RegistryApiAccessRecord, "MCPServerAccessRecord", TokenMintAuditRecord],
    ) -> None:
        """
        Write an audit record to MongoDB.

        This method is thread-safe. If MongoDB is not enabled or not
        available, the event is silently dropped to avoid impacting
        request processing.

        Args:
            record: The audit record to log (RegistryApiAccessRecord or MCPServerAccessRecord)
        """
        if not self.mongodb_enabled or not self._audit_repository:
            return

        async with self._lock:
            try:
                await self._audit_repository.insert(record)
            except Exception as e:
                logger.error(f"Failed to write audit event to MongoDB: {e}")
                # Don't raise - audit logging should not break request processing

    async def close(self) -> None:
        """
        Close the audit logger.

        This method exists for backward compatibility and cleanup.
        """
        logger.debug(f"Audit logger closed for stream: {self.stream_name}")

    @property
    def current_file_path(self) -> str | None:
        """Deprecated - returns None (no local files)."""
        return None

    @property
    def is_open(self) -> bool:
        """Check if the audit logger is operational."""
        return self.mongodb_enabled and self._audit_repository is not None
