"""
AuditLogger service for writing audit events to MongoDB.

This module provides the core audit logging service that writes
audit events to MongoDB for persistent storage and querying.
"""

import asyncio
import logging
from typing import TYPE_CHECKING, Optional, Union

from .models import MCPServerAccessRecord, RegistryApiAccessRecord, TokenMintAuditRecord

if TYPE_CHECKING:
    from ..repositories.audit_repository import AuditRepositoryBase

logger = logging.getLogger(__name__)


def _record_integrity_degradation(reason: str) -> None:
    """Increment the rate-alertable audit-integrity counter.

    Lazily imported (the `webhook_service` / `lifecycle_events` precedent) so
    that importing this module from the auth-server process -- which bootstraps
    its OWN meter provider in `auth_server.observability.meters` -- does not pull
    in a second one. Never raises: a metrics failure must not turn an audit
    problem into a request failure.

    `reason` is a code-supplied constant, never caller data, so the label
    cardinality stays bounded.
    """
    try:
        from ..observability.meters import audit_integrity_degraded_total

        audit_integrity_degraded_total.add(1, {"reason": reason})
    except Exception as exc:
        logger.warning("audit_integrity_degraded_total emit failed: %s", exc)


class NonDurableAuditError(RuntimeError):
    """Raised when audit logging would run without a durable sink and the
    deployment requires a durable audit trail.

    Fail-closed guard for the repudiation threat: best-effort JSON log lines are
    not a dependable audit trail (lost on restart, not queryable, rotated away).
    """


def enforce_durable_audit_sink(
    durable_sink_available: bool,
    require_durable: bool,
) -> None:
    """Enforce the durable-audit-sink policy, failing closed by default.

    Called during application startup after resolving whether a durable audit
    sink (MongoDB/DocumentDB) is actually available. When no durable sink is
    available:

    - if ``require_durable`` is True (the default deployment posture), raise
      :class:`NonDurableAuditError` so the application refuses to start rather
      than silently degrading to a non-durable audit trail;
    - if ``require_durable`` is False (explicit dev opt-out), emit a loud
      warning and allow startup to continue with best-effort log lines.

    Args:
        durable_sink_available: Whether a durable audit sink is available and
            wired up.
        require_durable: Whether the deployment requires a durable audit trail
            (``AUDIT_LOG_REQUIRE_DURABLE``).

    Raises:
        NonDurableAuditError: If no durable sink is available and
            ``require_durable`` is True.
    """
    if durable_sink_available:
        return

    if require_durable:
        logger.error(
            "Audit logging is enabled but no durable audit sink is available "
            "(MongoDB disabled or unreachable). Refusing to start with a "
            "non-durable audit trail. Provision the audit datastore, or set "
            "AUDIT_LOG_REQUIRE_DURABLE=false to explicitly accept a "
            "non-durable (best-effort log-line) audit trail."
        )
        raise NonDurableAuditError(
            "Non-durable audit trail refused: durable audit sink unavailable "
            "and AUDIT_LOG_REQUIRE_DURABLE is enabled. Provision the audit "
            "datastore or explicitly set AUDIT_LOG_REQUIRE_DURABLE=false."
        )

    logger.warning(
        "Audit logging is running WITHOUT a durable sink "
        "(AUDIT_LOG_REQUIRE_DURABLE=false). Audit records are best-effort log "
        "lines only and may be lost; this is unsafe for any repudiation-"
        "sensitive deployment. Enable MongoDB audit storage for a durable "
        "audit trail."
    )


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
        Write an audit record to the durable store (MongoDB).

        This method is thread-safe. If a durable sink is not available (audit
        disabled or MongoDB not wired up) the event is dropped — startup already
        fails closed on a missing durable sink when AUDIT_LOG_REQUIRE_DURABLE is
        set, so reaching this path means an operator explicitly opted into a
        non-durable trail.

        A durable-write failure at request time (transient MongoDB
        unavailability, etc.) does not fail the request — failing every API call
        on an audit blip would be a self-inflicted denial of service. Instead the
        dropped record is logged as a distinct CRITICAL event so the loss itself
        is loud and alertable (a durable audit trail must never lose a record
        silently). See the audit-logging docs for the retry/DLQ follow-up.

        Args:
            record: The audit record to log.
        """
        if not self.mongodb_enabled or not self._audit_repository:
            return

        async with self._lock:
            try:
                await self._audit_repository.insert(record)
            except Exception as e:
                # Never raise (would DoS the request path), but make the dropped
                # record loud and alertable rather than a quiet error line. Log
                # only identifying context, never the full record (avoids
                # leaking any sensitive field the record may carry).
                logger.critical(
                    "AUDIT RECORD DROPPED: durable write failed (log_type=%s request_id=%s): %s",
                    getattr(record, "log_type", "unknown"),
                    getattr(record, "request_id", "unknown"),
                    e,
                )
                _record_integrity_degradation("record_dropped")

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
