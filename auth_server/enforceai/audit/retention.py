from __future__ import annotations

import sqlite3
from datetime import (
    datetime,
    timedelta,
    timezone,
)
from pathlib import Path
from typing import Optional

from ..errors import (
    DependencyUnavailableError,
)
from ..stores.interfaces import (
    AuditStore,
)


def _ensure_aware_utc(
    value: datetime,
) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _vacuum_sqlite_db(
    *,
    db_path: Path,
) -> None:
    connection = sqlite3.connect(db_path)
    try:
        with connection:
            connection.execute("VACUUM")
    finally:
        connection.close()


def compute_cutoff(
    *,
    now: datetime,
    retention_days: int,
) -> Optional[datetime]:
    """Compute a UTC cutoff timestamp for time-based audit retention.

    Args:
        now: Current time. If tz-naive, treated as UTC.
        retention_days: Number of days to retain. `0` disables time deletion.

    Returns:
        A UTC-aware cutoff timestamp, or None when time retention is disabled.

    Raises:
        ValueError: If retention_days is negative.
    """
    if retention_days < 0:
        raise ValueError("retention_days must be non-negative")
    if retention_days == 0:
        return None

    normalized_now = _ensure_aware_utc(now).replace(microsecond=0)
    return normalized_now - timedelta(days=retention_days)


def enforce_time_retention(
    *,
    audit_store: AuditStore,
    cutoff: Optional[datetime],
) -> int:
    """Delete audit events older than the cutoff.

    Args:
        audit_store: AuditStore implementation.
        cutoff: Cutoff timestamp. None means no deletion.

    Returns:
        Number of deleted rows.

    Raises:
        DependencyUnavailableError: If the underlying store operation fails.
    """
    if cutoff is None:
        return 0

    normalized_cutoff = _ensure_aware_utc(cutoff).replace(microsecond=0)
    try:
        return audit_store.delete_events_older_than(
            cutoff=normalized_cutoff,
        )
    except Exception as exc:
        raise DependencyUnavailableError(
            "Audit retention time-based deletion failed",
        ) from exc


def enforce_size_retention(
    *,
    db_path: Path,
    audit_store: AuditStore,
    max_db_bytes: int,
    batch_size: int = 500,
) -> int:
    """Enforce a best-effort SQLite audit DB size cap by deleting oldest events.

    This function uses `db_path.stat().st_size` for size checks and compacts the
    DB via best-effort `VACUUM` after deletions.

    Args:
        db_path: SQLite DB file path.
        audit_store: AuditStore implementation backed by the same DB.
        max_db_bytes: Max DB size. `0` disables size deletion.
        batch_size: Number of events to delete per batch.

    Returns:
        Total number of deleted rows.

    Raises:
        ValueError: If parameters are invalid.
        DependencyUnavailableError: If the underlying store operation fails.
    """
    if max_db_bytes < 0:
        raise ValueError("max_db_bytes must be non-negative")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if max_db_bytes == 0:
        return 0
    if not db_path.exists():
        raise ValueError(f"db_path does not exist: {db_path}")

    deleted_total = 0
    while db_path.stat().st_size > max_db_bytes:
        try:
            deleted = audit_store.delete_oldest_events(limit=batch_size)
        except Exception as exc:
            raise DependencyUnavailableError(
                "Audit retention size-based deletion failed",
            ) from exc

        if deleted <= 0:
            break

        deleted_total += deleted
        try:
            _vacuum_sqlite_db(db_path=db_path)
        except Exception:
            continue

    return deleted_total

