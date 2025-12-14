"""
Unit tests for EnforceAI audit retention primitives (Stage 7.2).
"""

from __future__ import annotations

import sqlite3
import uuid
from datetime import (
    datetime,
    timedelta,
    timezone,
)
from pathlib import Path
from unittest.mock import Mock

import pytest

from auth_server.enforceai.audit.retention import (
    compute_cutoff,
    enforce_size_retention,
    enforce_time_retention,
)
from auth_server.enforceai.db.migrations import (
    upgrade_to_latest,
)
from auth_server.enforceai.stores.sqlite.audit_store import (
    SqliteAuditStore,
)


def _migrate_db(
    db_path: Path,
) -> None:
    connection = sqlite3.connect(db_path)
    try:
        upgrade_to_latest(connection)
    finally:
        connection.close()


@pytest.mark.unit
class TestAuditRetention:
    def test_compute_cutoff_retention_days_zero_returns_none(self) -> None:
        assert (
            compute_cutoff(
                now=datetime(2025, 1, 1, tzinfo=timezone.utc),
                retention_days=0,
            )
            is None
        )

    def test_compute_cutoff_positive_days_returns_utc_aware_datetime(self) -> None:
        now = datetime(2025, 1, 11, 12, 0, 0, tzinfo=timezone.utc)
        cutoff = compute_cutoff(
            now=now,
            retention_days=10,
        )
        assert cutoff == datetime(2025, 1, 1, 12, 0, 0, tzinfo=timezone.utc)

    def test_compute_cutoff_naive_now_is_treated_as_utc(self) -> None:
        cutoff = compute_cutoff(
            now=datetime(2025, 1, 11, 12, 0, 0),
            retention_days=10,
        )
        assert cutoff == datetime(2025, 1, 1, 12, 0, 0, tzinfo=timezone.utc)

    def test_enforce_time_retention_none_cutoff_is_noop(self) -> None:
        store = Mock()
        deleted = enforce_time_retention(
            audit_store=store,
            cutoff=None,
        )
        assert deleted == 0
        store.delete_events_older_than.assert_not_called()

    def test_enforce_time_retention_deletes_using_cutoff(self) -> None:
        store = Mock()
        store.delete_events_older_than.return_value = 3
        cutoff = datetime(2025, 1, 1, tzinfo=timezone.utc)

        deleted = enforce_time_retention(
            audit_store=store,
            cutoff=cutoff,
        )
        assert deleted == 3
        store.delete_events_older_than.assert_called_once()

    def test_enforce_size_retention_under_cap_no_deletes(
        self,
        enforceai_sqlite_db_path: Path,
    ) -> None:
        _migrate_db(enforceai_sqlite_db_path)

        store = Mock()
        current_size = enforceai_sqlite_db_path.stat().st_size
        deleted = enforce_size_retention(
            db_path=enforceai_sqlite_db_path,
            audit_store=store,
            max_db_bytes=current_size + 1024,
            batch_size=10,
        )
        assert deleted == 0
        store.delete_oldest_events.assert_not_called()

    def test_enforce_size_retention_deletes_in_batches_until_under_cap(
        self,
        enforceai_sqlite_db_path: Path,
    ) -> None:
        _migrate_db(enforceai_sqlite_db_path)
        store = SqliteAuditStore(db_path=enforceai_sqlite_db_path)

        baseline_size = enforceai_sqlite_db_path.stat().st_size
        cap = baseline_size + 50_000

        user_id = "https://issuer.example|sub-1"
        agent_id = str(uuid.uuid4())
        base_time = datetime(2025, 1, 1, tzinfo=timezone.utc)
        large_payload = "x" * 20_000

        for offset in range(120):
            store.append_event(
                occurred_at=base_time + timedelta(seconds=offset),
                user_id=user_id,
                agent_id=agent_id,
                action="tools/call",
                outcome="allow",
                request_id=f"req-{offset}",
                details={"payload": large_payload},
            )

        assert enforceai_sqlite_db_path.stat().st_size > cap

        deleted = enforce_size_retention(
            db_path=enforceai_sqlite_db_path,
            audit_store=store,
            max_db_bytes=cap,
            batch_size=25,
        )
        assert deleted > 0
        assert enforceai_sqlite_db_path.stat().st_size <= cap

    def test_enforce_size_retention_stops_when_deletes_return_zero(
        self,
        enforceai_sqlite_db_path: Path,
    ) -> None:
        _migrate_db(enforceai_sqlite_db_path)

        store = Mock()
        store.delete_oldest_events.return_value = 0

        deleted = enforce_size_retention(
            db_path=enforceai_sqlite_db_path,
            audit_store=store,
            max_db_bytes=1,
            batch_size=10,
        )
        assert deleted == 0
        store.delete_oldest_events.assert_called_once_with(limit=10)

