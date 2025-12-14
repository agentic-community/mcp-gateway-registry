"""
Unit tests for EnforceAI AuditStore SQLite implementation (Stage 1.5).
"""

from __future__ import annotations

import sqlite3
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

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
class TestSqliteAuditStore:
    def test_append_and_query_recent(
        self,
        enforceai_sqlite_db_path: Path,
    ) -> None:
        _migrate_db(enforceai_sqlite_db_path)
        store = SqliteAuditStore(db_path=enforceai_sqlite_db_path)

        user_id = "https://issuer.example|sub-1"
        agent_id = str(uuid.uuid4())
        t0 = datetime(2025, 1, 1, tzinfo=timezone.utc)
        t1 = t0 + timedelta(seconds=1)

        first = store.append_event(
            occurred_at=t0,
            user_id=user_id,
            agent_id=agent_id,
            action="tools/list",
            outcome="allow",
            request_id="req-1",
            details={"path": "/mcp", "token_hint": "redacted"},
        )
        assert first.event_id >= 1
        assert first.occurred_at == t0
        assert first.details == {"path": "/mcp", "token_hint": "redacted"}

        second = store.append_event(
            occurred_at=t1,
            user_id=user_id,
            agent_id=agent_id,
            action="tools/call",
            outcome="deny",
            request_id="req-2",
        )
        assert second.event_id > first.event_id

        recent = store.list_recent_events(
            user_id=user_id,
            limit=10,
        )
        assert [event.event_id for event in recent] == [
            second.event_id,
            first.event_id,
        ]

        filtered = store.list_recent_events(
            user_id=user_id,
            since=t1,
            limit=10,
        )
        assert [event.event_id for event in filtered] == [second.event_id]

    def test_does_not_print_sensitive_fields(
        self,
        capsys: pytest.CaptureFixture[str],
        caplog: pytest.LogCaptureFixture,
        enforceai_sqlite_db_path: Path,
    ) -> None:
        _migrate_db(enforceai_sqlite_db_path)
        store = SqliteAuditStore(db_path=enforceai_sqlite_db_path)

        store.append_event(
            occurred_at=datetime(2025, 1, 1, tzinfo=timezone.utc),
            user_id="https://issuer.example|sub-1",
            agent_id=str(uuid.uuid4()),
            action="auth/validate",
            outcome="deny",
            details={"api_key_secret": "should-not-appear"},
        )

        out = capsys.readouterr()
        assert "should-not-appear" not in out.out
        assert "should-not-appear" not in out.err

        combined_logs = "\n".join(record.message for record in caplog.records)
        assert "should-not-appear" not in combined_logs

    def test_requires_user_id_or_agent_id_filter(
        self,
        enforceai_sqlite_db_path: Path,
    ) -> None:
        _migrate_db(enforceai_sqlite_db_path)
        store = SqliteAuditStore(db_path=enforceai_sqlite_db_path)

        with pytest.raises(ValueError, match="At least one of user_id or agent_id"):
            store.list_recent_events()

        with pytest.raises(ValueError, match="limit must be positive"):
            store.list_recent_events(
                user_id="https://issuer.example|sub-1",
                limit=0,
            )

    def test_delete_events_older_than_deletes_expected_subset(
        self,
        enforceai_sqlite_db_path: Path,
    ) -> None:
        _migrate_db(enforceai_sqlite_db_path)
        store = SqliteAuditStore(db_path=enforceai_sqlite_db_path)

        user_id = "https://issuer.example|sub-1"
        agent_id = str(uuid.uuid4())
        base_time = datetime(2025, 1, 1, tzinfo=timezone.utc)

        for offset_seconds in range(5):
            store.append_event(
                occurred_at=base_time + timedelta(seconds=offset_seconds),
                user_id=user_id,
                agent_id=agent_id,
                action="tools/list",
                outcome="allow",
                request_id=f"req-{offset_seconds}",
            )

        cutoff = base_time + timedelta(seconds=2)
        deleted = store.delete_events_older_than(cutoff=cutoff)
        assert deleted == 2

        remaining = store.list_recent_events(
            user_id=user_id,
            limit=10,
        )
        assert [event.request_id for event in remaining] == [
            "req-4",
            "req-3",
            "req-2",
        ]

    def test_delete_events_older_than_accepts_naive_cutoff_as_utc(
        self,
        enforceai_sqlite_db_path: Path,
    ) -> None:
        _migrate_db(enforceai_sqlite_db_path)
        store = SqliteAuditStore(db_path=enforceai_sqlite_db_path)

        user_id = "https://issuer.example|sub-1"
        agent_id = str(uuid.uuid4())
        base_time = datetime(2025, 1, 1, tzinfo=timezone.utc)

        store.append_event(
            occurred_at=base_time,
            user_id=user_id,
            agent_id=agent_id,
            action="tools/list",
            outcome="allow",
            request_id="req-0",
        )
        store.append_event(
            occurred_at=base_time + timedelta(seconds=1),
            user_id=user_id,
            agent_id=agent_id,
            action="tools/list",
            outcome="allow",
            request_id="req-1",
        )

        deleted = store.delete_events_older_than(
            cutoff=datetime(2025, 1, 1, 0, 0, 1),
        )
        assert deleted == 1

        remaining = store.list_recent_events(
            user_id=user_id,
            limit=10,
        )
        assert [event.request_id for event in remaining] == ["req-1"]

    def test_delete_oldest_events_deletes_oldest_rows(
        self,
        enforceai_sqlite_db_path: Path,
    ) -> None:
        _migrate_db(enforceai_sqlite_db_path)
        store = SqliteAuditStore(db_path=enforceai_sqlite_db_path)

        user_id = "https://issuer.example|sub-1"
        agent_id = str(uuid.uuid4())
        base_time = datetime(2025, 1, 1, tzinfo=timezone.utc)

        for offset_seconds in range(5):
            store.append_event(
                occurred_at=base_time + timedelta(seconds=offset_seconds),
                user_id=user_id,
                agent_id=agent_id,
                action="tools/list",
                outcome="allow",
                request_id=f"req-{offset_seconds}",
            )

        deleted = store.delete_oldest_events(limit=2)
        assert deleted == 2

        remaining = store.list_recent_events(
            user_id=user_id,
            limit=10,
        )
        assert [event.request_id for event in remaining] == [
            "req-4",
            "req-3",
            "req-2",
        ]

    def test_delete_methods_return_zero_when_table_empty(
        self,
        enforceai_sqlite_db_path: Path,
    ) -> None:
        _migrate_db(enforceai_sqlite_db_path)
        store = SqliteAuditStore(db_path=enforceai_sqlite_db_path)

        assert store.delete_events_older_than(
            cutoff=datetime(2025, 1, 1, tzinfo=timezone.utc),
        ) == 0
        assert store.delete_oldest_events(limit=1) == 0

    def test_delete_oldest_events_requires_positive_limit(
        self,
        enforceai_sqlite_db_path: Path,
    ) -> None:
        _migrate_db(enforceai_sqlite_db_path)
        store = SqliteAuditStore(db_path=enforceai_sqlite_db_path)

        with pytest.raises(ValueError, match="limit must be positive"):
            store.delete_oldest_events(limit=0)
