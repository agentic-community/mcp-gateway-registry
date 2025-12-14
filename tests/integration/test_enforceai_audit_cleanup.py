"""
Stage 7.3 integration test: audit cleanup command against a real temp SQLite DB.
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import (
    datetime,
    timedelta,
    timezone,
)
from pathlib import Path

import pytest

from auth_server.enforceai.db.migrations import (
    upgrade_to_latest,
)
from auth_server.enforceai.stores.sqlite.audit_store import (
    SqliteAuditStore,
)
from cli import enforceai_audit_cleanup


def _migrate_db(
    db_path: Path,
) -> None:
    connection = sqlite3.connect(db_path)
    try:
        upgrade_to_latest(connection)
    finally:
        connection.close()


def _count_audit_events(
    *,
    db_path: Path,
) -> int:
    connection = sqlite3.connect(db_path)
    try:
        row = connection.execute("SELECT COUNT(*) FROM audit_events").fetchone()
    finally:
        connection.close()
    assert row is not None
    return int(row[0])


@pytest.mark.integration
def test_enforceai_audit_cleanup_deletes_old_events_and_outputs_summary(
    capsys: pytest.CaptureFixture[str],
    enforceai_sqlite_db_path: Path,
) -> None:
    _migrate_db(enforceai_sqlite_db_path)
    store = SqliteAuditStore(db_path=enforceai_sqlite_db_path)

    now = datetime.now(tz=timezone.utc).replace(microsecond=0)
    user_id = "https://issuer.example|sub-1"
    agent_id = str(uuid.uuid4())

    for offset in range(5):
        store.append_event(
            occurred_at=now - timedelta(days=10, seconds=offset),
            user_id=user_id,
            agent_id=agent_id,
            action="tools/call",
            outcome="allow",
            request_id=f"old-{offset}",
        )
    for offset in range(2):
        store.append_event(
            occurred_at=now - timedelta(hours=1, seconds=offset),
            user_id=user_id,
            agent_id=agent_id,
            action="tools/list",
            outcome="allow",
            request_id=f"recent-{offset}",
        )

    assert _count_audit_events(db_path=enforceai_sqlite_db_path) == 7

    code = enforceai_audit_cleanup.main(
        [
            "--db-path",
            str(enforceai_sqlite_db_path),
            "--retention-days",
            "1",
            "--max-db-bytes",
            "0",
            "--batch-size",
            "10",
        ],
        env={},
    )
    assert code == 0

    out = capsys.readouterr().out
    payload = json.loads(out)
    assert set(payload.keys()) == {
        "deleted_by_size",
        "deleted_by_time",
        "elapsed_seconds",
        "final_db_bytes",
        "finished_at",
        "started_at",
    }
    assert payload["deleted_by_time"] == 5
    assert payload["deleted_by_size"] == 0

    assert _count_audit_events(db_path=enforceai_sqlite_db_path) == 2

