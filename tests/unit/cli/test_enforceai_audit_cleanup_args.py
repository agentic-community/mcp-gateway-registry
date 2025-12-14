"""
Unit tests for EnforceAI audit cleanup CLI (Stage 7.3).
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


@pytest.mark.unit
class TestEnforceAIAuditCleanupArgs:
    def test_env_fallback_for_db_path(
        self,
        capsys: pytest.CaptureFixture[str],
        enforceai_sqlite_db_path: Path,
    ) -> None:
        _migrate_db(enforceai_sqlite_db_path)

        code = enforceai_audit_cleanup.main(
            ["--dry-run", "--retention-days", "0", "--max-db-bytes", "0"],
            env={
                enforceai_audit_cleanup.ENV_DB_PATH: str(enforceai_sqlite_db_path),
            },
        )
        assert code == 0
        out = capsys.readouterr().out
        payload = json.loads(out)
        assert payload["final_db_bytes"] >= 0

    def test_dry_run_does_not_delete(
        self,
        capsys: pytest.CaptureFixture[str],
        enforceai_sqlite_db_path: Path,
    ) -> None:
        _migrate_db(enforceai_sqlite_db_path)
        store = SqliteAuditStore(db_path=enforceai_sqlite_db_path)

        now = datetime.now(tz=timezone.utc).replace(microsecond=0)
        user_id = "https://issuer.example|sub-1"
        agent_id = str(uuid.uuid4())

        for offset in range(3):
            store.append_event(
                occurred_at=now - timedelta(days=10, seconds=offset),
                user_id=user_id,
                agent_id=agent_id,
                action="tools/list",
                outcome="allow",
                request_id=f"old-{offset}",
            )

        before_count = _count_audit_events(db_path=enforceai_sqlite_db_path)

        code = enforceai_audit_cleanup.main(
            [
                "--db-path",
                str(enforceai_sqlite_db_path),
                "--retention-days",
                "1",
                "--max-db-bytes",
                "0",
                "--dry-run",
            ],
            env={},
        )
        assert code == 0

        out = capsys.readouterr().out
        payload = json.loads(out)
        assert payload["deleted_by_time"] == 3

        after_count = _count_audit_events(db_path=enforceai_sqlite_db_path)
        assert after_count == before_count

    def test_missing_db_path_rejected(self) -> None:
        with pytest.raises(
            enforceai_audit_cleanup.CLIError,
            match="db path is required",
        ):
            enforceai_audit_cleanup.main(
                ["--dry-run"],
                env={},
            )

