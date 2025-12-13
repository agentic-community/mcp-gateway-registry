"""
Unit tests for EnforceAI SQLite migration runner (Stage 1.1).
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from auth_server.enforceai.db.migrations import (
    downgrade_one,
    upgrade_to_latest,
)


def _table_exists(
    connection: sqlite3.Connection,
    *,
    table_name: str,
) -> bool:
    row = connection.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
        (table_name,),
    ).fetchone()
    return row is not None


class TestEnforceAIMigrations:
    """Test suite for migration upgrade/downgrade behavior."""

    @pytest.mark.unit
    def test_upgrade_to_latest_is_idempotent(
        self,
        enforceai_sqlite_db_path: Path,
    ) -> None:
        connection = sqlite3.connect(enforceai_sqlite_db_path)
        try:
            upgrade_to_latest(connection)
            upgrade_to_latest(connection)

            rows = connection.execute(
                "SELECT version FROM schema_migrations ORDER BY version ASC"
            ).fetchall()
            assert [row[0] for row in rows] == ["0001_baseline"]

            assert _table_exists(connection, table_name="agents")
            assert _table_exists(connection, table_name="api_keys")
            assert _table_exists(connection, table_name="token_revocations")
            assert _table_exists(connection, table_name="audit_events")
        finally:
            connection.close()

    @pytest.mark.unit
    def test_downgrade_then_upgrade_restores_schema(
        self,
        enforceai_sqlite_db_path: Path,
    ) -> None:
        connection = sqlite3.connect(enforceai_sqlite_db_path)
        try:
            upgrade_to_latest(connection)

            rolled_back = downgrade_one(connection)
            assert rolled_back == "0001_baseline"

            assert not _table_exists(connection, table_name="agents")
            assert not _table_exists(connection, table_name="api_keys")
            assert not _table_exists(connection, table_name="token_revocations")
            assert not _table_exists(connection, table_name="audit_events")
            assert _table_exists(connection, table_name="schema_migrations")

            upgrade_to_latest(connection)
            assert _table_exists(connection, table_name="agents")
        finally:
            connection.close()
