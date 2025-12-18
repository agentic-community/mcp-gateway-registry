"""
Unit tests for EnforceAI egress allowlist SQLite store (Phase 2).
"""

from __future__ import annotations

import sqlite3
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
from auth_server.enforceai.stores.sqlite.egress_allowlist_store import (
    SqliteEgressAllowlistStore,
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
class TestSqliteEgressAllowlistStore:
    def test_crud_and_ttl_filtering(
        self,
        enforceai_sqlite_db_path: Path,
    ) -> None:
        _migrate_db(enforceai_sqlite_db_path)
        store = SqliteEgressAllowlistStore(db_path=enforceai_sqlite_db_path)

        now = datetime(2025, 1, 1, tzinfo=timezone.utc)

        created = store.create_entry(
            kind="hostname",
            value="example.com",
            comment="test",
            expires_at=now + timedelta(days=1),
        )
        assert created.entry_id >= 1
        assert created.kind == "hostname"
        assert created.value == "example.com"
        assert created.comment == "test"

        fetched = store.get_entry_by_id(entry_id=created.entry_id)
        assert fetched == created

        updated = store.update_entry(
            entry_id=created.entry_id,
            comment="updated",
        )
        assert updated is not None
        assert updated.entry_id == created.entry_id
        assert updated.comment == "updated"

        assert store.list_entries(include_expired=False, now=now) != []

        store.update_entry(
            entry_id=created.entry_id,
            expires_at=now - timedelta(seconds=1),
        )
        assert store.list_entries(include_expired=False, now=now) == []
        assert store.list_entries(include_expired=True, now=now) != []

        assert store.delete_entry(entry_id=created.entry_id) is True
        assert store.delete_entry(entry_id=created.entry_id) is False

