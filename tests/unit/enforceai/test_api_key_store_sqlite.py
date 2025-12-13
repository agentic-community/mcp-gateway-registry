"""
Unit tests for EnforceAI ApiKeyStore SQLite implementation (Stage 1.3).
"""

from __future__ import annotations

import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path

import pytest

from auth_server.enforceai.db.migrations import (
    upgrade_to_latest,
)
from auth_server.enforceai.stores.sqlite.api_key_store import (
    SqliteApiKeyStore,
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
class TestSqliteApiKeyStore:
    def test_create_get_revoke_happy_path(
        self,
        enforceai_sqlite_db_path: Path,
    ) -> None:
        _migrate_db(enforceai_sqlite_db_path)
        store = SqliteApiKeyStore(db_path=enforceai_sqlite_db_path)

        user_id = "https://issuer.example|sub-1"
        agent_id = str(uuid.uuid4())

        created = store.create_key(
            key_id="key-1",
            secret_hash="hash-abc",
            user_id=user_id,
            agent_id=agent_id,
            scopes=["mcp-servers-restricted/read"],
            expires_at=datetime(2030, 1, 1, tzinfo=timezone.utc),
        )
        assert created.key_id == "key-1"
        assert created.secret_hash == "hash-abc"
        assert created.user_id == user_id
        assert created.agent_id == agent_id
        assert created.scopes == ["mcp-servers-restricted/read"]
        assert created.expires_at == datetime(2030, 1, 1, tzinfo=timezone.utc)
        assert created.revoked_at is None
        assert created.last_used_at is None

        fetched = store.get_key_by_id(key_id="key-1")
        assert fetched == created

        revoked = store.revoke_key(key_id="key-1")
        assert revoked is not None
        assert revoked.revoked_at is not None

        revoked_again = store.revoke_key(key_id="key-1")
        assert revoked_again == revoked

    def test_scope_optional_behavior(
        self,
        enforceai_sqlite_db_path: Path,
    ) -> None:
        _migrate_db(enforceai_sqlite_db_path)
        store = SqliteApiKeyStore(db_path=enforceai_sqlite_db_path)

        record = store.create_key(
            key_id="key-1",
            secret_hash="hash-abc",
            user_id="https://issuer.example|sub-1",
            agent_id=str(uuid.uuid4()),
            scopes=None,
        )
        assert record.scopes is None

        fetched = store.get_key_by_id(key_id="key-1")
        assert fetched is not None
        assert fetched.scopes is None

    def test_last_used_at_update_does_not_log_secret(
        self,
        caplog: pytest.LogCaptureFixture,
        enforceai_sqlite_db_path: Path,
    ) -> None:
        _migrate_db(enforceai_sqlite_db_path)
        store = SqliteApiKeyStore(db_path=enforceai_sqlite_db_path)

        store.create_key(
            key_id="key-1",
            secret_hash="hash-super-secret",
            user_id="https://issuer.example|sub-1",
            agent_id=str(uuid.uuid4()),
        )

        caplog.clear()
        updated = store.update_last_used_at(
            key_id="key-1",
            last_used_at=datetime(2025, 1, 1, tzinfo=timezone.utc),
        )
        assert updated is not None
        assert updated.last_used_at == datetime(2025, 1, 1, tzinfo=timezone.utc)

        combined = "\n".join(record.message for record in caplog.records)
        assert "hash-super-secret" not in combined

    def test_list_keys_filters(
        self,
        enforceai_sqlite_db_path: Path,
    ) -> None:
        _migrate_db(enforceai_sqlite_db_path)
        store = SqliteApiKeyStore(db_path=enforceai_sqlite_db_path)

        user_a = "https://issuer.example|a"
        user_b = "https://issuer.example|b"
        agent_a = str(uuid.uuid4())
        agent_b = str(uuid.uuid4())

        key_a1 = store.create_key(
            key_id="key-a1",
            secret_hash="hash-a1",
            user_id=user_a,
            agent_id=agent_a,
        )
        store.create_key(
            key_id="key-a2",
            secret_hash="hash-a2",
            user_id=user_a,
            agent_id=agent_b,
        )
        store.create_key(
            key_id="key-b1",
            secret_hash="hash-b1",
            user_id=user_b,
            agent_id=agent_a,
        )

        with pytest.raises(ValueError, match="At least one of user_id"):
            store.list_keys()

        by_user = store.list_keys(user_id=user_a)
        assert [record.key_id for record in by_user] == ["key-a1", "key-a2"]

        by_agent = store.list_keys(agent_id=agent_a)
        assert [record.key_id for record in by_agent] == ["key-a1", "key-b1"]

        by_both = store.list_keys(user_id=user_a, agent_id=agent_a)
        assert by_both == [key_a1]

    def test_invalid_inputs_fail_fast(
        self,
        enforceai_sqlite_db_path: Path,
    ) -> None:
        _migrate_db(enforceai_sqlite_db_path)
        store = SqliteApiKeyStore(db_path=enforceai_sqlite_db_path)

        with pytest.raises(ValueError, match="user_id must be in"):
            store.create_key(
                key_id="key-1",
                secret_hash="hash",
                user_id="bad-user",
                agent_id=str(uuid.uuid4()),
            )

        with pytest.raises(ValueError, match="agent_id must be a UUIDv4"):
            store.create_key(
                key_id="key-1",
                secret_hash="hash",
                user_id="https://issuer.example|sub-1",
                agent_id="not-a-uuid",
            )

        with pytest.raises(ValueError, match="scopes must not contain empty"):
            store.create_key(
                key_id="key-1",
                secret_hash="hash",
                user_id="https://issuer.example|sub-1",
                agent_id=str(uuid.uuid4()),
                scopes=[" "],
            )

    def test_create_duplicate_key_fails(
        self,
        enforceai_sqlite_db_path: Path,
    ) -> None:
        _migrate_db(enforceai_sqlite_db_path)
        store = SqliteApiKeyStore(db_path=enforceai_sqlite_db_path)

        store.create_key(
            key_id="key-1",
            secret_hash="hash-abc",
            user_id="https://issuer.example|sub-1",
            agent_id=str(uuid.uuid4()),
        )

        with pytest.raises(ValueError, match="API key already exists"):
            store.create_key(
                key_id="key-1",
                secret_hash="hash-abc",
                user_id="https://issuer.example|sub-1",
                agent_id=str(uuid.uuid4()),
            )

