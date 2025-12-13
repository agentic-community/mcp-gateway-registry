"""
Unit tests for EnforceAI RevocationStore SQLite implementation (Stage 1.4).
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
from auth_server.enforceai.stores.sqlite.revocation_store import (
    SqliteRevocationStore,
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
class TestSqliteRevocationStore:
    def test_revoke_and_check(
        self,
        enforceai_sqlite_db_path: Path,
    ) -> None:
        _migrate_db(enforceai_sqlite_db_path)
        store = SqliteRevocationStore(db_path=enforceai_sqlite_db_path)

        jti = "jti-1"
        user_id = "https://issuer.example|sub-1"
        agent_id = str(uuid.uuid4())

        assert store.is_jti_revoked(jti=jti) is False

        record = store.revoke_jti(
            jti=jti,
            user_id=user_id,
            agent_id=agent_id,
            reason="manual",
        )
        assert record.jti == jti
        assert record.user_id == user_id
        assert record.agent_id == agent_id
        assert record.reason == "manual"

        assert store.is_jti_revoked(jti=jti) is True

    def test_revoke_is_idempotent(
        self,
        enforceai_sqlite_db_path: Path,
    ) -> None:
        _migrate_db(enforceai_sqlite_db_path)
        store = SqliteRevocationStore(db_path=enforceai_sqlite_db_path)

        agent_id = str(uuid.uuid4())
        first = store.revoke_jti(
            jti="jti-1",
            user_id="https://issuer.example|sub-1",
            agent_id=agent_id,
            reason="first",
        )
        second = store.revoke_jti(
            jti="jti-1",
            user_id="https://issuer.example|sub-1",
            agent_id=agent_id,
            reason="second",
        )
        assert second == first

    def test_check_respects_expires_at(
        self,
        enforceai_sqlite_db_path: Path,
    ) -> None:
        _migrate_db(enforceai_sqlite_db_path)
        store = SqliteRevocationStore(db_path=enforceai_sqlite_db_path)

        now = datetime(2025, 1, 1, tzinfo=timezone.utc)
        agent_id = str(uuid.uuid4())

        store.revoke_jti(
            jti="jti-expired",
            user_id="https://issuer.example|sub-1",
            agent_id=agent_id,
            expires_at=now - timedelta(seconds=1),
        )
        assert store.is_jti_revoked(
            jti="jti-expired",
            now=now,
        ) is False

        store.revoke_jti(
            jti="jti-active",
            user_id="https://issuer.example|sub-1",
            agent_id=agent_id,
            expires_at=now + timedelta(seconds=10),
        )
        assert store.is_jti_revoked(
            jti="jti-active",
            now=now,
        ) is True

    def test_list_by_agent_id(
        self,
        enforceai_sqlite_db_path: Path,
    ) -> None:
        _migrate_db(enforceai_sqlite_db_path)
        store = SqliteRevocationStore(db_path=enforceai_sqlite_db_path)

        agent_a = str(uuid.uuid4())
        agent_b = str(uuid.uuid4())

        store.revoke_jti(
            jti="jti-a1",
            user_id="https://issuer.example|sub-1",
            agent_id=agent_a,
        )
        store.revoke_jti(
            jti="jti-a2",
            user_id="https://issuer.example|sub-1",
            agent_id=agent_a,
        )
        store.revoke_jti(
            jti="jti-b1",
            user_id="https://issuer.example|sub-1",
            agent_id=agent_b,
        )

        listed = store.list_revocations_by_agent_id(agent_id=agent_a)
        assert [record.jti for record in listed] == ["jti-a1", "jti-a2"]

    def test_delete_expired_revocations(
        self,
        enforceai_sqlite_db_path: Path,
    ) -> None:
        _migrate_db(enforceai_sqlite_db_path)
        store = SqliteRevocationStore(db_path=enforceai_sqlite_db_path)

        now = datetime(2025, 1, 1, tzinfo=timezone.utc)
        agent_id = str(uuid.uuid4())

        store.revoke_jti(
            jti="jti-expired",
            user_id="https://issuer.example|sub-1",
            agent_id=agent_id,
            expires_at=now - timedelta(days=1),
        )
        store.revoke_jti(
            jti="jti-no-expiry",
            user_id="https://issuer.example|sub-1",
            agent_id=agent_id,
            expires_at=None,
        )
        store.revoke_jti(
            jti="jti-future",
            user_id="https://issuer.example|sub-1",
            agent_id=agent_id,
            expires_at=now + timedelta(days=1),
        )

        deleted = store.delete_expired_revocations(now=now)
        assert deleted == 1

        assert store.is_jti_revoked(jti="jti-expired", now=now) is False
        assert store.is_jti_revoked(jti="jti-no-expiry", now=now) is True
        assert store.is_jti_revoked(jti="jti-future", now=now) is True

