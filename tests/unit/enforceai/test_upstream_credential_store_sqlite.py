"""
Unit tests for EnforceAI upstream credential store SQLite implementation (Phase 1).
"""

from __future__ import annotations

import sqlite3
from datetime import (
    datetime,
    timezone,
)
from pathlib import Path

import pytest

from auth_server.enforceai.db.migrations import (
    upgrade_to_latest,
)
from auth_server.enforceai.stores.sqlite.upstream_credential_store import (
    SqliteUpstreamCredentialStore,
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
class TestSqliteUpstreamCredentialStore:
    def test_create_list_get_secret_revoke_happy_path(
        self,
        enforceai_sqlite_db_path: Path,
    ) -> None:
        _migrate_db(enforceai_sqlite_db_path)
        store = SqliteUpstreamCredentialStore(
            db_path=enforceai_sqlite_db_path,
            kek=b"\x22" * 32,
        )

        created = store.create_credential(
            server_path="/fininfo",
            credential_type="api-key",
            credential_binding="service",
            secret_payload={"api_key": "super-secret"},
            expires_at=datetime(2030, 1, 1, tzinfo=timezone.utc),
        )
        assert created.server_path == "/fininfo"
        assert created.credential_type == "api-key"
        assert created.credential_binding == "service"
        assert created.expires_at == datetime(2030, 1, 1, tzinfo=timezone.utc)
        assert created.revoked_at is None

        listed = store.list_credentials(server_path="/fininfo")
        assert [record.credential_id for record in listed] == [created.credential_id]

        fetched = store.get_credential_by_id(credential_id=created.credential_id)
        assert fetched == created

        secret = store.get_credential_secret(credential_id=created.credential_id)
        assert secret is not None
        assert secret.payload == {"api_key": "super-secret"}

        revoked = store.revoke_credential(credential_id=created.credential_id)
        assert revoked is not None
        assert revoked.revoked_at is not None

        revoked_again = store.revoke_credential(credential_id=created.credential_id)
        assert revoked_again == revoked

    def test_secret_is_not_stored_in_plaintext(
        self,
        enforceai_sqlite_db_path: Path,
    ) -> None:
        _migrate_db(enforceai_sqlite_db_path)
        store = SqliteUpstreamCredentialStore(
            db_path=enforceai_sqlite_db_path,
            kek=b"\x33" * 32,
        )

        created = store.create_credential(
            server_path="/sre-gateway",
            credential_type="oauth2",
            credential_binding="user",
            user_id="https://issuer.example|sub-1",
            secret_payload={"access_token": "tok-super-secret"},
        )

        connection = sqlite3.connect(enforceai_sqlite_db_path)
        try:
            row = connection.execute(
                """
                SELECT secret_ciphertext
                FROM upstream_credentials
                WHERE credential_id = ?
                """.strip(),
                (created.credential_id,),
            ).fetchone()
        finally:
            connection.close()

        assert row is not None
        ciphertext = row[0]
        assert ciphertext is not None
        assert b"tok-super-secret" not in bytes(ciphertext)

    def test_missing_secret_payload_roundtrips_as_none(
        self,
        enforceai_sqlite_db_path: Path,
    ) -> None:
        _migrate_db(enforceai_sqlite_db_path)
        store = SqliteUpstreamCredentialStore(
            db_path=enforceai_sqlite_db_path,
            kek=b"\x44" * 32,
        )

        created = store.create_credential(
            server_path="/mcp",
            credential_type="header-trust",
            credential_binding="service",
            secret_payload=None,
        )

        secret = store.get_credential_secret(credential_id=created.credential_id)
        assert secret is None

    def test_invalid_binding_fails_fast(
        self,
        enforceai_sqlite_db_path: Path,
    ) -> None:
        _migrate_db(enforceai_sqlite_db_path)
        store = SqliteUpstreamCredentialStore(
            db_path=enforceai_sqlite_db_path,
            kek=b"\x55" * 32,
        )

        with pytest.raises(ValueError, match="user binding requires user_id"):
            store.create_credential(
                server_path="/fininfo",
                credential_type="oauth2",
                credential_binding="user",
            )

        with pytest.raises(ValueError, match="credential_type=none"):
            store.create_credential(
                server_path="/fininfo",
                credential_type="none",
                credential_binding="service",
            )

