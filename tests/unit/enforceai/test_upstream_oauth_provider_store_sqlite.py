"""
Unit tests for EnforceAI upstream OAuth provider registry SQLite store (Phase 1).
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from auth_server.enforceai.db.migrations import (
    upgrade_to_latest,
)
from auth_server.enforceai.models.upstream_oauth_provider import (
    UpstreamOAuthProviderCreate,
    UpstreamOAuthProviderUpdate,
)
from auth_server.enforceai.stores.sqlite.upstream_oauth_provider_store import (
    SqliteUpstreamOAuthProviderStore,
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
class TestSqliteUpstreamOAuthProviderStore:
    def test_create_list_get_update_delete_happy_path(
        self,
        enforceai_sqlite_db_path: Path,
    ) -> None:
        _migrate_db(enforceai_sqlite_db_path)
        store = SqliteUpstreamOAuthProviderStore(
            db_path=enforceai_sqlite_db_path,
            kek=b"\x66" * 32,
        )

        created = store.create_provider(
            payload=UpstreamOAuthProviderCreate(
                provider_id="github",
                authorization_endpoint="https://example.com/auth",
                token_endpoint="https://example.com/token",
                client_id="client",
                client_secret="secret-1",
                default_scopes=["repo", "user:email"],
            )
        )
        assert created.provider.provider_id == "github"
        assert created.secret_present is True

        listed = store.list_providers()
        assert [item.provider.provider_id for item in listed] == ["github"]

        fetched = store.get_provider(provider_id="github")
        assert fetched == created

        secret = store.get_provider_secret_for_runtime(provider_id="github")
        assert secret == "secret-1"

        updated = store.update_provider(
            provider_id="github",
            payload=UpstreamOAuthProviderUpdate(
                client_secret="secret-2",
                extra_authorize_params={"prompt": "consent"},
            ),
        )
        assert updated is not None
        assert updated.provider.extra_authorize_params == {"prompt": "consent"}
        assert store.get_provider_secret_for_runtime(provider_id="github") == "secret-2"

        deleted = store.delete_provider(provider_id="github")
        assert deleted is True
        assert store.get_provider(provider_id="github") is None

    def test_secret_is_not_stored_in_plaintext(
        self,
        enforceai_sqlite_db_path: Path,
    ) -> None:
        _migrate_db(enforceai_sqlite_db_path)
        store = SqliteUpstreamOAuthProviderStore(
            db_path=enforceai_sqlite_db_path,
            kek=b"\x77" * 32,
        )

        created = store.create_provider(
            payload=UpstreamOAuthProviderCreate(
                provider_id="google",
                authorization_endpoint="https://example.com/auth",
                token_endpoint="https://example.com/token",
                client_id="client",
                client_secret="super-secret",
            )
        )

        connection = sqlite3.connect(enforceai_sqlite_db_path)
        try:
            row = connection.execute(
                """
                SELECT secret_ciphertext
                FROM upstream_oauth_providers
                WHERE provider_id = ?
                """.strip(),
                (created.provider.provider_id,),
            ).fetchone()
        finally:
            connection.close()

        assert row is not None
        ciphertext = row[0]
        assert ciphertext is not None
        assert b"super-secret" not in bytes(ciphertext)

