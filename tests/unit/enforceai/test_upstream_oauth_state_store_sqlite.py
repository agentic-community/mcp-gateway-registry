"""
Unit tests for EnforceAI upstream OAuth state store SQLite implementation (Phase 5).
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from auth_server.enforceai.db.migrations import (
    upgrade_to_latest,
)
from auth_server.enforceai.stores.sqlite.upstream_oauth_state_store import (
    SqliteUpstreamOAuthStateStore,
)
from auth_server.enforceai.upstream.oauth_flow import (
    consume_oauth_state,
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
class TestSqliteUpstreamOAuthStateStore:
    def test_create_and_consume_state_is_single_use(
        self,
        enforceai_sqlite_db_path: Path,
    ) -> None:
        _migrate_db(enforceai_sqlite_db_path)
        store = SqliteUpstreamOAuthStateStore(
            db_path=enforceai_sqlite_db_path,
            kek=b"\x11" * 32,
        )

        created = store.create_state(
            server_path="/fininfo",
            credential_type="oauth2",
            credential_binding="user",
            user_id="https://issuer.example|user-1",
            agent_id=None,
            provider="github",
            redirect_uri="http://localhost/enforceai/upstream/oauth/callback",
            ui_return_url="/credentials/upstream/oauth/callback",
            ttl_seconds=60,
            secret_payload={"code_verifier": "verifier-1"},
        )

        first = store.consume_state(state_id=created.state_id)
        assert first is not None
        record, secret = first
        assert record.state_id == created.state_id
        assert record.ui_return_url == "/credentials/upstream/oauth/callback"
        assert secret.payload["code_verifier"] == "verifier-1"

        second = store.consume_state(state_id=created.state_id)
        assert second is None

    def test_consume_oauth_state_binds_to_user_id(
        self,
        enforceai_sqlite_db_path: Path,
    ) -> None:
        _migrate_db(enforceai_sqlite_db_path)
        store = SqliteUpstreamOAuthStateStore(
            db_path=enforceai_sqlite_db_path,
            kek=b"\x22" * 32,
        )

        created = store.create_state(
            server_path="/fininfo",
            credential_type="provider-oauth",
            credential_binding="user",
            user_id="https://issuer.example|user-1",
            agent_id=None,
            provider="github",
            redirect_uri="http://localhost/enforceai/upstream/oauth/callback",
            ui_return_url=None,
            ttl_seconds=60,
            secret_payload={"code_verifier": "verifier-2"},
        )

        with pytest.raises(ValueError, match="does not match current user"):
            consume_oauth_state(
                state_store=store,
                state_id=created.state_id,
                actor_user_id="https://issuer.example|user-2",
            )

        # The state is consumed (deleted) even on mismatch to prevent replay.
        assert store.consume_state(state_id=created.state_id) is None

    def test_expired_state_is_rejected(
        self,
        enforceai_sqlite_db_path: Path,
    ) -> None:
        _migrate_db(enforceai_sqlite_db_path)
        store = SqliteUpstreamOAuthStateStore(
            db_path=enforceai_sqlite_db_path,
            kek=b"\x33" * 32,
        )

        created = store.create_state(
            server_path="/fininfo",
            credential_type="oidc",
            credential_binding="user",
            user_id="https://issuer.example|user-1",
            agent_id=None,
            provider="github",
            redirect_uri="http://localhost/enforceai/upstream/oauth/callback",
            ui_return_url=None,
            ttl_seconds=-1,
            secret_payload={"code_verifier": "verifier-3"},
        )

        assert store.consume_state(state_id=created.state_id) is None
