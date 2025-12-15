"""
Unit tests for EnforceAI SqliteUserStore.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from auth_server.enforceai.db.data_layer import (
    EnforceAIDataLayer,
)
from auth_server.enforceai.users.passwords import (
    hash_password,
    verify_password,
)


@pytest.mark.unit
class TestSqliteUserStore:
    def test_create_local_user_and_lookup(
        self,
        enforceai_sqlite_db_path: Path,
    ) -> None:
        layer = EnforceAIDataLayer(db_path=enforceai_sqlite_db_path)
        layer.initialize()
        stores = layer.build_stores()

        password_hash = hash_password("pw-1", salt=b"0" * 16).encoded
        created = stores.user_store.create_local_user(
            username="alice",
            email="alice@example.com",
            password_hash=password_hash,
            role="user",
        )
        assert created.user_id == "local|alice"
        assert created.auth_method == "password"
        assert created.username == "alice"
        assert created.email == "alice@example.com"
        assert created.password_hash == password_hash
        assert created.disabled_at is None

        loaded = stores.user_store.get_user_by_username(username="alice")
        assert loaded is not None
        assert loaded.user_id == "local|alice"
        assert verify_password("pw-1", loaded.password_hash or "") is True

    def test_upsert_oidc_user_updates_last_login_and_email(
        self,
        enforceai_sqlite_db_path: Path,
    ) -> None:
        layer = EnforceAIDataLayer(db_path=enforceai_sqlite_db_path)
        layer.initialize()
        stores = layer.build_stores()

        user_id = "https://issuer.example|sub-1"
        first = stores.user_store.upsert_oidc_user(
            user_id=user_id,
            email="first@example.com",
        )
        assert first.user_id == user_id
        assert first.auth_method == "oidc"
        assert first.email == "first@example.com"

        second = stores.user_store.upsert_oidc_user(
            user_id=user_id,
            email="second@example.com",
        )
        assert second.email == "second@example.com"
        assert second.updated_at >= first.updated_at

    def test_search_and_disable_user(
        self,
        enforceai_sqlite_db_path: Path,
    ) -> None:
        layer = EnforceAIDataLayer(db_path=enforceai_sqlite_db_path)
        layer.initialize()
        stores = layer.build_stores()

        stores.user_store.upsert_oidc_user(
            user_id="https://issuer.example|sub-1",
            email="bob@example.com",
        )
        password_hash = hash_password("pw-2", salt=b"1" * 16).encoded
        stores.user_store.create_local_user(
            username="alice",
            email="alice@example.com",
            password_hash=password_hash,
        )

        matches = stores.user_store.search_users(query="example.com", limit=10)
        assert len(matches) == 2

        disabled = stores.user_store.disable_user(user_id="local|alice")
        assert disabled is not None
        assert disabled.disabled_at is not None

