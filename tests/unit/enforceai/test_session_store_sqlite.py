"""
Unit tests for EnforceAI SqliteSessionStore.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from auth_server.enforceai.db.data_layer import (
    EnforceAIDataLayer,
)


@pytest.mark.unit
class TestSqliteSessionStore:
    def test_create_touch_revoke_roundtrip(
        self,
        enforceai_sqlite_db_path: Path,
    ) -> None:
        layer = EnforceAIDataLayer(db_path=enforceai_sqlite_db_path)
        layer.initialize()
        stores = layer.build_stores()

        created = stores.session_store.create_session(
            session_id="sess-1",
            user_id="https://issuer.example|sub-1",
            auth_method="oidc",
            expires_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        )
        assert created.session_id == "sess-1"
        assert created.revoked_at is None

        touched = stores.session_store.touch_session(
            session_id="sess-1",
            now=datetime(2025, 1, 2, tzinfo=timezone.utc),
        )
        assert touched is not None
        assert touched.last_seen_at == datetime(2025, 1, 2, tzinfo=timezone.utc)

        revoked = stores.session_store.revoke_session(
            session_id="sess-1",
            revoked_reason="logout",
        )
        assert revoked is not None
        assert revoked.revoked_at is not None
        assert revoked.revoked_reason == "logout"

