"""
End-to-end smoke test for the EnforceAI data layer (Stage 1.6).
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from pathlib import Path

import pytest

from auth_server.enforceai.db.connection import (
    sqlite_connection,
)
from auth_server.enforceai.db.data_layer import (
    EnforceAIDataLayer,
)


@pytest.mark.unit
class TestEnforceAIDataLayerSmoke:
    def test_data_layer_persists_across_reopen(
        self,
        enforceai_sqlite_db_path: Path,
    ) -> None:
        layer = EnforceAIDataLayer(db_path=enforceai_sqlite_db_path)
        layer.initialize()
        stores = layer.build_stores()

        user_id = "https://issuer.example|sub-1"
        agent_id = str(uuid.uuid4())

        created_agent = stores.agent_store.create_agent(
            user_id=user_id,
            agent_id=agent_id,
            scopes=["mcp-servers-restricted/read"],
            allowed_tools=["tools/list"],
            alias="agent-1",
        )
        assert created_agent.agent_id == agent_id

        created_key = stores.api_key_store.create_key(
            key_id="key-1",
            secret_hash="hash-1",
            user_id=user_id,
            agent_id=agent_id,
            scopes=None,
        )
        assert created_key.key_id == "key-1"

        stores.revocation_store.revoke_jti(
            jti="jti-1",
            user_id=user_id,
            agent_id=agent_id,
            reason="test",
        )
        assert stores.revocation_store.is_jti_revoked(jti="jti-1") is True

        event = stores.audit_store.append_event(
            occurred_at=datetime(2025, 1, 1, tzinfo=timezone.utc),
            user_id=user_id,
            agent_id=agent_id,
            action="tools/list",
            outcome="allow",
            details={"path": "/mcp"},
        )
        assert event.event_id >= 1

        reopened = EnforceAIDataLayer(db_path=enforceai_sqlite_db_path)
        reopened.initialize()
        reopened_stores = reopened.build_stores()

        loaded_agent = reopened_stores.agent_store.get_agent_by_id(agent_id=agent_id)
        assert loaded_agent is not None
        assert loaded_agent.alias == "agent-1"

        loaded_key = reopened_stores.api_key_store.get_key_by_id(key_id="key-1")
        assert loaded_key is not None
        assert loaded_key.secret_hash == "hash-1"

        assert reopened_stores.revocation_store.is_jti_revoked(jti="jti-1") is True

        recent = reopened_stores.audit_store.list_recent_events(
            user_id=user_id,
            limit=10,
        )
        assert recent[0].details == {"path": "/mcp"}

    def test_sqlite_pragmas_are_applied(
        self,
        enforceai_sqlite_db_path: Path,
    ) -> None:
        with sqlite_connection(enforceai_sqlite_db_path) as connection:
            row = connection.execute("PRAGMA foreign_keys").fetchone()
            assert row is not None
            assert row[0] == 1

