"""
Unit tests for EnforceAI AgentStore SQLite implementation (Stage 1.2).
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
from auth_server.enforceai.stores.sqlite.agent_store import (
    SqliteAgentStore,
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
class TestSqliteAgentStore:
    def test_crud_happy_path(
        self,
        enforceai_sqlite_db_path: Path,
    ) -> None:
        _migrate_db(enforceai_sqlite_db_path)
        store = SqliteAgentStore(db_path=enforceai_sqlite_db_path)

        user_id = "https://issuer.example|sub-1"
        agent_id = str(uuid.uuid4())

        created = store.create_agent(
            user_id=user_id,
            agent_id=agent_id,
            scopes=["mcp-servers-restricted/read"],
            allowed_tools=["tools/list"],
            alias="my-agent",
            metadata={"env": "test"},
        )
        assert created.user_id == user_id
        assert created.agent_id == agent_id
        assert created.scopes == ["mcp-servers-restricted/read"]
        assert created.allowed_tools == ["tools/list"]
        assert created.alias == "my-agent"
        assert created.metadata == {"env": "test"}
        assert created.revoked_at is None
        assert created.tokens_valid_after is None

        fetched = store.get_agent_by_id(agent_id=agent_id)
        assert fetched == created

        updated = store.update_agent(
            agent_id=agent_id,
            scopes=["mcp-servers-unrestricted/read"],
            allowed_tools=["tools/list", "tools/call"],
            alias="renamed",
            metadata={"env": "test", "version": 2},
        )
        assert updated is not None
        assert updated.scopes == ["mcp-servers-unrestricted/read"]
        assert updated.allowed_tools == ["tools/list", "tools/call"]
        assert updated.alias == "renamed"
        assert updated.metadata == {"env": "test", "version": 2}
        assert updated.created_at == created.created_at
        assert updated.updated_at >= created.updated_at

    def test_list_by_user_id_is_isolated(
        self,
        enforceai_sqlite_db_path: Path,
    ) -> None:
        _migrate_db(enforceai_sqlite_db_path)
        store = SqliteAgentStore(db_path=enforceai_sqlite_db_path)

        user_a = "https://issuer.example|a"
        user_b = "https://issuer.example|b"

        agent_a1 = store.create_agent(
            user_id=user_a,
            agent_id=str(uuid.uuid4()),
            scopes=["s1"],
        )
        agent_a2 = store.create_agent(
            user_id=user_a,
            agent_id=str(uuid.uuid4()),
            scopes=["s2"],
        )
        store.create_agent(
            user_id=user_b,
            agent_id=str(uuid.uuid4()),
            scopes=["s3"],
        )

        listed = store.list_agents_by_user_id(user_id=user_a)
        assert [record.agent_id for record in listed] == [
            agent_a1.agent_id,
            agent_a2.agent_id,
        ]

    def test_revoke_is_idempotent_and_persists(
        self,
        enforceai_sqlite_db_path: Path,
    ) -> None:
        _migrate_db(enforceai_sqlite_db_path)
        store = SqliteAgentStore(db_path=enforceai_sqlite_db_path)

        record = store.create_agent(
            user_id="https://issuer.example|sub-1",
            agent_id=str(uuid.uuid4()),
            scopes=["s1"],
        )

        revoked = store.revoke_agent(agent_id=record.agent_id)
        assert revoked is not None
        assert revoked.revoked_at is not None

        revoked_again = store.revoke_agent(agent_id=record.agent_id)
        assert revoked_again == revoked

        fetched = store.get_agent_by_id(agent_id=record.agent_id)
        assert fetched is not None
        assert fetched.revoked_at == revoked.revoked_at

    def test_tokens_valid_after_bump_is_monotonic(
        self,
        enforceai_sqlite_db_path: Path,
    ) -> None:
        _migrate_db(enforceai_sqlite_db_path)
        store = SqliteAgentStore(db_path=enforceai_sqlite_db_path)

        record = store.create_agent(
            user_id="https://issuer.example|sub-1",
            agent_id=str(uuid.uuid4()),
            scopes=["s1"],
        )
        t1 = datetime(2025, 1, 1, tzinfo=timezone.utc)
        t2 = datetime(2025, 1, 2, tzinfo=timezone.utc)

        bumped = store.bump_tokens_valid_after(
            agent_id=record.agent_id,
            tokens_valid_after=t1,
        )
        assert bumped is not None
        assert bumped.tokens_valid_after == t1

        bumped_earlier = store.bump_tokens_valid_after(
            agent_id=record.agent_id,
            tokens_valid_after=datetime(2024, 12, 31, tzinfo=timezone.utc),
        )
        assert bumped_earlier is not None
        assert bumped_earlier.tokens_valid_after == t1

        bumped_later = store.bump_tokens_valid_after(
            agent_id=record.agent_id,
            tokens_valid_after=t2,
        )
        assert bumped_later is not None
        assert bumped_later.tokens_valid_after == t2

    def test_invalid_inputs_fail_fast(
        self,
        enforceai_sqlite_db_path: Path,
    ) -> None:
        _migrate_db(enforceai_sqlite_db_path)
        store = SqliteAgentStore(db_path=enforceai_sqlite_db_path)

        with pytest.raises(ValueError, match="user_id must be in"):
            store.create_agent(
                user_id="not-a-user-id",
                agent_id=str(uuid.uuid4()),
                scopes=["s1"],
            )

        with pytest.raises(ValueError, match="agent_id must be a UUIDv4"):
            store.create_agent(
                user_id="https://issuer.example|sub-1",
                agent_id="not-a-uuid",
                scopes=["s1"],
            )

        with pytest.raises(ValueError, match="scopes must not contain empty"):
            store.create_agent(
                user_id="https://issuer.example|sub-1",
                agent_id=str(uuid.uuid4()),
                scopes=["  "],
            )

    def test_create_duplicate_agent_fails(
        self,
        enforceai_sqlite_db_path: Path,
    ) -> None:
        _migrate_db(enforceai_sqlite_db_path)
        store = SqliteAgentStore(db_path=enforceai_sqlite_db_path)

        agent_id = str(uuid.uuid4())
        store.create_agent(
            user_id="https://issuer.example|sub-1",
            agent_id=agent_id,
            scopes=["s1"],
        )

        with pytest.raises(ValueError, match="Agent already exists"):
            store.create_agent(
                user_id="https://issuer.example|sub-1",
                agent_id=agent_id,
                scopes=["s1"],
            )

