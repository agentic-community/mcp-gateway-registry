"""
Unit tests for EnforceAI AuditStore query_events method.
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
from auth_server.enforceai.stores.sqlite.audit_store import (
    SqliteAuditStore,
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
class TestAuditStoreQueryEvents:
    """Tests for SqliteAuditStore.query_events()"""

    def test_query_events_returns_events_in_descending_order(
        self,
        enforceai_sqlite_db_path: Path,
    ) -> None:
        _migrate_db(enforceai_sqlite_db_path)
        store = SqliteAuditStore(db_path=enforceai_sqlite_db_path)

        user_id = "https://issuer.example|sub-1"
        agent_id = str(uuid.uuid4())
        base_time = datetime(2025, 1, 1, tzinfo=timezone.utc)

        for i in range(5):
            store.append_event(
                occurred_at=base_time + timedelta(seconds=i),
                user_id=user_id,
                agent_id=agent_id,
                action="tools/list",
                outcome="allow",
                request_id=f"req-{i}",
            )

        result = store.query_events(user_id=user_id, limit=10)

        assert len(result.items) == 5
        event_ids = [e.event_id for e in result.items]
        assert event_ids == sorted(event_ids, reverse=True)
        assert result.next_cursor is None

    def test_query_events_filters_by_user_id(
        self,
        enforceai_sqlite_db_path: Path,
    ) -> None:
        _migrate_db(enforceai_sqlite_db_path)
        store = SqliteAuditStore(db_path=enforceai_sqlite_db_path)

        user1 = "https://issuer.example|user-1"
        user2 = "https://issuer.example|user-2"
        agent_id = str(uuid.uuid4())
        t0 = datetime(2025, 1, 1, tzinfo=timezone.utc)

        store.append_event(
            occurred_at=t0,
            user_id=user1,
            agent_id=agent_id,
            action="tools/list",
            outcome="allow",
        )
        store.append_event(
            occurred_at=t0 + timedelta(seconds=1),
            user_id=user2,
            agent_id=agent_id,
            action="tools/call",
            outcome="deny",
        )

        result = store.query_events(user_id=user1, limit=10)

        assert len(result.items) == 1
        assert result.items[0].user_id == user1

    def test_query_events_filters_by_agent_id(
        self,
        enforceai_sqlite_db_path: Path,
    ) -> None:
        _migrate_db(enforceai_sqlite_db_path)
        store = SqliteAuditStore(db_path=enforceai_sqlite_db_path)

        user_id = "https://issuer.example|user-1"
        agent1 = str(uuid.uuid4())
        agent2 = str(uuid.uuid4())
        t0 = datetime(2025, 1, 1, tzinfo=timezone.utc)

        store.append_event(
            occurred_at=t0,
            user_id=user_id,
            agent_id=agent1,
            action="tools/list",
            outcome="allow",
        )
        store.append_event(
            occurred_at=t0 + timedelta(seconds=1),
            user_id=user_id,
            agent_id=agent2,
            action="tools/call",
            outcome="deny",
        )

        result = store.query_events(user_id=user_id, agent_id=agent1, limit=10)

        assert len(result.items) == 1
        assert result.items[0].agent_id == agent1

    def test_query_events_filters_by_actions(
        self,
        enforceai_sqlite_db_path: Path,
    ) -> None:
        _migrate_db(enforceai_sqlite_db_path)
        store = SqliteAuditStore(db_path=enforceai_sqlite_db_path)

        user_id = "https://issuer.example|user-1"
        agent_id = str(uuid.uuid4())
        t0 = datetime(2025, 1, 1, tzinfo=timezone.utc)

        store.append_event(
            occurred_at=t0,
            user_id=user_id,
            agent_id=agent_id,
            action="tools/list",
            outcome="allow",
        )
        store.append_event(
            occurred_at=t0 + timedelta(seconds=1),
            user_id=user_id,
            agent_id=agent_id,
            action="tools/call",
            outcome="allow",
        )
        store.append_event(
            occurred_at=t0 + timedelta(seconds=2),
            user_id=user_id,
            agent_id=agent_id,
            action="management/agents/create",
            outcome="allow",
        )

        result = store.query_events(
            user_id=user_id,
            actions=["tools/list", "tools/call"],
            limit=10,
        )

        assert len(result.items) == 2
        actions = {e.action for e in result.items}
        assert actions == {"tools/list", "tools/call"}

    def test_query_events_filters_by_outcomes(
        self,
        enforceai_sqlite_db_path: Path,
    ) -> None:
        _migrate_db(enforceai_sqlite_db_path)
        store = SqliteAuditStore(db_path=enforceai_sqlite_db_path)

        user_id = "https://issuer.example|user-1"
        agent_id = str(uuid.uuid4())
        t0 = datetime(2025, 1, 1, tzinfo=timezone.utc)

        store.append_event(
            occurred_at=t0,
            user_id=user_id,
            agent_id=agent_id,
            action="tools/list",
            outcome="allow",
        )
        store.append_event(
            occurred_at=t0 + timedelta(seconds=1),
            user_id=user_id,
            agent_id=agent_id,
            action="tools/call",
            outcome="deny",
        )

        result = store.query_events(
            user_id=user_id,
            outcomes=["deny"],
            limit=10,
        )

        assert len(result.items) == 1
        assert result.items[0].outcome == "deny"

    def test_query_events_filters_by_request_id(
        self,
        enforceai_sqlite_db_path: Path,
    ) -> None:
        _migrate_db(enforceai_sqlite_db_path)
        store = SqliteAuditStore(db_path=enforceai_sqlite_db_path)

        user_id = "https://issuer.example|user-1"
        agent_id = str(uuid.uuid4())
        t0 = datetime(2025, 1, 1, tzinfo=timezone.utc)

        store.append_event(
            occurred_at=t0,
            user_id=user_id,
            agent_id=agent_id,
            action="tools/list",
            outcome="allow",
            request_id="req-abc",
        )
        store.append_event(
            occurred_at=t0 + timedelta(seconds=1),
            user_id=user_id,
            agent_id=agent_id,
            action="tools/call",
            outcome="allow",
            request_id="req-xyz",
        )

        result = store.query_events(
            user_id=user_id,
            request_id="req-abc",
            limit=10,
        )

        assert len(result.items) == 1
        assert result.items[0].request_id == "req-abc"

    def test_query_events_filters_by_server_in_details(
        self,
        enforceai_sqlite_db_path: Path,
    ) -> None:
        _migrate_db(enforceai_sqlite_db_path)
        store = SqliteAuditStore(db_path=enforceai_sqlite_db_path)

        user_id = "https://issuer.example|user-1"
        agent_id = str(uuid.uuid4())
        t0 = datetime(2025, 1, 1, tzinfo=timezone.utc)

        store.append_event(
            occurred_at=t0,
            user_id=user_id,
            agent_id=agent_id,
            action="tools/call",
            outcome="allow",
            details={"server": "sqlite", "tool": "query"},
        )
        store.append_event(
            occurred_at=t0 + timedelta(seconds=1),
            user_id=user_id,
            agent_id=agent_id,
            action="tools/call",
            outcome="allow",
            details={"server": "github", "tool": "create_issue"},
        )

        result = store.query_events(
            user_id=user_id,
            server="sqlite",
            limit=10,
        )

        assert len(result.items) == 1
        assert result.items[0].details["server"] == "sqlite"

    def test_query_events_filters_by_tool_in_details(
        self,
        enforceai_sqlite_db_path: Path,
    ) -> None:
        _migrate_db(enforceai_sqlite_db_path)
        store = SqliteAuditStore(db_path=enforceai_sqlite_db_path)

        user_id = "https://issuer.example|user-1"
        agent_id = str(uuid.uuid4())
        t0 = datetime(2025, 1, 1, tzinfo=timezone.utc)

        store.append_event(
            occurred_at=t0,
            user_id=user_id,
            agent_id=agent_id,
            action="tools/call",
            outcome="allow",
            details={"server": "sqlite", "tool": "query"},
        )
        store.append_event(
            occurred_at=t0 + timedelta(seconds=1),
            user_id=user_id,
            agent_id=agent_id,
            action="tools/call",
            outcome="allow",
            details={"server": "sqlite", "tool": "execute"},
        )

        result = store.query_events(
            user_id=user_id,
            tool="query",
            limit=10,
        )

        assert len(result.items) == 1
        assert result.items[0].details["tool"] == "query"

    def test_query_events_filters_by_time_range(
        self,
        enforceai_sqlite_db_path: Path,
    ) -> None:
        _migrate_db(enforceai_sqlite_db_path)
        store = SqliteAuditStore(db_path=enforceai_sqlite_db_path)

        user_id = "https://issuer.example|user-1"
        agent_id = str(uuid.uuid4())
        base_time = datetime(2025, 1, 1, tzinfo=timezone.utc)

        for i in range(10):
            store.append_event(
                occurred_at=base_time + timedelta(hours=i),
                user_id=user_id,
                agent_id=agent_id,
                action="tools/list",
                outcome="allow",
                request_id=f"req-{i}",
            )

        result = store.query_events(
            user_id=user_id,
            since=base_time + timedelta(hours=3),
            until=base_time + timedelta(hours=6),
            limit=10,
        )

        assert len(result.items) == 4
        request_ids = [e.request_id for e in result.items]
        assert set(request_ids) == {"req-3", "req-4", "req-5", "req-6"}

    def test_query_events_pagination_with_cursor(
        self,
        enforceai_sqlite_db_path: Path,
    ) -> None:
        _migrate_db(enforceai_sqlite_db_path)
        store = SqliteAuditStore(db_path=enforceai_sqlite_db_path)

        user_id = "https://issuer.example|user-1"
        agent_id = str(uuid.uuid4())
        base_time = datetime(2025, 1, 1, tzinfo=timezone.utc)

        for i in range(10):
            store.append_event(
                occurred_at=base_time + timedelta(seconds=i),
                user_id=user_id,
                agent_id=agent_id,
                action="tools/list",
                outcome="allow",
                request_id=f"req-{i}",
            )

        page1 = store.query_events(user_id=user_id, limit=3)

        assert len(page1.items) == 3
        assert page1.next_cursor is not None
        first_page_ids = [e.event_id for e in page1.items]

        page2 = store.query_events(
            user_id=user_id,
            limit=3,
            cursor=page1.next_cursor,
        )

        assert len(page2.items) == 3
        assert page2.next_cursor is not None
        second_page_ids = [e.event_id for e in page2.items]

        assert set(first_page_ids).isdisjoint(set(second_page_ids))
        assert max(second_page_ids) < min(first_page_ids)

        page3 = store.query_events(
            user_id=user_id,
            limit=3,
            cursor=page2.next_cursor,
        )

        assert len(page3.items) == 3
        assert page3.next_cursor is not None

        page4 = store.query_events(
            user_id=user_id,
            limit=3,
            cursor=page3.next_cursor,
        )

        assert len(page4.items) == 1
        assert page4.next_cursor is None

    def test_query_events_limit_capped_at_500(
        self,
        enforceai_sqlite_db_path: Path,
    ) -> None:
        _migrate_db(enforceai_sqlite_db_path)
        store = SqliteAuditStore(db_path=enforceai_sqlite_db_path)

        user_id = "https://issuer.example|user-1"
        agent_id = str(uuid.uuid4())
        t0 = datetime(2025, 1, 1, tzinfo=timezone.utc)

        store.append_event(
            occurred_at=t0,
            user_id=user_id,
            agent_id=agent_id,
            action="tools/list",
            outcome="allow",
        )

        result = store.query_events(user_id=user_id, limit=1000)
        assert len(result.items) == 1

    def test_query_events_returns_server_time(
        self,
        enforceai_sqlite_db_path: Path,
    ) -> None:
        _migrate_db(enforceai_sqlite_db_path)
        store = SqliteAuditStore(db_path=enforceai_sqlite_db_path)

        user_id = "https://issuer.example|user-1"
        agent_id = str(uuid.uuid4())
        t0 = datetime(2025, 1, 1, tzinfo=timezone.utc)

        store.append_event(
            occurred_at=t0,
            user_id=user_id,
            agent_id=agent_id,
            action="tools/list",
            outcome="allow",
        )

        before = datetime.now(timezone.utc)
        result = store.query_events(user_id=user_id, limit=10)
        after = datetime.now(timezone.utc)

        assert result.server_time >= before.replace(microsecond=0)
        assert result.server_time <= after.replace(microsecond=0) + timedelta(seconds=1)

    def test_query_events_no_filters_returns_all(
        self,
        enforceai_sqlite_db_path: Path,
    ) -> None:
        _migrate_db(enforceai_sqlite_db_path)
        store = SqliteAuditStore(db_path=enforceai_sqlite_db_path)

        user_id = "https://issuer.example|user-1"
        agent_id = str(uuid.uuid4())
        t0 = datetime(2025, 1, 1, tzinfo=timezone.utc)

        for i in range(3):
            store.append_event(
                occurred_at=t0 + timedelta(seconds=i),
                user_id=user_id,
                agent_id=agent_id,
                action="tools/list",
                outcome="allow",
            )

        result = store.query_events(limit=10)

        assert len(result.items) == 3

    def test_query_events_combines_multiple_filters(
        self,
        enforceai_sqlite_db_path: Path,
    ) -> None:
        _migrate_db(enforceai_sqlite_db_path)
        store = SqliteAuditStore(db_path=enforceai_sqlite_db_path)

        user_id = "https://issuer.example|user-1"
        agent_id = str(uuid.uuid4())
        t0 = datetime(2025, 1, 1, tzinfo=timezone.utc)

        store.append_event(
            occurred_at=t0,
            user_id=user_id,
            agent_id=agent_id,
            action="tools/call",
            outcome="allow",
            details={"server": "sqlite", "tool": "query"},
        )
        store.append_event(
            occurred_at=t0 + timedelta(seconds=1),
            user_id=user_id,
            agent_id=agent_id,
            action="tools/call",
            outcome="deny",
            details={"server": "sqlite", "tool": "execute"},
        )
        store.append_event(
            occurred_at=t0 + timedelta(seconds=2),
            user_id=user_id,
            agent_id=agent_id,
            action="tools/list",
            outcome="allow",
            details={"server": "sqlite"},
        )

        result = store.query_events(
            user_id=user_id,
            actions=["tools/call"],
            outcomes=["deny"],
            server="sqlite",
            limit=10,
        )

        assert len(result.items) == 1
        assert result.items[0].action == "tools/call"
        assert result.items[0].outcome == "deny"

    def test_query_events_empty_result(
        self,
        enforceai_sqlite_db_path: Path,
    ) -> None:
        _migrate_db(enforceai_sqlite_db_path)
        store = SqliteAuditStore(db_path=enforceai_sqlite_db_path)

        result = store.query_events(
            user_id="https://issuer.example|nonexistent",
            limit=10,
        )

        assert len(result.items) == 0
        assert result.next_cursor is None

    def test_query_events_invalid_cursor_ignored(
        self,
        enforceai_sqlite_db_path: Path,
    ) -> None:
        _migrate_db(enforceai_sqlite_db_path)
        store = SqliteAuditStore(db_path=enforceai_sqlite_db_path)

        user_id = "https://issuer.example|user-1"
        agent_id = str(uuid.uuid4())
        t0 = datetime(2025, 1, 1, tzinfo=timezone.utc)

        store.append_event(
            occurred_at=t0,
            user_id=user_id,
            agent_id=agent_id,
            action="tools/list",
            outcome="allow",
        )

        result = store.query_events(
            user_id=user_id,
            cursor="invalid-cursor",
            limit=10,
        )

        assert len(result.items) == 1
