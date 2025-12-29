from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from ...db.connection import (
    sqlite_connection,
)
from ...models.audit import (
    AuditEventRecord,
    AuditEventsQueryResult,
    DEFAULT_AUDIT_PAGE_SIZE,
    MAX_AUDIT_PAGE_SIZE,
)


def _validate_positive_limit(
    *,
    limit: int,
) -> None:
    if limit <= 0:
        raise ValueError("limit must be positive")


def _ensure_aware_utc(
    value: datetime,
) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _datetime_to_iso(
    value: Optional[datetime],
) -> Optional[str]:
    if value is None:
        return None
    value = _ensure_aware_utc(value).replace(microsecond=0)
    return value.isoformat().replace(
        "+00:00",
        "Z",
    )


def _datetime_from_iso(
    value: Optional[str],
) -> Optional[datetime]:
    if value is None:
        return None
    normalized = value.replace("Z", "+00:00")
    return datetime.fromisoformat(normalized)


def _json_dumps_optional(
    value: Optional[dict[str, object]],
) -> Optional[str]:
    if value is None:
        return None
    return json.dumps(value, separators=(",", ":"), sort_keys=True, default=str)


def _json_loads_optional(
    raw: Optional[str],
) -> Optional[dict[str, object]]:
    if raw is None:
        return None
    parsed = json.loads(raw)
    if parsed is None:
        return None
    if not isinstance(parsed, dict):
        raise ValueError("Invalid details_json stored for audit_events record")
    return parsed


class SqliteAuditStore:
    def __init__(
        self,
        *,
        db_path: Path,
    ) -> None:
        self._db_path = db_path

    def append_event(
        self,
        *,
        occurred_at: datetime,
        user_id: str,
        agent_id: str,
        action: str,
        outcome: str,
        request_id: Optional[str] = None,
        details: Optional[dict[str, object]] = None,
    ) -> AuditEventRecord:
        occurred_at = _ensure_aware_utc(occurred_at).replace(microsecond=0)
        validated = AuditEventRecord(
            event_id=1,
            occurred_at=occurred_at,
            user_id=user_id,
            agent_id=agent_id,
            action=action,
            outcome=outcome,
            request_id=request_id,
            details=details,
        )

        with sqlite_connection(self._db_path) as connection:
            cursor = connection.execute(
                """
                INSERT INTO audit_events(
                    occurred_at,
                    user_id,
                    agent_id,
                    action,
                    outcome,
                    request_id,
                    details_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """.strip(),
                (
                    _datetime_to_iso(validated.occurred_at),
                    validated.user_id,
                    validated.agent_id,
                    validated.action,
                    validated.outcome,
                    validated.request_id,
                    _json_dumps_optional(details),
                ),
            )
            event_id = int(cursor.lastrowid)

        stored = self._get_event_by_id(event_id=event_id)
        if stored is None:
            raise RuntimeError(
                "Audit event insert succeeded but record could not be read back"
            )
        return stored

    def list_recent_events(
        self,
        *,
        user_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        since: Optional[datetime] = None,
        until: Optional[datetime] = None,
        limit: int = 100,
    ) -> list[AuditEventRecord]:
        if user_id is None and agent_id is None:
            raise ValueError("At least one of user_id or agent_id must be provided")
        _validate_positive_limit(limit=limit)

        filters: list[str] = []
        params: list[object] = []

        if user_id is not None:
            filters.append("user_id = ?")
            params.append(user_id)
        if agent_id is not None:
            filters.append("agent_id = ?")
            params.append(agent_id)
        if since is not None:
            filters.append("occurred_at >= ?")
            params.append(_datetime_to_iso(_ensure_aware_utc(since).replace(microsecond=0)))
        if until is not None:
            filters.append("occurred_at <= ?")
            params.append(_datetime_to_iso(_ensure_aware_utc(until).replace(microsecond=0)))

        where_clause = " AND ".join(filters)
        query = (
            """
            SELECT
                id,
                occurred_at,
                user_id,
                agent_id,
                action,
                outcome,
                request_id,
                details_json
            FROM audit_events
            WHERE {where_clause}
            ORDER BY occurred_at DESC, id DESC
            LIMIT ?
            """.format(where_clause=where_clause).strip()
        )

        params.append(limit)
        with sqlite_connection(self._db_path) as connection:
            rows = connection.execute(
                query,
                tuple(params),
            ).fetchall()

        return [
            AuditEventRecord(
                event_id=row[0],
                occurred_at=_datetime_from_iso(row[1]),
                user_id=row[2],
                agent_id=row[3],
                action=row[4],
                outcome=row[5],
                request_id=row[6],
                details=_json_loads_optional(row[7]),
            )
            for row in rows
        ]

    def delete_events_older_than(
        self,
        *,
        cutoff: datetime,
    ) -> int:
        normalized_cutoff = _ensure_aware_utc(cutoff).replace(microsecond=0)
        cutoff_iso = _datetime_to_iso(normalized_cutoff)
        if cutoff_iso is None:
            raise RuntimeError("cutoff datetime conversion failed")

        with sqlite_connection(self._db_path) as connection:
            cursor = connection.execute(
                """
                DELETE FROM audit_events
                WHERE occurred_at < ?
                """.strip(),
                (cutoff_iso,),
            )
            deleted = cursor.rowcount

        return int(deleted) if deleted is not None else 0

    def delete_oldest_events(
        self,
        *,
        limit: int,
    ) -> int:
        _validate_positive_limit(limit=limit)

        with sqlite_connection(self._db_path) as connection:
            cursor = connection.execute(
                """
                DELETE FROM audit_events
                WHERE id IN (
                    SELECT id
                    FROM audit_events
                    ORDER BY occurred_at ASC, id ASC
                    LIMIT ?
                )
                """.strip(),
                (limit,),
            )
            deleted = cursor.rowcount

        return int(deleted) if deleted is not None else 0

    def _get_event_by_id(
        self,
        *,
        event_id: int,
    ) -> Optional[AuditEventRecord]:
        with sqlite_connection(self._db_path) as connection:
            row = connection.execute(
                """
                SELECT
                    id,
                    occurred_at,
                    user_id,
                    agent_id,
                    action,
                    outcome,
                    request_id,
                    details_json
                FROM audit_events
                WHERE id = ?
                """.strip(),
                (event_id,),
            ).fetchone()

        if row is None:
            return None

        return AuditEventRecord(
            event_id=row[0],
            occurred_at=_datetime_from_iso(row[1]),
            user_id=row[2],
            agent_id=row[3],
            action=row[4],
            outcome=row[5],
            request_id=row[6],
            details=_json_loads_optional(row[7]),
        )

    def query_events(
        self,
        *,
        user_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        actions: Optional[list[str]] = None,
        outcomes: Optional[list[str]] = None,
        request_id: Optional[str] = None,
        server: Optional[str] = None,
        tool: Optional[str] = None,
        since: Optional[datetime] = None,
        until: Optional[datetime] = None,
        limit: int = DEFAULT_AUDIT_PAGE_SIZE,
        cursor: Optional[str] = None,
        max_limit: int = MAX_AUDIT_PAGE_SIZE,
    ) -> AuditEventsQueryResult:
        """
        Query audit events with filtering and pagination.

        Args:
            user_id: Filter by user ID (required for self-service queries)
            agent_id: Filter by agent ID
            actions: Filter by action names (OR logic)
            outcomes: Filter by outcome values (OR logic)
            request_id: Filter by exact request ID match
            server: Filter by server (from details.server)
            tool: Filter by tool (from details.tool)
            since: Filter events after this time
            until: Filter events before this time
            limit: Maximum number of events to return
            cursor: Pagination cursor (event_id to start after)

        Returns:
            AuditEventsQueryResult with items, next_cursor, and server_time
        """
        _validate_positive_limit(limit=limit)
        _validate_positive_limit(limit=max_limit)
        if limit > max_limit:
            limit = max_limit

        filters: list[str] = []
        params: list[object] = []

        if user_id is not None:
            filters.append("user_id = ?")
            params.append(user_id)

        if agent_id is not None:
            filters.append("agent_id = ?")
            params.append(agent_id)

        if actions is not None and len(actions) > 0:
            placeholders = ",".join("?" for _ in actions)
            filters.append(f"action IN ({placeholders})")
            params.extend(actions)

        if outcomes is not None and len(outcomes) > 0:
            placeholders = ",".join("?" for _ in outcomes)
            filters.append(f"outcome IN ({placeholders})")
            params.extend(outcomes)

        if request_id is not None:
            filters.append("request_id = ?")
            params.append(request_id)

        if server is not None:
            filters.append("json_extract(details_json, '$.server') = ?")
            params.append(server)

        if tool is not None:
            filters.append("json_extract(details_json, '$.tool') = ?")
            params.append(tool)

        if since is not None:
            filters.append("occurred_at >= ?")
            params.append(
                _datetime_to_iso(_ensure_aware_utc(since).replace(microsecond=0))
            )

        if until is not None:
            filters.append("occurred_at <= ?")
            params.append(
                _datetime_to_iso(_ensure_aware_utc(until).replace(microsecond=0))
            )

        if cursor is not None:
            try:
                cursor_id = int(cursor)
                filters.append("id < ?")
                params.append(cursor_id)
            except ValueError:
                pass

        where_clause = " AND ".join(filters) if filters else "1=1"

        query = f"""
            SELECT
                id,
                occurred_at,
                user_id,
                agent_id,
                action,
                outcome,
                request_id,
                details_json
            FROM audit_events
            WHERE {where_clause}
            ORDER BY id DESC
            LIMIT ?
        """.strip()

        params.append(limit + 1)

        with sqlite_connection(self._db_path) as connection:
            rows = connection.execute(
                query,
                tuple(params),
            ).fetchall()

        has_more = len(rows) > limit
        if has_more:
            rows = rows[:limit]

        items = [
            AuditEventRecord(
                event_id=row[0],
                occurred_at=_datetime_from_iso(row[1]),
                user_id=row[2],
                agent_id=row[3],
                action=row[4],
                outcome=row[5],
                request_id=row[6],
                details=_json_loads_optional(row[7]),
            )
            for row in rows
        ]

        next_cursor: Optional[str] = None
        if has_more and items:
            next_cursor = str(items[-1].event_id)

        return AuditEventsQueryResult(
            items=items,
            next_cursor=next_cursor,
            server_time=datetime.now(timezone.utc).replace(microsecond=0),
        )
