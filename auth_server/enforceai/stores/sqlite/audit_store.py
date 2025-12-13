from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import sqlite3

from auth_server.enforceai.db.connection import (
    sqlite_connection,
)
from auth_server.enforceai.models.audit import (
    AuditEventRecord,
)


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
        if limit <= 0:
            raise ValueError("limit must be positive")

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
