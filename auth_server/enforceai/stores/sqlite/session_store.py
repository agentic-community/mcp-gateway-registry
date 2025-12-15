from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from ...db.connection import (
    sqlite_connection,
)
from ...models.session import (
    SessionRecord,
)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc).replace(microsecond=0)


def _ensure_utc(
    value: datetime,
) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc).replace(microsecond=0)


def _datetime_to_iso(
    value: datetime,
) -> str:
    normalized = _ensure_utc(value)
    return normalized.isoformat().replace("+00:00", "Z")


def _datetime_from_iso(
    value: str,
) -> datetime:
    normalized = value.replace("Z", "+00:00")
    return datetime.fromisoformat(normalized)


def _row_to_session(
    row: sqlite3.Row,
) -> SessionRecord:
    return SessionRecord(
        session_id=row["session_id"],
        user_id=row["user_id"],
        auth_method=row["auth_method"],
        created_at=_datetime_from_iso(row["created_at"]),
        expires_at=_datetime_from_iso(row["expires_at"]),
        last_seen_at=_datetime_from_iso(row["last_seen_at"]),
        revoked_at=_datetime_from_iso(row["revoked_at"]) if row["revoked_at"] else None,
        revoked_reason=row["revoked_reason"],
    )


class SqliteSessionStore:
    def __init__(
        self,
        *,
        db_path: Path,
    ) -> None:
        self._db_path = db_path

    def create_session(
        self,
        *,
        session_id: str,
        user_id: str,
        auth_method: str,
        expires_at: datetime,
        now: Optional[datetime] = None,
    ) -> SessionRecord:
        timestamp = _utc_now() if now is None else _ensure_utc(now)
        expires_at_value = _ensure_utc(expires_at)
        if expires_at_value <= timestamp:
            raise ValueError("expires_at must be in the future")

        created_at = _datetime_to_iso(timestamp)
        expires_at_iso = _datetime_to_iso(expires_at_value)
        last_seen_at = created_at

        with sqlite_connection(self._db_path) as connection:
            connection.row_factory = sqlite3.Row
            connection.execute(
                """
                INSERT INTO sessions(
                    session_id,
                    user_id,
                    auth_method,
                    created_at,
                    expires_at,
                    last_seen_at,
                    revoked_at,
                    revoked_reason
                ) VALUES (?, ?, ?, ?, ?, ?, NULL, NULL)
                """.strip(),
                (
                    session_id,
                    user_id,
                    auth_method,
                    created_at,
                    expires_at_iso,
                    last_seen_at,
                ),
            )
            row = connection.execute(
                "SELECT * FROM sessions WHERE session_id = ?",
                (session_id,),
            ).fetchone()
            assert row is not None
            return _row_to_session(row)

    def get_session_by_id(
        self,
        *,
        session_id: str,
    ) -> Optional[SessionRecord]:
        with sqlite_connection(self._db_path) as connection:
            connection.row_factory = sqlite3.Row
            row = connection.execute(
                "SELECT * FROM sessions WHERE session_id = ?",
                (session_id,),
            ).fetchone()
            if row is None:
                return None
            return _row_to_session(row)

    def touch_session(
        self,
        *,
        session_id: str,
        now: datetime,
    ) -> Optional[SessionRecord]:
        timestamp = _ensure_utc(now)
        touched = _datetime_to_iso(timestamp)

        with sqlite_connection(self._db_path) as connection:
            connection.row_factory = sqlite3.Row
            connection.execute(
                """
                UPDATE sessions
                SET last_seen_at = ?
                WHERE session_id = ? AND revoked_at IS NULL
                """.strip(),
                (
                    touched,
                    session_id,
                ),
            )
            row = connection.execute(
                "SELECT * FROM sessions WHERE session_id = ?",
                (session_id,),
            ).fetchone()
            if row is None:
                return None
            return _row_to_session(row)

    def revoke_session(
        self,
        *,
        session_id: str,
        revoked_at: Optional[datetime] = None,
        revoked_reason: Optional[str] = None,
    ) -> Optional[SessionRecord]:
        timestamp = _utc_now() if revoked_at is None else _ensure_utc(revoked_at)
        revoked_iso = _datetime_to_iso(timestamp)

        with sqlite_connection(self._db_path) as connection:
            connection.row_factory = sqlite3.Row
            connection.execute(
                """
                UPDATE sessions
                SET revoked_at = ?, revoked_reason = ?
                WHERE session_id = ? AND revoked_at IS NULL
                """.strip(),
                (
                    revoked_iso,
                    revoked_reason,
                    session_id,
                ),
            )
            row = connection.execute(
                "SELECT * FROM sessions WHERE session_id = ?",
                (session_id,),
            ).fetchone()
            if row is None:
                return None
            return _row_to_session(row)

