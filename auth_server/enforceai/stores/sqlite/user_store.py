from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from ...db.connection import (
    sqlite_connection,
)
from ...models.user import (
    UserRecord,
)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc).replace(microsecond=0)


def _datetime_to_iso(
    value: Optional[datetime],
) -> Optional[str]:
    if value is None:
        return None
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc).replace(microsecond=0).isoformat().replace(
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


def _row_to_user(
    row: sqlite3.Row,
) -> UserRecord:
    return UserRecord(
        user_id=row["user_id"],
        auth_method=row["auth_method"],
        username=row["username"],
        email=row["email"],
        password_hash=row["password_hash"],
        role=row["role"],
        created_at=_datetime_from_iso(row["created_at"]),
        updated_at=_datetime_from_iso(row["updated_at"]),
        last_login_at=_datetime_from_iso(row["last_login_at"]),
        disabled_at=_datetime_from_iso(row["disabled_at"]),
    )


class SqliteUserStore:
    def __init__(
        self,
        *,
        db_path: Path,
    ) -> None:
        self._db_path = db_path

    def upsert_oidc_user(
        self,
        *,
        user_id: str,
        email: str,
        role: str = "user",
        now: Optional[datetime] = None,
    ) -> UserRecord:
        timestamp = _utc_now() if now is None else now
        created_at = _datetime_to_iso(timestamp)
        updated_at = created_at
        last_login_at = created_at

        with sqlite_connection(self._db_path) as connection:
            connection.row_factory = sqlite3.Row
            connection.execute(
                """
                INSERT INTO users(
                    user_id,
                    auth_method,
                    username,
                    email,
                    password_hash,
                    role,
                    created_at,
                    updated_at,
                    last_login_at,
                    disabled_at
                ) VALUES (?, 'oidc', NULL, ?, NULL, ?, ?, ?, ?, NULL)
                ON CONFLICT(user_id) DO UPDATE SET
                    email=excluded.email,
                    role=users.role,
                    updated_at=excluded.updated_at,
                    last_login_at=excluded.last_login_at
                """.strip(),
                (
                    user_id,
                    email,
                    role,
                    created_at,
                    updated_at,
                    last_login_at,
                ),
            )
            row = connection.execute(
                "SELECT * FROM users WHERE user_id = ?",
                (user_id,),
            ).fetchone()
            assert row is not None
            return _row_to_user(row)

    def create_local_user(
        self,
        *,
        username: str,
        email: str,
        password_hash: str,
        role: str = "user",
        now: Optional[datetime] = None,
    ) -> UserRecord:
        timestamp = _utc_now() if now is None else now
        created_at = _datetime_to_iso(timestamp)
        updated_at = created_at
        user_id = f"local|{username}"

        with sqlite_connection(self._db_path) as connection:
            connection.row_factory = sqlite3.Row
            connection.execute(
                """
                INSERT INTO users(
                    user_id,
                    auth_method,
                    username,
                    email,
                    password_hash,
                    role,
                    created_at,
                    updated_at,
                    last_login_at,
                    disabled_at
                ) VALUES (?, 'password', ?, ?, ?, ?, ?, ?, NULL, NULL)
                """.strip(),
                (
                    user_id,
                    username,
                    email,
                    password_hash,
                    role,
                    created_at,
                    updated_at,
                ),
            )
            row = connection.execute(
                "SELECT * FROM users WHERE user_id = ?",
                (user_id,),
            ).fetchone()
            assert row is not None
            return _row_to_user(row)

    def get_user_by_id(
        self,
        *,
        user_id: str,
    ) -> Optional[UserRecord]:
        with sqlite_connection(self._db_path) as connection:
            connection.row_factory = sqlite3.Row
            row = connection.execute(
                "SELECT * FROM users WHERE user_id = ?",
                (user_id,),
            ).fetchone()
            if row is None:
                return None
            return _row_to_user(row)

    def get_user_by_username(
        self,
        *,
        username: str,
    ) -> Optional[UserRecord]:
        with sqlite_connection(self._db_path) as connection:
            connection.row_factory = sqlite3.Row
            row = connection.execute(
                "SELECT * FROM users WHERE username = ?",
                (username,),
            ).fetchone()
            if row is None:
                return None
            return _row_to_user(row)

    def search_users(
        self,
        *,
        query: str,
        limit: int = 50,
    ) -> list[UserRecord]:
        normalized = query.strip()
        if not normalized:
            return []

        like_value = f"%{normalized}%"

        with sqlite_connection(self._db_path) as connection:
            connection.row_factory = sqlite3.Row
            rows = connection.execute(
                """
                SELECT *
                FROM users
                WHERE email LIKE ? OR username LIKE ?
                ORDER BY last_login_at DESC, created_at DESC
                LIMIT ?
                """.strip(),
                (
                    like_value,
                    like_value,
                    limit,
                ),
            ).fetchall()
            return [_row_to_user(row) for row in rows]

    def disable_user(
        self,
        *,
        user_id: str,
        disabled_at: Optional[datetime] = None,
    ) -> Optional[UserRecord]:
        timestamp = _utc_now() if disabled_at is None else disabled_at
        disabled_at_iso = _datetime_to_iso(timestamp)
        updated_at_iso = disabled_at_iso

        with sqlite_connection(self._db_path) as connection:
            connection.row_factory = sqlite3.Row
            connection.execute(
                """
                UPDATE users
                SET disabled_at = ?, updated_at = ?
                WHERE user_id = ?
                """.strip(),
                (
                    disabled_at_iso,
                    updated_at_iso,
                    user_id,
                ),
            )
            row = connection.execute(
                "SELECT * FROM users WHERE user_id = ?",
                (user_id,),
            ).fetchone()
            if row is None:
                return None
            return _row_to_user(row)

    def update_password_hash(
        self,
        *,
        user_id: str,
        password_hash: str,
        updated_at: Optional[datetime] = None,
    ) -> Optional[UserRecord]:
        timestamp = _utc_now() if updated_at is None else updated_at
        updated_at_iso = _datetime_to_iso(timestamp)

        with sqlite_connection(self._db_path) as connection:
            connection.row_factory = sqlite3.Row
            connection.execute(
                """
                UPDATE users
                SET password_hash = ?, updated_at = ?
                WHERE user_id = ?
                """.strip(),
                (
                    password_hash,
                    updated_at_iso,
                    user_id,
                ),
            )
            row = connection.execute(
                "SELECT * FROM users WHERE user_id = ?",
                (user_id,),
            ).fetchone()
            if row is None:
                return None
            return _row_to_user(row)

