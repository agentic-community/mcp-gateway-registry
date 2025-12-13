from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import sqlite3

from auth_server.enforceai.db.connection import (
    sqlite_connection,
)
from auth_server.enforceai.models.api_key import (
    ApiKeyRecord,
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


def _json_dumps(
    value: object,
) -> str:
    return json.dumps(value, separators=(",", ":"), sort_keys=True)


def _json_loads_optional(
    raw: Optional[str],
) -> object:
    if raw is None:
        return None
    return json.loads(raw)


class SqliteApiKeyStore:
    def __init__(
        self,
        *,
        db_path: Path,
    ) -> None:
        self._db_path = db_path

    def create_key(
        self,
        *,
        key_id: str,
        secret_hash: str,
        user_id: str,
        agent_id: str,
        scopes: Optional[list[str]] = None,
        expires_at: Optional[datetime] = None,
    ) -> ApiKeyRecord:
        now = _utc_now()
        validated = ApiKeyRecord(
            key_id=key_id,
            secret_hash=secret_hash,
            user_id=user_id,
            agent_id=agent_id,
            scopes=scopes,
            expires_at=expires_at,
            revoked_at=None,
            created_at=now,
            last_used_at=None,
        )

        with sqlite_connection(self._db_path) as connection:
            try:
                connection.execute(
                    """
                    INSERT INTO api_keys(
                        key_id,
                        secret_hash,
                        user_id,
                        agent_id,
                        scopes_json,
                        expires_at,
                        revoked_at,
                        created_at,
                        last_used_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """.strip(),
                    (
                        validated.key_id,
                        validated.secret_hash,
                        validated.user_id,
                        validated.agent_id,
                        _json_dumps(validated.scopes) if validated.scopes is not None else None,
                        _datetime_to_iso(validated.expires_at),
                        None,
                        _datetime_to_iso(validated.created_at),
                        None,
                    ),
                )
            except sqlite3.IntegrityError as exc:
                raise ValueError(f"API key already exists: {validated.key_id}") from exc

        record = self.get_key_by_id(key_id=validated.key_id)
        if record is None:
            raise RuntimeError("API key insert succeeded but record could not be read back")
        return record

    def get_key_by_id(
        self,
        *,
        key_id: str,
    ) -> Optional[ApiKeyRecord]:
        with sqlite_connection(self._db_path) as connection:
            row = connection.execute(
                """
                SELECT
                    key_id,
                    secret_hash,
                    user_id,
                    agent_id,
                    scopes_json,
                    expires_at,
                    revoked_at,
                    created_at,
                    last_used_at
                FROM api_keys
                WHERE key_id = ?
                """.strip(),
                (key_id,),
            ).fetchone()

        if row is None:
            return None

        scopes = _json_loads_optional(row[4])
        if scopes is not None and not isinstance(scopes, list):
            raise ValueError("Invalid scopes_json stored for api_keys record")

        return ApiKeyRecord(
            key_id=row[0],
            secret_hash=row[1],
            user_id=row[2],
            agent_id=row[3],
            scopes=scopes,
            expires_at=_datetime_from_iso(row[5]),
            revoked_at=_datetime_from_iso(row[6]),
            created_at=_datetime_from_iso(row[7]),
            last_used_at=_datetime_from_iso(row[8]),
        )

    def list_keys(
        self,
        *,
        user_id: Optional[str] = None,
        agent_id: Optional[str] = None,
    ) -> list[ApiKeyRecord]:
        if user_id is None and agent_id is None:
            raise ValueError("At least one of user_id or agent_id must be provided")

        filters: list[str] = []
        params: list[object] = []
        if user_id is not None:
            filters.append("user_id = ?")
            params.append(user_id)
        if agent_id is not None:
            filters.append("agent_id = ?")
            params.append(agent_id)

        where_clause = " AND ".join(filters)
        query = (
            """
            SELECT
                key_id,
                secret_hash,
                user_id,
                agent_id,
                scopes_json,
                expires_at,
                revoked_at,
                created_at,
                last_used_at
            FROM api_keys
            WHERE {where_clause}
            ORDER BY created_at ASC
            """.format(where_clause=where_clause).strip()
        )

        with sqlite_connection(self._db_path) as connection:
            rows = connection.execute(
                query,
                tuple(params),
            ).fetchall()

        records: list[ApiKeyRecord] = []
        for row in rows:
            scopes = _json_loads_optional(row[4])
            if scopes is not None and not isinstance(scopes, list):
                raise ValueError("Invalid scopes_json stored for api_keys record")
            records.append(
                ApiKeyRecord(
                    key_id=row[0],
                    secret_hash=row[1],
                    user_id=row[2],
                    agent_id=row[3],
                    scopes=scopes,
                    expires_at=_datetime_from_iso(row[5]),
                    revoked_at=_datetime_from_iso(row[6]),
                    created_at=_datetime_from_iso(row[7]),
                    last_used_at=_datetime_from_iso(row[8]),
                )
            )
        return records

    def revoke_key(
        self,
        *,
        key_id: str,
        revoked_at: Optional[datetime] = None,
    ) -> Optional[ApiKeyRecord]:
        existing = self.get_key_by_id(key_id=key_id)
        if existing is None:
            return None
        if existing.revoked_at is not None:
            return existing

        now = revoked_at or _utc_now()
        if now.tzinfo is None:
            now = now.replace(tzinfo=timezone.utc)

        with sqlite_connection(self._db_path) as connection:
            connection.execute(
                """
                UPDATE api_keys
                SET revoked_at = ?
                WHERE key_id = ?
                """.strip(),
                (
                    _datetime_to_iso(now),
                    key_id,
                ),
            )

        return self.get_key_by_id(key_id=key_id)

    def update_last_used_at(
        self,
        *,
        key_id: str,
        last_used_at: datetime,
    ) -> Optional[ApiKeyRecord]:
        existing = self.get_key_by_id(key_id=key_id)
        if existing is None:
            return None

        if last_used_at.tzinfo is None:
            last_used_at = last_used_at.replace(tzinfo=timezone.utc)

        ApiKeyRecord(
            key_id=existing.key_id,
            secret_hash=existing.secret_hash,
            user_id=existing.user_id,
            agent_id=existing.agent_id,
            scopes=existing.scopes,
            expires_at=existing.expires_at,
            revoked_at=existing.revoked_at,
            created_at=existing.created_at,
            last_used_at=last_used_at,
        )

        with sqlite_connection(self._db_path) as connection:
            connection.execute(
                """
                UPDATE api_keys
                SET last_used_at = ?
                WHERE key_id = ?
                """.strip(),
                (
                    _datetime_to_iso(last_used_at),
                    key_id,
                ),
            )

        return self.get_key_by_id(key_id=key_id)
