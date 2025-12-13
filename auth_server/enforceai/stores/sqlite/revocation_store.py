from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import sqlite3

from auth_server.enforceai.db.connection import (
    sqlite_connection,
)
from auth_server.enforceai.models.revocation import (
    TokenRevocationRecord,
)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc).replace(microsecond=0)


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
    value = _ensure_aware_utc(value)
    return value.replace(microsecond=0).isoformat().replace(
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


class SqliteRevocationStore:
    def __init__(
        self,
        *,
        db_path: Path,
    ) -> None:
        self._db_path = db_path

    def revoke_jti(
        self,
        *,
        jti: str,
        user_id: str,
        agent_id: str,
        revoked_at: Optional[datetime] = None,
        expires_at: Optional[datetime] = None,
        reason: Optional[str] = None,
    ) -> TokenRevocationRecord:
        now = revoked_at or _utc_now()
        now = _ensure_aware_utc(now).replace(microsecond=0)
        if expires_at is not None:
            expires_at = _ensure_aware_utc(expires_at).replace(microsecond=0)

        validated = TokenRevocationRecord(
            jti=jti,
            user_id=user_id,
            agent_id=agent_id,
            revoked_at=now,
            expires_at=expires_at,
            reason=reason,
        )

        existing = self._get_revocation_by_jti(jti=validated.jti)
        if existing is not None:
            return existing

        with sqlite_connection(self._db_path) as connection:
            try:
                connection.execute(
                    """
                    INSERT INTO token_revocations(
                        jti,
                        user_id,
                        agent_id,
                        revoked_at,
                        expires_at,
                        reason
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    """.strip(),
                    (
                        validated.jti,
                        validated.user_id,
                        validated.agent_id,
                        _datetime_to_iso(validated.revoked_at),
                        _datetime_to_iso(validated.expires_at),
                        validated.reason,
                    ),
                )
            except sqlite3.IntegrityError as exc:
                raise ValueError(f"Token already revoked: {validated.jti}") from exc

        stored = self._get_revocation_by_jti(jti=validated.jti)
        if stored is None:
            raise RuntimeError(
                "Token revocation insert succeeded but record could not be read back"
            )
        return stored

    def is_jti_revoked(
        self,
        *,
        jti: str,
        now: Optional[datetime] = None,
    ) -> bool:
        record = self._get_revocation_by_jti(jti=jti)
        if record is None:
            return False

        if record.expires_at is None:
            return True

        effective_now = _ensure_aware_utc(now or _utc_now()).replace(microsecond=0)
        return record.expires_at > effective_now

    def list_revocations_by_agent_id(
        self,
        *,
        agent_id: str,
    ) -> list[TokenRevocationRecord]:
        with sqlite_connection(self._db_path) as connection:
            rows = connection.execute(
                """
                SELECT
                    jti,
                    user_id,
                    agent_id,
                    revoked_at,
                    expires_at,
                    reason
                FROM token_revocations
                WHERE agent_id = ?
                ORDER BY revoked_at ASC
                """.strip(),
                (agent_id,),
            ).fetchall()

        return [
            TokenRevocationRecord(
                jti=row[0],
                user_id=row[1],
                agent_id=row[2],
                revoked_at=_datetime_from_iso(row[3]),
                expires_at=_datetime_from_iso(row[4]),
                reason=row[5],
            )
            for row in rows
        ]

    def delete_expired_revocations(
        self,
        *,
        now: datetime,
    ) -> int:
        now = _ensure_aware_utc(now).replace(microsecond=0)
        with sqlite_connection(self._db_path) as connection:
            cursor = connection.execute(
                """
                DELETE FROM token_revocations
                WHERE expires_at IS NOT NULL AND expires_at <= ?
                """.strip(),
                (_datetime_to_iso(now),),
            )
            deleted = cursor.rowcount or 0
        return deleted

    def _get_revocation_by_jti(
        self,
        *,
        jti: str,
    ) -> Optional[TokenRevocationRecord]:
        with sqlite_connection(self._db_path) as connection:
            row = connection.execute(
                """
                SELECT
                    jti,
                    user_id,
                    agent_id,
                    revoked_at,
                    expires_at,
                    reason
                FROM token_revocations
                WHERE jti = ?
                """.strip(),
                (jti,),
            ).fetchone()

        if row is None:
            return None

        return TokenRevocationRecord(
            jti=row[0],
            user_id=row[1],
            agent_id=row[2],
            revoked_at=_datetime_from_iso(row[3]),
            expires_at=_datetime_from_iso(row[4]),
            reason=row[5],
        )
