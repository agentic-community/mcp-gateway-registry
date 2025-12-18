from __future__ import annotations

import sqlite3
import uuid
from datetime import (
    datetime,
    timedelta,
    timezone,
)
from pathlib import Path
from typing import (
    Any,
    Optional,
)

from ...crypto.upstream_secrets import (
    EncryptedSecretEnvelope,
    build_aad_for_upstream_oauth_state,
    decrypt_secret_payload,
    encrypt_secret_payload,
)
from ...db.connection import (
    sqlite_connection,
)
from ...models.upstream_oauth import (
    UpstreamOAuthCredentialType,
    UpstreamOAuthStateRecord,
    UpstreamOAuthStateSecret,
)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc).replace(microsecond=0)


def _datetime_to_iso(
    value: datetime,
) -> str:
    ts = value
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    return ts.astimezone(timezone.utc).replace(microsecond=0).isoformat().replace(
        "+00:00",
        "Z",
    )


def _datetime_from_iso(
    value: str,
) -> datetime:
    normalized = value.replace("Z", "+00:00")
    return datetime.fromisoformat(normalized)


class SqliteUpstreamOAuthStateStore:
    def __init__(
        self,
        *,
        db_path: Path,
        kek: bytes,
    ) -> None:
        if not kek:
            raise ValueError("kek must be non-empty")
        self._db_path = db_path
        self._kek = kek

    def create_state(
        self,
        *,
        server_path: str,
        credential_type: UpstreamOAuthCredentialType,
        credential_binding: str,
        user_id: str,
        agent_id: Optional[str],
        provider: str,
        redirect_uri: str,
        ttl_seconds: int,
        secret_payload: dict[str, Any],
    ) -> UpstreamOAuthStateRecord:
        now = _utc_now()
        expires_at = now + timedelta(seconds=ttl_seconds)
        state_id = str(uuid.uuid4())

        record = UpstreamOAuthStateRecord(
            state_id=state_id,
            server_path=server_path,
            credential_type=credential_type,
            credential_binding=credential_binding,
            user_id=user_id,
            agent_id=agent_id,
            provider=provider,
            redirect_uri=redirect_uri,
            created_at=now,
            expires_at=expires_at,
        )

        aad = build_aad_for_upstream_oauth_state(
            state_id=record.state_id,
            server_path=record.server_path,
            credential_type=record.credential_type,
        )
        envelope = encrypt_secret_payload(
            key=self._kek,
            aad=aad,
            payload=dict(secret_payload),
        )

        with sqlite_connection(self._db_path) as connection:
            try:
                connection.execute(
                    """
                    INSERT INTO upstream_oauth_states(
                        state_id,
                        server_path,
                        credential_type,
                        credential_binding,
                        user_id,
                        agent_id,
                        provider,
                        redirect_uri,
                        secret_version,
                        secret_nonce,
                        secret_ciphertext,
                        created_at,
                        expires_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """.strip(),
                    (
                        record.state_id,
                        record.server_path,
                        record.credential_type,
                        record.credential_binding,
                        record.user_id,
                        record.agent_id,
                        record.provider,
                        record.redirect_uri,
                        envelope.version,
                        envelope.nonce,
                        envelope.ciphertext,
                        _datetime_to_iso(record.created_at),
                        _datetime_to_iso(record.expires_at),
                    ),
                )
            except sqlite3.IntegrityError as exc:
                raise ValueError("Failed to insert upstream OAuth state") from exc

        return record

    def get_state(
        self,
        *,
        state_id: str,
    ) -> Optional[UpstreamOAuthStateRecord]:
        with sqlite_connection(self._db_path) as connection:
            row = connection.execute(
                """
                SELECT
                    state_id,
                    server_path,
                    credential_type,
                    credential_binding,
                    user_id,
                    agent_id,
                    provider,
                    redirect_uri,
                    created_at,
                    expires_at
                FROM upstream_oauth_states
                WHERE state_id = ?
                """.strip(),
                (state_id,),
            ).fetchone()

        if row is None:
            return None

        return UpstreamOAuthStateRecord(
            state_id=row[0],
            server_path=row[1],
            credential_type=row[2],
            credential_binding=row[3],
            user_id=row[4],
            agent_id=row[5],
            provider=row[6],
            redirect_uri=row[7],
            created_at=_datetime_from_iso(row[8]),
            expires_at=_datetime_from_iso(row[9]),
        )

    def consume_state(
        self,
        *,
        state_id: str,
    ) -> Optional[tuple[UpstreamOAuthStateRecord, UpstreamOAuthStateSecret]]:
        now = _utc_now()

        with sqlite_connection(self._db_path) as connection:
            connection.isolation_level = None
            try:
                connection.execute("BEGIN IMMEDIATE")
                row = connection.execute(
                    """
                    SELECT
                        state_id,
                        server_path,
                        credential_type,
                        credential_binding,
                        user_id,
                        agent_id,
                        provider,
                        redirect_uri,
                        secret_version,
                        secret_nonce,
                        secret_ciphertext,
                        created_at,
                        expires_at
                    FROM upstream_oauth_states
                    WHERE state_id = ?
                    """.strip(),
                    (state_id,),
                ).fetchone()

                if row is None:
                    connection.execute("COMMIT")
                    return None

                expires_at = _datetime_from_iso(row[12])
                connection.execute(
                    "DELETE FROM upstream_oauth_states WHERE state_id = ?",
                    (state_id,),
                )
                connection.execute("COMMIT")
            except Exception:
                connection.execute("ROLLBACK")
                raise
            finally:
                connection.isolation_level = ""

        if expires_at <= now:
            return None

        record = UpstreamOAuthStateRecord(
            state_id=row[0],
            server_path=row[1],
            credential_type=row[2],
            credential_binding=row[3],
            user_id=row[4],
            agent_id=row[5],
            provider=row[6],
            redirect_uri=row[7],
            created_at=_datetime_from_iso(row[11]),
            expires_at=expires_at,
        )

        secret_version = int(row[8])
        secret_nonce = bytes(row[9])
        secret_ciphertext = bytes(row[10])

        aad = build_aad_for_upstream_oauth_state(
            state_id=record.state_id,
            server_path=record.server_path,
            credential_type=record.credential_type,
        )
        payload = decrypt_secret_payload(
            key=self._kek,
            aad=aad,
            envelope=EncryptedSecretEnvelope(
                version=secret_version,
                nonce=secret_nonce,
                ciphertext=secret_ciphertext,
            ),
        )
        secret = UpstreamOAuthStateSecret(
            state_id=record.state_id,
            payload=payload,
        )
        return record, secret
