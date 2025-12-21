from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import (
    datetime,
    timezone,
)
from pathlib import Path
from typing import (
    Any,
    Optional,
)

from ...crypto.upstream_secrets import (
    EncryptedSecretEnvelope,
    build_aad_for_upstream_credential,
    decrypt_secret_payload,
    encrypt_secret_payload,
)
from ...db.connection import (
    sqlite_connection,
)
from ...models.upstream_credentials import (
    UpstreamCredentialRecord,
    UpstreamCredentialSecret,
)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _datetime_to_iso(
    value: Optional[datetime],
) -> Optional[str]:
    if value is None:
        return None
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc).isoformat().replace(
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


class SqliteUpstreamCredentialStore:
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

    def create_credential(
        self,
        *,
        server_path: str,
        credential_type: str,
        credential_binding: str,
        user_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        provider: Optional[str] = None,
        scopes: Optional[list[str]] = None,
        token_type: Optional[str] = None,
        expires_at: Optional[datetime] = None,
        secret_payload: Optional[dict[str, object]] = None,
    ) -> UpstreamCredentialRecord:
        now = _utc_now()
        credential_id = str(uuid.uuid4())

        validated = UpstreamCredentialRecord(
            credential_id=credential_id,
            server_path=server_path,
            credential_type=credential_type,
            credential_binding=credential_binding,
            user_id=user_id,
            agent_id=agent_id,
            provider=provider,
            scopes=scopes,
            token_type=token_type,
            expires_at=expires_at,
            revoked_at=None,
            last_used_at=None,
            created_at=now,
            updated_at=now,
        )

        secret_version: Optional[int] = None
        secret_nonce: Optional[bytes] = None
        secret_ciphertext: Optional[bytes] = None

        if secret_payload is not None:
            aad = build_aad_for_upstream_credential(
                credential_id=validated.credential_id,
                server_path=validated.server_path,
                credential_type=validated.credential_type,
            )
            envelope = encrypt_secret_payload(
                key=self._kek,
                aad=aad,
                payload=dict(secret_payload),
            )
            secret_version = envelope.version
            secret_nonce = envelope.nonce
            secret_ciphertext = envelope.ciphertext

        with sqlite_connection(self._db_path) as connection:
            try:
                connection.execute(
                    """
                    INSERT INTO upstream_credentials(
                        credential_id,
                        server_path,
                        credential_type,
                        credential_binding,
                        user_id,
                        agent_id,
                        provider,
                        scopes_json,
                        token_type,
                        secret_version,
                        secret_nonce,
                        secret_ciphertext,
                        created_at,
                        updated_at,
                        expires_at,
                        revoked_at,
                        last_used_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """.strip(),
                    (
                        validated.credential_id,
                        validated.server_path,
                        validated.credential_type,
                        validated.credential_binding,
                        validated.user_id,
                        validated.agent_id,
                        validated.provider,
                        _json_dumps(validated.scopes) if validated.scopes is not None else None,
                        validated.token_type,
                        secret_version,
                        secret_nonce,
                        secret_ciphertext,
                        _datetime_to_iso(validated.created_at),
                        _datetime_to_iso(validated.updated_at),
                        _datetime_to_iso(validated.expires_at),
                        None,
                        None,
                    ),
                )
            except sqlite3.IntegrityError as exc:
                raise ValueError("Failed to insert upstream credential") from exc

        record = self.get_credential_by_id(credential_id=validated.credential_id)
        if record is None:
            raise RuntimeError(
                "Upstream credential insert succeeded but record could not be read back"
            )
        return record

    def get_credential_by_id(
        self,
        *,
        credential_id: str,
    ) -> Optional[UpstreamCredentialRecord]:
        with sqlite_connection(self._db_path) as connection:
            row = connection.execute(
                """
                SELECT
                    credential_id,
                    server_path,
                    credential_type,
                    credential_binding,
                    user_id,
                    agent_id,
                    provider,
                    scopes_json,
                    token_type,
                    expires_at,
                    revoked_at,
                    last_used_at,
                    created_at,
                    updated_at
                FROM upstream_credentials
                WHERE credential_id = ?
                """.strip(),
                (credential_id,),
            ).fetchone()

        if row is None:
            return None

        scopes = _json_loads_optional(row[7])
        if scopes is not None and not isinstance(scopes, list):
            raise ValueError("Invalid scopes_json stored for upstream_credentials record")

        return UpstreamCredentialRecord(
            credential_id=row[0],
            server_path=row[1],
            credential_type=row[2],
            credential_binding=row[3],
            user_id=row[4],
            agent_id=row[5],
            provider=row[6],
            scopes=scopes,
            token_type=row[8],
            expires_at=_datetime_from_iso(row[9]),
            revoked_at=_datetime_from_iso(row[10]),
            last_used_at=_datetime_from_iso(row[11]),
            created_at=_datetime_from_iso(row[12]),
            updated_at=_datetime_from_iso(row[13]),
        )

    def get_credential_secret(
        self,
        *,
        credential_id: str,
    ) -> Optional[UpstreamCredentialSecret]:
        with sqlite_connection(self._db_path) as connection:
            row = connection.execute(
                """
                SELECT
                    credential_id,
                    server_path,
                    credential_type,
                    secret_version,
                    secret_nonce,
                    secret_ciphertext
                FROM upstream_credentials
                WHERE credential_id = ?
                """.strip(),
                (credential_id,),
            ).fetchone()

        if row is None:
            return None

        secret_version = row[3]
        secret_nonce = row[4]
        secret_ciphertext = row[5]

        if secret_version is None or secret_nonce is None or secret_ciphertext is None:
            return None

        aad = build_aad_for_upstream_credential(
            credential_id=row[0],
            server_path=row[1],
            credential_type=row[2],
        )
        payload = decrypt_secret_payload(
            key=self._kek,
            aad=aad,
            envelope=EncryptedSecretEnvelope(
                version=int(secret_version),
                nonce=bytes(secret_nonce),
                ciphertext=bytes(secret_ciphertext),
            ),
        )

        return UpstreamCredentialSecret(
            credential_id=row[0],
            payload=payload,
        )

    def list_credentials(
        self,
        *,
        server_path: Optional[str] = None,
        user_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        include_revoked: bool = False,
    ) -> list[UpstreamCredentialRecord]:
        filters: list[str] = []
        params: list[object] = []

        if server_path is not None:
            filters.append("server_path = ?")
            params.append(server_path)
        if user_id is not None:
            filters.append("user_id = ?")
            params.append(user_id)
        if agent_id is not None:
            filters.append("agent_id = ?")
            params.append(agent_id)
        if not include_revoked:
            filters.append("revoked_at IS NULL")

        where_clause = " AND ".join(filters) if filters else "1=1"
        query = (
            """
            SELECT
                credential_id,
                server_path,
                credential_type,
                credential_binding,
                user_id,
                agent_id,
                provider,
                scopes_json,
                token_type,
                expires_at,
                revoked_at,
                last_used_at,
                created_at,
                updated_at
            FROM upstream_credentials
            WHERE {where_clause}
            ORDER BY created_at ASC
            """.format(where_clause=where_clause).strip()
        )

        with sqlite_connection(self._db_path) as connection:
            rows = connection.execute(
                query,
                tuple(params),
            ).fetchall()

        records: list[UpstreamCredentialRecord] = []
        for row in rows:
            scopes = _json_loads_optional(row[7])
            if scopes is not None and not isinstance(scopes, list):
                raise ValueError("Invalid scopes_json stored for upstream_credentials record")
            records.append(
                UpstreamCredentialRecord(
                    credential_id=row[0],
                    server_path=row[1],
                    credential_type=row[2],
                    credential_binding=row[3],
                    user_id=row[4],
                    agent_id=row[5],
                    provider=row[6],
                    scopes=scopes,
                    token_type=row[8],
                    expires_at=_datetime_from_iso(row[9]),
                    revoked_at=_datetime_from_iso(row[10]),
                    last_used_at=_datetime_from_iso(row[11]),
                    created_at=_datetime_from_iso(row[12]),
                    updated_at=_datetime_from_iso(row[13]),
                )
            )
        return records

    def revoke_credential(
        self,
        *,
        credential_id: str,
        revoked_at: Optional[datetime] = None,
    ) -> Optional[UpstreamCredentialRecord]:
        existing = self.get_credential_by_id(credential_id=credential_id)
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
                UPDATE upstream_credentials
                SET revoked_at = ?, updated_at = ?
                WHERE credential_id = ?
                """.strip(),
                (
                    _datetime_to_iso(now),
                    _datetime_to_iso(_utc_now()),
                    credential_id,
                ),
            )

        return self.get_credential_by_id(credential_id=credential_id)

    def update_last_used_at(
        self,
        *,
        credential_id: str,
        last_used_at: datetime,
    ) -> Optional[UpstreamCredentialRecord]:
        existing = self.get_credential_by_id(credential_id=credential_id)
        if existing is None:
            return None

        ts = last_used_at
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)

        with sqlite_connection(self._db_path) as connection:
            connection.execute(
                """
                UPDATE upstream_credentials
                SET last_used_at = ?, updated_at = ?
                WHERE credential_id = ?
                """.strip(),
                (
                    _datetime_to_iso(ts),
                    _datetime_to_iso(_utc_now()),
                    credential_id,
                ),
            )

        return self.get_credential_by_id(credential_id=credential_id)

    def update_credential(
        self,
        *,
        credential_id: str,
        token_type: Optional[str] = None,
        scopes: Optional[list[str]] = None,
        expires_at: Optional[datetime] = None,
        secret_payload: Optional[dict[str, object]] = None,
    ) -> Optional[UpstreamCredentialRecord]:
        existing = self.get_credential_by_id(credential_id=credential_id)
        if existing is None:
            return None

        updated_token_type = existing.token_type if token_type is None else token_type
        updated_scopes = existing.scopes if scopes is None else scopes
        updated_expires_at = existing.expires_at if expires_at is None else expires_at

        secret_version: Optional[int] = None
        secret_nonce: Optional[bytes] = None
        secret_ciphertext: Optional[bytes] = None
        set_secret = secret_payload is not None
        if set_secret:
            aad = build_aad_for_upstream_credential(
                credential_id=existing.credential_id,
                server_path=existing.server_path,
                credential_type=existing.credential_type,
            )
            envelope = encrypt_secret_payload(
                key=self._kek,
                aad=aad,
                payload=dict(secret_payload),
            )
            secret_version = envelope.version
            secret_nonce = envelope.nonce
            secret_ciphertext = envelope.ciphertext

        with sqlite_connection(self._db_path) as connection:
            if set_secret:
                connection.execute(
                    """
                    UPDATE upstream_credentials
                    SET
                        scopes_json = ?,
                        token_type = ?,
                        expires_at = ?,
                        secret_version = ?,
                        secret_nonce = ?,
                        secret_ciphertext = ?,
                        updated_at = ?
                    WHERE credential_id = ?
                    """.strip(),
                    (
                        _json_dumps(updated_scopes) if updated_scopes is not None else None,
                        updated_token_type,
                        _datetime_to_iso(updated_expires_at),
                        secret_version,
                        secret_nonce,
                        secret_ciphertext,
                        _datetime_to_iso(_utc_now()),
                        credential_id,
                    ),
                )
            else:
                connection.execute(
                    """
                    UPDATE upstream_credentials
                    SET
                        scopes_json = ?,
                        token_type = ?,
                        expires_at = ?,
                        updated_at = ?
                    WHERE credential_id = ?
                    """.strip(),
                    (
                        _json_dumps(updated_scopes) if updated_scopes is not None else None,
                        updated_token_type,
                        _datetime_to_iso(updated_expires_at),
                        _datetime_to_iso(_utc_now()),
                        credential_id,
                    ),
                )

        return self.get_credential_by_id(credential_id=credential_id)
