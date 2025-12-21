from __future__ import annotations

import json
import sqlite3
from datetime import (
    datetime,
    timezone,
)
from pathlib import Path
from typing import (
    Optional,
)

from ...crypto.upstream_secrets import (
    EncryptedSecretEnvelope,
    build_aad_for_upstream_oauth_provider,
    decrypt_secret_payload,
    encrypt_secret_payload,
)
from ...db.connection import (
    sqlite_connection,
)
from ...models.upstream_oauth_provider import (
    UpstreamOAuthProviderCreate,
    UpstreamOAuthProviderPublic,
    UpstreamOAuthProviderRecord,
    UpstreamOAuthProviderUpdate,
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


class SqliteUpstreamOAuthProviderStore:
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

    def create_provider(
        self,
        *,
        payload: UpstreamOAuthProviderCreate,
    ) -> UpstreamOAuthProviderPublic:
        now = _utc_now()
        validated = UpstreamOAuthProviderRecord(
            provider_id=payload.provider_id,
            authorization_endpoint=payload.authorization_endpoint,
            token_endpoint=payload.token_endpoint,
            client_id=payload.client_id,
            default_scopes=payload.default_scopes,
            extra_authorize_params=payload.extra_authorize_params,
            created_at=now,
            updated_at=now,
        )

        aad = build_aad_for_upstream_oauth_provider(
            provider_id=validated.provider_id,
        )
        envelope = encrypt_secret_payload(
            key=self._kek,
            aad=aad,
            payload={"client_secret": payload.client_secret},
        )

        with sqlite_connection(self._db_path) as connection:
            try:
                connection.execute(
                    """
                    INSERT INTO upstream_oauth_providers(
                        provider_id,
                        authorization_endpoint,
                        token_endpoint,
                        client_id,
                        default_scopes_json,
                        extra_authorize_params_json,
                        secret_version,
                        secret_nonce,
                        secret_ciphertext,
                        created_at,
                        updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """.strip(),
                    (
                        validated.provider_id,
                        validated.authorization_endpoint,
                        validated.token_endpoint,
                        validated.client_id,
                        _json_dumps(validated.default_scopes),
                        _json_dumps(validated.extra_authorize_params),
                        envelope.version,
                        envelope.nonce,
                        envelope.ciphertext,
                        _datetime_to_iso(validated.created_at),
                        _datetime_to_iso(validated.updated_at),
                    ),
                )
            except sqlite3.IntegrityError as exc:
                raise ValueError(
                    f"Upstream OAuth provider already exists: {validated.provider_id}"
                ) from exc

        created = self.get_provider(provider_id=validated.provider_id)
        if created is None:
            raise RuntimeError(
                "Upstream OAuth provider insert succeeded but record could not be read back"
            )
        return created

    def get_provider(
        self,
        *,
        provider_id: str,
    ) -> Optional[UpstreamOAuthProviderPublic]:
        with sqlite_connection(self._db_path) as connection:
            row = connection.execute(
                """
                SELECT
                    provider_id,
                    authorization_endpoint,
                    token_endpoint,
                    client_id,
                    default_scopes_json,
                    extra_authorize_params_json,
                    secret_version,
                    secret_nonce,
                    secret_ciphertext,
                    created_at,
                    updated_at
                FROM upstream_oauth_providers
                WHERE provider_id = ?
                """.strip(),
                (provider_id,),
            ).fetchone()

        if row is None:
            return None

        default_scopes = _json_loads_optional(row[4])
        if default_scopes is None:
            default_scopes = []
        extra_authorize_params = _json_loads_optional(row[5])
        if extra_authorize_params is None:
            extra_authorize_params = {}

        secret_present = row[8] is not None

        return UpstreamOAuthProviderPublic(
            provider=UpstreamOAuthProviderRecord(
                provider_id=row[0],
                authorization_endpoint=row[1],
                token_endpoint=row[2],
                client_id=row[3],
                default_scopes=default_scopes,
                extra_authorize_params=extra_authorize_params,
                created_at=_datetime_from_iso(row[9]),
                updated_at=_datetime_from_iso(row[10]),
            ),
            secret_present=secret_present,
        )

    def list_providers(
        self,
    ) -> list[UpstreamOAuthProviderPublic]:
        with sqlite_connection(self._db_path) as connection:
            rows = connection.execute(
                """
                SELECT
                    provider_id,
                    authorization_endpoint,
                    token_endpoint,
                    client_id,
                    default_scopes_json,
                    extra_authorize_params_json,
                    secret_ciphertext,
                    created_at,
                    updated_at
                FROM upstream_oauth_providers
                ORDER BY provider_id ASC
                """.strip()
            ).fetchall()

        providers: list[UpstreamOAuthProviderPublic] = []
        for row in rows:
            default_scopes = _json_loads_optional(row[4])
            if default_scopes is None:
                default_scopes = []
            extra_authorize_params = _json_loads_optional(row[5])
            if extra_authorize_params is None:
                extra_authorize_params = {}

            providers.append(
                UpstreamOAuthProviderPublic(
                    provider=UpstreamOAuthProviderRecord(
                        provider_id=row[0],
                        authorization_endpoint=row[1],
                        token_endpoint=row[2],
                        client_id=row[3],
                        default_scopes=default_scopes,
                        extra_authorize_params=extra_authorize_params,
                        created_at=_datetime_from_iso(row[7]),
                        updated_at=_datetime_from_iso(row[8]),
                    ),
                    secret_present=row[6] is not None,
                )
            )

        return providers

    def update_provider(
        self,
        *,
        provider_id: str,
        payload: UpstreamOAuthProviderUpdate,
    ) -> Optional[UpstreamOAuthProviderPublic]:
        existing = self.get_provider(provider_id=provider_id)
        if existing is None:
            return None

        updated_at = _utc_now()
        merged = UpstreamOAuthProviderRecord(
            provider_id=existing.provider.provider_id,
            authorization_endpoint=payload.authorization_endpoint
            if payload.authorization_endpoint is not None
            else existing.provider.authorization_endpoint,
            token_endpoint=payload.token_endpoint
            if payload.token_endpoint is not None
            else existing.provider.token_endpoint,
            client_id=payload.client_id if payload.client_id is not None else existing.provider.client_id,
            default_scopes=payload.default_scopes
            if payload.default_scopes is not None
            else existing.provider.default_scopes,
            extra_authorize_params=payload.extra_authorize_params
            if payload.extra_authorize_params is not None
            else existing.provider.extra_authorize_params,
            created_at=existing.provider.created_at,
            updated_at=updated_at,
        )

        secret_version: Optional[int] = None
        secret_nonce: Optional[bytes] = None
        secret_ciphertext: Optional[bytes] = None

        if payload.client_secret is not None:
            aad = build_aad_for_upstream_oauth_provider(
                provider_id=merged.provider_id,
            )
            envelope = encrypt_secret_payload(
                key=self._kek,
                aad=aad,
                payload={"client_secret": payload.client_secret},
            )
            secret_version = envelope.version
            secret_nonce = envelope.nonce
            secret_ciphertext = envelope.ciphertext

        with sqlite_connection(self._db_path) as connection:
            if payload.client_secret is not None:
                connection.execute(
                    """
                    UPDATE upstream_oauth_providers
                    SET
                        authorization_endpoint = ?,
                        token_endpoint = ?,
                        client_id = ?,
                        default_scopes_json = ?,
                        extra_authorize_params_json = ?,
                        secret_version = ?,
                        secret_nonce = ?,
                        secret_ciphertext = ?,
                        updated_at = ?
                    WHERE provider_id = ?
                    """.strip(),
                    (
                        merged.authorization_endpoint,
                        merged.token_endpoint,
                        merged.client_id,
                        _json_dumps(merged.default_scopes),
                        _json_dumps(merged.extra_authorize_params),
                        secret_version,
                        secret_nonce,
                        secret_ciphertext,
                        _datetime_to_iso(merged.updated_at),
                        merged.provider_id,
                    ),
                )
            else:
                connection.execute(
                    """
                    UPDATE upstream_oauth_providers
                    SET
                        authorization_endpoint = ?,
                        token_endpoint = ?,
                        client_id = ?,
                        default_scopes_json = ?,
                        extra_authorize_params_json = ?,
                        updated_at = ?
                    WHERE provider_id = ?
                    """.strip(),
                    (
                        merged.authorization_endpoint,
                        merged.token_endpoint,
                        merged.client_id,
                        _json_dumps(merged.default_scopes),
                        _json_dumps(merged.extra_authorize_params),
                        _datetime_to_iso(merged.updated_at),
                        merged.provider_id,
                    ),
                )

        return self.get_provider(provider_id=merged.provider_id)

    def delete_provider(
        self,
        *,
        provider_id: str,
    ) -> bool:
        with sqlite_connection(self._db_path) as connection:
            cursor = connection.execute(
                "DELETE FROM upstream_oauth_providers WHERE provider_id = ?",
                (provider_id,),
            )
            return cursor.rowcount > 0

    def get_provider_secret_for_runtime(
        self,
        *,
        provider_id: str,
    ) -> Optional[str]:
        with sqlite_connection(self._db_path) as connection:
            row = connection.execute(
                """
                SELECT
                    secret_version,
                    secret_nonce,
                    secret_ciphertext
                FROM upstream_oauth_providers
                WHERE provider_id = ?
                """.strip(),
                (provider_id,),
            ).fetchone()

        if row is None:
            return None

        if row[2] is None:
            return None

        envelope = EncryptedSecretEnvelope(
            version=row[0],
            nonce=row[1],
            ciphertext=row[2],
        )
        aad = build_aad_for_upstream_oauth_provider(
            provider_id=provider_id,
        )
        payload = decrypt_secret_payload(
            key=self._kek,
            aad=aad,
            envelope=envelope,
        )
        raw_secret = payload.get("client_secret")
        if not isinstance(raw_secret, str) or not raw_secret.strip():
            raise ValueError("Stored upstream OAuth provider secret is missing client_secret")
        return raw_secret

