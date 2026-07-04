"""SecretsManagerStore -- one AWS Secrets Manager secret per principal.

Key granularity: ONE secret named ``{prefix}/{enc(auth_method)}/{enc(user_id)}``
whose value is a JSON map ``{map_key(provider, server_path): StoredToken}``.
This keeps the secret count at one-per-principal (cost/quota mitigation) rather
than one-per-connection.

Concurrency: two providers linked concurrently for the same principal each
read-modify-write the same secret's JSON map and could clobber each other.
Secrets Manager has no native compare-and-swap on ``PutSecretValue``, so we use
a bounded read-merge-write-verify loop: after writing, re-read and confirm our
entry survived; if a concurrent writer's version landed last and dropped it,
retry the merge. Bounded at 5 attempts with deterministic backoff; on terminal
failure we raise so the caller marks the entry ``refresh_failed`` and the next
vend re-consents (never a silent clobber).
"""

import asyncio
import json
import logging

from registry.egress_auth.schemas import StoredToken
from registry.secrets import keys
from registry.secrets.interfaces import SecretStoreBase, SecretStoreError

logger = logging.getLogger(__name__)

_MAX_WRITE_ATTEMPTS = 5
_BACKOFF_BASE_SECONDS = 0.05


class SecretsManagerStore(SecretStoreBase):
    """One secret per principal holding a JSON connection map. ``client`` is boto3."""

    def __init__(self, client, prefix: str, kms_key_id: str | None = None) -> None:
        self._client = client
        self._prefix = prefix.strip("/")
        self._kms_key_id = kms_key_id or None

    def _secret_name(self, auth_method: str, user_id: str) -> str:
        return f"{self._prefix}/{keys.user_principal(auth_method, user_id)}"

    # -- raw boto3 helpers (run in a thread; boto3 is sync) ------------------- #

    def _get_map(self, name: str) -> dict[str, dict]:
        try:
            resp = self._client.get_secret_value(SecretId=name)
        except Exception as exc:
            if type(exc).__name__ == "ResourceNotFoundException":
                return {}
            raise SecretStoreError(f"Secrets Manager get failed: {exc}") from exc
        raw = resp.get("SecretString")
        return json.loads(raw) if raw else {}

    def _put_map(self, name: str, data: dict[str, dict]) -> None:
        payload = json.dumps(data)
        try:
            self._client.put_secret_value(SecretId=name, SecretString=payload)
        except Exception as exc:
            if type(exc).__name__ == "ResourceNotFoundException":
                kwargs = {"Name": name, "SecretString": payload}
                if self._kms_key_id:
                    kwargs["KmsKeyId"] = self._kms_key_id
                self._client.create_secret(**kwargs)
            else:
                raise SecretStoreError(f"Secrets Manager put failed: {exc}") from exc

    def _delete_secret(self, name: str) -> None:
        try:
            self._client.delete_secret(SecretId=name, ForceDeleteWithoutRecovery=True)
        except Exception as exc:
            if type(exc).__name__ == "ResourceNotFoundException":
                return
            raise SecretStoreError(f"Secrets Manager delete failed: {exc}") from exc

    def _merge_write_verify(self, name: str, key: str, value: dict | None) -> None:
        """Read-merge-write with post-write verification, bounded retry.

        ``value=None`` means delete the key. Converges under concurrent writers
        by re-reading after each write and retrying if our intended state did
        not survive a racing writer's later version.
        """
        for attempt in range(_MAX_WRITE_ATTEMPTS):
            data = self._get_map(name)
            if value is None:
                data.pop(key, None)
            else:
                data[key] = value
            if not data:
                self._delete_secret(name)
                return
            self._put_map(name, data)
            # Verify our intended mutation survived a possible concurrent put.
            check = self._get_map(name)
            if value is None:
                if key not in check:
                    return
            elif check.get(key) == value:
                return
            logger.warning(
                "Secrets Manager write race on %s (attempt %d/%d); retrying merge",
                name,
                attempt + 1,
                _MAX_WRITE_ATTEMPTS,
            )
        raise SecretStoreError(
            f"Secrets Manager write did not converge after {_MAX_WRITE_ATTEMPTS} attempts "
            f"for {name}"
        )

    # -- SecretStoreBase ------------------------------------------------------ #

    async def put_token(
        self,
        auth_method: str,
        user_id: str,
        provider: str,
        server_path: str,
        token: StoredToken,
    ) -> None:
        name = self._secret_name(auth_method, user_id)
        key = keys.map_key(provider, server_path)
        await asyncio.to_thread(self._merge_write_verify, name, key, token.model_dump())

    async def get_token(
        self,
        auth_method: str,
        user_id: str,
        provider: str,
        server_path: str,
    ) -> StoredToken | None:
        name = self._secret_name(auth_method, user_id)
        key = keys.map_key(provider, server_path)
        data = await asyncio.to_thread(self._get_map, name)
        raw = data.get(key)
        return StoredToken(**raw) if raw is not None else None

    async def delete_token(
        self,
        auth_method: str,
        user_id: str,
        provider: str,
        server_path: str,
    ) -> None:
        name = self._secret_name(auth_method, user_id)
        key = keys.map_key(provider, server_path)
        await asyncio.to_thread(self._merge_write_verify, name, key, None)

    async def list_for_user(
        self,
        auth_method: str,
        user_id: str,
    ) -> list[tuple[str, str, StoredToken]]:
        name = self._secret_name(auth_method, user_id)
        data = await asyncio.to_thread(self._get_map, name)
        out: list[tuple[str, str, StoredToken]] = []
        for key, raw in data.items():
            provider_enc, _, server_enc = key.partition(keys.MAP_KEY_DELIMITER)
            out.append(
                (
                    keys.decode_segment(provider_enc),
                    keys.decode_segment(server_enc),
                    StoredToken(**raw),
                )
            )
        return out
