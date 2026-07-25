"""AWS Secrets Manager egress-token store.

The normal representation remains the legacy, backward-compatible JSON map in
one secret per canonical principal. If that map would exceed the operating size
target, it is replaced by a small versioned root manifest and deterministic
hash shards. Tokens never leave Secrets Manager; Mongo is used only for a
short-lived principal mutation lease because Secrets Manager has no CAS API.
"""

import asyncio
import hashlib
import json
import logging
import uuid
from collections.abc import Callable
from contextlib import asynccontextmanager
from typing import Protocol

from registry.egress_auth.schemas import StoredToken
from registry.secrets import keys
from registry.secrets.interfaces import SecretStoreBase, SecretStoreError

logger = logging.getLogger(__name__)

_AWS_SECRET_VALUE_MAX_BYTES = 65_536
_DEFAULT_TARGET_BYTES = 60 * 1024
_DEFAULT_MAX_SHARDS = 64
_MAX_PARALLEL_REQUESTS = 8
_LEASE_TTL_SECONDS = 120
_LEASE_WAIT_SECONDS = 30
_META_KEY = "_egress"
_SCHEMA_VERSION = 1
_HASH_ALGORITHM = "sha256-v1"
_STALE_READ_RETRIES = 3


class _StaleShardError(SecretStoreError):
    """A required overflow shard was absent mid-read.

    A reader can observe generation G1 in the root, then a concurrent
    same-principal commit flips the manifest to G2 and cleans up G1's shards
    before the reader fetches them. That is not corruption: re-reading the root
    yields the new generation (or an inline/empty layout). Readers translate a
    missing required shard into this signal and retry the whole read; it is never
    surfaced to the caller.
    """


class MutationLease(Protocol):
    """Distributed operational lease; implementations store no credentials."""

    async def acquire_lease(self, key: str, holder: str, ttl_seconds: int) -> bool: ...

    async def renew_lease(self, key: str, holder: str, ttl_seconds: int) -> bool: ...

    async def release_lease(self, key: str, holder: str) -> None: ...


class _LeaseState:
    def __init__(self) -> None:
        self.lost = False

    def ensure_held(self) -> None:
        if self.lost:
            raise SecretStoreError("Secrets Manager principal mutation lease was lost")


class SecretsManagerStore(SecretStoreBase):
    """One inline secret per principal, with transparent overflow hash shards."""

    def __init__(
        self,
        client,
        prefix: str,
        kms_key_id: str | None = None,
        mutation_lease: MutationLease | None = None,
        target_payload_bytes: int = _DEFAULT_TARGET_BYTES,
        max_shards: int = _DEFAULT_MAX_SHARDS,
    ) -> None:
        if not 1024 <= target_payload_bytes <= _AWS_SECRET_VALUE_MAX_BYTES:
            raise ValueError("target_payload_bytes must be between 1024 and 65536")
        if max_shards < 1:
            raise ValueError("max_shards must be positive")
        self._client = client
        self._prefix = prefix.strip("/")
        self._kms_key_id = kms_key_id or None
        self._mutation_lease = mutation_lease
        self._target_bytes = target_payload_bytes
        self._max_shards = max_shards
        self._local_mutation_lock = asyncio.Lock()
        self._request_limit = asyncio.Semaphore(_MAX_PARALLEL_REQUESTS)

    def _secret_name(self, auth_method: str, user_id: str) -> str:
        return f"{self._prefix}/{keys.user_principal(auth_method, user_id)}"

    @staticmethod
    def _serialize(document: dict) -> str:
        return json.dumps(document, ensure_ascii=False, separators=(",", ":"), sort_keys=True)

    @classmethod
    def _payload_size(cls, document: dict) -> int:
        return len(cls._serialize(document).encode("utf-8"))

    @staticmethod
    def _bucket(key: str, bucket_count: int) -> int:
        digest = hashlib.sha256(key.encode("utf-8")).digest()
        return int.from_bytes(digest[:8], "big") % bucket_count

    def _manifest(self, document: dict) -> dict | None:
        meta = document.get(_META_KEY)
        if meta is None:
            return None
        if set(document) != {_META_KEY} or not isinstance(meta, dict):
            raise SecretStoreError("Secrets Manager egress root manifest is malformed")
        if (
            meta.get("schema") != _SCHEMA_VERSION
            or meta.get("layout") != "sharded"
            or meta.get("hash") != _HASH_ALGORITHM
            or not isinstance(meta.get("generation"), str)
            or not isinstance(meta.get("bucket_count"), int)
            or not 1 <= meta["bucket_count"] <= self._max_shards
        ):
            raise SecretStoreError("Secrets Manager egress root manifest is unsupported")
        return meta

    def _shard_name(self, root_name: str, generation: str, bucket: int) -> str:
        return f"{root_name}/overflow/{generation}/{bucket}"

    def _shard_document(
        self, generation: str, bucket: int, bucket_count: int, entries: dict[str, dict]
    ) -> dict:
        return {
            _META_KEY: {
                "schema": _SCHEMA_VERSION,
                "layout": "shard",
                "generation": generation,
                "bucket": bucket,
                "bucket_count": bucket_count,
            },
            "entries": entries,
        }

    # -- raw boto3 helpers (all callers run these in a worker thread) -------- #

    def _get_document(self, name: str, required: bool = False) -> dict | None:
        try:
            resp = self._client.get_secret_value(SecretId=name)
        except Exception as exc:
            if type(exc).__name__ == "ResourceNotFoundException":
                if required:
                    raise SecretStoreError(f"Secrets Manager overflow shard is missing: {name}")
                return None
            raise SecretStoreError(f"Secrets Manager get failed: {exc}") from exc
        raw = resp.get("SecretString")
        if not raw:
            if required:
                raise SecretStoreError(f"Secrets Manager overflow shard is empty: {name}")
            return {}
        try:
            document = json.loads(raw)
        except (TypeError, json.JSONDecodeError) as exc:
            raise SecretStoreError(f"Secrets Manager value is not valid JSON: {name}") from exc
        if not isinstance(document, dict):
            raise SecretStoreError(f"Secrets Manager value is not a JSON object: {name}")
        return document

    def _put_document(self, name: str, document: dict) -> None:
        payload = self._serialize(document)
        size = len(payload.encode("utf-8"))
        if size > _AWS_SECRET_VALUE_MAX_BYTES:
            raise SecretStoreError(
                f"Secrets Manager payload exceeds {_AWS_SECRET_VALUE_MAX_BYTES} bytes"
            )
        try:
            self._client.put_secret_value(SecretId=name, SecretString=payload)
        except Exception as exc:
            if type(exc).__name__ != "ResourceNotFoundException":
                raise SecretStoreError(f"Secrets Manager put failed: {exc}") from exc
            kwargs = {"Name": name, "SecretString": payload}
            if self._kms_key_id:
                kwargs["KmsKeyId"] = self._kms_key_id
            try:
                self._client.create_secret(**kwargs)
            except Exception as create_exc:
                raise SecretStoreError(
                    f"Secrets Manager create failed: {create_exc}"
                ) from create_exc

    def _delete_secret(self, name: str) -> None:
        try:
            self._client.delete_secret(SecretId=name, ForceDeleteWithoutRecovery=True)
        except Exception as exc:
            if type(exc).__name__ == "ResourceNotFoundException":
                return
            raise SecretStoreError(f"Secrets Manager delete failed: {exc}") from exc

    async def _call(self, fn: Callable, *args):
        async with self._request_limit:
            return await asyncio.to_thread(fn, *args)

    # -- document decoding --------------------------------------------------- #

    def _decode_shard(
        self, document: dict, generation: str, bucket: int, bucket_count: int
    ) -> dict[str, dict]:
        meta = document.get(_META_KEY)
        entries = document.get("entries")
        if (
            set(document) != {_META_KEY, "entries"}
            or not isinstance(meta, dict)
            or meta.get("schema") != _SCHEMA_VERSION
            or meta.get("layout") != "shard"
            or meta.get("generation") != generation
            or meta.get("bucket") != bucket
            or meta.get("bucket_count") != bucket_count
            or not isinstance(entries, dict)
        ):
            raise SecretStoreError("Secrets Manager overflow shard is malformed")
        return entries

    async def _read_shard(self, root_name: str, meta: dict, bucket: int) -> dict[str, dict]:
        name = self._shard_name(root_name, meta["generation"], bucket)
        # A non-required read distinguishes "shard cleaned up because the manifest
        # already moved on" (retryable stale read) from a genuinely corrupt layout.
        document = await self._call(self._get_document, name, False)
        if document is None:
            raise _StaleShardError(f"Secrets Manager overflow shard vanished mid-read: {name}")
        return self._decode_shard(document, meta["generation"], bucket, meta["bucket_count"])

    async def _load_entries(self, root_name: str, root: dict | None) -> dict[str, dict]:
        if root is None:
            return {}
        meta = self._manifest(root)
        if meta is None:
            if not all(isinstance(value, dict) for value in root.values()):
                raise SecretStoreError("Secrets Manager legacy egress document is malformed")
            return dict(root)
        shards = await asyncio.gather(
            *(self._read_shard(root_name, meta, bucket) for bucket in range(meta["bucket_count"]))
        )
        entries: dict[str, dict] = {}
        for shard_entries in shards:
            overlap = entries.keys() & shard_entries.keys()
            if overlap:
                raise SecretStoreError("Secrets Manager overflow shards contain duplicate keys")
            entries.update(shard_entries)
        return entries

    # -- principal mutation lease ------------------------------------------- #

    @asynccontextmanager
    async def _mutation_guard(self, root_name: str):
        async with self._local_mutation_lock:
            state = _LeaseState()
            if self._mutation_lease is None:
                yield state
                return

            lease_key = (
                "secrets-manager-principal:" + hashlib.sha256(root_name.encode("utf-8")).hexdigest()
            )
            holder = uuid.uuid4().hex
            loop = asyncio.get_running_loop()
            deadline = loop.time() + _LEASE_WAIT_SECONDS
            delay = 0.05
            acquired = False
            while loop.time() < deadline:
                try:
                    acquired = await self._mutation_lease.acquire_lease(
                        lease_key, holder, _LEASE_TTL_SECONDS
                    )
                except Exception as exc:
                    raise SecretStoreError(
                        f"Secrets Manager mutation lease acquisition failed: {exc}"
                    ) from exc
                if acquired:
                    break
                await asyncio.sleep(delay)
                delay = min(delay * 2, 0.5)
            if not acquired:
                raise SecretStoreError("Timed out acquiring Secrets Manager mutation lease")

            async def _renew() -> None:
                try:
                    while True:
                        await asyncio.sleep(_LEASE_TTL_SECONDS / 3)
                        renewed = await self._mutation_lease.renew_lease(
                            lease_key, holder, _LEASE_TTL_SECONDS
                        )
                        if not renewed:
                            state.lost = True
                            return
                except asyncio.CancelledError:
                    raise
                except Exception:
                    state.lost = True
                    logger.exception("Secrets Manager mutation lease renewal failed")

            renewal = asyncio.create_task(_renew())
            try:
                yield state
                state.ensure_held()
            finally:
                renewal.cancel()
                try:
                    await renewal
                except asyncio.CancelledError:
                    pass
                try:
                    await self._mutation_lease.release_lease(lease_key, holder)
                except Exception:
                    logger.exception("Secrets Manager mutation lease release failed")

    # -- layout selection and atomic manifest commit ------------------------ #

    def _partition(self, entries: dict[str, dict], generation: str) -> list[dict]:
        bucket_count = 1
        while bucket_count <= self._max_shards:
            buckets: list[dict[str, dict]] = [{} for _ in range(bucket_count)]
            for key, value in entries.items():
                buckets[self._bucket(key, bucket_count)][key] = value
            documents = [
                self._shard_document(generation, index, bucket_count, bucket)
                for index, bucket in enumerate(buckets)
            ]
            if all(self._payload_size(document) <= self._target_bytes for document in documents):
                return documents
            bucket_count *= 2
        raise SecretStoreError(
            "Secrets Manager egress entries exceed the configured overflow shard capacity"
        )

    async def _cleanup_generation(self, root_name: str, meta: dict | None) -> None:
        if not meta:
            return
        results = await asyncio.gather(
            *(
                self._call(
                    self._delete_secret,
                    self._shard_name(root_name, meta["generation"], bucket),
                )
                for bucket in range(meta["bucket_count"])
            ),
            return_exceptions=True,
        )
        for result in results:
            if isinstance(result, Exception):
                logger.warning("Could not remove an obsolete Secrets Manager shard: %s", result)

    async def _commit_entries(
        self,
        root_name: str,
        entries: dict[str, dict],
        old_root: dict | None,
        lease_state: _LeaseState,
    ) -> None:
        old_meta = self._manifest(old_root) if old_root is not None else None
        lease_state.ensure_held()

        if not entries:
            await self._call(self._delete_secret, root_name)
            await self._cleanup_generation(root_name, old_meta)
            return

        # Preserve the existing bare-map format for ordinary principals. This
        # keeps rolling upgrades and all current one-secret values compatible.
        if self._payload_size(entries) <= self._target_bytes:
            await self._call(self._put_document, root_name, entries)
            written = await self._call(self._get_document, root_name, True)
            if written != entries:
                raise SecretStoreError("Secrets Manager inline write verification failed")
            await self._cleanup_generation(root_name, old_meta)
            return

        generation = uuid.uuid4().hex
        shard_documents = self._partition(entries, generation)
        bucket_count = len(shard_documents)
        shard_names = [
            self._shard_name(root_name, generation, bucket) for bucket in range(bucket_count)
        ]

        try:
            await asyncio.gather(
                *(
                    self._call(self._put_document, name, document)
                    for name, document in zip(shard_names, shard_documents, strict=True)
                )
            )
            checks = await asyncio.gather(
                *(self._call(self._get_document, name, True) for name in shard_names)
            )
            if checks != shard_documents:
                raise SecretStoreError("Secrets Manager overflow shard verification failed")
        except Exception:
            # The old root is still authoritative because the manifest has not
            # changed. Best-effort cleanup prevents failed staging from adding cost.
            await asyncio.gather(
                *(self._call(self._delete_secret, name) for name in shard_names),
                return_exceptions=True,
            )
            raise

        lease_state.ensure_held()
        manifest = {
            _META_KEY: {
                "schema": _SCHEMA_VERSION,
                "layout": "sharded",
                "hash": _HASH_ALGORITHM,
                "generation": generation,
                "bucket_count": bucket_count,
            }
        }
        # Manifest-last is the commit point: readers see either the complete old
        # layout or a fully written and verified new generation.
        await self._call(self._put_document, root_name, manifest)
        written_manifest = await self._call(self._get_document, root_name, True)
        if written_manifest != manifest:
            raise SecretStoreError("Secrets Manager overflow manifest verification failed")
        await self._cleanup_generation(root_name, old_meta)

    async def _mutate(self, root_name: str, key: str, value: dict | None) -> None:
        async with self._mutation_guard(root_name) as lease_state:
            root = await self._call(self._get_document, root_name)
            entries = await self._load_entries(root_name, root)
            if value is None:
                entries.pop(key, None)
            else:
                entries[key] = value
            await self._commit_entries(root_name, entries, root, lease_state)

    # -- SecretStoreBase ----------------------------------------------------- #

    async def put_token(
        self,
        auth_method: str,
        user_id: str,
        provider: str,
        server_path: str,
        token: StoredToken,
    ) -> None:
        root_name = self._secret_name(auth_method, user_id)
        await self._mutate(root_name, keys.map_key(provider, server_path), token.model_dump())

    async def _read_with_retry(self, operation: Callable):
        """Run a lock-free read, retrying if it races an overflow generation cleanup.

        A same-principal commit can delete the previous generation's shards
        between a reader observing the old manifest and fetching a shard. That is
        a transient stale read, not corruption, so re-read the whole root (which
        now points at the new generation) and try again a bounded number of times.
        """
        for attempt in range(_STALE_READ_RETRIES):
            try:
                return await operation()
            except _StaleShardError as exc:
                if attempt + 1 >= _STALE_READ_RETRIES:
                    raise SecretStoreError(
                        "Secrets Manager overflow read kept racing generation cleanup"
                    ) from exc
                logger.info(
                    "Secrets Manager overflow read raced a generation cleanup; "
                    "re-reading root (attempt %d/%d)",
                    attempt + 1,
                    _STALE_READ_RETRIES,
                )

    async def _get_token_once(self, root_name: str, key: str) -> StoredToken | None:
        root = await self._call(self._get_document, root_name)
        if root is None:
            return None
        meta = self._manifest(root)
        if meta is None:
            raw = root.get(key)
        else:
            bucket = self._bucket(key, meta["bucket_count"])
            raw = (await self._read_shard(root_name, meta, bucket)).get(key)
        return StoredToken(**raw) if raw is not None else None

    async def get_token(
        self,
        auth_method: str,
        user_id: str,
        provider: str,
        server_path: str,
    ) -> StoredToken | None:
        root_name = self._secret_name(auth_method, user_id)
        key = keys.map_key(provider, server_path)
        return await self._read_with_retry(lambda: self._get_token_once(root_name, key))

    async def delete_token(
        self,
        auth_method: str,
        user_id: str,
        provider: str,
        server_path: str,
    ) -> None:
        root_name = self._secret_name(auth_method, user_id)
        await self._mutate(root_name, keys.map_key(provider, server_path), None)

    async def _list_for_user_once(self, root_name: str) -> list[tuple[str, str, StoredToken]]:
        root = await self._call(self._get_document, root_name)
        entries = await self._load_entries(root_name, root)
        out: list[tuple[str, str, StoredToken]] = []
        for key, raw in entries.items():
            provider_enc, delimiter, server_enc = key.partition(keys.MAP_KEY_DELIMITER)
            if not delimiter or not provider_enc or not server_enc:
                raise SecretStoreError("Secrets Manager egress map key is malformed")
            out.append(
                (
                    keys.decode_segment(provider_enc),
                    keys.decode_segment(server_enc),
                    StoredToken(**raw),
                )
            )
        return out

    async def list_for_user(
        self,
        auth_method: str,
        user_id: str,
    ) -> list[tuple[str, str, StoredToken]]:
        root_name = self._secret_name(auth_method, user_id)
        return await self._read_with_retry(lambda: self._list_for_user_once(root_name))
