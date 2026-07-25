"""OpenBaoStore -- per-entry KV v2 SecretStore for per-user egress tokens.

Each connection is its own KV v2 entry at::

    {mount}/data/{prefix}/{enc(auth_method)}/{enc(user_id)}/{enc(provider)}/{enc(server_path)}

so there is no shared blob and no read-modify-write race between two providers
of the same principal. ``list_for_user`` walks the KV LIST under the principal
prefix. The same-key refresh race is handled with KV v2 CAS (``cas`` param).

Uses ``hvac`` (Vault/OpenBao API-compatible). Auth is configured by the
factory (token / kubernetes / approle) before the client reaches this class.
"""

import asyncio
import logging
from collections.abc import Callable
from typing import TypeVar

from registry.egress_auth.schemas import StoredToken
from registry.secrets import keys
from registry.secrets.interfaces import SecretStoreBase, SecretStoreError

logger = logging.getLogger(__name__)

_T = TypeVar("_T")

# Bounded backoff for transient Vault/OpenBao unavailability (see
# ``_is_transient_error``). A HA leader election or a pod restart clears within a
# few seconds, so a short exponential backoff (0.5, 1, 2, 4 -> ~7.5s total)
# rides through it without stranding the caller. Kept as module constants so
# tests can shrink the backoff.
_TRANSIENT_RETRIES = 4
_TRANSIENT_BACKOFF_BASE = 0.5


def _is_auth_expiry_error(exc: Exception) -> bool:
    """True if an hvac error looks like an expired/invalid Vault token.

    OpenBao role tokens are short-lived and hvac does not auto-renew, so a
    long-lived client eventually sees its token lapse. Vault then answers reads
    with 403 ``permission denied`` (hvac ``Forbidden``). We treat Forbidden /
    Unauthorized / 403 / "permission denied" as re-authenticatable. A genuine
    policy gap also surfaces as Forbidden, but re-login is cheap and a real
    policy error simply fails again on the retry, so the classification is safe.
    """
    name = type(exc).__name__
    if name in ("Forbidden", "Unauthorized", "InvalidRequest"):
        return True
    text = str(exc).lower()
    return "permission denied" in text or "403" in text or "invalid token" in text


def _is_transient_error(exc: Exception) -> bool:
    """True if an hvac/transport error looks like a transient Vault availability
    blip rather than a definitive failure.

    A Vault/OpenBao HA cluster momentarily rejects requests while it (re-)elects a
    leader -- e.g. after a pod restart (eviction / rollout / spot reclaim). During
    that window in-flight requests see one of:

      * ``connection refused`` / ``failed to establish a new connection`` -- the
        Service still routes to a pod that is terminating, or a standby redirects
        (307) to a now-dead leader ``api_addr``,
      * HTTP 5xx with ``local node not active but active cluster node not found``
        -- a standby that does not yet know who the active node is,
      * read timeouts.

    These clear within seconds once a leader is (re-)elected, so a short bounded
    retry (see ``_run``) turns a user-visible failure -- a consent whose token
    never persists, i.e. a silent "0 tools" -- into a transparent success. NOTE:
    this is deliberately disjoint from ``_is_auth_expiry_error`` (403/permission
    denied is NOT transient) so the two retry paths never overlap.
    """
    name = type(exc).__name__
    if name in (
        "InternalServerError",  # hvac 500
        "VaultDown",  # hvac 503
        "BadGateway",  # hvac 502
        "GatewayTimeout",  # hvac 504
        "ConnectionError",  # requests / urllib3
        "ConnectTimeout",
        "ReadTimeout",
        "Timeout",
        "MaxRetryError",  # urllib3
        "NewConnectionError",  # urllib3
    ):
        return True
    text = str(exc).lower()
    return (
        "local node not active" in text
        or "connection refused" in text
        or "max retries exceeded" in text
        or "failed to establish a new connection" in text
        or "read timed out" in text
        or "temporarily unavailable" in text
        or " 500" in text
        or " 502" in text
        or " 503" in text
        or " 504" in text
    )


class OpenBaoStore(SecretStoreBase):
    """Per-entry KV v2 store. ``client`` is a connected, authenticated hvac client.

    ``reauthenticate`` re-runs the configured login on ``client`` in place; the
    store calls it and retries once when an operation fails with an expired-token
    error (see ``_is_auth_expiry_error``), making a long-lived store self-healing
    against short Vault token TTLs. It also rides out transient Vault
    unavailability (HA leader election / pod restart -- see
    ``_is_transient_error``) with a short bounded backoff, so a consent's token
    write is not lost to a few-second cluster blip.
    """

    def __init__(
        self,
        client,
        mount_point: str,
        prefix: str,
        reauthenticate: Callable[[], None] | None = None,
    ) -> None:
        self._client = client
        self._mount = mount_point
        self._prefix = prefix.strip("/")
        self._reauthenticate = reauthenticate

    async def _run(self, fn: Callable[[], _T]) -> _T:
        """Run a blocking hvac call in a thread, with two independent recoveries:

        1. Expired Vault token (``_is_auth_expiry_error``) -> re-authenticate
           once and retry (short-lived role tokens; hvac does not auto-renew).
        2. Transient cluster unavailability (``_is_transient_error``, e.g. a HA
           leader election after a pod restart) -> bounded exponential backoff
           retry so a few-second blip does not surface as a hard failure (a
           consent whose token never persists == a silent "0 tools").

        A genuine error (bad path, real policy gap, malformed data) matches
        neither and is raised immediately by the callers as a SecretStoreError.
        """
        attempt = 0
        reauthed = False
        while True:
            try:
                return await asyncio.to_thread(fn)
            except Exception as exc:
                # 1) expired token: re-login once, then keep going.
                if not reauthed and self._reauthenticate is not None and _is_auth_expiry_error(exc):
                    logger.warning(
                        "OpenBao token rejected (%s); re-authenticating and retrying", exc
                    )
                    await asyncio.to_thread(self._reauthenticate)
                    reauthed = True
                    continue
                # 2) transient unavailability: bounded backoff retry.
                if attempt < _TRANSIENT_RETRIES and _is_transient_error(exc):
                    delay = _TRANSIENT_BACKOFF_BASE * (2**attempt)
                    attempt += 1
                    logger.warning(
                        "OpenBao request failed transiently (%s); retry %d/%d in %.1fs",
                        exc,
                        attempt,
                        _TRANSIENT_RETRIES,
                        delay,
                    )
                    await asyncio.sleep(delay)
                    continue
                raise

    def _rel_path(
        self,
        auth_method: str,
        user_id: str,
        provider: str,
        server_path: str,
    ) -> str:
        # hvac takes a path relative to the mount, so strip our prefix logic into
        # the KV "path" (it does NOT include the mount or the "/data/" infix).
        return (
            f"{self._prefix}/"
            f"{keys.encode_segment(auth_method)}/{keys.encode_segment(user_id)}/"
            f"{keys.encode_segment(provider)}/{keys.encode_segment(server_path)}"
        )

    def _principal_rel_prefix(self, auth_method: str, user_id: str) -> str:
        return f"{self._prefix}/{keys.encode_segment(auth_method)}/{keys.encode_segment(user_id)}"

    async def put_token(
        self,
        auth_method: str,
        user_id: str,
        provider: str,
        server_path: str,
        token: StoredToken,
    ) -> None:
        path = self._rel_path(auth_method, user_id, provider, server_path)

        def _write() -> None:
            self._client.secrets.kv.v2.create_or_update_secret(
                path=path,
                secret=token.model_dump(),
                mount_point=self._mount,
            )

        try:
            await self._run(_write)
        except Exception as exc:  # hvac raises various subclasses; fail closed
            raise SecretStoreError(f"OpenBao put failed: {exc}") from exc

    async def get_token(
        self,
        auth_method: str,
        user_id: str,
        provider: str,
        server_path: str,
    ) -> StoredToken | None:
        path = self._rel_path(auth_method, user_id, provider, server_path)

        def _read() -> dict | None:
            try:
                resp = self._client.secrets.kv.v2.read_secret_version(
                    path=path,
                    mount_point=self._mount,
                    raise_on_deleted_version=False,
                )
            except Exception as exc:
                # InvalidPath -> miss; everything else is a real error.
                if type(exc).__name__ == "InvalidPath":
                    return None
                raise
            data = (resp or {}).get("data", {}).get("data")
            return data

        try:
            raw = await self._run(_read)
        except Exception as exc:
            raise SecretStoreError(f"OpenBao get failed: {exc}") from exc
        return StoredToken(**raw) if raw else None

    async def delete_token(
        self,
        auth_method: str,
        user_id: str,
        provider: str,
        server_path: str,
    ) -> None:
        path = self._rel_path(auth_method, user_id, provider, server_path)

        def _delete() -> None:
            self._client.secrets.kv.v2.delete_metadata_and_all_versions(
                path=path,
                mount_point=self._mount,
            )

        try:
            await self._run(_delete)
        except Exception as exc:
            if type(exc).__name__ == "InvalidPath":
                return  # idempotent
            raise SecretStoreError(f"OpenBao delete failed: {exc}") from exc

    async def list_for_user(
        self,
        auth_method: str,
        user_id: str,
    ) -> list[tuple[str, str, StoredToken]]:
        principal = self._principal_rel_prefix(auth_method, user_id)

        def _walk() -> list[tuple[str, str, StoredToken]]:
            out: list[tuple[str, str, StoredToken]] = []
            try:
                providers = self._client.secrets.kv.v2.list_secrets(
                    path=principal, mount_point=self._mount
                )
            except Exception as exc:
                if type(exc).__name__ == "InvalidPath":
                    return out
                raise
            for provider_enc in (providers or {}).get("data", {}).get("keys", []):
                provider_enc = provider_enc.rstrip("/")
                try:
                    servers = self._client.secrets.kv.v2.list_secrets(
                        path=f"{principal}/{provider_enc}", mount_point=self._mount
                    )
                except Exception as exc:
                    if type(exc).__name__ == "InvalidPath":
                        continue
                    raise
                for server_enc in (servers or {}).get("data", {}).get("keys", []):
                    server_enc = server_enc.rstrip("/")
                    resp = self._client.secrets.kv.v2.read_secret_version(
                        path=f"{principal}/{provider_enc}/{server_enc}",
                        mount_point=self._mount,
                        raise_on_deleted_version=False,
                    )
                    raw = (resp or {}).get("data", {}).get("data")
                    if raw:
                        out.append(
                            (
                                keys.decode_segment(provider_enc),
                                keys.decode_segment(server_enc),
                                StoredToken(**raw),
                            )
                        )
            return out

        try:
            return await self._run(_walk)
        except Exception as exc:
            raise SecretStoreError(f"OpenBao list failed: {exc}") from exc
