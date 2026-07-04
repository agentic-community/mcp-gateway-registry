"""SecretStore abstract base class.

Pluggable secret store for per-user egress tokens, addressed by deterministic
namespacing on ``(auth_method, user_id, provider, server_path)``. This is the
single source of truth -- there is no companion app-DB table. Key granularity
(one-secret-per-principal map vs per-entry KV) is an implementation detail
hidden behind these methods.

Mirrors the ``registry.repositories.interfaces`` ABC style.
"""

from abc import ABC, abstractmethod

from registry.egress_auth.schemas import StoredToken


class SecretStoreError(Exception):
    """Base exception for SecretStore failures (store unreachable, write conflict)."""


class SecretStoreBase(ABC):
    """Abstract base class for per-user egress token storage."""

    @abstractmethod
    async def put_token(
        self,
        auth_method: str,
        user_id: str,
        provider: str,
        server_path: str,
        token: StoredToken,
    ) -> None:
        """Store/overwrite the token at the deterministic address.

        No ref is returned -- the address is recomputable from
        ``(auth_method, user_id, provider, server_path)``. Implementations that
        share a per-principal blob (Secrets Manager) MUST use optimistic
        concurrency so a concurrent ``put`` for a different
        ``(provider, server_path)`` of the same principal does not clobber.
        """

    @abstractmethod
    async def get_token(
        self,
        auth_method: str,
        user_id: str,
        provider: str,
        server_path: str,
    ) -> StoredToken | None:
        """Retrieve the token, or None on miss."""

    @abstractmethod
    async def delete_token(
        self,
        auth_method: str,
        user_id: str,
        provider: str,
        server_path: str,
    ) -> None:
        """Delete the token (idempotent -- deleting a missing entry is a no-op)."""

    @abstractmethod
    async def list_for_user(
        self,
        auth_method: str,
        user_id: str,
    ) -> list[tuple[str, str, StoredToken]]:
        """Enumerate a principal's connections as ``(provider, server_path, token)``.

        Backs ``GET /api/egress-auth/connections`` and the optional refresh
        sweep. OpenBao: KV LIST under the principal prefix. Secrets Manager:
        read the principal's one secret and iterate its JSON map.
        Returns an empty list on miss (e.g. a non-per-user caller with no
        addressable namespace).
        """
