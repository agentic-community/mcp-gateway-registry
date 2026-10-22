"""Per-user long-lived API key ("patch key") service.

Stores user-minted, non-expiring Bearer credentials in the shared
``patch_keys`` MongoDB collection. Only a SHA-256 hash of each key is
persisted; the plaintext is returned by :meth:`PatchKeyService.mint_key`
exactly once and never logged.

Two consumers share this module:

- the registry's console API (``registry.api.patch_key_routes``) for
  mint/list/revoke, authenticated as the logged-in console user;
- the auth server's ``/validate`` for verification, which resolves an
  ``wgk-`` prefixed Bearer token by hash and continues the request as the
  owning user (see ``auth_server/server.py::_validate_patch_key_token``).

Verification reads the database on every request with ``status == "active"``
in the query itself, so revocation takes effect on the very next call and
there is no cache to invalidate. Revocation is one-way: no code path flips a
revoked key back to active.

Tracked by the wire-platform-v1 fork (task 2.1).
"""

import hashlib
import logging
import secrets
import uuid
from datetime import datetime

from motor.motor_asyncio import AsyncIOMotorDatabase
from pymongo.errors import DuplicateKeyError

from registry.core.config import settings
from registry.schemas.patch_key import (
    PATCH_KEY_STATUS_ACTIVE,
    PATCH_KEY_STATUS_REVOKED,
    PatchKeyCreated,
    PatchKeyInfo,
)

logger = logging.getLogger(__name__)


COLLECTION_NAME: str = "patch_keys"

# Bearer tokens with this prefix are patch keys. The prefix is what keeps the
# /validate insertion cheap and non-interfering: any other token (JWT,
# REGISTRY_API_KEYS static key) fails the prefix test and the new code path is
# never entered, so the JWT and static-key mechanisms are bit-for-bit
# unchanged.
PATCH_KEY_PREFIX: str = "wgk-"

# 32 bytes of entropy (>= the 32-char minimum required of REGISTRY_API_KEYS
# values), URL-safe base64 => 43 chars after the prefix.
PATCH_KEY_ENTROPY_BYTES: int = 32

# Display prefix length INCLUDING the "wgk-" marker (enough for a human to
# tell two keys apart, far too little to brute-force the remainder).
_KEY_PREFIX_DISPLAY_LEN: int = 12


def hash_patch_key(plaintext: str) -> str:
    """Return the hex SHA-256 digest of a patch key plaintext.

    SHA-256 is appropriate here (not a password KDF): the key is a high-entropy
    random token, so offline brute-force of the hash is infeasible and a fast
    digest keeps the per-request /validate lookup cheap.
    """
    return hashlib.sha256(plaintext.encode("utf-8")).hexdigest()


def generate_patch_key() -> str:
    """Generate a new plaintext patch key (``wgk-`` + 43 url-safe chars)."""
    return f"{PATCH_KEY_PREFIX}{secrets.token_urlsafe(PATCH_KEY_ENTROPY_BYTES)}"


class PatchKeyQuotaExceeded(Exception):
    """Raised when the user already holds the configured max active keys."""

    def __init__(self, username: str, max_active: int) -> None:
        super().__init__(
            f"User already holds the maximum of {max_active} active patch keys; "
            "revoke one before minting another"
        )
        self.username = username
        self.max_active = max_active


class PatchKeyNotFound(Exception):
    """Raised when the requested key_id does not exist for the user."""


class PatchKeyAlreadyRevoked(Exception):
    """Raised when revoking a key that is already revoked."""


def _doc_to_info(doc: dict) -> PatchKeyInfo:
    """Build the metadata view from a stored document (drops hash/_id)."""
    return PatchKeyInfo(
        key_id=doc["key_id"],
        name=doc.get("name") or "",
        key_prefix=doc.get("key_prefix") or "",
        username=doc["username"],
        email=doc.get("email"),
        provider=doc.get("provider"),
        groups=list(doc.get("groups") or []),
        status=doc.get("status") or PATCH_KEY_STATUS_REVOKED,
        created_at=doc.get("created_at") or datetime.utcnow(),
        last_used_at=doc.get("last_used_at"),
        revoked_at=doc.get("revoked_at"),
    )


class PatchKeyService:
    """CRUD + verification for per-user long-lived API keys."""

    def __init__(self, db: AsyncIOMotorDatabase) -> None:
        self._collection = db[COLLECTION_NAME]

    async def ensure_indexes(self) -> None:
        """Create required indexes (idempotent).

        ``key_hash`` unique: two mints can never collide (cosmetic today --
        256-bit randomness -- but the index also backs the /validate lookup).
        ``username``: the per-user list/quota queries.
        """
        await self._collection.create_index("key_hash", unique=True)
        await self._collection.create_index("username")
        logger.info("Ensured indexes on %s (key_hash unique, username)", COLLECTION_NAME)

    async def count_active_keys(self, username: str) -> int:
        """Number of the user's keys currently in the active state."""
        return await self._collection.count_documents(
            {"username": username, "status": PATCH_KEY_STATUS_ACTIVE}
        )

    async def mint_key(
        self,
        *,
        username: str,
        groups: list[str],
        name: str,
        email: str | None = None,
        provider: str | None = None,
    ) -> PatchKeyCreated:
        """Mint a new patch key for ``username``.

        The plaintext appears only in the returned :class:`PatchKeyCreated`;
        the persisted document carries its hash. SECURITY: never log the
        return value's ``key`` -- callers must treat it as display-once.

        Raises:
            PatchKeyQuotaExceeded: if the user already holds
                ``patch_key_max_active_per_user`` active keys.
        """
        max_active = getattr(settings, "patch_key_max_active_per_user", 20)
        active = await self.count_active_keys(username)
        if active >= max_active:
            raise PatchKeyQuotaExceeded(username, max_active)

        plaintext = generate_patch_key()
        now = datetime.utcnow()
        doc = {
            "key_id": uuid.uuid4().hex,
            "key_hash": hash_patch_key(plaintext),
            "key_prefix": plaintext[:_KEY_PREFIX_DISPLAY_LEN],
            "name": name,
            "username": username,
            "email": email,
            # Snapshot: authorization follows the owner's mint-time groups.
            # Scope NAMES are re-resolved per request, so scope-mapping edits
            # apply immediately; group-membership edits need a fresh key.
            "groups": list(groups),
            "provider": provider,
            "status": PATCH_KEY_STATUS_ACTIVE,
            "created_at": now,
            "last_used_at": None,
            "revoked_at": None,
        }
        try:
            await self._collection.insert_one(dict(doc))
        except DuplicateKeyError as exc:  # pragma: no cover - 256-bit collision
            raise RuntimeError("patch key hash collision; retry the mint") from exc

        # Log identifiers only: key_id + username. The plaintext and its hash
        # never appear in any log line.
        logger.info(
            "Minted patch key key_id=%s for username=%s (active=%d)",
            doc["key_id"],
            username,
            active + 1,
        )
        return PatchKeyCreated(key=plaintext, info=_doc_to_info(doc))

    async def verify_key(self, plaintext: str) -> PatchKeyInfo | None:
        """Resolve an active key by plaintext, else ``None``.

        This is the auth-server /validate read path. The ``status == active``
        filter is part of the query, so a revoke is visible to the very next
        request with no cache layer. Database errors are caught here and
        treated as a miss (fail closed -> the caller 401s) so a datastore
        outage can never degrade into accept.
        """
        try:
            doc = await self._collection.find_one(
                {"key_hash": hash_patch_key(plaintext), "status": PATCH_KEY_STATUS_ACTIVE}
            )
        except Exception as exc:  # noqa: BLE001 - fail closed on any datastore error
            logger.error(
                "Patch key verification failed (%s); failing closed",
                type(exc).__name__,
            )
            return None
        if doc is None:
            return None
        return _doc_to_info(doc)

    async def touch_last_used(self, key_id: str) -> None:
        """Best-effort update of ``last_used_at`` (never fails the request)."""
        try:
            await self._collection.update_one(
                {"key_id": key_id},
                {"$set": {"last_used_at": datetime.utcnow()}},
            )
        except Exception as exc:  # noqa: BLE001 - metadata update is best-effort
            logger.warning(
                "Failed to update last_used_at for patch key key_id=%s (%s)",
                key_id,
                type(exc).__name__,
            )

    async def list_keys_for_user(self, username: str) -> list[PatchKeyInfo]:
        """All of the user's keys (active and revoked), newest first."""
        cursor = self._collection.find({"username": username}).sort("created_at", -1)
        docs = await cursor.to_list(length=None)
        return [_doc_to_info(d) for d in docs]

    async def revoke_key(self, *, key_id: str, username: str) -> PatchKeyInfo:
        """Revoke one of the user's keys. One-way: no un-revoke exists.

        Raises:
            PatchKeyNotFound: no key with this key_id belongs to this user.
            PatchKeyAlreadyRevoked: the key was already revoked.
        """
        doc = await self._collection.find_one({"key_id": key_id, "username": username})
        if doc is None:
            raise PatchKeyNotFound(key_id)
        if doc.get("status") != PATCH_KEY_STATUS_ACTIVE:
            raise PatchKeyAlreadyRevoked(key_id)

        now = datetime.utcnow()
        await self._collection.update_one(
            {"key_id": key_id, "username": username, "status": PATCH_KEY_STATUS_ACTIVE},
            {"$set": {"status": PATCH_KEY_STATUS_REVOKED, "revoked_at": now}},
        )
        logger.info(
            "Revoked patch key key_id=%s for username=%s (irreversible)",
            key_id,
            username,
        )
        updated = await self._collection.find_one({"key_id": key_id})
        return _doc_to_info(updated if updated is not None else doc)


_singleton: PatchKeyService | None = None


async def get_patch_key_service() -> PatchKeyService:
    """Module-level singleton getter bound to the active DocumentDB client.

    Mirrors ``get_user_group_management_service`` so the registry routes and
    the auth server's /validate share one service/collection instance.
    """
    global _singleton
    if _singleton is None:
        # Imported lazily: the auth server also reaches this module and must
        # not pay the documentdb import at module import time.
        from registry.repositories.documentdb.client import get_documentdb_client

        db = await get_documentdb_client()
        _singleton = PatchKeyService(db)
    return _singleton


def reset_patch_key_service_singleton() -> None:
    """Test hook: drop the singleton so the next getter re-binds to a new db."""
    global _singleton
    _singleton = None
