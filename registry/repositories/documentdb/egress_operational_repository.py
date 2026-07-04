"""Cross-replica operational state for the egress credential vault.

Holds NO secret/token material -- only:
  - single-use OAuth ``state`` nonces (replay guard),
  - per-(auth_method,user,provider,server) refresh leases (single-flight), and
  - OAuth AS-facade pending-authorize + auth-code correlation state (the
    IDE-driven egress consent spans replicas; see the facade methods below).

Storing operational state (not credentials) in the app DB is the defensible
property: the vault remains the single source of truth for tokens; this
collection only coordinates replicas. Two logical record kinds share one
collection, discriminated by ``kind`` ('nonce' | 'lease'):

  nonce: {_id: "nonce:<nonce>", kind, expires_at}            -- replay guard
  lease: {_id: "lease:<key>",   kind, holder, expires_at}    -- refresh single-flight

A TTL index on ``expires_at`` reaps both kinds; correctness does NOT depend on
the ~60s TTL sweep (it is a crashed-holder safety net) -- the lease comparison
and the nonce unique-insert are the real mechanisms.
"""

import logging
from datetime import UTC, datetime, timedelta

from motor.motor_asyncio import AsyncIOMotorCollection
from pymongo import ASCENDING, ReturnDocument
from pymongo.errors import DuplicateKeyError

from .client import get_collection_name, get_documentdb_client

logger = logging.getLogger(__name__)


def _is_expired(expires_at_dt) -> bool:
    """True if a stored expiry is in the past.

    BSON datetimes round-trip from Mongo/DocumentDB as timezone-NAIVE (UTC) even
    though we store tz-aware values, so a direct ``< datetime.now(UTC)`` compare
    raises ``TypeError: can't compare offset-naive and offset-aware datetimes``.
    Normalize a naive value to UTC-aware before comparing. None -> not expired."""
    if expires_at_dt is None:
        return False
    if expires_at_dt.tzinfo is None:
        expires_at_dt = expires_at_dt.replace(tzinfo=UTC)
    return expires_at_dt < datetime.now(UTC)


class EgressOperationalRepository:
    """Mongo-backed replay guard + refresh lease for the egress vault."""

    def __init__(self) -> None:
        self._collection: AsyncIOMotorCollection | None = None
        self._collection_name = get_collection_name("mcp_egress_operational")
        self._indexes_created = False

    async def _get_collection(self) -> AsyncIOMotorCollection:
        if self._collection is None:
            db = await get_documentdb_client()
            self._collection = db[self._collection_name]
            await self.ensure_indexes()
        return self._collection

    async def ensure_indexes(self) -> None:
        """Create the TTL index (idempotent). expires_at is an ISO8601 string;
        DocumentDB/Mongo TTL requires a BSON date, so we store a native datetime
        in ``expires_at_dt`` for the TTL and keep the ISO string for lease logic."""
        if self._indexes_created:
            return
        col = await self._get_collection()
        try:
            await col.create_index(
                [("expires_at_dt", ASCENDING)],
                expireAfterSeconds=0,
                name="ttl_expires_at",
            )
            self._indexes_created = True
            logger.info("Created TTL index for %s", self._collection_name)
        except Exception as e:  # index creation is best-effort; not fatal
            logger.warning("Could not create index for %s: %s", self._collection_name, e)

    # -- replay guard --------------------------------------------------------- #

    async def consume_nonce(self, nonce: str, ttl_seconds: int) -> bool:
        """Record a state nonce as used. Returns True if unused (now consumed),
        False if it was already present (replay).

        Atomic via the unique ``_id``: a concurrent second insert raises
        DuplicateKeyError, so exactly one caller wins."""
        col = await self._get_collection()
        now = datetime.now(UTC)
        expires = now + timedelta(seconds=ttl_seconds)
        try:
            await col.insert_one(
                {
                    "_id": f"nonce:{nonce}",
                    "kind": "nonce",
                    "expires_at_dt": expires,
                }
            )
            return True
        except DuplicateKeyError:
            return False

    # -- refresh lease (single-flight) ---------------------------------------- #

    async def acquire_lease(self, key: str, holder: str, ttl_seconds: int) -> bool:
        """Acquire the refresh lease for ``key`` (the canonical vault tuple,
        stringified). Returns True if acquired.

        Lease pattern (mirrors agent_batch_job_repository.claim_next_queued): the
        upsert matches a free or expired lease and stamps holder+expiry atomically;
        we then confirm we are the holder. Correctness is the lease comparison, not
        the TTL sweep."""
        col = await self._get_collection()
        now = datetime.now(UTC)
        expires = now + timedelta(seconds=ttl_seconds)
        doc = await col.find_one_and_update(
            {
                "_id": f"lease:{key}",
                "$or": [
                    {"holder": {"$exists": False}},
                    {"expires_at_dt": {"$lt": now}},
                ],
            },
            {
                "$set": {
                    "kind": "lease",
                    "holder": holder,
                    "expires_at_dt": expires,
                }
            },
            upsert=True,
            return_document=ReturnDocument.AFTER,
        )
        return bool(doc) and doc.get("holder") == holder

    async def release_lease(self, key: str, holder: str) -> None:
        """Release the lease iff still held by ``holder`` (idempotent). The holder
        guard prevents deleting a lease another replica reclaimed after ours lapsed."""
        col = await self._get_collection()
        await col.delete_one({"_id": f"lease:{key}", "holder": holder})
