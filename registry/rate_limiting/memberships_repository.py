"""CRUD + cached reads over the ``rate_limit_memberships`` collection.

Maps a caller (a user by username, or an agent by client_id) to the rate-limit
group name(s) it belongs to. This is **the only** source of a caller's rate-limit
groups: no IdP emits them, and they are deliberately kept out of the token's authz
groups (mixing them in could change a caller's scopes). The limiter resolves a
caller's rate-limit groups purely from this collection, keyed by the username /
client_id / sub taken from the validated token.

Read on the hot ``/validate`` path, so lookups are served from a small in-process
time-based cache (default ~30s TTL).
"""

import logging
import time

from motor.motor_asyncio import AsyncIOMotorCollection

from ..repositories.documentdb.client import (
    get_collection_name,
    get_documentdb_client,
)
from .models import RateLimitMembership

# Configure logging with basicConfig
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s,p%(process)s,{%(filename)s:%(lineno)d},%(levelname)s,%(message)s",
)
logger = logging.getLogger(__name__)

# Base name of the memberships collection (namespaced at runtime).
MEMBERSHIPS_COLLECTION_BASE: str = "rate_limit_memberships"

# Default in-process cache TTL for membership reads.
DEFAULT_CACHE_TTL_SECONDS: float = 30.0

# Last-observed member count per group, refreshed by the async count on every
# admin quarantine mutation / list call and by seeding. Read SYNCHRONOUSLY by the
# metrics ObservableGauge callback (OTel callbacks cannot await). Reflects the
# most recent DB-backed count this process observed; bounded to the reserved
# group names in practice.
_GROUP_MEMBER_COUNTS: dict[str, int] = {}

# Process-wide shared repository (used by the metrics gauge in the registry
# process, which is separate from the auth-server's rate limiter instance).
_shared_repo: "MembershipsRepository | None" = None


def get_shared_memberships_repo() -> "MembershipsRepository":
    """Return a lazily-created process-wide MembershipsRepository (metrics + seeding)."""
    global _shared_repo
    if _shared_repo is None:
        _shared_repo = MembershipsRepository()
    return _shared_repo


class MembershipsRepository:
    """Repository for rate-limit memberships with a small time-based read cache."""

    def __init__(
        self,
        cache_ttl_seconds: float = DEFAULT_CACHE_TTL_SECONDS,
    ) -> None:
        self._collection: AsyncIOMotorCollection | None = None
        self._collection_name = get_collection_name(MEMBERSHIPS_COLLECTION_BASE)
        self._cache_ttl_seconds = cache_ttl_seconds
        # cache: doc-id -> (expires_at_monotonic, groups)
        self._cache: dict[str, tuple[float, list[str]]] = {}

    async def _get_collection(self) -> AsyncIOMotorCollection:
        """Get the memberships collection singleton."""
        if self._collection is None:
            db = await get_documentdb_client()
            self._collection = db[self._collection_name]
        return self._collection

    def _cache_get(
        self,
        doc_id: str,
    ) -> list[str] | None:
        """Return cached groups for ``doc_id`` if still fresh, else None."""
        entry = self._cache.get(doc_id)
        if entry is None:
            return None
        expires_at, groups = entry
        if time.monotonic() >= expires_at:
            del self._cache[doc_id]
            return None
        return groups

    def _cache_put(
        self,
        doc_id: str,
        groups: list[str],
    ) -> None:
        """Store ``groups`` for ``doc_id`` with the configured TTL."""
        self._cache[doc_id] = (time.monotonic() + self._cache_ttl_seconds, groups)

    def invalidate_cache(self) -> None:
        """Drop all cached reads (called after a mutating admin operation)."""
        self._cache.clear()

    async def _groups_for_id(
        self,
        doc_id: str,
    ) -> list[str]:
        """Return the rate-limit groups for a single membership ``_id`` (cached)."""
        cached = self._cache_get(doc_id)
        if cached is not None:
            return cached
        collection = await self._get_collection()
        doc = await collection.find_one({"_id": doc_id, "enabled": True})
        groups = list(doc.get("groups", [])) if doc else []
        self._cache_put(doc_id, groups)
        return groups

    async def get_groups_for(
        self,
        username: str | None,
        client_id: str | None,
    ) -> list[str]:
        """Resolve the caller's rate-limit groups from the user and client memberships.

        Looks up ``user:<username>`` and ``client:<client_id>`` and unions their
        groups. This is the ONLY source of rate-limit groups; the token's authz
        groups claim is never consulted.
        """
        groups: list[str] = []
        if username:
            groups.extend(await self._groups_for_id(f"user:{username}"))
        if client_id:
            groups.extend(await self._groups_for_id(f"client:{client_id}"))
        # De-duplicate while preserving order.
        seen: set[str] = set()
        unique: list[str] = []
        for group in groups:
            if group not in seen:
                seen.add(group)
                unique.append(group)
        return unique

    async def is_target_quarantined(
        self,
        target_entity_type: str | None,
        target_name: str | None,
    ) -> bool:
        """True if the classified target is in the quarantine-targets group.

        One cached point-read on the target's membership (``server:<name>`` or
        ``agent:<path>``), reusing the same 30s cache as caller lookups -- so the
        steady-state cost is zero DB reads. Returns False when no target is
        classified (control-plane requests never reach here anyway).
        """
        from .models import QUARANTINE_TARGET_GROUP

        if not target_entity_type or not target_name:
            return False
        subject_type = "server" if target_entity_type == "mcp_server" else "agent"
        groups = await self._groups_for_id(f"{subject_type}:{target_name}")
        return QUARANTINE_TARGET_GROUP in groups

    async def count_group_members(
        self,
        group: str,
    ) -> int:
        """Count enabled memberships whose ``groups`` contain ``group`` (single query).

        Also refreshes the process-wide sync count cache the metrics ObservableGauge
        reads (OTel callbacks cannot await), so the gauge tracks the latest observed
        count. No per-item loop; one ``count_documents``.
        """
        collection = await self._get_collection()
        count = await collection.count_documents({"groups": group, "enabled": True})
        _GROUP_MEMBER_COUNTS[group] = count
        return count

    def count_group_members_cached(
        self,
        group: str,
    ) -> int:
        """Synchronously return the last observed member count for the metrics gauge."""
        return _GROUP_MEMBER_COUNTS.get(group, 0)

    async def list_group_members(
        self,
        group: str,
    ) -> list[RateLimitMembership]:
        """Return every enabled membership belonging to ``group`` (admin quarantine list)."""
        collection = await self._get_collection()
        memberships: list[RateLimitMembership] = []
        async for doc in collection.find({"groups": group, "enabled": True}):
            doc.pop("_id", None)
            doc.pop("enabled", None)
            try:
                memberships.append(RateLimitMembership(**doc))
            except Exception as exc:
                logger.warning(f"skipping malformed rate-limit membership: {exc}")
        return memberships

    async def upsert(
        self,
        membership: RateLimitMembership,
    ) -> RateLimitMembership:
        """Create or replace a membership; invalidate the read cache."""
        collection = await self._get_collection()
        doc = membership.model_dump()
        doc["_id"] = membership.build_id()
        doc["enabled"] = True
        await collection.replace_one({"_id": doc["_id"]}, doc, upsert=True)
        self.invalidate_cache()
        return membership

    async def delete(
        self,
        membership_id: str,
    ) -> bool:
        """Delete a membership by ``_id``; return True if a doc was removed."""
        collection = await self._get_collection()
        result = await collection.delete_one({"_id": membership_id})
        self.invalidate_cache()
        return result.deleted_count > 0

    async def get_by_id(
        self,
        membership_id: str,
    ) -> RateLimitMembership | None:
        """Return a single membership by ``_id``, or None if absent."""
        collection = await self._get_collection()
        doc = await collection.find_one({"_id": membership_id})
        if not doc:
            return None
        doc.pop("_id", None)
        doc.pop("enabled", None)
        try:
            return RateLimitMembership(**doc)
        except Exception as exc:
            logger.warning(f"skipping malformed rate-limit membership {membership_id}: {exc}")
            return None

    async def list_all(self) -> list[RateLimitMembership]:
        """Return every membership (admin listing; no cache)."""
        collection = await self._get_collection()
        memberships: list[RateLimitMembership] = []
        async for doc in collection.find({}):
            doc.pop("_id", None)
            doc.pop("enabled", None)
            try:
                memberships.append(RateLimitMembership(**doc))
            except Exception as exc:
                logger.warning(f"skipping malformed rate-limit membership: {exc}")
        return memberships
