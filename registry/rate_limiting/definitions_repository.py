"""CRUD + cached reads over the ``mcp_rate_limits`` definitions collection.

Definitions are read on the hot ``/validate`` path, so both list queries are
served from a tiny in-process time-based cache (default ~30s TTL). Steady-state
per-call cost for definitions is therefore zero DB reads; only the counter
upserts touch the DB.

Each definition is one ``RateLimitDefinition`` document keyed by
``<axis>:<entity_type>:<name>:<window_seconds>``. A subject may hold several
definitions at different windows (e.g. a burst cap and a daily volume cap).
"""

import logging
import time

from motor.motor_asyncio import AsyncIOMotorCollection
from pymongo import ReturnDocument

from ..repositories.documentdb.client import (
    get_collection_name,
    get_documentdb_client,
)
from .models import (
    SERVER_GROUP_ENTITY_TYPE,
    RateLimitDefinition,
)

# Configure logging with basicConfig
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s,p%(process)s,{%(filename)s:%(lineno)d},%(levelname)s,%(message)s",
)
logger = logging.getLogger(__name__)

# Base name of the definitions collection (namespaced at runtime).
DEFINITIONS_COLLECTION_BASE: str = "mcp_rate_limits"

# Default in-process cache TTL for definition reads.
DEFAULT_CACHE_TTL_SECONDS: float = 30.0


class DefinitionsRepository:
    """Repository for rate-limit definitions with a small time-based read cache."""

    def __init__(
        self,
        cache_ttl_seconds: float = DEFAULT_CACHE_TTL_SECONDS,
    ) -> None:
        self._collection: AsyncIOMotorCollection | None = None
        self._collection_name = get_collection_name(DEFINITIONS_COLLECTION_BASE)
        self._cache_ttl_seconds = cache_ttl_seconds
        # cache: query-key -> (expires_at_monotonic, list[RateLimitDefinition])
        self._cache: dict[str, tuple[float, list[RateLimitDefinition]]] = {}

    async def _get_collection(self) -> AsyncIOMotorCollection:
        """Get the definitions collection singleton."""
        if self._collection is None:
            db = await get_documentdb_client()
            self._collection = db[self._collection_name]
        return self._collection

    def _cache_get(
        self,
        cache_key: str,
    ) -> list[RateLimitDefinition] | None:
        """Return cached definitions for ``cache_key`` if still fresh, else None."""
        entry = self._cache.get(cache_key)
        if entry is None:
            return None
        expires_at, definitions = entry
        if time.monotonic() >= expires_at:
            del self._cache[cache_key]
            return None
        return definitions

    def _cache_put(
        self,
        cache_key: str,
        definitions: list[RateLimitDefinition],
    ) -> None:
        """Store ``definitions`` for ``cache_key`` with the configured TTL."""
        self._cache[cache_key] = (time.monotonic() + self._cache_ttl_seconds, definitions)

    def invalidate_cache(self) -> None:
        """Drop all cached reads (called after a mutating admin operation)."""
        self._cache.clear()

    async def _find_definitions(
        self,
        query: dict,
    ) -> list[RateLimitDefinition]:
        """Run a definitions query and parse each doc into a model (skipping bad docs)."""
        collection = await self._get_collection()
        definitions: list[RateLimitDefinition] = []
        async for doc in collection.find(query):
            doc.pop("_id", None)
            try:
                definitions.append(RateLimitDefinition(**doc))
            except Exception as exc:
                # A malformed definition must never break the auth path; skip and log.
                logger.warning(f"skipping malformed rate-limit definition: {exc}")
        return definitions

    async def list_caller_limits(
        self,
        entity_type: str,
        names: list[str],
    ) -> list[RateLimitDefinition]:
        """Return all enabled caller-axis definitions (all windows) for the given names.

        One bulk ``$in`` query, cached. Empty ``names`` short-circuits to no limits.
        """
        if not names:
            return []
        cache_key = f"caller:{entity_type}:{','.join(sorted(names))}"
        cached = self._cache_get(cache_key)
        if cached is not None:
            return cached
        query = {
            "axis": "caller",
            "entity_type": entity_type,
            "name": {"$in": names},
            "enabled": True,
        }
        definitions = await self._find_definitions(query)
        self._cache_put(cache_key, definitions)
        return definitions

    async def list_caller_target_limits(
        self,
        entity_type: str,
        names: list[str],
    ) -> list[RateLimitDefinition]:
        """Return all enabled caller_target-axis definitions (all windows) for the given names.

        Mirrors ``list_caller_limits`` with ``axis="caller_target"``. One bulk ``$in``
        query, cached. A member of such a group gets an INDEPENDENT quota per target
        (the limiter builds a composite counter subject).
        """
        if not names:
            return []
        cache_key = f"caller_target:{entity_type}:{','.join(sorted(names))}"
        cached = self._cache_get(cache_key)
        if cached is not None:
            return cached
        query = {
            "axis": "caller_target",
            "entity_type": entity_type,
            "name": {"$in": names},
            "enabled": True,
        }
        definitions = await self._find_definitions(query)
        self._cache_put(cache_key, definitions)
        return definitions

    async def list_target_limits(
        self,
        entity_type: str,
        name: str,
    ) -> list[RateLimitDefinition]:
        """Return all enabled target-axis definitions (all windows) for one target entity.

        For an ``mcp_server`` target this ALSO returns enabled ``server_group``
        definitions whose ``members`` array contains this server, so a server picks
        up any group limit it belongs to. Each such definition is keyed per this
        individual server by the limiter (``tgt:server_group:<server>:<window>``),
        giving per-member uniform buckets rather than a shared pool. The exact-name
        arm and the group arm are unioned via ``$or``; both a per-server and a
        per-group definition can apply at once (independent buckets, all enforced).
        """
        cache_key = f"target:{entity_type}:{name}"
        cached = self._cache_get(cache_key)
        if cached is not None:
            return cached
        # Exact single-target match always applies. Only server targets can belong
        # to a server_group, so the group arm is added only for mcp_server (agent /
        # tool / skill lookups keep their single-arm behavior, unchanged).
        or_clauses: list[dict] = [{"entity_type": entity_type, "name": name}]
        if entity_type == "mcp_server":
            or_clauses.append({"entity_type": SERVER_GROUP_ENTITY_TYPE, "members": name})
        query = {
            "axis": "target",
            "enabled": True,
            "$or": or_clauses,
        }
        definitions = await self._find_definitions(query)
        self._cache_put(cache_key, definitions)
        return definitions

    async def is_quarantine_group_enabled(
        self,
        group: str,
    ) -> bool:
        """Return whether a reserved quarantine group's sentinel definition is enabled.

        A disabled sentinel is the operator's GLOBAL kill-switch off-toggle: the
        limiter skips the quarantine short-circuit for that scope even if members
        exist. Served from the definitions read cache. An unseeded group (no doc at
        all) is treated as enabled so a fresh deploy still honors a hand-added
        membership; seeding normally creates the sentinel enabled.
        """
        cache_key = f"quarantine_sentinel:{group}"
        cached = self._cache_get(cache_key)
        if cached is None:
            # Query the sentinel WITHOUT the enabled filter so a disabled doc is
            # still returned (we need its enabled flag). _find_definitions parses
            # the model, which carries `enabled`.
            cached = await self._find_definitions({"axis": "quarantine", "name": group})
            self._cache_put(cache_key, cached)
        if not cached:
            return True  # unseeded -> honor a hand-added membership
        return cached[0].enabled

    async def upsert(
        self,
        definition: RateLimitDefinition,
    ) -> RateLimitDefinition:
        """Create or replace a definition; invalidate the read cache."""
        collection = await self._get_collection()
        doc = definition.model_dump()
        doc["_id"] = definition.build_id()
        await collection.replace_one({"_id": doc["_id"]}, doc, upsert=True)
        self.invalidate_cache()
        return definition

    async def delete(
        self,
        definition_id: str,
    ) -> bool:
        """Delete a definition by ``_id``; return True if a doc was removed."""
        collection = await self._get_collection()
        result = await collection.delete_one({"_id": definition_id})
        self.invalidate_cache()
        return result.deleted_count > 0

    async def list_all(self) -> list[RateLimitDefinition]:
        """Return every definition (admin listing; bypasses the enabled filter and cache)."""
        return await self._find_definitions({})

    async def get_by_id(
        self,
        definition_id: str,
    ) -> RateLimitDefinition | None:
        """Return a single definition by ``_id``, or None if absent (admin read; no cache)."""
        results = await self._find_definitions({"_id": definition_id})
        return results[0] if results else None

    async def set_enabled(
        self,
        definition_id: str,
        enabled: bool,
    ) -> RateLimitDefinition | None:
        """Toggle a definition's ``enabled`` flag in place; return the updated def or None."""
        collection = await self._get_collection()
        doc = await collection.find_one_and_update(
            {"_id": definition_id},
            {"$set": {"enabled": enabled}},
            return_document=ReturnDocument.AFTER,
        )
        self.invalidate_cache()
        if not doc:
            return None
        doc.pop("_id", None)
        return RateLimitDefinition(**doc)

    async def seed_reserved_groups(self) -> None:
        """Seed the two reserved quarantine group sentinels if absent (idempotent).

        Uses ``$setOnInsert`` upserts so concurrent registry replicas racing at
        startup converge to one doc each, and a restart NEVER clobbers an operator's
        enable/disable state. Runs regardless of ``RATE_LIMITING_ENABLED`` so the
        groups are always visible in the API/UI; enforcement is still feature-gated.
        """
        from .models import (
            MIN_WINDOW_SECONDS,
            QUARANTINE_CALLER_GROUP,
            QUARANTINE_TARGET_GROUP,
        )

        collection = await self._get_collection()
        sentinels = [
            RateLimitDefinition(
                axis="quarantine",
                entity_type="group",
                name=QUARANTINE_CALLER_GROUP,
                scope="caller",
                window_seconds=MIN_WINDOW_SECONDS,
            ),
            RateLimitDefinition(
                axis="quarantine",
                entity_type="group",
                name=QUARANTINE_TARGET_GROUP,
                scope="target",
                window_seconds=MIN_WINDOW_SECONDS,
            ),
        ]
        for sentinel in sentinels:
            doc = sentinel.model_dump()
            doc_id = sentinel.build_id()
            await collection.update_one(
                {"_id": doc_id},
                {"$setOnInsert": {**doc, "_id": doc_id}},
                upsert=True,
            )
        self.invalidate_cache()
        logger.info("Seeded reserved quarantine groups (idempotent)")
