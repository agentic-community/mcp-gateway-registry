"""
DocumentDB (MongoDB) implementation for custom entity type descriptors.

Stores admin-authored ``CustomTypeDescriptor`` documents keyed by the
immutable type name. Owns the in-process ``CustomTypeCache`` used by the
service/config/search hot paths.
"""

import asyncio
import logging
import time
from typing import Any

from motor.motor_asyncio import AsyncIOMotorCollection
from pymongo import ReturnDocument
from pymongo.errors import DuplicateKeyError

from ...core.config import settings
from ...schemas.custom_entity_models import CustomTypeDescriptor
from ...services.custom_entity_errors import CustomTypeAlreadyExistsError
from ..interfaces import CustomTypeRepositoryBase
from .client import get_collection_name, get_documentdb_client

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s,p%(process)s,{%(filename)s:%(lineno)d},%(levelname)s,%(message)s",
)
logger = logging.getLogger(__name__)


class CustomTypeCache:
    """In-process descriptor cache owned by the type repository.

    Exposes two distinct reads with different correctness contracts:

    - ``get_for_read``/``list_descriptors`` — best-effort, served from a
      lazily-refreshed snapshot (short TTL). Used by config/search hot paths
      where a brief cross-replica staleness window is acceptable.
    - ``get_for_write`` — AUTHORITATIVE, always hits Mongo. Used by
      create/update record so a type that exists but isn't yet in this
      replica's snapshot is never falsely reported missing (and a deleted
      type is never falsely reported present).
    """

    def __init__(
        self,
        repo: "DocumentDBCustomTypeRepository",
    ) -> None:
        self._repo = repo
        self._snapshot: dict[str, CustomTypeDescriptor] | None = None
        self._loaded_at: float = 0.0
        self._lock = asyncio.Lock()  # prevents thundering herd on cold start

    async def get_for_read(
        self,
        name: str,
    ) -> CustomTypeDescriptor | None:
        """Best-effort read from snapshot; falls through to Mongo on miss."""
        snap = await self._snapshot_fresh()
        if name in snap:
            return snap[name]
        found = await self._repo.get(name)
        if found is not None:
            # Don't mutate snapshot in-place — create a new dict.
            self._snapshot = {**snap, name: found}
        return found

    async def get_for_write(
        self,
        name: str,
    ) -> CustomTypeDescriptor | None:
        """AUTHORITATIVE read — always hits Mongo (used by create/update record)."""
        return await self._repo.get(name)

    async def list_descriptors(self) -> list[CustomTypeDescriptor]:
        """Return all descriptors from the (possibly refreshed) snapshot."""
        return list((await self._snapshot_fresh()).values())

    def invalidate(self) -> None:
        """Drop the cached snapshot; the next read reloads from Mongo."""
        self._snapshot = None

    async def _snapshot_fresh(self) -> dict[str, CustomTypeDescriptor]:
        """Return a fresh snapshot, reloading from Mongo if stale/empty."""
        ttl = settings.custom_type_cache_ttl_seconds
        if self._snapshot is None or (time.monotonic() - self._loaded_at) > ttl:
            async with self._lock:  # single-flight reload
                if self._snapshot is None or (time.monotonic() - self._loaded_at) > ttl:
                    self._snapshot = {d.name: d for d in await self._repo.list_all()}
                    self._loaded_at = time.monotonic()
        return self._snapshot


class DocumentDBCustomTypeRepository(CustomTypeRepositoryBase):
    """MongoDB implementation for custom type descriptor storage."""

    def __init__(self):
        self._collection: AsyncIOMotorCollection | None = None
        self._collection_name = get_collection_name("mcp_custom_types")
        self._indexes_created = False
        self._init_lock: asyncio.Lock | None = None
        self.cache = CustomTypeCache(self)

    async def _get_collection(self) -> AsyncIOMotorCollection:
        """Get the descriptor collection, creating indexes on first access."""
        if self._collection is not None:
            return self._collection
        if self._init_lock is None:
            self._init_lock = asyncio.Lock()
        async with self._init_lock:
            if self._collection is not None:
                return self._collection
            db = await get_documentdb_client()
            collection = db[self._collection_name]
            self._collection = collection
            await self.ensure_indexes()
            return self._collection

    async def ensure_indexes(self) -> None:
        """Create required indexes if not present.

        The descriptor ``_id`` is the type name (implicitly unique), so no
        additional index is strictly required; the method exists to satisfy
        the base contract and to anchor any future indexes.
        """
        if self._indexes_created:
            return
        self._indexes_created = True
        logger.info(f"Indexes ready for {self._collection_name} collection")

    async def create(
        self,
        descriptor: CustomTypeDescriptor,
    ) -> CustomTypeDescriptor:
        """Create a new type descriptor."""
        coll = await self._get_collection()
        doc = descriptor.model_dump(mode="json")
        doc["_id"] = descriptor.name
        try:
            await coll.insert_one(doc)
            logger.info(f"Created custom type: {descriptor.name}")
            return descriptor
        except DuplicateKeyError as e:
            logger.warning(f"Custom type already exists: {descriptor.name}")
            raise CustomTypeAlreadyExistsError(descriptor.name) from e

    async def get(
        self,
        name: str,
    ) -> CustomTypeDescriptor | None:
        """Get a type descriptor by name."""
        coll = await self._get_collection()
        doc = await coll.find_one({"_id": name})
        return CustomTypeDescriptor(**doc) if doc else None

    async def list_all(self) -> list[CustomTypeDescriptor]:
        """List all type descriptors, sorted by name."""
        coll = await self._get_collection()
        descriptors: list[CustomTypeDescriptor] = []
        async for doc in coll.find({}).sort("_id", 1):
            try:
                descriptors.append(CustomTypeDescriptor(**doc))
            except Exception as e:
                logger.error(f"Failed to parse custom type document: {e}")
        return descriptors

    async def delete(
        self,
        name: str,
    ) -> bool:
        """Delete a type descriptor."""
        coll = await self._get_collection()
        res = await coll.delete_one({"_id": name})
        if res.deleted_count > 0:
            logger.info(f"Deleted custom type: {name}")
            return True
        return False

    async def update_metadata(
        self,
        name: str,
        updates: dict[str, Any],
    ) -> CustomTypeDescriptor | None:
        """Update mutable metadata (display_name/description) of a type.

        Only the supplied keys are written via ``$set``; the immutable
        ``_id``/``name``/``fields`` are never touched. Returns the updated
        descriptor, or None if no type with this name exists.
        """
        if not updates:
            # Nothing to change; return the current descriptor (or None).
            return await self.get(name)
        coll = await self._get_collection()
        doc = await coll.find_one_and_update(
            {"_id": name},
            {"$set": updates},
            return_document=ReturnDocument.AFTER,
        )
        if doc is None:
            return None
        logger.info(f"Updated custom type metadata: {name} ({list(updates.keys())})")
        return CustomTypeDescriptor(**doc)

    async def count_types(self) -> int:
        """Count defined custom types (used by the metrics scrape gauge)."""
        coll = await self._get_collection()
        return await coll.count_documents({})
