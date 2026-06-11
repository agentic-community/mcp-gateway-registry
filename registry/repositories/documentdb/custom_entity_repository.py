"""
DocumentDB (MongoDB) implementation for custom entity records.

All custom records of every type live in a single ``mcp_custom_entities``
collection, scoped by the ``entity_type`` discriminator. List and count
accept an optional ``visibility_filter`` so the route layer can deliver a
``total_count`` that matches the slice the caller can actually see
(in-query visibility filtering, index-covered by ``{entity_type:1, visibility:1}``).
"""

import asyncio
import logging
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from motor.motor_asyncio import AsyncIOMotorCollection
from pymongo import ReturnDocument
from pymongo.errors import DuplicateKeyError

from ...schemas.custom_entity_models import CustomEntityRecord
from ..interfaces import CustomEntityRepositoryBase
from .client import get_collection_name, get_documentdb_client

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s,p%(process)s,{%(filename)s:%(lineno)d},%(levelname)s,%(message)s",
)
logger = logging.getLogger(__name__)


def _document_to_record(
    doc: dict[str, Any],
) -> CustomEntityRecord:
    """Convert a Mongo document to a CustomEntityRecord (drops _id)."""
    doc_copy = dict(doc)
    doc_copy.pop("_id", None)
    return CustomEntityRecord(**doc_copy)


class DocumentDBCustomEntityRepository(CustomEntityRepositoryBase):
    """MongoDB implementation for custom entity record storage."""

    def __init__(self):
        self._collection: AsyncIOMotorCollection | None = None
        self._collection_name = get_collection_name("mcp_custom_entities")
        self._indexes_created = False
        self._init_lock: asyncio.Lock | None = None

    async def _get_collection(self) -> AsyncIOMotorCollection:
        """Get the records collection, creating indexes on first access."""
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
        """Create the compound indexes backing list/count queries."""
        if self._indexes_created:
            return
        collection = await self._get_collection()
        try:
            # Backs list_paginated/count without a visibility filter (admin).
            await collection.create_index([("entity_type", 1), ("_id", 1)])
            # Backs list_paginated/count WITH a visibility filter.
            await collection.create_index([("entity_type", 1), ("visibility", 1)])
            self._indexes_created = True
            logger.info(f"Created indexes for {self._collection_name} collection")
        except Exception as e:
            logger.warning(f"Could not create indexes: {e}")

    async def create(
        self,
        record: CustomEntityRecord,
    ) -> CustomEntityRecord:
        """Create a new record (``_id`` = record.path)."""
        coll = await self._get_collection()
        doc = record.model_dump(mode="json")
        doc["_id"] = record.path  # /{type}/{uuid}
        try:
            await coll.insert_one(doc)
            logger.info(f"Created custom record: {record.path} ({record.entity_type})")
            return record
        except DuplicateKeyError:
            # uuid4 collision is astronomically unlikely; regenerate once
            # rather than leak an uncaught driver exception. A second
            # collision propagates as an uncaught error -> 500 (do NOT loop).
            record.path = f"/{record.entity_type}/{uuid4()}"
            doc["_id"] = record.path
            doc["path"] = record.path  # keep doc["path"] in sync with _id
            await coll.insert_one(doc)
            logger.info(f"Created custom record after uuid retry: {record.path}")
            return record

    async def get(
        self,
        path: str,
    ) -> CustomEntityRecord | None:
        """Get a record by its synthetic path."""
        coll = await self._get_collection()
        doc = await coll.find_one({"_id": path})
        if doc is None:
            return None
        return _document_to_record(doc)

    async def list_paginated(
        self,
        entity_type: str,
        skip: int = 0,
        limit: int = 100,
        visibility_filter: dict[str, Any] | None = None,
    ) -> list[CustomEntityRecord]:
        """List records of a type with DB-level pagination and optional filter."""
        coll = await self._get_collection()
        query: dict[str, Any] = {"entity_type": entity_type}
        if visibility_filter:
            query.update(visibility_filter)
        records: list[CustomEntityRecord] = []
        cursor = coll.find(query).sort("_id", 1).skip(skip).limit(limit)
        async for doc in cursor:
            try:
                records.append(_document_to_record(doc))
            except Exception as e:
                logger.error(f"Failed to parse custom record document: {e}")
        return records

    async def update(
        self,
        path: str,
        updates: dict[str, Any],
    ) -> CustomEntityRecord | None:
        """Update a record via ``$set`` of the given fields."""
        coll = await self._get_collection()
        updates["updated_at"] = datetime.now(UTC).isoformat()
        doc = await coll.find_one_and_update(
            {"_id": path},
            {"$set": updates},
            return_document=ReturnDocument.AFTER,
        )
        if doc is None:
            return None
        logger.info(f"Updated custom record: {path}")
        return _document_to_record(doc)

    async def delete(
        self,
        path: str,
    ) -> bool:
        """Delete a single record."""
        coll = await self._get_collection()
        res = await coll.delete_one({"_id": path})
        if res.deleted_count > 0:
            logger.info(f"Deleted custom record: {path}")
            return True
        return False

    async def delete_by_type(
        self,
        entity_type: str,
    ) -> int:
        """Bulk-delete all records of a type."""
        coll = await self._get_collection()
        res = await coll.delete_many({"entity_type": entity_type})
        logger.info(f"Deleted {res.deleted_count} records of type {entity_type}")
        return res.deleted_count

    async def count(
        self,
        entity_type: str,
        visibility_filter: dict[str, Any] | None = None,
    ) -> int:
        """Count records of a type, applying the SAME optional filter as list."""
        coll = await self._get_collection()
        query: dict[str, Any] = {"entity_type": entity_type}
        if visibility_filter:
            query.update(visibility_filter)
        return await coll.count_documents(query)
