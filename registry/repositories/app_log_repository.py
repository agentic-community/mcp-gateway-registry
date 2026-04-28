"""Read-only repository for querying application logs stored in MongoDB."""

import logging
from datetime import datetime
from typing import Any

from motor.motor_asyncio import AsyncIOMotorCollection

from .documentdb.client import get_collection_name, get_documentdb_client

logger = logging.getLogger(__name__)


class AppLogRepository:
    """Queries the ``application_logs`` collection written by MongoDBLogHandler."""

    def __init__(self):
        self._collection: AsyncIOMotorCollection | None = None
        self._collection_name = get_collection_name("application_logs")

    async def _get_collection(self) -> AsyncIOMotorCollection:
        if self._collection is None:
            db = await get_documentdb_client()
            self._collection = db[self._collection_name]
        return self._collection

    async def query(
        self,
        service: str | None = None,
        level_no: int | None = None,
        hostname: str | None = None,
        start: datetime | None = None,
        end: datetime | None = None,
        search: str | None = None,
        skip: int = 0,
        limit: int = 100,
    ) -> tuple[list[dict[str, Any]], int]:
        """Query application log entries with filtering and pagination.

        Args:
            service: Filter by service name (registry, auth-server).
            level_no: Minimum log level number (10=DEBUG, 20=INFO, etc.).
            hostname: Filter by pod/hostname.
            start: Only include entries at or after this timestamp.
            end: Only include entries at or before this timestamp.
            search: Substring search within the message field (pre-escaped).
            skip: Number of entries to skip (offset).
            limit: Maximum number of entries to return.

        Returns:
            Tuple of (list of log documents, total matching count).
        """
        collection = await self._get_collection()

        query_filter: dict[str, Any] = {}

        if service:
            query_filter["service"] = service
        if level_no is not None:
            query_filter["level_no"] = {"$gte": level_no}
        if hostname:
            query_filter["hostname"] = hostname

        time_filter: dict[str, Any] = {}
        if start:
            time_filter["$gte"] = start
        if end:
            time_filter["$lte"] = end
        if time_filter:
            query_filter["timestamp"] = time_filter

        if search:
            query_filter["message"] = {"$regex": search, "$options": "i"}

        try:
            if not query_filter:
                total = await collection.estimated_document_count()
            else:
                total = await collection.count_documents(query_filter)

            cursor = collection.find(query_filter).sort("timestamp", -1).skip(skip).limit(limit)

            results: list[dict[str, Any]] = []
            async for doc in cursor:
                doc.pop("_id", None)
                results.append(doc)

            return results, total
        except Exception as e:
            logger.error(f"Error querying application logs: {e}", exc_info=True)
            return [], 0

    async def get_distinct_services(self) -> list[str]:
        """Get list of distinct service names in the log collection."""
        collection = await self._get_collection()
        try:
            return await collection.distinct("service")
        except Exception as e:
            logger.error(f"Error fetching distinct services: {e}", exc_info=True)
            return []

    async def get_distinct_hostnames(self) -> list[str]:
        """Get list of distinct hostnames in the log collection."""
        collection = await self._get_collection()
        try:
            return await collection.distinct("hostname")
        except Exception as e:
            logger.error(f"Error fetching distinct hostnames: {e}", exc_info=True)
            return []
