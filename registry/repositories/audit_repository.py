"""
Audit repository for storing and querying audit events.

This module provides the abstract base class and DocumentDB implementation
for audit event storage, supporting the audit logging system's MongoDB
warm storage requirements.
"""

import logging
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Union

from motor.motor_asyncio import AsyncIOMotorCollection
from pymongo.errors import DuplicateKeyError

from ..audit.models import MCPServerAccessRecord, RegistryApiAccessRecord
from ..core.config import settings
from .documentdb.client import get_collection_name, get_documentdb_client


logger = logging.getLogger(__name__)

# Type alias for audit records
AuditRecord = Union[RegistryApiAccessRecord, MCPServerAccessRecord]


class AuditRepositoryBase(ABC):
    """
    Abstract base class for audit event data access.

    Implementations:
    - DocumentDBAuditRepository: MongoDB/DocumentDB storage with TTL
    """

    @abstractmethod
    async def find(
        self,
        query: Dict[str, Any],
        limit: int = 50,
        offset: int = 0,
        sort_field: str = "timestamp",
        sort_order: int = -1,
    ) -> List[Dict[str, Any]]:
        """
        Find audit events matching the query.

        Args:
            query: MongoDB query filter
            limit: Maximum number of results to return
            offset: Number of results to skip for pagination
            sort_field: Field to sort by (default: timestamp)
            sort_order: Sort order (-1 for descending, 1 for ascending)

        Returns:
            List of audit event documents
        """
        pass

    @abstractmethod
    async def find_one(
        self,
        query: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        """
        Find a single audit event matching the query.

        Args:
            query: MongoDB query filter

        Returns:
            Audit event document if found, None otherwise
        """
        pass

    @abstractmethod
    async def count(
        self,
        query: Dict[str, Any],
    ) -> int:
        """
        Count audit events matching the query.

        Args:
            query: MongoDB query filter

        Returns:
            Number of matching documents
        """
        pass

    @abstractmethod
    async def distinct(
        self,
        field: str,
        query: Optional[Dict[str, Any]] = None,
    ) -> List[str]:
        """
        Get distinct values for a field in audit events.

        Args:
            field: The document field path (e.g., 'identity.username')
            query: Optional filter query to scope the distinct values

        Returns:
            Sorted list of distinct string values
        """
        pass

    @abstractmethod
    async def aggregate(
        self,
        pipeline: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Run a MongoDB aggregation pipeline on audit events.

        Args:
            pipeline: MongoDB aggregation pipeline stages

        Returns:
            List of aggregation result documents
        """
        pass

    @abstractmethod
    async def insert(
        self,
        record: AuditRecord,
    ) -> bool:
        """
        Insert an audit event record.

        Args:
            record: The audit record to insert (RegistryApiAccessRecord or MCPServerAccessRecord)

        Returns:
            True if inserted successfully, False otherwise
        """
        pass


class DocumentDBAuditRepository(AuditRepositoryBase):
    """
    DocumentDB/MongoDB implementation of audit repository.

    Stores audit events in a MongoDB collection with TTL index
    for automatic expiration of old events.
    """

    def __init__(self):
        self._collection: Optional[AsyncIOMotorCollection] = None
        self._collection_name = get_collection_name("audit_events")

    async def _get_collection(self) -> AsyncIOMotorCollection:
        """Get DocumentDB collection."""
        if self._collection is None:
            db = await get_documentdb_client()
            self._collection = db[self._collection_name]
        return self._collection

    async def find(
        self,
        query: Dict[str, Any],
        limit: int = 50,
        offset: int = 0,
        sort_field: str = "timestamp",
        sort_order: int = -1,
    ) -> List[Dict[str, Any]]:
        """
        Find audit events matching the query.

        Args:
            query: MongoDB query filter
            limit: Maximum number of results to return
            offset: Number of results to skip for pagination
            sort_field: Field to sort by (default: timestamp)
            sort_order: Sort order (-1 for descending, 1 for ascending)

        Returns:
            List of audit event documents
        """
        logger.debug(
            f"DocumentDB READ: Finding audit events with query={query}, "
            f"limit={limit}, offset={offset}"
        )
        collection = await self._get_collection()

        try:
            cursor = collection.find(query)
            cursor = cursor.sort(sort_field, sort_order)
            cursor = cursor.skip(offset).limit(limit)

            events = []
            async for doc in cursor:
                # Convert _id to string if it's an ObjectId
                if "_id" in doc:
                    doc["_id"] = str(doc["_id"])
                # Motor returns naive datetimes; re-attach UTC for correct serialization
                if isinstance(doc.get("timestamp"), datetime) and doc["timestamp"].tzinfo is None:
                    doc["timestamp"] = doc["timestamp"].replace(tzinfo=timezone.utc)
                events.append(doc)

            logger.debug(f"DocumentDB READ: Found {len(events)} audit events")
            return events
        except Exception as e:
            logger.error(f"Error finding audit events: {e}", exc_info=True)
            return []

    async def find_one(
        self,
        query: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        """
        Find a single audit event matching the query.

        Args:
            query: MongoDB query filter

        Returns:
            Audit event document if found, None otherwise
        """
        logger.debug(f"DocumentDB READ: Finding single audit event with query={query}")
        collection = await self._get_collection()

        try:
            doc = await collection.find_one(query)
            if doc:
                # Convert _id to string if it's an ObjectId
                if "_id" in doc:
                    doc["_id"] = str(doc["_id"])
                # Motor returns naive datetimes; re-attach UTC for correct serialization
                if isinstance(doc.get("timestamp"), datetime) and doc["timestamp"].tzinfo is None:
                    doc["timestamp"] = doc["timestamp"].replace(tzinfo=timezone.utc)
                logger.debug(
                    f"DocumentDB READ: Found audit event with request_id={doc.get('request_id')}"
                )
            else:
                logger.debug("DocumentDB READ: Audit event not found")
            return doc
        except Exception as e:
            logger.error(f"Error finding audit event: {e}", exc_info=True)
            return None

    async def count(
        self,
        query: Dict[str, Any],
    ) -> int:
        """
        Count audit events matching the query.

        Args:
            query: MongoDB query filter

        Returns:
            Number of matching documents
        """
        logger.debug(f"DocumentDB READ: Counting audit events with query={query}")
        collection = await self._get_collection()

        try:
            count = await collection.count_documents(query)
            logger.debug(f"DocumentDB READ: Counted {count} audit events")
            return count
        except Exception as e:
            logger.error(f"Error counting audit events: {e}", exc_info=True)
            return 0

    async def distinct(
        self,
        field: str,
        query: Optional[Dict[str, Any]] = None,
    ) -> List[str]:
        """Get distinct values for a field in audit events."""
        logger.debug(f"DocumentDB READ: Getting distinct values for field={field}")
        collection = await self._get_collection()
        try:
            values = await collection.distinct(field, query or {})
            result = sorted([str(v) for v in values if v])
            logger.debug(f"DocumentDB READ: Found {len(result)} distinct values for {field}")
            return result
        except Exception as e:
            logger.error(f"Error getting distinct values for {field}: {e}", exc_info=True)
            return []

    async def aggregate(
        self,
        pipeline: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Run a MongoDB aggregation pipeline on audit events."""
        logger.debug(f"DocumentDB READ: Running aggregation pipeline with {len(pipeline)} stages")
        collection = await self._get_collection()
        try:
            results = []
            async for doc in collection.aggregate(pipeline):
                results.append(doc)
            logger.debug(f"DocumentDB READ: Aggregation returned {len(results)} results")
            return results
        except Exception as e:
            logger.error(f"Error running aggregation pipeline: {e}", exc_info=True)
            return []

    async def insert(
        self,
        record: AuditRecord,
    ) -> bool:
        """
        Insert an audit event record.

        Args:
            record: The audit record to insert (RegistryApiAccessRecord or MCPServerAccessRecord)

        Returns:
            True if inserted successfully or if the record already exists (duplicate request_id),
            False if an unexpected error occurs
        """
        logger.debug(f"DocumentDB WRITE: Inserting audit event with request_id={record.request_id}")
        collection = await self._get_collection()

        try:
            # Convert Pydantic model to dict
            doc = record.model_dump(mode="json")

            # Ensure timestamp is stored as datetime for TTL index
            if isinstance(doc.get("timestamp"), str):
                doc["timestamp"] = datetime.fromisoformat(doc["timestamp"].replace("Z", "+00:00"))

            await collection.insert_one(doc)
            logger.info(f"DocumentDB WRITE: Inserted audit event request_id={record.request_id}")
            return True
        except DuplicateKeyError:
            logger.debug(
                f"DocumentDB WRITE: Skipped duplicate audit event for request_id={record.request_id}. "
                f"This occurs when the same request_id is processed twice (auth validation + endpoint execution). "
                f"Returning True to not break the request."
            )
            return True
        except Exception as e:
            logger.error(f"Error inserting audit event: {e}", exc_info=True)
            return False
