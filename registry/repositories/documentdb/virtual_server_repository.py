"""
DocumentDB (MongoDB) implementation for virtual server repository.

Stores virtual MCP server configurations in MongoDB with the path
as the document _id, following the same patterns as skill and server
repositories.
"""

import logging
from datetime import UTC, datetime
from typing import (
    Any,
)

from motor.motor_asyncio import AsyncIOMotorCollection
from pymongo.errors import DuplicateKeyError

from ...exceptions import (
    VirtualServerAlreadyExistsError,
    VirtualServerServiceError,
)
from ...schemas.virtual_server_models import VirtualServerConfig
from ..interfaces import VirtualServerRepositoryBase
from .client import get_collection_name, get_documentdb_client

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s,p%(process)s,{%(filename)s:%(lineno)d},%(levelname)s,%(message)s",
)
logger = logging.getLogger(__name__)


def _config_to_document(
    config: VirtualServerConfig,
) -> dict[str, Any]:
    """Convert VirtualServerConfig to MongoDB document."""
    doc = config.model_dump(mode="json")
    doc["_id"] = config.path
    return doc


def _document_to_config(
    doc: dict[str, Any],
) -> VirtualServerConfig:
    """Convert MongoDB document to VirtualServerConfig."""
    doc_copy = dict(doc)
    doc_copy.pop("_id", None)
    return VirtualServerConfig(**doc_copy)


class DocumentDBVirtualServerRepository(VirtualServerRepositoryBase):
    """MongoDB implementation for virtual server storage."""

    def __init__(self):
        self._collection: AsyncIOMotorCollection | None = None
        self._collection_name = get_collection_name("virtual_servers")
        self._indexes_created = False

    async def _get_collection(self) -> AsyncIOMotorCollection:
        """Get DocumentDB collection, creating indexes on first access."""
        if self._collection is None:
            db = await get_documentdb_client()
            self._collection = db[self._collection_name]
            await self.ensure_indexes()
        return self._collection

    async def ensure_indexes(self) -> None:
        """Create required indexes if not present.

        Called automatically on first collection access via _get_collection().
        """
        if self._indexes_created:
            return

        if self._collection is None:
            return

        try:
            # Enabled state index for list_enabled queries
            await self._collection.create_index("is_enabled")

            # Tags index for filtering
            await self._collection.create_index("tags")

            # Server name index for search
            await self._collection.create_index("server_name")

            # Compound index for common query patterns
            await self._collection.create_index(
                [
                    ("is_enabled", 1),
                    ("server_name", 1),
                ]
            )

            self._indexes_created = True
            logger.info(f"Created indexes for {self._collection_name} collection")
        except Exception as e:
            logger.warning(f"Could not create indexes for {self._collection_name}: {e}")

    async def get(
        self,
        path: str,
    ) -> VirtualServerConfig | None:
        """Get a virtual server by path."""
        collection = await self._get_collection()
        doc = await collection.find_one({"_id": path})
        if doc:
            return _document_to_config(doc)
        return None

    async def list_all(self) -> list[VirtualServerConfig]:
        """List all virtual servers."""
        collection = await self._get_collection()
        configs = []
        cursor = collection.find({})
        async for doc in cursor:
            try:
                configs.append(_document_to_config(doc))
            except Exception as e:
                doc_id = doc.get("_id", "unknown")
                logger.error(f"Failed to parse virtual server document '{doc_id}': {e}")
        return configs

    async def list_enabled(self) -> list[VirtualServerConfig]:
        """List only enabled virtual servers."""
        collection = await self._get_collection()
        configs = []
        cursor = collection.find({"is_enabled": True})
        async for doc in cursor:
            try:
                configs.append(_document_to_config(doc))
            except Exception as e:
                doc_id = doc.get("_id", "unknown")
                logger.error(f"Failed to parse virtual server document '{doc_id}': {e}")
        return configs

    async def create(
        self,
        config: VirtualServerConfig,
    ) -> VirtualServerConfig:
        """Create a new virtual server."""
        collection = await self._get_collection()
        doc = _config_to_document(config)

        try:
            await collection.insert_one(doc)
            logger.info(f"Created virtual server: {config.path}")
            return config
        except DuplicateKeyError:
            logger.error(f"Virtual server already exists: {config.path}")
            raise VirtualServerAlreadyExistsError(config.path)
        except Exception as e:
            logger.error(f"Failed to create virtual server {config.path}: {e}")
            raise VirtualServerServiceError(f"Failed to create virtual server: {e}") from e

    async def update(
        self,
        path: str,
        updates: dict[str, Any],
    ) -> VirtualServerConfig | None:
        """Update a virtual server."""
        collection = await self._get_collection()
        updates["updated_at"] = datetime.now(UTC).isoformat()

        try:
            result = await collection.find_one_and_update(
                {"_id": path},
                {"$set": updates},
                return_document=True,
            )

            if result:
                logger.info(f"Updated virtual server: {path}")
                return _document_to_config(result)
            return None
        except Exception as e:
            logger.error(f"Failed to update virtual server {path}: {e}")
            raise VirtualServerServiceError(f"Failed to update virtual server: {e}") from e

    async def delete(
        self,
        path: str,
    ) -> bool:
        """Delete a virtual server."""
        collection = await self._get_collection()
        result = await collection.delete_one({"_id": path})
        if result.deleted_count > 0:
            logger.info(f"Deleted virtual server: {path}")
            return True
        return False

    async def get_state(
        self,
        path: str,
    ) -> bool:
        """Get virtual server enabled state."""
        collection = await self._get_collection()
        doc = await collection.find_one(
            {"_id": path},
            {"is_enabled": 1},
        )
        return doc.get("is_enabled", False) if doc else False

    async def set_state(
        self,
        path: str,
        enabled: bool,
    ) -> bool:
        """Set virtual server enabled state."""
        collection = await self._get_collection()
        result = await collection.update_one(
            {"_id": path},
            {
                "$set": {
                    "is_enabled": enabled,
                    "updated_at": datetime.now(UTC).isoformat(),
                }
            },
        )
        if result.modified_count > 0:
            logger.info(f"Set virtual server {path} enabled={enabled}")
            return True
        return False

    async def update_rating(
        self,
        path: str,
        num_stars: float,
        rating_details: list[dict[str, Any]],
    ) -> bool:
        """Update virtual server rating.

        Args:
            path: Virtual server path
            num_stars: Calculated average star rating
            rating_details: List of rating entries with user and rating

        Returns:
            True if update succeeded, False if server not found
        """
        collection = await self._get_collection()
        result = await collection.update_one(
            {"_id": path},
            {
                "$set": {
                    "num_stars": num_stars,
                    "rating_details": rating_details,
                    "updated_at": datetime.now(UTC).isoformat(),
                }
            },
        )
        if result.modified_count > 0 or result.matched_count > 0:
            logger.info(f"Updated rating for virtual server {path}: {num_stars:.2f} stars")
            return True
        return False

    async def get_rating(
        self,
        path: str,
    ) -> dict[str, Any] | None:
        """Get virtual server rating info.

        Args:
            path: Virtual server path

        Returns:
            Dict with num_stars and rating_details, or None if not found
        """
        collection = await self._get_collection()
        doc = await collection.find_one(
            {"_id": path},
            {"num_stars": 1, "rating_details": 1},
        )
        if doc:
            return {
                "num_stars": doc.get("num_stars", 0.0),
                "rating_details": doc.get("rating_details", []),
            }
        return None
