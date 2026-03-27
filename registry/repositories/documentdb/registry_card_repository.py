"""DocumentDB repository for registry card storage."""

import logging
from datetime import UTC, datetime

from motor.motor_asyncio import AsyncIOMotorCollection

from ...schemas.registry_card import RegistryCard
from ..interfaces import RegistryCardRepositoryBase
from .client import get_collection_name, get_documentdb_client

logger = logging.getLogger(__name__)


class DocumentDBRegistryCardRepository(RegistryCardRepositoryBase):
    """DocumentDB implementation of Registry Card repository."""

    def __init__(self):
        self._collection: AsyncIOMotorCollection | None = None
        self._collection_name = get_collection_name("registry_cards")
        logger.info(
            f"Initialized DocumentDB RegistryCardRepository with collection: "
            f"{self._collection_name}"
        )

    async def _get_collection(self) -> AsyncIOMotorCollection:
        """Get DocumentDB collection (lazy initialization)."""
        if self._collection is None:
            db = await get_documentdb_client()
            self._collection = db[self._collection_name]
        return self._collection

    async def get(self) -> RegistryCard | None:
        """Retrieve the Registry Card."""
        try:
            collection = await self._get_collection()

            doc = await collection.find_one({"_id": "default"})

            if not doc:
                logger.debug("No registry card found in database")
                return None

            doc.pop("_id", None)

            card = RegistryCard(**doc)
            logger.info("Retrieved registry card from DocumentDB")
            return card

        except Exception as e:
            logger.error(f"Error getting registry card: {e}", exc_info=True)
            return None

    async def save(
        self,
        card: RegistryCard,
    ) -> RegistryCard:
        """Save or update the Registry Card using upsert."""
        try:
            collection = await self._get_collection()

            card_id = "default"
            existing = await collection.find_one({"_id": card_id})

            doc = card.model_dump(mode="json")

            now = datetime.now(UTC).isoformat()
            if existing:
                doc["created_at"] = existing.get("created_at", now)
                doc["updated_at"] = now
            else:
                doc["created_at"] = now
                doc["updated_at"] = now

            doc["_id"] = card_id

            await collection.replace_one({"_id": card_id}, doc, upsert=True)

            logger.info("Saved registry card to DocumentDB")
            return card

        except Exception as e:
            logger.error(f"Error saving registry card: {e}", exc_info=True)
            raise

    async def exists(self) -> bool:
        """Check if Registry Card exists."""
        try:
            collection = await self._get_collection()

            result = await collection.find_one({"_id": "default"})

            exists = result is not None
            logger.debug(f"Registry card exists check: {exists}")
            return exists

        except Exception as e:
            logger.error(f"Error checking registry card existence: {e}", exc_info=True)
            return False
