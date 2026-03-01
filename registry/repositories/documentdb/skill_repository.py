"""
DocumentDB (MongoDB) implementation for skill repository.

Implements all recommendations:
- Index creation on initialization
- Batch operations for federation sync
- Database-level filtering
- Duplicate key handling
"""

import logging
from datetime import datetime
from typing import (
    Any,
    Dict,
    List,
    Optional,
)

from motor.motor_asyncio import AsyncIOMotorCollection

from pymongo.errors import DuplicateKeyError

from ...exceptions import (
    SkillAlreadyExistsError,
    SkillServiceError,
)
from ..interfaces import SkillRepositoryBase
from ...schemas.skill_models import SkillCard
from .client import get_collection_name, get_documentdb_client


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s,p%(process)s,{%(filename)s:%(lineno)d},%(levelname)s,%(message)s",
)
logger = logging.getLogger(__name__)


def _skill_to_document(
    skill: SkillCard,
) -> Dict[str, Any]:
    """Convert SkillCard to MongoDB document."""
    doc = skill.model_dump(mode="json")
    doc["_id"] = skill.path
    return doc


def _document_to_skill(
    doc: Dict[str, Any],
) -> SkillCard:
    """Convert MongoDB document to SkillCard."""
    doc_copy = dict(doc)
    doc_copy.pop("_id", None)
    return SkillCard(**doc_copy)


class DocumentDBSkillRepository(SkillRepositoryBase):
    """MongoDB implementation for skill storage."""

    def __init__(self):
        self._collection: Optional[AsyncIOMotorCollection] = None
        self._collection_name = get_collection_name("agent_skills")
        self._indexes_created = False

    async def _get_collection(self) -> AsyncIOMotorCollection:
        """Get DocumentDB collection."""
        if self._collection is None:
            db = await get_documentdb_client()
            self._collection = db[self._collection_name]
        return self._collection

    async def ensure_indexes(self) -> None:
        """Create required indexes if not present."""
        if self._indexes_created:
            return

        collection = await self._get_collection()

        try:
            # Name index (unique)
            await collection.create_index("name", unique=True)

            # Tags index for filtering
            await collection.create_index("tags")

            # Visibility index for access control
            await collection.create_index("visibility")

            # Registry name for federation queries
            await collection.create_index("registry_name")

            # Owner for private visibility
            await collection.create_index("owner")

            # Compound index for common query patterns
            await collection.create_index(
                [("visibility", 1), ("is_enabled", 1), ("registry_name", 1)]
            )

            self._indexes_created = True
            logger.info(f"Created indexes for {self._collection_name} collection")
        except Exception as e:
            logger.warning(f"Could not create indexes: {e}")

    async def get(
        self,
        path: str,
    ) -> Optional[SkillCard]:
        """Get a skill by path."""
        await self.ensure_indexes()
        collection = await self._get_collection()
        doc = await collection.find_one({"_id": path})
        if doc:
            return _document_to_skill(doc)
        return None

    async def list_all(
        self,
        skip: int = 0,
        limit: int = 100,
    ) -> List[SkillCard]:
        """List all skills with pagination.

        Args:
            skip: Number of records to skip (offset)
            limit: Maximum number of records to return

        Returns:
            List of SkillCard objects
        """
        await self.ensure_indexes()
        collection = await self._get_collection()
        skills = []
        cursor = collection.find({}).skip(skip).limit(limit)
        async for doc in cursor:
            try:
                skills.append(_document_to_skill(doc))
            except Exception as e:
                logger.error(f"Failed to parse skill document: {e}")
        return skills

    async def list_filtered(
        self,
        include_disabled: bool = False,
        tag: Optional[str] = None,
        visibility: Optional[str] = None,
        registry_name: Optional[str] = None,
    ) -> List[SkillCard]:
        """List skills with database-level filtering."""
        await self.ensure_indexes()
        collection = await self._get_collection()

        query: Dict[str, Any] = {}

        if not include_disabled:
            query["is_enabled"] = True

        if tag:
            query["tags"] = tag

        if visibility:
            query["visibility"] = visibility

        if registry_name:
            query["registry_name"] = registry_name

        skills = []
        cursor = collection.find(query)
        async for doc in cursor:
            try:
                skills.append(_document_to_skill(doc))
            except Exception as e:
                logger.error(f"Failed to parse skill document: {e}")
        return skills

    async def create(
        self,
        skill: SkillCard,
    ) -> SkillCard:
        """Create a new skill."""
        await self.ensure_indexes()
        collection = await self._get_collection()
        doc = _skill_to_document(skill)

        try:
            await collection.insert_one(doc)
            logger.info(f"Created skill: {skill.path}")
            return skill
        except DuplicateKeyError:
            logger.error(f"Skill already exists: {skill.path}")
            raise SkillAlreadyExistsError(skill.name)
        except Exception as e:
            logger.error(f"Failed to create skill {skill.path}: {e}")
            raise SkillServiceError(f"Failed to create skill: {e}") from e

    async def update(
        self,
        path: str,
        updates: Dict[str, Any],
    ) -> Optional[SkillCard]:
        """Update a skill."""
        await self.ensure_indexes()
        collection = await self._get_collection()
        updates["updated_at"] = datetime.utcnow().isoformat()

        try:
            result = await collection.find_one_and_update(
                {"_id": path}, {"$set": updates}, return_document=True
            )

            if result:
                logger.info(f"Updated skill: {path}")
                return _document_to_skill(result)
            return None
        except Exception as e:
            logger.error(f"Failed to update skill {path}: {e}")
            raise SkillServiceError(f"Failed to update skill: {e}") from e

    async def delete(
        self,
        path: str,
    ) -> bool:
        """Delete a skill."""
        await self.ensure_indexes()
        collection = await self._get_collection()
        result = await collection.delete_one({"_id": path})
        if result.deleted_count > 0:
            logger.info(f"Deleted skill: {path}")
            return True
        return False

    async def get_state(
        self,
        path: str,
    ) -> bool:
        """Get skill enabled state."""
        await self.ensure_indexes()
        collection = await self._get_collection()
        doc = await collection.find_one({"_id": path}, {"is_enabled": 1})
        return doc.get("is_enabled", False) if doc else False

    async def set_state(
        self,
        path: str,
        enabled: bool,
    ) -> bool:
        """Set skill enabled state."""
        await self.ensure_indexes()
        collection = await self._get_collection()
        result = await collection.update_one(
            {"_id": path},
            {"$set": {"is_enabled": enabled, "updated_at": datetime.utcnow().isoformat()}},
        )
        if result.modified_count > 0:
            logger.info(f"Set skill {path} enabled={enabled}")
            return True
        return False

    async def create_many(
        self,
        skills: List[SkillCard],
    ) -> List[SkillCard]:
        """Create multiple skills in single operation."""
        await self.ensure_indexes()
        collection = await self._get_collection()

        if not skills:
            return []

        docs = [_skill_to_document(s) for s in skills]

        try:
            await collection.insert_many(docs, ordered=False)
            logger.info(f"Created {len(skills)} skills in batch")
            return skills
        except Exception as e:
            logger.error(f"Failed to create skills in batch: {e}")
            raise SkillServiceError(f"Batch create failed: {e}") from e

    async def update_many(
        self,
        updates: Dict[str, Dict[str, Any]],
    ) -> int:
        """Update multiple skills by path, return count."""
        await self.ensure_indexes()
        collection = await self._get_collection()

        if not updates:
            return 0

        count = 0
        for path, update_data in updates.items():
            update_data["updated_at"] = datetime.utcnow().isoformat()
            result = await collection.update_one({"_id": path}, {"$set": update_data}, upsert=True)
            if result.modified_count > 0 or result.upserted_id:
                count += 1

        logger.info(f"Updated {count} skills in batch")
        return count

    async def count(self) -> int:
        """Get total count of skills.

        Returns:
            Total number of skills in the repository.
        """
        await self.ensure_indexes()
        collection = await self._get_collection()

        try:
            count = await collection.count_documents({})
            logger.debug(f"DocumentDB COUNT: Found {count} skills")
            return count
        except Exception as e:
            logger.error(f"Error counting skills in DocumentDB: {e}", exc_info=True)
            return 0
