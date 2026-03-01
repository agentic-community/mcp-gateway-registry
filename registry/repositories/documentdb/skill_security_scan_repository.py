"""DocumentDB-based repository for skill security scan results storage."""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from motor.motor_asyncio import AsyncIOMotorCollection

from ..interfaces import SkillSecurityScanRepositoryBase
from .client import get_collection_name, get_documentdb_client


logger = logging.getLogger(__name__)


class DocumentDBSkillSecurityScanRepository(SkillSecurityScanRepositoryBase):
    """DocumentDB implementation of skill security scan repository."""

    def __init__(self):
        self._collection: Optional[AsyncIOMotorCollection] = None
        self._collection_name = get_collection_name("mcp_skill_security_scans")

    async def _get_collection(self) -> AsyncIOMotorCollection:
        """Get DocumentDB collection."""
        if self._collection is None:
            db = await get_documentdb_client()
            self._collection = db[self._collection_name]
        return self._collection

    async def load_all(self) -> None:
        """Load all skill security scan results from DocumentDB."""
        logger.info(
            f"Loading skill security scans from DocumentDB collection: {self._collection_name}"
        )
        collection = await self._get_collection()

        try:
            count = await collection.count_documents({})
            logger.info(f"Loaded {count} skill security scan results from DocumentDB")
        except Exception as e:
            logger.error(f"Error loading skill security scans from DocumentDB: {e}", exc_info=True)

    async def get(
        self,
        skill_path: str,
    ) -> Optional[Dict[str, Any]]:
        """Get latest security scan result for a skill."""
        return await self.get_latest(skill_path)

    async def list_all(self) -> List[Dict[str, Any]]:
        """List all skill security scan results."""
        collection = await self._get_collection()

        try:
            cursor = collection.find({}).sort("scan_timestamp", -1)
            scans = []
            async for doc in cursor:
                doc.pop("_id", None)
                scans.append(doc)
            return scans
        except Exception as e:
            logger.error(f"Error listing skill security scans from DocumentDB: {e}", exc_info=True)
            return []

    async def create(
        self,
        scan_result: Dict[str, Any],
    ) -> bool:
        """Create/update a skill security scan result."""
        try:
            skill_path = scan_result.get("skill_path")
            if not skill_path:
                logger.error("Scan result must contain 'skill_path' field")
                return False

            collection = await self._get_collection()

            if "scan_timestamp" not in scan_result:
                scan_result["scan_timestamp"] = datetime.utcnow().isoformat()

            await collection.insert_one(scan_result)

            logger.info(f"Indexed skill security scan for {skill_path} in DocumentDB")
            return True
        except Exception as e:
            logger.error(f"Failed to index skill security scan in DocumentDB: {e}", exc_info=True)
            return False

    async def get_latest(
        self,
        skill_path: str,
    ) -> Optional[Dict[str, Any]]:
        """Get latest scan result for a skill."""
        try:
            collection = await self._get_collection()

            scan_doc = await collection.find_one(
                {"skill_path": skill_path}, sort=[("scan_timestamp", -1)]
            )

            if scan_doc:
                scan_doc.pop("_id", None)
                return scan_doc

            return None
        except Exception as e:
            logger.error(f"Failed to get latest skill scan from DocumentDB: {e}", exc_info=True)
            return None

    async def query_by_status(
        self,
        status: str,
    ) -> List[Dict[str, Any]]:
        """Query scan results by status."""
        try:
            collection = await self._get_collection()

            cursor = collection.find({"scan_status": status}).sort("scan_timestamp", -1)

            scans = []
            async for doc in cursor:
                doc.pop("_id", None)
                scans.append(doc)

            return scans
        except Exception as e:
            logger.error(
                f"Failed to query skill scans by status from DocumentDB: {e}", exc_info=True
            )
            return []
