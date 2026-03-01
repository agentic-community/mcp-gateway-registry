"""DocumentDB-based repository for security scan results storage."""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from motor.motor_asyncio import AsyncIOMotorCollection

from ..interfaces import SecurityScanRepositoryBase
from .client import get_collection_name, get_documentdb_client


logger = logging.getLogger(__name__)


class DocumentDBSecurityScanRepository(SecurityScanRepositoryBase):
    """DocumentDB implementation of security scan repository."""

    def __init__(self):
        self._collection: Optional[AsyncIOMotorCollection] = None
        self._collection_name = get_collection_name("mcp_security_scans")

    async def _get_collection(self) -> AsyncIOMotorCollection:
        """Get DocumentDB collection."""
        if self._collection is None:
            db = await get_documentdb_client()
            self._collection = db[self._collection_name]
        return self._collection

    async def load_all(self) -> None:
        """Load all security scan results from DocumentDB."""
        logger.info(f"Loading security scans from DocumentDB collection: {self._collection_name}")
        collection = await self._get_collection()

        try:
            count = await collection.count_documents({})
            logger.info(f"Loaded {count} security scan results from DocumentDB")
        except Exception as e:
            logger.error(f"Error loading security scans from DocumentDB: {e}", exc_info=True)

    async def get(
        self,
        server_path: str,
    ) -> Optional[Dict[str, Any]]:
        """Get latest security scan result for a server."""
        return await self.get_latest(server_path)

    async def list_all(self) -> List[Dict[str, Any]]:
        """List all security scan results."""
        collection = await self._get_collection()

        try:
            cursor = collection.find({}).sort("scan_timestamp", -1)
            scans = []
            async for doc in cursor:
                doc.pop("_id", None)
                scans.append(doc)
            return scans
        except Exception as e:
            logger.error(f"Error listing security scans from DocumentDB: {e}", exc_info=True)
            return []

    async def create(
        self,
        scan_result: Dict[str, Any],
    ) -> bool:
        """Create/update a security scan result."""
        try:
            path = scan_result.get("server_path") or scan_result.get("agent_path")
            if not path:
                logger.error("Scan result must contain either 'server_path' or 'agent_path' field")
                return False

            collection = await self._get_collection()

            if "agent_path" in scan_result and "server_path" not in scan_result:
                scan_result["server_path"] = scan_result["agent_path"]

            if "scan_timestamp" not in scan_result:
                scan_result["scan_timestamp"] = datetime.utcnow().isoformat()

            if "vulnerabilities" in scan_result and isinstance(
                scan_result["vulnerabilities"], list
            ):
                vuln_counts = {"critical": 0, "high": 0, "medium": 0, "low": 0}
                for vuln in scan_result["vulnerabilities"]:
                    severity = vuln.get("severity", "").lower()
                    if severity in vuln_counts:
                        vuln_counts[severity] += 1

                scan_result["total_vulnerabilities"] = len(scan_result["vulnerabilities"])
                scan_result["critical_count"] = vuln_counts["critical"]
                scan_result["high_count"] = vuln_counts["high"]
                scan_result["medium_count"] = vuln_counts["medium"]
                scan_result["low_count"] = vuln_counts["low"]

            await collection.insert_one(scan_result)

            logger.info(f"Indexed security scan for {path} in DocumentDB")
            return True
        except Exception as e:
            logger.error(f"Failed to index security scan in DocumentDB: {e}", exc_info=True)
            return False

    async def get_latest(
        self,
        server_path: str,
    ) -> Optional[Dict[str, Any]]:
        """Get latest scan result for a server."""
        try:
            collection = await self._get_collection()

            scan_doc = await collection.find_one(
                {"server_path": server_path}, sort=[("scan_timestamp", -1)]
            )

            if scan_doc:
                scan_doc.pop("_id", None)
                return scan_doc

            return None
        except Exception as e:
            logger.error(f"Failed to get latest scan from DocumentDB: {e}", exc_info=True)
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
            logger.error(f"Failed to query scans by status from DocumentDB: {e}", exc_info=True)
            return []
