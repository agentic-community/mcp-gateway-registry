"""DocumentDB-based repository for MCP server storage."""

import logging
from datetime import datetime
from typing import Any, Dict, Optional

from motor.motor_asyncio import AsyncIOMotorCollection
from pymongo.errors import DuplicateKeyError

from ...core.config import settings
from ..interfaces import ServerRepositoryBase
from .client import get_collection_name, get_documentdb_client


logger = logging.getLogger(__name__)


class DocumentDBServerRepository(ServerRepositoryBase):
    """DocumentDB implementation of server repository."""

    def __init__(self):
        self._collection: Optional[AsyncIOMotorCollection] = None
        self._collection_name = get_collection_name("mcp_servers")

    async def _get_collection(self) -> AsyncIOMotorCollection:
        """Get DocumentDB collection."""
        if self._collection is None:
            db = await get_documentdb_client()
            self._collection = db[self._collection_name]
        return self._collection

    async def load_all(self) -> None:
        """Load all servers from DocumentDB."""
        logger.info(f"Loading servers from DocumentDB collection: {self._collection_name}")
        collection = await self._get_collection()

        try:
            count = await collection.count_documents({})
            logger.info(f"Loaded {count} servers from DocumentDB")
        except Exception as e:
            logger.error(f"Error loading servers from DocumentDB: {e}", exc_info=True)

    async def get(
        self,
        path: str,
    ) -> Optional[Dict[str, Any]]:
        """Get server by path."""
        logger.debug(
            f"DocumentDB READ: Getting server with path='{path}' from collection '{self._collection_name}'"
        )
        collection = await self._get_collection()

        try:
            server_info = await collection.find_one({"_id": path})
            if server_info:
                server_info["path"] = server_info.pop("_id")
                logger.debug(
                    f"DocumentDB READ: Found server '{server_info.get('server_name', 'unknown')}' at '{path}'"
                )
            else:
                logger.debug(f"DocumentDB READ: Server not found at '{path}'")
            return server_info
        except Exception as e:
            logger.error(f"Error getting server '{path}' from DocumentDB: {e}", exc_info=True)
            return None

    async def list_all(self) -> Dict[str, Dict[str, Any]]:
        """List all servers."""
        logger.debug(
            f"DocumentDB READ: Listing all servers from collection '{self._collection_name}'"
        )
        collection = await self._get_collection()

        try:
            cursor = collection.find({})
            servers = {}
            async for doc in cursor:
                path = doc.pop("_id")
                doc["path"] = path
                servers[path] = doc
            logger.info(
                f"DocumentDB READ: Retrieved {len(servers)} servers from collection '{self._collection_name}'"
            )
            return servers
        except Exception as e:
            logger.error(f"Error listing servers from DocumentDB: {e}", exc_info=True)
            return {}

    async def list_by_source(
        self,
        source: str,
    ) -> Dict[str, Dict[str, Any]]:
        """List all servers from a specific federation source.

        Args:
            source: Federation source identifier (e.g., "anthropic")

        Returns:
            Dictionary mapping server path to server info
        """
        logger.debug(
            f"DocumentDB READ: Listing servers with source='{source}' from collection '{self._collection_name}'"
        )
        collection = await self._get_collection()

        try:
            cursor = collection.find({"source": source})
            servers = {}
            async for doc in cursor:
                path = doc.pop("_id")
                doc["path"] = path
                servers[path] = doc
            logger.info(
                f"DocumentDB READ: Retrieved {len(servers)} servers with source='{source}' from collection '{self._collection_name}'"
            )
            return servers
        except Exception as e:
            logger.error(
                f"Error listing servers by source '{source}' from DocumentDB: {e}", exc_info=True
            )
            return {}

    async def create(
        self,
        server_info: Dict[str, Any],
    ) -> bool:
        """Create a new server."""
        path = server_info["path"]
        logger.debug(
            f"DocumentDB WRITE: Creating server '{server_info.get('server_name', 'unknown')}' at '{path}' in collection '{self._collection_name}'"
        )
        collection = await self._get_collection()

        server_info["registered_at"] = datetime.utcnow().isoformat()
        server_info["updated_at"] = datetime.utcnow().isoformat()
        server_info.setdefault("is_enabled", False)

        try:
            doc = {**server_info}
            doc["_id"] = path
            doc.pop("path", None)

            await collection.insert_one(doc)
            logger.info(
                f"DocumentDB WRITE: Created server '{server_info['server_name']}' at '{path}'"
            )
            return True
        except DuplicateKeyError:
            logger.error(f"Server path '{path}' already exists in DocumentDB")
            return False
        except Exception as e:
            logger.error(f"Failed to create server in DocumentDB: {e}", exc_info=True)
            return False

    async def update(
        self,
        path: str,
        server_info: Dict[str, Any],
    ) -> bool:
        """Update an existing server."""
        logger.debug(
            f"DocumentDB WRITE: Updating server at '{path}' in collection '{self._collection_name}'"
        )
        collection = await self._get_collection()

        server_info["updated_at"] = datetime.utcnow().isoformat()

        try:
            doc = {**server_info}
            doc.pop("path", None)

            result = await collection.update_one({"_id": path}, {"$set": doc})

            if result.matched_count == 0:
                logger.error(f"Server at '{path}' not found in DocumentDB")
                return False

            logger.info(
                f"DocumentDB WRITE: Updated server '{server_info.get('server_name', 'unknown')}' at '{path}'"
            )
            return True
        except Exception as e:
            logger.error(f"Failed to update server in DocumentDB: {e}", exc_info=True)
            return False

    async def delete(
        self,
        path: str,
    ) -> bool:
        """Delete a server."""
        logger.debug(
            f"DocumentDB DELETE: Deleting server at '{path}' from collection '{self._collection_name}'"
        )
        collection = await self._get_collection()

        try:
            server_doc = await collection.find_one({"_id": path})
            if not server_doc:
                logger.error(f"Server at '{path}' not found in DocumentDB")
                return False

            server_name = server_doc.get("server_name", "Unknown")

            result = await collection.delete_one({"_id": path})

            if result.deleted_count == 0:
                logger.error(f"Failed to delete server at '{path}'")
                return False

            logger.info(f"DocumentDB DELETE: Deleted server '{server_name}' from '{path}'")
            return True
        except Exception as e:
            logger.error(f"Failed to delete server from DocumentDB: {e}", exc_info=True)
            return False

    async def delete_with_versions(
        self,
        path: str,
    ) -> int:
        """Delete a server and all its version documents.

        Deletes the active document at `path` and any inactive version
        documents with IDs matching `{path}:{version}`.

        Args:
            path: Server base path (e.g., "/context7")

        Returns:
            Number of documents deleted (0 if none found)
        """
        logger.debug(
            f"DocumentDB DELETE: Deleting server at '{path}' and all version documents "
            f"from collection '{self._collection_name}'"
        )
        collection = await self._get_collection()

        try:
            # Match the active document (exact path) and version documents (path:version)
            filter_query = {
                "$or": [
                    {"_id": path},
                    {"_id": {"$regex": f"^{path}:"}},
                ]
            }

            result = await collection.delete_many(filter_query)
            deleted_count = result.deleted_count

            if deleted_count == 0:
                logger.error(f"No documents found for server at '{path}'")
            else:
                logger.info(
                    f"DocumentDB DELETE: Deleted {deleted_count} document(s) "
                    f"for server at '{path}' (active + version documents)"
                )

            return deleted_count
        except Exception as e:
            logger.error(
                f"Failed to delete server and versions from DocumentDB: {e}",
                exc_info=True,
            )
            return 0

    async def get_state(
        self,
        path: str,
    ) -> bool:
        """Get server enabled/disabled state."""
        server_info = await self.get(path)
        if server_info:
            return server_info.get("is_enabled", False)
        return False

    async def set_state(
        self,
        path: str,
        enabled: bool,
    ) -> bool:
        """Set server enabled/disabled state."""
        collection = await self._get_collection()

        try:
            server_doc = await collection.find_one({"_id": path})
            if not server_doc:
                logger.error(f"Server at '{path}' not found in DocumentDB")
                return False

            server_name = server_doc.get("server_name", path)

            result = await collection.update_one(
                {"_id": path},
                {"$set": {"is_enabled": enabled, "updated_at": datetime.utcnow().isoformat()}},
            )

            if result.matched_count == 0:
                logger.error(f"Server at '{path}' not found")
                return False

            logger.info(f"Toggled '{server_name}' ({path}) to {enabled}")
            return True
        except Exception as e:
            logger.error(f"Failed to update server state in DocumentDB: {e}", exc_info=True)
            return False

    async def count(self) -> int:
        """Get total count of servers.

        Returns:
            Total number of servers in the repository.
        """
        logger.debug(
            f"DocumentDB COUNT: Counting servers in collection '{self._collection_name}'"
        )
        collection = await self._get_collection()

        try:
            count = await collection.count_documents({})
            logger.debug(f"DocumentDB COUNT: Found {count} servers")
            return count
        except Exception as e:
            logger.error(f"Error counting servers in DocumentDB: {e}", exc_info=True)
            return 0
