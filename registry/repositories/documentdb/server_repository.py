"""DocumentDB-based repository for MCP server storage."""

import asyncio
import logging
from datetime import datetime
from typing import Any

from motor.motor_asyncio import AsyncIOMotorCollection
from pymongo.errors import DuplicateKeyError

from ...utils.url_normalize import ENTITY_TYPE_SERVER, NORMALIZED_IDENTITY_URL_FIELD
from ..interfaces import ServerRepositoryBase
from ._identity_url_sidecar import (
    backfill_normalized_identity_url,
    ensure_normalized_identity_url_index,
    find_by_normalized_identity_url,
    populate_normalized_identity_url,
)
from .client import get_collection_name, get_documentdb_client

logger = logging.getLogger(__name__)


class DocumentDBServerRepository(ServerRepositoryBase):
    """DocumentDB implementation of server repository."""

    def __init__(self):
        self._collection: AsyncIOMotorCollection | None = None
        self._collection_name = get_collection_name("mcp_servers")
        # Guards the one-shot index/backfill block so two coroutines
        # racing on the first _get_collection call don't both run the
        # backfill cursor. The lock is constructed lazily inside the
        # method so it binds to the running event loop, not the loop
        # active at __init__ time (factories may run before the loop
        # exists).
        self._init_lock: asyncio.Lock | None = None

    async def _get_collection(self) -> AsyncIOMotorCollection:
        """Get DocumentDB collection.

        On first acquisition we also create the sparse index on the
        ``_identity_url_normalized`` sidecar field (used by
        :meth:`find_by_identity_url` for registration dedup as an
        indexed ``$eq`` lookup) and lazily backfill the sidecar onto
        documents registered before this field existed. The init
        block runs under an asyncio.Lock so concurrent callers don't
        race on the backfill cursor; the second caller observes the
        already-populated ``self._collection`` and returns
        immediately.
        """
        if self._collection is not None:
            return self._collection
        if self._init_lock is None:
            self._init_lock = asyncio.Lock()
        async with self._init_lock:
            if self._collection is not None:
                return self._collection
            db = await get_documentdb_client()
            collection = db[self._collection_name]
            await ensure_normalized_identity_url_index(
                collection,
                self._collection_name,
            )
            await backfill_normalized_identity_url(
                collection,
                self._collection_name,
                ENTITY_TYPE_SERVER,
            )
            # Publish only after the index + backfill are in place so
            # other coroutines never see a half-initialized state.
            self._collection = collection
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
    ) -> dict[str, Any] | None:
        """Get server by path.

        Normalizes paths to match both with and without trailing slashes.
        """
        logger.debug(
            f"DocumentDB READ: Getting server with path='{path}' from collection '{self._collection_name}'"
        )
        collection = await self._get_collection()

        try:
            server_info = await collection.find_one({"_id": path})

            # If not found, try alternate path (with/without trailing slash)
            if not server_info:
                if path.endswith("/"):
                    alternate_path = path.rstrip("/")
                else:
                    alternate_path = path + "/"

                logger.debug(f"DocumentDB READ: Trying alternate path '{alternate_path}'")
                server_info = await collection.find_one({"_id": alternate_path})

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

    async def list_all(
        self,
        exclude_tool_list: bool = False,
    ) -> dict[str, dict[str, Any]]:
        """List all servers.

        Args:
            exclude_tool_list: If True, project out the ``tool_list`` field so
                heavy tool schemas are not transferred from the database.
        """
        logger.debug(
            f"DocumentDB READ: Listing all servers from collection '{self._collection_name}' "
            f"(exclude_tool_list={exclude_tool_list})"
        )
        collection = await self._get_collection()

        projection = {"tool_list": 0} if exclude_tool_list else None

        try:
            cursor = collection.find({}, projection)
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

    async def list_paginated(
        self,
        skip: int = 0,
        limit: int = 100,
        exclude_tool_list: bool = False,
    ) -> dict[str, dict[str, Any]]:
        """List servers with DB-level skip/limit pagination.

        Args:
            skip: Number of documents to skip.
            limit: Maximum number of documents to return.
            exclude_tool_list: If True, project out the ``tool_list`` field so
                heavy tool schemas are not transferred from the database.

        Returns:
            Dictionary mapping server path to server info for the requested page.
        """
        logger.debug(
            f"DocumentDB READ: Listing paginated servers (skip={skip}, limit={limit}) "
            f"from collection '{self._collection_name}' (exclude_tool_list={exclude_tool_list})"
        )
        collection = await self._get_collection()

        projection = {"tool_list": 0} if exclude_tool_list else None

        try:
            cursor = collection.find({}, projection).sort("_id", 1).skip(skip).limit(limit)
            servers = {}
            async for doc in cursor:
                path = doc.pop("_id")
                doc["path"] = path
                servers[path] = doc
            logger.info(
                f"DocumentDB READ: Retrieved {len(servers)} servers (skip={skip}, limit={limit}) "
                f"from collection '{self._collection_name}'"
            )
            return servers
        except Exception as e:
            logger.error(f"Error listing paginated servers from DocumentDB: {e}", exc_info=True)
            return {}

    async def list_by_ids(
        self,
        paths: list[str],
    ) -> dict[str, dict[str, Any]]:
        """List servers whose _id is in the given set of paths.

        Only exact paths are matched, so version documents
        (``{path}:{version}``) are never returned.

        Args:
            paths: Exact server paths to fetch.

        Returns:
            Dictionary mapping server path to server info for found paths.
        """
        if not paths:
            return {}

        logger.debug(
            f"DocumentDB READ: Listing {len(paths)} servers by id "
            f"from collection '{self._collection_name}'"
        )
        collection = await self._get_collection()

        try:
            cursor = collection.find({"_id": {"$in": paths}})
            servers = {}
            async for doc in cursor:
                path = doc.pop("_id")
                doc["path"] = path
                servers[path] = doc
            logger.info(
                f"DocumentDB READ: Retrieved {len(servers)} of {len(paths)} requested servers "
                f"from collection '{self._collection_name}'"
            )
            return servers
        except Exception as e:
            logger.error(f"Error listing servers by ids from DocumentDB: {e}", exc_info=True)
            return {}

    async def list_by_source(
        self,
        source: str,
    ) -> dict[str, dict[str, Any]]:
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
        server_info: dict[str, Any],
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
            populate_normalized_identity_url(doc, ENTITY_TYPE_SERVER)

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
        server_info: dict[str, Any],
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
            populate_normalized_identity_url(doc, ENTITY_TYPE_SERVER)
            unset_ops: dict[str, str] = {}
            # update() may carry no proxy_pass_url at all (a partial
            # patch that doesn't touch the URL); leave the existing
            # sidecar alone in that case. But if proxy_pass_url is
            # explicitly cleared, drop the sidecar so the sparse
            # index doesn't keep a stale value.
            if (
                NORMALIZED_IDENTITY_URL_FIELD not in doc
                and "proxy_pass_url" in doc
                and not doc.get("proxy_pass_url")
            ):
                unset_ops[NORMALIZED_IDENTITY_URL_FIELD] = ""

            update_spec: dict[str, dict[str, Any]] = {"$set": doc}
            if unset_ops:
                update_spec["$unset"] = unset_ops
            result = await collection.update_one({"_id": path}, update_spec)

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

    async def get_all_states(self) -> dict[str, bool]:
        """Get enabled/disabled state for all servers in a single query."""
        collection = await self._get_collection()

        try:
            cursor = collection.find({}, {"_id": 1, "is_enabled": 1})
            states: dict[str, bool] = {}
            async for doc in cursor:
                server_path = doc.get("_id")
                if server_path:
                    states[server_path] = doc.get("is_enabled", False)
            return states
        except Exception as e:
            logger.error(f"Error getting all server states from DocumentDB: {e}", exc_info=True)
            return {}

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

    async def count(
        self,
        exclude_versions: bool = False,
    ) -> int:
        """Get total count of servers.

        Args:
            exclude_versions: When True, exclude version documents (``_id``
                containing ":") so the count reflects distinct servers only,
                matching the basis used by ``count_tools()``. Defaults to False
                to preserve the historical count of all documents.

        Returns:
            Total number of servers in the repository.
        """
        logger.debug(f"DocumentDB COUNT: Counting servers in collection '{self._collection_name}'")
        collection = await self._get_collection()

        query: dict[str, Any] = {"_id": {"$not": {"$regex": ":"}}} if exclude_versions else {}

        try:
            count = await collection.count_documents(query)
            logger.debug(f"DocumentDB COUNT: Found {count} servers")
            return count
        except Exception as e:
            logger.error(f"Error counting servers in DocumentDB: {e}", exc_info=True)
            return 0

    async def count_tools(self) -> int:
        """Get the total number of tools exposed across all servers.

        Tools are embedded in each server document under ``tool_list``; there is
        no separate tools collection. This sums the list sizes in a single
        aggregation (no N+1 loads). Version documents (``_id`` containing ":")
        are excluded so tools are not double-counted across versions.

        Returns:
            Total number of tools across all current servers.
        """
        logger.debug(f"DocumentDB COUNT: Counting tools in collection '{self._collection_name}'")
        collection = await self._get_collection()

        pipeline = [
            {"$match": {"_id": {"$not": {"$regex": ":"}}}},
            {"$group": {"_id": None, "total": {"$sum": {"$size": {"$ifNull": ["$tool_list", []]}}}}},
        ]

        try:
            cursor = collection.aggregate(pipeline)
            docs = await cursor.to_list(length=1)
            total = docs[0]["total"] if docs else 0
            logger.debug(f"DocumentDB COUNT: Found {total} tools")
            return total
        except Exception as e:
            logger.error(f"Error counting tools in DocumentDB: {e}", exc_info=True)
            return 0

    async def update_field(
        self,
        path: str,
        field: str,
        value: Any,
    ) -> bool:
        """Update a single field on a document."""
        collection = await self._get_collection()

        if value is None:
            result = await collection.update_one(
                {"_id": path},
                {"$unset": {field: ""}},
            )
        else:
            result = await collection.update_one(
                {"_id": path},
                {"$set": {field: value}},
            )

        return result.modified_count > 0

    async def find_with_filter(
        self,
        filter_dict: dict[str, Any],
        *,
        limit: int | None = None,
    ) -> dict[str, dict]:
        """Find documents matching a MongoDB-style filter."""
        collection = await self._get_collection()
        cursor = collection.find(filter_dict)
        if limit is not None:
            cursor = cursor.limit(limit)
        results = {}
        async for doc in cursor:
            doc_id = doc.pop("_id", None)
            if doc_id:
                results[doc_id] = doc
        return results

    async def find_by_identity_url(
        self,
        identity_url: str,
    ) -> dict[str, Any] | None:
        """Find a server whose ``proxy_pass_url`` normalizes to ``identity_url``.

        Indexed ``$eq`` lookup against the ``_identity_url_normalized``
        sidecar field, which write paths populate from
        ``proxy_pass_url`` at insert/update time. Single-document
        round trip; no full-collection scan.
        """
        collection = await self._get_collection()
        return await find_by_normalized_identity_url(collection, identity_url)
