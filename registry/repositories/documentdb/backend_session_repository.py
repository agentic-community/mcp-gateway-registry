"""
DocumentDB (MongoDB) implementation for backend session repository.

Stores per-client backend MCP session mappings in MongoDB with a TTL index
on last_used_at for automatic cleanup of idle sessions. Uses compound keys
(<client_session_id>:<backend_key>) as _id for fast lookups.
"""

import logging
from datetime import UTC, datetime

from motor.motor_asyncio import AsyncIOMotorCollection
from pymongo import ASCENDING
from pymongo.errors import DuplicateKeyError

from ..interfaces import BackendSessionRepositoryBase
from .client import get_collection_name, get_documentdb_client

# Session TTL: 1 hour of inactivity
SESSION_TTL_SECONDS: int = 3600

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s,p%(process)s,{%(filename)s:%(lineno)d},%(levelname)s,%(message)s",
)
logger = logging.getLogger(__name__)


def _make_backend_session_id(
    client_session_id: str,
    backend_key: str,
) -> str:
    """Build compound _id for a backend session document."""
    return f"{client_session_id}:{backend_key}"


def _make_client_session_id(
    client_session_id: str,
) -> str:
    """Build _id for a client session document."""
    return f"client:{client_session_id}"


class DocumentDBBackendSessionRepository(BackendSessionRepositoryBase):
    """MongoDB implementation for backend session storage."""

    def __init__(self):
        self._collection: AsyncIOMotorCollection | None = None
        self._collection_name = get_collection_name("backend_sessions")
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

        Creates:
        - TTL index on last_used_at (expires after SESSION_TTL_SECONDS)
        - Index on client_session_id for listing all backend sessions per client
        """
        if self._indexes_created:
            return

        if self._collection is None:
            return

        try:
            # TTL index: auto-delete documents after SESSION_TTL_SECONDS of inactivity
            await self._collection.create_index(
                [("last_used_at", ASCENDING)],
                expireAfterSeconds=SESSION_TTL_SECONDS,
                name="ttl_last_used_at",
            )

            # Index for querying all backend sessions for a given client session
            await self._collection.create_index(
                [("client_session_id", ASCENDING)],
                name="idx_client_session_id",
            )

            self._indexes_created = True
            logger.info(
                f"Created indexes for {self._collection_name} collection "
                f"(TTL={SESSION_TTL_SECONDS}s)"
            )
        except Exception as e:
            logger.warning(f"Could not create indexes for {self._collection_name}: {e}")

    async def get_backend_session(
        self,
        client_session_id: str,
        backend_key: str,
        user_id: str | None = None,
    ) -> str | None:
        """Get backend session ID and atomically bump last_used_at.

        Uses find_one_and_update so the TTL is refreshed on every access,
        keeping active sessions alive.

        When ``user_id`` is provided, the lookup also filters on the stored
        owner. This is defense in depth: although every in-router path to a
        backend session already passes the owner-bound client-session gate,
        binding the backend-session read to the owner too means a single missed
        gate cannot leak another user's live backend session ID. The owner
        match is in the same atomic query that refreshes the TTL.

        Args:
            client_session_id: Client-facing session ID
            backend_key: Backend location key
            user_id: Authenticated user identity the session must belong to.
                When None, ownership is not enforced (legacy behavior).

        Returns:
            Backend session ID if found (and owned by ``user_id`` when given),
            None otherwise
        """
        collection = await self._get_collection()
        doc_id = _make_backend_session_id(client_session_id, backend_key)

        query: dict[str, str] = {"_id": doc_id}
        # `is not None` is load-bearing: only None means "no owner filter". An
        # empty-string user_id must add an (always-non-matching) filter, NOT skip
        # it -- do not "simplify" this to `if user_id:` or an empty owner would
        # reopen the existence-only hijack gap.
        if user_id is not None:
            query["user_id"] = user_id

        result = await collection.find_one_and_update(
            query,
            {"$set": {"last_used_at": datetime.now(UTC)}},
        )

        if result:
            return result.get("backend_session_id")
        return None

    async def store_backend_session(
        self,
        client_session_id: str,
        backend_key: str,
        backend_session_id: str,
        user_id: str,
        virtual_server_path: str,
    ) -> None:
        """Store or update a backend session (upsert).

        Args:
            client_session_id: Client-facing session ID
            backend_key: Backend location key
            backend_session_id: Session ID from the backend MCP server
            user_id: User identity for audit
            virtual_server_path: Virtual server path
        """
        collection = await self._get_collection()
        doc_id = _make_backend_session_id(client_session_id, backend_key)
        now = datetime.now(UTC)

        doc = {
            "_id": doc_id,
            "client_session_id": client_session_id,
            "backend_key": backend_key,
            "backend_session_id": backend_session_id,
            "user_id": user_id,
            "virtual_server_path": virtual_server_path,
            "created_at": now,
            "last_used_at": now,
        }

        # Pin the owner in the upsert filter so a document with this _id owned by
        # a different user is never overwritten. client_session_id is already
        # owner-namespaced (each initialize mints a fresh vs-<uuid4> for one
        # user), so a mismatch cannot happen on the legitimate path; if it ever
        # did, the upsert tries to insert a new doc with a colliding _id and
        # raises DuplicateKeyError, which we swallow as a safe refusal rather
        # than clobbering another user's live backend session.
        try:
            await collection.replace_one(
                {"_id": doc_id, "user_id": user_id},
                doc,
                upsert=True,
            )
            logger.debug(f"Stored backend session: {doc_id} -> {backend_session_id}")
        except DuplicateKeyError:
            logger.warning(
                f"Refused to store backend session {doc_id} for user={user_id}: "
                f"_id already owned by a different user"
            )

    async def delete_backend_session(
        self,
        client_session_id: str,
        backend_key: str,
        user_id: str | None = None,
    ) -> None:
        """Delete a stale backend session.

        When ``user_id`` is provided, the delete is scoped to the owner so a
        caller can only invalidate its own backend session (defense in depth,
        symmetric with get/store). When None, ownership is not enforced
        (legacy behavior).

        Args:
            client_session_id: Client-facing session ID
            backend_key: Backend location key
            user_id: Authenticated user identity the session must belong to.
        """
        collection = await self._get_collection()
        doc_id = _make_backend_session_id(client_session_id, backend_key)

        query: dict[str, str] = {"_id": doc_id}
        # `is not None` is load-bearing -- see get_backend_session: only None
        # means "no owner filter"; an empty string must filter, not skip.
        if user_id is not None:
            query["user_id"] = user_id

        result = await collection.delete_one(query)
        if result.deleted_count > 0:
            logger.debug(f"Deleted backend session: {doc_id}")

    async def create_client_session(
        self,
        client_session_id: str,
        user_id: str,
        virtual_server_path: str,
    ) -> None:
        """Create a client session document for validation.

        Uses the same collection with a 'client:' prefix on _id to
        distinguish from backend session documents.

        Args:
            client_session_id: Generated client session ID
            user_id: User identity from auth context
            virtual_server_path: Virtual server path
        """
        collection = await self._get_collection()
        doc_id = _make_client_session_id(client_session_id)
        now = datetime.now(UTC)

        doc = {
            "_id": doc_id,
            "client_session_id": client_session_id,
            "user_id": user_id,
            "virtual_server_path": virtual_server_path,
            "created_at": now,
            "last_used_at": now,
        }

        await collection.insert_one(doc)
        logger.info(
            f"Created client session: {client_session_id} "
            f"for user={user_id} path={virtual_server_path}"
        )

    async def validate_client_session(
        self,
        client_session_id: str,
        user_id: str | None = None,
        virtual_server_path: str | None = None,
    ) -> bool:
        """Check if a client session exists, is owned by ``user_id``, and bump last_used_at.

        When ``user_id`` is provided, the lookup filters on the stored owner so a
        session belonging to a different user is treated as not found. This binds
        the session to the authenticated identity and prevents session hijacking
        via a guessed or stolen Mcp-Session-Id header (the match is done in the
        same atomic query that refreshes the TTL, so there is no check/use gap).

        When ``virtual_server_path`` is provided, the lookup also filters on it so
        a session minted for one virtual server cannot be replayed against
        another (issue #2: virtual-server binding).

        Args:
            client_session_id: Client-facing session ID
            user_id: Authenticated user identity the session must belong to.
                When None, ownership is not enforced (legacy behavior).
            virtual_server_path: Virtual server path the session was minted for.
                When None, the path is not enforced (legacy behavior).

        Returns:
            True if session exists (and matches ``user_id`` /
            ``virtual_server_path`` when given), False otherwise
        """
        collection = await self._get_collection()
        doc_id = _make_client_session_id(client_session_id)

        query: dict[str, str] = {"_id": doc_id}
        # `is not None` is load-bearing -- see get_backend_session: only None
        # means "no owner filter"; an empty string must filter, not skip.
        if user_id is not None:
            query["user_id"] = user_id
        if virtual_server_path is not None:
            query["virtual_server_path"] = virtual_server_path

        result = await collection.find_one_and_update(
            query,
            {"$set": {"last_used_at": datetime.now(UTC)}},
        )

        return result is not None
