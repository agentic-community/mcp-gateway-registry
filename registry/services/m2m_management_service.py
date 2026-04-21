"""Direct CRUD service for manually-registered M2M clients.

This service writes to the shared ``idp_m2m_clients`` collection without
calling any IdP Admin API. Records written by this service are tagged with
``provider == "manual"`` so they are distinguishable from IdP-synced records,
and only ``manual`` records can be modified or deleted via this API.

Tracked by issue #851.
"""

import logging
from datetime import datetime

from motor.motor_asyncio import AsyncIOMotorDatabase
from pymongo.errors import DuplicateKeyError

from registry.schemas.idp_m2m_client import (
    MANUAL_PROVIDER,
    IdPM2MClient,
    IdPM2MClientCreate,
    IdPM2MClientPatch,
)

logger = logging.getLogger(__name__)


COLLECTION_NAME: str = "idp_m2m_clients"


class M2MClientConflict(Exception):
    """Raised when a client_id already exists in the collection."""


class M2MClientNotFound(Exception):
    """Raised when the requested client_id does not exist."""


class M2MClientImmutable(Exception):
    """Raised when attempting to mutate a record owned by IdP sync."""


class M2MManagementService:
    """CRUD service for manually-registered M2M clients."""

    def __init__(
        self,
        db: AsyncIOMotorDatabase,
    ) -> None:
        self._collection = db[COLLECTION_NAME]

    async def ensure_indexes(self) -> None:
        """Create required indexes (idempotent).

        Creates a unique index on ``client_id`` to prevent duplicate registrations
        under concurrent POSTs.
        """
        await self._collection.create_index("client_id", unique=True)
        logger.info(
            "Ensured unique index on %s.client_id",
            COLLECTION_NAME,
        )

    async def create(
        self,
        payload: IdPM2MClientCreate,
        created_by: str | None,
    ) -> IdPM2MClient:
        """Insert a new manual M2M client record.

        Args:
            payload: Validated create request body.
            created_by: Username of the operator performing the action
                (captured from the authenticated user context).

        Returns:
            The persisted :class:`IdPM2MClient`.

        Raises:
            M2MClientConflict: If ``client_id`` already exists (unique index
                violation).
        """
        now = datetime.utcnow()
        doc: dict = {
            "client_id": payload.client_id,
            "name": payload.client_name,
            "description": payload.description,
            "groups": list(payload.groups),
            "enabled": True,
            "provider": MANUAL_PROVIDER,
            "idp_app_id": None,
            "created_by": created_by,
            "created_at": now,
            "updated_at": now,
        }
        try:
            await self._collection.insert_one(doc)
        except DuplicateKeyError as e:
            raise M2MClientConflict(payload.client_id) from e

        logger.info(
            "Registered manual M2M client client_id=%s name=%s groups=%s created_by=%s",
            payload.client_id,
            payload.client_name,
            payload.groups,
            created_by,
        )
        return IdPM2MClient(**doc)

    async def list_paged(
        self,
        provider: str | None = None,
        limit: int = 500,
        skip: int = 0,
    ) -> tuple[list[IdPM2MClient], int]:
        """Return a paginated slice of the collection.

        Args:
            provider: Optional filter on the ``provider`` field.
            limit: Maximum number of records to return on this page.
            skip: Number of records to skip (offset).

        Returns:
            Tuple of (items_on_page, total_matching_count).
        """
        query: dict = {}
        if provider is not None:
            query["provider"] = provider
        total = await self._collection.count_documents(query)
        cursor = self._collection.find(query).skip(skip).limit(limit)
        docs = await cursor.to_list(length=limit)
        return [IdPM2MClient(**d) for d in docs], total

    async def get(
        self,
        client_id: str,
    ) -> IdPM2MClient:
        """Fetch a single client by ``client_id``.

        Raises:
            M2MClientNotFound: If the record does not exist.
        """
        doc = await self._collection.find_one({"client_id": client_id})
        if doc is None:
            raise M2MClientNotFound(client_id)
        return IdPM2MClient(**doc)

    async def patch(
        self,
        client_id: str,
        payload: IdPM2MClientPatch,
    ) -> IdPM2MClient:
        """Update a manual M2M client record.

        Only records with ``provider == "manual"`` can be modified.

        Raises:
            M2MClientNotFound: If the record does not exist.
            M2MClientImmutable: If the record was written by IdP sync.
        """
        existing = await self._collection.find_one({"client_id": client_id})
        if existing is None:
            raise M2MClientNotFound(client_id)
        if existing.get("provider") != MANUAL_PROVIDER:
            raise M2MClientImmutable(client_id)

        # Pydantic v2: only fields explicitly set in the request body appear
        # in the dump. Callers clearing groups pass [] and it lands here too.
        provided = payload.model_dump(exclude_unset=True)

        field_map: dict[str, str] = {
            "client_name": "name",
            "groups": "groups",
            "description": "description",
            "enabled": "enabled",
        }
        updates: dict = {"updated_at": datetime.utcnow()}
        for request_field, storage_field in field_map.items():
            if request_field in provided:
                updates[storage_field] = provided[request_field]

        if len(updates) == 1:
            # No meaningful changes requested; return existing doc unchanged.
            return IdPM2MClient(**existing)

        await self._collection.update_one(
            {"client_id": client_id},
            {"$set": updates},
        )

        logger.info(
            "Updated manual M2M client client_id=%s fields=%s",
            client_id,
            sorted(updates.keys()),
        )
        return await self.get(client_id)

    async def delete(
        self,
        client_id: str,
    ) -> None:
        """Delete a manual M2M client record.

        Only records with ``provider == "manual"`` can be deleted.

        Raises:
            M2MClientNotFound: If the record does not exist.
            M2MClientImmutable: If the record was written by IdP sync.
        """
        existing = await self._collection.find_one({"client_id": client_id})
        if existing is None:
            raise M2MClientNotFound(client_id)
        if existing.get("provider") != MANUAL_PROVIDER:
            raise M2MClientImmutable(client_id)

        await self._collection.delete_one({"client_id": client_id})
        logger.info("Deleted manual M2M client client_id=%s", client_id)
