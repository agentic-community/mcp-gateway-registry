"""Direct CRUD service for user-to-group fallback records.

This service writes to the shared ``idp_user_groups`` collection without
calling any IdP Admin API. Records written by this service are tagged with
``provider == "manual"`` so they are distinguishable from IdP-synced records.

This service is the user-side mirror of :mod:`registry.services.m2m_management_service`
and is used as the authorization fallback when the JWT's groups claim is empty
for an IdP listed in ``IDP_USER_GROUP_FALLBACK_ENABLED_PROVIDERS`` (e.g.
PingFederate today).

Tracked by issue #1127.
"""

import logging
from datetime import datetime

from motor.motor_asyncio import AsyncIOMotorDatabase
from pymongo.errors import DuplicateKeyError

from registry.repositories.documentdb.client import get_documentdb_client
from registry.schemas.idp_user_group import (
    MANUAL_PROVIDER,
    IdPUserGroup,
    IdPUserGroupCreate,
    IdPUserGroupPatch,
)

logger = logging.getLogger(__name__)


COLLECTION_NAME: str = "idp_user_groups"


class UserGroupConflict(Exception):
    """Raised when a username already exists in the collection."""


class UserGroupNotFound(Exception):
    """Raised when the requested username does not exist."""


class UserGroupManagementService:
    """CRUD service for manually-registered user-to-group fallback records."""

    def __init__(
        self,
        db: AsyncIOMotorDatabase,
    ) -> None:
        self._collection = db[COLLECTION_NAME]

    async def ensure_indexes(self) -> None:
        """Create required indexes (idempotent).

        Creates a unique index on ``username`` to prevent duplicate registrations
        under concurrent POSTs.
        """
        await self._collection.create_index("username", unique=True)
        logger.info(
            "Ensured unique index on %s.username",
            COLLECTION_NAME,
        )

    async def register_user_group(
        self,
        payload: IdPUserGroupCreate,
        created_by: str | None,
    ) -> IdPUserGroup:
        """Insert a new manual user-group fallback record.

        Args:
            payload: Validated create request body.
            created_by: Username of the operator performing the action
                (captured from the authenticated user context).

        Returns:
            The persisted :class:`IdPUserGroup`.

        Raises:
            UserGroupConflict: If ``username`` already exists (unique index
                violation).
        """
        now = datetime.utcnow()
        doc: dict = {
            "username": payload.username,
            "groups": list(payload.groups),
            "enabled": True,
            "provider": MANUAL_PROVIDER,
            "email": payload.email,
            "created_by": created_by,
            "created_at": now,
            "updated_at": now,
        }
        try:
            await self._collection.insert_one(doc)
        except DuplicateKeyError as e:
            raise UserGroupConflict(payload.username) from e

        logger.info(
            "Registered manual user-group username=%s groups=%s created_by=%s",
            payload.username,
            payload.groups,
            created_by,
        )
        return IdPUserGroup(**doc)

    async def list_user_groups(
        self,
        skip: int = 0,
        limit: int = 500,
        provider: str | None = None,
        q: str | None = None,
    ) -> tuple[list[IdPUserGroup], int]:
        """Return a paginated slice of the collection.

        Args:
            skip: Number of records to skip (offset).
            limit: Maximum number of records to return on this page.
            provider: Optional filter on the ``provider`` field.
            q: Optional case-insensitive substring filter applied to
                ``username`` and ``email``.

        Returns:
            Tuple of (items_on_page, total_matching_count).
        """
        query: dict = {}
        if provider is not None:
            query["provider"] = provider
        if q:
            # Case-insensitive substring match against username/email. Anchored
            # to literal text via re.escape to avoid letting callers inject
            # regex metacharacters.
            import re as _re

            pattern = _re.escape(q)
            query["$or"] = [
                {"username": {"$regex": pattern, "$options": "i"}},
                {"email": {"$regex": pattern, "$options": "i"}},
            ]

        total = await self._collection.count_documents(query)
        cursor = self._collection.find(query).skip(skip).limit(limit)
        docs = await cursor.to_list(length=limit)
        return [IdPUserGroup(**d) for d in docs], total

    async def get_user_group(
        self,
        username: str,
    ) -> IdPUserGroup:
        """Fetch a single record by ``username``.

        Raises:
            UserGroupNotFound: If the record does not exist.
        """
        doc = await self._collection.find_one({"username": username})
        if doc is None:
            raise UserGroupNotFound(username)
        return IdPUserGroup(**doc)

    async def update_user_group(
        self,
        username: str,
        payload: IdPUserGroupPatch,
    ) -> IdPUserGroup:
        """Update fields on an existing user-group record.

        Raises:
            UserGroupNotFound: If the record does not exist.
        """
        existing = await self._collection.find_one({"username": username})
        if existing is None:
            raise UserGroupNotFound(username)

        # Pydantic v2: only fields explicitly set in the request body appear
        # in the dump. Callers clearing groups pass [] and it lands here too.
        provided = payload.model_dump(exclude_unset=True)

        field_map: dict[str, str] = {
            "groups": "groups",
            "email": "email",
            "enabled": "enabled",
        }
        updates: dict = {"updated_at": datetime.utcnow()}
        for request_field, storage_field in field_map.items():
            if request_field in provided:
                updates[storage_field] = provided[request_field]

        if len(updates) == 1:
            # No meaningful changes requested; return existing doc unchanged.
            return IdPUserGroup(**existing)

        await self._collection.update_one(
            {"username": username},
            {"$set": updates},
        )

        logger.info(
            "Updated manual user-group username=%s fields=%s",
            username,
            sorted(updates.keys()),
        )
        return await self.get_user_group(username)

    async def delete_user_group(
        self,
        username: str,
    ) -> None:
        """Delete a user-group record.

        Raises:
            UserGroupNotFound: If the record does not exist.
        """
        existing = await self._collection.find_one({"username": username})
        if existing is None:
            raise UserGroupNotFound(username)

        await self._collection.delete_one({"username": username})
        logger.info("Deleted manual user-group username=%s", username)

    async def get_user_groups_for_username(
        self,
        username: str,
    ) -> list[str]:
        """Return the groups associated with a username, or [] if no record.

        This is the read path used by the auth server to enrich a user context
        when the JWT's groups claim is empty for a fallback-enabled provider.

        Disabled records are treated as if they do not exist (returns []) so
        operators can revoke fallback memberships without deleting history.
        """
        doc = await self._collection.find_one({"username": username})
        if doc is None:
            return []
        if not doc.get("enabled", True):
            return []
        groups = doc.get("groups") or []
        return list(groups)


_singleton: UserGroupManagementService | None = None


async def get_user_group_management_service() -> UserGroupManagementService:
    """Module-level singleton getter for the user-group service.

    Mirrors the M2M getter pattern so callers in routes / auth code share a
    single service instance bound to the active DocumentDB client.
    """
    global _singleton
    if _singleton is None:
        db = await get_documentdb_client()
        _singleton = UserGroupManagementService(db)
    return _singleton
