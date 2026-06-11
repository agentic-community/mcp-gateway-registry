"""
Service layer for custom entity types.

Orchestrates validation, persistence, and search indexing for both type
descriptors and records — mirroring ``skill_service``'s save-then-index
pattern (indexing failure is logged, never fatal to the write).
"""

import logging

from ..api.custom_entity_visibility import (
    _build_visibility_filter,
    _require_owner_or_admin,
    _user_can_view,
)
from ..core.config import settings
from ..repositories.factory import (
    get_custom_entity_repository,
    get_custom_type_repository,
    get_search_repository,
)
from ..repositories.interfaces import (
    CustomEntityRepositoryBase,
    CustomTypeRepositoryBase,
    SearchRepositoryBase,
)
from ..schemas.custom_entity_models import (
    CustomEntityCreate,
    CustomEntityRecord,
    CustomEntityUpdate,
    CustomTypeDescriptor,
    CustomTypeUpdate,
)
from .custom_entity_errors import (
    CustomEntityNotFoundError,
    CustomEntityValidationError,
    CustomTypeHasRecordsError,
    CustomTypeLimitError,
    CustomTypeRecordCapError,
    UnknownCustomTypeError,
)
from .custom_entity_validator import validate_attributes
from .rating_service import (
    calculate_average_rating,
    update_rating_details,
    validate_rating,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s,p%(process)s,{%(filename)s:%(lineno)d},%(levelname)s,%(message)s",
)
logger = logging.getLogger(__name__)


class CustomEntityService:
    """Validate + persist + index custom type descriptors and records."""

    def __init__(self):
        self._types: CustomTypeRepositoryBase | None = None
        self._entities: CustomEntityRepositoryBase | None = None
        self._search: SearchRepositoryBase | None = None

    def _get_types(self) -> CustomTypeRepositoryBase:
        """Lazy init of the type-descriptor repository."""
        if self._types is None:
            self._types = get_custom_type_repository()
        return self._types

    def _get_entities(self) -> CustomEntityRepositoryBase:
        """Lazy init of the record repository."""
        if self._entities is None:
            self._entities = get_custom_entity_repository()
        return self._entities

    def _get_search(self) -> SearchRepositoryBase:
        """Lazy init of the search repository."""
        if self._search is None:
            self._search = get_search_repository()
        return self._search

    @property
    def cache(self):
        """The shared custom-type descriptor cache (owned by the type repo)."""
        return self._get_types().cache

    # --- Type descriptor operations -------------------------------------

    async def create_type(
        self,
        descriptor: CustomTypeDescriptor,
    ) -> CustomTypeDescriptor:
        """Persist a new type descriptor and invalidate the cache.

        Enforces the MAX_CUSTOM_TYPES limit (0 = unlimited) before creating.
        Like the record cap, this count-then-create check is best-effort: two
        concurrent type creates could both pass at the boundary. Acceptable
        because type creation is admin-only and low-frequency.
        """
        type_limit = settings.max_custom_types
        if type_limit > 0:
            current_types = await self._get_types().count_types()
            if current_types >= type_limit:
                raise CustomTypeLimitError(type_limit)

        created = await self._get_types().create(descriptor)
        self.cache.invalidate()
        logger.info(f"Defined custom type {created.name} ({len(created.fields)} fields)")
        return created

    async def update_type(
        self,
        type_name: str,
        request: CustomTypeUpdate,
    ) -> CustomTypeDescriptor | None:
        """Update a type's mutable metadata (display_name/description).

        The type name and field schema are immutable; only display_name and
        description can change. A field left None in the request is not touched.
        Returns the updated descriptor, or None if the type does not exist.
        Invalidates the descriptor cache so the new label propagates.
        """
        updates = request.model_dump(exclude_none=True)
        updated = await self._get_types().update_metadata(type_name, updates)
        if updated is not None:
            self.cache.invalidate()
            logger.info(f"Updated custom type {type_name} metadata: {list(updates.keys())}")
        return updated

    async def delete_type(
        self,
        type_name: str,
        force: bool,
    ) -> int:
        """Cascade-delete a type: embeddings -> records -> descriptor.

        Returns the number of records deleted. Raises CustomTypeHasRecordsError
        (409) when records exist and force is False.
        """
        entities = self._get_entities()
        count = await entities.count(type_name)
        if count > 0 and not force:
            raise CustomTypeHasRecordsError(type_name, count)
        # Order: embeddings -> records -> descriptor. If a mid-step fails the
        # descriptor still exists, so the type stays coherent and retryable.
        await self._get_search().delete_custom_entity_index_by_type(type_name)
        await entities.delete_by_type(type_name)
        await self._get_types().delete(type_name)
        self.cache.invalidate()
        logger.info(f"Deleted custom type {type_name} (cascaded {count} records)")
        return count

    async def list_types(self) -> list[CustomTypeDescriptor]:
        """Return all defined type descriptors (cache-backed, hot path)."""
        return await self.cache.list_descriptors()

    async def get_type(
        self,
        type_name: str,
    ) -> CustomTypeDescriptor | None:
        """Return a single type descriptor (cache-backed read, hot path).

        Uses the TTL cache: this is a read-only descriptor lookup (GET
        /custom-types/{name}, list-page rendering), where a brief
        cross-replica staleness window is acceptable. Write paths
        (create/update/delete record) use get_for_write for an authoritative
        read instead.
        """
        return await self.cache.get_for_read(type_name)

    # --- Record operations ----------------------------------------------

    async def create_record(
        self,
        type_name: str,
        request: CustomEntityCreate,
        owner: str,
    ) -> CustomEntityRecord:
        """Validate attributes, persist a record, then index it (non-fatal)."""
        descriptor = await self.cache.get_for_write(type_name)  # authoritative
        if descriptor is None:
            raise UnknownCustomTypeError(type_name)

        # Soft cap (0 = unlimited). This count-then-create check is best-effort:
        # concurrent creates can both read a count below the cap and both insert,
        # so the type may overshoot the cap slightly. It is NOT a hard
        # transactional guarantee -- it exists to stop runaway imports, not to
        # enforce an exact ceiling. Making it hard would need an atomic counter.
        cap = settings.max_custom_records_per_type
        if cap > 0:
            current_count = await self._get_entities().count(type_name)
            if current_count >= cap:
                raise CustomTypeRecordCapError(type_name, cap)

        cleaned = validate_attributes(descriptor, request.attributes)
        record = CustomEntityRecord(
            entity_type=type_name,
            name=request.name,
            description=request.description,
            owner=owner,  # server-derived, NEVER from client
            visibility=request.visibility,
            allowed_groups=request.allowed_groups,
            tags=request.tags,
            attributes=cleaned,
        )
        record.assign_path()
        created = await self._get_entities().create(record)
        try:
            await self._get_search().index_custom_entity(record=created, descriptor=descriptor)
        except Exception:
            logger.exception("Failed to index custom entity %s (record persisted)", created.path)
        return created

    async def update_record(
        self,
        type_name: str,
        path: str,
        request: CustomEntityUpdate,
        user_context: dict,
    ) -> CustomEntityRecord:
        """Authorize, merge+validate, persist, then re-index a record."""
        descriptor = await self.cache.get_for_write(type_name)  # authoritative
        if descriptor is None:
            raise UnknownCustomTypeError(type_name)

        existing = await self._get_entities().get(path)
        if existing is None or not _user_can_view(existing, user_context):
            raise CustomEntityNotFoundError(path)  # 404 — don't disclose existence
        _require_owner_or_admin(existing, user_context)  # 403 if viewable-but-not-owned

        updates: dict[str, object] = {}
        if request.name is not None:
            updates["name"] = request.name
        if request.description is not None:
            updates["description"] = request.description
        if request.visibility is not None:
            updates["visibility"] = request.visibility
        if request.allowed_groups is not None:
            updates["allowed_groups"] = request.allowed_groups
        if request.tags is not None:
            updates["tags"] = request.tags
        if request.attributes is not None:
            # Merge-then-validate: merge client patch into stored bag, then validate.
            merged_attrs = dict(existing.attributes)
            for key, val in request.attributes.items():
                if val is None:
                    merged_attrs.pop(key, None)  # explicit null = remove key
                else:
                    merged_attrs[key] = val  # overwrite / add
            updates["attributes"] = validate_attributes(descriptor, merged_attrs)

        # group-restricted invariant on the MERGED state (a PUT may set one side only).
        merged_visibility = updates.get("visibility", existing.visibility)
        merged_groups = updates.get("allowed_groups", existing.allowed_groups)
        if merged_visibility == "group-restricted" and not merged_groups:
            raise CustomEntityValidationError(
                "allowed_groups",
                "group-restricted visibility requires at least one allowed_group",
            )

        updated = await self._get_entities().update(path, updates)
        if updated is None:
            # Row vanished between the authz read and the update (concurrent delete).
            raise CustomEntityNotFoundError(path)
        try:
            await self._get_search().index_custom_entity(record=updated, descriptor=descriptor)
        except Exception:
            logger.exception("Failed to re-index custom entity %s (update persisted)", path)
        return updated

    async def delete_record(
        self,
        type_name: str,
        path: str,
        user_context: dict,
    ) -> None:
        """Authorize, delete the embedding, then delete the record."""
        existing = await self._get_entities().get(path)
        if existing is None or not _user_can_view(existing, user_context):
            raise CustomEntityNotFoundError(path)  # 404 — don't disclose existence
        _require_owner_or_admin(existing, user_context)  # 403 if viewable-but-not-owned
        await self._get_search().delete_custom_entity_index(path)
        deleted = await self._get_entities().delete(path)
        if not deleted:
            raise CustomEntityNotFoundError(path)
        logger.info(f"Deleted custom record {path} ({type_name})")

    async def get_record(
        self,
        path: str,
        user_context: dict,
    ) -> CustomEntityRecord:
        """Fetch a single record, enforcing visibility (404 if not viewable)."""
        record = await self._get_entities().get(path)
        if record is None or not _user_can_view(record, user_context):
            raise CustomEntityNotFoundError(path)
        return record

    async def update_rating(
        self,
        path: str,
        username: str,
        rating: int,
        user_context: dict,
    ) -> float:
        """Add/update a user's rating on a record; returns the new average.

        Any user who can VIEW the record may rate it (mirrors servers/agents).
        Raises CustomEntityNotFoundError (404) if the record is missing or not
        viewable, ValueError if the rating is out of range.
        """
        validate_rating(rating)
        existing = await self._get_entities().get(path)
        if existing is None or not _user_can_view(existing, user_context):
            raise CustomEntityNotFoundError(path)  # 404 — don't disclose existence

        updated_details, _ = update_rating_details(
            list(existing.rating_details), username, rating
        )
        average = calculate_average_rating(updated_details)
        updated = await self._get_entities().update(
            path,
            {"rating_details": updated_details, "num_stars": average},
        )
        if updated is None:
            raise CustomEntityNotFoundError(path)  # concurrent delete
        logger.info(f"Rated custom record {path}: {rating} by {username} (avg {average:.2f})")
        return average

    async def get_rating(
        self,
        path: str,
        user_context: dict,
    ) -> dict:
        """Return {num_stars, rating_details} for a viewable record (404 otherwise)."""
        record = await self._get_entities().get(path)
        if record is None or not _user_can_view(record, user_context):
            raise CustomEntityNotFoundError(path)
        return {
            "num_stars": record.num_stars,
            "rating_details": record.rating_details,
        }

    async def list_records(
        self,
        type_name: str,
        skip: int,
        limit: int,
        user_context: dict,
    ) -> tuple[list[CustomEntityRecord], int]:
        """List records of a type with in-query visibility filtering.

        Returns (page_slice, total_count) where total_count is the number of
        records THIS user can see — both use the SAME filter so the count
        matches the slice.
        """
        # Read path: a list page tolerates brief cross-replica staleness, so
        # use the TTL cache rather than an authoritative Mongo read per page.
        descriptor = await self.cache.get_for_read(type_name)
        if descriptor is None:
            raise UnknownCustomTypeError(type_name)
        visibility_filter = _build_visibility_filter(user_context)
        entities = self._get_entities()
        items = await entities.list_paginated(type_name, skip, limit, visibility_filter)
        total = await entities.count(type_name, visibility_filter)
        return items, total
