"""
Service layer for custom entity types.

Orchestrates validation, persistence, and search indexing for both type
descriptors and records — mirroring ``skill_service``'s save-then-index
pattern (indexing failure is logged, never fatal to the write).
"""

import logging
from typing import Any

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
from ..schemas.proxy_mixin import (
    clear_upstream_headers_on_repoint,
    validate_and_pin_proxy_target,
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

        # Encrypt static upstream auth headers for the proxy hop (plaintext
        # {name, value} -> {name, value_encrypted}); never persisted in plaintext.
        custom_headers_encrypted = None
        custom_header_names: list[str] = []
        custom_header_overridable_names: list[str] = []
        custom_headers_updated_at = None
        if getattr(request, "custom_headers", None):
            from ..utils.credential_encryption import (
                encrypt_custom_headers_in_server_dict,
                validate_custom_headers,
            )

            # Same header policy as the MCP-server route (reserved-name block +
            # count cap, + the per-header overridable rules): a custom entity must
            # not register a gateway-managed header for injection. Surface a
            # validation error as 400, not 500. Both validate and encrypt live under
            # the boundary so their ValueError (bad header, or SECRET_KEY missing)
            # maps to 4xx consistently with the skill create path.
            try:
                validate_custom_headers(request.custom_headers)
                _ch: dict[str, Any] = {"custom_headers": request.custom_headers}
                encrypt_custom_headers_in_server_dict(_ch)
            except ValueError as e:
                raise CustomEntityValidationError("custom_headers", str(e)) from e
            custom_headers_encrypted = _ch.get("custom_headers_encrypted")
            custom_header_names = _ch.get("custom_header_names", [])
            custom_header_overridable_names = _ch.get("custom_header_overridable_names", [])
            custom_headers_updated_at = _ch.get("custom_headers_updated_at")

        record = CustomEntityRecord(
            entity_type=type_name,
            name=request.name,
            description=request.description,
            owner=owner,  # server-derived, NEVER from client
            visibility=request.visibility,
            allowed_groups=request.allowed_groups,
            tags=request.tags,
            attributes=cleaned,
            # Gateway-proxy opt-in (validated on the request model; carried through).
            is_proxied=request.is_proxied,
            proxy_target_url=request.proxy_target_url,
            proxy_streaming=getattr(request, "proxy_streaming", False),
            custom_headers_encrypted=custom_headers_encrypted,
            custom_header_names=custom_header_names,
            custom_header_overridable_names=custom_header_overridable_names,
            custom_headers_updated_at=custom_headers_updated_at,
        )
        record.assign_path()
        # Gateway-proxy SSRF layer 2: when proxied, resolve the target hostname and
        # validate every resolved IP against the egress policy, then pin the IPs.
        # Rejects a metadata/private target at registration. Custom records carry
        # their own type token as the entity_type.
        if record.is_proxied:
            pin = await validate_and_pin_proxy_target(
                record.entity_type,
                {
                    "is_proxied": record.is_proxied,
                    "proxy_target_url": record.proxy_target_url,
                    "proxy_disabled_reason": record.proxy_disabled_reason,
                },
            )
            if pin:
                record.proxy_resolved_ips = pin["proxy_resolved_ips"]
                record.proxy_target_host = pin["proxy_target_host"]
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
        if request.is_proxied is not None:
            updates["is_proxied"] = request.is_proxied
        if request.proxy_target_url is not None:
            updates["proxy_target_url"] = request.proxy_target_url
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

        # Proxy target-required invariant on the MERGED state (a PUT may set
        # is_proxied without a target, or clear the target while proxied). Must
        # run BEFORE persist: the repo write-then-reconstruct would otherwise
        # store an invalid state, then CustomEntityRecord(**doc) raises on every
        # subsequent read and the record vanishes from listings.
        merged_is_proxied = updates.get("is_proxied", existing.is_proxied)
        merged_target = updates.get("proxy_target_url", existing.proxy_target_url)
        if merged_is_proxied and not merged_target and not existing.proxy_disabled_reason:
            raise CustomEntityValidationError(
                "proxy_target_url",
                "is_proxied=true requires a proxy_target_url for a custom entity",
            )

        # Gateway-proxy SSRF layer 2: when this update touches the proxy opt-in or
        # target, resolve+validate the MERGED target and refresh the pin bookkeeping
        # (or clear it when no longer a live proxy). Runs BEFORE persist so a
        # metadata/private target is rejected at update time. Re-enabling clears any
        # prior auto-disable.
        if "is_proxied" in updates or "proxy_target_url" in updates:
            pin = await validate_and_pin_proxy_target(
                existing.entity_type,
                {
                    "is_proxied": merged_is_proxied,
                    "proxy_target_url": merged_target,
                    "proxy_disabled_reason": None,
                },
            )
            updates["proxy_resolved_ips"] = pin.get("proxy_resolved_ips", [])
            updates["proxy_target_host"] = pin.get("proxy_target_host")
            updates["proxy_disabled_reason"] = None
            # Credential-misdirection guard: clear create-time upstream headers if
            # the effective target host changed (don't send the old host's secret
            # to a new host). See clear_upstream_headers_on_repoint.
            clear_upstream_headers_on_repoint(
                updates,
                existing_target=existing.proxy_target_url,
                new_target=str(merged_target) if merged_target else None,
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

    async def update_record_upstream_headers(
        self,
        type_name: str,
        path: str,
        custom_headers: list[dict],
        user_context: dict,
    ) -> CustomEntityRecord:
        """Rotate a record's upstream custom headers (proxy-hop credentials).

        Dedicated, narrowly-scoped mutation (the mirror of the skill rotation
        endpoint / the MCP-server auth-credential PATCH): validate + encrypt the
        full header set and persist ONLY the four header storage fields. An empty
        list clears all headers. Same authz as ``update_record`` (viewable +
        owner-or-admin). Raises CustomEntityValidationError (-> 400) on any policy
        violation so the plaintext-secret path never yields a 500.

        Note: rotation does NOT re-run the proxy-target repoint guard -- it only
        touches header fields, not the target, so there is no host change to
        misdirect against. The headers are always re-scoped to the CURRENT target.
        """
        existing = await self._get_entities().get(path)
        if existing is None or not _user_can_view(existing, user_context):
            raise CustomEntityNotFoundError(path)  # 404 — don't disclose existence
        _require_owner_or_admin(existing, user_context)  # 403 if viewable-but-not-owned

        from ..utils.credential_encryption import build_custom_headers_storage_fields

        try:
            updates = build_custom_headers_storage_fields(custom_headers)
        except ValueError as e:
            raise CustomEntityValidationError("custom_headers", str(e)) from e

        updated = await self._get_entities().update(path, updates)
        if updated is None:
            raise CustomEntityNotFoundError(path)  # concurrent delete
        try:
            descriptor = await self.cache.get_for_write(type_name)
            if descriptor is not None:
                await self._get_search().index_custom_entity(record=updated, descriptor=descriptor)
        except Exception:
            logger.exception(
                "Failed to re-index custom entity %s (header rotation persisted)", path
            )
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

        updated_details, _ = update_rating_details(list(existing.rating_details), username, rating)
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
        restrict_paths: list[str] | None = None,
    ) -> tuple[list[CustomEntityRecord], int]:
        """List records of a type with in-query visibility filtering.

        Returns (page_slice, total_count) where total_count is the number of
        records THIS user can see — both use the SAME filter so the count
        matches the slice.

        The discovery grant and the per-record visibility filter compose here:

        - ``restrict_paths=None`` — whole-type access (grant is ``"all"``/type
          name, or admin). Only the per-record visibility filter applies.
        - ``restrict_paths=[...]`` — the caller holds only specific records of
          this type (record-scoped ``list_<type>_entity``). The result is
          additionally constrained to those record paths, intersected with the
          visibility filter. An empty list yields no records.
        """
        # Read path: a list page tolerates brief cross-replica staleness, so
        # use the TTL cache rather than an authoritative Mongo read per page.
        descriptor = await self.cache.get_for_read(type_name)
        if descriptor is None:
            raise UnknownCustomTypeError(type_name)
        visibility_filter = _build_visibility_filter(user_context)

        if restrict_paths is not None:
            # Intersect the granted record paths with the visibility filter.
            # `_id` is the synthetic record path. Combine with $and so neither
            # clause clobbers the other (visibility_filter is None for admins,
            # but admins never pass restrict_paths, so this branch is non-admin).
            path_clause: dict = {"_id": {"$in": restrict_paths}}
            if visibility_filter:
                combined_filter: dict | None = {"$and": [path_clause, visibility_filter]}
            else:
                combined_filter = path_clause
        else:
            combined_filter = visibility_filter

        entities = self._get_entities()
        items = await entities.list_paginated(type_name, skip, limit, combined_filter)
        total = await entities.count(type_name, combined_filter)
        return items, total
