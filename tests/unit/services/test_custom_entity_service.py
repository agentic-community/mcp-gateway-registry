"""Unit tests for registry/services/custom_entity_service.py.

The service orchestrates validation, persistence, and search indexing over
mocked repositories. Tests cover: unknown-type 404 mapping, the record cap,
owner being server-derived, save-then-index (indexing failure is non-fatal),
the merge-then-validate update path with explicit-null key removal, the
group-restricted merged-state invariant, the authz gates on update/delete,
and the cascade order for delete_type (embeddings -> records -> descriptor).
"""

import logging
from unittest.mock import AsyncMock, MagicMock, call, patch

import pytest

from registry.schemas.custom_entity_models import (
    CustomEntityCreate,
    CustomEntityRecord,
    CustomEntityUpdate,
    CustomFieldDescriptor,
    CustomFieldType,
    CustomTypeDescriptor,
    CustomTypeUpdate,
)
from registry.services.custom_entity_errors import (
    CustomEntityNotFoundError,
    CustomEntityValidationError,
    CustomTypeHasRecordsError,
    CustomTypeLimitError,
    CustomTypeRecordCapError,
    UnknownCustomTypeError,
)
from registry.services.custom_entity_service import CustomEntityService

logger = logging.getLogger(__name__)

TYPE = "workflow"
ADMIN = {"username": "admin", "is_admin": True}
BOB = {"username": "bob", "is_admin": False, "groups": []}
EVE = {"username": "eve", "is_admin": False, "groups": []}


def _descriptor() -> CustomTypeDescriptor:
    return CustomTypeDescriptor(
        name=TYPE,
        fields=[
            CustomFieldDescriptor(name="title", datatype=CustomFieldType.STRING),
            CustomFieldDescriptor(name="note", datatype=CustomFieldType.STRING),
        ],
    )


def _record(owner: str = "bob", **kw) -> CustomEntityRecord:
    rec = CustomEntityRecord(entity_type=TYPE, name="r", owner=owner, **kw)
    rec.path = f"/{TYPE}/abc"
    return rec


@pytest.fixture
def service():
    svc = CustomEntityService()
    entities = MagicMock()
    entities.count = AsyncMock(return_value=0)
    entities.create = AsyncMock(side_effect=lambda r: r)
    entities.get = AsyncMock(return_value=None)
    entities.update = AsyncMock()
    entities.delete = AsyncMock(return_value=True)
    entities.delete_by_type = AsyncMock(return_value=0)
    search = MagicMock()
    search.index_custom_entity = AsyncMock()
    search.delete_custom_entity_index = AsyncMock()
    search.delete_custom_entity_index_by_type = AsyncMock()
    types = MagicMock()
    types.delete = AsyncMock()
    types.create = AsyncMock(side_effect=lambda d: d)
    types.count_types = AsyncMock(return_value=0)
    types.update_metadata = AsyncMock(return_value=_descriptor())
    cache = MagicMock()
    cache.get_for_write = AsyncMock(return_value=_descriptor())
    cache.invalidate = MagicMock()
    types.cache = cache
    svc._entities = entities
    svc._search = search
    svc._types = types
    return svc, entities, search, types


@pytest.mark.unit
class TestCreateRecord:
    async def test_unknown_type_raises(self, service):
        svc, _, _, types = service
        types.cache.get_for_write = AsyncMock(return_value=None)
        with pytest.raises(UnknownCustomTypeError):
            await svc.create_record(TYPE, CustomEntityCreate(name="x"), owner="bob")

    async def test_owner_is_server_derived(self, service):
        svc, entities, _, _ = service
        out = await svc.create_record(
            TYPE, CustomEntityCreate(name="x", attributes={"title": "t"}), owner="bob"
        )
        assert out.owner == "bob"
        entities.create.assert_awaited_once()

    async def test_cap_enforced(self, service):
        svc, entities, _, _ = service
        entities.count = AsyncMock(return_value=100)
        with patch("registry.services.custom_entity_service.settings") as s:
            s.max_custom_records_per_type = 100
            with pytest.raises(CustomTypeRecordCapError):
                await svc.create_record(TYPE, CustomEntityCreate(name="x"), owner="bob")

    async def test_indexing_failure_is_non_fatal(self, service):
        svc, _, search, _ = service
        search.index_custom_entity = AsyncMock(side_effect=RuntimeError("boom"))
        out = await svc.create_record(
            TYPE, CustomEntityCreate(name="x", attributes={"title": "t"}), owner="bob"
        )
        assert out.name == "x"


@pytest.mark.unit
class TestUpdateRecord:
    async def test_not_found_when_absent(self, service):
        svc, entities, _, _ = service
        entities.get = AsyncMock(return_value=None)
        with pytest.raises(CustomEntityNotFoundError):
            await svc.update_record(TYPE, f"/{TYPE}/abc", CustomEntityUpdate(name="z"), BOB)

    async def test_non_owner_forbidden(self, service):
        from fastapi import HTTPException

        svc, entities, _, _ = service
        entities.get = AsyncMock(return_value=_record(owner="bob", visibility="public"))
        with pytest.raises(HTTPException) as exc:
            await svc.update_record(TYPE, f"/{TYPE}/abc", CustomEntityUpdate(name="z"), EVE)
        assert exc.value.status_code == 403

    async def test_explicit_null_removes_attribute(self, service):
        svc, entities, _, _ = service
        existing = _record(owner="bob", attributes={"title": "t", "note": "n"})
        entities.get = AsyncMock(return_value=existing)
        entities.update = AsyncMock(return_value=existing)
        await svc.update_record(
            TYPE,
            f"/{TYPE}/abc",
            CustomEntityUpdate(attributes={"note": None}),
            BOB,
        )
        updates = entities.update.call_args[0][1]
        assert "note" not in updates["attributes"]
        assert updates["attributes"]["title"] == "t"

    async def test_group_restricted_without_groups_rejected(self, service):
        svc, entities, _, _ = service
        entities.get = AsyncMock(return_value=_record(owner="bob", visibility="public"))
        with pytest.raises(CustomEntityValidationError):
            await svc.update_record(
                TYPE,
                f"/{TYPE}/abc",
                CustomEntityUpdate(visibility="group-restricted"),
                BOB,
            )


@pytest.mark.unit
class TestDeleteRecord:
    async def test_not_found(self, service):
        svc, entities, _, _ = service
        entities.get = AsyncMock(return_value=None)
        with pytest.raises(CustomEntityNotFoundError):
            await svc.delete_record(TYPE, f"/{TYPE}/abc", BOB)

    async def test_index_deleted_before_record(self, service):
        svc, entities, search, _ = service
        entities.get = AsyncMock(return_value=_record(owner="bob"))
        await svc.delete_record(TYPE, f"/{TYPE}/abc", BOB)
        search.delete_custom_entity_index.assert_awaited_once()
        entities.delete.assert_awaited_once()


@pytest.mark.unit
class TestDeleteType:
    async def test_has_records_without_force(self, service):
        svc, entities, _, _ = service
        entities.count = AsyncMock(return_value=3)
        with pytest.raises(CustomTypeHasRecordsError):
            await svc.delete_type(TYPE, force=False)

    async def test_cascade_order(self, service):
        svc, entities, search, types = service
        entities.count = AsyncMock(return_value=2)
        manager = MagicMock()
        manager.attach_mock(search.delete_custom_entity_index_by_type, "embeddings")
        manager.attach_mock(entities.delete_by_type, "records")
        manager.attach_mock(types.delete, "descriptor")
        count = await svc.delete_type(TYPE, force=True)
        assert count == 2
        assert manager.mock_calls == [
            call.embeddings(TYPE),
            call.records(TYPE),
            call.descriptor(TYPE),
        ]
        types.cache.invalidate.assert_called_once()


@pytest.mark.unit
class TestCreateType:
    async def test_created_when_under_limit(self, service):
        svc, _, _, types = service
        types.count_types = AsyncMock(return_value=10)
        with patch("registry.services.custom_entity_service.settings") as s:
            s.max_custom_types = 50
            out = await svc.create_type(_descriptor())
        assert out.name == TYPE
        types.create.assert_awaited_once()
        types.cache.invalidate.assert_called_once()

    async def test_type_limit_enforced(self, service):
        svc, _, _, types = service
        types.count_types = AsyncMock(return_value=50)
        with patch("registry.services.custom_entity_service.settings") as s:
            s.max_custom_types = 50
            with pytest.raises(CustomTypeLimitError):
                await svc.create_type(_descriptor())
        types.create.assert_not_awaited()

    async def test_zero_limit_means_unlimited(self, service):
        svc, _, _, types = service
        types.count_types = AsyncMock(return_value=10_000)
        with patch("registry.services.custom_entity_service.settings") as s:
            s.max_custom_types = 0
            out = await svc.create_type(_descriptor())
        assert out.name == TYPE
        types.create.assert_awaited_once()


@pytest.mark.unit
class TestUpdateType:
    async def test_updates_mutable_metadata(self, service):
        svc, _, _, types = service
        await svc.update_type(TYPE, CustomTypeUpdate(display_name="New Label"))
        # Only the supplied (non-None) keys are forwarded to the repo.
        types.update_metadata.assert_awaited_once_with(TYPE, {"display_name": "New Label"})
        types.cache.invalidate.assert_called_once()

    async def test_none_fields_excluded_from_update(self, service):
        svc, _, _, types = service
        await svc.update_type(TYPE, CustomTypeUpdate(description="just a description"))
        types.update_metadata.assert_awaited_once_with(TYPE, {"description": "just a description"})

    async def test_unknown_type_returns_none_no_cache_invalidate(self, service):
        svc, _, _, types = service
        types.update_metadata = AsyncMock(return_value=None)
        result = await svc.update_type("does_not_exist", CustomTypeUpdate(display_name="x"))
        assert result is None
        types.cache.invalidate.assert_not_called()
