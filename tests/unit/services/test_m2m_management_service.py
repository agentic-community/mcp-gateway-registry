"""Unit tests for registry.services.m2m_management_service.

These tests mock the Motor collection so the service logic can be exercised
without a live MongoDB.
"""

import logging
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock

import pytest
from pymongo.errors import DuplicateKeyError

from registry.schemas.idp_m2m_client import (
    MANUAL_PROVIDER,
    IdPM2MClientCreate,
    IdPM2MClientPatch,
)
from registry.services.m2m_management_service import (
    COLLECTION_NAME,
    M2MClientConflict,
    M2MClientImmutable,
    M2MClientNotFound,
    M2MManagementService,
)

logger = logging.getLogger(__name__)


def _make_collection_mock() -> MagicMock:
    """Return a MagicMock that mimics an AsyncIOMotorCollection."""
    collection = MagicMock()
    collection.insert_one = AsyncMock()
    collection.find_one = AsyncMock()
    collection.update_one = AsyncMock()
    collection.delete_one = AsyncMock()
    collection.count_documents = AsyncMock()
    collection.create_index = AsyncMock()

    # find() returns a chainable cursor stub.
    cursor = MagicMock()
    cursor.skip = MagicMock(return_value=cursor)
    cursor.limit = MagicMock(return_value=cursor)
    cursor.to_list = AsyncMock()
    collection.find = MagicMock(return_value=cursor)
    collection._cursor = cursor
    return collection


@pytest.fixture
def mock_collection() -> MagicMock:
    return _make_collection_mock()


@pytest.fixture
def mock_db(mock_collection: MagicMock) -> MagicMock:
    db = MagicMock()
    db.__getitem__ = MagicMock(return_value=mock_collection)
    return db


@pytest.fixture
def service(mock_db: MagicMock) -> M2MManagementService:
    return M2MManagementService(mock_db)


@pytest.fixture
def sample_manual_doc() -> dict:
    """A manual-provider document as stored in MongoDB."""
    now = datetime.utcnow()
    return {
        "client_id": "test-client-id",
        "name": "Test Client",
        "description": "A test client",
        "groups": ["group-a"],
        "enabled": True,
        "provider": MANUAL_PROVIDER,
        "idp_app_id": None,
        "created_by": "alice",
        "created_at": now,
        "updated_at": now,
    }


@pytest.fixture
def sample_synced_doc() -> dict:
    """An IdP-synced document; must be immutable to this API."""
    now = datetime.utcnow()
    return {
        "client_id": "synced-client-id",
        "name": "Synced Client",
        "description": None,
        "groups": ["group-b"],
        "enabled": True,
        "provider": "okta",
        "idp_app_id": "0oa1100",
        "created_at": now,
        "updated_at": now,
    }


class TestEnsureIndexes:
    """Tests for ensure_indexes."""

    @pytest.mark.asyncio
    async def test_creates_unique_index_on_client_id(
        self,
        service: M2MManagementService,
        mock_collection: MagicMock,
    ) -> None:
        await service.ensure_indexes()

        mock_collection.create_index.assert_awaited_once_with("client_id", unique=True)


class TestCreate:
    """Tests for M2MManagementService.create."""

    @pytest.mark.asyncio
    async def test_inserts_document_with_manual_provider(
        self,
        service: M2MManagementService,
        mock_collection: MagicMock,
    ) -> None:
        payload = IdPM2MClientCreate(
            client_id="new-client-id",
            client_name="New Client",
            groups=["g1", "g2"],
            description="desc",
        )
        mock_collection.insert_one = AsyncMock()

        result = await service.create(payload, created_by="alice")

        mock_collection.insert_one.assert_awaited_once()
        inserted_doc = mock_collection.insert_one.await_args.args[0]
        assert inserted_doc["client_id"] == "new-client-id"
        assert inserted_doc["name"] == "New Client"
        assert inserted_doc["groups"] == ["g1", "g2"]
        assert inserted_doc["description"] == "desc"
        assert inserted_doc["provider"] == MANUAL_PROVIDER
        assert inserted_doc["created_by"] == "alice"
        assert inserted_doc["enabled"] is True
        assert inserted_doc["idp_app_id"] is None
        assert isinstance(inserted_doc["created_at"], datetime)
        assert isinstance(inserted_doc["updated_at"], datetime)
        assert result.client_id == "new-client-id"
        assert result.provider == MANUAL_PROVIDER

    @pytest.mark.asyncio
    async def test_raises_conflict_on_duplicate_key(
        self,
        service: M2MManagementService,
        mock_collection: MagicMock,
    ) -> None:
        mock_collection.insert_one = AsyncMock(side_effect=DuplicateKeyError("dup"))
        payload = IdPM2MClientCreate(
            client_id="dup-id",
            client_name="Dup",
        )

        with pytest.raises(M2MClientConflict):
            await service.create(payload, created_by=None)


class TestListPaged:
    """Tests for M2MManagementService.list_paged."""

    @pytest.mark.asyncio
    async def test_returns_items_and_total(
        self,
        service: M2MManagementService,
        mock_collection: MagicMock,
        sample_manual_doc: dict,
    ) -> None:
        mock_collection.count_documents = AsyncMock(return_value=1)
        mock_collection._cursor.to_list = AsyncMock(return_value=[sample_manual_doc])

        items, total = await service.list_paged(limit=10, skip=0)

        assert total == 1
        assert len(items) == 1
        assert items[0].client_id == "test-client-id"
        mock_collection.count_documents.assert_awaited_once_with({})

    @pytest.mark.asyncio
    async def test_filters_by_provider(
        self,
        service: M2MManagementService,
        mock_collection: MagicMock,
    ) -> None:
        mock_collection.count_documents = AsyncMock(return_value=0)
        mock_collection._cursor.to_list = AsyncMock(return_value=[])

        items, total = await service.list_paged(provider="manual", limit=10, skip=0)

        assert items == []
        assert total == 0
        mock_collection.count_documents.assert_awaited_once_with({"provider": "manual"})
        mock_collection.find.assert_called_once_with({"provider": "manual"})

    @pytest.mark.asyncio
    async def test_applies_skip_and_limit(
        self,
        service: M2MManagementService,
        mock_collection: MagicMock,
    ) -> None:
        mock_collection.count_documents = AsyncMock(return_value=0)
        mock_collection._cursor.to_list = AsyncMock(return_value=[])

        await service.list_paged(limit=25, skip=100)

        mock_collection._cursor.skip.assert_called_once_with(100)
        mock_collection._cursor.limit.assert_called_once_with(25)


class TestGet:
    """Tests for M2MManagementService.get."""

    @pytest.mark.asyncio
    async def test_returns_client_when_found(
        self,
        service: M2MManagementService,
        mock_collection: MagicMock,
        sample_manual_doc: dict,
    ) -> None:
        mock_collection.find_one = AsyncMock(return_value=sample_manual_doc)

        result = await service.get("test-client-id")

        assert result.client_id == "test-client-id"
        mock_collection.find_one.assert_awaited_once_with({"client_id": "test-client-id"})

    @pytest.mark.asyncio
    async def test_raises_not_found_when_missing(
        self,
        service: M2MManagementService,
        mock_collection: MagicMock,
    ) -> None:
        mock_collection.find_one = AsyncMock(return_value=None)

        with pytest.raises(M2MClientNotFound):
            await service.get("missing")


class TestPatch:
    """Tests for M2MManagementService.patch."""

    @pytest.mark.asyncio
    async def test_raises_not_found_when_missing(
        self,
        service: M2MManagementService,
        mock_collection: MagicMock,
    ) -> None:
        mock_collection.find_one = AsyncMock(return_value=None)

        with pytest.raises(M2MClientNotFound):
            await service.patch("missing", IdPM2MClientPatch(client_name="x"))

    @pytest.mark.asyncio
    async def test_raises_immutable_for_non_manual(
        self,
        service: M2MManagementService,
        mock_collection: MagicMock,
        sample_synced_doc: dict,
    ) -> None:
        mock_collection.find_one = AsyncMock(return_value=sample_synced_doc)

        with pytest.raises(M2MClientImmutable):
            await service.patch("synced-client-id", IdPM2MClientPatch(groups=["new-group"]))

    @pytest.mark.asyncio
    async def test_updates_only_provided_fields(
        self,
        service: M2MManagementService,
        mock_collection: MagicMock,
        sample_manual_doc: dict,
    ) -> None:
        mock_collection.find_one = AsyncMock(return_value=sample_manual_doc)

        await service.patch(
            "test-client-id",
            IdPM2MClientPatch(groups=["new-group"]),
        )

        mock_collection.update_one.assert_awaited_once()
        filter_arg, update_arg = mock_collection.update_one.await_args.args
        assert filter_arg == {"client_id": "test-client-id"}
        assert update_arg["$set"]["groups"] == ["new-group"]
        # client_name and description were not provided, must not be in $set.
        assert "name" not in update_arg["$set"]
        assert "description" not in update_arg["$set"]
        assert "enabled" not in update_arg["$set"]

    @pytest.mark.asyncio
    async def test_allows_clearing_groups_with_empty_list(
        self,
        service: M2MManagementService,
        mock_collection: MagicMock,
        sample_manual_doc: dict,
    ) -> None:
        mock_collection.find_one = AsyncMock(return_value=sample_manual_doc)

        await service.patch(
            "test-client-id",
            IdPM2MClientPatch(groups=[]),
        )

        _, update_arg = mock_collection.update_one.await_args.args
        assert update_arg["$set"]["groups"] == []

    @pytest.mark.asyncio
    async def test_no_op_patch_skips_update_call(
        self,
        service: M2MManagementService,
        mock_collection: MagicMock,
        sample_manual_doc: dict,
    ) -> None:
        mock_collection.find_one = AsyncMock(return_value=sample_manual_doc)

        # Empty patch (no fields set) should not call update_one.
        await service.patch("test-client-id", IdPM2MClientPatch())

        mock_collection.update_one.assert_not_awaited()


class TestDelete:
    """Tests for M2MManagementService.delete."""

    @pytest.mark.asyncio
    async def test_deletes_manual_record(
        self,
        service: M2MManagementService,
        mock_collection: MagicMock,
        sample_manual_doc: dict,
    ) -> None:
        mock_collection.find_one = AsyncMock(return_value=sample_manual_doc)

        await service.delete("test-client-id")

        mock_collection.delete_one.assert_awaited_once_with({"client_id": "test-client-id"})

    @pytest.mark.asyncio
    async def test_raises_not_found_when_missing(
        self,
        service: M2MManagementService,
        mock_collection: MagicMock,
    ) -> None:
        mock_collection.find_one = AsyncMock(return_value=None)

        with pytest.raises(M2MClientNotFound):
            await service.delete("missing")

    @pytest.mark.asyncio
    async def test_raises_immutable_for_non_manual(
        self,
        service: M2MManagementService,
        mock_collection: MagicMock,
        sample_synced_doc: dict,
    ) -> None:
        mock_collection.find_one = AsyncMock(return_value=sample_synced_doc)

        with pytest.raises(M2MClientImmutable):
            await service.delete("synced-client-id")

        mock_collection.delete_one.assert_not_awaited()


class TestClientIdValidation:
    """Tests for the IdPM2MClientCreate client_id validator."""

    def test_accepts_alphanumerics(self) -> None:
        IdPM2MClientCreate(client_id="abc123", client_name="x")

    def test_accepts_dash_underscore_dot_colon(self) -> None:
        IdPM2MClientCreate(client_id="abc-def_ghi.jkl:mno", client_name="x")

    def test_rejects_whitespace(self) -> None:
        with pytest.raises(ValueError):
            IdPM2MClientCreate(client_id="abc 123", client_name="x")

    def test_rejects_special_chars(self) -> None:
        with pytest.raises(ValueError):
            IdPM2MClientCreate(client_id="abc$123", client_name="x")

    def test_rejects_control_chars(self) -> None:
        with pytest.raises(ValueError):
            IdPM2MClientCreate(client_id="abc\x00123", client_name="x")


class TestCollectionName:
    """Sanity check that service writes to the right collection."""

    def test_collection_name_is_idp_m2m_clients(self) -> None:
        assert COLLECTION_NAME == "idp_m2m_clients"
