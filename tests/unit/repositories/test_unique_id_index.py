"""Unit tests for documentdb/_unique_id_index.py (#1276)."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from registry.repositories.documentdb._unique_id_index import (
    ID_INDEX_NAME,
    backfill_missing_id,
    ensure_unique_id_index,
    find_doc_by_id,
)


def _make_cursor(items: list[dict]) -> MagicMock:
    cursor = MagicMock()
    cursor.__aiter__ = lambda self: self
    cursor._items = items
    cursor._index = 0

    async def anext_impl(self):
        if self._index >= len(self._items):
            raise StopAsyncIteration
        item = self._items[self._index]
        self._index += 1
        return item

    cursor.__anext__ = anext_impl
    return cursor


@pytest.fixture
def mock_collection():
    collection = AsyncMock()
    collection.create_index = AsyncMock()
    collection.find_one = AsyncMock(return_value=None)
    collection.update_one = AsyncMock()
    collection.find = MagicMock(return_value=_make_cursor([]))
    return collection


class TestEnsureUniqueIdIndex:
    async def test_creates_unique_partial_index(self, mock_collection):
        await ensure_unique_id_index(mock_collection, "servers")
        mock_collection.create_index.assert_awaited_once_with(
            "id",
            name=ID_INDEX_NAME,
            unique=True,
            partialFilterExpression={"id": {"$exists": True}},
        )

    async def test_tolerates_index_creation_failure(self, mock_collection):
        # Engines that reject the index must not break collection init.
        mock_collection.create_index.side_effect = Exception("unsupported")
        # Should not raise.
        await ensure_unique_id_index(mock_collection, "servers")


class TestBackfillMissingId:
    async def test_assigns_id_to_each_legacy_doc(self, mock_collection):
        legacy = [{"_id": "/a"}, {"_id": "/b"}, {"_id": "/c"}]
        mock_collection.find = MagicMock(return_value=_make_cursor(legacy))

        await backfill_missing_id(mock_collection, "servers")

        assert mock_collection.update_one.await_count == 3
        # Every update sets a string id under $set.
        for call in mock_collection.update_one.await_args_list:
            _filter, update = call.args
            assert "id" in update["$set"]
            assert isinstance(update["$set"]["id"], str)
            assert update["$set"]["id"]

    async def test_no_updates_when_all_docs_have_id(self, mock_collection):
        mock_collection.find = MagicMock(return_value=_make_cursor([]))
        await backfill_missing_id(mock_collection, "servers")
        mock_collection.update_one.assert_not_awaited()

    async def test_tolerates_backfill_failure(self, mock_collection):
        mock_collection.find = MagicMock(side_effect=Exception("db down"))
        # Should not raise.
        await backfill_missing_id(mock_collection, "servers")


class TestFindDocById:
    async def test_returns_none_on_miss(self, mock_collection):
        mock_collection.find_one = AsyncMock(return_value=None)
        result = await find_doc_by_id(mock_collection, "nope")
        assert result is None
        mock_collection.find_one.assert_awaited_once_with({"id": "nope"})

    async def test_remaps_id_to_path_on_hit(self, mock_collection):
        stored = {"_id": "/my-server", "id": "arn:aws:x", "name": "srv"}
        mock_collection.find_one = AsyncMock(return_value=stored)

        result = await find_doc_by_id(mock_collection, "arn:aws:x")

        assert result is not None
        assert result["path"] == "/my-server"
        assert "_id" not in result
        assert result["id"] == "arn:aws:x"
