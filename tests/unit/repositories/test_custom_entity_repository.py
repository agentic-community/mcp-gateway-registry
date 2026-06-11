"""Unit tests for DocumentDBCustomEntityRepository.

The Mongo collection is mocked (no live DB) so these tests focus on the
repository's own logic: entity_type + visibility-filter query composition,
the _id=path convention, the duplicate-key uuid-retry path, and the
delete return-value contract.
"""

import logging
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pymongo.errors import DuplicateKeyError

from registry.repositories.documentdb.custom_entity_repository import (
    DocumentDBCustomEntityRepository,
)
from registry.schemas.custom_entity_models import CustomEntityRecord

logger = logging.getLogger(__name__)

TYPE = "workflow"


def _record(name: str = "r", path: str = "/workflow/abc") -> CustomEntityRecord:
    rec = CustomEntityRecord(entity_type=TYPE, name=name, owner="bob")
    rec.path = path
    return rec


class _AsyncCursor:
    """Minimal async cursor exposing the sort/skip/limit fluent chain."""

    def __init__(self, docs: list[dict[str, Any]]):
        self._docs = docs

    def sort(self, *_a, **_k):
        return self

    def skip(self, *_a, **_k):
        return self

    def limit(self, *_a, **_k):
        return self

    def __aiter__(self):
        async def _gen():
            for d in self._docs:
                yield d

        return _gen()


@pytest.fixture
def collection() -> MagicMock:
    coll = MagicMock()
    coll.insert_one = AsyncMock()
    coll.find_one = AsyncMock()
    coll.delete_one = AsyncMock()
    coll.delete_many = AsyncMock()
    coll.count_documents = AsyncMock()
    coll.find_one_and_update = AsyncMock()
    return coll


@pytest.fixture
def repo(collection):
    r = DocumentDBCustomEntityRepository()
    with patch.object(r, "_get_collection", new=AsyncMock(return_value=collection)):
        yield r


@pytest.mark.unit
class TestCreate:
    async def test_create_sets_id_to_path(self, repo, collection):
        rec = _record(path="/workflow/abc")
        out = await repo.create(rec)
        assert out is rec
        doc = collection.insert_one.call_args[0][0]
        assert doc["_id"] == "/workflow/abc"

    async def test_duplicate_key_regenerates_uuid_once(self, repo, collection):
        collection.insert_one = AsyncMock(side_effect=[DuplicateKeyError("dup"), None])
        rec = _record(path="/workflow/abc")
        await repo.create(rec)
        assert rec.path != "/workflow/abc"
        assert collection.insert_one.await_count == 2


@pytest.mark.unit
class TestListAndCount:
    async def test_list_without_filter_uses_entity_type_only(self, repo, collection):
        collection.find = MagicMock(return_value=_AsyncCursor([]))
        await repo.list_paginated(TYPE)
        assert collection.find.call_args[0][0] == {"entity_type": TYPE}

    async def test_list_merges_visibility_filter(self, repo, collection):
        collection.find = MagicMock(return_value=_AsyncCursor([]))
        vis = {"$or": [{"visibility": "public"}]}
        await repo.list_paginated(TYPE, visibility_filter=vis)
        query = collection.find.call_args[0][0]
        assert query["entity_type"] == TYPE
        assert query["$or"] == [{"visibility": "public"}]

    async def test_count_applies_same_filter(self, repo, collection):
        collection.count_documents = AsyncMock(return_value=5)
        vis = {"$or": [{"visibility": "public"}]}
        total = await repo.count(TYPE, visibility_filter=vis)
        assert total == 5
        assert collection.count_documents.call_args[0][0]["entity_type"] == TYPE
        assert collection.count_documents.call_args[0][0]["$or"] == vis["$or"]


@pytest.mark.unit
class TestDelete:
    async def test_delete_returns_true_when_removed(self, repo, collection):
        collection.delete_one = AsyncMock(return_value=MagicMock(deleted_count=1))
        assert await repo.delete("/workflow/abc") is True

    async def test_delete_returns_false_when_absent(self, repo, collection):
        collection.delete_one = AsyncMock(return_value=MagicMock(deleted_count=0))
        assert await repo.delete("/workflow/abc") is False

    async def test_delete_by_type_returns_count(self, repo, collection):
        collection.delete_many = AsyncMock(return_value=MagicMock(deleted_count=4))
        assert await repo.delete_by_type(TYPE) == 4


@pytest.mark.unit
class TestGet:
    async def test_get_missing_returns_none(self, repo, collection):
        collection.find_one = AsyncMock(return_value=None)
        assert await repo.get("/workflow/none") is None

    async def test_get_strips_id(self, repo, collection):
        collection.find_one = AsyncMock(
            return_value={"_id": "/workflow/abc", "entity_type": TYPE, "name": "r"}
        )
        rec = await repo.get("/workflow/abc")
        assert rec is not None
        assert rec.entity_type == TYPE
