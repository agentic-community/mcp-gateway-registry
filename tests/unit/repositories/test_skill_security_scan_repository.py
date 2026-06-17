"""Unit tests for documentdb/skill_security_scan_repository.py."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from registry.repositories.documentdb.skill_security_scan_repository import (
    DocumentDBSkillSecurityScanRepository,
)


def _make_cursor(items: list[dict]) -> MagicMock:
    cursor = MagicMock()
    cursor.sort = MagicMock(return_value=cursor)
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
    collection.count_documents = AsyncMock(return_value=0)
    collection.insert_one = AsyncMock()
    collection.find_one = AsyncMock(return_value=None)
    collection.create_index = AsyncMock()
    collection.find = MagicMock(return_value=_make_cursor([]))
    return collection


@pytest.fixture
def repo(mock_collection):
    r = DocumentDBSkillSecurityScanRepository.__new__(DocumentDBSkillSecurityScanRepository)
    r._collection = mock_collection
    r._collection_name = "mcp_skill_security_scans_test"
    r._indexes_created = True
    return r


class TestEnsureIndexes:
    async def test_creates_compound_index(self, repo, mock_collection):
        repo._indexes_created = False
        await repo.ensure_indexes()
        mock_collection.create_index.assert_called_once()
        assert repo._indexes_created is True

    async def test_skips_when_already_created(self, repo, mock_collection):
        await repo.ensure_indexes()
        mock_collection.create_index.assert_not_called()

    async def test_skips_when_no_collection(self):
        r = DocumentDBSkillSecurityScanRepository.__new__(DocumentDBSkillSecurityScanRepository)
        r._collection = None
        r._indexes_created = False
        await r.ensure_indexes()

    async def test_index_failure_is_swallowed(self, repo, mock_collection):
        repo._indexes_created = False
        mock_collection.create_index.side_effect = Exception("boom")
        await repo.ensure_indexes()
        assert repo._indexes_created is False


class TestLoadAll:
    async def test_logs_count(self, repo, mock_collection):
        mock_collection.count_documents.return_value = 3
        await repo.load_all()
        mock_collection.count_documents.assert_called_once_with({})

    async def test_error_is_swallowed(self, repo, mock_collection):
        mock_collection.count_documents.side_effect = Exception("db error")
        await repo.load_all()


class TestGet:
    async def test_delegates_to_get_latest(self, repo, mock_collection):
        mock_collection.find_one.return_value = {"_id": "x", "skill_path": "/s"}
        result = await repo.get("/s")
        assert result == {"skill_path": "/s"}


class TestListAll:
    async def test_returns_docs_without_id(self, repo, mock_collection):
        mock_collection.find.return_value = _make_cursor(
            [{"_id": "1", "skill_path": "/a"}, {"_id": "2", "skill_path": "/b"}]
        )
        result = await repo.list_all()
        assert result == [{"skill_path": "/a"}, {"skill_path": "/b"}]

    async def test_sorted_by_timestamp_desc(self, repo, mock_collection):
        cursor = _make_cursor([])
        mock_collection.find.return_value = cursor
        await repo.list_all()
        cursor.sort.assert_called_with("scan_timestamp", -1)

    async def test_error_returns_empty(self, repo, mock_collection):
        mock_collection.find.side_effect = Exception("db error")
        assert await repo.list_all() == []


class TestListLatest:
    async def test_groups_by_skill_path_unwrapping_doc(self, repo, mock_collection):
        mock_collection.aggregate = MagicMock(
            return_value=_make_cursor(
                [
                    {"_id": "/a", "doc": {"_id": "x", "skill_path": "/a", "low_severity": 2}},
                    {"_id": "/b", "doc": {"_id": "y", "skill_path": "/b", "critical_issues": 1}},
                ]
            )
        )
        result = await repo.list_latest()
        assert result == [
            {"skill_path": "/a", "low_severity": 2},
            {"skill_path": "/b", "critical_issues": 1},
        ]

    async def test_pipeline_sorts_desc_then_groups_first(self, repo, mock_collection):
        mock_collection.aggregate = MagicMock(return_value=_make_cursor([]))
        await repo.list_latest()
        pipeline = mock_collection.aggregate.call_args[0][0]
        assert pipeline[0] == {"$sort": {"scan_timestamp": -1}}
        assert pipeline[1]["$group"]["_id"] == "$skill_path"
        assert pipeline[1]["$group"]["doc"] == {"$first": "$$ROOT"}

    async def test_error_returns_empty(self, repo, mock_collection):
        mock_collection.aggregate = MagicMock(side_effect=Exception("db error"))
        assert await repo.list_latest() == []


class TestCreate:
    async def test_missing_skill_path_returns_false(self, repo, mock_collection):
        assert await repo.create({"scan_status": "done"}) is False
        mock_collection.insert_one.assert_not_called()

    async def test_inserts_with_skill_path(self, repo, mock_collection):
        assert await repo.create({"skill_path": "/s"}) is True
        mock_collection.insert_one.assert_called_once()

    async def test_adds_scan_timestamp_when_missing(self, repo, mock_collection):
        await repo.create({"skill_path": "/s"})
        inserted = mock_collection.insert_one.call_args[0][0]
        assert "scan_timestamp" in inserted

    async def test_preserves_existing_scan_timestamp(self, repo, mock_collection):
        await repo.create({"skill_path": "/s", "scan_timestamp": "2026-01-01"})
        inserted = mock_collection.insert_one.call_args[0][0]
        assert inserted["scan_timestamp"] == "2026-01-01"

    async def test_insert_error_returns_false(self, repo, mock_collection):
        mock_collection.insert_one.side_effect = Exception("db error")
        assert await repo.create({"skill_path": "/s"}) is False


class TestGetLatest:
    async def test_returns_none_when_not_found(self, repo, mock_collection):
        mock_collection.find_one.return_value = None
        assert await repo.get_latest("/s") is None

    async def test_filters_by_skill_path(self, repo, mock_collection):
        mock_collection.find_one.return_value = {"_id": "x", "skill_path": "/s"}
        await repo.get_latest("/s")
        filter_arg = mock_collection.find_one.call_args[0][0]
        assert filter_arg == {"skill_path": "/s"}

    async def test_strips_id(self, repo, mock_collection):
        mock_collection.find_one.return_value = {"_id": "x", "skill_path": "/s"}
        result = await repo.get_latest("/s")
        assert "_id" not in result

    async def test_error_returns_none(self, repo, mock_collection):
        mock_collection.find_one.side_effect = Exception("db error")
        assert await repo.get_latest("/s") is None


class TestQueryByStatus:
    async def test_filters_by_status(self, repo, mock_collection):
        cursor = _make_cursor([{"_id": "1", "scan_status": "failed"}])
        mock_collection.find.return_value = cursor
        result = await repo.query_by_status("failed")
        assert result == [{"scan_status": "failed"}]
        mock_collection.find.assert_called_with({"scan_status": "failed"})

    async def test_error_returns_empty(self, repo, mock_collection):
        mock_collection.find.side_effect = Exception("db error")
        assert await repo.query_by_status("failed") == []
