"""Unit tests for registry/repositories/documentdb/security_scan_repository.py."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from registry.repositories.documentdb.security_scan_repository import (
    DocumentDBSecurityScanRepository,
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
    r = DocumentDBSecurityScanRepository.__new__(DocumentDBSecurityScanRepository)
    r._collection = mock_collection
    r._collection_name = "mcp_security_scans_test"
    r._indexes_created = True
    return r


class TestEnsureIndexes:
    async def test_creates_compound_index(self, repo, mock_collection):
        repo._indexes_created = False
        await repo.ensure_indexes()
        mock_collection.create_index.assert_called_once()
        assert repo._indexes_created is True

    async def test_skips_when_already_created(self, repo, mock_collection):
        repo._indexes_created = True
        await repo.ensure_indexes()
        mock_collection.create_index.assert_not_called()

    async def test_skips_when_no_collection(self, mock_collection):
        r = DocumentDBSecurityScanRepository.__new__(DocumentDBSecurityScanRepository)
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
        mock_collection.count_documents.return_value = 7
        await repo.load_all()
        mock_collection.count_documents.assert_called_once_with({})

    async def test_error_is_swallowed(self, repo, mock_collection):
        mock_collection.count_documents.side_effect = Exception("db error")
        await repo.load_all()


class TestGet:
    async def test_delegates_to_get_latest(self, repo, mock_collection):
        mock_collection.find_one.return_value = {"_id": "x", "server_path": "/a"}
        result = await repo.get("/a")
        assert result == {"server_path": "/a"}


class TestListAll:
    async def test_returns_docs_without_id(self, repo, mock_collection):
        mock_collection.find.return_value = _make_cursor(
            [{"_id": "1", "server_path": "/a"}, {"_id": "2", "server_path": "/b"}]
        )
        result = await repo.list_all()
        assert result == [{"server_path": "/a"}, {"server_path": "/b"}]

    async def test_sorted_by_timestamp_desc(self, repo, mock_collection):
        cursor = _make_cursor([])
        mock_collection.find.return_value = cursor
        await repo.list_all()
        cursor.sort.assert_called_with("scan_timestamp", -1)

    async def test_error_returns_empty(self, repo, mock_collection):
        mock_collection.find.side_effect = Exception("db error")
        assert await repo.list_all() == []


class TestListLatest:
    async def test_groups_by_server_path_keeping_first_after_desc_sort(self, repo, mock_collection):
        # aggregate() returns one {_id, doc} group per path; the repo unwraps doc.
        mock_collection.aggregate = MagicMock(
            return_value=_make_cursor(
                [
                    {"_id": "/a", "doc": {"_id": "x", "server_path": "/a", "critical_issues": 0}},
                    {"_id": "/b", "doc": {"_id": "y", "server_path": "/b", "high_severity": 1}},
                ]
            )
        )
        result = await repo.list_latest()
        assert result == [
            {"server_path": "/a", "critical_issues": 0},
            {"server_path": "/b", "high_severity": 1},
        ]

    async def test_pipeline_sorts_desc_then_groups_first(self, repo, mock_collection):
        mock_collection.aggregate = MagicMock(return_value=_make_cursor([]))
        await repo.list_latest()
        pipeline = mock_collection.aggregate.call_args[0][0]
        assert pipeline[0] == {"$sort": {"scan_timestamp": -1}}
        assert pipeline[1]["$group"]["_id"] == "$server_path"
        assert pipeline[1]["$group"]["doc"] == {"$first": "$$ROOT"}

    async def test_error_returns_empty(self, repo, mock_collection):
        mock_collection.aggregate = MagicMock(side_effect=Exception("db error"))
        assert await repo.list_latest() == []


class TestCreate:
    async def test_missing_path_returns_false(self, repo, mock_collection):
        assert await repo.create({"scan_status": "done"}) is False
        mock_collection.insert_one.assert_not_called()

    async def test_inserts_with_server_path(self, repo, mock_collection):
        result = await repo.create({"server_path": "/a", "scan_status": "done"})
        assert result is True
        mock_collection.insert_one.assert_called_once()

    async def test_agent_path_copied_to_server_path(self, repo, mock_collection):
        await repo.create({"agent_path": "/agents/x"})
        inserted = mock_collection.insert_one.call_args[0][0]
        assert inserted["server_path"] == "/agents/x"

    async def test_adds_scan_timestamp_when_missing(self, repo, mock_collection):
        await repo.create({"server_path": "/a"})
        inserted = mock_collection.insert_one.call_args[0][0]
        assert "scan_timestamp" in inserted

    async def test_preserves_existing_scan_timestamp(self, repo, mock_collection):
        await repo.create({"server_path": "/a", "scan_timestamp": "2026-01-01"})
        inserted = mock_collection.insert_one.call_args[0][0]
        assert inserted["scan_timestamp"] == "2026-01-01"

    async def test_computes_vulnerability_counts(self, repo, mock_collection):
        await repo.create(
            {
                "server_path": "/a",
                "vulnerabilities": [
                    {"severity": "Critical"},
                    {"severity": "high"},
                    {"severity": "high"},
                    {"severity": "unknown"},
                ],
            }
        )
        inserted = mock_collection.insert_one.call_args[0][0]
        assert inserted["total_vulnerabilities"] == 4
        assert inserted["critical_count"] == 1
        assert inserted["high_count"] == 2
        assert inserted["medium_count"] == 0
        assert inserted["low_count"] == 0

    async def test_insert_error_returns_false(self, repo, mock_collection):
        mock_collection.insert_one.side_effect = Exception("db error")
        assert await repo.create({"server_path": "/a"}) is False


class TestGetLatest:
    async def test_returns_none_when_not_found(self, repo, mock_collection):
        mock_collection.find_one.return_value = None
        assert await repo.get_latest("/a") is None

    async def test_normalizes_trailing_slash(self, repo, mock_collection):
        mock_collection.find_one.return_value = {"_id": "x", "server_path": "/a"}
        await repo.get_latest("/a/")
        filter_arg = mock_collection.find_one.call_args[0][0]
        paths = [c["server_path"] for c in filter_arg["$or"]]
        assert "/a" in paths
        assert "/a/" in paths

    async def test_strips_id(self, repo, mock_collection):
        mock_collection.find_one.return_value = {"_id": "x", "server_path": "/a"}
        result = await repo.get_latest("/a")
        assert "_id" not in result

    async def test_error_returns_none(self, repo, mock_collection):
        mock_collection.find_one.side_effect = Exception("db error")
        assert await repo.get_latest("/a") is None


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
