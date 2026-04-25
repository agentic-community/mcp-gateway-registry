"""Unit tests for registry/repositories/app_log_repository.py."""

from datetime import UTC, datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from registry.repositories.app_log_repository import AppLogRepository


@pytest.fixture
def mock_collection():
    collection = AsyncMock()
    collection.count_documents = AsyncMock(return_value=0)
    collection.estimated_document_count = AsyncMock(return_value=0)
    collection.distinct = AsyncMock(return_value=[])

    cursor = MagicMock()
    cursor.sort = MagicMock(return_value=cursor)
    cursor.skip = MagicMock(return_value=cursor)
    cursor.limit = MagicMock(return_value=cursor)
    cursor.__aiter__ = lambda self: self
    cursor._items = []
    cursor._index = 0

    async def anext_impl(self):
        if self._index >= len(self._items):
            raise StopAsyncIteration
        item = self._items[self._index]
        self._index += 1
        return item

    cursor.__anext__ = anext_impl
    collection.find = MagicMock(return_value=cursor)
    return collection


@pytest.fixture
def repo(mock_collection):
    r = AppLogRepository.__new__(AppLogRepository)
    r._collection = mock_collection
    r._collection_name = "application_logs_test"
    return r


@pytest.fixture
def sample_docs() -> list[dict[str, Any]]:
    return [
        {
            "_id": "abc123",
            "timestamp": datetime(2026, 4, 24, 10, 0, 0, tzinfo=UTC),
            "hostname": "pod-abc",
            "service": "registry",
            "level": "INFO",
            "level_no": 20,
            "logger": "registry.main",
            "filename": "main.py",
            "lineno": 42,
            "process": 130,
            "message": "Server started",
            "created_at": datetime(2026, 4, 24, 10, 0, 0, tzinfo=UTC),
        },
    ]


class TestQuery:
    """Test the query method."""

    @pytest.mark.asyncio
    async def test_empty_result(self, repo, mock_collection):
        entries, total = await repo.query()
        assert entries == []
        assert total == 0
        mock_collection.estimated_document_count.assert_called_once()

    @pytest.mark.asyncio
    async def test_uses_estimated_count_when_no_filter(self, repo, mock_collection):
        mock_collection.estimated_document_count.return_value = 100
        _, total = await repo.query()
        assert total == 100
        mock_collection.count_documents.assert_not_called()

    @pytest.mark.asyncio
    async def test_uses_count_documents_with_filter(self, repo, mock_collection):
        mock_collection.count_documents.return_value = 5
        _, total = await repo.query(service="registry")
        assert total == 5
        mock_collection.estimated_document_count.assert_not_called()

    @pytest.mark.asyncio
    async def test_service_filter(self, repo, mock_collection):
        await repo.query(service="registry")
        filter_arg = mock_collection.find.call_args[0][0]
        assert filter_arg["service"] == "registry"

    @pytest.mark.asyncio
    async def test_level_no_gte_filter(self, repo, mock_collection):
        await repo.query(level_no=30)
        filter_arg = mock_collection.find.call_args[0][0]
        assert filter_arg["level_no"] == {"$gte": 30}

    @pytest.mark.asyncio
    async def test_hostname_filter(self, repo, mock_collection):
        await repo.query(hostname="pod-abc")
        filter_arg = mock_collection.find.call_args[0][0]
        assert filter_arg["hostname"] == "pod-abc"

    @pytest.mark.asyncio
    async def test_time_range_filter(self, repo, mock_collection):
        start = datetime(2026, 4, 24, 0, 0, 0, tzinfo=UTC)
        end = datetime(2026, 4, 24, 23, 59, 59, tzinfo=UTC)
        await repo.query(start=start, end=end)
        filter_arg = mock_collection.find.call_args[0][0]
        assert filter_arg["timestamp"]["$gte"] == start
        assert filter_arg["timestamp"]["$lte"] == end

    @pytest.mark.asyncio
    async def test_search_regex_filter(self, repo, mock_collection):
        await repo.query(search="timeout")
        filter_arg = mock_collection.find.call_args[0][0]
        assert filter_arg["message"] == {"$regex": "timeout", "$options": "i"}

    @pytest.mark.asyncio
    async def test_pagination(self, repo, mock_collection):
        await repo.query(skip=10, limit=25)
        cursor = mock_collection.find.return_value
        cursor.skip.assert_called_with(10)
        cursor.limit.assert_called_with(25)

    @pytest.mark.asyncio
    async def test_results_strip_id(self, repo, mock_collection, sample_docs):
        cursor = mock_collection.find.return_value
        cursor._items = sample_docs.copy()
        cursor._index = 0
        mock_collection.estimated_document_count.return_value = 1

        entries, _ = await repo.query()
        assert len(entries) == 1
        assert "_id" not in entries[0]

    @pytest.mark.asyncio
    async def test_sort_by_timestamp_descending(self, repo, mock_collection):
        await repo.query()
        cursor = mock_collection.find.return_value
        cursor.sort.assert_called_with("timestamp", -1)

    @pytest.mark.asyncio
    async def test_error_returns_empty(self, repo, mock_collection):
        mock_collection.find.side_effect = Exception("db error")
        entries, total = await repo.query()
        assert entries == []
        assert total == 0


class TestGetDistinctServices:
    """Test the get_distinct_services method."""

    @pytest.mark.asyncio
    async def test_returns_services(self, repo, mock_collection):
        mock_collection.distinct.return_value = ["registry", "auth-server"]
        result = await repo.get_distinct_services()
        assert result == ["registry", "auth-server"]
        mock_collection.distinct.assert_called_with("service")

    @pytest.mark.asyncio
    async def test_error_returns_empty(self, repo, mock_collection):
        mock_collection.distinct.side_effect = Exception("db error")
        result = await repo.get_distinct_services()
        assert result == []


class TestGetDistinctHostnames:
    """Test the get_distinct_hostnames method."""

    @pytest.mark.asyncio
    async def test_returns_hostnames(self, repo, mock_collection):
        mock_collection.distinct.return_value = ["pod-abc", "pod-def"]
        result = await repo.get_distinct_hostnames()
        assert result == ["pod-abc", "pod-def"]
        mock_collection.distinct.assert_called_with("hostname")

    @pytest.mark.asyncio
    async def test_error_returns_empty(self, repo, mock_collection):
        mock_collection.distinct.side_effect = Exception("db error")
        result = await repo.get_distinct_hostnames()
        assert result == []
