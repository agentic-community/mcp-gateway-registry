"""Unit tests for get_all_mapped_group_names on the scope repositories.

This method returns the union of every scope document's ``group_mappings``
array, i.e. every IdP group the registry grants access through. It backs the
login-time group filter (Design C): groups outside this set produce no scopes
and are dropped from the session.

The DocumentDB implementation issues a single projected query (live, not the
in-memory cache, so newly added mappings are reflected). The file implementation
returns the keys of the in-memory group_mappings dict.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from registry.repositories.documentdb.scope_repository import DocumentDBScopeRepository
from registry.repositories.file.scope_repository import FileScopeRepository


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
    collection.find = MagicMock(return_value=_make_cursor([]))
    return collection


@pytest.fixture
def doc_repo(mock_collection):
    r = DocumentDBScopeRepository.__new__(DocumentDBScopeRepository)
    r._collection = mock_collection
    r._collection_name = "mcp_scopes_test"
    r._scopes_cache = {}
    return r


class TestDocumentDBMappedGroups:
    async def test_unions_group_mappings_across_docs(self, doc_repo, mock_collection):
        mock_collection.find.return_value = _make_cursor(
            [
                {"_id": "registry-admins", "group_mappings": ["ai_admins", "platform-eng"]},
                {"_id": "registry-readonly", "group_mappings": ["all-eng", "ai_admins"]},
                {"_id": "no-groups", "group_mappings": []},
                {"_id": "missing-field"},
            ]
        )
        result = await doc_repo.get_all_mapped_group_names()
        assert result == {"ai_admins", "platform-eng", "all-eng"}

    async def test_uses_projection_query(self, doc_repo, mock_collection):
        mock_collection.find.return_value = _make_cursor([])
        await doc_repo.get_all_mapped_group_names()
        mock_collection.find.assert_called_once_with({}, {"group_mappings": 1})

    async def test_empty_collection_returns_empty_set(self, doc_repo, mock_collection):
        mock_collection.find.return_value = _make_cursor([])
        assert await doc_repo.get_all_mapped_group_names() == set()

    async def test_error_returns_empty_set(self, doc_repo, mock_collection):
        mock_collection.find.side_effect = Exception("db error")
        assert await doc_repo.get_all_mapped_group_names() == set()


class TestFileMappedGroups:
    @pytest.fixture
    def file_repo(self):
        r = FileScopeRepository.__new__(FileScopeRepository)
        r._scopes_data = {
            "group_mappings": {
                "registry-admins": ["scope-a"],
                "registry-readonly": ["scope-b"],
            },
            "UI-Scopes": {},
        }
        return r

    async def test_returns_mapping_keys(self, file_repo):
        result = await file_repo.get_all_mapped_group_names()
        assert result == {"registry-admins", "registry-readonly"}

    async def test_missing_group_mappings_returns_empty(self):
        r = FileScopeRepository.__new__(FileScopeRepository)
        r._scopes_data = {"UI-Scopes": {}}
        assert await r.get_all_mapped_group_names() == set()
