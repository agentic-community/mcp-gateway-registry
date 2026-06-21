"""Unit tests for the bulk scope-fetch methods on the scope repositories.

Covers ``get_server_scopes_bulk`` / ``get_ui_scopes_bulk``, added to collapse
the per-scope ``find_one`` fan-out on the ``/api/auth/me`` hot path into a
single ``$in`` query (one round-trip instead of one-per-scope, which dominated
latency on a remote Atlas cluster for users with many groups).

The DocumentDB implementation overrides the methods with a single query; the
file implementation inherits the base-class default that loops the per-scope
getters. Both must produce the same dict-keyed-by-scope, empties-omitted shape.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from registry.repositories.documentdb.scope_repository import (
    DocumentDBScopeRepository,
    _flatten_server_access,
)
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
def repo(mock_collection):
    r = DocumentDBScopeRepository.__new__(DocumentDBScopeRepository)
    r._collection = mock_collection
    r._collection_name = "mcp_scopes_test"
    r._scopes_cache = {}
    return r


class TestFlattenServerAccess:
    """The shared flatten helper must handle both on-disk formats so the
    single and bulk paths agree byte-for-byte."""

    def test_new_format_access_rules(self):
        access = [{"scope_name": "s", "access_rules": [{"server": "a"}, {"server": "b"}]}]
        assert _flatten_server_access(access) == [{"server": "a"}, {"server": "b"}]

    def test_old_direct_format(self):
        access = [{"server": "a", "methods": ["all"]}]
        assert _flatten_server_access(access) == [{"server": "a", "methods": ["all"]}]

    def test_skips_non_server_entries(self):
        access = [{"agent_permissions": ["x"]}, {"server": "a"}]
        assert _flatten_server_access(access) == [{"server": "a"}]

    def test_empty_returns_empty(self):
        assert _flatten_server_access([]) == []


class TestGetServerScopesBulk:
    async def test_empty_input_skips_query(self, repo, mock_collection):
        assert await repo.get_server_scopes_bulk([]) == {}
        mock_collection.find.assert_not_called()

    async def test_only_falsy_input_skips_query(self, repo, mock_collection):
        assert await repo.get_server_scopes_bulk(["", None]) == {}
        mock_collection.find.assert_not_called()

    async def test_uses_in_query_with_deduped_sorted_ids(self, repo, mock_collection):
        mock_collection.find.return_value = _make_cursor([])
        await repo.get_server_scopes_bulk(["b", "a", "b", ""])
        # One $in query, deduped and sorted, blanks dropped.
        mock_collection.find.assert_called_once_with({"_id": {"$in": ["a", "b"]}})

    async def test_returns_flattened_rules_keyed_by_id(self, repo, mock_collection):
        mock_collection.find.return_value = _make_cursor(
            [
                {
                    "_id": "scope-a",
                    "server_access": [{"scope_name": "scope-a", "access_rules": [{"server": "x"}]}],
                },
                {
                    "_id": "scope-b",
                    "server_access": [{"server": "y", "methods": ["all"]}],
                },
            ]
        )
        result = await repo.get_server_scopes_bulk(["scope-a", "scope-b"])
        assert result == {
            "scope-a": [{"server": "x"}],
            "scope-b": [{"server": "y", "methods": ["all"]}],
        }

    async def test_omits_scopes_with_no_rules(self, repo, mock_collection):
        mock_collection.find.return_value = _make_cursor(
            [
                {"_id": "scope-a", "server_access": [{"server": "x"}]},
                {"_id": "scope-empty", "server_access": []},
            ]
        )
        result = await repo.get_server_scopes_bulk(["scope-a", "scope-empty"])
        assert "scope-empty" not in result
        assert result == {"scope-a": [{"server": "x"}]}

    async def test_error_returns_empty(self, repo, mock_collection):
        mock_collection.find.side_effect = Exception("db error")
        assert await repo.get_server_scopes_bulk(["scope-a"]) == {}


class TestGetUIScopesBulk:
    async def test_empty_input_skips_query(self, repo, mock_collection):
        assert await repo.get_ui_scopes_bulk([]) == {}
        mock_collection.find.assert_not_called()

    async def test_uses_in_query_with_deduped_sorted_ids(self, repo, mock_collection):
        mock_collection.find.return_value = _make_cursor([])
        await repo.get_ui_scopes_bulk(["b", "a", "a"])
        mock_collection.find.assert_called_once_with({"_id": {"$in": ["a", "b"]}})

    async def test_returns_ui_permissions_keyed_by_id(self, repo, mock_collection):
        mock_collection.find.return_value = _make_cursor(
            [
                {"_id": "admin", "ui_permissions": {"list_service": ["all"]}},
                {"_id": "user", "ui_permissions": {"list_service": ["mcpgw"]}},
            ]
        )
        result = await repo.get_ui_scopes_bulk(["admin", "user"])
        assert result == {
            "admin": {"list_service": ["all"]},
            "user": {"list_service": ["mcpgw"]},
        }

    async def test_omits_scopes_with_no_permissions(self, repo, mock_collection):
        mock_collection.find.return_value = _make_cursor(
            [
                {"_id": "admin", "ui_permissions": {"list_service": ["all"]}},
                {"_id": "noperm", "ui_permissions": {}},
                {"_id": "missing"},
            ]
        )
        result = await repo.get_ui_scopes_bulk(["admin", "noperm", "missing"])
        assert result == {"admin": {"list_service": ["all"]}}

    async def test_error_returns_empty(self, repo, mock_collection):
        mock_collection.find.side_effect = Exception("db error")
        assert await repo.get_ui_scopes_bulk(["admin"]) == {}


class TestFileBackendBulkDefault:
    """The file backend has no override, so it uses the base-class default
    that loops the per-scope getters. Verify the same shape contract."""

    @pytest.fixture
    def file_repo(self):
        r = FileScopeRepository.__new__(FileScopeRepository)
        r._scopes_data = {
            "scope-a": [{"server": "x", "methods": ["all"]}],
            "scope-empty": [],
            "UI-Scopes": {
                "admin": {"list_service": ["all"]},
                "noperm": {},
            },
        }
        return r

    async def test_server_scopes_bulk_loops_singles(self, file_repo):
        result = await file_repo.get_server_scopes_bulk(["scope-a", "scope-empty", "missing"])
        assert result == {"scope-a": [{"server": "x", "methods": ["all"]}]}

    async def test_ui_scopes_bulk_loops_singles(self, file_repo):
        result = await file_repo.get_ui_scopes_bulk(["admin", "noperm", "missing"])
        assert result == {"admin": {"list_service": ["all"]}}


class TestGetGroupMappingsBulk:
    """DocumentDB overrides get_group_mappings_bulk with a single $in query
    returning the de-duplicated union of scope names across the groups."""

    async def test_empty_input_skips_query(self, repo, mock_collection):
        assert await repo.get_group_mappings_bulk([]) == []
        mock_collection.find.assert_not_called()

    async def test_only_falsy_input_skips_query(self, repo, mock_collection):
        assert await repo.get_group_mappings_bulk(["", None]) == []
        mock_collection.find.assert_not_called()

    async def test_uses_in_query_with_deduped_sorted_groups(self, repo, mock_collection):
        mock_collection.find.return_value = _make_cursor([])
        await repo.get_group_mappings_bulk(["g-b", "g-a", "g-b", ""])
        mock_collection.find.assert_called_once_with({"group_mappings": {"$in": ["g-a", "g-b"]}})

    async def test_returns_deduped_union_of_scope_ids(self, repo, mock_collection):
        # Two groups map to overlapping scopes; the scope _id appears once each.
        mock_collection.find.return_value = _make_cursor(
            [{"_id": "registry-admins"}, {"_id": "public-mcp-users"}]
        )
        result = await repo.get_group_mappings_bulk(["g-a", "g-b"])
        assert result == ["registry-admins", "public-mcp-users"]

    async def test_error_returns_empty(self, repo, mock_collection):
        mock_collection.find.side_effect = Exception("db error")
        assert await repo.get_group_mappings_bulk(["g-a"]) == []


class TestFileBackendGroupMappingsBulkDefault:
    """File backend inherits the base-class default: loops the single getter
    and unions the results, de-duplicated and order-stable."""

    @pytest.fixture
    def file_repo(self):
        r = FileScopeRepository.__new__(FileScopeRepository)
        r._scopes_data = {
            "group_mappings": {
                "g-a": ["read:servers", "read:tools"],
                "g-b": ["read:servers", "write:servers"],
            },
        }
        return r

    async def test_unions_and_dedupes(self, file_repo):
        result = await file_repo.get_group_mappings_bulk(["g-a", "g-b", "missing"])
        assert result == ["read:servers", "read:tools", "write:servers"]

    async def test_empty_returns_empty(self, file_repo):
        assert await file_repo.get_group_mappings_bulk([]) == []
