"""Unit tests for documentdb/server_repository.py."""

import re
from unittest.mock import AsyncMock, MagicMock

import pytest
from pymongo.errors import DuplicateKeyError

from registry.repositories.documentdb.server_repository import DocumentDBServerRepository


def _make_cursor(items: list[dict]) -> MagicMock:
    cursor = MagicMock()
    cursor.sort = MagicMock(return_value=cursor)
    cursor.skip = MagicMock(return_value=cursor)
    cursor.limit = MagicMock(return_value=cursor)
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

    async def to_list_impl(length=None):
        return list(items) if length is None else list(items)[:length]

    cursor.to_list = to_list_impl
    return cursor


@pytest.fixture
def mock_collection():
    collection = AsyncMock()
    collection.count_documents = AsyncMock(return_value=0)
    collection.find_one = AsyncMock(return_value=None)
    collection.insert_one = AsyncMock()
    collection.update_one = AsyncMock()
    collection.delete_one = AsyncMock()
    collection.delete_many = AsyncMock()
    collection.find = MagicMock(return_value=_make_cursor([]))
    return collection


@pytest.fixture
def repo(mock_collection):
    r = DocumentDBServerRepository.__new__(DocumentDBServerRepository)
    r._collection = mock_collection
    r._collection_name = "mcp_servers_test"
    r._init_lock = None
    return r


class TestLoadAll:
    async def test_logs_count(self, repo, mock_collection):
        mock_collection.count_documents.return_value = 5
        await repo.load_all()
        mock_collection.count_documents.assert_called_once_with({})

    async def test_error_swallowed(self, repo, mock_collection):
        mock_collection.count_documents.side_effect = Exception("db error")
        await repo.load_all()


class TestGet:
    async def test_found_by_exact_path(self, repo, mock_collection):
        mock_collection.find_one.return_value = {"_id": "/a", "server_name": "A"}
        result = await repo.get("/a")
        assert result["path"] == "/a"
        assert "_id" not in result

    async def test_falls_back_to_alternate_with_slash(self, repo, mock_collection):
        mock_collection.find_one.side_effect = [None, {"_id": "/a/", "server_name": "A"}]
        result = await repo.get("/a")
        assert result["path"] == "/a/"
        assert mock_collection.find_one.call_count == 2

    async def test_falls_back_to_alternate_without_slash(self, repo, mock_collection):
        mock_collection.find_one.side_effect = [None, {"_id": "/a", "server_name": "A"}]
        result = await repo.get("/a/")
        assert result["path"] == "/a"

    async def test_not_found_returns_none(self, repo, mock_collection):
        mock_collection.find_one.return_value = None
        assert await repo.get("/missing") is None

    async def test_error_returns_none(self, repo, mock_collection):
        mock_collection.find_one.side_effect = Exception("db error")
        assert await repo.get("/a") is None


class TestListAll:
    async def test_returns_servers_keyed_by_path(self, repo, mock_collection):
        mock_collection.find.return_value = _make_cursor(
            [{"_id": "/a", "server_name": "A"}, {"_id": "/b", "server_name": "B"}]
        )
        result = await repo.list_all()
        assert set(result.keys()) == {"/a", "/b"}
        assert result["/a"]["path"] == "/a"

    async def test_exclude_tool_list_projection(self, repo, mock_collection):
        await repo.list_all(exclude_tool_list=True)
        projection = mock_collection.find.call_args[0][1]
        assert projection == {"tool_list": 0}

    async def test_no_projection_by_default(self, repo, mock_collection):
        await repo.list_all()
        projection = mock_collection.find.call_args[0][1]
        assert projection is None

    async def test_error_returns_empty(self, repo, mock_collection):
        mock_collection.find.side_effect = Exception("db error")
        assert await repo.list_all() == {}


class TestListPaginated:
    async def test_applies_skip_and_limit(self, repo, mock_collection):
        cursor = _make_cursor([{"_id": "/a"}])
        mock_collection.find.return_value = cursor
        await repo.list_paginated(skip=10, limit=5)
        cursor.skip.assert_called_with(10)
        cursor.limit.assert_called_with(5)
        cursor.sort.assert_called_with("_id", 1)

    async def test_error_returns_empty(self, repo, mock_collection):
        mock_collection.find.side_effect = Exception("db error")
        assert await repo.list_paginated() == {}


class TestListByIds:
    async def test_empty_paths_returns_empty(self, repo, mock_collection):
        assert await repo.list_by_ids([]) == {}
        mock_collection.find.assert_not_called()

    async def test_filters_by_in_clause(self, repo, mock_collection):
        mock_collection.find.return_value = _make_cursor([{"_id": "/a"}])
        result = await repo.list_by_ids(["/a", "/b"])
        assert result["/a"]["path"] == "/a"
        assert mock_collection.find.call_args[0][0] == {"_id": {"$in": ["/a", "/b"]}}

    async def test_error_returns_empty(self, repo, mock_collection):
        mock_collection.find.side_effect = Exception("db error")
        assert await repo.list_by_ids(["/a"]) == {}


class TestListBySource:
    async def test_filters_by_source(self, repo, mock_collection):
        mock_collection.find.return_value = _make_cursor([{"_id": "/a", "source": "anthropic"}])
        result = await repo.list_by_source("anthropic")
        assert result["/a"]["path"] == "/a"
        assert mock_collection.find.call_args[0][0] == {"source": "anthropic"}

    async def test_error_returns_empty(self, repo, mock_collection):
        mock_collection.find.side_effect = Exception("db error")
        assert await repo.list_by_source("anthropic") == {}


class TestCreate:
    async def test_inserts_with_timestamps_and_id(self, repo, mock_collection):
        result = await repo.create({"path": "/a", "server_name": "A"})
        assert result is True
        doc = mock_collection.insert_one.call_args[0][0]
        assert doc["_id"] == "/a"
        assert "path" not in doc
        assert "registered_at" in doc
        assert "updated_at" in doc
        assert doc["is_enabled"] is False

    async def test_duplicate_key_returns_false(self, repo, mock_collection):
        mock_collection.insert_one.side_effect = DuplicateKeyError("dup")
        assert await repo.create({"path": "/a", "server_name": "A"}) is False

    async def test_generic_error_returns_false(self, repo, mock_collection):
        mock_collection.insert_one.side_effect = Exception("db error")
        assert await repo.create({"path": "/a", "server_name": "A"}) is False

    async def test_duplicate_id_raises_asset_id_conflict(self, repo, mock_collection):
        """A DuplicateKeyError on the id index -> AssetIdConflictError (#1276)."""
        from registry.exceptions import AssetIdConflictError

        mock_collection.insert_one.side_effect = DuplicateKeyError(
            "E11000 duplicate key", 11000, {"keyPattern": {"id": 1}}
        )
        with pytest.raises(AssetIdConflictError):
            await repo.create({"path": "/a", "server_name": "A", "id": "arn:aws:x"})

    async def test_duplicate_path_still_returns_false(self, repo, mock_collection):
        """A DuplicateKeyError on the path (_id) index keeps old behavior."""
        mock_collection.insert_one.side_effect = DuplicateKeyError(
            "E11000 duplicate key", 11000, {"keyPattern": {"_id": 1}}
        )
        assert await repo.create({"path": "/a", "server_name": "A"}) is False


class TestUpdate:
    async def test_updates_existing(self, repo, mock_collection):
        mock_collection.update_one.return_value = MagicMock(matched_count=1)
        result = await repo.update("/a", {"server_name": "A"})
        assert result is True
        spec = mock_collection.update_one.call_args[0][1]
        assert "$set" in spec
        assert "path" not in spec["$set"]

    async def test_not_found_returns_false(self, repo, mock_collection):
        mock_collection.update_one.return_value = MagicMock(matched_count=0)
        assert await repo.update("/a", {"server_name": "A"}) is False

    async def test_cleared_proxy_url_unsets_sidecar(self, repo, mock_collection):
        mock_collection.update_one.return_value = MagicMock(matched_count=1)
        await repo.update("/a", {"proxy_pass_url": ""})
        spec = mock_collection.update_one.call_args[0][1]
        assert "$unset" in spec

    async def test_error_returns_false(self, repo, mock_collection):
        mock_collection.update_one.side_effect = Exception("db error")
        assert await repo.update("/a", {"server_name": "A"}) is False


class TestDelete:
    async def test_deletes_existing(self, repo, mock_collection):
        mock_collection.find_one.return_value = {"_id": "/a", "server_name": "A"}
        mock_collection.delete_one.return_value = MagicMock(deleted_count=1)
        assert await repo.delete("/a") is True

    async def test_not_found_returns_false(self, repo, mock_collection):
        mock_collection.find_one.return_value = None
        assert await repo.delete("/a") is False

    async def test_delete_failure_returns_false(self, repo, mock_collection):
        mock_collection.find_one.return_value = {"_id": "/a"}
        mock_collection.delete_one.return_value = MagicMock(deleted_count=0)
        assert await repo.delete("/a") is False

    async def test_error_returns_false(self, repo, mock_collection):
        mock_collection.find_one.side_effect = Exception("db error")
        assert await repo.delete("/a") is False


class TestDeleteWithVersions:
    async def test_deletes_active_and_versions(self, repo, mock_collection):
        mock_collection.delete_many.return_value = MagicMock(deleted_count=3)
        result = await repo.delete_with_versions("/a")
        assert result == 3
        filter_query = mock_collection.delete_many.call_args[0][0]
        assert "$or" in filter_query

    async def test_none_found_returns_zero(self, repo, mock_collection):
        mock_collection.delete_many.return_value = MagicMock(deleted_count=0)
        assert await repo.delete_with_versions("/a") == 0

    async def test_error_returns_zero(self, repo, mock_collection):
        mock_collection.delete_many.side_effect = Exception("db error")
        assert await repo.delete_with_versions("/a") == 0

    async def test_regex_metacharacters_in_path_are_escaped(self, repo, mock_collection):
        """A path with regex metacharacters must match only its literal versions.

        Without escaping, a path like "/a.b" would let "." match any character
        (over-deleting "/aXb:v1"), and a path like "/(a+)+" could trigger ReDoS.
        """
        mock_collection.delete_many.return_value = MagicMock(deleted_count=1)

        malicious_path = "/a.*b(c+)+"
        await repo.delete_with_versions(malicious_path)

        filter_query = mock_collection.delete_many.call_args[0][0]
        version_clause = next(
            clause for clause in filter_query["$or"] if isinstance(clause.get("_id"), dict)
        )
        pattern = version_clause["_id"]["$regex"]

        # The anchor and separator stay outside the escape; the path is a literal.
        assert pattern == f"^{re.escape(malicious_path)}:"
        # No raw metacharacters leaked into the pattern body.
        assert ".*" not in pattern
        assert "(c+)+" not in pattern
        # The compiled pattern matches the intended version doc and nothing else.
        compiled = re.compile(pattern)
        assert compiled.match(f"{malicious_path}:1.0.0")
        assert not compiled.match("/aXXXb:1.0.0")

    async def test_exact_path_clause_is_literal_string(self, repo, mock_collection):
        """The active-document clause must be an exact string match, never a regex."""
        mock_collection.delete_many.return_value = MagicMock(deleted_count=1)
        await repo.delete_with_versions("/a.*")
        filter_query = mock_collection.delete_many.call_args[0][0]
        exact_clause = next(
            clause for clause in filter_query["$or"] if isinstance(clause.get("_id"), str)
        )
        assert exact_clause["_id"] == "/a.*"


class TestStateMethods:
    async def test_get_state_enabled(self, repo, mock_collection):
        mock_collection.find_one.return_value = {"_id": "/a", "is_enabled": True}
        assert await repo.get_state("/a") is True

    async def test_get_state_missing_returns_false(self, repo, mock_collection):
        mock_collection.find_one.return_value = None
        assert await repo.get_state("/a") is False

    async def test_get_all_states(self, repo, mock_collection):
        mock_collection.find.return_value = _make_cursor(
            [{"_id": "/a", "is_enabled": True}, {"_id": "/b", "is_enabled": False}]
        )
        states = await repo.get_all_states()
        assert states == {"/a": True, "/b": False}

    async def test_get_all_states_error_returns_empty(self, repo, mock_collection):
        mock_collection.find.side_effect = Exception("db error")
        assert await repo.get_all_states() == {}

    async def test_set_state_success(self, repo, mock_collection):
        mock_collection.find_one.return_value = {"_id": "/a", "server_name": "A"}
        mock_collection.update_one.return_value = MagicMock(matched_count=1)
        assert await repo.set_state("/a", True) is True

    async def test_set_state_not_found(self, repo, mock_collection):
        mock_collection.find_one.return_value = None
        assert await repo.set_state("/a", True) is False

    async def test_set_state_error_returns_false(self, repo, mock_collection):
        mock_collection.find_one.side_effect = Exception("db error")
        assert await repo.set_state("/a", True) is False


class TestCount:
    async def test_returns_count(self, repo, mock_collection):
        mock_collection.count_documents.return_value = 42
        assert await repo.count() == 42

    async def test_counts_all_documents_by_default(self, repo, mock_collection):
        mock_collection.count_documents.return_value = 42
        await repo.count()
        mock_collection.count_documents.assert_called_once_with({})

    async def test_exclude_versions_filters_version_docs(self, repo, mock_collection):
        mock_collection.count_documents.return_value = 5
        result = await repo.count(exclude_versions=True)
        assert result == 5
        # Version documents (_id containing ":") are excluded from the count
        mock_collection.count_documents.assert_called_once_with({"_id": {"$not": {"$regex": ":"}}})

    async def test_error_returns_zero(self, repo, mock_collection):
        mock_collection.count_documents.side_effect = Exception("db error")
        assert await repo.count() == 0


class TestCountTools:
    async def test_sums_tool_list_sizes(self, repo, mock_collection):
        mock_collection.aggregate = MagicMock(
            return_value=_make_cursor([{"_id": None, "total": 13}])
        )

        result = await repo.count_tools()

        assert result == 13
        # Version documents are excluded so tools are not double-counted
        pipeline = mock_collection.aggregate.call_args[0][0]
        assert pipeline[0] == {"$match": {"_id": {"$not": {"$regex": ":"}}}}

    async def test_empty_collection_returns_zero(self, repo, mock_collection):
        mock_collection.aggregate = MagicMock(return_value=_make_cursor([]))
        assert await repo.count_tools() == 0

    async def test_error_returns_zero(self, repo, mock_collection):
        mock_collection.aggregate = MagicMock(side_effect=Exception("db error"))
        assert await repo.count_tools() == 0


class TestUpdateField:
    async def test_set_value(self, repo, mock_collection):
        mock_collection.update_one.return_value = MagicMock(modified_count=1)
        assert await repo.update_field("/a", "tags", ["x"]) is True
        spec = mock_collection.update_one.call_args[0][1]
        assert spec == {"$set": {"tags": ["x"]}}

    async def test_none_value_unsets(self, repo, mock_collection):
        mock_collection.update_one.return_value = MagicMock(modified_count=1)
        await repo.update_field("/a", "tags", None)
        spec = mock_collection.update_one.call_args[0][1]
        assert spec == {"$unset": {"tags": ""}}

    async def test_no_modification_returns_false(self, repo, mock_collection):
        mock_collection.update_one.return_value = MagicMock(modified_count=0)
        assert await repo.update_field("/a", "tags", ["x"]) is False


class TestFindWithFilter:
    async def test_returns_matching_docs(self, repo, mock_collection):
        mock_collection.find.return_value = _make_cursor([{"_id": "/a", "x": 1}])
        result = await repo.find_with_filter({"x": 1})
        assert result == {"/a": {"x": 1}}

    async def test_applies_limit(self, repo, mock_collection):
        cursor = _make_cursor([])
        mock_collection.find.return_value = cursor
        await repo.find_with_filter({"x": 1}, limit=10)
        cursor.limit.assert_called_with(10)


class TestFindById:
    async def test_returns_none_on_miss(self, repo, mock_collection):
        mock_collection.find_one.return_value = None
        result = await repo.find_by_id("arn:aws:nope")
        assert result is None
        mock_collection.find_one.assert_awaited_with({"id": "arn:aws:nope"})

    async def test_remaps_id_to_path_on_hit(self, repo, mock_collection):
        mock_collection.find_one.return_value = {
            "_id": "/my-server",
            "id": "arn:aws:x",
            "server_name": "srv",
        }
        result = await repo.find_by_id("arn:aws:x")
        assert result is not None
        assert result["path"] == "/my-server"
        assert "_id" not in result
        assert result["id"] == "arn:aws:x"

    async def test_empty_id_returns_none_without_query(self, repo, mock_collection):
        mock_collection.find_one.reset_mock()
        result = await repo.find_by_id("")
        assert result is None
        mock_collection.find_one.assert_not_awaited()
