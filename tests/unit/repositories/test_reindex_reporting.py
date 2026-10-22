"""Reindex must report a path as failed when its entry was stored without a vector.

Before this fix, POST /api/admin/embeddings/reindex counted every path that did
not raise as a success. index_server/index_agent/index_skill catch an embedding
failure and store the entry with an empty vector, so a provider that rejected
every request still produced "N success, 0 failed". Found while testing Cohere
on Amazon Bedrock, which rejects texts over 2048 characters (issue #1826).
"""

from unittest.mock import (
    AsyncMock,
    MagicMock,
    patch,
)

import pytest

from registry.repositories.documentdb.search_repository import (
    REINDEX_NO_EMBEDDING_ERROR,
    DocumentDBSearchRepository,
    _reindex_detail,
)

SERVER_INFO: dict = {
    "server_name": "Demo",
    "description": "A demo server.",
    "tags": ["demo"],
    "tool_list": [{"name": "t", "description": "does a thing"}],
}


def _repo_with_collection() -> tuple[DocumentDBSearchRepository, MagicMock]:
    """Build a repository whose embeddings collection is an in-memory mock."""
    repo = DocumentDBSearchRepository.__new__(DocumentDBSearchRepository)
    collection = MagicMock()
    collection.find_one = AsyncMock(return_value=None)
    collection.replace_one = AsyncMock()
    collection.update_one = AsyncMock()
    repo._collection = collection
    return repo, collection


def _source_db(
    servers: dict | None = None,
) -> dict:
    """Build a fake database keyed by collection name, with only servers populated."""
    servers = servers or {}

    def _collection(docs: dict) -> MagicMock:
        col = MagicMock()
        col.find_one = AsyncMock(side_effect=lambda query: docs.get(query["_id"]))
        return col

    db = MagicMock()
    db.__getitem__.side_effect = lambda name: (
        _collection(servers) if "mcp_servers" in name else _collection({})
    )
    return db


@pytest.mark.unit
class TestReindexDetail:
    """The per-path result follows whether a vector was stored."""

    def test_embedded_is_success(self):
        assert _reindex_detail("/a", "mcp_server", True) == {
            "path": "/a",
            "entity_type": "mcp_server",
            "status": "success",
        }

    def test_not_embedded_is_failed_with_reason(self):
        detail = _reindex_detail("/a", "mcp_server", False)
        assert detail["status"] == "failed"
        assert detail["error"] == REINDEX_NO_EMBEDDING_ERROR


@pytest.mark.unit
class TestIndexServerReturnValue:
    """index_server says whether the entry was stored with an embedding."""

    @pytest.mark.asyncio
    async def test_true_when_embedded(self):
        repo, collection = _repo_with_collection()
        with patch.object(repo, "_embed_texts", AsyncMock(return_value=[[0.1, 0.2]])):
            assert await repo.index_server("/a", SERVER_INFO, is_enabled=True) is True
        assert collection.replace_one.await_args.args[1]["embedding"] == [0.1, 0.2]

    @pytest.mark.asyncio
    async def test_false_when_embedding_failed_but_entry_still_stored(self):
        repo, collection = _repo_with_collection()
        with patch.object(repo, "_embed_texts", AsyncMock(return_value=None)):
            assert await repo.index_server("/a", SERVER_INFO, is_enabled=True) is False
        # Stored anyway, with an empty vector, so keyword search still finds it.
        assert collection.replace_one.await_args.args[1]["embedding"] == []

    @pytest.mark.asyncio
    async def test_false_when_write_failed(self):
        repo, collection = _repo_with_collection()
        collection.replace_one = AsyncMock(side_effect=RuntimeError("db down"))
        with patch.object(repo, "_embed_texts", AsyncMock(return_value=[[0.1]])):
            assert await repo.index_server("/a", SERVER_INFO, is_enabled=True) is False

    @pytest.mark.asyncio
    async def test_true_when_skipped_because_unchanged_and_embedded(self):
        repo, collection = _repo_with_collection()
        with patch.object(repo, "_embed_texts", AsyncMock(return_value=[[0.1]])):
            await repo.index_server("/a", SERVER_INFO, is_enabled=True)
        stored = collection.replace_one.await_args.args[1]
        collection.find_one = AsyncMock(
            return_value={
                "content_hash": stored["content_hash"],
                "embedding": stored["embedding"],
                "is_enabled": True,
            }
        )
        embed = AsyncMock()
        with patch.object(repo, "_embed_texts", embed):
            result = await repo.index_server(
                "/a", SERVER_INFO, is_enabled=True, skip_if_unchanged=True
            )
        assert result is True
        embed.assert_not_awaited()


@pytest.mark.unit
class TestReindexPathsCounts:
    """reindex_paths counts a path stored without a vector as failed."""

    @pytest.mark.asyncio
    async def test_counts_follow_index_result(self):
        repo, _ = _repo_with_collection()
        db = _source_db({"/ok": {"_id": "/ok"}, "/too-long": {"_id": "/too-long"}})
        index_server = AsyncMock(side_effect=lambda path, *a, **k: path == "/ok")
        with (
            patch(
                "registry.repositories.documentdb.search_repository.get_documentdb_client",
                AsyncMock(return_value=db),
            ),
            patch.object(repo, "index_server", index_server),
        ):
            result = await repo.reindex_paths(["/ok", "/too-long", "/missing"])

        assert result["success"] == 1
        assert result["failed"] == 2
        by_path = {d["path"]: d for d in result["details"]}
        assert by_path["/ok"]["status"] == "success"
        assert by_path["/too-long"]["status"] == "failed"
        assert by_path["/too-long"]["error"] == REINDEX_NO_EMBEDDING_ERROR
        assert by_path["/missing"]["error"] == "Not found in any source collection"
