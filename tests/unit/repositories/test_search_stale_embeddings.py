"""Unit tests for stale embedding detection and admin cleanup (issue #1145)."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from registry.repositories.documentdb.search_repository import DocumentDBSearchRepository


def _make_repo() -> DocumentDBSearchRepository:
    repo = DocumentDBSearchRepository()
    repo._collection = AsyncMock()
    return repo


@pytest.mark.unit
class TestFindStaleEmbeddings:
    """Tests for find_stale_embeddings."""

    @pytest.mark.asyncio
    async def test_reports_orphaned_index_entries(self):
        repo = _make_repo()

        mock_source_col = MagicMock()
        mock_source_cursor = AsyncMock()
        mock_source_cursor.to_list = AsyncMock(return_value=[{"_id": "/server-a"}])
        mock_source_col.find.return_value = mock_source_cursor

        mock_embeddings_cursor = AsyncMock()
        mock_embeddings_cursor.to_list = AsyncMock(
            return_value=[
                {"_id": "/server-a", "entity_type": "mcp_server", "name": "A", "is_enabled": True},
                {"_id": "/ghost", "entity_type": "mcp_server", "name": "Ghost", "is_enabled": True},
            ]
        )
        mock_embeddings_col = MagicMock()
        mock_embeddings_col.find.return_value = mock_embeddings_cursor

        mock_db = MagicMock()
        mock_db.__getitem__ = MagicMock(return_value=mock_source_col)

        with (
            patch.object(repo, "_get_collection", new_callable=AsyncMock, return_value=mock_embeddings_col),
            patch(
                "registry.repositories.documentdb.search_repository.get_documentdb_client",
                new_callable=AsyncMock,
                return_value=mock_db,
            ),
            patch(
                "registry.repositories.documentdb.search_repository.get_collection_name",
                side_effect=lambda name: name,
            ),
        ):
            result = await repo.find_stale_embeddings()

        assert result["total_stale"] == 1
        assert result["stale"][0]["path"] == "/ghost"


@pytest.mark.unit
class TestRemoveStaleEmbeddings:
    """Tests for remove_stale_embeddings."""

    @pytest.mark.asyncio
    async def test_reports_removed_when_embedding_existed(self):
        # delete_one reporting deleted_count > 0 means a stale embedding was
        # actually removed -> status "removed".
        repo = _make_repo()
        repo._collection.delete_one = AsyncMock(return_value=MagicMock(deleted_count=1))

        result = await repo.remove_stale_embeddings(["/ghost-a", "/ghost-b"])

        assert result["removed"] == 2
        assert result["not_found"] == 0
        assert result["failed"] == 0
        assert result["total"] == 2
        assert all(d["status"] == "removed" for d in result["details"])

    @pytest.mark.asyncio
    async def test_reports_not_found_for_noop_path(self):
        # deleted_count == 0 means nothing was indexed at that path (typo or
        # already-clean) -> status "not_found", NOT a misleading success.
        repo = _make_repo()
        repo._collection.delete_one = AsyncMock(return_value=MagicMock(deleted_count=0))

        result = await repo.remove_stale_embeddings(["/does-not-exist"])

        assert result["removed"] == 0
        assert result["not_found"] == 1
        assert result["failed"] == 0
        assert result["details"][0]["status"] == "not_found"

    @pytest.mark.asyncio
    async def test_records_failure_when_delete_raises(self):
        repo = _make_repo()
        repo._collection.delete_one = AsyncMock(
            side_effect=[MagicMock(deleted_count=1), Exception("db error")]
        )

        result = await repo.remove_stale_embeddings(["/ghost-a", "/ghost-b"])

        assert result["removed"] == 1
        assert result["failed"] == 1
        assert result["details"][1]["status"] == "failed"


@pytest.mark.unit
class TestRemoveEntity:
    """Tests for remove_entity return value."""

    @pytest.mark.asyncio
    async def test_returns_true_when_deleted(self):
        repo = _make_repo()
        mock_collection = AsyncMock()
        mock_collection.delete_one.return_value = MagicMock(deleted_count=1)
        repo._collection = mock_collection

        with patch.object(repo, "_get_collection", AsyncMock(return_value=mock_collection)):
            assert await repo.remove_entity("/server-a") is True

    @pytest.mark.asyncio
    async def test_returns_true_when_not_found(self):
        repo = _make_repo()
        mock_collection = AsyncMock()
        mock_collection.delete_one.return_value = MagicMock(deleted_count=0)
        repo._collection = mock_collection

        with patch.object(repo, "_get_collection", AsyncMock(return_value=mock_collection)):
            assert await repo.remove_entity("/missing") is True

    @pytest.mark.asyncio
    async def test_returns_false_on_exception(self):
        repo = _make_repo()
        mock_collection = AsyncMock()
        mock_collection.delete_one.side_effect = RuntimeError("db down")
        repo._collection = mock_collection

        with patch.object(repo, "_get_collection", AsyncMock(return_value=mock_collection)):
            assert await repo.remove_entity("/server-a") is False
