"""Unit tests for skill health storage in the DocumentDB skill repository (issue #1952).

Covers the startup reset of inline skills that the old HEAD probe marked
unhealthy, and the ``fields`` projection on ``list_by_paths`` that search
uses to read health without loading whole skill documents.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from registry.repositories.documentdb import skill_repository
from registry.repositories.documentdb.skill_repository import (
    DocumentDBSkillRepository,
    _reset_inline_skill_health,
)

# The filter is the whole contract of the reset: only documents the content
# route serves inline (non-empty skill_md_content) and that are unhealthy.
EXPECTED_RESET_FILTER: dict = {
    "skill_md_content": {"$nin": [None, ""]},
    "health_status": "unhealthy",
}
EXPECTED_RESET_UPDATE: dict = {
    "$set": {"health_status": "unknown", "last_checked_time": None},
}


def _make_cursor(
    items: list[dict],
) -> MagicMock:
    """Build an async-iterable cursor that yields the given documents."""
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


def _repo_with_collection(
    collection: MagicMock,
) -> DocumentDBSkillRepository:
    """Build a repository whose collection is already initialized."""
    repo = DocumentDBSkillRepository()
    repo._collection = collection
    repo.ensure_indexes = AsyncMock()
    return repo


@pytest.fixture
def mock_collection():
    collection = AsyncMock()
    collection.update_many = AsyncMock(return_value=MagicMock(modified_count=0))
    collection.find = MagicMock(return_value=_make_cursor([]))
    return collection


class TestResetInlineSkillHealth:
    async def test_resets_only_inline_unhealthy_skills(self, mock_collection):
        await _reset_inline_skill_health(mock_collection, "agent_skills")

        mock_collection.update_many.assert_awaited_once_with(
            EXPECTED_RESET_FILTER,
            EXPECTED_RESET_UPDATE,
        )

    async def test_filter_excludes_url_backed_and_empty_inline_skills(self, mock_collection):
        # A missing field and an empty string both mean "not inline": the
        # content route fetches the URL for those, so a probe result stands.
        await _reset_inline_skill_health(mock_collection, "agent_skills")

        query = mock_collection.update_many.await_args.args[0]
        excluded = query["skill_md_content"]["$nin"]
        assert None in excluded
        assert "" in excluded

    async def test_filter_leaves_healthy_and_unknown_skills_alone(self, mock_collection):
        await _reset_inline_skill_health(mock_collection, "agent_skills")

        query = mock_collection.update_many.await_args.args[0]
        assert query["health_status"] == "unhealthy"

    async def test_logs_when_documents_are_reset(self, mock_collection, caplog):
        mock_collection.update_many.return_value = MagicMock(modified_count=3)

        with caplog.at_level("INFO"):
            await _reset_inline_skill_health(mock_collection, "agent_skills")

        assert "Reset health on 3 inline agent_skills documents" in caplog.text

    async def test_tolerates_update_failure(self, mock_collection, caplog):
        # A failure must not block collection init, mirroring backfill_missing_id.
        mock_collection.update_many.side_effect = Exception("db down")

        with caplog.at_level("WARNING"):
            await _reset_inline_skill_health(mock_collection, "agent_skills")

        assert "Inline skill health reset failed" in caplog.text

    async def test_runs_during_collection_init(self, mock_collection):
        repo = DocumentDBSkillRepository()
        repo.ensure_indexes = AsyncMock()
        db = MagicMock()
        db.__getitem__.return_value = mock_collection
        reset = AsyncMock()

        with (
            patch.object(skill_repository, "get_documentdb_client", AsyncMock(return_value=db)),
            patch.object(skill_repository, "backfill_normalized_identity_url", AsyncMock()),
            patch.object(skill_repository, "backfill_missing_id", AsyncMock()),
            patch.object(skill_repository, "ensure_unique_id_index", AsyncMock()),
            patch.object(skill_repository, "_reset_inline_skill_health", reset),
        ):
            await repo._get_collection()
            await repo._get_collection()

        reset.assert_awaited_once_with(mock_collection, repo._collection_name)


class TestListByPathsProjection:
    async def test_empty_field_list_projects_only_id(self, mock_collection):
        # MongoDB reads an empty projection as "return everything", which
        # would hand back encrypted credential fields. [] must mean _id only.
        repo = _repo_with_collection(mock_collection)

        await repo.list_by_paths(["/skills/a"], fields=[])

        projection = mock_collection.find.call_args.args[1]
        assert projection == {"_id": 1}

    async def test_named_fields_are_projected(self, mock_collection):
        repo = _repo_with_collection(mock_collection)

        await repo.list_by_paths(
            ["/skills/a"],
            fields=["health_status", "last_checked_time"],
        )

        projection = mock_collection.find.call_args.args[1]
        assert projection == {"health_status": 1, "last_checked_time": 1}

    async def test_no_fields_returns_whole_document(self, mock_collection):
        repo = _repo_with_collection(mock_collection)

        await repo.list_by_paths(["/skills/a"])

        projection = mock_collection.find.call_args.args[1]
        assert projection is None

    async def test_results_are_keyed_by_path(self, mock_collection):
        mock_collection.find = MagicMock(
            return_value=_make_cursor([{"_id": "/skills/a", "health_status": "healthy"}])
        )
        repo = _repo_with_collection(mock_collection)

        result = await repo.list_by_paths(["/skills/a"], fields=["health_status"])

        assert result == {"/skills/a": {"path": "/skills/a", "health_status": "healthy"}}
