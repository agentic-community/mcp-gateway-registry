"""MCP tool annotations in the search index and search results."""

import hashlib
from unittest.mock import (
    AsyncMock,
    MagicMock,
    patch,
)

import pytest

from registry.api.search_routes import (
    MatchingToolResult,
    ToolSearchResult,
)
from registry.repositories.documentdb.search_repository import (
    DocumentDBSearchRepository,
    _attach_tool_annotations,
)


def _repo() -> tuple[DocumentDBSearchRepository, MagicMock]:
    """Build a repository with an in-memory embeddings collection."""
    repo = DocumentDBSearchRepository.__new__(DocumentDBSearchRepository)
    collection = MagicMock()
    collection.find_one = AsyncMock(return_value=None)
    collection.replace_one = AsyncMock()
    repo._collection = collection
    return repo, collection


_SERVER: dict = {
    "server_name": "Notes",
    "description": "Note taking",
    "tool_list": [
        {"name": "read_notes", "description": "Read", "annotations": {"readOnlyHint": True}},
        {
            "name": "delete_note",
            "description": "Delete",
            "annotations": {"readOnlyHint": False, "destructiveHint": True},
            "outputSchema": {"type": "object"},
        },
        {"name": "ping", "description": "Ping"},
        {"name": "forged", "description": "Bad", "annotations": {"readOnlyHint": "true"}},
    ],
}


async def _index(
    server: dict,
) -> dict:
    """Index one server and return the document written to the collection."""
    repo, collection = _repo()
    with patch.object(repo, "_embed_with_trim_retry", AsyncMock(return_value=[0.1])):
        await repo.index_server("/notes", server, is_enabled=True)
    return collection.replace_one.await_args.args[1]


@pytest.mark.unit
class TestIndexServerAnnotations:
    """index_server stores validated annotations per tool."""

    @pytest.mark.asyncio
    async def test_indexed_tools_carry_annotations(self):
        stored = await _index(_SERVER)

        by_name = {tool["name"]: tool for tool in stored["tools"]}
        assert by_name["read_notes"]["annotations"] == {"readOnlyHint": True}
        assert by_name["delete_note"]["annotations"] == {
            "readOnlyHint": False,
            "destructiveHint": True,
        }
        assert by_name["ping"]["annotations"] is None
        assert by_name["forged"]["annotations"] is None

    @pytest.mark.asyncio
    async def test_output_schema_stays_out_of_index(self):
        stored = await _index(_SERVER)

        assert all("outputSchema" not in tool for tool in stored["tools"])


@pytest.mark.unit
class TestAnnotationContentHash:
    """The content hash covers annotations only when a server has them."""

    @pytest.mark.asyncio
    async def test_server_without_annotations_keeps_text_only_hash(self):
        plain = {"server_name": "Plain", "description": "x", "tool_list": [{"name": "a"}]}

        stored = await _index(plain)

        text_hash = hashlib.sha256(stored["text_for_embedding"].encode()).hexdigest()[:16]
        assert stored["content_hash"] == text_hash

    @pytest.mark.asyncio
    async def test_hint_flip_changes_hash(self):
        flipped = {
            **_SERVER,
            "tool_list": [{"name": "read_notes", "annotations": {"readOnlyHint": False}}],
        }
        original = {
            **_SERVER,
            "tool_list": [{"name": "read_notes", "annotations": {"readOnlyHint": True}}],
        }

        first = await _index(original)
        second = await _index(flipped)

        assert first["text_for_embedding"] == second["text_for_embedding"]
        assert first["content_hash"] != second["content_hash"]

    @pytest.mark.asyncio
    async def test_startup_reindex_refreshes_entry_written_without_annotations(self):
        """An entry stored with the old text-only hash is re-embedded, not skipped."""
        repo, collection = _repo()
        old = await _index(_SERVER)
        old_text_hash = hashlib.sha256(old["text_for_embedding"].encode()).hexdigest()[:16]
        collection.find_one = AsyncMock(
            return_value={"content_hash": old_text_hash, "is_enabled": True, "embedding": [0.1]}
        )
        embed = AsyncMock(return_value=[0.2])

        with patch.object(repo, "_embed_with_trim_retry", embed):
            await repo.index_server("/notes", _SERVER, is_enabled=True, skip_if_unchanged=True)

        embed.assert_awaited_once()
        collection.replace_one.assert_awaited_once()


@pytest.mark.unit
class TestAttachToolAnnotations:
    """_attach_tool_annotations copies annotations onto matching tools."""

    def test_adds_annotations_to_matching_tool(self):
        matching = [{"tool_name": "delete_note"}, {"tool_name": "ping"}]
        indexed = [
            {"name": "delete_note", "annotations": {"destructiveHint": True}},
            {"name": "ping", "annotations": None},
        ]

        _attach_tool_annotations(matching, indexed)

        assert matching[0]["annotations"] == {"destructiveHint": True}
        assert "annotations" not in matching[1]

    def test_handles_documents_indexed_before_the_field(self):
        matching = [{"tool_name": "delete_note"}]

        _attach_tool_annotations(matching, [{"name": "delete_note"}])

        assert matching == [{"tool_name": "delete_note"}]

    def test_invalid_indexed_annotations_not_attached(self):
        matching = [{"tool_name": "t"}]

        _attach_tool_annotations(matching, [{"name": "t", "annotations": {"x": 1}}])

        assert matching == [{"tool_name": "t"}]


@pytest.mark.unit
class TestSearchResultModels:
    """The search response models carry the field and default it to None."""

    def test_matching_tool_result(self):
        result = MatchingToolResult(
            tool_name="t", relevance_score=0.5, annotations={"readOnlyHint": False}
        )

        assert result.model_dump()["annotations"] == {"readOnlyHint": False}
        assert MatchingToolResult(tool_name="t", relevance_score=0.5).annotations is None

    def test_tool_search_result(self):
        result = ToolSearchResult(
            server_path="/notes",
            server_name="Notes",
            tool_name="t",
            relevance_score=0.5,
            annotations={"destructiveHint": True},
        )

        assert result.model_dump()["annotations"] == {"destructiveHint": True}
