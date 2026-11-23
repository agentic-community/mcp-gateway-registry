"""MCP Apps in the search index: app-only tools stay out, page addresses go in."""

import hashlib
from unittest.mock import (
    AsyncMock,
    MagicMock,
    patch,
)

import pytest

from registry.repositories.documentdb.search_repository import (
    DocumentDBSearchRepository,
    _attach_ui_resource_uris,
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
    "server_name": "Budget",
    "description": "Budget planning",
    "tool_list": [
        {
            "name": "open_budget",
            "description": "Open the budget",
            "ui_resource_uri": "ui://budget/view.html",
            "ui_visibility": ["model", "app"],
        },
        {
            "name": "save_budget",
            "description": "Called by the page",
            "ui_resource_uri": "ui://budget/view.html",
            "ui_visibility": ["app"],
        },
        {"name": "plain", "description": "No page"},
        {"name": "forged", "description": "Bad address", "ui_resource_uri": "javascript:x"},
    ],
}


async def _index(
    server: dict,
) -> dict:
    """Index one server and return the document written to the collection."""
    repo, collection = _repo()
    with patch.object(repo, "_embed_with_trim_retry", AsyncMock(return_value=[0.1])):
        await repo.index_server("/budget", server, is_enabled=True)
    return collection.replace_one.await_args.args[1]


@pytest.mark.unit
class TestIndexServerMcpApps:
    """index_server handling of MCP Apps tools."""

    @pytest.mark.asyncio
    async def test_app_only_tool_is_not_indexed(self):
        """A ui_visibility=["app"] tool is left out of the searchable tools."""
        stored = await _index(_SERVER)

        names = [tool["name"] for tool in stored["tools"]]
        assert "save_budget" not in names
        assert names == ["open_budget", "plain", "forged"]

    @pytest.mark.asyncio
    async def test_app_only_tool_is_not_embedded(self):
        """The app-only tool's text does not reach the embedding."""
        stored = await _index(_SERVER)

        assert "save_budget" not in stored["text_for_embedding"]
        assert "open_budget" in stored["text_for_embedding"]

    @pytest.mark.asyncio
    async def test_indexed_tool_carries_valid_page_address_only(self):
        """A valid ui:// address is stored; a forged one is dropped."""
        stored = await _index(_SERVER)

        by_name = {tool["name"]: tool for tool in stored["tools"]}
        assert by_name["open_budget"]["ui_resource_uri"] == "ui://budget/view.html"
        assert by_name["plain"]["ui_resource_uri"] is None
        assert by_name["forged"]["ui_resource_uri"] is None


@pytest.mark.unit
class TestMcpAppsContentHash:
    """The content hash covers page addresses only when a server has them."""

    @pytest.mark.asyncio
    async def test_server_without_pages_keeps_text_only_hash(self):
        """A server with no MCP App keeps the hash it had before this feature."""
        plain = {"server_name": "Plain", "description": "x", "tool_list": [{"name": "a"}]}

        stored = await _index(plain)

        text_hash = hashlib.sha256(stored["text_for_embedding"].encode()).hexdigest()[:16]
        assert stored["content_hash"] == text_hash

    @pytest.mark.asyncio
    async def test_startup_reindex_refreshes_entry_written_without_addresses(self):
        """An entry stored with the old text-only hash is re-embedded, not skipped."""
        repo, collection = _repo()
        old = await _index(_SERVER)
        old_text_hash = hashlib.sha256(old["text_for_embedding"].encode()).hexdigest()[:16]
        collection.find_one = AsyncMock(
            return_value={"content_hash": old_text_hash, "is_enabled": True, "embedding": [0.1]}
        )
        embed = AsyncMock(return_value=[0.2])

        with patch.object(repo, "_embed_with_trim_retry", embed):
            await repo.index_server("/budget", _SERVER, is_enabled=True, skip_if_unchanged=True)

        embed.assert_awaited_once()
        collection.replace_one.assert_awaited_once()


@pytest.mark.unit
class TestAttachUiResourceUris:
    """_attach_ui_resource_uris copies page addresses onto matching tools."""

    def test_adds_address_to_matching_tool(self):
        """A matching tool with a page gets its address."""
        matching = [{"tool_name": "open_budget"}, {"tool_name": "plain"}]
        indexed = [
            {"name": "open_budget", "ui_resource_uri": "ui://budget/view.html"},
            {"name": "plain", "ui_resource_uri": None},
        ]

        _attach_ui_resource_uris(matching, indexed)

        assert matching[0]["ui_resource_uri"] == "ui://budget/view.html"
        assert "ui_resource_uri" not in matching[1]

    def test_handles_documents_indexed_before_the_field(self):
        """Older index documents have no ui_resource_uri key, so nothing is added."""
        matching = [{"tool_name": "open_budget"}]

        _attach_ui_resource_uris(matching, [{"name": "open_budget"}])

        assert matching == [{"tool_name": "open_budget"}]
