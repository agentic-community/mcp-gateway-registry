"""Retry a failed embedding once with trimmed text, only when length can be the cause (#1917)."""

import hashlib
import logging
from unittest.mock import (
    AsyncMock,
    MagicMock,
    patch,
)

import pytest

from registry.embeddings.text_budget import EmbeddingText
from registry.repositories.documentdb.search_repository import DocumentDBSearchRepository

TARGET: int = 2000
SETTINGS_PATH: str = "registry.repositories.documentdb.search_repository.settings"


def _repo() -> tuple[DocumentDBSearchRepository, MagicMock]:
    """Build a repository with an in-memory embeddings collection."""
    repo = DocumentDBSearchRepository.__new__(DocumentDBSearchRepository)
    collection = MagicMock()
    collection.find_one = AsyncMock(return_value=None)
    collection.replace_one = AsyncMock()
    collection.update_one = AsyncMock()
    repo._collection = collection
    return repo, collection


def _long_text() -> EmbeddingText:
    """Build text well over TARGET, with names that must survive the trim."""
    return EmbeddingText(
        head=["Big Server", "Many tools."],
        items=[(f"tool_{i:02d}", "describes the tool at length " * 10) for i in range(40)],
    )


def _settings(
    retry_max: int | None,
) -> MagicMock:
    """Build a settings stand-in with the retry target and a model name."""
    fake = MagicMock()
    fake.embeddings_retry_max_input_chars = retry_max
    fake.embeddings_model_name = "openai/text-embedding-3-small"
    return fake


@pytest.mark.unit
class TestEmbedWithTrimRetry:
    """The single shared step every indexer uses."""

    @pytest.mark.asyncio
    async def test_success_on_first_attempt_sends_full_text_once(self):
        repo, _ = _repo()
        text = _long_text()
        embed = AsyncMock(return_value=[[0.1]])
        with patch.object(repo, "_embed_texts", embed), patch(SETTINGS_PATH, _settings(TARGET)):
            assert await repo._embed_with_trim_retry(text, context="server 'x'") == [0.1]
        assert embed.await_count == 1
        assert embed.await_args.args[0] == [text.full()]

    @pytest.mark.asyncio
    async def test_failure_under_target_is_not_retried(self):
        repo, _ = _repo()
        text = EmbeddingText(head=["Small server"])
        embed = AsyncMock(return_value=None)
        with patch.object(repo, "_embed_texts", embed), patch(SETTINGS_PATH, _settings(TARGET)):
            assert await repo._embed_with_trim_retry(text, context="server 'x'") == []
        assert embed.await_count == 1

    @pytest.mark.asyncio
    async def test_failure_over_target_retries_once_with_trimmed_text(self):
        repo, _ = _repo()
        text = _long_text()
        embed = AsyncMock(side_effect=[None, [[0.2]]])
        with patch.object(repo, "_embed_texts", embed), patch(SETTINGS_PATH, _settings(TARGET)):
            assert await repo._embed_with_trim_retry(text, context="server 'x'") == [0.2]
        assert embed.await_count == 2
        retried = embed.await_args_list[1].args[0][0]
        assert len(retried) <= TARGET
        assert all(name in retried for name, _ in text.items)

    @pytest.mark.asyncio
    async def test_second_failure_gives_up_without_looping(self):
        repo, _ = _repo()
        embed = AsyncMock(return_value=None)
        with patch.object(repo, "_embed_texts", embed), patch(SETTINGS_PATH, _settings(TARGET)):
            assert await repo._embed_with_trim_retry(_long_text(), context="server 'x'") == []
        assert embed.await_count == 2

    @pytest.mark.asyncio
    async def test_unset_setting_uses_the_model_default(self):
        repo, _ = _repo()
        fake = _settings(None)
        fake.embeddings_model_name = "bedrock/cohere.embed-english-v3"
        with patch(SETTINGS_PATH, fake):
            assert repo._retry_max_input_chars() == 2048

    @pytest.mark.asyncio
    async def test_warning_has_lengths_and_never_the_text(self, caplog):
        repo, _ = _repo()
        text = _long_text()
        embed = AsyncMock(side_effect=[None, [[0.2]]])
        with (
            caplog.at_level(logging.WARNING),
            patch.object(repo, "_embed_texts", embed),
            patch(SETTINGS_PATH, _settings(TARGET)),
        ):
            await repo._embed_with_trim_retry(text, context="server 'x'")
        warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warnings) == 1
        assert str(len(text.full())) in warnings[0]
        assert "describes the tool" not in warnings[0]


@pytest.mark.unit
class TestIndexServerKeepsFullTextHash:
    """A trimmed entry stores the full text's hash, so skip_if_unchanged skips it."""

    @pytest.mark.asyncio
    async def test_trimmed_entry_is_skipped_when_unchanged(self):
        repo, collection = _repo()
        server = {
            "server_name": "Big Server",
            "description": "Many tools.",
            "tool_list": [
                {"name": f"tool_{i:02d}", "description": "describes the tool at length " * 10}
                for i in range(40)
            ],
        }
        embed = AsyncMock(side_effect=[None, [[0.3]]])
        with patch.object(repo, "_embed_texts", embed), patch(SETTINGS_PATH, _settings(TARGET)):
            assert await repo.index_server("/big", server, is_enabled=True) is True
        stored = collection.replace_one.await_args.args[1]
        full_hash = hashlib.sha256(stored["text_for_embedding"].encode()).hexdigest()[:16]
        assert stored["content_hash"] == full_hash
        assert len(stored["text_for_embedding"]) > TARGET
        assert stored["embedding"] == [0.3]

        collection.find_one = AsyncMock(
            return_value={
                "content_hash": stored["content_hash"],
                "embedding": [0.3],
                "is_enabled": True,
            }
        )
        second = AsyncMock()
        with patch.object(repo, "_embed_texts", second), patch(SETTINGS_PATH, _settings(TARGET)):
            await repo.index_server("/big", server, is_enabled=True, skip_if_unchanged=True)
        second.assert_not_awaited()
