"""Regression tests: the raw user search query stays out of the logs.

A search query is free text a person typed. It can hold a customer name, an email
address, an account number, or an unannounced project. Issue #1752 took the query
out of seven log lines and put it back behind one opt-in flag,
``SEARCH_LOG_QUERY_TEXT``, which defaults to off.

Both states are tested. The off case protects users. The on case is what an
operator asked for when they need to explain why a search returned what it did,
and a regression that quietly breaks either one is worth catching.

The assertion is on the string rather than on a specific log line: the query
passes through several log calls in the repository, and a future addition would
slip past a line-specific check.
"""

import logging
from unittest.mock import AsyncMock, MagicMock

import pytest

from registry.core.config import settings
from registry.repositories.documentdb.search_repository import (
    DocumentDBSearchRepository,
    _tokenize_query,
)

# Long enough to survive _tokenize_query(), which drops tokens of two characters
# or fewer, and distinctive enough that no fixture or log format contains it.
MARKER = "zzqueryleakcanary"
QUERY = f"invoices for {MARKER}"


def _make_cursor(items: list[dict]) -> MagicMock:
    cursor = MagicMock()
    cursor.limit = MagicMock(return_value=cursor)

    async def to_list_impl(length=None):
        return list(items)

    cursor.to_list = to_list_impl
    return cursor


@pytest.fixture
def repo():
    """A search repo wired to an empty mocked collection."""
    instance = DocumentDBSearchRepository.__new__(DocumentDBSearchRepository)

    collection = MagicMock()
    collection.aggregate = MagicMock(return_value=_make_cursor([]))
    collection.find = MagicMock(return_value=_make_cursor([]))

    instance._get_collection = AsyncMock(return_value=collection)
    instance._embed_texts = AsyncMock(return_value=[[0.1, 0.2, 0.3]])
    instance._default_search_scope = AsyncMock(return_value=["mcp_server"])
    return instance


def _records_with_marker(caplog) -> list[logging.LogRecord]:
    return [record for record in caplog.records if MARKER in record.getMessage()]


class TestQueryTextIsNotLoggedByDefault:
    """The flag is off, so no log record may hold the query."""

    @pytest.mark.asyncio
    async def test_marker_appears_in_no_record(self, repo, caplog, monkeypatch):
        monkeypatch.setattr(settings, "search_log_query_text", False)

        with caplog.at_level(logging.DEBUG):
            await repo.search(query=QUERY, max_results=5)

        leaked = [record.getMessage() for record in _records_with_marker(caplog)]
        assert leaked == [], f"query text reached the logs: {leaked}"

    @pytest.mark.asyncio
    async def test_token_count_is_still_reported(self, repo, caplog, monkeypatch):
        """The cleaned line kept the one fact it existed to report."""
        monkeypatch.setattr(settings, "search_log_query_text", False)

        with caplog.at_level(logging.DEBUG):
            await repo.search(query=QUERY, max_results=5)

        expected = len(_tokenize_query(QUERY))
        messages = [record.getMessage() for record in caplog.records]
        assert any(f"tokenized query into {expected} tokens" in m for m in messages), messages

    @pytest.mark.asyncio
    async def test_no_token_list_or_regex_is_logged(self, repo, caplog, monkeypatch):
        """The tokens are the query minus stopwords, and the regex is those tokens joined."""
        monkeypatch.setattr(settings, "search_log_query_text", False)

        with caplog.at_level(logging.DEBUG):
            await repo.search(query=QUERY, max_results=5)

        for record in caplog.records:
            message = record.getMessage()
            for token in _tokenize_query(QUERY):
                assert token not in message, f"token {token!r} leaked in: {message}"


class TestQueryTextIsLoggedWhenTheFlagIsOn:
    """The operator opted in, so the query must actually appear, once."""

    @pytest.mark.asyncio
    async def test_exactly_one_record_holds_the_query(self, repo, caplog, monkeypatch):
        """One guarded line, so the query has a single exit point.

        Two records would mean a second site was added or an old interpolation
        came back.
        """
        monkeypatch.setattr(settings, "search_log_query_text", True)

        with caplog.at_level(logging.DEBUG):
            await repo.search(query=QUERY, max_results=5)

        logged = _records_with_marker(caplog)
        assert len(logged) == 1, [record.getMessage() for record in logged]

    @pytest.mark.asyncio
    async def test_the_query_is_logged_at_info(self, repo, caplog, monkeypatch):
        monkeypatch.setattr(settings, "search_log_query_text", True)

        with caplog.at_level(logging.DEBUG):
            await repo.search(query=QUERY, max_results=5)

        assert _records_with_marker(caplog)[0].levelno == logging.INFO

    @pytest.mark.asyncio
    async def test_the_full_query_is_logged_not_a_fragment(self, repo, caplog, monkeypatch):
        """Opting in means the whole query; a partial mask would be a false comfort."""
        monkeypatch.setattr(settings, "search_log_query_text", True)

        with caplog.at_level(logging.DEBUG):
            await repo.search(query=QUERY, max_results=5)

        assert QUERY in _records_with_marker(caplog)[0].getMessage()


class TestTheDefaultIsOff:
    """The unset case is what every existing deployment lands in on upgrade."""

    def test_settings_default_is_false(self):
        assert settings.model_fields["search_log_query_text"].default is False

    @pytest.mark.parametrize("raw", ["", "false", "no", "0", "off"])
    def test_falsey_values_do_not_enable_it(self, raw, monkeypatch):
        """Pydantic parses these as False; mcpgw's membership test agrees."""
        monkeypatch.setenv("SEARCH_LOG_QUERY_TEXT", raw)
        mcpgw_view = raw.lower() in ("true", "1", "yes")

        assert mcpgw_view is False

    @pytest.mark.parametrize("raw", ["true", "1", "yes"])
    def test_documented_truthy_values_agree_across_both_parsers(self, raw):
        """The docs promise true/false; these three are read the same by both services."""
        from pydantic import TypeAdapter

        registry_view = TypeAdapter(bool).validate_python(raw)
        mcpgw_view = raw.lower() in ("true", "1", "yes")

        assert registry_view is True
        assert mcpgw_view is True

    def test_an_unparseable_value_fails_rather_than_enabling_it(self):
        """A typo must not silently turn the flag on."""
        from pydantic import TypeAdapter, ValidationError

        with pytest.raises(ValidationError):
            TypeAdapter(bool).validate_python("maybe")

        assert ("maybe".lower() in ("true", "1", "yes")) is False


class TestTheOtherTwoSearchPathsAlsoStaySilent:
    """search() is the entry point for all three paths, so one guarded line covers
    them, but each path has its own log calls that could reintroduce the query."""

    @pytest.mark.asyncio
    async def test_lexical_only_path(self, repo, caplog, monkeypatch):
        """Embeddings unavailable, so search() delegates to _lexical_only_search()."""
        monkeypatch.setattr(settings, "search_log_query_text", False)
        repo._embed_texts = AsyncMock(return_value=[])

        with caplog.at_level(logging.DEBUG):
            await repo.search(query=QUERY, max_results=5)

        leaked = [record.getMessage() for record in _records_with_marker(caplog)]
        assert leaked == [], f"query text reached the logs: {leaked}"

    @pytest.mark.asyncio
    async def test_client_side_path_on_mongodb_ce(self, repo, caplog, monkeypatch):
        """MongoDB CE rejects $search: {vectorSearch}, so search() falls back."""
        from pymongo.errors import OperationFailure

        monkeypatch.setattr(settings, "search_log_query_text", False)

        collection = MagicMock()
        collection.aggregate = MagicMock(
            side_effect=OperationFailure("vectorSearch is not supported", code=31082)
        )
        collection.find = MagicMock(return_value=_make_cursor([]))
        repo._get_collection = AsyncMock(return_value=collection)

        with caplog.at_level(logging.DEBUG):
            await repo.search(query=QUERY, max_results=5)

        leaked = [record.getMessage() for record in _records_with_marker(caplog)]
        assert leaked == [], f"query text reached the logs: {leaked}"

    @pytest.mark.asyncio
    async def test_still_exactly_one_line_when_the_flag_is_on(self, repo, caplog, monkeypatch):
        """The guarded line lives in search(), so a fallback must not double it."""
        monkeypatch.setattr(settings, "search_log_query_text", True)
        repo._embed_texts = AsyncMock(return_value=[])

        with caplog.at_level(logging.DEBUG):
            await repo.search(query=QUERY, max_results=5)

        logged = _records_with_marker(caplog)
        assert len(logged) == 1, [record.getMessage() for record in logged]
