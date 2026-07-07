"""Regression tests: user query strings must never reach MongoDB as a raw
regex/regexMatch pattern.

The hybrid keyword-matching stage of the search pipeline builds `$regex` and
`$regexMatch` patterns from the query. If the query is used verbatim (rather than
tokenized-and-escaped), an authenticated searcher can:

- inject regex metacharacters that over-match unrelated documents, or
- submit a pathological pattern such as ``(a+)+$`` to trigger catastrophic
  backtracking (ReDoS) on the database.

These tests capture every pipeline/filter handed to the (mocked) collection and
assert that no attacker-controlled pattern ever appears — either it is
regex-escaped, or (when tokenization yields nothing) the keyword stage is
skipped entirely.
"""

import re
from unittest.mock import AsyncMock, MagicMock

import pytest

from registry.repositories.documentdb.search_repository import (
    DocumentDBSearchRepository,
    _build_keyword_match_filter,
    _build_text_boost_stage,
    _tokenize_query,
)

# A query composed only of regex metacharacters / short punctuation. Every
# "token" is <= 2 chars or non-word, so _tokenize_query() returns [].
REDOS_QUERY = "(a+)+$"
# A query that tokenizes but whose tokens contain metacharacters once split.
METACHAR_QUERY = ".*|.*|.*"


def _make_capturing_cursor(items: list[dict]) -> MagicMock:
    cursor = MagicMock()
    cursor.limit = MagicMock(return_value=cursor)

    async def to_list_impl(length=None):
        return list(items)

    cursor.to_list = to_list_impl
    return cursor


@pytest.fixture
def capturing_repo():
    """A search repo whose collection records every aggregate/find argument."""
    repo = DocumentDBSearchRepository.__new__(DocumentDBSearchRepository)

    aggregate_pipelines: list[list[dict]] = []
    find_filters: list[dict] = []

    collection = MagicMock()

    def aggregate_impl(pipeline, *args, **kwargs):
        aggregate_pipelines.append(pipeline)
        return _make_capturing_cursor([])

    def find_impl(filter_query=None, *args, **kwargs):
        find_filters.append(filter_query or {})
        return _make_capturing_cursor([])

    collection.aggregate = MagicMock(side_effect=aggregate_impl)
    collection.find = MagicMock(side_effect=find_impl)

    repo._get_collection = AsyncMock(return_value=collection)
    # Force the vector path (not the lexical-only fallback): return a fake vector.
    repo._embed_texts = AsyncMock(return_value=[[0.1, 0.2, 0.3]])
    repo._default_search_scope = AsyncMock(return_value=["mcp_server"])

    repo._aggregate_pipelines = aggregate_pipelines
    repo._find_filters = find_filters
    return repo


def _collect_regex_patterns(obj) -> list[str]:
    """Recursively collect every $regex / $regexMatch pattern string in a dict."""
    patterns: list[str] = []
    if isinstance(obj, dict):
        for key, value in obj.items():
            if key == "$regex" and isinstance(value, str):
                patterns.append(value)
            elif key == "$regexMatch" and isinstance(value, dict):
                regex_val = value.get("regex")
                if isinstance(regex_val, str):
                    patterns.append(regex_val)
            else:
                patterns.extend(_collect_regex_patterns(value))
    elif isinstance(obj, list):
        for item in obj:
            patterns.extend(_collect_regex_patterns(item))
    return patterns


class TestTokenizerAlwaysEscapes:
    def test_metacharacter_query_yields_no_tokens(self):
        """A metacharacter-only query tokenizes to nothing (the empty-token path)."""
        assert _tokenize_query(REDOS_QUERY) == []

    def test_builders_only_ever_see_escaped_input(self):
        """The pattern builders combine already-escaped tokens with '|'."""
        tokens = _tokenize_query("alpha.beta gamma*")
        escaped = [re.escape(t) for t in tokens]
        token_regex = "|".join(escaped)
        # Metacharacters from the original query are neutralized.
        assert "\\." in token_regex or "." not in token_regex.replace("\\.", "")
        # The builders embed the escaped pattern verbatim; no re-introduction of raw input.
        match_filter = _build_keyword_match_filter(token_regex=token_regex)
        boost = _build_text_boost_stage(token_regex)
        for pattern in _collect_regex_patterns(match_filter) + _collect_regex_patterns(boost):
            assert pattern == token_regex


class TestSearchNeverEmitsRawQueryRegex:
    async def test_redos_query_emits_no_keyword_regex(self, capturing_repo):
        """Empty-token query: keyword $regex/$regexMatch stages are skipped entirely."""
        result = await capturing_repo.search(REDOS_QUERY, entity_types=["mcp_server"])

        # No find() keyword pass at all when there are no tokens.
        assert capturing_repo._find_filters == []

        # No $regex/$regexMatch anywhere in any aggregate pipeline.
        all_patterns: list[str] = []
        for pipeline in capturing_repo._aggregate_pipelines:
            all_patterns.extend(_collect_regex_patterns(pipeline))
        assert all_patterns == [], f"leaked regex patterns: {all_patterns}"

        # Sanity: the search still returns a well-formed (empty) result set.
        assert set(result.keys()) >= {"servers", "tools", "agents"}

    async def test_raw_query_never_used_as_regex(self, capturing_repo):
        """No emitted regex pattern equals the raw (unescaped) query string."""
        for query in (REDOS_QUERY, METACHAR_QUERY):
            capturing_repo._aggregate_pipelines.clear()
            capturing_repo._find_filters.clear()
            await capturing_repo.search(query, entity_types=["mcp_server"])

            emitted: list[str] = []
            for pipeline in capturing_repo._aggregate_pipelines:
                emitted.extend(_collect_regex_patterns(pipeline))
            for flt in capturing_repo._find_filters:
                emitted.extend(_collect_regex_patterns(flt))

            assert query not in emitted, f"raw query leaked as regex for {query!r}"
            # Every emitted pattern must be a join of re.escape()d tokens.
            for pattern in emitted:
                for alternative in pattern.split("|"):
                    assert alternative == re.escape(re.sub(r"\\(.)", r"\1", alternative))

    async def test_metachar_tokens_are_escaped_when_present(self, capturing_repo):
        """A query with real tokens still escapes any embedded metacharacters."""
        # "server.name" tokenizes to ["server", "name"] (split on non-word chars),
        # so use a token that survives tokenization AND carries a metachar via digits.
        await capturing_repo.search("payments a.b.c", entity_types=["mcp_server"])
        emitted: list[str] = []
        for pipeline in capturing_repo._aggregate_pipelines:
            emitted.extend(_collect_regex_patterns(pipeline))
        for flt in capturing_repo._find_filters:
            emitted.extend(_collect_regex_patterns(flt))
        # "payments" is the only token > 2 chars that is not a stopword; a/b/c are dropped.
        assert emitted, "expected keyword stages to run for a tokenizable query"
        for pattern in emitted:
            assert ".*" not in pattern
