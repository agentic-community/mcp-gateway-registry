"""One vector search per hybrid search, over-requested past the post-filters.

Issue #1751. The repository used to run one ``$search`` aggregation per entity
type, awaited in sequence. That cost up to five round trips and bought nothing,
because ``$search`` selects the nearest ``k`` documents *before* any ``$match``
runs: every pipeline ran the same global search and only differed in which type
it kept afterwards. Measured on DocumentDB, the union of all per-type pipelines
was exactly ``k``, never ``k`` per type.

Three things are pinned here, each of which regressed silently before:

1. Exactly one ``$search`` stage per search, whatever the entity-type count.
2. ``k`` over-requests to cover what the post-filters discard, capped by
   ``efSearch`` (the HNSW queue cannot return more candidates than it holds).
3. No ``$sort`` on ``text_boost`` inside the pipeline, so the documents reach
   ``_reciprocal_rank_fusion()`` in similarity order, which is the input its
   docstring promises.

Nothing in the suite asserted the old per-type behaviour, which is why removing
it broke no test. These exist so the reverse is not true.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from registry.core.config import settings
from registry.repositories.documentdb.search_repository import DocumentDBSearchRepository


def _make_cursor(items: list[dict]) -> MagicMock:
    cursor = MagicMock()
    cursor.limit = MagicMock(return_value=cursor)

    async def to_list_impl(length=None):
        return list(items)

    cursor.to_list = to_list_impl
    return cursor


@pytest.fixture
def repo_and_pipelines():
    """A repo whose aggregate() calls are captured rather than executed."""
    instance = DocumentDBSearchRepository.__new__(DocumentDBSearchRepository)
    pipelines: list[list[dict]] = []

    def capture(pipeline, *args, **kwargs):
        pipelines.append(pipeline)
        return _make_cursor([])

    collection = MagicMock()
    collection.aggregate = MagicMock(side_effect=capture)
    collection.find = MagicMock(return_value=_make_cursor([]))
    collection.count_documents = AsyncMock(return_value=400)

    instance._get_collection = AsyncMock(return_value=collection)
    instance._embed_texts = AsyncMock(return_value=[[0.1, 0.2, 0.3]])
    instance._doc_count_cache = None
    return instance, pipelines


def _search_stages(pipelines: list[list[dict]]) -> list[dict]:
    """Every $search stage across every captured pipeline."""
    return [stage for pipeline in pipelines for stage in pipeline if "$search" in stage]


class TestOneQueryNotPerType:
    """The round-trip count must not scale with the entity-type count."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "entity_types",
        [
            ["mcp_server"],
            ["mcp_server", "a2a_agent"],
            ["mcp_server", "a2a_agent", "skill", "virtual_server"],
        ],
    )
    async def test_one_search_stage_regardless_of_type_count(
        self,
        repo_and_pipelines,
        entity_types,
    ) -> None:
        repo, pipelines = repo_and_pipelines
        repo._default_search_scope = AsyncMock(return_value=entity_types)

        await repo.search("time in tokyo", entity_types=entity_types, max_results=10)

        assert len(_search_stages(pipelines)) == 1

    @pytest.mark.asyncio
    async def test_all_requested_types_go_into_one_match(self, repo_and_pipelines) -> None:
        """The types are filtered with a single $in rather than N pipelines."""
        repo, pipelines = repo_and_pipelines
        types = ["mcp_server", "a2a_agent", "skill"]
        repo._default_search_scope = AsyncMock(return_value=types)

        await repo.search("time in tokyo", entity_types=types, max_results=10)

        vector_pipeline = next(p for p in pipelines if any("$search" in s for s in p))
        type_matches = [
            s["$match"]["entity_type"]
            for s in vector_pipeline
            if "$match" in s and "entity_type" in s["$match"]
        ]
        assert type_matches == [{"$in": types}]

    @pytest.mark.asyncio
    async def test_pipeline_does_not_sort_by_text_boost(self, repo_and_pipelines) -> None:
        """Sorting here would hand RRF a list ordered by keyword score.

        _reciprocal_rank_fusion() treats list position as the vector rank, so the
        documents must arrive in similarity order. The old pipeline sorted by
        text_boost, which silently made the keyword signal drive the vector rank.
        """
        repo, pipelines = repo_and_pipelines
        repo._default_search_scope = AsyncMock(return_value=["mcp_server"])

        await repo.search("time in tokyo", max_results=10)

        vector_pipeline = next(p for p in pipelines if any("$search" in s for s in p))
        sorts = [s["$sort"] for s in vector_pipeline if "$sort" in s]
        assert all("text_boost" not in sort for sort in sorts)


class TestOverRequest:
    """k must cover what the post-filters throw away, and respect the queue."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize("max_results", [1, 5, 10, 50])
    async def test_k_is_max_results_times_overrequest_capped_by_the_budget(
        self,
        repo_and_pipelines,
        max_results,
    ) -> None:
        repo, pipelines = repo_and_pipelines
        repo._default_search_scope = AsyncMock(return_value=["mcp_server"])

        await repo.search("time in tokyo", max_results=max_results)

        vector = _search_stages(pipelines)[0]["$search"]["vectorSearch"]
        expected = min(
            max_results * settings.vector_search_overrequest,
            settings.vector_search_ef_search,
        )
        assert vector["k"] == expected

    @pytest.mark.asyncio
    @pytest.mark.parametrize("max_results", [1, 5, 10, 20, 50])
    async def test_ef_search_is_the_configured_budget(
        self,
        repo_and_pipelines,
        max_results,
    ) -> None:
        """efSearch is a query-time parameter, currently the configured ceiling.

        Sizing it down for small k was tried and reverted. Measured on DocumentDB
        with efSearch pinned at 1000, vector-stage latency tracked k (52ms at
        k=20, 61ms at k=200, 155ms at k=1000), so the cost is materialising
        documents rather than graph traversal.
        """
        repo, pipelines = repo_and_pipelines
        repo._default_search_scope = AsyncMock(return_value=["mcp_server"])

        await repo.search("time in tokyo", max_results=max_results)

        vector = _search_stages(pipelines)[0]["$search"]["vectorSearch"]
        assert vector["efSearch"] == settings.vector_search_ef_search
        assert vector["efSearch"] >= vector["k"], "the queue must hold at least k"


class TestNoSearchableTypes:
    """entity_types=["tool"] has no standalone documents to match."""

    @pytest.mark.asyncio
    async def test_tool_only_request_skips_the_vector_search(
        self,
        repo_and_pipelines,
    ) -> None:
        """Paying for an ANN traversal whose $match rejects everything is waste."""
        repo, pipelines = repo_and_pipelines
        repo._default_search_scope = AsyncMock(return_value=["mcp_server"])
        repo._lexical_only_search = AsyncMock(
            return_value={
                "servers": [],
                "tools": [],
                "agents": [],
                "skills": [],
                "virtual_servers": [],
                "custom": [],
            }
        )

        await repo.search("time in tokyo", entity_types=["tool"], max_results=10)

        assert _search_stages(pipelines) == []
        repo._lexical_only_search.assert_awaited_once()
