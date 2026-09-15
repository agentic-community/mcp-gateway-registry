"""Unit tests for the absolute similarity carried alongside relevance_score.

``relevance_score`` answers "where did this hit rank"; under RRF fusion it is
min-max rescaled so the best hit is 1.0 regardless of how alike the two things
actually are. ``similarity_score`` answers "how alike are they", which is what
the duplicate-check advisory needs (issue #1696).

Covers:
- The top RRF hit normalizes to 1.0 while its similarity stays low
- Hits are matched to their source document by path
- Entries with no path (tools lifted out of a parent server) are left alone
- A query that could not be embedded stamps nothing
"""

from registry.repositories.documentdb.search_repository import (
    _attach_similarity_scores,
    _normalize_scores,
    _reciprocal_rank_fusion,
)

QUERY = [1.0, 0.0, 0.0]
NEAR = [0.96, 0.28, 0.0]
FAR = [0.26, 0.97, 0.0]


def _doc(path: str, embedding: list[float]) -> dict:
    return {"_id": path, "path": path, "name": path.rsplit("/", 1)[-1], "embedding": embedding}


def test_display_score_of_one_can_accompany_a_low_similarity() -> None:
    """The regression this file exists for: rank 1.0 does not mean similar."""
    far = _doc("/servers/payroll", FAR)
    scored = _reciprocal_rank_fusion([far, _doc("/servers/weather", FAR)], [])
    normalized = _normalize_scores(scored, max_results=30)

    assert normalized[0][1] == 1.0

    grouped = {"servers": [{"path": "/servers/payroll", "relevance_score": 1.0}]}
    _attach_similarity_scores(grouped, normalized, QUERY)

    assert grouped["servers"][0]["relevance_score"] == 1.0
    assert grouped["servers"][0]["similarity_score"] < 0.4


def test_similarity_reflects_the_embedding_not_the_ranking() -> None:
    selected = [(_doc("/servers/near", NEAR), 1.0), (_doc("/servers/far", FAR), 0.0)]
    grouped = {
        "servers": [{"path": "/servers/near"}, {"path": "/servers/far"}],
    }
    _attach_similarity_scores(grouped, selected, QUERY)

    near, far = grouped["servers"]
    assert near["similarity_score"] > 0.9
    assert far["similarity_score"] < 0.4


def test_entries_without_a_path_are_left_untouched() -> None:
    selected = [(_doc("/servers/near", NEAR), 1.0)]
    grouped = {
        "servers": [{"path": "/servers/near"}],
        "tools": [{"server_path": "/servers/near", "tool_name": "do_thing"}],
    }
    _attach_similarity_scores(grouped, selected, QUERY)

    assert "similarity_score" in grouped["servers"][0]
    assert "similarity_score" not in grouped["tools"][0]


def test_unknown_path_is_left_untouched() -> None:
    selected = [(_doc("/servers/near", NEAR), 1.0)]
    grouped = {"servers": [{"path": "/servers/somewhere-else"}]}
    _attach_similarity_scores(grouped, selected, QUERY)

    assert "similarity_score" not in grouped["servers"][0]


def test_missing_query_embedding_stamps_nothing() -> None:
    selected = [(_doc("/servers/near", NEAR), 1.0)]
    grouped = {"servers": [{"path": "/servers/near"}]}
    _attach_similarity_scores(grouped, selected, None)

    assert "similarity_score" not in grouped["servers"][0]


def test_document_without_an_embedding_scores_zero() -> None:
    selected = [({"_id": "x", "path": "/servers/no-vector"}, 1.0)]
    grouped = {"servers": [{"path": "/servers/no-vector"}]}
    _attach_similarity_scores(grouped, selected, QUERY)

    assert grouped["servers"][0]["similarity_score"] == 0.0


# ---------------------------------------------------------------------------
# Issue #1752: the absolute similarity must survive the response model.
# Before this fix the number was computed and then dropped by Pydantic, because
# no response model declared the field.
# ---------------------------------------------------------------------------


def test_server_response_model_carries_similarity_score() -> None:
    from registry.api.search_routes import ServerSearchResult

    result = ServerSearchResult(
        path="/servers/near",
        server_name="near",
        relevance_score=1.0,
        similarity_score=0.6412,
    )

    assert result.model_dump()["similarity_score"] == 0.6412


def test_similarity_score_defaults_to_none_when_not_embedded() -> None:
    from registry.api.search_routes import ServerSearchResult

    result = ServerSearchResult(path="/servers/near", server_name="near", relevance_score=1.0)

    assert result.similarity_score is None


def test_similarity_score_accepts_a_negative_cosine() -> None:
    """Cosine similarity ranges [-1, 1]; a 0..1 bound would reject opposite vectors."""
    from registry.api.search_routes import ServerSearchResult

    result = ServerSearchResult(
        path="/servers/opposite",
        server_name="opposite",
        relevance_score=0.1,
        similarity_score=-0.83,
    )

    assert result.similarity_score == -0.83


def test_every_search_result_model_declares_it() -> None:
    from registry.api.search_routes import (
        AgentSearchResult,
        CustomEntitySearchResult,
        ServerSearchResult,
        SkillSearchResult,
        ToolSearchResult,
        VirtualServerSearchResult,
    )

    for model in (
        ServerSearchResult,
        ToolSearchResult,
        AgentSearchResult,
        SkillSearchResult,
        VirtualServerSearchResult,
        CustomEntitySearchResult,
    ):
        assert "similarity_score" in model.model_fields, model.__name__


def test_matching_tool_without_a_score_is_rejected() -> None:
    """A missing tool score means a search path stopped grading (issue #1752)."""
    import pytest
    from pydantic import ValidationError

    from registry.api.search_routes import MatchingToolResult

    with pytest.raises(ValidationError):
        MatchingToolResult(tool_name="get_current_time", description="Return the time")


def test_matching_tool_with_a_graded_score_is_accepted() -> None:
    from registry.api.search_routes import MatchingToolResult

    tool = MatchingToolResult(
        tool_name="get_current_time",
        description="Return the time",
        relevance_score=0.31,
    )

    assert tool.relevance_score == 0.31
