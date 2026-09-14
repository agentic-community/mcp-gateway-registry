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
