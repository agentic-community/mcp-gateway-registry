"""Unit tests for Reciprocal Rank Fusion (RRF) scoring.

Tests the _reciprocal_rank_fusion() function that replaces the broken
additive score formula which saturated everything to 1.0.

Covers:
- Basic two-list fusion
- Documents appearing in only one list (missing embeddings scenario)
- Documents appearing in both lists get boosted
- Correct ranking order
- Empty inputs
- Score differentiation (no saturation)
- Industry-standard k=60 behavior
"""

from registry.repositories.documentdb.search_repository import (
    RRF_K,
    SCORE_DISPLAY_FLOOR,
    _normalize_scores,
    _reciprocal_rank_fusion,
    _score_tool_relevance,
)


def _make_doc(
    doc_id: str,
    entity_type: str = "mcp_server",
    name: str | None = None,
) -> dict:
    """Create a minimal document dict for testing."""
    return {
        "_id": doc_id,
        "path": f"/{doc_id}",
        "entity_type": entity_type,
        "name": name or doc_id,
    }


class TestReciprocalRankFusion:
    """Tests for _reciprocal_rank_fusion()."""

    def test_empty_both_lists(self):
        """Empty inputs produce empty output."""
        result = _reciprocal_rank_fusion([], [])
        assert result == []

    def test_empty_vector_list(self):
        """Documents only in keyword list still appear in results."""
        kw_docs = [_make_doc("a"), _make_doc("b"), _make_doc("c")]
        result = _reciprocal_rank_fusion([], kw_docs)

        assert len(result) == 3
        ids = [doc["_id"] for doc, _ in result]
        assert ids == ["a", "b", "c"]

    def test_empty_keyword_list(self):
        """Documents only in vector list still appear in results."""
        vec_docs = [_make_doc("x"), _make_doc("y")]
        result = _reciprocal_rank_fusion(vec_docs, [])

        assert len(result) == 2
        ids = [doc["_id"] for doc, _ in result]
        assert ids == ["x", "y"]

    def test_document_in_both_lists_ranks_higher(self):
        """A document appearing in both lists gets a higher score than one in only one list."""
        shared_doc = _make_doc("shared")
        vec_only = _make_doc("vec_only")
        kw_only = _make_doc("kw_only")

        vector_ranked = [shared_doc, vec_only]
        keyword_ranked = [shared_doc, kw_only]

        result = _reciprocal_rank_fusion(vector_ranked, keyword_ranked)

        scores = {doc["_id"]: score for doc, score in result}
        assert scores["shared"] > scores["vec_only"]
        assert scores["shared"] > scores["kw_only"]

    def test_rank_1_in_both_beats_rank_1_in_one(self):
        """Doc ranked #1 in both lists scores higher than doc ranked #1 in only one."""
        top_both = _make_doc("top_both")
        top_vec = _make_doc("top_vec")
        top_kw = _make_doc("top_kw")

        vector_ranked = [top_both, top_vec]
        keyword_ranked = [top_both, top_kw]

        result = _reciprocal_rank_fusion(vector_ranked, keyword_ranked)

        scores = {doc["_id"]: score for doc, score in result}
        expected_top_score = 2.0 / (RRF_K + 1)
        assert abs(scores["top_both"] - expected_top_score) < 1e-10

    def test_scores_are_differentiated(self):
        """Documents at different combined rank positions get different scores.

        Unlike the old formula where everything saturated to 1.0,
        RRF produces meaningful score differences based on rank positions.
        """
        docs = [_make_doc(f"doc_{i}") for i in range(10)]

        vector_ranked = docs
        keyword_ranked = docs

        result = _reciprocal_rank_fusion(vector_ranked, keyword_ranked)

        scores = [score for _, score in result]
        unique_scores = set(scores)
        assert len(unique_scores) == len(scores)
        assert scores[0] > scores[-1]

    def test_k60_score_range(self):
        """With k=60 (default), scores are in expected range."""
        docs = [_make_doc(f"d{i}") for i in range(10)]

        result = _reciprocal_rank_fusion(docs, docs)

        max_possible = 2.0 / (60 + 1)
        min_possible = 2.0 / (60 + 10)

        for _, score in result:
            assert 0 < score <= max_possible
            assert score >= min_possible

    def test_missing_embedding_document_still_found(self):
        """Documents without embeddings (only in keyword list) are findable.

        This is the key fix: the old formula gave these docs score 0.5
        (from (0+1)/2) which saturated to 1.0 with any text_boost,
        or excluded them entirely. RRF ranks them by keyword position.
        """
        embedded_server = _make_doc("embedded", "mcp_server")
        no_embedding_server = _make_doc("no_embed", "mcp_server")

        vector_ranked = [embedded_server]
        keyword_ranked = [no_embedding_server, embedded_server]

        result = _reciprocal_rank_fusion(vector_ranked, keyword_ranked)

        ids = [doc["_id"] for doc, _ in result]
        assert "no_embed" in ids
        assert "embedded" in ids

    def test_result_order_is_descending(self):
        """Results are sorted by score descending."""
        docs = [_make_doc(f"d{i}") for i in range(5)]
        result = _reciprocal_rank_fusion(docs, list(reversed(docs)))

        scores = [score for _, score in result]
        assert scores == sorted(scores, reverse=True)

    def test_custom_k_value(self):
        """Custom k parameter changes score sensitivity."""
        docs = [_make_doc(f"d{i}") for i in range(5)]

        result_k2 = _reciprocal_rank_fusion(docs, docs, k=2)
        result_k60 = _reciprocal_rank_fusion(docs, docs, k=60)

        scores_k2 = [s for _, s in result_k2]
        scores_k60 = [s for _, s in result_k60]

        spread_k2 = scores_k2[0] - scores_k2[-1]
        spread_k60 = scores_k60[0] - scores_k60[-1]

        assert spread_k2 > spread_k60

    def test_large_result_set(self):
        """RRF handles hundreds of documents efficiently."""
        vec_docs = [_make_doc(f"v{i}") for i in range(200)]
        kw_docs = [_make_doc(f"k{i}") for i in range(100)]

        result = _reciprocal_rank_fusion(vec_docs, kw_docs)

        assert len(result) == 300
        scores = [s for _, s in result]
        assert scores == sorted(scores, reverse=True)

    def test_duplicate_ids_across_lists_merged(self):
        """Same document in both lists is merged (not duplicated)."""
        doc_a = _make_doc("same_id")
        doc_b = _make_doc("same_id")

        result = _reciprocal_rank_fusion([doc_a], [doc_b])

        assert len(result) == 1
        assert result[0][0]["_id"] == "same_id"

    def test_realistic_scenario(self):
        """Simulate the Expedia 'atlassian' query scenario.

        Atlassian server has no embedding (only in keyword list).
        Other agents/skills have embeddings and keyword matches.
        With RRF, the Atlassian server should appear in results.
        """
        atlassian_server = _make_doc("atlassian", "mcp_server", "Atlassian")
        agent_1 = _make_doc("agent_jira", "a2a_agent", "Jira Agent")
        agent_2 = _make_doc("agent_conf", "a2a_agent", "Confluence Agent")
        skill_1 = _make_doc("skill_jira", "skill", "Jira Skill")

        vector_ranked = [agent_1, agent_2, skill_1]
        keyword_ranked = [atlassian_server, agent_1, skill_1, agent_2]

        result = _reciprocal_rank_fusion(vector_ranked, keyword_ranked)

        ids = [doc["_id"] for doc, _ in result]
        assert "atlassian" in ids

        scores = {doc["_id"]: score for doc, score in result}
        assert scores["atlassian"] > 0


class TestNormalizeScores:
    """Tests for _normalize_scores()."""

    def test_empty_input(self):
        """Empty list returns empty."""
        assert _normalize_scores([]) == []

    def test_single_result(self):
        """Single result normalizes to 1.0."""
        result = _normalize_scores([(_make_doc("a"), 0.016)])
        assert result[0][1] == 1.0

    def test_two_results(self):
        """Two results kept when fewer than max_results even if below floor."""
        result = _normalize_scores([
            (_make_doc("top"), 0.033),
            (_make_doc("bot"), 0.016),
        ], max_results=10)
        assert result[0][1] == 1.0
        assert len(result) == 2

    def test_preserves_order(self):
        """Normalization preserves descending order."""
        result = _normalize_scores([
            (_make_doc("a"), 0.033),
            (_make_doc("b"), 0.025),
            (_make_doc("c"), 0.020),
            (_make_doc("d"), 0.016),
        ])
        scores = [s for _, s in result]
        assert scores == sorted(scores, reverse=True)

    def test_filters_below_floor_when_enough_results(self):
        """Results below floor are excluded when enough remain above it."""
        result = _normalize_scores([
            (_make_doc(f"d{i}"), 0.03 - i * 0.001) for i in range(20)
        ], max_results=5)
        for _, score in result:
            assert SCORE_DISPLAY_FLOOR <= score <= 1.0
        assert len(result) < 20

    def test_keeps_below_floor_when_too_few(self):
        """Results below floor are kept when dropping would leave fewer than max_results."""
        result = _normalize_scores([
            (_make_doc("top"), 0.033),
            (_make_doc("mid"), 0.020),
            (_make_doc("bot"), 0.016),
        ], max_results=10)
        assert len(result) == 3

    def test_equal_scores_all_get_one(self):
        """If all scores are equal, all get 1.0."""
        result = _normalize_scores([
            (_make_doc("a"), 0.02),
            (_make_doc("b"), 0.02),
            (_make_doc("c"), 0.02),
        ])
        for _, score in result:
            assert score == 1.0


class TestScoreToolRelevance:
    """Tests for _score_tool_relevance()."""

    def test_no_tokens(self):
        """Empty tokens returns 0."""
        assert _score_tool_relevance("search", "searches stuff", []) == 0.0

    def test_name_match(self):
        """Token matching tool name gives high score."""
        score = _score_tool_relevance("web_search_exa", "Web search", ["search"])
        assert score > 0.5

    def test_description_only_match(self):
        """Token matching only description gives lower score."""
        score = _score_tool_relevance("get_data", "Fetches search results", ["search"])
        assert 0.0 < score < 0.8

    def test_no_match(self):
        """No token overlap returns 0."""
        score = _score_tool_relevance("get_weather", "Returns forecast", ["database"])
        assert score == 0.0

    def test_multiple_token_match(self):
        """Multiple tokens matching gives higher score."""
        score_one = _score_tool_relevance(
            "search_docs", "Search documentation", ["search"]
        )
        score_both = _score_tool_relevance(
            "search_docs", "Search documentation", ["search", "docs"]
        )
        assert score_both > score_one

    def test_score_capped_at_one(self):
        """Score never exceeds 1.0."""
        score = _score_tool_relevance(
            "search_web_crawl_fetch",
            "search web crawl fetch extract data",
            ["search", "web", "crawl", "fetch", "extract", "data"],
        )
        assert score <= 1.0

    def test_no_inherited_server_score(self):
        """Tools that don't match query get 0, not a server-inherited score."""
        score = _score_tool_relevance("getStats", "Get activity stats", ["strava"])
        assert score == 0.0
