"""Unit tests for search result distribution logic.

Tests the _distribute_results() function and _tool_extraction_limit() helper
that replace the old hardcoded cap of 3 per entity type with global ranking
and competitive soft caps.

Covers:
- Empty results
- Single-type dominance (no artificial cap when no competition)
- Multi-type competition with soft cap enforcement
- Soft cap lifted when no other types remain
- Edge cases (max_results=1, max_results >= total)
- Backward compatibility with default max_results=10
- Tool extraction limit scaling
"""

import math

from registry.repositories.documentdb.search_repository import (
    SOFT_CAP_RATIO,
    _distribute_results,
    _tool_extraction_limit,
)

# =============================================================================
# HELPERS
# =============================================================================


def _make_doc(
    entity_type: str,
    name: str,
    score: float,
) -> tuple[dict, float]:
    """Create a (doc, score) tuple for testing.

    Args:
        entity_type: Entity type string (e.g. "mcp_server")
        name: Document name for identification
        score: Relevance score

    Returns:
        Tuple of (doc_dict, score)
    """
    return (
        {"entity_type": entity_type, "name": name},
        score,
    )


def _make_servers(
    count: int,
    start_score: float = 0.95,
    step: float = 0.02,
) -> list[tuple[dict, float]]:
    """Create a list of server result tuples with descending scores.

    Args:
        count: Number of servers to create
        start_score: Score of the first server
        step: Score decrement per server

    Returns:
        List of (doc, score) tuples sorted by score descending
    """
    return [
        _make_doc("mcp_server", f"server-{i}", round(start_score - i * step, 4))
        for i in range(count)
    ]


def _make_agents(
    count: int,
    start_score: float = 0.80,
    step: float = 0.05,
) -> list[tuple[dict, float]]:
    """Create a list of agent result tuples with descending scores."""
    return [
        _make_doc("a2a_agent", f"agent-{i}", round(start_score - i * step, 4)) for i in range(count)
    ]


def _make_tools(
    count: int,
    start_score: float = 0.75,
    step: float = 0.05,
) -> list[tuple[dict, float]]:
    """Create a list of tool result tuples with descending scores."""
    return [
        _make_doc("mcp_tool", f"tool-{i}", round(start_score - i * step, 4)) for i in range(count)
    ]


def _make_skills(
    count: int,
    start_score: float = 0.70,
    step: float = 0.05,
) -> list[tuple[dict, float]]:
    """Create a list of skill result tuples with descending scores."""
    return [
        _make_doc("skill", f"skill-{i}", round(start_score - i * step, 4)) for i in range(count)
    ]


def _count_types(
    results: list[tuple[dict, float]],
) -> dict[str, int]:
    """Count results per entity type.

    Args:
        results: List of (doc, score) tuples

    Returns:
        Dict mapping entity_type to count
    """
    counts: dict[str, int] = {}
    for doc, _ in results:
        entity_type = doc.get("entity_type", "")
        counts[entity_type] = counts.get(entity_type, 0) + 1
    return counts


# =============================================================================
# TESTS: _distribute_results()
# =============================================================================


class TestDistributeResults:
    """Tests for the _distribute_results() function."""

    def test_empty_results(self):
        """Empty input returns empty output."""
        result = _distribute_results([], 10)
        assert result == []

    def test_zero_max_results(self):
        """max_results=0 returns empty output."""
        scored = _make_servers(5)
        result = _distribute_results(scored, 0)
        assert result == []

    def test_single_type_no_cap(self):
        """Only servers in results -- all slots go to servers, no artificial limit."""
        servers = _make_servers(20)
        result = _distribute_results(servers, 10)

        assert len(result) == 10
        counts = _count_types(result)
        assert counts["mcp_server"] == 10

    def test_single_type_respects_max_results(self):
        """20 servers with max_results=10 returns exactly 10."""
        servers = _make_servers(20)
        result = _distribute_results(servers, 10)
        assert len(result) == 10

    def test_single_type_fewer_than_max(self):
        """5 servers with max_results=10 returns all 5."""
        servers = _make_servers(5)
        result = _distribute_results(servers, 10)

        assert len(result) == 5
        counts = _count_types(result)
        assert counts["mcp_server"] == 5

    def test_mixed_types_global_ranking(self):
        """Higher-relevance items win regardless of type."""
        # 3 servers at 0.95, 0.93, 0.91
        # 3 agents at 0.80, 0.75, 0.70
        scored = _make_servers(3) + _make_agents(3)
        scored.sort(key=lambda x: x[1], reverse=True)

        result = _distribute_results(scored, 6)
        assert len(result) == 6

        # First 3 should be servers (highest scores)
        for doc, _ in result[:3]:
            assert doc["entity_type"] == "mcp_server"

    def test_soft_cap_enforced(self):
        """Dominant type capped at 60% when other types have results."""
        # 10 servers (0.95 to 0.77) + 5 agents (0.80 to 0.60)
        servers = _make_servers(10)
        agents = _make_agents(5)
        scored = servers + agents
        scored.sort(key=lambda x: x[1], reverse=True)

        max_results = 10
        soft_cap = math.ceil(max_results * SOFT_CAP_RATIO)  # 6

        result = _distribute_results(scored, max_results)
        assert len(result) == max_results

        counts = _count_types(result)
        # Servers should be capped at soft_cap since agents are competing
        assert counts["mcp_server"] <= soft_cap
        # Agents should have gotten some slots
        assert counts.get("a2a_agent", 0) > 0

    def test_soft_cap_lifted_when_no_competition(self):
        """Cap removed when only one type remains in the tail."""
        # 15 servers (high scores) + 1 agent (lower score)
        servers = _make_servers(15, start_score=0.95, step=0.02)
        agents = [_make_doc("a2a_agent", "agent-0", 0.50)]
        scored = servers + agents
        scored.sort(key=lambda x: x[1], reverse=True)

        max_results = 10
        soft_cap = math.ceil(max_results * SOFT_CAP_RATIO)  # 6

        result = _distribute_results(scored, max_results)
        assert len(result) == max_results

        counts = _count_types(result)
        # Agent should be included (it's in the top candidates)
        # But servers should get more than soft_cap since after the agent
        # there are no more agents remaining, so the cap is lifted
        assert counts["mcp_server"] >= soft_cap

    def test_max_results_1(self):
        """Edge case: max_results=1 returns exactly 1 result."""
        scored = _make_servers(5) + _make_agents(3)
        scored.sort(key=lambda x: x[1], reverse=True)

        result = _distribute_results(scored, 1)
        assert len(result) == 1
        # Should be the highest scored item
        assert result[0][1] == max(s for _, s in scored)

    def test_max_results_equals_total(self):
        """max_results >= total results returns all results."""
        servers = _make_servers(3)
        agents = _make_agents(2)
        scored = servers + agents
        scored.sort(key=lambda x: x[1], reverse=True)

        result = _distribute_results(scored, 100)
        assert len(result) == 5  # All results returned

    def test_backward_compatible_default(self):
        """max_results=10 with mixed types produces diverse results."""
        servers = _make_servers(8, start_score=0.95)
        agents = _make_agents(5, start_score=0.80)
        tools = _make_tools(4, start_score=0.75)
        scored = servers + agents + tools
        scored.sort(key=lambda x: x[1], reverse=True)

        result = _distribute_results(scored, 10)
        assert len(result) == 10

        counts = _count_types(result)
        # Should have diversity -- at least 2 types present
        assert len(counts) >= 2
        # No type should have more than 6 (soft cap for max_results=10)
        for count in counts.values():
            assert count <= math.ceil(10 * SOFT_CAP_RATIO)

    def test_entity_types_filter_single(self):
        """When only one entity_type in input, all slots go to it."""
        agents = _make_agents(20, start_score=0.90, step=0.01)
        result = _distribute_results(agents, 15)

        assert len(result) == 15
        counts = _count_types(result)
        assert counts["a2a_agent"] == 15

    def test_results_contain_highest_scores(self):
        """Selected results include the highest-scoring items available."""
        scored = _make_servers(5) + _make_agents(5)
        scored.sort(key=lambda x: x[1], reverse=True)

        result = _distribute_results(scored, 8)
        result_scores = sorted([s for _, s in result], reverse=True)
        # The top score from the input should be in the results
        assert result_scores[0] == scored[0][1]
        assert len(result) == 8

    def test_three_types_competing(self):
        """Three entity types compete fairly."""
        servers = _make_servers(10, start_score=0.95)
        agents = _make_agents(8, start_score=0.85)
        tools = _make_tools(6, start_score=0.75)
        scored = servers + agents + tools
        scored.sort(key=lambda x: x[1], reverse=True)

        result = _distribute_results(scored, 15)
        assert len(result) == 15

        counts = _count_types(result)
        soft_cap = math.ceil(15 * SOFT_CAP_RATIO)  # 9

        # All three types should be represented
        assert len(counts) == 3
        # No type exceeds soft cap (since all three have results)
        for count in counts.values():
            assert count <= soft_cap

    def test_five_types_all_present(self):
        """All five entity types get fair representation."""
        servers = _make_servers(5, start_score=0.95)
        agents = _make_agents(5, start_score=0.85)
        tools = _make_tools(5, start_score=0.75)
        skills = _make_skills(5, start_score=0.65)
        virtual = [
            _make_doc("virtual_server", f"vs-{i}", round(0.60 - i * 0.05, 4)) for i in range(5)
        ]
        scored = servers + agents + tools + skills + virtual
        scored.sort(key=lambda x: x[1], reverse=True)

        result = _distribute_results(scored, 20)
        assert len(result) == 20

        counts = _count_types(result)
        # All 5 types should be present
        assert len(counts) == 5

    def test_small_max_results_5(self):
        """max_results=5 with mixed types -- soft_cap=3."""
        servers = _make_servers(8, start_score=0.95)
        agents = _make_agents(5, start_score=0.80)
        scored = servers + agents
        scored.sort(key=lambda x: x[1], reverse=True)

        result = _distribute_results(scored, 5)
        assert len(result) == 5

        counts = _count_types(result)
        soft_cap = math.ceil(5 * SOFT_CAP_RATIO)  # 3
        # Servers should be capped at 3 since agents are competing
        assert counts["mcp_server"] <= soft_cap
        # Agents should get remaining slots
        assert counts.get("a2a_agent", 0) > 0


# =============================================================================
# TESTS: _tool_extraction_limit()
# =============================================================================


class TestToolExtractionLimit:
    """Tests for the _tool_extraction_limit() helper."""

    def test_default_max_results(self):
        """max_results=10 gives ceil(10*0.6)=6, which is >=3."""
        result = _tool_extraction_limit(10)
        assert result == 6

    def test_small_max_results(self):
        """max_results=1 still returns at least 3 (backward compat)."""
        result = _tool_extraction_limit(1)
        assert result == 3

    def test_max_results_3(self):
        """max_results=3 gives ceil(3*0.6)=2, floor is 3."""
        result = _tool_extraction_limit(3)
        assert result == 3

    def test_large_max_results(self):
        """max_results=50 gives ceil(50*0.6)=30."""
        result = _tool_extraction_limit(50)
        assert result == 30

    def test_never_below_3(self):
        """Tool limit never goes below 3 regardless of max_results."""
        for max_results in range(1, 10):
            assert _tool_extraction_limit(max_results) >= 3

    def test_scales_with_max_results(self):
        """Larger max_results produces larger tool limit."""
        limit_10 = _tool_extraction_limit(10)
        limit_50 = _tool_extraction_limit(50)
        assert limit_50 > limit_10


# =============================================================================
# TESTS: SOFT_CAP_RATIO constant
# =============================================================================


class TestSoftCapRatio:
    """Tests for the SOFT_CAP_RATIO constant value."""

    def test_ratio_value(self):
        """SOFT_CAP_RATIO is 0.6 as designed."""
        assert SOFT_CAP_RATIO == 0.6

    def test_ratio_produces_expected_caps(self):
        """Verify soft cap values for common max_results values."""
        assert math.ceil(10 * SOFT_CAP_RATIO) == 6
        assert math.ceil(5 * SOFT_CAP_RATIO) == 3
        assert math.ceil(50 * SOFT_CAP_RATIO) == 30
        assert math.ceil(1 * SOFT_CAP_RATIO) == 1
