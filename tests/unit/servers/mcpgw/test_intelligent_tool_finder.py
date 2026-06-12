"""Unit tests for intelligent_tool_finder in servers/mcpgw/server.py.

Tests verify the fix for GitHub Issue #682: top_n parameter was ignored
due to wrong field names in the HTTP request and missing client-side truncation.
"""

import sys
import types
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# The mcpgw server depends on `fastmcp` which is not installed in the main
# project venv. Stub it out before importing the server module.
# FastMCP.tool() is a decorator — make it a passthrough so the original
# async functions remain callable.
_fastmcp_stub = types.ModuleType("fastmcp")
_fastmcp_stub.Context = type("Context", (), {})
_mock_mcp = MagicMock()
_mock_mcp.tool.return_value = lambda fn: fn  # decorator is a no-op
_fastmcp_stub.FastMCP = MagicMock(return_value=_mock_mcp)
sys.modules["fastmcp"] = _fastmcp_stub

# Force re-import of the server module with the stub in place
sys.modules.pop("servers.mcpgw.server", None)

# Add servers/mcpgw to sys.path so that `from models import ...` works
# when importing servers.mcpgw.server
_mcpgw_path = str(Path(__file__).resolve().parents[4] / "servers" / "mcpgw")
if _mcpgw_path not in sys.path:
    sys.path.insert(0, _mcpgw_path)

from servers.mcpgw.server import (
    _build_discovery_receipt,
    _dedupe_candidates,
    _validate_top_n,
    intelligent_tool_finder,
    search_registry,
)


def _make_mock_response(servers=None, status_code=200):
    """Create a mock httpx response with the given servers payload."""
    mock_resp = MagicMock()
    mock_resp.status_code = status_code
    mock_resp.raise_for_status = MagicMock()
    mock_resp.json.return_value = {"servers": servers or []}
    return mock_resp


def _make_server_with_tools(n_tools, server_name="test-server", path="/test"):
    """Create a mock server dict with n_tools matching_tools."""
    return {
        "server_name": server_name,
        "path": path,
        "matching_tools": [
            {
                "tool_name": f"tool_{i}",
                "description": f"Tool {i} description",
                "relevance_score": round(1.0 - i * 0.05, 2),
            }
            for i in range(n_tools)
        ],
    }


async def _call_with_mocked_registry(tool_func, mock_response, capture=None, **kwargs):
    """Call a registry search tool with mocked HTTP client and token."""
    captured_kwargs = {}

    async def mock_post(url, **post_kwargs):
        captured_kwargs.update(post_kwargs)
        return mock_response

    mock_client = AsyncMock()
    mock_client.post = mock_post
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)

    with (
        patch("servers.mcpgw.server.httpx.AsyncClient", return_value=mock_client),
        patch("servers.mcpgw.server._extract_bearer_token", return_value="test-token"),
    ):
        result = await tool_func(**kwargs)

    if capture is not None:
        capture.update(captured_kwargs)

    return result


async def _call_finder(
    mock_response,
    query="test",
    top_n=None,
    capture=None,
    include_discovery_receipt=False,
):
    """Helper to call intelligent_tool_finder with mocked HTTP client and token.

    Args:
        mock_response: The mock httpx response to return from POST.
        query: Search query string.
        top_n: Number of results (omit to use default).
        capture: If provided, a dict that will be populated with the POST kwargs.

    Returns:
        The result dict from intelligent_tool_finder.
    """
    kwargs = {"query": query}
    if top_n is not None:
        kwargs["top_n"] = top_n
    if include_discovery_receipt:
        kwargs["include_discovery_receipt"] = True
    return await _call_with_mocked_registry(
        intelligent_tool_finder,
        mock_response,
        capture=capture,
        **kwargs,
    )


async def _call_search_registry(
    mock_response,
    query="test",
    max_results=10,
    capture=None,
    include_discovery_receipt=False,
):
    """Helper to call search_registry with mocked HTTP client and token."""
    return await _call_with_mocked_registry(
        search_registry,
        mock_response,
        capture=capture,
        query=query,
        max_results=max_results,
        include_discovery_receipt=include_discovery_receipt,
    )


# ---------------------------------------------------------------------------
# test_request_payload_uses_correct_field_names
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_request_payload_uses_correct_field_names():
    """Verify POST body uses max_results and entity_types (not top_k / entity_type)."""
    mock_resp = _make_mock_response(servers=[])
    captured = {}

    await _call_finder(mock_resp, query="test", top_n=7, capture=captured)

    body = captured["json"]
    assert "max_results" in body
    assert body["max_results"] == 7
    assert "entity_types" in body
    assert body["entity_types"] == ["mcp_server", "tool", "a2a_agent", "skill", "virtual_server"]
    assert "top_k" not in body
    assert "entity_type" not in body


# ---------------------------------------------------------------------------
# test_top_n_limits_results
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_top_n_limits_results():
    """With 10 tools available and top_n=3, only 3 results should be returned."""
    server = _make_server_with_tools(10)
    mock_resp = _make_mock_response(servers=[server])

    result = await _call_finder(mock_resp, top_n=3)

    assert len(result["results"]) == 3
    assert result["total_results"] == 3


# ---------------------------------------------------------------------------
# test_top_n_default_value
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_top_n_default_value():
    """Without specifying top_n, default (5) should limit results."""
    server = _make_server_with_tools(10)
    mock_resp = _make_mock_response(servers=[server])

    result = await _call_finder(mock_resp)  # no top_n → default 5

    assert len(result["results"]) <= 5


# ---------------------------------------------------------------------------
# test_top_n_equals_result_count
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_top_n_equals_result_count():
    """When registry returns exactly top_n tools, all should be returned."""
    server = _make_server_with_tools(3)
    mock_resp = _make_mock_response(servers=[server])

    result = await _call_finder(mock_resp, top_n=3)

    assert len(result["results"]) == 3
    assert result["total_results"] == 3


# ---------------------------------------------------------------------------
# test_top_n_greater_than_results
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_top_n_greater_than_results():
    """When registry returns fewer than top_n, return what's available (no padding)."""
    server = _make_server_with_tools(2)
    mock_resp = _make_mock_response(servers=[server])

    result = await _call_finder(mock_resp, top_n=10)

    assert len(result["results"]) == 2


# ---------------------------------------------------------------------------
# test_top_n_validation_rejects_out_of_bounds
# ---------------------------------------------------------------------------


def test_top_n_validation_rejects_out_of_bounds():
    """_validate_top_n rejects values outside [1, 50] and accepts boundaries."""
    with pytest.raises(ValueError):
        _validate_top_n(0)

    with pytest.raises(ValueError):
        _validate_top_n(51)

    with pytest.raises(ValueError):
        _validate_top_n(-1)

    assert _validate_top_n(50) == 50
    assert _validate_top_n(1) == 1


# ---------------------------------------------------------------------------
# test_total_results_matches_truncated_list
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_total_results_matches_truncated_list():
    """total_results must equal len(results) after truncation to top_n."""
    # 2 servers × 4 tools each = 8 total tools
    server_a = _make_server_with_tools(4, server_name="server-a", path="/a")
    server_b = _make_server_with_tools(4, server_name="server-b", path="/b")
    mock_resp = _make_mock_response(servers=[server_a, server_b])

    result = await _call_finder(mock_resp, top_n=5)

    assert result["total_results"] == len(result["results"])
    assert result["total_results"] == 5


# ---------------------------------------------------------------------------
# test_discovery_receipt_records_exposed_and_withheld_tools
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_discovery_receipt_records_exposed_and_withheld_tools():
    """Opt-in receipt shows visible results and candidates kept out by top_n."""
    server = _make_server_with_tools(4, server_name="server-a", path="/a")
    mock_resp = _make_mock_response(servers=[server])

    result = await _call_finder(
        mock_resp,
        query="weather lookup",
        top_n=2,
        include_discovery_receipt=True,
    )

    receipt = result["discovery_receipt"]
    assert receipt["event"] == "registry.discovery_receipt"
    assert receipt["query"] == "weather lookup"
    assert receipt["limits"] == {"max_results": 2}
    assert receipt["exposed_results"] == [
        {"asset_type": "tool", "service_path": "/a", "name": "tool_0", "similarity_score": 1.0},
        {"asset_type": "tool", "service_path": "/a", "name": "tool_1", "similarity_score": 0.95},
    ]
    assert receipt["withheld"] == {
        "candidate_result_count": 2,
        "reason": "outside_intent_or_budget",
        "top_withheld": [
            {"asset_type": "tool", "service_path": "/a", "name": "tool_2", "similarity_score": 0.9},
            {
                "asset_type": "tool",
                "service_path": "/a",
                "name": "tool_3",
                "similarity_score": 0.85,
            },
        ],
    }
    assert receipt["stop_reason"] == "results_returned"


# ---------------------------------------------------------------------------
# test_discovery_receipt_is_opt_in
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_discovery_receipt_is_opt_in():
    """Default responses stay backward-compatible without receipt tokens."""
    server = _make_server_with_tools(1)
    mock_resp = _make_mock_response(servers=[server])

    finder_result = await _call_finder(mock_resp, top_n=1)
    assert "discovery_receipt" not in finder_result

    search_result = await _call_search_registry(mock_resp, max_results=1)
    assert "discovery_receipt" not in search_result


# ---------------------------------------------------------------------------
# test_search_registry_receipt_counts_withheld_matching_tools
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_search_registry_receipt_counts_withheld_matching_tools():
    """search_registry receipt counts all matching tools, not just servers."""
    server = _make_server_with_tools(4, server_name="server-a", path="/a")
    mock_resp = _make_mock_response(servers=[server])

    result = await _call_search_registry(
        mock_resp,
        query="weather lookup",
        max_results=2,
        include_discovery_receipt=True,
    )

    receipt = result["discovery_receipt"]
    assert result["total_results"] == 1
    assert receipt["exposed_results"] == [
        {"asset_type": "tool", "service_path": "/a", "name": "tool_0", "similarity_score": 1.0},
        {"asset_type": "tool", "service_path": "/a", "name": "tool_1", "similarity_score": 0.95},
    ]
    assert receipt["withheld"]["candidate_result_count"] == 2
    assert receipt["withheld"]["reason"] == "outside_intent_or_budget"
    assert receipt["withheld"]["top_withheld"] == [
        {"asset_type": "tool", "service_path": "/a", "name": "tool_2", "similarity_score": 0.9},
        {"asset_type": "tool", "service_path": "/a", "name": "tool_3", "similarity_score": 0.85},
    ]


# ---------------------------------------------------------------------------
# test_search_registry_receipt_counts_top_level_tools
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_search_registry_receipt_counts_top_level_tools():
    """search_registry receipt also handles top-level tool search results."""
    mock_resp = _make_mock_response()
    mock_resp.json.return_value = {
        "servers": [],
        "tools": [
            {"server_path": "/a", "tool_name": "tool_0", "relevance_score": 1.0},
            {"server_path": "/a", "tool_name": "tool_1", "relevance_score": 0.95},
            {"server_path": "/b", "tool_name": "tool_2", "relevance_score": 0.9},
        ],
        "agents": [],
        "skills": [],
    }

    result = await _call_search_registry(
        mock_resp,
        query="weather lookup",
        max_results=2,
        include_discovery_receipt=True,
    )

    receipt = result["discovery_receipt"]
    assert result["total_results"] == 3
    assert len(receipt["exposed_results"]) == 2
    assert receipt["withheld"]["candidate_result_count"] == 1


# ---------------------------------------------------------------------------
# test_search_registry_receipt_counts_agents_and_skills
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_search_registry_receipt_counts_agents_and_skills():
    """search_registry receipt accounts for agents and skills, not just tools."""
    mock_resp = _make_mock_response()
    mock_resp.json.return_value = {
        "servers": [],
        "tools": [{"server_path": "/a", "tool_name": "tool_0", "relevance_score": 1.0}],
        "agents": [{"path": "/agent", "agent_name": "agent_0", "relevance_score": 0.9}],
        "skills": [{"path": "/skill", "skill_name": "skill_0", "relevance_score": 0.8}],
    }

    result = await _call_search_registry(
        mock_resp,
        query="weather lookup",
        max_results=2,
        include_discovery_receipt=True,
    )

    receipt = result["discovery_receipt"]
    assert result["total_results"] == 3
    assert receipt["exposed_results"] == [
        {"asset_type": "tool", "service_path": "/a", "name": "tool_0", "similarity_score": 1.0},
        {
            "asset_type": "agent",
            "service_path": "/agent",
            "name": "agent_0",
            "similarity_score": 0.9,
        },
    ]
    assert receipt["withheld"]["candidate_result_count"] == 1


# ---------------------------------------------------------------------------
# test_discovery_receipt_helper_does_not_leak_payloads
# ---------------------------------------------------------------------------


def test_discovery_receipt_helper_does_not_leak_payloads():
    """Helper keeps receipts to query, counts, limits, scores, and status fields."""
    receipt = _build_discovery_receipt(
        query="find private customer records",
        limit=1,
        exposed_results=[
            {
                "asset_type": "tool",
                "service_path": "/crm",
                "name": "search_customers",
                "similarity_score": 0.9,
            }
        ],
        withheld_results=[
            {
                "asset_type": "tool",
                "service_path": "/crm",
                "name": "list_orders",
                "similarity_score": 0.4,
            },
            {
                "asset_type": "tool",
                "service_path": "/crm",
                "name": "list_tickets",
                "similarity_score": 0.3,
            },
        ],
        status="success",
        stop_reason="results_returned",
    )

    assert "raw_args" not in receipt
    assert "raw_result" not in receipt
    assert receipt["withheld"]["candidate_result_count"] == 2
    assert receipt["withheld"]["top_withheld"] == [
        {
            "asset_type": "tool",
            "service_path": "/crm",
            "name": "list_orders",
            "similarity_score": 0.4,
        },
        {
            "asset_type": "tool",
            "service_path": "/crm",
            "name": "list_tickets",
            "similarity_score": 0.3,
        },
    ]


# ---------------------------------------------------------------------------
# test_search_registry_receipt_dedupes_tool_in_both_arrays
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_search_registry_receipt_dedupes_tool_in_both_arrays():
    """A tool returned in both tools[] and matching_tools is counted once.

    The real registry returns a matched tool in both arrays. Without dedup the
    receipt double-counts it: exposed_results lists it twice and withheld is
    inflated. This is the case the pre-fix unit tests missed.
    """
    mock_resp = _make_mock_response()
    mock_resp.json.return_value = {
        "servers": [
            {
                "server_name": "server-a",
                "path": "/a",
                "matching_tools": [
                    {"tool_name": "get_weather", "relevance_score": 0.45},
                ],
            }
        ],
        "tools": [
            {"server_path": "/a", "tool_name": "get_weather", "relevance_score": 0.45},
        ],
        "agents": [],
        "skills": [],
    }

    result = await _call_search_registry(
        mock_resp,
        query="weather forecast",
        max_results=2,
        include_discovery_receipt=True,
    )

    receipt = result["discovery_receipt"]
    # One unique tool: exposed once, nothing withheld.
    assert receipt["exposed_results"] == [
        {
            "asset_type": "tool",
            "service_path": "/a",
            "name": "get_weather",
            "similarity_score": 0.45,
        },
    ]
    assert receipt["withheld"]["candidate_result_count"] == 0
    assert receipt["withheld"]["top_withheld"] == []


# ---------------------------------------------------------------------------
# test_dedupe_candidates_keeps_higher_score
# ---------------------------------------------------------------------------


def test_dedupe_candidates_keeps_higher_score():
    """Dedup keeps one entry per (asset_type, service_path, name), higher score."""
    candidates = [
        {"asset_type": "tool", "service_path": "/a", "name": "foo", "similarity_score": 0.3},
        {"asset_type": "tool", "service_path": "/a", "name": "foo", "similarity_score": 0.9},
        {"asset_type": "tool", "service_path": "/b", "name": "bar", "similarity_score": 0.5},
    ]

    deduped = _dedupe_candidates(candidates)

    assert deduped == [
        {"asset_type": "tool", "service_path": "/a", "name": "foo", "similarity_score": 0.9},
        {"asset_type": "tool", "service_path": "/b", "name": "bar", "similarity_score": 0.5},
    ]


# ---------------------------------------------------------------------------
# test_discovery_receipt_caps_top_withheld_items
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_discovery_receipt_caps_top_withheld_items():
    """top_withheld is capped (MAX_WITHHELD_ITEMS) while the count stays full."""
    server = _make_server_with_tools(10, server_name="server-a", path="/a")
    mock_resp = _make_mock_response(servers=[server])

    result = await _call_search_registry(
        mock_resp,
        query="many tools",
        max_results=1,
        include_discovery_receipt=True,
    )

    receipt = result["discovery_receipt"]
    # 10 unique tools, 1 exposed, 9 withheld; only 5 itemized.
    assert receipt["withheld"]["candidate_result_count"] == 9
    assert len(receipt["withheld"]["top_withheld"]) == 5
    # The itemized ones are the highest-scoring withheld (tool_1..tool_5).
    assert [item["name"] for item in receipt["withheld"]["top_withheld"]] == [
        "tool_1",
        "tool_2",
        "tool_3",
        "tool_4",
        "tool_5",
    ]
