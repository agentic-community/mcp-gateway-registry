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

from servers.mcpgw.server import _validate_top_n, intelligent_tool_finder


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


async def _call_finder(mock_response, query="test", top_n=None, capture=None):
    """Helper to call intelligent_tool_finder with mocked HTTP client and token.

    Args:
        mock_response: The mock httpx response to return from POST.
        query: Search query string.
        top_n: Number of results (omit to use default).
        capture: If provided, a dict that will be populated with the POST kwargs.

    Returns:
        The result dict from intelligent_tool_finder.
    """
    captured_kwargs = {}

    async def mock_post(url, **kwargs):
        captured_kwargs.update(kwargs)
        return mock_response

    mock_client = AsyncMock()
    mock_client.post = mock_post
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)

    with (
        patch("servers.mcpgw.server.httpx.AsyncClient", return_value=mock_client),
        patch("servers.mcpgw.server._extract_bearer_token", return_value="test-token"),
    ):
        if top_n is not None:
            result = await intelligent_tool_finder(query=query, top_n=top_n)
        else:
            result = await intelligent_tool_finder(query=query)

    if capture is not None:
        capture.update(captured_kwargs)

    return result


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
    assert body["entity_types"] == ["mcp_server", "tool"]
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
