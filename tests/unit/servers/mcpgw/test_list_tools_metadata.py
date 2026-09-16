"""Unit tests for metadata passthrough on the mcpgw catalog listing tools.

list_services, list_agents and list_skills reshape each registry record through
a local Pydantic model before returning it. Those models used to omit the
`metadata` subdocument, so anything a registrant attached to an asset —
ownership, provenance, configuration — was dropped by the MCP layer even though
the registry had returned it in full. These tests pin the passthrough.
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
if "fastmcp" not in sys.modules:
    _fastmcp_stub = types.ModuleType("fastmcp")
    _fastmcp_stub.Context = type("Context", (), {})
    _mock_mcp = MagicMock()
    _mock_mcp.tool.return_value = lambda fn: fn  # decorator is a no-op
    _fastmcp_stub.FastMCP = MagicMock(return_value=_mock_mcp)
    sys.modules["fastmcp"] = _fastmcp_stub

# Add servers/mcpgw to sys.path so that `from models import ...` works
_mcpgw_path = str(Path(__file__).resolve().parents[4] / "servers" / "mcpgw")
if _mcpgw_path not in sys.path:
    sys.path.insert(0, _mcpgw_path)

# Import (not re-import) the server module: popping it from sys.modules here
# would break sibling test modules that already hold a reference to the
# previously imported module object, since patch() targets would then resolve
# to a different module than the tools under test came from.

from servers.mcpgw.server import (  # noqa: E402
    list_agents,
    list_services,
    list_skills,
)

OWNERSHIP = {"owner_person": "someone@example.com", "owner_team": "platform"}


async def _call_with_mocked_get(tool_func, payload):
    """Call a listing tool with the registry GET returning `payload`."""
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.raise_for_status = MagicMock()
    mock_response.json.return_value = payload

    async def mock_get(url, **kwargs):
        return mock_response

    mock_client = AsyncMock()
    mock_client.get = mock_get
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)

    with (
        patch("servers.mcpgw.server.httpx.AsyncClient", return_value=mock_client),
        patch("servers.mcpgw.server._extract_bearer_token", return_value="test-token"),
    ):
        return await tool_func()


@pytest.mark.asyncio
async def test_list_services_keeps_server_metadata():
    """A server's metadata answers "who owns this"; do not discard it."""
    payload = {
        "servers": [
            {
                "display_name": "Billing",
                "path": "/billing",
                "description": "Billing tools",
                "is_enabled": True,
                "metadata": OWNERSHIP,
            }
        ]
    }

    result = await _call_with_mocked_get(list_services, payload)

    assert result["status"] == "success"
    assert result["services"][0]["metadata"] == OWNERSHIP


@pytest.mark.asyncio
async def test_list_agents_keeps_agent_metadata():
    payload = {"agents": [{"name": "sre", "description": "SRE agent", "metadata": OWNERSHIP}]}

    result = await _call_with_mocked_get(list_agents, payload)

    assert result["agents"][0]["metadata"] == OWNERSHIP


@pytest.mark.asyncio
async def test_list_skills_keeps_skill_metadata():
    """Skills nest free-form keys under metadata.extra, so keep the whole object."""
    skill_metadata = {"author": "someone", "version": "1.0.0", "extra": OWNERSHIP}
    payload = {
        "skills": [
            {
                "path": "/skills/deploy",
                "name": "deploy",
                "description": "Deploy a service",
                "metadata": skill_metadata,
            }
        ]
    }

    result = await _call_with_mocked_get(list_skills, payload)

    assert result["skills"][0]["metadata"] == skill_metadata


@pytest.mark.asyncio
async def test_records_without_metadata_stay_valid():
    """Metadata is optional: a registry that sends none must not fail the call."""
    payload = {
        "servers": [
            {"display_name": "Bare", "path": "/bare", "is_enabled": False},
        ]
    }

    result = await _call_with_mocked_get(list_services, payload)

    assert result["status"] == "success"
    assert result["services"][0]["metadata"] is None
