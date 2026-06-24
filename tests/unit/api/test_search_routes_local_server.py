"""Tests for the deployment='local' branch in semantic_search.

For local servers, the search response should suppress endpoint_url (there's
no nginx route to GET) and propagate deployment + local_runtime so consumers
can construct a stdio launch recipe.
"""

from typing import Any
from unittest.mock import AsyncMock, Mock, patch

import pytest
from fastapi import Request

from registry.api.search_routes import SemanticSearchRequest, semantic_search


@pytest.fixture
def admin_ctx() -> dict[str, Any]:
    return {
        "username": "admin",
        "is_admin": True,
        "groups": ["mcp-registry-admin"],
        "scopes": ["mcp-servers-unrestricted/read"],
        "accessible_servers": ["*"],
        "accessible_tools": {"*": {"*"}},
        "accessible_agents": ["all"],
    }


@pytest.fixture
def mock_request():
    req = Mock(spec=Request)
    req.state = Mock()
    return req


@pytest.fixture
def mock_repo():
    return AsyncMock()


@pytest.mark.unit
@pytest.mark.asyncio
class TestSemanticSearchLocalServer:
    async def test_local_server_endpoint_url_is_none_and_runtime_propagated(
        self,
        mock_request,
        mock_repo,
        admin_ctx,
    ):
        """A search hit for a local server should set endpoint_url=None on the
        result (no gateway URL to GET) and surface deployment + local_runtime
        so the client can build a stdio launch recipe."""
        local_runtime = {
            "type": "npx",
            "package": "@acme/local-mcp",
            "version": "1.0.0",
            "env": {"LOG_LEVEL": "info"},
            "required_env": ["API_KEY"],
        }
        results = {
            "servers": [
                {
                    "path": "/local-mcp",
                    "server_name": "local-mcp",
                    "description": "stdio server",
                    "tags": ["local"],
                    "num_tools": 0,
                    "is_enabled": True,
                    "relevance_score": 0.9,
                    "match_context": "local stdio",
                    "matching_tools": [],
                    "deployment": "local",
                    "local_runtime": local_runtime,
                }
            ],
            "tools": [],
            "agents": [],
        }
        mock_repo.search = AsyncMock(return_value=results)

        # The handler calls server_service.get_server_info to look up trust
        # metadata and the full tool_list. Return a matching shape so the
        # local-server branch is the only thing distinguishing this from the
        # remote case.
        async def fake_get_server_info(path):
            return {
                "path": path,
                "server_name": "local-mcp",
                "deployment": "local",
                "local_runtime": local_runtime,
                "tool_list": [],
                "is_enabled": True,
            }

        with patch(
            "registry.api.search_routes.server_service.get_server_info",
            new=AsyncMock(side_effect=fake_get_server_info),
        ):
            request = SemanticSearchRequest(query="local", max_results=5)
            response = await semantic_search(mock_request, request, admin_ctx, mock_repo)

        assert len(response.servers) == 1
        result = response.servers[0]
        assert result.deployment == "local"
        assert result.local_runtime == local_runtime
        # The local-server branch suppresses endpoint_url — clients should use
        # local_runtime to build a launch recipe instead of trying to GET a URL.
        assert result.endpoint_url is None

    async def test_remote_server_endpoint_url_unchanged(
        self,
        mock_request,
        mock_repo,
        admin_ctx,
    ):
        """Sanity check: remote servers still get a populated endpoint_url and
        no local_runtime, ensuring the local branch hasn't broken the default."""
        results = {
            "servers": [
                {
                    "path": "/remote-mcp",
                    "server_name": "remote-mcp",
                    "description": "http server",
                    "tags": ["remote"],
                    "num_tools": 1,
                    "is_enabled": True,
                    "relevance_score": 0.9,
                    "match_context": "remote http",
                    "matching_tools": [],
                    "deployment": "remote",
                    # no local_runtime
                }
            ],
            "tools": [],
            "agents": [],
        }
        mock_repo.search = AsyncMock(return_value=results)

        async def fake_get_server_info(path):
            return {
                "path": path,
                "server_name": "remote-mcp",
                "deployment": "remote",
                "proxy_pass_url": "http://upstream",
                "tool_list": [],
                "is_enabled": True,
            }

        with patch(
            "registry.api.search_routes.server_service.get_server_info",
            new=AsyncMock(side_effect=fake_get_server_info),
        ):
            request = SemanticSearchRequest(query="remote", max_results=5)
            response = await semantic_search(mock_request, request, admin_ctx, mock_repo)

        assert len(response.servers) == 1
        result = response.servers[0]
        assert result.deployment == "remote"
        assert result.local_runtime is None
