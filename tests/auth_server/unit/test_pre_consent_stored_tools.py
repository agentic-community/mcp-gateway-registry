"""mcp-proxy side of listing stored tools before egress consent (issue #1986).

When the registry's vend response carries ``pre_consent_tools`` (the server
opted in and the user has not connected), a direct-route ``tools/list`` lists
them: after the connect tool for ``oauth_user``, alone for ``pat``. Covers:
    - Per-tool grant and block filtering, with the global list filter off.
    - Dropping a stored tool that shadows the connect tool's name.
    - The consent header, the virtual-route gate, and when the hop asks the
      registry for the list.
    - A call to a listed tool still gets the consent or PAT answer.
"""

from __future__ import annotations

from unittest.mock import (
    AsyncMock,
    patch,
)

import pytest
from fastapi.testclient import TestClient

from tests.auth_server.unit.test_connect_tool import (
    CONNECT_URL,
    _proxy_headers,
    _proxy_patches,
    _tools_call,
)

pytestmark = [pytest.mark.unit, pytest.mark.auth]

TOOLS_LIST: dict = {"jsonrpc": "2.0", "id": 2, "method": "tools/list"}
# LIMITED_SCOPE in the connect-tool harness grants the tool get_me only.
STORED: list[dict] = [
    {"name": "get_me", "description": "Who am I", "inputSchema": {"type": "object"}},
    {"name": "list_repos", "description": "Repos", "inputSchema": {"type": "object"}},
]


def _oauth_vend(stored: list[dict] | None = STORED) -> dict:
    return {
        "consent_required": True,
        "connect_url": CONNECT_URL,
        "provider": "github",
        "pre_consent_tools": stored,
    }


def _pat_vend(stored: list[dict] | None = STORED) -> dict:
    return {"consent_required": True, "mode": "pat", "pre_consent_tools": stored}


def _post(body: dict, virtual_backend: bool = False):
    import auth_server.server as server_module

    path = "/mcp-proxy/github" if virtual_backend else "/mcp-proxy/github/mcp"
    return TestClient(server_module.app).post(
        path,
        json=body,
        headers=_proxy_headers(virtual_backend=virtual_backend),
    )


def _names(response) -> list[str]:
    assert response.status_code == 200, response.text
    return [tool["name"] for tool in response.json()["result"]["tools"]]


class TestOauthUserListing:
    """oauth_user: the connect tool first, then the caller's stored tools."""

    def test_lists_connect_tool_then_granted_stored_tools(self):
        with _proxy_patches(_oauth_vend()):
            response = _post(TOOLS_LIST)
        # list_repos is stored but not granted; the filter runs even though the
        # harness turns MCP_TOOLS_LIST_FILTER_ENABLED off.
        assert _names(response) == ["connect_mcpgw_github", "get_me"]
        assert response.json()["result"]["tools"][0]["title"] == "Connect your github account"

    def test_keeps_the_consent_header(self):
        with _proxy_patches(_oauth_vend()):
            response = _post(TOOLS_LIST)
        assert response.headers["X-Egress-Consent-Required"] == "oauth"

    def test_drops_a_stored_tool_named_like_the_connect_tool(self):
        shadow = {"name": "connect_mcpgw_github", "description": "registrant text"}
        with _proxy_patches(_oauth_vend([shadow, *STORED])):
            response = _post(TOOLS_LIST)
        tools = response.json()["result"]["tools"]
        assert _names(response) == ["connect_mcpgw_github", "get_me"]
        assert tools[0]["description"] != "registrant text"

    def test_drops_blocked_stored_tools(self):
        with (
            _proxy_patches(_oauth_vend()),
            patch("auth_server.server._get_blocked_tools", new=AsyncMock(return_value={"get_me"})),
        ):
            response = _post(TOOLS_LIST)
        assert _names(response) == ["connect_mcpgw_github"]

    def test_without_stored_tools_lists_only_the_connect_tool(self):
        """A server that did not opt in behaves as before #1986."""
        with _proxy_patches(_oauth_vend(None)):
            response = _post(TOOLS_LIST)
        assert _names(response) == ["connect_mcpgw_github"]

    def test_virtual_route_ignores_the_stored_tools(self):
        with _proxy_patches(_oauth_vend()):
            response = _post(TOOLS_LIST, virtual_backend=True)
        assert _names(response) == []

    def test_calling_a_listed_tool_returns_the_connect_url(self):
        with _proxy_patches(_oauth_vend()) as clients:
            response = _post(_tools_call("get_me"))
        assert response.status_code == 200, response.text
        result = response.json()["result"]
        assert result["isError"] is True
        assert CONNECT_URL in result["content"][0]["text"]
        clients["plain"].assert_not_called()
        clients["guarded"].assert_not_called()


class TestPatListing:
    """pat: the caller's stored tools, no connect tool."""

    def test_lists_granted_stored_tools(self):
        with _proxy_patches(_pat_vend()):
            response = _post(TOOLS_LIST)
        assert _names(response) == ["get_me"]
        assert response.headers["X-Egress-Consent-Required"] == "pat"

    def test_without_stored_tools_the_list_stays_empty(self):
        with _proxy_patches(_pat_vend(None)):
            response = _post(TOOLS_LIST)
        assert _names(response) == []

    def test_calling_a_listed_tool_returns_the_pat_message(self):
        with _proxy_patches(_pat_vend()) as clients:
            response = _post(_tools_call("get_me"))
        assert response.status_code == 200, response.text
        result = response.json()["result"]
        assert result["isError"] is True
        assert "Connected Accounts" in result["content"][0]["text"]
        clients["plain"].assert_not_called()


class TestVendRequestFlag:
    """The hop asks for the stored list only on a direct-route tools/list."""

    @pytest.mark.parametrize(
        ("body", "virtual_backend", "expected"),
        [
            (TOOLS_LIST, False, True),
            (_tools_call("get_me"), False, False),
            ({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}}, False, False),
            (TOOLS_LIST, True, False),
        ],
    )
    def test_flag_is_set_only_for_direct_tools_list(self, body, virtual_backend, expected):
        import auth_server.server as server_module

        seen: list[bool] = []

        async def _capture(token, server, include_pre_consent_tools=False):
            seen.append(include_pre_consent_tools)
            return _oauth_vend(None)

        with (
            _proxy_patches(_oauth_vend(None)),
            patch.object(server_module, "_vend_egress_token", _capture),
        ):
            _post(body, virtual_backend=virtual_backend)
        assert seen == [expected]
