"""Tests for the synthetic per-server connect tool (issue #1860).

Before a user connects an oauth_user server's provider account, tools/list
returns one gateway-owned tool, ``connect_mcpgw_<server>``. Covers:
    - The tool name and the exact-match rule.
    - The scope rule: on a direct route the connect tool is authorized like
      tools/list, never on a virtual route, never for any other name.
    - The pre-consent local answers (tools/list, server/discover).
    - The mcp-proxy hop: the connect tool is answered locally and is never
      forwarded upstream, with or without a vaulted credential.
    - The /validate wiring of the scope rule.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from unittest.mock import (
    AsyncMock,
    MagicMock,
    patch,
)

import jwt
import pytest
from fastapi.testclient import TestClient

pytestmark = [pytest.mark.unit, pytest.mark.auth]

MARKER: str = "verified-marker-for-test"  # pragma: allowlist secret
CONNECT_URL: str = "https://gw.example.com/oauth2/egress/connect?server=%2Fgithub"
LIMITED_SCOPE: str = "github-limited"
# Access to the server (initialize, tools/list) but a grant for one tool only.
LIMITED_SCOPE_CONFIG: list[dict] = [
    {"server": "github", "methods": ["initialize", "tools/list"], "tools": ["get_me"]}
]


def _limited_scope_repo() -> AsyncMock:
    """Scope repository granting LIMITED_SCOPE_CONFIG for LIMITED_SCOPE only."""
    repo = AsyncMock()

    async def _get_server_scopes(scope_name: str):
        if scope_name == LIMITED_SCOPE:
            return LIMITED_SCOPE_CONFIG
        return []

    async def _get_server_scopes_bulk(scope_names: list[str]):
        return {s: LIMITED_SCOPE_CONFIG for s in scope_names if s == LIMITED_SCOPE}

    repo.get_server_scopes.side_effect = _get_server_scopes
    repo.get_server_scopes_bulk.side_effect = _get_server_scopes_bulk
    return repo


def _proxy_headers(
    server_name: str = "github",
    virtual_backend: bool = False,
) -> dict:
    """The X-Internal-Token and registration headers nginx sends to /mcp-proxy."""
    from auth_server.internal_request_token import mint_mcp_proxy_token

    token = mint_mcp_proxy_token(
        subject="test-user",
        scopes=[LIMITED_SCOPE],
        server_name=server_name,
        upstream_url="https://api.githubcopilot.com/mcp",
        virtual_backend=virtual_backend,
    )
    return {
        "X-Internal-Token": token,
        "X-Registered-Server-Path": "/" + server_name,
        "X-Registered-Route-Mode": "virtual" if virtual_backend else "direct",
    }


def _legacy_proxy_headers(server_name: str = "github") -> dict:
    """Headers carrying a hop token minted before the route claims existed.

    The token has no ``version_id`` and no ``virtual_backend``. The route check
    still accepts it as a direct route, but it must not earn the connect-tool
    exemption: only an explicit ``virtual_backend: False`` does.
    """
    from auth_server.internal_request_token import (
        MCP_PROXY_AUDIENCE,
        MCP_PROXY_TOKEN_USE,
        _mint_internal_token,
    )

    token = _mint_internal_token(
        audience=MCP_PROXY_AUDIENCE,
        subject="test-user",
        scopes=[LIMITED_SCOPE],
        extra_claims={
            "server": server_name,
            "upstream_url": "https://api.githubcopilot.com/mcp",
            "auth_method": "",
            "egress_user": "",
            "token_use": MCP_PROXY_TOKEN_USE,
        },
    )
    return {"X-Internal-Token": token}


def _tools_call(name: str) -> dict:
    return {"jsonrpc": "2.0", "id": 3, "method": "tools/call", "params": {"name": name}}


@contextmanager
def _proxy_patches(vend: dict | None):
    """Patch the proxy hop's collaborators and expose the upstream clients.

    Yields a dict of the two upstream client factories, so a test can assert
    that neither was used (nothing was forwarded).
    """
    import auth_server.server as server_module

    async def _vend(token, server):
        return vend

    plain_client = MagicMock(side_effect=AssertionError("forwarded with the plain client"))
    guarded_client = MagicMock(side_effect=AssertionError("forwarded with the guarded client"))
    with (
        patch.object(server_module.settings, "egress_auth_enabled", vend is not None),
        patch.object(server_module.settings, "egress_consent_use_elicitation", False),
        patch.object(server_module, "_vend_egress_token", _vend),
        patch.object(server_module, "_read_mcp_filter_enabled", return_value=False),
        patch("auth_server.server.get_scope_repository", return_value=_limited_scope_repo()),
        patch("auth_server.server._get_blocked_tools", new=AsyncMock(return_value=set())),
        patch("auth_server.server.httpx.AsyncClient", plain_client),
        patch("registry.utils.url_guard.shared_guarded_async_client", guarded_client),
    ):
        yield {"plain": plain_client, "guarded": guarded_client}


def _consent_vend() -> dict:
    return {"consent_required": True, "connect_url": CONNECT_URL, "provider": "github"}


def _connected_vend() -> dict:
    return {"access_token": "vaulted-upstream-token", "provider": "github"}


class TestConnectToolName:
    """The name is derived from the registered server path."""

    def test_name_is_the_prefix_plus_the_server_path(self):
        from auth_server.server import _connect_tool_name

        assert _connect_tool_name("/github/") == "connect_mcpgw_github"

    def test_characters_mcp_does_not_allow_are_replaced(self):
        from auth_server.server import _connect_tool_name

        assert _connect_tool_name("team/a b") == "connect_mcpgw_team_a_b"

    def test_name_is_truncated_to_the_mcp_limit(self):
        from auth_server.server import _connect_tool_name

        assert len(_connect_tool_name("x" * 300)) == 128


class TestScopeCheckTarget:
    """Only this server's exact connect tool, on a direct route, is relaxed."""

    def test_connect_tool_on_a_direct_route_is_checked_as_tools_list(self):
        from auth_server.server import _scope_check_target

        target = _scope_check_target("github", "tools/call", "connect_mcpgw_github", True)
        assert target == ("tools/list", None)

    def test_connect_tool_on_a_virtual_route_is_checked_as_a_tool(self):
        from auth_server.server import _scope_check_target

        target = _scope_check_target("github", "tools/call", "connect_mcpgw_github", False)
        assert target == ("tools/call", "connect_mcpgw_github")

    @pytest.mark.parametrize(
        "tool_name",
        ["connect_mcpgw_other", "connect_mcpgw_github_extra", "connect_mcpgw_", "get_me"],
    )
    def test_any_other_name_is_checked_as_a_tool(self, tool_name):
        from auth_server.server import _scope_check_target

        target = _scope_check_target("github", "tools/call", tool_name, True)
        assert target == ("tools/call", tool_name)

    def test_a_non_string_tool_name_is_not_the_connect_tool(self):
        from auth_server.server import _is_connect_tool_call

        assert _is_connect_tool_call("github", "tools/call", ["connect_mcpgw_github"]) is False


class TestPreConsentLocalAnswers:
    """tools/list and server/discover while consent is pending."""

    def test_tools_list_returns_the_connect_tool_and_keeps_the_marker(self):
        import json

        import auth_server.server as server_module

        tool = server_module._connect_tool_definition("github", "GitHub")
        resp = server_module._pre_consent_local_response(
            "tools/list", 1, {"method": "tools/list"}, "oauth", connect_tool=tool
        )
        tools = json.loads(resp.body)["result"]["tools"]
        assert [t["name"] for t in tools] == ["connect_mcpgw_github"]
        assert tools[0]["inputSchema"] == {"type": "object", "properties": {}}
        assert tools[0]["annotations"]["readOnlyHint"] is True
        assert resp.headers[server_module.EGRESS_CONSENT_REQUIRED_HEADER] == "oauth"

    def test_description_tells_the_model_when_to_call_and_what_follows(self):
        """The model only learns the server's real tools exist from this text."""
        import auth_server.server as server_module

        description = server_module._connect_tool_definition("github", "GitHub")["description"]
        assert "tools stay hidden until the user connects" in description
        assert "when the user asks for anything this server handles" in description
        assert "Show the link to the user" in description
        assert "reconnect this server" in description
        assert "this tool is no longer listed" in description

    def test_initialize_instructions_name_the_reconnect_step(self):
        """initialize gives the same next steps as the connect tool."""
        import json

        import auth_server.server as server_module

        resp = server_module._pre_consent_local_response(
            "initialize",
            1,
            {"method": "initialize", "params": {}},
            "oauth",
            connect_url=CONNECT_URL,
            provider="github",
        )
        instructions = json.loads(resp.body)["result"]["instructions"]
        assert CONNECT_URL in instructions
        assert "reconnect this server in your MCP client" in instructions

    def test_other_lists_stay_empty_when_a_connect_tool_is_given(self):
        import json

        import auth_server.server as server_module

        tool = server_module._connect_tool_definition("github", "GitHub")
        resp = server_module._pre_consent_local_response(
            "resources/list", 1, {"method": "resources/list"}, "oauth", connect_tool=tool
        )
        assert json.loads(resp.body)["result"]["resources"] == []

    @pytest.mark.parametrize("mode", ["pat", "oauth"])
    def test_server_discover_is_method_not_found(self, mode):
        import json

        import auth_server.server as server_module

        resp = server_module._pre_consent_local_response(
            "server/discover", 2, {"method": "server/discover"}, mode
        )
        assert json.loads(resp.body)["error"]["code"] == -32601


class TestMcpProxyConnectTool:
    """The proxy hop lists the connect tool and never forwards a call to it."""

    def test_pre_consent_tools_list_on_a_direct_route_lists_the_connect_tool(self):
        import auth_server.server as server_module

        with _proxy_patches(_consent_vend()):
            response = TestClient(server_module.app).post(
                "/mcp-proxy/github/mcp",
                json={"jsonrpc": "2.0", "id": 1, "method": "tools/list"},
                headers=_proxy_headers(),
            )
        assert response.status_code == 200, response.text
        names = [tool["name"] for tool in response.json()["result"]["tools"]]
        assert names == ["connect_mcpgw_github"]

    def test_pre_consent_tools_list_on_a_virtual_route_stays_empty(self):
        import auth_server.server as server_module

        with _proxy_patches(_consent_vend()):
            response = TestClient(server_module.app).post(
                "/mcp-proxy/github",
                json={"jsonrpc": "2.0", "id": 1, "method": "tools/list"},
                headers=_proxy_headers(virtual_backend=True),
            )
        assert response.status_code == 200, response.text
        assert response.json()["result"]["tools"] == []

    def test_pre_consent_connect_call_returns_the_connect_url(self):
        """A scope with no grant for the connect tool still reaches consent."""
        import auth_server.server as server_module

        with _proxy_patches(_consent_vend()) as clients:
            response = TestClient(server_module.app).post(
                "/mcp-proxy/github/mcp",
                json=_tools_call("connect_mcpgw_github"),
                headers=_proxy_headers(),
            )
        assert response.status_code == 200, response.text
        result = response.json()["result"]
        assert result["isError"] is True
        text = result["content"][0]["text"]
        assert CONNECT_URL in text
        # Running the connect tool again only says "already connected", so its
        # reply points the user at reconnecting the server instead.
        assert "reconnect this server" in text
        assert "load its github tools" in text
        assert "run this tool again" not in text
        clients["plain"].assert_not_called()
        clients["guarded"].assert_not_called()

    def test_pre_consent_ordinary_tool_call_still_asks_to_run_it_again(self):
        """A real tool keeps the retry wording; only the connect tool changes."""
        import auth_server.server as server_module

        with _proxy_patches(_consent_vend()):
            response = TestClient(server_module.app).post(
                "/mcp-proxy/github/mcp",
                json=_tools_call("get_me"),
                headers=_proxy_headers(),
            )
        assert response.status_code == 200, response.text
        text = response.json()["result"]["content"][0]["text"]
        assert CONNECT_URL in text
        assert "run this tool again" in text

    def test_connected_connect_call_is_answered_locally_not_forwarded(self):
        """With a vaulted credential the connect tool must not reach upstream."""
        import auth_server.server as server_module

        with _proxy_patches(_connected_vend()) as clients:
            response = TestClient(server_module.app).post(
                "/mcp-proxy/github/mcp",
                json=_tools_call("connect_mcpgw_github"),
                headers=_proxy_headers(),
            )
        assert response.status_code == 200, response.text
        result = response.json()["result"]
        assert result["isError"] is False
        assert "already connected" in result["content"][0]["text"]
        clients["plain"].assert_not_called()
        clients["guarded"].assert_not_called()

    def test_connect_call_on_a_non_egress_server_is_answered_locally(self):
        import auth_server.server as server_module

        with _proxy_patches(None) as clients:
            response = TestClient(server_module.app).post(
                "/mcp-proxy/github/mcp",
                json=_tools_call("connect_mcpgw_github"),
                headers=_proxy_headers(),
            )
        assert response.status_code == 200, response.text
        assert "does not need" in response.json()["result"]["content"][0]["text"]
        clients["plain"].assert_not_called()

    def test_connect_call_on_a_virtual_route_needs_a_tool_grant(self):
        import auth_server.server as server_module

        with _proxy_patches(_connected_vend()) as clients:
            response = TestClient(server_module.app).post(
                "/mcp-proxy/github",
                json=_tools_call("connect_mcpgw_github"),
                headers=_proxy_headers(virtual_backend=True),
            )
        assert response.status_code == 403
        clients["guarded"].assert_not_called()

    def test_hop_token_without_the_route_claim_gets_no_exemption(self):
        """A token predating ``virtual_backend`` fails closed: no tool grant, 403."""
        import auth_server.server as server_module

        with _proxy_patches(_consent_vend()) as clients:
            response = TestClient(server_module.app).post(
                "/mcp-proxy/github/mcp",
                json=_tools_call("connect_mcpgw_github"),
                headers=_legacy_proxy_headers(),
            )
        assert response.status_code == 403, response.text
        clients["plain"].assert_not_called()
        clients["guarded"].assert_not_called()

    def test_hop_token_without_the_route_claim_is_not_offered_the_tool(self):
        """The same token sees the pre-consent empty list, never the tool."""
        import auth_server.server as server_module

        with _proxy_patches(_consent_vend()):
            response = TestClient(server_module.app).post(
                "/mcp-proxy/github/mcp",
                json={"jsonrpc": "2.0", "id": 1, "method": "tools/list"},
                headers=_legacy_proxy_headers(),
            )
        assert response.status_code == 200, response.text
        assert response.json()["result"]["tools"] == []

    @pytest.mark.parametrize("tool_name", ["delete_repo", "connect_mcpgw_other"])
    def test_an_ungranted_tool_is_still_denied(self, tool_name):
        import auth_server.server as server_module

        with _proxy_patches(_connected_vend()) as clients:
            response = TestClient(server_module.app).post(
                "/mcp-proxy/github/mcp",
                json=_tools_call(tool_name),
                headers=_proxy_headers(),
            )
        assert response.status_code == 403
        clients["guarded"].assert_not_called()


class TestValidateConnectTool:
    """/validate applies the same scope rule, keyed on the nginx route mode."""

    def _self_signed_token(self, secret_key: str) -> str:
        now = int(time.time())
        claims = {
            "iss": "mcp-auth-server",
            "aud": "mcp-registry",
            "sub": "alice",
            "preferred_username": "alice",
            "groups": [],
            "scope": LIMITED_SCOPE,
            "token_use": "access",
            "token_kind": "user",
            "auth_method": "oauth2",
            "provider": "keycloak",
            "iat": now,
            "exp": now + 3600,
        }
        return jwt.encode(claims, secret_key, algorithm="HS256")

    def _validate(
        self,
        auth_env_vars,
        tool_name: str,
        route_mode: str,
    ):
        import auth_server.server as server_module

        server_module.SECRET_KEY = auth_env_vars["SECRET_KEY"]
        validator = server_module.SimplifiedCognitoValidator()
        provider = MagicMock()
        provider.validate_token.side_effect = validator.validate_self_signed_token
        provider.get_provider_info.return_value = {"provider_type": "self_signed"}
        token = self._self_signed_token(server_module.SECRET_KEY)
        url = "https://example.com/github/mcp"
        with (
            patch("auth_server.server.get_scope_repository", return_value=_limited_scope_repo()),
            patch("auth_server.server.get_auth_provider", return_value=provider),
            patch("auth_server.server._get_blocked_tools", new=AsyncMock(return_value=set())),
            patch.object(server_module.settings, "auth_server_nginx_marker_secret", MARKER),
        ):
            return TestClient(server_module.app).get(
                "/validate",
                headers={
                    "Authorization": f"Bearer {token}",
                    "X-Original-URL": url,
                    "X-Virtual-Original-URL": url if route_mode == "virtual" else "",
                    "X-Body": (
                        '{"jsonrpc":"2.0","id":1,"method":"tools/call",'
                        f'"params":{{"name":"{tool_name}"}}}}'
                    ),
                    "X-Resolved-Upstream": "https://api.githubcopilot.com/mcp",
                    "X-Validate-Source-Secret": MARKER,
                    "X-Validate-Binding-Secret": MARKER,
                    "X-Registered-Server-Path": "/github",
                    "X-Registered-Route-Mode": route_mode,
                },
            )

    def test_connect_tool_on_a_direct_route_is_allowed(self, auth_env_vars):
        response = self._validate(auth_env_vars, "connect_mcpgw_github", "direct")
        assert response.status_code == 200, response.text

    def test_ungranted_tool_on_a_direct_route_is_denied(self, auth_env_vars):
        response = self._validate(auth_env_vars, "delete_repo", "direct")
        assert response.status_code == 403

    def test_connect_tool_on_a_virtual_route_is_denied(self, auth_env_vars):
        response = self._validate(auth_env_vars, "connect_mcpgw_github", "virtual")
        assert response.status_code == 403
