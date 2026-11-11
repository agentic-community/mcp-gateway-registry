"""Tests for the Origin check on cookie-authenticated MCP requests (#1996)
and the JSON-RPC body of a refused MCP request (#2006).

Each endpoint test sends /validate the headers nginx sends: X-Original-URL,
X-Original-Method, X-Body and the client's own Cookie and Origin headers.
"""

import json
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient
from itsdangerous import URLSafeTimedSerializer

GATEWAY_ORIGIN: str = "https://gw.example.com"
TRUSTED: frozenset[str] = frozenset({GATEWAY_ORIGIN})
DIRECT_URL: str = "https://gw.example.com/test-server/mcp"
VIRTUAL_URL: str = "https://gw.example.com/virtual/compat-test/mcp"
API_URL: str = "https://gw.example.com/api/servers"
LIST_BODY: str = '{"jsonrpc":"2.0","id":4,"method":"tools/list","params":{}}'
CALL_BODY: str = (
    '{"jsonrpc":"2.0","id":"req-9","method":"tools/call",'
    '"params":{"name":"delete_everything","arguments":{}}}'
)


async def _fake_resolve(
    session_id: str,
) -> dict:
    return {
        "session_id": session_id,
        "username": "testuser",
        "groups": ["users", "developers"],
        "provider": "cognito",
        "auth_method": "oauth2",
    }


def _cookie_validate(
    auth_env_vars: dict,
    cookie: str,
    scope_repository: object,
    original_url: str,
    method: str,
    origin: str | None,
    body: str | None = LIST_BODY,
) -> object:
    """Call /validate as nginx would for a cookie-authenticated request."""
    import auth_server.server as server_module

    headers = {
        "Cookie": f"mcp_gateway_session={cookie}",
        "X-Original-URL": original_url,
        "X-Original-Method": method,
    }
    if origin is not None:
        headers["Origin"] = origin
    if body is not None:
        headers["X-Body"] = body
    with (
        patch("auth_server.server.get_scope_repository", return_value=scope_repository),
        patch(
            "auth_server.server.signer",
            URLSafeTimedSerializer(auth_env_vars["SECRET_KEY"]),
        ),
        patch("session_store.resolve_session", _fake_resolve),
        patch("auth_server.server._gateway_trusted_origins", return_value=TRUSTED),
        patch("auth_server.server.get_auth_provider"),
    ):
        return TestClient(server_module.app).get("/validate", headers=headers)


def _error_body(
    response: object,
) -> dict:
    return json.loads(response.headers["X-Mcp-Error-Body"])


class TestCookieOriginCheck:
    """A cookie-authenticated MCP request must come from the gateway's origin."""

    @pytest.fixture(autouse=True)
    def _setup(
        self,
        auth_env_vars,
        valid_session_cookie,
        mock_scope_repository_with_data,
    ):
        self.env = auth_env_vars
        self.cookie = valid_session_cookie
        self.repo = mock_scope_repository_with_data

    def _call(
        self,
        original_url: str,
        method: str,
        origin: str | None,
        body: str | None = LIST_BODY,
    ) -> object:
        return _cookie_validate(
            self.env, self.cookie, self.repo, original_url, method, origin, body
        )

    def test_rejects_cross_origin_post_on_direct_route(self):
        response = self._call(DIRECT_URL, "POST", "https://evil.example")
        assert response.status_code == 403
        assert "X-Internal-Token" not in response.headers
        assert _error_body(response)["error"]["data"]["reason"] == "origin_refused"

    def test_rejects_cross_origin_post_on_virtual_route(self):
        response = self._call(VIRTUAL_URL, "POST", "https://evil.example")
        assert response.status_code == 403
        assert "X-Internal-Token" not in response.headers

    @pytest.mark.parametrize(
        "origin",
        [None, "", "null", "not a url", "https://gw.example.com/path", "ftp://gw.example.com"],
    )
    def test_rejects_post_with_missing_null_or_malformed_origin(
        self,
        origin,
    ):
        response = self._call(DIRECT_URL, "POST", origin)
        assert response.status_code == 403

    def test_rejects_untrusted_origin_on_a_get(self):
        response = self._call(DIRECT_URL, "GET", "https://evil.example", body=None)
        assert response.status_code == 403

    def test_allows_same_origin_post(self):
        response = self._call(DIRECT_URL, "POST", GATEWAY_ORIGIN)
        assert response.status_code == 200

    def test_allows_same_origin_with_explicit_default_port(self):
        response = self._call(DIRECT_URL, "POST", "https://gw.example.com:443")
        assert response.status_code == 200

    def test_allows_get_without_origin(self):
        response = self._call(DIRECT_URL, "GET", None, body=None)
        assert response.status_code == 200

    def test_refusal_names_method_and_request_id(self):
        response = self._call(DIRECT_URL, "POST", "https://evil.example")
        error = _error_body(response)
        assert error["id"] == 4
        assert "tools/list" in error["error"]["message"]
        assert "/test-server/" in error["error"]["message"]

    def test_registry_api_routes_keep_their_own_check(self):
        response = self._call(API_URL, "POST", "https://evil.example", body=None)
        assert response.status_code != 403 or "X-Mcp-Error-Body" not in response.headers


class TestRefuseCrossOriginHelper:
    """Unit tests for _refuse_cross_origin_cookie_request."""

    def _request(
        self,
        origin: str | None,
    ) -> object:
        from starlette.requests import Request

        headers = [(b"origin", origin.encode())] if origin is not None else []
        return Request({"type": "http", "headers": headers})

    def test_bearer_request_without_origin_is_not_checked(self):
        from auth_server.server import _refuse_cross_origin_cookie_request

        with patch("auth_server.server._gateway_trusted_origins", return_value=TRUSTED):
            _refuse_cross_origin_cookie_request(
                self._request(None), "keycloak", DIRECT_URL, "POST", {}, "test-server"
            )

    def test_cookie_request_from_other_origin_raises_403(self):
        from fastapi import HTTPException

        from auth_server.server import _refuse_cross_origin_cookie_request

        with (
            patch("auth_server.server._gateway_trusted_origins", return_value=TRUSTED),
            pytest.raises(HTTPException) as raised,
        ):
            _refuse_cross_origin_cookie_request(
                self._request("https://evil.example"),
                "session_cookie",
                DIRECT_URL,
                "POST",
                json.loads(CALL_BODY),
                "test-server",
            )
        assert raised.value.status_code == 403
        body = json.loads(raised.value.headers["X-Mcp-Error-Body"])
        assert body["id"] == "req-9"
        assert body["error"]["data"]["tool"] == "delete_everything"


class TestGatewayTrustedOrigins:
    """The trusted origins come from configuration, never from the request."""

    def test_reads_configured_urls_and_drops_path_and_default_port(
        self,
        monkeypatch,
    ):
        from auth_server.server import _gateway_trusted_origins

        monkeypatch.setenv("AUTH_SERVER_EXTERNAL_URL", "https://GW.example.com:443/")
        monkeypatch.setenv("REGISTRY_URL", "http://localhost/registry")
        origins = _gateway_trusted_origins()
        assert "https://gw.example.com" in origins
        assert "http://localhost" in origins

    def test_ignores_an_unparseable_url(
        self,
        monkeypatch,
    ):
        from auth_server.server import _gateway_trusted_origins

        monkeypatch.setenv("AUTH_SERVER_EXTERNAL_URL", "not a url")
        monkeypatch.setenv("REGISTRY_URL", "")
        assert "not a url" not in _gateway_trusted_origins()


class TestDeniedErrorBody:
    """A scope denial carries a JSON-RPC error naming the method and tool."""

    def test_scope_denial_names_the_tool(
        self,
        auth_env_vars,
        valid_session_cookie,
        mock_scope_repository_with_data,
    ):
        response = _cookie_validate(
            auth_env_vars,
            valid_session_cookie,
            mock_scope_repository_with_data,
            "https://gw.example.com/server-nobody-has/mcp",
            "POST",
            GATEWAY_ORIGIN,
            body=CALL_BODY,
        )
        assert response.status_code == 403
        error = _error_body(response)
        assert error["id"] == "req-9"
        assert error["error"]["code"] == -32003
        assert error["error"]["data"]["method"] == "tools/call"
        assert error["error"]["data"]["tool"] == "delete_everything"
        assert "/server-nobody-has/" in error["error"]["message"]

    def test_denied_headers_skip_a_notification(self):
        from auth_server.server import _mcp_denied_headers

        payload = {"jsonrpc": "2.0", "method": "notifications/initialized"}
        assert _mcp_denied_headers(payload, "not_permitted", "x", None, "s") == {}

    def test_denied_header_value_has_no_line_breaks(self):
        from auth_server.server import _mcp_denied_headers

        payload = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {"name": "a\r\nX-Injected: 1"},
        }
        headers = _mcp_denied_headers(payload, "not_permitted", "tools/call", "a\r\nX: 1", "s")
        value = headers["X-Mcp-Error-Body"]
        assert "\r" not in value and "\n" not in value
