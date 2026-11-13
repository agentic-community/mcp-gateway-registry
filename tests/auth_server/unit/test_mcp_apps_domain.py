"""MCP Apps: the gateway rewrites a page's Claude ui.domain to match its own URL.

Claude shows an MCP App page only when ``_meta.ui.domain`` equals a hash of the
connector URL. A backend hashes its own URL; behind the gateway the user
connected to the gateway's URL, so the proxy recomputes the value on
``resources/read`` replies. Unit tests cover the helpers; route tests drive the
real mcp_proxy handler with a mocked upstream, like test_sse_stream_truncation.
"""

import json
import os
from collections.abc import AsyncIterator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

os.environ.setdefault("SECRET_KEY", "test-secret-key-that-is-definitely-long-enough-32b")

from fastapi.testclient import TestClient  # noqa: E402

import auth_server.server as server  # noqa: E402
from auth_server.mcp_apps_domain import (  # noqa: E402
    claude_ui_domain,
    connector_url,
    gateway_public_origins,
    rewrite_ui_domains,
)

pytestmark = pytest.mark.unit

SERVER_NAME: str = "drawio"
UPSTREAM_URL: str = "https://mcp.draw.io/mcp"
GATEWAY_URL: str = "https://gw.example.com"
# draw.io's real value: the hash of its own URL, checked against Claude's rule.
DRAWIO_DOMAIN: str = "5604f724840cf26674b8350fa1891c47.claudemcpcontent.com"


def _page_reply(
    domain: str | None,
) -> dict:
    """A resources/read reply for one MCP App page."""
    meta = {"ui": {"domain": domain, "csp": {"connectDomains": ["https://a.example"]}}}
    if domain is None:
        meta = {"ui": {}}
    return {
        "jsonrpc": "2.0",
        "id": 3,
        "result": {
            "contents": [
                {
                    "uri": "ui://drawio/mcp-app.html",
                    "mimeType": "text/html;profile=mcp-app",
                    "text": "<html></html>",
                    "_meta": meta,
                }
            ]
        },
    }


def _sse(
    payload: dict,
) -> bytes:
    """Frame one JSON-RPC reply as a streamable-HTTP SSE event."""
    return f"event: message\ndata: {json.dumps(payload)}\n\n".encode()


def _domain_in(
    body: bytes,
) -> str | None:
    """Read the first content item's ui.domain from a JSON or SSE body."""
    text = body.decode()
    for line in text.split("\n"):
        if line.startswith("data:"):
            text = line[len("data:") :]
            break
    item = json.loads(text)["result"]["contents"][0]
    return item["_meta"]["ui"].get("domain")


class TestHelpers:
    """The pure helpers in auth_server/mcp_apps_domain.py."""

    def test_claude_domain_matches_known_value(self):
        """The hash rule reproduces the value draw.io computes for its own URL."""
        assert claude_ui_domain(UPSTREAM_URL) == DRAWIO_DOMAIN

    def test_connector_url_keeps_trailing_slash_and_query(self):
        """Claude hashes the exact string, so both must survive."""
        assert connector_url(GATEWAY_URL, "drawio/mcp/") == f"{GATEWAY_URL}/drawio/mcp/"
        assert connector_url(GATEWAY_URL + "/", "/drawio/mcp", "a=1") == (
            f"{GATEWAY_URL}/drawio/mcp?a=1"
        )

    def test_public_origins_prefer_registry_url(self, monkeypatch):
        """REGISTRY_URL comes first; any path in either setting is dropped."""
        monkeypatch.setenv("REGISTRY_URL", "https://gw.example.com/registry")
        monkeypatch.setenv("AUTH_SERVER_EXTERNAL_URL", "http://localhost:8888")

        assert gateway_public_origins() == ["https://gw.example.com", "http://localhost:8888"]

    def test_public_origins_empty_without_settings(self, monkeypatch):
        """With neither setting there is no origin, so nothing is rewritten."""
        monkeypatch.delenv("REGISTRY_URL", raising=False)
        monkeypatch.delenv("AUTH_SERVER_EXTERNAL_URL", raising=False)

        assert gateway_public_origins() == []

    def test_keeps_a_value_already_right_for_a_configured_url(self):
        """A backend value in keep_domains is not replaced."""
        body = json.dumps(_page_reply("right.claudemcpcontent.com")).encode()

        result = rewrite_ui_domains(
            body,
            is_sse=False,
            new_domain="other.claudemcpcontent.com",
            keep_domains=frozenset({"right.claudemcpcontent.com"}),
        )

        assert result is None

    def test_sse_rewrites_past_a_non_json_data_line(self):
        """One unreadable data line is kept and the reply event is still rewritten."""
        body = b"data: not-json\n\n" + _sse(_page_reply(DRAWIO_DOMAIN))

        rewritten = rewrite_ui_domains(body, is_sse=True, new_domain="x.claudemcpcontent.com")

        assert rewritten.startswith(b"data: not-json\n")
        reply_lines = [
            line[len("data:") :]
            for line in rewritten.decode().split("\n")
            if line.startswith("data: {")
        ]
        assert len(reply_lines) == 1
        item = json.loads(reply_lines[0])["result"]["contents"][0]
        assert item["_meta"]["ui"]["domain"] == "x.claudemcpcontent.com"

    def test_rewrites_json_reply(self):
        """A Claude-format domain is replaced, and the rest of _meta is kept."""
        body = json.dumps(_page_reply(DRAWIO_DOMAIN)).encode()

        rewritten = rewrite_ui_domains(body, is_sse=False, new_domain="x.claudemcpcontent.com")

        assert _domain_in(rewritten) == "x.claudemcpcontent.com"
        item = json.loads(rewritten)["result"]["contents"][0]
        assert item["_meta"]["ui"]["csp"] == {"connectDomains": ["https://a.example"]}

    def test_leaves_other_hosts_domains_alone(self):
        """A domain not in Claude's format is not touched."""
        body = json.dumps(_page_reply("sandbox.other-host.example")).encode()

        assert rewrite_ui_domains(body, is_sse=False, new_domain="x.claudemcpcontent.com") is None

    def test_page_without_domain_is_unchanged(self):
        """No ui.domain means nothing to rewrite."""
        body = json.dumps(_page_reply(None)).encode()

        assert rewrite_ui_domains(body, is_sse=False, new_domain="x.claudemcpcontent.com") is None

    def test_unreadable_reply_is_unchanged(self):
        """A body that is not JSON is left for the caller to forward as is."""
        assert rewrite_ui_domains(b"<html>", is_sse=False, new_domain="x") is None
        assert rewrite_ui_domains(b"data: {bad\n\n", is_sse=True, new_domain="x") is None
        assert rewrite_ui_domains(b"\xff\xfe", is_sse=False, new_domain="x") is None


def _upstream(
    body: bytes,
    content_type: str = "text/event-stream",
    status_code: int = 200,
) -> MagicMock:
    """The httpx.Response surface mcp_proxy streams from."""
    upstream = MagicMock()
    upstream.status_code = status_code
    upstream.headers = {"content-type": content_type}

    async def _aiter(
        chunk_size: int = 64 * 1024,
    ) -> AsyncIterator[bytes]:
        yield body

    upstream.aiter_bytes = _aiter
    return upstream


def _patch_upstream(
    upstream_response: MagicMock,
):
    """Patch httpx.AsyncClient so mcp_proxy's stream chain yields this response."""
    stream_cm = AsyncMock()
    stream_cm.__aenter__ = AsyncMock(return_value=upstream_response)
    stream_cm.__aexit__ = AsyncMock(return_value=False)
    client = AsyncMock()
    client.stream = MagicMock(return_value=stream_cm)
    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__ = AsyncMock(return_value=False)
    return patch("auth_server.server.httpx.AsyncClient", return_value=client)


def _patch_scope_repo_allow_all():
    """Grant the wildcard scope so the handler's re-authorization passes."""
    repo = AsyncMock()

    async def _get_server_scopes(scope_name: str):
        if scope_name == "admin:all":
            return [{"server": "*", "methods": ["*"], "tools": ["*"]}]
        return []

    repo.get_server_scopes.side_effect = _get_server_scopes
    return patch("auth_server.server.get_scope_repository", return_value=repo)


def _proxy_token_headers() -> dict:
    """Mint the X-Internal-Token nginx's /validate hop would forward."""
    from auth_server.internal_request_token import mint_mcp_proxy_token

    token = mint_mcp_proxy_token(
        subject="test-user",
        scopes=["admin:all"],
        server_name=SERVER_NAME,
        upstream_url=UPSTREAM_URL,
    )
    return {
        "X-Internal-Token": token,
        "X-Registered-Server-Path": "/" + SERVER_NAME,
        "X-Registered-Route-Mode": "direct",
    }


class TestMcpProxyRewritesUiDomain:
    """The route-level outcome for MCP App page reads through the gateway."""

    def _post(
        self,
        upstream_body: bytes,
        method: str = "resources/read",
        content_type: str = "text/event-stream",
        params: dict | None = None,
        status_code: int = 200,
        path_suffix: str = "/mcp",
    ):
        with (
            _patch_upstream(_upstream(upstream_body, content_type, status_code)),
            _patch_scope_repo_allow_all(),
            patch.object(server, "_read_mcp_filter_enabled", return_value=False),
            # tools/call checks blocked tools in the database; keep the test off it,
            # as test_server.py does, so it does not depend on worker event loops.
            patch.object(server, "_get_blocked_tools", new=AsyncMock(return_value=set())),
        ):
            client = TestClient(server.app)
            return client.post(
                f"/mcp-proxy/{SERVER_NAME}{path_suffix}",
                json={
                    "jsonrpc": "2.0",
                    "id": 3,
                    "method": method,
                    "params": params or {"uri": "ui://drawio/mcp-app.html"},
                },
                headers=_proxy_token_headers(),
            )

    @pytest.fixture(autouse=True)
    def _gateway_url(self, monkeypatch):
        """Configure the public gateway URL the way a deployment does."""
        monkeypatch.setenv("REGISTRY_URL", GATEWAY_URL)
        monkeypatch.delenv("AUTH_SERVER_EXTERNAL_URL", raising=False)

    def test_page_domain_is_rewritten_to_the_gateway_url(self, monkeypatch):
        """The SSE page reply carries the hash of the URL the client connected to."""

        response = self._post(_sse(_page_reply(DRAWIO_DOMAIN)))

        assert response.status_code == 200
        expected = claude_ui_domain(f"{GATEWAY_URL}/{SERVER_NAME}/mcp")
        assert _domain_in(response.content) == expected

    def test_json_page_reply_is_rewritten(self, monkeypatch):
        """A plain JSON reply is rewritten the same way."""

        response = self._post(
            json.dumps(_page_reply(DRAWIO_DOMAIN)).encode(),
            content_type="application/json",
        )

        expected = claude_ui_domain(f"{GATEWAY_URL}/{SERVER_NAME}/mcp")
        assert _domain_in(response.content) == expected

    def test_other_methods_pass_through_unchanged(self, monkeypatch):
        """Only resources/read is rewritten; a tool result is forwarded byte for byte."""
        body = _sse(_page_reply(DRAWIO_DOMAIN))

        response = self._post(
            body,
            method="tools/call",
            params={"name": "create_diagram", "arguments": {}},
        )

        assert response.status_code == 200

        assert response.content == body

    def test_no_public_url_configured_leaves_reply_unchanged(self, monkeypatch):
        """Without a configured gateway URL there is nothing to hash."""
        monkeypatch.delenv("REGISTRY_URL", raising=False)
        body = _sse(_page_reply(DRAWIO_DOMAIN))

        response = self._post(body)

        assert response.content == body

    def test_error_reply_is_unchanged(self):
        """A non-2xx resources/read reply is forwarded as the backend sent it."""
        body = json.dumps(_page_reply(DRAWIO_DOMAIN)).encode()

        response = self._post(body, content_type="application/json", status_code=500)

        assert response.content == body

    def test_hash_uses_the_forwarded_path_and_query(self):
        """A trailing slash and a query string are part of the hashed URL."""
        response = self._post(_sse(_page_reply(DRAWIO_DOMAIN)), path_suffix="/mcp/?v=2")

        expected = claude_ui_domain(f"{GATEWAY_URL}/{SERVER_NAME}/mcp/?v=2")
        assert _domain_in(response.content) == expected


class TestRewriteGate:
    """_should_rewrite_mcp_apps_ui_domain: which replies are rewritten."""

    def test_direct_successful_page_read_is_rewritten(self):
        assert server._should_rewrite_mcp_apps_ui_domain(True, "resources/read", 200) is True

    def test_virtual_backend_hop_is_not_rewritten(self):
        """On a virtual server the client connected to the virtual URL."""
        assert server._should_rewrite_mcp_apps_ui_domain(False, "resources/read", 200) is False

    def test_other_methods_and_errors_are_not_rewritten(self):
        assert server._should_rewrite_mcp_apps_ui_domain(True, "tools/call", 200) is False
        assert server._should_rewrite_mcp_apps_ui_domain(True, "resources/read", 404) is False
        assert server._should_rewrite_mcp_apps_ui_domain(True, None, 200) is False
