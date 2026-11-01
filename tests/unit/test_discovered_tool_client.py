"""Tests for agents/discovered_tool_client.py: search, origin guard, and call setup."""

import asyncio
import json
import sys
from pathlib import Path

import httpx
import pytest

AGENTS_DIR = Path(__file__).resolve().parents[2] / "agents"
if str(AGENTS_DIR) not in sys.path:
    sys.path.insert(0, str(AGENTS_DIR))

import discovered_tool_client as dtc  # noqa: E402

REGISTRY_URL = "https://registry.example.com"


def _search_response() -> dict:
    return {
        "tools": [
            {
                "server_path": "/cloudflare-docs",
                "tool_name": "search_cloudflare_documentation",
                "endpoint_url": "https://registry.example.com/cloudflare-docs/mcp",
                "description": "Search the Cloudflare docs",
                "inputSchema": {"type": "object", "properties": {"query": {"type": "string"}}},
            },
            {
                "server_path": "/local-only",
                "tool_name": "local_tool",
                "endpoint_url": None,
            },
        ]
    }


def _patch_async_client(
    monkeypatch: pytest.MonkeyPatch,
    handler,
) -> None:
    """Make every httpx.AsyncClient in the module use a mock transport."""
    real_client = httpx.AsyncClient

    def factory(*args, **kwargs):
        kwargs["transport"] = httpx.MockTransport(handler)
        return real_client(*args, **kwargs)

    monkeypatch.setattr(dtc.httpx, "AsyncClient", factory)


class TestRequireRegistryOrigin:
    """The token must only ever go to the registry's own origin."""

    def test_accepts_same_origin(self):
        dtc._require_registry_origin(f"{REGISTRY_URL}/cloudflare-docs/mcp", REGISTRY_URL)

    def test_accepts_case_and_default_port_variants(self):
        dtc._require_registry_origin(
            "HTTPS://Registry.Example.com:443/cloudflare-docs/mcp", REGISTRY_URL
        )

    def test_rejects_other_host(self):
        with pytest.raises(ValueError, match="not the registry origin"):
            dtc._require_registry_origin("https://evil.example.net/mcp", REGISTRY_URL)

    def test_rejects_lookalike_subdomain(self):
        with pytest.raises(ValueError, match="not the registry origin"):
            dtc._require_registry_origin("https://registry.example.com.evil.net/mcp", REGISTRY_URL)

    def test_rejects_other_port(self):
        with pytest.raises(ValueError, match="not the registry origin"):
            dtc._require_registry_origin("https://registry.example.com:8443/mcp", REGISTRY_URL)

    def test_rejects_scheme_downgrade(self):
        with pytest.raises(ValueError, match="not the registry origin"):
            dtc._require_registry_origin("http://registry.example.com/mcp", REGISTRY_URL)

    def test_rejects_userinfo(self):
        with pytest.raises(ValueError, match="userinfo"):
            dtc._require_registry_origin("https://registry.example.com@evil.net/mcp", REGISTRY_URL)

    def test_rejects_non_http_scheme(self):
        with pytest.raises(ValueError, match="http or https"):
            dtc._require_registry_origin("file:///etc/passwd", REGISTRY_URL)


class TestParseTools:
    """Search results become DiscoveredTool objects."""

    def test_keeps_tools_with_endpoint_and_maps_schema(self):
        tools = dtc._parse_tools(_search_response())

        assert [t.tool_name for t in tools] == ["search_cloudflare_documentation"]
        assert tools[0].input_schema["properties"]["query"]["type"] == "string"

    def test_reads_matching_tools_under_servers_with_the_server_url(self):
        response = {
            "servers": [
                {
                    "path": "/ai-registry/cloudflare-docs",
                    "endpoint_url": "https://registry.example.com/ai-registry/cloudflare-docs/mcp",
                    "matching_tools": [
                        {"tool_name": "search_cloudflare_documentation", "inputSchema": None}
                    ],
                }
            ],
            "virtual_servers": [
                {
                    "path": "/virtual/dev-tools",
                    "endpoint_url": "https://registry.example.com/virtual/dev-tools/mcp",
                    "matching_tools": [{"tool_name": "lookup"}],
                }
            ],
        }

        tools = dtc._parse_tools(response)

        assert [(t.server_path, t.tool_name) for t in tools] == [
            ("/ai-registry/cloudflare-docs", "search_cloudflare_documentation"),
            ("/virtual/dev-tools", "lookup"),
        ]
        assert tools[0].endpoint_url.endswith("/ai-registry/cloudflare-docs/mcp")

    def test_keeps_a_tool_listed_twice_once(self):
        response = _search_response()
        response["servers"] = [
            {
                "path": "/cloudflare-docs",
                "endpoint_url": "https://registry.example.com/cloudflare-docs/mcp",
                "matching_tools": [{"tool_name": "search_cloudflare_documentation"}],
            }
        ]

        tools = dtc._parse_tools(response)

        assert [t.tool_name for t in tools] == ["search_cloudflare_documentation"]

    def test_skips_server_tools_when_the_server_has_no_endpoint(self):
        response = {"servers": [{"path": "/local", "matching_tools": [{"tool_name": "x"}]}]}

        assert dtc._parse_tools(response) == []

    def test_empty_or_missing_tools_array(self):
        assert dtc._parse_tools({}) == []
        assert dtc._parse_tools({"tools": None}) == []


class TestGetToken:
    """The token comes from REGISTRY_TOKEN only."""

    def test_reads_token(self, monkeypatch):
        monkeypatch.setenv("REGISTRY_TOKEN", " abc ")
        assert dtc._get_token() == "abc"

    def test_missing_token_names_the_variable(self, monkeypatch):
        monkeypatch.delenv("REGISTRY_TOKEN", raising=False)
        with pytest.raises(ValueError, match="REGISTRY_TOKEN"):
            dtc._get_token()


class TestSearchTools:
    """search_tools sends the user's token and does not filter by entity type.

    A tool-only filter returns no tools, because search lifts tools out of
    their server's results rather than indexing them on their own.
    """

    def test_sends_bearer_without_entity_filter(self, monkeypatch):
        seen = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen["url"] = str(request.url)
            seen["auth"] = request.headers.get("authorization")
            seen["body"] = request.read().decode()
            return httpx.Response(200, json=_search_response())

        _patch_async_client(monkeypatch, handler)
        tools = asyncio.run(dtc.search_tools(REGISTRY_URL + "/", "tok", "docs", 3))

        assert seen["url"] == "https://registry.example.com/api/search/semantic"
        assert seen["auth"] == "Bearer tok"
        assert "entity_types" not in json.loads(seen["body"])
        assert len(tools) == 1

    def test_raises_on_401(self, monkeypatch):
        _patch_async_client(monkeypatch, lambda request: httpx.Response(401))
        with pytest.raises(httpx.HTTPStatusError):
            asyncio.run(dtc.search_tools(REGISTRY_URL, "expired", "docs"))

    def test_rejects_out_of_range_max_results(self):
        with pytest.raises(ValueError, match="max_results"):
            asyncio.run(dtc.search_tools(REGISTRY_URL, "tok", "docs", 0))


class TestCallDiscoveredTool:
    """call_discovered_tool refuses a foreign endpoint before any network call."""

    def test_refuses_foreign_endpoint_without_sending_token(self, monkeypatch):
        def handler(request: httpx.Request) -> httpx.Response:
            raise AssertionError("no request may be sent to a foreign origin")

        _patch_async_client(monkeypatch, handler)
        tool = dtc.DiscoveredTool(
            server_path="/evil",
            tool_name="steal",
            endpoint_url="https://evil.example.net/mcp",
        )
        with pytest.raises(ValueError, match="not the registry origin"):
            asyncio.run(dtc.call_discovered_tool(tool, REGISTRY_URL, "tok"))
