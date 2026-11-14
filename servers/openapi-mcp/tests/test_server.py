"""Tests for the control API, the dispatcher and the hub lifecycle."""

import base64

import httpx
import hubs
import pytest
import server
from conftest import (
    SYNC_SECRET,
    mcp_post_body,
    parse_mcp_response,
)

from registry.auth.openapi_mcp_sync import mint_sync_token

MCP_HEADERS: dict = {
    "content-type": "application/json",
    "accept": "application/json, text/event-stream",
    "mcp-protocol-version": "2025-11-25",
}
# Headers nginx adds to a proxied MCP request. None may reach a third-party API.
GATEWAY_HEADERS: dict = {
    "x-user": "alice",
    "x-username": "alice",
    "x-scopes": "mcp-servers-unrestricted/execute",
    "x-forwarded-for": "10.0.0.5",
    "x-forwarded-host": "registry.example.com",
    "x-real-ip": "10.0.0.5",
    "x-resolved-upstream": "http://openapi-mcp:9876/s/demo/mcp",
    "x-validate-source-secret": "nginx-marker",
    "x-auth-method": "keycloak",
    "x-client-id": "agent",
    "user-agent": "claude-code/2.0",
    "x-custom-trace": "abc",
}


def _client() -> httpx.AsyncClient:
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=server.app), base_url="http://openapi-mcp"
    )


def _auth(
    purpose: str = "test",
) -> dict:
    return {"authorization": f"Bearer {mint_sync_token(SYNC_SECRET, purpose)}"}


def _target(
    spec: dict,
    name: str = "items",
    **auth,
) -> dict:
    return {"name": name, "base_url": "https://api.example.com/v1", "spec": spec, **auth}


def _state(
    targets: list[dict],
    generation: int = 1,
    slug: str = "demo",
) -> dict:
    return {
        "slug": slug,
        "name": "Demo",
        "description": "demo server",
        "generation": generation,
        "targets": targets,
    }


async def _put(
    client: httpx.AsyncClient,
    state: dict,
) -> httpx.Response:
    return await client.put(
        f"/internal/servers/{state['slug']}", json=state, headers=_auth("put_server")
    )


async def _call_tool(
    client: httpx.AsyncClient,
    tool: str,
    arguments: dict,
    extra_headers: dict | None = None,
) -> dict:
    headers = {**MCP_HEADERS, **(extra_headers or {})}
    body = mcp_post_body("tools/call", {"name": tool, "arguments": arguments})
    response = await client.post("/s/demo/mcp", content=body, headers=headers)
    assert response.status_code == 200, response.text
    return parse_mcp_response(response)


class TestControlApiAuth:
    """Every /internal route requires a fresh sync token."""

    @pytest.mark.parametrize(
        ("method", "path"),
        [
            ("PUT", "/internal/servers/demo"),
            ("DELETE", "/internal/servers/demo"),
            ("POST", "/internal/preview"),
            ("GET", "/internal/status"),
        ],
    )
    async def test_missing_token_is_401(self, method, path):
        async with _client() as client:
            response = await client.request(method, path, json={})

        assert response.status_code == 401

    async def test_reused_token_is_401(self):
        headers = _auth("status")
        async with _client() as client:
            first = await client.get("/internal/status", headers=headers)
            second = await client.get("/internal/status", headers=headers)

        assert (first.status_code, second.status_code) == (200, 401)

    async def test_health_needs_no_token(self):
        async with _client() as client:
            response = await client.get("/health")

        assert response.json()["status"] == "ok"


class TestHubLifecycle:
    """Push, status, generations, isolation and delete."""

    async def test_push_reports_tools_and_status(self, items_spec, backend):
        async with _client() as client:
            response = await _put(client, _state([_target(items_spec)]))
            status = await client.get("/internal/status", headers=_auth())

        assert response.json()["targets"]["items"]["status"] == "ok"
        assert "items_getItem" in response.json()["targets"]["items"]["tools"]
        assert status.json()["servers"]["demo"]["generation"] == 1

    async def test_stale_generation_ignored(self, items_spec, backend):
        async with _client() as client:
            await _put(client, _state([_target(items_spec)], generation=3))
            response = await _put(client, _state([], generation=2))

        assert response.json()["generation"] == 3
        assert "items" in response.json()["targets"]

    async def test_failing_target_does_not_block_others(self, items_spec, backend):
        bad = _target({"openapi": "3.1.0", "paths": {"/x": {"get": "broken"}}}, name="broken")
        async with _client() as client:
            response = await _put(client, _state([bad, _target(items_spec)]))

        targets = response.json()["targets"]
        assert targets["broken"]["status"] == "error"
        assert targets["items"]["status"] == "ok"

    async def test_swap_keeps_old_hub_open_during_grace(self, items_spec, backend):
        async with _client() as client:
            await _put(client, _state([_target(items_spec)], generation=1))
            old_clients = list(hubs.get_hub("demo").clients)
            await _put(client, _state([_target(items_spec)], generation=2))

        assert not any(c.is_closed for c in old_clients)

    async def test_swap_closes_old_clients_after_grace(self, items_spec, backend, monkeypatch):
        import asyncio

        monkeypatch.setattr(hubs, "RETIRED_HUB_GRACE_SECONDS", 0.05)
        async with _client() as client:
            await _put(client, _state([_target(items_spec)], generation=1))
            old_clients = list(hubs.get_hub("demo").clients)
            await _put(client, _state([_target(items_spec)], generation=2))
        await asyncio.sleep(0.3)

        assert all(c.is_closed for c in old_clients)

    async def test_request_in_flight_survives_a_swap(self, items_spec, monkeypatch):
        import asyncio

        import guarded_transport
        import httpx as _httpx

        release = asyncio.Event()

        async def _slow_backend(request):
            await release.wait()
            return _httpx.Response(200, json={"id": 1})

        monkeypatch.setattr(
            guarded_transport.url_guard,
            "GuardedAsyncTransport",
            lambda **kw: _httpx.MockTransport(_slow_backend),
        )
        async with _client() as client:
            await _put(client, _state([_target(items_spec)], generation=1))
            call = asyncio.create_task(_call_tool(client, "items_getItem", {"item_id": 1}))
            await asyncio.sleep(0.2)
            await _put(client, _state([_target(items_spec)], generation=2))
            release.set()
            result = await call

        assert result["result"]["structuredContent"] == {"id": 1}

    async def test_delete_then_404(self, items_spec, backend):
        async with _client() as client:
            await _put(client, _state([_target(items_spec)]))
            deleted = await client.delete("/internal/servers/demo", headers=_auth())
            again = await client.delete("/internal/servers/demo", headers=_auth())
            mcp = await client.post("/s/demo/mcp", content="{}", headers=MCP_HEADERS)

        assert (deleted.status_code, again.status_code, mcp.status_code) == (204, 404, 404)

    async def test_slug_mismatch_is_400(self, items_spec):
        state = _state([_target(items_spec)])
        async with _client() as client:
            response = await client.put("/internal/servers/other", json=state, headers=_auth())

        assert response.status_code == 400


class TestToolCalls:
    """Calls through /s/<slug>/mcp reach the backend with the right headers."""

    async def test_gateway_headers_never_reach_backend(self, items_spec, backend):
        target = _target(
            items_spec,
            auth_scheme="api_key_header",
            auth_header_name="X-Api-Key",
            auth_credential="k1",
        )
        async with _client() as client:
            await _put(client, _state([target]))
            await _call_tool(client, "items_getItem", {"item_id": 7}, GATEWAY_HEADERS)

        sent = backend.requests[0].headers
        leaked = {k for k, v in GATEWAY_HEADERS.items() if sent.get(k) == v}
        assert leaked == set()
        assert sent["x-api-key"] == "k1"

    async def test_unknown_argument_rejected(self, items_spec, backend):
        async with _client() as client:
            await _put(client, _state([_target(items_spec)]))
            result = await _call_tool(
                client, "items_getItem", {"item_id": 7, "current_weather": True}
            )

        assert result["result"]["isError"] is True
        assert backend.requests == []

    async def test_array_response_is_wrapped(self, items_spec, backend):
        async with _client() as client:
            await _put(client, _state([_target(items_spec)]))
            result = await _call_tool(client, "items_listItems", {})

        assert result["result"]["structuredContent"] == {"result": [{"id": 1}, {"id": 2}]}

    async def test_response_breaking_output_schema_still_returns(self, items_spec, backend):
        async with _client() as client:
            await _put(client, _state([_target(items_spec)]))
            result = await _call_tool(client, "items_getItem", {"item_id": 7})

        assert result["result"].get("isError") is not True
        assert result["result"]["structuredContent"] == {"id": 7}


class TestAuthSchemes:
    """Each auth scheme sends its credential to the backend."""

    @pytest.mark.parametrize(
        ("auth", "check"),
        [
            (
                {
                    "auth_scheme": "api_key_header",
                    "auth_header_name": "api_key",
                    "auth_credential": "k1",
                },
                lambda r: r.headers["api_key"] == "k1",
            ),
            (
                {
                    "auth_scheme": "api_key_query",
                    "auth_query_param": "apikey",
                    "auth_credential": "k2",
                },
                lambda r: r.url.params["apikey"] == "k2",
            ),
            (
                {"auth_scheme": "bearer", "auth_credential": "t3"},
                lambda r: r.headers["authorization"] == "Bearer t3",
            ),
            (
                {"auth_scheme": "basic", "auth_username": "u", "auth_credential": "p4"},
                lambda r: (
                    r.headers["authorization"] == "Basic " + base64.b64encode(b"u:p4").decode()
                ),
            ),
            ({"auth_scheme": "none"}, lambda r: "authorization" not in r.headers),
        ],
    )
    async def test_scheme_reaches_backend(self, items_spec, backend, auth, check):
        async with _client() as client:
            await _put(client, _state([_target(items_spec, **auth)]))
            await _call_tool(client, "items_getItem", {"item_id": 7})

        assert check(backend.requests[0])
        assert str(backend.requests[0].url).startswith("https://api.example.com/v1/items/7")
