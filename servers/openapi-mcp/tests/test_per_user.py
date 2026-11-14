"""Per-user credentials: the dispatcher verifies the gateway's envelope and each
per-user target sends the caller's own key (issue #2022)."""

import pytest
from conftest import (
    SYNC_SECRET,
    mcp_post_body,
    parse_mcp_response,
)
from test_server import (
    MCP_HEADERS,
    _client,
    _put,
    _state,
    _target,
)

from registry.auth.openapi_mcp_sync import (
    ENVELOPE_HEADER,
    mint_credential_envelope,
)

BASE_URL: str = "https://api.example.com/v1"


def _per_user_target(
    spec: dict,
    name: str = "items",
) -> dict:
    return _target(
        spec,
        name=name,
        auth_scheme="api_key_header",
        auth_header_name="X-Api-Key",
        auth_credential_mode="per_user",
    )


def _envelope(
    credentials: dict,
    slug: str = "demo",
    secret: str = SYNC_SECRET,
) -> dict:
    return {ENVELOPE_HEADER: mint_credential_envelope(secret, slug, credentials)}


async def _call(
    client,
    tool: str,
    extra_headers: dict | None = None,
):
    body = mcp_post_body("tools/call", {"name": tool, "arguments": {"item_id": 7}})
    return await client.post(
        "/s/demo/mcp", content=body, headers={**MCP_HEADERS, **(extra_headers or {})}
    )


class TestPerUserKeys:
    async def test_each_user_sends_their_own_key(self, items_spec, backend):
        async with _client() as client:
            await _put(client, _state([_per_user_target(items_spec)]))
            await _call(
                client,
                "items_getItem",
                _envelope({"items": {"key": "alice-key", "base_url": BASE_URL}}),
            )
            await _call(
                client,
                "items_getItem",
                _envelope({"items": {"key": "bob-key", "base_url": BASE_URL}}),
            )

        assert [r.headers["x-api-key"] for r in backend.requests] == ["alice-key", "bob-key"]

    async def test_bearer_scheme(self, items_spec, backend):
        target = _target(items_spec, auth_scheme="bearer", auth_credential_mode="per_user")
        async with _client() as client:
            await _put(client, _state([target]))
            await _call(
                client, "items_getItem", _envelope({"items": {"key": "tok", "base_url": BASE_URL}})
            )

        assert backend.requests[0].headers["authorization"] == "Bearer tok"

    async def test_missing_key_is_a_tool_error_and_no_call(self, items_spec, backend):
        async with _client() as client:
            await _put(client, _state([_per_user_target(items_spec)]))
            response = await _call(client, "items_getItem")

        result = parse_mcp_response(response)["result"]
        assert result["isError"] is True
        assert "No API key for 'items'" in result["content"][0]["text"]
        assert backend.requests == []

    async def test_key_for_another_destination_is_refused(self, items_spec, backend):
        async with _client() as client:
            await _put(client, _state([_per_user_target(items_spec)]))
            response = await _call(
                client,
                "items_getItem",
                _envelope({"items": {"key": "k", "base_url": "https://evil.example.com"}}),
            )

        assert parse_mcp_response(response)["result"]["isError"] is True
        assert backend.requests == []

    async def test_envelope_never_reaches_backend(self, items_spec, backend):
        async with _client() as client:
            await _put(client, _state([_per_user_target(items_spec)]))
            await _call(
                client, "items_getItem", _envelope({"items": {"key": "k", "base_url": BASE_URL}})
            )

        assert ENVELOPE_HEADER.lower() not in {k.lower() for k in backend.requests[0].headers}

    async def test_shared_target_needs_no_envelope(self, items_spec, backend):
        shared = _target(
            items_spec,
            name="shared",
            auth_scheme="api_key_header",
            auth_header_name="X-Api-Key",
            auth_credential="team-key",
        )
        async with _client() as client:
            await _put(client, _state([_per_user_target(items_spec), shared]))
            await _call(client, "shared_getItem")

        assert backend.requests[0].headers["x-api-key"] == "team-key"


class TestEnvelopeChecks:
    @pytest.mark.parametrize(
        "headers",
        [
            {ENVELOPE_HEADER: "not-a-jwt"},
            {
                ENVELOPE_HEADER: mint_credential_envelope(
                    "another-secret-0123456789abcdef-0123456789", "demo", {}
                )
            },
            {ENVELOPE_HEADER: mint_credential_envelope(SYNC_SECRET, "other-server", {})},
        ],
    )
    async def test_bad_envelope_is_401(self, items_spec, backend, headers):
        async with _client() as client:
            await _put(client, _state([_per_user_target(items_spec)]))
            response = await _call(client, "items_getItem", headers)

        assert response.status_code == 401
        assert backend.requests == []

    async def test_replayed_envelope_is_401(self, items_spec, backend):
        headers = _envelope({"items": {"key": "k", "base_url": BASE_URL}})
        async with _client() as client:
            await _put(client, _state([_per_user_target(items_spec)]))
            first = await _call(client, "items_getItem", headers)
            second = await _call(client, "items_getItem", headers)

        assert (first.status_code, second.status_code) == (200, 401)
