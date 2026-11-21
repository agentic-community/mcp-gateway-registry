"""A backend that echoes the target's credential must not hand it to the model (issue #2022).

api.nasa.gov repeats the caller's api_key inside the "links" URLs of its
responses, so without redaction a per-user key reached the agent's context.
"""

import json
from urllib.parse import quote

import guarded_transport
import httpx
import pytest
from conftest import (
    mcp_post_body,
    parse_mcp_response,
)
from guarded_transport import (
    redact,
    secret_forms,
)
from test_per_user import (
    BASE_URL,
    _envelope,
    _per_user_target,
)
from test_server import (
    MCP_HEADERS,
    _client,
    _put,
    _state,
    _target,
)

SECRET: str = "s3cr3t/key+with=chars"


class EchoingBackend:
    """Echo the request URL and auth headers in the body and a response header."""

    def __init__(self) -> None:
        self.requests: list[httpx.Request] = []

    def handler(
        self,
        request: httpx.Request,
    ) -> httpx.Response:
        self.requests.append(request)
        echoed = {
            "self": str(request.url),
            "next": f"{BASE_URL}/items?page=2&api_key={quote(SECRET, safe='')}",
            "raw": SECRET,
            "auth": request.headers.get("authorization", ""),
        }
        return httpx.Response(200, json={"id": 7, "links": echoed}, headers={"x-echo-key": SECRET})


@pytest.fixture
def echo_backend(monkeypatch) -> EchoingBackend:
    recorder = EchoingBackend()

    def _factory(
        **kwargs,
    ) -> httpx.MockTransport:
        return httpx.MockTransport(recorder.handler)

    monkeypatch.setattr(guarded_transport.url_guard, "GuardedAsyncTransport", _factory)
    return recorder


async def _call_text(
    client,
    tool: str,
    extra_headers: dict | None = None,
) -> str:
    body = mcp_post_body("tools/call", {"name": tool, "arguments": {"item_id": 7}})
    response = await client.post(
        "/s/demo/mcp", content=body, headers={**MCP_HEADERS, **(extra_headers or {})}
    )
    return json.dumps(parse_mcp_response(response))


def _assert_no_secret(
    text: str,
) -> None:
    assert SECRET not in text
    assert quote(SECRET, safe="") not in text
    assert "[REDACTED]" in text


class TestRedactHelpers:
    def test_forms_cover_raw_and_url_encoded(self):
        forms = secret_forms(SECRET)

        assert SECRET.encode() in forms
        assert quote(SECRET, safe="").encode() in forms

    def test_short_or_empty_secret_is_not_redacted(self):
        assert secret_forms("abc", "", None) == ()

    def test_basic_form_included(self):
        forms = secret_forms("password1", basic_username="user")

        assert b"dXNlcjpwYXNzd29yZDE=" in forms

    def test_redact_replaces_every_occurrence(self):
        data = b"a=" + SECRET.encode() + b"&b=" + SECRET.encode()

        assert redact(data, secret_forms(SECRET)) == b"a=[REDACTED]&b=[REDACTED]"


class TestEchoedCredentials:
    async def test_per_user_key_redacted(self, items_spec, echo_backend):
        async with _client() as client:
            await _put(client, _state([_per_user_target(items_spec)]))
            text = await _call_text(
                client,
                "items_getItem",
                _envelope({"items": {"key": SECRET, "base_url": BASE_URL}}),
            )

        assert echo_backend.requests[0].headers["x-api-key"] == SECRET
        _assert_no_secret(text)

    async def test_shared_query_key_redacted(self, items_spec, echo_backend):
        target = _target(
            items_spec,
            auth_scheme="api_key_query",
            auth_query_param="api_key",
            auth_credential=SECRET,
        )
        async with _client() as client:
            await _put(client, _state([target]))
            text = await _call_text(client, "items_getItem")

        assert echo_backend.requests[0].url.params["api_key"] == SECRET
        _assert_no_secret(text)

    async def test_shared_basic_credential_redacted(self, items_spec, echo_backend):
        target = _target(
            items_spec,
            auth_scheme="basic",
            auth_username="svc",
            auth_credential=SECRET,
        )
        async with _client() as client:
            await _put(client, _state([target]))
            text = await _call_text(client, "items_getItem")

        assert echo_backend.requests[0].headers["authorization"].startswith("Basic ")
        _assert_no_secret(text)
        assert echo_backend.requests[0].headers["authorization"].split()[1] not in text
