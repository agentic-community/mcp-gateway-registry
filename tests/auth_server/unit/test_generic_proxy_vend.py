"""Unit tests for _vend_generic_upstream_headers (generic egress vend hop).

Covers the internal service-token mint, the httpx POST to the registry
egress-internal endpoint, and every fail-closed branch plus the success
coercion path.
"""

import os
from unittest.mock import patch

import httpx
import pytest

os.environ.setdefault("SECRET_KEY", "test-secret-key-that-is-definitely-long-enough-32b")

import auth_server.server as server  # noqa: E402

pytestmark = pytest.mark.unit


class _FakeResponse:
    def __init__(self, status_code, json_value=None, json_exc=None):
        self.status_code = status_code
        self._json_value = json_value
        self._json_exc = json_exc

    def json(self):
        if self._json_exc is not None:
            raise self._json_exc
        return self._json_value


class _FakeAsyncClient:
    """Async-context-manager stand-in for httpx.AsyncClient."""

    def __init__(self, *, response=None, post_exc=None):
        self._response = response
        self._post_exc = post_exc
        self.post_calls = []

    def __call__(self, *args, **kwargs):
        # httpx.AsyncClient(timeout=...) construction; capture and return self.
        return self

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def post(self, url, **kwargs):
        self.post_calls.append((url, kwargs))
        if self._post_exc is not None:
            raise self._post_exc
        return self._response


def _patch_client(client):
    return patch.object(server.httpx, "AsyncClient", client)


def _patch_mint(**kwargs):
    return patch("registry.auth.internal.generate_internal_token", **kwargs)


async def test_mint_failure_returns_none():
    with _patch_mint(side_effect=ValueError("bad claims")):
        result = await server._vend_generic_upstream_headers("gtok", "server", "/svc")
    assert result is None


async def test_transport_error_returns_none():
    client = _FakeAsyncClient(post_exc=httpx.ConnectError("boom"))
    with _patch_mint(return_value="svc-token"), _patch_client(client):
        result = await server._vend_generic_upstream_headers("gtok", "server", "/svc")
    assert result is None
    assert client.post_calls, "httpx path must be reached"


async def test_non_200_returns_none():
    client = _FakeAsyncClient(response=_FakeResponse(503, json_value={"headers": {}}))
    with _patch_mint(return_value="svc-token"), _patch_client(client):
        result = await server._vend_generic_upstream_headers("gtok", "server", "/svc")
    assert result is None


async def test_json_decode_error_returns_none():
    client = _FakeAsyncClient(response=_FakeResponse(200, json_exc=ValueError("nope")))
    with _patch_mint(return_value="svc-token"), _patch_client(client):
        result = await server._vend_generic_upstream_headers("gtok", "server", "/svc")
    assert result is None


async def test_headers_not_dict_returns_none():
    client = _FakeAsyncClient(response=_FakeResponse(200, json_value={"headers": ["nope"]}))
    with _patch_mint(return_value="svc-token"), _patch_client(client):
        result = await server._vend_generic_upstream_headers("gtok", "server", "/svc")
    assert result is None


async def test_success_coerces_and_filters():
    payload = {
        "headers": {
            "X-Api-Key": "secret",
            "X-Num": 42,  # non-string value coerced to str
            123: "dropped",  # non-string key dropped defensively
        },
        "overridable_names": ["X-Api-Key", 999, None, "X-Other"],
    }
    client = _FakeAsyncClient(response=_FakeResponse(200, json_value=payload))
    with _patch_mint(return_value="svc-token"), _patch_client(client):
        result = await server._vend_generic_upstream_headers("gtok", "server", "/svc")

    assert result is not None
    defaults, overridable = result
    assert defaults == {"X-Api-Key": "secret", "X-Num": "42"}
    assert 123 not in defaults and "123" not in defaults
    assert overridable == ["X-Api-Key", "X-Other"]

    # The request carried the minted service token + forwarded generic token.
    url, kwargs = client.post_calls[0]
    assert url.endswith("/_egress_internal/generic-upstream-headers")
    assert kwargs["headers"]["Authorization"] == "Bearer svc-token"
    assert kwargs["headers"]["X-Internal-Token-Generic"] == "gtok"
    assert kwargs["json"] == {"entity_type": "server", "registered_path": "/svc"}


async def test_success_missing_overridable_defaults_empty():
    payload = {"headers": {"X-Api-Key": "secret"}}  # no overridable_names key
    client = _FakeAsyncClient(response=_FakeResponse(200, json_value=payload))
    with _patch_mint(return_value="svc-token"), _patch_client(client):
        result = await server._vend_generic_upstream_headers("gtok", "server", "/svc")

    assert result is not None
    defaults, overridable = result
    assert defaults == {"X-Api-Key": "secret"}
    assert overridable == []
