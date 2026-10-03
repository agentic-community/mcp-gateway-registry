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
    """Stand-in for the pooled client returned by shared_plain_async_client."""

    def __init__(self, *, response=None, post_exc=None):
        self._response = response
        self._post_exc = post_exc
        self.post_calls = []

    async def post(self, url, **kwargs):
        self.post_calls.append((url, kwargs))
        if self._post_exc is not None:
            raise self._post_exc
        return self._response


def _patch_client(client):
    """Patch the pooled plain client this hop uses.

    The hop deliberately uses ``shared_plain_async_client()`` rather than a bare
    ``httpx.AsyncClient``, matching its sibling egress-token vend: a bare client
    defaults to ``trust_env=True``, so an operator-set HTTP_PROXY would send this
    in-cluster POST to a corporate forward proxy (issue #1832). Patch the same
    way ``_patch_vend_httpx`` in test_server.py does, so both vend test files
    agree on what they are standing in for.
    """
    return patch("registry.utils.url_guard.shared_plain_async_client", return_value=client)


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


async def test_uses_the_pooled_plain_client_not_a_bare_one():
    """The hop must be proxy-blind (issue #1832).

    A bare ``httpx.AsyncClient`` defaults to ``trust_env=True``, so an
    operator-set HTTP_PROXY would hand this in-cluster POST to the corporate
    forward proxy and the vend would fail unless NO_PROXY happened to cover the
    internal registry URL. ``shared_plain_async_client`` passes a custom
    transport, which is what keeps the request direct.
    """
    client = _FakeAsyncClient(response=_FakeResponse(200, json_value={"headers": {}}))
    with (
        _patch_mint(return_value="svc-token"),
        _patch_client(client) as pooled,
        patch.object(server.httpx, "AsyncClient") as bare,
    ):
        await server._vend_generic_upstream_headers("gtok", "server", "/svc")

    pooled.assert_called_once_with()
    bare.assert_not_called()


async def test_passes_the_vend_timeout_per_request():
    """A pooled client's default timeout is a fallback, so the hop must pass its own."""
    client = _FakeAsyncClient(response=_FakeResponse(200, json_value={"headers": {}}))
    with _patch_mint(return_value="svc-token"), _patch_client(client):
        await server._vend_generic_upstream_headers("gtok", "server", "/svc")

    _url, kwargs = client.post_calls[0]
    assert kwargs["timeout"] == server._egress_vend_timeout_seconds()
