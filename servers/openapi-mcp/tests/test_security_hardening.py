"""Hardening from the PR #2039 security review (issue #2022).

Each test drives a real backend client from hubs._backend_client against a
mock backend behind the real transport, so the checks run where they ship.
"""

import gzip
import json
import logging

import guarded_transport
import httpx
import hubs
import pytest
from guarded_transport import (
    BackendRequestFailedError,
    BackendUrlBlockedError,
    secret_forms,
)
from models import HubTargetState
from settings import (
    ContainerSettings,
    GuardSettings,
)

from registry.utils import url_guard

BASE: str = "https://api.example.com/approved/v1"
SPEC: dict = {"openapi": "3.0.3", "info": {"title": "t", "version": "1"}, "paths": {}}


class Backend:
    """Records requests; answers with whatever the test sets."""

    def __init__(self) -> None:
        self.requests: list[httpx.Request] = []
        self.response = httpx.Response(200, json={"ok": True})

    def handler(
        self,
        request: httpx.Request,
    ) -> httpx.Response:
        self.requests.append(request)
        return self.response


@pytest.fixture
def backend(monkeypatch) -> Backend:
    recorder = Backend()
    monkeypatch.setattr(
        guarded_transport.url_guard,
        "GuardedAsyncTransport",
        lambda **kwargs: httpx.MockTransport(recorder.handler),
    )
    return recorder


def _client(
    base_url: str = BASE,
    **auth,
):
    target = HubTargetState(name="api", base_url=base_url, spec=SPEC, **auth)
    return hubs._backend_client(target, ContainerSettings())


class TestCookies:
    async def test_a_backend_cookie_never_reaches_the_next_caller(self, backend):
        backend.response = httpx.Response(200, json={}, headers={"set-cookie": "session=alice"})
        async with _client() as client:
            first = await client.get("/me")
            await client.get("/me")

        assert "set-cookie" not in first.headers
        assert "cookie" not in backend.requests[1].headers


class TestDestination:
    @pytest.mark.parametrize(
        "path",
        ["/../../other/collect", "/%2e%2e/%2E%2E/x", "/..%2f..%2fx", "/a/%5c..%5cx", "/../v10/x"],
    )
    async def test_a_url_outside_the_base_path_is_refused_before_sending(self, backend, path):
        async with _client(
            auth_scheme="api_key_header", auth_header_name="X-Key", auth_credential="k" * 12
        ) as client:
            with pytest.raises(BackendUrlBlockedError):
                await client.get(path)

        assert backend.requests == []

    async def test_a_sibling_prefix_is_not_inside_the_base(self, backend):
        async with _client(base_url="https://api.example.com/v1") as client:
            with pytest.raises(BackendUrlBlockedError):
                await client.get("https://api.example.com/v10/x")

    async def test_ordinary_paths_under_the_base_are_sent(self, backend):
        async with _client() as client:
            await client.get("/items/7", params={"q": "a.b"})

        assert backend.requests[0].url.path == "/approved/v1/items/7"


class TestHttps:
    async def test_a_credentialed_call_over_http_is_refused(self, backend):
        async with _client(
            base_url="http://api.example.com/v1",
            auth_scheme="bearer",
            auth_credential="token-value-123",
        ) as client:
            with pytest.raises(BackendUrlBlockedError, match="HTTPS"):
                await client.get("/x")

        assert backend.requests == []

    async def test_an_operator_trusted_host_may_use_http(self, backend, monkeypatch, request):
        monkeypatch.setattr(url_guard, "settings", GuardSettings(ssrf_allowed_hosts="internal-api"))
        url_guard._proxy_allowlist.cache_clear()
        request.addfinalizer(url_guard._proxy_allowlist.cache_clear)
        async with _client(
            base_url="http://internal-api:8080/v1",
            auth_scheme="bearer",
            auth_credential="token-value-123",
        ) as client:
            await client.get("/x")

        assert len(backend.requests) == 1

    async def test_an_unauthenticated_target_may_use_http(self, backend):
        async with _client(base_url="http://api.example.com/v1") as client:
            await client.get("/x")

        assert len(backend.requests) == 1


class TestCompression:
    async def test_identity_is_requested_and_a_compressed_response_refused(self, backend):
        backend.response = httpx.Response(
            200, content=gzip.compress(b"x" * 10_000), headers={"content-encoding": "gzip"}
        )
        async with _client() as client:
            with pytest.raises(BackendRequestFailedError, match="compressed"):
                await client.get("/big")

        assert backend.requests[0].headers["accept-encoding"] == "identity"


class TestJsonEscapedRedaction:
    @pytest.mark.parametrize("secret", ['pa"ss\\\\word-1', "key/with/slashes"])
    async def test_a_json_escaped_echo_is_redacted(self, backend, secret):
        backend.response = httpx.Response(
            200, content=json.dumps({"echo": secret, "url": f"https://x/?k={secret}"}).encode()
        )
        async with _client(auth_scheme="bearer", auth_credential=secret) as client:
            response = await client.get("/echo")

        assert secret not in json.dumps(response.json(), ensure_ascii=False)
        assert "[REDACTED]" in response.text

    def test_escaped_forms_cover_quotes_and_non_ascii(self):
        forms = secret_forms('pa"ss-clé-secrète')

        assert b'pa\\"ss-cl\\u00e9-secr\\u00e8te' in forms


class TestLogging:
    async def test_a_blocked_backend_url_is_not_logged(self, caplog, monkeypatch):
        caplog.set_level(logging.WARNING)
        async with _client(
            base_url="http://169.254.169.254/x",
            auth_scheme="api_key_query",
            auth_query_param="api_key",
            auth_credential="secret-query-key",
        ) as client:
            with pytest.raises(BackendUrlBlockedError):
                await client.get("/latest")

        assert "secret-query-key" not in caplog.text
        assert "169.254.169.254" not in caplog.text
