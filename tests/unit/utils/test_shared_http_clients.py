"""Unit tests for the pooled egress HTTP client accessors in url_guard.

Covers the connection-pooling contract: process-lifetime sharing keyed by
(profile, verify), cookie non-persistence (no cross-request/user bleed), the
keep-alive reconnect helper, and lifespan teardown/rebuild.
"""

from __future__ import annotations

import httpx
import pytest

from registry.utils import url_guard
from registry.utils.url_guard import (
    CREDENTIALED_OAUTH_PROFILE,
    PROXY_PROFILE,
    GuardedAsyncTransport,
    post_with_reconnect,
    shared_guarded_async_client,
    shared_plain_async_client,
)


class _StubSettings:
    """Minimal settings stub: pool knobs + the SSRF allowlist knobs the profiles read."""

    egress_http_pool_max_connections = 42
    egress_http_pool_max_keepalive = 7
    egress_http_pool_keepalive_expiry_seconds = 11.0
    egress_http_pool_connect_retries = 3
    ssrf_allowed_hosts = ""
    ssrf_allowed_cidrs = ""


@pytest.fixture(autouse=True)
async def _pool_env(monkeypatch):
    monkeypatch.setattr(url_guard, "settings", _StubSettings())
    url_guard.reset_shared_clients_for_tests()
    yield
    await url_guard.aclose_shared_clients()
    url_guard.reset_shared_clients_for_tests()


async def test_guarded_client_shared_per_profile_and_verify():
    a = shared_guarded_async_client(profile=PROXY_PROFILE)
    assert shared_guarded_async_client(profile=PROXY_PROFILE) is a
    # different profile -> different pooled client
    assert shared_guarded_async_client(profile=CREDENTIALED_OAUTH_PROFILE) is not a
    # verify is part of the key: a verify=False client is never reused where
    # verification is expected
    assert shared_guarded_async_client(profile=PROXY_PROFILE, verify=False) is not a


async def test_guarded_client_wraps_guarded_transport_with_profile():
    client = shared_guarded_async_client(profile=CREDENTIALED_OAUTH_PROFILE)
    assert isinstance(client._transport, GuardedAsyncTransport)
    # The SSRF guard is preserved: the shared client carries the profile's
    # per-request validate+pin transport.
    assert client._transport._guard_profile is CREDENTIALED_OAUTH_PROFILE


async def test_plain_client_shared():
    a = shared_plain_async_client()
    assert shared_plain_async_client() is a


async def test_pool_limits_reflect_settings():
    limits = url_guard._pool_limits()
    assert limits.max_connections == 42
    assert limits.max_keepalive_connections == 7
    assert limits.keepalive_expiry == 11.0


async def test_no_default_auth_header():
    # No shared default identity header: credentials ride per-request only.
    for client in (
        shared_guarded_async_client(profile=PROXY_PROFILE),
        shared_plain_async_client(),
    ):
        assert "authorization" not in {k.lower() for k in client.headers}


async def test_no_cookie_persistence_uses_no_store_jar():
    # Cookies are never stored (a no-store jar), so a Set-Cookie can never be
    # replayed onto a later OR concurrent request sharing the pooled client.
    from registry.utils.url_guard import _NoStoreCookieJar

    for client in (
        shared_plain_async_client(),
        shared_guarded_async_client(profile=PROXY_PROFILE),
    ):
        assert isinstance(client.cookies.jar, _NoStoreCookieJar)
        # Even an explicit set is dropped -> nothing to replay, no shared state to race.
        client.cookies.set("sid", "secret", domain="example.com")
        assert len(client.cookies) == 0
        # A real Set-Cookie extracted from a response is likewise not stored.
        resp = httpx.Response(
            200,
            headers=[("set-cookie", "sid=leaked; Domain=example.com; Path=/")],
            request=httpx.Request("GET", "https://example.com/"),
        )
        client.cookies.extract_cookies(resp)
        assert len(client.cookies) == 0


async def test_rebuilds_after_close():
    a = shared_plain_async_client()
    await a.aclose()
    b = shared_plain_async_client()
    assert b is not a
    assert not b.is_closed


async def test_aclose_closes_all_and_accessor_rebuilds():
    guarded = shared_guarded_async_client(profile=PROXY_PROFILE)
    plain = shared_plain_async_client()
    await url_guard.aclose_shared_clients()
    assert guarded.is_closed
    assert plain.is_closed
    # accessor lazily rebuilds a fresh open client
    assert shared_guarded_async_client(profile=PROXY_PROFILE) is not guarded


class _FakeClient:
    """Stands in for a pooled client to exercise post_with_reconnect."""

    def __init__(self, *, fail_times: int, exc: Exception):
        self._fail_times = fail_times
        self._exc = exc
        self.calls = 0

    async def post(self, url, **kwargs):
        self.calls += 1
        if self.calls <= self._fail_times:
            raise self._exc
        return httpx.Response(200, request=httpx.Request("POST", url))


async def test_post_with_reconnect_retries_once_and_signals():
    resets = []
    client = _FakeClient(fail_times=1, exc=httpx.RemoteProtocolError("server disconnected"))
    resp = await post_with_reconnect(
        client, "https://idp/token", data={"a": "b"}, on_reset=lambda: resets.append(1)
    )
    assert resp.status_code == 200
    assert client.calls == 2  # one failure, one successful retry
    assert resets == [1]  # on_reset fired exactly once


async def test_post_with_reconnect_no_retry_on_success():
    client = _FakeClient(fail_times=0, exc=httpx.ConnectError("unused"))
    resp = await post_with_reconnect(client, "https://idp/token")
    assert resp.status_code == 200
    assert client.calls == 1


async def test_post_with_reconnect_reraises_when_retry_also_fails():
    client = _FakeClient(fail_times=2, exc=httpx.RemoteProtocolError("down"))
    with pytest.raises(httpx.RemoteProtocolError):
        await post_with_reconnect(client, "https://idp/token")
    assert client.calls == 2  # original + one retry, then gives up


async def test_post_with_reconnect_does_not_retry_connect_error():
    # A connect failure is retried by the transport's ``retries=``, not here, so
    # post_with_reconnect must NOT double-retry it -- it propagates on the first call.
    client = _FakeClient(fail_times=1, exc=httpx.ConnectError("connect"))
    with pytest.raises(httpx.ConnectError):
        await post_with_reconnect(client, "https://idp/token")
    assert client.calls == 1  # no reconnect retry for a connect failure


async def test_pinning_coalesces_hostnames_sharing_an_ip(monkeypatch):
    # The guard rewrites the connect host to the resolved IP, so httpcore pools by
    # IP: two different hostnames that resolve to the SAME public IP pin to the same
    # origin -- the mechanism behind cross-hostname connection coalescing on a shared
    # client. The Host header and TLS SNI stay the ORIGINAL hostname per request, so
    # routing and cert identity are unaffected and each request is independently
    # validated+pinned (no SSRF bypass).
    from registry.utils import url_guard as ug

    async def _fake_resolve(hostname, port, allowlist, *, allow_private=False):
        return ["203.0.113.7"]  # one shared public IP for every hostname

    monkeypatch.setattr(ug, "_resolve_public_ips_async", _fake_resolve)

    transport = shared_guarded_async_client(profile=PROXY_PROFILE)._transport
    pinned_hosts = []
    for host in ("a.example.com", "b.example.com"):
        pinned = await transport._pin_request_async(httpx.Request("POST", f"https://{host}/mcp"))
        pinned_hosts.append(pinned.url.host)
        assert pinned.headers["host"] == host  # Host preserved (routing)
        assert pinned.extensions["sni_hostname"] == host  # SNI preserved (TLS identity)

    # Both hostnames pinned to the same IP -> same httpcore pool origin -> coalesce.
    assert pinned_hosts == ["203.0.113.7", "203.0.113.7"]
