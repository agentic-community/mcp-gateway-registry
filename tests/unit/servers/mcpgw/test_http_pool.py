"""Unit tests for the pooled mcpgw egress client (servers/mcpgw/http_pool.py).

Covers what pooling must guarantee for a client shared by every mcpgw tool call and
by every concurrent caller: one instance per process, limits/retries taken from the
EGRESS_HTTP_POOL_* env vars (with the registry's clamp and bounds semantics), no
cookie persistence, a clean rebuild after close, and the single transparent retry
when a pooled keep-alive was closed while idle.

The mcpgw image does not contain the `registry` package, so this module is a
deliberate port of registry/utils/url_guard.py's pooled accessors; these tests are
the mcpgw-side counterpart of tests/unit/utils/test_shared_http_clients.py.
"""

from __future__ import annotations

import asyncio
import contextlib
import ipaddress
import socket
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

import httpx
import pytest

# servers/mcpgw must be importable as a flat directory because server.py does
# `from http_pool import ...` (the container runs with /app as the root).
_mcpgw_path = str(Path(__file__).resolve().parents[4] / "servers" / "mcpgw")
if _mcpgw_path not in sys.path:
    sys.path.insert(0, _mcpgw_path)

import servers.mcpgw.http_pool as http_pool  # noqa: E402

_POOL_ENV_VARS = (
    "EGRESS_HTTP_POOL_MAX_CONNECTIONS",
    "EGRESS_HTTP_POOL_MAX_KEEPALIVE",
    "EGRESS_HTTP_POOL_KEEPALIVE_EXPIRY_SECONDS",
    "EGRESS_HTTP_POOL_CONNECT_RETRIES",
)


@pytest.fixture(autouse=True)
def _clean_pool(monkeypatch):
    """Start every test with no pooled client and no inherited pool env."""
    for name in _POOL_ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    http_pool.reset_shared_client_for_tests()
    yield
    http_pool.reset_shared_client_for_tests()


@pytest.fixture
async def keepalive_server():
    """Minimal HTTP/1.1 keep-alive server on 127.0.0.1 (IPv4 only).

    Yields ``(port, connections)``; ``connections`` records one entry per TCP
    connection the server accepted, which is what pooling is supposed to minimize.
    """
    connections: list[object] = []

    async def _handle(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        connections.append(writer.get_extra_info("peername"))
        try:
            while True:
                head = await reader.readuntil(b"\r\n\r\n")
                length = 0
                for line in head.split(b"\r\n"):
                    if line.lower().startswith(b"content-length:"):
                        length = int(line.split(b":", 1)[1])
                if length:
                    await reader.readexactly(length)
                writer.write(b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\n\r\n{}")
                await writer.drain()
        except (asyncio.IncompleteReadError, ConnectionError):
            pass
        finally:
            writer.close()

    server = await asyncio.start_server(_handle, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    try:
        yield port, connections
    finally:
        await http_pool.aclose_shared_client()
        server.close()
        await server.wait_closed()


async def test_sequential_tool_calls_reuse_one_connection(monkeypatch, keepalive_server):
    # The point of the change: every tool call goes through the accessor, and they
    # must share warm keep-alives instead of opening a connection each.
    port, connections = keepalive_server
    monkeypatch.setenv("REGISTRY_BASE_URL", f"http://registry.test:{port}")
    monkeypatch.setattr(socket, "getaddrinfo", _fake_getaddrinfo("127.0.0.1"))

    for _ in range(5):
        response = await http_pool.shared_async_client().get(
            f"http://registry.test:{port}/api/servers", timeout=5.0
        )
        assert response.status_code == 200

    assert len(connections) == 1


async def test_unreachable_first_answer_does_not_defeat_reuse(monkeypatch, keepalive_server):
    # Regression: answers were always tried in resolver order, so with `localhost` ->
    # ::1 (no listener) + 127.0.0.1, every request opened a fresh failed connect to ::1
    # first, and those failed connections evicted the idle keep-alive to 127.0.0.1 --
    # one new TCP connection per request. MAX_KEEPALIVE=1 makes the eviction
    # deterministic; at the default it shows up under concurrency instead.
    port, connections = keepalive_server
    monkeypatch.setenv("REGISTRY_BASE_URL", f"http://registry.test:{port}")
    monkeypatch.setenv("EGRESS_HTTP_POOL_MAX_KEEPALIVE", "1")
    monkeypatch.setattr(socket, "getaddrinfo", _fake_getaddrinfo("::1", "127.0.0.1"))

    for _ in range(5):
        response = await http_pool.shared_async_client().get(
            f"http://registry.test:{port}/api/servers", timeout=5.0
        )
        assert response.status_code == 200

    assert len(connections) == 1


async def test_rebuilds_after_close():
    first = http_pool.shared_async_client()
    await first.aclose()
    second = http_pool.shared_async_client()
    assert second is not first
    assert not second.is_closed


async def test_aclose_closes_and_accessor_rebuilds():
    client = http_pool.shared_async_client()
    await http_pool.aclose_shared_client()
    assert client.is_closed
    assert http_pool.shared_async_client() is not client


async def test_aclose_is_noop_without_a_pool():
    # Lifespan shutdown must not raise when no tool ever made a request.
    await http_pool.aclose_shared_client()


async def test_pool_limits_default_to_registry_defaults():
    limits = http_pool._pool_limits()
    assert limits.max_connections == 100
    assert limits.max_keepalive_connections == 20
    assert limits.keepalive_expiry == 30.0


async def test_pool_limits_read_env(monkeypatch):
    monkeypatch.setenv("EGRESS_HTTP_POOL_MAX_CONNECTIONS", "42")
    monkeypatch.setenv("EGRESS_HTTP_POOL_MAX_KEEPALIVE", "7")
    monkeypatch.setenv("EGRESS_HTTP_POOL_KEEPALIVE_EXPIRY_SECONDS", "11.5")
    limits = http_pool._pool_limits()
    assert limits.max_connections == 42
    assert limits.max_keepalive_connections == 7
    assert limits.keepalive_expiry == 11.5


async def test_max_keepalive_clamped_to_max_connections(monkeypatch):
    # Same clamp as registry/core/config.py: httpx must never hold more idle
    # connections than the pool ceiling allows. Asserted on the LIVE pool, because
    # a correct helper is worthless if the client does not receive its value.
    monkeypatch.setenv("EGRESS_HTTP_POOL_MAX_CONNECTIONS", "5")
    monkeypatch.setenv("EGRESS_HTTP_POOL_MAX_KEEPALIVE", "50")
    pool = http_pool.shared_async_client()._transport._pool
    assert pool._max_connections == 5
    assert pool._max_keepalive_connections == 5


@pytest.mark.parametrize("bad", ["not-a-number", "0", "-1", "100000", ""])
async def test_out_of_range_or_malformed_value_falls_back_to_default(monkeypatch, bad):
    # A typo must not produce an unbounded or degenerate pool; fail safe to default.
    monkeypatch.setenv("EGRESS_HTTP_POOL_MAX_CONNECTIONS", bad)
    assert http_pool.shared_async_client()._transport._pool._max_connections == 100


@pytest.mark.parametrize(("value", "expected"), [("0", 0), ("3", 3), ("9", 1), ("x", 1)])
async def test_connect_retries_bounded(monkeypatch, value, expected):
    monkeypatch.setenv("EGRESS_HTTP_POOL_CONNECT_RETRIES", value)
    assert http_pool._connect_retries() == expected
    assert http_pool.shared_async_client()._transport._pool._retries == expected


async def test_keepalive_expiry_zero_retires_idle_connections(monkeypatch):
    # The documented rollback lever: expiry 0 keeps the pooled client object but
    # retires an idle connection immediately, so no request reuses one.
    monkeypatch.setenv("EGRESS_HTTP_POOL_KEEPALIVE_EXPIRY_SECONDS", "0")
    assert http_pool._pool_limits().keepalive_expiry == 0.0
    assert http_pool.shared_async_client()._transport._pool._keepalive_expiry == 0.0


async def test_configured_limits_reach_the_connection_pool(monkeypatch):
    # Regression guard: httpx.AsyncClient DROPS its own ``limits=`` when an explicit
    # ``transport=`` is passed, so asserting _pool_limits() alone would pass while
    # every EGRESS_HTTP_POOL_* value stayed inert (httpx defaults 100/20/5s).
    monkeypatch.setenv("EGRESS_HTTP_POOL_MAX_CONNECTIONS", "9")
    monkeypatch.setenv("EGRESS_HTTP_POOL_MAX_KEEPALIVE", "4")
    monkeypatch.setenv("EGRESS_HTTP_POOL_KEEPALIVE_EXPIRY_SECONDS", "2.5")
    monkeypatch.setenv("EGRESS_HTTP_POOL_CONNECT_RETRIES", "3")
    pool = http_pool.shared_async_client()._transport._pool
    assert pool._max_connections == 9
    assert pool._max_keepalive_connections == 4
    assert pool._keepalive_expiry == 2.5
    assert pool._retries == 3


async def test_client_carries_no_shared_identity_header():
    # Credentials ride per-request headers only, so a pooled connection never
    # carries one caller's identity into another caller's request.
    client = http_pool.shared_async_client()
    lowered = {k.lower() for k in client.headers}
    assert "authorization" not in lowered
    assert "x-authorization" not in lowered


async def test_cookies_are_never_persisted():
    client = http_pool.shared_async_client()
    assert isinstance(client.cookies.jar, http_pool._NoStoreCookieJar)

    request = httpx.Request("GET", "https://registry.example/api/servers")
    response = httpx.Response(200, headers={"Set-Cookie": "session=abc; Path=/"}, request=request)
    client.cookies.extract_cookies(response)
    assert len(client.cookies) == 0


# ---------------------------------------------------------------------------
# SSRF guard: validate + pin per request, before pool checkout.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("address", "reason"),
    [
        ("169.254.169.254", "cloud/workload credential endpoint"),  # EC2 IMDS
        ("169.254.170.2", "cloud/workload credential endpoint"),  # ECS task creds
        ("169.254.170.23", "cloud/workload credential endpoint"),  # EKS Pod Identity
        ("fd00:ec2::254", "cloud/workload credential endpoint"),  # IMDS over IPv6
        ("fd00:ec2::23", "cloud/workload credential endpoint"),
        ("100.100.100.200", "cloud/workload credential endpoint"),  # Alibaba
        ("169.254.1.1", "link-local"),
        ("fe80::1", "link-local"),
        ("0.0.0.0", "unspecified address"),
        ("224.0.0.1", "multicast"),
        ("240.0.0.1", "reserved"),
    ],
)
async def test_hard_denied_addresses(address, reason):
    # These stay denied even though mcpgw must allow private/in-cluster targets --
    # that is the whole point of checking them before the category relaxations.
    assert http_pool._ip_denial_reason(ipaddress.ip_address(address)) == reason


@pytest.mark.parametrize(
    "address",
    [
        "10.0.1.5",  # in-cluster service IP
        "172.17.0.3",  # docker bridge
        "192.168.1.10",
        "100.64.0.1",  # CGNAT / overlay
        "127.0.0.1",  # dev / stdio mode default (REGISTRY_BASE_URL=http://localhost)
        "::1",
        "203.0.113.10",  # ordinary public address
    ],
)
async def test_allowed_addresses(address):
    assert http_pool._ip_denial_reason(ipaddress.ip_address(address)) is None


@pytest.mark.parametrize(
    "wrapper",
    [
        "::ffff:169.254.169.254",  # IPv4-mapped
        "64:ff9b::a9fe:a9fe",  # NAT64
        "2002:a9fe:a9fe::",  # 6to4
        "2001:0:0:0:0:0:5601:5601",  # Teredo (XOR'd 169.254.169.254)
        "fd00:ec2::254%eth0",  # scope id must not defeat the exact match
    ],
)
async def test_ipv6_wrappers_embedding_metadata_are_denied(wrapper):
    # Python classifies these wrappers as neither link-local nor private, so without
    # unwrapping they would reach the metadata endpoint.
    assert http_pool._ip_denial_reason(ipaddress.ip_address(wrapper)) is not None


def _fake_getaddrinfo(*addresses):
    def _inner(host, port, *args, **kwargs):
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (addr, port)) for addr in addresses]

    return _inner


@pytest.fixture
def captured_requests(monkeypatch):
    """Run the guard but stop before any real connection is made."""
    seen: list[httpx.Request] = []

    async def _stub_parent(self, request):
        seen.append(request)
        return httpx.Response(200, request=request)

    monkeypatch.setattr(httpx.AsyncHTTPTransport, "handle_async_request", _stub_parent)
    return seen


async def test_request_is_pinned_to_resolved_ip(monkeypatch, captured_requests):
    monkeypatch.setattr(socket, "getaddrinfo", _fake_getaddrinfo("10.0.1.5"))
    transport = http_pool._GuardedAsyncTransport()
    await transport.handle_async_request(httpx.Request("GET", "http://registry:8080/api/servers"))

    pinned = captured_requests[0]
    # Connect host becomes the validated IP -- which is also what keys the pool, so a
    # rebound hostname cannot reuse the previous host's connection.
    assert pinned.url.host == "10.0.1.5"
    # Routing and certificate identity keep the original hostname.
    assert pinned.headers["Host"] == "registry:8080"
    assert pinned.extensions["sni_hostname"] == "registry"


async def test_resolved_metadata_address_fails_closed(monkeypatch, captured_requests):
    monkeypatch.setattr(socket, "getaddrinfo", _fake_getaddrinfo("169.254.169.254"))
    transport = http_pool._GuardedAsyncTransport()
    with pytest.raises(http_pool.EgressTargetError, match="credential endpoint"):
        await transport.handle_async_request(
            httpx.Request("GET", "http://registry:8080/api/servers")
        )
    assert captured_requests == []  # never reached the connection


async def test_any_blocked_answer_fails_closed(monkeypatch, captured_requests):
    # A resolver returning a good answer AND a metadata answer must not be salvaged
    # by answer ordering.
    monkeypatch.setattr(socket, "getaddrinfo", _fake_getaddrinfo("10.0.1.5", "169.254.169.254"))
    transport = http_pool._GuardedAsyncTransport()
    with pytest.raises(http_pool.EgressTargetError):
        await transport.handle_async_request(
            httpx.Request("GET", "http://registry:8080/api/servers")
        )
    assert captured_requests == []


async def test_literal_metadata_host_is_denied(captured_requests):
    transport = http_pool._GuardedAsyncTransport()
    with pytest.raises(http_pool.EgressTargetError, match="credential endpoint"):
        await transport.handle_async_request(
            httpx.Request("GET", "http://169.254.169.254/latest/meta-data/")
        )
    assert captured_requests == []


async def test_literal_private_host_is_allowed(captured_requests):
    transport = http_pool._GuardedAsyncTransport()
    await transport.handle_async_request(httpx.Request("GET", "http://127.0.0.1:8080/api/servers"))
    assert captured_requests[0].url.host == "127.0.0.1"


async def test_non_http_scheme_is_denied(captured_requests):
    transport = http_pool._GuardedAsyncTransport()
    with pytest.raises(http_pool.EgressTargetError, match="scheme"):
        await transport.handle_async_request(httpx.Request("GET", "file://etc/passwd"))
    assert captured_requests == []


async def test_pooled_client_enforces_the_guard(monkeypatch, captured_requests):
    # End to end through the accessor the tools use: the pooled client is guarded,
    # so a metadata-resolving REGISTRY_BASE_URL never gets the bearer token.
    monkeypatch.setenv("REGISTRY_BASE_URL", "http://registry:8080")
    monkeypatch.setattr(socket, "getaddrinfo", _fake_getaddrinfo("169.254.170.2"))
    client = http_pool.shared_async_client()
    # match= so this can never pass for the wrong reason (e.g. a destination reject).
    with pytest.raises(http_pool.EgressTargetError, match="credential endpoint"):
        await client.get(
            "http://registry:8080/api/servers", headers={"Authorization": "Bearer secret"}
        )
    assert captured_requests == []


async def test_guard_runs_per_request_not_per_client(monkeypatch, captured_requests):
    # The pooled client is long-lived; validation must re-run on every request, so a
    # host that starts resolving to a denied address is rejected even after a
    # successful earlier request on the same client.
    monkeypatch.setenv("REGISTRY_BASE_URL", "http://registry:8080")
    client = http_pool.shared_async_client()
    monkeypatch.setattr(socket, "getaddrinfo", _fake_getaddrinfo("10.0.1.5"))
    resp = await client.get("http://registry:8080/api/servers")
    assert resp.status_code == 200

    monkeypatch.setattr(socket, "getaddrinfo", _fake_getaddrinfo("169.254.169.254"))
    with pytest.raises(http_pool.EgressTargetError, match="credential endpoint"):
        await client.get("http://registry:8080/api/servers")


class _FakeClient:
    """Stands in for a pooled client to exercise request_with_reconnect."""

    def __init__(self, fail_times: int, exc: Exception) -> None:
        self._fail_times = fail_times
        self._exc = exc
        self.calls: list[tuple[str, str]] = []

    async def request(self, method: str, url: str, **kwargs: object) -> httpx.Response:
        self.calls.append((method, url))
        if len(self.calls) <= self._fail_times:
            raise self._exc
        return httpx.Response(200, request=httpx.Request(method, url))


async def test_reconnect_retries_once_and_signals_reset():
    resets: list[str] = []
    client = _FakeClient(fail_times=1, exc=httpx.RemoteProtocolError("closed while idle"))
    resp = await http_pool.request_with_reconnect(
        client,  # type: ignore[arg-type]
        "POST",
        "https://registry.example/api/search/semantic",
        on_reset=lambda: resets.append("hit"),
    )
    assert resp.status_code == 200
    assert len(client.calls) == 2  # original + one retry
    assert resets == ["hit"]  # counted exactly once


async def test_reconnect_no_retry_on_success():
    client = _FakeClient(fail_times=0, exc=httpx.RemoteProtocolError("unused"))
    await http_pool.request_with_reconnect(
        client,  # type: ignore[arg-type]
        "GET",
        "https://registry.example/api/servers",
    )
    assert len(client.calls) == 1


async def test_reconnect_reraises_when_retry_also_fails():
    client = _FakeClient(fail_times=2, exc=httpx.RemoteProtocolError("down"))
    with pytest.raises(httpx.RemoteProtocolError):
        await http_pool.request_with_reconnect(
            client,  # type: ignore[arg-type]
            "GET",
            "https://registry.example/api/servers",
        )
    assert len(client.calls) == 2  # original + one retry, then fails closed


async def test_reconnect_does_not_retry_connect_error():
    # Connect failures are the transport's `retries=` job; retrying here too would
    # double the attempts on an unreachable upstream.
    client = _FakeClient(fail_times=1, exc=httpx.ConnectError("refused"))
    with pytest.raises(httpx.ConnectError):
        await http_pool.request_with_reconnect(
            client,  # type: ignore[arg-type]
            "GET",
            "https://registry.example/api/servers",
        )
    assert len(client.calls) == 1


async def test_reconnect_forwards_request_kwargs():
    captured: dict[str, object] = {}

    class _Recorder:
        async def request(self, method: str, url: str, **kwargs: object) -> httpx.Response:
            captured.update(kwargs)
            return httpx.Response(200, request=httpx.Request(method, url))

    await http_pool.request_with_reconnect(
        _Recorder(),  # type: ignore[arg-type]
        "POST",
        "https://registry.example/api/search/semantic",
        json={"query": "q"},
        headers={"X-Authorization": "Bearer t"},
        timeout=30.0,
    )
    # Per-request timeout and credential must reach the pooled client untouched.
    assert captured["timeout"] == 30.0
    assert captured["json"] == {"query": "q"}
    assert captured["headers"] == {"X-Authorization": "Bearer t"}


def _install_fastmcp_stub():
    """Install a MagicMock-based fastmcp stub and return it (fastmcp is not in this venv)."""
    stub = types.ModuleType("fastmcp")
    stub.Context = type("Context", (), {})
    mock_server = MagicMock()
    mock_server.tool.return_value = lambda fn: fn
    mock_server.custom_route.return_value = lambda fn: fn
    stub.FastMCP = MagicMock(return_value=mock_server)
    sys.modules["fastmcp"] = stub
    return stub


def _import_server_module():
    """Import servers.mcpgw.server, stubbing fastmcp only if nothing else already did."""
    if "fastmcp" not in sys.modules:
        _install_fastmcp_stub()

    import servers.mcpgw.server as mcpgw_server

    return mcpgw_server


@contextlib.contextmanager
def _freshly_imported_server():
    """Re-import servers.mcpgw.server under a recording stub, then restore sys.modules.

    Needed whenever an assertion depends on IMPORT-TIME state (the ``FastMCP(...)``
    call kwargs, the module-level URL constants read from env): the module object other
    tests already hold was built under whatever env existed then.
    """
    saved = {
        name: sys.modules.get(name)
        for name in ("fastmcp", "servers.mcpgw.server", "server", "http_pool")
    }
    stub = _install_fastmcp_stub()
    sys.modules.pop("servers.mcpgw.server", None)
    sys.modules.pop("server", None)
    try:
        import servers.mcpgw.server as fresh_server

        yield fresh_server, stub
    finally:
        for name, module in saved.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


async def test_server_lifespan_closes_the_pool():
    """The FastMCP lifespan shutdown half closes the pooled client."""
    mcpgw_server = _import_server_module()

    # server.py imports the flat `http_pool` module, which is the same file but a
    # distinct module object from `servers.mcpgw.http_pool`; assert against the one
    # the server actually uses.
    import http_pool as server_http_pool

    client = server_http_pool.shared_async_client()
    async with mcpgw_server._lifespan(None):
        assert not client.is_closed
    assert client.is_closed
    server_http_pool.reset_shared_client_for_tests()


async def test_registry_reset_is_attributed_to_the_registry_site(monkeypatch):
    """A pooled keep-alive reset on a tool call is retried and counted as mcpgw_registry."""
    mcpgw_server = _import_server_module()

    calls = {"n": 0}

    class _FlakyOnce:
        async def request(self, method, url, **kwargs):
            calls["n"] += 1
            if calls["n"] == 1:
                raise httpx.RemoteProtocolError("server disconnected without response")
            response = httpx.Response(
                200,
                json={"servers": [], "tools": [], "agents": [], "skills": []},
                request=httpx.Request(method, url),
            )
            return response

    sites: list[str] = []
    monkeypatch.setattr(mcpgw_server, "shared_async_client", lambda: _FlakyOnce())
    monkeypatch.setattr(mcpgw_server, "record_egress_conn_reset", sites.append)
    monkeypatch.setattr(mcpgw_server, "REGISTRY_API_TOKEN", "placeholder-token")

    result = await mcpgw_server.search_registry("docs search")

    # The caller never sees the reset, and the metric attributes it to the hop.
    assert result["status"] == "success", result
    assert calls["n"] == 2
    assert sites == ["mcpgw_registry"]


async def test_m2m_token_reset_is_attributed_to_the_token_site(monkeypatch):
    """The Keycloak token POST retries once and counts under mcpgw_m2m_token."""
    mcpgw_server = _import_server_module()

    calls = {"n": 0}

    class _FlakyOnce:
        async def request(self, method, url, **kwargs):
            calls["n"] += 1
            if calls["n"] == 1:
                raise httpx.RemoteProtocolError("server disconnected without response")
            return httpx.Response(
                200,
                json={"access_token": "placeholder-access-token", "expires_in": 300},
                request=httpx.Request(method, url),
            )

    sites: list[str] = []
    monkeypatch.setattr(mcpgw_server, "shared_async_client", lambda: _FlakyOnce())
    monkeypatch.setattr(mcpgw_server, "record_egress_conn_reset", sites.append)

    manager = mcpgw_server._M2MTokenManager(
        token_url="http://keycloak:8080/realms/r/protocol/openid-connect/token",
        client_id="mcp-gateway-m2m",
        client_secret="placeholder-secret",
    )
    token = await manager.get_token()

    assert token == "placeholder-access-token"
    assert calls["n"] == 2
    assert sites == ["mcpgw_m2m_token"]


# ---------------------------------------------------------------------------
# Multi-address fallback: validated answers are tried in order, under ONE budget.
# ---------------------------------------------------------------------------


@pytest.fixture
def attempted(monkeypatch):
    """Snapshot each pinned connect attempt; the first N addresses refuse the connection.

    The guard mutates ONE request object across attempts (as the registry's mixin
    does), so each attempt must be snapshotted at the time it is made -- holding the
    request objects would show every entry with the last attempt's values.
    """
    seen: list[dict[str, object]] = []
    refuse: dict[str, int] = {"first": 0}

    async def _stub_parent(self, request):
        timeout = dict(request.extensions.get("timeout") or {})
        seen.append(
            {
                "host": request.url.host,
                "http_host": request.headers.get("Host"),
                "sni": request.extensions.get("sni_hostname"),
                "timeout": timeout,
            }
        )
        if len(seen) <= refuse["first"]:
            raise httpx.ConnectError("connection refused", request=request)
        return httpx.Response(200, request=request)

    monkeypatch.setattr(httpx.AsyncHTTPTransport, "handle_async_request", _stub_parent)
    return seen, refuse


async def test_falls_back_to_the_next_validated_address(monkeypatch, attempted):
    # The dual-stack case that pinning answer[0] would break: `localhost` -> ::1 with
    # an IPv4-only listener still has to connect.
    seen, refuse = attempted
    refuse["first"] = 1
    monkeypatch.setattr(socket, "getaddrinfo", _fake_getaddrinfo("::1", "127.0.0.1"))
    transport = http_pool._GuardedAsyncTransport()

    response = await transport.handle_async_request(
        httpx.Request("GET", "http://localhost:8080/api/servers")
    )

    assert response.status_code == 200
    assert [a["host"] for a in seen] == ["::1", "127.0.0.1"]
    # Host and SNI stay the configured hostname on the fallback attempt too.
    assert seen[-1]["http_host"] == "localhost:8080"
    assert seen[-1]["sni"] == "localhost"


async def test_preferred_address_is_dropped_when_dns_no_longer_returns_it(monkeypatch, attempted):
    # The remembered address is only a reordering hint, never a cache: a request whose
    # fresh resolution no longer contains it must go to the new answer (rebind-safe).
    seen, refuse = attempted
    refuse["first"] = 1
    transport = http_pool._GuardedAsyncTransport()
    monkeypatch.setattr(socket, "getaddrinfo", _fake_getaddrinfo("::1", "10.0.1.5"))
    await transport.handle_async_request(httpx.Request("GET", "http://registry:8080/api"))
    await transport.handle_async_request(httpx.Request("GET", "http://registry:8080/api"))

    monkeypatch.setattr(socket, "getaddrinfo", _fake_getaddrinfo("10.0.9.9"))
    await transport.handle_async_request(httpx.Request("GET", "http://registry:8080/api"))

    # ::1 refused once; the second request went straight to the address that worked;
    # the third followed DNS to the new address.
    assert [a["host"] for a in seen] == ["::1", "10.0.1.5", "10.0.1.5", "10.0.9.9"]


async def test_reraises_last_error_when_every_address_fails(monkeypatch, attempted):
    seen, refuse = attempted
    refuse["first"] = 2
    monkeypatch.setattr(socket, "getaddrinfo", _fake_getaddrinfo("::1", "127.0.0.1"))
    transport = http_pool._GuardedAsyncTransport()

    with pytest.raises(httpx.ConnectError):
        await transport.handle_async_request(
            httpx.Request("GET", "http://localhost:8080/api/servers")
        )
    assert len(seen) == 2  # tried every validated address, then failed closed


async def test_connect_budget_is_split_across_addresses(monkeypatch, attempted):
    # Unguarded httpx walks the addresses itself inside ONE connect budget. Pinning
    # replaces that, so the budget must be divided -- otherwise a 30s call against a
    # dead dual-stack upstream would hang for 30s PER address (times connect retries).
    seen, refuse = attempted
    refuse["first"] = 1
    monkeypatch.setattr(socket, "getaddrinfo", _fake_getaddrinfo("::1", "127.0.0.1"))
    transport = http_pool._GuardedAsyncTransport()

    request = httpx.Request("GET", "http://localhost:8080/api/servers")
    request.extensions = {"timeout": {"connect": 30.0, "read": 30.0}}
    await transport.handle_async_request(request)

    assert [a["timeout"]["connect"] for a in seen] == [15.0, 15.0]
    # Only the connect phase is divided; read/write budgets are untouched.
    assert seen[-1]["timeout"]["read"] == 30.0


async def test_single_address_keeps_the_full_connect_budget(monkeypatch, attempted):
    seen, _refuse = attempted
    monkeypatch.setattr(socket, "getaddrinfo", _fake_getaddrinfo("10.0.1.5"))
    transport = http_pool._GuardedAsyncTransport()

    request = httpx.Request("GET", "http://registry:8080/api/servers")
    request.extensions = {"timeout": {"connect": 30.0}}
    await transport.handle_async_request(request)

    assert seen[0]["timeout"]["connect"] == 30.0


async def test_duplicate_resolver_answers_do_not_shrink_the_budget(monkeypatch, attempted):
    # getaddrinfo commonly returns the same address for several socktypes/protocols.
    seen, _refuse = attempted
    monkeypatch.setattr(socket, "getaddrinfo", _fake_getaddrinfo("10.0.1.5", "10.0.1.5"))
    transport = http_pool._GuardedAsyncTransport()

    request = httpx.Request("GET", "http://registry:8080/api/servers")
    request.extensions = {"timeout": {"connect": 30.0}}
    await transport.handle_async_request(request)

    assert len(seen) == 1
    assert seen[0]["timeout"]["connect"] == 30.0


# ---------------------------------------------------------------------------
# Destination allowlist: only the (host, port) pairs this process is configured for.
# ---------------------------------------------------------------------------


async def test_allowed_destinations_derived_from_env(monkeypatch):
    monkeypatch.setenv("REGISTRY_BASE_URL", "http://registry:8080")
    monkeypatch.setenv("KEYCLOAK_INTERNAL_URL", "https://keycloak.internal")
    assert http_pool._allowed_destinations() == frozenset(
        {("registry", 8080), ("keycloak.internal", 443)}
    )


async def test_allowed_destinations_cover_both_env_defaults(monkeypatch):
    """Unset env must still admit BOTH destinations the server builds URLs from.

    Regression: when this defaulted Keycloak to "" but ``server.py`` defaulted it to
    http://keycloak:8080, an M2M or OIDC deployment that left KEYCLOAK_INTERNAL_URL
    unset (the documented default) had every token POST and JWKS fetch rejected by its
    own guard -- i.e. every authenticated registry call failed.
    """
    monkeypatch.delenv("REGISTRY_BASE_URL", raising=False)
    monkeypatch.delenv("KEYCLOAK_INTERNAL_URL", raising=False)
    assert http_pool._allowed_destinations() == frozenset({("localhost", 80), ("keycloak", 8080)})


async def test_server_url_defaults_match_the_allowlist(monkeypatch):
    """Every URL the server builds must be admitted by the guard under default env.

    Re-imports the server module with the env cleared, because ``REGISTRY_URL`` and
    ``KEYCLOAK_INTERNAL_URL`` are import-time constants: this is what catches someone
    re-hardcoding a default in one module but not the other.
    """
    monkeypatch.delenv("REGISTRY_BASE_URL", raising=False)
    monkeypatch.delenv("KEYCLOAK_INTERNAL_URL", raising=False)
    allowed = http_pool._allowed_destinations()

    with _freshly_imported_server() as (fresh_server, _stub):
        for url in (fresh_server.REGISTRY_URL, fresh_server.KEYCLOAK_INTERNAL_URL):
            assert http_pool._destination(url) in allowed, url


async def test_pooled_client_enforces_the_env_destination_allowlist(monkeypatch, captured_requests):
    # The accessor must build its guard from REGISTRY_BASE_URL / KEYCLOAK_INTERNAL_URL:
    # both configured destinations go out, anything else fails before any connect.
    monkeypatch.setenv("REGISTRY_BASE_URL", "http://registry:8080")
    monkeypatch.setenv("KEYCLOAK_INTERNAL_URL", "http://keycloak:8080")
    monkeypatch.setattr(socket, "getaddrinfo", _fake_getaddrinfo("10.0.1.5"))
    client = http_pool.shared_async_client()

    await client.get("http://registry:8080/api/servers")
    await client.post("http://keycloak:8080/realms/r/protocol/openid-connect/token")
    with pytest.raises(http_pool.EgressTargetError, match="not a configured mcpgw destination"):
        await client.get("http://attacker.example:8080/api/servers")

    assert [r.headers["Host"] for r in captured_requests] == ["registry:8080", "keycloak:8080"]


async def test_m2m_token_post_passes_the_guard_on_default_keycloak_url(monkeypatch, attempted):
    """The live M2M path with KEYCLOAK_INTERNAL_URL unset must reach the upstream."""
    seen, _refuse = attempted
    monkeypatch.delenv("KEYCLOAK_INTERNAL_URL", raising=False)
    monkeypatch.setenv("REGISTRY_BASE_URL", "http://registry:8080")
    monkeypatch.setattr(socket, "getaddrinfo", _fake_getaddrinfo("10.0.2.9"))

    client = http_pool.shared_async_client()
    await client.post(
        f"{http_pool.DEFAULT_KEYCLOAK_INTERNAL_URL}/realms/r/protocol/openid-connect/token",
        data={"grant_type": "client_credentials"},
        timeout=15.0,
    )

    assert [a["http_host"] for a in seen] == ["keycloak:8080"]
    assert seen[0]["host"] == "10.0.2.9"


async def test_unconfigured_destination_is_rejected(captured_requests):
    transport = http_pool._GuardedAsyncTransport(
        allowed_destinations=frozenset({("registry", 8080)})
    )
    with pytest.raises(http_pool.EgressTargetError, match="not a configured mcpgw destination"):
        await transport.handle_async_request(httpx.Request("GET", "http://evil.example/steal"))
    # Same host on a different port is a different destination.
    with pytest.raises(http_pool.EgressTargetError, match="not a configured mcpgw destination"):
        await transport.handle_async_request(httpx.Request("GET", "http://registry:9999/api"))
    assert captured_requests == []


async def test_configured_destination_is_allowed(monkeypatch, captured_requests):
    monkeypatch.setattr(socket, "getaddrinfo", _fake_getaddrinfo("10.0.1.5"))
    transport = http_pool._GuardedAsyncTransport(
        allowed_destinations=frozenset({("registry", 8080)})
    )
    await transport.handle_async_request(httpx.Request("GET", "http://registry:8080/api/servers"))
    assert captured_requests[0].url.host == "10.0.1.5"


async def test_url_userinfo_is_rejected(captured_requests):
    # copy_with(host=ip) would carry userinfo onto the pinned request; credentials
    # belong in a per-request header.
    transport = http_pool._GuardedAsyncTransport()
    with pytest.raises(http_pool.EgressTargetError, match="userinfo"):
        await transport.handle_async_request(
            httpx.Request("GET", "http://user:pass@registry:8080/api/servers")
        )
    assert captured_requests == []


# ---------------------------------------------------------------------------
# Drift guard: the denial table is duplicated from url_guard by necessity.
# ---------------------------------------------------------------------------


async def test_credential_endpoint_set_matches_url_guard():
    from registry.utils import url_guard

    assert http_pool._CREDENTIAL_ENDPOINT_IPS == url_guard._CREDENTIAL_ENDPOINT_IPS


@pytest.mark.parametrize(
    "address",
    [
        "169.254.169.254",
        "fd00:ec2::254",
        "169.254.170.2",
        "169.254.170.23",
        "fd00:ec2::23",
        "100.100.100.200",
        "::ffff:169.254.169.254",
        "64:ff9b::a9fe:a9fe",
        "2002:a9fe:a9fe::",
        "fe80::1",
        "0.0.0.0",
        "224.0.0.1",
        "240.0.0.1",
        "10.0.1.5",
        "100.64.0.1",
        "127.0.0.1",
        "::1",
        "203.0.113.10",
    ],
)
async def test_classifier_matches_url_guard_allow_private_profile(address):
    # mcpgw's classifier must stay equivalent to the registry's with allow_private=True
    # (its EGRESS_UPSTREAM_PROFILE semantics). This pins the intended relationship, so
    # a new denial added on either side cannot silently skip the other.
    from registry.utils import url_guard

    ip = ipaddress.ip_address(address)
    assert http_pool._ip_denial_reason(ip) == url_guard._ip_denial_reason(ip, allow_private=True)


async def test_blocked_target_surfaces_as_a_failed_tool_call(monkeypatch):
    """EgressTargetError must reach the caller as a failed tool result, not a crash.

    The class subclasses ValueError specifically so the tools' ``except ValueError``
    branch handles it; that ordering (ValueError before HTTPStatusError before
    Exception) is the contract being pinned here.
    """
    mcpgw_server = _import_server_module()
    monkeypatch.setattr(mcpgw_server, "REGISTRY_API_TOKEN", "placeholder-token")
    monkeypatch.setattr(socket, "getaddrinfo", _fake_getaddrinfo("169.254.169.254"))

    import http_pool as server_http_pool

    server_http_pool.reset_shared_client_for_tests()
    try:
        result = await mcpgw_server.search_registry("docs search")
    finally:
        await server_http_pool.aclose_shared_client()

    assert result["status"] == "failed"
    assert "credential endpoint" in result["error"]
    assert result["total_results"] == 0


async def test_budget_split_applies_on_the_live_client_path(monkeypatch, attempted):
    """The split must hold for a real ``client.get(..., timeout=...)`` call.

    The other split test hand-builds ``request.extensions``; this one goes through
    httpx, which is what builds the timeout dict in production.
    """
    seen, refuse = attempted
    refuse["first"] = 1
    monkeypatch.setenv("REGISTRY_BASE_URL", "http://registry:8080")
    monkeypatch.setattr(socket, "getaddrinfo", _fake_getaddrinfo("::1", "10.0.1.5"))

    client = http_pool.shared_async_client()
    await client.get("http://registry:8080/api/servers", timeout=30.0)

    assert [a["timeout"]["connect"] for a in seen] == [15.0, 15.0]
    assert seen[-1]["timeout"]["read"] == 30.0
