"""End-to-end tests for forward-proxy egress through a real stub proxy (#1832).

The unit suite in ``tests/unit/utils/test_url_guard_forward_proxy.py`` covers the
routing decision with mocks. These tests exercise the decision through real
``httpx`` and ``httpcore`` against a stub forward proxy, which is the only way to
catch the failure mode that matters most here: the guard deciding correctly but
wiring ``httpx.Proxy`` into the delegate transport wrongly, so no request ever
reaches the proxy. A mock cannot see that.

The stub is a ~50-line asyncio server, so no Squid, no tinyproxy, and no new
dependency. It records every request line it receives, which is what the
assertions read.

What is deliberately NOT covered here: a TLS-intercepting proxy. Proving that
end to end needs a certificate authority and a re-signed upstream certificate.
The CA-bundle code paths that can be tested without one (missing path raises,
malformed PEM raises, the bundle loads on top of the default roots) are in the
unit suite.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from registry.exceptions import UrlValidationError
from registry.utils import url_guard

pytestmark = pytest.mark.asyncio

# The target hostname is made to resolve to a public address so the guard routes
# it through the proxy. The stub proxy ignores the address and answers locally.
PUBLIC_IP = "93.184.216.34"
PRIVATE_IP = "10.0.0.5"
UPSTREAM_HOST = "upstream.example"


class StubForwardProxy:
    """A minimal HTTP forward proxy that records what it is asked to do.

    Handles both shapes a forward proxy sees:

    - ``CONNECT host:port`` for an https target. Answers ``200 Connection
      established`` and then closes, because carrying the tunnel further would
      need a certificate for the upstream. The recorded request line is the
      assertion: it proves the tunnel was requested by HOSTNAME rather than by a
      pinned IP, which is what lets a real tunnel verify the certificate.
    - ``GET http://host/path`` (absolute-URI forward) for an http target.
      Answers with a canned response, so a full round trip can be asserted.
    """

    def __init__(self, forward_status: int = 401) -> None:
        self.forward_status = forward_status
        self.request_lines: list[str] = []
        self._server: asyncio.Server | None = None

    @property
    def url(self) -> str:
        assert self._server is not None, "proxy not started"
        host, port = self._server.sockets[0].getsockname()[:2]
        return f"http://{host}:{port}"

    async def __aenter__(self) -> "StubForwardProxy":
        self._server = await asyncio.start_server(self._handle, "127.0.0.1", 0)
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()

    async def _handle(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        try:
            raw = await asyncio.wait_for(reader.readline(), timeout=5)
            if not raw:
                return
            request_line = raw.decode("latin-1").strip()
            self.request_lines.append(request_line)

            # Drain the headers so the client's write completes.
            while True:
                header = await asyncio.wait_for(reader.readline(), timeout=5)
                if header in (b"\r\n", b"\n", b""):
                    break

            if request_line.upper().startswith("CONNECT"):
                writer.write(b"HTTP/1.1 200 Connection established\r\n\r\n")
            else:
                body = b'{"jsonrpc":"2.0","error":{"code":-32001,"message":"unauthorized"}}'
                writer.write(
                    f"HTTP/1.1 {self.forward_status} Unauthorized\r\n".encode()
                    + b"Content-Type: application/json\r\n"
                    + f"Content-Length: {len(body)}\r\n".encode()
                    + b"Connection: close\r\n\r\n"
                    + body
                )
            await writer.drain()
        except (TimeoutError, ConnectionError):
            pass
        finally:
            writer.close()


def _reset_caches() -> None:
    url_guard._skill_allowlist.cache_clear()
    url_guard._proxy_allowlist.cache_clear()
    url_guard._builtin_airegistry_tools_allowlist.cache_clear()
    url_guard._credentialed_oauth_allowlist.cache_clear()
    url_guard._forward_proxy_config.cache_clear()
    url_guard._forward_proxy_ssl_context.cache_clear()


@pytest.fixture(autouse=True)
def _clear_caches():
    _reset_caches()
    yield
    _reset_caches()


def _settings(ssrf_allowed_hosts: str = ""):
    s = MagicMock()
    s.github_extra_hosts = ""
    s.ssrf_allowed_hosts = ssrf_allowed_hosts
    s.ssrf_allowed_cidrs = ""
    s.gateway_proxy_allow_private_targets = False
    s.egress_oauth_trusted_idp_hosts = ""
    return s


def _resolves_to(*ips: str):
    """Stub the event loop resolver, leaving the real classifier in place."""
    return patch.object(
        asyncio.get_event_loop(),
        "getaddrinfo",
        new=AsyncMock(return_value=[(2, 1, 6, "", (ip, 443)) for ip in ips]),
    )


class TestRealProxyTransport:
    async def test_https_target_issues_connect_naming_the_hostname(self):
        """The reporter's path: an https target reaches the proxy as a CONNECT.

        The hostname in the CONNECT line is the whole point. A pinned request
        would read ``CONNECT 93.184.216.34:443``, and the tunnel would then verify
        the upstream certificate against an IP and fail.
        """
        async with StubForwardProxy() as proxy:
            env = {
                "EGRESS_FORWARD_PROXY_ENABLED": "true",
                "HTTPS_PROXY": proxy.url,
            }
            with (
                patch.dict("os.environ", env, clear=True),
                patch.object(url_guard, "settings", _settings()),
                _resolves_to(PUBLIC_IP),
            ):
                client = url_guard.guarded_async_client(
                    profile=url_guard.PROXY_PROFILE, timeout=5.0
                )
                async with client:
                    # The stub closes after answering the CONNECT, so the TLS
                    # handshake inside the tunnel cannot complete. That is
                    # expected: the assertion is the CONNECT line.
                    with pytest.raises(httpx.HTTPError):
                        await client.post(f"https://{UPSTREAM_HOST}/mcp", json={})

        assert proxy.request_lines, "the proxy was never contacted"
        assert proxy.request_lines[0] == f"CONNECT {UPSTREAM_HOST}:443 HTTP/1.1"

    async def test_http_target_round_trips_through_the_proxy(self):
        """A full request/response through the absolute-URI forward path.

        This is the reporter's success criterion in miniature: a proxy-only
        upstream answers 401 instead of the request timing out.
        """
        async with StubForwardProxy(forward_status=401) as proxy:
            env = {
                "EGRESS_FORWARD_PROXY_ENABLED": "true",
                "HTTP_PROXY": proxy.url,
            }
            with (
                patch.dict("os.environ", env, clear=True),
                patch.object(url_guard, "settings", _settings()),
                _resolves_to(PUBLIC_IP),
            ):
                client = url_guard.guarded_async_client(
                    profile=url_guard.PROXY_PROFILE, timeout=5.0
                )
                async with client:
                    response = await client.post(f"http://{UPSTREAM_HOST}/mcp", json={})

        assert response.status_code == 401
        assert proxy.request_lines[0] == f"POST http://{UPSTREAM_HOST}/mcp HTTP/1.1"

    async def test_a_401_through_the_proxy_counts_as_reachable(self):
        """The health gate's own verdict, driven through a real proxied client.

        ``_try_ping_without_auth`` treats 200/400/401/403 as reachable, which is
        what turns the reporter's 401 into a HEALTHY server and therefore into a
        live nginx location block instead of a commented-out one.
        """
        from registry.health.service import HealthMonitoringService

        async with StubForwardProxy(forward_status=401) as proxy:
            env = {
                "EGRESS_FORWARD_PROXY_ENABLED": "true",
                "HTTP_PROXY": proxy.url,
            }
            with (
                patch.dict("os.environ", env, clear=True),
                patch.object(url_guard, "settings", _settings()),
                _resolves_to(PUBLIC_IP),
            ):
                client = url_guard.guarded_async_client(
                    profile=url_guard.PROXY_PROFILE, timeout=5.0
                )
                async with client:
                    # Unbound call on an uninitialized instance: the method only
                    # uses its client and endpoint arguments, so constructing the
                    # full service (and its websocket manager) is unnecessary.
                    reachable = await HealthMonitoringService._try_ping_without_auth(
                        HealthMonitoringService.__new__(HealthMonitoringService),
                        client,
                        f"http://{UPSTREAM_HOST}/mcp",
                    )

        assert reachable is True
        assert proxy.request_lines, "the ping never reached the proxy"

    async def test_internal_target_never_reaches_the_proxy(self):
        """An internal target keeps its pin and is dialed directly.

        No NO_PROXY entry is configured, so this is the classification-derived
        routing rule doing its job: an in-cluster MCP server is not sent to a
        corporate proxy that cannot reach it.
        """
        async with StubForwardProxy() as proxy:
            env = {
                "EGRESS_FORWARD_PROXY_ENABLED": "true",
                "HTTP_PROXY": proxy.url,
                "HTTPS_PROXY": proxy.url,
            }
            with (
                patch.dict("os.environ", env, clear=True),
                patch.object(url_guard, "settings", _settings(ssrf_allowed_hosts=UPSTREAM_HOST)),
                _resolves_to(PRIVATE_IP),
            ):
                client = url_guard.guarded_async_client(
                    profile=url_guard.PROXY_PROFILE, timeout=2.0
                )
                async with client:
                    # Nothing listens on the pinned private address, so this
                    # fails. The assertion is that it failed DIRECTLY rather
                    # than via the proxy.
                    with pytest.raises(httpx.HTTPError):
                        await client.post(f"http://{UPSTREAM_HOST}/mcp", json={})

        assert proxy.request_lines == []

    async def test_metadata_resolution_is_denied_before_any_connect(self):
        """Classification still runs in full on the proxied path.

        The regression test for the mount bypass: had the fix been implemented by
        passing ``proxy=`` to the enclosing client, httpx would have mounted a
        plain transport under ``all://`` that shadows the guarded one, and this
        request would have been happily tunneled.
        """
        async with StubForwardProxy() as proxy:
            env = {
                "EGRESS_FORWARD_PROXY_ENABLED": "true",
                "HTTPS_PROXY": proxy.url,
            }
            with (
                patch.dict("os.environ", env, clear=True),
                patch.object(url_guard, "settings", _settings()),
                _resolves_to("169.254.169.254"),
            ):
                client = url_guard.guarded_async_client(
                    profile=url_guard.PROXY_PROFILE, timeout=5.0
                )
                async with client:
                    with pytest.raises(UrlValidationError):
                        await client.post(f"https://{UPSTREAM_HOST}/mcp", json={})

        assert proxy.request_lines == []

    async def test_flag_off_never_reaches_the_proxy(self):
        """With HTTP_PROXY exported but the flag off, nothing changes."""
        async with StubForwardProxy() as proxy:
            env = {"HTTP_PROXY": proxy.url, "HTTPS_PROXY": proxy.url}
            with (
                patch.dict("os.environ", env, clear=True),
                patch.object(url_guard, "settings", _settings()),
                _resolves_to("127.0.0.1"),
            ):
                client = url_guard.guarded_async_client(
                    profile=url_guard.PROXY_PROFILE, timeout=2.0
                )
                async with client:
                    with pytest.raises((httpx.HTTPError, UrlValidationError)):
                        await client.post(f"http://{UPSTREAM_HOST}/mcp", json={})

        assert proxy.request_lines == []
