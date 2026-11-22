"""MCP SDK connections run on the pinned, SSRF-guarded httpx2 transport.

_assert_mcp_url_fetchable refuses a blocked target early, but it resolves DNS in
one step and the SDK connects in another. The SDK client must therefore carry the
guarded transport, so a hostname that rebinds in between is refused at connect.
The transport-detection probes used the SDK's own default client and are covered
here too.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from registry.core import mcp_client
from registry.exceptions import UrlValidationError
from registry.utils import url_guard
from registry.utils.url_guard import PROXY_PROFILE
from registry.utils.url_guard_httpx2 import GuardedHttpx2AsyncTransport

pytestmark = pytest.mark.unit

PUBLIC_IP: str = "93.184.216.34"
METADATA_IP: str = "169.254.169.254"


def _resolve_to(
    *ips: str,
):
    def _stub(host, port, *args, **kw):
        return [(2, 1, 6, "", (ip, port)) for ip in ips]

    return _stub


@pytest.fixture(autouse=True)
def _settings():
    s = MagicMock()
    s.github_extra_hosts = ""
    s.ssrf_allowed_hosts = ""
    s.ssrf_allowed_cidrs = ""
    s.gateway_proxy_allow_private_targets = False
    url_guard._proxy_allowlist.cache_clear()
    with patch.object(url_guard, "settings", s):
        yield
    url_guard._proxy_allowlist.cache_clear()


@asynccontextmanager
async def _fake_connection(*args, **kwargs):
    yield MagicMock()


async def test_guarded_client_refuses_a_hostname_that_rebinds_to_metadata():
    async with mcp_client._guarded_http_client(PROXY_PROFILE) as client:
        with patch.object(url_guard.socket, "getaddrinfo", _resolve_to(METADATA_IP)):
            with pytest.raises(UrlValidationError):
                await client.get("https://mcp.example.com/mcp")


async def test_guarded_client_carries_the_guarded_transport():
    async with mcp_client._guarded_http_client(PROXY_PROFILE) as client:
        assert isinstance(client._transport, GuardedHttpx2AsyncTransport)


async def test_streamable_http_probe_runs_on_a_guarded_client():
    seen = {}

    def _streamable(url, http_client=None, **kwargs):
        seen["http_client"] = http_client
        return _fake_connection()

    with (
        patch.object(mcp_client, "_assert_mcp_url_fetchable", return_value=True),
        patch.object(mcp_client, "streamable_http_client", _streamable),
    ):
        result = await mcp_client.detect_server_transport("https://mcp.example.com")

    assert result == "streamable-http"
    assert isinstance(seen["http_client"]._transport, GuardedHttpx2AsyncTransport)


async def test_sse_probe_runs_on_a_guarded_client():
    seen = {}

    def _sse(url, httpx_client_factory=None, **kwargs):
        seen["factory"] = httpx_client_factory
        return _fake_connection()

    def _streamable_fails(*args, **kwargs):
        raise ConnectionError("no streamable endpoint")

    with (
        patch.object(mcp_client, "_assert_mcp_url_fetchable", return_value=True),
        patch.object(mcp_client, "streamable_http_client", _streamable_fails),
        patch.object(mcp_client, "sse_client", _sse),
    ):
        result = await mcp_client.detect_server_transport("https://mcp.example.com")

    assert result == "sse"
    async with seen["factory"]() as client:
        assert isinstance(client._transport, GuardedHttpx2AsyncTransport)


async def test_probes_do_not_connect_when_the_target_is_blocked():
    streamable = MagicMock()

    with (
        patch.object(mcp_client, "_assert_mcp_url_fetchable", return_value=False),
        patch.object(mcp_client, "streamable_http_client", streamable),
    ):
        await mcp_client.detect_server_transport("https://mcp.example.com")

    streamable.assert_not_called()


async def test_sse_discovery_uses_a_guarded_client_factory():
    seen = {}

    def _sse(url, headers=None, httpx_client_factory=None, **kwargs):
        seen["factory"] = httpx_client_factory
        return MagicMock()

    with (
        patch.object(mcp_client, "sse_client", _sse),
        patch.object(mcp_client, "_discover_tools", AsyncMock(return_value=([], None))),
    ):
        await mcp_client._discover_sse("https://mcp.example.com/sse", {})

    async with seen["factory"]() as client:
        assert isinstance(client._transport, GuardedHttpx2AsyncTransport)


def test_profile_for_an_ordinary_server_is_the_proxy_profile():
    assert mcp_client._mcp_guard_profile("https://mcp.example.com/mcp") is PROXY_PROFILE


async def test_streamable_http_discovery_uses_a_guarded_client():
    seen = {}

    def _streamable(url, http_client=None, **kwargs):
        seen["http_client"] = http_client
        return MagicMock()

    with (
        patch.object(mcp_client, "streamable_http_client", _streamable),
        patch.object(mcp_client, "_discover_tools", AsyncMock(return_value=([], None))),
    ):
        await mcp_client._discover_streamable_http("https://mcp.example.com/mcp", {})

    assert isinstance(seen["http_client"]._transport, GuardedHttpx2AsyncTransport)
