"""Security tests for outbound A2A calls to registry-discovered agents.

Covers two related credential-leak / SSRF concerns:

1. The URL guard blocks non-http(s) schemes and hosts that resolve to
   private / loopback / cloud-metadata addresses (fail closed).
2. The remote-agent client never forwards the caller's registry-scoped token
   to an untrusted remote agent -- only an explicitly supplied per-target
   delegation token is ever attached, and the agent-card fetch is validated,
   redirect-disabled, and short-timeout.

Run from the ``agents/a2a`` directory:

    uv run --with pytest --with pytest-asyncio pytest test/test_remote_agent_security.py -q
"""

import sys
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

_SRC = Path(__file__).resolve().parent.parent / "src" / "travel-assistant-agent"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import url_guard  # noqa: E402
from models import DiscoveredAgent  # noqa: E402
from remote_agent_client import (  # noqa: E402
    _REMOTE_AGENT_TIMEOUT_SECONDS,
    RemoteAgentCache,
    RemoteAgentClient,
)
from url_guard import UnsafeUrlError, assert_fetchable  # noqa: E402


class TestUrlGuard:
    """Tests for the outbound SSRF URL guard."""

    def test_rejects_non_http_scheme(self):
        with pytest.raises(UnsafeUrlError):
            assert_fetchable("file:///etc/passwd")

    def test_rejects_empty_url(self):
        with pytest.raises(UnsafeUrlError):
            assert_fetchable("")

    def test_rejects_missing_host(self):
        with pytest.raises(UnsafeUrlError):
            assert_fetchable("https://")

    def test_rejects_loopback(self):
        with pytest.raises(UnsafeUrlError):
            assert_fetchable("http://127.0.0.1:9000/")

    def test_rejects_localhost(self):
        with pytest.raises(UnsafeUrlError):
            assert_fetchable("http://localhost:9000/")

    def test_rejects_cloud_metadata(self):
        with pytest.raises(UnsafeUrlError):
            assert_fetchable("http://169.254.169.254/latest/meta-data/")

    def test_rejects_rfc1918(self):
        with pytest.raises(UnsafeUrlError):
            assert_fetchable("http://10.1.2.3/mcp")

    def test_rejects_ipv4_mapped_ipv6_private(self):
        with pytest.raises(UnsafeUrlError):
            assert_fetchable("http://[::ffff:10.0.0.1]/mcp")

    def test_allows_public_ip(self):
        # 93.184.216.34 (example.com range) is public; no DNS needed for an IP.
        assert assert_fetchable("https://93.184.216.34/mcp") == "https://93.184.216.34/mcp"

    def test_public_hostname_resolving_to_private_is_blocked(self):
        # Simulate a hostname that resolves to a private address (DNS rebinding
        # style). getaddrinfo returns a private IP -> guard must fail closed.
        fake_resolution = [(2, 1, 6, "", ("10.0.0.5", 443))]
        with patch.object(url_guard.socket, "getaddrinfo", return_value=fake_resolution):
            with pytest.raises(UnsafeUrlError):
                assert_fetchable("https://evil.example.com/mcp")


class TestRemoteAgentClientCredentials:
    """The registry token must never reach an untrusted remote agent."""

    def test_defaults_to_no_credential(self):
        client = RemoteAgentClient(
            agent_url="https://remote.example.com/",
            agent_name="Remote",
            agent_id="/remote",
        )
        assert client.delegation_token is None

    def test_delegation_token_is_stored_verbatim(self):
        client = RemoteAgentClient(
            agent_url="https://remote.example.com/",
            agent_name="Remote",
            agent_id="/remote",
            delegation_token="scoped-delegation-token",
        )
        assert client.delegation_token == "scoped-delegation-token"

    def test_cache_does_not_forward_registry_token(self):
        """cache_discovered_agents attaches no credential by default."""
        cache = RemoteAgentCache()
        agents = [
            DiscoveredAgent(name="Remote", path="/remote", url="https://remote.example.com/"),
        ]
        cache.cache_discovered_agents(agents)
        client = cache.get("/remote")
        assert client is not None
        assert client.delegation_token is None

    def test_cache_uses_per_target_delegation_provider(self):
        """A provider mints a target-bound token; it is stored, not the registry token."""
        cache = RemoteAgentCache()
        agents = [
            DiscoveredAgent(name="Remote", path="/remote", url="https://remote.example.com/"),
        ]

        def _provider(agent: DiscoveredAgent) -> str:
            return f"delegation-for-{agent.path}"

        cache.cache_discovered_agents(agents, delegation_token_provider=_provider)
        client = cache.get("/remote")
        assert client.delegation_token == "delegation-for-/remote"

    def test_timeout_is_bounded(self):
        assert _REMOTE_AGENT_TIMEOUT_SECONDS <= 60


class TestRemoteAgentClientInitialization:
    """The agent-card fetch validates the URL, disables redirects, and sends
    only an explicit delegation token."""

    @pytest.mark.asyncio
    async def test_ensure_initialized_blocks_unsafe_url(self):
        client = RemoteAgentClient(
            agent_url="http://169.254.169.254/",
            agent_name="Evil",
            agent_id="/evil",
        )
        with pytest.raises(UnsafeUrlError):
            await client._ensure_initialized()
        # No httpx client should have been created for a blocked destination.
        assert client.httpx_client is None

    @pytest.mark.asyncio
    async def test_ensure_initialized_no_authorization_without_delegation(self):
        captured = {}

        class _FakeResolver:
            def __init__(self, httpx_client, base_url):
                captured["headers"] = dict(httpx_client.headers)
                captured["follow_redirects"] = httpx_client.follow_redirects

            async def get_agent_card(self):
                return object()

        client = RemoteAgentClient(
            agent_url="https://93.184.216.34/",
            agent_name="Remote",
            agent_id="/remote",
        )
        with (
            patch("remote_agent_client.A2ACardResolver", _FakeResolver),
            patch("remote_agent_client.ClientFactory") as mock_factory,
        ):
            mock_factory.return_value.create.return_value = AsyncMock()
            await client._ensure_initialized()

        header_names = {k.lower() for k in captured["headers"]}
        assert "authorization" not in header_names
        assert captured["follow_redirects"] is False

    @pytest.mark.asyncio
    async def test_ensure_initialized_sends_only_delegation_token(self):
        captured = {}

        class _FakeResolver:
            def __init__(self, httpx_client, base_url):
                captured["headers"] = dict(httpx_client.headers)

            async def get_agent_card(self):
                return object()

        client = RemoteAgentClient(
            agent_url="https://93.184.216.34/",
            agent_name="Remote",
            agent_id="/remote",
            delegation_token="scoped-delegation-token",
        )
        with (
            patch("remote_agent_client.A2ACardResolver", _FakeResolver),
            patch("remote_agent_client.ClientFactory") as mock_factory,
        ):
            mock_factory.return_value.create.return_value = AsyncMock()
            await client._ensure_initialized()

        assert captured["headers"].get("authorization") == "Bearer scoped-delegation-token"
