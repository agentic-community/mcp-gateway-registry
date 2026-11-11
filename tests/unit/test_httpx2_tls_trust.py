"""httpx2 clients get the registry's TLS trust.

MCP SDK 2.x talks through httpx2, whose default SSL context reads only the
operating system trust store. The registry image ships none, so without these
settings every HTTPS MCP server fails certificate verification during tool
discovery and security scans.
"""

import ssl
from unittest.mock import MagicMock, patch

import certifi
import pytest

from registry.core import mcp_client
from registry.utils import url_guard


@pytest.fixture(autouse=True)
def _fresh_contexts():
    url_guard.egress_ssl_context.cache_clear()
    url_guard._forward_proxy_ssl_context.cache_clear()
    yield
    url_guard.egress_ssl_context.cache_clear()
    url_guard._forward_proxy_ssl_context.cache_clear()


def test_egress_context_loads_certifi_roots_and_is_cached():
    context = url_guard.egress_ssl_context()

    assert isinstance(context, ssl.SSLContext)
    assert context.verify_mode == ssl.CERT_REQUIRED
    assert context.check_hostname is True
    assert context.cert_store_stats()["x509_ca"] > 100
    assert url_guard.egress_ssl_context() is context


def test_egress_context_reuses_the_forward_proxy_bundle_context():
    proxy_context = ssl.create_default_context()
    with patch.object(url_guard, "_forward_proxy_ssl_context", return_value=proxy_context):
        assert url_guard.egress_ssl_context() is proxy_context


def test_discovery_client_and_sse_factory_use_the_egress_context():
    seen = []

    class _Client:
        def __init__(self, **kwargs):
            seen.append(kwargs)

    with patch.object(mcp_client.httpx2, "AsyncClient", _Client):
        mcp_client._verified_http_client(headers={"a": "b"})

    assert seen[0]["verify"] is url_guard.egress_ssl_context()
    assert seen[0]["timeout"].read == mcp_client.SDK_DEFAULT_SSE_READ_TIMEOUT_SECONDS


@pytest.mark.asyncio
async def test_streamable_http_discovery_passes_the_egress_context():
    seen = {}

    class _Client:
        def __init__(self, **kwargs):
            seen.update(kwargs)

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

    async def _no_discovery(transport, mode):
        return [], None

    with (
        patch.object(mcp_client.httpx2, "AsyncClient", _Client),
        patch.object(mcp_client, "streamable_http_client", MagicMock()),
        patch.object(mcp_client, "_discover_tools", _no_discovery),
    ):
        await mcp_client._discover_streamable_http("https://h.example/mcp", {})

    assert seen["verify"] is url_guard.egress_ssl_context()


def _scanner_env(monkeypatch, preset: str | None) -> dict:
    from registry.services.security_scanner import SecurityScannerService

    if preset is None:
        monkeypatch.delenv("SSL_CERT_FILE", raising=False)
    else:
        monkeypatch.setenv("SSL_CERT_FILE", preset)
    completed = MagicMock(stdout="[]", stderr="", returncode=0)
    with patch("registry.services.security_scanner.subprocess.run", return_value=completed) as run:
        SecurityScannerService()._run_mcp_scanner("https://h.example/mcp", "yara", timeout=5)
    return run.call_args.kwargs["env"]


def test_scanner_subprocess_gets_certifi_when_unset(monkeypatch):
    assert _scanner_env(monkeypatch, None)["SSL_CERT_FILE"] == certifi.where()


def test_scanner_subprocess_keeps_an_operator_bundle(monkeypatch):
    assert _scanner_env(monkeypatch, "/etc/custom/ca.pem")["SSL_CERT_FILE"] == "/etc/custom/ca.pem"
