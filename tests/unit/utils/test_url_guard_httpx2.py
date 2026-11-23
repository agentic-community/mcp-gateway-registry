"""The httpx2 guarded transport and the profile the MCP hop uses.

The MCP SDK runs on httpx2, so its connections need their own pinned transport.
These tests pin the properties that matter: a blocked target is refused before any
connection, the connect IP is the one validated for that request, and a hostname
that rebinds between two requests is caught on the second.
"""

from __future__ import annotations

import ssl
from unittest.mock import AsyncMock, MagicMock, patch

import httpx2
import pytest

from registry.exceptions import UrlValidationError
from registry.utils import url_guard, url_guard_httpx2
from registry.utils.url_guard import EGRESS_UPSTREAM_PROFILE, PROXY_PROFILE
from registry.utils.url_guard_httpx2 import GuardedHttpx2AsyncTransport

PUBLIC_IP: str = "93.184.216.34"
PRIVATE_IP: str = "10.0.0.5"
METADATA_IP: str = "169.254.169.254"
ISTIO_RESERVED_IP: str = "240.240.0.7"


def _settings(
    ssrf_allowed_cidrs: str = "",
    ssrf_allowed_hosts: str = "",
    allow_private: bool = False,
) -> MagicMock:
    s = MagicMock()
    s.github_extra_hosts = ""
    s.ssrf_allowed_hosts = ssrf_allowed_hosts
    s.ssrf_allowed_cidrs = ssrf_allowed_cidrs
    s.gateway_proxy_allow_private_targets = allow_private
    return s


def _resolve_to(
    *ips: str,
):
    def _stub(host, port, *args, **kw):
        return [(2, 1, 6, "", (ip, port)) for ip in ips]

    return _stub


@pytest.fixture(autouse=True)
def _fresh_state():
    def _reset() -> None:
        url_guard._proxy_allowlist.cache_clear()

    _reset()
    with patch.object(url_guard, "settings", _settings()):
        yield
    _reset()


@pytest.fixture
def sent():
    """Capture what reaches the real connection pool instead of dialing it."""
    send = AsyncMock(return_value=httpx2.Response(200))
    with patch.object(httpx2.AsyncHTTPTransport, "handle_async_request", send):
        yield send


def _request(
    url: str,
) -> httpx2.Request:
    return httpx2.Request("POST", url)


@pytest.mark.asyncio
async def test_refuses_a_metadata_address_before_connecting(sent):
    transport = GuardedHttpx2AsyncTransport(guard_profile=PROXY_PROFILE)

    with pytest.raises(UrlValidationError):
        await transport.handle_async_request(_request(f"http://{METADATA_IP}/latest/meta-data"))

    sent.assert_not_called()


@pytest.mark.asyncio
async def test_refuses_a_hostname_that_resolves_private(sent):
    transport = GuardedHttpx2AsyncTransport(guard_profile=PROXY_PROFILE)

    with patch.object(url_guard.socket, "getaddrinfo", _resolve_to(PRIVATE_IP)):
        with pytest.raises(UrlValidationError):
            await transport.handle_async_request(_request("https://mcp.example.com/mcp"))

    sent.assert_not_called()


@pytest.mark.asyncio
async def test_connects_to_the_validated_ip_and_keeps_host_and_sni(sent):
    transport = GuardedHttpx2AsyncTransport(guard_profile=PROXY_PROFILE)

    with patch.object(url_guard.socket, "getaddrinfo", _resolve_to(PUBLIC_IP)):
        await transport.handle_async_request(_request("https://mcp.example.com/mcp"))

    pinned = sent.call_args.args[0]
    assert pinned.url.host == PUBLIC_IP
    assert pinned.headers["Host"] == "mcp.example.com"
    assert pinned.extensions["sni_hostname"] == "mcp.example.com"


@pytest.mark.asyncio
async def test_a_rebind_after_the_first_request_is_refused_on_the_next(sent):
    """The check-then-connect gap: every request is resolved and classified afresh."""
    transport = GuardedHttpx2AsyncTransport(guard_profile=PROXY_PROFILE)

    with patch.object(url_guard.socket, "getaddrinfo", _resolve_to(PUBLIC_IP)):
        await transport.handle_async_request(_request("https://mcp.example.com/mcp"))
    with patch.object(url_guard.socket, "getaddrinfo", _resolve_to(METADATA_IP)):
        with pytest.raises(UrlValidationError):
            await transport.handle_async_request(_request("https://mcp.example.com/mcp"))

    assert sent.call_count == 1


@pytest.mark.asyncio
async def test_a_mixed_answer_set_is_refused(sent):
    """One private answer among public ones is enough to refuse the host."""
    transport = GuardedHttpx2AsyncTransport(guard_profile=PROXY_PROFILE)

    with patch.object(url_guard.socket, "getaddrinfo", _resolve_to(PUBLIC_IP, PRIVATE_IP)):
        with pytest.raises(UrlValidationError):
            await transport.handle_async_request(_request("https://mcp.example.com/mcp"))

    sent.assert_not_called()


@pytest.mark.asyncio
async def test_aclose_closes_the_pool():
    transport = GuardedHttpx2AsyncTransport(guard_profile=PROXY_PROFILE)

    await transport.aclose()


class TestProxyProfileAllowlist:
    """The MCP hop's profile: a private backend needs the operator allowlist.

    This is the profile the registry's health check applies to the same
    registration, so a backend that is healthy is reachable and nothing else is.
    """

    @pytest.mark.asyncio
    async def test_private_backend_is_refused_without_an_allowlist(self, sent):
        transport = GuardedHttpx2AsyncTransport(guard_profile=PROXY_PROFILE)

        with patch.object(url_guard.socket, "getaddrinfo", _resolve_to(PRIVATE_IP)):
            with pytest.raises(UrlValidationError):
                await transport.handle_async_request(_request("http://currenttime:8000/mcp"))

        sent.assert_not_called()

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "settings",
        [
            pytest.param(_settings(ssrf_allowed_hosts="currenttime"), id="SSRF_ALLOWED_HOSTS"),
            pytest.param(_settings(ssrf_allowed_cidrs="10.0.0.0/8"), id="SSRF_ALLOWED_CIDRS"),
            pytest.param(_settings(allow_private=True), id="GATEWAY_PROXY_ALLOW_PRIVATE_TARGETS"),
        ],
    )
    async def test_an_allowlisted_private_backend_is_reachable_and_pinned(self, sent, settings):
        transport = GuardedHttpx2AsyncTransport(guard_profile=PROXY_PROFILE)

        with (
            patch.object(url_guard, "settings", settings),
            patch.object(url_guard.socket, "getaddrinfo", _resolve_to(PRIVATE_IP)),
        ):
            await transport.handle_async_request(_request("http://currenttime:8000/mcp"))

        assert sent.call_args.args[0].url.host == PRIVATE_IP

    @pytest.mark.asyncio
    async def test_an_allowlisted_host_does_not_admit_another_private_host(self, sent):
        transport = GuardedHttpx2AsyncTransport(guard_profile=PROXY_PROFILE)

        with (
            patch.object(url_guard, "settings", _settings(ssrf_allowed_hosts="currenttime")),
            patch.object(url_guard.socket, "getaddrinfo", _resolve_to(PRIVATE_IP)),
        ):
            with pytest.raises(UrlValidationError):
                await transport.handle_async_request(_request("http://openbao:8200/v1/sys"))

        sent.assert_not_called()

    @pytest.mark.asyncio
    @pytest.mark.parametrize("ip", ["169.254.169.254", "169.254.170.2", "100.100.100.200"])
    async def test_metadata_endpoints_stay_denied_for_an_allowlisted_host(self, sent, ip):
        transport = GuardedHttpx2AsyncTransport(guard_profile=PROXY_PROFILE)

        with (
            patch.object(
                url_guard,
                "settings",
                _settings(ssrf_allowed_hosts="currenttime", allow_private=True),
            ),
            patch.object(url_guard.socket, "getaddrinfo", _resolve_to(ip)),
        ):
            with pytest.raises(UrlValidationError):
                await transport.handle_async_request(_request("http://currenttime:8000/mcp"))

        sent.assert_not_called()

    @pytest.mark.asyncio
    async def test_a_public_name_that_rebinds_private_is_refused(self, sent):
        """The attack the hop exists to stop: healthy at check time, private at connect."""
        transport = GuardedHttpx2AsyncTransport(guard_profile=PROXY_PROFILE)

        with patch.object(url_guard.socket, "getaddrinfo", _resolve_to(PUBLIC_IP)):
            await transport.handle_async_request(_request("https://mcp.example.com/mcp"))
        with patch.object(url_guard.socket, "getaddrinfo", _resolve_to(PRIVATE_IP)):
            with pytest.raises(UrlValidationError):
                await transport.handle_async_request(_request("https://mcp.example.com/mcp"))

        assert sent.call_count == 1

    @pytest.mark.asyncio
    async def test_reserved_range_needs_an_operator_cidr(self, sent):
        transport = GuardedHttpx2AsyncTransport(guard_profile=PROXY_PROFILE)

        with (
            patch.object(url_guard, "settings", _settings(allow_private=True)),
            patch.object(url_guard.socket, "getaddrinfo", _resolve_to(ISTIO_RESERVED_IP)),
        ):
            with pytest.raises(UrlValidationError):
                await transport.handle_async_request(_request("http://backend.mesh:8000/mcp"))

        sent.assert_not_called()

    @pytest.mark.asyncio
    async def test_operator_cidr_admits_the_istio_reserved_range(self, sent):
        transport = GuardedHttpx2AsyncTransport(guard_profile=PROXY_PROFILE)

        with (
            patch.object(url_guard, "settings", _settings("240.240.0.0/16")),
            patch.object(url_guard.socket, "getaddrinfo", _resolve_to(ISTIO_RESERVED_IP)),
        ):
            await transport.handle_async_request(_request("http://backend.mesh:8000/mcp"))

        assert sent.call_args.args[0].url.host == ISTIO_RESERVED_IP

    def test_profile_is_not_credential_bearing(self):
        assert PROXY_PROFILE.credential_bearing is False


# ---------------------------------------------------------------------------
# Forward proxy. The SDK client passes its own transport, so httpx2 never reads
# HTTP(S)_PROXY itself; this transport's delegate is the only proxy path.
# ---------------------------------------------------------------------------

PROXY_URL: str = "http://proxyuser:proxypass@corp-proxy.internal:3128"
HTTPS_PROXY_URL: str = "https://corp-proxy.internal:8443"
_PROXY_ENV_NAMES: tuple[str, ...] = (
    "EGRESS_FORWARD_PROXY_ENABLED",
    "EGRESS_FORWARD_PROXY_CA_BUNDLE",
    "HTTP_PROXY",
    "http_proxy",
    "HTTPS_PROXY",
    "https_proxy",
    "NO_PROXY",
    "no_proxy",
)


def _ca_bundle(
    tmp_path,
):
    """Write a one-certificate PEM file taken from the certifi bundle."""
    import certifi

    marker = "-----END CERTIFICATE-----"
    path = tmp_path / "corporate-ca.pem"
    path.write_text(certifi.contents().split(marker)[0] + marker + "\n")
    return path


@pytest.fixture
def proxy_env(monkeypatch):
    """Clear every proxy variable, then return a setter that applies new ones."""
    for name in _PROXY_ENV_NAMES:
        monkeypatch.delenv(name, raising=False)

    def _reset() -> None:
        url_guard._forward_proxy_config.cache_clear()
        url_guard._forward_proxy_ssl_context.cache_clear()

    def _apply(**values: str) -> None:
        for name, value in values.items():
            monkeypatch.setenv(name, value)
        _reset()

    _reset()
    yield _apply
    _reset()


@pytest.fixture
def delegate():
    """Stand in for the proxy delegate and record what it is handed."""
    stub = MagicMock()
    stub.handle_async_request = AsyncMock(return_value=httpx2.Response(200))
    return stub


def _proxied_transport(
    profile,
    delegate_stub: MagicMock,
) -> GuardedHttpx2AsyncTransport:
    transport = GuardedHttpx2AsyncTransport(guard_profile=profile)
    transport._proxy_delegate = MagicMock(return_value=delegate_stub)  # type: ignore[method-assign]
    return transport


class TestForwardProxy:
    @pytest.mark.asyncio
    async def test_a_public_target_goes_to_the_proxy_unpinned(self, sent, delegate, proxy_env):
        """A CONNECT tunnel needs the hostname, so the request reaches the proxy as sent."""
        proxy_env(EGRESS_FORWARD_PROXY_ENABLED="true", HTTPS_PROXY=PROXY_URL)
        transport = _proxied_transport(PROXY_PROFILE, delegate)

        with patch.object(url_guard.socket, "getaddrinfo", _resolve_to(PUBLIC_IP)):
            await transport.handle_async_request(_request("https://mcp.example.com/mcp"))

        transport._proxy_delegate.assert_called_once_with(PROXY_URL)
        assert delegate.handle_async_request.call_args.args[0].url.host == "mcp.example.com"
        sent.assert_not_called()

    @pytest.mark.asyncio
    async def test_an_http_target_uses_the_http_proxy(self, sent, delegate, proxy_env):
        proxy_env(EGRESS_FORWARD_PROXY_ENABLED="true", HTTP_PROXY=PROXY_URL)
        transport = _proxied_transport(PROXY_PROFILE, delegate)

        with patch.object(url_guard.socket, "getaddrinfo", _resolve_to(PUBLIC_IP)):
            await transport.handle_async_request(_request("http://mcp.example.com/mcp"))

        transport._proxy_delegate.assert_called_once_with(PROXY_URL)
        sent.assert_not_called()

    @pytest.mark.asyncio
    async def test_an_internal_target_stays_direct_and_pinned_without_no_proxy(
        self, sent, delegate, proxy_env
    ):
        proxy_env(EGRESS_FORWARD_PROXY_ENABLED="true", HTTP_PROXY=PROXY_URL)
        transport = _proxied_transport(PROXY_PROFILE, delegate)

        with (
            patch.object(url_guard, "settings", _settings(ssrf_allowed_hosts="backend.internal")),
            patch.object(url_guard.socket, "getaddrinfo", _resolve_to(PRIVATE_IP)),
        ):
            await transport.handle_async_request(_request("http://backend.internal:8000/mcp"))

        delegate.handle_async_request.assert_not_called()
        assert sent.call_args.args[0].url.host == PRIVATE_IP

    @pytest.mark.asyncio
    async def test_a_no_proxy_host_stays_direct_and_pinned(self, sent, delegate, proxy_env):
        proxy_env(
            EGRESS_FORWARD_PROXY_ENABLED="true",
            HTTPS_PROXY=PROXY_URL,
            NO_PROXY="mcp.example.com",
        )
        transport = _proxied_transport(PROXY_PROFILE, delegate)

        with patch.object(url_guard.socket, "getaddrinfo", _resolve_to(PUBLIC_IP)):
            await transport.handle_async_request(_request("https://mcp.example.com/mcp"))

        delegate.handle_async_request.assert_not_called()
        assert sent.call_args.args[0].url.host == PUBLIC_IP

    @pytest.mark.asyncio
    async def test_a_metadata_answer_is_refused_before_the_proxy(self, sent, delegate, proxy_env):
        """The proxy would resolve the name itself, so the guard must classify first."""
        proxy_env(EGRESS_FORWARD_PROXY_ENABLED="true", HTTPS_PROXY=PROXY_URL)
        transport = _proxied_transport(EGRESS_UPSTREAM_PROFILE, delegate)

        with patch.object(url_guard.socket, "getaddrinfo", _resolve_to(METADATA_IP)):
            with pytest.raises(UrlValidationError):
                await transport.handle_async_request(_request("https://rebind.example.com/mcp"))

        delegate.handle_async_request.assert_not_called()
        sent.assert_not_called()

    @pytest.mark.asyncio
    async def test_a_redirect_hop_is_revalidated_on_the_proxied_path(
        self, sent, delegate, proxy_env
    ):
        """httpx2 sends each redirect hop through the transport, so it is classified."""
        proxy_env(EGRESS_FORWARD_PROXY_ENABLED="true", HTTPS_PROXY=PROXY_URL)
        delegate.handle_async_request = AsyncMock(
            return_value=httpx2.Response(
                307, headers={"location": "https://rebind.example.com/mcp"}
            )
        )
        transport = _proxied_transport(PROXY_PROFILE, delegate)
        answers = {"mcp.example.com": PUBLIC_IP, "rebind.example.com": METADATA_IP}

        def _resolve(host, port, *args, **kw):
            return [(2, 1, 6, "", (answers[host], port))]

        with patch.object(url_guard.socket, "getaddrinfo", _resolve):
            async with httpx2.AsyncClient(transport=transport, follow_redirects=True) as client:
                with pytest.raises(UrlValidationError):
                    await client.post("https://mcp.example.com/mcp")

        assert delegate.handle_async_request.call_count == 1

    @pytest.mark.asyncio
    async def test_flag_off_ignores_proxy_variables(self, sent, delegate, proxy_env):
        """Documents the opt-in: HTTPS_PROXY alone does not route guarded traffic."""
        proxy_env(HTTPS_PROXY=PROXY_URL)
        transport = _proxied_transport(PROXY_PROFILE, delegate)

        with patch.object(url_guard.socket, "getaddrinfo", _resolve_to(PUBLIC_IP)):
            await transport.handle_async_request(_request("https://mcp.example.com/mcp"))

        delegate.handle_async_request.assert_not_called()
        assert sent.call_args.args[0].url.host == PUBLIC_IP

    @pytest.mark.asyncio
    async def test_proxy_profile_allows_http_through_the_proxy(self, sent, delegate, proxy_env):
        """Nothing secret rides the ordinary hop, so plain http may be forwarded."""
        proxy_env(EGRESS_FORWARD_PROXY_ENABLED="true", HTTP_PROXY=PROXY_URL)
        transport = _proxied_transport(PROXY_PROFILE, delegate)

        with patch.object(url_guard.socket, "getaddrinfo", _resolve_to(PUBLIC_IP)):
            await transport.handle_async_request(_request("http://mcp.example.com/mcp"))

        delegate.handle_async_request.assert_called_once()

    @pytest.mark.asyncio
    async def test_a_credential_bearing_profile_refuses_http_through_the_proxy(
        self, sent, delegate, proxy_env
    ):
        proxy_env(EGRESS_FORWARD_PROXY_ENABLED="true", HTTP_PROXY=PROXY_URL)
        transport = _proxied_transport(EGRESS_UPSTREAM_PROFILE, delegate)

        with patch.object(url_guard.socket, "getaddrinfo", _resolve_to(PUBLIC_IP)):
            with pytest.raises(UrlValidationError):
                await transport.handle_async_request(_request("http://mcp.example.com/mcp"))

        delegate.handle_async_request.assert_not_called()


class TestProxyDelegate:
    def test_the_delegate_is_built_once_per_proxy_url(self, proxy_env):
        transport = GuardedHttpx2AsyncTransport(guard_profile=PROXY_PROFILE)

        first = transport._proxy_delegate(PROXY_URL)

        assert transport._proxy_delegate(PROXY_URL) is first
        assert isinstance(first, httpx2.AsyncHTTPTransport)

    def test_proxy_basic_auth_comes_from_the_url_userinfo(self, proxy_env):
        proxy = url_guard_httpx2._build_httpx2_proxy(PROXY_URL)

        assert proxy.auth == ("proxyuser", "proxypass")
        assert proxy.url.username == ""

    def test_an_https_proxy_gets_the_ca_bundle_context(self, tmp_path, proxy_env):
        proxy_env(EGRESS_FORWARD_PROXY_CA_BUNDLE=str(_ca_bundle(tmp_path)))

        proxy = url_guard_httpx2._build_httpx2_proxy(HTTPS_PROXY_URL)

        assert isinstance(proxy.ssl_context, ssl.SSLContext)

    def test_an_http_proxy_gets_no_proxy_ssl_context(self, tmp_path, proxy_env):
        """httpcore refuses proxy_ssl_context for an http proxy (issue #1849)."""
        proxy_env(EGRESS_FORWARD_PROXY_CA_BUNDLE=str(_ca_bundle(tmp_path)))

        proxy = url_guard_httpx2._build_httpx2_proxy(PROXY_URL)

        assert proxy.ssl_context is None

    @pytest.mark.parametrize("proxy_url", [PROXY_URL, HTTPS_PROXY_URL])
    def test_the_delegate_constructs_with_a_ca_bundle(self, tmp_path, proxy_env, proxy_url):
        """httpcore raises while BUILDING a bad delegate, so build one for real."""
        proxy_env(EGRESS_FORWARD_PROXY_CA_BUNDLE=str(_ca_bundle(tmp_path)))
        transport = GuardedHttpx2AsyncTransport(guard_profile=PROXY_PROFILE)

        assert transport._proxy_delegate(proxy_url) is not None

    def test_the_delegate_keeps_the_callers_tls_context(self, proxy_env):
        """The SDK passes the registry's egress context; the delegate must not swap it."""
        context = ssl.create_default_context()
        transport = GuardedHttpx2AsyncTransport(guard_profile=PROXY_PROFILE, verify=context)

        kwargs = url_guard._delegate_kwargs_with_ca_bundle(transport._delegate_kwargs)

        assert kwargs["verify"] is context

    @pytest.mark.asyncio
    async def test_aclose_closes_every_delegate(self, proxy_env):
        transport = GuardedHttpx2AsyncTransport(guard_profile=PROXY_PROFILE)
        stub = MagicMock()
        stub.aclose = AsyncMock()
        transport._proxy_delegates[PROXY_URL] = stub

        await transport.aclose()

        stub.aclose.assert_awaited_once()
        assert transport._proxy_delegates == {}
