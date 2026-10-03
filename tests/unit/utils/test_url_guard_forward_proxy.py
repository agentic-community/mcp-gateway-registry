"""Unit tests for forward-proxy-aware egress in the SSRF guard (issue #1832).

The guard's transports pass a custom transport to httpx, which disables httpx's
own environment-proxy support, so a registry with no direct internet egress can
never reach an external MCP server. These tests pin the opt-in proxy path:

- the flag defaults off and an off flag changes nothing,
- a proxied request is still resolved and classified in full before any CONNECT,
- routing is derived from the classification, so an internal target keeps its pin
  with no NO_PROXY entry,
- NO_PROXY, per-scheme proxy selection, and malformed-proxy fail-closed,
- credential-bearing profiles refuse cleartext through a proxy,
- the Settings field and the guard's own env reader cannot disagree,
- the CA-bundle context adds to the default roots rather than replacing them.
"""

import ssl
from unittest.mock import AsyncMock, MagicMock, patch

import certifi
import httpx
import pytest
from pydantic import BaseModel

from registry.exceptions import UrlValidationError
from registry.utils import url_guard

# A public address and an internal one, used as resolver answers throughout.
PUBLIC_IP = "93.184.216.34"
PRIVATE_IP = "10.0.0.5"
METADATA_IP = "169.254.169.254"

PROXY_URL = "http://corp-proxy.internal:3128"


def _reset_caches() -> None:
    url_guard._skill_allowlist.cache_clear()
    url_guard._proxy_allowlist.cache_clear()
    url_guard._builtin_airegistry_tools_allowlist.cache_clear()
    url_guard._credentialed_oauth_allowlist.cache_clear()
    url_guard._forward_proxy_config.cache_clear()
    url_guard._forward_proxy_ssl_context.cache_clear()


@pytest.fixture(autouse=True)
def _clear_caches():
    """Both the allowlist factories and the proxy config are lru_cached."""
    _reset_caches()
    yield
    _reset_caches()


def _resolve_to(*ips: str):
    """getaddrinfo stub resolving any host to the given IP(s)."""

    def _stub(host, port, **kw):
        return [(2, 1, 6, "", (ip, port)) for ip in ips]

    return _stub


def _stub_async_dns(*ips: str):
    """Patch the event loop's resolver, leaving the real classifier in place.

    Preferred over patching ``_resolve_public_ips_async`` whenever the test is
    about whether a blocked answer is DENIED: stubbing that function out would
    remove the very check under test.
    """
    import asyncio

    return patch.object(
        asyncio.get_running_loop(),
        "getaddrinfo",
        new=AsyncMock(return_value=[(2, 1, 6, "", (ip, 443)) for ip in ips]),
    )


def _settings(
    github_extra_hosts="",
    ssrf_allowed_hosts="",
    ssrf_allowed_cidrs="",
    gateway_proxy_allow_private_targets=False,
    egress_oauth_trusted_idp_hosts="",
):
    s = MagicMock()
    s.github_extra_hosts = github_extra_hosts
    s.ssrf_allowed_hosts = ssrf_allowed_hosts
    s.ssrf_allowed_cidrs = ssrf_allowed_cidrs
    s.gateway_proxy_allow_private_targets = gateway_proxy_allow_private_targets
    s.egress_oauth_trusted_idp_hosts = egress_oauth_trusted_idp_hosts
    return s


def _settings_with_pool():
    """Settings stub carrying the pool fields shared_plain_async_client reads."""
    s = _settings()
    s.egress_http_pool_connect_retries = 1
    s.egress_http_pool_max_connections = 100
    s.egress_http_pool_max_keepalive = 20
    s.egress_http_pool_keepalive_expiry_seconds = 30
    return s


def _proxy_env(
    *,
    enabled: str | None = "true",
    http_proxy: str | None = PROXY_URL,
    https_proxy: str | None = PROXY_URL,
    no_proxy: str | None = None,
    ca_bundle: str | None = None,
) -> dict[str, str]:
    """Build the environment the guard reads, omitting unset keys."""
    env: dict[str, str] = {}
    if enabled is not None:
        env["EGRESS_FORWARD_PROXY_ENABLED"] = enabled
    if http_proxy is not None:
        env["HTTP_PROXY"] = http_proxy
    if https_proxy is not None:
        env["HTTPS_PROXY"] = https_proxy
    if no_proxy is not None:
        env["NO_PROXY"] = no_proxy
    if ca_bundle is not None:
        env["EGRESS_FORWARD_PROXY_CA_BUNDLE"] = ca_bundle
    return env


# ---------------------------------------------------------------------------
# Settings / env-reader parity. Written first: a config page that disagrees with
# the guard is worse than the feature being off, and two readers is the design's
# accepted risk (url_guard must not force Settings to be built).
# ---------------------------------------------------------------------------


class _BoolProbe(BaseModel):
    """A bool field with the same Pydantic coercion rules as the real Settings field."""

    value: bool


class TestSettingsParity:
    def test_settings_default_matches_guard_default(self):
        from registry.core.config import Settings

        assert Settings.model_fields["egress_forward_proxy_enabled"].default is False
        assert Settings.model_fields["egress_forward_proxy_ca_bundle"].default == ""
        # The guard, with nothing in the environment, must agree.
        with patch.dict("os.environ", {}, clear=True):
            assert url_guard._forward_proxy_config().enabled is False

    @pytest.mark.parametrize(
        "spelling",
        # Every spelling Pydantic's bool coercion accepts, plus the casing and
        # whitespace variants an operator realistically types into a values file.
        ["1", "true", "True", "TRUE", "  true  ", "t", "T", "yes", "Yes", "y", "on", "On"],
    )
    def test_truthy_spellings_agree(self, spelling):
        assert _BoolProbe(value=spelling.strip()).value is True, "probe assumption"
        with patch.dict("os.environ", _proxy_env(enabled=spelling), clear=True):
            url_guard._forward_proxy_config.cache_clear()
            assert url_guard._forward_proxy_config().enabled is True

    @pytest.mark.parametrize(
        "spelling",
        ["0", "false", "False", "FALSE", "f", "no", "n", "off", "Off", "", "   "],
    )
    def test_falsy_spellings_agree(self, spelling):
        if spelling.strip():
            assert _BoolProbe(value=spelling.strip()).value is False, "probe assumption"
        with patch.dict("os.environ", _proxy_env(enabled=spelling), clear=True):
            url_guard._forward_proxy_config.cache_clear()
            assert url_guard._forward_proxy_config().enabled is False

    def test_garbage_value_is_not_enabled(self):
        """An unrecognized value must fail closed to disabled, not to enabled."""
        with patch.dict("os.environ", _proxy_env(enabled="maybe"), clear=True):
            url_guard._forward_proxy_config.cache_clear()
            assert url_guard._forward_proxy_config().enabled is False


# ---------------------------------------------------------------------------
# The guarantee that matters: the proxied path still validates.
# ---------------------------------------------------------------------------


class TestProxiedPathStillGuards:
    async def test_proxied_request_still_validates_and_classifies(self):
        """A proxied target resolving to the metadata IP is denied before CONNECT.

        This is the regression test for the mount bypass: setting proxy= on the
        enclosing client would shadow the guarded transport entirely and let this
        request through with no checks at all. The REAL resolver path runs here
        (only the DNS answer is stubbed), so the assertion covers classification
        rather than a mock.
        """
        transport = url_guard.GuardedAsyncTransport(guard_profile=url_guard.PROXY_PROFILE)
        request = httpx.Request("GET", "https://evil.example/x")
        with (
            patch.dict("os.environ", _proxy_env(), clear=True),
            patch.object(url_guard, "settings", _settings()),
            _stub_async_dns(METADATA_IP),
        ):
            with pytest.raises(UrlValidationError):
                await transport._route_request_async(request)

    async def test_proxied_request_denies_private_resolution(self):
        """One blocked resolver answer denies the whole request."""
        transport = url_guard.GuardedAsyncTransport(guard_profile=url_guard.SKILL_PROFILE)
        request = httpx.Request("GET", "https://acme.example/x")
        with (
            patch.dict("os.environ", _proxy_env(), clear=True),
            patch.object(url_guard, "settings", _settings()),
            _stub_async_dns(PUBLIC_IP, PRIVATE_IP),
        ):
            with pytest.raises(UrlValidationError):
                await transport._route_request_async(request)

    async def test_proxied_request_keeps_hostname(self):
        """The request handed to the delegate still names the host.

        A CONNECT tunnel builds its TLS handshake with server_hostname from the
        request URL and never reads the sni_hostname extension, so a pinned
        request through a tunnel would verify the certificate against an IP.
        """
        transport = url_guard.GuardedAsyncTransport(guard_profile=url_guard.PROXY_PROFILE)
        request = httpx.Request("GET", "https://acme.example/x")
        with (
            patch.dict("os.environ", _proxy_env(), clear=True),
            patch.object(url_guard, "settings", _settings()),
            patch.object(
                url_guard, "_resolve_public_ips_async", new=AsyncMock(return_value=[PUBLIC_IP])
            ),
        ):
            routed = await transport._route_request_async(request)

        assert routed.proxy_url == PROXY_URL
        assert routed.request.url.host == "acme.example"
        assert "sni_hostname" not in routed.request.extensions

    async def test_redirect_hop_is_revalidated_on_proxied_path(self):
        """Each hop re-enters the transport, so hop 2 to metadata is denied.

        httpx re-invokes the transport per redirect, which is simulated here by
        routing two requests through the same transport instance.
        """
        transport = url_guard.GuardedAsyncTransport(guard_profile=url_guard.PROXY_PROFILE)
        with (
            patch.dict("os.environ", _proxy_env(), clear=True),
            patch.object(url_guard, "settings", _settings()),
        ):
            with patch.object(
                url_guard, "_resolve_public_ips_async", new=AsyncMock(return_value=[PUBLIC_IP])
            ):
                first = await transport._route_request_async(
                    httpx.Request("GET", "https://acme.example/x")
                )
            assert first.proxy_url == PROXY_URL

            # Hop 2: a Location pointing at the metadata literal.
            with pytest.raises(UrlValidationError):
                await transport._route_request_async(
                    httpx.Request("GET", f"http://{METADATA_IP}/latest/meta-data/")
                )


# ---------------------------------------------------------------------------
# Routing is derived from the classification, not from an operator list.
# ---------------------------------------------------------------------------


class TestRoutingScope:
    async def test_flag_off_is_byte_for_byte_unchanged(self):
        """With HTTP_PROXY set but the flag unset, the request is still pinned."""
        transport = url_guard.GuardedAsyncTransport(guard_profile=url_guard.SKILL_PROFILE)
        request = httpx.Request("GET", "https://acme.example/x")
        with (
            patch.dict("os.environ", _proxy_env(enabled=None), clear=True),
            patch.object(url_guard, "settings", _settings()),
            patch.object(
                url_guard, "_resolve_public_ips_async", new=AsyncMock(return_value=[PUBLIC_IP])
            ),
        ):
            routed = await transport._route_request_async(request)

        assert routed.proxy_url is None
        assert routed.request.url.host == PUBLIC_IP
        assert routed.request.headers["Host"] == "acme.example"
        assert routed.request.extensions["sni_hostname"] == "acme.example"
        assert transport._proxy_delegates == {}

    async def test_public_resolving_host_is_proxied(self):
        transport = url_guard.GuardedAsyncTransport(guard_profile=url_guard.PROXY_PROFILE)
        with (
            patch.dict("os.environ", _proxy_env(), clear=True),
            patch.object(url_guard, "settings", _settings()),
            patch.object(
                url_guard, "_resolve_public_ips_async", new=AsyncMock(return_value=[PUBLIC_IP])
            ),
        ):
            routed = await transport._route_request_async(
                httpx.Request("GET", "https://acme.example/x")
            )
        assert routed.proxy_url == PROXY_URL

    async def test_internal_resolving_host_stays_direct_without_no_proxy(self):
        """The scoping rule: an allowlisted internal host keeps its pin.

        No NO_PROXY entry is needed. This is what makes the feature
        self-maintaining: registering an internal MCP server tomorrow does not
        require editing NO_PROXY, and the rebinding relaxation never applies to it.
        """
        transport = url_guard.GuardedAsyncTransport(guard_profile=url_guard.PROXY_PROFILE)
        with (
            patch.dict("os.environ", _proxy_env(), clear=True),
            patch.object(
                url_guard, "settings", _settings(ssrf_allowed_hosts="internal.mcp.example")
            ),
            patch.object(
                url_guard, "_resolve_public_ips_async", new=AsyncMock(return_value=[PRIVATE_IP])
            ),
        ):
            routed = await transport._route_request_async(
                httpx.Request("GET", "https://internal.mcp.example/x")
            )

        assert routed.proxy_url is None
        assert routed.request.url.host == PRIVATE_IP
        assert routed.request.headers["Host"] == "internal.mcp.example"
        assert transport._proxy_delegates == {}

    async def test_mixed_public_and_internal_answers_stay_direct(self):
        """A mixed answer set counts as internal: preserve the control."""
        transport = url_guard.GuardedAsyncTransport(guard_profile=url_guard.PROXY_PROFILE)
        with (
            patch.dict("os.environ", _proxy_env(), clear=True),
            patch.object(url_guard, "settings", _settings(ssrf_allowed_hosts="split.example")),
            patch.object(
                url_guard,
                "_resolve_public_ips_async",
                new=AsyncMock(return_value=[PUBLIC_IP, PRIVATE_IP]),
            ),
        ):
            routed = await transport._route_request_async(
                httpx.Request("GET", "https://split.example/x")
            )
        assert routed.proxy_url is None
        assert routed.request.url.host == PUBLIC_IP  # pinned to the first validated answer

    async def test_internal_ip_literal_target_stays_direct(self):
        """An allowlisted private literal is dialed as-is, with no rewrite."""
        transport = url_guard.GuardedAsyncTransport(guard_profile=url_guard.EGRESS_UPSTREAM_PROFILE)
        with (
            patch.dict("os.environ", _proxy_env(), clear=True),
            patch.object(url_guard, "settings", _settings()),
        ):
            routed = await transport._route_request_async(
                httpx.Request("GET", f"https://{PRIVATE_IP}:8443/mcp")
            )
        assert routed.proxy_url is None
        assert routed.request.url.host == PRIVATE_IP

    async def test_addresses_resolved_once_per_request(self):
        """One lookup feeds both the access decision and the routing decision.

        Resolving twice would waste a lookup and open a window for the two
        answers to disagree, which is the TOCTOU gap the pin exists to close.
        """
        transport = url_guard.GuardedAsyncTransport(guard_profile=url_guard.PROXY_PROFILE)
        resolver = AsyncMock(return_value=[PUBLIC_IP])
        with (
            patch.dict("os.environ", _proxy_env(), clear=True),
            patch.object(url_guard, "settings", _settings()),
            patch.object(url_guard, "_resolve_public_ips_async", new=resolver),
        ):
            await transport._route_request_async(httpx.Request("GET", "https://acme.example/x"))
        resolver.assert_awaited_once()

    def test_sync_transport_proxies_a_public_target(self):
        """GuardedTransport routes identically to GuardedAsyncTransport."""
        transport = url_guard.GuardedTransport(guard_profile=url_guard.PROXY_PROFILE)
        with (
            patch.dict("os.environ", _proxy_env(), clear=True),
            patch.object(url_guard, "settings", _settings()),
            patch.object(url_guard.socket, "getaddrinfo", _resolve_to(PUBLIC_IP)),
        ):
            proxied = transport._route_request(httpx.Request("GET", "https://acme.example/x"))
        assert proxied.proxy_url == PROXY_URL
        assert proxied.request.url.host == "acme.example"

    def test_sync_transport_keeps_an_internal_target_direct(self):
        transport = url_guard.GuardedTransport(guard_profile=url_guard.PROXY_PROFILE)
        with (
            patch.dict("os.environ", _proxy_env(), clear=True),
            patch.object(url_guard, "settings", _settings(ssrf_allowed_hosts="internal.example")),
            patch.object(url_guard.socket, "getaddrinfo", _resolve_to(PRIVATE_IP)),
        ):
            direct = transport._route_request(httpx.Request("GET", "https://internal.example/x"))
        assert direct.proxy_url is None
        assert direct.request.url.host == PRIVATE_IP


# ---------------------------------------------------------------------------
# NO_PROXY and per-scheme selection.
# ---------------------------------------------------------------------------


class TestNoProxyAndSchemes:
    @pytest.mark.parametrize(
        "no_proxy,hostname",
        [
            # Exact-host forms. "mcpgw-server" is deliberately not used: it is a
            # reserved hostname rejected by every ordinary profile.
            ("auth-server", "auth-server"),
            ("AUTH-SERVER", "auth-server"),
            ("acme.example:443", "acme.example"),
            ("svc.cluster.local", "a.svc.cluster.local"),
            (".svc.cluster.local", "a.svc.cluster.local"),
            ("*", "acme.example"),
            ("other.example,acme.example", "acme.example"),
        ],
    )
    async def test_no_proxy_forms_stay_direct_and_pinned(self, no_proxy, hostname):
        transport = url_guard.GuardedAsyncTransport(guard_profile=url_guard.PROXY_PROFILE)
        with (
            patch.dict("os.environ", _proxy_env(no_proxy=no_proxy), clear=True),
            patch.object(url_guard, "settings", _settings(ssrf_allowed_hosts=hostname)),
            patch.object(
                url_guard, "_resolve_public_ips_async", new=AsyncMock(return_value=[PUBLIC_IP])
            ),
        ):
            routed = await transport._route_request_async(
                httpx.Request("GET", f"https://{hostname}/x")
            )
        # Public-resolving, so only NO_PROXY can have kept it direct.
        assert routed.proxy_url is None
        assert routed.request.url.host == PUBLIC_IP

    async def test_no_proxy_suffix_does_not_match_a_bare_substring(self):
        """'example.com' must not match 'notexample.com'."""
        transport = url_guard.GuardedAsyncTransport(guard_profile=url_guard.PROXY_PROFILE)
        with (
            patch.dict("os.environ", _proxy_env(no_proxy="example.com"), clear=True),
            patch.object(url_guard, "settings", _settings()),
            patch.object(
                url_guard, "_resolve_public_ips_async", new=AsyncMock(return_value=[PUBLIC_IP])
            ),
        ):
            routed = await transport._route_request_async(
                httpx.Request("GET", "https://notexample.com/x")
            )
        assert routed.proxy_url == PROXY_URL

    async def test_no_proxy_match_is_recorded_as_a_direct_route(self):
        """A NO_PROXY exclusion must show up as route="direct" in the metric.

        Found by running the feature against a live stack: the metric had no
        route="direct" series at all, even though a healthy NO_PROXY-matched
        server was being probed every cycle, because the early return skipped
        the recording. That made "the flag is on but everything is going direct"
        invisible, which is the one question the counter exists to answer.
        """
        transport = url_guard.GuardedAsyncTransport(guard_profile=url_guard.PROXY_PROFILE)
        with (
            patch.dict("os.environ", _proxy_env(no_proxy="acme.example"), clear=True),
            patch.object(url_guard, "settings", _settings()),
            patch.object(
                url_guard, "_resolve_public_ips_async", new=AsyncMock(return_value=[PUBLIC_IP])
            ),
            patch.object(url_guard, "_record_egress_route") as record,
        ):
            routed = await transport._route_request_async(
                httpx.Request("GET", "https://acme.example/x")
            )
        assert routed.proxy_url is None
        record.assert_called_once_with("proxy", "direct", "ok")

    async def test_flag_off_records_nothing(self):
        """The flag-off path must stay free of observability cost."""
        transport = url_guard.GuardedAsyncTransport(guard_profile=url_guard.PROXY_PROFILE)
        with (
            patch.dict("os.environ", _proxy_env(enabled=None), clear=True),
            patch.object(url_guard, "settings", _settings()),
            patch.object(
                url_guard, "_resolve_public_ips_async", new=AsyncMock(return_value=[PUBLIC_IP])
            ),
            patch.object(url_guard, "_record_egress_route") as record,
        ):
            await transport._route_request_async(httpx.Request("GET", "https://acme.example/x"))
        record.assert_not_called()

    async def test_no_proxy_for_this_scheme_is_recorded_as_direct(self):
        transport = url_guard.GuardedAsyncTransport(guard_profile=url_guard.PROXY_PROFILE)
        with (
            patch.dict("os.environ", _proxy_env(http_proxy=None), clear=True),
            patch.object(url_guard, "settings", _settings()),
            patch.object(
                url_guard, "_resolve_public_ips_async", new=AsyncMock(return_value=[PUBLIC_IP])
            ),
            patch.object(url_guard, "_record_egress_route") as record,
        ):
            await transport._route_request_async(httpx.Request("GET", "http://acme.example/x"))
        record.assert_called_once_with("proxy", "direct", "ok")

    def test_cidr_shaped_no_proxy_entry_warns(self, caplog):
        """A CIDR is not expanded by curl or httpx, so it silently fails to match."""
        with caplog.at_level("WARNING"):
            patterns = url_guard._parse_no_proxy("10.0.0.0/8,acme.example")
        assert "10.0.0.0/8" in patterns  # kept, but flagged
        assert any("CIDR" in record.message for record in caplog.records)

    async def test_https_target_uses_https_proxy_http_target_uses_http_proxy(self):
        transport = url_guard.GuardedAsyncTransport(guard_profile=url_guard.PROXY_PROFILE)
        env = _proxy_env(http_proxy="http://plain:3128", https_proxy="http://tunnel:3129")
        with (
            patch.dict("os.environ", env, clear=True),
            patch.object(url_guard, "settings", _settings()),
            patch.object(
                url_guard, "_resolve_public_ips_async", new=AsyncMock(return_value=[PUBLIC_IP])
            ),
        ):
            https_routed = await transport._route_request_async(
                httpx.Request("GET", "https://acme.example/x")
            )
            http_routed = await transport._route_request_async(
                httpx.Request("GET", "http://acme.example/x")
            )
        assert https_routed.proxy_url == "http://tunnel:3129"
        assert http_routed.proxy_url == "http://plain:3128"

    async def test_flag_on_with_only_https_proxy_sends_http_target_direct(self):
        """No warning: a deployment with only HTTPS_PROXY is a normal configuration."""
        transport = url_guard.GuardedAsyncTransport(guard_profile=url_guard.PROXY_PROFILE)
        with (
            patch.dict("os.environ", _proxy_env(http_proxy=None), clear=True),
            patch.object(url_guard, "settings", _settings()),
            patch.object(
                url_guard, "_resolve_public_ips_async", new=AsyncMock(return_value=[PUBLIC_IP])
            ),
        ):
            routed = await transport._route_request_async(
                httpx.Request("GET", "http://acme.example/x")
            )
        assert routed.proxy_url is None
        assert routed.request.url.host == PUBLIC_IP  # still pinned

    async def test_flag_on_without_proxy_env_warns_and_stays_direct(self, caplog):
        """Does not raise; logs one WARNING and behaves exactly as disabled."""
        transport = url_guard.GuardedAsyncTransport(guard_profile=url_guard.PROXY_PROFILE)
        env = _proxy_env(http_proxy=None, https_proxy=None)
        with (
            patch.dict("os.environ", env, clear=True),
            patch.object(url_guard, "settings", _settings()),
            patch.object(
                url_guard, "_resolve_public_ips_async", new=AsyncMock(return_value=[PUBLIC_IP])
            ),
            caplog.at_level("WARNING"),
        ):
            routed = await transport._route_request_async(
                httpx.Request("GET", "https://acme.example/x")
            )
        assert routed.proxy_url is None
        assert routed.request.url.host == PUBLIC_IP
        assert any("neither HTTP_PROXY nor HTTPS_PROXY" in r.message for r in caplog.records)

    def test_upper_case_env_wins_over_lower_case(self):
        """Matches curl. Deliberate; see the _env_first docstring."""
        with patch.dict(
            "os.environ",
            {
                "EGRESS_FORWARD_PROXY_ENABLED": "true",
                "HTTPS_PROXY": "http://upper:3128",
                "https_proxy": "http://lower:3128",
            },
            clear=True,
        ):
            assert url_guard._forward_proxy_config().https_proxy == "http://upper:3128"

    def test_lower_case_env_is_read_when_upper_is_absent(self):
        with patch.dict(
            "os.environ",
            {"EGRESS_FORWARD_PROXY_ENABLED": "true", "https_proxy": "http://lower:3128"},
            clear=True,
        ):
            config = url_guard._forward_proxy_config()
        assert config.enabled is True
        assert config.https_proxy == "http://lower:3128"


# ---------------------------------------------------------------------------
# Credential-bearing profiles, and fail-closed on a bad proxy URL.
# ---------------------------------------------------------------------------


class TestProfileRulesAndFailClosed:
    @pytest.mark.parametrize(
        "profile_name",
        ["FEDERATION_PROFILE", "EGRESS_UPSTREAM_PROFILE", "CREDENTIALED_OAUTH_PROFILE"],
    )
    def test_credential_bearing_profiles_are_marked(self, profile_name):
        assert getattr(url_guard, profile_name).credential_bearing is True

    @pytest.mark.parametrize(
        "profile_name",
        ["SKILL_PROFILE", "PROXY_PROFILE", "BUILTIN_AIREGISTRY_TOOLS_PROFILE"],
    )
    def test_non_credential_profiles_are_not_marked(self, profile_name):
        assert getattr(url_guard, profile_name).credential_bearing is False

    async def test_credential_bearing_profile_refuses_http_via_proxy(self):
        """An http target would hand the credential to the proxy in cleartext."""
        transport = url_guard.GuardedAsyncTransport(guard_profile=url_guard.EGRESS_UPSTREAM_PROFILE)
        with (
            patch.dict("os.environ", _proxy_env(), clear=True),
            patch.object(url_guard, "settings", _settings()),
            patch.object(
                url_guard, "_resolve_public_ips_async", new=AsyncMock(return_value=[PUBLIC_IP])
            ),
        ):
            with pytest.raises(UrlValidationError) as exc:
                await transport._route_request_async(
                    httpx.Request("GET", "http://upstream.example/mcp")
                )
        # The remedy must be in the message, or an operator reads this as the
        # feature being broken rather than as a configuration gap.
        assert "NO_PROXY" in str(exc.value)

    async def test_credential_bearing_profile_allows_https_via_proxy(self):
        transport = url_guard.GuardedAsyncTransport(guard_profile=url_guard.EGRESS_UPSTREAM_PROFILE)
        with (
            patch.dict("os.environ", _proxy_env(), clear=True),
            patch.object(url_guard, "settings", _settings()),
            patch.object(
                url_guard, "_resolve_public_ips_async", new=AsyncMock(return_value=[PUBLIC_IP])
            ),
        ):
            routed = await transport._route_request_async(
                httpx.Request("GET", "https://upstream.example/mcp")
            )
        assert routed.proxy_url == PROXY_URL

    async def test_credential_bearing_http_internal_target_is_unaffected(self):
        """The refusal only applies to a PROXIED target.

        An in-cluster http upstream resolves internal, so it goes direct and
        never reaches the cleartext check. This is why marking
        EGRESS_UPSTREAM_PROFILE credential-bearing does not break the common
        private-http MCP upstream.
        """
        transport = url_guard.GuardedAsyncTransport(guard_profile=url_guard.EGRESS_UPSTREAM_PROFILE)
        with (
            patch.dict("os.environ", _proxy_env(), clear=True),
            patch.object(url_guard, "settings", _settings()),
            patch.object(
                url_guard, "_resolve_public_ips_async", new=AsyncMock(return_value=[PRIVATE_IP])
            ),
        ):
            routed = await transport._route_request_async(
                httpx.Request("GET", "http://upstream.internal/mcp")
            )
        assert routed.proxy_url is None
        assert routed.request.url.host == PRIVATE_IP

    async def test_non_credential_profile_allows_http_via_proxy(self):
        transport = url_guard.GuardedAsyncTransport(guard_profile=url_guard.SKILL_PROFILE)
        with (
            patch.dict("os.environ", _proxy_env(), clear=True),
            patch.object(url_guard, "settings", _settings()),
            patch.object(
                url_guard, "_resolve_public_ips_async", new=AsyncMock(return_value=[PUBLIC_IP])
            ),
        ):
            routed = await transport._route_request_async(
                httpx.Request("GET", "http://raw.example/file.md")
            )
        assert routed.proxy_url == PROXY_URL

    @pytest.mark.parametrize(
        "bad_proxy",
        ["ftp://proxy:3128", "socks5://proxy:1080", "proxy-without-scheme:3128", "http://", "::::"],
    )
    def test_malformed_proxy_url_fails_closed(self, bad_proxy):
        """Raise rather than silently dialing direct, which would hide the typo
        behind the same ConnectTimeout the operator is trying to fix."""
        with pytest.raises(UrlValidationError):
            url_guard._validate_proxy_url(bad_proxy)

    def test_valid_proxy_urls_pass(self):
        for good in [
            "http://proxy:3128",
            "https://proxy.corp.example:8443",
            "http://user:pass@proxy:3128",
        ]:
            url_guard._validate_proxy_url(good)  # must not raise

    def test_malformed_proxy_url_error_does_not_leak_userinfo(self):
        with pytest.raises(UrlValidationError) as exc:
            url_guard._validate_proxy_url("ftp://user:sup3rsecret@proxy:3128")
        assert "sup3rsecret" not in str(exc.value)

    async def test_malformed_proxy_raises_at_request_time(self):
        transport = url_guard.GuardedAsyncTransport(guard_profile=url_guard.PROXY_PROFILE)
        with (
            patch.dict("os.environ", _proxy_env(https_proxy="ftp://proxy:3128"), clear=True),
            patch.object(url_guard, "settings", _settings()),
        ):
            with pytest.raises(UrlValidationError):
                await transport._route_request_async(httpx.Request("GET", "https://acme.example/x"))

    def test_proxy_url_is_never_logged_raw(self, caplog):
        env = _proxy_env(
            http_proxy="http://bob:sup3rsecret@proxy:3128",
            https_proxy="http://bob:sup3rsecret@proxy:3128",
        )
        with patch.dict("os.environ", env, clear=True), caplog.at_level("DEBUG"):
            url_guard._forward_proxy_config()
        text = caplog.text
        assert "sup3rsecret" not in text
        assert "bob" not in text
        assert "proxy:3128" in text  # the host is still useful and safe to log


# ---------------------------------------------------------------------------
# Delegate transports: construction, dispatch, caching, teardown.
# ---------------------------------------------------------------------------


class TestDelegateTransports:
    def test_httpx_transport_accepts_proxy_keyword(self):
        """Pins the httpx floor: 'proxy=' was renamed from 'proxies' in 0.26."""
        import inspect

        for cls in (httpx.HTTPTransport, httpx.AsyncHTTPTransport):
            assert "proxy" in inspect.signature(cls.__init__).parameters

    async def test_proxied_request_goes_to_the_delegate_not_the_direct_path(self):
        transport = url_guard.GuardedAsyncTransport(guard_profile=url_guard.PROXY_PROFILE)
        delegate = MagicMock()
        delegate.handle_async_request = AsyncMock(return_value=httpx.Response(401))
        with (
            patch.dict("os.environ", _proxy_env(), clear=True),
            patch.object(url_guard, "settings", _settings()),
            patch.object(
                url_guard, "_resolve_public_ips_async", new=AsyncMock(return_value=[PUBLIC_IP])
            ),
            patch.object(transport, "_proxy_delegate", return_value=delegate) as factory,
            patch.object(
                httpx.AsyncHTTPTransport, "handle_async_request", new=AsyncMock()
            ) as direct,
        ):
            response = await transport.handle_async_request(
                httpx.Request("GET", "https://acme.example/mcp")
            )

        assert response.status_code == 401
        factory.assert_called_once_with(PROXY_URL)
        delegate.handle_async_request.assert_awaited_once()
        direct.assert_not_awaited()

    async def test_direct_request_does_not_touch_a_delegate(self):
        transport = url_guard.GuardedAsyncTransport(guard_profile=url_guard.PROXY_PROFILE)
        with (
            patch.dict("os.environ", _proxy_env(enabled=None), clear=True),
            patch.object(url_guard, "settings", _settings()),
            patch.object(
                url_guard, "_resolve_public_ips_async", new=AsyncMock(return_value=[PUBLIC_IP])
            ),
            patch.object(
                httpx.AsyncHTTPTransport,
                "handle_async_request",
                new=AsyncMock(return_value=httpx.Response(200)),
            ) as direct,
        ):
            await transport.handle_async_request(httpx.Request("GET", "https://acme.example/x"))
        direct.assert_awaited_once()
        assert transport._proxy_delegates == {}

    async def test_delegate_is_created_once_per_proxy_url(self):
        transport = url_guard.GuardedAsyncTransport(guard_profile=url_guard.PROXY_PROFILE)
        with patch.dict("os.environ", _proxy_env(), clear=True):
            first = transport._proxy_delegate(PROXY_URL)
            second = transport._proxy_delegate(PROXY_URL)
            third = transport._proxy_delegate("http://other-proxy:3128")
        assert first is second
        assert third is not first
        assert len(transport._proxy_delegates) == 2
        await transport.aclose()

    async def test_delegate_inherits_verify_and_retries(self):
        transport = url_guard.GuardedAsyncTransport(
            guard_profile=url_guard.PROXY_PROFILE, verify=False, retries=3
        )
        with patch.dict("os.environ", _proxy_env(), clear=True):
            transport._proxy_delegate(PROXY_URL)
        assert transport._delegate_kwargs["verify"] is False
        assert transport._delegate_kwargs["retries"] == 3
        await transport.aclose()

    async def test_delegates_are_closed_on_aclose(self):
        transport = url_guard.GuardedAsyncTransport(guard_profile=url_guard.PROXY_PROFILE)
        delegate = MagicMock()
        delegate.aclose = AsyncMock()
        transport._proxy_delegates[PROXY_URL] = delegate
        await transport.aclose()
        delegate.aclose.assert_awaited_once()
        assert transport._proxy_delegates == {}

    def test_sync_delegates_are_closed_on_close(self):
        transport = url_guard.GuardedTransport(guard_profile=url_guard.PROXY_PROFILE)
        delegate = MagicMock()
        transport._proxy_delegates[PROXY_URL] = delegate
        transport.close()
        delegate.close.assert_called_once()
        assert transport._proxy_delegates == {}

    def test_proxy_basic_auth_survives_proxy_object(self):
        """httpx.Proxy extracts userinfo into raw_auth itself."""
        with patch.dict("os.environ", {}, clear=True):
            proxy = url_guard._build_proxy("http://bob:sup3rsecret@proxy:3128")
        assert proxy.raw_auth == (b"bob", b"sup3rsecret")
        assert proxy.url.host == "proxy"


# ---------------------------------------------------------------------------
# CA bundle for a TLS-intercepting proxy.
# ---------------------------------------------------------------------------


def _single_cert_pem(tmp_path):
    """Write a one-certificate PEM file, taken from the certifi bundle."""
    marker = "-----END CERTIFICATE-----"
    first = certifi.contents().split(marker)[0] + marker + "\n"
    path = tmp_path / "corporate-ca.pem"
    path.write_text(first)
    return path


class TestCaBundle:
    def test_ca_bundle_unset_returns_default_verify(self):
        """An empty path leaves httpx to build its own context: no change."""
        with patch.dict("os.environ", {}, clear=True):
            assert url_guard._forward_proxy_ssl_context() is True

    def test_ca_bundle_loads_on_top_of_default_roots(self, tmp_path):
        """What SSL_CERT_FILE would have broken.

        ssl.create_default_context(cafile=...) REPLACES the trust store, so an
        operator pointing SSL_CERT_FILE at their one corporate CA loses TLS to
        every direct-path target. A bundle of one cert must still leave the
        default roots loaded.
        """
        path = _single_cert_pem(tmp_path)
        with patch.dict("os.environ", {"EGRESS_FORWARD_PROXY_CA_BUNDLE": str(path)}, clear=True):
            context = url_guard._forward_proxy_ssl_context()
        assert isinstance(context, ssl.SSLContext)
        # Far more than the single certificate in the bundle itself.
        assert len(context.get_ca_certs()) > 1

    def test_ca_bundle_missing_path_raises(self, tmp_path):
        missing = str(tmp_path / "nope.pem")
        with patch.dict("os.environ", {"EGRESS_FORWARD_PROXY_CA_BUNDLE": missing}, clear=True):
            with pytest.raises(UrlValidationError):
                url_guard._forward_proxy_ssl_context()

    def test_ca_bundle_malformed_pem_raises(self, tmp_path):
        path = tmp_path / "not-a-pem.txt"
        path.write_text("this is not a certificate\n")
        with patch.dict("os.environ", {"EGRESS_FORWARD_PROXY_CA_BUNDLE": str(path)}, clear=True):
            with pytest.raises(UrlValidationError):
                url_guard._forward_proxy_ssl_context()

    def test_ca_bundle_context_is_cached(self, tmp_path):
        """An SSLContext hashes by identity, and the shared-client pool is keyed
        on (profile, verify), so a fresh context per call would multiply the pool."""
        path = _single_cert_pem(tmp_path)
        with patch.dict("os.environ", {"EGRESS_FORWARD_PROXY_CA_BUNDLE": str(path)}, clear=True):
            assert url_guard._forward_proxy_ssl_context() is url_guard._forward_proxy_ssl_context()

    def test_https_proxy_delegate_gets_proxy_ssl_context(self, tmp_path):
        """A bare URL string leaves httpx.Proxy.ssl_context None, so an https
        proxy signed by the internal CA would fail even with the upstream fixed."""
        path = _single_cert_pem(tmp_path)
        with patch.dict("os.environ", {"EGRESS_FORWARD_PROXY_CA_BUNDLE": str(path)}, clear=True):
            proxy = url_guard._build_proxy("https://proxy.corp.example:8443")
        assert isinstance(proxy.ssl_context, ssl.SSLContext)

    def test_ca_bundle_replaces_default_verify_on_the_delegate(self, tmp_path):
        """The bundle must reach the upstream leg inside the tunnel too."""
        path = _single_cert_pem(tmp_path)
        with patch.dict("os.environ", {"EGRESS_FORWARD_PROXY_CA_BUNDLE": str(path)}, clear=True):
            kwargs = url_guard._delegate_kwargs_with_ca_bundle({"verify": True, "retries": 2})
        assert isinstance(kwargs["verify"], ssl.SSLContext)
        assert kwargs["retries"] == 2

    def test_explicit_caller_verify_is_left_alone(self, tmp_path):
        path = _single_cert_pem(tmp_path)
        with patch.dict("os.environ", {"EGRESS_FORWARD_PROXY_CA_BUNDLE": str(path)}, clear=True):
            assert url_guard._delegate_kwargs_with_ca_bundle({"verify": False})["verify"] is False
            assert (
                url_guard._delegate_kwargs_with_ca_bundle({"verify": "/other/ca.pem"})["verify"]
                == "/other/ca.pem"
            )

    def test_startup_validation_raises_on_a_bad_bundle(self, tmp_path):
        """validate_forward_proxy_config is called from each app's lifespan, so a
        typo fails the process rather than the first proxied request."""
        missing = str(tmp_path / "nope.pem")
        with patch.dict("os.environ", {"EGRESS_FORWARD_PROXY_CA_BUNDLE": missing}, clear=True):
            with pytest.raises(UrlValidationError):
                url_guard.validate_forward_proxy_config()

    def test_startup_validation_checks_the_bundle_even_when_the_flag_is_off(self, tmp_path):
        """A typo must surface before the flag is flipped, not after."""
        missing = str(tmp_path / "nope.pem")
        env = {"EGRESS_FORWARD_PROXY_ENABLED": "false", "EGRESS_FORWARD_PROXY_CA_BUNDLE": missing}
        with patch.dict("os.environ", env, clear=True):
            with pytest.raises(UrlValidationError):
                url_guard.validate_forward_proxy_config()

    def test_startup_validation_is_a_noop_when_unconfigured(self):
        with patch.dict("os.environ", {}, clear=True):
            url_guard.validate_forward_proxy_config()  # must not raise


# ---------------------------------------------------------------------------
# The PLAIN egress client (issue #1834).
#
# shared_plain_async_client serves operator-configured, already-trusted targets:
# the login IdP token/userinfo endpoints and the in-cluster registry vends.
# Pooling the login callback behind an explicit transport removed httpx's
# environment-proxy support, which is a complete sign-in outage on a proxy-only
# network with a hosted IdP.
#
# This transport must route PUBLIC targets through the proxy while adding NO
# pinning and NO SSRF denial: its whole purpose is reaching targets the guard
# rejects, such as an in-cluster http://keycloak:8080 at a private address.
# ---------------------------------------------------------------------------


class TestPlainProxyRoutedTransport:
    async def test_flag_off_is_a_no_op(self):
        """Byte-for-byte the previous behavior: no resolution, no delegate."""
        transport = url_guard.PlainProxyRoutedAsyncTransport()
        with (
            patch.dict("os.environ", _proxy_env(enabled=None), clear=True),
            patch.object(url_guard, "_resolves_exclusively_public_async") as resolver,
        ):
            proxy = await transport._proxy_url_for(
                httpx.Request("POST", "https://login.microsoftonline.com/t/oauth2/v2.0/token")
            )
        assert proxy is None
        resolver.assert_not_called()
        assert transport._proxy_delegates == {}

    async def test_public_idp_is_routed_through_the_proxy(self):
        """The #1834 fix: a hosted IdP token endpoint reaches the proxy."""
        transport = url_guard.PlainProxyRoutedAsyncTransport()
        with (
            patch.dict("os.environ", _proxy_env(), clear=True),
            _stub_async_dns(PUBLIC_IP),
        ):
            proxy = await transport._proxy_url_for(
                httpx.Request("POST", "https://login.microsoftonline.com/t/oauth2/v2.0/token")
            )
        assert proxy == PROXY_URL

    async def test_in_cluster_idp_stays_direct_with_no_no_proxy_entry(self):
        """An in-cluster Keycloak needs no configuration to stay direct.

        This is the property that makes the two internal vends safe: they are
        direct by construction, not because an operator remembered to exclude
        them. A plain http:// private target must also NOT raise, which is what
        would happen if this transport inherited the guard.
        """
        transport = url_guard.PlainProxyRoutedAsyncTransport()
        with (
            patch.dict("os.environ", _proxy_env(no_proxy=None), clear=True),
            _stub_async_dns(PRIVATE_IP),
        ):
            proxy = await transport._proxy_url_for(
                httpx.Request(
                    "POST", "http://keycloak:8080/realms/mcp/protocol/openid-connect/token"
                )
            )
        assert proxy is None
        assert transport._proxy_delegates == {}

    async def test_in_cluster_vend_stays_direct_with_no_no_proxy_entry(self):
        transport = url_guard.PlainProxyRoutedAsyncTransport()
        with (
            patch.dict("os.environ", _proxy_env(no_proxy=None), clear=True),
            _stub_async_dns(PRIVATE_IP),
        ):
            proxy = await transport._proxy_url_for(
                httpx.Request("POST", "http://registry:8091/_egress_internal/egress-token")
            )
        assert proxy is None

    async def test_no_proxy_override_keeps_a_public_target_direct(self):
        transport = url_guard.PlainProxyRoutedAsyncTransport()
        with (
            patch.dict("os.environ", _proxy_env(no_proxy="login.microsoftonline.com"), clear=True),
            _stub_async_dns(PUBLIC_IP),
        ):
            proxy = await transport._proxy_url_for(
                httpx.Request("POST", "https://login.microsoftonline.com/t/oauth2/v2.0/token")
            )
        assert proxy is None

    async def test_no_proxy_for_this_scheme_stays_direct(self):
        transport = url_guard.PlainProxyRoutedAsyncTransport()
        with (
            patch.dict("os.environ", _proxy_env(http_proxy=None), clear=True),
            _stub_async_dns(PUBLIC_IP),
        ):
            proxy = await transport._proxy_url_for(
                httpx.Request("POST", "http://public-idp.example/token")
            )
        assert proxy is None

    async def test_resolution_failure_falls_back_to_direct_without_raising(self):
        """A new failure mode would be worse than the one it replaces.

        The guarded client fails closed on a resolution error because it is an
        access decision there. Here resolution is only a routing hint, so a
        failure dials direct and lets httpx surface its own error.
        """
        transport = url_guard.PlainProxyRoutedAsyncTransport()
        with (
            patch.dict("os.environ", _proxy_env(), clear=True),
            patch.object(
                url_guard.asyncio.get_event_loop(),
                "getaddrinfo",
                new=AsyncMock(side_effect=url_guard.socket.gaierror("nxdomain")),
            ),
        ):
            proxy = await transport._proxy_url_for(
                httpx.Request("POST", "https://gone.example/token")
            )
        assert proxy is None

    async def test_mixed_public_and_internal_answers_stay_direct(self):
        transport = url_guard.PlainProxyRoutedAsyncTransport()
        with (
            patch.dict("os.environ", _proxy_env(), clear=True),
            _stub_async_dns(PUBLIC_IP, PRIVATE_IP),
        ):
            proxy = await transport._proxy_url_for(
                httpx.Request("POST", "https://split.example/token")
            )
        assert proxy is None

    async def test_public_http_target_refuses_cleartext(self):
        """An IdP token POST carries the operator client_secret.

        httpcore's plain forward path sends the absolute URI and merges headers,
        so the proxy would see the secret. Fail closed, and name the remedy.
        """
        transport = url_guard.PlainProxyRoutedAsyncTransport()
        with (
            patch.dict("os.environ", _proxy_env(), clear=True),
            _stub_async_dns(PUBLIC_IP),
        ):
            with pytest.raises(UrlValidationError) as exc:
                await transport._proxy_url_for(
                    httpx.Request("POST", "http://public-idp.example/token")
                )
        assert "NO_PROXY" in str(exc.value)
        assert "cleartext" in str(exc.value)

    async def test_public_ip_literal_is_proxied(self):
        transport = url_guard.PlainProxyRoutedAsyncTransport()
        with patch.dict("os.environ", _proxy_env(), clear=True):
            proxy = await transport._proxy_url_for(
                httpx.Request("POST", f"https://{PUBLIC_IP}/token")
            )
        assert proxy == PROXY_URL

    async def test_private_ip_literal_stays_direct(self):
        transport = url_guard.PlainProxyRoutedAsyncTransport()
        with patch.dict("os.environ", _proxy_env(), clear=True):
            proxy = await transport._proxy_url_for(
                httpx.Request("POST", f"http://{PRIVATE_IP}:8091/_egress_internal/egress-token")
            )
        assert proxy is None

    async def test_metadata_ip_literal_stays_direct_and_is_not_denied(self):
        """This transport does NOT make access decisions.

        A metadata address classifies as internal, so it is routed direct rather
        than denied. Denying here would be a behavior change on a client whose
        callers are static operator config, and the guarded client is what
        protects request-derived targets.
        """
        transport = url_guard.PlainProxyRoutedAsyncTransport()
        with patch.dict("os.environ", _proxy_env(), clear=True):
            proxy = await transport._proxy_url_for(
                httpx.Request("GET", f"http://{METADATA_IP}/latest/meta-data/")
            )
        assert proxy is None

    async def test_request_is_handed_over_unmodified(self):
        """No pinning: the host must survive so the tunnel can verify the cert."""
        transport = url_guard.PlainProxyRoutedAsyncTransport()
        delegate = MagicMock()
        delegate.handle_async_request = AsyncMock(return_value=httpx.Response(200))
        request = httpx.Request("POST", "https://login.microsoftonline.com/t/oauth2/v2.0/token")
        with (
            patch.dict("os.environ", _proxy_env(), clear=True),
            _stub_async_dns(PUBLIC_IP),
            patch.object(transport, "_proxy_delegate", return_value=delegate),
        ):
            await transport.handle_async_request(request)
        forwarded = delegate.handle_async_request.await_args.args[0]
        assert forwarded.url.host == "login.microsoftonline.com"
        assert "sni_hostname" not in forwarded.extensions

    async def test_direct_path_does_not_touch_a_delegate(self):
        transport = url_guard.PlainProxyRoutedAsyncTransport()
        with (
            patch.dict("os.environ", _proxy_env(), clear=True),
            _stub_async_dns(PRIVATE_IP),
            patch.object(
                httpx.AsyncHTTPTransport,
                "handle_async_request",
                new=AsyncMock(return_value=httpx.Response(200)),
            ) as direct,
        ):
            await transport.handle_async_request(
                httpx.Request("POST", "http://keycloak:8080/token")
            )
        direct.assert_awaited_once()
        assert transport._proxy_delegates == {}

    async def test_routing_is_recorded_under_the_plain_profile(self):
        transport = url_guard.PlainProxyRoutedAsyncTransport()
        with (
            patch.dict("os.environ", _proxy_env(), clear=True),
            _stub_async_dns(PUBLIC_IP),
            patch.object(url_guard, "_record_egress_route") as record,
        ):
            await transport._proxy_url_for(
                httpx.Request("POST", "https://login.microsoftonline.com/t/oauth2/v2.0/token")
            )
        record.assert_called_once_with("plain", "proxied", "ok")

    async def test_delegates_are_closed_on_aclose(self):
        transport = url_guard.PlainProxyRoutedAsyncTransport()
        delegate = MagicMock()
        delegate.aclose = AsyncMock()
        transport._proxy_delegates[PROXY_URL] = delegate
        await transport.aclose()
        delegate.aclose.assert_awaited_once()
        assert transport._proxy_delegates == {}

    def test_shared_plain_client_uses_the_proxy_routed_transport(self):
        """The regression guard: a bare AsyncHTTPTransport here is issue #1834."""
        with patch.object(url_guard, "settings", _settings_with_pool()):
            url_guard.reset_shared_clients_for_tests()
            client = url_guard.shared_plain_async_client()
        assert isinstance(client._transport, url_guard.PlainProxyRoutedAsyncTransport)
        url_guard.reset_shared_clients_for_tests()


class TestResolvesExclusivelyPublic:
    """The routing classifier used by the plain transport. It must never raise."""

    async def test_public_hostname(self):
        with _stub_async_dns(PUBLIC_IP):
            assert await url_guard._resolves_exclusively_public_async("acme.example", 443) is True

    async def test_private_hostname(self):
        with _stub_async_dns(PRIVATE_IP):
            assert await url_guard._resolves_exclusively_public_async("keycloak", 8080) is False

    async def test_mixed_answers_are_not_exclusively_public(self):
        with _stub_async_dns(PUBLIC_IP, PRIVATE_IP):
            assert await url_guard._resolves_exclusively_public_async("split.example", 443) is False

    async def test_resolution_failure_returns_none(self):
        with patch.object(
            url_guard.asyncio.get_event_loop(),
            "getaddrinfo",
            new=AsyncMock(side_effect=url_guard.socket.gaierror("nxdomain")),
        ):
            assert await url_guard._resolves_exclusively_public_async("gone.example", 443) is None

    async def test_timeout_returns_none(self):
        with patch.object(
            url_guard.asyncio.get_event_loop(),
            "getaddrinfo",
            new=AsyncMock(side_effect=TimeoutError()),
        ):
            assert await url_guard._resolves_exclusively_public_async("slow.example", 443) is None

    @pytest.mark.parametrize(
        "ip,expected",
        [
            (PUBLIC_IP, True),
            (PRIVATE_IP, False),
            (METADATA_IP, False),
            ("127.0.0.1", False),
            ("100.64.0.1", False),  # CGNAT
            ("2001:4860:4860::8888", True),  # public IPv6
            ("::1", False),
        ],
    )
    async def test_ip_literals_need_no_resolution(self, ip, expected):
        assert await url_guard._resolves_exclusively_public_async(ip, 443) is expected


class TestCredentialEndpointBypassIsMandatory:
    """The proxy must not be able to carry a cloud credential request.

    Fails closed at startup rather than warning, because the blocked combination
    is never a correct configuration and the leak would otherwise happen on the
    first boto3 call, long after anyone read the log.
    """

    ALL_THREE = "169.254.169.254,169.254.170.2,169.254.170.23"

    def test_flag_on_without_the_exclusions_refuses_to_start(self):
        env = _proxy_env(no_proxy="localhost,127.0.0.1,keycloak")
        with patch.dict("os.environ", env, clear=True):
            with pytest.raises(UrlValidationError) as exc:
                url_guard.validate_forward_proxy_config()
        message = str(exc.value)
        assert "Refusing to start" in message
        # The message must carry both the reason and the exact remedy.
        assert "169.254.169.254" in message
        assert "NO_PROXY" in message
        assert "cleartext" in message

    def test_the_imds_flag_is_not_accepted_as_a_substitute(self):
        """AWS_EC2_METADATA_DISABLED covers only 169.254.169.254.

        botocore reads it in IMDSFetcher alone; ContainerProvider is gated purely
        on AWS_CONTAINER_CREDENTIALS_RELATIVE_URI, so the ECS and EKS
        task-credential endpoints stay exposed. Accepting the flag here would
        leave two of the three addresses proxied.
        """
        env = _proxy_env(no_proxy="localhost")
        env["AWS_EC2_METADATA_DISABLED"] = "true"
        with patch.dict("os.environ", env, clear=True):
            with pytest.raises(UrlValidationError) as exc:
                url_guard.validate_forward_proxy_config()
        assert "not sufficient" in str(exc.value).lower()

    def test_partial_exclusion_still_refuses(self):
        """Excluding EC2 IMDS alone leaves the container endpoints exposed."""
        env = _proxy_env(no_proxy="169.254.169.254,localhost")
        with patch.dict("os.environ", env, clear=True):
            with pytest.raises(UrlValidationError) as exc:
                url_guard.validate_forward_proxy_config()
        message = str(exc.value)
        assert "169.254.170.2" in message
        assert "169.254.170.23" in message

    def test_all_three_excluded_starts_cleanly(self):
        env = _proxy_env(no_proxy=f"localhost,127.0.0.1,{self.ALL_THREE},keycloak")
        with patch.dict("os.environ", env, clear=True):
            url_guard.validate_forward_proxy_config()  # must not raise

    def test_no_proxy_wildcard_satisfies_the_requirement(self):
        """NO_PROXY=* excludes everything, including the credential endpoints."""
        with patch.dict("os.environ", _proxy_env(no_proxy="*"), clear=True):
            url_guard.validate_forward_proxy_config()  # must not raise

    def test_flag_off_does_not_require_anything(self):
        """An unconfigured deployment is unaffected: no proxy, no hazard."""
        with patch.dict("os.environ", _proxy_env(enabled=None, no_proxy=None), clear=True):
            url_guard.validate_forward_proxy_config()  # must not raise

    def test_flag_on_with_no_proxy_configured_does_not_require_anything(self):
        """The flag alone is inert: _forward_proxy_config returns disabled."""
        env = _proxy_env(http_proxy=None, https_proxy=None, no_proxy=None)
        with patch.dict("os.environ", env, clear=True):
            url_guard.validate_forward_proxy_config()  # must not raise

    def test_the_recommended_docs_value_satisfies_the_check(self):
        """Guards against the docs and the code drifting apart.

        This is the value docs/forward-proxy-egress.md and the FAQ recommend. If
        someone trims it, this test fails rather than a deployment refusing to
        boot.
        """
        recommended = (
            "localhost,127.0.0.1,169.254.169.254,169.254.170.2,169.254.170.23,"
            ".svc.cluster.local,mcpgw-server,auth-server,keycloak"
        )
        with patch.dict("os.environ", _proxy_env(no_proxy=recommended), clear=True):
            url_guard.validate_forward_proxy_config()  # must not raise
