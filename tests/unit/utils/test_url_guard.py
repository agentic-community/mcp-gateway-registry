"""Unit tests for the hardened SSRF URL guard (registry.utils.url_guard).

Covers, for both validation profiles:
- scheme rejection (only http/https, optional https-only),
- private / loopback / link-local / reserved / multicast / unspecified / cloud
  metadata rejection,
- IPv4-mapped IPv6 unwrapping,
- nginx metacharacter rejection for proxy_pass_url,
- operator allowlist bypass (github_extra_hosts for skills; ssrf_allowed_hosts /
  ssrf_allowed_cidrs for proxy targets),
- DNS-rebinding defeat: the pinned transport rewrites the connection target to a
  validated IP, preserving Host + SNI, and re-validates literal IPs.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from registry.exceptions import UrlValidationError
from registry.utils import url_guard


def _reset_caches() -> None:
    url_guard._skill_allowlist.cache_clear()
    url_guard._proxy_allowlist.cache_clear()
    url_guard._builtin_airegistry_tools_allowlist.cache_clear()


@pytest.fixture(autouse=True)
def _clear_allowlist_caches():
    _reset_caches()
    yield
    _reset_caches()


def _resolve_to(*ips: str):
    """getaddrinfo stub resolving any host to the given IP(s)."""

    def _stub(host, port, **kw):
        return [(2, 1, 6, "", (ip, port)) for ip in ips]

    return _stub


def _settings(github_extra_hosts="", ssrf_allowed_hosts="", ssrf_allowed_cidrs=""):
    s = MagicMock()
    s.github_extra_hosts = github_extra_hosts
    s.ssrf_allowed_hosts = ssrf_allowed_hosts
    s.ssrf_allowed_cidrs = ssrf_allowed_cidrs
    return s


# ---------------------------------------------------------------------------
# Scheme validation
# ---------------------------------------------------------------------------


class TestScheme:
    @pytest.mark.parametrize("url", ["ftp://x/y", "file:///etc/passwd", "gopher://x", "//x/y"])
    def test_non_http_scheme_rejected(self, url):
        with pytest.raises(UrlValidationError):
            url_guard.validate_url(url)

    def test_http_rejected_when_https_required(self):
        with patch.object(url_guard, "settings", _settings()):
            with patch.object(url_guard.socket, "getaddrinfo", _resolve_to("93.184.216.34")):
                with pytest.raises(UrlValidationError):
                    url_guard.validate_url("http://acme.com/x", require_https=True)

    def test_missing_host_rejected(self):
        with pytest.raises(UrlValidationError):
            url_guard.validate_url("https:///nohost")

    def test_empty_url_rejected(self):
        with pytest.raises(UrlValidationError):
            url_guard.validate_url("")

    @pytest.mark.parametrize(
        "url",
        [
            "https://user@example.com/path",
            "https://user:secret@example.com/path",
            "https://example.com/path#fragment",
        ],
    )
    def test_userinfo_and_fragments_rejected(self, url):
        with pytest.raises(UrlValidationError):
            url_guard.validate_url(url, resolve=False)

    def test_query_is_preserved_in_normalized_identity(self):
        assert (
            url_guard.normalize_url_identity(
                "HTTPS://Example.COM:443/v1/resource?tenant=a&mode=full"
            )
            == "https://example.com/v1/resource?tenant=a&mode=full"
        )


# ---------------------------------------------------------------------------
# Private / metadata blocking
# ---------------------------------------------------------------------------


class TestPrivateAndMetadata:
    @pytest.mark.parametrize(
        "ip",
        [
            "127.0.0.1",
            "10.1.2.3",
            "192.168.1.5",
            "172.16.0.9",
            "169.254.169.254",
            "100.100.100.200",
            "::ffff:100.100.100.200",
            "64:ff9b::6464:64c8",
            "2002:6464:64c8::",
            "0.0.0.0",
            "224.0.0.1",
            "::1",
            "::ffff:10.0.0.1",
            "fe80::1",
            "fc00::1",
        ],
    )
    def test_blocked_targets_rejected(self, ip):
        with patch.object(url_guard, "settings", _settings()):
            with patch.object(url_guard.socket, "getaddrinfo", _resolve_to(ip)):
                with pytest.raises(UrlValidationError):
                    url_guard.validate_url("https://evil.example/x")

    def test_public_ip_allowed(self):
        with patch.object(url_guard, "settings", _settings()):
            with patch.object(url_guard.socket, "getaddrinfo", _resolve_to("93.184.216.34")):
                assert url_guard.validate_url("https://acme.com/x") == ["93.184.216.34"]

    def test_any_private_in_resolution_set_rejected(self):
        """If a host resolves to a public AND a private IP, it is rejected."""
        with patch.object(url_guard, "settings", _settings()):
            with patch.object(
                url_guard.socket, "getaddrinfo", _resolve_to("93.184.216.34", "10.0.0.1")
            ):
                with pytest.raises(UrlValidationError):
                    url_guard.validate_url("https://acme.com/x")

    def test_literal_private_ip_host_rejected(self):
        with patch.object(url_guard, "settings", _settings()):
            with pytest.raises(UrlValidationError):
                url_guard.validate_url("http://169.254.169.254/latest/meta-data/")

    def test_literal_public_ip_host_allowed(self):
        with patch.object(url_guard, "settings", _settings()):
            assert url_guard.validate_url("https://93.184.216.34/x") == ["93.184.216.34"]

    def test_dns_failure_fails_closed(self):
        def _boom(host, port, **kw):
            raise url_guard.socket.gaierror("nope")

        with patch.object(url_guard, "settings", _settings()):
            with patch.object(url_guard.socket, "getaddrinfo", _boom):
                with pytest.raises(UrlValidationError):
                    url_guard.validate_url("https://acme.com/x")


# ---------------------------------------------------------------------------
# Carrier-grade NAT (RFC 6598) explicit block
# ---------------------------------------------------------------------------


class TestCgnatBlock:
    """CGNAT (100.64.0.0/10) must be blocked independently of Python's is_private.

    These pin the exact range so a runtime/semantics change fails loudly here
    rather than silently re-opening an SSRF pivot to a shared-address-space host.
    """

    @pytest.mark.parametrize(
        "ip",
        [
            "100.64.0.1",
            "100.64.0.0",
            "100.100.50.1",
            "100.127.255.254",
            "::ffff:100.64.0.1",  # IPv4-mapped IPv6 form
        ],
    )
    def test_cgnat_ip_is_blocked(self, ip):
        assert url_guard._is_blocked_ip(ip, url_guard._Allowlist()) is True

    @pytest.mark.parametrize(
        "ip",
        [
            "100.63.255.255",  # just below the range
            "100.128.0.1",  # just above the range
        ],
    )
    def test_addresses_adjacent_to_cgnat_range_are_public(self, ip):
        assert url_guard._is_blocked_ip(ip, url_guard._Allowlist()) is False

    def test_cgnat_range_pinned_exactly(self):
        """The pinned network must be exactly 100.64.0.0/10 (RFC 6598)."""
        import ipaddress

        assert url_guard._CGNAT_NET == ipaddress.ip_network("100.64.0.0/10")

    def test_cgnat_literal_host_rejected(self):
        with patch.object(url_guard, "settings", _settings()):
            with pytest.raises(UrlValidationError):
                url_guard.validate_url("https://100.64.0.1/x")

    def test_cgnat_resolution_rejected(self):
        with patch.object(url_guard, "settings", _settings()):
            with patch.object(url_guard.socket, "getaddrinfo", _resolve_to("100.64.0.1")):
                with pytest.raises(UrlValidationError):
                    url_guard.validate_url("https://sneaky.example/x")


# ---------------------------------------------------------------------------
# nginx metacharacters
# ---------------------------------------------------------------------------


class TestNginxMetacharacters:
    @pytest.mark.parametrize(
        "url",
        [
            'http://evil.com/"; } location / { proxy_pass http://x;',
            "http://evil.com/\n}",
            "http://evil.com/a;b",
            "http://evil.com/${x}",
            "http://evil.com/a b",
        ],
    )
    def test_metacharacters_rejected(self, url):
        with pytest.raises(UrlValidationError):
            url_guard.validate_url(url, reject_nginx_metacharacters=True)

    def test_validate_proxy_pass_url_rejects_injection(self):
        with pytest.raises(UrlValidationError):
            url_guard.validate_proxy_pass_url('http://x/";}')

    def test_validate_proxy_pass_url_rejects_metadata_literal(self):
        with patch.object(url_guard, "settings", _settings()):
            with pytest.raises(UrlValidationError):
                url_guard.validate_proxy_pass_url("http://169.254.169.254/")

    def test_validate_proxy_pass_url_rejects_private_literal(self):
        with patch.object(url_guard, "settings", _settings()):
            with pytest.raises(UrlValidationError):
                url_guard.validate_proxy_pass_url("http://10.0.0.1/mcp")

    def test_validate_proxy_pass_url_rejects_bad_scheme(self):
        with patch.object(url_guard, "settings", _settings()):
            with pytest.raises(UrlValidationError):
                url_guard.validate_proxy_pass_url("ftp://acme.com/mcp")

    def test_validate_proxy_pass_url_allows_hostname_without_dns(self):
        """Registration-time validation does not resolve DNS (structural only)."""

        def _boom(*a, **k):
            raise AssertionError("registration validation must not perform DNS resolution")

        with patch.object(url_guard, "settings", _settings()):
            with patch.object(url_guard.socket, "getaddrinfo", _boom):
                url_guard.validate_proxy_pass_url("https://acme.com/mcp")
                url_guard.validate_agent_url("https://agent.example.com/a2a")

    def test_validate_proxy_pass_url_allows_exact_builtin_registration(self):
        with patch.object(url_guard, "settings", _settings()):
            url_guard.validate_proxy_pass_url(
                "http://mcpgw-server:8003/", server_path="/airegistry-tools/"
            )

    @pytest.mark.parametrize(
        "path,target",
        [
            ("/other/", "http://mcpgw-server:8003/"),
            ("/airegistry-tools/", "http://mcpgw-server:8004/"),
            ("/airegistry-tools/", "http://mcpgw-server:8003/other"),
        ],
    )
    def test_validate_proxy_pass_url_rejects_inexact_builtin_registration(self, path, target):
        with patch.object(url_guard, "settings", _settings()):
            with pytest.raises(UrlValidationError):
                url_guard.validate_proxy_pass_url(target, server_path=path)

    @pytest.mark.parametrize(
        "path",
        [
            "/evil; }",
            "/a\nb",
            "/x${y}",
            "/a b",
            '/"quote',
            "/has#comment",
        ],
    )
    def test_validate_server_path_rejects_metacharacters(self, path):
        with pytest.raises(UrlValidationError):
            url_guard.validate_server_path(path)

    @pytest.mark.parametrize(
        "path",
        ["/github", "/tools/currenttime", "/a-b_c.d/leaf"],
    )
    def test_validate_server_path_allows_normal_paths(self, path):
        url_guard.validate_server_path(path)  # does not raise

    def test_validate_server_path_rejects_empty(self):
        with pytest.raises(UrlValidationError):
            url_guard.validate_server_path("")

    @pytest.mark.parametrize(
        "path",
        [
            "/all",
            "all",
            "/ALL",
            "/All",
            "all/",
            "//all//",
            "/*",
            "*",
        ],
    )
    def test_validate_server_path_rejects_reserved_wildcard_names(self, path):
        """Reserved cross-server wildcard names (all/*), any case or slash
        wrapping, must be rejected (privilege escalation)."""
        with pytest.raises(UrlValidationError):
            url_guard.validate_server_path(path)

    @pytest.mark.parametrize(
        "path",
        [
            "/github",
            "/my-server",
            "/fininfo",
            "/all-tools",
            "/all/leaf",
            "/callthing",
        ],
    )
    def test_validate_server_path_allows_adjacent_names(self, path):
        """Only the EXACT reserved names are blocked; superstrings and paths
        that merely contain 'all' as a segment or substring stay valid."""
        url_guard.validate_server_path(path)  # does not raise

    @pytest.mark.parametrize("path", ["/", "//", "///"])
    def test_validate_server_path_rejects_root_and_slashes_only(self, path):
        """A slashes-only path normalizes to an empty server name and, after the
        trailing-slash location normalization (issue #1501), renders as a
        gateway-wide `location /` block that subjects every URL to the /validate
        auth subrequest. No real server registers at the root, so reject it."""
        with pytest.raises(UrlValidationError):
            url_guard.validate_server_path(path)


# ---------------------------------------------------------------------------
# Allowlist bypass behaviour
# ---------------------------------------------------------------------------


class TestAllowlists:
    def test_skill_profile_does_not_bypass_public_forge_domain(self):
        """github.com is NOT auto-trusted; it must pass IP validation."""
        with patch.object(url_guard, "settings", _settings(github_extra_hosts="")):
            with patch.object(url_guard.socket, "getaddrinfo", _resolve_to("10.0.0.5")):
                with pytest.raises(UrlValidationError):
                    url_guard.validate_url("https://github.com/x", profile=url_guard.SKILL_PROFILE)

    def test_skill_profile_ghes_host_resolves_classifies_and_pins(self):
        with patch.object(url_guard, "settings", _settings(github_extra_hosts="github.corp")):
            with patch.object(url_guard.socket, "getaddrinfo", _resolve_to("10.0.0.8")):
                assert url_guard.validate_url(
                    "https://github.corp/x", profile=url_guard.SKILL_PROFILE
                ) == ["10.0.0.8"]

    def test_proxy_profile_host_allowlist_resolves_classifies_and_pins(self):
        with patch.object(url_guard, "settings", _settings(ssrf_allowed_hosts="mcpgw,localhost")):
            with patch.object(url_guard.socket, "getaddrinfo", _resolve_to("10.1.2.3")):
                assert url_guard.validate_url(
                    "http://mcpgw:8000/mcp", profile=url_guard.PROXY_PROFILE
                ) == ["10.1.2.3"]

    def test_proxy_profile_mcpgw_is_reserved_not_globally_trusted(self):
        with patch.object(url_guard, "settings", _settings(ssrf_allowed_hosts="")):
            with pytest.raises(UrlValidationError, match="reserved"):
                url_guard.validate_url("http://mcpgw-server:8003/", profile=url_guard.PROXY_PROFILE)

    def test_exact_builtin_identity_selects_supported_profile(self):
        profile = url_guard.proxy_profile_for_entity_target(
            "mcp_server", "/airegistry-tools/", "http://mcpgw-server:8003/"
        )
        assert profile is url_guard.BUILTIN_AIREGISTRY_TOOLS_PROFILE
        with patch.object(url_guard, "settings", _settings(ssrf_allowed_hosts="")):
            with patch.object(url_guard.socket, "getaddrinfo", _resolve_to("10.0.0.9")):
                assert url_guard.validate_url("http://mcpgw-server:8003/", profile=profile) == [
                    "10.0.0.9"
                ]

    @pytest.mark.parametrize(
        "entity_type,path,target",
        [
            ("skill", "/skills/fake", "http://mcpgw-server:8003/"),
            ("a2a_agent", "/agents/fake", "http://mcpgw-server:8003/"),
            ("workflow", "/workflow/fake", "http://mcpgw-server:8003/"),
            ("mcp_server", "/not-airegistry-tools/", "http://mcpgw-server:8003/"),
            ("mcp_server", "/airegistry-tools/", "http://mcpgw-server:8004/"),
            ("mcp_server", "/airegistry-tools/", "http://mcpgw-server:8003/other"),
        ],
    )
    def test_arbitrary_entity_cannot_select_builtin_trust(self, entity_type, path, target):
        assert (
            url_guard.proxy_profile_for_entity_target(entity_type, path, target)
            is url_guard.PROXY_PROFILE
        )

    def test_proxy_profile_demo_servers_not_trusted_by_default(self):
        """Demo servers (currenttime, realserverfaketools) are opt-in and are NOT
        in the built-in trust set: they resolve normally and a private IP is
        blocked unless the operator allowlists them."""
        with patch.object(url_guard, "settings", _settings(ssrf_allowed_hosts="")):
            with patch.object(url_guard.socket, "getaddrinfo", _resolve_to("10.0.0.7")):
                for host in ("currenttime-server", "realserverfaketools-server"):
                    with pytest.raises(UrlValidationError):
                        url_guard.validate_url(
                            f"http://{host}:8000/mcp", profile=url_guard.PROXY_PROFILE
                        )

    @pytest.mark.parametrize(
        "credential_ip",
        [
            "169.254.169.254",
            "169.254.170.2",
            "169.254.170.23",
            "100.100.100.200",
            "::ffff:100.100.100.200",
            "64:ff9b::6464:64c8",
            "2002:6464:64c8::",
            "fd00:ec2::23",
            "fd00:ec2::23%eth0",
            "fd00:ec2::254",
            "fd00:ec2::254%eth0",
        ],
    )
    def test_proxy_profile_trusted_host_never_permits_credential_endpoint(self, credential_ip):
        with patch.object(url_guard, "settings", _settings(ssrf_allowed_hosts="internal.corp")):
            with patch.object(url_guard.socket, "getaddrinfo", _resolve_to(credential_ip)):
                with pytest.raises(UrlValidationError):
                    url_guard.validate_url(
                        "http://internal.corp/mcp", profile=url_guard.PROXY_PROFILE
                    )

    def test_proxy_profile_cidr_allowlist_permits_private(self):
        with patch.object(url_guard, "settings", _settings(ssrf_allowed_cidrs="10.0.0.0/8")):
            with patch.object(url_guard.socket, "getaddrinfo", _resolve_to("10.1.2.3")):
                assert url_guard.validate_url(
                    "http://internal.corp/mcp", profile=url_guard.PROXY_PROFILE
                ) == ["10.1.2.3"]

    @pytest.mark.parametrize(
        "credential_ip,cidr",
        [
            ("169.254.169.254", "169.254.0.0/16"),
            ("169.254.170.2", "169.254.0.0/16"),
            ("169.254.170.23", "169.254.0.0/16"),
            ("100.100.100.200", "100.64.0.0/10"),
            ("fd00:ec2::23", "fd00:ec2::/64"),
            ("fd00:ec2::23%eth0", "fd00:ec2::/64"),
            ("fd00:ec2::254", "fd00:ec2::/64"),
            ("fd00:ec2::254%eth0", "fd00:ec2::/64"),
        ],
    )
    def test_proxy_profile_cidr_allowlist_never_permits_credential_endpoints(
        self, credential_ip, cidr
    ):
        with patch.object(url_guard, "settings", _settings(ssrf_allowed_cidrs=cidr)):
            with patch.object(url_guard.socket, "getaddrinfo", _resolve_to(credential_ip)):
                with pytest.raises(UrlValidationError):
                    url_guard.validate_url("http://sneaky.corp/x", profile=url_guard.PROXY_PROFILE)

    @pytest.mark.parametrize("host", ["fd00:ec2::23%25eth0", "fd00:ec2::254%25eth0"])
    def test_proxy_profile_cidr_allowlist_never_permits_scoped_literal(self, host):
        with patch.object(url_guard, "settings", _settings(ssrf_allowed_cidrs="fd00:ec2::/64")):
            with pytest.raises(UrlValidationError):
                url_guard.validate_url(f"http://[{host}]/x", profile=url_guard.PROXY_PROFILE)

    def test_skill_allowlist_does_not_leak_into_proxy_profile(self):
        with patch.object(url_guard, "settings", _settings(github_extra_hosts="github.corp")):
            with patch.object(url_guard.socket, "getaddrinfo", _resolve_to("10.0.0.5")):
                with pytest.raises(UrlValidationError):
                    url_guard.validate_url("https://github.corp/x", profile=url_guard.PROXY_PROFILE)


# ---------------------------------------------------------------------------
# Pinned transport (DNS-rebinding defeat)
# ---------------------------------------------------------------------------


class TestPinnedTransport:
    def test_pin_rewrites_host_to_validated_ip(self):
        with patch.object(url_guard, "settings", _settings()):
            with patch.object(url_guard.socket, "getaddrinfo", _resolve_to("93.184.216.34")):
                transport = url_guard.GuardedAsyncTransport(guard_profile=url_guard.SKILL_PROFILE)
                request = httpx.Request("GET", "https://acme.com/path")
                pinned = transport._pin_request(request)

        # Connection target rewritten to the validated public IP.
        assert pinned.url.host == "93.184.216.34"
        # Host header and SNI preserve the original hostname (so TLS + vhost work).
        assert pinned.headers["Host"] == "acme.com"
        assert pinned.extensions["sni_hostname"] == "acme.com"

    async def test_async_pin_uses_async_resolver_and_rewrites(self):
        with patch.object(url_guard, "settings", _settings()):
            transport = url_guard.GuardedAsyncTransport(guard_profile=url_guard.SKILL_PROFILE)
            request = httpx.Request("GET", "https://acme.com/path")
            with patch.object(
                url_guard,
                "_resolve_public_ips_async",
                new=AsyncMock(return_value=["93.184.216.34"]),
            ) as resolver:
                pinned = await transport._pin_request_async(request)
        resolver.assert_awaited_once()
        assert pinned.url.host == "93.184.216.34"
        assert pinned.headers["Host"] == "acme.com"
        assert pinned.extensions["sni_hostname"] == "acme.com"

    def test_pin_blocks_private_resolution(self):
        with patch.object(url_guard, "settings", _settings()):
            with patch.object(url_guard.socket, "getaddrinfo", _resolve_to("10.0.0.5")):
                transport = url_guard.GuardedAsyncTransport(guard_profile=url_guard.SKILL_PROFILE)
                request = httpx.Request("GET", "https://acme.com/path")
                with pytest.raises(UrlValidationError):
                    transport._pin_request(request)

    def test_pin_blocks_metadata_literal_ip(self):
        with patch.object(url_guard, "settings", _settings()):
            transport = url_guard.GuardedAsyncTransport(guard_profile=url_guard.PROXY_PROFILE)
            request = httpx.Request("GET", "http://169.254.169.254/latest/meta-data/")
            with pytest.raises(UrlValidationError):
                transport._pin_request(request)

    def test_pin_rebinding_between_check_and_connect_is_defeated(self):
        """A host that validated once but rebinds to a private IP is still blocked.

        The transport resolves+validates at connect time, so a rebind after an
        earlier validate_url() call cannot slip a private IP through.
        """
        with patch.object(url_guard, "settings", _settings()):
            # First: passes an out-of-band pre-check (public IP).
            with patch.object(url_guard.socket, "getaddrinfo", _resolve_to("93.184.216.34")):
                url_guard.validate_url("https://rebind.example/x")

            # Then: the host rebinds to a private IP. The transport re-resolves
            # inside _pin_request and blocks it.
            with patch.object(url_guard.socket, "getaddrinfo", _resolve_to("10.0.0.5")):
                transport = url_guard.GuardedAsyncTransport(guard_profile=url_guard.SKILL_PROFILE)
                request = httpx.Request("GET", "https://rebind.example/x")
                with pytest.raises(UrlValidationError):
                    transport._pin_request(request)

    def test_sync_transport_pins_too(self):
        with patch.object(url_guard, "settings", _settings()):
            with patch.object(url_guard.socket, "getaddrinfo", _resolve_to("93.184.216.34")):
                transport = url_guard.GuardedTransport(guard_profile=url_guard.SKILL_PROFILE)
                request = httpx.Request("GET", "https://acme.com/path")
                pinned = transport._pin_request(request)
        assert pinned.url.host == "93.184.216.34"

    def test_proxy_profile_rebind_to_metadata_is_defeated(self):
        """The generic-proxy fetch path (PROXY_PROFILE) blocks a rebind to the
        metadata IP at connect time — the core Step-2 guarantee."""
        with patch.object(url_guard, "settings", _settings()):
            with patch.object(url_guard.socket, "getaddrinfo", _resolve_to("93.184.216.34")):
                url_guard.validate_url("https://dash.example/x", profile=url_guard.PROXY_PROFILE)
            with patch.object(url_guard.socket, "getaddrinfo", _resolve_to("169.254.169.254")):
                transport = url_guard.GuardedAsyncTransport(guard_profile=url_guard.PROXY_PROFILE)
                request = httpx.Request("GET", "https://dash.example/x")
                with pytest.raises(UrlValidationError):
                    transport._pin_request(request)

    def test_guarded_client_denies_redirect_to_metadata_ip(self):
        """End-to-end: a public host that 302-redirects to the metadata IP is
        denied at the SECOND hop.

        httpx's own redirect-following machinery re-issues the redirected
        request through GuardedTransport, whose _pin_request rejects the
        metadata literal before any connection is made. This documents the
        per-redirect guarantee through the real httpx.Client path (not just a
        direct _pin_request call)."""
        metadata_url = "http://169.254.169.254/latest/meta-data/"
        seen: list[str] = []

        def _fake_base_handle(self, request):
            # Records every hop that reaches the underlying (network) transport.
            # The metadata hop must never appear here: the guard blocks it in
            # _pin_request, before super().handle_request runs.
            seen.append(request.url.host)
            return httpx.Response(302, headers={"Location": metadata_url})

        with patch.object(url_guard, "settings", _settings()):
            with patch.object(url_guard.socket, "getaddrinfo", _resolve_to("93.184.216.34")):
                with patch.object(httpx.HTTPTransport, "handle_request", _fake_base_handle):
                    with url_guard.guarded_client(
                        profile=url_guard.PROXY_PROFILE,
                        follow_redirects=True,
                    ) as client:
                        with pytest.raises(UrlValidationError):
                            client.get("https://public.example/start")

        # Only the first (public) hop reached the network; the metadata IP the
        # redirect pointed at was blocked before a second connection.
        assert seen == ["93.184.216.34"]


class TestProxyProfileAllowlistTiers:
    """Explicit trust cannot reopen hard-denied destination categories."""

    @pytest.mark.parametrize("ip", ["169.254.10.10", "0.0.0.0", "fe80::1", "::"])
    def test_explicit_cidr_does_not_override_link_local_or_unspecified(self, ip):
        cidr = "169.254.0.0/16" if ":" not in ip else "::/0"
        with patch.object(url_guard, "settings", _settings(ssrf_allowed_cidrs=cidr)):
            with pytest.raises(UrlValidationError):
                url_guard.validate_url(
                    f"http://[{ip}]/x" if ":" in ip else f"http://{ip}/x",
                    profile=url_guard.PROXY_PROFILE,
                    resolve=False,
                )

    def test_obfuscated_metadata_literal_denied_via_proxy_profile(self):
        with patch.object(url_guard, "settings", _settings()):
            for spelling in ("http://0xA9FEA9FE/x", "http://2852039166/x"):
                with pytest.raises(UrlValidationError):
                    url_guard.validate_url(spelling, profile=url_guard.PROXY_PROFILE, resolve=False)


class TestBuiltinExactOutboundIdentity:
    @pytest.mark.parametrize(
        "actual",
        [
            "http://other.example/mcp",
            "http://mcpgw-server:8004/mcp",
            "http://mcpgw-server:8003/other",
            "http://mcpgw-server:8003/mcp?tenant=other",
        ],
    )
    def test_registered_builtin_rejects_different_actual_identity(self, actual):
        with pytest.raises(UrlValidationError, match="exact built-in"):
            url_guard.proxy_profile_for_entity_target(
                "mcp_server",
                "/airegistry-tools/",
                "http://mcpgw-server:8003/",
                actual,
            )

    def test_missing_registered_target_cannot_select_builtin_trust(self):
        profile = url_guard.proxy_profile_for_entity_target(
            "mcp_server",
            "/airegistry-tools/",
            None,
            "http://mcpgw-server:8003/",
        )
        assert profile is url_guard.PROXY_PROFILE
        with pytest.raises(UrlValidationError, match="reserved"):
            url_guard.validate_url(
                "http://mcpgw-server:8003/",
                profile=profile,
                resolve=False,
            )

    @pytest.mark.parametrize(
        "actual",
        [
            "http://mcpgw-server:8003/",
            "http://mcpgw-server:8003/mcp",
            "http://mcpgw-server:8003/mcp/",
        ],
    )
    def test_registered_builtin_accepts_only_known_exact_identities(self, actual):
        assert (
            url_guard.proxy_profile_for_entity_target(
                "mcp_server",
                "/airegistry-tools/",
                "http://mcpgw-server:8003/",
                actual,
            )
            is url_guard.BUILTIN_AIREGISTRY_TOOLS_PROFILE
        )


class TestCredentialedOAuthProfile:
    def test_profile_has_empty_allowlist_and_requires_https(self):
        profile = url_guard.CREDENTIALED_OAUTH_PROFILE
        allowlist = profile.allowlist_factory()
        assert profile.name == "credentialed-oauth"
        assert profile.require_https is True
        assert allowlist.hosts == frozenset()
        assert allowlist.cidrs == ()

    def test_proxy_allowlists_cannot_relax_oauth_token_target(self):
        with patch.object(
            url_guard,
            "settings",
            _settings(
                ssrf_allowed_hosts="token.internal",
                ssrf_allowed_cidrs="10.0.0.0/8",
            ),
        ):
            with patch.object(url_guard.socket, "getaddrinfo", _resolve_to("10.0.0.8")):
                with pytest.raises(UrlValidationError):
                    url_guard.validate_url(
                        "https://token.internal/oauth/token",
                        profile=url_guard.CREDENTIALED_OAUTH_PROFILE,
                    )

    def test_transport_rejects_http_even_for_public_host(self):
        transport = url_guard.GuardedAsyncTransport(
            guard_profile=url_guard.CREDENTIALED_OAUTH_PROFILE
        )
        with pytest.raises(UrlValidationError):
            transport._pin_request(httpx.Request("POST", "http://93.184.216.34/token"))
