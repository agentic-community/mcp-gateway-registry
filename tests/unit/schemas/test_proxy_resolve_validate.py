"""Unit tests for the DNS-aware resolve-and-validate proxy guard (SSRF layer 2).

resolve_and_validate_proxy_target resolves a proxy_target_url's hostname and
validates EVERY resolved IP against the egress policy — the check
_assert_egress_allowed deliberately skips for hostnames. These lock in:
- literal-IP targets are category-checked without DNS;
- a hostname resolving to a metadata/private IP is rejected (SSRF), regardless of
  the public IP it might also resolve to;
- gateway_proxy_allow_private_targets relaxes private but NEVER link-local;
- validate_and_pin_proxy_target returns the pin fields only for a live target and
  is a no-op for federated/disabled/targetless entities.
"""

import socket
from contextlib import contextmanager
from unittest.mock import patch

import pytest

from registry.schemas.proxy_mixin import (
    EgressPolicyError,
    resolve_and_validate_proxy_target,
    validate_and_pin_proxy_target,
)

pytestmark = pytest.mark.unit


def _addrinfo(*ips):
    """Build getaddrinfo-shaped tuples for the given IP strings."""
    out = []
    for ip in ips:
        family = socket.AF_INET6 if ":" in ip else socket.AF_INET
        out.append((family, socket.SOCK_STREAM, socket.IPPROTO_TCP, "", (ip, 443)))
    return out


@contextmanager
def _patch_settings(allow_private=False):
    # resolve_and_validate_proxy_target delegates to url_guard, which reads the
    # allowlist through its OWN imported `settings` reference and caches it
    # (@lru_cache). Patch url_guard.settings directly and clear the cache so the
    # flag flip is honored; also patch the config.settings the helper reads.
    from registry.utils import url_guard

    with patch("registry.core.config.settings") as s:
        s.gateway_proxy_allow_private_targets = allow_private
        s.ssrf_allowed_hosts = ""
        s.ssrf_allowed_cidrs = ""
        with patch.object(url_guard, "settings", s):
            url_guard._proxy_allowlist.cache_clear()
            try:
                yield s
            finally:
                url_guard._proxy_allowlist.cache_clear()


class TestLiteralIpTargets:
    async def test_public_ip_literal_passes_without_dns(self):
        with _patch_settings():
            with patch("socket.getaddrinfo") as gai:
                host, ips = await resolve_and_validate_proxy_target("https://8.8.8.8/")
                gai.assert_not_called()  # literal IP => no DNS
        assert host is None
        assert ips == ["8.8.8.8"]

    async def test_metadata_ip_literal_denied(self):
        with _patch_settings(allow_private=True):  # allow_private must NOT relax link-local
            with pytest.raises(EgressPolicyError):
                await resolve_and_validate_proxy_target("http://169.254.169.254/")

    async def test_private_ip_literal_denied_by_default(self):
        with _patch_settings(allow_private=False):
            with pytest.raises(EgressPolicyError):
                await resolve_and_validate_proxy_target("http://10.0.0.5/")

    async def test_private_ip_literal_allowed_when_flag_set(self):
        with _patch_settings(allow_private=True):
            host, ips = await resolve_and_validate_proxy_target("http://10.0.0.5/")
        assert ips == ["10.0.0.5"]


class TestHostnameResolution:
    async def test_hostname_all_public_passes_and_pins(self):
        with _patch_settings():
            with patch("socket.getaddrinfo", return_value=_addrinfo("93.184.216.34")):
                host, ips = await resolve_and_validate_proxy_target("https://ok.example/")
        assert host == "ok.example"
        assert ips == ["93.184.216.34"]

    async def test_hostname_resolving_to_metadata_rejected(self):
        # DNS-rebind style: the name resolves to the metadata IP now.
        with _patch_settings():
            with patch("socket.getaddrinfo", return_value=_addrinfo("169.254.169.254")):
                with pytest.raises(EgressPolicyError):
                    await resolve_and_validate_proxy_target("https://rebind.example/")

    async def test_hostname_with_any_denied_ip_rejected(self):
        # Multiple A records; one is private -> reject the whole target.
        with _patch_settings():
            with patch("socket.getaddrinfo", return_value=_addrinfo("93.184.216.34", "10.0.0.5")):
                with pytest.raises(EgressPolicyError):
                    await resolve_and_validate_proxy_target("https://mixed.example/")

    async def test_unresolvable_hostname_raises_valueerror(self):
        # url_guard surfaces a DNS failure as UrlValidationError; the proxy adapter
        # wraps it as EgressPolicyError (a ValueError) so it maps to a 4xx.
        with _patch_settings():
            with patch("socket.getaddrinfo", side_effect=socket.gaierror("nxdomain")):
                with pytest.raises(ValueError, match="[Rr]esolution failed"):
                    await resolve_and_validate_proxy_target("https://nope.example/")

    async def test_non_http_scheme_rejected(self):
        with _patch_settings():
            with pytest.raises(ValueError, match="scheme"):
                await resolve_and_validate_proxy_target("gopher://x/")


class TestValidateAndPin:
    async def test_no_op_when_not_proxied(self):
        with _patch_settings():
            out = await validate_and_pin_proxy_target(
                "skill", {"is_proxied": False, "proxy_target_url": "https://x/"}
            )
        assert out == {}

    async def test_no_op_when_federated(self):
        with _patch_settings():
            out = await validate_and_pin_proxy_target(
                "skill",
                {
                    "is_proxied": True,
                    "proxy_target_url": "https://x/",
                    "sync_metadata": {"is_federated": True},
                },
            )
        assert out == {}

    async def test_pins_resolved_ips_for_live_target(self):
        with _patch_settings():
            with patch("socket.getaddrinfo", return_value=_addrinfo("93.184.216.34")):
                out = await validate_and_pin_proxy_target(
                    "skill", {"is_proxied": True, "proxy_target_url": "https://ok.example/"}
                )
        assert out == {
            "proxy_resolved_ips": ["93.184.216.34"],
            "proxy_target_host": "ok.example",
        }

    async def test_raises_on_denied_target(self):
        with _patch_settings():
            with patch("socket.getaddrinfo", return_value=_addrinfo("169.254.169.254")):
                with pytest.raises(EgressPolicyError):
                    await validate_and_pin_proxy_target(
                        "skill", {"is_proxied": True, "proxy_target_url": "https://evil.example/"}
                    )
