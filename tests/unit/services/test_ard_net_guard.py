"""Unit tests for the ARD ingestion SSRF guard (issue #1296)."""

from unittest.mock import patch

import pytest

from registry.services import ard_net_guard as g
from registry.services.ard_search_service import ArdValidationError


def _resolve_to(ip: str):
    """Return a getaddrinfo stub that resolves any host to ``ip``."""
    return lambda host, port, **kw: [(2, 1, 6, "", (ip, port))]


class TestAssertFetchable:
    def test_rejects_non_https(self):
        with pytest.raises(ArdValidationError):
            g.assert_fetchable("http://acme.com/.well-known/ai-catalog.json")

    def test_rejects_missing_host(self):
        with pytest.raises(ArdValidationError):
            g.assert_fetchable("https:///nohost")

    def test_allows_public_ip(self):
        with patch.object(g.socket, "getaddrinfo", _resolve_to("93.184.216.34")):
            assert g.assert_fetchable("https://acme.com/x") == "https://acme.com/x"

    @pytest.mark.parametrize(
        "ip",
        ["127.0.0.1", "10.1.2.3", "192.168.1.5", "172.16.0.9", "169.254.169.254",
         "0.0.0.0", "::ffff:10.0.0.1"],
    )
    def test_blocks_private_and_metadata(self, ip):
        family = 10 if ":" in ip else 2  # AF_INET6 vs AF_INET
        stub = lambda host, port, **kw: [(family, 1, 6, "", (ip, port))]
        with patch.object(g.socket, "getaddrinfo", stub):
            with pytest.raises(ArdValidationError):
                g.assert_fetchable("https://evil.example/x")

    def test_same_domain_allows_subdomain(self):
        with patch.object(g.socket, "getaddrinfo", _resolve_to("93.184.216.34")):
            assert g.assert_fetchable("https://sub.acme.com/x", allowed_domain="acme.com")

    def test_same_domain_blocks_other_domain(self):
        with patch.object(g.socket, "getaddrinfo", _resolve_to("93.184.216.34")):
            with pytest.raises(ArdValidationError):
                g.assert_fetchable("https://evil.com/x", allowed_domain="acme.com")
