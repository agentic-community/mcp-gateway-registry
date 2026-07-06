"""
Unit tests for registry.utils.request_utils.

Validates IP extraction and sanitization from proxied requests.
"""

from unittest.mock import MagicMock

from registry.utils.request_utils import get_client_ip


def _make_request(headers=None, client_host="127.0.0.1", client=None):
    """Create a minimal mock FastAPI Request."""
    request = MagicMock()
    request.headers = headers or {}
    if client is False:
        request.client = None
    else:
        request.client = MagicMock()
        request.client.host = client_host
    return request


class TestGetClientIp:
    """Tests for get_client_ip utility function."""

    def test_returns_first_ip_from_forwarded_for(self):
        """Should return the first IP from X-Forwarded-For header."""
        request = _make_request(
            headers={"X-Forwarded-For": "33.111.22.33, 10.0.0.1"},
        )
        assert get_client_ip(request) == "33.111.22.33"

    def test_returns_single_forwarded_for_ip(self):
        """Should handle a single IP in X-Forwarded-For."""
        request = _make_request(
            headers={"X-Forwarded-For": "192.168.1.1"},
        )
        assert get_client_ip(request) == "192.168.1.1"

    def test_falls_back_to_client_host_when_no_header(self):
        """Should use request.client.host when X-Forwarded-For is absent."""
        request = _make_request(client_host="10.0.0.5")
        assert get_client_ip(request) == "10.0.0.5"

    def test_returns_unknown_when_no_client(self):
        """Should return 'unknown' when both header and client are missing."""
        request = _make_request(client=False)
        assert get_client_ip(request) == "unknown"

    def test_rejects_malformed_forwarded_for(self):
        """Should ignore non-IP values in X-Forwarded-For and fall back."""
        request = _make_request(
            headers={"X-Forwarded-For": "<script>alert(1)</script>"},
            client_host="10.0.0.1",
        )
        assert get_client_ip(request) == "10.0.0.1"

    def test_rejects_arbitrary_string_in_header(self):
        """Should ignore random strings in X-Forwarded-For."""
        request = _make_request(
            headers={"X-Forwarded-For": "not-an-ip, 10.1.2.3"},
            client_host="10.0.0.1",
        )
        assert get_client_ip(request) == "10.0.0.1"

    def test_handles_ipv6_address(self):
        """Should accept valid IPv6 addresses in X-Forwarded-For."""
        request = _make_request(
            headers={"X-Forwarded-For": "2001:db8::1, 10.1.2.3"},
        )
        assert get_client_ip(request) == "2001:db8::1"

    def test_handles_whitespace_around_ip(self):
        """Should strip whitespace from the extracted IP."""
        request = _make_request(
            headers={"X-Forwarded-For": "  33.111.22.33 , 10.0.0.1"},
        )
        assert get_client_ip(request) == "33.111.22.33"

    def test_empty_forwarded_for_falls_back(self):
        """Should fall back to client.host when header is empty string."""
        request = _make_request(
            headers={"X-Forwarded-For": ""},
            client_host="10.0.0.1",
        )
        assert get_client_ip(request) == "10.0.0.1"
