"""
Unit tests for registry.utils.request_utils.

Validates IP extraction and sanitization from proxied requests.
"""

from unittest.mock import MagicMock

from registry.utils.request_utils import (
    get_client_ip,
    is_sensitive_header,
    redact_sensitive_headers,
)


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


class TestRedactSensitiveHeaders:
    """Tests for header redaction used by diagnostic header dumps."""

    def test_redacts_authorization_and_cookie(self):
        """Authorization and Cookie values are never logged."""
        redacted = redact_sensitive_headers(
            {"Authorization": "Bearer abc", "Cookie": "session=xyz", "Accept": "*/*"}
        )
        assert redacted["Authorization"] == "[REDACTED]"
        assert redacted["Cookie"] == "[REDACTED]"
        assert redacted["Accept"] == "*/*"

    def test_redacts_the_skill_parse_credential_header(self):
        """The X-Auth-Credential header used by parse-skill-md is redacted."""
        redacted = redact_sensitive_headers({"X-Auth-Credential": "super-secret-token"})
        assert redacted["X-Auth-Credential"] == "[REDACTED]"
        assert "super-secret-token" not in str(redacted)

    def test_redacts_variant_credential_headers(self):
        """Variant credential-bearing header names are redacted (fail closed)."""
        headers = {
            "X-Api-Key": "k1",
            "X-Access-Token": "t1",
            "X-Client-Secret": "s1",
            "Proxy-Authorization": "p1",
            "X-User-Password": "pw1",
        }
        redacted = redact_sensitive_headers(headers)
        for name in headers:
            assert redacted[name] == "[REDACTED]", name

    def test_leaves_nonsensitive_headers_intact(self):
        """Ordinary headers pass through unmodified."""
        headers = {
            "User-Agent": "curl/8",
            "Accept": "application/json",
            "X-Forwarded-For": "10.0.0.1",
            "Content-Length": "42",
        }
        assert redact_sensitive_headers(headers) == headers

    def test_is_sensitive_header_case_insensitive(self):
        """Sensitivity detection ignores case."""
        assert is_sensitive_header("AUTHORIZATION")
        assert is_sensitive_header("x-auth-credential")
        assert is_sensitive_header("X-Auth-Credential")
        assert not is_sensitive_header("User-Agent")
