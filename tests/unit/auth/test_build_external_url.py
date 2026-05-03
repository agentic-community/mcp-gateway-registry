"""Tests for _build_external_url helper in auth routes.

Verifies that OAuth2 redirect URIs include ROOT_PATH when path-based
routing is enabled (issue #500).
"""

from unittest.mock import MagicMock

from registry.auth import routes
from registry.auth.routes import _build_external_url


def _make_request(
    host: str = "example.com",
    scheme: str = "https",
    cloudfront_proto: str = "",
    x_forwarded_proto: str = "",
) -> MagicMock:
    """Create a mock Request with configurable headers and scheme."""
    request = MagicMock()
    header_dict = {
        "host": host,
        "x-cloudfront-forwarded-proto": cloudfront_proto,
        "x-forwarded-proto": x_forwarded_proto,
    }
    request.headers = MagicMock()
    request.headers.get = lambda key, default="": header_dict.get(key, default)
    request.url = MagicMock()
    request.url.scheme = scheme
    return request


class TestBuildExternalUrlWithoutRootPath:
    """Tests when ROOT_PATH is empty (subdomain mode, Docker, ECS)."""

    def setup_method(self):
        self._original = routes._ROOT_PATH
        routes._ROOT_PATH = ""

    def teardown_method(self):
        routes._ROOT_PATH = self._original

    def test_basic_url(self):
        request = _make_request()
        result = _build_external_url(request, "/logout")
        assert result == "https://example.com/logout"

    def test_root_path_url(self):
        request = _make_request()
        result = _build_external_url(request, "/")
        assert result == "https://example.com/"

    def test_empty_path(self):
        request = _make_request()
        result = _build_external_url(request)
        assert result == "https://example.com"

    def test_http_scheme(self):
        request = _make_request(scheme="http")
        result = _build_external_url(request, "/logout")
        assert result == "http://example.com/logout"


class TestBuildExternalUrlWithRootPath:
    """Tests when ROOT_PATH is set (path-based routing, EKS)."""

    def setup_method(self):
        self._original = routes._ROOT_PATH
        routes._ROOT_PATH = "/registry"

    def teardown_method(self):
        routes._ROOT_PATH = self._original

    def test_logout_includes_root_path(self):
        request = _make_request()
        result = _build_external_url(request, "/logout")
        assert result == "https://example.com/registry/logout"

    def test_root_includes_root_path(self):
        request = _make_request()
        result = _build_external_url(request, "/")
        assert result == "https://example.com/registry/"

    def test_empty_path_with_root(self):
        request = _make_request()
        result = _build_external_url(request)
        assert result == "https://example.com/registry"

    def test_deep_root_path(self):
        routes._ROOT_PATH = "/app/registry"
        request = _make_request()
        result = _build_external_url(request, "/logout")
        assert result == "https://example.com/app/registry/logout"


class TestBuildExternalUrlSchemeDetection:
    """Tests for HTTPS detection from proxy headers."""

    def setup_method(self):
        self._original = routes._ROOT_PATH
        routes._ROOT_PATH = ""

    def teardown_method(self):
        routes._ROOT_PATH = self._original

    def test_cloudfront_header_forces_https(self):
        request = _make_request(scheme="http", cloudfront_proto="https")
        result = _build_external_url(request, "/logout")
        assert result.startswith("https://")

    def test_x_forwarded_proto_forces_https(self):
        request = _make_request(scheme="http", x_forwarded_proto="https")
        result = _build_external_url(request, "/logout")
        assert result.startswith("https://")

    def test_request_scheme_https(self):
        request = _make_request(scheme="https")
        result = _build_external_url(request, "/logout")
        assert result.startswith("https://")

    def test_all_http_stays_http(self):
        request = _make_request(scheme="http")
        result = _build_external_url(request, "/logout")
        assert result.startswith("http://")


class TestBuildExternalUrlLocalhostHandling:
    """Tests for localhost special case (adds port if missing)."""

    def setup_method(self):
        self._original = routes._ROOT_PATH
        routes._ROOT_PATH = ""

    def teardown_method(self):
        routes._ROOT_PATH = self._original

    def test_localhost_without_port_gets_default(self):
        request = _make_request(host="localhost", scheme="http")
        result = _build_external_url(request, "/logout")
        assert result == "http://localhost:7860/logout"

    def test_localhost_with_port_preserved(self):
        request = _make_request(host="localhost:3000", scheme="http")
        result = _build_external_url(request, "/logout")
        assert result == "http://localhost:3000/logout"

    def test_localhost_with_root_path(self):
        routes._ROOT_PATH = "/registry"
        request = _make_request(host="localhost", scheme="http")
        result = _build_external_url(request, "/logout")
        assert result == "http://localhost:7860/registry/logout"


class TestBuildExternalUrlPathNormalization:
    """Tests that path argument is normalized correctly."""

    def setup_method(self):
        self._original = routes._ROOT_PATH
        routes._ROOT_PATH = ""

    def teardown_method(self):
        routes._ROOT_PATH = self._original

    def test_path_without_leading_slash_gets_one(self):
        request = _make_request()
        result = _build_external_url(request, "logout")
        assert result == "https://example.com/logout"

    def test_path_with_leading_slash_preserved(self):
        request = _make_request()
        result = _build_external_url(request, "/logout")
        assert result == "https://example.com/logout"
