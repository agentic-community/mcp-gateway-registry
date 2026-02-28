"""Unit tests for URL scheme validation."""

import pytest
from cli.mcp_utils import _validate_url_scheme


class TestUrlValidation:
    """Tests for _validate_url_scheme function."""

    def test_allows_http(self):
        """Test that http:// URLs are allowed."""
        _validate_url_scheme("http://example.com")
        _validate_url_scheme("http://localhost:8080")
        _validate_url_scheme("http://192.168.1.1/api")
        # Should not raise

    def test_allows_https(self):
        """Test that https:// URLs are allowed."""
        _validate_url_scheme("https://example.com")
        _validate_url_scheme("https://api.github.com")
        _validate_url_scheme("https://secure.example.com:443/path")
        # Should not raise

    def test_blocks_file_scheme(self):
        """Test that file:// URLs are blocked."""
        with pytest.raises(ValueError, match="Invalid URL scheme 'file'"):
            _validate_url_scheme("file:///etc/passwd")

    def test_blocks_ftp_scheme(self):
        """Test that ftp:// URLs are blocked."""
        with pytest.raises(ValueError, match="Invalid URL scheme 'ftp'"):
            _validate_url_scheme("ftp://example.com")

    def test_blocks_gopher_scheme(self):
        """Test that gopher:// URLs are blocked."""
        with pytest.raises(ValueError, match="Invalid URL scheme 'gopher'"):
            _validate_url_scheme("gopher://example.com")

    def test_blocks_javascript_scheme(self):
        """Test that javascript: URLs are blocked."""
        with pytest.raises(ValueError, match="Invalid URL scheme 'javascript'"):
            _validate_url_scheme("javascript:alert(1)")

    def test_blocks_data_scheme(self):
        """Test that data: URLs are blocked."""
        with pytest.raises(ValueError, match="Invalid URL scheme 'data'"):
            _validate_url_scheme("data:text/html,<script>alert(1)</script>")

    def test_error_message_format(self):
        """Test that error message includes scheme and allowed schemes."""
        try:
            _validate_url_scheme("ftp://example.com")
            pytest.fail("Should have raised ValueError")
        except ValueError as e:
            error_msg = str(e)
            assert "ftp" in error_msg
            assert "http" in error_msg
            assert "https" in error_msg
