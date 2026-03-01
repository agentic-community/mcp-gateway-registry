"""
Unit tests for registry.core.endpoint_utils module.

This module tests the endpoint URL resolution utilities, including
custom endpoint support and backward compatibility with default /mcp and /sse suffixes.
"""

import pytest

from registry.core.endpoint_utils import (
    get_endpoint_url,
    get_endpoint_url_from_server_info,
    _url_contains_transport_path,
)


# =============================================================================
# TEST CLASS: URL Contains Transport Path Detection
# =============================================================================


@pytest.mark.unit
@pytest.mark.core
class TestUrlContainsTransportPath:
    """Test _url_contains_transport_path helper function."""

    def test_url_ending_with_mcp(self) -> None:
        """URL ending with /mcp should be detected."""
        assert _url_contains_transport_path("http://server.com/mcp") is True

    def test_url_ending_with_sse(self) -> None:
        """URL ending with /sse should be detected."""
        assert _url_contains_transport_path("http://server.com/sse") is True

    def test_url_with_mcp_in_path(self) -> None:
        """URL with /mcp/ in path should be detected."""
        assert _url_contains_transport_path("http://server.com/mcp/v1") is True

    def test_url_with_sse_in_path(self) -> None:
        """URL with /sse/ in path should be detected."""
        assert _url_contains_transport_path("http://server.com/sse/v1") is True

    def test_url_without_transport_path(self) -> None:
        """URL without transport path should not be detected."""
        assert _url_contains_transport_path("http://server.com/api") is False

    def test_url_with_custom_path(self) -> None:
        """URL with custom path should not be detected."""
        assert _url_contains_transport_path("http://server.com/use-case") is False


# =============================================================================
# TEST CLASS: get_endpoint_url for Streamable HTTP
# =============================================================================


@pytest.mark.unit
@pytest.mark.core
class TestGetEndpointUrlStreamableHttp:
    """Test get_endpoint_url function for streamable-http transport."""

    def test_explicit_mcp_endpoint_takes_priority(self) -> None:
        """Explicit mcp_endpoint should be used when provided."""
        result = get_endpoint_url(
            proxy_pass_url="http://server.com/api",
            transport_type="streamable-http",
            mcp_endpoint="http://custom.server.com/use-case",
        )
        assert result == "http://custom.server.com/use-case"

    def test_explicit_mcp_endpoint_preserves_trailing_slash(self) -> None:
        """Explicit mcp_endpoint should preserve trailing slash (changed behavior)."""
        result = get_endpoint_url(
            proxy_pass_url="http://server.com/api",
            transport_type="streamable-http",
            mcp_endpoint="http://custom.server.com/use-case/",
        )
        # Changed: Now preserves trailing slash for servers that require it
        assert result == "http://custom.server.com/use-case/"

    def test_url_with_mcp_used_as_is(self) -> None:
        """URL already containing /mcp should be used as-is."""
        result = get_endpoint_url(
            proxy_pass_url="http://server.com/mcp",
            transport_type="streamable-http",
        )
        assert result == "http://server.com/mcp"

    def test_url_with_mcp_in_path_used_as_is(self) -> None:
        """URL with /mcp/ in path should be used as-is."""
        result = get_endpoint_url(
            proxy_pass_url="http://server.com/mcp/v1",
            transport_type="streamable-http",
        )
        assert result == "http://server.com/mcp/v1"

    def test_plain_url_gets_mcp_appended(self) -> None:
        """Plain URL without transport path should get /mcp appended."""
        result = get_endpoint_url(
            proxy_pass_url="http://server.com/api",
            transport_type="streamable-http",
        )
        assert result == "http://server.com/api/mcp"

    def test_url_with_trailing_slash_handled(self) -> None:
        """URL with trailing slash should be handled correctly."""
        result = get_endpoint_url(
            proxy_pass_url="http://server.com/api/",
            transport_type="streamable-http",
        )
        assert result == "http://server.com/api/mcp"

    def test_default_transport_is_streamable_http(self) -> None:
        """Default transport type should be streamable-http."""
        result = get_endpoint_url(
            proxy_pass_url="http://server.com/api",
        )
        assert result == "http://server.com/api/mcp"


# =============================================================================
# TEST CLASS: get_endpoint_url for SSE
# =============================================================================


@pytest.mark.unit
@pytest.mark.core
class TestGetEndpointUrlSse:
    """Test get_endpoint_url function for SSE transport."""

    def test_explicit_sse_endpoint_takes_priority(self) -> None:
        """Explicit sse_endpoint should be used when provided."""
        result = get_endpoint_url(
            proxy_pass_url="http://server.com/api",
            transport_type="sse",
            sse_endpoint="http://custom.server.com/events",
        )
        assert result == "http://custom.server.com/events"

    def test_explicit_sse_endpoint_preserves_trailing_slash(self) -> None:
        """Explicit sse_endpoint should preserve trailing slash (changed behavior)."""
        result = get_endpoint_url(
            proxy_pass_url="http://server.com/api",
            transport_type="sse",
            sse_endpoint="http://custom.server.com/events/",
        )
        # Changed: Now preserves trailing slash for servers that require it
        assert result == "http://custom.server.com/events/"

    def test_url_with_sse_used_as_is(self) -> None:
        """URL already ending with /sse should be used as-is."""
        result = get_endpoint_url(
            proxy_pass_url="http://server.com/sse",
            transport_type="sse",
        )
        assert result == "http://server.com/sse"

    def test_url_with_sse_in_path_used_as_is(self) -> None:
        """URL with /sse/ in path should be used as-is."""
        result = get_endpoint_url(
            proxy_pass_url="http://server.com/sse/v1",
            transport_type="sse",
        )
        assert result == "http://server.com/sse/v1"

    def test_plain_url_gets_sse_appended(self) -> None:
        """Plain URL without transport path should get /sse appended."""
        result = get_endpoint_url(
            proxy_pass_url="http://server.com/api",
            transport_type="sse",
        )
        assert result == "http://server.com/api/sse"

    def test_url_with_trailing_slash_handled(self) -> None:
        """URL with trailing slash should be handled correctly."""
        result = get_endpoint_url(
            proxy_pass_url="http://server.com/api/",
            transport_type="sse",
        )
        assert result == "http://server.com/api/sse"


# =============================================================================
# TEST CLASS: get_endpoint_url_from_server_info
# =============================================================================


@pytest.mark.unit
@pytest.mark.core
class TestGetEndpointUrlFromServerInfo:
    """Test get_endpoint_url_from_server_info function."""

    def test_extracts_proxy_pass_url(self) -> None:
        """Should extract proxy_pass_url from server_info."""
        server_info = {"proxy_pass_url": "http://server.com/api"}
        result = get_endpoint_url_from_server_info(server_info)
        assert result == "http://server.com/api/mcp"

    def test_uses_mcp_endpoint_from_server_info(self) -> None:
        """Should use mcp_endpoint when present in server_info."""
        server_info = {
            "proxy_pass_url": "http://server.com/api",
            "mcp_endpoint": "http://custom.server.com/use-case",
        }
        result = get_endpoint_url_from_server_info(server_info, transport_type="streamable-http")
        assert result == "http://custom.server.com/use-case"

    def test_uses_sse_endpoint_from_server_info(self) -> None:
        """Should use sse_endpoint when present in server_info."""
        server_info = {
            "proxy_pass_url": "http://server.com/api",
            "sse_endpoint": "http://custom.server.com/events",
        }
        result = get_endpoint_url_from_server_info(server_info, transport_type="sse")
        assert result == "http://custom.server.com/events"

    def test_raises_on_missing_proxy_pass_url(self) -> None:
        """Should raise ValueError if proxy_pass_url is missing."""
        server_info = {"server_name": "test"}
        with pytest.raises(ValueError, match="proxy_pass_url"):
            get_endpoint_url_from_server_info(server_info)

    def test_handles_none_endpoint_fields(self) -> None:
        """Should handle None values for endpoint fields."""
        server_info = {
            "proxy_pass_url": "http://server.com/api",
            "mcp_endpoint": None,
            "sse_endpoint": None,
        }
        result = get_endpoint_url_from_server_info(server_info)
        assert result == "http://server.com/api/mcp"


# =============================================================================
# TEST CLASS: Real-World Scenarios
# =============================================================================


@pytest.mark.unit
@pytest.mark.core
class TestRealWorldScenarios:
    """Test real-world scenarios mentioned in the issue."""

    def test_custom_use_case_endpoint(self) -> None:
        """Test custom endpoint like mcp.myorg.com/use-case."""
        server_info = {
            "proxy_pass_url": "http://mcp.myorg.com/use-case",
            "mcp_endpoint": "http://mcp.myorg.com/use-case",
        }
        result = get_endpoint_url_from_server_info(server_info)
        assert result == "http://mcp.myorg.com/use-case"

    def test_multiple_servers_same_host(self) -> None:
        """Test multiple MCP servers on same host with different paths."""
        server1 = {
            "proxy_pass_url": "http://myorg.com/mcp-1",
            "mcp_endpoint": "http://myorg.com/mcp-1",
        }
        server2 = {
            "proxy_pass_url": "http://myorg.com/mcp-2",
            "mcp_endpoint": "http://myorg.com/mcp-2",
        }
        result1 = get_endpoint_url_from_server_info(server1)
        result2 = get_endpoint_url_from_server_info(server2)
        assert result1 == "http://myorg.com/mcp-1"
        assert result2 == "http://myorg.com/mcp-2"

    def test_backward_compatibility_no_explicit_endpoint(self) -> None:
        """Test backward compatibility when no explicit endpoint is set."""
        server_info = {
            "proxy_pass_url": "http://server.com/api",
        }
        result = get_endpoint_url_from_server_info(server_info)
        assert result == "http://server.com/api/mcp"

    def test_backward_compatibility_url_already_has_mcp(self) -> None:
        """Test backward compatibility when URL already has /mcp."""
        server_info = {
            "proxy_pass_url": "http://server.com/api/mcp",
        }
        result = get_endpoint_url_from_server_info(server_info)
        assert result == "http://server.com/api/mcp"


# =============================================================================
# TEST CLASS: Trailing Slash Preservation (Issue #539 Fix)
# =============================================================================


@pytest.mark.unit
@pytest.mark.core
class TestTrailingSlashPreservation:
    """Test trailing slash preservation for servers that require it (e.g., Hydrata)."""

    def test_hydrata_url_with_trailing_slash(self) -> None:
        """Hydrata URL with /mcp/ should preserve trailing slash."""
        result = get_endpoint_url(
            proxy_pass_url="https://hydrata.com/mcp/",
            transport_type="streamable-http",
        )
        # Critical fix: Preserve trailing slash to avoid 301 redirect → 405 error
        assert result == "https://hydrata.com/mcp/"

    def test_url_with_mcp_and_trailing_slash(self) -> None:
        """URL ending with /mcp/ should preserve trailing slash."""
        result = get_endpoint_url(
            proxy_pass_url="https://example.com/mcp/",
            transport_type="streamable-http",
        )
        assert result == "https://example.com/mcp/"

    def test_url_with_sse_and_trailing_slash(self) -> None:
        """URL ending with /sse/ should preserve trailing slash."""
        result = get_endpoint_url(
            proxy_pass_url="https://example.com/sse/",
            transport_type="sse",
        )
        assert result == "https://example.com/sse/"

    def test_url_with_mcp_in_middle_and_trailing_slash(self) -> None:
        """URL with /mcp/ in middle and trailing slash should preserve it."""
        result = get_endpoint_url(
            proxy_pass_url="https://example.com/mcp/v1/",
            transport_type="streamable-http",
        )
        assert result == "https://example.com/mcp/v1/"

    def test_url_without_transport_still_strips_slash(self) -> None:
        """URL without transport path should still strip trailing slash before appending."""
        result = get_endpoint_url(
            proxy_pass_url="http://server.com/api/",
            transport_type="streamable-http",
        )
        assert result == "http://server.com/api/mcp"


# =============================================================================
# TEST CLASS: Production URL Patterns (All 4 Patterns)
# =============================================================================


@pytest.mark.unit
@pytest.mark.core
class TestProductionUrlPatterns:
    """Test all 4 URL patterns found in production database."""

    def test_pattern1_no_transport_no_slash(self) -> None:
        """Pattern 1: No transport, no slash (1 server in prod)."""
        # Example: http://localhost:3000
        result = get_endpoint_url("http://localhost:3000")
        assert result == "http://localhost:3000/mcp"

    def test_pattern2_no_transport_with_slash(self) -> None:
        """Pattern 2: No transport, has slash (7 servers in prod)."""
        # Examples: http://mcpgw-server:8003/, http://currenttime-server:8000/
        result1 = get_endpoint_url("http://mcpgw-server:8003/")
        result2 = get_endpoint_url("http://currenttime-server:8000/")
        assert result1 == "http://mcpgw-server:8003/mcp"
        assert result2 == "http://currenttime-server:8000/mcp"

    def test_pattern3_has_transport_no_slash(self) -> None:
        """Pattern 3: Has transport, no slash (10 servers in prod)."""
        # Examples: https://docs.mcp.cloudflare.com/mcp, https://mcp.context7.com/mcp
        result1 = get_endpoint_url("https://docs.mcp.cloudflare.com/mcp")
        result2 = get_endpoint_url("https://mcp.context7.com/mcp")
        result3 = get_endpoint_url("https://mcp.cloudflare.com/mcp")
        assert result1 == "https://docs.mcp.cloudflare.com/mcp"
        assert result2 == "https://mcp.context7.com/mcp"
        assert result3 == "https://mcp.cloudflare.com/mcp"

    def test_pattern4_has_transport_with_slash(self) -> None:
        """Pattern 4: Has transport, has slash (1 server in prod - Hydrata)."""
        # Example: https://hydrata.com/mcp/
        result = get_endpoint_url("https://hydrata.com/mcp/")
        # This is the ONLY pattern with changed behavior (fix for issue #539)
        assert result == "https://hydrata.com/mcp/"

    def test_all_patterns_except_pattern4_unchanged(self) -> None:
        """Verify patterns 1-3 have identical behavior to old code."""
        # Pattern 1
        assert get_endpoint_url("http://localhost:3000") == "http://localhost:3000/mcp"

        # Pattern 2
        assert get_endpoint_url("http://mcpgw-server:8003/") == "http://mcpgw-server:8003/mcp"

        # Pattern 3
        assert get_endpoint_url("https://docs.mcp.cloudflare.com/mcp") == "https://docs.mcp.cloudflare.com/mcp"

        # Only Pattern 4 has different behavior (this is the fix)
        assert get_endpoint_url("https://hydrata.com/mcp/") == "https://hydrata.com/mcp/"


# =============================================================================
# TEST CLASS: Regression Prevention
# =============================================================================


@pytest.mark.unit
@pytest.mark.core
class TestRegressionPrevention:
    """Tests to prevent regression of the Hydrata 301/405 bug."""

    def test_prevents_301_redirect_issue(self) -> None:
        """Ensure URLs with trailing slash don't cause 301 redirects.

        Background: When POSTing to https://hydrata.com/mcp (no slash),
        Hydrata redirects with 301 to https://hydrata.com/mcp/ (with slash).
        HTTP clients change POST to GET on 301 redirects, causing 405 errors.

        Fix: Preserve trailing slash so we POST directly to the correct URL.
        """
        url_with_slash = "https://hydrata.com/mcp/"
        result = get_endpoint_url(url_with_slash)

        # Should return URL with slash to avoid redirect
        assert result == "https://hydrata.com/mcp/"
        assert not result.endswith("/mcp")  # Should NOT strip the slash

    def test_explicit_endpoint_with_slash_preserved(self) -> None:
        """Explicit endpoints with trailing slash should be preserved."""
        result = get_endpoint_url(
            proxy_pass_url="http://localhost:3000",
            mcp_endpoint="https://hydrata.com/mcp/",
        )
        assert result == "https://hydrata.com/mcp/"

    def test_cloudflare_docs_url_unchanged(self) -> None:
        """Cloudflare docs URL without slash should remain unchanged."""
        result = get_endpoint_url("https://docs.mcp.cloudflare.com/mcp")
        assert result == "https://docs.mcp.cloudflare.com/mcp"

    def test_context7_url_unchanged(self) -> None:
        """Context7 URL without slash should remain unchanged."""
        result = get_endpoint_url("https://mcp.context7.com/mcp")
        assert result == "https://mcp.context7.com/mcp"
