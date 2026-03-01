"""Centralized endpoint URL resolution utilities.

This module provides functions for resolving MCP and SSE endpoint URLs
from server configuration, supporting custom endpoints while maintaining
backward compatibility with the default /mcp and /sse suffixes.
"""

import logging
from typing import (
    Any,
)

logger = logging.getLogger(__name__)


def _url_contains_transport_path(url: str) -> bool:
    """Check if URL already contains a transport-specific path.

    Args:
        url: The URL to check.

    Returns:
        True if URL contains /mcp or /sse path segments.
    """
    return url.endswith("/mcp") or url.endswith("/sse") or "/mcp/" in url or "/sse/" in url


def get_endpoint_url(
    proxy_pass_url: str,
    transport_type: str = "streamable-http",
    mcp_endpoint: str | None = None,
    sse_endpoint: str | None = None,
) -> str:
    """Resolve the actual endpoint URL for health checks and client connections.

    This function follows a priority-based resolution:
    1. If mcp_endpoint/sse_endpoint is explicitly set, use it directly
    2. If proxy_pass_url already contains /mcp or /sse, use as-is
    3. Otherwise, append the default suffix (/mcp or /sse)

    Args:
        proxy_pass_url: The base proxy URL for the server.
        transport_type: The transport type - "streamable-http" or "sse".
        mcp_endpoint: Optional explicit endpoint URL for streamable-http.
        sse_endpoint: Optional explicit endpoint URL for SSE.

    Returns:
        The resolved endpoint URL.
    """
    # Only strip trailing slash if URL doesn't already contain transport path
    # Some servers like Hydrata require the trailing slash
    if _url_contains_transport_path(proxy_pass_url):
        base_url = proxy_pass_url
    else:
        base_url = proxy_pass_url.rstrip("/")

    if transport_type == "sse":
        # Priority 1: Explicit sse_endpoint
        if sse_endpoint:
            logger.debug(f"Using explicit sse_endpoint: {sse_endpoint}")
            return sse_endpoint

        # Priority 2: URL already contains transport path
        if base_url.endswith("/sse") or "/sse/" in base_url:
            logger.debug(f"URL already contains /sse: {base_url}")
            return base_url

        # Priority 3: Append default suffix
        endpoint = f"{base_url}/sse"
        logger.debug(f"Appending /sse suffix: {endpoint}")
        return endpoint

    else:
        # streamable-http (default)
        # Priority 1: Explicit mcp_endpoint
        if mcp_endpoint:
            logger.debug(f"Using explicit mcp_endpoint: {mcp_endpoint}")
            return mcp_endpoint

        # Priority 2: URL already contains transport path
        if _url_contains_transport_path(base_url):
            logger.debug(f"URL already contains transport path: {base_url}")
            return base_url

        # Priority 3: Append default suffix
        endpoint = f"{base_url}/mcp"
        logger.debug(f"Appending /mcp suffix: {endpoint}")
        return endpoint


def get_endpoint_url_from_server_info(
    server_info: dict[str, Any],
    transport_type: str = "streamable-http",
) -> str:
    """Resolve endpoint URL from a server_info dictionary.

    Convenience wrapper around get_endpoint_url that extracts
    the relevant fields from a server_info dict.

    Args:
        server_info: Dictionary containing server configuration.
        transport_type: The transport type - "streamable-http" or "sse".

    Returns:
        The resolved endpoint URL.

    Raises:
        ValueError: If proxy_pass_url is missing from server_info.
    """
    proxy_pass_url = server_info.get("proxy_pass_url")
    if not proxy_pass_url:
        raise ValueError("server_info must contain proxy_pass_url")

    return get_endpoint_url(
        proxy_pass_url=proxy_pass_url,
        transport_type=transport_type,
        mcp_endpoint=server_info.get("mcp_endpoint"),
        sse_endpoint=server_info.get("sse_endpoint"),
    )
