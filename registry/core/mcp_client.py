"""
MCP Client Service

Handles connections to MCP servers and tool list retrieval.
Copied directly from main_old.py working implementation.
"""

import asyncio
import logging
import re
from typing import (
    Any,
    TypedDict,
)

# MCP Client imports
from mcp import ClientSession
from mcp.client.sse import sse_client
from mcp.client.streamable_http import streamablehttp_client

from ..common.log_redaction import redact_url

logger = logging.getLogger(__name__)


def _assert_mcp_url_fetchable(
    url: str,
) -> bool:
    """Fail-closed SSRF check before an MCP SDK connection is opened.

    The MCP SDK transports (``streamablehttp_client`` / ``sse_client``) build
    their own httpx client, so they cannot use the pinned guarded transport that
    protects the registry's direct fetches. This helper re-validates the target
    against the proxy profile (public-only unless the operator allowlisted the
    internal target) immediately before a connection is opened, so a
    ``proxy_pass_url`` that resolves to a private/metadata address is rejected
    *before* any decrypted credential is built or sent. It resolves DNS at call
    time, closing the window between registration-time validation and this
    fetch. Any validation error results in refusal (returns ``False``).

    Args:
        url: The MCP endpoint / base URL about to be connected to.

    Returns:
        True if the URL passed validation and may be connected to, else False.
    """
    from ..exceptions import UrlValidationError
    from ..utils.url_guard import PROXY_PROFILE, validate_url

    try:
        validate_url(url, profile=PROXY_PROFILE)
        return True
    except UrlValidationError as e:
        logger.warning("MCP connection blocked by SSRF guard for %s: %s", redact_url(url), e)
        return False
    except Exception as e:  # pragma: no cover - defensive, fail closed
        logger.warning("MCP connection blocked (validation error) for %s: %s", redact_url(url), e)
        return False


class MCPServerInfo(TypedDict, total=False):
    """Server info returned from MCP initialize response."""

    name: str
    version: str


class MCPConnectionResult(TypedDict, total=False):
    """Result of connecting to an MCP server."""

    tools: list[dict]
    server_info: MCPServerInfo


def normalize_sse_endpoint_url(endpoint_url: str) -> str:
    """
    Normalize SSE endpoint URLs by removing mount path prefixes.

    For example:
    - Input: "/currenttime/messages/?session_id=123"
    - Output: "/messages/?session_id=123"

    Args:
        endpoint_url: The endpoint URL from the SSE event data

    Returns:
        The normalized URL with mount path stripped
    """
    if not endpoint_url:
        return endpoint_url

    # Pattern to match mount paths like /currenttime/, /mcpgw/, etc.
    # We look for paths that start with /word/ followed by messages/
    mount_path_pattern = r"^(/[^/]+)(/messages/.*)"

    match = re.match(mount_path_pattern, endpoint_url)
    if match:
        mount_path = match.group(1)  # e.g., "/currenttime"
        rest_of_url = match.group(2)  # e.g., "/messages/?session_id=123"

        logger.debug(
            f"Stripping mount path '{mount_path}' from endpoint URL: {redact_url(endpoint_url)}"
        )
        return rest_of_url

    # If no mount path pattern detected, return as-is
    return endpoint_url


import httpx


def _build_headers_for_server(
    server_info: dict = None,
    destination_url: str | None = None,
) -> dict[str, str]:
    """
    Build HTTP headers for server requests by merging server-specific headers.

    Args:
        server_info: Server configuration dictionary
        destination_url: The URL the resulting headers will be sent to. When any
            encrypted secret (custom headers or an auth credential) would be
            attached, this destination is re-validated through the shared SSRF
            guard before decryption. If it is missing or fails validation, the
            decrypted secrets are NOT attached (fail closed) so a mutated /
            unvalidated backend can never receive them. Non-secret headers
            (Accept/Content-Type and plaintext server headers) are unaffected.

    Returns:
        Headers dictionary with server-specific headers
    """
    # Start with default MCP headers (required by some servers like Cloudflare)
    headers = {"Accept": "application/json, text/event-stream", "Content-Type": "application/json"}

    # Merge server-specific headers if present
    logger.info(
        f"[AUTH DEBUG] _build_headers_for_server called, server_info is None: {server_info is None}"
    )
    if server_info:
        logger.info(f"[AUTH DEBUG] server_info keys: {list(server_info.keys())}")
        server_headers = server_info.get("headers", [])
        if server_headers and isinstance(server_headers, list):
            for header_dict in server_headers:
                if isinstance(header_dict, dict):
                    from ..common.log_redaction import redact_headers

                    headers.update(header_dict)
                    logger.debug(
                        f"Added server headers to MCP client: {redact_headers(header_dict)}"
                    )

        # Gate every decrypted secret on a fresh SSRF re-validation of the exact
        # destination the headers will be sent to. Callers validate before
        # connecting, but self-guarding here means a future caller that forgets
        # cannot leak a decrypted credential to a private/metadata address. If we
        # have a secret to attach but no validated destination, fail closed and
        # attach no secret.
        _has_secret = bool(
            server_info.get("custom_headers_encrypted")
            or (
                server_info.get("auth_scheme", "none") != "none"
                and server_info.get("auth_credential_encrypted")
            )
        )
        _destination_safe = bool(destination_url) and _assert_mcp_url_fetchable(destination_url)
        if _has_secret and not _destination_safe:
            logger.warning(
                "Not attaching decrypted headers/credential for '%s': destination "
                "missing or failed SSRF re-validation.",
                server_info.get("service_path", "unknown"),
            )
            return headers

        # Custom headers go first; auth_scheme below overwrites name collisions
        encrypted_custom = server_info.get("custom_headers_encrypted")
        if encrypted_custom:
            from ..utils.credential_encryption import decrypt_custom_headers

            decrypted = decrypt_custom_headers(encrypted_custom)
            for entry in decrypted:
                headers[entry["name"]] = entry["value"]
            logger.debug(
                f"Merged {len(decrypted)} custom headers into outbound request "
                f"(names only): {[e['name'] for e in decrypted]}"
            )

        # Inject auth header from encrypted credentials (if present)
        auth_scheme = server_info.get("auth_scheme", "none")
        encrypted_credential = server_info.get("auth_credential_encrypted")

        logger.debug(
            f"[AUTH DEBUG] auth_scheme: {auth_scheme}, has_credential: {bool(encrypted_credential)}"
        )

        if auth_scheme != "none" and encrypted_credential:
            from ..utils.credential_encryption import decrypt_credential

            credential = decrypt_credential(encrypted_credential)
            if credential:
                if auth_scheme == "bearer":
                    header_name = server_info.get("auth_header_name", "Authorization")
                    headers[header_name] = f"Bearer {credential}"
                    logger.debug("Added Bearer auth header for MCP client")
                elif auth_scheme == "api_key":
                    header_name = server_info.get("auth_header_name", "X-API-Key")
                    headers[header_name] = credential
                    logger.debug(f"Added API key header '{header_name}' for MCP client")
            else:
                logger.warning(
                    f"Could not decrypt credential for "
                    f"'{server_info.get('service_path', 'unknown')}'. "
                    f"MCP client will proceed without auth."
                )

    return headers


def normalize_sse_endpoint_url_for_request(url_str: str) -> str:
    """
    Normalize URLs in HTTP requests by removing mount paths.
    Example: http://localhost:8000/currenttime/messages/... -> http://localhost:8000/messages/...
    """
    if "/messages/" not in url_str:
        return url_str

    # Pattern to match URLs like http://host:port/mount_path/messages/...
    import re

    pattern = r"(https?://[^/]+)/([^/]+)(/messages/.*)"
    match = re.match(pattern, url_str)

    if match:
        base_url = match.group(1)  # http://host:port
        mount_path = match.group(2)  # currenttime, mcpgw, etc.
        messages_path = match.group(3)  # /messages/...

        # Skip common paths that aren't mount paths
        if mount_path in ["api", "static", "health"]:
            return url_str

        normalized = f"{base_url}{messages_path}"
        logger.debug(f"Normalized request URL: {url_str} -> {normalized}")
        return normalized

    return url_str


async def detect_server_transport_aware(base_url: str, server_info: dict = None) -> str:
    """
    Detect which transport a server supports by checking configuration and testing endpoints.
    Uses server_info supported_transports if available, otherwise falls back to auto-detection.

    Args:
        base_url: The base URL of the MCP server
        server_info: Optional server configuration dict containing supported_transports

    Returns:
        The preferred transport type ("sse" or "streamable-http")
    """
    # If URL already has a transport endpoint, detect from it
    if base_url.endswith("/sse") or "/sse/" in base_url:
        logger.debug(f"Server URL {redact_url(base_url)} already has SSE endpoint")
        return "sse"
    elif base_url.endswith("/mcp") or "/mcp/" in base_url:
        logger.debug(f"Server URL {redact_url(base_url)} already has MCP endpoint")
        return "streamable-http"

    # Use server configuration if available
    if server_info:
        supported_transports = server_info.get("supported_transports", [])
        logger.debug(f"Server configuration specifies supported transports: {supported_transports}")

        # Prefer SSE if it's the only option or explicitly listed first
        if supported_transports == ["sse"]:
            logger.debug("Server only supports SSE transport")
            return "sse"
        elif (
            supported_transports
            and "sse" in supported_transports
            and "streamable-http" not in supported_transports
        ):
            logger.debug("Server supports SSE but not streamable-http")
            return "sse"
        elif supported_transports and "streamable-http" in supported_transports:
            logger.debug("Server supports streamable-http (preferred)")
            return "streamable-http"

    # Fall back to auto-detection
    return await detect_server_transport(base_url)


async def detect_server_transport(base_url: str) -> str:
    """
    Detect which transport a server supports by testing endpoints.
    Returns the preferred transport type.
    """
    # If URL already has a transport endpoint, detect from it
    if base_url.endswith("/sse") or "/sse/" in base_url:
        logger.debug(f"Server URL {redact_url(base_url)} already has SSE endpoint")
        return "sse"
    elif base_url.endswith("/mcp") or "/mcp/" in base_url:
        logger.debug(f"Server URL {redact_url(base_url)} already has MCP endpoint")
        return "streamable-http"

    # Fail closed on SSRF before probing the target with the (unpinnable) SDK
    # client. Both probe endpoints share this host, so validating base_url once
    # covers them. Default to streamable-http (no connection) when blocked.
    if not _assert_mcp_url_fetchable(base_url):
        return "streamable-http"

    # Test streamable-http first (default preference)
    try:
        mcp_url = base_url.rstrip("/") + "/mcp/"
        async with streamablehttp_client(url=mcp_url) as connection:
            logger.debug(f"Server at {redact_url(base_url)} supports streamable-http transport")
            return "streamable-http"
    except Exception as e:
        logger.debug(f"Streamable-HTTP test failed for {redact_url(base_url)}: {e}")

    # Fallback to SSE
    try:
        sse_url = base_url.rstrip("/") + "/sse"
        async with sse_client(sse_url) as connection:
            logger.debug(f"Server at {redact_url(base_url)} supports SSE transport")
            return "sse"
    except Exception as e:
        logger.debug(f"SSE test failed for {redact_url(base_url)}: {e}")

    # Default to streamable-http if detection fails
    logger.warning(
        f"Could not detect transport for {redact_url(base_url)}, defaulting to streamable-http"
    )
    return "streamable-http"


async def get_tools_from_server_with_transport(
    base_url: str, transport: str = "auto"
) -> list[dict] | None:
    """
    Connects to an MCP server using the specified transport, lists tools, and returns their details.

    Args:
        base_url: The base URL of the MCP server (e.g., http://localhost:8000).
        transport: Transport type ("streamable-http", "sse", or "auto")

    Returns:
        A list of tool detail dictionaries, or None if connection/retrieval fails.
    """
    if not base_url:
        logger.error("MCP Check Error: Base URL is empty.")
        return None

    # Auto-detect transport if needed
    if transport == "auto":
        transport = await detect_server_transport(base_url)

    logger.info(
        f"Attempting to connect to MCP server at {redact_url(base_url)} using {transport} transport..."
    )

    try:
        if transport == "streamable-http":
            return await _get_tools_streamable_http(base_url)
        elif transport == "sse":
            return await _get_tools_sse(base_url)
        else:
            logger.error(f"Unsupported transport type: {transport}")
            return None

    except Exception as e:
        logger.error(
            f"MCP Check Error: Failed to get tool list from {redact_url(base_url)} with {transport}: {type(e).__name__} - {e}"
        )
        return None


async def _get_tools_streamable_http(base_url: str, server_info: dict = None) -> list[dict] | None:
    """Get tools using streamable-http transport"""
    # Check if server_info has explicit mcp_endpoint
    explicit_endpoint = server_info.get("mcp_endpoint") if server_info else None

    # Fail closed on SSRF BEFORE decrypting/building credential headers: a
    # target that resolves to a private/metadata address must never receive the
    # server's decrypted backend credentials. Validate the actual endpoint about
    # to be connected to (explicit endpoint if set, else the base URL).
    if not _assert_mcp_url_fetchable(explicit_endpoint or base_url):
        return None

    # Build headers for the server (destination re-validated inside before any
    # decrypted secret is attached).
    headers = _build_headers_for_server(server_info, destination_url=explicit_endpoint or base_url)

    # If explicit endpoint is provided, use it directly (single attempt)
    if explicit_endpoint:
        mcp_url = explicit_endpoint
        logger.info(f"MCP Client: Using explicit mcp_endpoint: {redact_url(mcp_url)}")

        # Handle servers imported from anthropic by adding required query parameter
        if (
            server_info
            and "tags" in server_info
            and "anthropic-registry" in server_info.get("tags", [])
        ):
            if "?" not in mcp_url:
                mcp_url += "?instance_id=default"
            elif "instance_id=" not in mcp_url:
                mcp_url += "&instance_id=default"

        try:
            async with streamablehttp_client(url=mcp_url, headers=headers) as (
                read,
                write,
                get_session_id,
            ):
                async with ClientSession(read, write) as session:
                    await asyncio.wait_for(session.initialize(), timeout=10.0)
                    tools_response = await asyncio.wait_for(session.list_tools(), timeout=15.0)
                    result = _extract_tool_details(tools_response)
                    return result
        except Exception as e:
            logger.error(
                f"MCP Check Error: Streamable-HTTP connection failed to {redact_url(mcp_url)}: {e}"
            )
            return None

    # If URL already has MCP endpoint, use it directly
    if base_url.endswith("/mcp") or "/mcp/" in base_url:
        mcp_url = base_url
        # Don't add trailing slash - some servers like Cloudflare reject it

        # Handle streamable-http and sse servers imported from anthropic by adding required query parameter
        if (
            server_info
            and "tags" in server_info
            and "anthropic-registry" in server_info.get("tags", [])
        ):
            if "?" not in mcp_url:
                mcp_url += "?instance_id=default"
            elif "instance_id=" not in mcp_url:
                mcp_url += "&instance_id=default"
        else:
            logger.debug(f"Not a Strata server, URL unchanged: {redact_url(mcp_url)}")

        logger.debug(f"About to connect to: {redact_url(mcp_url)}")
        try:
            async with streamablehttp_client(url=mcp_url, headers=headers) as (
                read,
                write,
                get_session_id,
            ):
                async with ClientSession(read, write) as session:
                    await asyncio.wait_for(session.initialize(), timeout=10.0)
                    tools_response = await asyncio.wait_for(session.list_tools(), timeout=15.0)

                    result = _extract_tool_details(tools_response)
                    return result
        except Exception as e:
            logger.error(
                f"MCP Check Error: Streamable-HTTP connection failed to {redact_url(base_url)}: {e}"
            )

            return None
    else:
        # Try with /mcp suffix first, then without if it fails
        endpoints_to_try = [base_url.rstrip("/") + "/mcp/", base_url.rstrip("/") + "/"]

        for mcp_url in endpoints_to_try:
            try:
                logger.info(f"MCP Client: Trying streamable-http endpoint: {redact_url(mcp_url)}")
                async with streamablehttp_client(url=mcp_url, headers=headers) as (
                    read,
                    write,
                    get_session_id,
                ):
                    async with ClientSession(read, write) as session:
                        await asyncio.wait_for(session.initialize(), timeout=10.0)
                        tools_response = await asyncio.wait_for(session.list_tools(), timeout=15.0)

                        logger.info(f"MCP Client: Successfully connected to {redact_url(mcp_url)}")
                        return _extract_tool_details(tools_response)

            except TimeoutError:
                logger.error(
                    f"MCP Check Error: Timeout during streamable-http session with {redact_url(mcp_url)}."
                )
                if mcp_url == endpoints_to_try[0]:
                    continue
                return None
            except Exception as e:
                logger.error(
                    f"MCP Check Error: Streamable-HTTP connection failed to {redact_url(mcp_url)}: {e}"
                )
                if mcp_url == endpoints_to_try[0]:
                    continue
                return None

    return None


async def _get_tools_sse(base_url: str, server_info: dict = None) -> list[dict] | None:
    """Get tools using SSE transport (legacy method with patches)"""
    # Check if server_info has explicit sse_endpoint
    explicit_endpoint = server_info.get("sse_endpoint") if server_info else None

    # Resolve SSE endpoint URL
    if explicit_endpoint:
        sse_url = explicit_endpoint
        logger.info(f"MCP Client: Using explicit sse_endpoint: {redact_url(sse_url)}")
    elif base_url.endswith("/sse") or "/sse/" in base_url:
        sse_url = base_url
    else:
        sse_url = base_url.rstrip("/") + "/sse"

    secure_prefix = "s" if sse_url.startswith("https://") else ""
    mcp_server_url = f"http{secure_prefix}://{sse_url[len(f'http{secure_prefix}://') :]}"

    # Fail closed on SSRF BEFORE decrypting/building credential headers. This
    # validates the ACTUAL connection target (mcp_server_url), whose host is
    # taken verbatim from the explicit sse_endpoint when one is set, so an
    # sse_endpoint pointing at a private/metadata/loopback address is rejected
    # before any credential is built or attached.
    if not _assert_mcp_url_fetchable(mcp_server_url):
        return None

    # Build headers for the server (destination re-validated inside before any
    # decrypted secret is attached).
    headers = _build_headers_for_server(server_info, destination_url=mcp_server_url)

    try:
        # Monkey patch httpx to fix mount path issues (legacy SSE support)
        original_request = httpx.AsyncClient.request

        async def patched_request(self, method, url, **kwargs):
            if isinstance(url, str) and "/messages/" in url:
                url = normalize_sse_endpoint_url_for_request(url)
            elif hasattr(url, "__str__") and "/messages/" in str(url):
                url = normalize_sse_endpoint_url_for_request(str(url))
            return await original_request(self, method, url, **kwargs)

        httpx.AsyncClient.request = patched_request  # type: ignore[method-assign]  # legacy SSE monkeypatch

        try:
            async with sse_client(mcp_server_url, headers=headers) as (read, write):
                async with ClientSession(read, write, sampling_callback=None) as session:
                    await asyncio.wait_for(session.initialize(), timeout=10.0)
                    tools_response = await asyncio.wait_for(session.list_tools(), timeout=15.0)

                    return _extract_tool_details(tools_response)
        finally:
            httpx.AsyncClient.request = original_request  # type: ignore[method-assign]  # restore monkeypatch

    except TimeoutError:
        logger.error(f"MCP Check Error: Timeout during SSE session with {redact_url(base_url)}.")
        return None
    except Exception as e:
        logger.error(f"MCP Check Error: SSE connection failed to {redact_url(base_url)}: {e}")
        return None


def _extract_tool_details(tools_response) -> list[dict]:
    """Extract tool details from MCP tools response."""
    tool_details_list: list[dict[str, Any]] = []

    if tools_response and hasattr(tools_response, "tools"):
        for tool in tools_response.tools:
            tool_name = getattr(tool, "name", "Unknown Name")
            tool_desc = getattr(tool, "description", None) or getattr(tool, "__doc__", None)

            # Log tool description for debugging
            desc_preview = repr(tool_desc)[:100] if tool_desc else "None"
            logger.debug(f"Tool '{tool_name}' description: {desc_preview}")

            # Parse docstring into sections
            parsed_desc = {
                "main": "No description available.",
                "args": None,
                "returns": None,
                "raises": None,
            }
            if tool_desc:
                tool_desc = tool_desc.strip()
                lines = tool_desc.split("\n")
                main_desc_lines: list[str] = []
                current_section = "main"
                section_content: list[str] = []

                for line in lines:
                    stripped_line = line.strip()
                    if stripped_line.startswith("Args:"):
                        parsed_desc["main"] = "\n".join(main_desc_lines).strip()
                        current_section = "args"
                        section_content = [stripped_line[len("Args:") :].strip()]
                    elif stripped_line.startswith("Returns:"):
                        if current_section != "main":
                            parsed_desc[current_section] = "\n".join(section_content).strip()
                        else:
                            parsed_desc["main"] = "\n".join(main_desc_lines).strip()
                        current_section = "returns"
                        section_content = [stripped_line[len("Returns:") :].strip()]
                    elif stripped_line.startswith("Raises:"):
                        if current_section != "main":
                            parsed_desc[current_section] = "\n".join(section_content).strip()
                        else:
                            parsed_desc["main"] = "\n".join(main_desc_lines).strip()
                        current_section = "raises"
                        section_content = [stripped_line[len("Raises:") :].strip()]
                    elif current_section == "main":
                        main_desc_lines.append(line.strip())
                    else:
                        section_content.append(line.strip())

                # Add the last collected section
                if current_section != "main":
                    parsed_desc[current_section] = "\n".join(section_content).strip()
                elif not parsed_desc["main"] and main_desc_lines:
                    parsed_desc["main"] = "\n".join(main_desc_lines).strip()

                # Ensure main description has content
                if not parsed_desc["main"] and (
                    parsed_desc["args"] or parsed_desc["returns"] or parsed_desc["raises"]
                ):
                    parsed_desc["main"] = "(No primary description provided)"
            else:
                parsed_desc["main"] = "No description available."

            tool_schema = getattr(tool, "inputSchema", {})

            tool_details_list.append(
                {
                    "name": tool_name,
                    "description": tool_desc or "",
                    "parsed_description": parsed_desc,
                    "schema": tool_schema,
                }
            )

    tool_names = [tool["name"] for tool in tool_details_list]
    logger.info(
        f"Successfully retrieved details for {len(tool_details_list)} tools: {', '.join(tool_names)}"
    )
    return tool_details_list


async def get_tools_from_server_with_server_info(
    base_url: str, server_info: dict = None
) -> list[dict] | None:
    """
    Get tools from server using server configuration to determine optimal transport.

    Args:
        base_url: The base URL of the MCP server (e.g., http://localhost:8000).
        server_info: Optional server configuration dict containing supported_transports

    Returns:
        A list of tool detail dictionaries (keys: name, description, schema),
        or None if connection/retrieval fails.
    """

    if not base_url:
        logger.error("MCP Check Error: Base URL is empty.")
        return None

    # Use transport-aware detection
    transport = await detect_server_transport_aware(base_url, server_info)

    logger.info(
        f"Attempting to connect to MCP server at {redact_url(base_url)} using {transport} transport (server-info aware)..."
    )

    try:
        if transport == "streamable-http":
            return await _get_tools_streamable_http(base_url, server_info)
        elif transport == "sse":
            return await _get_tools_sse(base_url, server_info)
        else:
            logger.error(f"Unsupported transport type: {transport}")
            return None

    except Exception as e:
        logger.error(
            f"MCP Check Error: Failed to get tool list from {redact_url(base_url)} with {transport}: {type(e).__name__} - {e}"
        )
        return None


async def get_mcp_connection_result(
    base_url: str, server_info: dict = None
) -> MCPConnectionResult | None:
    """
    Connect to MCP server and return both tools and server info.

    This function performs the MCP initialize handshake and extracts
    the serverInfo (name, version) from the response along with tools.

    Args:
        base_url: The base URL of the MCP server
        server_info: Optional server configuration dict

    Returns:
        MCPConnectionResult with tools and server_info, or None on failure
    """
    if not base_url:
        logger.error("MCP Check Error: Base URL is empty.")
        return None

    # Determine the MCP endpoint URL
    explicit_endpoint = server_info.get("mcp_endpoint") if server_info else None
    explicit_sse_endpoint = server_info.get("sse_endpoint") if server_info else None

    # Fail closed on SSRF BEFORE any transport probe or credential build: a
    # target that resolves to a private/metadata address must never receive the
    # server's decrypted backend credentials. Validate BOTH override endpoint
    # fields (mcp_endpoint and sse_endpoint) here because either one can be the
    # actual connection target below depending on the negotiated transport; the
    # SDK client is unpinnable, so this is the fetch-time re-validation.
    if not _assert_mcp_url_fetchable(explicit_endpoint or base_url):
        return None
    if explicit_sse_endpoint and not _assert_mcp_url_fetchable(explicit_sse_endpoint):
        return None

    # Use transport-aware detection
    transport = await detect_server_transport_aware(base_url, server_info)

    logger.info(
        f"Getting MCP connection result from {redact_url(base_url)} using {transport} transport..."
    )

    # Build headers for the server (destination re-validated inside before any
    # decrypted secret is attached).
    headers = _build_headers_for_server(server_info, destination_url=explicit_endpoint or base_url)

    if explicit_endpoint:
        mcp_url = explicit_endpoint
    elif base_url.endswith("/mcp") or "/mcp/" in base_url:
        mcp_url = base_url
    else:
        mcp_url = base_url.rstrip("/") + "/mcp/"

    # Handle anthropic-registry servers
    if (
        server_info
        and "tags" in server_info
        and "anthropic-registry" in server_info.get("tags", [])
    ):
        if "?" not in mcp_url:
            mcp_url += "?instance_id=default"
        elif "instance_id=" not in mcp_url:
            mcp_url += "&instance_id=default"

    try:
        if transport == "streamable-http":
            async with streamablehttp_client(url=mcp_url, headers=headers) as (
                read,
                write,
                get_session_id,
            ):
                async with ClientSession(read, write) as session:
                    # Capture the initialize result which contains serverInfo
                    init_result = await asyncio.wait_for(session.initialize(), timeout=10.0)
                    tools_response = await asyncio.wait_for(session.list_tools(), timeout=15.0)

                    tools = _extract_tool_details(tools_response)

                    # Extract server info from initialize result
                    mcp_server_info: MCPServerInfo = {}
                    if (
                        init_result
                        and hasattr(init_result, "serverInfo")
                        and init_result.serverInfo
                    ):
                        if hasattr(init_result.serverInfo, "name"):
                            mcp_server_info["name"] = init_result.serverInfo.name
                        if hasattr(init_result.serverInfo, "version"):
                            mcp_server_info["version"] = init_result.serverInfo.version

                    if mcp_server_info:
                        logger.info(
                            f"MCP Server Info from {redact_url(base_url)}: "
                            f"name={mcp_server_info.get('name')}, "
                            f"version={mcp_server_info.get('version')}"
                        )

                    return MCPConnectionResult(tools=tools or [], server_info=mcp_server_info)

        elif transport == "sse":
            # For SSE transport
            sse_endpoint = server_info.get("sse_endpoint") if server_info else None
            if sse_endpoint:
                sse_url = sse_endpoint
            else:
                sse_url = base_url.rstrip("/") + "/sse"

            async with sse_client(url=sse_url, headers=headers) as (read, write):
                async with ClientSession(read, write) as session:
                    # Capture the initialize result which contains serverInfo
                    init_result = await asyncio.wait_for(session.initialize(), timeout=10.0)
                    tools_response = await asyncio.wait_for(session.list_tools(), timeout=15.0)

                    tools = _extract_tool_details(tools_response)

                    # Extract server info from initialize result
                    mcp_server_info = MCPServerInfo()
                    if (
                        init_result
                        and hasattr(init_result, "serverInfo")
                        and init_result.serverInfo
                    ):
                        if hasattr(init_result.serverInfo, "name"):
                            mcp_server_info["name"] = init_result.serverInfo.name
                        if hasattr(init_result.serverInfo, "version"):
                            mcp_server_info["version"] = init_result.serverInfo.version

                    if mcp_server_info:
                        logger.info(
                            f"MCP Server Info from {redact_url(base_url)}: "
                            f"name={mcp_server_info.get('name')}, "
                            f"version={mcp_server_info.get('version')}"
                        )

                    return MCPConnectionResult(tools=tools or [], server_info=mcp_server_info)

        else:
            logger.error(f"Unsupported transport type: {transport}")
            return None

    except TimeoutError:
        logger.error(f"MCP Check Error: Timeout connecting to {redact_url(mcp_url)}")
        return None
    except Exception as e:
        logger.error(
            f"MCP Check Error: Failed to get connection result from {redact_url(base_url)}: "
            f"{type(e).__name__} - {e}"
        )
        return None


class MCPClientService:
    """Service wrapper for the MCP client function to maintain compatibility."""

    async def get_tools_from_server_with_server_info(
        self, base_url: str, server_info: dict = None
    ) -> list[dict] | None:
        """Wrapper method that uses server configuration for transport selection."""
        return await get_tools_from_server_with_server_info(base_url, server_info)

    async def get_mcp_connection_result(
        self, base_url: str, server_info: dict = None
    ) -> MCPConnectionResult | None:
        """Get both tools and server info from MCP server."""
        return await get_mcp_connection_result(base_url, server_info)


# Global MCP client service instance
mcp_client_service = MCPClientService()
