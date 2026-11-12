"""
MCP Client Service

Handles connections to MCP servers and tool list retrieval.
Copied directly from main_old.py working implementation.
"""

import asyncio
import logging
import re

# MCP SDK 2.x. Client(mode="auto") speaks MCP 2026-07-28 and falls back to the
# initialize handshake for servers on 2025-11-25 and earlier. The SDK's HTTP
# stack is httpx2, a separate package from the httpx the rest of the registry
# uses.
import ssl
from typing import (
    Any,
    Literal,
    TypedDict,
)

import httpx2
from mcp import Client
from mcp.client.sse import sse_client
from mcp.client.streamable_http import streamable_http_client
from mcp.types import (
    Implementation,
    Tool,
)
from pydantic import ValidationError

from ..common.log_redaction import redact_url
from .backend_oauth import RESOLVED_BEARER_KEY as _BACKEND_OAUTH_TOKEN_KEY
from .backend_oauth import with_bearer as _with_backend_oauth
from .mcp_apps import (
    mcp_apps_client_extensions,
    tool_ui_fields,
)

logger = logging.getLogger(__name__)

# One budget covers connecting, version negotiation and listing every page of
# tools. It replaces the old separate 10s initialize and 15s list_tools limits.
MCP_DISCOVERY_TIMEOUT_SECONDS: float = 25.0
# A server that keeps returning nextCursor stops being paged after this many
# pages, so a buggy or hostile server cannot hold discovery open.
MAX_TOOL_LIST_PAGES: int = 50
# HTTP client timeouts for the SDK transport. The overall budget above is the
# real limit; these only bound a single stalled read or connect.
MCP_HTTP_CONNECT_TIMEOUT_SECONDS: float = 10.0
MCP_HTTP_READ_TIMEOUT_SECONDS: float = 25.0
# The SDK's own client defaults (mcp.shared._httpx_utils), kept for the SSE
# transport, whose stream stays open between events.
SDK_DEFAULT_TIMEOUT_SECONDS: float = 30.0
SDK_DEFAULT_SSE_READ_TIMEOUT_SECONDS: float = 300.0
MCP_DISCOVERY_CLIENT_NAME: str = "mcp-gateway-registry"
MCP_DISCOVERY_CLIENT_VERSION: str = "2.0.0"

# Negotiation mode per transport. Streamable HTTP probes server/discover and
# falls back to initialize. The HTTP+SSE transport predates 2026-07-28, so it
# always uses the handshake.
_TransportMode = Literal["auto", "legacy"]


def _assert_mcp_url_fetchable(
    url: str,
    server_info: dict[str, Any] | None = None,
) -> bool:
    """Fail-closed SSRF check before an MCP SDK connection is opened.

    The MCP SDK transports (``streamable_http_client`` / ``sse_client``) run on
    their own httpx2 client, so they cannot use the pinned guarded transport that
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
    from ..utils.url_guard import (
        proxy_profile_for_entity_target,
        validate_url,
    )

    identity = server_info or {}
    entity_path = identity.get("path") or identity.get("service_path")
    registered_target = identity.get("proxy_pass_url")
    try:
        profile = proxy_profile_for_entity_target("mcp_server", entity_path, registered_target, url)
        validate_url(url, profile=profile)
        return True
    except UrlValidationError:
        logger.warning(
            f"MCP connection blocked by SSRF guard for {redact_url(url)}",
        )
        return False
    except Exception as exc:  # pragma: no cover - defensive, fail closed
        logger.warning(
            f"MCP connection blocked for {redact_url(url)} validation_type={type(exc).__name__}",
        )
        return False


def _describe_validation_error(
    error: ValidationError,
) -> str:
    """Summarize a pydantic error from the SDK without dumping tool schemas.

    The MCP SDK validates the whole tools/list result at once, so one tool
    whose schema the SDK rejects fails the listing. Name where it failed, for
    example "tools.4.inputSchema.type", so the operator can find the tool.

    Args:
        error: The ValidationError the SDK raised.

    Returns:
        A short description with the error count and the first failing field.
    """
    errors = error.errors()
    first = ".".join(str(part) for part in errors[0]["loc"]) if errors else "unknown"
    return f"{len(errors)} validation error(s), first at {first}"


def _server_info_dict(
    implementation: Implementation | None,
) -> "MCPServerInfo":
    """Convert the SDK's server identity into the registry's MCPServerInfo.

    Args:
        implementation: Server identity from the SDK. None when a 2026-07-28
            server did not identify itself.

    Returns:
        MCPServerInfo with name and version when known, else empty.
    """
    info: MCPServerInfo = {}
    if implementation is None:
        return info
    if implementation.name:
        info["name"] = implementation.name
    if implementation.version:
        info["version"] = implementation.version
    return info


async def _list_all_tools(
    client: Client,
) -> list[Tool]:
    """List every tool, following nextCursor up to MAX_TOOL_LIST_PAGES pages.

    Args:
        client: An entered MCP SDK client.

    Returns:
        All tools across pages, in server order.
    """
    tools: list[Tool] = []
    cursor: str | None = None
    for page in range(1, MAX_TOOL_LIST_PAGES + 1):
        result = await client.list_tools(cursor=cursor)
        tools.extend(result.tools)
        cursor = result.next_cursor
        if not cursor:
            return tools
        logger.debug(f"tools/list page {page} returned a cursor; fetching the next page")
    logger.warning(
        f"tools/list still returned a cursor after {MAX_TOOL_LIST_PAGES} pages; "
        f"keeping the first {len(tools)} tools"
    )
    return tools


async def _discover_tools(
    transport: Any,
    mode: _TransportMode,
) -> tuple[list[Tool], Implementation | None]:
    """Connect over an SDK transport, negotiate, and list every tool.

    In "auto" mode the SDK sends server/discover first. A 2026-07-28 server
    answers it and gets stateless requests. A handshake-era server does not,
    so the SDK falls back to initialize. Response caching is off, because
    discovery must see the server's current list. The client declares MCP Apps
    support, because a server that follows that spec hides its UI tools from a
    client that does not.

    Args:
        transport: An SDK transport context manager, not yet entered.
        mode: "auto" for Streamable HTTP, "legacy" for HTTP+SSE.

    Returns:
        The tools and the server identity (None if the server sent none).

    Raises:
        TimeoutError: The whole exchange took longer than
            MCP_DISCOVERY_TIMEOUT_SECONDS.
        Exception: Any connection, protocol or validation error from the SDK.
    """
    client = Client(
        transport,
        mode=mode,
        cache=None,
        client_info=Implementation(
            name=MCP_DISCOVERY_CLIENT_NAME,
            version=MCP_DISCOVERY_CLIENT_VERSION,
        ),
        extensions=mcp_apps_client_extensions(),
    )
    async with asyncio.timeout(MCP_DISCOVERY_TIMEOUT_SECONDS):
        async with client:
            logger.debug(f"MCP session negotiated protocol_version={client.protocol_version}")
            tools = await _list_all_tools(client)
            return tools, client.server_info


def _tls_context() -> ssl.SSLContext:
    """The registry's TLS trust for httpx2 (imported here, as url_guard is above)."""
    from ..utils.url_guard import egress_ssl_context

    return egress_ssl_context()


def _verified_http_client(
    headers: dict[str, str] | None = None,
    timeout: httpx2.Timeout | None = None,
    auth: httpx2.Auth | None = None,
) -> httpx2.AsyncClient:
    """The SDK's client factory, with the registry's TLS trust.

    Same defaults as mcp.shared._httpx_utils.create_mcp_http_client; only the
    certificate trust differs (see egress_ssl_context).
    """
    if timeout is None:
        timeout = httpx2.Timeout(
            SDK_DEFAULT_TIMEOUT_SECONDS,
            read=SDK_DEFAULT_SSE_READ_TIMEOUT_SECONDS,
        )
    return httpx2.AsyncClient(
        headers=headers,
        timeout=timeout,
        auth=auth,
        verify=_tls_context(),
    )


async def _discover_streamable_http(
    mcp_url: str,
    headers: dict[str, str],
) -> tuple[list[Tool], Implementation | None]:
    """Discover tools over Streamable HTTP, speaking either protocol era.

    The SDK only takes headers through its own httpx2 client, so build one
    that carries them. The SDK does not close a client it was given, so this
    function closes it.

    Args:
        mcp_url: The exact MCP endpoint, already checked by
            _assert_mcp_url_fetchable.
        headers: Headers built for this exact destination.

    Returns:
        The tools and the server identity.
    """
    timeout = httpx2.Timeout(
        MCP_HTTP_READ_TIMEOUT_SECONDS,
        connect=MCP_HTTP_CONNECT_TIMEOUT_SECONDS,
    )
    async with httpx2.AsyncClient(
        headers=headers,
        timeout=timeout,
        verify=_tls_context(),
    ) as http_client:
        transport = streamable_http_client(mcp_url, http_client=http_client)
        return await _discover_tools(transport, mode="auto")


async def _discover_sse(
    sse_url: str,
    headers: dict[str, str],
) -> tuple[list[Tool], Implementation | None]:
    """Discover tools over the deprecated HTTP+SSE transport (handshake only).

    Args:
        sse_url: The exact SSE endpoint, already checked by
            _assert_mcp_url_fetchable.
        headers: Headers built for this exact destination.

    Returns:
        The tools and the server identity.
    """
    transport = sse_client(sse_url, headers=headers, httpx_client_factory=_verified_http_client)
    return await _discover_tools(transport, mode="legacy")


def _connection_result(
    base_url: str,
    sdk_tools: list[Tool],
    implementation: Implementation | None,
) -> "MCPConnectionResult":
    """Build the MCPConnectionResult for get_mcp_connection_result.

    Args:
        base_url: The server's base URL, for the log line.
        sdk_tools: Tools from the SDK.
        implementation: Server identity from the SDK, if the server sent one.

    Returns:
        MCPConnectionResult with tools and server_info.
    """
    mcp_server_info = _server_info_dict(implementation)
    if mcp_server_info:
        logger.info(
            f"MCP Server Info from {redact_url(base_url)}: "
            f"name={mcp_server_info.get('name')}, "
            f"version={mcp_server_info.get('version')}"
        )
    tools = _extract_tool_details(sdk_tools)
    return MCPConnectionResult(tools=tools or [], server_info=mcp_server_info)


def _root_cause(
    error: BaseException,
) -> BaseException:
    """Return the first non-group exception inside an ExceptionGroup.

    The SDK runs transports in anyio task groups, so a failure such as a
    tools/list ValidationError reaches the caller wrapped in an ExceptionGroup.
    Logging only "ExceptionGroup" hides the reason.

    Args:
        error: The exception the caller caught.

    Returns:
        The first leaf exception, or the error itself when it is not a group.
    """
    while isinstance(error, BaseExceptionGroup) and error.exceptions:
        error = error.exceptions[0]
    return error


def _log_discovery_failure(
    prefix: str,
    url: str,
    error: Exception,
) -> None:
    """Log a failed discovery attempt without leaking credentials or schemas.

    Args:
        prefix: What was being attempted, for the log line.
        url: The endpoint; redacted before logging.
        error: The exception raised.
    """
    cause = _root_cause(error)
    detail = type(cause).__name__
    if isinstance(cause, ValidationError):
        detail = f"{detail}: {_describe_validation_error(cause)}"
    logger.error(f"{prefix} endpoint={redact_url(url)} type={detail}")


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

        logger.debug(f"Stripping MCP SSE mount path={mount_path}")
        return rest_of_url

    # If no mount path pattern detected, return as-is
    return endpoint_url


def _build_headers_for_server(
    server_info: dict | None = None,
    destination_url: str | None = None,
) -> dict[str, str]:
    """Build MCP headers, attaching server values only to a validated target.

    Protocol headers are always returned. Every server-provided header,
    encrypted custom header, and auth credential is withheld unless the exact
    destination passes the entity-bound URL guard before any decryption occurs.
    """
    headers = {
        "Accept": "application/json, text/event-stream",
        "Content-Type": "application/json",
    }
    if not server_info:
        return headers

    has_server_values = bool(
        server_info.get("headers")
        or server_info.get("custom_headers_encrypted")
        or (
            server_info.get("auth_scheme", "none") != "none"
            and server_info.get("auth_credential_encrypted")
        )
        or server_info.get(_BACKEND_OAUTH_TOKEN_KEY)
    )
    destination_safe = bool(destination_url) and _assert_mcp_url_fetchable(
        destination_url,
        server_info,
    )
    if has_server_values and not destination_safe:
        logger.warning(
            f"MCP headers withheld service={server_info.get('service_path') or server_info.get('path') or 'unknown'} destination={redact_url(destination_url or '')} reason=URL validation failed",
        )
        return headers

    server_headers = server_info.get("headers", [])
    if isinstance(server_headers, list):
        for header_dict in server_headers:
            if isinstance(header_dict, dict):
                headers.update(header_dict)
                logger.debug(
                    f"Added server header names to MCP client names={sorted(str(name) for name in header_dict)}",
                )

    encrypted_custom = server_info.get("custom_headers_encrypted")
    if encrypted_custom:
        from ..utils.credential_encryption import decrypt_custom_headers

        decrypted = decrypt_custom_headers(encrypted_custom)
        for entry in decrypted:
            headers[entry["name"]] = entry["value"]
        logger.debug(
            f"Merged encrypted custom headers into MCP request count={len(decrypted):d} names={sorted(entry['name'] for entry in decrypted)}",
        )

    # A pre-resolved OAuth bearer is the only credential when it is present: it is
    # set by the async caller (backend_oauth.with_bearer), whose tiers all bow out
    # when an explicit static auth_scheme is configured, so this cannot shadow an
    # operator's scan token. This builder is synchronous, so the token must already
    # be resolved+stashed. Gated by the same destination check above.
    #
    # ALWAYS 'Authorization'. An OAuth bearer is not interchangeable with a static
    # credential, so it must not inherit auth_header_name: an api_key server has
    # that set to e.g. 'X-API-Key', which would emit `X-API-Key: Bearer <oauth>` --
    # an RFC 6750 bearer under an API-key header name, with a scheme prefix the
    # api_key path deliberately omits. No upstream consumes that correctly.
    resolved_bearer = server_info.get(_BACKEND_OAUTH_TOKEN_KEY)
    if resolved_bearer:
        headers["Authorization"] = f"Bearer {resolved_bearer}"
        logger.debug("Added resolved OAuth bearer header to MCP request")
        return headers
    auth_scheme = server_info.get("auth_scheme", "none")
    encrypted_credential = server_info.get("auth_credential_encrypted")
    if auth_scheme != "none" and encrypted_credential:
        from ..utils.credential_encryption import decrypt_credential

        credential = decrypt_credential(encrypted_credential)
        if credential:
            if auth_scheme == "bearer":
                header_name = server_info.get("auth_header_name", "Authorization")
                headers[header_name] = f"Bearer {credential}"
                logger.debug("Added bearer auth header to MCP request")
            elif auth_scheme == "api_key":
                header_name = server_info.get("auth_header_name", "X-API-Key")
                headers[header_name] = credential
                logger.debug(f"Added API-key header name={header_name} to MCP request")
        else:
            logger.warning(
                f"MCP credential decryption failed service={server_info.get('service_path') or server_info.get('path') or 'unknown'}",
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
        logger.debug(
            f"Normalized MCP request URL from={redact_url(url_str)} to={redact_url(normalized)}",
        )
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
        async with streamable_http_client(mcp_url) as connection:
            logger.debug(f"Server at {redact_url(base_url)} supports streamable-http transport")
            return "streamable-http"
    except Exception as e:
        logger.debug(
            f"Streamable-HTTP probe failed endpoint={redact_url(base_url)} type={type(e).__name__}",
        )

    # Fallback to SSE
    try:
        sse_url = base_url.rstrip("/") + "/sse"
        async with sse_client(sse_url) as connection:
            logger.debug(f"Server at {redact_url(base_url)} supports SSE transport")
            return "sse"
    except Exception as e:
        logger.debug(
            f"SSE probe failed endpoint={redact_url(base_url)} type={type(e).__name__}",
        )

    # Default to streamable-http if detection fails
    logger.warning(
        f"Could not detect transport endpoint={redact_url(base_url)}; defaulting to streamable-http",
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
        f"Attempting MCP connection endpoint={redact_url(base_url)} transport={transport}",
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
            f"MCP Check Error: Failed to get tool list from {redact_url(base_url)} with {transport}: {type(e).__name__}"
        )
        return None


async def _get_tools_streamable_http(base_url: str, server_info: dict = None) -> list[dict] | None:
    """Get tools using streamable-http transport"""
    # Check if server_info has explicit mcp_endpoint
    explicit_endpoint = server_info.get("mcp_endpoint") if server_info else None

    # Derive and validate each exact destination before building/decrypting its
    # credential headers. A base URL check is insufficient when the actual MCP
    # endpoint adds a path or query parameter.

    # If explicit endpoint is provided, use it directly (single attempt)
    if explicit_endpoint:
        mcp_url = explicit_endpoint
        logger.info(f"MCP client using explicit endpoint={redact_url(mcp_url)}")

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

        if not _assert_mcp_url_fetchable(mcp_url, server_info):
            return None
        headers = _build_headers_for_server(server_info, destination_url=mcp_url)

        try:
            tools, _ = await _discover_streamable_http(mcp_url, headers)
            return _extract_tool_details(tools)
        except Exception as e:
            _log_discovery_failure("Streamable-HTTP connection failed", mcp_url, e)
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
            logger.debug(f"MCP endpoint unchanged={redact_url(mcp_url)}")

        logger.debug(f"MCP connection starting endpoint={redact_url(mcp_url)}")
        if not _assert_mcp_url_fetchable(mcp_url, server_info):
            return None
        headers = _build_headers_for_server(server_info, destination_url=mcp_url)
        try:
            tools, _ = await _discover_streamable_http(mcp_url, headers)
            return _extract_tool_details(tools)
        except Exception as e:
            _log_discovery_failure("Streamable-HTTP connection failed", base_url, e)
            return None
    else:
        # Try with /mcp suffix first, then without if it fails
        endpoints_to_try = [base_url.rstrip("/") + "/mcp/", base_url.rstrip("/") + "/"]

        for mcp_url in endpoints_to_try:
            if not _assert_mcp_url_fetchable(mcp_url, server_info):
                continue
            headers = _build_headers_for_server(server_info, destination_url=mcp_url)
            try:
                logger.info(f"Trying streamable-http endpoint={redact_url(mcp_url)}")
                tools, _ = await _discover_streamable_http(mcp_url, headers)
                logger.info(f"MCP connection succeeded endpoint={redact_url(mcp_url)}")
                return _extract_tool_details(tools)

            except TimeoutError:
                logger.error(
                    f"MCP streamable-http timeout endpoint={redact_url(mcp_url)}",
                )
                if mcp_url == endpoints_to_try[0]:
                    continue
                return None
            except Exception as e:
                _log_discovery_failure(
                    "MCP Check Error: Streamable-HTTP connection failed", mcp_url, e
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
        logger.info(f"MCP client using explicit SSE endpoint={redact_url(sse_url)}")
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
    if not _assert_mcp_url_fetchable(mcp_server_url, server_info):
        return None

    # Build headers for the server (destination re-validated inside before any
    # decrypted secret is attached).
    headers = _build_headers_for_server(server_info, destination_url=mcp_server_url)

    try:
        # Monkey patch the SDK's HTTP client to fix mount path issues (legacy SSE
        # support). SDK 2.x uses httpx2; patching httpx would silently do nothing.
        original_request = httpx2.AsyncClient.request

        async def patched_request(self, method, url, **kwargs):
            if isinstance(url, str) and "/messages/" in url:
                url = normalize_sse_endpoint_url_for_request(url)
            elif hasattr(url, "__str__") and "/messages/" in str(url):
                url = normalize_sse_endpoint_url_for_request(str(url))
            return await original_request(self, method, url, **kwargs)

        httpx2.AsyncClient.request = patched_request  # type: ignore[method-assign]  # legacy SSE monkeypatch

        try:
            tools, _ = await _discover_sse(mcp_server_url, headers)
            return _extract_tool_details(tools)
        finally:
            httpx2.AsyncClient.request = original_request  # type: ignore[method-assign]  # restore monkeypatch

    except TimeoutError:
        logger.error(
            f"MCP timeout during SSE session endpoint={redact_url(base_url)}",
        )
        return None
    except Exception as e:
        _log_discovery_failure("SSE connection failed", base_url, e)
        return None


def _extract_tool_details(
    tools: list[Tool],
) -> list[dict]:
    """Extract tool details from the SDK's tool list.

    SDK 2.x models use snake_case attributes (input_schema); only the JSON
    alias is inputSchema, so getattr(tool, "inputSchema") would find nothing
    and silently store an empty schema.

    Args:
        tools: Tools returned by tools/list, across all pages.

    Returns:
        One dict per tool with name, description, parsed_description, schema,
        plus ui_resource_uri and ui_visibility for a tool with an MCP App page.
    """
    tool_details_list: list[dict[str, Any]] = []

    if tools:
        for tool in tools:
            tool_name = tool.name or "Unknown Name"
            tool_desc = tool.description

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

            tool_schema = tool.input_schema or {}

            tool_detail = {
                "name": tool_name,
                "description": tool_desc or "",
                "parsed_description": parsed_desc,
                "schema": tool_schema,
                # The MCP field name, which the virtual-server mapping, the tool
                # catalog and search read. "schema" stays for ToolDescription.
                "inputSchema": tool_schema,
            }
            # MCP Apps: ui_resource_uri and ui_visibility, only for a tool with a page.
            tool_detail.update(tool_ui_fields(tool))
            tool_details_list.append(tool_detail)

    logger.info(f"Successfully retrieved tool details count={len(tool_details_list):d}")
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

    # Validate the destination BEFORE resolving a credential, matching
    # get_mcp_connection_result below. Resolving first decrypts the stored
    # client_secret and POSTs the operator's token endpoint, so an
    # attacker-registered unsafe proxy_pass_url could drive IdP token minting even
    # though the header builder would then refuse to attach the result. Nothing leaks
    # either way, but there is no reason to do the work -- or the mint -- for a
    # destination we are about to reject.
    if not _assert_mcp_url_fetchable(base_url, server_info):
        return None

    # Resolve+cache an OAuth2 client_credentials bearer (no-op unless
    # auth_scheme == 'oauth') so the synchronous header builder can attach it.
    server_info = await _with_backend_oauth(server_info) if server_info else server_info

    # Use transport-aware detection
    transport = await detect_server_transport_aware(base_url, server_info)

    logger.info(
        f"Attempting MCP connection endpoint={redact_url(base_url)} transport={transport} server_info=true",
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
            f"MCP Check Error: Failed to get tool list from {redact_url(base_url)} with {transport}: {type(e).__name__}"
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
    if not _assert_mcp_url_fetchable(explicit_endpoint or base_url, server_info):
        return None
    if explicit_sse_endpoint and not _assert_mcp_url_fetchable(explicit_sse_endpoint, server_info):
        return None

    # Resolve+cache an OAuth2 client_credentials bearer (no-op unless
    # auth_scheme == 'oauth') so the synchronous header builder can attach it.
    server_info = await _with_backend_oauth(server_info) if server_info else server_info

    # Use transport-aware detection
    transport = await detect_server_transport_aware(base_url, server_info)

    logger.info(
        f"Getting MCP connection result endpoint={redact_url(base_url)} transport={transport}",
    )

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

    if transport == "streamable-http":
        if not _assert_mcp_url_fetchable(mcp_url, server_info):
            return None
        headers = _build_headers_for_server(server_info, destination_url=mcp_url)

    try:
        if transport == "streamable-http":
            sdk_tools, implementation = await _discover_streamable_http(mcp_url, headers)
            return _connection_result(base_url, sdk_tools, implementation)

        elif transport == "sse":
            # For SSE transport
            sse_endpoint = server_info.get("sse_endpoint") if server_info else None
            if sse_endpoint:
                sse_url = sse_endpoint
            else:
                sse_url = base_url.rstrip("/") + "/sse"
            if not _assert_mcp_url_fetchable(sse_url, server_info):
                return None
            headers = _build_headers_for_server(server_info, destination_url=sse_url)

            sdk_tools, implementation = await _discover_sse(sse_url, headers)
            return _connection_result(base_url, sdk_tools, implementation)

        else:
            logger.error(f"Unsupported transport type: {transport}")
            return None

    except TimeoutError:
        logger.error(f"MCP connection timeout endpoint={redact_url(mcp_url)}")
        return None
    except Exception as e:
        _log_discovery_failure("MCP connection-result failure", base_url, e)
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
