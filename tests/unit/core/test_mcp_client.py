"""
Unit tests for registry/core/mcp_client.py

Tests the MCPClientService for tool discovery and server connections.
"""

import contextlib
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from registry.core.mcp_client import (
    MCPClientService,
    _build_headers_for_server,
    _extract_tool_details,
    _get_tools_sse,
    _get_tools_streamable_http,
    detect_server_transport,
    detect_server_transport_aware,
    get_tools_from_server_with_server_info,
    get_tools_from_server_with_transport,
    mcp_client_service,
    normalize_sse_endpoint_url,
    normalize_sse_endpoint_url_for_request,
)

# =============================================================================
# TEST FIXTURES
# =============================================================================


@pytest.fixture
def mock_server_info():
    """Create mock server info."""
    return {
        "server_name": "test-server",
        "supported_transports": ["streamable-http"],
        "headers": [{"X-Custom-Header": "custom-value"}],
        "tags": [],
    }


@pytest.fixture
def mock_tools_response():
    """Create mock tools response from MCP server."""
    mock_tool = MagicMock()
    mock_tool.name = "test_tool"
    mock_tool.description = """Test tool for testing.

    Args:
        param1: First parameter
        param2: Second parameter

    Returns:
        Result of the operation

    Raises:
        ValueError: If parameters are invalid
    """
    mock_tool.inputSchema = {
        "type": "object",
        "properties": {
            "param1": {"type": "string"},
            "param2": {"type": "integer"},
        },
    }

    mock_response = MagicMock()
    mock_response.tools = [mock_tool]
    return mock_response


@pytest.fixture
def mock_client_session():
    """Create mock MCP ClientSession."""
    session = AsyncMock()
    session.initialize = AsyncMock()
    session.list_tools = AsyncMock()
    return session


# =============================================================================
# NORMALIZE_SSE_ENDPOINT_URL TESTS
# =============================================================================


@pytest.mark.unit
def test_normalize_sse_endpoint_url_with_mount_path():
    """Test normalizing SSE endpoint URL with mount path."""
    url = "/fininfo/messages/?session_id=123"
    result = normalize_sse_endpoint_url(url)

    assert result == "/messages/?session_id=123"


@pytest.mark.unit
def test_normalize_sse_endpoint_url_without_mount_path():
    """Test normalizing SSE endpoint URL without mount path."""
    url = "/messages/?session_id=123"
    result = normalize_sse_endpoint_url(url)

    assert result == "/messages/?session_id=123"


@pytest.mark.unit
def test_normalize_sse_endpoint_url_empty():
    """Test normalizing empty SSE endpoint URL."""
    result = normalize_sse_endpoint_url("")

    assert result == ""


@pytest.mark.unit
def test_normalize_sse_endpoint_url_complex_path():
    """Test normalizing complex SSE endpoint URL."""
    url = "/currenttime/messages/?session_id=abc-123&param=value"
    result = normalize_sse_endpoint_url(url)

    assert result == "/messages/?session_id=abc-123&param=value"


# =============================================================================
# NORMALIZE_SSE_ENDPOINT_URL_FOR_REQUEST TESTS
# =============================================================================


@pytest.mark.unit
def test_normalize_sse_endpoint_url_for_request_with_mount():
    """Test normalizing request URL with mount path."""
    url = "http://localhost:8000/currenttime/messages/?session_id=123"
    result = normalize_sse_endpoint_url_for_request(url)

    assert result == "http://localhost:8000/messages/?session_id=123"


@pytest.mark.unit
def test_normalize_sse_endpoint_url_for_request_without_mount():
    """Test normalizing request URL without mount path."""
    url = "http://localhost:8000/messages/?session_id=123"
    result = normalize_sse_endpoint_url_for_request(url)

    assert result == "http://localhost:8000/messages/?session_id=123"


@pytest.mark.unit
def test_normalize_sse_endpoint_url_for_request_api_path():
    """Test normalizing request URL with common API path."""
    url = "http://localhost:8000/api/messages/?session_id=123"
    result = normalize_sse_endpoint_url_for_request(url)

    # Should not normalize 'api' as mount path
    assert result == "http://localhost:8000/api/messages/?session_id=123"


@pytest.mark.unit
def test_normalize_sse_endpoint_url_for_request_no_messages():
    """Test normalizing request URL without /messages/ path."""
    url = "http://localhost:8000/api/data"
    result = normalize_sse_endpoint_url_for_request(url)

    assert result == "http://localhost:8000/api/data"


# =============================================================================
# BUILD_HEADERS_FOR_SERVER TESTS
# =============================================================================


@pytest.mark.unit
def test_build_headers_for_server_with_custom_headers():
    """Test building headers with custom server headers."""
    server_info = {
        "headers": [
            {"X-Custom-1": "value1"},
            {"X-Custom-2": "value2"},
        ]
    }

    headers = _build_headers_for_server(server_info)

    assert "Accept" in headers
    assert "Content-Type" in headers
    assert headers["X-Custom-1"] == "value1"
    assert headers["X-Custom-2"] == "value2"


@pytest.mark.unit
def test_build_headers_for_server_no_custom_headers():
    """Test building headers without custom server headers."""
    headers = _build_headers_for_server(None)

    assert "Accept" in headers
    assert "Content-Type" in headers
    assert headers["Accept"] == "application/json, text/event-stream"


@pytest.mark.unit
def test_build_headers_for_server_empty_headers():
    """Test building headers with empty headers list."""
    server_info = {"headers": []}

    headers = _build_headers_for_server(server_info)

    assert "Accept" in headers
    assert "Content-Type" in headers


# =============================================================================
# DETECT_SERVER_TRANSPORT TESTS
# =============================================================================


@pytest.mark.unit
@pytest.mark.asyncio
async def test_detect_server_transport_explicit_sse():
    """Test detecting transport when URL has /sse endpoint."""
    url = "http://localhost:8000/sse"
    result = await detect_server_transport(url)

    assert result == "sse"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_detect_server_transport_explicit_mcp():
    """Test detecting transport when URL has /mcp endpoint."""
    url = "http://localhost:8000/mcp"
    result = await detect_server_transport(url)

    assert result == "streamable-http"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_detect_server_transport_streamable_http_success():
    """Test detecting transport with successful streamable-http connection."""
    url = "http://localhost:8000"

    with patch("registry.core.mcp_client.streamablehttp_client") as mock_client:
        mock_client.return_value.__aenter__.return_value = MagicMock()

        result = await detect_server_transport(url)

        assert result == "streamable-http"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_detect_server_transport_sse_fallback():
    """Test detecting transport with SSE fallback."""
    url = "http://localhost:8000"

    with patch("registry.core.mcp_client.streamablehttp_client") as mock_streamable:
        mock_streamable.side_effect = Exception("Connection failed")

        with patch("registry.core.mcp_client.sse_client") as mock_sse:
            mock_sse.return_value.__aenter__.return_value = MagicMock()

            result = await detect_server_transport(url)

            assert result == "sse"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_detect_server_transport_default():
    """Test detecting transport defaults to streamable-http."""
    url = "http://localhost:8000"

    with patch("registry.core.mcp_client.streamablehttp_client") as mock_streamable:
        mock_streamable.side_effect = Exception("Connection failed")

        with patch("registry.core.mcp_client.sse_client") as mock_sse:
            mock_sse.side_effect = Exception("Connection failed")

            result = await detect_server_transport(url)

            assert result == "streamable-http"


# =============================================================================
# DETECT_SERVER_TRANSPORT_AWARE TESTS
# =============================================================================


@pytest.mark.unit
@pytest.mark.asyncio
async def test_detect_server_transport_aware_with_config():
    """Test transport detection using server configuration."""
    url = "http://localhost:8000"
    server_info = {"supported_transports": ["sse"]}

    result = await detect_server_transport_aware(url, server_info)

    assert result == "sse"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_detect_server_transport_aware_prefer_streamable():
    """Test transport detection prefers streamable-http."""
    url = "http://localhost:8000"
    server_info = {"supported_transports": ["sse", "streamable-http"]}

    result = await detect_server_transport_aware(url, server_info)

    assert result == "streamable-http"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_detect_server_transport_aware_explicit_url():
    """Test transport detection with explicit URL endpoint."""
    url = "http://localhost:8000/sse"
    server_info = {"supported_transports": ["streamable-http"]}

    result = await detect_server_transport_aware(url, server_info)

    # URL takes precedence
    assert result == "sse"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_detect_server_transport_aware_no_config():
    """Test transport detection without server config."""
    url = "http://localhost:8000"

    with patch("registry.core.mcp_client.detect_server_transport", return_value="streamable-http"):
        result = await detect_server_transport_aware(url, None)

        assert result == "streamable-http"


# =============================================================================
# EXTRACT_TOOL_DETAILS TESTS
# =============================================================================


@pytest.mark.unit
def test_extract_tool_details(mock_tools_response):
    """Test extracting tool details from MCP response."""
    result = _extract_tool_details(mock_tools_response)

    assert len(result) == 1
    assert result[0]["name"] == "test_tool"
    assert "parsed_description" in result[0]
    assert result[0]["parsed_description"]["main"] == "Test tool for testing."
    assert "param1" in result[0]["parsed_description"]["args"]
    assert "schema" in result[0]
    # Verify raw description is also stored
    assert "description" in result[0]
    assert "Test tool for testing" in result[0]["description"]


@pytest.mark.unit
def test_extract_tool_details_no_description():
    """Test extracting tool details with no description."""
    mock_tool = MagicMock()
    mock_tool.name = "simple_tool"
    mock_tool.description = None
    mock_tool.__doc__ = None  # MagicMock has its own __doc__; clear it
    mock_tool.inputSchema = {}

    mock_response = MagicMock()
    mock_response.tools = [mock_tool]

    result = _extract_tool_details(mock_response)

    assert len(result) == 1
    assert result[0]["name"] == "simple_tool"
    assert result[0]["parsed_description"]["main"] == "No description available."


@pytest.mark.unit
def test_extract_tool_details_empty_response():
    """Test extracting tool details from empty response."""
    mock_response = MagicMock()
    mock_response.tools = []

    result = _extract_tool_details(mock_response)

    assert len(result) == 0


@pytest.mark.unit
def test_extract_tool_details_complex_docstring():
    """Test extracting tool details with complex docstring."""
    mock_tool = MagicMock()
    mock_tool.name = "complex_tool"
    mock_tool.description = """
    Main description line 1.
    Main description line 2.

    Args:
        arg1: Description of arg1
        arg2: Description of arg2

    Returns:
        Description of return value

    Raises:
        ValueError: When something goes wrong
        TypeError: When type is incorrect
    """
    mock_tool.inputSchema = {}

    mock_response = MagicMock()
    mock_response.tools = [mock_tool]

    result = _extract_tool_details(mock_response)

    assert len(result) == 1
    parsed = result[0]["parsed_description"]
    assert "Main description" in parsed["main"]
    assert "arg1" in parsed["args"]
    assert "return value" in parsed["returns"]
    assert "ValueError" in parsed["raises"]


# =============================================================================
# GET_TOOLS_STREAMABLE_HTTP TESTS
# =============================================================================


@pytest.mark.unit
@pytest.mark.asyncio
async def test_get_tools_streamable_http_success(mock_server_info, mock_tools_response):
    """Test getting tools via streamable-http successfully."""
    url = "http://localhost:8000/mcp"

    mock_session = AsyncMock()
    mock_session.initialize = AsyncMock()
    mock_session.list_tools = AsyncMock(return_value=mock_tools_response)

    with patch("registry.core.mcp_client.streamablehttp_client") as mock_client:
        mock_client.return_value.__aenter__.return_value = (MagicMock(), MagicMock(), MagicMock())

        with patch("registry.core.mcp_client.ClientSession") as mock_session_class:
            mock_session_class.return_value.__aenter__.return_value = mock_session

            result = await _get_tools_streamable_http(url, mock_server_info)

            assert result is not None
            assert len(result) == 1
            assert result[0]["name"] == "test_tool"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_get_tools_streamable_http_timeout():
    """Test getting tools via streamable-http with timeout."""
    url = "http://localhost:8000/mcp"

    mock_session = AsyncMock()
    mock_session.initialize = AsyncMock(side_effect=TimeoutError())

    with patch("registry.core.mcp_client.streamablehttp_client") as mock_client:
        mock_client.return_value.__aenter__.return_value = (MagicMock(), MagicMock(), MagicMock())

        with patch("registry.core.mcp_client.ClientSession") as mock_session_class:
            mock_session_class.return_value.__aenter__.return_value = mock_session

            result = await _get_tools_streamable_http(url, None)

            assert result is None


@pytest.mark.unit
@pytest.mark.asyncio
async def test_get_tools_streamable_http_anthropic_registry():
    """Test getting tools from Anthropic registry server."""
    url = "http://localhost:8000/mcp"
    server_info = {
        "tags": ["anthropic-registry"],
        "headers": [],
    }

    mock_session = AsyncMock()
    mock_session.initialize = AsyncMock()
    mock_session.list_tools = AsyncMock(return_value=MagicMock(tools=[]))

    with patch("registry.core.mcp_client.streamablehttp_client") as mock_client:
        # Capture the URL passed to streamablehttp_client
        captured_urls = []

        @contextlib.asynccontextmanager
        async def mock_cm(*args, **kwargs):
            captured_urls.append(kwargs.get("url"))
            yield (MagicMock(), MagicMock(), MagicMock())

        mock_client.side_effect = mock_cm

        with patch("registry.core.mcp_client.ClientSession") as mock_session_class:
            mock_session_class.return_value.__aenter__.return_value = mock_session

            await _get_tools_streamable_http(url, server_info)

            # Verify instance_id parameter was added
            assert len(captured_urls) > 0
            assert any("instance_id=default" in u for u in captured_urls)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_get_tools_streamable_http_fallback_endpoints():
    """Test getting tools trying multiple endpoints."""
    url = "http://localhost:8000"

    mock_session = AsyncMock()
    mock_session.initialize = AsyncMock()
    mock_session.list_tools = AsyncMock(return_value=MagicMock(tools=[]))

    call_count = 0

    def mock_client_side_effect(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            # First attempt fails
            raise Exception("Connection failed")
        else:
            # Second attempt succeeds
            return (MagicMock(), MagicMock(), MagicMock())

    with patch("registry.core.mcp_client.streamablehttp_client") as mock_client:
        mock_client.return_value.__aenter__.side_effect = mock_client_side_effect

        with patch("registry.core.mcp_client.ClientSession") as mock_session_class:
            mock_session_class.return_value.__aenter__.return_value = mock_session

            await _get_tools_streamable_http(url, None)

            # Should try /mcp/ first, then / (root)
            assert call_count == 2


# =============================================================================
# GET_TOOLS_SSE TESTS
# =============================================================================


@pytest.mark.unit
@pytest.mark.asyncio
async def test_get_tools_sse_success(mock_tools_response):
    """Test getting tools via SSE successfully."""
    url = "http://localhost:8000/sse"

    mock_session = AsyncMock()
    mock_session.initialize = AsyncMock()
    mock_session.list_tools = AsyncMock(return_value=mock_tools_response)

    with patch("registry.core.mcp_client.sse_client") as mock_client:
        mock_client.return_value.__aenter__.return_value = (MagicMock(), MagicMock())

        with patch("registry.core.mcp_client.ClientSession") as mock_session_class:
            mock_session_class.return_value.__aenter__.return_value = mock_session

            result = await _get_tools_sse(url, None)

            assert result is not None
            assert len(result) == 1


@pytest.mark.unit
@pytest.mark.asyncio
async def test_get_tools_sse_timeout():
    """Test getting tools via SSE with timeout."""
    url = "http://localhost:8000/sse"

    mock_session = AsyncMock()
    mock_session.initialize = AsyncMock(side_effect=TimeoutError())

    with patch("registry.core.mcp_client.sse_client") as mock_client:
        mock_client.return_value.__aenter__.return_value = (MagicMock(), MagicMock())

        with patch("registry.core.mcp_client.ClientSession") as mock_session_class:
            mock_session_class.return_value.__aenter__.return_value = mock_session

            result = await _get_tools_sse(url, None)

            assert result is None


@pytest.mark.unit
@pytest.mark.asyncio
async def test_get_tools_sse_connection_error():
    """Test getting tools via SSE with connection error."""
    url = "http://localhost:8000/sse"

    with patch("registry.core.mcp_client.sse_client") as mock_client:
        mock_client.return_value.__aenter__.side_effect = Exception("Connection failed")

        result = await _get_tools_sse(url, None)

        assert result is None


@pytest.mark.unit
@pytest.mark.asyncio
async def test_get_tools_sse_url_normalization():
    """Test getting tools via SSE with URL normalization."""
    url = "http://localhost:8000"

    mock_session = AsyncMock()
    mock_session.initialize = AsyncMock()
    mock_session.list_tools = AsyncMock(return_value=MagicMock(tools=[]))

    captured_url = None

    @contextlib.asynccontextmanager
    async def mock_cm(url_arg, *args, **kwargs):
        nonlocal captured_url
        captured_url = url_arg
        yield (MagicMock(), MagicMock())

    with patch("registry.core.mcp_client.sse_client") as mock_client:
        mock_client.side_effect = mock_cm

        with patch("registry.core.mcp_client.ClientSession") as mock_session_class:
            mock_session_class.return_value.__aenter__.return_value = mock_session

            await _get_tools_sse(url, None)

            # Should append /sse to URL
            assert captured_url is not None
            assert captured_url.endswith("/sse")


# =============================================================================
# GET_TOOLS_FROM_SERVER_WITH_TRANSPORT TESTS
# =============================================================================


@pytest.mark.unit
@pytest.mark.asyncio
async def test_get_tools_from_server_with_transport_auto():
    """Test getting tools with auto transport detection."""
    url = "http://localhost:8000"

    with patch("registry.core.mcp_client.detect_server_transport", return_value="streamable-http"):
        with patch("registry.core.mcp_client._get_tools_streamable_http", return_value=[]):
            result = await get_tools_from_server_with_transport(url, "auto")

            assert result == []


@pytest.mark.unit
@pytest.mark.asyncio
async def test_get_tools_from_server_with_transport_streamable_http():
    """Test getting tools with explicit streamable-http transport."""
    url = "http://localhost:8000"

    with patch("registry.core.mcp_client._get_tools_streamable_http", return_value=[]) as mock_get:
        result = await get_tools_from_server_with_transport(url, "streamable-http")

        mock_get.assert_awaited_once()
        assert result == []


@pytest.mark.unit
@pytest.mark.asyncio
async def test_get_tools_from_server_with_transport_sse():
    """Test getting tools with explicit SSE transport."""
    url = "http://localhost:8000"

    with patch("registry.core.mcp_client._get_tools_sse", return_value=[]) as mock_get:
        result = await get_tools_from_server_with_transport(url, "sse")

        mock_get.assert_awaited_once()
        assert result == []


@pytest.mark.unit
@pytest.mark.asyncio
async def test_get_tools_from_server_with_transport_unsupported():
    """Test getting tools with unsupported transport."""
    url = "http://localhost:8000"

    result = await get_tools_from_server_with_transport(url, "invalid-transport")

    assert result is None


@pytest.mark.unit
@pytest.mark.asyncio
async def test_get_tools_from_server_with_transport_empty_url():
    """Test getting tools with empty URL."""
    result = await get_tools_from_server_with_transport("", "auto")

    assert result is None


# =============================================================================
# GET_TOOLS_FROM_SERVER_WITH_SERVER_INFO TESTS
# =============================================================================


@pytest.mark.unit
@pytest.mark.asyncio
async def test_get_tools_from_server_with_server_info_success(mock_server_info):
    """Test getting tools with server info successfully."""
    url = "http://localhost:8000"

    with patch(
        "registry.core.mcp_client.detect_server_transport_aware", return_value="streamable-http"
    ):
        with patch(
            "registry.core.mcp_client._get_tools_streamable_http", return_value=[]
        ) as mock_get:
            result = await get_tools_from_server_with_server_info(url, mock_server_info)

            mock_get.assert_awaited_once()
            assert result == []


@pytest.mark.unit
@pytest.mark.asyncio
async def test_get_tools_from_server_with_server_info_empty_url():
    """Test getting tools with server info but empty URL."""
    result = await get_tools_from_server_with_server_info("", {"supported_transports": ["sse"]})

    assert result is None


@pytest.mark.unit
@pytest.mark.asyncio
async def test_get_tools_from_server_with_server_info_exception():
    """Test getting tools with server info when exception occurs in detect_server_transport_aware.

    Note: Due to a bug in mcp_client.py, exceptions from detect_server_transport_aware
    are not caught (it's called before the try block). See:
    .scratchpad/fixes/registry/fix-mcp-client-exception-handling.md

    This test verifies the actual behavior (exception propagates).
    When the bug is fixed, this test should expect result == None instead.
    """
    url = "http://localhost:8000"

    with patch(
        "registry.core.mcp_client.detect_server_transport_aware",
        side_effect=Exception("Test error"),
    ):
        # Actual behavior: exception propagates (not caught)
        # Expected behavior (when bug is fixed): should return None
        with pytest.raises(Exception, match="Test error"):
            await get_tools_from_server_with_server_info(url, None)


# =============================================================================
# MCPCLIENTSERVICE TESTS
# =============================================================================


@pytest.mark.unit
@pytest.mark.asyncio
async def test_mcp_client_service_wrapper(mock_server_info):
    """Test MCPClientService wrapper method."""
    service = MCPClientService()
    url = "http://localhost:8000"

    with patch(
        "registry.core.mcp_client.get_tools_from_server_with_server_info",
        return_value=[{"name": "tool1"}],
    ) as mock_get:
        result = await service.get_tools_from_server_with_server_info(url, mock_server_info)

        mock_get.assert_awaited_once_with(url, mock_server_info)
        assert len(result) == 1
        assert result[0]["name"] == "tool1"


@pytest.mark.unit
def test_mcp_client_service_global_instance():
    """Test that global mcp_client_service instance exists."""
    assert mcp_client_service is not None
    assert isinstance(mcp_client_service, MCPClientService)


# =============================================================================
# INTEGRATION-STYLE TESTS
# =============================================================================


@pytest.mark.unit
@pytest.mark.asyncio
async def test_full_tool_discovery_flow_streamable_http(mock_server_info, mock_tools_response):
    """Test complete tool discovery flow for streamable-http."""
    url = "http://localhost:8000"

    mock_session = AsyncMock()
    mock_session.initialize = AsyncMock()
    mock_session.list_tools = AsyncMock(return_value=mock_tools_response)

    with patch("registry.core.mcp_client.streamablehttp_client") as mock_client:
        mock_client.return_value.__aenter__.return_value = (MagicMock(), MagicMock(), MagicMock())

        with patch("registry.core.mcp_client.ClientSession") as mock_session_class:
            mock_session_class.return_value.__aenter__.return_value = mock_session

            # Full flow: detect transport -> get tools
            with patch(
                "registry.core.mcp_client.detect_server_transport_aware",
                return_value="streamable-http",
            ):
                result = await get_tools_from_server_with_server_info(url, mock_server_info)

                assert result is not None
                assert len(result) == 1
                assert result[0]["name"] == "test_tool"
                assert "parsed_description" in result[0]


@pytest.mark.unit
@pytest.mark.asyncio
async def test_full_tool_discovery_flow_sse(mock_tools_response):
    """Test complete tool discovery flow for SSE."""
    url = "http://localhost:8000"
    server_info = {"supported_transports": ["sse"], "headers": []}

    mock_session = AsyncMock()
    mock_session.initialize = AsyncMock()
    mock_session.list_tools = AsyncMock(return_value=mock_tools_response)

    with patch("registry.core.mcp_client.sse_client") as mock_client:
        mock_client.return_value.__aenter__.return_value = (MagicMock(), MagicMock())

        with patch("registry.core.mcp_client.ClientSession") as mock_session_class:
            mock_session_class.return_value.__aenter__.return_value = mock_session

            # Full flow: detect transport -> get tools
            with patch(
                "registry.core.mcp_client.detect_server_transport_aware", return_value="sse"
            ):
                result = await get_tools_from_server_with_server_info(url, server_info)

                assert result is not None
                assert len(result) == 1
