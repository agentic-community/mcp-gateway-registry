"""
Unit tests for registry/health/service.py

Tests the HealthMonitoringService and HighPerformanceWebSocketManager.
"""

import asyncio
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import httpx
import pytest
from fastapi import WebSocket

from registry.constants import HealthStatus
from registry.health.service import (
    HealthMonitoringService,
    HighPerformanceWebSocketManager,
)

# =============================================================================
# TEST FIXTURES
# =============================================================================


@pytest.fixture
def mock_websocket():
    """Create a mock WebSocket connection."""
    ws = AsyncMock(spec=WebSocket)
    ws.client = MagicMock()
    ws.client.host = "127.0.0.1"
    ws.accept = AsyncMock()
    ws.send_text = AsyncMock()
    ws.close = AsyncMock()
    return ws


@pytest.fixture
def ws_manager():
    """Create a HighPerformanceWebSocketManager instance."""
    return HighPerformanceWebSocketManager()


@pytest.fixture
def health_service():
    """Create a HealthMonitoringService instance."""
    service = HealthMonitoringService()
    return service


@pytest.fixture
def mock_server_info():
    """Create mock server info."""
    return {
        "server_name": "test-server",
        "proxy_pass_url": "http://localhost:8000/mcp",
        "supported_transports": ["streamable-http"],
        "headers": [{"X-Test-Header": "test-value"}],
        "tool_list": [{"name": "test_tool", "description": "A test tool"}],
        "num_tools": 1,
        "is_enabled": True,
    }


# =============================================================================
# HIGHPERFORMANCEWEBSOCKETMANAGER TESTS
# =============================================================================


@pytest.mark.unit
@pytest.mark.asyncio
async def test_ws_manager_add_connection_success(ws_manager, mock_websocket):
    """Test adding a WebSocket connection successfully."""
    with patch.object(ws_manager, "_send_initial_status_optimized", new=AsyncMock()):
        success = await ws_manager.add_connection(mock_websocket)

        assert success is True
        assert mock_websocket in ws_manager.connections
        assert mock_websocket in ws_manager.connection_metadata
        mock_websocket.accept.assert_awaited_once()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_ws_manager_add_connection_at_capacity(ws_manager, mock_settings):
    """Test adding connection when at capacity limit."""
    # Set low limit for testing
    mock_settings.max_websocket_connections = 1

    with patch("registry.health.service.settings", mock_settings):
        ws1 = AsyncMock(spec=WebSocket)
        ws1.client = MagicMock(host="127.0.0.1")
        ws2 = AsyncMock(spec=WebSocket)
        ws2.client = MagicMock(host="127.0.0.2")

        with patch.object(ws_manager, "_send_initial_status_optimized", new=AsyncMock()):
            # Add first connection - should succeed
            success1 = await ws_manager.add_connection(ws1)
            assert success1 is True

            # Add second connection - should fail
            success2 = await ws_manager.add_connection(ws2)
            assert success2 is False
            ws2.close.assert_awaited_once()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_ws_manager_remove_connection(ws_manager, mock_websocket):
    """Test removing a WebSocket connection."""
    ws_manager.connections.add(mock_websocket)
    ws_manager.connection_metadata[mock_websocket] = {"connected_at": 123456}

    await ws_manager.remove_connection(mock_websocket)

    assert mock_websocket not in ws_manager.connections
    assert mock_websocket not in ws_manager.connection_metadata


@pytest.mark.unit
@pytest.mark.asyncio
async def test_ws_manager_broadcast_update_no_connections(ws_manager):
    """Test broadcast with no active connections."""
    await ws_manager.broadcast_update("test-path", {"status": "healthy"})
    # Should not raise any errors


@pytest.mark.unit
@pytest.mark.asyncio
async def test_ws_manager_broadcast_update_rate_limiting(ws_manager, mock_websocket, mock_settings):
    """Test that broadcasts are rate-limited."""
    mock_settings.websocket_broadcast_interval_ms = 1000  # 1 second

    with patch("registry.health.service.settings", mock_settings):
        ws_manager.connections.add(mock_websocket)

        # First broadcast should go through
        await ws_manager.broadcast_update("test-path", {"status": "healthy"})

        # Immediate second broadcast should be queued (not sent)
        await ws_manager.broadcast_update("test-path-2", {"status": "unhealthy"})

        # Check that update was queued
        assert "test-path-2" in ws_manager.pending_updates


@pytest.mark.unit
@pytest.mark.asyncio
async def test_ws_manager_safe_send_message_success(ws_manager, mock_websocket):
    """Test safe message sending."""
    message = "test message"
    result = await ws_manager._safe_send_message(mock_websocket, message)

    assert result is True
    mock_websocket.send_text.assert_awaited_once_with(message)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_ws_manager_safe_send_message_timeout(ws_manager, mock_websocket):
    """Test safe message sending with timeout."""
    mock_websocket.send_text.side_effect = TimeoutError()

    result = await ws_manager._safe_send_message(mock_websocket, "test")

    assert isinstance(result, TimeoutError)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_ws_manager_send_to_connections_optimized(ws_manager):
    """Test optimized sending to multiple connections."""
    # Create mock connections
    connections = []
    for i in range(5):
        ws = AsyncMock(spec=WebSocket)
        ws.client = MagicMock(host=f"127.0.0.{i}")
        connections.append(ws)
        ws_manager.connections.add(ws)

    data = {"test": "data"}

    with patch.object(ws_manager, "_safe_send_message", return_value=True) as mock_send:
        await ws_manager._send_to_connections_optimized(data)

        # Should have sent to all connections
        assert mock_send.call_count == len(connections)


@pytest.mark.unit
def test_ws_manager_get_stats(ws_manager):
    """Test getting WebSocket manager statistics."""
    ws_manager.broadcast_count = 10
    ws_manager.failed_send_count = 2

    stats = ws_manager.get_stats()

    assert stats["active_connections"] == 0
    assert stats["total_broadcasts"] == 10
    assert stats["failed_sends"] == 2


# =============================================================================
# HEALTHMONITORINGSERVICE TESTS
# =============================================================================


@pytest.mark.unit
@pytest.mark.asyncio
async def test_health_service_initialize(health_service):
    """Test health service initialization."""
    with patch.object(health_service, "_run_health_checks", return_value=AsyncMock()):
        await health_service.initialize()

        assert health_service.health_check_task is not None


@pytest.mark.unit
@pytest.mark.asyncio
async def test_health_service_shutdown(health_service):
    """Test health service shutdown."""

    # Create a proper asyncio Task
    async def dummy_task():
        while True:
            await asyncio.sleep(1)

    # Create and immediately cancel the task
    task = asyncio.create_task(dummy_task())
    health_service.health_check_task = task

    # Add mock connections
    mock_ws = AsyncMock(spec=WebSocket)
    mock_ws.close = AsyncMock()
    health_service.websocket_manager.connections.add(mock_ws)

    await health_service.shutdown()

    # Task should be cancelled
    assert task.cancelled()
    mock_ws.close.assert_awaited_once()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_health_service_add_websocket_connection(health_service, mock_websocket):
    """Test adding WebSocket connection to health service."""
    with patch.object(
        health_service.websocket_manager, "add_connection", return_value=True
    ) as mock_add:
        success = await health_service.add_websocket_connection(mock_websocket)

        assert success is True
        mock_add.assert_awaited_once_with(mock_websocket)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_health_service_remove_websocket_connection(health_service, mock_websocket):
    """Test removing WebSocket connection from health service."""
    with patch.object(health_service.websocket_manager, "remove_connection") as mock_remove:
        await health_service.remove_websocket_connection(mock_websocket)

        mock_remove.assert_awaited_once_with(mock_websocket)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_health_service_broadcast_health_update_no_connections(health_service):
    """Test broadcasting health update with no connections."""
    # Should not raise any errors
    await health_service.broadcast_health_update()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_health_service_broadcast_health_update_specific_service(
    health_service, mock_server_info
):
    """Test broadcasting health update for specific service."""
    service_path = "/test-server"

    with patch("registry.services.server_service.server_service") as mock_server_service:
        mock_server_service.get_server_info = AsyncMock(return_value=mock_server_info)

        # Add a mock connection
        mock_ws = AsyncMock(spec=WebSocket)
        health_service.websocket_manager.connections.add(mock_ws)

        with patch.object(health_service.websocket_manager, "broadcast_update") as mock_broadcast:
            await health_service.broadcast_health_update(service_path)

            mock_broadcast.assert_awaited_once()
            # Check that service_path was passed
            args = mock_broadcast.call_args
            assert args[0][0] == service_path


@pytest.mark.unit
@pytest.mark.asyncio
async def test_health_service_get_cached_health_data(health_service):
    """Test getting cached health data."""
    with patch("registry.services.server_service.server_service") as mock_server_service:
        mock_server_service.get_all_servers = AsyncMock(
            return_value={"/test-server": {"server_name": "test", "proxy_pass_url": "http://test"}}
        )

        data = await health_service._get_cached_health_data()

        assert isinstance(data, dict)
        assert "/test-server" in data


@pytest.mark.unit
def test_health_service_get_websocket_stats(health_service):
    """Test getting WebSocket statistics."""
    health_service.websocket_manager.broadcast_count = 5

    stats = health_service.get_websocket_stats()

    assert "active_connections" in stats
    assert "total_broadcasts" in stats


@pytest.mark.unit
@pytest.mark.asyncio
async def test_health_service_check_server_endpoint_transport_aware_healthy(
    health_service, mock_server_info
):
    """Test checking server endpoint that is healthy."""
    proxy_url = "http://localhost:8000/mcp"

    mock_client = AsyncMock(spec=httpx.AsyncClient)
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_client.post.return_value = mock_response

    with patch.object(health_service, "_initialize_mcp_session", return_value="session-123"):
        is_healthy, status = await health_service._check_server_endpoint_transport_aware(
            mock_client, proxy_url, mock_server_info
        )

        assert is_healthy is True
        assert status == HealthStatus.HEALTHY


@pytest.mark.unit
@pytest.mark.asyncio
async def test_health_service_check_server_endpoint_missing_url(health_service, mock_server_info):
    """Test checking server endpoint with missing URL."""
    mock_client = AsyncMock(spec=httpx.AsyncClient)

    is_healthy, status = await health_service._check_server_endpoint_transport_aware(
        mock_client, "", mock_server_info
    )

    assert is_healthy is False
    assert status == HealthStatus.UNHEALTHY_MISSING_PROXY_URL


@pytest.mark.unit
@pytest.mark.asyncio
async def test_health_service_check_server_endpoint_stdio_transport(
    health_service, mock_server_info
):
    """Test checking server with stdio transport (should skip check)."""
    mock_server_info["supported_transports"] = ["stdio"]
    mock_client = AsyncMock(spec=httpx.AsyncClient)

    is_healthy, status = await health_service._check_server_endpoint_transport_aware(
        mock_client, "http://localhost:8000", mock_server_info
    )

    assert is_healthy is True
    assert status == HealthStatus.UNKNOWN


@pytest.mark.unit
def test_health_service_build_headers_for_server(health_service, mock_server_info):
    """Test building headers for server requests."""
    headers = health_service._build_headers_for_server(mock_server_info)

    assert "Accept" in headers
    assert "Content-Type" in headers
    assert headers["X-Test-Header"] == "test-value"


@pytest.mark.unit
def test_health_service_build_headers_with_session_id(health_service, mock_server_info):
    """Test building headers with session ID."""
    headers = health_service._build_headers_for_server(mock_server_info, include_session_id=True)

    assert "Mcp-Session-Id" in headers
    # Should be a valid UUID
    import uuid

    try:
        uuid.UUID(headers["Mcp-Session-Id"])
        assert True
    except ValueError:
        pytest.fail("Session ID is not a valid UUID")


@pytest.mark.unit
@pytest.mark.asyncio
async def test_health_service_initialize_mcp_session_success(health_service):
    """Test initializing MCP session successfully."""
    mock_client = AsyncMock(spec=httpx.AsyncClient)
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.headers = {"Mcp-Session-Id": "server-session-123"}
    mock_client.post.return_value = mock_response

    session_id = await health_service._initialize_mcp_session(
        mock_client, "http://localhost:8000/mcp", {}
    )

    assert session_id == "server-session-123"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_health_service_initialize_mcp_session_failure(health_service):
    """Test initializing MCP session with failure."""
    mock_client = AsyncMock(spec=httpx.AsyncClient)
    mock_response = MagicMock()
    mock_response.status_code = 500
    mock_response.text = "Internal Server Error"
    mock_client.post.return_value = mock_response

    session_id = await health_service._initialize_mcp_session(
        mock_client, "http://localhost:8000/mcp", {}
    )

    assert session_id is None


@pytest.mark.unit
@pytest.mark.asyncio
async def test_health_service_try_ping_without_auth_success(health_service):
    """Test ping without auth when server is reachable."""
    mock_client = AsyncMock(spec=httpx.AsyncClient)
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_client.post.return_value = mock_response

    result = await health_service._try_ping_without_auth(mock_client, "http://localhost:8000/mcp")

    assert result is True


@pytest.mark.unit
@pytest.mark.asyncio
async def test_health_service_try_ping_without_auth_failure(health_service):
    """Test ping without auth when server is unreachable."""
    mock_client = AsyncMock(spec=httpx.AsyncClient)
    mock_client.post.side_effect = httpx.ConnectError("Connection refused")

    result = await health_service._try_ping_without_auth(mock_client, "http://localhost:8000/mcp")

    assert result is False


@pytest.mark.unit
def test_health_service_is_mcp_endpoint_healthy_200(health_service):
    """Test MCP endpoint health check with 200 status."""
    mock_response = MagicMock()
    mock_response.status_code = 200

    result = health_service._is_mcp_endpoint_healthy(mock_response)

    assert result is True


@pytest.mark.unit
def test_health_service_is_mcp_endpoint_healthy_400_with_session_error(health_service):
    """Test MCP endpoint health check with 400 and session error."""
    mock_response = MagicMock()
    mock_response.status_code = 400
    mock_response.json.return_value = {
        "jsonrpc": "2.0",
        "id": "server-error",
        "error": {"code": -32600, "message": "Missing session ID"},
    }

    result = health_service._is_mcp_endpoint_healthy(mock_response)

    assert result is True


@pytest.mark.unit
def test_health_service_is_mcp_endpoint_healthy_streamable_200(health_service):
    """Test streamable-http endpoint health check with 200 status."""
    mock_response = MagicMock()
    mock_response.status_code = 200

    result = health_service._is_mcp_endpoint_healthy_streamable(mock_response)

    assert result is True


@pytest.mark.unit
def test_health_service_is_mcp_endpoint_healthy_streamable_400_with_jsonrpc_error(
    health_service,
):
    """Test streamable-http endpoint health check with 400 and JSON-RPC error."""
    mock_response = MagicMock()
    mock_response.status_code = 400
    mock_response.json.return_value = {"error": {"code": -32600}}

    result = health_service._is_mcp_endpoint_healthy_streamable(mock_response)

    assert result is True


@pytest.mark.unit
@pytest.mark.asyncio
async def test_health_service_perform_immediate_health_check(health_service, mock_server_info):
    """Test performing immediate health check."""
    service_path = "/test-server"

    with patch("registry.services.server_service.server_service") as mock_server_service:
        mock_server_service.get_server_info = AsyncMock(return_value=mock_server_info)
        mock_server_service.get_enabled_services = AsyncMock(return_value=[service_path])

        with patch.object(
            health_service,
            "_check_server_endpoint_transport_aware",
            return_value=(True, HealthStatus.HEALTHY),
        ):
            with patch("registry.core.nginx_service.nginx_service") as mock_nginx:
                mock_nginx.generate_config_async = AsyncMock()

                status, last_checked = await health_service.perform_immediate_health_check(
                    service_path
                )

                assert status == HealthStatus.HEALTHY
                assert isinstance(last_checked, datetime)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_health_service_check_single_service_status_changed(health_service, mock_server_info):
    """Test checking single service when status changes."""
    service_path = "/test-server"
    health_service.server_health_status[service_path] = HealthStatus.UNHEALTHY_TIMEOUT

    mock_client = AsyncMock(spec=httpx.AsyncClient)

    with patch.object(
        health_service,
        "_check_server_endpoint_transport_aware",
        return_value=(True, HealthStatus.HEALTHY),
    ):
        with patch.object(health_service, "_update_tools_background"):
            status_changed = await health_service._check_single_service(
                mock_client, service_path, mock_server_info
            )

            assert status_changed is True
            assert health_service.server_health_status[service_path] == HealthStatus.HEALTHY


@pytest.mark.unit
@pytest.mark.asyncio
async def test_health_service_update_tools_background(health_service, mock_server_info):
    """Test updating tools in background."""
    service_path = "/test-server"
    proxy_url = "http://localhost:8000/mcp"

    # Mock the server_info to not have tool_list initially
    mock_server_info_copy = mock_server_info.copy()
    mock_server_info_copy["tool_list"] = []
    mock_server_info_copy["num_tools"] = 0

    with patch("registry.core.mcp_client.mcp_client_service") as mock_mcp:
        mock_mcp.get_mcp_connection_result = AsyncMock(
            return_value={
                "tools": [{"name": "test_tool", "description": "Test"}],
                "server_info": {"name": "test-server", "version": "1.0.0"},
            }
        )

        with patch("registry.services.server_service.server_service") as mock_server_service:
            # First call returns server info without tools, second call returns it with tools
            mock_server_service.get_server_info = AsyncMock(return_value=mock_server_info_copy)
            mock_server_service.update_server = AsyncMock()

            with patch("registry.utils.scopes_manager.update_server_scopes", new=AsyncMock()):
                # Add small sleep to allow background coroutine to run
                await health_service._update_tools_background(service_path, proxy_url)
                await asyncio.sleep(0.01)

                # Should have called update_server
                mock_server_service.update_server.assert_called_once()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_health_service_get_all_health_status(health_service, mock_server_info):
    """Test getting all health status."""
    with patch("registry.services.server_service.server_service") as mock_server_service:
        mock_server_service.get_all_servers = AsyncMock(
            return_value={"/test-server": mock_server_info}
        )

        all_status = await health_service.get_all_health_status()

        assert isinstance(all_status, dict)
        assert "/test-server" in all_status
        assert "status" in all_status["/test-server"]


@pytest.mark.unit
def test_health_service_get_service_health_data_fast(health_service, mock_server_info):
    """Test getting service health data fast."""
    service_path = "/test-server"
    health_service.server_health_status[service_path] = HealthStatus.HEALTHY

    health_data = health_service._get_service_health_data_fast(service_path, mock_server_info)

    assert health_data["status"] == HealthStatus.HEALTHY
    assert health_data["num_tools"] == 1


@pytest.mark.unit
def test_health_service_get_service_health_data_disabled(health_service, mock_server_info):
    """Test getting service health data for disabled service."""
    service_path = "/test-server"

    # Set is_enabled to False in server_info
    mock_server_info["is_enabled"] = False

    health_data = health_service._get_service_health_data_fast(service_path, mock_server_info)

    assert health_data["status"] == "disabled"


# =============================================================================
# ADDITIONAL TESTS FOR MISSING COVERAGE
# =============================================================================


@pytest.mark.unit
@pytest.mark.asyncio
async def test_ws_manager_add_connection_exception(ws_manager, mock_websocket):
    """Test adding connection when exception occurs."""
    mock_websocket.accept.side_effect = Exception("Connection error")

    success = await ws_manager.add_connection(mock_websocket)

    assert success is False


@pytest.mark.unit
@pytest.mark.asyncio
async def test_ws_manager_send_initial_status_optimized_with_cached_data(
    ws_manager, mock_websocket
):
    """Test sending initial status with cached data."""
    with patch("registry.health.service.health_service") as mock_health_service:
        mock_health_service._get_cached_health_data = AsyncMock(return_value={"test": "data"})

        await ws_manager._send_initial_status_optimized(mock_websocket)

        mock_websocket.send_text.assert_awaited_once()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_ws_manager_send_initial_status_optimized_exception(ws_manager, mock_websocket):
    """Test sending initial status when exception occurs."""
    mock_websocket.send_text.side_effect = Exception("Send failed")

    with patch("registry.health.service.health_service") as mock_health_service:
        mock_health_service._get_cached_health_data = AsyncMock(return_value={"test": "data"})
        with patch.object(ws_manager, "remove_connection", new=AsyncMock()) as mock_remove:
            await ws_manager._send_initial_status_optimized(mock_websocket)

            mock_remove.assert_awaited_once_with(mock_websocket)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_ws_manager_broadcast_update_single_service(ws_manager, mock_websocket):
    """Test broadcast update for single service."""
    ws_manager.connections.add(mock_websocket)
    ws_manager.last_broadcast_time = 0

    with patch.object(ws_manager, "_send_to_connections_optimized", new=AsyncMock()) as mock_send:
        await ws_manager.broadcast_update("test-path", {"status": "healthy"})

        mock_send.assert_awaited_once()
        call_args = mock_send.call_args[0][0]
        assert "test-path" in call_args


@pytest.mark.unit
@pytest.mark.asyncio
async def test_ws_manager_broadcast_update_with_pending_updates(
    ws_manager, mock_websocket, mock_settings
):
    """Test broadcast update with pending updates batch."""
    mock_settings.websocket_broadcast_interval_ms = 10
    mock_settings.websocket_max_batch_size = 5

    with patch("registry.health.service.settings", mock_settings):
        ws_manager.connections.add(mock_websocket)
        ws_manager.last_broadcast_time = 0
        ws_manager.pending_updates = {
            "path1": {"status": "healthy"},
            "path2": {"status": "unhealthy"},
        }

        with patch.object(
            ws_manager, "_send_to_connections_optimized", new=AsyncMock()
        ) as mock_send:
            await ws_manager.broadcast_update()

            mock_send.assert_awaited_once()
            # Pending updates should be sent
            call_args = mock_send.call_args[0][0]
            assert "path1" in call_args or "path2" in call_args


@pytest.mark.unit
@pytest.mark.asyncio
async def test_ws_manager_broadcast_update_full_status(ws_manager, mock_websocket):
    """Test broadcast update with full status when no pending updates."""
    ws_manager.connections.add(mock_websocket)
    ws_manager.last_broadcast_time = 0

    with patch("registry.health.service.health_service") as mock_health_service:
        mock_health_service._get_cached_health_data = AsyncMock(return_value={"full": "status"})

        with patch.object(
            ws_manager, "_send_to_connections_optimized", new=AsyncMock()
        ) as mock_send:
            await ws_manager.broadcast_update()

            mock_send.assert_awaited_once()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_ws_manager_send_to_connections_no_connections(ws_manager):
    """Test sending to connections when no connections exist."""
    data = {"test": "data"}

    # Should not raise any errors
    await ws_manager._send_to_connections_optimized(data)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_ws_manager_send_to_connections_with_failures(ws_manager):
    """Test sending to connections with some failures."""
    # Create connections where some will fail
    good_ws = AsyncMock(spec=WebSocket)
    good_ws.client = MagicMock(host="127.0.0.1")
    bad_ws = AsyncMock(spec=WebSocket)
    bad_ws.client = MagicMock(host="127.0.0.2")

    ws_manager.connections.add(good_ws)
    ws_manager.connections.add(bad_ws)

    data = {"test": "data"}

    with patch.object(ws_manager, "_safe_send_message") as mock_send:
        mock_send.side_effect = [True, Exception("Send failed")]

        with patch.object(ws_manager, "_cleanup_failed_connections", new=AsyncMock()):
            await ws_manager._send_to_connections_optimized(data)

            assert len(ws_manager.failed_connections) > 0


@pytest.mark.unit
@pytest.mark.asyncio
async def test_ws_manager_cleanup_failed_connections(ws_manager):
    """Test cleanup of failed connections."""
    mock_ws = AsyncMock(spec=WebSocket)
    ws_manager.connections.add(mock_ws)
    ws_manager.failed_connections.add(mock_ws)

    await ws_manager._cleanup_failed_connections()

    assert mock_ws not in ws_manager.connections
    assert len(ws_manager.failed_connections) == 0


@pytest.mark.unit
@pytest.mark.asyncio
async def test_ws_manager_cleanup_failed_connections_empty(ws_manager):
    """Test cleanup with no failed connections."""
    # Should not raise any errors
    await ws_manager._cleanup_failed_connections()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_ws_manager_safe_send_message_exception(ws_manager, mock_websocket):
    """Test safe send message with general exception."""
    mock_websocket.send_text.side_effect = RuntimeError("Connection closed")

    result = await ws_manager._safe_send_message(mock_websocket, "test")

    assert isinstance(result, Exception)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_health_service_shutdown_no_task(health_service):
    """Test shutdown when no health check task exists."""
    health_service.health_check_task = None

    # Should not raise any errors
    await health_service.shutdown()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_health_service_shutdown_with_connection_errors(health_service):
    """Test shutdown with connection close errors."""
    mock_ws1 = AsyncMock(spec=WebSocket)
    mock_ws1.close.side_effect = Exception("Close failed")
    mock_ws2 = AsyncMock(spec=WebSocket)
    mock_ws2.close = AsyncMock()

    health_service.websocket_manager.connections.add(mock_ws1)
    health_service.websocket_manager.connections.add(mock_ws2)

    # Should handle exceptions gracefully
    await health_service.shutdown()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_health_service_add_websocket_connection_failure(health_service, mock_websocket):
    """Test adding WebSocket connection when it fails."""
    with patch.object(health_service.websocket_manager, "add_connection", return_value=False):
        success = await health_service.add_websocket_connection(mock_websocket)

        assert success is False


@pytest.mark.unit
@pytest.mark.asyncio
async def test_health_service_broadcast_health_update_full(health_service):
    """Test broadcasting full health update."""
    mock_ws = AsyncMock(spec=WebSocket)
    health_service.websocket_manager.connections.add(mock_ws)

    with patch.object(health_service.websocket_manager, "broadcast_update") as mock_broadcast:
        await health_service.broadcast_health_update()

        mock_broadcast.assert_awaited_once_with()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_health_service_broadcast_health_update_no_server_info(health_service):
    """Test broadcasting health update when server info not found."""
    service_path = "/missing-server"
    mock_ws = AsyncMock(spec=WebSocket)
    health_service.websocket_manager.connections.add(mock_ws)

    with patch("registry.services.server_service.server_service") as mock_server_service:
        mock_server_service.get_server_info = AsyncMock(return_value=None)

        # Should not raise errors
        await health_service.broadcast_health_update(service_path)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_health_service_get_cached_health_data_with_valid_cache(health_service):
    """Test getting cached health data when cache is still valid."""
    from time import time

    # Set up valid cache
    health_service._cached_health_data = {"test": "data"}
    health_service._cache_timestamp = time()

    data = await health_service._get_cached_health_data()

    assert data == {"test": "data"}


@pytest.mark.unit
@pytest.mark.asyncio
async def test_health_service_run_health_checks_loop(health_service):
    """Test health check loop execution."""
    call_count = 0

    async def mock_perform_health_checks():
        nonlocal call_count
        call_count += 1
        if call_count >= 2:
            # Raise CancelledError directly to stop the loop
            raise asyncio.CancelledError()

    with patch.object(
        health_service, "_perform_health_checks", side_effect=mock_perform_health_checks
    ):
        with patch("asyncio.sleep", new=AsyncMock()):
            try:
                await health_service._run_health_checks()
            except asyncio.CancelledError:
                pass

            assert call_count >= 2


@pytest.mark.unit
@pytest.mark.asyncio
async def test_health_service_run_health_checks_with_exception(health_service, mock_settings):
    """Test health check loop handles exceptions."""
    mock_settings.health_check_interval_seconds = 0.01

    call_count = 0

    async def mock_perform_with_error():
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise Exception("Health check error")
        else:
            # Raise CancelledError directly to stop the loop after error recovery
            raise asyncio.CancelledError()

    with patch("registry.health.service.settings", mock_settings):
        with patch.object(
            health_service, "_perform_health_checks", side_effect=mock_perform_with_error
        ):
            with patch("asyncio.sleep", new=AsyncMock()):
                try:
                    await health_service._run_health_checks()
                except asyncio.CancelledError:
                    pass

                assert call_count >= 1


@pytest.mark.unit
@pytest.mark.asyncio
async def test_health_service_perform_health_checks_no_services(health_service):
    """Test performing health checks when no services are enabled."""
    with patch("registry.services.server_service.server_service") as mock_server_service:
        mock_server_service.get_enabled_services = AsyncMock(return_value=[])

        # Should not raise errors
        await health_service._perform_health_checks()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_health_service_perform_health_checks_many_services(health_service, mock_server_info):
    """Test performing health checks on many services."""
    with patch("registry.services.server_service.server_service") as mock_server_service:
        # Multiple services to trigger debug logging
        mock_server_service.get_enabled_services = AsyncMock(
            return_value=["/service1", "/service2", "/service3"]
        )
        mock_server_service.get_server_info = AsyncMock(return_value=mock_server_info)

        with patch.object(health_service, "_check_single_service", return_value=False):
            await health_service._perform_health_checks()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_health_service_perform_health_checks_status_changed(
    health_service, mock_server_info
):
    """Test performing health checks when status changes."""
    with patch("registry.services.server_service.server_service") as mock_server_service:
        mock_server_service.get_enabled_services = AsyncMock(return_value=["/test-server"])
        mock_server_service.get_server_info = AsyncMock(return_value=mock_server_info)

        with patch.object(health_service, "_check_single_service", return_value=True):
            with patch.object(
                health_service, "broadcast_health_update", new=AsyncMock()
            ) as mock_broadcast:
                with patch("registry.core.nginx_service.nginx_service") as mock_nginx:
                    mock_nginx.generate_config_async = AsyncMock()

                    await health_service._perform_health_checks()

                    mock_broadcast.assert_awaited_once()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_health_service_perform_health_checks_nginx_error(health_service, mock_server_info):
    """Test performing health checks when nginx regeneration fails."""
    with patch("registry.services.server_service.server_service") as mock_server_service:
        mock_server_service.get_enabled_services = AsyncMock(return_value=["/test-server"])
        mock_server_service.get_server_info = AsyncMock(return_value=mock_server_info)

        with patch.object(health_service, "_check_single_service", return_value=True):
            with patch.object(health_service, "broadcast_health_update", new=AsyncMock()):
                with patch("registry.core.nginx_service.nginx_service") as mock_nginx:
                    mock_nginx.generate_config_async = AsyncMock(
                        side_effect=Exception("Nginx error")
                    )

                    # Should handle exception gracefully
                    await health_service._perform_health_checks()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_health_service_check_single_service_timeout(health_service, mock_server_info):
    """Test checking single service with timeout."""
    service_path = "/test-server"
    mock_client = AsyncMock(spec=httpx.AsyncClient)

    with patch.object(
        health_service,
        "_check_server_endpoint_transport_aware",
        side_effect=httpx.TimeoutException("Timeout"),
    ):
        await health_service._check_single_service(mock_client, service_path, mock_server_info)

        assert health_service.server_health_status[service_path] == HealthStatus.UNHEALTHY_TIMEOUT


@pytest.mark.unit
@pytest.mark.asyncio
async def test_health_service_check_single_service_connection_error(
    health_service, mock_server_info
):
    """Test checking single service with connection error."""
    service_path = "/test-server"
    mock_client = AsyncMock(spec=httpx.AsyncClient)

    with patch.object(
        health_service,
        "_check_server_endpoint_transport_aware",
        side_effect=httpx.ConnectError("Connection failed"),
    ):
        await health_service._check_single_service(mock_client, service_path, mock_server_info)

        assert (
            health_service.server_health_status[service_path]
            == HealthStatus.UNHEALTHY_CONNECTION_ERROR
        )


@pytest.mark.unit
@pytest.mark.asyncio
async def test_health_service_check_single_service_generic_error(health_service, mock_server_info):
    """Test checking single service with generic error."""
    service_path = "/test-server"
    mock_client = AsyncMock(spec=httpx.AsyncClient)

    with patch.object(
        health_service,
        "_check_server_endpoint_transport_aware",
        side_effect=ValueError("Something went wrong"),
    ):
        await health_service._check_single_service(mock_client, service_path, mock_server_info)

        assert "error: ValueError" in health_service.server_health_status[service_path]


@pytest.mark.unit
@pytest.mark.asyncio
async def test_health_service_check_single_service_first_time_healthy(
    health_service, mock_server_info
):
    """Test checking service for the first time when healthy."""
    service_path = "/test-server"
    health_service.server_health_status[service_path] = HealthStatus.UNKNOWN

    mock_client = AsyncMock(spec=httpx.AsyncClient)

    with patch.object(
        health_service,
        "_check_server_endpoint_transport_aware",
        return_value=(True, HealthStatus.HEALTHY),
    ):
        with patch.object(health_service, "_update_tools_background"):
            status_changed = await health_service._check_single_service(
                mock_client, service_path, mock_server_info
            )

            # Should trigger tool fetch on first healthy check
            assert status_changed is True


@pytest.mark.unit
@pytest.mark.asyncio
async def test_health_service_check_single_service_transition_to_healthy(
    health_service, mock_server_info
):
    """Test service transitioning from unhealthy to healthy."""
    service_path = "/test-server"
    health_service.server_health_status[service_path] = HealthStatus.UNHEALTHY_TIMEOUT

    mock_client = AsyncMock(spec=httpx.AsyncClient)

    with patch.object(
        health_service,
        "_check_server_endpoint_transport_aware",
        return_value=(True, HealthStatus.HEALTHY),
    ):
        with patch.object(health_service, "_update_tools_background"):
            status_changed = await health_service._check_single_service(
                mock_client, service_path, mock_server_info
            )

            assert status_changed is True


@pytest.mark.unit
@pytest.mark.asyncio
async def test_health_service_check_single_service_already_healthy_no_tools(
    health_service, mock_server_info
):
    """Test service that is already healthy but has no tools."""
    service_path = "/test-server"
    health_service.server_health_status[service_path] = HealthStatus.HEALTHY

    # Remove tools from server info
    mock_server_info_no_tools = mock_server_info.copy()
    mock_server_info_no_tools["tool_list"] = []

    mock_client = AsyncMock(spec=httpx.AsyncClient)

    with patch.object(
        health_service,
        "_check_server_endpoint_transport_aware",
        return_value=(True, HealthStatus.HEALTHY),
    ):
        with patch.object(health_service, "_update_tools_background"):
            status_changed = await health_service._check_single_service(
                mock_client, service_path, mock_server_info_no_tools
            )

            # Should still fetch tools if none exist
            assert status_changed is False


@pytest.mark.unit
def test_health_service_build_headers_for_server_no_headers(health_service):
    """Test building headers when server has no custom headers."""
    server_info = {
        "server_name": "test-server",
        "proxy_pass_url": "http://localhost:8000/mcp",
    }

    headers = health_service._build_headers_for_server(server_info)

    assert "Accept" in headers
    assert "Content-Type" in headers


@pytest.mark.unit
def test_health_service_build_headers_for_server_invalid_headers(health_service):
    """Test building headers when server has invalid headers."""
    server_info = {
        "server_name": "test-server",
        "proxy_pass_url": "http://localhost:8000/mcp",
        "headers": "invalid_string",
    }

    headers = health_service._build_headers_for_server(server_info)

    # Should still return base headers
    assert "Accept" in headers
    assert "Content-Type" in headers


@pytest.mark.unit
@pytest.mark.asyncio
async def test_health_service_initialize_mcp_session_no_server_session_id(health_service):
    """Test initializing MCP session when server doesn't return session ID."""
    mock_client = AsyncMock(spec=httpx.AsyncClient)
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.headers = {}
    mock_client.post.return_value = mock_response

    session_id = await health_service._initialize_mcp_session(
        mock_client, "http://localhost:8000/mcp", {}
    )

    # Should generate client-side session ID
    assert session_id is not None
    import uuid

    uuid.UUID(session_id)  # Verify it's a valid UUID


@pytest.mark.unit
@pytest.mark.asyncio
async def test_health_service_initialize_mcp_session_exception(health_service):
    """Test initializing MCP session with exception."""
    mock_client = AsyncMock(spec=httpx.AsyncClient)
    mock_client.post.side_effect = Exception("Network error")

    session_id = await health_service._initialize_mcp_session(
        mock_client, "http://localhost:8000/mcp", {}
    )

    assert session_id is None


@pytest.mark.unit
@pytest.mark.asyncio
async def test_health_service_try_ping_without_auth_auth_errors(health_service):
    """Test ping without auth when server returns auth errors."""
    mock_client = AsyncMock(spec=httpx.AsyncClient)
    mock_response = MagicMock()
    mock_response.status_code = 401
    mock_client.post.return_value = mock_response

    result = await health_service._try_ping_without_auth(mock_client, "http://localhost:8000/mcp")

    assert result is True


@pytest.mark.unit
@pytest.mark.asyncio
async def test_health_service_try_ping_without_auth_server_error(health_service):
    """Test ping without auth when server returns error."""
    mock_client = AsyncMock(spec=httpx.AsyncClient)
    mock_response = MagicMock()
    mock_response.status_code = 500
    mock_client.post.return_value = mock_response

    result = await health_service._try_ping_without_auth(mock_client, "http://localhost:8000/mcp")

    assert result is False


@pytest.mark.unit
@pytest.mark.asyncio
async def test_health_service_check_server_endpoint_sse_transport(health_service, mock_server_info):
    """Test checking server endpoint with SSE transport."""
    mock_server_info["supported_transports"] = ["sse"]
    proxy_url = "http://localhost:8000"

    mock_client = AsyncMock(spec=httpx.AsyncClient)
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_client.get.return_value = mock_response

    with patch.object(health_service, "_is_mcp_endpoint_healthy", return_value=True):
        is_healthy, status = await health_service._check_server_endpoint_transport_aware(
            mock_client, proxy_url, mock_server_info
        )

        assert is_healthy is True
        assert status == HealthStatus.HEALTHY


@pytest.mark.unit
@pytest.mark.asyncio
async def test_health_service_check_server_endpoint_sse_timeout(health_service, mock_server_info):
    """Test checking server endpoint with SSE transport timeout."""
    mock_server_info["supported_transports"] = ["sse"]
    proxy_url = "http://localhost:8000"

    mock_client = AsyncMock(spec=httpx.AsyncClient)
    mock_client.get.side_effect = TimeoutError()

    is_healthy, status = await health_service._check_server_endpoint_transport_aware(
        mock_client, proxy_url, mock_server_info
    )

    # SSE timeout is considered healthy
    assert is_healthy is True


@pytest.mark.unit
@pytest.mark.asyncio
async def test_health_service_check_server_endpoint_url_with_mcp(health_service, mock_server_info):
    """Test checking server endpoint when URL already has /mcp."""
    proxy_url = "http://localhost:8000/mcp"

    mock_client = AsyncMock(spec=httpx.AsyncClient)
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_client.post.return_value = mock_response

    with patch.object(health_service, "_initialize_mcp_session", return_value="session-123"):
        is_healthy, status = await health_service._check_server_endpoint_transport_aware(
            mock_client, proxy_url, mock_server_info
        )

        assert is_healthy is True
        assert status == HealthStatus.HEALTHY


@pytest.mark.unit
@pytest.mark.asyncio
async def test_health_service_check_server_endpoint_auth_failure(health_service, mock_server_info):
    """Test checking server endpoint with auth failure."""
    proxy_url = "http://localhost:8000/mcp"

    mock_client = AsyncMock(spec=httpx.AsyncClient)
    mock_response = MagicMock()
    mock_response.status_code = 401
    mock_client.get.return_value = mock_response

    with patch.object(health_service, "_try_ping_without_auth", return_value=True):
        is_healthy, status = await health_service._check_server_endpoint_transport_aware(
            mock_client, proxy_url, mock_server_info
        )

        assert is_healthy is True


@pytest.mark.unit
def test_health_service_get_service_health_data_fast_transitioning_from_disabled(
    health_service, mock_server_info
):
    """Test getting service health data when transitioning from disabled."""
    service_path = "/test-server"
    health_service.server_health_status[service_path] = "disabled"

    health_data = health_service._get_service_health_data_fast(service_path, mock_server_info)

    # Should transition to checking
    assert health_data["status"] == HealthStatus.CHECKING


@pytest.mark.unit
def test_health_service_get_service_health_data_legacy_method(health_service, mock_server_info):
    """Test legacy _get_service_health_data method."""
    service_path = "/test-server"
    health_service.server_health_status[service_path] = HealthStatus.HEALTHY

    health_data = health_service._get_service_health_data(service_path, mock_server_info)

    assert health_data["status"] == HealthStatus.HEALTHY
