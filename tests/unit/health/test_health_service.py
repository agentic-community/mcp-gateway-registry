"""
Unit tests for health monitoring service.
"""

import asyncio
from datetime import datetime, timezone
from unittest.mock import AsyncMock, patch

import pytest

from registry.health.service import HealthMonitoringService


@pytest.mark.unit
@pytest.mark.health
class TestHealthMonitoringService:
    """Test suite for HealthMonitoringService."""

    def test_init(self, health_service: HealthMonitoringService) -> None:
        """Initializes with empty state."""
        assert health_service.server_health_status == {}
        assert health_service.server_last_check_time == {}
        assert health_service.websocket_manager.connections == set()
        assert health_service.health_check_task is None

    @pytest.mark.asyncio
    async def test_initialize_starts_health_task(
        self,
        health_service: HealthMonitoringService,
    ) -> None:
        """Starts background health check task."""
        with patch("asyncio.create_task") as mock_create_task:
            mock_task = AsyncMock()

            def _create_task_and_close(coro):
                coro.close()
                return mock_task

            mock_create_task.side_effect = _create_task_and_close

            await health_service.initialize()

            assert mock_create_task.called
            assert health_service.health_check_task == mock_task

    @pytest.mark.asyncio
    async def test_shutdown_cancels_task_and_closes_connections(
        self,
        health_service: HealthMonitoringService,
    ) -> None:
        """Cancels background task and closes all websocket connections."""
        health_service.health_check_task = asyncio.create_task(asyncio.sleep(1000))

        mock_conn1 = AsyncMock()
        mock_conn1.close = AsyncMock()
        mock_conn2 = AsyncMock()
        mock_conn2.close = AsyncMock()
        health_service.websocket_manager.connections = {mock_conn1, mock_conn2}

        await health_service.shutdown()

        assert health_service.health_check_task.cancelled()
        mock_conn1.close.assert_called_once()
        mock_conn2.close.assert_called_once()
        assert health_service.websocket_manager.connections == set()

    @pytest.mark.asyncio
    async def test_add_websocket_connection_delegates_to_manager(
        self,
        health_service: HealthMonitoringService,
        mock_websocket,
    ) -> None:
        """Delegates adding to websocket manager."""
        health_service.websocket_manager.add_connection = AsyncMock(return_value=True)

        result = await health_service.add_websocket_connection(mock_websocket)

        assert result is True
        health_service.websocket_manager.add_connection.assert_awaited_once_with(
            mock_websocket
        )

    @pytest.mark.asyncio
    async def test_remove_websocket_connection_delegates_to_manager(
        self,
        health_service: HealthMonitoringService,
        mock_websocket,
    ) -> None:
        """Delegates removal to websocket manager."""
        health_service.websocket_manager.remove_connection = AsyncMock()

        await health_service.remove_websocket_connection(mock_websocket)

        health_service.websocket_manager.remove_connection.assert_awaited_once_with(
            mock_websocket
        )

    def test_get_service_health_data_fast_enabled(
        self,
        health_service: HealthMonitoringService,
    ) -> None:
        """Returns cached status and tool count for enabled services."""
        last_check = datetime(2023, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        health_service.server_health_status["/test"] = "healthy"
        health_service.server_last_check_time["/test"] = last_check

        with patch("registry.services.server_service.server_service") as mock_server_service:
            mock_server_service.is_service_enabled.return_value = True

            result = health_service._get_service_health_data_fast(
                "/test",
                {"num_tools": 10},
            )

        assert result["status"] == "healthy"
        assert result["last_checked_iso"] == last_check.isoformat()
        assert result["num_tools"] == 10

    def test_get_service_health_data_fast_disabled_sets_disabled(
        self,
        health_service: HealthMonitoringService,
    ) -> None:
        """Returns disabled status for disabled services."""
        with patch("registry.services.server_service.server_service") as mock_server_service:
            mock_server_service.is_service_enabled.return_value = False

            result = health_service._get_service_health_data_fast(
                "/test",
                {"num_tools": 10},
            )

        assert result["status"] == "disabled"
        assert health_service.server_health_status["/test"] == "disabled"

    def test_get_all_health_status(
        self,
        health_service: HealthMonitoringService,
    ) -> None:
        """Returns health status for all registered servers."""
        with patch("registry.services.server_service.server_service") as mock_server_service:
            mock_server_service.get_all_servers.return_value = {
                "/test1": {"num_tools": 1},
                "/test2": {"num_tools": 2},
            }
            mock_server_service.is_service_enabled.return_value = True

            result = health_service.get_all_health_status()

        assert result["/test1"]["num_tools"] == 1
        assert result["/test2"]["num_tools"] == 2
