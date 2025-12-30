import asyncio
import logging
import httpx
from datetime import datetime, timezone
from typing import Dict, Optional
from fastapi import WebSocket
from time import time

from ..core.config import settings
from registry.constants import HealthStatus

from .mcp_transport_checks import McpTransportChecker
from .websocket_manager import HighPerformanceWebSocketManager

logger = logging.getLogger(__name__)


class HealthMonitoringService:
    """Optimized health monitoring service for high-scale WebSocket operations."""

    def __init__(self):
        self.server_health_status: Dict[str, str] = {}
        self.server_last_check_time: Dict[str, datetime] = {}

        # High-performance WebSocket manager
        self.websocket_manager = HighPerformanceWebSocketManager(
            get_cached_health_data=self._get_cached_health_data,
        )

        self._mcp_transport_checker = McpTransportChecker()

        # Background task management
        self.health_check_task: Optional[asyncio.Task] = None

        # Performance optimizations
        self._cached_health_data: Dict = {}
        self._cache_timestamp = 0
        self._cache_ttl = settings.websocket_cache_ttl_seconds

    async def initialize(self):
        """Initialize the health monitoring service."""
        logger.info("Initializing health monitoring service...")

        # Start background health checks
        self.health_check_task = asyncio.create_task(self._run_health_checks())

        logger.info("Health monitoring service initialized!")

    async def shutdown(self):
        """Shutdown the health monitoring service."""
        # Cancel background tasks
        if self.health_check_task:
            self.health_check_task.cancel()
            try:
                await self.health_check_task
            except asyncio.CancelledError:
                pass

        # Close all WebSocket connections
        connections = list(self.websocket_manager.connections)
        close_tasks = []
        remove_tasks = []
        for conn in connections:
            try:
                close_tasks.append(conn.close())
            except Exception:
                logger.debug("Failed to schedule WebSocket close", exc_info=True)
            try:
                remove_tasks.append(self.websocket_manager.remove_connection(conn))
            except Exception:
                logger.debug(
                    "Failed to schedule WebSocket removal from manager",
                    exc_info=True,
                )

        if close_tasks:
            await asyncio.gather(*close_tasks, return_exceptions=True)
        if remove_tasks:
            await asyncio.gather(*remove_tasks, return_exceptions=True)

        logger.info("Health monitoring service shutdown complete")

    async def add_websocket_connection(self, websocket: WebSocket):
        """Add a new WebSocket connection and send initial health status."""
        success = await self.websocket_manager.add_connection(websocket)
        if success:
            logger.info(f"WebSocket client connected: {websocket.client}")
        return success

    async def remove_websocket_connection(self, websocket: WebSocket):
        """Remove a WebSocket connection."""
        await self.websocket_manager.remove_connection(websocket)
        logger.info(f"WebSocket connection removed: {websocket.client}")

    async def _send_initial_status(self, websocket: WebSocket):
        """Send initial health status to a newly connected WebSocket client."""
        # This method is kept for compatibility but delegates to the optimized manager
        await self.websocket_manager._send_initial_status_optimized(websocket)

    async def broadcast_health_update(self, service_path: Optional[str] = None):
        """Broadcast health status updates to all connected WebSocket clients."""
        if not self.websocket_manager.connections:
            return

        from ..services.server_service import server_service

        if service_path:
            # Single service update - get data efficiently
            server_info = server_service.get_server_info(service_path)
            if server_info:
                health_data = self._get_service_health_data_fast(service_path, server_info)
                await self.websocket_manager.broadcast_update(service_path, health_data)
        else:
            # Full update - use cached data
            await self.websocket_manager.broadcast_update()

    def _get_cached_health_data(self) -> Dict:
        """Get cached health data to avoid expensive operations during WebSocket sends."""
        current_time = time()

        # Return cached data if still valid
        if (current_time - self._cache_timestamp) < self._cache_ttl and self._cached_health_data:
            return self._cached_health_data

        # Rebuild cache
        from ..services.server_service import server_service
        all_servers = server_service.get_all_servers()

        data = {}
        for path, server_info in all_servers.items():
            data[path] = self._get_service_health_data_fast(path, server_info)

        self._cached_health_data = data
        self._cache_timestamp = current_time
        return data

    def get_websocket_stats(self) -> Dict:
        """Get WebSocket performance statistics."""
        return self.websocket_manager.get_stats()

    async def _run_health_checks(self):
        from .health_loop import run_health_checks

        await run_health_checks(self)

    async def _perform_health_checks(self):
        from .health_loop import perform_health_checks

        await perform_health_checks(self)

    async def _check_single_service(self, client: httpx.AsyncClient, service_path: str, server_info: Dict) -> bool:
        from .health_loop import check_single_service

        return await check_single_service(self, client, service_path, server_info)


    async def _check_server_endpoint_transport_aware(
        self,
        client: httpx.AsyncClient,
        proxy_pass_url: str,
        server_info: Dict,
    ) -> tuple[bool, str]:
        """Check server endpoint using transport-aware logic."""
        return await self._mcp_transport_checker._check_server_endpoint_transport_aware(
            client,
            proxy_pass_url,
            server_info,
        )

    async def _update_tools_background(self, service_path: str, proxy_pass_url: str):
        from .tool_discovery import update_tools_background

        await update_tools_background(self, service_path, proxy_pass_url)

    def get_all_health_status(self) -> Dict:
        """Get health status for all services."""
        from ..services.server_service import server_service

        all_servers = server_service.get_all_servers()

        data = {}
        for path, server_info in all_servers.items():
            data[path] = self._get_service_health_data_fast(path, server_info)

        return data

    async def perform_immediate_health_check(self, service_path: str) -> tuple[str, datetime | None]:
        """Perform an immediate health check for a single service."""
        from ..services.server_service import server_service

        server_info = server_service.get_server_info(service_path)
        if not server_info:
            return "error: server not registered", None

        proxy_pass_url = server_info.get("proxy_pass_url")

        # Record check time
        last_checked_time = datetime.now(timezone.utc)
        self.server_last_check_time[service_path] = last_checked_time

        if not proxy_pass_url:
            current_status = HealthStatus.UNHEALTHY_MISSING_PROXY_URL
            self.server_health_status[service_path] = current_status
            logger.info(f"Health check skipped for {service_path}: Missing URL.")
            return current_status, last_checked_time

        previous_status = self.server_health_status.get(service_path, HealthStatus.UNKNOWN)
        self.server_health_status[service_path] = HealthStatus.CHECKING

        try:
            from .health_loop import check_single_service

            async with httpx.AsyncClient(timeout=httpx.Timeout(settings.health_check_timeout_seconds)) as client:
                await check_single_service(
                    self,
                    client,
                    service_path,
                    server_info,
                    previous_status_override=previous_status,
                )
        except Exception as e:
            current_status = f"error: {type(e).__name__}"
            logger.error(
                "Unexpected error during health check for %s: %s",
                service_path,
                e,
                exc_info=True,
            )
        else:
            current_status = self.server_health_status.get(service_path, HealthStatus.UNKNOWN)

        # Update the status
        self.server_health_status[service_path] = current_status
        logger.info(f"Final health status for {service_path}: {current_status}")

        # Regenerate nginx configuration if status changed
        if previous_status != current_status:
            try:
                from ..core.nginx_service import nginx_service
                enabled_servers = {
                    path: server_service.get_server_info(path)
                    for path in server_service.get_enabled_services()
                }
                await nginx_service.generate_config_async(enabled_servers)
                logger.info(f"Nginx configuration regenerated due to status change for {service_path}: {previous_status} -> {current_status}")
            except Exception as e:
                logger.error(f"Failed to regenerate nginx configuration after immediate health check: {e}")

        return current_status, last_checked_time

    def get_service_health_data(
        self,
        service_path: str,
        server_info: Optional[Dict] = None,
    ) -> Dict:
        """Get health data for a specific service.

        Args:
            service_path: Service path (e.g. `/myservice`).
            server_info: Optional server info dict to avoid repeated lookups.

        Returns:
            Dictionary with health fields (status, last_checked_iso, num_tools).
        """
        if server_info is None:
            from ..services.server_service import server_service

            server_info = server_service.get_server_info(service_path) or {}

        return self._get_service_health_data_fast(
            service_path,
            server_info,
        )

    def _get_service_health_data(self, service_path: str) -> Dict:
        """Get health data for a specific service - legacy method, use _get_service_health_data_fast for better performance."""
        return self.get_service_health_data(service_path)

    def _get_service_health_data_fast(self, service_path: str, server_info: Dict) -> Dict:
        """Get health data for a specific service - optimized version."""
        from ..services.server_service import server_service

        # Quick enabled check using cached server_info if possible
        is_enabled = server_service.is_service_enabled(service_path)

        if not is_enabled:
            status = "disabled"
            self.server_health_status[service_path] = "disabled"
        else:
            # Use cached status, only update if transitioning from disabled
            cached_status = self.server_health_status.get(service_path, "unknown")
            if cached_status == "disabled":
                status = HealthStatus.CHECKING
                self.server_health_status[service_path] = HealthStatus.CHECKING
            else:
                status = cached_status

        # Use pre-fetched server_info instead of calling get_server_info again
        last_checked_dt = self.server_last_check_time.get(service_path)
        last_checked_iso = last_checked_dt.isoformat() if last_checked_dt else None
        num_tools = server_info.get("num_tools", 0) if server_info else 0

        return {
            "status": status,
            "last_checked_iso": last_checked_iso,
            "num_tools": num_tools
        }


# Global health monitoring service instance
health_service = HealthMonitoringService()
