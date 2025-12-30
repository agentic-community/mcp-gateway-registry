from __future__ import annotations

import asyncio
import json
import logging
from time import (
    time,
)
from typing import (
    Callable,
    Dict,
    Optional,
    Set,
)

from fastapi import (
    WebSocket,
)

from ..core.config import (
    settings,
)

logger = logging.getLogger(__name__)


class HighPerformanceWebSocketManager:
    """High-performance WebSocket manager for 400-1000+ concurrent connections."""

    def __init__(
        self,
        get_cached_health_data: Callable[[], Dict],
    ):
        self._get_cached_health_data = get_cached_health_data

        self.connections: Set[WebSocket] = set()
        self.connection_metadata: Dict[WebSocket, Dict] = {}

        # Rate limiting and batching
        self.pending_updates: Dict[str, Dict] = {}  # service_path -> latest_data
        self.last_broadcast_time = 0
        self.min_broadcast_interval = settings.websocket_broadcast_interval_ms / 1000.0
        self.max_batch_size = settings.websocket_max_batch_size

        # Connection health tracking
        self.failed_connections: Set[WebSocket] = set()
        self.cleanup_task: Optional[asyncio.Task] = None

        # Performance metrics
        self.broadcast_count = 0
        self.failed_send_count = 0

    async def add_connection(
        self,
        websocket: WebSocket,
    ) -> bool:
        """Add a new WebSocket connection with connection limits."""
        try:
            if len(self.connections) >= settings.max_websocket_connections:
                logger.warning("Connection limit reached: %s", len(self.connections))
                await websocket.close(code=1008, reason="Server at capacity")
                return False

            await websocket.accept()
            self.connections.add(websocket)
            self.connection_metadata[websocket] = {
                "connected_at": time(),
                "last_ping": time(),
                "client_ip": getattr(websocket.client, "host", "unknown")
                if websocket.client
                else "unknown",
            }

            logger.debug(
                "WebSocket connected: %s total connections",
                len(self.connections),
            )

            await self._send_initial_status_optimized(websocket)
            return True

        except Exception as exc:
            logger.error("Error adding WebSocket connection: %s", exc)
            return False

    async def remove_connection(
        self,
        websocket: WebSocket,
    ):
        """Remove a WebSocket connection."""
        self.connections.discard(websocket)
        self.connection_metadata.pop(websocket, None)
        self.failed_connections.discard(websocket)

        logger.debug(
            "WebSocket disconnected: %s total connections",
            len(self.connections),
        )

    async def _send_initial_status_optimized(
        self,
        websocket: WebSocket,
    ):
        """Send initial status using cached data to avoid blocking."""
        try:
            cached_data = self._get_cached_health_data()
            if cached_data:
                await websocket.send_text(json.dumps(cached_data))
        except Exception as exc:
            logger.warning("Failed to send initial status: %s", exc)
            await self.remove_connection(websocket)

    async def broadcast_update(
        self,
        service_path: Optional[str] = None,
        health_data: Optional[Dict] = None,
    ):
        """High-performance broadcasting with batching and rate limiting."""
        if not self.connections:
            return

        current_time = time()

        # Rate limiting: prevent too frequent broadcasts
        if current_time - self.last_broadcast_time < self.min_broadcast_interval:
            if service_path and health_data:
                self.pending_updates[service_path] = health_data
            return

        if service_path and health_data:
            broadcast_data = {service_path: health_data}
        else:
            if self.pending_updates:
                batch_data = dict(list(self.pending_updates.items())[: self.max_batch_size])
                broadcast_data = batch_data
                for key in batch_data.keys():
                    self.pending_updates.pop(key, None)
            else:
                broadcast_data = self._get_cached_health_data()

        if broadcast_data:
            await self._send_to_connections_optimized(broadcast_data)
            self.last_broadcast_time = current_time

    async def _send_to_connections_optimized(
        self,
        data: Dict,
    ):
        """Optimized concurrent sending with automatic cleanup."""
        if not self.connections:
            return

        message = json.dumps(data)
        connections_list = list(self.connections)

        chunk_size = 100
        for i in range(0, len(connections_list), chunk_size):
            chunk = connections_list[i : i + chunk_size]

            tasks = [self._safe_send_message(conn, message) for conn in chunk]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            for conn, result in zip(chunk, results):
                if isinstance(result, Exception):
                    self.failed_connections.add(conn)
                    self.failed_send_count += 1

        if self.failed_connections:
            asyncio.create_task(self._cleanup_failed_connections())

        self.broadcast_count += 1

    async def _safe_send_message(
        self,
        connection: WebSocket,
        message: str,
    ):
        """Send message with timeout and error handling."""
        try:
            await asyncio.wait_for(
                connection.send_text(message),
                timeout=settings.websocket_send_timeout_seconds,
            )
            return True
        except asyncio.TimeoutError:
            return TimeoutError("Send timeout")
        except Exception as exc:
            return exc

    async def _cleanup_failed_connections(
        self,
    ):
        """Cleanup failed connections without blocking main operations."""
        failed_count = len(self.failed_connections)
        if failed_count == 0:
            return

        for conn in list(self.failed_connections):
            await self.remove_connection(conn)

        logger.info("Cleaned up %s failed WebSocket connections", failed_count)

    def get_stats(self) -> Dict:
        """Get performance statistics."""
        return {
            "active_connections": len(self.connections),
            "pending_updates": len(self.pending_updates),
            "total_broadcasts": self.broadcast_count,
            "failed_sends": self.failed_send_count,
            "failed_connections": len(self.failed_connections),
        }

